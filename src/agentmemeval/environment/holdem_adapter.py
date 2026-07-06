"""
模块说明：本模块实现一个独立的本地 Texas Hold'em 环境适配器。
核心职责：提供发牌、盲注、轮流行动、合法动作、街道推进、摊牌和结算。
输入与输出：输入 TableSpec、seed 与 ActionDecision，输出观察、事件和 HandResult。
依赖边界：不依赖官方仓库代码，不依赖外部扑克引擎，不依赖 LLM 或记忆模块。
不负责：不进行实验指标聚合，不保存工件，不处理现金局外部买入规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.core.domain import (
    ActionDecision,
    AgentId,
    AgentObservation,
    HandResult,
    LegalAction,
    LegalActionSet,
    PlayerPublicState,
    StepResult,
    TableSpec,
)
from agentmemeval.core.errors import EnvironmentError
from agentmemeval.core.seeds import make_rng
from agentmemeval.environment.action_guard import ActionGuard
from agentmemeval.environment.hand_evaluator import evaluate_best, split_side_pots

RANKS = "23456789TJQKA"
SUITS = "cdhs"
PHASES = ("preflop", "flop", "turn", "river")


@dataclass(slots=True)
class _PlayerState:
    """
    功能：保存环境内部玩家状态。
    参数：
        agent_id：Agent 标识。
        seat：座位号。
        stack：剩余筹码。
        hole_cards：私有手牌。
        current_bet：本街下注。
        total_committed：本手累计下注。
        folded：是否弃牌。
    返回：内部状态对象。
    副作用：无。
    异常：无。
    设计说明：私有手牌只存在于环境内部，观察构造时按可见性过滤。
    """

    agent_id: AgentId
    seat: int
    stack: int
    hole_cards: list[str] = field(default_factory=list)
    current_bet: int = 0
    total_committed: int = 0
    folded: bool = False

    @property
    def all_in(self) -> bool:
        """
        功能：判断玩家是否全下。
        参数：无。
        返回：布尔值。
        副作用：无。
        异常：无。
        设计说明：全下玩家不再获得行动权，但仍可能参与摊牌。
        """

        return self.stack <= 0 and not self.folded


class HoldemEnvironment:
    """
    功能：独立本地德州扑克环境。
    参数：无。
    返回：环境实例。
    副作用：运行中维护牌局状态。
    异常：非法状态或非法动作时抛出 EnvironmentError。
    设计说明：本实现覆盖实验闭环所需核心规则，复杂边池行为在文档中标为待增强。
    """

    def __init__(self) -> None:
        """
        功能：初始化空环境。
        参数：无。
        返回：无。
        副作用：创建内部状态容器。
        异常：无。
        设计说明：具体牌桌由 reset 注入，方便实验层反复复用。
        """

        self.table_spec: TableSpec | None = None
        self.seed = 0
        self.rng = make_rng(0)
        self.players: list[_PlayerState] = []
        self.players_by_id: dict[AgentId, _PlayerState] = {}
        self.deck: list[str] = []
        self.community_cards: list[str] = []
        self.pot = 0
        self.phase_index = 0
        self.current_bet = 0
        self.current_seat: int | None = None
        self.dealer_index = 0
        self.acted: set[int] = set()
        self.raise_count = 0
        self.action_history: list[dict[str, object]] = []
        self.finished = True
        self.hand_id = "uninitialized"
        self.starting_stacks: dict[AgentId, int] = {}
        self.final_result: HandResult | None = None
        self._guard = ActionGuard()
        self._hand_counter = 0

    def reset(self, table_spec: TableSpec, seed: int) -> None:
        """
        功能：重置牌桌并开始一手牌。
        参数：
            table_spec：桌面配置与当前筹码。
            seed：本手随机 seed。
        返回：无。
        副作用：重置并发牌、贴盲。
        异常：少于两名玩家或盲注配置非法时抛出 EnvironmentError。
        设计说明：实验层用当前筹码构造 TableSpec，因此环境不保存跨手全局状态。
        """

        if len(table_spec.agent_ids) < 2:
            raise EnvironmentError("一张桌至少需要两名 Agent")
        if table_spec.small_blind <= 0 or table_spec.big_blind < table_spec.small_blind:
            raise EnvironmentError("盲注配置非法：big_blind 必须不小于 small_blind")
        self.table_spec = table_spec
        self.seed = seed
        self.rng = make_rng(seed, table_spec.table_id)
        self._hand_counter += 1
        self.hand_id = f"{table_spec.table_id}-h{self._hand_counter}-{seed}"
        self.players = [
            _PlayerState(agent_id=agent_id, seat=seat, stack=table_spec.starting_stacks[agent_id])
            for seat, agent_id in enumerate(table_spec.agent_ids)
        ]
        self.players_by_id = {player.agent_id: player for player in self.players}
        self.starting_stacks = {player.agent_id: player.stack for player in self.players}
        self.deck = [rank + suit for rank in RANKS for suit in SUITS]
        self.rng.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.phase_index = 0
        self.current_bet = 0
        self.acted = set()
        self.raise_count = 0
        self.action_history = []
        self.finished = False
        self.final_result = None
        self.dealer_index = (self._hand_counter - 1) % len(self.players)
        for player in self.players:
            player.hole_cards = [self.deck.pop(), self.deck.pop()]
            player.current_bet = 0
            player.total_committed = 0
            player.folded = player.stack <= 0
        active = [player for player in self.players if player.stack > 0]
        if len(active) < 2:
            self.finished = True
            self.current_seat = None
            self._settle_without_showdown()
            return
        self._post_blinds()
        self.current_seat = self._first_preflop_seat()
        self._skip_unable_actors()

    def current_agent_id(self) -> AgentId | None:
        """
        功能：返回当前行动者 AgentId。
        参数：无。
        返回：AgentId 或 None。
        副作用：无。
        异常：无。
        设计说明：场景循环以该方法作为是否继续行动的真相源。
        """

        if self.current_seat is None:
            return None
        player = self._player_by_seat(self.current_seat)
        return player.agent_id if player else None

    def current_observation(self, agent_id: AgentId) -> AgentObservation:
        """
        功能：构造指定 Agent 的合法可见观察。
        参数：
            agent_id：观察者。
        返回：AgentObservation。
        副作用：无。
        异常：未知 Agent 时抛出 EnvironmentError。
        设计说明：只把观察者自己的 hole cards 放入观察，公共牌只包含已揭示部分。
        """

        if self.table_spec is None:
            raise EnvironmentError("环境尚未 reset")
        player = self.players_by_id.get(agent_id)
        if player is None:
            raise EnvironmentError(f"未知 Agent：{agent_id}")
        players = [
            PlayerPublicState(
                agent_id=item.agent_id,
                seat=item.seat,
                stack=item.stack,
                current_bet=item.current_bet,
                total_committed=item.total_committed,
                folded=item.folded,
                all_in=item.all_in,
                busted=item.stack <= 0 and item.folded,
            )
            for item in self.players
        ]
        return AgentObservation(
            agent_id=agent_id,
            table_id=self.table_spec.table_id,
            hand_id=self.hand_id,
            phase=self._phase,
            seat=player.seat,
            hole_cards=list(player.hole_cards),
            community_cards=list(self.community_cards),
            pot=self.pot,
            current_bet=self.current_bet,
            to_call=max(0, self.current_bet - player.current_bet),
            players=players,
            action_history=list(self.action_history),
            legal_actions=self.legal_actions(agent_id),
            seed=self.seed,
        )

    def legal_actions(self, agent_id: AgentId) -> LegalActionSet:
        """
        功能：返回指定 Agent 当前合法动作。
        参数：
            agent_id：Agent 标识。
        返回：LegalActionSet。
        副作用：无。
        异常：无。
        设计说明：非当前行动者得到空集合，环境执行仍会二次校验。
        """

        player = self.players_by_id.get(agent_id)
        if (
            self.finished
            or self.table_spec is None
            or player is None
            or self.current_seat != player.seat
            or player.folded
            or player.stack <= 0
        ):
            return LegalActionSet(actions=[])
        actions: list[LegalAction] = [LegalAction("fold")]
        to_call = max(0, self.current_bet - player.current_bet)
        if to_call == 0:
            actions.append(LegalAction("check"))
        else:
            actions.append(LegalAction("call"))
        if self.raise_count < self.table_spec.max_raises_per_street and player.stack > to_call:
            min_to = self.current_bet + self.table_spec.big_blind
            max_to = player.current_bet + player.stack
            if max_to >= min_to:
                actions.append(LegalAction("raise", min_amount=min_to, max_amount=max_to))
            elif max_to > self.current_bet:
                actions.append(
                    LegalAction("raise", min_amount=max_to, max_amount=max_to, reopens=False)
                )
        return LegalActionSet(actions=actions)

    def step(self, agent_id: AgentId, action: ActionDecision) -> StepResult:
        """
        功能：执行当前行动者动作并推进环境。
        参数：
            agent_id：行动者。
            action：候选动作。
        返回：StepResult。
        副作用：更新筹码、底池、行动历史和街道。
        异常：非当前行动者或非法动作时抛出 EnvironmentError。
        设计说明：环境作为最后防线，即使上层已 guard 也重新校验。
        """

        if self.finished:
            raise EnvironmentError("本手已结束，不能继续执行动作")
        if self.current_agent_id() != agent_id:
            raise EnvironmentError(f"当前行动者不是 {agent_id}")
        legal = self.legal_actions(agent_id)
        guard = self._guard.guard(action, legal, strict=True)
        player = self.players_by_id[agent_id]
        to_call = max(0, self.current_bet - player.current_bet)
        pot_before = self.pot
        effective_raise = False
        committed = 0
        if guard.action.action_type == "fold":
            player.folded = True
        elif guard.action.action_type == "check":
            committed = 0
        elif guard.action.action_type == "call":
            committed = min(to_call, player.stack)
            self._commit(player, committed)
        elif guard.action.action_type == "raise":
            target = guard.action.amount
            if target is None:
                raise EnvironmentError("raise 动作缺少 amount")
            committed = min(max(0, target - player.current_bet), player.stack)
            old_bet = self.current_bet
            self._commit(player, committed)
            if player.current_bet > old_bet:
                self.current_bet = player.current_bet
                rule = legal.rule_for("raise")
                effective_raise = bool(rule and rule.reopens)
                if effective_raise:
                    self.raise_count += 1
                    self.acted = set()
        self.acted.add(player.seat)
        event = {
            "event": "action",
            "table_id": self.table_spec.table_id if self.table_spec else "",
            "hand_id": self.hand_id,
            "phase": self._phase,
            "agent_id": agent_id,
            "action_type": guard.action.action_type,
            "amount": guard.action.amount,
            "committed": committed,
            "to_call": to_call,
            "pot_before": pot_before,
            "pot_after": self.pot,
            "effective_raise": effective_raise,
        }
        self.action_history.append(event)
        self._advance_after_action(player.seat)
        return StepResult(event=event, hand_finished=self.finished)

    def is_hand_finished(self) -> bool:
        """
        功能：返回本手是否结束。
        参数：无。
        返回：布尔值。
        副作用：无。
        异常：无。
        设计说明：场景层通过该方法控制行动循环。
        """

        return self.finished

    def finalize_hand(self) -> HandResult:
        """
        功能：返回本手结算结果。
        参数：无。
        返回：HandResult。
        副作用：若尚未构造结算结果，会执行兜底结算。
        异常：无。
        设计说明：调用方只在 hand_finished 后调用；兜底是为了增强健壮性。
        """

        if self.final_result is None:
            if not self.finished:
                self._run_showdown()
            else:
                self._settle_without_showdown()
        assert self.final_result is not None
        return self.final_result

    @property
    def _phase(self) -> str:
        """
        功能：返回当前街道名称。
        参数：无。
        返回：阶段字符串。
        副作用：无。
        异常：无。
        设计说明：内部用索引推进，外部始终看到稳定字符串。
        """

        if self.finished:
            return "hand_over"
        return PHASES[min(self.phase_index, len(PHASES) - 1)]

    def _post_blinds(self) -> None:
        """
        功能：按按钮位置贴小盲和大盲。
        参数：无。
        返回：无。
        副作用：更新两个玩家筹码与底池。
        异常：无。
        设计说明：heads-up 时按钮位为小盲；多人时按钮下家为小盲。
        """

        if self.table_spec is None:
            return
        active = [player for player in self.players if player.stack > 0]
        if len(active) == 2:
            sb = self.players[self.dealer_index]
            bb = self._next_active_player(sb.seat)
        else:
            sb = self._next_active_player(self.players[self.dealer_index].seat)
            bb = self._next_active_player(sb.seat)
        self._commit(sb, min(sb.stack, self.table_spec.small_blind))
        self._commit(bb, min(bb.stack, self.table_spec.big_blind))
        self.current_bet = max(sb.current_bet, bb.current_bet)

    def _first_preflop_seat(self) -> int | None:
        """
        功能：确定翻牌前首个行动座位。
        参数：无。
        返回：座位号或 None。
        副作用：无。
        异常：无。
        设计说明：heads-up 由小盲先行动，多人由大盲后第一位行动。
        """

        active = [player for player in self.players if player.stack > 0 and not player.folded]
        if len(active) < 2:
            return None
        dealer = self.players[self.dealer_index]
        if len(active) == 2:
            return dealer.seat
        sb = self._next_active_player(dealer.seat)
        bb = self._next_active_player(sb.seat)
        return self._next_active_player(bb.seat).seat

    def _commit(self, player: _PlayerState, amount: int) -> None:
        """
        功能：让玩家向底池投入筹码。
        参数：
            player：玩家状态。
            amount：投入数量。
        返回：无。
        副作用：修改玩家筹码、下注线和底池。
        异常：无。
        设计说明：所有 call/raise/盲注都通过该函数，避免筹码更新分散。
        """

        real = max(0, min(amount, player.stack))
        player.stack -= real
        player.current_bet += real
        player.total_committed += real
        self.pot += real

    def _advance_after_action(self, previous_seat: int) -> None:
        """
        功能：动作后决定是否结束本手、进入下一街或换到下一行动者。
        参数：
            previous_seat：刚行动的座位。
        返回：无。
        副作用：更新 current_seat、finished 或公共牌。
        异常：无。
        设计说明：投注闭合条件与行动者选择集中在这里，减少规则分叉。
        """

        live = [player for player in self.players if not player.folded]
        if len(live) <= 1:
            self._settle_without_showdown()
            return
        can_act = [player for player in live if player.stack > 0]
        if not can_act:
            self._deal_remaining_and_showdown()
            return
        round_closed = all(player.current_bet == self.current_bet for player in can_act) and all(
            player.seat in self.acted for player in can_act
        )
        if round_closed:
            self._advance_street()
            return
        self.current_seat = self._next_actor_seat(previous_seat)
        self._skip_unable_actors()

    def _advance_street(self) -> None:
        """
        功能：进入下一条街并揭示公共牌。
        参数：无。
        返回：无。
        副作用：修改公共牌、下注线和行动者。
        异常：无。
        设计说明：公共牌只在街道推进时加入 observation，避免提前泄露。
        """

        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        self.raise_count = 0
        self.acted = set()
        self.phase_index += 1
        if self.phase_index == 1:
            self.community_cards.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])
        elif self.phase_index in (2, 3):
            self.community_cards.append(self.deck.pop())
        elif self.phase_index >= 4:
            self._run_showdown()
            return
        can_act = [player for player in self.players if not player.folded and player.stack > 0]
        if not can_act:
            self._deal_remaining_and_showdown()
            return
        dealer = self.players[self.dealer_index]
        self.current_seat = self._next_actor_seat(dealer.seat)
        self._skip_unable_actors()

    def _deal_remaining_and_showdown(self) -> None:
        """
        功能：所有剩余玩家全下时发完公共牌并摊牌。
        参数：无。
        返回：无。
        副作用：补齐公共牌并结算。
        异常：无。
        设计说明：无人可行动时不再制造空转动作。
        """

        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        self.phase_index = 4
        self._run_showdown()

    def _settle_without_showdown(self) -> None:
        """
        功能：在只剩一名未弃牌玩家时直接结算。
        参数：无。
        返回：无。
        副作用：派奖并设置 final_result。
        异常：无。
        设计说明：非摊牌结束不公开任何玩家私牌。
        """

        if self.table_spec is None:
            return
        live = [player for player in self.players if not player.folded]
        winner = live[0] if live else max(self.players, key=lambda item: item.stack)
        winner.stack += self.pot
        winners = [{"agent_id": winner.agent_id, "amount": self.pot, "reason": "last_unfolded"}]
        self.pot = 0
        self.finished = True
        self.current_seat = None
        self.final_result = self._make_result(winners=winners, showdown_visible_cards={})

    def _run_showdown(self) -> None:
        """
        功能：执行摊牌结算。
        参数：无。
        返回：无。
        副作用：根据牌力分配底池并设置 final_result。
        异常：无。
        设计说明：迁入原版 side-pot 分层思路，短码 all-in 时按贡献层派奖。
        """

        contenders = [player for player in self.players if not player.folded]
        ranks = {
            player.agent_id: evaluate_best(player.hole_cards + self.community_cards)
            for player in contenders
        }
        contributions = {player.agent_id: player.total_committed for player in self.players}
        folded = {player.agent_id for player in self.players if player.folded}
        winners = split_side_pots(contributions, folded, ranks)
        for payout in winners:
            player = self.players_by_id[str(payout["agent_id"])]
            player.stack += int(payout["amount"])
        visible = {player.agent_id: list(player.hole_cards) for player in contenders}
        showdown_ranks = {agent_id: rank.class_name for agent_id, rank in ranks.items()}
        self.pot = 0
        self.finished = True
        self.current_seat = None
        self.final_result = self._make_result(
            winners=winners,
            showdown_visible_cards=visible,
            showdown_ranks=showdown_ranks,
        )

    def _make_result(
        self,
        winners: list[dict[str, object]],
        showdown_visible_cards: dict[AgentId, list[str]],
        showdown_ranks: dict[AgentId, str] | None = None,
    ) -> HandResult:
        """
        功能：根据当前内部状态构造 HandResult。
        参数：
            winners：赢家摘要。
            showdown_visible_cards：按规则可见的摊牌牌面。
        返回：HandResult。
        副作用：无。
        异常：无。
        设计说明：奖励按本手初始筹码和结束筹码差计算，支持跨桌总筹码同步。
        """

        if self.table_spec is None:
            raise EnvironmentError("环境尚未 reset")
        final_stacks = {player.agent_id: player.stack for player in self.players}
        rewards = {
            agent_id: final_stacks[agent_id] - self.starting_stacks[agent_id]
            for agent_id in final_stacks
        }
        return HandResult(
            table_id=self.table_spec.table_id,
            hand_id=self.hand_id,
            rewards=rewards,
            final_stacks=final_stacks,
            public_actions=list(self.action_history),
            winners=winners,
            showdown_visible_cards=showdown_visible_cards,
            showdown_ranks=showdown_ranks or {},
        )

    def _skip_unable_actors(self) -> None:
        """
        功能：跳过不能行动的玩家。
        参数：无。
        返回：无。
        副作用：可能更新 current_seat 或直接推进街道。
        异常：无。
        设计说明：盲注后短码全下或街道开始无人可行动时需要兜底。
        """

        if self.current_seat is None or self.finished:
            return
        start = self.current_seat
        while True:
            player = self._player_by_seat(self.current_seat)
            if player and not player.folded and player.stack > 0:
                return
            self.current_seat = self._next_actor_seat(self.current_seat)
            if self.current_seat == start:
                self._advance_street()
                return

    def _next_active_player(self, seat: int) -> _PlayerState:
        """
        功能：返回指定座位之后下一名未出局玩家。
        参数：
            seat：起始座位。
        返回：玩家状态。
        副作用：无。
        异常：找不到玩家时抛出 EnvironmentError。
        设计说明：盲注和按钮规则都需要沿座位环查找。
        """

        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(seat + offset) % len(self.players)]
            if candidate.stack > 0 and not candidate.folded:
                return candidate
        raise EnvironmentError("没有可参与本手的玩家")

    def _next_actor_seat(self, seat: int) -> int | None:
        """
        功能：返回下一名可行动玩家座位。
        参数：
            seat：起始座位。
        返回：座位号或 None。
        副作用：无。
        异常：无。
        设计说明：可行动意味着未弃牌且仍有筹码。
        """

        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(seat + offset) % len(self.players)]
            if not candidate.folded and candidate.stack > 0:
                return candidate.seat
        return None

    def _player_by_seat(self, seat: int | None) -> _PlayerState | None:
        """
        功能：根据座位查找玩家。
        参数：
            seat：座位号。
        返回：玩家状态或 None。
        副作用：无。
        异常：无。
        设计说明：座位数较小，线性查找足够且更易读。
        """

        if seat is None:
            return None
        for player in self.players:
            if player.seat == seat:
                return player
        return None
