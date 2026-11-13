from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.path1_v1.core.domain import (
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
from agentmemeval.path1_v1.core.errors import EnvironmentError
from agentmemeval.path1_v1.core.seeds import make_rng
from agentmemeval.path1_v1.environment.action_guard import ActionGuard
from agentmemeval.path1_v1.environment.hand_evaluator import evaluate_best, split_side_pots

RANKS = "23456789TJQKA"
SUITS = "cdhs"
PHASES = ("preflop", "flop", "turn", "river")


@dataclass(slots=True)
class _PlayerState:
    agent_id: AgentId
    seat: int
    stack: int
    hole_cards: list[str] = field(default_factory=list)
    current_bet: int = 0
    total_committed: int = 0
    folded: bool = False

    @property
    def all_in(self) -> bool:

        return self.stack <= 0 and not self.folded


class HoldemEnvironment:
    def __init__(self) -> None:

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
        self.small_blind_agent_id: AgentId | None = None
        self.big_blind_agent_id: AgentId | None = None
        self.acted: set[int] = set()
        self.raise_count = 0
        self.last_raise_size = 0
        self.action_history: list[dict[str, object]] = []
        self.finished = True
        self.hand_id = "uninitialized"
        self.starting_stacks: dict[AgentId, int] = {}
        self.final_result: HandResult | None = None
        self._guard = ActionGuard()
        self._hand_counter = 0

    def reset(self, table_spec: TableSpec, seed: int) -> None:

        if len(table_spec.agent_ids) < 2:
            raise EnvironmentError("a table requires at least two agents")
        if table_spec.small_blind <= 0 or table_spec.big_blind < table_spec.small_blind:
            raise EnvironmentError("invalid blinds: big_blind must be at least small_blind")
        self.table_spec = table_spec
        self.seed = seed
        self.rng = make_rng(seed, table_spec.table_id)
        self._hand_counter += 1
        hand_number = max(1, int(table_spec.hand_number))
        self.hand_id = f"{table_spec.table_id}-h{hand_number}-{seed}"
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
        self.last_raise_size = table_spec.big_blind
        self.action_history = []
        self.finished = False
        self.final_result = None
        self.dealer_index = int(table_spec.dealer_index) % len(self.players)
        self.small_blind_agent_id = None
        self.big_blind_agent_id = None
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
        configured_dealer = self.players[self.dealer_index]
        if configured_dealer.stack <= 0 or configured_dealer.folded:
            self.dealer_index = self._next_active_player(configured_dealer.seat).seat
        self._post_blinds()
        self.current_seat = self._first_preflop_seat()
        self._skip_unable_actors()

    def current_agent_id(self) -> AgentId | None:

        if self.current_seat is None:
            return None
        player = self._player_by_seat(self.current_seat)
        return player.agent_id if player else None

    def current_observation(self, agent_id: AgentId) -> AgentObservation:

        if self.table_spec is None:
            raise EnvironmentError("environment has not been reset")
        player = self.players_by_id.get(agent_id)
        if player is None:
            raise EnvironmentError(f"unknown agent: {agent_id}")
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
            dealer_seat=self.players[self.dealer_index].seat,
            small_blind_seat=next(
                (item.seat for item in self.players if item.agent_id == self.small_blind_agent_id),
                None,
            ),
            big_blind_seat=next(
                (item.seat for item in self.players if item.agent_id == self.big_blind_agent_id),
                None,
            ),
            small_blind_amount=self.table_spec.small_blind,
            big_blind_amount=self.table_spec.big_blind,
        )

    def legal_actions(self, agent_id: AgentId) -> LegalActionSet:

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
        already_acted_facing_short_allin = to_call > 0 and player.seat in self.acted
        can_raise_again = (
            self.table_spec.max_raises_per_street <= 0
            or self.raise_count < self.table_spec.max_raises_per_street
        )
        if can_raise_again and player.stack > to_call and not already_acted_facing_short_allin:
            min_to = self.current_bet + self.last_raise_size
            max_to = player.current_bet + player.stack
            if max_to >= min_to:
                actions.append(LegalAction("raise", min_amount=min_to, max_amount=max_to))
            elif max_to > self.current_bet:
                actions.append(
                    LegalAction("raise", min_amount=max_to, max_amount=max_to, reopens=False)
                )
        return LegalActionSet(actions=actions)

    def step(self, agent_id: AgentId, action: ActionDecision) -> StepResult:

        if self.finished:
            raise EnvironmentError("the hand is finished; no further action is allowed")
        if self.current_agent_id() != agent_id:
            raise EnvironmentError(f"{agent_id} is not the current actor")
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
                raise EnvironmentError("raise action is missing amount")
            committed = min(max(0, target - player.current_bet), player.stack)
            old_bet = self.current_bet
            previous_last_raise = self.last_raise_size
            self._commit(player, committed)
            if player.current_bet > old_bet:
                raise_increment = player.current_bet - old_bet
                self.current_bet = player.current_bet
                rule = legal.rule_for("raise")
                effective_raise = bool(
                    rule and rule.reopens and raise_increment >= previous_last_raise
                )
                if effective_raise:
                    self.last_raise_size = raise_increment
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
            "last_raise_size": self.last_raise_size,
        }
        self.action_history.append(event)
        self._advance_after_action(player.seat)
        return StepResult(event=event, hand_finished=self.finished)

    def is_hand_finished(self) -> bool:

        return self.finished

    def finalize_hand(self) -> HandResult:

        if self.final_result is None:
            if not self.finished:
                self._run_showdown()
            else:
                self._settle_without_showdown()
        assert self.final_result is not None
        return self.final_result

    @property
    def _phase(self) -> str:

        if self.finished:
            return "hand_over"
        return PHASES[min(self.phase_index, len(PHASES) - 1)]

    def _post_blinds(self) -> None:

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
        self.small_blind_agent_id = sb.agent_id
        self.big_blind_agent_id = bb.agent_id
        self.current_bet = max(sb.current_bet, bb.current_bet)

    def _first_preflop_seat(self) -> int | None:

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

        real = max(0, min(amount, player.stack))
        player.stack -= real
        player.current_bet += real
        player.total_committed += real
        self.pot += real

    def _advance_after_action(self, previous_seat: int) -> None:

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

        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        self.raise_count = 0
        self.last_raise_size = self.table_spec.big_blind if self.table_spec else 0
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

        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        self.phase_index = 4
        self._run_showdown()

    def _settle_without_showdown(self) -> None:

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

        contenders = [player for player in self.players if not player.folded]
        ranks = {
            player.agent_id: evaluate_best(player.hole_cards + self.community_cards)
            for player in contenders
        }
        contributions = {player.agent_id: player.total_committed for player in self.players}
        folded = {player.agent_id for player in self.players if player.folded}
        odd_chip_order = [
            self.players[(self.dealer_index + offset) % len(self.players)].agent_id
            for offset in range(1, len(self.players) + 1)
        ]
        winners = split_side_pots(
            contributions,
            folded,
            ranks,
            odd_chip_order=odd_chip_order,
        )
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

        if self.table_spec is None:
            raise EnvironmentError("environment has not been reset")
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
            community_cards=list(self.community_cards),
        )

    def _skip_unable_actors(self) -> None:

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

        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(seat + offset) % len(self.players)]
            if candidate.stack > 0 and not candidate.folded:
                return candidate
        raise EnvironmentError("no player is eligible to enter the hand")

    def _next_actor_seat(self, seat: int) -> int | None:

        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(seat + offset) % len(self.players)]
            if not candidate.folded and candidate.stack > 0:
                return candidate.seat
        return None

    def _player_by_seat(self, seat: int | None) -> _PlayerState | None:

        if seat is None:
            return None
        for player in self.players:
            if player.seat == seat:
                return player
        return None
