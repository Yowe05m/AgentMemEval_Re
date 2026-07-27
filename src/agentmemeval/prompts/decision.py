"""
模块说明：本模块负责渲染动作决策提示词。
核心职责：把合法可见观察、合法动作和记忆上下文转换为简洁提示。
输入与输出：输入 AgentObservation 与 MemoryContext，输出 system/user prompt 字符串。
依赖边界：只依赖领域对象，不依赖具体 Provider 或环境内部状态。
不负责：不解析 LLM 输出，不校验动作合法性。
"""

from __future__ import annotations

import re

from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord, MemoryContext
from agentmemeval.core.poker_position import (
    active_public_players,
    public_position_label,
)
from agentmemeval.environment.decision_facts import (
    build_decision_facts,
    chips_to_big_blinds,
)
from agentmemeval.environment.raise_sizing import RaiseSizingPlan

PROMPT_TEMPLATE_VERSION = "2026-07-27-v9-chip-bb-unit-explicit"

BASE_SYSTEM_PROMPT = (
    "你是一个德州扑克决策 Agent。目标是在遵守规则的前提下最大化长期期望筹码收益，"
    "不要把保守、探索或任何人格特征机械映射成固定动作。\n"
    "你只能使用用户消息中给出的可见信息。先比较牌力与听牌、位置、底池赔率、"
    "有效筹码和本手公开行动，再从用户列出的合法动作中选择。\n"
    "用户消息中未以 *_big_blinds 命名的数值筹码字段一律是 chips；只有显式的 "
    "*_big_blinds 数值才可表述为 BB，禁止把筹码原值直接改称 BB。\n"
    "用户消息中的确定性牌型、听牌和成本分析由规则引擎计算，优先级高于你的自行识别、"
    "人格偏好和历史记忆；不要声称不存在的对子、两对、同花、顺子或听牌。\n"
    "历史记忆只记录过去发生的状态、动作和结果，不证明旧动作最优；不得把一次历史动作、"
    "模型旧理由或单手回报表述成已验证的期望值规律。\n"
    "经验文档中的动作建议是待验证假设，不含未选择动作的反事实结果；即使文档使用“总是”、"
    "“必然”或“优先”等处方词，也不得机械复制。必须根据当前规则引擎分析和可见局面重新比较合法动作。\n"
    "若经验规则的适用条件与当前局面匹配，且其失效条件均未触发，应把该规则建议作为优先候选之一；"
    "supported 规则的证据权重高于【待验证】规则，但二者都不能覆盖规则引擎事实或合法动作约束。\n"
    "请输出严格 JSON："
    '{"action_type": "fold|check|call|raise", "amount": null 或整数, '
    '"confidence": 0到1, "reason_summary": "一句简短理由", '
    '"intent": "一句简短策略意图"}。\n'
    "action_type 必须来自本次给出的合法动作。raise 的 amount 表示加注到的总额，"
    "必须位于给出的闭区间内；fold、check、call 的 amount 必须是 null。\n"
    "reason_summary 必须与最终 action_type 一致，只概括关键可见证据。\n"
    "intent 与 reason_summary 含义不同：intent 只写你希望通过本次动作实现的短期策略目标"
    "或传达的桌面形象，例如‘用价值线争取跟注’、‘代表强牌施压’或‘保持紧凶形象’；"
    "不得复述牌张、牌力判断、长推理或任何对手标签，最多 60 个汉字。\n"
    "接近最大 raise 金额通常意味着全下；不要仅因探索性、位置优势或对手可能诈唬而用边缘牌全下。\n"
    "不要输出长思维链，不要假设你看到了对手私有手牌。"
)


def render_system_prompt(context: MemoryContext) -> str:
    """
    功能：渲染系统提示词。
    参数：
        context：记忆上下文。
    返回：系统提示词。
    副作用：无。
    异常：无。
    设计说明：人格决策提示在系统层注入，普通记忆证据在用户层注入。
    """

    persona_prompt = context.persona.get("decision_prompt")
    if persona_prompt:
        return f"{context.persona.get('description', '')}\n{persona_prompt}\n\n{BASE_SYSTEM_PROMPT}"
    return BASE_SYSTEM_PROMPT


def render_user_prompt(
    observation: AgentObservation,
    context: MemoryContext,
    raise_sizing: RaiseSizingPlan | None = None,
) -> str:
    """
    功能：渲染用户提示词。
    参数：
        observation：合法可见观察。
        context：记忆上下文。
    返回：用户提示词。
    副作用：无。
    异常：无。
    设计说明：提示词只使用 AgentObservation，不访问环境私有状态。
    """

    player_labels = _relative_player_labels(observation)
    lines = [
        "## 当前局面",
        f"- 阶段：{observation.phase}",
        f"- 我的座位：{observation.seat}",
        f"- 我的相对位置：{public_position_label(observation)}",
        "- 公开位置锚点："
        f"BTN seat={observation.dealer_seat}, "
        f"SB seat={observation.small_blind_seat}, "
        f"BB seat={observation.big_blind_seat}",
        "- 盲注金额："
        f"SB={_render_blind_amount(observation.small_blind_amount)}, "
        f"BB={_render_blind_amount(observation.big_blind_amount)}",
        f"- 我的手牌：{' '.join(observation.hole_cards)}",
        f"- 公共牌：{' '.join(observation.community_cards) or '无'}",
        f"- 底池：{_render_chip_amount(observation.pot, observation.big_blind_amount)}",
        "- 当前下注线："
        f"{_render_chip_amount(observation.current_bet, observation.big_blind_amount)}",
        f"- 需要补齐：{_render_chip_amount(observation.to_call, observation.big_blind_amount)}",
        "",
        "## 规则引擎确定性分析（必须以此为准）",
        *_render_deterministic_analysis(observation),
        "",
        "## 玩家公开状态",
    ]
    for player in observation.players:
        if player.busted:
            continue
        lines.append(
            f"- {player_labels.get(player.agent_id, '某对手')}: seat={player.seat}, "
            f"stack={_render_chip_amount(player.stack, observation.big_blind_amount)}, "
            "bet="
            f"{_render_chip_amount(player.current_bet, observation.big_blind_amount)}, "
            f"folded={player.folded}, all_in={player.all_in}"
        )
    lines.extend(["", "## 合法动作"])
    for action in observation.legal_actions.actions:
        if action.action_type == "raise":
            lines.append(
                "- raise: amount（raise-to 总额）范围 "
                f"[{_render_chip_amount(action.min_amount, observation.big_blind_amount)}, "
                f"{_render_chip_amount(action.max_amount, observation.big_blind_amount)}], "
                f"reopens={action.reopens}"
            )
        else:
            lines.append(f"- {action.action_type}")
    if raise_sizing is not None and raise_sizing.allowed_amounts is not None:
        amounts = ", ".join(
            _render_chip_amount(amount, observation.big_blind_amount)
            for amount in raise_sizing.allowed_amounts
        )
        lines.append(
            f"- 本地离散 raise-to 候选：[{amounts}]；选择 raise 时 amount 只能取其中一个。"
        )
    lines.extend(["", "## 近期公开行动"])
    if observation.action_history:
        for event in observation.action_history[-12:]:
            amount = event.get("amount")
            rendered_amount = (
                _render_chip_amount(amount, observation.big_blind_amount)
                if type(amount) is int
                else ""
            )
            lines.append(
                f"- {event.get('phase')} "
                f"{player_labels.get(str(event.get('agent_id')), '某对手')} "
                f"{event.get('action_type')} {rendered_amount}".rstrip()
            )
    else:
        lines.append("- 无")
    lines.extend(["", render_memory_section(context)])
    return "\n".join(lines)


def render_facts_section(context: MemoryContext) -> str:
    """Render exactly the factual-memory subsection injected into a decision prompt."""

    return _render_facts(context.facts)


def render_experience_section(context: MemoryContext) -> str:
    """Render exactly the experience subsection injected into a decision prompt."""

    if context.experience is None:
        return "### 经验文档\n无"
    return f"### 经验文档\n{context.experience.body}"


def render_memory_section(context: MemoryContext) -> str:
    """Render the complete memory section through the same path as the live prompt."""

    sections = [
        "## 记忆",
        render_facts_section(context),
        render_experience_section(context),
    ]
    if context.persona:
        sections.extend(
            [
                "### 人格提示",
                str(context.persona.get("description", "")),
            ]
        )
    return "\n".join(sections)


def _render_deterministic_analysis(observation: AgentObservation) -> list[str]:
    """Render authoritative made-hand, draw, and stack-risk facts for small models."""

    facts = build_decision_facts(observation)
    draw = facts["draw"]
    hero_draw = draw["hero_draw"]
    board_threat = draw["board_completion_threat"]
    call = facts["call"]
    best_cards = " ".join(facts["best_cards"])
    lines = [f"- 当前已成牌：{facts['made_hand_class']}；规则引擎最佳牌：{best_cards}。"]
    if observation.phase == "preflop":
        lines.append(
            "- 听牌状态：翻牌前不计算翻牌后 outs；当前 outs=0 不代表起手牌没有继续价值，"
            "应依据起手牌结构、位置、入池成本和有效筹码判断。"
        )
    elif observation.phase == "river":
        lines.append("- 听牌状态：河牌已结束，没有未来补牌；outs=0。")
    else:
        lines.append(
            "- 我方听牌：flush_draw={flush_draw}，straight_draw={straight_draw}，"
            "去重名义 outs={outs}，待发公共牌={cards_to_come}；"
            "这些是我方底牌参与的一张牌型补全张，不等同于 clean/winning outs 或胜率。".format(
                **hero_draw,
                cards_to_come=draw["cards_to_come"],
            )
        )
        if int(board_threat["outs"]) > 0:
            lines.append(
                "- 公共牌补全威胁：flush={flush_draw}，straight={straight_draw}，"
                "名义补全张={outs}；这是共享牌面变化，不能冒充我方听牌。".format(**board_threat)
            )
    call_cost = int(call["call_cost"])
    if call_cost:
        rendered_call_cost = _render_chip_amount(
            call_cost, observation.big_blind_amount
        )
        rendered_to_call = _render_chip_amount(
            observation.to_call, observation.big_blind_amount
        )
        rendered_pot_before_call = _render_chip_amount(
            int(call["contestable_pot_before_call"]), observation.big_blind_amount
        )
        rendered_pot_after_call = _render_chip_amount(
            int(call["contestable_pot_after_call"]), observation.big_blind_amount
        )
        lines.append(
            "- 跟注成本：实际最多投入 "
            f"{rendered_call_cost}"
            "（显示需补齐 "
            f"{rendered_to_call}）；"
            "跟注前可争夺底池 "
            f"{rendered_pot_before_call}，"
            "跟注后可争夺底池 "
            f"{rendered_pot_after_call}；"
            f"需要约 {float(call['required_equity']):.1%} 胜率；"
            f"占剩余筹码 {float(call['stack_fraction']):.1%}；"
            f"风险标签={call['risk_label']}，all_in={call['is_all_in']}；"
            f"{'这是全下跟注。' if call['is_all_in'] else '跟注后仍有剩余筹码。'}"
        )
    else:
        lines.append("- 跟注成本：0，可以过牌时不要虚构跟注成本。")
    lines.append(
        f"- 桌面风险：可争夺底池口径 SPR={float(facts['spr']):.2f}，"
        f"跟注后 SPR={float(facts['spr_after_call']):.2f}；"
        "策略有效筹码="
        f"{_render_chip_amount(int(facts['effective_stack']), observation.big_blind_amount)}"
        "（各对手范围 "
        f"{_render_chip_amount(int(facts['effective_stack_min']), observation.big_blind_amount)}"
        " 至 "
        f"{_render_chip_amount(int(facts['effective_stack_max']), observation.big_blind_amount)}"
        "，取最大可争夺额），"
        f"底池仍有效玩家={facts['multiway_players']}（对手 {facts['active_opponents']}）。"
    )

    raise_rule = observation.legal_actions.rule_for("raise")
    if raise_rule and raise_rule.min_amount is not None and raise_rule.max_amount is not None:
        self_state = next(
            player for player in observation.players if player.agent_id == observation.agent_id
        )
        min_cost = min(
            self_state.stack,
            max(0, raise_rule.min_amount - self_state.current_bet),
        )
        max_cost = min(
            self_state.stack,
            max(0, raise_rule.max_amount - self_state.current_bet),
        )
        lines.append(
            "- 加注风险：最小额外投入 "
            f"{_render_chip_amount(min_cost, observation.big_blind_amount)}，"
            "最大额外投入 "
            f"{_render_chip_amount(max_cost, observation.big_blind_amount)}；"
            f"最大 raise {'等同全下' if max_cost >= self_state.stack else '不是全下'}。"
        )
    return lines


def _render_blind_amount(amount: int | None) -> str:
    """Render an optional blind amount without inventing a blind structure."""

    if type(amount) is not int or amount <= 0:
        return "未提供"
    return f"{amount} chips"


def _render_chip_amount(amount: int, big_blind_amount: int | None) -> str:
    """Render chips first and append a deterministic BB equivalent when available."""

    rendered = f"{amount} chips"
    big_blinds = chips_to_big_blinds(amount, big_blind_amount)
    if big_blinds is None:
        return rendered
    formatted_big_blinds = f"{big_blinds:.4f}".rstrip("0").rstrip(".")
    return f"{rendered}（={formatted_big_blinds} BB）"


def _render_facts(facts: list[FactualMemoryRecord]) -> str:
    """
    功能：渲染事实证据列表。
    参数：
        facts：事实记录。
    返回：事实文本。
    副作用：无。
    异常：无。
    设计说明：只渲染事实摘要、动作和回报，不包含原始长回复。
    """

    if not facts:
        return "### 事实证据\n无"
    lines = [
        "### 事实证据",
        "以下仅为历史观察、动作和结果；不是最优策略标签，也不是期望值估计。",
    ]
    for fact in facts:
        fact_label = fact.record_id.removeprefix(f"{fact.agent_id}-")
        lines.append(
            f"- {fact_label}: {_sanitize_agent_ids(fact.state_summary, fact.agent_id)}; "
            f"{_sanitize_agent_ids(fact.action_summary, fact.agent_id)}; "
            f"final_reward={fact.final_reward}"
        )
    return "\n".join(lines)


def _relative_player_labels(observation: AgentObservation) -> dict[str, str]:
    """按相对座位生成每手匿名标签，避免跨手绑定稳定 Agent ID。"""

    labels = {observation.agent_id: "我"}
    active_players = active_public_players(observation)
    active_seats = [player.seat for player in active_players]
    if observation.seat in active_seats:
        hero_index = active_seats.index(observation.seat)
        active_players = active_players[hero_index + 1 :] + active_players[:hero_index]
    opponents = [player for player in active_players if player.agent_id != observation.agent_id]
    for index, player in enumerate(opponents, start=1):
        labels[player.agent_id] = f"下家{index}"
    return labels


_AGENT_ID_PATTERN = re.compile(r"\b(?:agent|heldout)_\d+\b", re.IGNORECASE)


def _sanitize_agent_ids(text: str, owner_id: str) -> str:
    """从长期记忆文本中移除稳定玩家 ID，只保留自我/某对手语义。"""

    def replace(match: re.Match[str]) -> str:
        return "我" if match.group(0).lower() == owner_id.lower() else "某对手"

    return _AGENT_ID_PATTERN.sub(replace, str(text))
