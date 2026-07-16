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
from agentmemeval.environment.decision_facts import build_decision_facts
from agentmemeval.environment.raise_sizing import RaiseSizingPlan

PROMPT_TEMPLATE_VERSION = "2026-07-15-v4-authoritative-facts"

BASE_SYSTEM_PROMPT = (
    "你是一个德州扑克决策 Agent。目标是在遵守规则的前提下最大化长期期望筹码收益，"
    "不要把保守、探索或任何人格特征机械映射成固定动作。\n"
    "你只能使用用户消息中给出的可见信息。先比较牌力与听牌、位置、底池赔率、"
    "有效筹码和本手公开行动，再从用户列出的合法动作中选择。\n"
    "用户消息中的确定性牌型、听牌和成本分析由规则引擎计算，优先级高于你的自行识别、"
    "人格偏好和历史记忆；不要声称不存在的对子、两对、同花、顺子或听牌。\n"
    "请输出严格 JSON："
    '{"action_type": "fold|check|call|raise", "amount": null 或整数, '
    '"confidence": 0到1, "reason_summary": "一句简短理由"}。\n'
    "action_type 必须来自本次给出的合法动作。raise 的 amount 表示加注到的总额，"
    "必须位于给出的闭区间内；fold、check、call 的 amount 必须是 null。\n"
    "reason_summary 必须与最终 action_type 一致，只概括关键可见证据。\n"
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
        f"- 我的手牌：{' '.join(observation.hole_cards)}",
        f"- 公共牌：{' '.join(observation.community_cards) or '无'}",
        f"- 底池：{observation.pot}",
        f"- 当前下注线：{observation.current_bet}",
        f"- 需要补齐：{observation.to_call}",
        "",
        "## 规则引擎确定性分析（必须以此为准）",
        *_render_deterministic_analysis(observation),
        "",
        "## 玩家公开状态",
    ]
    for player in observation.players:
        lines.append(
            f"- {player_labels.get(player.agent_id, '某对手')}: seat={player.seat}, "
            f"stack={player.stack}, "
            f"bet={player.current_bet}, folded={player.folded}, all_in={player.all_in}"
        )
    lines.extend(["", "## 合法动作"])
    for action in observation.legal_actions.actions:
        if action.action_type == "raise":
            lines.append(
                f"- raise: amount 范围 [{action.min_amount}, {action.max_amount}], "
                f"reopens={action.reopens}"
            )
        else:
            lines.append(f"- {action.action_type}")
    if raise_sizing is not None and raise_sizing.allowed_amounts is not None:
        amounts = ", ".join(str(amount) for amount in raise_sizing.allowed_amounts)
        lines.append(
            f"- 本地离散 raise-to 候选：[{amounts}]；选择 raise 时 amount 只能取其中一个。"
        )
    lines.extend(["", "## 近期公开行动"])
    if observation.action_history:
        for event in observation.action_history[-12:]:
            lines.append(
                f"- {event.get('phase')} "
                f"{player_labels.get(str(event.get('agent_id')), '某对手')} "
                f"{event.get('action_type')} {event.get('amount') or ''}".rstrip()
            )
    else:
        lines.append("- 无")
    lines.extend(["", "## 记忆"])
    lines.append(_render_facts(context.facts))
    if context.experience is not None:
        lines.append("### 经验文档")
        lines.append(context.experience.body)
    else:
        lines.append("### 经验文档\n无")
    if context.persona:
        lines.append("### 人格提示")
        lines.append(str(context.persona.get("description", "")))
    return "\n".join(lines)


def _render_deterministic_analysis(observation: AgentObservation) -> list[str]:
    """Render authoritative made-hand, draw, and stack-risk facts for small models."""

    facts = build_decision_facts(observation)
    draw = facts["draw"]
    call = facts["call"]
    best_cards = " ".join(facts["best_cards"])
    lines = [f"- 当前已成牌：{facts['made_hand_class']}；规则引擎最佳牌：{best_cards}。"]
    if observation.phase == "river":
        lines.append("- 听牌状态：河牌已结束，没有未来补牌；outs=0。")
    else:
        lines.append(
            "- 听牌状态：flush_draw={flush_draw}，straight_draw={straight_draw}，"
            "去重 outs={outs}，待发公共牌={cards_to_come}。".format(**draw)
        )
    call_cost = int(call["call_cost"])
    if call_cost:
        lines.append(
            f"- 跟注成本：实际最多投入 {call_cost}（显示需补齐 {observation.to_call}）；"
            f"需要约 {float(call['required_equity']):.1%} 胜率；"
            f"占剩余筹码 {float(call['stack_fraction']):.1%}；"
            f"风险标签={call['risk_label']}，all_in={call['is_all_in']}；"
            f"{'这是全下跟注。' if call['is_all_in'] else '跟注后仍有剩余筹码。'}"
        )
    else:
        lines.append("- 跟注成本：0，可以过牌时不要虚构跟注成本。")
    lines.append(
        f"- 桌面风险：SPR={float(facts['spr']):.2f}，有效筹码={facts['effective_stack']}，"
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
            f"- 加注风险：最小额外投入 {min_cost}，最大额外投入 {max_cost}；"
            f"最大 raise {'等同全下' if max_cost >= self_state.stack else '不是全下'}。"
        )
    return lines


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
    lines = ["### 事实证据"]
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
    opponents = sorted(
        (player for player in observation.players if player.agent_id != observation.agent_id),
        key=lambda player: (player.seat - observation.seat) % max(1, len(observation.players)),
    )
    for index, player in enumerate(opponents, start=1):
        labels[player.agent_id] = f"下家{index}"
    return labels


_AGENT_ID_PATTERN = re.compile(r"\b(?:agent|heldout)_\d+\b", re.IGNORECASE)


def _sanitize_agent_ids(text: str, owner_id: str) -> str:
    """从长期记忆文本中移除稳定玩家 ID，只保留自我/某对手语义。"""

    def replace(match: re.Match[str]) -> str:
        return "我" if match.group(0).lower() == owner_id.lower() else "某对手"

    return _AGENT_ID_PATTERN.sub(replace, str(text))
