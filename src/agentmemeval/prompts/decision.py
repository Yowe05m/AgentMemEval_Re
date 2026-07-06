"""
模块说明：本模块负责渲染动作决策提示词。
核心职责：把合法可见观察、合法动作和记忆上下文转换为简洁提示。
输入与输出：输入 AgentObservation 与 MemoryContext，输出 system/user prompt 字符串。
依赖边界：只依赖领域对象，不依赖具体 Provider 或环境内部状态。
不负责：不解析 LLM 输出，不校验动作合法性。
"""

from __future__ import annotations

from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord, MemoryContext

BASE_SYSTEM_PROMPT = (
    "你是一个德州扑克 Agent。你只能使用用户消息中给出的可见信息。\n"
    "请输出严格 JSON："
    '{"action_type": "fold|check|call|raise", "amount": null 或整数, '
    '"confidence": 0到1, "reason_summary": "一句简短理由"}。\n'
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


def render_user_prompt(observation: AgentObservation, context: MemoryContext) -> str:
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

    lines = [
        "## 当前局面",
        f"- 桌号：{observation.table_id}",
        f"- 手牌：{observation.hand_id}",
        f"- 阶段：{observation.phase}",
        f"- 我的座位：{observation.seat}",
        f"- 我的手牌：{' '.join(observation.hole_cards)}",
        f"- 公共牌：{' '.join(observation.community_cards) or '无'}",
        f"- 底池：{observation.pot}",
        f"- 当前下注线：{observation.current_bet}",
        f"- 需要补齐：{observation.to_call}",
        "",
        "## 玩家公开状态",
    ]
    for player in observation.players:
        lines.append(
            f"- {player.agent_id}: seat={player.seat}, stack={player.stack}, "
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
    lines.extend(["", "## 近期公开行动"])
    if observation.action_history:
        for event in observation.action_history[-12:]:
            lines.append(
                f"- {event.get('phase')} {event.get('agent_id')} "
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
        lines.append(
            f"- {fact.record_id}: {fact.state_summary}; {fact.action_summary}; "
            f"final_reward={fact.final_reward}"
        )
    return "\n".join(lines)
