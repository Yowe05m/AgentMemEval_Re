"""
模块说明：本模块提供单手牌运行的共享工具。
核心职责：让固定桌、泛化桌和换桌场景共用环境推进、事件记录和轨迹构造逻辑。
输入与输出：输入 Agent、筹码、桌面配置和 seed，输出 HandResult 并更新筹码。
依赖边界：依赖环境协议、Agent 协议和工件管理器，不依赖具体场景配置。
不负责：不生成换桌排程，不计算最终指标。
"""

from __future__ import annotations

from collections import defaultdict

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.core.domain import (
    ActionDecision,
    DecisionEvent,
    HandResult,
    HandTrajectory,
    TableSpec,
)
from agentmemeval.environment.holdem_adapter import HoldemEnvironment
from agentmemeval.storage.artifacts import ArtifactManager


def run_single_hand(
    agents: dict[str, LLMDecisionAgent],
    table_id: str,
    agent_ids: list[str],
    stacks: dict[str, int],
    seed: int,
    stage: str,
    small_blind: int,
    big_blind: int,
    max_raises_per_street: int,
    update_memory: bool,
    artifacts: ArtifactManager,
    dealer_index: int = 0,
    hand_number: int = 1,
    max_actions: int = 200,
) -> HandResult:
    """
    功能：运行一手牌并记录标准工件。
    参数：
        agents：Agent 查找表。
        table_id：桌号。
        agent_ids：入座 Agent。
        stacks：全局筹码表，函数会更新参与者筹码。
        seed：本手 seed。
        stage：train/test/rotation 等阶段。
        small_blind：小盲。
        big_blind：大盲。
        max_raises_per_street：每街最大加注次数。
        update_memory：是否把手牌轨迹写入记忆。
        artifacts：工件管理器。
        max_actions：单手最大动作数。
    返回：HandResult。
    副作用：推进环境、写事件与手牌摘要、更新 Agent 记忆和 stacks。
    异常：环境或 Provider 错误向上抛出。
    设计说明：所有场景共用该函数，保证日志和记忆更新口径一致。
    """

    table_spec = TableSpec(
        table_id=table_id,
        agent_ids=agent_ids,
        starting_stacks={agent_id: max(0, int(stacks[agent_id])) for agent_id in agent_ids},
        small_blind=small_blind,
        big_blind=big_blind,
        max_raises_per_street=max_raises_per_street,
        dealer_index=dealer_index,
        hand_number=hand_number,
    )
    env = HoldemEnvironment()
    env.reset(table_spec, seed=seed)
    decision_events: dict[str, list[DecisionEvent]] = defaultdict(list)
    action_steps = 0
    while not env.is_hand_finished():
        current = env.current_agent_id()
        if current is None:
            break
        observation = env.current_observation(current)
        action, memory_context, metadata = agents[current].decide(observation)
        raw_payload = metadata.get("raw_decision", action.to_dict())
        raw_decision = ActionDecision(**raw_payload) if isinstance(raw_payload, dict) else action
        step_result = env.step(current, action)
        event = DecisionEvent(
            agent_id=current,
            table_id=table_id,
            hand_id=observation.hand_id,
            observation=observation,
            decision=raw_decision,
            committed_action=action,
            memory_context=memory_context,
            llm_metadata=metadata,
        )
        agents[current].observe_decision(event)
        decision_events[current].append(event)
        artifacts.log_event(
            {
                **step_result.event,
                "stage": stage,
                "guard_repaired": metadata.get("guard_repaired", False),
                "guard_errors": metadata.get("guard_errors", []),
                "fallback_used": metadata.get("fallback_used", False),
                "llm": metadata.get("llm", {}),
                "prompt": metadata.get("prompt", {}),
                "raise_sizing": metadata.get("raise_sizing", {}),
                "raw_decision": metadata.get("raw_decision", {}),
                "memory_context": memory_context.to_dict(),
            }
        )
        action_steps += 1
        if action_steps >= max_actions:
            artifacts.log_event(
                {
                    "event": "max_actions_reached",
                    "stage": stage,
                    "table_id": table_id,
                    "hand_id": observation.hand_id,
                    "max_actions": max_actions,
                }
            )
            break
    result = env.finalize_hand()
    stacks.update(result.final_stacks)
    for agent_id in agent_ids:
        trajectory = HandTrajectory(
            agent_id=agent_id,
            table_id=table_id,
            hand_id=result.hand_id,
            decision_events=list(decision_events.get(agent_id, [])),
            public_actions=result.public_actions,
            final_reward=result.rewards.get(agent_id, 0),
            final_stack=result.final_stacks.get(agent_id, stacks.get(agent_id, 0)),
            showdown_visible_cards=result.showdown_visible_cards,
            summary=_summarize_trajectory(agent_id, result, decision_events.get(agent_id, [])),
        )
        if update_memory:
            agents[agent_id].observe_hand_finished(trajectory)
    artifacts.log_hand(
        {
            "event": "hand_summary",
            "stage": stage,
            "table_id": table_id,
            "hand_id": result.hand_id,
            "seed": seed,
            "hand_number": hand_number,
            "dealer_index": env.dealer_index,
            "dealer_agent_id": agent_ids[env.dealer_index],
            "small_blind_agent_id": env.small_blind_agent_id,
            "big_blind_agent_id": env.big_blind_agent_id,
            "agent_ids": list(agent_ids),
            "rewards": dict(result.rewards),
            "final_stacks": dict(result.final_stacks),
            "winners": list(result.winners),
            "showdown_ranks": dict(result.showdown_ranks),
            "action_count": len(result.public_actions),
            "memory_updated": update_memory,
        }
    )
    return result


def _summarize_trajectory(
    agent_id: str,
    result: HandResult,
    events: list[DecisionEvent] | None,
) -> str:
    """
    功能：为记忆更新生成短轨迹摘要。
    参数：
        agent_id：Agent 标识。
        result：手牌结果。
        events：该 Agent 的决策事件。
    返回：摘要文本。
    副作用：无。
    异常：无。
    设计说明：摘要不包含对手未公开私牌，只描述动作和本手回报。
    """

    event_list = events or []
    actions = ", ".join(
        f"{event.observation.phase}:{event.committed_action.action_type}"
        for event in event_list
    ) or "无决策"
    return (
        f"{agent_id} 在 {result.hand_id} 的动作为 {actions}；"
        f"最终净收益 {result.rewards.get(agent_id, 0)}；"
        f"摊牌牌型 {result.showdown_ranks.get(agent_id, '未摊牌')}。"
    )
