"""
模块说明：本模块提供单手牌运行的共享工具。
核心职责：让固定桌、泛化桌和换桌场景共用环境推进、事件记录和轨迹构造逻辑。
输入与输出：输入 Agent、筹码、桌面配置和 seed，输出 HandResult 并更新筹码。
依赖边界：依赖环境协议、Agent 协议和工件管理器，不依赖具体场景配置。
不负责：不生成换桌排程，不计算最终指标。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.audit.decision_trace import build_and_persist_decision_trace
from agentmemeval.core.domain import (
    ActionDecision,
    DecisionEvent,
    HandResult,
    HandTrajectory,
    TableSpec,
)
from agentmemeval.core.poker_position import public_position_label
from agentmemeval.environment.action_risk import build_call_risk
from agentmemeval.environment.decision_facts import build_decision_facts
from agentmemeval.environment.holdem_adapter import HoldemEnvironment
from agentmemeval.environment.raise_sizing import build_raise_sizing_plan
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
    hand_metadata: dict[str, Any] | None = None,
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
    starting_stacks = dict(table_spec.starting_stacks)
    env.reset(table_spec, seed=seed)
    decision_events: dict[str, list[DecisionEvent]] = defaultdict(list)
    action_steps = 0
    while not env.is_hand_finished():
        current = env.current_agent_id()
        if current is None:
            break
        observation = env.current_observation(current)
        call_risk = build_call_risk(observation)
        decision_facts = build_decision_facts(observation)
        action, memory_context, metadata = agents[current].decide(observation)
        decision_index = len(decision_events[current])
        experience_reference = agents[current].experience_reference()
        decision_trace = build_and_persist_decision_trace(
            run_root=artifacts.run_dir,
            observation=observation,
            memory_context=memory_context,
            raise_sizing=build_raise_sizing_plan(
                observation,
                agents[current].raise_sizing_policy,
            ),
            decision_index=decision_index,
            expected_prompt=metadata.get("prompt", {}),
            experience_reference=experience_reference,
        )
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
                "call_risk": call_risk.to_dict(),
                "decision_facts": decision_facts,
                "facing_effective_raise": any(
                    prior.get("phase") == observation.phase
                    and bool(prior.get("effective_raise", False))
                    for prior in observation.action_history
                ),
                "raw_decision": metadata.get("raw_decision", {}),
                "committed_action": action.to_dict(),
                "rewrite_reason": metadata.get("guard_errors", []),
                "risk_gate_applied": False,
                "decision_index": decision_index,
                "observation_sha256": decision_trace["observation"]["sha256"],
                "decision_trace": decision_trace,
                "memory_context": _compact_memory_context(
                    memory_context,
                    experience_reference,
                ),
                **(hand_metadata or {}),
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
            raise RuntimeError(
                f"hand {observation.hand_id} exceeded max_actions={max_actions}; "
                "refusing to force an incomplete board to showdown"
            )
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
            won_amount=sum(
                int(payout.get("amount", 0) or 0)
                for payout in result.winners
                if str(payout.get("agent_id", "")) == agent_id
            ),
            final_community_cards=list(result.community_cards),
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
            "starting_stacks": starting_stacks,
            "rewards": dict(result.rewards),
            "final_stacks": dict(result.final_stacks),
            "winners": list(result.winners),
            "showdown_ranks": dict(result.showdown_ranks),
            "action_count": len(result.public_actions),
            "memory_updated": update_memory,
            **(hand_metadata or {}),
        }
    )
    return result


def _compact_memory_context(
    memory_context: object,
    experience_reference: dict[str, object] | None,
) -> dict[str, object]:
    """Persist facts plus an immutable experience reference, never the full Markdown body."""

    payload = memory_context.to_dict()
    experience = getattr(memory_context, "experience", None)
    if experience is None:
        payload["experience"] = None
        return payload
    reference = dict(experience_reference or {})
    reference.setdefault("version", int(experience.version))
    reference.setdefault("sha256", hashlib.sha256(str(experience.body).encode("utf-8")).hexdigest())
    reference.setdefault("chars", len(str(experience.body)))
    reference.setdefault("path", None)
    payload["experience"] = reference
    return payload


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
    decision_lines: list[str] = []
    for event in event_list:
        observation = event.observation
        self_state = next(
            (player for player in observation.players if player.agent_id == agent_id),
            None,
        )
        amount = (
            f" to {event.committed_action.amount}"
            if event.committed_action.amount is not None
            else ""
        )
        fallback_used = bool(event.llm_metadata.get("fallback_used"))
        intent = "" if fallback_used else event.decision.intent.strip()[:60]
        intent_status = (
            "invalidated_by_guard_fallback"
            if fallback_used
            else "model_self_report_not_ground_truth"
        )
        decision_lines.append(
            "{phase}[position={position},hole={hole},board={board},pot={pot},"
            "to_call={to_call},stack={stack},intent={intent!r},"
            "intent_status={intent_status}]=>{action}{amount}".format(
                phase=observation.phase,
                position=public_position_label(observation),
                hole=" ".join(observation.hole_cards) or "none",
                board=" ".join(observation.community_cards) or "none",
                pot=observation.pot,
                to_call=observation.to_call,
                stack=self_state.stack if self_state is not None else "unknown",
                intent=intent,
                intent_status=intent_status,
                action=event.committed_action.action_type,
                amount=amount,
            )
        )
    decisions = " | ".join(decision_lines) or "无决策"
    won_amount = sum(
        int(payout.get("amount", 0) or 0)
        for payout in result.winners
        if str(payout.get("agent_id", "")) == agent_id
    )
    return (
        f"决策轨迹：{decisions}；"
        f"从底池赢回={won_amount}，"
        f"终局净收益={result.rewards.get(agent_id, 0):+d}，"
        f"终局牌型={result.showdown_ranks.get(agent_id, '未摊牌')}，"
        f"公开摊牌={bool(result.showdown_visible_cards)}。"
    )
