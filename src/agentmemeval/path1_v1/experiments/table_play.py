from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agentmemeval.path1_v1.agents.base import LLMDecisionAgent
from agentmemeval.path1_v1.audit.decision_trace import build_and_persist_decision_trace
from agentmemeval.path1_v1.core.domain import (
    ActionDecision,
    DecisionEvent,
    HandResult,
    HandTrajectory,
    TableSpec,
)
from agentmemeval.path1_v1.core.poker_position import public_position_label
from agentmemeval.path1_v1.environment.action_risk import build_call_risk
from agentmemeval.path1_v1.environment.decision_facts import build_decision_facts
from agentmemeval.path1_v1.environment.holdem_adapter import HoldemEnvironment
from agentmemeval.path1_v1.environment.raise_sizing import build_raise_sizing_plan
from agentmemeval.path1_v1.storage.artifacts import ArtifactManager


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
    on_settled: Callable[[HandResult], None] | None = None,
    direct_rng_seed: int | None = None,
) -> HandResult:

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
    if direct_rng_seed is None:
        env.reset(table_spec, seed=seed)
    else:
        env.reset(table_spec, seed=seed, direct_rng_seed=direct_rng_seed)
    # Audit the actual exogenous deal before any policy can affect the trajectory.
    # Private cards remain hashed audit data and never enter another player's prompt.
    deal_sha256 = hashlib.sha256(
        json.dumps(
            {
                "hole_cards_by_seat": [list(player.hole_cards) for player in env.players],
                "remaining_deck": list(env.deck),
                "dealer_index": env.dealer_index,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
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
                "deal_sha256": deal_sha256,
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
    # Path1 persists cash flow before a revision can fail. The callback must not
    # mutate agents or the result; an archival error is fatal, never a fallback.
    if on_settled is not None:
        on_settled(result)
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
            "deal_sha256": deal_sha256,
            **({"actual_rng_seed": env.actual_rng_seed,
                "actual_rng_algorithm": env.actual_rng_algorithm}
               if direct_rng_seed is not None else {}),
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
        intent = "" if fallback_used else event.decision.intent.strip()[:40]
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
    decisions = " | ".join(decision_lines) or "no decisions"
    won_amount = sum(
        int(payout.get("amount", 0) or 0)
        for payout in result.winners
        if str(payout.get("agent_id", "")) == agent_id
    )
    return (
        f"Decision trace: {decisions}; "
        f"pot returned={won_amount}; "
        f"net reward={result.rewards.get(agent_id, 0):+d}; "
        f"showdown rank={result.showdown_ranks.get(agent_id, 'not shown')}; "
        f"public showdown={bool(result.showdown_visible_cards)}."
    )
