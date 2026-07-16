"""Deterministic call-risk and made-hand diagnostics for experiment audit."""

from __future__ import annotations

from dataclasses import dataclass

from agentmemeval.core.domain import AgentObservation
from agentmemeval.environment.decision_facts import build_decision_facts


@dataclass(frozen=True, slots=True)
class CallRisk:
    stack_before: int
    call_cost: int
    stack_fraction: float
    required_equity: float
    is_all_in: bool
    made_hand_class: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stack_before": self.stack_before,
            "call_cost": self.call_cost,
            "stack_fraction": self.stack_fraction,
            "required_equity": self.required_equity,
            "is_all_in": self.is_all_in,
            "made_hand_class": self.made_hand_class,
        }


def build_call_risk(observation: AgentObservation) -> CallRisk:
    """Compute the real capped call cost instead of trusting the displayed to_call."""

    facts = build_decision_facts(observation)
    call = facts["call"]
    return CallRisk(
        stack_before=int(call["stack_before"]),
        call_cost=int(call["call_cost"]),
        stack_fraction=float(call["stack_fraction"]),
        required_equity=float(call["required_equity"]),
        is_all_in=bool(call["is_all_in"]),
        made_hand_class=str(facts["made_hand_class"]),
    )


def observation_made_hand_class(observation: AgentObservation) -> str:
    return str(build_decision_facts(observation)["made_hand_class"])
