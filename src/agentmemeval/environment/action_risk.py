"""Deterministic call-risk and made-hand diagnostics for experiment audit."""

from __future__ import annotations

from dataclasses import dataclass

from agentmemeval.core.domain import AgentObservation
from agentmemeval.environment.hand_evaluator import evaluate_best


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

    self_state = next(
        player for player in observation.players if player.agent_id == observation.agent_id
    )
    stack_before = max(0, self_state.stack)
    call_cost = min(max(0, observation.to_call), stack_before)
    stack_fraction = call_cost / max(1, stack_before)
    required_equity = call_cost / max(1, observation.pot + call_cost)
    return CallRisk(
        stack_before=stack_before,
        call_cost=call_cost,
        stack_fraction=stack_fraction,
        required_equity=required_equity,
        is_all_in=call_cost > 0 and call_cost >= stack_before,
        made_hand_class=observation_made_hand_class(observation),
    )


def observation_made_hand_class(observation: AgentObservation) -> str:
    cards = [*observation.hole_cards, *observation.community_cards]
    if len(cards) >= 5:
        return evaluate_best(cards).class_name
    paired = observation.hole_cards[0][0] == observation.hole_cards[1][0]
    return "Pocket Pair" if paired else "Unpaired"
