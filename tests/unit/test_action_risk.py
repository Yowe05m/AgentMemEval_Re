"""Tests for deterministic all-in call diagnostics."""

from agentmemeval.core.domain import (
    AgentObservation,
    LegalAction,
    LegalActionSet,
    PlayerPublicState,
)
from agentmemeval.environment.action_risk import build_call_risk


def test_call_risk_caps_to_call_at_remaining_stack() -> None:
    observation = AgentObservation(
        agent_id="agent_00",
        table_id="risk",
        hand_id="risk-h1",
        phase="river",
        seat=0,
        hole_cards=["3s", "2s"],
        community_cards=["Td", "4c", "Jc", "2c", "9h"],
        pot=2200,
        current_bet=1960,
        to_call=1960,
        players=[
            PlayerPublicState("agent_00", 0, 920, 0, 80, False, False),
            PlayerPublicState("agent_01", 1, 0, 1960, 2000, False, True),
        ],
        action_history=[],
        legal_actions=LegalActionSet([LegalAction("fold"), LegalAction("call")]),
        seed=7,
    )

    risk = build_call_risk(observation)
    assert risk.call_cost == 920
    assert risk.stack_fraction == 1.0
    assert risk.is_all_in is True
    assert risk.made_hand_class == "Pair"
    assert round(risk.required_equity, 3) == 0.295
