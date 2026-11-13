"""Tests for deterministic all-in call diagnostics."""

import pytest

from agentmemeval.core.domain import (
    AgentObservation,
    LegalAction,
    LegalActionSet,
    PlayerPublicState,
)
from agentmemeval.environment.action_risk import build_call_risk
from agentmemeval.environment.hand_evaluator import evaluate_best, split_side_pots


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
            PlayerPublicState("folded", 2, 880, 0, 120, True, False),
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
    # Hero's total eligibility is 80 + 920 = 1000. Villain's unmatched 1000
    # cannot be won; the folded player's 120 remains dead money in hero's pot.
    assert risk.contestable_pot_before_call == 80 + 1000 + 120
    assert risk.contestable_pot_after_call == 1000 + 1000 + 120
    assert risk.required_equity == pytest.approx(920 / 2120)
    assert risk.required_equity != pytest.approx(920 / (2200 + 920))
    payouts = split_side_pots(
        contributions={"hero": 1000, "villain": 2000, "folded": 120},
        folded={"folded"},
        ranks={
            "hero": evaluate_best(["As", "Ah", "2c", "3d", "4s", "8h", "9c"]),
            "villain": evaluate_best(["Ks", "Kh", "2c", "3d", "4s", "8h", "9c"]),
        },
    )
    assert sum(p["amount"] for p in payouts if p["agent_id"] == "hero") == 2120
