"""Tests for native and local discrete raise-to action spaces."""

from agentmemeval.core.domain import (
    AgentObservation,
    LegalAction,
    LegalActionSet,
    PlayerPublicState,
)
from agentmemeval.environment.raise_sizing import build_raise_sizing_plan


def make_observation(*, pot: int = 3, stack: int = 1000) -> AgentObservation:
    return AgentObservation(
        agent_id="agent_00",
        table_id="sizing",
        hand_id="sizing-h1",
        phase="preflop",
        seat=0,
        hole_cards=["3s", "2s"],
        community_cards=[],
        pot=pot,
        current_bet=2,
        to_call=2,
        players=[
            PlayerPublicState("agent_00", 0, stack, 0, 0, False, False),
            PlayerPublicState("agent_01", 1, 998, 2, 2, False, False),
        ],
        action_history=[],
        legal_actions=LegalActionSet(
            [
                LegalAction("fold"),
                LegalAction("call"),
                LegalAction("raise", 4, stack),
            ]
        ),
        seed=7,
    )


def test_local_discrete_excludes_deep_stack_preflop_all_in() -> None:
    plan = build_raise_sizing_plan(make_observation(), "local_discrete")

    assert plan.allowed_amounts == (4, 7)
    assert plan.stack_to_pot_ratio > 1.5


def test_local_discrete_keeps_all_in_when_stack_is_short_relative_to_pot() -> None:
    plan = build_raise_sizing_plan(
        make_observation(pot=600, stack=500),
        "local_discrete",
    )

    assert plan.allowed_amounts is not None
    assert 500 in plan.allowed_amounts
    assert plan.stack_to_pot_ratio <= 1.5


def test_native_no_limit_does_not_restrict_raise_amounts() -> None:
    plan = build_raise_sizing_plan(make_observation(), "native_no_limit")

    assert plan.allowed_amounts is None
