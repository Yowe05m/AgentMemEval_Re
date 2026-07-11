import pytest

from agentmemeval.core.domain import ActionDecision, TableSpec
from agentmemeval.environment.hand_evaluator import (
    evaluate_best,
    score_five,
    split_side_pots,
)
from agentmemeval.environment.holdem_adapter import HoldemEnvironment

HAND_CLASS_CASES = [
    (["As", "Ks", "Qs", "Js", "Ts"], "Straight Flush", (8, (14,))),
    (["9c", "9d", "9h", "9s", "2d"], "Four of a Kind", (7, (9, 2))),
    (["Tc", "Td", "Th", "3s", "3d"], "Full House", (6, (10, 3))),
    (["Ac", "Jc", "9c", "5c", "2c"], "Flush", (5, (14, 11, 9, 5, 2))),
    (["As", "2d", "3h", "4c", "5s"], "Straight", (4, (5,))),
    (["Qc", "Qd", "Qh", "8s", "2d"], "Three of a Kind", (3, (12, 8, 2))),
    (["Jc", "Jd", "4h", "4s", "9d"], "Two Pair", (2, (11, 4, 9))),
    (["Ac", "Ad", "Kh", "8s", "2d"], "Pair", (1, (14, 13, 8, 2))),
    (["Ac", "Kd", "9h", "5s", "2d"], "High Card", (0, (14, 13, 9, 5, 2))),
]


@pytest.mark.parametrize(("cards", "class_name", "score"), HAND_CLASS_CASES)
def test_golden_hand_classes_cover_standard_rankings(
    cards: list[str], class_name: str, score: tuple[int, tuple[int, ...]]
) -> None:
    rank = evaluate_best(cards)
    assert rank.class_name == class_name
    assert rank.score == score


def test_best_five_uses_kickers_and_normalizes_card_codes() -> None:
    stronger = evaluate_best(["as", "Ad", "Kh", "8s", "2d", "3c", "4h"])
    weaker = evaluate_best(["Ac", "Ah", "Qh", "8d", "2s", "3d", "4c"])
    assert stronger.class_name == "Pair"
    assert weaker.class_name == "Pair"
    assert stronger.score > weaker.score
    assert stronger.best_cards[0] == "As"


def test_evaluator_rejects_invalid_duplicate_and_too_many_cards() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_best(["As", "As", "Kd", "Qh", "Jc"])
    with pytest.raises(ValueError, match="invalid card code"):
        score_five(["1s", "Kd", "Qh", "Jc", "9d"])
    with pytest.raises(ValueError, match="expected 5 to 7 cards"):
        evaluate_best(["As", "Kd", "Qh", "Jc", "9d", "8s", "7c", "6h"])


def test_side_pot_split_handles_folded_players_and_remainder() -> None:
    ranks = {
        "short": evaluate_best(["As", "Ah", "2c", "3d", "4s", "8h", "9c"]),
        "deep": evaluate_best(["Ks", "Kh", "2c", "3d", "4s", "8h", "9c"]),
        "caller": evaluate_best(["Qs", "Qh", "2c", "3d", "4s", "8h", "9c"]),
    }
    payouts = split_side_pots(
        contributions={"short": 10, "deep": 30, "caller": 30, "folder": 30},
        folded={"folder"},
        ranks=ranks,
    )
    by_agent = _sum_payouts(payouts)
    assert by_agent == {"short": 40, "deep": 60}

    tied = {
        "a": evaluate_best(["As", "Ah", "Kd", "Qc", "9s", "4h", "2c"]),
        "b": evaluate_best(["Ad", "Ac", "Kh", "Qd", "9c", "4s", "2h"]),
        "c": evaluate_best(["Ks", "Kh", "Qh", "Jc", "8d", "4c", "2d"]),
    }
    tied_payouts = split_side_pots(
        contributions={"a": 5, "b": 5, "c": 5},
        folded=set(),
        ranks=tied,
    )
    assert _sum_payouts(tied_payouts) == {"a": 8, "b": 7}


def test_side_pot_refunds_when_no_layer_has_eligible_contender() -> None:
    payouts = split_side_pots(
        contributions={"a": 5, "b": 5},
        folded={"a", "b"},
        ranks={},
    )
    assert sum(int(payout["amount"]) for payout in payouts) == 10
    assert {payout["reason"] for payout in payouts} == {"uncontested_refund"}


def test_min_raise_tracks_last_effective_raise_size() -> None:
    env = _make_env({"a": 100, "b": 100, "c": 100})
    assert env.current_agent_id() == "a"

    env.step("a", ActionDecision("raise", amount=10))
    raise_rule = env.legal_actions("b").rule_for("raise")

    assert raise_rule is not None
    assert raise_rule.min_amount == 18


def test_short_all_in_raise_does_not_reopen_action_to_prior_actors() -> None:
    env = _make_env({"a": 100, "b": 100, "c": 14})
    env.step("a", ActionDecision("raise", amount=10))
    env.step("b", ActionDecision("call"))

    short_raise_rule = env.legal_actions("c").rule_for("raise")
    assert short_raise_rule is not None
    assert short_raise_rule.min_amount == 14
    assert short_raise_rule.max_amount == 14
    assert short_raise_rule.reopens is False

    result = env.step("c", ActionDecision("raise", amount=14))

    assert result.event["effective_raise"] is False
    assert env.current_agent_id() == "a"
    assert env.legal_actions("a").rule_for("raise") is None
    env.step("a", ActionDecision("call"))
    assert env.legal_actions("b").rule_for("raise") is None


def test_full_all_in_raise_reopens_action_and_updates_min_raise() -> None:
    env = _make_env({"a": 100, "b": 100, "c": 100})
    env.step("a", ActionDecision("raise", amount=10))
    env.step("b", ActionDecision("call"))

    result = env.step("c", ActionDecision("raise", amount=22))
    next_raise_rule = env.legal_actions("a").rule_for("raise")

    assert result.event["effective_raise"] is True
    assert next_raise_rule is not None
    assert next_raise_rule.min_amount == 34


def test_optional_treys_crosscheck_matches_original_engine_class_names() -> None:
    treys = pytest.importorskip("treys")
    evaluator = treys.Evaluator()

    for cards, _, _ in HAND_CLASS_CASES:
        hand = [treys.Card.new(card) for card in cards[:2]]
        board = [treys.Card.new(card) for card in cards[2:]]
        treys_score = evaluator.evaluate(board, hand)
        treys_class = evaluator.class_to_string(evaluator.get_rank_class(treys_score))
        assert evaluate_best(cards).class_name == treys_class


def _make_env(stacks: dict[str, int]) -> HoldemEnvironment:
    env = HoldemEnvironment()
    env.reset(
        TableSpec(
            table_id="rules",
            agent_ids=list(stacks),
            starting_stacks=stacks,
            small_blind=1,
            big_blind=2,
            max_raises_per_street=0,
        ),
        seed=7,
    )
    return env


def _sum_payouts(payouts: list[dict[str, object]]) -> dict[str, int]:
    by_agent: dict[str, int] = {}
    for payout in payouts:
        agent_id = str(payout["agent_id"])
        by_agent[agent_id] = by_agent.get(agent_id, 0) + int(payout["amount"])
    return by_agent
