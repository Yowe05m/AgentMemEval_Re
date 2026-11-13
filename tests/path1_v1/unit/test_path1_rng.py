"""Actual engine RNG must retain the independently assigned Path1 seed bits."""

import random

import pytest

from agentmemeval.path1_v1.core.domain import TableSpec
from agentmemeval.path1_v1.core.errors import EnvironmentError
from agentmemeval.path1_v1.core.seeds import derive_seed
from agentmemeval.path1_v1.environment.holdem_adapter import (
    DIRECT_RNG_ALGORITHM,
    RANKS,
    SUITS,
    HoldemEnvironment,
)


def spec():
    ids = [f"agent_{i:02d}" for i in range(8)]
    return TableSpec(table_id="path1_collision", agent_ids=ids,
                     starting_stacks=dict.fromkeys(ids, 1000), small_blind=1, big_blind=2,
                     max_raises_per_street=4, dealer_index=0, hand_number=1)


def actual_deal(env):
    return [list(p.hole_cards) for p in env.players], list(env.deck)


def expected_deal(seed):
    deck = [rank + suit for rank in RANKS for suit in SUITS]
    random.Random(seed).shuffle(deck)
    holes = [[deck.pop(), deck.pop()] for _ in range(8)]
    return holes, deck


def test_explicit_rng_retains_high_bits_and_records_actual_seed():
    seed = 2**62 + 273
    env = HoldemEnvironment()
    env.reset(spec(), seed=12, direct_rng_seed=seed)
    assert env.actual_rng_seed == seed
    assert env.actual_rng_algorithm == DIRECT_RNG_ALGORITHM
    assert actual_deal(env) == expected_deal(seed)
    assert actual_deal(env) != expected_deal(seed % 2**32)


def test_legacy_known_collision_is_preserved_only_on_legacy_path():
    a, b = 103760, 121546
    assert derive_seed(a, "path1_collision") == derive_seed(b, "path1_collision") == 1248277998
    first, second = HoldemEnvironment(), HoldemEnvironment()
    first.reset(spec(), seed=a)
    second.reset(spec(), seed=b)
    assert actual_deal(first) == actual_deal(second) == expected_deal(1248277998)
    assert first.actual_rng_algorithm == "legacy-sha256-table-id-int32-v1"
    first.reset(spec(), seed=a, direct_rng_seed=a)
    second.reset(spec(), seed=b, direct_rng_seed=b)
    assert actual_deal(first) == expected_deal(a)
    assert actual_deal(second) == expected_deal(b)
    assert actual_deal(first) != actual_deal(second)


def test_paired_assignment_reuses_exact_deal_without_sharing_mutable_rng():
    a, b = HoldemEnvironment(), HoldemEnvironment()
    a.reset(spec(), seed=18, direct_rng_seed=2**61 + 17)
    b.reset(spec(), seed=18, direct_rng_seed=2**61 + 17)
    assert actual_deal(a) == actual_deal(b)
    assert a.rng is not b.rng
    before = b.rng.getstate()
    a.rng.random()
    assert b.rng.getstate() == before


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_direct_rng_boundaries_never_call_legacy_hash(monkeypatch, seed):
    # Construction retains the legacy unused placeholder RNG. Only reset's
    # actual direct-seed path must avoid secondary derivation before dealing.
    env = HoldemEnvironment()

    def forbidden(*args, **kwargs):
        raise AssertionError("direct seed was rehashed through legacy RNG")

    monkeypatch.setattr("agentmemeval.path1_v1.environment.holdem_adapter.make_rng", forbidden)
    monkeypatch.setattr("agentmemeval.path1_v1.environment.holdem_adapter.derive_seed", forbidden)
    env.reset(spec(), seed=9, direct_rng_seed=seed)
    assert env.actual_rng_seed == seed
    assert actual_deal(env) == expected_deal(seed)


@pytest.mark.parametrize("seed", [True, False, 1.0, -1, 2**63, "1"])
def test_direct_rng_rejects_invalid_seed_before_dealing(seed):
    env = HoldemEnvironment()
    with pytest.raises(EnvironmentError, match="direct RNG seed"):
        env.reset(spec(), seed=1, direct_rng_seed=seed)
    assert env.players == []
