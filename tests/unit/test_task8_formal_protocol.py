from __future__ import annotations

import copy

import pytest

from agentmemeval.config.loader import validate_config
from agentmemeval.core.domain import MemorySnapshot
from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import (
    build_clone_audit,
    build_heldout_schedule_manifest,
    clone_memory_branches,
    derive_formal_hand_seed,
    verify_schedule_manifest,
)


def _config(experiment: dict[str, object]) -> dict[str, object]:
    return {
        "provider": {"provider": "mock", "model": "mock-deterministic-v1"},
        "table": {"starting_stack": 100, "small_blind": 1, "big_blind": 2},
        "agent": {"mechanism": "expr", "memory_scope": "per_agent"},
        "experiment": {"scenario": "fixed_evolving_table", "seed": 1, **experiment},
    }


def test_checkpoint_set_accepts_exact_task8_points() -> None:
    validate_config(_config({"train_hands": 300, "checkpoint_set": [30, 75, 150, 300]}))


@pytest.mark.parametrize(
    "points",
    [[], [30, 30, 300], [75, 30, 300], [0, 300], [30, 301], [30, 150]],
)
def test_checkpoint_set_fails_closed(points: list[int]) -> None:
    with pytest.raises(ConfigError):
        validate_config(_config({"train_hands": 300, "checkpoint_set": points}))


def test_checkpoint_set_rejects_ambiguous_legacy_interval() -> None:
    with pytest.raises(ConfigError, match="不得同时配置"):
        validate_config(
            _config(
                {
                    "train_hands": 300,
                    "checkpoint_set": [30, 75, 150, 300],
                    "checkpoint_interval": 30,
                }
            )
        )


def test_checkpoint_specific_test_hands_are_validated() -> None:
    validate_config(
        _config(
            {
                "train_hands": 300,
                "checkpoint_set": [30, 75, 150, 300],
                "checkpoint_test_hands_by_checkpoint": {
                    "30": 50,
                    "75": 50,
                    "150": 50,
                    "300": 200,
                },
            }
        )
    )
    with pytest.raises(ConfigError, match="未声明 checkpoint"):
        validate_config(
            _config(
                {
                    "train_hands": 300,
                    "checkpoint_set": [30, 75, 150, 300],
                    "checkpoint_test_hands_by_checkpoint": {600: 1},
                }
            )
        )


def test_crn_seed_exposes_no_identity_parameters() -> None:
    expected = derive_formal_hand_seed(101, "heldout", 300, "H01", 7, "deal_and_opponent")
    assert expected == derive_formal_hand_seed(
        101, "heldout", 300, "H01", 7, "deal_and_opponent"
    )
    assert expected != derive_formal_hand_seed(
        101, "heldout", 300, "H02", 7, "deal_and_opponent"
    )


def test_three_heldout_schedules_are_reproducible_distinct_and_hash_locked() -> None:
    kwargs = {
        "root_seed": 101,
        "checkpoint_set": [30, 75, 150, 300],
        "table_set": ["H01", "H02", "H03"],
        "hands_by_checkpoint": {30: 1, 75: 1, 150: 1, 300: 2},
        "table_size": 8,
        "roster_identity": "task8-natural-rosters-v1",
    }
    first = build_heldout_schedule_manifest(**kwargs)
    second = build_heldout_schedule_manifest(**kwargs)
    assert first == second
    assert verify_schedule_manifest(first) == first["schedule_sha256"]
    first_seeds = {
        row["table_id"]: row["deal_and_opponent_seed"]
        for row in first["rows"]
        if row["checkpoint_hand"] == 30
    }
    assert len(set(first_seeds.values())) == 3
    tampered = copy.deepcopy(first)
    tampered["rows"][0]["dealer_index"] += 1
    with pytest.raises(ConfigError, match="SHA-256"):
        verify_schedule_manifest(tampered)


def test_formal_three_tables_require_distinct_complete_rosters() -> None:
    experiment = {
        "run_mode": "formal",
        "train_hands": 300,
        "checkpoint_set": [30, 75, 150, 300],
        "heldout_table_set": ["H01", "H02", "H03"],
        "heldout_table_rosters": {
            "H01": {"mechanism": "no_memory", "roster": "natural-a"},
            "H02": {"mechanism": "no_memory", "roster": "natural-b"},
            "H03": {"mechanism": "no_memory", "roster": "natural-c"},
        },
    }
    validate_config(_config(experiment))
    duplicate = copy.deepcopy(experiment)
    duplicate["heldout_table_rosters"]["H03"] = duplicate["heldout_table_rosters"]["H02"]
    with pytest.raises(ConfigError, match="不同的自然 roster"):
        validate_config(_config(duplicate))


def test_memory_branches_are_deep_isolated_and_without_is_complete() -> None:
    snapshot = MemorySnapshot(
        mechanism="fact_expr_async",
        agent_id="agent_00",
        scope="per_agent",
        payload={
            "fact": {
                "schema_version": 5,
                "records": [{"record_id": "f1"}],
                "admission_log": [{"status": "admitted"}],
                "admission_counts": {"admitted": 1},
                "retrieval_audit_log": [{"query": "x"}],
            },
            "expr": {
                "history": [{"version": 1, "body": "learned"}],
                "revision_log": [{"version": 1}],
                "skipped_trajectory_hand_ids": ["h1"],
            },
            "sweep_log": [{"hand": 1}],
            "evidence_review_queue": [{"id": 1}],
            "fact_state": {"f1": {"weight": 1}},
            "hand_counter": 5,
            "eligible_hand_counter": 4,
            "skipped_trajectory_hand_ids": ["h2"],
        },
    )
    branches = clone_memory_branches(snapshot)
    branches["Online"].payload["fact"]["records"].append({"record_id": "f2"})
    assert len(snapshot.payload["fact"]["records"]) == 1
    assert len(branches["Frozen"].payload["fact"]["records"]) == 1
    without = branches["Without"].payload
    assert without["fact"]["records"] == []
    assert without["expr"]["history"] == []
    assert without["sweep_log"] == []
    assert without["evidence_review_queue"] == []
    assert without["fact_state"] == {}
    assert without["hand_counter"] == 0
    audit = build_clone_audit(snapshot, branches)
    assert len(audit["parent_checkpoint_sha256"]) == 64
    assert audit["branches"]["Without"]["transform"] == "whole_memory_removal"


def test_unknown_without_transform_is_pending_review_fail_closed() -> None:
    snapshot = MemorySnapshot("future_memory", "agent_00", "per_agent", {"x": 1})
    with pytest.raises(ConfigError, match="pending_review"):
        clone_memory_branches(snapshot)
