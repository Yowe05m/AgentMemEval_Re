from __future__ import annotations

import json
from pathlib import Path

import yaml

from agentmemeval.evaluation.decision_point_gate import (
    build_decision_point_smoke_gate,
)


def _write_run(tmp_path: Path, *, malformed_score: bool = False) -> Path:
    run = tmp_path / "run"
    snapshots = run / "memory_snapshots"
    snapshots.mkdir(parents=True)
    config = {
        "agent": {"retrieval_unit": "decision_point_max_v1"},
        "experiment": {
            "train_hands": 1,
            "test_hands": 1,
            "agent_roster": [{"agent_id": "fact_00", "mechanism": "fact"}],
        },
    }
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    json_files = {
        "manifest.json": {
            "metadata": {"code": {"commit": "expected", "dirty": False}}
        },
        "metrics.json": {"primary_metrics": {}},
        "protocol_audit.json": {
            "execution_health": {
                "valid": True,
                "fallback_count": 0,
                "memory_revision_fallback_count": 0,
                "reward_conservation_violation_count": 0,
                "stack_conservation_violation_count": 0,
            }
        },
        "checkpoint_generalization.json": {"results": []},
        "experiment_result.json": {"status": "complete"},
    }
    for name, value in json_files.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    (run / "report.md").write_text("complete\n", encoding="utf-8")
    score = {
        "record_id": "r1",
        "retrieval_unit": "decision_point_max_v1",
        "matched_decision_index": None if malformed_score else 0,
        "matched_phase": "preflop",
    }
    event = {
        "event": "action",
        "fallback_used": False,
        "memory_context": {"metadata": {"retrieval_scores": [score]}},
    }
    (run / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    hands = [
        {
            "stage": "train",
            "hand_id": "h1",
            "memory_updated": True,
            "starting_stacks": {"a": 10, "b": 10},
            "final_stacks": {"a": 11, "b": 9},
            "rewards": {"a": 1, "b": -1},
        },
        {
            "stage": "test",
            "hand_id": "h2",
            "memory_updated": False,
            "starting_stacks": {"a": 11, "b": 9},
            "final_stacks": {"a": 10, "b": 10},
            "rewards": {"a": -1, "b": 1},
        },
    ]
    (run / "hand_summaries.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in hands), encoding="utf-8"
    )
    snapshot = {
        "mechanism": "fact",
        "payload": {
            "schema_version": 6,
            "retrieval_unit": "decision_point_max_v1",
            "records": [
                {
                    "source": {
                        "decisions": [
                            {
                                "phase": "preflop",
                                "retrieval_query": "phase=preflop",
                                "features": ["phase:preflop"],
                            }
                        ]
                    }
                }
            ],
        },
    }
    (snapshots / "fact_00_final.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    return run


def test_decision_point_smoke_gate_accepts_complete_evidence(tmp_path: Path) -> None:
    audit = build_decision_point_smoke_gate(
        _write_run(tmp_path),
        expected_code_sha="expected",
        expected_train_hands=1,
        expected_test_hands=1,
    )

    assert audit["status"] == "ready_to_start_v8_calibration_pilot"
    assert audit["blockers"] == []
    assert audit["evidence"]["event_audit"]["retrieval_score_count"] == 1


def test_decision_point_smoke_gate_rejects_malformed_match(tmp_path: Path) -> None:
    audit = build_decision_point_smoke_gate(
        _write_run(tmp_path, malformed_score=True),
        expected_code_sha="expected",
        expected_train_hands=1,
        expected_test_hands=1,
    )

    assert audit["status"] == "no_go"
    assert any("malformed decision-point" in item for item in audit["blockers"])


def test_decision_point_smoke_gate_expands_test_hands_per_target(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path)
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    config["experiment"]["evaluate_all_train_agents"] = True
    config["experiment"]["agent_roster"].append(
        {"agent_id": "no_memory_00", "mechanism": "no_memory"}
    )
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    hands = (run / "hand_summaries.jsonl").read_text(encoding="utf-8").splitlines()
    extra = json.loads(hands[-1])
    extra["hand_id"] = "h3"
    hands.append(json.dumps(extra))
    (run / "hand_summaries.jsonl").write_text(
        "\n".join(hands) + "\n", encoding="utf-8"
    )

    audit = build_decision_point_smoke_gate(
        run,
        expected_code_sha="expected",
        expected_train_hands=1,
        expected_test_hands=1,
    )

    assert audit["status"] == "ready_to_start_v8_calibration_pilot"
    hand_audit = audit["evidence"]["hand_audit"]
    assert hand_audit["configured_test_hands_per_target"] == 1
    assert hand_audit["evaluation_target_count"] == 2
    assert hand_audit["expected_total_test_hands"] == 2
