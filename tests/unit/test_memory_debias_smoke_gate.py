from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml


def _gate_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "task4"
        / "gate_memory_debias_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("task4_memory_debias_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, *, intent: bool = False, collapsed: bool = False) -> Path:
    run = tmp_path / "run"
    snapshots = run / "memory_snapshots"
    snapshots.mkdir(parents=True)
    config = {"agent": {"reject_single_preflop_fold": True}}
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    manifest = {
        "metadata": {
            "code": {"commit": "expected-sha", "dirty": False},
            "prompts": {
                "decision_version": "2026-07-17-v5-nonnormative-memory"
            },
        }
    }
    metrics_agents = {
        agent_id: {
            "vpip": 0.0 if collapsed and index < 3 else 0.30,
            "fold_rate": 1.0 if collapsed and index < 3 else 0.40,
        }
        for index, agent_id in enumerate(
            ["fact_00", "fact_01", "sync_00", "sync_01", "async_00", "async_01"]
        )
    }
    json_files = {
        "manifest.json": manifest,
        "metrics.json": {
            "primary_metrics": {"stage_per_agent": {"train": metrics_agents}}
        },
        "protocol_audit.json": {"execution_health": {"valid": True}},
        "checkpoint_generalization.json": {"results": []},
        "experiment_result.json": {"status": "complete"},
    }
    for name, data in json_files.items():
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    (run / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "hand_summaries.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "report.md").write_text("complete\n", encoding="utf-8")
    for agent_id in metrics_agents:
        mechanism = (
            "fact"
            if agent_id.startswith("fact_")
            else "fact_expr_sync"
            if agent_id.startswith("sync_")
            else "fact_expr_async"
        )
        fact = {
            "schema_version": 5,
            "reject_single_preflop_fold": True,
            "admission_counts": {
                "reason:single_preflop_fold_low_information": 1
            },
            "records": [
                {
                    "source": {
                        "decisions": [
                            {"observed_action": "call", **({"intent": "x"} if intent else {})}
                        ]
                    }
                }
            ],
        }
        payload = fact if mechanism == "fact" else {"fact": fact}
        snapshot = {
            "mechanism": mechanism,
            "agent_id": agent_id,
            "scope": "per_agent",
            "payload": payload,
        }
        (snapshots / f"{agent_id}_final.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )
    return run


def test_memory_debias_smoke_gate_accepts_integrated_repair_evidence(
    tmp_path: Path,
) -> None:
    audit = _gate_module().build_gate(
        _run(tmp_path),
        expected_code_sha="expected-sha",
    )
    assert audit["status"] == "ready_to_start_v5_pilot"
    assert audit["blockers"] == []
    assert audit["evidence"]["snapshot_audit"]["factual_snapshot_count"] == 6


def test_memory_debias_smoke_gate_rejects_normative_intent_and_systematic_collapse(
    tmp_path: Path,
) -> None:
    audit = _gate_module().build_gate(
        _run(tmp_path, intent=True, collapsed=True),
        expected_code_sha="expected-sha",
    )
    assert audit["status"] == "no_go"
    assert any("intent" in blocker for blocker in audit["blockers"])
    assert any("systematic" in blocker for blocker in audit["blockers"])
