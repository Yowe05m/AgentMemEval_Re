from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _gate_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools/task4/gate_campaign_p_before_e.py"
    spec = importlib.util.spec_from_file_location("task4_campaign_p_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(revision_fallback_count: int = 0) -> dict[str, object]:
    values = {
        "vpip": 0.30,
        "fold_rate": 0.40,
        "voluntary_participation_hands": 10,
        "all_in_hand_rate": 0.05,
        "bust_hand_rate": 0.01,
        "hand_reward_sensitivity": {"share_of_absolute_reward_activity": 0.30},
        "memory": {
            "empty_retrieval_rate": 0.20,
            "max_structural_signature_share": 0.15,
            "revision_fallback_count": revision_fallback_count,
        },
    }
    return {
        "primary_metrics": {
            "stage_per_agent": {
                "train": {"expr_00": dict(values)},
                "test": {"expr_00": dict(values)},
            }
        }
    }


def _campaign(tmp_path: Path, *, dirty: bool = False, revision_fallback: int = 0) -> Path:
    campaign = tmp_path / "campaign"
    run_dir = campaign / "runs" / "mixed__s1__a01"
    run_dir.mkdir(parents=True)
    campaign_manifest = {
        "campaign": {
            "campaign_id": "p-gate-test",
            "seeds": [1],
            "conditions": [{"condition_id": "mixed", "target_mechanism": "mixed"}],
        }
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(campaign_manifest), encoding="utf-8"
    )
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"t\tmixed\tmixed\t1\t1\tcomplete\tr1\t{run_dir}\t\t\n",
        encoding="utf-8",
    )
    runtime = {
        "metadata": {
            "code": {"commit": "expected-sha", "dirty": dirty},
            "gpu": {"devices": [{"name": "gpu", "driver": "driver"}]},
            "model_service_runtime": {
                "status": "verified",
                "torch_cuda_version": "12.8",
                "vllm_version": "vllm",
            },
            "model": {"name": "model", "revision": "revision", "weights_hash": "hash"},
            "service": {
                "provider": "openai_compatible",
                "service_startup_parameters": {"max_model_len": 16384},
            },
            "embedding": {"name": "embedding", "revision": "revision"},
            "prompts": {"decision_version": "version", "decision_system_sha256": "hash"},
        }
    }
    json_files = {
        "manifest.json": runtime,
        "metrics.json": _metrics(revision_fallback),
        "protocol_audit.json": {"execution_health": {"valid": True}},
        "checkpoint_generalization.json": {"results": []},
        "experiment_result.json": {"status": "complete"},
    }
    for name, data in json_files.items():
        (run_dir / name).write_text(json.dumps(data), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (run_dir / "hand_summaries.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "report.md").write_text("complete\n", encoding="utf-8")
    return campaign


def test_campaign_p_gate_accepts_complete_clean_homogeneous_evidence(
    tmp_path: Path,
) -> None:
    audit = _gate_module().build_gate(
        _campaign(tmp_path),
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
    )
    assert audit["status"] == "ready_to_start_campaign_e"
    assert audit["blockers"] == []
    assert audit["behavior_freeze_preview"]["status"] == "frozen"
    assert len(audit["leaf_evidence"][0]["sha256"]) == 8


def test_campaign_p_gate_rejects_dirty_or_revision_fallback_evidence(
    tmp_path: Path,
) -> None:
    audit = _gate_module().build_gate(
        _campaign(tmp_path, dirty=True, revision_fallback=1),
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
    )
    assert audit["status"] == "no_go"
    assert any("dirty worktree" in blocker for blocker in audit["blockers"])
    assert any("revision fallbacks" in blocker for blocker in audit["blockers"])
