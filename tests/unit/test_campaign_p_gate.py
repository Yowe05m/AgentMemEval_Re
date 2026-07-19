from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml

from agentmemeval.evaluation.aggregation import aggregate_metrics


def _gate_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools/task4/gate_campaign_p_before_e.py"
    spec = importlib.util.spec_from_file_location("task4_campaign_p_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(
    seed: int,
    revision_fallback_count: int = 0,
) -> dict[str, object]:
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
    effects = {
        1: {"expr": 10.0, "fact_expr_async": 4.0, "fact_expr_sync": 8.0},
        2: {"expr": -5.0, "fact_expr_async": -2.0, "fact_expr_sync": 1.0},
    }[seed]
    return {
        "run_validity": {"paper_eligible": False},
        "primary_metrics": {
            "per_agent": {"expr_00": {"bb_per_100": 0.0, "chip_delta": 0.0}},
            "stage_per_agent": {
                "train": {"expr_00": dict(values)},
                "test": {"expr_00": dict(values)},
            },
            "table_run_estimand": {
                "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
                "seed": seed,
                "run_id": f"mixed__s{seed}__a01",
                "endpoint": "final_test_bb_per_100",
                "baseline_mechanism": "fact",
                "effects_vs_baseline": effects,
                "statistical_plan_status": "pending_pilot_power_calibration",
                "multiple_comparison_method": "holm",
                "required_seed_pairs": None,
            },
        }
    }


def _campaign(
    tmp_path: Path,
    *,
    dirty: bool = False,
    revision_fallback: int = 0,
) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign_manifest = {
        "campaign": {
            "campaign_id": "p-gate-test",
            "seeds": [1, 2],
            "conditions": [{"condition_id": "mixed", "target_mechanism": "mixed"}],
        }
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(campaign_manifest), encoding="utf-8"
    )
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        + "".join(
            "t\tmixed\tmixed\t"
            f"{seed}\t1\tcomplete\tmixed__s{seed}__a01\t"
            f"{campaign / 'runs' / f'mixed__s{seed}__a01'}\t\t\n"
            for seed in (1, 2)
        ),
        encoding="utf-8",
    )
    for seed in (1, 2):
        run_dir = campaign / "runs" / f"mixed__s{seed}__a01"
        run_dir.mkdir(parents=True)
        runtime = {
            "run_id": f"mixed__s{seed}__a01",
            "seed": seed,
            "output_dir": str(run_dir),
            "config_snapshot_path": str(run_dir / "resolved_config.yaml"),
            "metadata": {
                "code": {"commit": "expected-sha", "dirty": dirty},
                "gpu": {"devices": [{"name": "gpu", "driver": "driver"}]},
                "model_service_runtime": {
                    "status": "verified",
                    "torch_cuda_version": "12.8",
                    "vllm_version": "vllm",
                },
                "model": {
                    "name": "model",
                    "revision": "revision",
                    "weights_hash": "hash",
                },
                "service": {
                    "provider": "openai_compatible",
                    "service_startup_parameters": {"max_model_len": 16384},
                },
                "embedding": {"name": "embedding", "revision": "revision"},
                "prompts": {
                    "decision_version": "version",
                    "decision_system_sha256": "decision-hash",
                    "experience_update_sha256": "experience-hash",
                },
            }
        }
        json_files = {
            "manifest.json": runtime,
            "metrics.json": _metrics(seed, revision_fallback),
            "protocol_audit.json": {
                "evaluation_target_ids": ["expr_00"],
                "execution_health": {
                    "valid": True,
                    "status": "passed",
                    "fallback_count": 0,
                    "memory_revision_fallback_count": 0,
                    "reward_conservation_violation_count": 0,
                    "stack_conservation_violation_count": 0,
                },
            },
            "checkpoint_generalization.json": {"results": []},
            "experiment_result.json": {"status": "complete"},
        }
        for name, data in json_files.items():
            (run_dir / name).write_text(json.dumps(data), encoding="utf-8")
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment": {
                        "campaign_id": "p-gate-test",
                        "campaign_condition_id": "mixed",
                        "seed": seed,
                        "run_id": f"mixed__s{seed}__a01",
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "hand_summaries.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "report.md").write_text("complete\n", encoding="utf-8")
    aggregate_path = campaign / "campaign_aggregate_test.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "status": "descriptive_only",
                "completed_run_count": 2,
                "expected_run_count": 2,
                "runtime_homogeneity": {
                    "homogeneous": True,
                    "identity": {
                        "code": [["commit", "expected-sha"], ["dirty", False]],
                        "prompts": [
                            ["decision_version", "version"],
                            ["decision_system_sha256", "decision-hash"],
                            ["experience_update_sha256", "experience-hash"],
                        ],
                    },
                },
                "design": "mixed_table",
                "aggregate_metrics": aggregate_metrics(
                    [_metrics(1), _metrics(2)]
                ),
            }
        ),
        encoding="utf-8",
    )
    return campaign, aggregate_path


def test_campaign_p_gate_accepts_complete_clean_homogeneous_evidence(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )
    assert audit["status"] == "ready_to_start_campaign_e"
    assert audit["schema_version"] == "task4_campaign_p_before_e_gate_v5"
    assert audit["blockers"] == []
    assert audit["behavior_freeze_preview"]["status"] == "frozen"
    assert (
        audit["campaign_p_power_diagnostic"]["status"]
        == "p_side_power_diagnostic_ready_not_joint_freeze"
    )
    assert (
        audit["campaign_p_power_diagnostic"]["joint_p_e_power_freeze_complete"]
        is False
    )
    assert len(audit["leaf_evidence"][0]["sha256"]) == 8
    assert len(audit["leaf_evidence"]) == 2
    assert audit["aggregate_metrics_match_canonical_leaf_rebuild"] is True
    assert (
        audit["observed_aggregate_metrics_sha256"]
        == audit["rebuilt_aggregate_metrics_sha256"]
    )


def test_campaign_p_gate_rejects_dirty_or_revision_fallback_evidence(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(
        tmp_path, dirty=True, revision_fallback=1
    )
    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )
    assert audit["status"] == "no_go"
    assert any("dirty worktree" in blocker for blocker in audit["blockers"])
    assert any("revision fallbacks" in blocker for blocker in audit["blockers"])


def test_campaign_p_gate_rejects_explicit_execution_prompt_and_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    run_dir = campaign / "runs" / "mixed__s1__a01"
    protocol_path = run_dir / "protocol_audit.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["execution_health"]["fallback_count"] = 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["prompts"]["decision_version"] = "wrong-version"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_data = json.loads(aggregate.read_text(encoding="utf-8"))
    aggregate_data["runtime_homogeneity"]["identity"]["code"][0][1] = "wrong-sha"
    aggregate.write_text(json.dumps(aggregate_data), encoding="utf-8")

    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )

    assert audit["status"] == "no_go"
    assert any("execution fallback_count" in item for item in audit["blockers"])
    assert any("prompt identity mismatch" in item for item in audit["blockers"])
    assert any("aggregate code SHA mismatch" in item for item in audit["blockers"])


def test_campaign_p_gate_rejects_cross_campaign_or_malformed_aggregate(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    external_dir = tmp_path / "other"
    external_dir.mkdir()
    external_aggregate = external_dir / aggregate.name
    external_aggregate.write_text(
        aggregate.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = json.loads(external_aggregate.read_text(encoding="utf-8"))
    data["completed_run_count"] = "not-an-integer"
    data["aggregate_metrics"]["paired_estimand_descriptive"][
        "effects_by_mechanism"
    ]["expr"] = [float("nan"), 1.0]
    external_aggregate.write_text(json.dumps(data), encoding="utf-8")

    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=external_aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )

    assert audit["status"] == "no_go"
    assert any("directly inside campaign_dir" in item for item in audit["blockers"])
    assert any("aggregate matrix mismatch" in item for item in audit["blockers"])
    assert any("non-finite values" in item for item in audit["blockers"])


def test_campaign_p_gate_rejects_paired_contract_or_seed_mismatch(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    data = json.loads(aggregate.read_text(encoding="utf-8"))
    paired = data["aggregate_metrics"]["paired_estimand_descriptive"]
    paired["matched_seeds"] = [2, 1]
    paired["endpoint"] = "wrong_endpoint"
    paired["multiple_comparison_method"] = "none"
    data["runtime_homogeneity"]["identity"]["prompts"][0][1] = "wrong-version"
    aggregate.write_text(json.dumps(data), encoding="utf-8")

    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )

    assert audit["status"] == "no_go"
    assert any("matched seeds mismatch" in item for item in audit["blockers"])
    assert any("endpoint mismatch" in item for item in audit["blockers"])
    assert any(
        "multiple_comparison_method mismatch" in item
        for item in audit["blockers"]
    )
    assert any(
        "aggregate prompt identity mismatch" in item
        for item in audit["blockers"]
    )
    assert any(
        "aggregate metrics do not match" in item for item in audit["blockers"]
    )


def test_campaign_p_gate_rejects_external_or_mismatched_leaf_identity(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    state_path = campaign / "state.tsv"
    state = state_path.read_text(encoding="utf-8")
    external = tmp_path / "external" / "mixed__s1__a01"
    external.parent.mkdir()
    state_path.write_text(
        state.replace(
            str(campaign / "runs" / "mixed__s1__a01"),
            str(external),
        ),
        encoding="utf-8",
    )
    run_two = campaign / "runs" / "mixed__s2__a01"
    manifest_path = run_two / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = run_two / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["campaign_id"] = "other-campaign"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    audit = _gate_module().build_gate(
        campaign,
        aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_max_model_len=16384,
        expected_prompts={
            "decision_version": "version",
            "decision_system_sha256": "decision-hash",
            "experience_update_sha256": "experience-hash",
        },
    )

    assert audit["status"] == "no_go"
    assert any("canonical campaign leaf" in item for item in audit["blockers"])
    assert any("manifest seed mismatch" in item for item in audit["blockers"])
    assert any(
        "resolved config campaign_id mismatch" in item
        for item in audit["blockers"]
    )
