from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path, PurePosixPath
from types import ModuleType

import yaml

from agentmemeval.experiments.campaign import build_campaign_aggregate_payload


def _gate_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools/task4/gate_campaign_p_before_e.py"
    spec = importlib.util.spec_from_file_location("task4_campaign_p_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate_correction_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/task4/rebuild_campaign_aggregate_run_local_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task4_campaign_aggregate_cache_correction", path
    )
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


def _checkpoint_results(seed: int) -> list[dict[str, object]]:
    effects = {
        1: {
            "fact": 0.0,
            "expr": 10.0,
            "fact_expr_async": 4.0,
            "fact_expr_sync": 8.0,
        },
        2: {
            "fact": 0.0,
            "expr": -5.0,
            "fact_expr_async": -2.0,
            "fact_expr_sync": 1.0,
        },
    }[seed]
    return [
        {
            "checkpoint_hand": 100,
            "mechanism": mechanism,
            "bb_per_100": value,
            "test_chip_per_hand": value / 100.0,
            "train_bb_per_100": value / 2.0,
            "train_chip_per_hand": value / 200.0,
            "generalization_gap_bb_per_100": value / 3.0,
        }
        for mechanism, value in effects.items()
    ]


def _campaign(
    tmp_path: Path,
    *,
    dirty: bool = False,
    revision_fallback: int = 0,
) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign_manifest = {
        "schema_version": "agentmemeval_campaign_v1",
        "campaign": {
            "campaign_id": "p-gate-test",
            "design": "mixed_table",
            "seeds": [1, 2],
            "conditions": [{"condition_id": "mixed", "target_mechanism": "mixed"}],
        },
        "base_config": {
            "experiment": {
                "run_mode": "smoke",
                "primary_baseline_mechanism": "fact",
                "statistical_plan_status": "pending_pilot_power_calibration",
                "multiple_comparison_method": "holm",
                "required_seed_pairs": None,
            }
        },
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
                "embedding": {
                    "name": "embedding",
                    "revision": "revision",
                    "cache_namespace_template": str(
                        run_dir / "embedding_cache" / "{agent_id}.json"
                    ),
                },
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
            "checkpoint_generalization.json": {
                "results": _checkpoint_results(seed)
            },
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
    completed_runs = [
        {
            "condition_id": "mixed",
            "target_mechanism": "mixed",
            "seed": str(seed),
            "attempt": "1",
            "status": "complete",
            "run_id": f"mixed__s{seed}__a01",
            "run_dir": str(campaign / "runs" / f"mixed__s{seed}__a01"),
        }
        for seed in (1, 2)
    ]
    aggregate_payload = build_campaign_aggregate_payload(
        campaign_manifest["campaign"],
        campaign_manifest["base_config"],
        completed_runs,
    )
    aggregate_path = campaign / "campaign_aggregate_test.json"
    aggregate_path.write_text(
        json.dumps(aggregate_payload),
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
    assert audit["status"] == "ready_to_start_campaign_e", audit["blockers"]
    assert audit["schema_version"] == "task4_campaign_p_before_e_gate_v7"
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
    aggregate_data = json.loads(aggregate.read_text(encoding="utf-8"))
    assert (
        "auxiliary_table_run_estimands"
        in aggregate_data["aggregate_metrics"]
    )
    assert audit["aggregate_metrics_match_canonical_leaf_rebuild"] is True
    assert (
        audit["observed_aggregate_metrics_sha256"]
        == audit["rebuilt_aggregate_metrics_sha256"]
    )
    assert (
        audit["runtime_homogeneity_match_canonical_leaf_rebuild"] is True
    )
    assert (
        audit["observed_runtime_homogeneity_sha256"]
        == audit["rebuilt_runtime_homogeneity_sha256"]
    )


def test_campaign_p_gate_rejects_noncanonical_embedding_cache_namespace(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    manifest_path = campaign / "runs" / "mixed__s2__a01" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["embedding"]["cache_namespace_template"] = (
        "/shared/embedding_cache/{agent_id}.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

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
    assert any(
        "embedding cache namespace mismatch" in blocker
        for blocker in audit["blockers"]
    )


def test_campaign_aggregate_cache_correction_changes_only_runtime_identity(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    original_root = PurePosixPath(str(campaign).replace("\\", "/"))
    for run_dir in sorted((campaign / "runs").iterdir()):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["embedding"]["cache_namespace_template"] = str(
            original_root
            / "runs"
            / run_dir.name
            / "embedding_cache"
            / "{agent_id}.json"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_data = json.loads(aggregate.read_text(encoding="utf-8"))
    aggregate_data["runtime_homogeneity"] = {
        "homogeneous": False,
        "run_count": 2,
        "mismatches": {"embedding": ["run-a-cache", "run-b-cache"]},
        "identity": None,
        "formal_aggregation_allowed": False,
    }
    aggregate.write_text(json.dumps(aggregate_data), encoding="utf-8")

    audit, corrected = _aggregate_correction_module().build_correction(
        campaign,
        original_campaign_dir=str(original_root),
        observed_aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_run_count=2,
    )

    assert audit["status"] == "verified_isolated_cache_path_correction"
    assert audit["blockers"] == []
    assert audit["aggregate_payload_unchanged_outside_runtime_homogeneity"] is True
    assert corrected["runtime_homogeneity"]["homogeneous"] is True
    assert (
        audit["observed_aggregate_metrics_sha256"]
        == audit["corrected_aggregate_metrics_sha256"]
    )


def test_campaign_aggregate_cache_correction_normalizes_json_seed_keys() -> None:
    module = _aggregate_correction_module()

    rebuilt = {"condition_units": {"fact": {2026072401: {"value": 1.0}}}}
    observed = {"condition_units": {"fact": {"2026072401": {"value": 1.0}}}}

    assert rebuilt != observed
    assert module._json_normalize(rebuilt) == observed


def test_campaign_aggregate_cache_correction_rejects_other_payload_changes(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    original_root = PurePosixPath(str(campaign).replace("\\", "/"))
    for run_dir in sorted((campaign / "runs").iterdir()):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["embedding"]["cache_namespace_template"] = str(
            original_root
            / "runs"
            / run_dir.name
            / "embedding_cache"
            / "{agent_id}.json"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_data = json.loads(aggregate.read_text(encoding="utf-8"))
    aggregate_data["runtime_homogeneity"] = {
        "homogeneous": False,
        "run_count": 2,
        "mismatches": {"embedding": ["run-a-cache", "run-b-cache"]},
        "identity": None,
        "formal_aggregation_allowed": False,
    }
    aggregate_data["completed_run_count"] = 999
    aggregate.write_text(json.dumps(aggregate_data), encoding="utf-8")

    audit, _corrected = _aggregate_correction_module().build_correction(
        campaign,
        original_campaign_dir=str(original_root),
        observed_aggregate_path=aggregate,
        expected_code_sha="expected-sha",
        expected_run_count=2,
    )

    assert audit["status"] == "no_go"
    assert any("outside runtime_homogeneity" in item for item in audit["blockers"])


def test_campaign_p_gate_accepts_failed_attempt_followed_by_complete_retry(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    retry = _copy_attempt(campaign, seed=1, attempt=2)
    state_path = campaign / "state.tsv"
    state = state_path.read_text(encoding="utf-8")
    first_complete = (
        f"t\tmixed\tmixed\t1\t1\tcomplete\tmixed__s1__a01\t"
        f"{campaign / 'runs' / 'mixed__s1__a01'}\t\t\n"
    )
    failed_then_retry = (
        f"t\tmixed\tmixed\t1\t1\tfailed\tmixed__s1__a01\t"
        f"{campaign / 'runs' / 'mixed__s1__a01'}\tinfrastructure\tfailed\n"
        f"t2\tmixed\tmixed\t1\t2\tcomplete\t{retry.name}\t{retry}\t\t\n"
    )
    state_path.write_text(
        state.replace(first_complete, failed_then_retry),
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

    assert audit["status"] == "ready_to_start_campaign_e", audit["blockers"]
    assert audit["failed_state_rows"] == 1
    assert audit["superseded_failed_state_rows"] == 1
    assert audit["latest_failed_matrix_units"] == 0
    assert any(item["run_id"] == retry.name for item in audit["leaf_evidence"])


def test_campaign_p_gate_rejects_multiple_completed_attempts(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    retry = _copy_attempt(campaign, seed=1, attempt=2)
    with (campaign / "state.tsv").open("a", encoding="utf-8") as handle:
        handle.write(
            f"t2\tmixed\tmixed\t1\t2\tcomplete\t{retry.name}\t{retry}\t\t\n"
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
    assert any(
        "multiple completed attempts" in item for item in audit["blockers"]
    )


def test_campaign_p_gate_rejects_nonpositive_attempt(tmp_path: Path) -> None:
    campaign, aggregate = _campaign(tmp_path)
    state = campaign / "state.tsv"
    state.write_text(
        state.read_text(encoding="utf-8").replace(
            "\t1\tcomplete\t",
            "\t0\tcomplete\t",
            1,
        ),
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
    assert any("malformed state rows: 1" in item for item in audit["blockers"])


def test_campaign_p_gate_rejects_failed_then_complete_same_attempt(
    tmp_path: Path,
) -> None:
    campaign, aggregate = _campaign(tmp_path)
    state = campaign / "state.tsv"
    complete = (
        f"t\tmixed\tmixed\t1\t1\tcomplete\tmixed__s1__a01\t"
        f"{campaign / 'runs' / 'mixed__s1__a01'}\t\t\n"
    )
    failed_then_complete = (
        f"t0\tmixed\tmixed\t1\t1\tfailed\tmixed__s1__a01\t"
        f"{campaign / 'runs' / 'mixed__s1__a01'}\tinfrastructure\tfailed\n"
        f"{complete}"
    )
    state.write_text(
        state.read_text(encoding="utf-8").replace(
            complete,
            failed_then_complete,
        ),
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
    assert any(
        "failed state precedes completion within latest attempt" in item
        for item in audit["blockers"]
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
    assert any(
        "runtime homogeneity does not match" in item
        for item in audit["blockers"]
    )


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
    data["schema_version"] = "wrong-schema"
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
    assert any("aggregate schema mismatch" in item for item in audit["blockers"])
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


def _copy_attempt(campaign: Path, *, seed: int, attempt: int) -> Path:
    source = campaign / "runs" / f"mixed__s{seed}__a01"
    run_id = f"mixed__s{seed}__a{attempt:02d}"
    destination = campaign / "runs" / run_id
    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_id
    manifest["output_dir"] = str(destination)
    manifest["config_snapshot_path"] = str(destination / "resolved_config.yaml")
    manifest["metadata"]["embedding"]["cache_namespace_template"] = str(
        destination / "embedding_cache" / "{agent_id}.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = destination / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["run_id"] = run_id
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return destination
