"""Build an immutable, fail-closed Campaign P gate before Campaign E starts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.evaluation.aggregation import (
    aggregate_metrics,
    validate_runtime_homogeneity,
)
from agentmemeval.evaluation.pilot import (
    PRIMARY_MDE_BB_PER_100,
    SENSITIVITY_MDES_BB_PER_100,
    calibrate_behavior_thresholds,
)
from agentmemeval.evaluation.statistics import estimate_paired_seed_requirement

REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)
EXPECTED_PAIRED_MECHANISMS = {
    "expr",
    "fact_expr_async",
    "fact_expr_sync",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    parser.add_argument("--expected-decision-version", required=True)
    parser.add_argument("--expected-decision-system-sha256", required=True)
    parser.add_argument("--expected-experience-update-sha256", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    aggregate_path = Path(args.aggregate).resolve()
    output = Path(args.output).resolve()
    audit = build_gate(
        campaign_dir,
        aggregate_path=aggregate_path,
        expected_code_sha=str(args.expected_code_sha),
        expected_max_model_len=int(args.expected_max_model_len),
        expected_prompts={
            "decision_version": str(args.expected_decision_version),
            "decision_system_sha256": str(args.expected_decision_system_sha256),
            "experience_update_sha256": str(
                args.expected_experience_update_sha256
            ),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "ready_to_start_campaign_e" else 2


def build_gate(
    campaign_dir: Path,
    *,
    aggregate_path: Path,
    expected_code_sha: str,
    expected_max_model_len: int,
    expected_prompts: dict[str, str],
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    aggregate_path = aggregate_path.resolve()
    manifest_path = campaign_dir / "campaign_manifest.json"
    state_path = campaign_dir / "state.tsv"
    manifest = _read_json(manifest_path)
    aggregate = _read_json(aggregate_path)
    campaign = manifest.get("campaign", {})
    conditions = campaign.get("conditions") or [
        {"condition_id": "mixed_table", "target_mechanism": "mixed"}
    ]
    seeds = campaign.get("seeds", [])
    expected = len(conditions) * len(seeds)
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    blockers: list[str] = []
    expected_identities = {
        (str(condition.get("condition_id", "")), int(seed))
        for condition in conditions
        if isinstance(condition, dict)
        for seed in seeds
    }
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    malformed_state_rows = 0
    for row in rows:
        try:
            identity = (str(row.get("condition_id", "")), int(row.get("seed", "")))
            int(row.get("attempt", ""))
        except (TypeError, ValueError):
            malformed_state_rows += 1
            continue
        grouped.setdefault(identity, []).append(row)
    if malformed_state_rows:
        blockers.append(f"malformed state rows: {malformed_state_rows}")
    extras = sorted(set(grouped) - expected_identities)
    missing = sorted(expected_identities - set(grouped))
    if extras:
        blockers.append(f"unexpected matrix identities: {extras}")
    if missing:
        blockers.append(f"missing matrix identities: {missing}")

    completed: list[dict[str, str]] = []
    latest_failed_matrix_units = 0
    superseded_failed_state_rows = 0
    for identity in sorted(expected_identities):
        identity_rows = grouped.get(identity, [])
        if not identity_rows:
            continue
        maximum_attempt = max(int(row["attempt"]) for row in identity_rows)
        latest_attempt_rows = [
            row for row in identity_rows if int(row["attempt"]) == maximum_attempt
        ]
        latest = latest_attempt_rows[-1]
        completed_attempts = {
            int(row["attempt"])
            for row in identity_rows
            if row.get("status") == "complete"
        }
        if len(completed_attempts) > 1:
            blockers.append(
                f"multiple completed attempts for {identity}: "
                f"{sorted(completed_attempts)}"
            )
        failed_in_latest_attempt = sum(
            row.get("status") == "failed" for row in latest_attempt_rows
        )
        if failed_in_latest_attempt and latest.get("status") == "complete":
            blockers.append(
                f"failed state precedes completion within latest attempt for "
                f"{identity}"
            )
        superseded_failed_state_rows += sum(
            row.get("status") == "failed"
            and int(row["attempt"]) < maximum_attempt
            for row in identity_rows
        )
        if latest.get("status") == "complete":
            completed.append(latest)
        else:
            if latest.get("status") == "failed":
                latest_failed_matrix_units += 1
            blockers.append(
                f"latest attempt is not complete for {identity}: "
                f"attempt={maximum_attempt}, status={latest.get('status')}"
            )
    if manifest.get("schema_version") != "agentmemeval_campaign_v1":
        blockers.append(
            f"campaign manifest schema mismatch: {manifest.get('schema_version')}"
        )
    if (
        aggregate_path.parent != campaign_dir
        or not aggregate_path.name.startswith("campaign_aggregate_")
        or aggregate_path.suffix != ".json"
    ):
        blockers.append(
            "aggregate must be a campaign_aggregate_*.json file directly "
            "inside campaign_dir"
        )
    if len(completed) != expected:
        blockers.append(f"complete matrix mismatch: {len(completed)}/{expected}")
    power_diagnostic = _campaign_p_power_diagnostic(
        aggregate,
        expected_run_count=expected,
        expected_code_sha=expected_code_sha,
        expected_seeds=[int(seed) for seed in seeds],
        expected_prompts=expected_prompts,
    )
    blockers.extend(str(item) for item in power_diagnostic["blockers"])

    metrics_list: list[dict[str, Any]] = []
    evaluation_target_ids_by_run: list[list[str]] = []
    leaf_evidence: list[dict[str, Any]] = []
    runtime_identities: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    runs_root = (campaign_dir / "runs").resolve()
    for row in completed:
        run_id = str(row.get("run_id", ""))
        row_run_dir = Path(str(row.get("run_dir", ""))).resolve()
        expected_run_dir = (campaign_dir / "runs" / run_id).resolve()
        if (
            not run_id
            or not expected_run_dir.is_relative_to(runs_root)
            or row_run_dir != expected_run_dir
        ):
            blockers.append(
                f"{run_id or '<missing-run-id>'} run_dir is not the canonical "
                f"campaign leaf: {row.get('run_dir')}"
            )
            continue
        run_dir = expected_run_dir
        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if not (run_dir / name).is_file() or (run_dir / name).stat().st_size < 1
        ]
        if missing:
            blockers.append(f"{row['run_id']} missing artifacts: {missing}")
            continue
        metrics = _read_json(run_dir / "metrics.json")
        protocol = _read_json(run_dir / "protocol_audit.json")
        run_manifest = _read_json(run_dir / "manifest.json")
        run_manifests.append(run_manifest)
        resolved_config = _read_yaml(run_dir / "resolved_config.yaml")
        _audit_leaf_identity(
            blockers,
            row=row,
            campaign_id=str(campaign.get("campaign_id", "")),
            run_dir=run_dir,
            run_manifest=run_manifest,
            resolved_config=resolved_config,
        )
        metrics_list.append(metrics)
        evaluation_targets = protocol.get("evaluation_target_ids", [])
        if not isinstance(evaluation_targets, list) or not evaluation_targets:
            blockers.append(f"{row['run_id']} lacks evaluation_target_ids")
            evaluation_target_ids_by_run.append([])
        else:
            evaluation_target_ids_by_run.append(
                [str(value) for value in evaluation_targets]
            )
        execution = protocol.get("execution_health", {})
        if not isinstance(execution, dict) or execution.get("valid") is not True:
            blockers.append(f"{row['run_id']} execution health invalid")
        else:
            if execution.get("status") != "passed":
                blockers.append(
                    f"{row['run_id']} execution health status is "
                    f"{execution.get('status')}"
                )
            for key in (
                "fallback_count",
                "memory_revision_fallback_count",
                "reward_conservation_violation_count",
                "stack_conservation_violation_count",
            ):
                observed_count = _safe_int(execution.get(key))
                if observed_count != 0:
                    blockers.append(
                        f"{row['run_id']} execution {key}: "
                        f"{execution.get(key)}"
                    )
        revision_fallbacks = _revision_fallback_count(metrics)
        if revision_fallbacks:
            blockers.append(
                f"{row['run_id']} deterministic experience revision fallbacks: "
                f"{revision_fallbacks}"
            )
        runtime = _runtime_identity(run_manifest)
        runtime_identities.append(runtime)
        observed_code = dict(runtime.get("code", {})).get("commit")
        if str(observed_code) != expected_code_sha:
            blockers.append(
                f"{row['run_id']} code SHA mismatch: {observed_code}"
            )
        if dict(runtime.get("code", {})).get("dirty") is not False:
            blockers.append(f"{row['run_id']} was created from a dirty worktree")
        service = json.loads(str(runtime.get("service", "{}")))
        observed_context = dict(service.get("service_startup_parameters", {})).get(
            "max_model_len"
        )
        if int(observed_context or 0) != expected_max_model_len:
            blockers.append(
                f"{row['run_id']} max_model_len mismatch: {observed_context}"
            )
        observed_prompts = dict(runtime.get("prompts", {}))
        for key, expected_value in expected_prompts.items():
            if str(observed_prompts.get(key)) != expected_value:
                blockers.append(
                    f"{row['run_id']} prompt identity mismatch for {key}: "
                    f"{observed_prompts.get(key)}"
                )
        leaf_evidence.append(
            {
                "condition_id": row["condition_id"],
                "seed": int(row["seed"]),
                "attempt": int(row["attempt"]),
                "run_id": row["run_id"],
                "run_dir": str(run_dir),
                "execution_health": execution,
                "memory_revision_fallback_count": revision_fallbacks,
                "sha256": {
                    name: _sha256(run_dir / name)
                    for name in REQUIRED_ARTIFACTS
                },
            }
        )

    canonical_runtime = {
        json.dumps(identity, ensure_ascii=False, sort_keys=True)
        for identity in runtime_identities
    }
    if len(canonical_runtime) != 1:
        blockers.append(
            f"runtime identity is not homogeneous: {len(canonical_runtime)} identities"
        )
    behavior = calibrate_behavior_thresholds(
        metrics_list,
        evaluation_target_ids_by_run,
    )
    if behavior.get("status") != "frozen":
        blockers.extend(str(item) for item in behavior.get("blockers", []))
        if not behavior.get("blockers"):
            blockers.append(f"behavior gate status is {behavior.get('status')}")
    rebuilt_aggregate_metrics = aggregate_metrics(metrics_list)
    observed_aggregate_metrics = aggregate.get("aggregate_metrics")
    observed_aggregate_metrics_sha256 = _json_sha256(observed_aggregate_metrics)
    rebuilt_aggregate_metrics_sha256 = _json_sha256(rebuilt_aggregate_metrics)
    aggregate_metrics_match = (
        observed_aggregate_metrics_sha256 == rebuilt_aggregate_metrics_sha256
    )
    if not aggregate_metrics_match:
        blockers.append(
            "aggregate metrics do not match a canonical rebuild from campaign leaves"
        )
    rebuilt_runtime_homogeneity = validate_runtime_homogeneity(run_manifests)
    observed_runtime_homogeneity = aggregate.get("runtime_homogeneity")
    observed_runtime_homogeneity_sha256 = _json_sha256(
        observed_runtime_homogeneity
    )
    rebuilt_runtime_homogeneity_sha256 = _json_sha256(
        rebuilt_runtime_homogeneity
    )
    runtime_homogeneity_match = (
        observed_runtime_homogeneity_sha256
        == rebuilt_runtime_homogeneity_sha256
    )
    if not runtime_homogeneity_match:
        blockers.append(
            "aggregate runtime homogeneity does not match a canonical rebuild "
            "from campaign leaf manifests"
        )

    return {
        "schema_version": "task4_campaign_p_before_e_gate_v7",
        "campaign_dir": str(campaign_dir),
        "campaign_id": campaign.get("campaign_id"),
        "expected_matrix_units": expected,
        "completed_matrix_units": len(completed),
        "failed_state_rows": sum(
            row.get("status") == "failed" for row in rows
        ),
        "superseded_failed_state_rows": superseded_failed_state_rows,
        "latest_failed_matrix_units": latest_failed_matrix_units,
        "ignored_noncomplete_state_rows": len(rows) - len(completed),
        "expected_code_sha": expected_code_sha,
        "expected_max_model_len": expected_max_model_len,
        "expected_prompts": expected_prompts,
        "runtime_identity": runtime_identities[0] if len(canonical_runtime) == 1 else None,
        "behavior_freeze_preview": behavior,
        "campaign_p_power_diagnostic": power_diagnostic,
        "campaign_manifest_sha256": _sha256(manifest_path),
        "state_tsv_sha256": _sha256(state_path),
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": _sha256(aggregate_path),
        "aggregate_metrics_match_canonical_leaf_rebuild": aggregate_metrics_match,
        "observed_aggregate_metrics_sha256": observed_aggregate_metrics_sha256,
        "rebuilt_aggregate_metrics_sha256": rebuilt_aggregate_metrics_sha256,
        "runtime_homogeneity_match_canonical_leaf_rebuild": (
            runtime_homogeneity_match
        ),
        "observed_runtime_homogeneity_sha256": (
            observed_runtime_homogeneity_sha256
        ),
        "rebuilt_runtime_homogeneity_sha256": (
            rebuilt_runtime_homogeneity_sha256
        ),
        "leaf_evidence": sorted(
            leaf_evidence, key=lambda item: (item["condition_id"], item["seed"])
        ),
        "blockers": blockers,
        "status": "ready_to_start_campaign_e" if not blockers else "no_go",
    }


def _campaign_p_power_diagnostic(
    aggregate: dict[str, Any],
    *,
    expected_run_count: int,
    expected_code_sha: str,
    expected_seeds: list[int],
    expected_prompts: dict[str, str],
) -> dict[str, Any]:
    """Audit P-side paired variance without claiming a joint P/E formal freeze."""

    blockers: list[str] = []
    if aggregate.get("schema_version") != "agentmemeval_campaign_aggregate_v1":
        blockers.append(
            f"aggregate schema mismatch: {aggregate.get('schema_version')}"
        )
    completed = _safe_int(aggregate.get("completed_run_count"))
    expected = _safe_int(aggregate.get("expected_run_count"))
    if completed != expected_run_count or expected != expected_run_count:
        blockers.append(
            "aggregate matrix mismatch: "
            f"completed={completed}, expected={expected}, gate={expected_run_count}"
        )
    if aggregate.get("status") != "descriptive_only":
        blockers.append("aggregate status must be descriptive_only")
    if aggregate.get("design") != "mixed_table":
        blockers.append(
            f"aggregate design mismatch: {aggregate.get('design')}"
        )
    homogeneity = aggregate.get("runtime_homogeneity", {})
    if not isinstance(homogeneity, dict) or homogeneity.get("homogeneous") is not True:
        blockers.append("aggregate runtime homogeneity is not verified")
    else:
        identity = homogeneity.get("identity", {})
        code = _pairs_to_dict(
            identity.get("code", {}) if isinstance(identity, dict) else {}
        )
        if str(code.get("commit")) != expected_code_sha:
            blockers.append(
                f"aggregate code SHA mismatch: {code.get('commit')}"
            )
        if code.get("dirty") is not False:
            blockers.append("aggregate runtime identity was dirty")
        prompts = _pairs_to_dict(
            identity.get("prompts", {}) if isinstance(identity, dict) else {}
        )
        for key, expected_value in expected_prompts.items():
            if str(prompts.get(key)) != expected_value:
                blockers.append(
                    f"aggregate prompt identity mismatch for {key}: "
                    f"{prompts.get(key)}"
                )
    paired = (
        aggregate.get("aggregate_metrics", {})
        .get("paired_estimand_descriptive", {})
    )
    if not isinstance(paired, dict) or paired.get("status") != "descriptive_only":
        blockers.append("aggregate paired estimand is unavailable")
        paired = {}
    if _safe_int(paired.get("independent_seed_count")) != expected_run_count:
        blockers.append(
            "paired estimand independent seed count mismatch: "
            f"{paired.get('independent_seed_count')}/{expected_run_count}"
        )
    paired_contract = {
        "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
        "endpoint": "final_test_bb_per_100",
        "baseline_mechanism": "fact",
        "multiple_comparison_method": "holm",
    }
    for key, expected_value in paired_contract.items():
        if paired.get(key) != expected_value:
            blockers.append(
                f"paired estimand {key} mismatch: {paired.get(key)}"
            )
    raw_matched_seeds = paired.get("matched_seeds")
    if not isinstance(raw_matched_seeds, list):
        blockers.append("paired estimand matched_seeds is unavailable")
    else:
        matched_seeds = [_safe_int(seed) for seed in raw_matched_seeds]
        if (
            any(seed is None for seed in matched_seeds)
            or matched_seeds != expected_seeds
        ):
            blockers.append(
                "paired estimand matched seeds mismatch: "
                f"{raw_matched_seeds}/{expected_seeds}"
            )
    effects_by_mechanism = paired.get("effects_by_mechanism", {})
    plans: dict[str, Any] = {}
    if not isinstance(effects_by_mechanism, dict) or not effects_by_mechanism:
        blockers.append("paired estimand has no mechanism effects")
    else:
        observed_mechanisms = {str(value) for value in effects_by_mechanism}
        if observed_mechanisms != EXPECTED_PAIRED_MECHANISMS:
            blockers.append(
                "paired estimand mechanisms mismatch: "
                f"{sorted(observed_mechanisms)}/"
                f"{sorted(EXPECTED_PAIRED_MECHANISMS)}"
            )
        for mechanism, raw_effects in sorted(effects_by_mechanism.items()):
            if not isinstance(raw_effects, list):
                blockers.append(f"{mechanism} paired effects is not a list")
                continue
            try:
                effects = [float(value) for value in raw_effects]
            except (TypeError, ValueError):
                blockers.append(
                    f"{mechanism} paired effects contains non-numeric values"
                )
                continue
            if not all(math.isfinite(value) for value in effects):
                blockers.append(
                    f"{mechanism} paired effects contains non-finite values"
                )
                continue
            if len(effects) != expected_run_count:
                blockers.append(
                    f"{mechanism} paired effects mismatch: "
                    f"{len(effects)}/{expected_run_count}"
                )
                continue
            plans[str(mechanism)] = {
                "effects": effects,
                "sensitivity_by_mde_bb_per_100": {
                    str(mde): estimate_paired_seed_requirement(effects, mde)
                    for mde in SENSITIVITY_MDES_BB_PER_100
                },
            }
    primary_requirements = [
        int(
            plan["sensitivity_by_mde_bb_per_100"][
                str(PRIMARY_MDE_BB_PER_100)
            ]["required_seed_pairs_normal_approximation"]
        )
        for plan in plans.values()
    ]
    return {
        "schema_version": "task4_campaign_p_power_diagnostic_v1",
        "primary_endpoint": "final_test_bb_per_100",
        "primary_mde_bb_per_100": PRIMARY_MDE_BB_PER_100,
        "sensitivity_mdes_bb_per_100": list(SENSITIVITY_MDES_BB_PER_100),
        "alpha": 0.05,
        "power": 0.80,
        "planning_method": "paired_normal_approximation_for_planning_only",
        "contrasts": plans,
        "required_seed_pairs_primary_max_across_p_only": (
            max(primary_requirements)
            if primary_requirements and not blockers
            else None
        ),
        "joint_p_e_power_freeze_complete": False,
        "joint_freeze_note": (
            "Campaign E independent pilot is still required before the joint "
            "P/E formal seed plan can be frozen."
        ),
        "blockers": blockers,
        "status": (
            "p_side_power_diagnostic_ready_not_joint_freeze"
            if not blockers
            else "blocked_invalid_or_incomplete_campaign_p_aggregate"
        ),
    }


def _pairs_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        try:
            return {str(key): item for key, item in value}
        except (TypeError, ValueError):
            return {}
    return {}


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    return {
        "code": metadata.get("code", {}),
        "gpu": metadata.get("gpu", {}),
        "model_service_runtime": metadata.get("model_service_runtime", {}),
        "model": metadata.get("model", {}),
        "service": json.dumps(
            metadata.get("service", {}), ensure_ascii=False, sort_keys=True
        ),
        "embedding": json.dumps(
            metadata.get("embedding", {}), ensure_ascii=False, sort_keys=True
        ),
        "prompts": metadata.get("prompts", {}),
    }


def _revision_fallback_count(metrics: dict[str, Any]) -> int:
    primary = metrics.get("primary_metrics", {})
    stage_per_agent = primary.get("stage_per_agent", {})
    tables = (
        list(stage_per_agent.values())
        if isinstance(stage_per_agent, dict) and stage_per_agent
        else [primary.get("per_agent", {})]
    )
    maximum_by_agent: dict[str, int] = {}
    for per_agent in tables:
        if not isinstance(per_agent, dict):
            continue
        for agent_id, values in per_agent.items():
            memory = values.get("memory", {}) if isinstance(values, dict) else {}
            count = int(memory.get("revision_fallback_count", 0))
            key = str(agent_id)
            maximum_by_agent[key] = max(maximum_by_agent.get(key, 0), count)
    return sum(maximum_by_agent.values())


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML object: {path}")
    return data


def _audit_leaf_identity(
    blockers: list[str],
    *,
    row: dict[str, str],
    campaign_id: str,
    run_dir: Path,
    run_manifest: dict[str, Any],
    resolved_config: dict[str, Any],
) -> None:
    run_id = str(row.get("run_id", ""))
    seed = _safe_int(row.get("seed"))
    manifest_contract = {
        "run_id": run_id,
        "seed": seed,
        "output_dir": str(run_dir),
        "config_snapshot_path": str(run_dir / "resolved_config.yaml"),
    }
    for key, expected in manifest_contract.items():
        observed = run_manifest.get(key)
        if observed != expected:
            blockers.append(
                f"{run_id} manifest {key} mismatch: {observed}/{expected}"
            )

    experiment = resolved_config.get("experiment")
    if not isinstance(experiment, dict):
        blockers.append(f"{run_id} resolved config lacks experiment object")
        return
    config_contract = {
        "campaign_id": campaign_id,
        "campaign_condition_id": str(row.get("condition_id", "")),
        "seed": seed,
        "run_id": run_id,
    }
    for key, expected in config_contract.items():
        observed = experiment.get(key)
        if observed != expected:
            blockers.append(
                f"{run_id} resolved config {key} mismatch: {observed}/{expected}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
