"""Independent-pilot power planning without silently authorizing formal runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.evaluation.degeneracy import (
    evaluate_behavior_health,
    revision_fallback_count,
)
from agentmemeval.evaluation.relevance_review import REVIEW_POLICY
from agentmemeval.evaluation.statistics import estimate_paired_seed_requirement

PRIMARY_MDE_BB_PER_100 = 5.0
SENSITIVITY_MDES_BB_PER_100 = (3.0, 5.0, 10.0)
BEHAVIOR_FREEZE_POLICY = {
    "lower_quantile": 0.05,
    "upper_quantile": 0.95,
    "rate_margin": 0.02,
    "concentration_margin": 0.05,
    "single_hand_reward_activity_role": "diagnostic_only_not_behavior_exclusion",
    "domain_floors": {"min_vpip": 0.02},
    "domain_caps": {
        "max_fold_rate": 0.98,
        "max_all_in_hand_rate": 0.50,
        "max_bust_hand_rate": 0.50,
        "max_single_hand_reward_activity_share": 0.75,
        "max_empty_retrieval_rate": 0.98,
        "max_structural_signature_share": 0.95,
    },
}
PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS = {
    # Documentation, tests, and Pilot/Formal post-processing do not participate
    # in table play. Keep this list exact rather than allowing whole directories.
    "README.md",
    "configs/campaigns/task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml",
    "configs/campaigns/task4_campaign_p_strict_model_substituted.yaml",
    "src/agentmemeval/cli/main.py",
    "src/agentmemeval/evaluation/campaign_progress.py",
    "src/agentmemeval/evaluation/campaign_reporting.py",
    "src/agentmemeval/evaluation/formal_freeze.py",
    "src/agentmemeval/evaluation/pilot.py",
    "src/agentmemeval/evaluation/relevance_review.py",
    "src/agentmemeval/evaluation/runtime_lock.py",
    "src/agentmemeval/experiments/campaign.py",
    # The admission change is confined to the formal/frozen-preflight branch;
    # ordinary Pilot admission returns before the V2 runtime-lock check.
    "src/agentmemeval/experiments/admission.py",
    "src/agentmemeval/storage/archive.py",
    "src/agentmemeval/storage/run_map.py",
    "src/agentmemeval/storage/snapshot_archive.py",
    "tests/unit/test_archive_manifest.py",
    "tests/unit/test_campaign_p_gate.py",
    "tests/unit/test_campaign_progress.py",
    "tests/unit/test_campaign_reporting.py",
    "tests/unit/test_config_validation.py",
    "tests/unit/test_formal_freeze.py",
    "tests/unit/test_pilot_power_plan.py",
    "tests/unit/test_protocol_admission.py",
    "tests/unit/test_relevance_review.py",
    "tests/unit/test_run_map.py",
    "tests/unit/test_runtime_lock.py",
    "tests/unit/test_snapshot_archive.py",
    "tests/integration/test_campaign.py",
    "tools/task4/audit_pilot_prelaunch_code_paths.py",
    "tools/task4/audit_pilot_runtime_equivalence.py",
    "tools/task4/build_formal_runtime_lock.py",
    "tools/task4/campaign_progress.py",
    "tools/task4/gate_campaign_p_before_e.py",
    "tools/task4/retrieval_relevance_review.py",
    "tools/task4/snapshot_archive.py",
    "tools/task4/start_campaign_e_v7_pilot.sh",
}
PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256 = {
    # The only execution-adjacent P→E change is the reviewed seed-major matrix
    # scheduler. Binding the exact Git patch prevents this path-level exception
    # from authorizing any later leaf execution or aggregation change.
    "src/agentmemeval/experiments/campaign.py": (
        "f261385922bde8f0294c164d6990b5bf5a424032e67748574b8c429d774141c3"
    ),
}
EXECUTION_ZERO_FIELDS = (
    "fallback_count",
    "memory_revision_fallback_count",
    "reward_conservation_violation_count",
    "stack_conservation_violation_count",
)
EXPECTED_P_MECHANISMS = {
    "expr",
    "fact_expr_async",
    "fact_expr_sync",
}
EXPECTED_E_CONDITIONS = {
    "fact_target",
    "expr_target",
    "sync_target",
    "async_target",
}


def build_pilot_power_plan(
    campaign_p: dict[str, Any],
    campaign_e: dict[str, Any],
    runtime_equivalence_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the pre-registered P/E seed plan from complete pilot aggregates."""

    blockers: list[str] = []
    _validate_pilot_aggregate(campaign_p, "campaign_p", blockers)
    _validate_pilot_aggregate(campaign_e, "campaign_e", blockers)
    p_runtime = campaign_p.get("runtime_homogeneity", {})
    e_runtime = campaign_e.get("runtime_homogeneity", {})
    p_identity = p_runtime.get("identity") if isinstance(p_runtime, dict) else None
    e_identity = e_runtime.get("identity") if isinstance(e_runtime, dict) else None
    runtime_identity_mode = "missing_or_invalid"
    if not isinstance(p_identity, dict) or not isinstance(e_identity, dict):
        blockers.append("campaign P/E runtime identities are missing")
    elif p_identity != e_identity:
        runtime_blockers = _runtime_equivalence_blockers(
            p_identity,
            e_identity,
            runtime_equivalence_audit,
        )
        blockers.extend(runtime_blockers)
        runtime_identity_mode = (
            "pilot_only_verified_execution_equivalence"
            if not runtime_blockers
            else "mismatch_unverified"
        )
    else:
        runtime_identity_mode = "exact_identity"
    contrasts: dict[str, list[float]] = {}
    p_estimand_container = (
        campaign_p.get("aggregate_metrics", {})
        .get("paired_estimand_descriptive", {})
    )
    p_estimand = (
        p_estimand_container.get("effects_by_mechanism", {})
        if isinstance(p_estimand_container, dict)
        else {}
    )
    e_comparisons = campaign_e.get("paired_comparisons", {})
    endpoint = str(campaign_e.get("primary_endpoint", "final_test_bb_per_100"))
    p_seeds = _validate_power_contracts(
        campaign_p,
        campaign_e,
        p_estimand_container,
        p_estimand,
        e_comparisons,
        endpoint,
        blockers,
    )
    if isinstance(p_estimand, dict):
        for mechanism, effects in sorted(p_estimand.items()):
            parsed_effects = _float_list(effects)
            if len(parsed_effects) != len(p_seeds):
                blockers.append(
                    f"campaign_p:{mechanism}_vs_fact effect count mismatch: "
                    f"{len(parsed_effects)}/{len(p_seeds)}"
                )
            contrasts[f"campaign_p:{mechanism}_vs_fact"] = parsed_effects

    if isinstance(e_comparisons, dict):
        for condition, comparison in sorted(e_comparisons.items()):
            metric = (
                comparison.get("metrics", {}).get(endpoint, {})
                if isinstance(comparison, dict)
                else {}
            )
            effects = _float_list(
                metric.get("effects", {}) if isinstance(metric, dict) else {}
            )
            matched_seeds = _int_list(
                metric.get("matched_seeds", {}) if isinstance(metric, dict) else {}
            )
            if matched_seeds != p_seeds:
                blockers.append(
                    f"campaign_e:{condition}_vs_no_memory matched seeds mismatch: "
                    f"{matched_seeds}/{p_seeds}"
                )
            if len(effects) != len(p_seeds):
                blockers.append(
                    f"campaign_e:{condition}_vs_no_memory effect count mismatch: "
                    f"{len(effects)}/{len(p_seeds)}"
                )
            contrasts[f"campaign_e:{condition}_vs_no_memory"] = effects

    if not contrasts:
        blockers.append("pilot aggregates contain no paired primary-endpoint contrasts")
    plans: dict[str, dict[str, Any]] = {}
    for name, effects in contrasts.items():
        if len(effects) < 2:
            blockers.append(f"{name} has fewer than two paired pilot effects")
            continue
        sensitivity = {
            str(mde): estimate_paired_seed_requirement(effects, mde)
            for mde in SENSITIVITY_MDES_BB_PER_100
        }
        plans[name] = {
            "effects": effects,
            "sensitivity_by_mde_bb_per_100": sensitivity,
        }

    primary_requirements = [
        int(
            plan["sensitivity_by_mde_bb_per_100"][str(PRIMARY_MDE_BB_PER_100)][
                "required_seed_pairs_normal_approximation"
            ]
        )
        for plan in plans.values()
    ]
    required = max(primary_requirements) if primary_requirements and not blockers else None
    return {
        "schema_version": "agentmemeval_pilot_power_plan_v1",
        "primary_endpoint": "final_test_bb_per_100",
        "primary_mde_bb_per_100": PRIMARY_MDE_BB_PER_100,
        "sensitivity_mdes_bb_per_100": list(SENSITIVITY_MDES_BB_PER_100),
        "alpha": 0.05,
        "power": 0.80,
        "contrasts": plans,
        "required_seed_pairs_primary_max_across_p_and_e": required,
        "blockers": blockers,
        "status": (
            "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
            if not blockers
            else "blocked_invalid_or_incomplete_pilot"
        ),
        "planning_method": "paired_normal_approximation_for_planning_only",
        "no_silent_resource_cap": True,
        "runtime_identity_mode": runtime_identity_mode,
        "runtime_equivalence_audit_schema_version": (
            runtime_equivalence_audit.get("schema_version")
            if runtime_identity_mode == "pilot_only_verified_execution_equivalence"
            and runtime_equivalence_audit is not None
            else None
        ),
        "formal_homogeneity_not_granted": (
            runtime_identity_mode == "pilot_only_verified_execution_equivalence"
        ),
    }


def build_pilot_freeze_proposal(
    campaign_p: dict[str, Any],
    campaign_e: dict[str, Any],
    campaign_p_metrics: list[dict[str, Any]],
    campaign_p_protocol_audits: list[dict[str, Any]],
    campaign_e_metrics: list[dict[str, Any]],
    campaign_e_protocol_audits: list[dict[str, Any]],
    retrieval_review_audit: dict[str, Any],
    runtime_equivalence_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine power, behavior, execution, and retrieval freeze gates."""

    power_plan = build_pilot_power_plan(
        campaign_p,
        campaign_e,
        runtime_equivalence_audit,
    )
    p_targets, p_target_blockers = _evaluation_target_ids_by_run(
        campaign_p_protocol_audits,
        "campaign_p",
    )
    e_targets, e_target_blockers = _evaluation_target_ids_by_run(
        campaign_e_protocol_audits,
        "campaign_e",
    )
    target_blockers = [*p_target_blockers, *e_target_blockers]
    behavior = (
        calibrate_behavior_thresholds(
            [*campaign_p_metrics, *campaign_e_metrics],
            [*p_targets, *e_targets],
            [
                *[
                    f"campaign_p run {index}"
                    for index in range(len(campaign_p_metrics))
                ],
                *[
                    f"campaign_e run {index}"
                    for index in range(len(campaign_e_metrics))
                ],
            ],
        )
        if not target_blockers
        else {
            "status": "blocked_missing_evaluation_targets",
            "blockers": target_blockers,
            "policy": BEHAVIOR_FREEZE_POLICY,
            "thresholds": {},
        }
    )
    execution_blockers = _execution_health_blockers(
        campaign_p_protocol_audits,
        "campaign_p",
    )
    execution_blockers.extend(
        _execution_health_blockers(
            campaign_e_protocol_audits,
            "campaign_e",
        )
    )
    retrieval_blockers = [
        str(item) for item in retrieval_review_audit.get("blockers", [])
    ]
    if retrieval_review_audit.get("review_status") != "human_labels_verified":
        retrieval_blockers.append("retrieval relevance review lacks verified human labels")
    if retrieval_review_audit.get("retrieval_threshold_status") != "frozen":
        retrieval_blockers.append("retrieval relevance threshold is not frozen")
    if (
        retrieval_review_audit.get("schema_version")
        != "task4_retrieval_relevance_audit_v2"
    ):
        retrieval_blockers.append("retrieval relevance audit schema is not V2")
    if set(retrieval_review_audit.get("source_designs", [])) != {
        "mixed_table",
        "target_vs_seven_no_memory",
    }:
        retrieval_blockers.append(
            "retrieval relevance audit lacks complete P/E source designs"
        )
    if _safe_int(retrieval_review_audit.get("source_campaign_count")) != 2:
        retrieval_blockers.append(
            "retrieval relevance audit source campaign count is not 2"
        )
    source_evidence = retrieval_review_audit.get("source_evidence", [])
    if (
        not isinstance(source_evidence, list)
        or len(source_evidence) != 2
        or any(
            not isinstance(source, dict)
            or source.get("matrix_complete") is not True
            or not str(source.get("campaign_dir", "")).strip()
            or _safe_int(source.get("expected_state_rows")) is None
            or _safe_int(source.get("expected_state_rows")) < 1
            or _safe_int(source.get("completed_state_rows"))
            != _safe_int(source.get("expected_state_rows"))
            or not _is_sha256(source.get("campaign_manifest_sha256"))
            or not _is_sha256(source.get("state_tsv_sha256"))
            or not isinstance(source.get("event_sources"), list)
            or len(source.get("event_sources", []))
            != _safe_int(source.get("expected_state_rows"))
            or any(
                not isinstance(event, dict)
                or not str(event.get("run_id", "")).strip()
                or not _is_sha256(event.get("events_sha256"))
                for event in source.get("event_sources", [])
            )
            for source in source_evidence
        )
    ):
        retrieval_blockers.append(
            "retrieval relevance audit source evidence is incomplete"
        )
    if retrieval_review_audit.get("source_rebuild_verified") is not True:
        retrieval_blockers.append(
            "retrieval relevance review pack was not verified by source rebuild"
        )
    if (
        retrieval_review_audit.get("source_rebuild_content_sha256")
        != retrieval_review_audit.get("review_pack_content_sha256")
    ):
        retrieval_blockers.append(
            "retrieval relevance source-rebuild hash does not match review pack"
        )
    if retrieval_review_audit.get("review_policy_sha256") != _json_sha256(
        REVIEW_POLICY
    ):
        retrieval_blockers.append(
            "retrieval relevance audit review policy hash is invalid"
        )
    pack_hash = retrieval_review_audit.get("review_pack_content_sha256")
    if not _is_sha256(pack_hash):
        retrieval_blockers.append(
            "retrieval relevance review-pack content hash is invalid"
        )
    evidence = retrieval_review_audit.get("input_evidence", {})
    if not isinstance(evidence, dict):
        retrieval_blockers.append("retrieval relevance input evidence is missing")
    else:
        for key in ("review_key_sha256", "labels_sha256"):
            if not _is_sha256(evidence.get(key)):
                retrieval_blockers.append(
                    f"retrieval relevance input evidence {key} is invalid"
                )
        if _safe_int(evidence.get("label_row_count")) != _safe_int(
            retrieval_review_audit.get("labeled_row_count")
        ):
            retrieval_blockers.append(
                "retrieval relevance label-row evidence count mismatch"
            )
        if (_safe_int(evidence.get("human_reviewer_count")) or 0) < 1:
            retrieval_blockers.append(
                "retrieval relevance audit has no bound human reviewer"
            )
        reviewer_hashes = evidence.get("human_reviewer_ids_sha256", [])
        if (
            not isinstance(reviewer_hashes, list)
            or len(reviewer_hashes)
            != (_safe_int(evidence.get("human_reviewer_count")) or 0)
            or any(
                not _is_sha256(value)
                for value in reviewer_hashes
            )
        ):
            retrieval_blockers.append(
                "retrieval relevance reviewer identity hashes are invalid"
            )
    retrieval_score = retrieval_review_audit.get("minimum_retrieval_score")
    if retrieval_score is None:
        retrieval_blockers.append("retrieval relevance audit has no selected threshold")
    execution_blockers.extend(
        f"campaign_p run {index} used deterministic experience revision fallback"
        for index, metrics in enumerate(campaign_p_metrics)
        if revision_fallback_count(metrics) > 0
    )
    execution_blockers.extend(
        f"campaign_e run {index} used deterministic experience revision fallback"
        for index, metrics in enumerate(campaign_e_metrics)
        if revision_fallback_count(metrics) > 0
    )
    expected_p = int(campaign_p.get("expected_run_count", 0))
    expected_e = int(campaign_e.get("expected_run_count", 0))
    if len(campaign_p_metrics) != expected_p:
        execution_blockers.append(
            f"campaign_p metrics count mismatch: {len(campaign_p_metrics)}/{expected_p}"
        )
    if len(campaign_p_protocol_audits) != expected_p:
        execution_blockers.append(
            "campaign_p protocol audit count mismatch: "
            f"{len(campaign_p_protocol_audits)}/{expected_p}"
        )
    if len(campaign_e_protocol_audits) != expected_e:
        execution_blockers.append(
            "campaign_e protocol audit count mismatch: "
            f"{len(campaign_e_protocol_audits)}/{expected_e}"
        )
    if len(campaign_e_metrics) != expected_e:
        execution_blockers.append(
            f"campaign_e metrics count mismatch: {len(campaign_e_metrics)}/{expected_e}"
        )
    blockers = [
        *list(power_plan["blockers"]),
        *list(behavior["blockers"]),
        *execution_blockers,
        *retrieval_blockers,
    ]
    return {
        "schema_version": "agentmemeval_pilot_freeze_proposal_v1",
        "power_plan": power_plan,
        "behavior_freeze": behavior,
        "retrieval_freeze": {
            "retrieval_threshold_status": (
                "frozen" if not retrieval_blockers else "blocked"
            ),
            "minimum_retrieval_score": retrieval_score,
            "reason": "independent outcome-blind human relevance review",
            "review_audit_schema_version": retrieval_review_audit.get(
                "schema_version"
            ),
            "source_designs": retrieval_review_audit.get("source_designs"),
            "review_key_sha256": (
                evidence.get("review_key_sha256")
                if isinstance(evidence, dict)
                else None
            ),
            "labels_sha256": (
                evidence.get("labels_sha256")
                if isinstance(evidence, dict)
                else None
            ),
        },
        "execution_blockers": execution_blockers,
        "retrieval_blockers": retrieval_blockers,
        "required_seed_pairs": power_plan[
            "required_seed_pairs_primary_max_across_p_and_e"
        ],
        "blockers": blockers,
        "status": (
            "ready_to_generate_immutable_formal_configs"
            if not blockers
            else "no_go_pilot_freeze_blocked"
        ),
    }


def build_pilot_freeze_proposal_from_paths(
    campaign_p_aggregate_path: str | Path,
    campaign_e_aggregate_path: str | Path,
    campaign_p_dir: str | Path,
    campaign_e_dir: str | Path,
    retrieval_review_audit_path: str | Path,
    runtime_equivalence_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load only completed P/E leaf evidence and build the immutable proposal."""

    p_aggregate = _read_json(Path(campaign_p_aggregate_path))
    e_aggregate = _read_json(Path(campaign_e_aggregate_path))
    p_completed, p_evidence = _completed_state_rows(campaign_p_dir, "campaign_p")
    e_completed, e_evidence = _completed_state_rows(campaign_e_dir, "campaign_e")
    metrics = [_read_json(Path(row["run_dir"]) / "metrics.json") for row in p_completed]
    p_audits = [
        _read_json(Path(row["run_dir"]) / "protocol_audit.json")
        for row in p_completed
    ]
    e_audits = [
        _read_json(Path(row["run_dir"]) / "protocol_audit.json")
        for row in e_completed
    ]
    e_metrics = [_read_json(Path(row["run_dir"]) / "metrics.json") for row in e_completed]
    retrieval_review = _read_json(Path(retrieval_review_audit_path))
    runtime_equivalence = (
        _read_json(Path(runtime_equivalence_audit_path))
        if runtime_equivalence_audit_path is not None
        else None
    )
    proposal = build_pilot_freeze_proposal(
        p_aggregate,
        e_aggregate,
        metrics,
        p_audits,
        e_metrics,
        e_audits,
        retrieval_review,
        runtime_equivalence,
    )
    proposal["campaign_p_evidence"] = p_evidence
    proposal["campaign_e_evidence"] = e_evidence
    proposal["campaign_p_aggregate_evidence"] = {
        "path": str(Path(campaign_p_aggregate_path).resolve()),
        "sha256": _sha256_file(Path(campaign_p_aggregate_path)),
        "schema_version": p_aggregate.get("schema_version"),
    }
    proposal["campaign_e_aggregate_evidence"] = {
        "path": str(Path(campaign_e_aggregate_path).resolve()),
        "sha256": _sha256_file(Path(campaign_e_aggregate_path)),
        "schema_version": e_aggregate.get("schema_version"),
    }
    proposal["campaign_p_leaf_evidence"] = _leaf_evidence(p_completed)
    proposal["campaign_e_leaf_evidence"] = _leaf_evidence(e_completed)
    proposal["retrieval_review_evidence"] = {
        "path": str(Path(retrieval_review_audit_path).resolve()),
        "sha256": _sha256_file(Path(retrieval_review_audit_path)),
        "schema_version": retrieval_review.get("schema_version"),
    }
    proposal["runtime_equivalence_evidence"] = (
        {
            "path": str(Path(runtime_equivalence_audit_path).resolve()),
            "sha256": _sha256_file(Path(runtime_equivalence_audit_path)),
            "schema_version": runtime_equivalence.get("schema_version"),
        }
        if runtime_equivalence_audit_path is not None
        and runtime_equivalence is not None
        else None
    )
    return proposal


def build_pilot_runtime_equivalence_audit(
    campaign_p: dict[str, Any],
    campaign_e: dict[str, Any],
    changed_paths: list[str],
    changed_path_diff_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Prove Pilot execution equivalence across orchestration-only commits."""

    blockers: list[str] = []
    p_identity = dict(campaign_p.get("runtime_homogeneity", {})).get("identity")
    e_identity = dict(campaign_e.get("runtime_homogeneity", {})).get("identity")
    if not isinstance(p_identity, dict) or not isinstance(e_identity, dict):
        blockers.append("campaign P/E runtime identities are missing")
        p_identity = {}
        e_identity = {}
    p_code = _pairs_to_dict(p_identity.get("code", {}))
    e_code = _pairs_to_dict(e_identity.get("code", {}))
    p_commit = str(p_code.get("commit", ""))
    e_commit = str(e_code.get("commit", ""))
    if not p_commit or not e_commit:
        blockers.append("campaign P/E code commits are missing")
    if p_code.get("dirty") is not False or e_code.get("dirty") is not False:
        blockers.append("campaign P/E runtime identity includes a dirty worktree")
    p_non_code = _without_code_identity(p_identity)
    e_non_code = _without_code_identity(e_identity)
    if p_non_code != e_non_code:
        blockers.append("campaign P/E non-code runtime identities differ")
    normalized_paths: list[str] = []
    invalid_paths: list[str] = []
    for path in changed_paths:
        candidate = str(path).replace("\\", "/")
        parts = PurePosixPath(candidate).parts
        if (
            not candidate
            or candidate.startswith(("/", "./"))
            or re.match(r"^[A-Za-z]:/", candidate)
            or any(part in {".", ".."} for part in parts)
        ):
            invalid_paths.append(candidate)
            continue
        normalized_paths.append(candidate)
    normalized_paths = sorted(set(normalized_paths))
    invalid_paths = sorted(set(invalid_paths))
    if invalid_paths:
        blockers.append(f"invalid changed paths: {invalid_paths}")
    disallowed = sorted(
        set(normalized_paths) - PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS
    )
    if disallowed:
        blockers.append(f"execution-relevant or unregistered paths changed: {disallowed}")
    diff_hash_blockers = _required_diff_hash_blockers(
        normalized_paths,
        changed_path_diff_sha256,
    )
    blockers.extend(diff_hash_blockers)
    return {
        "schema_version": "task4_pilot_runtime_equivalence_audit_v1",
        "campaign_p_code_sha": p_commit,
        "campaign_e_code_sha": e_commit,
        "campaign_p_runtime_identity_sha256": _json_sha256(p_identity),
        "campaign_e_runtime_identity_sha256": _json_sha256(e_identity),
        "non_code_runtime_identity_sha256": (
            _json_sha256(p_non_code) if p_non_code == e_non_code else None
        ),
        "changed_paths": normalized_paths,
        "invalid_changed_paths": invalid_paths,
        "allowed_changed_paths_policy": sorted(
            PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS
        ),
        "disallowed_changed_paths": disallowed,
        "changed_path_diff_sha256": dict(changed_path_diff_sha256 or {}),
        "required_diff_sha256_policy": dict(
            PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256
        ),
        "required_diff_sha256_match": not diff_hash_blockers,
        "formal_homogeneity_not_granted": True,
        "blockers": blockers,
        "status": (
            "verified_execution_runtime_equivalent_for_pilot_power_only"
            if not blockers
            else "no_go_runtime_equivalence_unverified"
        ),
    }


def build_pilot_prelaunch_code_audit(
    campaign_p_code_sha: str,
    campaign_e_code_sha: str,
    changed_paths: list[str],
    changed_path_diff_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed on unregistered code changes before Campaign E starts."""

    blockers: list[str] = []
    sha_pattern = re.compile(r"[0-9a-f]{40}")
    if sha_pattern.fullmatch(campaign_p_code_sha) is None:
        blockers.append("campaign P code SHA must be a full lowercase SHA-1")
    if sha_pattern.fullmatch(campaign_e_code_sha) is None:
        blockers.append("campaign E code SHA must be a full lowercase SHA-1")
    normalized_paths: list[str] = []
    invalid_paths: list[str] = []
    for path in changed_paths:
        candidate = str(path).replace("\\", "/")
        parts = PurePosixPath(candidate).parts
        if (
            not candidate
            or candidate.startswith(("/", "./"))
            or re.match(r"^[A-Za-z]:/", candidate)
            or any(part in {".", ".."} for part in parts)
        ):
            invalid_paths.append(candidate)
            continue
        normalized_paths.append(candidate)
    normalized_paths = sorted(set(normalized_paths))
    invalid_paths = sorted(set(invalid_paths))
    if invalid_paths:
        blockers.append(f"invalid changed paths: {invalid_paths}")
    disallowed = sorted(
        set(normalized_paths) - PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS
    )
    if disallowed:
        blockers.append(f"execution-relevant or unregistered paths changed: {disallowed}")
    diff_hash_blockers = _required_diff_hash_blockers(
        normalized_paths,
        changed_path_diff_sha256,
    )
    blockers.extend(diff_hash_blockers)
    return {
        "schema_version": "task4_pilot_prelaunch_code_path_audit_v1",
        "campaign_p_code_sha": campaign_p_code_sha,
        "campaign_e_code_sha": campaign_e_code_sha,
        "changed_paths": normalized_paths,
        "invalid_changed_paths": invalid_paths,
        "allowed_changed_paths_policy": sorted(
            PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS
        ),
        "allowed_changed_paths_policy_sha256": _json_sha256(
            sorted(PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS)
        ),
        "disallowed_changed_paths": disallowed,
        "changed_path_diff_sha256": dict(changed_path_diff_sha256 or {}),
        "required_diff_sha256_policy": dict(
            PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256
        ),
        "required_diff_sha256_match": not diff_hash_blockers,
        "runtime_equivalence_not_yet_granted": True,
        "formal_homogeneity_not_granted": True,
        "blockers": blockers,
        "status": (
            "verified_code_paths_safe_to_launch_campaign_e_pilot"
            if not blockers
            else "no_go_code_paths_changed"
        ),
    }


def _completed_state_rows(
    campaign_dir: str | Path, label: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    directory = Path(campaign_dir).resolve()
    state_path = directory / "state.tsv"
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        local_run = directory / "runs" / str(row.get("run_id", ""))
        if local_run.is_dir():
            row["run_dir"] = str(local_run.resolve())
    completed = [row for row in rows if row.get("status") == "complete"]
    identities = [
        (str(row.get("condition_id", "")), str(row.get("seed", "")))
        for row in completed
    ]
    duplicate_identities = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    if duplicate_identities:
        raise ValueError(f"{label} has duplicate completed matrix units: {duplicate_identities}")
    return completed, {
        "campaign_dir": str(directory),
        "completed_state_rows": len(completed),
        "unique_completed_matrix_units": len(set(identities)),
        "completed_seeds": sorted(
            {int(row["seed"]) for row in completed if str(row.get("seed", "")).isdigit()}
        ),
        "ignored_noncomplete_state_rows": len(rows) - len(completed),
    }


def calibrate_behavior_thresholds(
    metrics_list: list[dict[str, Any]],
    agent_ids_by_run: list[list[str]] | None = None,
    run_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the outcome-independent quantile-plus-domain-cap freeze policy."""

    if agent_ids_by_run is not None and len(agent_ids_by_run) != len(metrics_list):
        raise ValueError(
            "agent_ids_by_run length must match metrics_list length: "
            f"{len(agent_ids_by_run)} != {len(metrics_list)}"
        )
    if run_labels is not None and len(run_labels) != len(metrics_list):
        raise ValueError(
            "run_labels length must match metrics_list length: "
            f"{len(run_labels)} != {len(metrics_list)}"
        )
    samples: dict[str, list[float]] = {
        "vpip": [],
        "fold_rate": [],
        "all_in_hand_rate": [],
        "bust_hand_rate": [],
        "single_hand_reward_activity_share": [],
        "empty_retrieval_rate": [],
        "structural_signature_share": [],
    }
    evaluated_ids_by_run: list[list[str] | str] = []
    for run_index, metrics in enumerate(metrics_list):
        selected_ids = (
            set(str(value) for value in agent_ids_by_run[run_index])
            if agent_ids_by_run is not None
            else set()
        )
        evaluated_ids_by_run.append(sorted(selected_ids) if selected_ids else "all")
        primary = dict(metrics.get("primary_metrics", {}))
        stage_per_agent = primary.get("stage_per_agent", {})
        tables = (
            list(stage_per_agent.values())
            if isinstance(stage_per_agent, dict) and stage_per_agent
            else [primary.get("per_agent", {})]
        )
        for per_agent in tables:
            if not isinstance(per_agent, dict):
                continue
            for agent_id, values in per_agent.items():
                if selected_ids and str(agent_id) not in selected_ids:
                    continue
                if not isinstance(values, dict):
                    continue
                samples["vpip"].append(float(values.get("vpip", 0.0)))
                samples["fold_rate"].append(float(values.get("fold_rate", 0.0)))
                samples["all_in_hand_rate"].append(
                    float(values.get("all_in_hand_rate", 0.0))
                )
                samples["bust_hand_rate"].append(
                    float(values.get("bust_hand_rate", 0.0))
                )
                sensitivity = dict(values.get("hand_reward_sensitivity", {}))
                samples["single_hand_reward_activity_share"].append(
                    float(sensitivity.get("share_of_absolute_reward_activity", 0.0))
                )
                memory = dict(values.get("memory", {}))
                samples["empty_retrieval_rate"].append(
                    float(memory.get("empty_retrieval_rate", 0.0))
                )
                samples["structural_signature_share"].append(
                    float(memory.get("max_structural_signature_share", 0.0))
                )

    blockers = [name for name, values in samples.items() if len(values) < 2]
    if blockers:
        return {
            "status": "blocked_insufficient_behavior_samples",
            "blockers": [f"insufficient samples for {name}" for name in blockers],
            "policy": BEHAVIOR_FREEZE_POLICY,
            "thresholds": {},
            "sample_counts": {name: len(values) for name, values in samples.items()},
        }
    lower = float(BEHAVIOR_FREEZE_POLICY["lower_quantile"])
    upper = float(BEHAVIOR_FREEZE_POLICY["upper_quantile"])
    rate_margin = float(BEHAVIOR_FREEZE_POLICY["rate_margin"])
    concentration_margin = float(BEHAVIOR_FREEZE_POLICY["concentration_margin"])
    caps = dict(BEHAVIOR_FREEZE_POLICY["domain_caps"])
    thresholds = {
        "min_vpip": max(
            float(BEHAVIOR_FREEZE_POLICY["domain_floors"]["min_vpip"]),
            _quantile(samples["vpip"], lower) - rate_margin,
        ),
        "max_fold_rate": min(
            float(caps["max_fold_rate"]),
            _quantile(samples["fold_rate"], upper) + rate_margin,
        ),
        "min_voluntary_participation_hands": 1,
        "max_all_in_hand_rate": min(
            float(caps["max_all_in_hand_rate"]),
            _quantile(samples["all_in_hand_rate"], upper) + rate_margin,
        ),
        "max_bust_hand_rate": min(
            float(caps["max_bust_hand_rate"]),
            _quantile(samples["bust_hand_rate"], upper) + rate_margin,
        ),
        "max_single_hand_reward_activity_share": min(
            float(caps["max_single_hand_reward_activity_share"]),
            _quantile(samples["single_hand_reward_activity_share"], upper)
            + concentration_margin,
        ),
        "single_hand_reward_activity_diagnostic_only": True,
        "max_empty_retrieval_rate": min(
            float(caps["max_empty_retrieval_rate"]),
            _quantile(samples["empty_retrieval_rate"], upper)
            + concentration_margin,
        ),
        "max_structural_signature_share": min(
            float(caps["max_structural_signature_share"]),
            _quantile(samples["structural_signature_share"], upper)
            + concentration_margin,
        ),
    }
    thresholds = {
        name: round(float(value), 6) if not isinstance(value, int) else value
        for name, value in thresholds.items()
    }
    domain_thresholds = {
        "min_vpip": float(BEHAVIOR_FREEZE_POLICY["domain_floors"]["min_vpip"]),
        "max_fold_rate": float(caps["max_fold_rate"]),
        "min_voluntary_participation_hands": 1,
        "max_all_in_hand_rate": float(caps["max_all_in_hand_rate"]),
        "max_bust_hand_rate": float(caps["max_bust_hand_rate"]),
        "max_single_hand_reward_activity_share": float(
            caps["max_single_hand_reward_activity_share"]
        ),
        "single_hand_reward_activity_diagnostic_only": True,
        "max_empty_retrieval_rate": float(caps["max_empty_retrieval_rate"]),
        "max_structural_signature_share": float(
            caps["max_structural_signature_share"]
        ),
    }
    audits = [
        evaluate_behavior_health(
            metrics,
            {
                "behavior_threshold_status": "frozen",
                "behavior_thresholds": domain_thresholds,
            },
            agent_ids,
        )
        for run_index, metrics in enumerate(metrics_list)
        for agent_ids in [
            agent_ids_by_run[run_index] if agent_ids_by_run is not None else None
        ]
    ]
    failed = [index for index, audit in enumerate(audits) if audit.get("status") != "passed"]
    freeze_blockers = [
        (
            f"{run_labels[index]} fails frozen behavior gates"
            if run_labels is not None
            else f"run {index} fails frozen behavior gates"
        )
        for index in failed
    ]
    return {
        "status": "frozen" if not freeze_blockers else "blocked_pilot_behavior_degenerate",
        "blockers": freeze_blockers,
        "policy": BEHAVIOR_FREEZE_POLICY,
        "thresholds": thresholds,
        "pilot_domain_gate_thresholds": domain_thresholds,
        "sample_counts": {name: len(values) for name, values in samples.items()},
        "evaluated_agent_ids_by_run": evaluated_ids_by_run,
        "run_labels": (
            list(run_labels)
            if run_labels is not None
            else [f"run {index}" for index in range(len(metrics_list))]
        ),
        "failed_run_indexes": failed,
    }


def _evaluation_target_ids_by_run(
    protocol_audits: list[dict[str, Any]],
    label: str,
) -> tuple[list[list[str]], list[str]]:
    targets_by_run: list[list[str]] = []
    blockers: list[str] = []
    for index, audit in enumerate(protocol_audits):
        targets = audit.get("evaluation_target_ids", [])
        if not isinstance(targets, list) or not targets:
            blockers.append(
                f"{label} protocol audit {index} lacks evaluation_target_ids"
            )
            targets_by_run.append([])
        else:
            targets_by_run.append([str(value) for value in targets])
    return targets_by_run, blockers


def _execution_health_blockers(
    protocol_audits: list[dict[str, Any]],
    label: str,
) -> list[str]:
    blockers: list[str] = []
    for index, audit in enumerate(protocol_audits):
        execution = audit.get("execution_health", {})
        if not isinstance(execution, dict) or execution.get("valid") is not True:
            blockers.append(f"{label} run {index} execution health is not valid")
            continue
        if execution.get("status") != "passed":
            blockers.append(
                f"{label} run {index} execution health status is "
                f"{execution.get('status')}"
            )
        for key in EXECUTION_ZERO_FIELDS:
            value = _safe_int(execution.get(key))
            if value != 0:
                blockers.append(
                    f"{label} run {index} execution {key}: "
                    f"{execution.get(key)}"
                )
    return blockers


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _validate_pilot_aggregate(
    aggregate: dict[str, Any], label: str, blockers: list[str]
) -> None:
    completed = _safe_int(aggregate.get("completed_run_count"))
    expected = _safe_int(aggregate.get("expected_run_count"))
    if expected is None or expected < 1 or completed != expected:
        blockers.append(f"{label} matrix is incomplete: {completed}/{expected}")
    homogeneity = aggregate.get("runtime_homogeneity", {})
    if not isinstance(homogeneity, dict) or homogeneity.get("homogeneous") is not True:
        blockers.append(f"{label} runtime is heterogeneous or unverified")
    if str(aggregate.get("status")) != "descriptive_only":
        blockers.append(f"{label} must be a complete descriptive-only pilot")


def _validate_power_contracts(
    campaign_p: dict[str, Any],
    campaign_e: dict[str, Any],
    p_estimand_container: Any,
    p_estimand: Any,
    e_comparisons: Any,
    endpoint: str,
    blockers: list[str],
) -> list[int]:
    p_contract = {
        "design": "mixed_table",
    }
    e_contract = {
        "design": "target_vs_seven_no_memory",
        "estimand": "same_seed_cross_condition_target_effect_vs_no_memory",
        "baseline_condition_id": "no_memory_target",
        "primary_endpoint": "final_test_bb_per_100",
        "multiple_comparison_method": "holm",
    }
    for key, expected in p_contract.items():
        if campaign_p.get(key) != expected:
            blockers.append(f"campaign_p {key} mismatch: {campaign_p.get(key)}")
    for key, expected in e_contract.items():
        if campaign_e.get(key) != expected:
            blockers.append(f"campaign_e {key} mismatch: {campaign_e.get(key)}")
    if endpoint != "final_test_bb_per_100":
        blockers.append(f"campaign_e primary endpoint mismatch: {endpoint}")
    if not isinstance(p_estimand_container, dict):
        blockers.append("campaign_p paired estimand is missing")
        return []
    paired_contract = {
        "status": "descriptive_only",
        "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
        "endpoint": "final_test_bb_per_100",
        "baseline_mechanism": "fact",
        "multiple_comparison_method": "holm",
    }
    for key, expected in paired_contract.items():
        if p_estimand_container.get(key) != expected:
            blockers.append(
                f"campaign_p paired estimand {key} mismatch: "
                f"{p_estimand_container.get(key)}"
            )
    p_seeds = _int_list(p_estimand_container.get("matched_seeds"))
    if not p_seeds:
        blockers.append("campaign_p paired estimand matched_seeds is missing")
    if _safe_int(p_estimand_container.get("independent_seed_count")) != len(p_seeds):
        blockers.append(
            "campaign_p independent seed count does not match matched_seeds"
        )
    observed_p = set(p_estimand) if isinstance(p_estimand, dict) else set()
    if observed_p != EXPECTED_P_MECHANISMS:
        blockers.append(
            "campaign_p contrasts mismatch: "
            f"{sorted(str(value) for value in observed_p)}/"
            f"{sorted(EXPECTED_P_MECHANISMS)}"
        )
    observed_e = set(e_comparisons) if isinstance(e_comparisons, dict) else set()
    if observed_e != EXPECTED_E_CONDITIONS:
        blockers.append(
            "campaign_e contrasts mismatch: "
            f"{sorted(str(value) for value in observed_e)}/"
            f"{sorted(EXPECTED_E_CONDITIONS)}"
        )
    return p_seeds


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return []
    return result if all(math.isfinite(value) for value in result) else []


def _leaf_evidence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        metrics_path = run_dir / "metrics.json"
        protocol_path = run_dir / "protocol_audit.json"
        evidence.append(
            {
                "condition_id": row.get("condition_id"),
                "seed": _safe_int(row.get("seed")),
                "attempt": _safe_int(row.get("attempt")),
                "run_id": row.get("run_id"),
                "run_dir": str(run_dir),
                "metrics_sha256": _sha256_file(metrics_path),
                "protocol_audit_sha256": _sha256_file(protocol_path),
            }
        )
    return sorted(
        evidence,
        key=lambda item: (
            str(item["condition_id"]),
            int(item["seed"] or 0),
            int(item["attempt"] or 0),
        ),
    )


def _int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    result = [_safe_int(value) for value in values]
    return [int(value) for value in result if value is not None]


def _runtime_equivalence_blockers(
    p_identity: dict[str, Any],
    e_identity: dict[str, Any],
    audit: dict[str, Any] | None,
) -> list[str]:
    if audit is None:
        return ["campaign P/E runtime identities differ"]
    blockers: list[str] = []
    if audit.get("schema_version") != "task4_pilot_runtime_equivalence_audit_v1":
        blockers.append("pilot runtime-equivalence audit schema is invalid")
    if (
        audit.get("status")
        != "verified_execution_runtime_equivalent_for_pilot_power_only"
        or audit.get("blockers") != []
    ):
        blockers.append("pilot runtime-equivalence audit is not verified")
    if audit.get("formal_homogeneity_not_granted") is not True:
        blockers.append("pilot runtime-equivalence audit lacks formal-use prohibition")
    p_code = _pairs_to_dict(p_identity.get("code", {}))
    e_code = _pairs_to_dict(e_identity.get("code", {}))
    expected = {
        "campaign_p_code_sha": str(p_code.get("commit", "")),
        "campaign_e_code_sha": str(e_code.get("commit", "")),
        "campaign_p_runtime_identity_sha256": _json_sha256(p_identity),
        "campaign_e_runtime_identity_sha256": _json_sha256(e_identity),
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            blockers.append(f"pilot runtime-equivalence audit mismatch for {key}")
    if _without_code_identity(p_identity) != _without_code_identity(e_identity):
        blockers.append("campaign P/E non-code runtime identities differ")
    if audit.get("disallowed_changed_paths") != []:
        blockers.append("pilot runtime-equivalence audit has disallowed changed paths")
    reported_changed_paths = audit.get("changed_paths")
    if not isinstance(reported_changed_paths, list):
        blockers.append("pilot runtime-equivalence audit lacks changed_paths")
    else:
        normalized = {
            str(path).replace("\\", "/").lstrip("./")
            for path in reported_changed_paths
        }
        if normalized - PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS:
            blockers.append(
                "pilot runtime-equivalence audit reports unregistered changed paths"
            )
    if audit.get("allowed_changed_paths_policy") != sorted(
        PILOT_RUNTIME_EQUIVALENCE_ALLOWED_CHANGED_PATHS
    ):
        blockers.append("pilot runtime-equivalence audit policy is stale or altered")
    if audit.get("required_diff_sha256_policy") != dict(
        PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256
    ):
        blockers.append(
            "pilot runtime-equivalence required diff-hash policy is stale or altered"
        )
    if audit.get("required_diff_sha256_match") is not True:
        blockers.append(
            "pilot runtime-equivalence required diff hashes are not verified"
        )
    reported_diff_hashes = audit.get("changed_path_diff_sha256")
    if not isinstance(reported_diff_hashes, dict):
        blockers.append(
            "pilot runtime-equivalence audit lacks changed_path_diff_sha256"
        )
    elif isinstance(reported_changed_paths, list):
        blockers.extend(
            _required_diff_hash_blockers(
                [
                    str(path).replace("\\", "/")
                    for path in reported_changed_paths
                ],
                {str(key): str(value) for key, value in reported_diff_hashes.items()},
            )
        )
    return blockers


def _without_code_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "code"}


def _required_diff_hash_blockers(
    changed_paths: list[str],
    changed_path_diff_sha256: dict[str, str] | None,
) -> list[str]:
    observed = dict(changed_path_diff_sha256 or {})
    blockers: list[str] = []
    for path, expected_hash in PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256.items():
        if path not in changed_paths:
            continue
        actual_hash = str(observed.get(path, ""))
        if actual_hash != expected_hash:
            blockers.append(
                f"registered execution-adjacent path diff hash mismatch for "
                f"{path}: {actual_hash or '<missing>'}"
            )
    return blockers


def _pairs_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        try:
            return {str(key): item for key, item in value}
        except (TypeError, ValueError):
            return {}
    return {}


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)
