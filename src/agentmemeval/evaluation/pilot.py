"""Independent-pilot power planning without silently authorizing formal runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from agentmemeval.evaluation.degeneracy import (
    evaluate_behavior_health,
    revision_fallback_count,
)
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


def build_pilot_power_plan(
    campaign_p: dict[str, Any], campaign_e: dict[str, Any]
) -> dict[str, Any]:
    """Build the pre-registered P/E seed plan from complete pilot aggregates."""

    blockers: list[str] = []
    _validate_pilot_aggregate(campaign_p, "campaign_p", blockers)
    _validate_pilot_aggregate(campaign_e, "campaign_e", blockers)
    p_runtime = campaign_p.get("runtime_homogeneity", {})
    e_runtime = campaign_e.get("runtime_homogeneity", {})
    p_identity = p_runtime.get("identity") if isinstance(p_runtime, dict) else None
    e_identity = e_runtime.get("identity") if isinstance(e_runtime, dict) else None
    if not isinstance(p_identity, dict) or not isinstance(e_identity, dict):
        blockers.append("campaign P/E runtime identities are missing")
    elif p_identity != e_identity:
        blockers.append("campaign P/E runtime identities differ")
    contrasts: dict[str, list[float]] = {}

    p_estimand = (
        campaign_p.get("aggregate_metrics", {})
        .get("paired_estimand_descriptive", {})
        .get("effects_by_mechanism", {})
    )
    if isinstance(p_estimand, dict):
        for mechanism, effects in sorted(p_estimand.items()):
            contrasts[f"campaign_p:{mechanism}_vs_fact"] = _float_list(effects)

    e_comparisons = campaign_e.get("paired_comparisons", {})
    endpoint = str(campaign_e.get("primary_endpoint", "final_test_bb_per_100"))
    if isinstance(e_comparisons, dict):
        for condition, comparison in sorted(e_comparisons.items()):
            effects = (
                comparison.get("metrics", {}).get(endpoint, {}).get("effects", [])
                if isinstance(comparison, dict)
                else []
            )
            contrasts[f"campaign_e:{condition}_vs_no_memory"] = _float_list(effects)

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
    }


def build_pilot_freeze_proposal(
    campaign_p: dict[str, Any],
    campaign_e: dict[str, Any],
    campaign_p_metrics: list[dict[str, Any]],
    campaign_p_protocol_audits: list[dict[str, Any]],
    campaign_e_metrics: list[dict[str, Any]],
    campaign_e_protocol_audits: list[dict[str, Any]],
    retrieval_review_audit: dict[str, Any],
) -> dict[str, Any]:
    """Combine power, behavior, execution, and retrieval freeze gates."""

    power_plan = build_pilot_power_plan(campaign_p, campaign_e)
    behavior = calibrate_behavior_thresholds(
        campaign_p_metrics,
        _evaluation_target_ids_by_run(campaign_p_protocol_audits),
    )
    execution_blockers = [
        f"campaign_p run {index} execution health is not valid"
        for index, audit in enumerate(campaign_p_protocol_audits)
        if dict(audit.get("execution_health", {})).get("valid") is not True
    ]
    execution_blockers.extend(
        f"campaign_e run {index} execution health is not valid"
        for index, audit in enumerate(campaign_e_protocol_audits)
        if dict(audit.get("execution_health", {})).get("valid") is not True
    )
    retrieval_blockers = [
        str(item) for item in retrieval_review_audit.get("blockers", [])
    ]
    if retrieval_review_audit.get("review_status") != "human_labels_verified":
        retrieval_blockers.append("retrieval relevance review lacks verified human labels")
    if retrieval_review_audit.get("retrieval_threshold_status") != "frozen":
        retrieval_blockers.append("retrieval relevance threshold is not frozen")
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
    proposal = build_pilot_freeze_proposal(
        p_aggregate,
        e_aggregate,
        metrics,
        p_audits,
        e_metrics,
        e_audits,
        retrieval_review,
    )
    proposal["campaign_p_evidence"] = p_evidence
    proposal["campaign_e_evidence"] = e_evidence
    proposal["retrieval_review_evidence"] = {
        "path": str(Path(retrieval_review_audit_path).resolve()),
        "schema_version": retrieval_review.get("schema_version"),
    }
    return proposal


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
) -> dict[str, Any]:
    """Apply the outcome-independent quantile-plus-domain-cap freeze policy."""

    if agent_ids_by_run is not None and len(agent_ids_by_run) != len(metrics_list):
        raise ValueError(
            "agent_ids_by_run length must match metrics_list length: "
            f"{len(agent_ids_by_run)} != {len(metrics_list)}"
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
    freeze_blockers = [f"campaign_p run {index} fails frozen behavior gates" for index in failed]
    return {
        "status": "frozen" if not freeze_blockers else "blocked_pilot_behavior_degenerate",
        "blockers": freeze_blockers,
        "policy": BEHAVIOR_FREEZE_POLICY,
        "thresholds": thresholds,
        "pilot_domain_gate_thresholds": domain_thresholds,
        "sample_counts": {name: len(values) for name, values in samples.items()},
        "evaluated_agent_ids_by_run": evaluated_ids_by_run,
        "failed_run_indexes": failed,
    }


def _evaluation_target_ids_by_run(
    protocol_audits: list[dict[str, Any]],
) -> list[list[str]]:
    targets_by_run: list[list[str]] = []
    for index, audit in enumerate(protocol_audits):
        targets = audit.get("evaluation_target_ids", [])
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                f"campaign_p protocol audit {index} lacks evaluation_target_ids"
            )
        targets_by_run.append([str(value) for value in targets])
    return targets_by_run


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
    completed = int(aggregate.get("completed_run_count", 0))
    expected = int(aggregate.get("expected_run_count", 0))
    if expected < 1 or completed != expected:
        blockers.append(f"{label} matrix is incomplete: {completed}/{expected}")
    homogeneity = aggregate.get("runtime_homogeneity", {})
    if not isinstance(homogeneity, dict) or homogeneity.get("homogeneous") is not True:
        blockers.append(f"{label} runtime is heterogeneous or unverified")
    if str(aggregate.get("status")) != "descriptive_only":
        blockers.append(f"{label} must be a complete descriptive-only pilot")


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]
