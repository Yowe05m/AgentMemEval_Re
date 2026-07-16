"""Independent-pilot power planning without silently authorizing formal runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from agentmemeval.evaluation.degeneracy import evaluate_behavior_health
from agentmemeval.evaluation.statistics import estimate_paired_seed_requirement

PRIMARY_MDE_BB_PER_100 = 5.0
SENSITIVITY_MDES_BB_PER_100 = (3.0, 5.0, 10.0)
BEHAVIOR_FREEZE_POLICY = {
    "lower_quantile": 0.05,
    "upper_quantile": 0.95,
    "rate_margin": 0.02,
    "concentration_margin": 0.05,
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
) -> dict[str, Any]:
    """Combine power, behavior, execution, and retrieval freeze gates."""

    power_plan = build_pilot_power_plan(campaign_p, campaign_e)
    behavior = calibrate_behavior_thresholds(campaign_p_metrics)
    execution_blockers = [
        f"campaign_p run {index} execution health is not valid"
        for index, audit in enumerate(campaign_p_protocol_audits)
        if dict(audit.get("execution_health", {})).get("valid") is not True
    ]
    expected_p = int(campaign_p.get("expected_run_count", 0))
    if len(campaign_p_metrics) != expected_p:
        execution_blockers.append(
            f"campaign_p metrics count mismatch: {len(campaign_p_metrics)}/{expected_p}"
        )
    if len(campaign_p_protocol_audits) != expected_p:
        execution_blockers.append(
            "campaign_p protocol audit count mismatch: "
            f"{len(campaign_p_protocol_audits)}/{expected_p}"
        )
    blockers = [
        *list(power_plan["blockers"]),
        *list(behavior["blockers"]),
        *execution_blockers,
    ]
    return {
        "schema_version": "agentmemeval_pilot_freeze_proposal_v1",
        "power_plan": power_plan,
        "behavior_freeze": behavior,
        "retrieval_freeze": {
            "retrieval_threshold_status": "frozen",
            "minimum_retrieval_score": 0.0,
            "reason": (
                "pilot has no independent human relevance labels; freeze zero rather "
                "than tune retrieval on reward or test outcomes"
            ),
        },
        "execution_blockers": execution_blockers,
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
) -> dict[str, Any]:
    """Load only completed P leaf evidence and build the immutable proposal."""

    p_aggregate = _read_json(Path(campaign_p_aggregate_path))
    e_aggregate = _read_json(Path(campaign_e_aggregate_path))
    state_path = Path(campaign_p_dir) / "state.tsv"
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = [row for row in rows if row.get("status") == "complete"]
    metrics = [_read_json(Path(row["run_dir"]) / "metrics.json") for row in completed]
    audits = [
        _read_json(Path(row["run_dir"]) / "protocol_audit.json") for row in completed
    ]
    proposal = build_pilot_freeze_proposal(p_aggregate, e_aggregate, metrics, audits)
    proposal["campaign_p_evidence"] = {
        "campaign_dir": str(Path(campaign_p_dir).resolve()),
        "completed_state_rows": len(completed),
        "ignored_noncomplete_state_rows": len(rows) - len(completed),
    }
    return proposal


def calibrate_behavior_thresholds(
    metrics_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the outcome-independent quantile-plus-domain-cap freeze policy."""

    samples: dict[str, list[float]] = {
        "vpip": [],
        "fold_rate": [],
        "all_in_hand_rate": [],
        "bust_hand_rate": [],
        "single_hand_reward_activity_share": [],
        "empty_retrieval_rate": [],
        "structural_signature_share": [],
    }
    for metrics in metrics_list:
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
            for values in per_agent.values():
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
    audits = [
        evaluate_behavior_health(
            metrics,
            {"behavior_threshold_status": "frozen", "behavior_thresholds": thresholds},
        )
        for metrics in metrics_list
    ]
    failed = [index for index, audit in enumerate(audits) if audit.get("status") != "passed"]
    freeze_blockers = [f"campaign_p run {index} fails frozen behavior gates" for index in failed]
    return {
        "status": "frozen" if not freeze_blockers else "blocked_pilot_behavior_degenerate",
        "blockers": freeze_blockers,
        "policy": BEHAVIOR_FREEZE_POLICY,
        "thresholds": thresholds,
        "sample_counts": {name: len(values) for name, values in samples.items()},
        "failed_run_indexes": failed,
    }


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
