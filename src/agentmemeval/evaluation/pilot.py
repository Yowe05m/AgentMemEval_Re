"""Independent-pilot power planning without silently authorizing formal runs."""

from __future__ import annotations

from typing import Any

from agentmemeval.evaluation.statistics import estimate_paired_seed_requirement

PRIMARY_MDE_BB_PER_100 = 5.0
SENSITIVITY_MDES_BB_PER_100 = (3.0, 5.0, 10.0)


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
