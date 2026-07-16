"""Pre-registered behavior-health framework for excluding degenerate runs."""

from __future__ import annotations

from typing import Any


def evaluate_behavior_health(
    metrics: dict[str, Any],
    experiment_config: dict[str, Any],
    agent_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen thresholds, or report that pilot calibration is still required."""

    status = str(experiment_config.get("behavior_threshold_status", "pending_pilot"))
    thresholds = experiment_config.get("behavior_thresholds", {})
    audit: dict[str, Any] = {
        "threshold_status": status,
        "thresholds": thresholds if isinstance(thresholds, dict) else {},
        "checks": [],
        "degenerate": False,
        "valid_for_main_table": False,
    }
    if status != "frozen":
        audit["status"] = "pending_independent_pilot_calibration"
        audit["calibration_required"] = True
        return audit
    if not isinstance(thresholds, dict) or not thresholds:
        audit["status"] = "invalid_threshold_configuration"
        audit["degenerate"] = True
        audit["checks"].append(
            {"scope": "run", "metric": "behavior_thresholds", "passed": False}
        )
        return audit

    primary = metrics.get("primary_metrics", {})
    stage_per_agent = primary.get("stage_per_agent", {})
    source_tables = (
        [(str(stage), values) for stage, values in stage_per_agent.items()]
        if stage_per_agent
        else [("combined", primary.get("per_agent", {}))]
    )
    checks: list[dict[str, Any]] = []
    selected_ids = set(agent_ids or [])
    for stage, per_agent in source_tables:
        for agent_id, values in per_agent.items():
            if selected_ids and str(agent_id) not in selected_ids:
                continue
            scope = f"{agent_id}:{stage}"
            _agent_behavior_checks(checks, scope, values, thresholds)
    audit["evaluated_agent_ids"] = sorted(selected_ids) if selected_ids else "all"
    audit["evaluated_stages"] = [stage for stage, _values in source_tables]
    audit["checks"] = checks
    audit["degenerate"] = any(not item["passed"] for item in checks)
    audit["valid_for_main_table"] = bool(checks) and not audit["degenerate"]
    audit["status"] = "passed" if audit["valid_for_main_table"] else "degenerate"
    audit["calibration_required"] = False
    return audit


def _agent_behavior_checks(
    checks: list[dict[str, Any]],
    scope: str,
    values: dict[str, Any],
    thresholds: dict[str, Any],
) -> None:
    _minimum_check(checks, scope, "vpip", values, thresholds, "min_vpip")
    _maximum_check(checks, scope, "fold_rate", values, thresholds, "max_fold_rate")
    _minimum_check(
        checks,
        scope,
        "voluntary_participation_hands",
        values,
        thresholds,
        "min_voluntary_participation_hands",
    )
    _maximum_check(
        checks,
        scope,
        "all_in_hand_rate",
        values,
        thresholds,
        "max_all_in_hand_rate",
    )
    _maximum_check(
        checks,
        scope,
        "bust_hand_rate",
        values,
        thresholds,
        "max_bust_hand_rate",
    )
    sensitivity = values.get("hand_reward_sensitivity", {})
    _maximum_check(
        checks,
        scope,
        "share_of_absolute_reward_activity",
        sensitivity,
        thresholds,
        "max_single_hand_reward_activity_share",
    )
    memory = values.get("memory", {})
    _maximum_check(
        checks,
        scope,
        "empty_retrieval_rate",
        memory,
        thresholds,
        "max_empty_retrieval_rate",
    )
    _maximum_check(
        checks,
        scope,
        "max_structural_signature_share",
        memory,
        thresholds,
        "max_structural_signature_share",
    )
    street_thresholds = thresholds.get("min_street_coverage", {})
    if isinstance(street_thresholds, dict):
        coverage = values.get("street_coverage", {})
        for street, minimum in street_thresholds.items():
            actual = float(coverage.get(street, 0.0))
            checks.append(
                {
                    "scope": scope,
                    "metric": f"street_coverage.{street}",
                    "operator": ">=",
                    "threshold": float(minimum),
                    "actual": actual,
                    "passed": actual >= float(minimum),
                }
            )


def evaluate_execution_health(
    hand_summaries: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Reject pilot/formal interpretation when fallback or chip conservation fails."""

    quality = (
        metrics.get("exploratory_metrics", {})
        .get("decision_quality", {})
        .get("combined", {})
    )
    fallback_count = int(quality.get("fallback_count", 0))
    reward_violations: list[str] = []
    stack_violations: list[str] = []
    for hand in hand_summaries:
        hand_id = str(hand.get("hand_id", "unknown"))
        rewards = hand.get("rewards", {}) or {}
        if sum(int(value) for value in rewards.values()) != 0:
            reward_violations.append(hand_id)
        starting = hand.get("starting_stacks", {}) or {}
        final = hand.get("final_stacks", {}) or {}
        if sum(int(value) for value in starting.values()) != sum(
            int(value) for value in final.values()
        ):
            stack_violations.append(hand_id)
    valid = not fallback_count and not reward_violations and not stack_violations
    return {
        "valid": valid,
        "fallback_count": fallback_count,
        "reward_conservation_violation_count": len(reward_violations),
        "reward_conservation_violation_hand_ids": reward_violations,
        "stack_conservation_violation_count": len(stack_violations),
        "stack_conservation_violation_hand_ids": stack_violations,
        "status": "passed" if valid else "invalid_execution",
    }


def build_run_validity(
    admission: dict[str, Any],
    behavior_health: dict[str, Any],
    execution_health: dict[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    """Combine pre-run admission and post-run checks into one main-table decision."""

    paper_eligible = (
        bool(admission.get("paper_eligible_at_start"))
        and bool(execution_health.get("valid"))
        and bool(behavior_health.get("valid_for_main_table"))
    )
    return {
        "run_mode": run_mode,
        "execution_valid": bool(execution_health.get("valid")),
        "behavior_valid": bool(behavior_health.get("valid_for_main_table")),
        "paper_eligible": paper_eligible,
        "status": "valid_for_main_table" if paper_eligible else "not_for_main_table",
    }


def _minimum_check(
    checks: list[dict[str, Any]],
    scope: str,
    metric: str,
    values: dict[str, Any],
    thresholds: dict[str, Any],
    threshold_name: str,
) -> None:
    if threshold_name not in thresholds:
        return
    actual = float(values.get(metric, 0.0))
    threshold = float(thresholds[threshold_name])
    checks.append(
        {
            "scope": str(scope),
            "metric": metric,
            "operator": ">=",
            "threshold": threshold,
            "actual": actual,
            "passed": actual >= threshold,
        }
    )


def _maximum_check(
    checks: list[dict[str, Any]],
    scope: str,
    metric: str,
    values: dict[str, Any],
    thresholds: dict[str, Any],
    threshold_name: str,
) -> None:
    if threshold_name not in thresholds:
        return
    actual = float(values.get(metric, 0.0))
    threshold = float(thresholds[threshold_name])
    checks.append(
        {
            "scope": str(scope),
            "metric": metric,
            "operator": "<=",
            "threshold": threshold,
            "actual": actual,
            "passed": actual <= threshold,
        }
    )
