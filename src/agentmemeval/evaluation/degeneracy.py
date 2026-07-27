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
    audit["degenerate"] = any(
        not item["passed"] and item.get("hard_gate", True) for item in checks
    )
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
        hard_gate=not bool(
            thresholds.get("single_hand_reward_activity_diagnostic_only", False)
        ),
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
    memory_revision_fallback_count = revision_fallback_count(metrics)
    experience_health = experience_revision_health(metrics)
    memory_revision_failure_count = (
        experience_health["revision_failure_count"]
        + experience_health["revision_diagnostic_gap_count"]
    )
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
    valid = (
        not fallback_count
        and not memory_revision_fallback_count
        and not memory_revision_failure_count
        and not experience_health["contract_violation_count"]
        and not experience_health["invalid_document_count"]
        and not experience_health["unrealized_injection_count"]
        and not reward_violations
        and not stack_violations
    )
    return {
        "valid": valid,
        "fallback_count": fallback_count,
        "memory_revision_fallback_count": memory_revision_fallback_count,
        # Keep the established execution-health schema stable.  Missing
        # rejection diagnostics are execution failures and are included in this
        # counter; fully diagnosed fail-closed rejections remain visible in the
        # separate validation-rejection counter but are non-blocking.
        "memory_revision_failure_count": memory_revision_failure_count,
        "memory_revision_validation_rejection_count": experience_health[
            "validation_rejection_count"
        ],
        "memory_revision_contract_violation_count": experience_health[
            "contract_violation_count"
        ],
        "invalid_experience_document_count": experience_health[
            "invalid_document_count"
        ],
        "unrealized_experience_injection_count": experience_health[
            "unrealized_injection_count"
        ],
        "reward_conservation_violation_count": len(reward_violations),
        "reward_conservation_violation_hand_ids": reward_violations,
        "stack_conservation_violation_count": len(stack_violations),
        "stack_conservation_violation_hand_ids": stack_violations,
        "status": "passed" if valid else "invalid_execution",
    }


def revision_fallback_count(metrics: dict[str, Any]) -> int:
    """Count deterministic substitutes once per agent across train/test stages."""

    primary = metrics.get("primary_metrics", {})
    stage_per_agent = primary.get("stage_per_agent", {})
    tables = (
        list(stage_per_agent.values())
        if isinstance(stage_per_agent, dict) and stage_per_agent
        else []
    )
    per_agent = primary.get("per_agent", {})
    if isinstance(per_agent, dict) and per_agent:
        tables.append(per_agent)
    maximum_by_agent: dict[str, int] = {}
    for per_agent in tables:
        if not isinstance(per_agent, dict):
            continue
        for agent_id, values in per_agent.items():
            if not isinstance(values, dict):
                continue
            memory = values.get("memory", {})
            count = (
                int(memory.get("revision_fallback_count", 0))
                if isinstance(memory, dict)
                else 0
            )
            key = str(agent_id)
            maximum_by_agent[key] = max(maximum_by_agent.get(key, 0), count)
    return sum(maximum_by_agent.values())


def experience_revision_health(metrics: dict[str, Any]) -> dict[str, int]:
    """Aggregate structural Expr health without looking at reward direction or effects."""

    primary = metrics.get("primary_metrics", {})
    stage_per_agent = primary.get("stage_per_agent", {})
    tables = (
        list(stage_per_agent.values())
        if isinstance(stage_per_agent, dict) and stage_per_agent
        else []
    )
    per_agent = primary.get("per_agent", {})
    if isinstance(per_agent, dict) and per_agent:
        tables.append(per_agent)
    maximums: dict[str, dict[str, int]] = {}
    latest_validity: dict[str, tuple[bool, bool]] = {}
    experience_mechanisms = {"expr", "fact_expr_sync", "fact_expr_async"}
    for table in tables:
        if not isinstance(table, dict):
            continue
        for agent_id, values in table.items():
            if not isinstance(values, dict):
                continue
            memory = values.get("memory", {})
            if not isinstance(memory, dict):
                continue
            if str(memory.get("mechanism", "")) not in experience_mechanisms:
                continue
            key = str(agent_id)
            counters = maximums.setdefault(key, {})
            for field in (
                "revision_failure_count",
                "revision_validation_rejection_count",
                "revision_contract_violation_count",
            ):
                counters[field] = max(
                    counters.get(field, 0), int(memory.get(field, 0))
                )
            counters["revision_diagnostic_gap_count"] = max(
                counters.get("revision_diagnostic_gap_count", 0),
                _revision_diagnostic_gap_count(memory),
            )
            latest_validity[key] = (
                bool(memory.get("experience_document_valid", False)),
                bool(memory.get("experience_injection_current", False)),
            )
    return {
        "revision_failure_count": sum(
            values.get("revision_failure_count", 0) for values in maximums.values()
        ),
        "validation_rejection_count": sum(
            values.get("revision_validation_rejection_count", 0)
            for values in maximums.values()
        ),
        "revision_diagnostic_gap_count": sum(
            values.get("revision_diagnostic_gap_count", 0)
            for values in maximums.values()
        ),
        "contract_violation_count": sum(
            values.get("revision_contract_violation_count", 0)
            for values in maximums.values()
        ),
        "invalid_document_count": sum(
            not document_valid for document_valid, _injected in latest_validity.values()
        ),
        "unrealized_injection_count": sum(
            not injected for _document_valid, injected in latest_validity.values()
        ),
    }


def _revision_diagnostic_gap_count(memory: dict[str, Any]) -> int:
    """Fail closed when guarded rejection accounting is absent or inconsistent."""

    rejection_count = max(0, int(memory.get("revision_validation_rejection_count", 0)))
    if (
        "revision_rejection_diagnostic_count" in memory
        or "revision_rejection_missing_diagnostic_count" in memory
    ):
        complete = max(0, int(memory.get("revision_rejection_diagnostic_count", 0)))
        missing = max(
            0,
            int(memory.get("revision_rejection_missing_diagnostic_count", 0)),
        )
        final_gap = missing + abs(rejection_count - complete - missing)
    else:
        # Historical metrics did not prove that rejected proposals retained the
        # required body-free fingerprint and reason.
        final_gap = rejection_count

    initial_count = max(
        0,
        int(memory.get("revision_repair_initial_rejection_count", 0)),
    )
    if (
        "revision_repair_initial_diagnostic_count" in memory
        or "revision_repair_initial_missing_diagnostic_count" in memory
    ):
        initial_complete = max(
            0,
            int(memory.get("revision_repair_initial_diagnostic_count", 0)),
        )
        initial_missing = max(
            0,
            int(memory.get("revision_repair_initial_missing_diagnostic_count", 0)),
        )
        initial_gap = initial_missing + abs(
            initial_count - initial_complete - initial_missing
        )
    else:
        initial_gap = initial_count
    return final_gap + initial_gap


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
    *,
    hard_gate: bool = True,
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
            "hard_gate": hard_gate,
        }
    )
