from __future__ import annotations

from agentmemeval.evaluation.pilot import build_pilot_power_plan


def _p() -> dict[str, object]:
    return {
        "status": "descriptive_only",
        "completed_run_count": 3,
        "expected_run_count": 3,
        "runtime_homogeneity": {"homogeneous": True},
        "aggregate_metrics": {
            "paired_estimand_descriptive": {
                "effects_by_mechanism": {
                    "expr": [1.0, 3.0, 5.0],
                    "fact_expr_sync": [2.0, 4.0, 6.0],
                }
            }
        },
    }


def _e() -> dict[str, object]:
    return {
        "status": "descriptive_only",
        "completed_run_count": 6,
        "expected_run_count": 6,
        "runtime_homogeneity": {"homogeneous": True},
        "primary_endpoint": "final_test_bb_per_100",
        "paired_comparisons": {
            "fact_target": {
                "metrics": {
                    "final_test_bb_per_100": {"effects": [2.0, 5.0, 8.0]}
                }
            }
        },
    }


def test_pilot_power_plan_uses_max_requirement_without_capping() -> None:
    plan = build_pilot_power_plan(_p(), _e())
    assert plan["status"] == (
        "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
    )
    assert plan["primary_mde_bb_per_100"] == 5.0
    assert plan["sensitivity_mdes_bb_per_100"] == [3.0, 5.0, 10.0]
    requirements = [
        item["sensitivity_by_mde_bb_per_100"]["5.0"][
            "required_seed_pairs_normal_approximation"
        ]
        for item in plan["contrasts"].values()
    ]
    assert plan["required_seed_pairs_primary_max_across_p_and_e"] == max(requirements)
    assert plan["no_silent_resource_cap"] is True


def test_pilot_power_plan_blocks_incomplete_matrix() -> None:
    campaign_e = _e()
    campaign_e["completed_run_count"] = 5
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert plan["required_seed_pairs_primary_max_across_p_and_e"] is None
    assert "campaign_e matrix is incomplete: 5/6" in plan["blockers"]
