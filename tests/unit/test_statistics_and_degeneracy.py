from __future__ import annotations

import pytest

from agentmemeval.evaluation.aggregation import (
    aggregate_metrics,
    build_table_run_estimand,
    validate_runtime_homogeneity,
)
from agentmemeval.evaluation.degeneracy import (
    evaluate_behavior_health,
    evaluate_execution_health,
)
from agentmemeval.evaluation.statistics import (
    estimate_paired_seed_requirement,
    holm_adjust,
    paired_sign_flip_p_value,
    summarize_values,
)


def _metrics(vpip: float = 0.30, fold_rate: float = 0.40) -> dict[str, object]:
    return {
        "primary_metrics": {
            "per_agent": {
                "fact_00": {
                    "vpip": vpip,
                    "fold_rate": fold_rate,
                    "voluntary_participation_hands": 8,
                    "all_in_hand_rate": 0.05,
                    "bust_hand_rate": 0.0,
                    "street_coverage": {"preflop": 1.0, "flop": 0.25},
                    "hand_reward_sensitivity": {
                        "share_of_absolute_reward_activity": 0.35
                    },
                    "memory": {
                        "empty_retrieval_rate": 0.20,
                        "max_structural_signature_share": 0.10,
                    },
                }
            }
        }
    }


def test_behavior_health_pending_pilot_never_enters_main_table() -> None:
    audit = evaluate_behavior_health(_metrics(), {"behavior_threshold_status": "pending_pilot"})
    assert audit["calibration_required"] is True
    assert audit["valid_for_main_table"] is False


def test_behavior_health_frozen_thresholds_reject_degenerate_agent() -> None:
    config = {
        "behavior_threshold_status": "frozen",
        "behavior_thresholds": {"min_vpip": 0.10, "max_fold_rate": 0.80},
    }
    audit = evaluate_behavior_health(_metrics(vpip=0.02, fold_rate=0.95), config)
    assert audit["degenerate"] is True
    assert audit["valid_for_main_table"] is False
    assert {item["metric"] for item in audit["checks"] if not item["passed"]} == {
        "vpip",
        "fold_rate",
    }


def test_execution_health_rejects_fallback_and_conservation_violation() -> None:
    metrics = {
        "exploratory_metrics": {
            "decision_quality": {"combined": {"fallback_count": 1}}
        }
    }
    hands = [
        {
            "hand_id": "h1",
            "rewards": {"a": 5, "b": -4},
            "starting_stacks": {"a": 100, "b": 100},
            "final_stacks": {"a": 105, "b": 96},
        }
    ]
    audit = evaluate_execution_health(hands, metrics)
    assert audit["valid"] is False
    assert audit["fallback_count"] == 1
    assert audit["reward_conservation_violation_hand_ids"] == ["h1"]
    assert audit["stack_conservation_violation_hand_ids"] == ["h1"]


def test_execution_health_rejects_experience_revision_fallback_once_per_agent() -> None:
    metrics = _metrics()
    per_agent = metrics["primary_metrics"]["per_agent"]  # type: ignore[index]
    per_agent["fact_00"]["memory"]["revision_fallback_count"] = 2  # type: ignore[index]
    audit = evaluate_execution_health([], metrics)
    assert audit["valid"] is False
    assert audit["fallback_count"] == 0
    assert audit["memory_revision_fallback_count"] == 2


def test_student_t_interval_is_used_for_small_samples() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])
    assert summary["ci95_method"] == "student_t"
    assert summary["ci95_critical_value"] == pytest.approx(4.303)


def test_paired_power_plan_is_explicitly_planning_only() -> None:
    plan = estimate_paired_seed_requirement([1.0, 2.0, 4.0], 1.0)
    assert plan["required_seed_pairs_normal_approximation"] >= 2
    assert plan["formal_status"] == "requires_preregistered_A7_estimand_and_final_review"


def test_A7_R_collapses_agents_within_table_before_seed_effect() -> None:
    rows = [
        {"checkpoint_hand": 10, "mechanism": "fact", "bb_per_100": 1.0},
        {"checkpoint_hand": 10, "mechanism": "fact", "bb_per_100": 3.0},
        {"checkpoint_hand": 10, "mechanism": "expr", "bb_per_100": 8.0},
        {"checkpoint_hand": 10, "mechanism": "expr", "bb_per_100": 4.0},
    ]
    unit = build_table_run_estimand(
        rows,
        seed=7,
        run_id="run-7",
        endpoint="final_test_bb_per_100",
        baseline_mechanism="fact",
        statistical_plan_status="pending_pilot_power_calibration",
        multiple_comparison_method="holm",
        required_seed_pairs=None,
    )
    assert unit["mechanism_values"] == {"expr": 6.0, "fact": 2.0}
    assert unit["effects_vs_baseline"] == {"expr": 4.0}
    assert unit["seed"] == 7


def test_formal_A7_R_aggregation_uses_one_effect_per_seed_and_holm() -> None:
    metrics = []
    for seed, effect in ((1, 2.0), (2, 4.0), (3, 6.0)):
        metrics.append(
            {
                "primary_metrics": {
                    "per_agent": {},
                    "table_run_estimand": {
                        "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
                        "seed": seed,
                        "endpoint": "final_test_bb_per_100",
                        "baseline_mechanism": "fact",
                        "effects_vs_baseline": {"expr": effect, "sync": effect / 2},
                        "statistical_plan_status": "frozen",
                        "multiple_comparison_method": "holm",
                        "required_seed_pairs": 3,
                    },
                },
                "run_validity": {"paper_eligible": True},
            }
        )
    aggregate = aggregate_metrics(metrics)
    main = aggregate["main_table"]
    assert main["status"] == "ready"
    assert main["independent_seed_count"] == 3
    assert main["effects_by_mechanism"] == {
        "expr": [2.0, 4.0, 6.0],
        "sync": [1.0, 2.0, 3.0],
    }
    assert main["metrics"]["expr"]["n"] == 3.0
    assert main["metrics"]["expr"]["adjusted_p_value"] is not None


def test_duplicate_seed_is_blocked_from_main_table() -> None:
    unit = {
        "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
        "seed": 1,
        "endpoint": "final_test_bb_per_100",
        "baseline_mechanism": "fact",
        "effects_vs_baseline": {"expr": 1.0},
        "statistical_plan_status": "frozen",
        "multiple_comparison_method": "holm",
        "required_seed_pairs": 2,
    }
    metrics = [
        {"primary_metrics": {"per_agent": {}, "table_run_estimand": unit},
         "run_validity": {"paper_eligible": True}},
        {"primary_metrics": {"per_agent": {}, "table_run_estimand": dict(unit)},
         "run_validity": {"paper_eligible": True}},
    ]
    assert aggregate_metrics(metrics)["main_table"]["status"] == (
        "blocked_duplicate_seed_units"
    )


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    assert paired_sign_flip_p_value([1.0, 2.0, 3.0]) == pytest.approx(0.25)
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_runtime_homogeneity_detects_hardware_mix() -> None:
    def manifest(gpu: str) -> dict[str, object]:
        return {
            "metadata": {
                "gpu": {"devices": [{"name": gpu, "driver": "1", "pci_bus_id": "0"}]},
                "cuda": {"torch_cuda_version": "12.8"},
                "model": {"name": "m", "revision": "r", "weights_hash": "h"},
                "service": {"port": 8000},
            }
        }

    audit = validate_runtime_homogeneity([manifest("RTX 5090"), manifest("RTX 4090")])
    assert audit["homogeneous"] is False
    assert "gpu" in audit["mismatches"]


def test_runtime_homogeneity_uses_model_service_cuda_and_vllm() -> None:
    def manifest(cuda: str, vllm: str) -> dict[str, object]:
        return {
            "metadata": {
                "gpu": {
                    "devices": [
                        {"name": "RTX 4090", "driver": "1", "pci_bus_id": "0"}
                    ]
                },
                "cuda": {"collection_error": "ModuleNotFoundError"},
                "model_service_runtime": {
                    "status": "verified",
                    "torch_cuda_version": cuda,
                    "vllm_version": vllm,
                },
                "model": {"name": "m", "revision": "r", "weights_hash": "h"},
                "service": {"port": 8000},
            }
        }

    audit = validate_runtime_homogeneity(
        [manifest("13.0", "0.23.1"), manifest("13.0", "0.23.2")]
    )
    assert audit["homogeneous"] is False
    assert audit["mismatches"] == {"vllm_runtime": ["0.23.1", "0.23.2"]}
