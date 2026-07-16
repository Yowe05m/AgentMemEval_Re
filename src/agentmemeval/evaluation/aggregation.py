"""
模块说明：本模块负责从单次或多次运行指标中生成聚合指标。
核心职责：对 Agent 级收益和 BB/100 计算跨样本统计。
输入与输出：输入 metrics 字典列表，输出 aggregate_metrics 字典。
依赖边界：依赖 statistics 工具，不依赖文件系统。
不负责：不发现 outputs 目录，不绘图。
"""

from __future__ import annotations

import statistics
from typing import Any

from agentmemeval.evaluation.statistics import (
    holm_adjust,
    paired_sign_flip_p_value,
    summarize_values,
)

ENDPOINT_FIELDS = {
    "final_test_bb_per_100": "bb_per_100",
    "final_test_chip_per_hand": "test_chip_per_hand",
}


def build_table_run_estimand(
    checkpoint_results: list[dict[str, Any]],
    *,
    seed: int,
    run_id: str,
    endpoint: str,
    baseline_mechanism: str,
    statistical_plan_status: str,
    multiple_comparison_method: str,
    required_seed_pairs: int | None,
) -> dict[str, Any]:
    """Collapse same-table target agents into one mechanism value for one seed/run."""

    field = ENDPOINT_FIELDS.get(endpoint)
    if field is None:
        raise ValueError(f"unsupported primary endpoint: {endpoint}")
    if not checkpoint_results:
        raise ValueError("primary estimand requires checkpoint generalization results")
    final_checkpoint = max(int(item["checkpoint_hand"]) for item in checkpoint_results)
    final_rows = [
        item
        for item in checkpoint_results
        if int(item["checkpoint_hand"]) == final_checkpoint
    ]
    by_mechanism: dict[str, list[float]] = {}
    for row in final_rows:
        by_mechanism.setdefault(str(row["mechanism"]), []).append(float(row[field]))
    mechanism_values = {
        mechanism: statistics.mean(values)
        for mechanism, values in sorted(by_mechanism.items())
    }
    if baseline_mechanism not in mechanism_values:
        raise ValueError(
            f"primary baseline mechanism {baseline_mechanism!r} is absent from final test"
        )
    baseline_value = mechanism_values[baseline_mechanism]
    effects = {
        mechanism: value - baseline_value
        for mechanism, value in mechanism_values.items()
        if mechanism != baseline_mechanism
    }
    return {
        "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
        "independent_unit": "one complete table/run within seed",
        "seed": int(seed),
        "run_id": run_id,
        "final_checkpoint_hand": final_checkpoint,
        "endpoint": endpoint,
        "baseline_mechanism": baseline_mechanism,
        "within_table_aggregation": "arithmetic_mean_across_same_mechanism_agents",
        "mechanism_values": mechanism_values,
        "effects_vs_baseline": effects,
        "statistical_plan_status": statistical_plan_status,
        "multiple_comparison_method": multiple_comparison_method,
        "required_seed_pairs": required_seed_pairs,
    }


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能：聚合一组运行指标。
    参数：
        metrics_list：metrics 字典列表。
    返回：聚合指标。
    副作用：无。
    异常：无。
    设计说明：本阶段 smoke 多为单 seed，也保留多 seed 聚合接口。
    """

    bb_values: list[float] = []
    chip_values: list[float] = []
    for metrics in metrics_list:
        per_agent = metrics.get("primary_metrics", {}).get("per_agent", {})
        for item in per_agent.values():
            bb_values.append(float(item.get("bb_per_100", 0.0)))
            chip_values.append(float(item.get("chip_delta", 0.0)))
    eligible_units = [
        metrics["primary_metrics"]["table_run_estimand"]
        for metrics in metrics_list
        if metrics.get("run_validity", {}).get("paper_eligible")
        and isinstance(
            metrics.get("primary_metrics", {}).get("table_run_estimand"), dict
        )
    ]
    all_units = [
        metrics["primary_metrics"]["table_run_estimand"]
        for metrics in metrics_list
        if isinstance(
            metrics.get("primary_metrics", {}).get("table_run_estimand"), dict
        )
    ]
    main_table = _aggregate_table_run_units(eligible_units)
    descriptive_estimand = _aggregate_table_run_units(all_units, inference=False)
    return {
        "bb_per_100": summarize_values(bb_values),
        "chip_delta": summarize_values(chip_values),
        "sample_count": len(metrics_list),
        "main_table_included_run_count": sum(
            bool(metrics.get("run_validity", {}).get("paper_eligible"))
            for metrics in metrics_list
        ),
        "descriptive_only": True,
        "main_table": main_table,
        "paired_estimand_descriptive": descriptive_estimand,
        "inference_status": main_table["status"],
        "legacy_agent_pool_warning": (
            "Agent-level values are descriptive only; same-table agents are not independent."
        ),
    }


def _aggregate_table_run_units(
    units: list[dict[str, Any]], *, inference: bool = True
) -> dict[str, Any]:
    """Aggregate exactly one table/run effect per unique seed."""

    if not units:
        return {
            "status": "no_paper_eligible_runs" if inference else "no_estimand_units",
            "included_run_count": 0,
            "independent_seed_count": 0,
            "metrics": None,
        }
    identity_fields = (
        "design",
        "endpoint",
        "baseline_mechanism",
        "multiple_comparison_method",
        "required_seed_pairs",
    )
    identities = {
        tuple(unit.get(field) for field in identity_fields) for unit in units
    }
    if len(identities) != 1:
        return {
            "status": "blocked_inconsistent_estimand_configuration",
            "included_run_count": len(units),
            "independent_seed_count": 0,
            "metrics": None,
        }
    seeds = [int(unit["seed"]) for unit in units]
    if len(set(seeds)) != len(seeds):
        return {
            "status": "blocked_duplicate_seed_units",
            "included_run_count": len(units),
            "independent_seed_count": len(set(seeds)),
            "metrics": None,
            "duplicate_seeds": sorted(seed for seed in set(seeds) if seeds.count(seed) > 1),
        }
    effect_sets = [set(unit.get("effects_vs_baseline", {})) for unit in units]
    if len({tuple(sorted(items)) for items in effect_sets}) != 1:
        return {
            "status": "blocked_inconsistent_mechanism_family",
            "included_run_count": len(units),
            "independent_seed_count": len(seeds),
            "metrics": None,
        }
    mechanisms = sorted(set.intersection(*effect_sets)) if effect_sets else []
    effects = {
        mechanism: [float(unit["effects_vs_baseline"][mechanism]) for unit in units]
        for mechanism in mechanisms
    }
    summaries = {name: summarize_values(values) for name, values in effects.items()}
    raw_p = (
        {name: paired_sign_flip_p_value(values) for name, values in effects.items()}
        if inference and effects
        else {}
    )
    method = str(units[0].get("multiple_comparison_method", "holm"))
    adjusted_p = holm_adjust(raw_p) if raw_p and method == "holm" else raw_p
    for name in summaries:
        summaries[name]["raw_p_value"] = raw_p.get(name)
        summaries[name]["adjusted_p_value"] = adjusted_p.get(name)
    statuses = {str(unit.get("statistical_plan_status")) for unit in units}
    design, endpoint, baseline, correction, required_seed_pairs = next(iter(identities))
    enough_seeds = (
        isinstance(required_seed_pairs, int) and len(seeds) >= required_seed_pairs
    )
    if not inference:
        status = "descriptive_only"
    elif statuses != {"frozen"}:
        status = "blocked_statistical_plan_not_frozen"
    elif not enough_seeds:
        status = "insufficient_preregistered_seed_pairs"
    else:
        status = "ready"
    return {
        "status": status,
        "included_run_count": len(units),
        "independent_seed_count": len(seeds),
        "matched_seeds": sorted(seeds),
        "design": design,
        "endpoint": endpoint,
        "baseline_mechanism": baseline,
        "multiple_comparison_method": correction,
        "paired_test_method": (
            "two_sided_exact_sign_flip_up_to_20_pairs_else_"
            "deterministic_monte_carlo_100000"
        ),
        "required_seed_pairs": required_seed_pairs,
        "effects_by_mechanism": effects,
        "metrics": summaries,
    }


def validate_runtime_homogeneity(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare hardware and service identities before any cross-run formal aggregation."""

    fields = {
        "code": lambda item: tuple(
            sorted(item.get("metadata", {}).get("code", {}).items())
        ),
        "gpu": lambda item: tuple(
            (device.get("name"), device.get("driver"), device.get("pci_bus_id"))
            for device in item.get("metadata", {}).get("gpu", {}).get("devices", [])
        ),
        "cuda": lambda item: (
            item.get("metadata", {})
            .get("model_service_runtime", {})
            .get("torch_cuda_version")
            or item.get("metadata", {}).get("cuda", {}).get("torch_cuda_version")
        ),
        "vllm_runtime": lambda item: (
            item.get("metadata", {})
            .get("model_service_runtime", {})
            .get("vllm_version")
        ),
        "model": lambda item: tuple(
            sorted(item.get("metadata", {}).get("model", {}).items())
        ),
        "service": lambda item: repr(item.get("metadata", {}).get("service", {})),
        "embedding": lambda item: repr(
            item.get("metadata", {}).get("embedding", {})
        ),
        "prompts": lambda item: tuple(
            sorted(item.get("metadata", {}).get("prompts", {}).items())
        ),
    }
    mismatches: dict[str, list[object]] = {}
    identity: dict[str, object] = {}
    for name, getter in fields.items():
        values = [getter(manifest) for manifest in manifests]
        unique = list(dict.fromkeys(values))
        if len(unique) > 1:
            mismatches[name] = unique
        elif unique:
            identity[name] = unique[0]
    return {
        "homogeneous": not mismatches,
        "run_count": len(manifests),
        "mismatches": mismatches,
        "identity": identity if manifests and not mismatches else None,
        "formal_aggregation_allowed": bool(manifests) and not mismatches,
    }
