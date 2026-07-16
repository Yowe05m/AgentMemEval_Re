"""
模块说明：本模块提供基础统计函数。
核心职责：计算均值、标准差和简易置信区间。
输入与输出：输入数值列表，输出统计字典。
依赖边界：只依赖标准库 math。
不负责：不做显著性检验，不解释研究结论。
"""

from __future__ import annotations

import math
import random
import statistics
from itertools import product

T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def summarize_values(
    values: list[float], *, bootstrap_samples: int = 2000, bootstrap_seed: int = 20260715
) -> dict[str, float | str]:
    """
    功能：汇总一组数值。
    参数：
        values：数值列表。
    返回：包含 n、mean、std、ci95_low、ci95_high 的字典。
    副作用：无。
    异常：无。
    设计说明：小样本 smoke 也返回结构完整的统计字段。
    """

    n = len(values)
    if n == 0:
        return {
            "n": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0,
            "ci95_low": 0.0, "ci95_high": 0.0,
            "bootstrap_ci95_low": 0.0, "bootstrap_ci95_high": 0.0,
        }
    mean = sum(values) / n
    if n == 1:
        return {
            "n": 1.0, "mean": mean, "median": mean, "std": 0.0,
            "ci95_low": mean, "ci95_high": mean,
            "bootstrap_ci95_low": mean, "bootstrap_ci95_high": mean,
        }
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(variance)
    critical = T_CRITICAL_975.get(n - 1, 1.96)
    half_width = critical * std / math.sqrt(n)
    rng = random.Random(bootstrap_seed)
    bootstrap_means = sorted(
        statistics.mean(rng.choices(values, k=n)) for _ in range(max(1, bootstrap_samples))
    )
    low_index = int((len(bootstrap_means) - 1) * 0.025)
    high_index = int((len(bootstrap_means) - 1) * 0.975)
    return {
        "n": float(n),
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_method": "student_t",
        "ci95_critical_value": critical,
        "bootstrap_ci95_low": bootstrap_means[low_index],
        "bootstrap_ci95_high": bootstrap_means[high_index],
    }


def summarize_paired_effects(
    left_by_seed: dict[int, float], right_by_seed: dict[int, float]
) -> dict[str, object]:
    """Summarize left-minus-right effects over exactly matched seeds."""

    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    differences = [float(left_by_seed[seed]) - float(right_by_seed[seed]) for seed in seeds]
    return {
        "definition": "left minus right on matched seed",
        "matched_seeds": seeds,
        "effects": differences,
        "summary": summarize_values(differences),
    }


def paired_sign_flip_p_value(
    differences: list[float], *, monte_carlo_samples: int = 100_000
) -> float:
    """Return a two-sided exact or deterministic Monte Carlo sign-flip p-value."""

    values = [float(value) for value in differences]
    if not values:
        raise ValueError("paired sign-flip test requires at least one difference")
    observed = abs(statistics.mean(values))
    if len(values) > 20:
        rng = random.Random(20260716)
        extreme = sum(
            abs(statistics.mean(rng.choice((-1.0, 1.0)) * value for value in values))
            >= observed - 1e-12
            for _ in range(monte_carlo_samples)
        )
        return (extreme + 1) / (monte_carlo_samples + 1)
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(
            statistics.mean(
                sign * value for sign, value in zip(signs, values, strict=True)
            )
        )
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust a preregistered family of p-values with Holm's step-down method."""

    if any(value < 0.0 or value > 1.0 for value in p_values.values()):
        raise ValueError("p-values must be within [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - index) * value))
        adjusted[name] = running
    return adjusted


def estimate_paired_seed_requirement(
    pilot_differences: list[float],
    minimum_meaningful_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float | int | str]:
    """Plan, but never silently authorize, seed count from an independent paired pilot."""

    if len(pilot_differences) < 2:
        raise ValueError("paired power planning requires at least two pilot differences")
    if minimum_meaningful_effect <= 0:
        raise ValueError("minimum_meaningful_effect must be positive")
    if alpha != 0.05 or power != 0.80:
        raise ValueError("current audited approximation is frozen to alpha=0.05, power=0.80")
    std = statistics.stdev(pilot_differences)
    if std == 0:
        required = 2
    else:
        required = math.ceil(((1.96 + 0.841621) * std / minimum_meaningful_effect) ** 2)
    return {
        "pilot_pair_count": len(pilot_differences),
        "pilot_difference_std": std,
        "minimum_meaningful_effect": minimum_meaningful_effect,
        "alpha": alpha,
        "power": power,
        "required_seed_pairs_normal_approximation": max(2, required),
        "method": "paired_normal_approximation_for_planning_only",
        "formal_status": "requires_preregistered_A7_estimand_and_final_review",
    }
