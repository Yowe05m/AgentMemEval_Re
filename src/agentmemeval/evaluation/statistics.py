"""
模块说明：本模块提供基础统计函数。
核心职责：计算均值、标准差和简易置信区间。
输入与输出：输入数值列表，输出统计字典。
依赖边界：只依赖标准库 math。
不负责：不做显著性检验，不解释研究结论。
"""

from __future__ import annotations

import math


def summarize_values(values: list[float]) -> dict[str, float]:
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
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(n)
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }
