"""
模块说明：本模块负责从单次或多次运行指标中生成聚合指标。
核心职责：对 Agent 级收益和 BB/100 计算跨样本统计。
输入与输出：输入 metrics 字典列表，输出 aggregate_metrics 字典。
依赖边界：依赖 statistics 工具，不依赖文件系统。
不负责：不发现 outputs 目录，不绘图。
"""

from __future__ import annotations

from typing import Any

from agentmemeval.evaluation.statistics import summarize_values


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
    return {
        "bb_per_100": summarize_values(bb_values),
        "chip_delta": summarize_values(chip_values),
        "sample_count": len(metrics_list),
    }
