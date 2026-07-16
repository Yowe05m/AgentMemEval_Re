"""
模块说明：本模块负责生成和重建 Markdown 报告。
核心职责：把指标、聚合结果、图表和限制说明写入 report.md。
输入与输出：输入工件目录或指标字典，输出 Markdown 文本/文件。
依赖边界：依赖 JSONL 存储和指标函数，不依赖实验场景。
不负责：不运行 Provider，不修改记忆快照。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentmemeval.analysis.plots import plot_stack_curves
from agentmemeval.config.loader import load_raw_config
from agentmemeval.evaluation.aggregation import aggregate_metrics
from agentmemeval.evaluation.degeneracy import (
    build_run_validity,
    evaluate_behavior_health,
    evaluate_execution_health,
)
from agentmemeval.evaluation.metrics import compute_metrics
from agentmemeval.storage.jsonl_store import JsonlStore


def build_report_text(
    run_id: str,
    scenario: str,
    metrics: dict[str, Any],
    aggregate: dict[str, Any],
    plot_paths: list[str],
    notes: list[str],
) -> str:
    """
    功能：生成 Markdown 报告正文。
    参数：
        run_id：运行 ID。
        scenario：场景名称。
        metrics：运行指标。
        aggregate：聚合指标。
        plot_paths：图表路径。
        notes：限制说明。
    返回：Markdown 文本。
    副作用：无。
    异常：无。
    设计说明：报告明确区分主要指标和探索性指标，避免过度解读 smoke 结果。
    """

    counters = metrics.get("run_counters", {})
    lines = [
        f"# 实验报告：{run_id}",
        "",
        f"- 场景：{scenario}",
        f"- 手牌数：{counters.get('hands', 0)}",
        f"- 行动数：{counters.get('actions', 0)}",
        "",
        "## 主要指标",
    ]
    for agent_id, item in metrics.get("primary_metrics", {}).get("per_agent", {}).items():
        lines.append(
            f"- {agent_id}: chip_delta={item.get('chip_delta')}, "
            f"BB/100={item.get('bb_per_100'):.2f}, win_rate={item.get('win_rate'):.2f}"
        )
    lines.extend(
        [
            "",
            "## 探索性指标",
            "- 对手多样性字段："
            f"{bool(metrics.get('exploratory_metrics', {}).get('opponent_diversity'))}",
            f"- 聚合 BB/100 均值：{aggregate.get('bb_per_100', {}).get('mean', 0.0):.2f}",
            f"- 主表准入：{metrics.get('run_validity', {}).get('status', 'unknown')}",
            "- A7 统计设计："
            f"{metrics.get('primary_metrics', {}).get('table_run_estimand', {}).get('design', 'not_configured')}",  # noqa: E501
            "- A7 独立单位：每个 seed 的完整 table/run；桌内同机制 Agent 不作为独立 n。",
            f"- 跨 seed 主表状态：{aggregate.get('main_table', {}).get('status', 'unknown')}",
            "",
            "## 图表",
        ]
    )
    for path in plot_paths:
        lines.append(f"- {path}")
    lines.extend(["", "## 限制与假设"])
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def rebuild_report(run_dir: str | Path, big_blind: int = 2) -> dict[str, Any]:
    """
    功能：从原始工件重建指标、图表和报告。
    参数：
        run_dir：运行输出目录。
        big_blind：大盲数值。
    返回：包含 metrics、aggregate、report_path 的字典。
    副作用：重写 metrics.json、aggregate_metrics.json、plots 和 report.md。
    异常：文件缺失或 JSON 错误由标准库抛出。
    设计说明：满足“可从原始运行结果重新生成报告”的验收项。
    """

    root = Path(run_dir)
    hands = JsonlStore(root / "hand_summaries.jsonl").read_all()
    events = JsonlStore(root / "events.jsonl").read_all()
    exposure_path = root / "exposure_stats.json"
    exposure_stats = (
        json.loads(exposure_path.read_text(encoding="utf-8"))
        if exposure_path.exists()
        else None
    )
    existing_metrics_path = root / "metrics.json"
    existing_metrics = (
        json.loads(existing_metrics_path.read_text(encoding="utf-8"))
        if existing_metrics_path.exists()
        else {}
    )
    existing_per_agent = (
        existing_metrics.get("primary_metrics", {}).get("per_agent", {})
    )
    memory_metrics = {
        str(agent_id): dict(values.get("memory", {}))
        for agent_id, values in existing_per_agent.items()
        if isinstance(values, dict)
    }
    metrics = compute_metrics(
        hands,
        events,
        big_blind=big_blind,
        memory_metrics=memory_metrics,
        exposure_stats=exposure_stats,
    )
    existing_estimand = existing_metrics.get("primary_metrics", {}).get(
        "table_run_estimand"
    )
    if isinstance(existing_estimand, dict):
        metrics["primary_metrics"]["table_run_estimand"] = existing_estimand
    resolved_path = root / "resolved_config.yaml"
    config = load_raw_config(resolved_path) if resolved_path.exists() else {}
    experiment = dict(config.get("experiment", {}))
    protocol_path = root / "protocol_audit.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    behavior_health = evaluate_behavior_health(
        metrics,
        experiment,
        [str(item) for item in protocol.get("evaluation_target_ids", [])],
    )
    execution_health = evaluate_execution_health(hands, metrics)
    admission = dict(experiment.get("admission_audit", {}))
    metrics["behavior_health"] = behavior_health
    metrics["execution_health"] = execution_health
    metrics["run_validity"] = build_run_validity(
        admission,
        behavior_health,
        execution_health,
        str(experiment.get("run_mode", "smoke")),
    )
    aggregate = aggregate_metrics([metrics])
    plot = plot_stack_curves(hands, root / "plots")
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    report = build_report_text(
        run_id=str(manifest.get("run_id", root.name)),
        scenario=str(manifest.get("scenario", "unknown")),
        metrics=metrics,
        aggregate=aggregate,
        plot_paths=[plot],
        notes=["该报告由 report 命令从 JSONL 原始工件重建。"],
    )
    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"metrics": metrics, "aggregate_metrics": aggregate, "report_path": str(report_path)}
