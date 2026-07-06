"""
模块说明：本模块导出评估入口。
核心职责：提供指标、聚合和报告函数的稳定导入路径。
输入与输出：无直接运行输入输出。
依赖边界：只导入轻量函数。
不负责：不运行实验。
"""

from agentmemeval.evaluation.aggregation import aggregate_metrics
from agentmemeval.evaluation.metrics import compute_metrics
from agentmemeval.evaluation.reporting import build_report_text, rebuild_report

__all__ = ["compute_metrics", "aggregate_metrics", "build_report_text", "rebuild_report"]
