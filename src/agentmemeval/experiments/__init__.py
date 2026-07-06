"""
模块说明：本模块导出实验运行入口。
核心职责：提供 run_config 和场景注册函数的稳定路径。
输入与输出：无直接运行输入输出。
依赖边界：只导入轻量入口。
不负责：不自动运行实验。
"""

from agentmemeval.experiments.registry import get_scenario
from agentmemeval.experiments.runner import run_config, run_resolved_config

__all__ = ["run_config", "run_resolved_config", "get_scenario"]
