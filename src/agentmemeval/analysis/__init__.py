"""
模块说明：本模块导出分析绘图工具。
核心职责：提供图表生成函数的稳定导入路径。
输入与输出：无直接运行输入输出。
依赖边界：仅导入 plots 模块函数。
不负责：不计算指标。
"""

from agentmemeval.analysis.plots import plot_stack_curves

__all__ = ["plot_stack_curves"]
