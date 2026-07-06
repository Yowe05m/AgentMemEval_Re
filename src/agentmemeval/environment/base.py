"""
模块说明：本模块提供环境层的公共导出。
核心职责：让外部代码通过 environment.base 引用 PokerEnvironment 协议。
输入与输出：无直接运行输入输出。
依赖边界：只转发核心协议，不引入具体环境实现。
不负责：不创建环境，不处理配置。
"""

from agentmemeval.core.protocols import PokerEnvironment

__all__ = ["PokerEnvironment"]
