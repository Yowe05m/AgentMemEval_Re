"""
模块说明：本模块导出人格配置工具。
核心职责：为实验配置和文档提供默认 persona 字典。
输入与输出：输入名称，输出配置。
依赖边界：复用 memory 层默认人格文本。
不负责：不做人格筛选，不创建 Agent。
"""

from agentmemeval.memory.personality_driven import DEFAULT_PERSONAS


def list_default_personas() -> list[str]:
    """
    功能：列出内置人格示例名称。
    参数：无。
    返回：名称列表。
    副作用：无。
    异常：无。
    设计说明：README 和 doctor 可展示示例，但用户仍可配置任意 persona。
    """

    return sorted(DEFAULT_PERSONAS)
