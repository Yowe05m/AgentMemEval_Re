"""
模块说明：本模块提供人格提示词示例。
核心职责：暴露论文中使用的四类 MBTI 示例，同时允许配置覆盖。
输入与输出：输入 persona 名称，输出提示词字典。
依赖边界：复用 memory.personality_driven 的默认示例。
不负责：不限制用户只能使用 MBTI。
"""

from agentmemeval.memory.personality_driven import DEFAULT_PERSONAS


def get_persona_prompt(name: str) -> dict[str, str]:
    """
    功能：获取人格提示词配置。
    参数：
        name：人格名称。
    返回：提示词字典。
    副作用：无。
    异常：无。
    设计说明：未知 persona 返回通用占位，支持配置注册任意人格。
    """

    return DEFAULT_PERSONAS.get(
        name,
        {
            "description": name,
            "decision_prompt": "按该 persona 的偏好决策。",
            "filter_prompt": "按该 persona 的偏好筛选记忆。",
            "update_prompt": "按该 persona 的偏好更新记忆。",
        },
    )
