"""
模块说明：本模块提供观察对象的安全检查工具。
核心职责：检测提示词或事实记忆是否包含不应暴露的私有牌信息。
输入与输出：输入观察对象与文本，输出布尔或抛出中文错误。
依赖边界：只依赖领域对象和标准字符串处理。
不负责：不生成观察，不推进环境，不判断牌力。
"""

from __future__ import annotations

from agentmemeval.core.domain import AgentObservation
from agentmemeval.core.errors import EnvironmentError


def assert_observation_has_no_private_leak(
    observation: AgentObservation,
    forbidden_cards: dict[str, list[str]],
    rendered_text: str,
) -> None:
    """
    功能：断言渲染文本没有泄露对手私有手牌。
    参数：
        observation：当前观察者的可见观察。
        forbidden_cards：按对手 ID 给出的不应可见牌。
        rendered_text：待检查提示词或记忆文本。
    返回：无。
    副作用：无。
    异常：发现泄露时抛出 EnvironmentError。
    设计说明：该函数用于测试和开发自检，避免新提示词模板绕过可见性边界。
    """

    own = set(observation.hole_cards)
    for other_id, cards in forbidden_cards.items():
        for card in cards:
            if card not in own and card in rendered_text:
                raise EnvironmentError(f"观察文本泄露了 {other_id} 的私有手牌 {card}")


def observation_to_compact_text(observation: AgentObservation) -> str:
    """
    功能：把观察转换为适合日志和检索的短文本。
    参数：
        observation：合法可见观察。
    返回：短文本摘要。
    副作用：无。
    异常：无。
    设计说明：检索特征只来自可见信息，避免事实库引入上帝视角。
    """

    legal = ",".join(sorted(observation.legal_actions.types()))
    return (
        f"phase={observation.phase}; table={observation.table_id}; "
        f"hole={' '.join(observation.hole_cards)}; board={' '.join(observation.community_cards)}; "
        f"pot={observation.pot}; to_call={observation.to_call}; legal={legal}"
    )
