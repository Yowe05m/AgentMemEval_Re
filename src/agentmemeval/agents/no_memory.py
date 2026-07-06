"""
模块说明：本模块实现 NoMemoryAgent 基线。
核心职责：提供不读写记忆但仍经过统一 LLM 决策管线的 Agent。
输入与输出：输入 Agent 配置和 Provider，输出 Agent 实例。
依赖边界：复用 LLMDecisionAgent 与 NullMemory。
不负责：不实现额外策略分支。
"""

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.core.protocols import LLMClient
from agentmemeval.memory.base import NullMemory


class NoMemoryAgent(LLMDecisionAgent):
    """
    功能：构造无记忆基线 Agent。
    参数：
        agent_id：Agent 标识。
        llm_client：Provider 实例。
        model：模型名称。
    返回：NoMemoryAgent。
    副作用：无。
    异常：无。
    设计说明：显式类名方便配置、测试和报告引用。
    """

    def __init__(self, agent_id: str, llm_client: LLMClient, model: str) -> None:
        """
        功能：初始化无记忆 Agent。
        参数：
            agent_id：Agent 标识。
            llm_client：Provider。
            model：模型名称。
        返回：无。
        副作用：创建 NullMemory。
        异常：无。
        设计说明：基线与其他 Agent 共享动作校验和提示词渲染。
        """

        super().__init__(agent_id, NullMemory(agent_id), llm_client, model=model)
