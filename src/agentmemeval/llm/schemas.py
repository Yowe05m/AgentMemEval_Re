"""
模块说明：本模块定义 LLM 层的请求与响应结构。
核心职责：让 Agent、Provider 和日志共享同一组结构化字段。
输入与输出：输入为观察、记忆与提示词，输出为 Provider 元数据。
依赖边界：依赖核心领域对象，不依赖具体厂商 SDK。
不负责：不执行网络请求，不校验扑克动作合法性。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.core.domain import AgentObservation, JsonDict, MemoryContext


@dataclass(slots=True)
class LLMRequest:
    """
    功能：表示一次结构化 LLM 调用请求。
    参数：
        observation：当前合法可见观察。
        memory_context：注入的记忆上下文。
        system_prompt：系统提示词。
        user_prompt：用户提示词。
        provider_config：Provider 运行配置。
        metadata：实验层附加元数据。
    返回：请求对象。
    副作用：无。
    异常：无。
    设计说明：真实 Provider 和 mock Provider 使用同一请求对象，便于替换。
    """

    observation: AgentObservation
    memory_context: MemoryContext
    system_prompt: str
    user_prompt: str
    provider_config: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class LLMCallStats:
    """
    功能：记录一次 Provider 调用的成本代理与稳定性信息。
    参数：
        provider：Provider 名称。
        model：模型名称。
        retries：重试次数。
        elapsed_ms：耗时毫秒。
        prompt_tokens：输入 token 估计或厂商返回值。
        completion_tokens：输出 token 估计或厂商返回值。
        raw_saved：是否保存原始响应。
    返回：调用统计对象。
    副作用：无。
    异常：无。
    设计说明：即使 mock 没有真实 token，也记录可比较的代理指标。
    """

    provider: str
    model: str
    retries: int = 0
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_saved: bool = False

    def to_dict(self) -> JsonDict:
        """
        功能：转换为 JSON 友好字典。
        参数：无。
        返回：字典。
        副作用：无。
        异常：无。
        设计说明：避免存储层依赖 dataclass 细节。
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "retries": self.retries,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "raw_saved": self.raw_saved,
        }
