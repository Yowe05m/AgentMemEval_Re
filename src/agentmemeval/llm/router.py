"""
模块说明：本模块负责根据配置创建 LLM Provider。
核心职责：集中注册 mock、openai-compatible 和各厂商占位 Provider。
输入与输出：输入 Provider 配置字典，输出 LLMClient。
依赖边界：只在创建时导入具体 Provider，不让 Agent 处理厂商差异。
不负责：不读取实验 YAML，不执行真实 smoke test。
"""

from __future__ import annotations

from agentmemeval.core.errors import ProviderError
from agentmemeval.core.protocols import LLMClient
from agentmemeval.llm.mock import MockLLMClient
from agentmemeval.llm.providers.openai_compatible import OpenAICompatibleClient
from agentmemeval.llm.providers.placeholders import (
    PLACEHOLDER_PROVIDERS,
    PlaceholderProviderClient,
)


def build_llm_client(config: dict[str, object]) -> LLMClient:
    """
    功能：根据配置创建 Provider 实例。
    参数：
        config：Provider 配置。
    返回：LLMClient。
    副作用：无网络副作用。
    异常：未知 Provider 时抛出 ProviderError。
    设计说明：上层只拿到统一协议，不写 `if provider == ...` 的业务分支。
    """

    provider = str(config.get("provider", "mock"))
    if provider == "mock":
        return MockLLMClient(config)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    if provider in PLACEHOLDER_PROVIDERS:
        return PlaceholderProviderClient(config)
    raise ProviderError(f"未知 Provider：{provider}")


def provider_health(config: dict[str, object]) -> dict[str, object]:
    """
    功能：创建 Provider 并返回健康检查结果。
    参数：
        config：Provider 配置。
    返回：健康检查字典。
    副作用：不进行真实模型生成。
    异常：未知 Provider 时抛出 ProviderError。
    设计说明：CLI doctor 使用该函数给出可执行的接入状态。
    """

    return build_llm_client(config).healthcheck()
