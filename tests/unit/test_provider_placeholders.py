"""
模块说明：本模块测试占位 Provider 注册表。
核心职责：确保国内 LLM 占位 Provider 可被 doctor 路由识别，且不会伪造真实调用。
输入与输出：输入 provider 配置，输出 pytest 断言结果。
依赖边界：不访问网络，不读取真实密钥。
不负责：不测试真实厂商 API。
"""

import pytest

from agentmemeval.core.errors import ProviderError
from agentmemeval.llm.providers.placeholders import PLACEHOLDER_PROVIDER_INFO
from agentmemeval.llm.router import (
    PLACEHOLDER_PROVIDERS,
    build_llm_client,
    provider_health,
)

DOMESTIC_PLACEHOLDERS = {
    "dashscope",
    "baidu_qianfan",
    "tencent_hunyuan",
    "volcengine_doubao",
    "zhipu_glm",
    "moonshot_kimi",
    "minimax",
}

REMOVED_DOMESTIC_PLACEHOLDERS = {
    "baichuan",
    "lingyi_yi",
    "iflytek_spark",
    "stepfun",
    "sensenova",
    "siliconflow",
    "modelscope",
    "internlm",
}


def test_domestic_placeholder_providers_are_registered() -> None:
    """
    功能：验证新增国内 Provider 名称都进入路由白名单。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：新增占位应能被 doctor 识别，而不是落入未知 Provider。
    """

    assert DOMESTIC_PLACEHOLDERS <= PLACEHOLDER_PROVIDERS
    assert DOMESTIC_PLACEHOLDERS <= set(PLACEHOLDER_PROVIDER_INFO)
    assert REMOVED_DOMESTIC_PLACEHOLDERS.isdisjoint(PLACEHOLDER_PROVIDERS)
    assert REMOVED_DOMESTIC_PLACEHOLDERS.isdisjoint(PLACEHOLDER_PROVIDER_INFO)


def test_placeholder_healthcheck_exposes_access_hints() -> None:
    """
    功能：验证占位 Provider 的健康检查给出接入提示。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：占位不是可用实现，但要给用户后续接入线索。
    """

    health = provider_health({"provider": "moonshot_kimi"})
    assert health["available"] is False
    assert health["status"] == "placeholder"
    assert health["canonical_provider"] == "moonshot_kimi"
    assert health["api_key_env"] == "MOONSHOT_API_KEY"
    assert "Kimi" in str(health["display_name"])


def test_placeholder_alias_is_normalized() -> None:
    """
    功能：验证常见模型族别名可以规范到主 Provider 名。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：用户配置 doubao、kimi、ernie 这类俗称时也能得到清楚提示。
    """

    health = provider_health({"provider": "doubao"})
    assert health["provider"] == "doubao"
    assert health["canonical_provider"] == "volcengine_doubao"
    assert health["api_key_env"] == "ARK_API_KEY"


def test_placeholder_generation_fails_explicitly() -> None:
    """
    功能：验证占位 Provider 不会伪造真实生成。
    参数：无。
    返回：无。
    副作用：无。
    异常：预期抛出 ProviderError。
    设计说明：没有真实密钥 smoke test 前，只允许 healthcheck，不允许 run 假装成功。
    """

    client = build_llm_client({"provider": "zhipu_glm"})
    with pytest.raises(ProviderError, match="已预留注册位"):
        client.generate_structured(object(), dict)
