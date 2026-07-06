"""
模块说明：本模块提供尚未真实验证厂商的 Provider 注册位。
核心职责：清晰声明 openai、anthropic、google、xai、deepseek、qwen 的接入状态。
输入与输出：输入 Provider 配置，输出 healthcheck 或明确错误。
依赖边界：不导入任何厂商 SDK，不读取真实密钥值。
不负责：不伪造真实调用，不在默认测试中访问网络。
"""

from __future__ import annotations

from typing import TypeVar

from agentmemeval.core.errors import ProviderError

T = TypeVar("T")


class PlaceholderProviderClient:
    """
    功能：表示已注册但未在当前环境验证的真实厂商 Provider。
    参数：
        config：Provider 配置。
    返回：占位 Provider 实例。
    副作用：无。
    异常：调用生成时抛出 ProviderError。
    设计说明：用户能看到接入路径和环境变量，但测试不会依赖真实 API。
    """

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """
        功能：初始化占位 Provider。
        参数：
            config：包含 provider、model、api_key_env 等字段。
        返回：无。
        副作用：保存配置。
        异常：无。
        设计说明：显式区分“已注册接口”和“已真实验证”。
        """

        self.config = config or {}
        self.provider = str(self.config.get("provider", "unknown"))
        self.model = str(self.config.get("model", ""))

    def generate_structured(self, request: object, schema: type[T]) -> T:
        """
        功能：拒绝未验证 Provider 的默认真实调用。
        参数：
            request：调用请求。
            schema：目标结构。
        返回：不会返回。
        副作用：无。
        异常：始终抛出 ProviderError。
        设计说明：没有密钥和官方 SDK 验证时，不能声称已完成真实适配。
        """

        raise ProviderError(
            f"{self.provider} Provider 已预留注册位，但当前实现未用真实密钥验证。"
        )

    def healthcheck(self) -> dict[str, object]:
        """
        功能：返回接入说明。
        参数：无。
        返回：健康检查字典。
        副作用：无。
        异常：无。
        设计说明：doctor 命令能展示后续接入所需的环境变量。
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "available": False,
            "offline": False,
            "api_key_env": self.config.get("api_key_env", f"{self.provider.upper()}_API_KEY"),
            "base_url_env": self.config.get("base_url_env", f"{self.provider.upper()}_BASE_URL"),
            "message": "已注册占位；需要按 docs/development.md 完成真实适配和 smoke test。",
        }
