"""
模块说明：本模块提供尚未真实验证厂商的 Provider 注册位。
核心职责：集中声明国内外主流 LLM Provider 的占位接入状态和后续接入线索。
输入与输出：输入 Provider 配置，输出 healthcheck 或明确错误。
依赖边界：不导入任何厂商 SDK，不读取真实密钥值。
不负责：不伪造真实调用，不在默认测试中访问网络。
"""

from __future__ import annotations

from typing import TypeVar

from agentmemeval.core.errors import ProviderError

T = TypeVar("T")

PLACEHOLDER_PROVIDER_INFO: dict[str, dict[str, str]] = {
    "openai": {
        "display_name": "OpenAI",
        "model_family": "GPT / o-series",
        "default_model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "docs_url": "https://platform.openai.com/docs",
        "region": "global",
    },
    "anthropic": {
        "display_name": "Anthropic",
        "model_family": "Claude",
        "default_model": "claude-sonnet-4",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "docs_url": "https://docs.anthropic.com/",
        "region": "global",
    },
    "google": {
        "display_name": "Google",
        "model_family": "Gemini",
        "default_model": "gemini-2.5-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "GOOGLE_BASE_URL",
        "docs_url": "https://ai.google.dev/gemini-api/docs",
        "region": "global",
    },
    "xai": {
        "display_name": "xAI",
        "model_family": "Grok",
        "default_model": "grok-4",
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_BASE_URL",
        "docs_url": "https://docs.x.ai/",
        "region": "global",
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "model_family": "DeepSeek V/R 系列",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "docs_url": "https://api-docs.deepseek.com/",
        "region": "china",
    },
    "qwen": {
        "display_name": "通义千问 Qwen",
        "model_family": "Qwen",
        "default_model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "docs_url": "https://help.aliyun.com/zh/model-studio/",
        "region": "china",
    },
    "dashscope": {
        "display_name": "阿里云百炼 / DashScope",
        "model_family": "Qwen 及百炼模型广场",
        "default_model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "docs_url": "https://help.aliyun.com/zh/model-studio/",
        "region": "china",
    },
    "baidu_qianfan": {
        "display_name": "百度千帆 / 文心 ERNIE",
        "model_family": "ERNIE / 文心",
        "default_model": "ernie-4.5-turbo-128k",
        "api_key_env": "QIANFAN_API_KEY",
        "base_url_env": "QIANFAN_BASE_URL",
        "docs_url": "https://cloud.baidu.com/product/wenxinworkshop",
        "region": "china",
    },
    "tencent_hunyuan": {
        "display_name": "腾讯混元",
        "model_family": "Hunyuan / Tencent HY",
        "default_model": "hunyuan-turbo",
        "api_key_env": "TENCENT_HUNYUAN_API_KEY",
        "base_url_env": "TENCENT_HUNYUAN_BASE_URL",
        "docs_url": "https://cloud.tencent.com/product/hunyuan",
        "region": "china",
    },
    "volcengine_doubao": {
        "display_name": "火山方舟 / 豆包",
        "model_family": "Doubao / Seed",
        "default_model": "doubao-seed-1-6",
        "api_key_env": "ARK_API_KEY",
        "base_url_env": "ARK_BASE_URL",
        "docs_url": "https://www.volcengine.com/docs/82379/1263482",
        "region": "china",
    },
    "zhipu_glm": {
        "display_name": "智谱 GLM",
        "model_family": "GLM",
        "default_model": "glm-4.5",
        "api_key_env": "ZHIPUAI_API_KEY",
        "base_url_env": "ZHIPUAI_BASE_URL",
        "docs_url": "https://docs.bigmodel.cn/",
        "region": "china",
    },
    "moonshot_kimi": {
        "display_name": "Moonshot AI / Kimi",
        "model_family": "Kimi",
        "default_model": "kimi-k2.7-code",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_BASE_URL",
        "docs_url": "https://platform.moonshot.cn/docs/",
        "region": "china",
    },
    "minimax": {
        "display_name": "MiniMax",
        "model_family": "MiniMax M/Hailuo 系列",
        "default_model": "minimax-text-01",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_BASE_URL",
        "docs_url": "https://platform.minimaxi.com/",
        "region": "china",
    },
}

PLACEHOLDER_PROVIDER_ALIASES: dict[str, str] = {
    "aliyun_bailian": "dashscope",
    "bailian": "dashscope",
    "ernie": "baidu_qianfan",
    "wenxin": "baidu_qianfan",
    "hunyuan": "tencent_hunyuan",
    "doubao": "volcengine_doubao",
    "volcengine_ark": "volcengine_doubao",
    "glm": "zhipu_glm",
    "zhipu": "zhipu_glm",
    "kimi": "moonshot_kimi",
    "moonshot": "moonshot_kimi",
}

PLACEHOLDER_PROVIDERS = frozenset(
    set(PLACEHOLDER_PROVIDER_INFO) | set(PLACEHOLDER_PROVIDER_ALIASES)
)


def canonical_provider_name(provider: str) -> str:
    """
    功能：把别名规范化为占位 Provider 主名称。
    参数：
        provider：配置中的 provider 名称。
    返回：规范 provider 名称。
    副作用：无。
    异常：无。
    设计说明：允许用户按模型族或平台俗称配置，如 kimi、doubao、ernie。
    """

    return PLACEHOLDER_PROVIDER_ALIASES.get(provider, provider)


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
        self.canonical_provider = canonical_provider_name(self.provider)
        self.info = PLACEHOLDER_PROVIDER_INFO.get(self.canonical_provider, {})
        self.model = str(self.config.get("model", self.info.get("default_model", "")))

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
            "canonical_provider": self.canonical_provider,
            "display_name": self.info.get("display_name", self.provider),
            "model_family": self.info.get("model_family", ""),
            "model": self.model,
            "available": False,
            "offline": False,
            "status": "placeholder",
            "region": self.info.get("region", "unknown"),
            "docs_url": self.info.get("docs_url", ""),
            "api_key_env": self.config.get(
                "api_key_env",
                self.info.get("api_key_env", f"{self.provider.upper()}_API_KEY"),
            ),
            "base_url_env": self.config.get(
                "base_url_env",
                self.info.get("base_url_env", f"{self.provider.upper()}_BASE_URL"),
            ),
            "message": (
                "已注册占位；需要按 docs/0705_05_扩展开发指南.md "
                "完成真实适配和 smoke test。"
            ),
        }
