"""
模块说明：本模块提供 OpenAI-compatible Chat Completions Provider 骨架。
核心职责：通过环境变量读取 base URL 与 API Key，并返回统一结构化动作。
输入与输出：输入 LLMRequest，输出 ActionDecision。
依赖边界：只使用标准库 urllib，不绑定 openai 官方 SDK。
不负责：不验证所有兼容厂商差异，不在无密钥时执行真实请求。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import TypeVar

from agentmemeval.core.domain import ActionDecision
from agentmemeval.core.errors import ProviderError
from agentmemeval.environment.action_guard import coerce_decision
from agentmemeval.llm.retry import retry_call
from agentmemeval.llm.schemas import LLMRequest

T = TypeVar("T")


class OpenAICompatibleClient:
    """
    功能：调用兼容 OpenAI Chat Completions 的接口。
    参数：
        config：Provider 配置。
    返回：Provider 实例。
    副作用：真实调用时访问网络。
    异常：缺少环境变量或请求失败时抛出 ProviderError。
    设计说明：使用标准 HTTP 接口，后续可替换为官方 SDK 实现。
    """

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """
        功能：初始化兼容 Provider。
        参数：
            config：包含 model、api_key_env、base_url_env 等字段。
        返回：无。
        副作用：读取配置但不读取密钥值。
        异常：无。
        设计说明：doctor 能报告缺失项，避免导入时就失败。
        """

        self.config = config or {}
        self.provider = str(self.config.get("provider", "openai_compatible"))
        self.model = str(self.config.get("model", ""))
        self.api_key_env = str(self.config.get("api_key_env", "OPENAI_API_KEY"))
        self.base_url_env = str(self.config.get("base_url_env", "OPENAI_BASE_URL"))
        self.max_retries = int(self.config.get("max_retries", 1))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 30))

    def generate_structured(self, request: LLMRequest, schema: type[T]) -> T:
        """
        功能：向兼容接口请求结构化 JSON。
        参数：
            request：LLM 请求。
            schema：目标结构，目前支持 ActionDecision。
        返回：schema 实例。
        副作用：访问网络。
        异常：缺少密钥、响应非 JSON 或 schema 不支持时抛出 ProviderError。
        设计说明：默认要求模型输出 JSON，返回后仍交给 ActionGuard 校验。
        """

        if schema is not ActionDecision:
            raise ProviderError(f"{self.provider} 暂不支持 schema：{schema!r}")
        api_key = os.environ.get(self.api_key_env)
        base_url = os.environ.get(self.base_url_env)
        if not api_key or not base_url:
            raise ProviderError(
                f"{self.provider} 缺少环境变量 {self.api_key_env} 或 {self.base_url_env}"
            )

        def _call() -> ActionDecision:
            content = self._post(base_url, api_key, request)
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"模型响应不是 JSON：{content[:200]}") from exc
            return coerce_decision(payload)

        result, _ = retry_call(_call, self.max_retries)
        return result  # type: ignore[return-value]

    def healthcheck(self) -> dict[str, object]:
        """
        功能：报告 Provider 配置是否具备真实调用条件。
        参数：无。
        返回：健康检查字典。
        副作用：不访问网络。
        异常：无。
        设计说明：无密钥环境下仍能说明接入缺口，而不是伪造可用。
        """

        available = bool(os.environ.get(self.api_key_env) and os.environ.get(self.base_url_env))
        return {
            "provider": self.provider,
            "model": self.model,
            "available": available,
            "offline": False,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "message": "仅在提供密钥和 base URL 后才会执行真实 smoke test。",
        }

    def _post(self, base_url: str, api_key: str, request: LLMRequest) -> str:
        """
        功能：执行一次 Chat Completions HTTP 请求。
        参数：
            base_url：兼容接口根地址。
            api_key：密钥。
            request：LLM 请求。
        返回：模型 message content。
        副作用：访问网络。
        异常：HTTP 或响应结构错误时抛出 ProviderError。
        设计说明：不保存原始响应，避免默认持久化敏感内容。
        """

        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": float(self.config.get("temperature", 0.2)),
            "max_tokens": int(self.config.get("max_output_tokens", 256)),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.provider} HTTP 请求失败：{exc}") from exc
        elapsed = (time.perf_counter() - started) * 1000
        try:
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.provider} 响应结构无法解析，耗时 {elapsed:.1f}ms") from exc
        return str(content)
