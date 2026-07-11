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
        self.api_key_required = bool(self.config.get("api_key_required", True))
        self.max_retries = int(self.config.get("max_retries", 1))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 30))
        self.structured_output_mode = str(self.config.get("structured_output_mode", "json_object"))

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
        if not base_url:
            raise ProviderError(f"{self.provider} 缺少环境变量 {self.base_url_env}")
        if self.api_key_required and not api_key:
            raise ProviderError(f"{self.provider} 缺少环境变量 {self.api_key_env}")

        def _call() -> ActionDecision:
            content = self._post(base_url, api_key or "", request)
            try:
                payload = _load_json_object(content)
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

        api_key_present = bool(os.environ.get(self.api_key_env))
        base_url_present = bool(os.environ.get(self.base_url_env))
        available = bool(base_url_present and (api_key_present or not self.api_key_required))
        return {
            "provider": self.provider,
            "model": self.model,
            "available": available,
            "offline": False,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "api_key_required": self.api_key_required,
            "message": (
                "base URL 已配置；本地服务可不要求 API key。"
                if available and not self.api_key_required
                else "仅在提供密钥和 base URL 后才会执行真实调用。"
            ),
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
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        response_format = _response_format(self.structured_output_mode, request)
        if response_format is not None:
            payload["response_format"] = response_format
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        http_request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = f"{exc}"
            if error_body.strip():
                detail = f"{detail}；响应：{error_body[:500]}"
            raise ProviderError(f"{self.provider} HTTP 请求失败：{detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.provider} HTTP 请求失败：{exc}") from exc
        elapsed = (time.perf_counter() - started) * 1000
        try:
            parsed = json.loads(body)
            message = parsed["choices"][0]["message"]
            content = message.get("content") or ""
            if not content and message.get("reasoning_content"):
                raise ProviderError(
                    f"{self.provider} 只返回 reasoning_content，未返回最终 content；"
                    "请在 LM Studio 关闭 Think/Reasoning，或提高 max_output_tokens 后重试。"
                )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.provider} 响应结构无法解析，耗时 {elapsed:.1f}ms") from exc
        return str(content)


def _response_format(
    mode: str,
    request: LLMRequest | None = None,
) -> dict[str, object] | None:
    """
    功能：根据配置生成兼容接口的结构化输出参数。
    参数：
        mode：json_object、json_schema 或 prompt_only/none。
        request：可选当前请求，用于把 schema 收紧到本次合法动作与 raise 范围。
    返回：response_format 字典或 None。
    副作用：无。
    异常：无。
    设计说明：LM Studio 0.4.x 支持 json_schema；部分在线 API 支持 json_object。
    """

    if mode in {"none", "prompt_only", "disabled"}:
        return None
    if mode == "json_schema":
        legal_types = ["fold", "check", "call", "raise"]
        raise_rule = None
        if request is not None:
            legal_types = [
                action.action_type for action in request.observation.legal_actions.actions
            ] or legal_types
            raise_rule = request.observation.legal_actions.rule_for("raise")
        amount_schema: dict[str, object] = {"type": "null"}
        if "raise" in legal_types:
            integer_schema: dict[str, object] = {"type": "integer"}
            raise_sizing = request.metadata.get("raise_sizing", {}) if request else {}
            allowed_amounts = (
                raise_sizing.get("allowed_amounts")
                if isinstance(raise_sizing, dict)
                else None
            )
            if isinstance(allowed_amounts, list) and allowed_amounts:
                integer_schema["enum"] = [int(amount) for amount in allowed_amounts]
            else:
                if raise_rule is not None and raise_rule.min_amount is not None:
                    integer_schema["minimum"] = raise_rule.min_amount
                if raise_rule is not None and raise_rule.max_amount is not None:
                    integer_schema["maximum"] = raise_rule.max_amount
            amount_schema = {
                "anyOf": [
                    integer_schema,
                    {"type": "null"},
                ]
            }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "action_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": legal_types,
                        },
                        "amount": amount_schema,
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reason_summary": {
                            "type": "string",
                            "maxLength": 300,
                        },
                    },
                    "required": [
                        "action_type",
                        "amount",
                        "confidence",
                        "reason_summary",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    return {"type": "json_object"}


def _load_json_object(content: str) -> dict[str, object]:
    """
    功能：从模型文本中解析第一个 JSON 对象。
    参数：
        content：模型 message content。
    返回：JSON 对象字典。
    副作用：无。
    异常：无法解析时抛出 JSONDecodeError。
    设计说明：本地小模型偶尔会输出 fenced JSON 或前后解释，保守提取首个完整对象。
    """

    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _find_first_json_object(stripped)
        parsed = json.loads(candidate)
    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("top-level JSON is not an object", stripped, 0)


def _find_first_json_object(text: str) -> str:
    """
    功能：在文本中定位第一个括号平衡的 JSON 对象片段。
    参数：
        text：可能包含说明或 Markdown fence 的模型输出。
    返回：JSON 对象字符串。
    副作用：无。
    异常：找不到完整对象时抛出 JSONDecodeError。
    设计说明：只做括号和字符串状态机，不用正则贪婪截断嵌套对象。
    """

    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("JSON object start not found", text, 0)
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise json.JSONDecodeError("JSON object end not found", text, start)
