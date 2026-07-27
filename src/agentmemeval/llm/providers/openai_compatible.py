"""
模块说明：本模块提供 OpenAI-compatible Chat Completions Provider 骨架。
核心职责：通过环境变量读取 base URL 与 API Key，并返回统一结构化动作。
输入与输出：输入 LLMRequest，输出 ActionDecision。
依赖边界：只使用标准库 urllib，不绑定 openai 官方 SDK。
不负责：不验证所有兼容厂商差异，不在无密钥时执行真实请求。
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TypeVar

from agentmemeval.core.domain import ActionDecision
from agentmemeval.core.errors import ProviderError
from agentmemeval.core.model_identity import canonical_model_family
from agentmemeval.environment.action_guard import coerce_decision
from agentmemeval.llm.retry import retry_call
from agentmemeval.llm.schemas import LLMRequest

T = TypeVar("T")

EXPERIENCE_PROVIDER_DIAGNOSTICS_VERSION = "openai_compatible_http_calls_v1"
RESPONSE_MODEL_IDENTITY_VERSION = "openai_compatible_response_model_identity_v1"
_RESPONSE_MODEL_IDENTITY_POLICIES = {
    "off",
    "strict_exact",
    "if_present_allowlisted_alias",
}
_SAFE_MODEL_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}\Z")
_SENSITIVE_MODEL_TOKEN = re.compile(
    r"(?:^|[._:/+\-])(?:bearer|api[_-]?key|secret|access[_-]?token|sk-)",
    re.IGNORECASE,
)


def _is_experience_revision_schema(value: str) -> bool:
    """Recognize versioned experience schemas without falling into the action schema."""

    return value in {"experience_revision_v1", "experience_revision_v2"}


def _is_semantic_review_schema(value: str) -> bool:
    """Recognize independent semantic-review schemas without Expr side effects."""

    return value in {"strategy_semantic_review_v1", "strategy_semantic_review_v2"}


def _is_diagnostic_dict_schema(value: str) -> bool:
    """Return whether safe diagnostics must be attached to a dict response."""

    return _is_experience_revision_schema(value) or _is_semantic_review_schema(value)


@dataclass(frozen=True)
class _CompletionResult:
    """OpenAI-compatible 单次完成结果及停止原因。"""

    content: str
    finish_reason: str
    reasoning_present: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_model: str | None = None
    response_model_report_status: str = "unreported"


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
        self.experience_structured_output_mode = str(
            self.config.get(
                "experience_structured_output_mode",
                self.structured_output_mode,
            )
        )
        self.thinking_mode = str(self.config.get("thinking_mode", "")).strip()
        self.reasoning_effort = str(
            self.config.get("reasoning_effort", "")
        ).strip()
        self.response_model_identity_policy = str(
            self.config.get("response_model_identity_policy", "off")
        ).strip()
        if self.response_model_identity_policy not in _RESPONSE_MODEL_IDENTITY_POLICIES:
            raise ProviderError(
                "response_model_identity_policy must be one of: "
                "off, strict_exact, if_present_allowlisted_alias",
                retryable=False,
            )
        self._safe_requested_model = _safe_configured_model(self.model)
        raw_aliases = self.config.get("response_model_aliases", [])
        self.response_model_aliases = _normalize_response_model_aliases(
            raw_aliases,
            requested_model=self._safe_requested_model,
            policy=self.response_model_identity_policy,
        )
        self._last_call_diagnostics: dict[str, object] = {}
        self._identity_status_counts: dict[str, int] = {}
        self._observed_response_models: set[str] = set()

    @property
    def last_call_diagnostics(self) -> dict[str, object]:
        """Return the latest content-free call diagnostics for any schema path."""

        return copy.deepcopy(self._last_call_diagnostics)

    def response_identity_summary(self) -> dict[str, object]:
        """Summarize service-reported model identities without claiming attestation."""

        return {
            "schema_version": RESPONSE_MODEL_IDENTITY_VERSION,
            "identity_source": "chat_completions_top_level_model_self_report",
            "cryptographic_attestation": False,
            "policy": self.response_model_identity_policy,
            "requested_model": self._safe_requested_model,
            "allowlisted_aliases": list(self.response_model_aliases),
            "completed_response_count": sum(self._identity_status_counts.values()),
            "status_counts": dict(sorted(self._identity_status_counts.items())),
            "observed_response_models": sorted(self._observed_response_models),
        }

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

        if schema not in {ActionDecision, dict}:
            raise ProviderError(f"{self.provider} 暂不支持 schema：{schema!r}")
        api_key = os.environ.get(self.api_key_env)
        base_url = os.environ.get(self.base_url_env)
        if not base_url:
            raise ProviderError(f"{self.provider} 缺少环境变量 {self.base_url_env}")
        if self.api_key_required and not api_key:
            raise ProviderError(f"{self.provider} 缺少环境变量 {self.api_key_env}")

        metadata = getattr(request, "metadata", {})
        response_schema = str(metadata.get("response_schema", ""))
        attach_dict_diagnostics = schema is dict and _is_diagnostic_dict_schema(
            response_schema
        )
        # Every HTTP schema path is observed.  Expr token-budget enforcement is
        # still scoped separately inside _post_observed.
        call_diagnostics: list[dict[str, object]] = []
        outer_attempt = 0

        def _call() -> ActionDecision | dict[str, object]:
            nonlocal outer_attempt
            outer_attempt += 1
            max_tokens = (
                int(
                    self.config.get(
                        "experience_max_output_tokens",
                        3072,
                    )
                )
                if schema is dict and _is_experience_revision_schema(response_schema)
                else None
            )
            completion = self._post_observed(
                base_url,
                api_key or "",
                request,
                stage="primary",
                outer_attempt=outer_attempt,
                call_diagnostics=call_diagnostics,
                max_tokens=max_tokens,
            )
            try:
                payload = _load_json_object(completion.content)
                _mark_latest_json_status(call_diagnostics, "valid")
            except json.JSONDecodeError as first_error:
                _mark_latest_json_status(call_diagnostics, "invalid")
                payload = self._repair_or_recover_json(
                    base_url,
                    api_key or "",
                    request,
                    completion,
                    first_error,
                    outer_attempt=outer_attempt,
                    call_diagnostics=call_diagnostics,
                )
            if attach_dict_diagnostics:
                payload["_provider_diagnostics"] = _build_provider_diagnostics(
                    call_diagnostics,
                    retry_count=max(0, outer_attempt - 1),
                )
            if schema is ActionDecision:
                _validate_live_action_payload(payload)
                decision = coerce_decision(payload)
                decision.raw_response["_provider_diagnostics"] = (
                    _build_provider_diagnostics(
                        call_diagnostics,
                        retry_count=max(0, outer_attempt - 1),
                    )
                )
                return decision
            return payload

        try:
            result, retry_count = retry_call(_call, self.max_retries)
        except Exception as exc:  # noqa: BLE001
            diagnostics = _build_provider_diagnostics(
                call_diagnostics,
                retry_count=max(0, outer_attempt - 1),
            )
            self._last_call_diagnostics = copy.deepcopy(diagnostics)
            raise ProviderError(
                str(exc),
                diagnostics=diagnostics,
                retryable=bool(getattr(exc, "retryable", True)),
            ) from exc
        diagnostics = _build_provider_diagnostics(
            call_diagnostics,
            retry_count=retry_count,
        )
        self._last_call_diagnostics = copy.deepcopy(diagnostics)
        if isinstance(result, dict) and attach_dict_diagnostics:
            result["_provider_diagnostics"] = copy.deepcopy(diagnostics)
        elif isinstance(result, ActionDecision):
            result.raw_response["_provider_diagnostics"] = copy.deepcopy(diagnostics)
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
            "structured_output_mode": self.structured_output_mode,
            "experience_structured_output_mode": (
                self.experience_structured_output_mode
            ),
            "thinking_mode": self.thinking_mode or None,
            "reasoning_effort": self.reasoning_effort or None,
            "response_model_identity": self.response_identity_summary(),
            "message": (
                "base URL 已配置；本地服务可不要求 API key。"
                if available and not self.api_key_required
                else "仅在提供密钥和 base URL 后才会执行真实调用。"
            ),
        }

    def _repair_or_recover_json(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequest,
        completion: _CompletionResult,
        first_error: json.JSONDecodeError,
        *,
        outer_attempt: int,
        call_diagnostics: list[dict[str, object]] | None,
    ) -> dict[str, object]:
        """用紧凑请求修复截断 JSON，失败后保守恢复完整动作字段。"""

        metadata = getattr(request, "metadata", {})
        if _is_experience_revision_schema(str(metadata.get("response_schema", ""))):
            max_chars = int(metadata.get("experience_max_chars", 2400))
            aux_chars = int(metadata.get("experience_aux_max_chars", 240))
            schema_version = str(metadata.get("response_schema", ""))
            repair_instruction = (
                "重新生成结构化经验 patch：operation 只能是 keep 或 patch；keep 时 "
                "patches=[]；patch 中必须完整保留所有 typed fields；fact_assessments 必须"
                "逐条覆盖给定 fact IDs。"
                if schema_version == "experience_revision_v2"
                else (
                    f"重新生成紧凑经验修订：new_md 最多 {max_chars} 字符，"
                    "keep=true 时 new_md 必须为空或逐字等于旧文档；keep=false 时"
                    "必须返回从 '# 我的经验' 开始的完整五章节 Markdown 全文，"
                    "不得返回 patch。fact ID 数组只能引用本请求提供的 fact_ids。"
                )
            )
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是经验 JSON 压缩修复器。只输出符合 response schema 的完整 "
                        "JSON，不输出解释或思维过程。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        repair_instruction
                        + f"calibration_note 与 self_check 各最多 {aux_chars} 字符。"
                        "必须闭合 JSON，并保留全部必需字段。\n\n" + request.user_prompt
                    ),
                },
            ]
            repair_error: Exception | None = None
            try:
                repaired = self._post_observed(
                    base_url,
                    api_key,
                    request,
                    stage="json_repair",
                    outer_attempt=outer_attempt,
                    call_diagnostics=call_diagnostics,
                    messages=repair_messages,
                    max_tokens=int(
                        self.config.get("experience_repair_max_output_tokens", 2048)
                    ),
                )
                repaired_payload = _load_json_object(repaired.content)
                _mark_latest_json_status(call_diagnostics, "valid")
                repaired_payload["_provider_repair_used"] = True
                return repaired_payload
            except ProviderError as exc:
                if not exc.retryable:
                    raise
                repair_error = exc
            except json.JSONDecodeError as exc:
                _mark_latest_json_status(call_diagnostics, "invalid")
                repair_error = exc
            raise ProviderError(
                "经验修订响应不是完整 JSON；"
                f"finish_reason={completion.finish_reason or 'unknown'}；"
                f"紧凑重生成失败={repair_error}"
            ) from first_error

        if _is_semantic_review_schema(str(metadata.get("response_schema", ""))):
            # A reviewer response has its own request-bound schema.  Feeding it
            # into the action JSON repair prompt would silently change the task.
            raise ProviderError(
                "semantic_review_response_invalid_json；"
                f"finish_reason={completion.finish_reason or 'unknown'}"
            ) from first_error

        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是 JSON 修复器。只输出一个符合 schema 的 JSON 对象，不要解释。"
                    "reason_summary 最多 40 个汉字，intent 最多 60 个汉字；"
                    "intent 是策略目标或希望传达的形象，不是行动理由；不得输出思维过程。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "修复下面未完成或格式错误的动作 JSON。保留已经明确的动作字段；"
                    "若理由被截断，将其改写为一句极短摘要。\n"
                    + completion.content
                ),
            },
        ]
        repair_error: Exception | None = None
        try:
            repaired = self._post_observed(
                base_url,
                api_key,
                request,
                stage="json_repair",
                outer_attempt=outer_attempt,
                call_diagnostics=call_diagnostics,
                messages=repair_messages,
                max_tokens=int(self.config.get("repair_max_output_tokens", 256)),
            )
            return _load_json_object(repaired.content)
        except (ProviderError, json.JSONDecodeError) as exc:
            repair_error = exc

        recovered = _recover_truncated_action_object(completion.content)
        if recovered is not None:
            return recovered

        reason = completion.finish_reason or "unknown"
        raise ProviderError(
            "模型响应不是完整 JSON；"
            f"finish_reason={reason}；原始响应={completion.content[:200]}；"
            f"紧凑修复失败={repair_error}"
        ) from first_error

    def _post_observed(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequest,
        *,
        stage: str,
        outer_attempt: int,
        call_diagnostics: list[dict[str, object]] | None,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> _CompletionResult:
        """Execute one HTTP call and record only safe call-level metadata."""

        post_kwargs: dict[str, object] = {}
        if messages is not None:
            post_kwargs["messages"] = messages
        if max_tokens is not None:
            post_kwargs["max_tokens"] = max_tokens
        try:
            completion = self._post(
                base_url,
                api_key,
                request,
                **post_kwargs,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            if call_diagnostics is not None:
                call_diagnostics.append(
                    _failed_http_call_diagnostic(
                        call_index=len(call_diagnostics) + 1,
                        outer_attempt=outer_attempt,
                        stage=stage,
                        error=exc,
                        requested_model=self._safe_requested_model,
                        identity_policy=self.response_model_identity_policy,
                    )
                )
            raise
        if call_diagnostics is not None:
            call_diagnostic = _completed_http_call_diagnostic(
                call_index=len(call_diagnostics) + 1,
                outer_attempt=outer_attempt,
                stage=stage,
                completion=completion,
                requested_model=self._safe_requested_model,
                identity_policy=self.response_model_identity_policy,
            )
            call_diagnostics.append(call_diagnostic)
            self._enforce_response_model_identity(
                completion,
                call_diagnostic,
                call_diagnostics=call_diagnostics,
            )
            request_metadata = getattr(request, "metadata", {})
            if not isinstance(request_metadata, dict):
                request_metadata = {}
            if _is_experience_revision_schema(
                str(request_metadata.get("response_schema", ""))
            ):
                _enforce_experience_prompt_token_budget(
                    request,
                    completion,
                    call_diagnostic,
                )
        return completion

    def _enforce_response_model_identity(
        self,
        completion: _CompletionResult,
        call_diagnostic: dict[str, object],
        *,
        call_diagnostics: list[dict[str, object]],
    ) -> str:
        """Evaluate one service self-report under the configured fail-closed policy."""

        status, failure_code = _response_model_identity_status(
            requested_model=self._safe_requested_model,
            response_model=completion.response_model,
            report_status=completion.response_model_report_status,
            policy=self.response_model_identity_policy,
            aliases=self.response_model_aliases,
        )
        call_diagnostic["response_model_identity_status"] = status
        call_diagnostic["response_model_identity_verified"] = status in {
            "verified_exact",
            "verified_allowlisted_alias",
        }
        call_diagnostic["response_model_exact_match"] = status == "verified_exact"
        self._identity_status_counts[status] = (
            self._identity_status_counts.get(status, 0) + 1
        )
        if completion.response_model is not None:
            self._observed_response_models.add(completion.response_model)
        if failure_code is None:
            return status
        call_diagnostic["failure_code"] = failure_code
        diagnostics = _build_provider_diagnostics(
            call_diagnostics,
            retry_count=max(0, max(
                (int(item.get("outer_attempt", 0)) for item in call_diagnostics),
                default=1,
            ) - 1),
        )
        self._last_call_diagnostics = copy.deepcopy(diagnostics)
        raise ProviderError(
            f"response_model_identity_rejected:{status}",
            diagnostics=diagnostics,
            retryable=False,
        )

    def _post(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequest,
        *,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> _CompletionResult:
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
        response_schema = str(request.metadata.get("response_schema", ""))
        is_experience_revision = _is_experience_revision_schema(response_schema)
        default_temperature = self.config.get("temperature", 0.2)
        request_temperature = (
            self.config.get("experience_temperature", default_temperature)
            if is_experience_revision
            else default_temperature
        )
        payload = {
            "model": self.model,
            "temperature": float(request_temperature),
            "max_tokens": max_tokens or int(self.config.get("max_output_tokens", 256)),
            "messages": messages
            or [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        response_mode = (
            self.experience_structured_output_mode
            if is_experience_revision
            else self.structured_output_mode
        )
        response_format = _response_format(response_mode, request)
        if response_format is not None:
            payload["response_format"] = response_format
        if self.thinking_mode:
            payload["thinking"] = {"type": self.thinking_mode}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
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
            response_model, response_model_report_status = (
                _extract_safe_response_model(parsed)
            )
            choice = parsed["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            finish_reason = str(choice.get("finish_reason") or "")
            usage = parsed.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            prompt_tokens = _optional_int(usage.get("prompt_tokens"))
            completion_tokens = _optional_int(usage.get("completion_tokens"))
            if not str(content).strip():
                reasoning_present = bool(message.get("reasoning_content"))
                raise ProviderError(
                    f"{self.provider} empty_content=true；"
                    f"reasoning_present={str(reasoning_present).lower()}；"
                    f"finish_reason={finish_reason or 'unknown'}；"
                    "未返回最终 content；若存在思维内容也拒绝把它当作结构化 JSON。"
                )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.provider} 响应结构无法解析，耗时 {elapsed:.1f}ms") from exc
        return _CompletionResult(
            content=str(content),
            finish_reason=finish_reason,
            reasoning_present=bool(message.get("reasoning_content")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_model=response_model,
            response_model_report_status=response_model_report_status,
        )


def _completed_http_call_diagnostic(
    *,
    call_index: int,
    outer_attempt: int,
    stage: str,
    completion: _CompletionResult,
    requested_model: str | None,
    identity_policy: str,
) -> dict[str, object]:
    """Describe a completed HTTP call without retaining model content."""

    prompt_tokens = completion.prompt_tokens
    completion_tokens = completion.completion_tokens
    return {
        "call_index": call_index,
        "outer_attempt": outer_attempt,
        "stage": stage,
        "status": "completed",
        "finish_reason": _safe_finish_reason(completion.finish_reason),
        "reasoning_present": bool(completion.reasoning_present),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_available": type(prompt_tokens) is int and prompt_tokens > 0,
        "json_status": "not_checked",
        "budget_status": "not_checked",
        "failure_code": None,
        "response_model_identity_schema_version": RESPONSE_MODEL_IDENTITY_VERSION,
        "response_model_identity_source": (
            "chat_completions_top_level_model_self_report"
        ),
        "response_model_identity_policy": identity_policy,
        "requested_model": requested_model,
        "response_model": completion.response_model,
        "response_model_report_status": completion.response_model_report_status,
        "response_model_identity_status": "not_checked",
        "response_model_identity_verified": False,
        "response_model_exact_match": False,
    }


def _failed_http_call_diagnostic(
    *,
    call_index: int,
    outer_attempt: int,
    stage: str,
    error: Exception,
    requested_model: str | None,
    identity_policy: str,
) -> dict[str, object]:
    """Mark an HTTP failure explicitly while omitting its error text and body."""

    return {
        "call_index": call_index,
        "outer_attempt": outer_attempt,
        "stage": stage,
        "status": "failed",
        "finish_reason": None,
        "reasoning_present": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "usage_available": False,
        "json_status": "not_available",
        "budget_status": "not_available",
        "failure_code": (
            "provider_error" if isinstance(error, ProviderError) else "unexpected_error"
        ),
        "response_model_identity_schema_version": RESPONSE_MODEL_IDENTITY_VERSION,
        "response_model_identity_source": (
            "chat_completions_top_level_model_self_report"
        ),
        "response_model_identity_policy": identity_policy,
        "requested_model": requested_model,
        "response_model": None,
        "response_model_report_status": "no_completed_response",
        "response_model_identity_status": "not_available",
        "response_model_identity_verified": False,
        "response_model_exact_match": False,
    }


def _mark_latest_json_status(
    call_diagnostics: list[dict[str, object]] | None,
    status: str,
) -> None:
    """Annotate the latest completed call with a bounded JSON parse outcome."""

    if call_diagnostics and call_diagnostics[-1].get("status") == "completed":
        call_diagnostics[-1]["json_status"] = status


def _enforce_experience_prompt_token_budget(
    request: LLMRequest,
    completion: _CompletionResult,
    call_diagnostic: dict[str, object],
) -> None:
    """Apply the selected budget policy to one completed experience HTTP call."""

    raw_budget = request.metadata.get("experience_prompt_token_budget")
    if type(raw_budget) is not int or raw_budget < 1:
        call_diagnostic["budget_status"] = "invalid_budget"
        raise ProviderError(
            "experience_prompt_token_budget must be a positive integer",
            retryable=False,
        )
    prompt_tokens = completion.prompt_tokens
    if type(prompt_tokens) is not int or prompt_tokens < 1:
        call_diagnostic["budget_status"] = "usage_missing"
        raise ProviderError(
            "experience_prompt_token_usage_missing",
            retryable=False,
        )
    if prompt_tokens > raw_budget:
        if (
            request.metadata.get("experience_prompt_budget_policy")
            == "advisory_after_completed_response"
        ):
            # The provider has already accepted the context and returned a
            # complete response.  Main-style paper Expr may retain that result;
            # compaction and the audit still surface the excess for diagnosis.
            call_diagnostic["budget_status"] = "exceeded_advisory"
            return
        call_diagnostic["budget_status"] = "exceeded"
        raise ProviderError(
            "experience_prompt_token_budget_exceeded: "
            f"{prompt_tokens} > {raw_budget}",
            retryable=False,
        )
    call_diagnostic["budget_status"] = "within_budget"


def _build_provider_diagnostics(
    call_diagnostics: list[dict[str, object]],
    *,
    retry_count: int,
) -> dict[str, object]:
    """Build the safe internal diagnostics attached to an experience payload/error."""

    calls = [dict(item) for item in call_diagnostics]
    completed = [item for item in calls if item.get("status") == "completed"]
    last_completed = completed[-1] if completed else {}
    last_call = calls[-1] if calls else {}
    observed_prompt_tokens = [
        int(item["prompt_tokens"])
        for item in completed
        if type(item.get("prompt_tokens")) is int
    ]
    identity_status_counts: dict[str, int] = {}
    observed_response_models: set[str] = set()
    for item in completed:
        status = item.get("response_model_identity_status")
        if isinstance(status, str):
            identity_status_counts[status] = identity_status_counts.get(status, 0) + 1
        response_model = item.get("response_model")
        if isinstance(response_model, str) and _SAFE_MODEL_TOKEN.fullmatch(response_model):
            observed_response_models.add(response_model)
    return {
        "schema_version": EXPERIENCE_PROVIDER_DIAGNOSTICS_VERSION,
        "token_accounting": "per_http_call_v1",
        "retry_count": max(0, retry_count),
        "outer_attempt_count": max(
            (int(item.get("outer_attempt", 0)) for item in calls),
            default=0,
        ),
        "http_call_count": len(calls),
        "completed_http_call_count": len(completed),
        "failed_http_call_count": sum(item.get("status") == "failed" for item in calls),
        "json_repair_call_count": sum(item.get("stage") == "json_repair" for item in calls),
        # A transport/provider failure has no completed response whose usage can
        # be audited.  Keep it in failed_http_call_count instead of double
        # labelling the same call as both failed and usage-missing.
        "usage_missing_call_count": sum(
            item.get("status") == "completed"
            and item.get("usage_available") is not True
            for item in calls
        ),
        "prompt_tokens_total_observed": sum(observed_prompt_tokens),
        "prompt_tokens_max_observed": max(observed_prompt_tokens, default=0),
        # Legacy summary fields describe the final completed HTTP call only.
        "finish_reason": last_completed.get("finish_reason"),
        "reasoning_present": bool(last_completed.get("reasoning_present", False)),
        "prompt_tokens": last_completed.get("prompt_tokens"),
        "completion_tokens": last_completed.get("completion_tokens"),
        "response_model_identity_schema_version": RESPONSE_MODEL_IDENTITY_VERSION,
        "response_model_identity_source": (
            "chat_completions_top_level_model_self_report"
        ),
        "response_model_identity_cryptographic_attestation": False,
        "response_model_identity_policy": last_call.get(
            "response_model_identity_policy"
        ),
        "requested_model": last_call.get("requested_model"),
        "response_model": last_call.get("response_model"),
        "response_model_report_status": last_call.get(
            "response_model_report_status"
        ),
        "response_model_identity_status": last_call.get(
            "response_model_identity_status"
        ),
        "response_model_identity_verified": bool(
            last_call.get("response_model_identity_verified", False)
        ),
        "response_model_exact_match": bool(
            last_call.get("response_model_exact_match", False)
        ),
        "response_model_identity_status_counts": dict(
            sorted(identity_status_counts.items())
        ),
        "observed_response_models": sorted(observed_response_models),
        "calls": calls,
    }


def _safe_finish_reason(value: object) -> str:
    """Keep only a bounded provider finish-reason token, never arbitrary content."""

    text = str(value or "unknown")
    return (
        text
        if text
        in {
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
            "unknown",
        }
        else "other"
    )


def _optional_int(value: object) -> int | None:
    """Return a safe token count without making provider usage metadata mandatory."""

    return value if type(value) is int and value >= 0 else None


def _safe_configured_model(value: object) -> str | None:
    """Return a persistence-safe configured model token, or None for legacy values."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return (
        text
        if _SAFE_MODEL_TOKEN.fullmatch(text)
        and _SENSITIVE_MODEL_TOKEN.search(text) is None
        else None
    )


def _normalize_response_model_aliases(
    value: object,
    *,
    requested_model: str | None,
    policy: str,
) -> tuple[str, ...]:
    """Validate an exact, same-family alias allowlist for identity enforcement."""

    if policy != "off" and requested_model is None:
        raise ProviderError(
            "response model identity enforcement requires a safe non-empty model",
            retryable=False,
        )
    if value is None:
        raw_aliases: list[object] = []
    elif isinstance(value, (list, tuple)):
        raw_aliases = list(value)
    elif policy == "off":
        raw_aliases = []
    else:
        raise ProviderError(
            "response_model_aliases must be a list of exact model tokens",
            retryable=False,
        )
    aliases: list[str] = []
    for raw_alias in raw_aliases:
        alias = _safe_configured_model(raw_alias)
        if alias is None:
            if policy == "off":
                continue
            raise ProviderError(
                "response_model_aliases contains an unsafe model token",
                retryable=False,
            )
        if alias not in aliases:
            aliases.append(alias)
    if requested_model is not None and requested_model not in aliases:
        aliases.insert(0, requested_model)
    if policy == "strict_exact" and requested_model is not None:
        return (requested_model,)
    if policy == "if_present_allowlisted_alias" and requested_model is not None:
        requested_family = canonical_model_family(requested_model)
        if any(canonical_model_family(alias) != requested_family for alias in aliases):
            raise ProviderError(
                "response_model_aliases must remain within the requested model family",
                retryable=False,
            )
    return tuple(aliases)


def _extract_safe_response_model(parsed: object) -> tuple[str | None, str]:
    """Read Chat Completions top-level model without retaining arbitrary text."""

    if not isinstance(parsed, dict) or "model" not in parsed:
        return None, "unreported"
    response_model = _safe_configured_model(parsed.get("model"))
    if response_model is None:
        return None, "invalid_reported_value"
    return response_model, "reported_valid"


def _response_model_identity_status(
    *,
    requested_model: str | None,
    response_model: str | None,
    report_status: str,
    policy: str,
    aliases: tuple[str, ...],
) -> tuple[str, str | None]:
    """Return a bounded identity outcome and optional fail-closed code."""

    effective_report_status = (
        "reported_valid" if response_model is not None else report_status
    )
    if effective_report_status == "invalid_reported_value":
        if policy == "off":
            return "reported_invalid_unverified", None
        return "rejected_invalid_reported_value", "response_model_invalid_report"
    if response_model is None:
        if policy == "strict_exact":
            return "rejected_unreported", "response_model_unreported"
        return "unreported", None
    if policy == "off":
        return "reported_unverified", None
    if requested_model is None:
        return "rejected_missing_requested_model", "requested_model_invalid"
    if policy == "strict_exact":
        if response_model == requested_model:
            return "verified_exact", None
        return "rejected_exact_mismatch", "response_model_exact_mismatch"

    requested_family = canonical_model_family(requested_model)
    response_family = canonical_model_family(response_model)
    if response_family != requested_family:
        return "rejected_cross_family", "response_model_cross_family"
    if response_model not in aliases:
        return "rejected_not_allowlisted", "response_model_not_allowlisted"
    if response_model == requested_model:
        return "verified_exact", None
    return "verified_allowlisted_alias", None


def _experience_patch_response_format(request: LLMRequest) -> dict[str, object]:
    """Build the strict v2 patch schema; Markdown never crosses this output boundary."""

    metadata = request.metadata
    aux_chars = int(metadata.get("experience_aux_max_chars", 240))
    trajectory_ids = [str(value) for value in metadata.get("trajectory_ids", [])]
    fact_ids = [str(value) for value in metadata.get("fact_ids", [])]

    def id_array(
        values: list[str],
        *,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> dict[str, object]:
        item_schema: dict[str, object] = {"type": "string"}
        if values:
            item_schema["enum"] = values
        return {
            "type": "array",
            "items": item_schema,
            "minItems": minimum,
            "maxItems": len(values) if maximum is None else maximum,
            "uniqueItems": True,
        }

    sections = [
        "起手牌",
        "翻牌后",
        "转牌 / 河牌",
        "对手类型应对",
        "注码与位置",
    ]
    required_patch_fields = [
        "section",
        "op",
        "rule_id",
        "when",
        "guidance",
        "invalidators",
        "basis",
        "status",
        "supporting_trajectory_ids",
        "supporting_fact_ids",
    ]

    def patch_schema(operation: str) -> dict[str, object]:
        is_delete = operation == "delete"
        rule_id = (
            {"type": "string", "minLength": 1, "maxLength": 64}
            if operation in {"revise", "delete"}
            else {"type": "string", "maxLength": 0}
        )
        string_list = {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 120},
            "minItems": 0 if is_delete else 1,
            "maxItems": 0 if is_delete else 4,
        }
        properties: dict[str, object] = {
            "section": {
                "type": "string",
                "enum": sections,
            },
            "op": {"type": "string", "enum": [operation]},
            "rule_id": rule_id,
            "when": string_list,
            "guidance": {
                "type": "string",
                "minLength": 0 if is_delete else 1,
                "maxLength": 0 if is_delete else 240,
            },
            "invalidators": dict(string_list),
            "basis": {
                "type": "string",
                "enum": ["observed_evidence", "model_prior"],
            },
            "status": {"type": "string", "enum": ["tentative", "supported"]},
            "supporting_trajectory_ids": id_array(
                trajectory_ids,
                maximum=0 if is_delete else len(trajectory_ids),
            ),
            "supporting_fact_ids": id_array(
                fact_ids,
                maximum=0 if is_delete else len(fact_ids),
            ),
        }
        return {
            "type": "object",
            "properties": properties,
            "required": required_patch_fields,
            "additionalProperties": False,
        }

    patch_item_schema = {
        "anyOf": [
            patch_schema("add"),
            patch_schema("revise"),
            patch_schema("delete"),
        ]
    }
    fact_id_schema: dict[str, object] = {"type": "string"}
    if fact_ids:
        fact_id_schema["enum"] = fact_ids
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "experience_revision_v2",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["keep", "patch"]},
                    "patches": {
                        "type": "array",
                        "items": patch_item_schema,
                        "maxItems": 8,
                    },
                    "fact_assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fact_id": fact_id_schema,
                                "label": {
                                    "type": "string",
                                    "enum": [
                                        "supporting",
                                        "contradicting",
                                        "noise",
                                        "irrelevant",
                                    ],
                                },
                            },
                            "required": ["fact_id", "label"],
                            "additionalProperties": False,
                        },
                        "minItems": len(fact_ids),
                        "maxItems": len(fact_ids),
                    },
                    "calibration_note": {
                        "type": "string",
                        "maxLength": aux_chars,
                    },
                    "self_check": {"type": "string", "maxLength": aux_chars},
                },
                "required": [
                    "operation",
                    "patches",
                    "fact_assessments",
                    "calibration_note",
                    "self_check",
                ],
                "additionalProperties": False,
            },
        },
    }


def _experience_full_document_response_format(
    request: LLMRequest,
) -> dict[str, object]:
    """Build the strict v1 whole-document schema using fact IDs only."""

    metadata = request.metadata
    max_chars = int(metadata.get("experience_max_chars", 2400))
    aux_chars = int(metadata.get("experience_aux_max_chars", 240))
    fact_ids = [str(value) for value in metadata.get("fact_ids", [])]
    require_fact_classification = bool(
        metadata.get("experience_fact_classification_fields", False)
    )
    fact_id_schema: dict[str, object] = {"type": "string"}
    if fact_ids:
        fact_id_schema["enum"] = fact_ids
    fact_id_array = {
        "type": "array",
        "items": fact_id_schema,
        "minItems": 0,
        "maxItems": len(fact_ids),
        "uniqueItems": True,
    }
    properties: dict[str, object] = {
        "keep": {"type": "boolean"},
        "new_md": {
            "type": "string",
            "minLength": 0,
            "maxLength": max_chars,
        },
        "calibration_note": {
            "type": "string",
            "maxLength": aux_chars,
        },
        "self_check": {
            "type": "string",
            "maxLength": aux_chars,
        },
        "supporting_fact_ids": fact_id_array,
    }
    required = [
        "keep",
        "new_md",
        "calibration_note",
        "self_check",
        "supporting_fact_ids",
    ]
    if require_fact_classification:
        properties.update(
            {
                "contradicting_fact_ids": dict(fact_id_array),
                "noise_fact_ids": dict(fact_id_array),
            }
        )
        required.extend(["contradicting_fact_ids", "noise_fact_ids"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "experience_revision_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _metadata_json_schema_response_format(
    request: LLMRequest,
) -> dict[str, object] | None:
    """Wrap a request-bound semantic-review schema without importing memory code."""

    response_schema = str(request.metadata.get("response_schema", ""))
    if response_schema not in {
        "strategy_semantic_review_v1",
        "strategy_semantic_review_v2",
    }:
        return None
    raw_schema = request.metadata.get("response_json_schema")
    if not isinstance(raw_schema, dict):
        raise ProviderError(
            f"{response_schema} requires object metadata.response_json_schema"
        )
    try:
        schema = json.loads(
            json.dumps(
                raw_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ProviderError("response_json_schema must be strict JSON data") from exc
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ProviderError(
            f"{response_schema} response_json_schema must be a closed object"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_schema,
            "strict": True,
            "schema": schema,
        },
    }


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
    异常：语义审阅请求缺少或携带无效 JSON Schema 时抛出 ProviderError。
    设计说明：LM Studio 0.4.x 支持 json_schema；部分在线 API 支持 json_object。
    """

    if request is not None:
        metadata_format = _metadata_json_schema_response_format(request)
        if metadata_format is not None:
            return metadata_format
    if mode in {"none", "prompt_only", "disabled"}:
        return None
    if mode == "json_schema":
        if (
            request is not None
            and request.metadata.get("response_schema") == "experience_revision_v2"
        ):
            return _experience_patch_response_format(request)
        if (
            request is not None
            and request.metadata.get("response_schema") == "experience_revision_v1"
        ):
            return _experience_full_document_response_format(request)
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
                        "intent": {
                            "type": "string",
                            "maxLength": 60,
                        },
                    },
                    "required": [
                        "action_type",
                        "amount",
                        "confidence",
                        "reason_summary",
                        "intent",
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


def _validate_live_action_payload(payload: dict[str, object]) -> None:
    """Require the paper-semantic explicit intent on every live action response."""

    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise ProviderError("动作响应缺少非空 intent；不得用 reason_summary 代替")
    if len(intent) > 60:
        raise ProviderError(f"动作响应 intent 过长：{len(intent)} > 60")


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


def _recover_truncated_action_object(content: str) -> dict[str, object] | None:
    """仅当三个关键动作字段均完整时，从截断 JSON 中保守恢复动作。"""

    action_match = re.search(
        r'"action_type"\s*:\s*"(fold|check|call|raise)"',
        content,
    )
    amount_match = re.search(r'"amount"\s*:\s*(null|-?\d+)', content)
    confidence_match = re.search(
        r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
        content,
    )
    if not (action_match and amount_match and confidence_match):
        return None

    amount_text = amount_match.group(1)
    confidence = float(confidence_match.group(1))
    if not 0.0 <= confidence <= 1.0:
        return None
    return {
        "action_type": action_match.group(1),
        "amount": None if amount_text == "null" else int(amount_text),
        "confidence": confidence,
        "reason_summary": "模型理由被截断，已恢复完整动作字段并交由规则校验",
        "intent": "",
        "provider_recovered_from_truncation": True,
    }
