from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from agentmemeval.path1_v1.core.domain import ActionDecision
from agentmemeval.path1_v1.core.errors import ProviderError
from agentmemeval.path1_v1.core.model_identity import canonical_model_family
from agentmemeval.path1_v1.environment.action_guard import coerce_decision
from agentmemeval.path1_v1.llm.request_identity import RequestIdentityBinding
from agentmemeval.path1_v1.llm.retry import retry_call
from agentmemeval.path1_v1.llm.schemas import LLMRequest
from agentmemeval.path1_v1.llm.terminal_policy import (
    PATH1_TERMINAL_POLICY,
    Path1ResponseContractError,
    Path1TerminalError,
    active_path1_budget,
    path1_logical_request,
    require_path1_terminal_policy,
    terminal_from_exception,
)

T = TypeVar("T")

EXPERIENCE_PROVIDER_DIAGNOSTICS_VERSION = "openai_compatible_http_calls_v1"
RESPONSE_MODEL_IDENTITY_VERSION = "openai_compatible_response_model_identity_v1"
ACTION_SEMANTIC_REPAIR_SYSTEM_PROMPT = (
    "You repair JSON. Output one object matching the schema and no explanation. "
    "reason_summary is limited to 40 characters. intent must be non-empty and "
    "limited to 40 characters. intent is a strategic objective or projected table "
    "image, not an action justification. Do not output a reasoning trace."
)
ACTION_SEMANTIC_REPAIR_USER_PREFIX = (
    "Repair the incomplete, malformed, or schema-invalid action JSON below. "
    "Preserve action fields that are already explicit. If the summary is truncated, "
    "rewrite it as one very short sentence.\n"
)
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
_DEFAULT_UNSUPPORTED_JSON_SCHEMA_KEYWORDS = frozenset({"uniqueItems"})


class ProviderExchangeAuditError(BaseException):
    """Run-fatal opt-in archival failure, deliberately outside model fallback handlers.

    Normal transport/JSON errors remain ProviderError. Only inability to persist
    required raw evidence uses this exception, so broad revision ``except Exception``
    handlers cannot silently convert missing evidence into a keep operation.
    """


class ProviderExchangeRecorder:
    """One append-only recorder shared by a run's clients and deep-copied agents."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._call_index = 0

    def __deepcopy__(self, memo: dict[int, object]) -> ProviderExchangeRecorder:
        memo[id(self)] = self
        return self

    def next_call_index(self) -> int:
        with self._lock:
            self._call_index += 1
            return self._call_index

    def append(self, record: dict[str, object]) -> None:
        try:
            line = json.dumps(
                {"schema_version": "provider-exchange-v1",
                 "recorded_at_utc": datetime.now(UTC).isoformat(), **record},
                ensure_ascii=False,
            ) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            raise ProviderExchangeAuditError(
                "provider_exchange_audit_write_failed; stopping run to preserve evidence integrity"
            ) from exc


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
    content: str
    finish_reason: str
    reasoning_present: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_model: str | None = None
    response_model_report_status: str = "unreported"
    estimated_prompt_tokens_upper_bound: int | None = None
    reserved_output_tokens: int | None = None
    resolved_context_limit: int | None = None


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, object] | None = None) -> None:

        self.config = config or {}
        self._terminal_policy_version = self.config.get("terminal_policy", "legacy")
        if not isinstance(self._terminal_policy_version, str) or (
            self._terminal_policy_version not in {"legacy", PATH1_TERMINAL_POLICY}
        ):
            raise Path1TerminalError("configuration", "unknown_terminal_policy")
        self.provider = str(self.config.get("provider", "openai_compatible"))
        self.model = str(self.config.get("model", ""))
        self.api_key_env = str(self.config.get("api_key_env", "OPENAI_API_KEY"))
        self.base_url_env = str(self.config.get("base_url_env", "OPENAI_BASE_URL"))
        self.api_key_required = bool(self.config.get("api_key_required", True))
        self.max_retries = int(self.config.get(
            "max_retries", 0 if self._terminal_policy_version == PATH1_TERMINAL_POLICY else 1,
        ))
        if self._terminal_policy_version == PATH1_TERMINAL_POLICY and self.max_retries != 0:
            raise Path1TerminalError("configuration", "transport_retries_must_be_zero")
        self.timeout_seconds = float(self.config.get("timeout_seconds", 30))
        self.structured_output_mode = str(self.config.get("structured_output_mode", "json_object"))
        self.experience_structured_output_mode = str(
            self.config.get(
                "experience_structured_output_mode",
                self.structured_output_mode,
            )
        )
        raw_unsupported_keywords = self.config.get(
            "json_schema_unsupported_keywords",
            sorted(_DEFAULT_UNSUPPORTED_JSON_SCHEMA_KEYWORDS),
        )
        if not isinstance(raw_unsupported_keywords, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_unsupported_keywords
        ):
            raise ProviderError(
                "json_schema_unsupported_keywords must be a list of non-empty strings",
                retryable=False,
            )
        self.json_schema_unsupported_keywords = {item.strip() for item in raw_unsupported_keywords}
        self.thinking_mode = str(self.config.get("thinking_mode", "")).strip()
        self.reasoning_effort = str(self.config.get("reasoning_effort", "")).strip()
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
        self._exchange_recorder: ProviderExchangeRecorder | None = None
        self._exchange_client_role = "primary"
        self._request_identity_binding: RequestIdentityBinding | None = None

    def bind_request_identity(self, binding: RequestIdentityBinding) -> None:
        if (type(binding) is not RequestIdentityBinding
                or self._request_identity_binding is not None):
            raise Path1TerminalError("configuration", "request_identity_rebind_or_wrong_type")
        if self._terminal_policy_version != PATH1_TERMINAL_POLICY:
            raise Path1TerminalError("configuration", "request_identity_requires_strict_policy")
        self._request_identity_binding = binding

    def bind_exchange_recorder(
        self, recorder: ProviderExchangeRecorder, *, client_role: str = "primary"
    ) -> None:
        """Enable raw archival explicitly; auth headers/endpoints are never recorded."""

        if client_role not in {"primary", "experience_reviewer"}:
            raise ValueError("unsupported provider exchange client role")
        self._exchange_recorder = recorder
        self._exchange_client_role = client_role

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

        claim = getattr(request, "metadata", {}).get("terminal_policy")
        if claim is not None and claim != self._terminal_policy_version:
            raise Path1TerminalError("configuration", "request_policy_claim_not_enabled")
        if self._terminal_policy_version == PATH1_TERMINAL_POLICY:
            require_path1_terminal_policy(self)
            return self._generate_path1(request, schema)
        if active_path1_budget() is not None:
            raise Path1TerminalError("configuration", "legacy_client_inside_path1_scope")
        return self._generate_legacy(request, schema)

    @property
    def terminal_policy_version(self) -> str:
        return str(self._terminal_policy_version)

    def _generate_path1(self, request: LLMRequest, schema: type[T]) -> T:
        """One initial request plus one repair shared with the memory layer."""
        purpose = "revision" if schema is dict else "action"
        with path1_logical_request(self, purpose) as budget:
            calls: list[dict[str, object]] = []
            try:
                if schema not in {dict, ActionDecision} or (
                    schema is dict and request.metadata.get("response_schema")
                    != "experience_revision_v2"
                ):
                    raise Path1TerminalError("configuration", "unsupported_path1_schema")
                base_url = os.environ.get(self.base_url_env)
                api_key = os.environ.get(self.api_key_env)
                if not base_url or (self.api_key_required and not api_key):
                    raise Path1TerminalError("configuration", "missing_provider_environment")
                stage = "memory_contract_repair" if request.metadata.get(
                    "experience_semantic_repair"
                ) is True else "primary"
                messages = None
                preserve_action: dict[str, object] | None = None
                while True:
                    repair = stage != "primary"
                    token_key = (
                        "experience_repair_max_output_tokens" if repair
                        else "experience_max_output_tokens"
                    ) if purpose == "revision" else (
                        "repair_max_output_tokens" if repair else "max_output_tokens"
                    )
                    default_tokens = 4096 if repair else 8192
                    completion = self._post_observed(
                        base_url, api_key or "", request, stage=stage, outer_attempt=1,
                        call_diagnostics=calls, messages=messages,
                        max_tokens=int(self.config.get(
                            token_key, default_tokens if purpose == "revision" else 256,
                        )),
                    )
                    try:
                        payload = _path1_load_object(completion)
                        if preserve_action is not None:
                            payload = {**preserve_action, "intent": payload.get("intent"),
                                       "_provider_semantic_repair_used": True}
                        if schema is ActionDecision:
                            _path1_validate_action(payload)
                        _mark_latest_json_status(calls, "valid")
                        break
                    except Path1ResponseContractError as exc:
                        _mark_latest_json_status(calls, "invalid")
                        if not budget.repair_available:
                            raise
                        if schema is ActionDecision and exc.diagnostics["failure_code"] == (
                            "action_intent_invalid"
                        ):
                            preserve_action = dict(payload)
                            stage = "action_semantic_repair"
                        else:
                            stage = "json_repair"
                        messages = _path1_repair_messages(request, completion, purpose)
                diagnostics = _build_provider_diagnostics(calls, retry_count=0)
                diagnostics["logical_request"] = budget.receipt()
                self._last_call_diagnostics = copy.deepcopy(diagnostics)
                if schema is dict:
                    payload["_provider_diagnostics"] = diagnostics
                    payload["_provider_repair_used"] = budget.contract_repairs > 0
                    return payload  # type: ignore[return-value]
                decision = coerce_decision(payload)
                decision.raw_response["_provider_diagnostics"] = diagnostics
                return decision  # type: ignore[return-value]
            except Path1ResponseContractError as exc:
                diagnostics = _build_provider_diagnostics(calls, retry_count=0)
                diagnostics.update(exc.diagnostics)
                diagnostics["logical_request"] = budget.receipt()
                self._last_call_diagnostics = copy.deepcopy(diagnostics)
                if purpose == "action":
                    raise Path1TerminalError(
                        "state", "action_response_contract_exhausted", diagnostics=diagnostics,
                    ) from exc
                raise Path1ResponseContractError(
                    str(exc.diagnostics["failure_code"]), diagnostics=diagnostics,
                ) from exc
            except ProviderExchangeAuditError as exc:
                raise Path1TerminalError("audit", "exchange_persistence_failed", diagnostics={
                    "logical_request": budget.receipt(),
                }) from exc
            except Exception as exc:
                terminal = terminal_from_exception(exc, stage="provider")
                diagnostics = _build_provider_diagnostics(calls, retry_count=0)
                diagnostics.update(terminal.diagnostics)
                diagnostics["logical_request"] = budget.receipt()
                terminal.diagnostics = diagnostics
                self._last_call_diagnostics = copy.deepcopy(diagnostics)
                if terminal is exc:
                    raise
                raise terminal from exc

    def _generate_legacy(self, request: LLMRequest, schema: type[T]) -> T:

        if schema not in {ActionDecision, dict}:
            raise ProviderError(f"{self.provider} does not support schema: {schema!r}")
        api_key = os.environ.get(self.api_key_env)
        base_url = os.environ.get(self.base_url_env)
        if not base_url:
            raise ProviderError(f"{self.provider} lacks environment variable {self.base_url_env}")
        if self.api_key_required and not api_key:
            raise ProviderError(f"{self.provider} lacks environment variable {self.api_key_env}")

        metadata = getattr(request, "metadata", {})
        response_schema = str(metadata.get("response_schema", ""))
        attach_dict_diagnostics = schema is dict and _is_diagnostic_dict_schema(response_schema)
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
                try:
                    _validate_live_action_payload(payload)
                except ProviderError as first_error:
                    repaired_payload = self._repair_or_recover_json(
                        base_url,
                        api_key or "",
                        request,
                        completion,
                        first_error,
                        outer_attempt=outer_attempt,
                        call_diagnostics=call_diagnostics,
                        repair_stage="action_semantic_repair",
                    )
                    # Semantic repair may supply only the missing bounded intent;
                    # it must not replace the model's already parsed poker action.
                    payload["intent"] = repaired_payload.get("intent")
                    payload["_provider_semantic_repair_used"] = True
                    _mark_latest_json_status(call_diagnostics, "valid")
                    _validate_live_action_payload(payload)
                decision = coerce_decision(payload)
                if payload.get("_provider_semantic_repair_used") is True:
                    decision.raw_response["_provider_semantic_repair_used"] = True
                decision.raw_response["_provider_diagnostics"] = _build_provider_diagnostics(
                    call_diagnostics,
                    retry_count=max(0, outer_attempt - 1),
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
            "experience_structured_output_mode": (self.experience_structured_output_mode),
            "thinking_mode": self.thinking_mode or None,
            "reasoning_effort": self.reasoning_effort or None,
            "response_model_identity": self.response_identity_summary(),
            "message": (
                "A base URL is configured; a local service may not require an API key."
                if available and not self.api_key_required
                else "Real calls require both a base URL and an API key."
            ),
        }

    def _repair_or_recover_json(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequest,
        completion: _CompletionResult,
        first_error: Exception,
        *,
        outer_attempt: int,
        call_diagnostics: list[dict[str, object]] | None,
        repair_stage: str = "json_repair",
    ) -> dict[str, object]:

        metadata = getattr(request, "metadata", {})
        if _is_experience_revision_schema(str(metadata.get("response_schema", ""))):
            max_chars = int(metadata.get("experience_max_chars", 2400))
            aux_chars = int(metadata.get("experience_aux_max_chars", 240))
            schema_version = str(metadata.get("response_schema", ""))
            formal_repair = _active_provider_repair_localization()
            if formal_repair is not None:
                if schema_version != "experience_revision_v2":
                    raise ProviderError(
                        "formal exact profile does not disclose full-document JSON repair"
                    )
                repair_instruction = _localized_value(
                    formal_repair,
                    "experience_patch_instruction",
                )
                repair_system = _localized_value(formal_repair, "experience_system")
                repair_suffix = _localized_value(
                    formal_repair,
                    "experience_suffix",
                ).format(aux_chars=aux_chars)
            else:
                repair_instruction = (
                    "Regenerate the structured experience patch. operation must be keep or "
                    "patch; keep requires patches=[]; every patch must retain all typed fields; "
                    "fact_assessments must cover every provided fact ID."
                    if schema_version == "experience_revision_v2"
                    else (
                        f"Regenerate a compact experience revision: new_md is limited to "
                        f"{max_chars} characters. With keep=true, new_md must be empty or exactly "
                        "match the old document. With keep=false, return the complete five-section "
                        "Markdown document beginning with '# My Experience'; do not return a "
                        "patch. Fact-ID arrays may reference only fact_ids supplied in this "
                        "request."
                    )
                )
                repair_system = (
                    "You repair compact experience JSON. Output one complete JSON object matching "
                    "the response schema, with no explanation or reasoning trace."
                )
                repair_suffix = (
                    f"calibration_note and self_check are each limited to {aux_chars} characters. "
                    "Close the JSON object and preserve every required field.\n\n"
                )
            repair_messages = [
                {
                    "role": "system",
                    "content": repair_system,
                },
                {
                    "role": "user",
                    "content": repair_instruction + repair_suffix + request.user_prompt,
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
                    max_tokens=int(self.config.get("experience_repair_max_output_tokens", 2048)),
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
                "experience revision response is not complete JSON; "
                f"finish_reason={completion.finish_reason or 'unknown'}; "
                f"compact regeneration failed={repair_error}"
            ) from first_error

        if _is_semantic_review_schema(str(metadata.get("response_schema", ""))):
            # A reviewer response has its own request-bound schema.  Feeding it
            # into the action JSON repair prompt would silently change the task.
            raise ProviderError(
                "semantic_review_response_invalid_json; "
                f"finish_reason={completion.finish_reason or 'unknown'}"
            ) from first_error

        formal_repair = _active_provider_repair_localization()
        if formal_repair is not None:
            action_system = _localized_value(formal_repair, "action_system")
            action_user = _localized_value(formal_repair, "action_user")
        else:
            action_system = ACTION_SEMANTIC_REPAIR_SYSTEM_PROMPT
            action_user = ACTION_SEMANTIC_REPAIR_USER_PREFIX
        repair_messages = [
            {
                "role": "system",
                "content": action_system,
            },
            {
                "role": "user",
                "content": action_user + completion.content,
            },
        ]
        repair_error: Exception | None = None
        try:
            repaired = self._post_observed(
                base_url,
                api_key,
                request,
                stage=repair_stage,
                outer_attempt=outer_attempt,
                call_diagnostics=call_diagnostics,
                messages=repair_messages,
                max_tokens=int(self.config.get("repair_max_output_tokens", 256)),
            )
            repaired_payload = _load_json_object(repaired.content)
            _mark_latest_json_status(call_diagnostics, "valid")
            return repaired_payload
        except json.JSONDecodeError as exc:
            _mark_latest_json_status(call_diagnostics, "invalid")
            repair_error = exc
        except ProviderError as exc:
            repair_error = exc

        recovered = _recover_truncated_action_object(completion.content)
        if recovered is not None:
            return recovered

        reason = completion.finish_reason or "unknown"
        raise ProviderError(
            "model response is not complete JSON; "
            f"finish_reason={reason}; raw response={completion.content[:200]}; "
            f"compact repair failed={repair_error}"
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

        budget = active_path1_budget()
        if budget is not None:
            budget.reserve(stage)
        post_kwargs: dict[str, object] = {}
        if messages is not None:
            post_kwargs["messages"] = messages
        if max_tokens is not None:
            post_kwargs["max_tokens"] = max_tokens
        if self._exchange_recorder is not None:
            post_kwargs["exchange_context"] = {
                "stage": stage,
                "outer_attempt": outer_attempt,
                "structured_call_index": len(call_diagnostics or []) + 1,
                **({"terminal_policy": PATH1_TERMINAL_POLICY,
                    "logical_request_id": budget.logical_request_id,
                    "request_kind": "initial" if stage == "primary" else "contract_repair"}
                   if budget is not None else {}),
            }
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
            if budget is not None:
                budget.completed_model_responses += 1
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
            if _is_experience_revision_schema(str(request_metadata.get("response_schema", ""))):
                try:
                    _enforce_experience_prompt_token_budget(request, completion, call_diagnostic)
                except ProviderError as exc:
                    if budget is not None:
                        raise Path1TerminalError("state", "experience_prompt_budget_exceeded",
                                                 diagnostics=exc.diagnostics) from exc
                    raise
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
        self._identity_status_counts[status] = self._identity_status_counts.get(status, 0) + 1
        if completion.response_model is not None:
            self._observed_response_models.add(completion.response_model)
        if failure_code is None:
            return status
        call_diagnostic["failure_code"] = failure_code
        diagnostics = _build_provider_diagnostics(
            call_diagnostics,
            retry_count=max(
                0,
                max(
                    (int(item.get("outer_attempt", 0)) for item in call_diagnostics),
                    default=1,
                )
                - 1,
            ),
        )
        self._last_call_diagnostics = copy.deepcopy(diagnostics)
        if active_path1_budget() is not None:
            raise Path1TerminalError("state", "response_model_identity_rejected",
                                     diagnostics=diagnostics)
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
        exchange_context: dict[str, object] | None = None,
    ) -> _CompletionResult:

        url = base_url.rstrip("/") + "/chat/completions"
        response_schema = str(request.metadata.get("response_schema", ""))
        is_experience_revision = _is_experience_revision_schema(response_schema)
        default_temperature = self.config.get("temperature", 0.2)
        request_temperature = (
            self.config.get("experience_temperature", default_temperature)
            if is_experience_revision
            else default_temperature
        )
        resolved_messages = messages or [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        reserved_output_tokens = max_tokens or int(self.config.get("max_output_tokens", 256))
        payload = {
            "model": self.model,
            "temperature": float(request_temperature),
            "max_tokens": reserved_output_tokens,
            "messages": resolved_messages,
        }
        request_identity = None
        if self._request_identity_binding is not None:
            budget = active_path1_budget()
            if budget is None or exchange_context is None:
                raise Path1TerminalError("state", "identity_request_outside_logical_budget")
            request_identity = self._request_identity_binding.issue(
                request, budget, str(exchange_context["stage"]),
            )
            # These are actual OpenAI-compatible HTTP body fields, not prompt
            # annotations. The user ID links the full non-secret audit identity.
            payload["seed"] = request_identity["sampler_seed"]
            payload["user"] = request_identity["request_id"]
            if type(self.config.get("enable_thinking")) is not bool:
                raise Path1TerminalError("configuration", "thinking_policy_not_explicit")
            payload["chat_template_kwargs"] = {"enable_thinking": self.config["enable_thinking"]}
        response_mode = (
            self.experience_structured_output_mode
            if is_experience_revision
            else self.structured_output_mode
        )
        response_format = _response_format(response_mode, request)
        if response_format is not None:
            response_format = _remove_json_schema_keywords(
                response_format,
                self.json_schema_unsupported_keywords,
            )
            payload["response_format"] = response_format
        resolved_context_limit = _resolved_context_limit(
            self.config,
            request,
            is_experience_revision=is_experience_revision,
        )
        estimated_prompt_tokens = _estimate_prompt_token_upper_bound(
            resolved_messages,
            response_format,
        )
        if (
            resolved_context_limit is not None
            and estimated_prompt_tokens + reserved_output_tokens > resolved_context_limit
        ):
            if active_path1_budget() is not None:
                raise Path1TerminalError("state", "context_budget_preflight_failed", diagnostics={
                    "estimated_prompt_tokens_upper_bound": estimated_prompt_tokens,
                    "reserved_output_tokens": reserved_output_tokens,
                    "resolved_context_limit": resolved_context_limit,
                })
            raise ProviderError(
                "context_budget_preflight_failed: "
                f"estimated_prompt_tokens_upper_bound={estimated_prompt_tokens}; "
                f"reserved_output_tokens={reserved_output_tokens}; "
                f"resolved_context_limit={resolved_context_limit}",
                retryable=False,
            )
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
        exchange: dict[str, object] | None = None
        if self._exchange_recorder is not None:
            exchange = {
                "exchange_id": uuid4().hex,
                "call_index": self._exchange_recorder.next_call_index(),
                "client_role": self._exchange_client_role,
                "purpose": response_schema or "action_decision",
                "stage": "direct",
                "agent_id": getattr(request.observation, "agent_id", None),
                "table_id": getattr(request.observation, "table_id", None),
                "hand_id": getattr(request.observation, "hand_id", None),
                **(exchange_context or {}),
                **({"path1_request_identity": request_identity}
                   if request_identity is not None else {}),
            }
            self._record_exchange(
                exchange, "request", base_url, api_key,
                request_payload=payload, elapsed_ms=0.0,
            )
        started = time.perf_counter()
        response_bytes: bytes | None = None
        try:
            if active_path1_budget() is not None:
                active_path1_budget().http_started += 1
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
                body = response_bytes.decode("utf-8")
            if active_path1_budget() is not None:
                active_path1_budget().completed_http_responses += 1
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                self._record_exchange(
                    exchange, "failed", base_url, api_key,
                    failure_category="http_error_body_unavailable", http_status=exc.code,
                    response_text=None, elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                raise
            self._record_exchange(
                exchange, "failed", base_url, api_key,
                failure_category="http_error", http_status=exc.code,
                response_text=error_body, elapsed_ms=(time.perf_counter() - started) * 1000,
                **_raw_response_usage(error_body),
            )
            if active_path1_budget() is not None:
                raise Path1TerminalError("service", "http_error", diagnostics={
                    "http_status": exc.code,
                }) from exc
            detail = f"{exc}"
            if error_body.strip():
                detail = f"{detail}; response: {error_body[:500]}"
            fail_fast = exc.code == 400 or any(
                marker in error_body.lower()
                for marker in (
                    "grammar error",
                    "unimplemented keys",
                    "context length",
                    "context_length",
                    "maximum context",
                )
            )
            raise ProviderError(
                f"{self.provider} HTTP request failed: {detail}",
                retryable=not fail_fast,
            ) from exc
        except urllib.error.URLError as exc:
            self._record_exchange(
                exchange, "failed", base_url, api_key, failure_category="transport_error",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            if active_path1_budget() is not None:
                raise Path1TerminalError("transport", "url_error", diagnostics={
                    "exception_type": type(exc.reason).__name__,
                }) from exc
            raise ProviderError(f"{self.provider} HTTP request failed: {exc}") from exc
        except Exception as exc:
            self._record_exchange(
                exchange, "failed", base_url, api_key, failure_category="response_io_error",
                response_text=(
                    response_bytes.decode("utf-8", errors="replace")
                    if response_bytes is not None else None
                ),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            if active_path1_budget() is not None and isinstance(exc, TimeoutError):
                raise Path1TerminalError("transport", "request_timeout") from exc
            raise
        elapsed = (time.perf_counter() - started) * 1000
        try:
            parsed = json.loads(body)
            response_model, response_model_report_status = _extract_safe_response_model(parsed)
            choice = parsed["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            if active_path1_budget() is None:
                content = content or ""
            finish_reason = str(choice.get("finish_reason") or "")
            usage = parsed.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            prompt_tokens = _optional_int(usage.get("prompt_tokens"))
            completion_tokens = _optional_int(usage.get("completion_tokens"))
            if active_path1_budget() is not None and not isinstance(content, str):
                raise TypeError("model message content is not text")
            if not str(content).strip() and active_path1_budget() is None:
                reasoning_present = bool(message.get("reasoning_content"))
                raise ProviderError(
                    f"{self.provider} empty_content=true; "
                    f"reasoning_present={str(reasoning_present).lower()}; "
                    f"finish_reason={finish_reason or 'unknown'}; "
                    "response lacks final content; reasoning content is not accepted as JSON."
                )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self._record_exchange(
                exchange, "failed", base_url, api_key, failure_category="response_envelope_invalid",
                response_text=body, elapsed_ms=elapsed, **_raw_response_usage(body),
            )
            if active_path1_budget() is not None:
                raise Path1TerminalError("service", "response_envelope_invalid") from exc
            raise ProviderError(
                f"cannot parse {self.provider} response after {elapsed:.1f} ms"
            ) from exc
        except ProviderError:
            self._record_exchange(
                exchange, "failed", base_url, api_key, failure_category="response_content_invalid",
                response_text=body, elapsed_ms=elapsed, **_raw_response_usage(body),
            )
            raise
        self._record_exchange(
            exchange, "completed", base_url, api_key,
            response_text=body, elapsed_ms=elapsed, **_raw_response_usage(body),
        )
        return _CompletionResult(
            content=str(content),
            finish_reason=finish_reason,
            reasoning_present=bool(message.get("reasoning_content")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_model=response_model,
            response_model_report_status=response_model_report_status,
            estimated_prompt_tokens_upper_bound=estimated_prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
            resolved_context_limit=resolved_context_limit,
        )

    def _record_exchange(
        self, exchange: dict[str, object] | None, event: str,
        base_url: str, api_key: str, **fields: object,
    ) -> None:
        if self._exchange_recorder is None or exchange is None:
            return
        try:
            record = {**exchange, "event": event, **fields}
            endpoint = urllib.parse.urlsplit(base_url)
            secrets = [api_key, base_url]
            for value in (endpoint.username, endpoint.password):
                if value:
                    secrets.extend((value, urllib.parse.unquote(value)))
            secrets.extend(value for _, value in urllib.parse.parse_qsl(endpoint.query) if value)
            variants = {value for value in secrets if value}
            variants.update(
                json.dumps(value, ensure_ascii=False)[1:-1] for value in secrets if value
            )
            redaction_applied = False

            def redact(value: object) -> object:
                nonlocal redaction_applied
                if isinstance(value, str):
                    result = value
                    for secret in sorted(variants, key=len, reverse=True):
                        result = result.replace(secret, "[REDACTED_CONNECTION_SECRET]")
                    redaction_applied = redaction_applied or result != value
                    return result
                if isinstance(value, dict):
                    return {key: redact(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [redact(item) for item in value]
                return value

            safe_record = {key: redact(value) for key, value in record.items()}
            safe_record["credential_redaction_applied"] = redaction_applied
            self._exchange_recorder.append(safe_record)
        except Exception as exc:
            raise ProviderExchangeAuditError(
                "provider_exchange_audit_serialization_failed; stopping run"
            ) from exc


def _raw_response_usage(body: str) -> dict[str, object]:
    """Copy service-reported usage without inventing token counts for missing fields."""

    try:
        parsed = json.loads(body)
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
        finish_reasons = [
            item.get("finish_reason") for item in choices if isinstance(item, dict)
        ] if isinstance(choices, list) else []
    except (TypeError, ValueError):
        usage, finish_reasons = None, []
    return {"usage": usage, "finish_reasons": finish_reasons}


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
    diagnostic = {
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
        "response_model_identity_source": ("chat_completions_top_level_model_self_report"),
        "response_model_identity_policy": identity_policy,
        "requested_model": requested_model,
        "response_model": completion.response_model,
        "response_model_report_status": completion.response_model_report_status,
        "response_model_identity_status": "not_checked",
        "response_model_identity_verified": False,
        "response_model_exact_match": False,
    }
    for key, value in (
        (
            "estimated_prompt_tokens_upper_bound",
            completion.estimated_prompt_tokens_upper_bound,
        ),
        ("reserved_output_tokens", completion.reserved_output_tokens),
        ("resolved_context_limit", completion.resolved_context_limit),
    ):
        if value is not None:
            diagnostic[key] = value
    return diagnostic


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
        "response_model_identity_source": ("chat_completions_top_level_model_self_report"),
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
            f"experience_prompt_token_budget_exceeded: {prompt_tokens} > {raw_budget}",
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
        int(item["prompt_tokens"]) for item in completed if type(item.get("prompt_tokens")) is int
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
        "action_semantic_repair_call_count": sum(
            item.get("stage") == "action_semantic_repair" for item in calls
        ),
        # A transport/provider failure has no completed response whose usage can
        # be audited.  Keep it in failed_http_call_count instead of double
        # labelling the same call as both failed and usage-missing.
        "usage_missing_call_count": sum(
            item.get("status") == "completed" and item.get("usage_available") is not True
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
        "response_model_identity_source": ("chat_completions_top_level_model_self_report"),
        "response_model_identity_cryptographic_attestation": False,
        "response_model_identity_policy": last_call.get("response_model_identity_policy"),
        "requested_model": last_call.get("requested_model"),
        "response_model": last_call.get("response_model"),
        "response_model_report_status": last_call.get("response_model_report_status"),
        "response_model_identity_status": last_call.get("response_model_identity_status"),
        "response_model_identity_verified": bool(
            last_call.get("response_model_identity_verified", False)
        ),
        "response_model_exact_match": bool(last_call.get("response_model_exact_match", False)),
        "response_model_identity_status_counts": dict(sorted(identity_status_counts.items())),
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
        if _SAFE_MODEL_TOKEN.fullmatch(text) and _SENSITIVE_MODEL_TOKEN.search(text) is None
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

    effective_report_status = "reported_valid" if response_model is not None else report_status
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


def _remove_json_schema_keywords(
    value: object,
    unsupported_keywords: set[str],
) -> object:
    """Copy JSON data while removing provider-declared unsupported schema keywords.

    This changes only the provider grammar. Request-bound identifiers and duplicate
    arrays remain subject to application validation before a revision is applied.
    """

    if isinstance(value, dict):
        return {
            key: _remove_json_schema_keywords(item, unsupported_keywords)
            for key, item in value.items()
            if key not in unsupported_keywords
        }
    if isinstance(value, list):
        return [_remove_json_schema_keywords(item, unsupported_keywords) for item in value]
    return value


def _resolved_context_limit(
    config: dict[str, object],
    request: LLMRequest,
    *,
    is_experience_revision: bool,
) -> int | None:
    """Resolve the hard request context limit without consulting provider errors."""

    candidates = [
        request.metadata.get("resolved_context_limit"),
        config.get("context_limit"),
    ]
    if is_experience_revision:
        candidates.append(request.metadata.get("experience_prompt_token_budget"))
    for value in candidates:
        if value is None:
            continue
        if type(value) is not int or value < 1:
            raise ProviderError(
                "resolved_context_limit must be a positive integer",
                retryable=False,
            )
        return value
    return None


def _estimate_prompt_token_upper_bound(
    messages: list[dict[str, str]],
    response_format: dict[str, object] | None,
) -> int:
    """Return a tokenizer-independent UTF-8 byte upper bound for the prompt."""

    envelope: dict[str, object] = {"messages": messages}
    if response_format is not None:
        envelope["response_format"] = response_format
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return max(1, len(encoded))


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
        }

    from agentmemeval.path1_v1.prompts.profiles import active_experience_section_labels

    sections = list(active_experience_section_labels())
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


def _active_provider_repair_localization() -> dict[str, object] | None:
    from agentmemeval.path1_v1.prompts.profiles import active_prompt_localization

    localization = active_prompt_localization()
    if localization is None:
        return None
    value = localization.get("provider_repair")
    if not isinstance(value, dict):
        raise ProviderError("formal provider-repair localization is missing")
    return value


def _localized_value(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ProviderError(f"formal provider-repair localization is missing: {key}")
    return value


def _experience_full_document_response_format(
    request: LLMRequest,
) -> dict[str, object]:
    """Build the strict v1 whole-document schema using fact IDs only."""

    metadata = request.metadata
    max_chars = int(metadata.get("experience_max_chars", 2400))
    aux_chars = int(metadata.get("experience_aux_max_chars", 240))
    fact_ids = [str(value) for value in metadata.get("fact_ids", [])]
    require_fact_classification = bool(metadata.get("experience_fact_classification_fields", False))
    fact_id_schema: dict[str, object] = {"type": "string"}
    if fact_ids:
        fact_id_schema["enum"] = fact_ids
    fact_id_array = {
        "type": "array",
        "items": fact_id_schema,
        "minItems": 0,
        "maxItems": len(fact_ids),
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
        raise ProviderError(f"{response_schema} requires object metadata.response_json_schema")
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
        raise ProviderError(f"{response_schema} response_json_schema must be a closed object")
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
                raise_sizing.get("allowed_amounts") if isinstance(raise_sizing, dict) else None
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
                            "maxLength": 40,
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


def _path1_load_object(completion: _CompletionResult) -> dict[str, object]:
    if completion.finish_reason == "length":
        raise Path1ResponseContractError("response_truncated")
    if completion.finish_reason != "stop":
        raise Path1TerminalError("service", "unexpected_finish_reason")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError("non-finite JSON number")

    try:
        parsed = json.loads(completion.content, object_pairs_hook=pairs,
                            parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise Path1ResponseContractError("response_json_invalid") from exc
    if not isinstance(parsed, dict):
        raise Path1ResponseContractError("response_not_object")
    if any(key.startswith("_") for key in parsed):
        raise Path1ResponseContractError("response_reserved_field")
    return parsed


def _path1_validate_action(payload: dict[str, object]) -> None:
    fields = {"action_type", "amount", "confidence", "reason_summary", "intent"}
    supplied = set(payload) - {"_provider_semantic_repair_used"}
    amount, confidence = payload.get("amount"), payload.get("confidence")
    reason = payload.get("reason_summary")
    if (supplied != fields or not isinstance(payload.get("action_type"), str)
            or (amount is not None and type(amount) is not int)
            or type(confidence) not in {int, float} or not math.isfinite(confidence)
            or not 0 <= confidence <= 1 or not isinstance(reason, str) or len(reason) > 300):
        raise Path1ResponseContractError("action_shape_invalid")
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip() or len(intent) > 40:
        raise Path1ResponseContractError("action_intent_invalid")


def _path1_repair_messages(
    request: LLMRequest, completion: _CompletionResult, purpose: str,
) -> list[dict[str, str]]:
    # Retain exactly the original visible information; no hidden state is added.
    instructions = (
        "Repair the completed response's JSON/contract failure. Return one complete "
        "JSON object, no markdown. Preserve explicit action fields when repairing intent. "
        "Use only the original request below. Do not invent evidence. "
        f"Request purpose: {purpose}."
    )
    return [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt},
        {"role": "assistant", "content": completion.content},
        {"role": "user", "content": instructions},
    ]


def _load_json_object(content: str) -> dict[str, object]:

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
        raise ProviderError(
            "action response requires non-empty intent; reason_summary is not a substitute"
        )
    if len(intent) > 40:
        raise ProviderError(f"action response intent is too long: {len(intent)} > 40")


def _find_first_json_object(text: str) -> str:

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
        "reason_summary": (
            "The model summary was truncated; action fields were recovered for validation."
        ),
        "intent": "",
        "provider_recovered_from_truncation": True,
    }
