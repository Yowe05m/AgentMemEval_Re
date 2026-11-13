"""Path1's shared logical request budget and bounded failure categories.

The context is process-local execution state, never a user-supplied metadata
claim. One memory revision may span two provider calls but has one repair slot.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from agentmemeval.path1_v1.core.errors import ProviderError

PATH1_TERMINAL_POLICY = "path1_strict_v1"


class Path1TerminalError(ProviderError):
    """Run-fatal error; no automatic retry or successful-completer substitution."""

    def __init__(self, category: str, code: str, *, diagnostics: dict | None = None) -> None:
        if category not in {"configuration", "service", "transport", "state", "audit", "unknown"}:
            raise ValueError("unsupported Path1 failure category")
        super().__init__(
            f"path1_terminal:{category}:{code}",
            retryable=False,
            diagnostics={
                **(diagnostics or {}),
                "terminal_policy": PATH1_TERMINAL_POLICY,
                "failure_category": category,
                "failure_code": code,
                "automatic_retry_allowed": False,
                "external_failure_proven": False,
                "terminal": True,
            },
        )


class Path1ResponseContractError(ProviderError):
    """Completed model response failed its contract; revision may preserve memory."""

    def __init__(self, code: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(
            f"path1_response_contract:{code}",
            retryable=False,
            diagnostics={
                **(diagnostics or {}),
                "terminal_policy": PATH1_TERMINAL_POLICY,
                "failure_category": "completed_response_contract",
                "failure_code": code,
                "automatic_retry_allowed": False,
                "external_failure_proven": False,
                "terminal": False,
            },
        )


@dataclass
class LogicalRequestBudget:
    owner_id: int
    purpose: str
    logical_request_id: str = field(default_factory=lambda: uuid4().hex)
    initial_requests: int = 0
    contract_repairs: int = 0
    http_started: int = 0
    completed_http_responses: int = 0
    completed_model_responses: int = 0
    stages: list[str] = field(default_factory=list)

    @property
    def repair_available(self) -> bool:
        return self.initial_requests == 1 and self.contract_repairs == 0

    def reserve(self, stage: str) -> None:
        if self.initial_requests == 0:
            if stage != "primary":
                raise Path1TerminalError("state", "repair_without_initial_request")
            self.initial_requests = 1
        else:
            if stage == "primary":
                raise Path1TerminalError("state", "repeated_initial_request")
            if self.completed_model_responses == 0:
                raise Path1TerminalError("state", "repair_without_completed_response")
            if not self.repair_available:
                raise Path1ResponseContractError(
                    "repair_budget_exhausted",
                    diagnostics={
                        "logical_request": self.receipt(),
                    },
                )
            self.contract_repairs += 1
        self.stages.append(stage)

    def receipt(self) -> dict[str, object]:
        return {
            "policy": PATH1_TERMINAL_POLICY,
            "logical_request_id": self.logical_request_id,
            "purpose": self.purpose,
            "initial_requests": self.initial_requests,
            "contract_repairs": self.contract_repairs,
            "max_contract_repairs": 1,
            "transport_retries": 0,
            "http_started": self.http_started,
            "completed_http_responses": self.completed_http_responses,
            "completed_model_responses": self.completed_model_responses,
            "stages": list(self.stages),
            "truncated_action_recovery": False,
        }


_ACTIVE: ContextVar[LogicalRequestBudget | None] = ContextVar("path1_request_budget", default=None)


def active_path1_budget() -> LogicalRequestBudget | None:
    return _ACTIVE.get()


def require_path1_terminal_policy(client: object) -> dict[str, object]:
    """Verify the implemented provider path, not a metadata capability claim.

    This is a local source capability check, not backend/GPU/token admission.
    The recorder must already be bound before this check or any live request.
    """
    from agentmemeval.path1_v1.llm.providers.openai_compatible import (
        OpenAICompatibleClient,
        ProviderExchangeRecorder,
    )

    if (
        type(client) is not OpenAICompatibleClient
        or getattr(client.generate_structured, "__func__", None)
        is not OpenAICompatibleClient.generate_structured
    ):
        raise Path1TerminalError("configuration", "unverified_provider_implementation")
    if (
        client.terminal_policy_version != PATH1_TERMINAL_POLICY
        or client.config.get("terminal_policy") != PATH1_TERMINAL_POLICY
    ):
        raise Path1TerminalError("configuration", "path1_policy_not_enabled")
    if client.max_retries != 0:
        raise Path1TerminalError("configuration", "transport_retry_not_disabled")
    if type(client._exchange_recorder) is not ProviderExchangeRecorder:
        raise Path1TerminalError("audit", "exchange_recorder_not_bound")
    return {
        "policy": PATH1_TERMINAL_POLICY,
        "max_contract_repairs": 1,
        "transport_retries": 0,
        "truncated_action_recovery": False,
        "raw_exchange_recording": True,
    }


@contextmanager
def path1_logical_request(owner: object, purpose: str) -> Iterator[LogicalRequestBudget]:
    if purpose not in {"action", "revision"}:
        raise Path1TerminalError("configuration", "unsupported_request_purpose")
    current = _ACTIVE.get()
    if current is not None:
        if current.owner_id != id(owner) or current.purpose != purpose:
            raise Path1TerminalError("state", "cross_client_or_purpose_request_scope")
        yield current
        return
    budget = LogicalRequestBudget(id(owner), purpose)
    token = _ACTIVE.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE.reset(token)


def terminal_from_exception(exc: Exception, *, stage: str) -> Path1TerminalError:
    if isinstance(exc, Path1TerminalError):
        return exc
    category = "state" if isinstance(exc, (MemoryError, RecursionError)) else "unknown"
    return Path1TerminalError(
        category,
        "unexpected_exception",
        diagnostics={
            "exception_type": type(exc).__name__,
            "stage": stage,
        },
    )
