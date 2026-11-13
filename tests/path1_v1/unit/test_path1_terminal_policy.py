from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.core.domain import (
    ActionDecision,
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    LegalAction,
    LegalActionSet,
    MemoryContext,
    PlayerPublicState,
)
from agentmemeval.path1_v1.llm.providers import openai_compatible as provider
from agentmemeval.path1_v1.llm.schemas import LLMRequest
from agentmemeval.path1_v1.llm.terminal_policy import (
    PATH1_TERMINAL_POLICY,
    Path1TerminalError,
    active_path1_budget,
    require_path1_terminal_policy,
)
from agentmemeval.path1_v1.memory.experiential import (
    ExperientialMemory,
    is_experience_revision_operational_failure,
)


def request():
    observation = AgentObservation(
        "hero",
        "table",
        "hand1",
        "preflop",
        0,
        ["As", "Kd"],
        [],
        3,
        1,
        1,
        [
            PlayerPublicState("hero", 0, 1000, 0, 0, False, False),
            PlayerPublicState("villain", 1, 998, 2, 2, False, False),
        ],
        [],
        LegalActionSet([LegalAction("fold"), LegalAction("call")]),
        7,
        small_blind_amount=1,
        big_blind_amount=2,
    )
    return LLMRequest(observation, MemoryContext(), "system visible only", "user visible only")


def action(intent="Preserve optionality"):
    return {
        "action_type": "call",
        "amount": None,
        "confidence": 0.7,
        "reason_summary": "Price is adequate",
        "intent": intent,
    }


def response(content, finish="stop", model="fixture"):
    return json.dumps(
        {
            "model": model,
            "choices": [
                {
                    "message": {
                        "content": (content if isinstance(content, str) else json.dumps(content))
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    ).encode()


def client(monkeypatch, tmp_path, **config):
    monkeypatch.setenv("PATH1_TEST_URL", "http://unused.example/v1")
    instance = provider.OpenAICompatibleClient(
        {
            "provider": "openai_compatible",
            "model": "fixture",
            "base_url_env": "PATH1_TEST_URL",
            "api_key_env": "PATH1_TEST_KEY",
            "api_key_required": False,
            "terminal_policy": PATH1_TERMINAL_POLICY,
            "max_retries": 0,
            "response_model_identity_policy": "strict_exact",
            **config,
        }
    )
    instance.bind_exchange_recorder(provider.ProviderExchangeRecorder(tmp_path / "http.jsonl"))
    return instance


def mock_http(monkeypatch, replies):
    calls = []
    responses = iter(replies)

    def urlopen(req, **kwargs):
        calls.append(json.loads(req.data))
        reply = next(responses)
        if isinstance(reply, BaseException):
            raise reply
        return io.BytesIO(reply)

    monkeypatch.setattr(provider.urllib.request, "urlopen", urlopen)
    return calls


def memory(instance):
    return ExperientialMemory(
        "hero",
        revision_strategy="llm",
        llm_client=instance,
        model="fixture",
        prompt_token_budget=65536,
    )


def trajectory():
    req = request()
    decision = ActionDecision("call", intent="Preserve optionality")
    event = DecisionEvent(
        "hero", "table", "hand1", req.observation, decision, decision, MemoryContext()
    )
    return HandTrajectory("hero", "table", "hand1", [event], [], 0, 1000, {}, "fixture")


def keep():
    return {
        "operation": "keep",
        "patches": [],
        "fact_assessments": [],
        "calibration_note": "No new evidence",
        "self_check": "Keep unchanged",
    }


def invalid_patch():
    return {**keep(), "operation": "patch"}


def test_capability_requires_actual_implementation_and_recorder(monkeypatch, tmp_path):
    with pytest.raises(Path1TerminalError, match="unverified_provider"):
        require_path1_terminal_policy(
            SimpleNamespace(terminal_policy_version=PATH1_TERMINAL_POLICY)
        )
    instance = client(monkeypatch, tmp_path)
    assert require_path1_terminal_policy(instance)["transport_retries"] == 0
    instance._exchange_recorder = None
    with pytest.raises(Path1TerminalError, match="exchange_recorder_not_bound"):
        instance.generate_structured(request(), ActionDecision)


def test_policy_cannot_be_claimed_by_metadata_or_enabled_with_retry(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path, terminal_policy="legacy")
    req = request()
    req.metadata["terminal_policy"] = PATH1_TERMINAL_POLICY
    with pytest.raises(Path1TerminalError, match="claim_not_enabled"):
        instance.generate_structured(req, ActionDecision)
    with pytest.raises(Path1TerminalError, match="transport_retries_must_be_zero"):
        client(monkeypatch, tmp_path, max_retries=1)


def test_action_initial_and_one_semantic_repair_are_audited(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(
        monkeypatch, [response(action("")), response({**action(), "action_type": "fold"})]
    )
    result = instance.generate_structured(request(), ActionDecision)
    assert result.action_type == "call"
    receipt = result.raw_response["_provider_diagnostics"]["logical_request"]
    assert receipt["initial_requests"] == receipt["contract_repairs"] == 1
    assert receipt["http_started"] == receipt["completed_model_responses"] == len(calls) == 2
    records = [json.loads(line) for line in (tmp_path / "http.jsonl").read_text().splitlines()]
    assert {row["logical_request_id"] for row in records} == {receipt["logical_request_id"]}
    assert [row["request_kind"] for row in records[::2]] == ["initial", "contract_repair"]
    assert calls[1]["messages"][:2] == calls[0]["messages"]
    assert active_path1_budget() is None


@pytest.mark.parametrize(
    "first,second",
    [
        (response("{bad"), response(action(""))),
        (response(action("")), response("{bad")),
        (response(action(), "length"), response('{"action_type":"call","amount":null', "length")),
        (response('{"action_type":"call","action_type":"fold"}'), response("{}")),
        (response("prefix " + json.dumps(action())), response("{}")),
    ],
)
def test_action_format_and_semantic_repairs_share_one_slot(monkeypatch, tmp_path, first, second):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [first, second, response(action())])
    with pytest.raises(Path1TerminalError, match="action_response_contract_exhausted") as caught:
        instance.generate_structured(request(), ActionDecision)
    assert len(calls) == 2
    assert caught.value.retryable is False
    assert caught.value.diagnostics["logical_request"]["truncated_action_recovery"] is False


@pytest.mark.parametrize(
    "failure,category",
    [
        (TimeoutError("timeout"), "transport"),
        (urllib.error.URLError("connection reset"), "transport"),
        (
            urllib.error.HTTPError("http://unused", 503, "out of memory", {}, io.BytesIO(b"OOM")),
            "service",
        ),
        (MemoryError("out of memory"), "state"),
        (RuntimeError("unknown"), "unknown"),
        (b'{"choices":[]}', "service"),
        (response(action(), model="wrong-model"), "state"),
        (response(action(), finish="content_filter"), "service"),
    ],
)
def test_non_contract_failures_never_retry_or_become_keep(monkeypatch, tmp_path, failure, category):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [failure, response(keep())])
    mem = memory(instance)
    before = mem.current.to_dict()
    with pytest.raises(Path1TerminalError) as caught:
        mem._revise_from_window([trajectory()])
    assert len(calls) == 1
    assert caught.value.diagnostics["failure_category"] == category
    assert caught.value.diagnostics["external_failure_proven"] is False
    assert caught.value.retryable is False
    assert mem.current.to_dict() == before
    assert not mem.revision_log
    assert active_path1_budget() is None


def test_audit_failure_stops_before_network(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    instance.bind_exchange_recorder(
        provider.ProviderExchangeRecorder(tmp_path / "missing" / "http.jsonl")
    )
    calls = mock_http(monkeypatch, [response(keep())])
    with pytest.raises(Path1TerminalError, match="audit:exchange_persistence_failed"):
        memory(instance)._revise_from_window([trajectory()])
    assert not calls


@pytest.mark.parametrize(
    "replies,stages,status",
    [
        ([response("{bad"), response(invalid_patch())], ["primary", "json_repair"], "rejected"),
        (
            [response(invalid_patch()), response("{bad")],
            ["primary", "memory_contract_repair"],
            "response_contract_failed",
        ),
        (
            [response("{bad"), response("{bad")],
            ["primary", "json_repair"],
            "response_contract_failed",
        ),
        (
            [response(invalid_patch()), response(invalid_patch())],
            ["primary", "memory_contract_repair"],
            "rejected",
        ),
        (
            [response(invalid_patch()), response(keep())],
            ["primary", "memory_contract_repair"],
            "kept_by_model",
        ),
    ],
)
def test_memory_and_provider_repair_share_two_total_requests(
    monkeypatch, tmp_path, replies, stages, status
):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [*replies, response(keep())])
    mem = memory(instance)
    before = mem.current.to_dict()
    result = mem._revise_from_window([trajectory()])
    assert len(calls) == 2
    assert result["validation_status"] == status
    assert result["terminal_policy_receipt"]["stages"] == stages
    assert result["terminal_policy_receipt"]["contract_repairs"] == 1
    assert result["fallback_used"] is False
    assert not is_experience_revision_operational_failure(result)
    assert mem.current.to_dict() == before
    assert active_path1_budget() is None


def test_failure_during_memory_repair_is_fatal_after_two_calls(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(
        monkeypatch, [response(invalid_patch()), TimeoutError("timeout"), response(keep())]
    )
    with pytest.raises(Path1TerminalError, match="transport:request_timeout") as caught:
        memory(instance)._revise_from_window([trajectory()])
    assert len(calls) == 2
    assert caught.value.diagnostics["logical_request"]["contract_repairs"] == 1


def test_unknown_memory_validation_exception_is_not_completed_response_failure(
    monkeypatch, tmp_path
):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [response(keep())])
    mem = memory(instance)

    def broken_validator(*args, **kwargs):
        raise RuntimeError("validator state bug")

    monkeypatch.setattr(mem, "_revision_from_structured_patch", broken_validator)
    with pytest.raises(Path1TerminalError, match="unknown:unexpected_exception"):
        mem._revise_from_window([trajectory()])
    assert len(calls) == 1


def test_completed_contract_keep_receipt_survives_compact_commit(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    mock_http(monkeypatch, [response("{bad"), response("{bad")])
    mem = memory(instance)
    before = mem.current.to_dict()
    result = mem._revise_from_window([trajectory()])
    mem.commit_revision([trajectory()], result)
    assert mem.current.to_dict() == before
    compact = mem.revision_log[-1]
    assert compact["terminal_outcome"] == "response_contract_preserved_current"
    assert compact["terminal_policy_receipt"]["completed_model_responses"] == 2
    assert not is_experience_revision_operational_failure(compact)


def test_valid_revision_uses_one_request_and_no_repair(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [response(keep())])
    result = memory(instance)._revise_from_window([trajectory()])
    assert len(calls) == 1
    assert result["validation_status"] == "kept_by_model"
    assert result["terminal_policy_receipt"]["contract_repairs"] == 0
    assert result["terminal_outcome"] == "completed_revision"


def test_context_preflight_does_not_send_or_repair(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path, context_limit=64)
    calls = mock_http(monkeypatch, [response(keep())])
    with pytest.raises(Path1TerminalError, match="state:context_budget_preflight_failed") as caught:
        memory(instance)._revise_from_window([trajectory()])
    assert not calls
    receipt = caught.value.diagnostics["logical_request"]
    assert receipt["http_started"] == receipt["contract_repairs"] == 0


@pytest.mark.parametrize(
    "schema,semantics",
    [
        ("full_document_v1", "paper_faithful_v1"),
        ("structured_patch_v2", "paper_faithful_v1"),
    ],
)
def test_unsupported_memory_mode_is_rejected_before_http(monkeypatch, tmp_path, schema, semantics):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [response(keep())])
    mem = ExperientialMemory(
        "hero",
        revision_strategy="llm",
        llm_client=instance,
        semantics_mode=semantics,
        revision_contract=schema,
    )
    with pytest.raises(Path1TerminalError, match="unsupported_path1_memory_contract"):
        mem._revise_from_window([trajectory()])
    assert not calls


def test_legacy_transport_failure_still_has_legacy_keep_behavior(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path, terminal_policy="legacy")
    calls = mock_http(monkeypatch, [TimeoutError("timeout")])
    result = memory(instance)._revise_from_window([trajectory()])
    assert len(calls) == 1
    assert result["keep"] is result["fallback_used"] is True
    assert "terminal_policy_receipt" not in result
    assert is_experience_revision_operational_failure(result)


def test_budget_resets_for_next_logical_call(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [response(action("")), response(action()), response(action())])
    first = instance.generate_structured(request(), ActionDecision)
    second = instance.generate_structured(request(), ActionDecision)
    first_receipt = first.raw_response["_provider_diagnostics"]["logical_request"]
    second_receipt = second.raw_response["_provider_diagnostics"]["logical_request"]
    assert len(calls) == 3
    assert first_receipt["logical_request_id"] != second_receipt["logical_request_id"]
    assert first_receipt["contract_repairs"] == 1
    assert second_receipt["contract_repairs"] == 0


def test_completed_empty_response_can_repair_but_non_text_envelope_cannot(monkeypatch, tmp_path):
    instance = client(monkeypatch, tmp_path)
    calls = mock_http(monkeypatch, [response(""), response(keep())])
    result = memory(instance)._revise_from_window([trajectory()])
    assert len(calls) == 2
    assert result["validation_status"] == "kept_by_model"
    raw = json.loads(response(keep()))
    raw["choices"][0]["message"]["content"] = []
    calls = mock_http(monkeypatch, [json.dumps(raw).encode()])
    with pytest.raises(Path1TerminalError, match="service:response_envelope_invalid"):
        memory(instance)._revise_from_window([trajectory()])
    assert len(calls) == 1
