from __future__ import annotations

import json
from typing import Any

import pytest

from agentmemeval.path1_v1.core.domain import (
    ActionDecision,
    AgentObservation,
    LegalAction,
    LegalActionSet,
    MemoryContext,
)
from agentmemeval.path1_v1.llm.providers import openai_compatible as provider_module
from agentmemeval.path1_v1.llm.schemas import LLMRequest


@pytest.mark.parametrize("invalid_intent", ["", "x" * 41])
def test_action_intent_semantic_repair_preserves_primary_action(
    monkeypatch: pytest.MonkeyPatch,
    invalid_intent: str,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_BASE_URL", "http://unused")
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-only")
    client = provider_module.OpenAICompatibleClient(
        {
            "provider": "openai_compatible",
            "model": "test-model",
            "base_url_env": "TEST_OPENAI_BASE_URL",
            "api_key_env": "TEST_OPENAI_API_KEY",
            "api_key_required": True,
            "max_retries": 0,
            "structured_output_mode": "json_object",
            "response_model_identity_policy": "off",
        }
    )
    primary = {
        "action_type": "call",
        "amount": None,
        "confidence": 0.7,
        "reason_summary": "Call with adequate pot odds.",
        "intent": invalid_intent,
    }
    repair = {
        "action_type": "fold",
        "amount": None,
        "confidence": 0.1,
        "reason_summary": "This replacement action must be ignored.",
        "intent": "Preserve optionality.",
    }
    completions = iter(
        [
            provider_module._CompletionResult(json.dumps(primary), "stop"),
            provider_module._CompletionResult(json.dumps(repair), "stop"),
        ]
    )

    def fake_post(*_args: object, **_kwargs: object) -> Any:
        return next(completions)

    monkeypatch.setattr(client, "_post", fake_post)
    decision = client.generate_structured(_request(), ActionDecision)

    assert decision.action_type == "call"
    assert decision.confidence == 0.7
    assert decision.reason_summary == "Call with adequate pot odds."
    assert decision.intent == "Preserve optionality."
    assert decision.raw_response["_provider_semantic_repair_used"] is True
    diagnostics = decision.raw_response["_provider_diagnostics"]
    assert diagnostics["http_call_count"] == 2
    assert diagnostics["json_repair_call_count"] == 0
    assert diagnostics["action_semantic_repair_call_count"] == 1


def _request() -> LLMRequest:
    observation = AgentObservation(
        agent_id="agent_00",
        table_id="semantic-repair-fixture",
        hand_id="semantic-repair-fixture-h1",
        phase="preflop",
        seat=0,
        hole_cards=["As", "Kd"],
        community_cards=[],
        pot=3,
        current_bet=1,
        to_call=1,
        players=[],
        action_history=[],
        legal_actions=LegalActionSet([LegalAction("fold"), LegalAction("call")]),
        seed=7,
    )
    return LLMRequest(
        observation=observation,
        memory_context=MemoryContext(),
        system_prompt="fixture",
        user_prompt="fixture",
    )
