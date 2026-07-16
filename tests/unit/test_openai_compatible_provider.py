"""
模块说明：本模块测试 OpenAI-compatible Provider 的本地模型兼容路径。
核心职责：覆盖无 API key 本地服务、provider-only doctor 配置和宽松 JSON 解析。
输入与输出：输入离线 monkeypatch 配置，输出 pytest 断言结果。
依赖边界：不访问网络，不读取真实密钥。
不负责：不测试具体 vLLM、Ollama 或 LMDeploy 服务。
"""

import json

import pytest

from agentmemeval.cli.main import main
from agentmemeval.core.domain import (
    ActionDecision,
    AgentObservation,
    LegalAction,
    LegalActionSet,
    MemoryContext,
    PlayerPublicState,
)
from agentmemeval.core.errors import ProviderError
from agentmemeval.llm.providers.openai_compatible import (
    OpenAICompatibleClient,
    _CompletionResult,
    _response_format,
)
from agentmemeval.llm.schemas import LLMRequest


def test_provider_only_config_can_drive_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    """
    功能：验证 doctor 可以读取 provider-only 配置且不会被默认 mock 覆盖。
    参数：
        capsys：pytest 输出捕获夹具。
    返回：无。
    副作用：执行 CLI main。
    异常：断言失败时由 pytest 报告。
    设计说明：本地模型联调通常先检查 provider YAML，而不是完整实验配置。
    """

    code = main(["doctor", "--config", "configs/providers/openai_compatible.yaml"])
    assert code == 0
    health = json.loads(capsys.readouterr().out)
    assert health["provider"] == "openai_compatible"
    assert health["model"] == "example-compatible-model"


def test_local_service_can_skip_api_key_and_parse_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证本地 OpenAI-compatible 服务可配置为不要求 API key。
    参数：
        monkeypatch：pytest 环境变量和方法替换夹具。
    返回：无。
    副作用：设置临时环境变量。
    异常：断言失败时由 pytest 报告。
    设计说明：AutoDL 上的 vLLM/Ollama 服务经常只需要本机 base URL。
    """

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        {
            "provider": "openai_compatible",
            "model": "local-test-model",
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "api_key_required": False,
        }
    )
    monkeypatch.setattr(
        client,
        "_post",
        lambda base_url, api_key, request: _CompletionResult(
            (
                "```json\n"
                '{"action_type": "call", "amount": null, "confidence": 0.8, '
                '"reason_summary": "跟注观察"}\n'
                "```"
            ),
            "stop",
        ),
    )

    health = client.healthcheck()
    assert health["available"] is True
    assert health["api_key_required"] is False
    decision = client.generate_structured(object(), ActionDecision)  # type: ignore[arg-type]
    assert decision.action_type == "call"
    assert decision.amount is None


def test_truncated_json_uses_compact_repair_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次输出截断时，应以短提示修复 JSON，而不是直接终止实验。"""

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAICompatibleClient(
        {
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_required": False,
            "max_retries": 0,
        }
    )
    completions = iter(
        [
            _CompletionResult(
                '{"action_type":"call","amount":null,"confidence":0.9,'
                '"reason_summary":"过长理由',
                "length",
            ),
            _CompletionResult(
                '{"action_type":"call","amount":null,"confidence":0.9,'
                '"reason_summary":"赔率可接受"}',
                "stop",
            ),
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> _CompletionResult:
        calls.append(kwargs)
        return next(completions)

    monkeypatch.setattr(client, "_post", fake_post)
    decision = client.generate_structured(object(), ActionDecision)  # type: ignore[arg-type]

    assert decision.action_type == "call"
    assert decision.reason_summary == "赔率可接受"
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == 256
    assert "JSON 修复器" in calls[1]["messages"][0]["content"]  # type: ignore[index]


def test_truncated_json_conservatively_recovers_complete_action_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修复请求也失败时，只恢复截断前已经完整返回的动作字段。"""

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAICompatibleClient(
        {
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_required": False,
            "max_retries": 0,
        }
    )
    completions = iter(
        [
            _CompletionResult(
                '{"action_type":"fold","amount":null,"confidence":0.7,'
                '"reason_summary":"未完成',
                "length",
            ),
            _CompletionResult("仍然不是 JSON", "length"),
        ]
    )
    monkeypatch.setattr(client, "_post", lambda *args, **kwargs: next(completions))

    decision = client.generate_structured(object(), ActionDecision)  # type: ignore[arg-type]

    assert decision.action_type == "fold"
    assert decision.confidence == 0.7
    assert decision.raw_response["provider_recovered_from_truncation"] is True


def test_irrecoverable_truncation_reports_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键字段不完整时应保留 length 诊断信息并安全失败。"""

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAICompatibleClient(
        {
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_required": False,
            "max_retries": 0,
        }
    )
    completions = iter(
        [
            _CompletionResult('{"action_type":"call', "length"),
            _CompletionResult("坏响应", "length"),
        ]
    )
    monkeypatch.setattr(client, "_post", lambda *args, **kwargs: next(completions))

    with pytest.raises(ProviderError, match="finish_reason=length"):
        client.generate_structured(object(), ActionDecision)  # type: ignore[arg-type]


def _experience_request() -> LLMRequest:
    return LLMRequest(
        observation=object(),  # type: ignore[arg-type]
        memory_context=MemoryContext(),
        system_prompt="experience system",
        user_prompt="experience evidence",
        metadata={
            "response_schema": "experience_revision_v1",
            "evidence_ids": ["h1", "h2"],
            "experience_max_chars": 1600,
            "experience_aux_max_chars": 240,
        },
    )


def _experience_payload() -> str:
    return json.dumps(
        {
            "keep": False,
            "new_md": "# 经验\n保持简洁。",
            "calibration_note": "已校准",
            "self_check": "仅使用可见证据",
            "supporting_fact_ids": ["h1"],
            "contradicting_fact_ids": [],
            "noise_fact_ids": ["h2"],
        },
        ensure_ascii=False,
    )


def test_experience_revision_uses_independent_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAICompatibleClient(
        {
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_required": False,
            "max_retries": 0,
            "max_output_tokens": 256,
            "experience_max_output_tokens": 3072,
        }
    )
    calls: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> _CompletionResult:
        calls.append(kwargs)
        return _CompletionResult(_experience_payload(), "stop")

    monkeypatch.setattr(client, "_post", fake_post)
    payload = client.generate_structured(_experience_request(), dict)
    assert payload["new_md"] == "# 经验\n保持简洁。"
    assert calls[0]["max_tokens"] == 3072


def test_experience_revision_truncation_uses_compact_schema_aware_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAICompatibleClient(
        {
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_required": False,
            "max_retries": 0,
            "structured_output_mode": "json_schema",
            "experience_max_output_tokens": 3072,
            "experience_repair_max_output_tokens": 2048,
        }
    )
    completions = iter(
        [
            _CompletionResult('{"keep": false, "new_md": "未闭合', "length"),
            _CompletionResult(_experience_payload(), "stop"),
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> _CompletionResult:
        calls.append(kwargs)
        return next(completions)

    monkeypatch.setattr(client, "_post", fake_post)
    payload = client.generate_structured(_experience_request(), dict)
    assert payload["keep"] is False
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 3072
    assert calls[1]["max_tokens"] == 2048
    assert "经验 JSON 压缩修复器" in calls[1]["messages"][0]["content"]  # type: ignore[index]


def test_experience_revision_schema_bounds_long_text_and_evidence_arrays() -> None:
    response_format = _response_format("json_schema", _experience_request())
    assert response_format is not None
    schema = response_format["json_schema"]["schema"]  # type: ignore[index]
    properties = schema["properties"]  # type: ignore[index]
    assert properties["new_md"]["minLength"] == 1
    assert properties["new_md"]["maxLength"] == 1600
    assert properties["calibration_note"]["maxLength"] == 240
    assert properties["self_check"]["maxLength"] == 240
    assert properties["supporting_fact_ids"]["maxItems"] == 2


def test_online_compatible_provider_still_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证默认 OpenAI-compatible 配置仍要求 API key。
    参数：
        monkeypatch：pytest 环境变量夹具。
    返回：无。
    副作用：设置临时环境变量。
    异常：预期抛出 ProviderError。
    设计说明：本地宽松模式不能改变在线 API 的安全默认值。
    """

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleClient({"provider": "openai_compatible", "model": "online"})

    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        client.generate_structured(object(), ActionDecision)  # type: ignore[arg-type]


def test_json_schema_uses_current_legal_actions_and_raise_bounds() -> None:
    """结构化输出应在生成阶段约束动作 enum、置信度和 raise 区间。"""

    observation = AgentObservation(
        agent_id="agent_00",
        table_id="table",
        hand_id="hand",
        phase="preflop",
        seat=0,
        hole_cards=["As", "Ah"],
        community_cards=[],
        pot=3,
        current_bet=2,
        to_call=2,
        players=[PlayerPublicState("agent_00", 0, 100, 0, 0, False, False)],
        action_history=[],
        legal_actions=LegalActionSet(
            [LegalAction("fold"), LegalAction("call"), LegalAction("raise", 4, 100)]
        ),
        seed=7,
    )
    request = LLMRequest(
        observation=observation,
        memory_context=MemoryContext(),
        system_prompt="system",
        user_prompt="user",
    )
    response_format = _response_format("json_schema", request)
    assert response_format is not None
    schema = response_format["json_schema"]["schema"]  # type: ignore[index]
    properties = schema["properties"]  # type: ignore[index]
    assert properties["action_type"]["enum"] == ["fold", "call", "raise"]
    integer_amount = properties["amount"]["anyOf"][0]
    assert integer_amount["minimum"] == 4
    assert integer_amount["maximum"] == 100
    assert properties["confidence"]["minimum"] == 0.0
    assert properties["confidence"]["maximum"] == 1.0


def test_json_schema_uses_discrete_raise_amount_enum() -> None:
    observation = AgentObservation(
        agent_id="agent_00",
        table_id="table",
        hand_id="hand",
        phase="preflop",
        seat=0,
        hole_cards=["3s", "2s"],
        community_cards=[],
        pot=3,
        current_bet=2,
        to_call=2,
        players=[PlayerPublicState("agent_00", 0, 1000, 0, 0, False, False)],
        action_history=[],
        legal_actions=LegalActionSet(
            [LegalAction("fold"), LegalAction("call"), LegalAction("raise", 4, 1000)]
        ),
        seed=7,
    )
    request = LLMRequest(
        observation=observation,
        memory_context=MemoryContext(),
        system_prompt="system",
        user_prompt="user",
        metadata={
            "raise_sizing": {
                "policy": "local_discrete",
                "allowed_amounts": [4, 7],
            }
        },
    )

    response_format = _response_format("json_schema", request)
    assert response_format is not None
    schema = response_format["json_schema"]["schema"]  # type: ignore[index]
    integer_amount = schema["properties"]["amount"]["anyOf"][0]  # type: ignore[index]
    assert integer_amount == {"type": "integer", "enum": [4, 7]}
