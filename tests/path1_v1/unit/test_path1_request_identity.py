from __future__ import annotations

import copy
import json
from dataclasses import replace
from importlib.resources import files

import pytest

from agentmemeval.path1_v1.core.domain import ActionDecision
from agentmemeval.path1_v1.llm import request_identity as identity
from agentmemeval.path1_v1.llm.terminal_policy import Path1TerminalError
from agentmemeval.path1_v1.memory.experiential import ExperientialMemory
from agentmemeval.path1_v1.operations.worker import StrictPath1ClientFactory


def source():
    return json.loads(
        (
            files("agentmemeval.path1_v1") / "assets/path1_source_contract.json"
        ).read_text(encoding="utf-8")
    )


def context(**overrides):
    return identity.EpisodeRequestContext(
        **{
            "campaign_id": "fixture",
            "phase": "E1",
            "artifact_id": "artifact_0001",
            "seed_namespace": "isolated",
            "root_seed": 1729,
            "stage": "formation",
            "episode_index": 0,
            "arm": "formation",
            "actor_id": "agent_00",
            "client_seed": 811,
            **overrides,
        }
    )


def request(ctx):
    from path1_v1.unit.test_path1_terminal_policy import request as original

    req = original()
    req.observation.agent_id = ctx.actor_id
    req.observation.players[0].agent_id = ctx.actor_id
    req.observation.table_id = f"{ctx.artifact_id}_{ctx.stage}_{ctx.episode_index:04d}"
    req.observation.hand_id = req.observation.table_id + "_0001"
    return req


def response(content):
    from path1_v1.unit.test_path1_terminal_policy import response as envelope

    return envelope(content, model="qwen/qwen3.5-9b")


def test_actual_http_body_seed_role_index_temperature_and_repair_budget(tmp_path, monkeypatch):
    from path1_v1.unit.test_path1_terminal_policy import action, mock_http

    monkeypatch.setenv("PATH1_MODEL_BASE_URL", "http://unused.example/v1")
    factory = StrictPath1ClientFactory(source())
    ctx = context()
    client = factory.create_episode_client(ctx, tmp_path / "http.jsonl")
    calls = mock_http(monkeypatch, [response(action("")), response(action()), response(action())])
    client.generate_structured(request(ctx), ActionDecision)
    client.generate_structured(request(ctx), ActionDecision)
    records = [json.loads(line) for line in (tmp_path / "http.jsonl").read_text().splitlines()]
    starts = [row for row in records if row["event"] == "request"]
    assert len(starts) == len(calls) == 3
    ids = [row["path1_request_identity"] for row in starts]
    assert [(row["logical_index"], row["request_index"]) for row in ids] == [(0, 0), (0, 1), (1, 0)]
    assert {row["role"] for row in ids} == {"hero_action"}
    assert ids[0]["logical_request_id"] == ids[1]["logical_request_id"]
    assert ids[0]["logical_request_id"] != ids[2]["logical_request_id"]
    assert len({call["seed"] for call in calls}) == 3
    for call, row, start in zip(calls, ids, starts, strict=True):
        assert call["seed"] == row["sampler_seed"]
        assert 0 <= call["seed"] < 2**63
        assert call["user"] == row["request_id"]
        assert call["temperature"] == 0.0 and call["max_tokens"] == 256
        assert call["chat_template_kwargs"] == {"enable_thinking": False}
        assert start["request_payload"] == call
        assert row["backend_seed_execution_verified"] is False
        assert {key: row[key] for key in ("episode_index", "actor_id", "arm")} == {
            "episode_index": 0,
            "actor_id": "agent_00",
            "arm": "formation",
        }


def test_memory_repair_uses_revision_seed_clock_and_contract_output_budgets(tmp_path, monkeypatch):
    from path1_v1.unit.test_path1_terminal_policy import (
        action,
        invalid_patch,
        keep,
        mock_http,
        trajectory,
    )

    monkeypatch.setenv("PATH1_MODEL_BASE_URL", "http://unused.example/v1")
    factory = StrictPath1ClientFactory(source())
    ctx = context()
    client = factory.create_episode_client(ctx, tmp_path / "http.jsonl")
    calls = mock_http(
        monkeypatch,
        [response(action()), response(invalid_patch()), response(keep()), response(action())],
    )
    client.generate_structured(request(ctx), ActionDecision)
    mem = ExperientialMemory(
        "agent_00",
        revision_strategy="llm",
        llm_client=client,
        model="qwen/qwen3.5-9b",
        prompt_token_budget=65536,
    )
    trace = trajectory()
    trace.agent_id = "agent_00"
    trace.decision_events[0].observation = request(ctx).observation
    revision = mem._revise_from_window([trace])
    client.generate_structured(request(ctx), ActionDecision)
    assert revision["terminal_policy_receipt"]["stages"] == ["primary", "memory_contract_repair"]
    assert [(call["temperature"], call["max_tokens"]) for call in calls] == [
        (0.0, 256),
        (1.0, 8192),
        (1.0, 4096),
        (0.0, 256),
    ]
    rows = factory.bindings[0].records
    assert [(row["role"], row["logical_index"], row["request_index"]) for row in rows] == [
        ("hero_action", 0, 0),
        ("hero_revision", 0, 0),
        ("hero_revision", 0, 1),
        ("hero_action", 1, 0),
    ]
    assert len({call["seed"] for call in calls}) == 4


def test_episode_actor_arm_and_artifact_seeds_are_order_independent(tmp_path, monkeypatch):
    from path1_v1.unit.test_path1_terminal_policy import action, mock_http

    monkeypatch.setenv("PATH1_MODEL_BASE_URL", "http://unused.example/v1")
    contexts = [
        context(
            stage="evaluation", arm=arm, actor_id=actor, episode_index=episode, artifact_id=artifact
        )
        for arm, actor, episode, artifact in [
            ("retained", "agent_00", 0, "artifact_0001"),
            ("removed", "agent_00", 0, "artifact_0001"),
            ("retained", "opponent_01", 0, "artifact_0001"),
            ("retained", "agent_00", 1, "artifact_0001"),
            ("retained", "agent_00", 0, "artifact_0002"),
        ]
    ]
    observed = []
    for run, order in enumerate((contexts, list(reversed(contexts)))):
        factory = StrictPath1ClientFactory(source())
        mock_http(monkeypatch, [response(action()) for _ in contexts])
        mapping = {}
        for index, ctx in enumerate(order):
            client = factory.create_episode_client(ctx, tmp_path / f"r{run}_{index}.jsonl")
            client.generate_structured(request(ctx), ActionDecision)
            row = factory.bindings[-1].records[0]
            mapping[row["request_id"]] = row["sampler_seed"]
        observed.append(mapping)
    assert observed[0] == observed[1]
    assert len(set(observed[0].values())) == len(contexts)


def test_sampler_collision_and_forbidden_seed_never_redraw(monkeypatch):
    ledger = identity.SamplerSeedLedger()
    first = ledger.derive({"label": "a"})
    assert first == ledger.derive({"label": "a"})
    with pytest.raises(Path1TerminalError, match="collision_no_redraw"):
        identity.SamplerSeedLedger(forbidden_seeds={first}).derive({"label": "a"})
    monkeypatch.setattr(identity, "_sha", lambda value: "a" * 64)
    ledger = identity.SamplerSeedLedger()
    ledger.derive({"label": "a"})
    with pytest.raises(Path1TerminalError, match="collision_no_redraw"):
        ledger.derive({"label": "b"})
    assert len(ledger.entries) == 1


@pytest.mark.parametrize("problem", ["actor", "episode", "metadata", "seed", "updater"])
def test_invalid_request_identity_never_reaches_http(tmp_path, monkeypatch, problem):
    from path1_v1.unit.test_path1_terminal_policy import action, mock_http

    monkeypatch.setenv("PATH1_MODEL_BASE_URL", "http://unused.example/v1")
    ctx = context(stage="evaluation", arm="retained") if problem == "updater" else context()
    factory = StrictPath1ClientFactory(source())
    client = factory.create_episode_client(ctx, tmp_path / "http.jsonl")
    req = request(ctx)
    schema = ActionDecision
    if problem == "actor":
        req.observation.agent_id = "unbound_actor"
    elif problem == "episode":
        req.observation.table_id = "another_episode"
    elif problem == "metadata":
        req.metadata["sampler_seed"] = 123
    elif problem == "seed":
        req.provider_config["seed"] = 123
    else:
        schema = dict
        req.metadata["response_schema"] = "experience_revision_v2"
    calls = mock_http(monkeypatch, [response(action())])
    with pytest.raises(Path1TerminalError):
        client.generate_structured(req, schema)
    assert not calls


def test_unbound_or_wrong_episode_capability_is_rejected(tmp_path):
    ctx = context()
    factory = StrictPath1ClientFactory(source())
    client = factory.create_episode_client(ctx, tmp_path / "http.jsonl")
    identity.require_path1_request_identity(client, ctx)
    with pytest.raises(Path1TerminalError, match="context_mismatch"):
        identity.require_path1_request_identity(client, replace(ctx, episode_index=2))
    with pytest.raises(Path1TerminalError, match="rebind"):
        client.bind_request_identity(identity.RequestIdentityBinding(ctx, factory.ledger))
    with pytest.raises(Path1TerminalError, match="duplicate_episode_actor_client"):
        factory.create_episode_client(ctx, tmp_path / "duplicate.jsonl")


def test_source_factory_rejects_unreviewed_parameter_drift():
    changed = copy.deepcopy(source())
    changed["provider"]["temperature"] = 1.0
    with pytest.raises(ValueError, match="unchanged source contract"):
        StrictPath1ClientFactory(changed)


def test_runtime_gate_is_rechecked_before_each_actual_http_request(tmp_path, monkeypatch):
    from path1_v1.unit.test_path1_terminal_policy import action, mock_http

    monkeypatch.setenv("PATH1_MODEL_BASE_URL", "http://unused.example/v1")
    checks = []

    def live_gate():
        checks.append(1)
        if len(checks) == 2:
            raise Path1TerminalError("state", "runtime_deadline_expired")

    factory = StrictPath1ClientFactory(source(), before_request=live_gate)
    ctx = context()
    client = factory.create_episode_client(ctx, tmp_path / "http.jsonl")
    calls = mock_http(monkeypatch, [response(action("")), response(action())])
    with pytest.raises(Path1TerminalError, match="runtime_deadline_expired"):
        client.generate_structured(request(ctx), ActionDecision)
    assert len(checks) == 2 and len(calls) == 1
    assert len(factory.bindings[0].records) == 1
