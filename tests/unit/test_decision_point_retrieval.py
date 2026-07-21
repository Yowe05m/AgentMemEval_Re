from __future__ import annotations

import pytest

from agentmemeval.config.loader import ConfigError, load_config, validate_config
from agentmemeval.core.domain import FactualMemoryRecord
from agentmemeval.memory.factual import FactualMemory
from agentmemeval.memory.rag import (
    SemanticScore,
    build_retrieval_query,
    hybrid_top_k_records,
)
from agentmemeval.memory.retrievers import observation_features
from tests.unit.test_memory import make_observation, make_trajectory


class ExactTextBackend:
    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]:
        return [
            SemanticScore(float(document == query), float(document == query), None, None)
            for document in documents
        ]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("not used")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("not used")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("not used")

    def audit_metadata(self) -> dict[str, object]:
        return {"backend": "exact_text_test"}


def _record(
    record_id: str,
    *,
    terminal_text: str,
    decision_query: str,
    decision_features: list[str],
) -> FactualMemoryRecord:
    return FactualMemoryRecord(
        record_id=record_id,
        agent_id="agent_00",
        table_id="table_a",
        hand_id=record_id,
        scope="per_agent",
        state_summary=terminal_text,
        action_summary="call",
        final_reward=1,
        features=list(decision_features),
        source={
            "fact_text": terminal_text,
            "decisions": [
                {
                    "phase": "preflop",
                    "retrieval_query": decision_query,
                    "features": list(decision_features),
                }
            ],
        },
    )


def test_decision_point_unit_ranks_matching_decision_not_terminal_text() -> None:
    observation = make_observation()
    query = build_retrieval_query(observation)
    features = observation_features(observation)
    decision_match = _record(
        "decision_match",
        terminal_text="unrelated terminal narrative",
        decision_query=query,
        decision_features=features,
    )
    terminal_match = _record(
        "terminal_match",
        terminal_text=query,
        decision_query="phase=river hole=['2c', '7d'] board=['As'] pot=99 to_call=20",
        decision_features=["phase:river", "players:2", "pot:large", "to_call:large"],
    )
    backend = ExactTextBackend()

    terminal_scores = hybrid_top_k_records(
        observation,
        [decision_match, terminal_match],
        k=2,
        semantic_weight=1.0,
        feature_weight=0.0,
        embedding_backend=backend,
        retrieval_unit="hand_terminal_v1",
    )
    decision_scores = hybrid_top_k_records(
        observation,
        [decision_match, terminal_match],
        k=2,
        semantic_weight=1.0,
        feature_weight=0.0,
        embedding_backend=backend,
        retrieval_unit="decision_point_max_v1",
    )

    assert terminal_scores[0].record.record_id == "terminal_match"
    assert decision_scores[0].record.record_id == "decision_match"
    assert decision_scores[0].retrieval_unit == "decision_point_max_v1"
    assert decision_scores[0].matched_decision_index == 0
    assert decision_scores[0].matched_phase == "preflop"


def test_factual_memory_persists_versioned_decision_views() -> None:
    memory = FactualMemory("agent_00", retrieval_unit="decision_point_max_v1")

    memory.on_hand_finished(make_trajectory())

    decision = memory.records[0].source["decisions"][0]
    assert decision["retrieval_query"] == build_retrieval_query(make_observation())
    assert decision["features"] == observation_features(make_observation())
    snapshot = memory.snapshot()
    assert snapshot.payload["schema_version"] == 6
    assert snapshot.payload["retrieval_unit"] == "decision_point_max_v1"
    restored = FactualMemory("agent_00")
    restored.restore(snapshot)
    assert restored.retrieval_unit == "decision_point_max_v1"


def test_unknown_retrieval_unit_fails_closed() -> None:
    with pytest.raises(ValueError, match="retrieval_unit"):
        FactualMemory("agent_00", retrieval_unit="post_hoc_best")


def test_config_rejects_unknown_retrieval_unit() -> None:
    config = load_config("configs/experiments/task4_campaign_p_smoke_v8_decision_point.yaml")
    config["agent"]["retrieval_unit"] = "post_hoc_best"

    with pytest.raises(ConfigError, match="retrieval_unit"):
        validate_config(config)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/experiments/task4_campaign_p_pilot_v8_decision_point.yaml",
        "configs/experiments/task4_campaign_e_pilot_v8_decision_point.yaml",
        "configs/experiments/task4_campaign_p_smoke_v8_decision_point.yaml",
    ],
)
def test_v8_decision_point_configs_resolve_and_validate(config_path: str) -> None:
    config = load_config(config_path)

    assert config["agent"]["retrieval_unit"] == "decision_point_max_v1"


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/experiments/task4_campaign_p_pilot_v8_decision_point.yaml",
        "configs/experiments/task4_campaign_e_pilot_v8_decision_point.yaml",
    ],
)
def test_expedited_v8_pilot_scale_is_explicit(config_path: str) -> None:
    experiment = load_config(config_path)["experiment"]

    assert experiment["train_hands"] == 60
    assert experiment["test_hands"] == 20
    assert experiment["checkpoint_interval"] == 60
    assert experiment["checkpoint_test_hands"] == 20
    assert experiment["required_seed_pairs"] == 2
