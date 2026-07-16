from __future__ import annotations

import pytest

from agentmemeval.config.loader import load_config, validate_config
from agentmemeval.core.errors import ConfigError


def _valid_config() -> dict[str, object]:
    return {
        "experiment": {"scenario": "fixed_evolving_table", "seed": 1, "table_size": 4},
        "provider": {"name": "mock"},
        "table": {"small_blind": 1, "big_blind": 2, "starting_stack": 100},
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("experiment", "table_size", 1),
        ("experiment", "train_hands", -1),
        ("table", "starting_stack", 1),
        ("table", "lifecycle", "unknown"),
    ],
)
def test_validate_config_rejects_invalid_ranges(
    section: str,
    field: str,
    value: object,
) -> None:
    config = _valid_config()
    config[section][field] = value  # type: ignore[index]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_accepts_minimal_valid_config() -> None:
    validate_config(_valid_config())


def test_validate_config_rejects_strategy_risk_gate() -> None:
    config = _valid_config()
    config["agent"] = {"strategy_risk_gate": "high_cost_fold"}
    with pytest.raises(ConfigError, match="策略风险门控"):
        validate_config(config)


def test_validate_config_requires_versioned_instructed_semantic_embedding() -> None:
    config = _valid_config()
    config["agent"] = {
        "embedding_backend": "openai_compatible",
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "embedding_revision": "fixed-revision",
    }
    with pytest.raises(ConfigError, match="embedding_query_instruction"):
        validate_config(config)


def test_paper_main_config_contains_reviewed_protocol_choices() -> None:
    config = load_config("configs/experiments/paper_exp1_mixed_local.yaml")
    agent = config["agent"]
    experiment = config["experiment"]

    assert agent["embedding_model"] == "Qwen/Qwen3-Embedding-4B"
    assert agent["embedding_revision"] == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert agent["strategy_risk_gate"] == "disabled"
    assert experiment["checkpoint_test_hands"] == 50
    assert experiment["run_mode"] == "pilot"
    assert agent["retrieval_threshold_status"] == "pending_pilot"
    assert experiment["statistical_plan_status"] == "pending_pilot_power_calibration"
    assert experiment["primary_estimand"] == (
        "same_seed_table_run_mechanism_effect_vs_baseline"
    )
    assert experiment["primary_baseline_mechanism"] == "fact"
    assert experiment["multiple_comparison_method"] == "holm"


def test_task4_real_pilot_has_independent_experience_revision_budget() -> None:
    config = load_config("configs/experiments/task4_real_pilot_base.yaml")
    assert config["provider"]["max_output_tokens"] == 2048
    assert config["provider"]["experience_max_output_tokens"] == 3072
    assert config["provider"]["experience_repair_max_output_tokens"] == 2048
    assert config["provider"]["service_startup_parameters"]["max_model_len"] == 16384


def test_validate_config_rejects_incomplete_A7_R_preregistration() -> None:
    config = _valid_config()
    config["experiment"].update(  # type: ignore[union-attr]
        {
            "primary_estimand": "same_seed_table_run_mechanism_effect_vs_baseline",
            "primary_endpoint": "final_test_bb_per_100",
        }
    )
    with pytest.raises(ConfigError, match="primary_baseline_mechanism"):
        validate_config(config)


def test_validate_config_rejects_unimplemented_shared_memory_scope() -> None:
    config = _valid_config()
    config["agent"] = {"memory_scope": "global"}
    with pytest.raises(ConfigError, match="共享记忆尚未实现"):
        validate_config(config)


def test_validate_config_rejects_persona_outside_smoke() -> None:
    config = _valid_config()
    config["experiment"]["run_mode"] = "pilot"  # type: ignore[index]
    config["agent"] = {"persona": "INTJ"}
    with pytest.raises(ConfigError, match="Exp2 人格机制已延期"):
        validate_config(config)


def test_frozen_retrieval_threshold_requires_numeric_value() -> None:
    config = _valid_config()
    config["agent"] = {"retrieval_threshold_status": "frozen"}
    with pytest.raises(ConfigError, match="minimum_retrieval_score"):
        validate_config(config)
