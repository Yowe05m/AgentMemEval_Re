from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agentmemeval.config.loader import load_config, validate_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.campaign import (
    _read_campaign_yaml,
    _validate_campaign_spec,
)
from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT
from agentmemeval.storage.artifacts import collect_runtime_metadata


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


def test_validate_config_accepts_bgem3_native_hybrid_without_instruction() -> None:
    config = _valid_config()
    config["agent"] = {
        "embedding_backend": "bgem3_hybrid_http",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "fixed-revision",
        "embedding_weights_hash": "fixed-weights",
        "embedding_tokenizer_revision": "fixed-tokenizer",
        "embedding_base_url_env": "BGEM3_BASE_URL",
        "embedding_query_policy": "raw_symmetric_no_instruction",
        "embedding_hybrid_weights": [0.4, 0.2, 0.4],
        "embedding_candidate_depth": 1000,
        "embedding_colbert_rerank_depth": 1000,
        "embedding_final_top_k_policy": "agent_roster_top_k",
        "embedding_cache_schema_version": "bgem3_native_document_repr_v1",
        "embedding_cache_path": "outputs/cache/{agent_id}",
        "embedding_service_startup_parameters": {
            "model_path": "/model",
            "service_script": "/service.py",
            "python": "/python",
            "dtype": "float16",
            "normalize_embeddings": True,
            "query_max_length": 256,
            "passage_max_length": 1024,
            "cache_capacity": 4096,
            "cache_schema_version": "bgem3_native_document_repr_v1",
            "flagembedding_version": "1.4.0",
        },
    }
    validate_config(config)


def test_validate_config_rejects_qwen_instruction_for_bgem3_native_hybrid() -> None:
    config = load_config("configs/experiments/task4_real_pilot_base_bgem3_native_528.yaml")
    config["agent"]["embedding_query_instruction"] = "Instruct: Qwen-style retrieval"
    with pytest.raises(ConfigError, match="禁止 Qwen-style"):
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


def test_528_bgem3_v6_uses_native_hybrid_and_same_campaign_seeds() -> None:
    config = load_config(
        "configs/experiments/task4_campaign_p_pilot_bgem3_native_528.yaml"
    )
    agent = config["agent"]
    assert agent["embedding_backend"] == "bgem3_hybrid_http"
    assert agent["embedding_model"] == "BAAI/bge-m3"
    assert agent["embedding_query_policy"] == "raw_symmetric_no_instruction"
    assert "embedding_query_instruction" not in agent
    assert agent["embedding_hybrid_weights"] == [0.4, 0.2, 0.4]

    campaign = _read_campaign_yaml(
        Path("configs/campaigns/task4_campaign_p_pilot_parallel_v6_bgem3_native_528.yaml")
    )
    assert campaign["campaign"]["seeds"] == list(range(2026072101, 2026072109))


def test_528_bgem3_v7_mirrors_848_contract_and_freezes_preflight() -> None:
    config = load_config("configs/experiments/task6_campaign_p_v7_bgem3_native_528.yaml")
    agent = config["agent"]
    experiment = config["experiment"]
    provider = config["provider"]
    assert provider["model_tokenizer_revision"] == provider["model_revision"]
    assert provider["service_startup_parameters"]["gpu_memory_utilization"] == 0.62
    assert provider["service_startup_parameters"]["quantization"] is None
    assert provider["service_startup_parameters"]["vllm_use_flashinfer_sampler"] is False
    assert agent["embedding_hybrid_weights"] == [0.4, 0.2, 0.4]
    assert agent["embedding_candidate_depth"] == 1000
    assert agent["embedding_colbert_rerank_depth"] == 1000
    assert agent["embedding_final_top_k_policy"] == "agent_roster_top_k"
    assert agent["embedding_cache_schema_version"] == "bgem3_native_document_repr_v1"
    assert experiment["train_hands"] == 150
    assert experiment["checkpoint_test_hands"] == 50
    assert experiment["behavior_threshold_status"] == "frozen"
    assert experiment["behavior_thresholds"]["min_vpip"] == 0.02
    assert [item["agent_id"] for item in experiment["agent_roster"]] == [
        "fact_00",
        "fact_01",
        "expr_00",
        "expr_01",
        "sync_00",
        "sync_01",
        "async_00",
        "async_01",
    ]

    preflight = load_config(
        "configs/experiments/task6_campaign_p_v7_bgem3_preflight_528.yaml"
    )
    preflight_experiment = preflight["experiment"]
    assert preflight_experiment["seed"] == 2026072399
    assert preflight_experiment["run_mode"] == "smoke"
    assert preflight_experiment["run_id"] == "task6_bgem3_v7_preflight_s2026072399_a02"
    assert preflight_experiment["output_root"] == "outputs/preflights"
    assert preflight_experiment["train_hands"] == 20
    assert preflight_experiment["checkpoint_interval"] == 20
    assert preflight_experiment["checkpoint_test_hands"] == 5
    assert preflight_experiment["train_hands"] + (
        len(preflight_experiment["agent_roster"])
        * preflight_experiment["checkpoint_test_hands"]
    ) == 60
    assert preflight_experiment["not_for_analysis"] is True

    campaign = _read_campaign_yaml(
        Path("configs/campaigns/task4_campaign_p_pilot_parallel_v7_bgem3_native_528.yaml")
    )
    spec = campaign["campaign"]
    assert spec["seeds"] == list(range(2026072301, 2026072309))
    assert spec["max_parallel_runs"] == 4
    assert "counterfactual_calibrated" in spec["protocol_label"]
    assert "not_for_main_table" in spec["protocol_label"]


def test_528_bgem3_v7_prompt_identity_matches_848_true_leaf_manifest() -> None:
    assert PROMPT_TEMPLATE_VERSION == "2026-07-19-v6-counterfactual-calibrated-memory"
    assert hashlib.sha256(BASE_SYSTEM_PROMPT.encode("utf-8")).hexdigest() == (
        "9cd2f157225e14bfee9113c3af01a2ff4fff839aeb68dcfd8f11740bd8647800"
    )
    assert hashlib.sha256(EXPERIENCE_UPDATE_PROMPT.encode("utf-8")).hexdigest() == (
        "7788fa2f85adca9710cf20f2fc95769db1b2b93ee60f9a5236a430b87d4ad382"
    )


def test_528_bgem3_v7_manifest_metadata_seals_threshold_and_embedding_identity() -> None:
    config = load_config("configs/experiments/task6_campaign_p_v7_bgem3_native_528.yaml")
    metadata = collect_runtime_metadata(config, Path.cwd())
    assert metadata["protocol"]["behavior_threshold_status"] == "frozen"
    assert len(metadata["protocol"]["behavior_threshold_sha256"]) == 64
    assert metadata["model"]["tokenizer_revision"] == config["provider"]["model_revision"]
    assert metadata["embedding"]["tokenizer_revision"] == config["agent"][
        "embedding_revision"
    ]
    assert metadata["embedding"]["cache_schema_version"] == (
        "bgem3_native_document_repr_v1"
    )
    assert metadata["embedding"]["cache_namespace_template"].endswith("{agent_id}")


def test_task6_diff_register_has_no_unexplained_paths() -> None:
    register = yaml.safe_load(
        Path("configs/audits/task6_bgem3_v7_diff_register.yaml").read_text(
            encoding="utf-8"
        )
    )
    categories = register["categories"]
    classified = {
        path
        for name in (
            "bge_m3_required_changes",
            "v7_prompt_required_changes",
            "postprocessing_and_audit_changes",
        )
        for path in categories[name]["paths"]
    }
    expected = {
        "README.md",
        "configs/audits/task6_848_campaign_p_v7_reference_identity.json",
        "configs/audits/task6_bgem3_v7_diff_register.yaml",
        "configs/campaigns/task4_campaign_p_pilot_parallel_v6_bgem3_native_528.yaml",
        "configs/campaigns/task4_campaign_p_pilot_parallel_v7_bgem3_native_528.yaml",
        "configs/experiments/task4_campaign_p_pilot_bgem3_native_528.yaml",
        "configs/experiments/task4_real_pilot_base_bgem3_native_528.yaml",
        "configs/experiments/task6_campaign_p_v7_bgem3_native_528.yaml",
        "configs/experiments/task6_campaign_p_v7_bgem3_preflight_528.yaml",
        "configs/experiments/task6_real_pilot_base_v7_bgem3_native_528.yaml",
        "pyproject.toml",
        "src/agentmemeval/config/loader.py",
        "src/agentmemeval/experiments/admission.py",
        "src/agentmemeval/experiments/fixed_table.py",
        "src/agentmemeval/memory/bgem3_contract.py",
        "src/agentmemeval/memory/factual.py",
        "src/agentmemeval/memory/rag.py",
        "src/agentmemeval/storage/artifacts.py",
        "tests/unit/test_bgem3_contract.py",
        "tests/unit/test_config_validation.py",
        "tests/unit/test_rag_and_evaluator.py",
        "tools/bgem3_hybrid_server.py",
        "tools/task6/gate_bgem3_v7_preflight.py",
    }
    assert classified == expected
    assert categories["unexplained_changes"]["paths"] == []
    assert register["admission"]["unexplained_change_count"] == 0


def test_task6_848_reference_identity_is_sealed_and_matches_prompt_contract() -> None:
    reference = json.loads(
        Path(
            "configs/audits/task6_848_campaign_p_v7_reference_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert reference["status"] == "sealed_read_only_reference"
    assert reference["prompts"]["decision_version"] == PROMPT_TEMPLATE_VERSION
    assert reference["prompts"]["decision_system_sha256"] == hashlib.sha256(
        BASE_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    assert reference["prompts"]["experience_update_sha256"] == hashlib.sha256(
        EXPERIENCE_UPDATE_PROMPT.encode("utf-8")
    ).hexdigest()
    assert reference["protocol"]["seeds"] == list(range(2026072301, 2026072309))


def test_task4_target_scoped_pilot_campaigns_are_valid_and_seed_paired() -> None:
    root = Path(__file__).resolve().parents[2]
    campaign_paths = [
        root
        / "configs/campaigns/task4_campaign_p_pilot_parallel_v6_target_scoped.yaml",
        root
        / "configs/campaigns/task4_campaign_e_pilot_parallel_v6_target_scoped.yaml",
    ]
    seed_sets: list[list[int]] = []
    for path in campaign_paths:
        raw = _read_campaign_yaml(path)
        spec = raw["campaign"]
        base_path = (path.parent / str(spec["base_experiment_config"])).resolve()
        _validate_campaign_spec(spec, load_config(base_path))
        seed_sets.append([int(seed) for seed in spec["seeds"]])
        assert spec["max_parallel_runs"] == 4
        assert "not_for_main_table" in spec["protocol_label"]
    assert seed_sets[0] == seed_sets[1]
    assert len(seed_sets[0]) == 8


def test_task4_counterfactual_v7_pilot_campaigns_are_valid_and_seed_paired() -> None:
    root = Path(__file__).resolve().parents[2]
    campaign_paths = [
        root
        / "configs/campaigns/"
        "task4_campaign_p_pilot_parallel_v7_counterfactual_calibrated.yaml",
        root
        / "configs/campaigns/"
        "task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml",
    ]
    seed_sets: list[list[int]] = []
    for path in campaign_paths:
        raw = _read_campaign_yaml(path)
        spec = raw["campaign"]
        base_path = (path.parent / str(spec["base_experiment_config"])).resolve()
        _validate_campaign_spec(spec, load_config(base_path))
        seed_sets.append([int(seed) for seed in spec["seeds"]])
        assert spec["max_parallel_runs"] == 4
        assert "counterfactual_calibrated" in spec["protocol_label"]
        assert "not_for_main_table" in spec["protocol_label"]
        if spec["design"] == "target_vs_seven_no_memory":
            assert spec["matrix_order"] == "seed_major"
        else:
            assert spec.get("matrix_order", "condition_major") == "condition_major"
    assert seed_sets[0] == seed_sets[1]
    assert seed_sets[0] == list(range(2026072301, 2026072309))


def test_task4_memory_debias_smoke_is_real_roster_and_not_for_paper() -> None:
    config = load_config(
        "configs/experiments/task4_campaign_p_memory_debias_smoke.yaml"
    )
    experiment = config["experiment"]
    assert experiment["run_mode"] == "pilot"
    assert "not_for_main_table" in experiment["protocol_label"]
    assert experiment["seed"] == 2026071899
    assert experiment["train_hands"] == 30
    assert experiment["checkpoint_test_hands"] == 10
    assert len(experiment["agent_roster"]) == 8
    assert config["agent"]["embedding_backend"] == "openai_compatible"
    assert config["agent"]["reject_single_preflop_fold"] is True


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
