"""
模块说明：本模块负责 YAML 配置加载、继承合并和基础校验。
核心职责：把实验配置解析为单一 resolved config，供运行器快照保存。
输入与输出：输入配置路径，输出合并后的字典。
依赖边界：依赖 PyYAML 与标准库 pathlib，不依赖实验模块。
不负责：不创建 Provider，不运行实验。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.core.bgem3_contract import (
    BGE_M3_DENSE_NORM_TOLERANCE,
    BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    BGE_M3_NORMALIZATION_IDENTITY,
    BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION,
)
from agentmemeval.core.errors import ConfigError
from agentmemeval.core.memory_mechanism import (
    EXPERIENCE_MECHANISMS,
    FACTUAL_MECHANISMS,
    resolve_memory_mechanism,
)
from agentmemeval.core.model_identity import canonical_model_family

_EXPERIENCE_SEMANTIC_REVIEW_MODES = frozenset(
    {"disabled", "independent_model_v1"}
)


def load_config(path: str | Path) -> dict[str, Any]:
    """
    功能：加载 YAML 配置并处理 extends。
    参数：
        path：配置文件路径。
    返回：合并后的配置字典。
    副作用：读取文件。
    异常：文件不存在或 YAML 结构非法时抛出 ConfigError。
    设计说明：配置集中在 YAML，避免实验参数散落在 Python 常量中。
    """

    config = load_raw_config(path)
    validate_config(config)
    return config


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """
    功能：加载 YAML 配置并处理 extends，但不做实验字段校验。
    参数：
        path：配置文件路径。
    返回：合并后的原始配置字典。
    副作用：读取文件。
    异常：文件不存在或 YAML 结构非法时抛出 ConfigError。
    设计说明：doctor 需要读取 provider-only 配置；run 仍使用 load_config 做完整校验。
    """

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在：{config_path}")
    config = _read_yaml(config_path)
    parent_name = config.pop("extends", None)
    if parent_name:
        parent_path = (config_path.parent / str(parent_name)).resolve()
        parent = load_raw_config(parent_path)
        config = deep_merge(parent, config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    功能：校验运行所需的关键配置字段。
    参数：
        config：配置字典。
    返回：无。
    副作用：无。
    异常：缺字段时抛出 ConfigError。
    设计说明：尽早失败，避免实验跑到一半才发现缺少 seed 或 scenario。
    """

    if "experiment" not in config:
        raise ConfigError("配置缺少 experiment 段")
    if "provider" not in config:
        raise ConfigError("配置缺少 provider 段")
    experiment = config["experiment"]
    provider = config["provider"]
    if not isinstance(experiment, dict):
        raise ConfigError("experiment 必须是映射")
    if not isinstance(provider, dict):
        raise ConfigError("provider 必须是映射")
    thinking_mode = str(provider.get("thinking_mode", "")).strip()
    if thinking_mode not in {"", "enabled", "disabled"}:
        raise ConfigError("provider.thinking_mode 必须为 enabled 或 disabled")
    for output_mode_field in (
        "structured_output_mode",
        "experience_structured_output_mode",
    ):
        output_mode = str(provider.get(output_mode_field, "json_object")).strip()
        if output_mode not in {
            "json_object",
            "json_schema",
            "none",
            "prompt_only",
            "disabled",
        }:
            raise ConfigError(f"provider.{output_mode_field} 不是受支持的输出模式")
    if "scenario" not in experiment:
        raise ConfigError("experiment.scenario 不能为空")
    if "seed" not in experiment:
        raise ConfigError("experiment.seed 不能为空")
    if not str(experiment["scenario"]).strip():
        raise ConfigError("experiment.scenario 不能为空字符串")
    try:
        int(experiment["seed"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("experiment.seed 必须是整数") from exc
    for field in ("train_hands", "test_hands", "checkpoint_test_hands"):
        if field in experiment and int(experiment[field]) < 0:
            raise ConfigError(f"experiment.{field} 不能为负数")
    _validate_checkpoint_protocol(experiment)
    _validate_heldout_table_set(experiment)
    if "table_size" in experiment and int(experiment["table_size"]) < 2:
        raise ConfigError("experiment.table_size 必须至少为 2")
    table = config.get("table", {})
    if not isinstance(table, dict):
        raise ConfigError("table 必须是映射")
    small_blind = int(table.get("small_blind", 1))
    big_blind = int(table.get("big_blind", 2))
    starting_stack = int(table.get("starting_stack", 1000))
    if small_blind <= 0 or big_blind <= small_blind:
        raise ConfigError("盲注必须满足 0 < small_blind < big_blind")
    if starting_stack < big_blind:
        raise ConfigError("table.starting_stack 不能小于 big_blind")
    lifecycle = str(table.get("lifecycle", "tournament_elimination"))
    if lifecycle not in {"tournament_elimination", "continuous_rebuy"}:
        raise ConfigError(f"未知 table.lifecycle：{lifecycle}")
    agent_sections = ("agent", "opponent_agent", "heldout_agent")
    run_mode = str(experiment.get("run_mode", "smoke"))
    if run_mode not in {"smoke", "pilot", "formal"}:
        raise ConfigError(f"未知 experiment.run_mode：{run_mode}")
    for section in agent_sections:
        section_config = config.get(section, {})
        if not isinstance(section_config, dict):
            raise ConfigError(f"{section} 必须是映射")
        strategy_risk_gate = str(section_config.get("strategy_risk_gate", "disabled"))
        if strategy_risk_gate != "disabled":
            raise ConfigError(
                f"主实现不支持策略风险门控；{section}.strategy_risk_gate 必须为 disabled"
            )
        memory_scope = str(section_config.get("memory_scope", "per_agent"))
        if memory_scope != "per_agent":
            raise ConfigError(f"共享记忆尚未实现；{section}.memory_scope 必须为 per_agent")
        _validate_memory_contract(section, section_config, run_mode)
        if section_config.get("persona") and run_mode != "smoke":
            raise ConfigError("Exp2 人格机制已延期；persona 配置只能用于 not_for_paper smoke")
    agent = config.get("agent", {})
    if _resolved_config_mechanism("agent", agent) in FACTUAL_MECHANISMS:
        _validate_embedding_retrieval_contract("agent", agent)
    for section, effective_config in _effective_secondary_agent_configs(config):
        if _resolved_config_mechanism(section, effective_config) in FACTUAL_MECHANISMS:
            _validate_embedding_retrieval_contract(section, effective_config)
        if section.startswith("experiment.heldout_table_rosters."):
            _validate_memory_contract(section, effective_config, run_mode)
            if str(effective_config.get("memory_scope", "per_agent")) != "per_agent":
                raise ConfigError(f"{section}.memory_scope 必须为 per_agent")
            if str(effective_config.get("strategy_risk_gate", "disabled")) != "disabled":
                raise ConfigError(f"{section}.strategy_risk_gate 必须为 disabled")
            if effective_config.get("persona") and run_mode != "smoke":
                raise ConfigError("Exp2 人格机制已延期；heldout persona 只能用于 smoke")
    primary_estimand = str(experiment.get("primary_estimand", "") or "")
    if primary_estimand:
        supported_estimands = {
            "same_seed_table_run_mechanism_effect_vs_baseline",
            "same_seed_cross_condition_target_effect_vs_no_memory",
        }
        if primary_estimand not in supported_estimands:
            raise ConfigError(f"未知 experiment.primary_estimand：{primary_estimand}")
        if str(experiment.get("primary_endpoint", "")) not in {
            "final_test_bb_per_100",
            "final_test_chip_per_hand",
        }:
            raise ConfigError("A7-R 需要受支持的 experiment.primary_endpoint")
        if not str(experiment.get("primary_baseline_mechanism", "")).strip():
            raise ConfigError("A7-R 需要 experiment.primary_baseline_mechanism")
        if primary_estimand == "same_seed_table_run_mechanism_effect_vs_baseline":
            if str(experiment.get("within_table_mechanism_aggregation", "")) != ("arithmetic_mean"):
                raise ConfigError("A7-R 桌内同机制 Agent 必须预注册 arithmetic_mean 聚合")
        else:
            if str(experiment.get("within_table_mechanism_aggregation", "")) != (
                "single_target_condition"
            ):
                raise ConfigError("Campaign E 每个条件必须只贡献一个 target 统计单位")
            if str(experiment.get("cross_condition_aggregation", "")) != ("paired_by_seed"):
                raise ConfigError("Campaign E 必须按 seed 跨条件配对")
            if str(experiment.get("primary_baseline_mechanism", "")) != "no_memory":
                raise ConfigError("Campaign E 的预注册基线必须为 no_memory")
            target_id = str(experiment.get("target_agent_id", ""))
            targets = [str(item) for item in experiment.get("evaluation_target_ids", [])]
            if not target_id or targets != [target_id]:
                raise ConfigError("Campaign E 必须仅评估明确的单一 target_agent_id")
        if str(experiment.get("multiple_comparison_method", "")) != "holm":
            raise ConfigError("当前 B1-R 实现要求 experiment.multiple_comparison_method=holm")
        statistical_status = str(experiment.get("statistical_plan_status", ""))
        required_seed_pairs = experiment.get("required_seed_pairs")
        if statistical_status == "frozen" and (
            required_seed_pairs is None or int(required_seed_pairs) < 2
        ):
            raise ConfigError("冻结统计计划需要至少 2 个 required_seed_pairs")
    roster = experiment.get("agent_roster", [])
    if isinstance(roster, list):
        roster_mechanisms = {
            str(item.get("mechanism", "")) for item in roster if isinstance(item, dict)
        }
        baseline = str(experiment.get("primary_baseline_mechanism", ""))
        if (
            primary_estimand == "same_seed_table_run_mechanism_effect_vs_baseline"
            and baseline not in roster_mechanisms
        ):
            raise ConfigError("experiment.primary_baseline_mechanism 必须存在于 agent_roster")
        for item in roster:
            gate = item.get("strategy_risk_gate", "disabled") if isinstance(item, dict) else None
            if gate is not None and str(gate) != "disabled":
                raise ConfigError("experiment.agent_roster 不得启用策略风险门控")
            if isinstance(item, dict):
                scope = str(item.get("memory_scope", agent.get("memory_scope", "per_agent")))
                if scope != "per_agent":
                    raise ConfigError("experiment.agent_roster 仅支持 per_agent memory_scope")
                nested_override = item.get("config", {})
                if nested_override is not None and not isinstance(nested_override, dict):
                    raise ConfigError("experiment.agent_roster.config 必须是映射")
                merged_agent_config = {
                    **agent,
                    **dict(nested_override or {}),
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {"agent_id", "config"}
                    },
                }
                _validate_memory_contract(
                    "experiment.agent_roster",
                    merged_agent_config,
                    run_mode,
                )
                if item.get("persona") and run_mode != "smoke":
                    raise ConfigError("Exp2 人格机制已延期；roster persona 只能用于 smoke")
    _validate_experience_reviewer_contract(config)


def _effective_secondary_agent_configs(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve every non-primary agent config exactly as the runtime consumes it."""

    resolved: list[tuple[str, dict[str, Any]]] = []
    for section in ("opponent_agent", "heldout_agent"):
        raw = config.get(section)
        if isinstance(raw, dict):
            resolved.append((section, dict(raw)))

    base_agent = dict(config.get("agent", {}))
    experiment = dict(config.get("experiment", {}))
    roster = experiment.get("agent_roster")
    if isinstance(roster, list):
        for index, raw_item in enumerate(roster):
            if not isinstance(raw_item, dict):
                continue
            nested = raw_item.get("config", {})
            if nested is not None and not isinstance(nested, dict):
                continue
            override = dict(nested or {})
            override.update(
                {key: value for key, value in raw_item.items() if key not in {"agent_id", "config"}}
            )
            agent_id = str(raw_item.get("agent_id", index))
            resolved.append(
                (
                    f"experiment.agent_roster[{agent_id}]",
                    {**base_agent, **override},
                )
            )

    heldout_rosters = experiment.get("heldout_table_rosters")
    if isinstance(heldout_rosters, dict):
        for table_id, raw_roster in sorted(heldout_rosters.items()):
            if isinstance(raw_roster, dict):
                resolved.append(
                    (
                        f"experiment.heldout_table_rosters.{table_id}",
                        dict(raw_roster),
                    )
                )
    return resolved


def _resolved_config_mechanism(section: str, config: dict[str, Any]) -> str:
    try:
        return resolve_memory_mechanism(config)
    except ValueError as exc:
        raise ConfigError(f"{section}: {exc}") from exc


def _validate_experience_reviewer_contract(config: dict[str, Any]) -> None:
    """Require an actually independent reviewer for every enabled experience path."""

    agent_configs = [
        ("agent", dict(config.get("agent", {}))),
        *_effective_secondary_agent_configs(config),
    ]
    requiring_sections = [
        section
        for section, agent_config in agent_configs
        if _resolved_config_mechanism(section, agent_config) in EXPERIENCE_MECHANISMS
        and str(
            agent_config.get("experience_semantic_review_mode", "disabled")
        ).strip()
        == "independent_model_v1"
    ]
    if not requiring_sections:
        return

    reviewer_provider = config.get("experience_reviewer_provider")
    if not isinstance(reviewer_provider, dict):
        sections = ", ".join(requiring_sections)
        raise ConfigError(
            "experience_semantic_review_mode=independent_model_v1 要求顶层 "
            f"experience_reviewer_provider 映射；启用位置：{sections}"
        )
    main_provider = config["provider"]
    main_model = main_provider.get("model")
    if not isinstance(main_model, str) or not main_model.strip():
        raise ConfigError(
            "experience_semantic_review_mode=independent_model_v1 要求 "
            "provider.model 为显式非空字符串"
        )
    reviewer_model = reviewer_provider.get("model")
    if not isinstance(reviewer_model, str) or not reviewer_model.strip():
        raise ConfigError(
            "experience_semantic_review_mode=independent_model_v1 要求 "
            "experience_reviewer_provider.model 为显式非空字符串"
        )
    main_family = canonical_model_family(main_model)
    reviewer_family = canonical_model_family(reviewer_model)
    if main_family == reviewer_family:
        raise ConfigError(
            "experience_reviewer_provider.model 必须与 provider.model 属于不同模型族；"
            f"规范化后均为 {main_family}"
        )


def _validate_embedding_retrieval_contract(
    section: str,
    agent: dict[str, Any],
) -> None:
    """Apply the same retrieval/backend contract to every effective agent config."""

    retrieval_unit = str(agent.get("retrieval_unit", "hand_terminal_v1"))
    if retrieval_unit not in {"hand_terminal_v1", "decision_point_max_v1"}:
        raise ConfigError(f"未知 {section}.retrieval_unit：{retrieval_unit}")
    if "top_k" in agent:
        raw_top_k = agent["top_k"]
        if type(raw_top_k) is not int:
            raise ConfigError(f"{section}.top_k 必须为正整数")
        top_k = raw_top_k
        if top_k < 1:
            raise ConfigError(f"{section}.top_k 必须为正整数")

    embedding_backend = str(agent.get("embedding_backend", "hash"))
    if embedding_backend not in {"hash", "openai_compatible", "bgem3_hybrid_http"}:
        raise ConfigError(f"未知 {section}.embedding_backend：{embedding_backend}")
    if (
        "embedding_api_key_required" in agent
        and type(agent["embedding_api_key_required"]) is not bool
    ):
        raise ConfigError(f"{section}.embedding_api_key_required 必须严格为布尔值")
    if "embedding_timeout_seconds" in agent:
        timeout = agent["embedding_timeout_seconds"]
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ConfigError(f"{section}.embedding_timeout_seconds 必须为正的有限数值")
    if embedding_backend == "openai_compatible":
        for field in (
            "embedding_model",
            "embedding_revision",
            "embedding_query_instruction",
        ):
            if not str(agent.get(field, "")).strip():
                raise ConfigError(f"真实 embedding backend 缺少 {section}.{field}")
        if "embedding_output_dimensions" in agent:
            output_dimensions = agent["embedding_output_dimensions"]
            if type(output_dimensions) is not int or output_dimensions < 1:
                raise ConfigError(f"{section}.embedding_output_dimensions 必须为正整数")
    if embedding_backend == "bgem3_hybrid_http":
        for field in (
            "embedding_model",
            "embedding_revision",
            "embedding_weights_hash",
            "embedding_tokenizer_revision",
            "embedding_base_url_env",
            "embedding_cache_schema_version",
            "embedding_cache_path",
            "embedding_final_top_k_policy",
        ):
            if not str(agent.get(field, "")).strip():
                raise ConfigError(f"BGE-M3 hybrid backend 缺少 {section}.{field}")
        weights_hash = agent["embedding_weights_hash"]
        if (
            not isinstance(weights_hash, str)
            or len(weights_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in weights_hash)
        ):
            raise ConfigError(f"{section}.embedding_weights_hash 必须为 64 位十六进制 SHA-256")
        query_instruction = agent.get("embedding_query_instruction")
        if query_instruction is not None and str(query_instruction).strip():
            raise ConfigError("BGE-M3 hybrid backend 禁止 Qwen-style embedding_query_instruction")
        if str(agent.get("embedding_query_policy", "")) != "raw_symmetric_no_instruction":
            raise ConfigError(
                f"{section} BGE-M3 hybrid backend 要求 "
                "embedding_query_policy=raw_symmetric_no_instruction"
            )
        raw_weights = agent.get("embedding_hybrid_weights")
        if not isinstance(raw_weights, list) or len(raw_weights) != 3:
            raise ConfigError(f"{section} BGE-M3 hybrid backend 要求三个 embedding_hybrid_weights")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_weights
        ):
            raise ConfigError(f"{section}.embedding_hybrid_weights 必须是数值")
        try:
            weights = [float(value) for value in raw_weights]
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{section}.embedding_hybrid_weights 必须是数值") from exc
        if (
            any(not math.isfinite(value) or value < 0 for value in weights)
            or not math.isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ConfigError(f"{section}.embedding_hybrid_weights 必须为有限非负数且总和大于零")
        if weights != [0.4, 0.2, 0.4]:
            raise ConfigError(f"{section}.embedding_hybrid_weights 必须冻结为 [0.4, 0.2, 0.4]")
        raw_candidate_depth = agent.get("embedding_candidate_depth", 0)
        raw_rerank_depth = agent.get("embedding_colbert_rerank_depth", 0)
        if type(raw_candidate_depth) is not int or type(raw_rerank_depth) is not int:
            raise ConfigError(
                f"{section}.embedding_candidate_depth 与 embedding_colbert_rerank_depth 必须是整数"
            )
        candidate_depth = raw_candidate_depth
        rerank_depth = raw_rerank_depth
        if (
            candidate_depth < 1
            or candidate_depth > 1000
            or rerank_depth < 1
            or rerank_depth > candidate_depth
        ):
            raise ConfigError(
                f"{section} BGE-M3 depth 必须满足 "
                "1 <= colbert_rerank_depth <= candidate_depth <= 1000"
            )
        if rerank_depth != candidate_depth:
            raise ConfigError(
                "当前 BGE-M3 服务会对全部候选执行 ColBERT；"
                f"{section}.embedding_colbert_rerank_depth 必须等于 "
                "embedding_candidate_depth"
            )
        if str(agent.get("embedding_final_top_k_policy")) != "agent_roster_top_k":
            raise ConfigError(
                f"当前 BGE-M3 检索仅实现 {section}.embedding_final_top_k_policy=agent_roster_top_k"
            )
        startup = agent.get("embedding_service_startup_parameters")
        if not isinstance(startup, dict):
            raise ConfigError(
                f"BGE-M3 hybrid backend 缺少 {section}.embedding_service_startup_parameters"
            )
        required_startup = (
            "model_path",
            "service_script",
            "python",
            "dtype",
            "normalize_embeddings",
            "query_max_length",
            "passage_max_length",
            "cache_capacity",
            "cache_schema_version",
            "flagembedding_version",
        )
        strict_v2_contract = str(agent["embedding_cache_schema_version"]).endswith("_v2")
        if strict_v2_contract:
            required_startup += (
                "query_policy",
                "retrieval_modes",
                "hybrid_weights",
                "candidate_depth",
                "colbert_rerank_depth",
                "final_top_k_policy",
                "runtime_identity",
                "server_source_sha256",
                "output_contract_schema_version",
                "expected_dense_dimensions",
                "normalization_identity",
                "dense_norm_tolerance",
            )
        for field in required_startup:
            if field not in startup or startup[field] in (None, ""):
                raise ConfigError(
                    "BGE-M3 hybrid backend 缺少 "
                    f"{section}.embedding_service_startup_parameters.{field}"
                )
        if str(startup["cache_schema_version"]) != str(agent["embedding_cache_schema_version"]):
            raise ConfigError(f"BGE-M3 cache schema 在 {section} 与启动参数间不一致")
        for field in (
            "model_path",
            "service_script",
            "python",
        ):
            if not isinstance(startup[field], str) or not startup[field].strip():
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters.{field} 必须是非空字符串"
                )
        if "runtime_identity" in startup and (
            not isinstance(startup["runtime_identity"], str)
            or not startup["runtime_identity"].strip()
            or startup["runtime_identity"].strip() == "unspecified"
        ):
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters.runtime_identity "
                "必须是显式非空身份"
            )
        if startup["dtype"] != "float16":
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters.dtype 必须为 float16"
            )
        if startup["normalize_embeddings"] is not True:
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters."
                "normalize_embeddings 必须严格为 true"
            )
        if strict_v2_contract:
            if (
                startup["output_contract_schema_version"]
                != BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION
            ):
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters."
                    "output_contract_schema_version 与代码合同不一致"
                )
            if (
                type(startup["expected_dense_dimensions"]) is not int
                or startup["expected_dense_dimensions"]
                != BGE_M3_EXPECTED_DENSE_DIMENSIONS
            ):
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters."
                    f"expected_dense_dimensions 必须冻结为 {BGE_M3_EXPECTED_DENSE_DIMENSIONS}"
                )
            if startup["normalization_identity"] != BGE_M3_NORMALIZATION_IDENTITY:
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters."
                    "normalization_identity 与 normalize_embeddings=True 合同不一致"
                )
            raw_norm_tolerance = startup["dense_norm_tolerance"]
            if (
                isinstance(raw_norm_tolerance, bool)
                or not isinstance(raw_norm_tolerance, (int, float))
                or not math.isfinite(float(raw_norm_tolerance))
                or not math.isclose(
                    float(raw_norm_tolerance),
                    BGE_M3_DENSE_NORM_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters."
                    f"dense_norm_tolerance 必须冻结为 {BGE_M3_DENSE_NORM_TOLERANCE}"
                )
        for field in (
            "query_max_length",
            "passage_max_length",
            "cache_capacity",
        ):
            value = startup[field]
            if type(value) is not int or value < 1:
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters.{field} 必须为正整数"
                )
        if "batch_size" in startup and (
            type(startup["batch_size"]) is not int or startup["batch_size"] < 1
        ):
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters.batch_size 必须为正整数"
            )
        if str(startup["flagembedding_version"]) != "1.4.0":
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters."
                "flagembedding_version 必须冻结为 1.4.0"
            )
        if "query_policy" in startup and startup["query_policy"] != agent["embedding_query_policy"]:
            raise ConfigError(f"{section} BGE-M3 query_policy 在顶层与启动参数间不一致")
        if "retrieval_modes" in startup and startup["retrieval_modes"] != [
            "dense",
            "sparse",
            "colbert",
        ]:
            raise ConfigError(
                f"{section}.embedding_service_startup_parameters.retrieval_modes "
                "必须为 [dense, sparse, colbert]"
            )
        if "hybrid_weights" in startup and startup["hybrid_weights"] != raw_weights:
            raise ConfigError(f"{section} BGE-M3 hybrid_weights 在顶层与启动参数间不一致")
        if "candidate_depth" in startup and (
            type(startup["candidate_depth"]) is not int
            or startup["candidate_depth"] != candidate_depth
        ):
            raise ConfigError(f"{section} BGE-M3 candidate_depth 在顶层与启动参数间不一致")
        if "colbert_rerank_depth" in startup and (
            type(startup["colbert_rerank_depth"]) is not int
            or startup["colbert_rerank_depth"] != rerank_depth
        ):
            raise ConfigError(f"{section} BGE-M3 colbert_rerank_depth 在顶层与启动参数间不一致")
        if (
            "final_top_k_policy" in startup
            and startup["final_top_k_policy"] != agent["embedding_final_top_k_policy"]
        ):
            raise ConfigError(f"{section} BGE-M3 final_top_k_policy 在顶层与启动参数间不一致")
        if "server_source_sha256" in startup:
            source_sha = startup["server_source_sha256"]
            if (
                not isinstance(source_sha, str)
                or len(source_sha) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in source_sha)
            ):
                raise ConfigError(
                    f"{section}.embedding_service_startup_parameters."
                    "server_source_sha256 必须为 64 位十六进制 SHA-256"
                )

    threshold_status = str(agent.get("retrieval_threshold_status", "pending_pilot"))
    if threshold_status not in {"pending_pilot", "frozen"}:
        raise ConfigError(f"{section}.retrieval_threshold_status 必须为 pending_pilot 或 frozen")
    minimum = agent.get("minimum_retrieval_score")
    if threshold_status == "frozen" and minimum is None:
        raise ConfigError(f"冻结检索阈值时必须提供 {section}.minimum_retrieval_score")
    if minimum is not None:
        if type(minimum) not in {int, float}:
            raise ConfigError(f"{section}.minimum_retrieval_score 必须是有限数值")
        parsed_minimum = float(minimum)
        if not math.isfinite(parsed_minimum):
            raise ConfigError(f"{section}.minimum_retrieval_score 必须是有限数值")


def _strict_config_integer(value: object, field: str) -> int:
    """Accept YAML/Python integers only; never coerce bool, float, or strings."""

    if type(value) is not int:
        raise ConfigError(f"{field} 必须是整数，禁止 bool、浮点数或数值字符串")
    return value


def _validate_memory_contract(
    section: str,
    config: dict[str, Any],
    run_mode: str,
) -> None:
    """Validate versioned memory semantics before any model or BGE service call."""

    try:
        mechanism = resolve_memory_mechanism(config)
    except ValueError as exc:
        raise ConfigError(f"{section}: {exc}") from exc
    semantic_review_mode = str(
        config.get("experience_semantic_review_mode", "disabled")
    ).strip()
    if semantic_review_mode not in _EXPERIENCE_SEMANTIC_REVIEW_MODES:
        raise ConfigError(
            f"{section}.experience_semantic_review_mode 必须为 disabled 或 "
            "independent_model_v1"
        )
    semantics_mode = str(
        config.get(
            "experience_semantics_mode",
            config.get("memory_semantics_mode", "quality_filtered_v2"),
        )
    )
    if semantics_mode not in {"paper_faithful_v1", "quality_filtered_v2"}:
        raise ConfigError(f"未知 {section}.experience_semantics_mode：{semantics_mode}")
    revision_strategy = str(
        config.get("experience_revision_strategy", "deterministic")
    ).strip()
    revision_contract = str(
        config.get(
            "experience_revision_contract",
            "full_document_v1" if semantics_mode == "paper_faithful_v1" else "structured_patch_v2",
        )
    )
    if revision_contract not in {"full_document_v1", "structured_patch_v2"}:
        raise ConfigError(
            f"{section}.experience_revision_contract 必须为 full_document_v1 或 structured_patch_v2"
        )
    if semantic_review_mode == "independent_model_v1":
        if revision_strategy != "llm":
            raise ConfigError(
                f"{section}.experience_semantic_review_mode=independent_model_v1 要求 "
                "experience_revision_strategy=llm"
            )
        if revision_contract != "full_document_v1":
            raise ConfigError(
                f"{section}.experience_semantic_review_mode=independent_model_v1 要求 "
                "experience_revision_contract=full_document_v1"
            )
    document_contract = str(
        config.get("experience_document_contract", "five_section_actionable_v1")
    )
    if document_contract not in {
        "five_section_actionable_v1",
        "transferable_rulebook_v2",
        "paper_five_section_v1",
    }:
        raise ConfigError(
            f"{section}.experience_document_contract 必须为 "
            "five_section_actionable_v1、transferable_rulebook_v2 或 paper_five_section_v1"
        )
    failure_policy = str(config.get("experience_failure_policy", "fail_closed_keep"))
    if failure_policy != "fail_closed_keep":
        raise ConfigError(f"{section}.experience_failure_policy 必须为 fail_closed_keep")
    max_chars = _strict_config_integer(
        config.get("experience_max_chars", 2400),
        f"{section}.experience_max_chars",
    )
    prompt_token_budget = _strict_config_integer(
        config.get("experience_prompt_token_budget", 32768),
        f"{section}.experience_prompt_token_budget",
    )
    update_period = _strict_config_integer(
        config.get("experience_update_period", 1),
        f"{section}.experience_update_period",
    )
    if max_chars < 512:
        raise ConfigError(f"{section}.experience_max_chars 必须至少为 512")
    if prompt_token_budget < 1:
        raise ConfigError(f"{section}.experience_prompt_token_budget 必须为正整数")
    if update_period < 1:
        raise ConfigError(f"{section}.experience_update_period 必须至少为 1")

    is_async = mechanism == "fact_expr_async"
    expected_top_k = 30 if mechanism == "fact" else 14 if is_async else 15
    top_k: int | None = None
    if mechanism in FACTUAL_MECHANISMS:
        top_k = _strict_config_integer(
            config.get("top_k", expected_top_k),
            f"{section}.top_k",
        )
        if top_k < 1:
            raise ConfigError(f"{section}.top_k 必须为正整数")
        duplicate_window = _strict_config_integer(
            config.get(
                "fact_duplicate_window",
                0 if semantics_mode == "paper_faithful_v1" else 50,
            ),
            f"{section}.fact_duplicate_window",
        )
        if duplicate_window < 0:
            raise ConfigError(f"{section}.fact_duplicate_window 不能为负数")
    else:
        duplicate_window = 0

    window_size: int | None = None
    if mechanism in EXPERIENCE_MECHANISMS:
        window_size = _strict_config_integer(
            config.get("window_size", 5 if is_async else 30),
            f"{section}.window_size",
        )
        if window_size < 1:
            raise ConfigError(f"{section}.window_size 必须为正整数")

    sweep_every: int | None = None
    evidence_k: int | None = None
    if is_async:
        sweep_every = _strict_config_integer(
            config.get("sweep_every", 5),
            f"{section}.sweep_every",
        )
        evidence_k = _strict_config_integer(
            config.get("evidence_k", 30),
            f"{section}.evidence_k",
        )
        if sweep_every < 1 or evidence_k < 1:
            raise ConfigError(f"{section}.sweep_every/evidence_k 必须为正整数")
        if window_size is None or window_size < sweep_every:
            raise ConfigError(f"{section}.window_size 必须不小于 sweep_every")

    if mechanism in FACTUAL_MECHANISMS:
        for field in (
            "reject_zero_reward_preflop_fold",
            "reject_single_preflop_fold",
            "retrieval_signature_dedup",
        ):
            if field in config and type(config[field]) is not bool:
                raise ConfigError(f"{section}.{field} 必须严格为布尔值")
        raw_max_records = config.get(
            "max_records", 0 if semantics_mode == "paper_faithful_v1" else 500
        )
        normalized_max_records = (
            0
            if raw_max_records is None
            else _strict_config_integer(raw_max_records, f"{section}.max_records")
        )
        if normalized_max_records < 0:
            raise ConfigError(f"{section}.max_records 必须是非负整数、0 或 null")

        raw_semantic_weight = config.get(
            "semantic_weight",
            1.0 if semantics_mode == "paper_faithful_v1" else 0.65,
        )
        raw_feature_weight = config.get(
            "feature_weight",
            0.0 if semantics_mode == "paper_faithful_v1" else 0.35,
        )
        default_semantic_weight = 1.0 if semantics_mode == "paper_faithful_v1" else 0.65
        default_feature_weight = 0.0 if semantics_mode == "paper_faithful_v1" else 0.35
        semantic_value = (
            default_semantic_weight if raw_semantic_weight is None else raw_semantic_weight
        )
        feature_value = default_feature_weight if raw_feature_weight is None else raw_feature_weight
        if type(semantic_value) not in {int, float} or type(feature_value) not in {
            int,
            float,
        }:
            raise ConfigError(f"{section}.semantic_weight/feature_weight 必须是数值")
        semantic_weight = float(semantic_value)
        feature_weight = float(feature_value)
        if (
            not math.isfinite(semantic_weight)
            or not math.isfinite(feature_weight)
            or semantic_weight < 0
            or feature_weight < 0
            or semantic_weight + feature_weight <= 0
        ):
            raise ConfigError(
                f"{section}.semantic_weight/feature_weight 必须为有限非负数且总和大于零"
            )
        if semantics_mode == "paper_faithful_v1":
            if str(config.get("retrieval_backend", "hybrid_rag")) != "hybrid_rag":
                raise ConfigError(
                    f"{section} paper_faithful_v1 要求 "
                    "retrieval_backend=hybrid_rag 并使用纯 embedding 排序"
                )
            if normalized_max_records != 0:
                raise ConfigError(
                    f"{section} paper_faithful_v1 要求 max_records=0 或 null（不截断）"
                )
            if semantic_weight != 1.0 or feature_weight != 0.0:
                raise ConfigError(
                    f"{section} paper_faithful_v1 要求纯 embedding 排序："
                    "semantic_weight=1.0, feature_weight=0.0"
                )
    if is_async:
        async_numbers = {
            "salience_threshold": config.get("salience_threshold", 0.03),
            "salience_mirror_threshold": config.get("salience_mirror_threshold", 0.30),
            "mirror_prob": config.get(
                "mirror_prob", 0.30 if semantics_mode == "paper_faithful_v1" else 0.20
            ),
            "stability_init": config.get("stability_init", 10.0),
            "stability_min": config.get("stability_min", 0.5),
            "stability_max": config.get("stability_max", 50.0),
        }
        parsed_async_numbers: dict[str, float] = {}
        for field, value in async_numbers.items():
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ConfigError(f"{section}.{field} 必须为有限数值")
            parsed_async_numbers[field] = float(value)
        if not (
            0.0
            <= parsed_async_numbers["salience_threshold"]
            <= parsed_async_numbers["salience_mirror_threshold"]
            <= 1.0
        ):
            raise ConfigError(f"{section}.salience 阈值必须满足 0 <= main <= mirror <= 1")
        if not 0.0 <= parsed_async_numbers["mirror_prob"] <= 1.0:
            raise ConfigError(f"{section}.mirror_prob 必须在 [0, 1]")
        if not (
            0.0
            < parsed_async_numbers["stability_min"]
            <= parsed_async_numbers["stability_init"]
            <= parsed_async_numbers["stability_max"]
        ):
            raise ConfigError(f"{section}.stability 必须满足 0 < min <= init <= max")
    if (
        mechanism in EXPERIENCE_MECHANISMS
        and str(config.get("experience_revision_strategy", "deterministic")) == "llm"
        and run_mode not in {"smoke", "pilot", "formal"}
    ):
        raise ConfigError(f"{section} 的 LLM experience revision 需要受支持的 run_mode")
    if semantics_mode != "paper_faithful_v1":
        return

    if revision_contract != "full_document_v1":
        raise ConfigError(
            f"{section} paper_faithful_v1 必须使用 "
            "experience_revision_contract=full_document_v1；"
            "structured_patch_v2 只保留为非论文兼容模式"
        )

    if str(config.get("retrieval_unit", "hand_terminal_v1")) != "hand_terminal_v1":
        raise ConfigError(
            f"{section} paper_faithful_v1 必须使用 hand_terminal_v1；"
            "decision_point_max_v1 属于 calibrated/quality 协议"
        )
    if mechanism in FACTUAL_MECHANISMS:
        if duplicate_window != 0:
            raise ConfigError(f"{section} paper_faithful_v1 禁止 Fact 写入去重")
        for field in (
            "reject_zero_reward_preflop_fold",
            "reject_single_preflop_fold",
            "retrieval_signature_dedup",
        ):
            if bool(config.get(field, False)):
                raise ConfigError(f"{section} paper_faithful_v1 要求 {field}=false")
    if mechanism in FACTUAL_MECHANISMS and top_k != expected_top_k:
        raise ConfigError(
            f"{section} paper_faithful_v1 的 {mechanism} top_k 必须为 {expected_top_k}"
        )
    if mechanism in EXPERIENCE_MECHANISMS:
        if str(config.get("experience_revision_strategy", "deterministic")) != "llm":
            raise ConfigError(
                f"{section} paper_faithful_v1 必须使用 experience_revision_strategy=llm"
            )
        expected_window = 5 if is_async else 30
        if window_size != expected_window:
            raise ConfigError(
                f"{section} paper_faithful_v1 的 window_size 必须为 {expected_window}"
            )
        if not is_async and update_period != 1:
            raise ConfigError(f"{section} paper_faithful_v1 的 experience_update_period 必须为 1")
        if is_async:
            mirror_prob = config.get("mirror_prob", 0.30)
            if type(mirror_prob) not in {int, float} or not math.isfinite(float(mirror_prob)):
                raise ConfigError(f"{section}.mirror_prob 必须为有限数值")
            if float(mirror_prob) != 0.30:
                raise ConfigError(f"{section} paper_faithful_v1 的 mirror_prob 必须为 0.30")
            if sweep_every != 5:
                raise ConfigError(f"{section} paper_faithful_v1 的 sweep_every 必须为 5")
            if evidence_k != 30:
                raise ConfigError(f"{section} paper_faithful_v1 的 evidence_k 必须为 30")


def resolve_checkpoint_set(experiment: dict[str, Any]) -> list[int] | None:
    """Return a validated explicit checkpoint set, or ``None`` for interval mode."""

    raw = experiment.get("checkpoint_set")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ConfigError("experiment.checkpoint_set 必须是非空整数列表")
    if "checkpoint_interval" in experiment and int(experiment.get("checkpoint_interval", 0)) > 0:
        raise ConfigError("checkpoint_set 与正数 checkpoint_interval 不得同时配置")
    try:
        points = [int(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ConfigError("experiment.checkpoint_set 必须只包含整数") from exc
    if any(value <= 0 for value in points):
        raise ConfigError("experiment.checkpoint_set 必须只包含正整数")
    if points != sorted(set(points)):
        raise ConfigError("experiment.checkpoint_set 必须严格递增且无重复")
    train_hands = int(experiment.get("train_hands", 0))
    if train_hands <= 0:
        raise ConfigError("显式 checkpoint_set 要求 experiment.train_hands 为正整数")
    if points[-1] > train_hands:
        raise ConfigError("experiment.checkpoint_set 不得超过 train_hands")
    if points[-1] != train_hands:
        raise ConfigError("experiment.checkpoint_set 必须包含最终 train_hands")
    return points


def resolve_heldout_table_set(experiment: dict[str, Any]) -> list[str]:
    """Return held-out table identifiers in their frozen declaration order."""

    raw = experiment.get("heldout_table_set", ["H01"])
    if not isinstance(raw, list) or not raw:
        raise ConfigError("experiment.heldout_table_set 必须是非空列表")
    tables = [str(value).strip() for value in raw]
    if any(not value for value in tables):
        raise ConfigError("experiment.heldout_table_set 不得包含空 table_id")
    if len(tables) != len(set(tables)):
        raise ConfigError("experiment.heldout_table_set 不得重复")
    return tables


def _validate_checkpoint_protocol(experiment: dict[str, Any]) -> None:
    checkpoint_set = resolve_checkpoint_set(experiment)
    if "checkpoint_interval" in experiment:
        try:
            interval = int(experiment["checkpoint_interval"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("experiment.checkpoint_interval 必须是整数") from exc
        if interval < 0:
            raise ConfigError("experiment.checkpoint_interval 不能为负数")
    raw_hands = experiment.get("checkpoint_test_hands_by_checkpoint")
    if raw_hands is not None:
        if checkpoint_set is None:
            raise ConfigError("checkpoint_test_hands_by_checkpoint 要求显式 checkpoint_set")
        if not isinstance(raw_hands, dict):
            raise ConfigError("checkpoint_test_hands_by_checkpoint 必须是映射")
        try:
            mapped = {int(key): int(value) for key, value in raw_hands.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigError("checkpoint test hands 的键和值必须是整数") from exc
        if set(mapped) - set(checkpoint_set):
            raise ConfigError("checkpoint test hands 含未声明 checkpoint")
        if any(value < 0 for value in mapped.values()):
            raise ConfigError("checkpoint test hands 不能为负数")


def _validate_heldout_table_set(experiment: dict[str, Any]) -> None:
    tables = resolve_heldout_table_set(experiment)
    run_mode = str(experiment.get("run_mode", "smoke"))
    rosters = experiment.get("heldout_table_rosters")
    if run_mode == "formal" and len(tables) > 1:
        if experiment.get("heldout_roster_identity"):
            raise ConfigError("Formal 多 held-out tables 禁止 scalar heldout_roster_identity 覆盖")
        if not isinstance(rosters, dict) or set(rosters) != set(tables):
            raise ConfigError("Formal heldout_table_rosters 必须完整覆盖 heldout_table_set")
        if any(not isinstance(rosters[table_id], dict) for table_id in tables):
            raise ConfigError("每个 held-out table roster 必须是配置映射")
        identities = [
            yaml.safe_dump(rosters[table_id], allow_unicode=True, sort_keys=True)
            for table_id in tables
        ]
        if len(identities) != len(set(identities)):
            raise ConfigError("Formal H01/H02/H03 必须使用不同的自然 roster identity")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    功能：递归合并两个配置字典。
    参数：
        base：基础配置。
        override：覆盖配置。
    返回：合并结果。
    副作用：无。
    异常：无。
    设计说明：实验配置只写差异，resolved 快照保存完整结果。
    """

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def dump_yaml(data: dict[str, Any]) -> str:
    """
    功能：把配置字典转为稳定 YAML 文本。
    参数：
        data：配置字典。
    返回：YAML 字符串。
    副作用：无。
    异常：无。
    设计说明：resolved_config.yaml 由同一函数生成，便于比较。
    """

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    """
    功能：读取 YAML 文件。
    参数：
        path：文件路径。
    返回：字典。
    副作用：读取文件。
    异常：YAML 顶层不是字典时抛出 ConfigError。
    设计说明：私有函数集中处理文件格式错误。
    """

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败：{path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML 顶层必须是对象：{path}")
    return raw
