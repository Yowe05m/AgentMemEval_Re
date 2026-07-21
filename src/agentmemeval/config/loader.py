"""
模块说明：本模块负责 YAML 配置加载、继承合并和基础校验。
核心职责：把实验配置解析为单一 resolved config，供运行器快照保存。
输入与输出：输入配置路径，输出合并后的字典。
依赖边界：依赖 PyYAML 与标准库 pathlib，不依赖实验模块。
不负责：不创建 Provider，不运行实验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentmemeval.core.errors import ConfigError


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
            raise ConfigError(
                f"共享记忆尚未实现；{section}.memory_scope 必须为 per_agent"
            )
        if section_config.get("persona") and run_mode != "smoke":
            raise ConfigError("Exp2 人格机制已延期；persona 配置只能用于 not_for_paper smoke")
    agent = config.get("agent", {})
    retrieval_unit = str(agent.get("retrieval_unit", "hand_terminal_v1"))
    if retrieval_unit not in {"hand_terminal_v1", "decision_point_max_v1"}:
        raise ConfigError(f"未知 agent.retrieval_unit：{retrieval_unit}")
    embedding_backend = str(agent.get("embedding_backend", "hash"))
    if embedding_backend not in {"hash", "openai_compatible", "bgem3_hybrid_http"}:
        raise ConfigError(f"未知 agent.embedding_backend：{embedding_backend}")
    if embedding_backend == "openai_compatible":
        for field in ("embedding_model", "embedding_revision", "embedding_query_instruction"):
            if not str(agent.get(field, "")).strip():
                raise ConfigError(f"真实 embedding backend 缺少 agent.{field}")
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
                raise ConfigError(f"BGE-M3 hybrid backend 缺少 agent.{field}")
        query_instruction = agent.get("embedding_query_instruction")
        if query_instruction is not None and str(query_instruction).strip():
            raise ConfigError("BGE-M3 hybrid backend 禁止 Qwen-style embedding_query_instruction")
        if str(agent.get("embedding_query_policy", "")) != "raw_symmetric_no_instruction":
            raise ConfigError(
                "BGE-M3 hybrid backend 要求 embedding_query_policy=raw_symmetric_no_instruction"
            )
        raw_weights = agent.get("embedding_hybrid_weights")
        if not isinstance(raw_weights, list) or len(raw_weights) != 3:
            raise ConfigError("BGE-M3 hybrid backend 要求三个 embedding_hybrid_weights")
        try:
            weights = [float(value) for value in raw_weights]
        except (TypeError, ValueError) as exc:
            raise ConfigError("embedding_hybrid_weights 必须是数值") from exc
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ConfigError("embedding_hybrid_weights 必须非负且总和大于零")
        try:
            candidate_depth = int(agent.get("embedding_candidate_depth", 0))
            rerank_depth = int(agent.get("embedding_colbert_rerank_depth", 0))
        except (TypeError, ValueError) as exc:
            raise ConfigError("BGE-M3 candidate/rerank depth 必须是整数") from exc
        if candidate_depth < 1 or rerank_depth < 1 or rerank_depth > candidate_depth:
            raise ConfigError(
                "BGE-M3 depth 必须满足 1 <= colbert_rerank_depth <= candidate_depth"
            )
        startup = agent.get("embedding_service_startup_parameters")
        if not isinstance(startup, dict):
            raise ConfigError("BGE-M3 hybrid backend 缺少 embedding_service_startup_parameters")
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
        for field in required_startup:
            if field not in startup or startup[field] in (None, ""):
                raise ConfigError(
                    f"BGE-M3 hybrid backend 缺少 embedding_service_startup_parameters.{field}"
                )
        if str(startup["cache_schema_version"]) != str(
            agent["embedding_cache_schema_version"]
        ):
            raise ConfigError("BGE-M3 cache schema 在 agent 与启动参数间不一致")
    threshold_status = str(agent.get("retrieval_threshold_status", "pending_pilot"))
    if threshold_status not in {"pending_pilot", "frozen"}:
        raise ConfigError("agent.retrieval_threshold_status 必须为 pending_pilot 或 frozen")
    if threshold_status == "frozen" and agent.get("minimum_retrieval_score") is None:
        raise ConfigError("冻结检索阈值时必须提供 agent.minimum_retrieval_score")
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
            if str(experiment.get("within_table_mechanism_aggregation", "")) != (
                "arithmetic_mean"
            ):
                raise ConfigError("A7-R 桌内同机制 Agent 必须预注册 arithmetic_mean 聚合")
        else:
            if str(experiment.get("within_table_mechanism_aggregation", "")) != (
                "single_target_condition"
            ):
                raise ConfigError("Campaign E 每个条件必须只贡献一个 target 统计单位")
            if str(experiment.get("cross_condition_aggregation", "")) != (
                "paired_by_seed"
            ):
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
                if item.get("persona") and run_mode != "smoke":
                    raise ConfigError("Exp2 人格机制已延期；roster persona 只能用于 smoke")


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
