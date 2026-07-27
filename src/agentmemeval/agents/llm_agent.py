"""
模块说明：本模块根据配置创建不同记忆机制的 LLM Agent。
核心职责：把机制名映射到 NoMemory、Fact、Expr、Sync、Async 和人格 wrapper。
输入与输出：输入 agent_id、机制配置和 Provider，输出 Agent 实例。
依赖边界：依赖记忆模块和通用 Agent，不依赖实验场景。
不负责：不读取 YAML 文件，不创建 Provider。
"""

from __future__ import annotations

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.core.errors import ConfigError
from agentmemeval.core.memory_mechanism import (
    FACTUAL_MECHANISMS,
    resolve_memory_mechanism,
)
from agentmemeval.core.protocols import LLMClient, MemoryMechanism
from agentmemeval.memory import (
    ExperientialMemory,
    FactExprAsyncMemory,
    FactExprSyncMemory,
    FactualMemory,
    NullMemory,
    PersonalityDrivenMemory,
)
from agentmemeval.memory.personality_driven import DEFAULT_PERSONAS
from agentmemeval.memory.rag import build_embedding_backend


def build_memory(
    agent_id: str,
    config: dict[str, object],
    llm_client: LLMClient | None = None,
    model: str = "",
    experience_reviewer_client: LLMClient | None = None,
    experience_reviewer_model: str = "",
) -> MemoryMechanism:
    """
    功能：根据配置创建记忆机制。
    参数：
        agent_id：Agent 标识。
        config：Agent 或 memory 配置。
    返回：MemoryMechanism。
    副作用：无。
    异常：未知机制时抛出 ConfigError。
    设计说明：机制构建集中在这里，实验控制流不需要因机制新增而修改。
    """

    try:
        mechanism = resolve_memory_mechanism(config)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    scope = config.get("memory_scope", config.get("scope", "per_agent"))
    semantics_mode = str(
        config.get(
            "experience_semantics_mode",
            config.get("memory_semantics_mode", "quality_filtered_v2"),
        )
    )
    paper_faithful = semantics_mode == "paper_faithful_v1"
    if mechanism == "fact":
        default_top_k = 30 if paper_faithful else 8
    elif mechanism == "fact_expr_sync":
        default_top_k = 15 if paper_faithful else 8
    elif mechanism == "fact_expr_async":
        default_top_k = 14 if paper_faithful else 8
    else:
        default_top_k = 8
    default_window_size = (
        5 if paper_faithful and mechanism == "fact_expr_async" else 30 if paper_faithful else 8
    )
    top_k = int(config.get("top_k", default_top_k))
    window_size = int(config.get("window_size", default_window_size))
    raw_max_records = config.get("max_records", None if paper_faithful else 500)
    max_records = None if raw_max_records is None else int(raw_max_records)
    retrieval_backend = str(config.get("retrieval_backend", "hybrid_rag"))
    experience_max_chars = int(config.get("experience_max_chars", 2400))
    raw_experience_prompt_token_budget = config.get(
        "experience_prompt_token_budget", 32768
    )
    if (
        type(raw_experience_prompt_token_budget) is not int
        or raw_experience_prompt_token_budget < 1
    ):
        raise ConfigError("experience_prompt_token_budget 必须为正整数")
    experience_prompt_token_budget = raw_experience_prompt_token_budget
    experience_update_period = int(config.get("experience_update_period", 1))
    revision_contract = str(
        config.get(
            "experience_revision_contract",
            "full_document_v1" if paper_faithful else "structured_patch_v2",
        )
    )
    document_contract = str(
        config.get("experience_document_contract", "five_section_actionable_v1")
    )
    failure_policy = str(config.get("experience_failure_policy", "fail_closed_keep"))
    raw_minimum_score = config.get("minimum_retrieval_score")
    raw_semantic_weight = config.get("semantic_weight", 1.0 if paper_faithful else 0.65)
    raw_feature_weight = config.get("feature_weight", 0.0 if paper_faithful else 0.35)
    fact_options = {
        "retrieval_unit": str(config.get("retrieval_unit", "hand_terminal_v1")),
        "minimum_retrieval_score": (
            None if raw_minimum_score is None else float(raw_minimum_score)
        ),
        "retrieval_threshold_status": str(
            config.get("retrieval_threshold_status", "pending_pilot")
        ),
        "semantic_weight": float(
            (1.0 if paper_faithful else 0.65)
            if raw_semantic_weight is None
            else raw_semantic_weight
        ),
        "feature_weight": float(
            (0.0 if paper_faithful else 0.35) if raw_feature_weight is None else raw_feature_weight
        ),
        "semantics_mode": semantics_mode,
    }
    optional_fact_options = {
        "fact_duplicate_window": ("duplicate_window", int),
        "reject_zero_reward_preflop_fold": (
            "reject_zero_reward_preflop_fold",
            bool,
        ),
        "reject_single_preflop_fold": ("reject_single_preflop_fold", bool),
        "retrieval_signature_dedup": ("retrieval_signature_dedup", bool),
    }
    for config_key, (option_key, converter) in optional_fact_options.items():
        if config_key in config:
            fact_options[option_key] = converter(config[config_key])
    embedding_backend = (
        build_embedding_backend(config, agent_id) if mechanism in FACTUAL_MECHANISMS else None
    )
    if mechanism == "no_memory":
        memory: MemoryMechanism = NullMemory(agent_id, scope=scope)  # type: ignore[arg-type]
    elif mechanism == "fact":
        memory = FactualMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
            embedding_backend=embedding_backend,
            **fact_options,
        )
    elif mechanism == "expr":
        memory = ExperientialMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            window_size=window_size,
            revision_strategy=str(config.get("experience_revision_strategy", "deterministic")),
            llm_client=llm_client,
            model=model,
            max_chars=experience_max_chars,
            prompt_token_budget=experience_prompt_token_budget,
            update_period=experience_update_period,
            semantics_mode=semantics_mode,
            revision_contract=revision_contract,
            document_contract=document_contract,
            failure_policy=failure_policy,
            semantic_review_mode=str(
                config.get("experience_semantic_review_mode", "disabled")
            ),
            semantic_reviewer_client=experience_reviewer_client,
            semantic_reviewer_model=experience_reviewer_model,
        )
    elif mechanism == "fact_expr_sync":
        memory = FactExprSyncMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            window_size=window_size,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
            embedding_backend=embedding_backend,
            fact_options=fact_options,
            revision_strategy=str(config.get("experience_revision_strategy", "deterministic")),
            llm_client=llm_client,
            model=model,
            max_chars=experience_max_chars,
            prompt_token_budget=experience_prompt_token_budget,
            update_period=experience_update_period,
            semantics_mode=semantics_mode,
            revision_contract=revision_contract,
            document_contract=document_contract,
            failure_policy=failure_policy,
            semantic_review_mode=str(
                config.get("experience_semantic_review_mode", "disabled")
            ),
            semantic_reviewer_client=experience_reviewer_client,
            semantic_reviewer_model=experience_reviewer_model,
        )
    elif mechanism == "fact_expr_async":
        memory = FactExprAsyncMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            window_size=window_size,
            sweep_every=int(config.get("sweep_every", 5 if paper_faithful else 3)),
            evidence_k=int(config.get("evidence_k", 30 if paper_faithful else 6)),
            max_records=max_records,
            salience_threshold=float(config.get("salience_threshold", 0.03)),
            salience_mirror_threshold=float(config.get("salience_mirror_threshold", 0.30)),
            mirror_prob=float(config.get("mirror_prob", 0.30 if paper_faithful else 0.20)),
            stability_init=float(config.get("stability_init", 10.0)),
            stability_min=float(config.get("stability_min", 0.5)),
            stability_max=float(config.get("stability_max", 50.0)),
            embedding_backend=embedding_backend,
            fact_options=fact_options,
            revision_strategy=str(config.get("experience_revision_strategy", "deterministic")),
            llm_client=llm_client,
            model=model,
            max_chars=experience_max_chars,
            prompt_token_budget=experience_prompt_token_budget,
            semantics_mode=semantics_mode,
            revision_contract=revision_contract,
            document_contract=document_contract,
            failure_policy=failure_policy,
            semantic_review_mode=str(
                config.get("experience_semantic_review_mode", "disabled")
            ),
            semantic_reviewer_client=experience_reviewer_client,
            semantic_reviewer_model=experience_reviewer_model,
        )
    else:
        raise ConfigError(f"未知记忆机制：{mechanism}")
    persona = config.get("persona")
    if persona:
        persona_name = str(persona)
        persona_config = dict(
            config.get("persona_config", {}) or DEFAULT_PERSONAS.get(persona_name, {})
        )
        memory = PersonalityDrivenMemory(memory, persona_name, persona_config)
    return memory


def build_agent(
    agent_id: str,
    config: dict[str, object],
    llm_client: LLMClient,
    model: str,
    experience_reviewer_client: LLMClient | None = None,
    experience_reviewer_model: str = "",
) -> LLMDecisionAgent:
    """
    功能：根据配置创建通用 LLM Agent。
    参数：
        agent_id：Agent 标识。
        config：Agent 配置。
        llm_client：Provider。
        model：模型名称。
    返回：LLMDecisionAgent。
    副作用：无。
    异常：未知机制时抛出 ConfigError。
    设计说明：除 NoMemory 的显式类外，其余机制共享同一 Agent 管线。
    """

    memory = build_memory(
        agent_id,
        config,
        llm_client=llm_client,
        model=model,
        experience_reviewer_client=experience_reviewer_client,
        experience_reviewer_model=experience_reviewer_model,
    )
    return LLMDecisionAgent(
        agent_id,
        memory,
        llm_client,
        model=model,
        raise_sizing_policy=str(config.get("raise_sizing_policy", "native_no_limit")),
    )
