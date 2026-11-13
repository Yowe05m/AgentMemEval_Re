from __future__ import annotations

from agentmemeval.path1_v1.agents.base import LLMDecisionAgent
from agentmemeval.path1_v1.core.errors import ConfigError
from agentmemeval.path1_v1.core.memory_mechanism import resolve_memory_mechanism
from agentmemeval.path1_v1.core.protocols import LLMClient, MemoryMechanism
from agentmemeval.path1_v1.memory.base import NullMemory
from agentmemeval.path1_v1.memory.experiential import ExperientialMemory


def build_memory(
    agent_id: str,
    config: dict[str, object],
    llm_client: LLMClient | None = None,
    model: str = "",
    experience_reviewer_client: LLMClient | None = None,
    experience_reviewer_model: str = "",
) -> MemoryMechanism:

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
    raw_experience_prompt_token_budget = config.get("experience_prompt_token_budget", 32768)
    if (
        type(raw_experience_prompt_token_budget) is not int
        or raw_experience_prompt_token_budget < 1
    ):
        raise ConfigError("experience_prompt_token_budget must be a positive integer")
    experience_prompt_token_budget = raw_experience_prompt_token_budget
    raw_revision_prompt_window_size = config.get("experience_revision_prompt_window_size")
    if raw_revision_prompt_window_size is not None and (
        type(raw_revision_prompt_window_size) is not int
        or raw_revision_prompt_window_size < 1
        or raw_revision_prompt_window_size > window_size
    ):
        raise ConfigError(
            "experience_revision_prompt_window_size must be an integer between 1 and window_size"
        )
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
    _ = top_k, max_records, retrieval_backend, fact_options
    if mechanism == "no_memory":
        memory: MemoryMechanism = NullMemory(agent_id, scope=scope)  # type: ignore[arg-type]
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
            revision_prompt_window_size=raw_revision_prompt_window_size,
            update_period=experience_update_period,
            semantics_mode=semantics_mode,
            revision_contract=revision_contract,
            document_contract=document_contract,
            failure_policy=failure_policy,
            semantic_review_mode=str(config.get("experience_semantic_review_mode", "disabled")),
            semantic_reviewer_client=experience_reviewer_client,
            semantic_reviewer_model=experience_reviewer_model,
        )
    else:
        raise ConfigError(f"unsupported Path1 v1 mechanism: {mechanism}")
    if config.get("persona"):
        raise ConfigError("persona wrappers are not part of the publication experiment package")
    return memory


def build_agent(
    agent_id: str,
    config: dict[str, object],
    llm_client: LLMClient,
    model: str,
    experience_reviewer_client: LLMClient | None = None,
    experience_reviewer_model: str = "",
) -> LLMDecisionAgent:

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
