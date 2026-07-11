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


def build_memory(agent_id: str, config: dict[str, object]) -> MemoryMechanism:
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

    mechanism = str(config.get("mechanism", config.get("type", "no_memory")))
    scope = config.get("memory_scope", config.get("scope", "per_agent"))
    top_k = int(config.get("top_k", 8))
    window_size = int(config.get("window_size", 8))
    max_records = int(config.get("max_records", 500))
    retrieval_backend = str(config.get("retrieval_backend", "hybrid_rag"))
    if mechanism in {"no_memory", "none", "naive"}:
        memory: MemoryMechanism = NullMemory(agent_id, scope=scope)  # type: ignore[arg-type]
    elif mechanism in {"fact", "FactAgent"}:
        memory = FactualMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
        )
    elif mechanism in {"expr", "ExprAgent"}:
        memory = ExperientialMemory(agent_id, scope=scope, window_size=window_size)  # type: ignore[arg-type]
    elif mechanism in {"fact_expr_sync", "fxsync", "FactExprSync"}:
        memory = FactExprSyncMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            window_size=window_size,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
        )
    elif mechanism in {"fact_expr_async", "fxasync", "FactExprAsync"}:
        memory = FactExprAsyncMemory(
            agent_id,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
            window_size=window_size,
            sweep_every=int(config.get("sweep_every", 3)),
            evidence_k=int(config.get("evidence_k", 6)),
            max_records=max_records,
            salience_threshold=float(config.get("salience_threshold", 0.03)),
            salience_mirror_threshold=float(config.get("salience_mirror_threshold", 0.30)),
            mirror_prob=float(config.get("mirror_prob", 0.20)),
            stability_init=float(config.get("stability_init", 10.0)),
            stability_min=float(config.get("stability_min", 0.5)),
            stability_max=float(config.get("stability_max", 50.0)),
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

    memory = build_memory(agent_id, config)
    return LLMDecisionAgent(
        agent_id,
        memory,
        llm_client,
        model=model,
        raise_sizing_policy=str(config.get("raise_sizing_policy", "native_no_limit")),
    )
