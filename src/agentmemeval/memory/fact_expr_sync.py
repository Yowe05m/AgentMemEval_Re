"""
模块说明：本模块实现事实记忆与经验记忆同步并行机制。
核心职责：决策时同时提供事实检索和经验文档，手牌结束后分别更新两类记忆。
输入与输出：输入观察与轨迹，输出合并 MemoryContext 和组合快照。
依赖边界：组合 FactualMemory 与 ExperientialMemory，不依赖 LLM 或环境内部状态。
不负责：不让经验更新读取未来不可见信息，不决定动作。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
)
from agentmemeval.core.protocols import LLMClient
from agentmemeval.memory.experiential import ExperientialMemory
from agentmemeval.memory.factual import FactualMemory
from agentmemeval.memory.rag import EmbeddingBackend

MEMORY_SEMANTICS_MODES = {"quality_filtered_v2", "paper_faithful_v1"}


class FactExprSyncMemory:
    """
    功能：组合事实和经验两类记忆。
    参数：
        agent_id：所属 Agent。
        scope：记忆作用域。
        top_k：事实检索条数。
        window_size：经验窗口。
        semantics_mode：论文忠实或质量过滤语义。
    返回：同步组合记忆。
    副作用：手牌结束时更新两个子记忆。
    异常：无。
    设计说明：两个子记忆并行更新，经验修订只接收本手已准入的可见事实。
    """

    name = "fact_expr_sync"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        top_k: int = 8,
        window_size: int = 8,
        max_records: int | None = 500,
        retrieval_backend: str = "hybrid_rag",
        embedding_backend: EmbeddingBackend | None = None,
        revision_strategy: str = "deterministic",
        llm_client: LLMClient | None = None,
        model: str = "",
        fact_options: dict[str, object] | None = None,
        semantics_mode: str = "quality_filtered_v2",
        max_chars: int = 2400,
        prompt_token_budget: int = 32768,
        update_period: int = 1,
        revision_contract: str | None = None,
        document_contract: str = "five_section_actionable_v1",
        failure_policy: str = "fail_closed_keep",
        semantic_review_mode: str = "disabled",
        semantic_reviewer_client: LLMClient | None = None,
        semantic_reviewer_model: str = "",
    ) -> None:
        """
        功能：初始化组合记忆。
        参数：
            agent_id：所属 Agent。
            scope：作用域。
            top_k：事实检索条数。
            window_size：经验窗口。
            max_records：事实容量。
            semantics_mode：论文忠实或质量过滤语义。
        返回：无。
        副作用：创建子记忆。
        异常：无。
        设计说明：组合层只协调生命周期，算法仍在子模块内。
        """

        if semantics_mode not in MEMORY_SEMANTICS_MODES:
            raise ValueError(f"未知 memory semantics_mode：{semantics_mode}")
        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.semantics_mode = semantics_mode
        self.frozen = False
        resolved_fact_options = dict(fact_options or {})
        nested_semantics_mode = str(
            resolved_fact_options.pop("semantics_mode", semantics_mode)
        )
        if nested_semantics_mode != semantics_mode:
            raise ValueError(
                "Fact semantics_mode 与组合记忆不匹配："
                f"{nested_semantics_mode} != {semantics_mode}"
            )
        self.fact = FactualMemory(
            agent_id,
            scope=scope,
            top_k=top_k,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
            embedding_backend=embedding_backend,
            semantics_mode=semantics_mode,
            **resolved_fact_options,
        )
        self.expr = ExperientialMemory(
            agent_id,
            scope=scope,
            window_size=window_size,
            max_chars=max_chars,
            prompt_token_budget=prompt_token_budget,
            update_period=update_period,
            revision_strategy=revision_strategy,
            llm_client=llm_client,
            model=model,
            semantics_mode=semantics_mode,
            revision_contract=revision_contract,
            document_contract=document_contract,
            failure_policy=failure_policy,
            semantic_review_mode=semantic_review_mode,
            semantic_reviewer_client=semantic_reviewer_client,
            semantic_reviewer_model=semantic_reviewer_model,
        )

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：合并事实检索和经验文档。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：更新事实记忆的最近检索记录。
        异常：无。
        设计说明：合并时保留两个子机制元数据，便于报告追踪。
        """

        fact_context = self.fact.build_context(observation)
        expr_context = self.expr.build_context(observation)
        return MemoryContext(
            facts=fact_context.facts,
            experience=expr_context.experience,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "semantics_mode": self.semantics_mode,
                "fact": fact_context.metadata,
                "expr": expr_context.metadata,
            },
        )

    def bind_persistence(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path,
    ) -> None:
        """Bind canonical Fact and Expr artifacts side by side for this Agent."""

        if self.frozen:
            raise RuntimeError("frozen Sync memory cannot bind writable persistence")
        self.fact.bind_persistence(directory, reference_root=reference_root)
        self.expr.bind_persistence(directory, reference_root=reference_root)

    def facts_reference(self) -> dict[str, object]:
        """Expose the Fact child's compact canonical-document reference."""

        return self.fact.facts_reference()

    def experience_reference(self) -> dict[str, object]:
        """Expose the Expr child's compact canonical-document reference."""

        return self.expr.experience_reference()

    def set_frozen(self, frozen: bool) -> None:
        """Freeze both child memories for held-out evaluation."""

        if frozen:
            self.fact._assert_canonical_facts_current()
            self.expr._assert_canonical_document_current()
        self.frozen = bool(frozen)
        self.fact.set_frozen(self.frozen)
        self.expr.set_frozen(self.frozen)

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：把决策事件转发给子记忆。
        参数：
            event：决策事件。
        返回：无。
        副作用：当前子记忆默认不即时写入。
        异常：无。
        设计说明：统一生命周期，便于未来加入短期缓存机制。
        """

        self.fact.on_decision_committed(event)
        self.expr.on_decision_committed(event)

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：手牌结束后同步更新事实和经验。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：写入事实记录并可能产生经验新版本。
        异常：无。
        设计说明：事实先执行质量准入，经验修订仅使用本手新准入的可见事实。
        """

        if self.frozen:
            return
        self.expr.record_trajectory(trajectory)
        before_fact_ids = {record.record_id for record in self.fact.records}
        self.fact.on_hand_finished(trajectory)
        if self.semantics_mode == "paper_faithful_v1":
            if trajectory.decision_events:
                self.expr.on_hand_finished(trajectory)
            return
        if self.fact.last_admission_status == "admitted":
            admitted_facts = [
                record
                for record in self.fact.records
                if record.record_id not in before_fact_ids
            ]
            self.expr.on_hand_finished(trajectory, evidence_records=admitted_facts)

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出组合记忆快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：子快照 payload 嵌套保存，恢复时仍走子模块 restore。
        """

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "fact": deepcopy(self.fact.snapshot().payload),
                "expr": deepcopy(self.expr.snapshot().payload),
                "semantics_mode": self.semantics_mode,
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：恢复组合记忆。
        参数：
            snapshot：组合快照。
        返回：无。
        副作用：恢复两个子记忆状态。
        异常：无。
        设计说明：子快照机制名称显式写回，保证 restore 边界清晰。
        """

        if self.frozen:
            raise RuntimeError("frozen Sync memory cannot restore writable state")
        if snapshot.mechanism != self.name:
            raise ValueError(
                f"Sync snapshot mechanism mismatch: {snapshot.mechanism} != {self.name}"
            )
        if snapshot.agent_id != self.agent_id:
            raise ValueError(
                f"Sync snapshot agent_id mismatch: {snapshot.agent_id} != {self.agent_id}"
            )
        if snapshot.scope != self.scope:
            raise ValueError(
                f"Sync snapshot scope mismatch: {snapshot.scope} != {self.scope}"
            )
        if not isinstance(snapshot.payload, dict):
            raise ValueError("Sync snapshot payload must be an object")
        raw_fact_payload = snapshot.payload.get("fact")
        raw_expr_payload = snapshot.payload.get("expr")
        if not isinstance(raw_fact_payload, dict) or not isinstance(
            raw_expr_payload, dict
        ):
            raise ValueError("Sync snapshot fact/expr payloads must be objects")
        snapshot_mode = snapshot.payload.get("semantics_mode", self.semantics_mode)
        if not isinstance(snapshot_mode, str):
            raise ValueError("Sync snapshot semantics_mode must be a string")
        if snapshot_mode not in MEMORY_SEMANTICS_MODES:
            raise ValueError(f"快照含未知 memory semantics_mode：{snapshot_mode}")
        if snapshot_mode != self.semantics_mode:
            raise ValueError(
                "memory semantics_mode 不匹配："
                f"instance={self.semantics_mode}, snapshot={snapshot_mode}"
            )
        restored_fact = deepcopy(self.fact)
        restored_expr = deepcopy(self.expr)
        persisted_experience_path = restored_expr.experience_path
        restored_expr.experience_path = None
        restored_fact.restore(
            MemorySnapshot(
                "fact",
                self.agent_id,
                self.scope,
                deepcopy(raw_fact_payload),
            )
        )
        restored_expr.restore(
            MemorySnapshot(
                "expr",
                self.agent_id,
                self.scope,
                deepcopy(raw_expr_payload),
            )
        )
        restored_expr.experience_path = persisted_experience_path
        self.expr._atomic_write_experience(restored_expr.current.body)
        self.fact = restored_fact
        self.expr = restored_expr

    def metrics(self) -> dict[str, object]:
        """
        功能：返回组合记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：聚合事实数量和经验更新次数，保留机制名称。
        """

        fact = self.fact.metrics()
        expr = self.expr.metrics()
        return {
            "mechanism": self.name,
            "semantics_mode": self.semantics_mode,
            "fact_count": fact["fact_count"],
            "eligible_fact_count": fact.get("eligible_fact_count", fact["fact_count"]),
            "excluded_fallback_fact_count": fact.get("excluded_fallback_fact_count", 0),
            "experience_updates": expr["experience_updates"],
            "experience_version": expr.get("experience_version", 1),
            "skipped_fallback_trajectories": expr.get("skipped_fallback_trajectories", 0),
            "experience_chars": expr.get("experience_chars", 0),
            "revision_count": expr.get("revision_count", 0),
            "revision_strategy": expr.get("revision_strategy"),
            "revision_model": expr.get("revision_model", ""),
            "prompt_token_budget": expr.get("prompt_token_budget", 0),
            "revision_fallback_count": expr.get("revision_fallback_count", 0),
            "revision_attempt_count": expr.get("revision_attempt_count", 0),
            "revision_applied_count": expr.get("revision_applied_count", 0),
            "revision_keep_count": expr.get("revision_keep_count", 0),
            "revision_failed_count": expr.get("revision_failed_count", 0),
            "revision_failure_count": expr.get("revision_failure_count", 0),
            "revision_rejected_count": expr.get("revision_rejected_count", 0),
            "revision_validation_rejection_count": expr.get(
                "revision_validation_rejection_count", 0
            ),
            "revision_rejection_diagnostic_count": expr.get(
                "revision_rejection_diagnostic_count", 0
            ),
            "revision_rejection_missing_diagnostic_count": expr.get(
                "revision_rejection_missing_diagnostic_count", 0
            ),
            "revision_repair_initial_rejection_count": expr.get(
                "revision_repair_initial_rejection_count", 0
            ),
            "revision_repair_initial_diagnostic_count": expr.get(
                "revision_repair_initial_diagnostic_count", 0
            ),
            "revision_repair_initial_missing_diagnostic_count": expr.get(
                "revision_repair_initial_missing_diagnostic_count", 0
            ),
            "revision_contract_violation_count": expr.get(
                "revision_contract_violation_count", 0
            ),
            "revision_repair_count": expr.get("revision_repair_count", 0),
            "experience_document_valid": expr.get("experience_document_valid", False),
            "experience_document_validation_errors": expr.get(
                "experience_document_validation_errors", []
            ),
            "experience_document_validation_findings": expr.get(
                "experience_document_validation_findings", []
            ),
            "experience_document_advisory_findings": expr.get(
                "experience_document_advisory_findings", []
            ),
            "experience_document_advisory_count": expr.get(
                "experience_document_advisory_count", 0
            ),
            "experience_rule_counts": expr.get("experience_rule_counts", {}),
            "experience_rule_count": expr.get("experience_rule_count", 0),
            "experience_near_duplicate_advisory_count": expr.get(
                "experience_near_duplicate_advisory_count", 0
            ),
            "experience_context_build_count": expr.get(
                "experience_context_build_count", 0
            ),
            "last_injected_experience_version": expr.get(
                "last_injected_experience_version"
            ),
            "experience_injection_current": expr.get(
                "experience_injection_current", False
            ),
            "revision_contract": expr.get("revision_contract"),
            "document_contract": expr.get("document_contract"),
            "failure_policy": expr.get("failure_policy"),
            "last_revision": expr.get("last_revision", {}),
            "last_retrieved_fact_ids": fact.get("last_retrieved_fact_ids", []),
        }
