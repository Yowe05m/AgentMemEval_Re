"""
模块说明：本模块实现事实即时写入、经验周期性巩固的异步组合机制。
核心职责：每手写入事实，按配置周期使用近期轨迹和事实证据更新经验文档。
输入与输出：输入观察与轨迹，输出合并 MemoryContext、快照和异步日志。
依赖边界：组合事实与经验模块，不依赖真实 LLM；巩固证据来自可见事实记录。
不负责：不启动后台线程，不在决策中写入未来信息。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
    utc_now_iso,
)
from agentmemeval.core.protocols import LLMClient
from agentmemeval.core.seeds import make_rng
from agentmemeval.memory.base import filter_records_by_scope, trajectory_quality
from agentmemeval.memory.experiential import (
    EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
    ExperientialMemory,
    experience_failure_diagnostic,
    summarize_experience_proposal,
)
from agentmemeval.memory.factual import FactualMemory, _score_summary
from agentmemeval.memory.rag import (
    EmbeddingBackend,
    build_retrieval_query,
    hybrid_top_k_records,
    retrieval_document_template,
)
from agentmemeval.memory.retrievers import top_k_records

MEMORY_SEMANTICS_MODES = {"quality_filtered_v2", "paper_faithful_v1"}
PAPER_DECISION_MAIN_K = 14
PAPER_DECISION_TOTAL_K = 15
PAPER_MIRROR_CANDIDATE_K = 5
PAPER_MIRROR_PROB = 0.30
PAPER_SWEEP_SIMILAR_K = 20
PAPER_SWEEP_DIVERSE_PER_BUCKET = 2
PAPER_SWEEP_IMPORTANT_K = 10
PAPER_SWEEP_TOTAL_K = 30
PAPER_FAITHFUL_APPLIED_CLASSIFICATION_STATUS = "paper_faithful_provisional_applied"
PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS = "pending_human_review"
NO_VALID_STABILITY_PROPOSAL_CLASSIFICATION_STATUS = "no_valid_stability_proposal"


class FactExprAsyncMemory:
    """
    功能：实现 FactExprAsync 的周期性经验巩固。
    参数：
        agent_id：所属 Agent。
        scope：作用域。
        top_k：决策时事实检索条数。
        window_size：近期轨迹窗口。
        sweep_every：异步巩固周期。
        evidence_k：巩固时事实证据数量。
        semantics_mode：论文忠实或质量过滤语义。
    返回：异步组合记忆。
    副作用：手牌结束时写事实，周期触发经验版本。
    异常：无。
    设计说明：异步是逻辑周期而非线程，保证离线复现实验确定性。
    """

    name = "fact_expr_async"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        top_k: int = 8,
        window_size: int = 8,
        sweep_every: int = 3,
        evidence_k: int = 6,
        max_records: int | None = 500,
        salience_threshold: float = 0.03,
        salience_mirror_threshold: float = 0.30,
        mirror_prob: float = 0.20,
        stability_init: float = 10.0,
        stability_min: float = 0.5,
        stability_max: float = 50.0,
        embedding_backend: EmbeddingBackend | None = None,
        revision_strategy: str = "deterministic",
        llm_client: LLMClient | None = None,
        model: str = "",
        fact_options: dict[str, object] | None = None,
        semantics_mode: str = "quality_filtered_v2",
        max_chars: int = 2400,
        prompt_token_budget: int = 32768,
        revision_contract: str | None = None,
        document_contract: str = "five_section_actionable_v1",
        failure_policy: str = "fail_closed_keep",
        semantic_review_mode: str = "disabled",
        semantic_reviewer_client: LLMClient | None = None,
        semantic_reviewer_model: str = "",
    ) -> None:
        """
        功能：初始化异步组合记忆。
        参数：
            agent_id：所属 Agent。
            scope：作用域。
            top_k：决策检索条数。
            window_size：近期轨迹窗口。
            sweep_every：巩固周期。
            evidence_k：证据数量。
            max_records：事实容量。
            semantics_mode：论文忠实或质量过滤语义。
        返回：无。
        副作用：创建子记忆和 sweep 日志。
        异常：无。
        设计说明：所有周期参数都来自配置，便于做消融。
        """

        if semantics_mode not in MEMORY_SEMANTICS_MODES:
            raise ValueError(f"未知 memory semantics_mode：{semantics_mode}")
        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.semantics_mode = semantics_mode
        resolved_fact_options = dict(fact_options or {})
        nested_semantics_mode = str(resolved_fact_options.pop("semantics_mode", semantics_mode))
        if nested_semantics_mode != semantics_mode:
            raise ValueError(
                f"Fact semantics_mode 与组合记忆不匹配：{nested_semantics_mode} != {semantics_mode}"
            )
        resolved_top_k = PAPER_DECISION_MAIN_K if semantics_mode == "paper_faithful_v1" else top_k
        self.fact = FactualMemory(
            agent_id,
            scope=scope,
            top_k=resolved_top_k,
            max_records=max_records,
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
            update_period=10**9,
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
        self.window_size = window_size
        self.sweep_every = sweep_every
        self.evidence_k = evidence_k
        self.salience_threshold = salience_threshold
        self.salience_mirror_threshold = salience_mirror_threshold
        self.mirror_prob = (
            PAPER_MIRROR_PROB
            if semantics_mode == "paper_faithful_v1" and mirror_prob == 0.20
            else mirror_prob
        )
        self.stability_init = stability_init
        self.stability_min = stability_min
        self.stability_max = stability_max
        _validate_async_parameters(
            window_size=self.window_size,
            sweep_every=self.sweep_every,
            evidence_k=self.evidence_k,
            salience_threshold=self.salience_threshold,
            salience_mirror_threshold=self.salience_mirror_threshold,
            mirror_prob=self.mirror_prob,
            stability_init=self.stability_init,
            stability_min=self.stability_min,
            stability_max=self.stability_max,
        )
        self.recent: list[HandTrajectory] = []
        self.sweep_log: list[dict[str, object]] = []
        self.evidence_review_queue: list[dict[str, object]] = []
        self.fact_state: dict[str, dict[str, object]] = {}
        self.hand_counter = 0
        self.eligible_hand_counter = 0
        self.eligible_hands_since_sweep = 0
        self.discarded_partial_sweep_eligible_hands = 0
        self.skipped_trajectory_hand_ids: list[str] = []
        self.persistence_directory: Path | None = None
        self.persistence_reference_root: Path | None = None
        self.fact_state_path: Path | None = None
        self.sweep_log_path: Path | None = None
        self.frozen = False

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：合并当前事实检索结果与经验文档。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：更新事实检索元数据。
        异常：无。
        设计说明：显著性先筛选并参与排序，再复用 Fact 的门槛、去重和 top-k 合同。
        """

        self._assert_async_persistence_current()
        scoped_candidates = self._scoped_eligible_records(observation)
        candidates = [
            record
            for record in scoped_candidates
            if self._salience(record.record_id, self.hand_counter + 1) >= self.salience_threshold
        ]
        retrieval_query = build_retrieval_query(observation)
        active_document_template = (
            "{record.features}"
            if self.fact.retrieval_backend == "feature_jaccard"
            else retrieval_document_template(
                self.fact.embedding_backend,
                self.fact.retrieval_unit,
            )
        )
        scored_records: list[tuple[object, float, dict[str, object]]] = []
        if self.fact.retrieval_backend == "feature_jaccard":
            base_scores = top_k_records(observation, candidates, len(candidates))
            for record, feature_score in base_scores:
                salience = self._salience(record.record_id, self.hand_counter + 1)
                score = float(feature_score) + math.log(max(salience, 1e-6))
                scored_records.append(
                    (
                        record,
                        score,
                        {
                            "record_id": record.record_id,
                            "score": score,
                            "feature": float(feature_score),
                            "semantic": None,
                            "salience": salience,
                            "retrieval_unit": self.fact.retrieval_unit,
                            "matched_decision_index": None,
                            "matched_phase": None,
                        },
                    )
                )
            scored_records.sort(
                key=lambda item: (item[1], item[0].created_at, item[0].record_id),
                reverse=True,
            )
        else:
            scored_rag = hybrid_top_k_records(
                observation,
                candidates,
                len(candidates),
                semantic_weight=self.fact.semantic_weight,
                feature_weight=self.fact.feature_weight,
                salience_fn=lambda record_id: self._salience(record_id, self.hand_counter + 1),
                embedding_backend=self.fact.embedding_backend,
                retrieval_unit=self.fact.retrieval_unit,
            )
            scored_records = [
                (
                    item.record,
                    item.score,
                    {
                        "record_id": item.record.record_id,
                        "score": item.score,
                        "semantic": item.semantic,
                        "dense": item.dense,
                        "sparse": item.sparse,
                        "colbert": item.colbert,
                        "feature": item.feature,
                        "salience": item.salience,
                        "retrieval_unit": item.retrieval_unit,
                        "matched_decision_index": item.matched_decision_index,
                        "matched_phase": item.matched_phase,
                    },
                )
                for item in scored_rag
            ]
        main_records, mirror_overflow, below_threshold, duplicate_excluded = (
            self._select_fact_records(scored_records)
        )
        mirror_record, mirror_candidate_count = self._mirror_record(
            observation,
            mirror_overflow,
        )
        selected_records = list(main_records)
        if mirror_record is not None:
            selected_records.append(mirror_record)
        if self.semantics_mode == "paper_faithful_v1":
            selected_records = selected_records[:PAPER_DECISION_TOTAL_K]
        retrieved = [record for record, _score, _metadata in selected_records]
        self.fact.last_scores = [
            {
                **metadata,
                "retrieval_path": ("seeded_mirror" if mirror_record is item else "salience_main"),
            }
            for item in selected_records
            for _record, _score, metadata in [item]
        ]
        self.fact.retrieval_request_count += 1
        self.fact.retrieval_below_threshold_count += below_threshold
        self.fact.retrieval_duplicate_excluded_count += duplicate_excluded
        if not retrieved:
            self.fact.empty_retrieval_count += 1
        score_summary = _score_summary([score for _record, score, _metadata in scored_records])
        audit = {
            "hand_id": observation.hand_id,
            "query": retrieval_query,
            "retrieval_backend": self.fact.retrieval_backend,
            "retrieval_unit": self.fact.retrieval_unit,
            "document_template": active_document_template,
            "retrieval_policy": "async_salience_mirror_v1",
            "scoped_candidate_count": len(scoped_candidates),
            "candidate_count": len(candidates),
            "salience_excluded_count": len(scoped_candidates) - len(candidates),
            "returned_count": len(retrieved),
            "main_returned_count": len(main_records),
            "mirror_candidate_count": mirror_candidate_count,
            "mirror_injected": mirror_record[0].record_id if mirror_record else None,
            "below_threshold_count": below_threshold,
            "duplicate_signature_excluded_count": duplicate_excluded,
            "score_summary": score_summary,
        }
        self.fact.retrieval_audit_log.append(audit)
        self.fact.retrieval_audit_log = self.fact.retrieval_audit_log[
            -self.fact.admission_log_limit :
        ]
        self.fact.last_retrieval = [record.record_id for record in retrieved]
        if not self.frozen:
            candidate_state = deepcopy(self.fact_state)
            for record in retrieved:
                state = candidate_state.get(record.record_id)
                if state is not None:
                    state["last_accessed_hand"] = self.hand_counter + 1
                    state["access_count"] = int(state.get("access_count", 0)) + 1
            self._write_fact_state_candidate(
                candidate_state,
                hand_counter=self.hand_counter + 1,
            )
            self.fact_state = candidate_state
        expr_context = self.expr.build_context(observation)
        return MemoryContext(
            facts=retrieved,
            experience=expr_context.experience,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "semantics_mode": self.semantics_mode,
                "frozen": self.frozen,
                "fact": {
                    "mechanism": self.fact.name,
                    "scope": self.scope,
                    "retrieval_backend": self.fact.retrieval_backend,
                    "retrieval_unit": self.fact.retrieval_unit,
                    "query": retrieval_query,
                    "document_template": active_document_template,
                    "retrieval_policy": "async_salience_mirror_v1",
                    "retrieved_fact_ids": list(self.fact.last_retrieval),
                    "retrieval_scores": list(self.fact.last_scores),
                    "scoped_candidate_count": len(scoped_candidates),
                    "candidate_count": len(candidates),
                    "salience_excluded_count": len(scoped_candidates) - len(candidates),
                    "returned_count": len(retrieved),
                    "main_returned_count": len(main_records),
                    "mirror_candidate_count": mirror_candidate_count,
                    "mirror_injected": mirror_record[0].record_id if mirror_record else None,
                    "below_threshold_count": below_threshold,
                    "duplicate_signature_excluded_count": duplicate_excluded,
                    "minimum_retrieval_score": self.fact.minimum_retrieval_score,
                    "retrieval_threshold_status": self.fact.retrieval_threshold_status,
                    "candidate_score_summary": score_summary,
                    "salience_threshold": self.salience_threshold,
                    "salience_mirror_threshold": self.salience_mirror_threshold,
                    "excluded_fallback_fact_count": self.fact.admission_counts.get(
                        "reason:provider_fallback", 0
                    ),
                    "embedding": self.fact.embedding_backend.audit_metadata(),
                },
                "expr": expr_context.metadata,
                "sweep_every": self.sweep_every,
            },
        )

    def bind_persistence(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path,
    ) -> None:
        """Bind Fact, Expr, current state, and append-only sweep artifacts."""

        if self.frozen:
            raise RuntimeError("frozen Async memory cannot bind writable persistence")
        resolved_directory, resolved_root = _resolve_async_persistence_paths(
            directory,
            reference_root,
        )
        state_path = resolved_directory / "facts_state.jsonl"
        sweep_path = resolved_directory / "sweep_log.jsonl"
        expected_state = self._canonical_fact_state_bytes()
        expected_sweeps = _canonical_sweep_log_bytes(self.sweep_log)
        for path, expected in (
            (state_path, expected_state),
            (sweep_path, expected_sweeps),
        ):
            if path.is_symlink():
                raise ValueError(f"{path.name} must not be a symbolic link")
            if path.exists():
                _assert_regular_exact_file(path, expected)

        self.fact.bind_persistence(resolved_directory, reference_root=resolved_root)
        self.expr.bind_persistence(resolved_directory, reference_root=resolved_root)
        resolved_directory = self.fact.persistence_directory or resolved_directory
        resolved_root = self.fact.persistence_reference_root or resolved_root
        state_path = resolved_directory / "facts_state.jsonl"
        sweep_path = resolved_directory / "sweep_log.jsonl"
        for path, expected in (
            (state_path, expected_state),
            (sweep_path, expected_sweeps),
        ):
            if not path.exists():
                _materialize_new_exact_file(path, expected)
            else:
                _assert_regular_exact_file(path, expected)
        self.persistence_directory = resolved_directory
        self.persistence_reference_root = resolved_root
        self.fact_state_path = state_path
        self.sweep_log_path = sweep_path
        self._assert_async_persistence_current()

    def facts_reference(self) -> dict[str, object]:
        """Expose the Fact child's compact canonical-document reference."""

        return self.fact.facts_reference()

    def facts_state_reference(self) -> dict[str, object]:
        """Return a compact pointer to the atomic current-state view."""

        payload = self._canonical_fact_state_bytes()
        relative_path: str | None = None
        if self.fact_state_path is not None:
            self._assert_async_persistence_current()
            if self.persistence_reference_root is None:  # pragma: no cover - invariant guard
                raise RuntimeError("Async persistence root is missing")
            relative_path = self.fact_state_path.relative_to(
                self.persistence_reference_root
            ).as_posix()
        return {
            "path": relative_path,
            "record_count": len(self.fact_state),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def sweep_log_reference(self) -> dict[str, object]:
        """Return a compact pointer to the append-only sweep audit."""

        payload = _canonical_sweep_log_bytes(self.sweep_log)
        relative_path: str | None = None
        if self.sweep_log_path is not None:
            self._assert_async_persistence_current()
            if self.persistence_reference_root is None:  # pragma: no cover - invariant guard
                raise RuntimeError("Async persistence root is missing")
            relative_path = self.sweep_log_path.relative_to(
                self.persistence_reference_root
            ).as_posix()
        return {
            "path": relative_path,
            "entry_count": len(self.sweep_log),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def experience_reference(self) -> dict[str, object]:
        """Expose the Expr child's compact canonical-document reference."""

        return self.expr.experience_reference()

    def set_frozen(self, frozen: bool = True) -> None:
        """Freeze or unfreeze all long-term Async memory evolution."""

        if frozen:
            self._assert_async_persistence_current()
        self.frozen = bool(frozen)
        for child in (self.fact, self.expr):
            setter = getattr(child, "set_frozen", None)
            if callable(setter):
                setter(self.frozen)

    def _canonical_fact_state_bytes(
        self,
        state: dict[str, dict[str, object]] | None = None,
        *,
        hand_counter: int | None = None,
        experience_version: int | None = None,
    ) -> bytes:
        """Render the complete validated state table as deterministic JSONL."""

        candidate = self.fact_state if state is None else state
        active_fact_ids = {
            record.record_id
            for record in self.fact.records
            if record.source.get("memory_eligible", True)
        }
        resolved_hand_counter = self.hand_counter if hand_counter is None else hand_counter
        if candidate:
            resolved_hand_counter = max(
                resolved_hand_counter,
                max(int(item.get("last_accessed_hand", 0)) for item in candidate.values()),
            )
        validated = _validated_restored_fact_state(
            candidate,
            active_fact_ids=active_fact_ids,
            hand_counter=resolved_hand_counter,
            stability_min=self.stability_min,
            stability_max=self.stability_max,
            max_experience_version=(
                self.expr.current.version
                if experience_version is None
                else experience_version
            ),
        )
        return _canonical_fact_state_bytes(validated)

    def _write_fact_state_candidate(
        self,
        state: dict[str, dict[str, object]],
        *,
        hand_counter: int,
        experience_version: int | None = None,
        expected_current: bytes | None = None,
    ) -> None:
        """Atomically replace only the current-state view, never its fact history."""

        if self.fact_state_path is None:
            return
        if self.frozen:
            raise RuntimeError("frozen Async memory cannot update facts_state.jsonl")
        expected = (
            self._canonical_fact_state_bytes()
            if expected_current is None
            else expected_current
        )
        _assert_regular_exact_file(self.fact_state_path, expected)
        candidate = self._canonical_fact_state_bytes(
            state,
            hand_counter=hand_counter,
            experience_version=experience_version,
        )
        _atomic_replace_exact_file(self.fact_state_path, candidate)

    def _append_persisted_sweep(self, entry: dict[str, object]) -> None:
        """Append one completed sweep through an atomic prefix-preserving replace."""

        if self.sweep_log_path is None:
            return
        if self.frozen:
            raise RuntimeError("frozen Async memory cannot append sweep_log.jsonl")
        current = _canonical_sweep_log_bytes(self.sweep_log)
        _assert_regular_exact_file(self.sweep_log_path, current)
        candidate = current + _canonical_json_line(_compact_sweep_log_entry(entry))
        _atomic_replace_exact_file(self.sweep_log_path, candidate)

    def _assert_async_persistence_current(self) -> None:
        """Fail closed if any bound Async long-term artifact has diverged."""

        self.fact._assert_canonical_facts_current()
        self.expr._assert_canonical_document_current()
        if self.fact_state_path is not None:
            _assert_regular_exact_file(
                self.fact_state_path,
                self._canonical_fact_state_bytes(),
            )
        if self.sweep_log_path is not None:
            _assert_regular_exact_file(
                self.sweep_log_path,
                _canonical_sweep_log_bytes(self.sweep_log),
            )

    def _rollback_fact_checkpoint(
        self,
        snapshot: MemorySnapshot,
        canonical_bytes: bytes,
    ) -> None:
        """Roll back an uncommitted fact when its required state row cannot persist."""

        if self.fact.facts_path is not None:
            self.fact._atomic_replace_facts_file(canonical_bytes)
        self.fact.restore(snapshot)

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：接收已提交决策。
        参数：
            event：决策事件。
        返回：无。
        副作用：无。
        异常：无。
        设计说明：异步巩固也只在手牌结束时触发。
        """

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：写事实并在周期到达时巩固经验。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：追加事实、更新近期窗口、可能追加经验版本和 sweep 日志。
        异常：无。
        设计说明：sweep 日志记录触发手牌、窗口、证据 ID 和版本变化。
        """

        if self.frozen:
            return

        self._assert_async_persistence_current()
        self.expr.record_trajectory(trajectory)
        quality = trajectory_quality(trajectory)
        fact_snapshot_before = self.fact.snapshot()
        fact_bytes_before = self.fact._assert_canonical_facts_current()
        state_bytes_before = self._canonical_fact_state_bytes()
        before_ids = {record.record_id for record in self.fact.records}
        self.fact.on_hand_finished(trajectory)
        candidate_state = deepcopy(self.fact_state)
        for record in self.fact.records:
            if (
                record.record_id not in before_ids
                and record.record_id not in candidate_state
                and record.source.get("memory_eligible", True)
            ):
                candidate_state[record.record_id] = {
                    "stability": self.stability_init,
                    "last_accessed_hand": self.hand_counter + 1,
                    "access_count": 0,
                    "linked_exp_revs": [],
                }
        active_fact_ids = {
            record.record_id
            for record in self.fact.records
            if record.source.get("memory_eligible", True)
        }
        candidate_state = {
            record_id: state
            for record_id, state in candidate_state.items()
            if record_id in active_fact_ids
        }
        candidate_hand_counter = self.hand_counter + 1
        try:
            self._write_fact_state_candidate(
                candidate_state,
                hand_counter=candidate_hand_counter,
                expected_current=state_bytes_before,
            )
        except Exception as exc:
            try:
                self._rollback_fact_checkpoint(fact_snapshot_before, fact_bytes_before)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O boundary
                raise RuntimeError(
                    "Async fact-state write failed and fact rollback also failed: "
                    f"{type(exc).__name__}"
                ) from rollback_exc
            raise
        self.fact_state = candidate_state
        self.hand_counter = candidate_hand_counter
        if not trajectory.decision_events or (
            self.semantics_mode != "paper_faithful_v1" and not quality["memory_eligible"]
        ):
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        if (
            self.semantics_mode == "quality_filtered_v2"
            and self.fact.last_admission_status != "admitted"
        ):
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        self.eligible_hand_counter += 1
        self.eligible_hands_since_sweep += 1
        self.recent.append(trajectory)
        self.recent = self.recent[-self.window_size :]
        if self.eligible_hands_since_sweep < self.sweep_every:
            return
        self.eligible_hands_since_sweep = 0
        sweep_recent = self.recent[-self.sweep_every :]
        trigger_observation = trajectory.decision_events[-1].observation
        evidence_groups = self._recall_for_sweep(trigger_observation)
        evidence_by_id = {
            record.record_id: record for records in evidence_groups.values() for record in records
        }
        evidence = list(evidence_by_id.values())
        old_version = self.expr.current.version
        sweep_state_before = deepcopy(self.fact_state)
        sweep_state_bytes_before = self._canonical_fact_state_bytes()
        sweep_bytes_before = _canonical_sweep_log_bytes(self.sweep_log)
        review_queue_before = deepcopy(self.evidence_review_queue)
        expr_snapshot_before = self.expr.snapshot()
        expr_trajectories_before = deepcopy(self.expr.trajectories)
        experience_log_path = (
            self.expr.experience_log_store.path
            if self.expr.experience_log_store is not None
            else None
        )
        experience_log_bytes_before = (
            experience_log_path.read_bytes()
            if experience_log_path is not None and experience_log_path.exists()
            else None
        )
        revision = self._sweep(
            evidence,
            recent=sweep_recent,
            evidence_groups=(
                evidence_groups if self.semantics_mode == "paper_faithful_v1" else None
            ),
        )
        new_version = self.expr.current.version
        revision_failure = (
            experience_failure_diagnostic(revision.get("failure"))
            if revision.get("failure") is not None
            else None
        )
        sweep_entry = {
                "trigger_hand_id": trajectory.hand_id,
                "semantics_mode": self.semantics_mode,
                "recent_window_hand_ids": [item.hand_id for item in sweep_recent],
                "evidence_fact_ids": [item.record_id for item in evidence],
                "evidence_groups": {
                    name: [record.record_id for record in records]
                    for name, records in evidence_groups.items()
                },
                "old_experience_version": old_version,
                "new_experience_version": new_version,
                "supporting_fact_ids": revision["supporting_fact_ids"],
                "contradicting_fact_ids": revision["contradicting_fact_ids"],
                "noise_fact_ids": revision["noise_fact_ids"],
                "irrelevant_fact_ids": revision.get("irrelevant_fact_ids", []),
                "ambiguous_fact_ids": revision.get("ambiguous_fact_ids", []),
                "revision_keep": bool(revision.get("keep", True)),
                "revision_fallback_used": bool(revision.get("fallback_used", False)),
                "experience_revision_created": bool(
                    revision.get("experience_revision_created", False)
                ),
                "revision_application_status": revision.get("application_status"),
                "revision_validation_status": revision.get("validation_status"),
                "revision_failure": (
                    revision_failure["preview"] if revision_failure is not None else None
                ),
                "revision_failure_code": (
                    revision_failure["code"] if revision_failure is not None else None
                ),
                "revision_failure_sha256": (
                    revision_failure["sha256"] if revision_failure is not None else None
                ),
                "revision_failure_chars": (
                    revision_failure["chars"] if revision_failure is not None else 0
                ),
                "revision_validation_issue_codes": [
                    str(item.get("code", ""))
                    for item in (revision.get("validation_findings", []) or [])
                    if isinstance(item, dict)
                ],
                "revision_initial_validation_issue_codes": [
                    str(item.get("code", ""))
                    for item in (revision.get("initial_validation_findings", []) or [])
                    if isinstance(item, dict)
                ],
                "revision_proposal_summary": revision.get("proposal_summary", {}),
                "revision_initial_proposal_summary": revision.get("initial_proposal_summary", {}),
                "evidence_classification_status": revision.get(
                    "evidence_classification_status", "not_requested"
                ),
                "human_review_required": bool(
                    revision.get("human_review_required", False)
                ),
                "stability_application_policy": revision.get(
                    "stability_application_policy", "no_valid_stability_proposal"
                ),
                "proposed_stability_adjustments": revision.get(
                    "proposed_stability_adjustments", []
                ),
                "created_at": utc_now_iso(),
            }
        try:
            self._write_fact_state_candidate(
                self.fact_state,
                hand_counter=self.hand_counter,
                experience_version=new_version,
                expected_current=sweep_state_bytes_before,
            )
            self._append_persisted_sweep(sweep_entry)
        except Exception as exc:
            try:
                if self.fact_state_path is not None:
                    _atomic_replace_exact_file(
                        self.fact_state_path,
                        sweep_state_bytes_before,
                    )
                if self.sweep_log_path is not None:
                    _atomic_replace_exact_file(self.sweep_log_path, sweep_bytes_before)
                if experience_log_path is not None and experience_log_bytes_before is not None:
                    _atomic_replace_exact_file(
                        experience_log_path,
                        experience_log_bytes_before,
                    )
                self.expr.restore(expr_snapshot_before)
                self.expr.trajectories = expr_trajectories_before
                self.fact_state = sweep_state_before
                self.evidence_review_queue = review_queue_before
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O boundary
                raise RuntimeError(
                    "Async sweep persistence failed and rollback also failed: "
                    f"{type(exc).__name__}"
                ) from rollback_exc
            raise
        self.sweep_log.append(sweep_entry)

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出异步组合快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：额外保存 sweep 日志，支持审计异步触发是否符合配置。
        """

        self._assert_async_persistence_current()
        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "fact": deepcopy(self.fact.snapshot().payload),
                "expr": deepcopy(self.expr.snapshot().payload),
                "semantics_mode": self.semantics_mode,
                "window_size": self.window_size,
                "sweep_every": self.sweep_every,
                "evidence_k": self.evidence_k,
                "salience_threshold": self.salience_threshold,
                "salience_mirror_threshold": self.salience_mirror_threshold,
                "mirror_prob": self.mirror_prob,
                "stability_init": self.stability_init,
                "stability_min": self.stability_min,
                "stability_max": self.stability_max,
                "sweep_log": deepcopy(self.sweep_log),
                "evidence_review_queue": deepcopy(self.evidence_review_queue),
                "fact_state": deepcopy(self.fact_state),
                "hand_counter": self.hand_counter,
                "eligible_hand_counter": self.eligible_hand_counter,
                "eligible_hands_since_sweep": self.eligible_hands_since_sweep,
                "skipped_trajectory_hand_ids": list(self.skipped_trajectory_hand_ids),
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：恢复异步组合记忆。
        参数：
            snapshot：快照。
        返回：无。
        副作用：恢复事实、经验和 sweep 元数据。
        异常：无。
        设计说明：泛化测试恢复后近期窗口清空，避免训练流水继续影响测试巩固。
        """

        if self.frozen:
            raise RuntimeError("frozen Async memory cannot restore writable state")
        if snapshot.mechanism != self.name:
            raise ValueError(
                f"memory mechanism 不匹配：instance={self.name}, snapshot={snapshot.mechanism}"
            )
        if snapshot.agent_id != self.agent_id:
            raise ValueError(
                f"memory agent_id 不匹配：instance={self.agent_id}, snapshot={snapshot.agent_id}"
            )
        if snapshot.scope != self.scope:
            raise ValueError(
                f"memory scope 不匹配：instance={self.scope}, snapshot={snapshot.scope}"
            )
        if not isinstance(snapshot.payload, dict):
            raise ValueError("Async snapshot payload 必须是对象")
        payload = snapshot.payload
        raw_fact_payload = payload.get("fact")
        raw_expr_payload = payload.get("expr")
        if not isinstance(raw_fact_payload, dict) or not isinstance(raw_expr_payload, dict):
            raise ValueError("Async snapshot fact/expr payloads 必须是对象")
        snapshot_mode = payload.get("semantics_mode", self.semantics_mode)
        if not isinstance(snapshot_mode, str):
            raise ValueError("Async snapshot semantics_mode 必须是字符串")
        if snapshot_mode not in MEMORY_SEMANTICS_MODES:
            raise ValueError(f"快照含未知 memory semantics_mode：{snapshot_mode}")
        if snapshot_mode != self.semantics_mode:
            raise ValueError(
                "memory semantics_mode 不匹配："
                f"instance={self.semantics_mode}, snapshot={snapshot_mode}"
            )
        restored_parameters = {
            "window_size": payload.get("window_size", self.window_size),
            "sweep_every": payload.get("sweep_every", self.sweep_every),
            "evidence_k": payload.get("evidence_k", self.evidence_k),
            "salience_threshold": payload.get("salience_threshold", self.salience_threshold),
            "salience_mirror_threshold": payload.get(
                "salience_mirror_threshold", self.salience_mirror_threshold
            ),
            "mirror_prob": payload.get("mirror_prob", self.mirror_prob),
            "stability_init": payload.get("stability_init", self.stability_init),
            "stability_min": payload.get("stability_min", self.stability_min),
            "stability_max": payload.get("stability_max", self.stability_max),
        }
        _validate_async_parameters(
            window_size=restored_parameters["window_size"],  # type: ignore[arg-type]
            sweep_every=restored_parameters["sweep_every"],  # type: ignore[arg-type]
            evidence_k=restored_parameters["evidence_k"],  # type: ignore[arg-type]
            salience_threshold=restored_parameters["salience_threshold"],  # type: ignore[arg-type]
            salience_mirror_threshold=restored_parameters["salience_mirror_threshold"],  # type: ignore[arg-type]
            mirror_prob=restored_parameters["mirror_prob"],  # type: ignore[arg-type]
            stability_init=restored_parameters["stability_init"],  # type: ignore[arg-type]
            stability_min=restored_parameters["stability_min"],  # type: ignore[arg-type]
            stability_max=restored_parameters["stability_max"],  # type: ignore[arg-type]
        )
        live_parameters = {
            "window_size": self.window_size,
            "sweep_every": self.sweep_every,
            "evidence_k": self.evidence_k,
            "salience_threshold": self.salience_threshold,
            "salience_mirror_threshold": self.salience_mirror_threshold,
            "mirror_prob": self.mirror_prob,
            "stability_init": self.stability_init,
            "stability_min": self.stability_min,
            "stability_max": self.stability_max,
        }
        mismatched_parameters = sorted(
            key for key, value in restored_parameters.items() if value != live_parameters[key]
        )
        if mismatched_parameters:
            raise ValueError(
                "Async snapshot semantic config mismatch: " + ", ".join(mismatched_parameters)
            )

        raw_sweep_log = payload.get("sweep_log", [])
        if not isinstance(raw_sweep_log, list) or any(
            not isinstance(item, dict) for item in raw_sweep_log
        ):
            raise ValueError("Async sweep_log 必须是对象列表")
        restored_sweep_log = [_compact_sweep_log_entry(deepcopy(item)) for item in raw_sweep_log]
        raw_review_queue = payload.get("evidence_review_queue", [])
        if not isinstance(raw_review_queue, list) or any(
            not isinstance(item, dict) for item in raw_review_queue
        ):
            raise ValueError("Async evidence_review_queue 必须是对象列表")
        restored_review_queue = [
            _normalized_evidence_review_item(deepcopy(item), semantics_mode=snapshot_mode)
            for item in raw_review_queue
        ]
        raw_fact_state = payload.get("fact_state", {})
        if not isinstance(raw_fact_state, dict):
            raise ValueError("Async fact_state 必须是对象")
        hand_counter = payload.get("hand_counter", 0)
        eligible_hand_counter = payload.get("eligible_hand_counter", hand_counter)
        if (
            type(hand_counter) is not int
            or type(eligible_hand_counter) is not int
            or hand_counter < 0
            or not 0 <= eligible_hand_counter <= hand_counter
        ):
            raise ValueError("Async hand counters 非法")
        legacy_phase = eligible_hand_counter % self.sweep_every
        snapshot_phase = payload.get("eligible_hands_since_sweep", legacy_phase)
        if type(snapshot_phase) is not int:
            raise ValueError("Async eligible_hands_since_sweep 非法")
        if not 0 <= snapshot_phase < self.sweep_every:
            raise ValueError("Async eligible_hands_since_sweep 非法")
        raw_skipped_ids = payload.get("skipped_trajectory_hand_ids", [])
        if (
            not isinstance(raw_skipped_ids, list)
            or any(not isinstance(item, str) or not item for item in raw_skipped_ids)
            or len(set(raw_skipped_ids)) != len(raw_skipped_ids)
        ):
            raise ValueError("Async skipped_trajectory_hand_ids 非法")

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
        active_fact_ids = {
            record.record_id
            for record in restored_fact.records
            if record.source.get("memory_eligible", True)
        }
        restored_fact_state = _validated_restored_fact_state(
            raw_fact_state,
            active_fact_ids=active_fact_ids,
            hand_counter=hand_counter,
            stability_min=self.stability_min,
            stability_max=self.stability_max,
            max_experience_version=restored_expr.current.version,
        )
        restored_sweep_log = _validated_restored_sweep_log(
            restored_sweep_log,
            stability_min=self.stability_min,
            stability_max=self.stability_max,
            max_experience_version=restored_expr.current.version,
        )
        if self.fact_state_path is not None:
            restored_fact_state_bytes = _canonical_fact_state_bytes(restored_fact_state)
            current_fact_state_bytes = self._canonical_fact_state_bytes()
            _assert_regular_exact_file(self.fact_state_path, current_fact_state_bytes)
            if current_fact_state_bytes:
                _assert_regular_exact_file(
                    self.fact_state_path,
                    restored_fact_state_bytes,
                )
            elif restored_fact_state_bytes:
                # A fresh formal run binds an empty state view before restoring
                # its clone snapshot.  Materialize only that verified-empty
                # destination; non-empty divergent state remains fail-closed.
                _atomic_replace_exact_file(
                    self.fact_state_path,
                    restored_fact_state_bytes,
                )
        if self.sweep_log_path is not None:
            restored_sweep_log_bytes = _canonical_sweep_log_bytes(restored_sweep_log)
            current_sweep_log_bytes = _canonical_sweep_log_bytes(self.sweep_log)
            _assert_regular_exact_file(self.sweep_log_path, current_sweep_log_bytes)
            if current_sweep_log_bytes:
                _assert_regular_exact_file(
                    self.sweep_log_path,
                    restored_sweep_log_bytes,
                )
            elif restored_sweep_log_bytes:
                # Apply the same empty-only materialization rule to the
                # append-only sweep prefix restored with the clone snapshot.
                _atomic_replace_exact_file(
                    self.sweep_log_path,
                    restored_sweep_log_bytes,
                )

        self.expr._atomic_write_experience(restored_expr.current.body)
        self.fact = restored_fact
        self.expr = restored_expr
        self.sweep_log = restored_sweep_log
        self.evidence_review_queue = restored_review_queue
        self.fact_state = restored_fact_state
        self.hand_counter = hand_counter
        self.eligible_hand_counter = eligible_hand_counter
        # Partial recent trajectories are intentionally not persisted. Reset
        # the cadence phase as well so an online restore must collect a full
        # fresh sweep window instead of firing after only the old remainder.
        self.discarded_partial_sweep_eligible_hands = snapshot_phase
        self.eligible_hands_since_sweep = 0
        self.skipped_trajectory_hand_ids = list(raw_skipped_ids)
        self.recent = []

    def metrics(self) -> dict[str, object]:
        """
        功能：返回异步记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：报告展示事实数量、经验更新和 sweep 次数。
        """

        self._assert_async_persistence_current()
        fact = self.fact.metrics()
        expr = self.expr.metrics()
        pending_review_count = sum(
            _evidence_human_review_required(item) for item in self.evidence_review_queue
        )
        paper_faithful_applied_count = sum(
            str(item.get("classification_status", ""))
            == PAPER_FAITHFUL_APPLIED_CLASSIFICATION_STATUS
            for item in self.evidence_review_queue
        )
        provisional_classification_count = sum(
            item.get("provisional_label")
            in {"supporting", "contradicting", "noise", "irrelevant", "ambiguous"}
            for item in self.evidence_review_queue
        )
        pending_stability_proposal_count = sum(
            item.get("proposed_stability") is not None
            and item.get("provisional_applied_to_stability") is not True
            for item in self.evidence_review_queue
        )
        original_baseline_applied_stability_count = sum(
            item.get("provisional_applied_to_stability") is True
            for item in self.evidence_review_queue
        )
        return {
            "mechanism": self.name,
            "semantics_mode": self.semantics_mode,
            "frozen": self.frozen,
            "fact_count": fact["fact_count"],
            "eligible_fact_count": fact.get("eligible_fact_count", fact["fact_count"]),
            "excluded_fallback_fact_count": fact.get("excluded_fallback_fact_count", 0),
            "experience_updates": expr["experience_updates"],
            "experience_version": expr.get("experience_version", 1),
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
            "revision_contract_violation_count": expr.get("revision_contract_violation_count", 0),
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
            "experience_context_build_count": expr.get("experience_context_build_count", 0),
            "last_injected_experience_version": expr.get("last_injected_experience_version"),
            "experience_injection_current": expr.get("experience_injection_current", False),
            "revision_contract": expr.get("revision_contract"),
            "document_contract": expr.get("document_contract"),
            "failure_policy": expr.get("failure_policy"),
            "last_revision": expr.get("last_revision", {}),
            "async_sweeps": len(self.sweep_log),
            "last_retrieved_fact_ids": fact.get("last_retrieved_fact_ids", []),
            "salience_threshold": self.salience_threshold,
            "eligible_hands_since_sweep": self.eligible_hands_since_sweep,
            "discarded_partial_sweep_eligible_hands": (self.discarded_partial_sweep_eligible_hands),
            "fact_state_count": len(self.fact_state),
            "skipped_fallback_trajectories": len(self.skipped_trajectory_hand_ids),
            "skipped_fact_admission_trajectories": len(self.skipped_trajectory_hand_ids),
            "evidence_classification_status": _aggregate_evidence_classification_status(
                self.evidence_review_queue
            ),
            "evidence_human_review_required": pending_review_count > 0,
            "evidence_review_queue_count": len(self.evidence_review_queue),
            "evidence_pending_human_review_count": pending_review_count,
            "evidence_paper_faithful_provisional_applied_count": (
                paper_faithful_applied_count
            ),
            "evidence_provisional_classification_count": provisional_classification_count,
            "evidence_pending_stability_proposal_count": pending_stability_proposal_count,
            "evidence_original_baseline_applied_stability_count": (
                original_baseline_applied_stability_count
            ),
            "evidence_review_queue": list(self.evidence_review_queue),
            "facts_reference": fact.get("facts_reference", self.fact.facts_reference()),
            "facts_state_reference": self.facts_state_reference(),
            "sweep_log_reference": self.sweep_log_reference(),
        }

    def _sweep(
        self,
        evidence: list[object],
        *,
        recent: list[HandTrajectory] | None = None,
        evidence_groups: dict[str, list[object]] | None = None,
    ) -> dict[str, object]:
        """
        功能：用近期轨迹和事实证据生成经验新版本。
        参数：
            evidence：事实证据列表。
            evidence_groups：保留首次召回路径归属的可选分组。
        返回：无。
        副作用：追加经验版本。
        异常：无。
        设计说明：当前为确定性巩固；真实 LLM 巩固可替换该函数。
        """

        sweep_recent = list(recent if recent is not None else self.recent[-self.sweep_every :])
        evidence_ids = [
            str(getattr(item, "record_id", ""))
            for item in evidence
            if str(getattr(item, "record_id", ""))
        ]
        allowed_evidence_ids = set(evidence_ids)
        supporting: list[str] = []
        contradicting: list[str] = []
        noise: list[str] = []
        irrelevant: list[str] = []
        ambiguous: list[str] = []
        classification_source = "deterministic_unreviewed"
        classification_reweight_allowed = False
        if self.expr.revision_strategy == "llm":
            revision_method = self.expr._revise_from_window
            revision_parameters = inspect.signature(revision_method).parameters
            revision_kwargs: dict[str, object] = {"evidence_records": evidence}
            if evidence_groups is not None and (
                "evidence_groups" in revision_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in revision_parameters.values()
                )
            ):
                revision_kwargs["evidence_groups"] = evidence_groups
            revision = revision_method(sweep_recent, **revision_kwargs)
            raw_supporting_ids = revision.get("supporting_fact_ids", [])
            if (
                self.expr.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
                and revision.get("validation_status") != "rejected"
                and isinstance(raw_supporting_ids, list)
                and len(raw_supporting_ids) == 1
            ):
                proposed_body = str(revision.get("new_md", ""))
                revision.update(
                    {
                        "keep": True,
                        "new_md": self.expr.current.body,
                        "proposed_new_md": proposed_body,
                        "validation_status": "rejected",
                        "failure": (
                            "invalid_fact_assessments: transferable Async supporting "
                            "evidence requires at least two facts"
                        ),
                        "contract_violation": True,
                        "applied_patches": [],
                        "proposal_summary": (
                            summarize_experience_proposal(self.expr.current.body, proposed_body)
                            if proposed_body
                            else {}
                        ),
                        "validation_findings": [
                            {
                                "code": "async_single_supporting_fact",
                                "message": (
                                    "transferable Async supporting evidence requires "
                                    "at least two facts"
                                ),
                                "section": "",
                                "bullet_index": None,
                                "excerpt": "",
                            }
                        ],
                    }
                )
            classification_reweight_allowed = bool(
                revision.get("revision_strategy")
                in {
                    "llm_full_document_revision",
                    "llm_structured_revision",
                    "llm_structured_patch_revision",
                }
                and not revision.get("fallback_used", False)
                and revision.get("validation_status")
                in {"accepted", "kept_by_model", "empty_noop", "unchanged"}
                and all(
                    isinstance(revision.get(key), list)
                    for key in (
                        "supporting_fact_ids",
                        "contradicting_fact_ids",
                        "noise_fact_ids",
                    )
                )
                and not (
                    self.expr.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
                    and len(revision.get("supporting_fact_ids", [])) == 1
                )
            )
            if classification_reweight_allowed:
                supporting = _validated_revision_fact_ids(
                    revision, "supporting_fact_ids", allowed_evidence_ids
                )
                contradicting = _validated_revision_fact_ids(
                    revision, "contradicting_fact_ids", allowed_evidence_ids
                )
                noise = _validated_revision_fact_ids(
                    revision, "noise_fact_ids", allowed_evidence_ids
                )
                irrelevant = _validated_revision_fact_ids(
                    revision, "irrelevant_fact_ids", allowed_evidence_ids
                )
                category_sets = [
                    set(supporting),
                    set(contradicting),
                    set(noise),
                    set(irrelevant),
                ]
                ambiguous_set = {
                    record_id
                    for index, category in enumerate(category_sets)
                    for other in category_sets[index + 1 :]
                    for record_id in category & other
                }
                ambiguous = [record_id for record_id in evidence_ids if record_id in ambiguous_set]
                supporting = [
                    record_id for record_id in supporting if record_id not in ambiguous_set
                ]
                contradicting = [
                    record_id for record_id in contradicting if record_id not in ambiguous_set
                ]
                noise = [record_id for record_id in noise if record_id not in ambiguous_set]
                irrelevant = [
                    record_id for record_id in irrelevant if record_id not in ambiguous_set
                ]
                classification_source = "llm_revision"
            elif revision.get("fallback_used", False):
                classification_source = "llm_revision_fallback_unreviewed"
            else:
                classification_source = "llm_revision_invalid_unreviewed"
            revision.update(
                {
                    "sweep_evidence_fact_ids": evidence_ids,
                    "sweep_supporting_fact_ids": supporting,
                    "sweep_contradicting_fact_ids": contradicting,
                    "sweep_noise_fact_ids": noise,
                    "sweep_irrelevant_fact_ids": irrelevant,
                    "ambiguous_fact_ids": ambiguous,
                    "sweep_evidence_groups": {
                        name: [
                            str(getattr(item, "record_id", ""))
                            for item in records
                            if str(getattr(item, "record_id", ""))
                        ]
                        for name, records in (evidence_groups or {}).items()
                    },
                }
            )
        else:
            revision = self.expr._deterministic_revision(sweep_recent)
        apply_provisional_stability = bool(
            classification_reweight_allowed
            and self.semantics_mode == "paper_faithful_v1"
        )
        evidence_classification_status, human_review_required = (
            _evidence_classification_outcome(
                has_evidence=bool(evidence),
                semantics_mode=self.semantics_mode,
                provisional_stability_applied=apply_provisional_stability,
            )
        )
        stability_application_policy = (
            "paper_faithful_v1_original_provisional_reweight"
            if apply_provisional_stability
            else "pending_human_review_proposal_only"
            if classification_reweight_allowed
            else "no_valid_stability_proposal"
        )
        proposed_stability_adjustments = self._proposed_stability_adjustments(
            supporting,
            contradicting,
            noise,
            applied=apply_provisional_stability,
        )
        self._classify_evidence(
            evidence,
            recent=sweep_recent,
            supporting=supporting,
            contradicting=contradicting,
            noise=noise,
            irrelevant=irrelevant,
            ambiguous=ambiguous,
            classification_source=classification_source,
            proposed_stability_adjustments=proposed_stability_adjustments,
            stability_application_policy=stability_application_policy,
            classification_status=evidence_classification_status,
            human_review_required=human_review_required,
        )
        revision.update(
            {
                "supporting_fact_ids": supporting,
                "contradicting_fact_ids": contradicting,
                "noise_fact_ids": noise,
                "irrelevant_fact_ids": irrelevant,
                "ambiguous_fact_ids": ambiguous,
                "experience_revision_created": False,
                "evidence_classification_status": evidence_classification_status,
                "human_review_required": human_review_required,
                "provisional_classification_source": classification_source,
                "stability_application_policy": stability_application_policy,
                "proposed_stability_adjustments": proposed_stability_adjustments,
            }
        )
        revision_number = self.expr.commit_revision(sweep_recent, revision)
        revision["experience_revision_created"] = revision_number is not None
        if apply_provisional_stability:
            self._reweight_fact_state(
                supporting,
                contradicting,
                noise,
                revision_number,
            )
        elif classification_reweight_allowed:
            self._link_fact_state_revision(
                [*supporting, *contradicting, *noise],
                revision_number,
            )
        return revision

    def _salience(self, record_id: str, current_hand: int) -> float:
        """
        功能：计算事实显著性。
        参数：
            record_id：事实 ID。
            current_hand：当前手牌序号。
        返回：0 到 1 附近的显著性。
        副作用：无。
        异常：无。
        设计说明：迁入原版 exp(-gap/stability) 的记忆曲线。
        """

        state = self.fact_state.get(record_id)
        if not state:
            return 0.0
        stability_value = state.get("stability")
        last_accessed_value = state.get("last_accessed_hand")
        if (
            isinstance(stability_value, bool)
            or not isinstance(stability_value, (int, float))
            or not math.isfinite(float(stability_value))
            or not self.stability_min <= float(stability_value) <= self.stability_max
            or isinstance(last_accessed_value, bool)
            or not isinstance(last_accessed_value, int)
            or last_accessed_value < 0
        ):
            return 0.0
        gap = max(0, current_hand - last_accessed_value)
        stability = float(stability_value)
        return math.exp(-gap / stability)

    def _select_fact_records(
        self,
        scored_records: list[tuple[object, float, dict[str, object]]],
    ) -> tuple[
        list[tuple[object, float, dict[str, object]]],
        list[tuple[object, float, dict[str, object]]],
        int,
        int,
    ]:
        """Apply Fact's threshold/signature contract and retain audited mirror overflow."""

        configured_top_k = self.fact.top_k
        self.fact.top_k = len(scored_records)
        try:
            eligible, below_threshold, duplicate_excluded = self.fact._select_diverse_records(
                scored_records
            )
        finally:
            self.fact.top_k = configured_top_k
        main_count = (
            PAPER_DECISION_MAIN_K
            if self.semantics_mode == "paper_faithful_v1"
            else max(0, configured_top_k)
        )
        return (
            list(eligible[:main_count]),
            list(eligible[main_count:]),
            below_threshold,
            duplicate_excluded,
        )

    def _scoped_eligible_records(self, observation: AgentObservation) -> list[object]:
        eligible = [
            record for record in self.fact.records if record.source.get("memory_eligible", True)
        ]
        return list(filter_records_by_scope(eligible, observation, self.scope))

    def _active_records(self, observation: AgentObservation) -> list[object]:
        """
        功能：返回当前显著性仍可召回的事实。
        参数：
            observation：当前观察。
        返回：事实列表。
        副作用：无。
        异常：无。
        设计说明：初期事实状态为空时不过滤，保证冷启动可用。
        """

        current_hand = (
            self.hand_counter
            if self.semantics_mode == "paper_faithful_v1"
            else self.hand_counter + 1
        )
        return [
            record
            for record in self._scoped_eligible_records(observation)
            if self._salience(record.record_id, current_hand) >= self.salience_threshold
        ]

    def _mirror_record(
        self,
        observation: AgentObservation,
        candidates: list[tuple[object, float, dict[str, object]]],
    ) -> tuple[tuple[object, float, dict[str, object]] | None, int]:
        """
        功能：确定性注入一条边缘显著性的 mirror 事实。
        参数：
            observation：当前观察。
            candidates：通过 Fact 门槛与签名去重、但未进入主路 top-k 的候选。
        返回：可选 mirror 记录及候选数量。
        副作用：无。
        异常：无。
        设计说明：对应原版 mirror recall，但用 seed 化随机保证复现。
        """

        if self.frozen:
            return None, 0

        edge_pool = [
            item
            for item in candidates
            if self.salience_threshold
            <= self._salience(item[0].record_id, self.hand_counter + 1)
            < self.salience_mirror_threshold
        ]
        if self.semantics_mode == "paper_faithful_v1":
            original_by_id = {item[0].record_id: item for item in edge_pool}
            edge_records = [item[0] for item in edge_pool]
            if self.fact.retrieval_backend == "feature_jaccard":
                feature_ranked = top_k_records(
                    observation,
                    edge_records,
                    PAPER_MIRROR_CANDIDATE_K,
                )
                pool = []
                for record, feature_score in feature_ranked:
                    original = original_by_id[record.record_id]
                    pool.append(
                        (
                            record,
                            float(feature_score),
                            {
                                **original[2],
                                "score": float(feature_score),
                                "semantic": None,
                                "feature": float(feature_score),
                                "salience": self._salience(record.record_id, self.hand_counter + 1),
                            },
                        )
                    )
            else:
                mirror_ranked = hybrid_top_k_records(
                    observation,
                    edge_records,
                    PAPER_MIRROR_CANDIDATE_K,
                    semantic_weight=self.fact.semantic_weight,
                    feature_weight=self.fact.feature_weight,
                    salience_fn=None,
                    embedding_backend=self.fact.embedding_backend,
                    retrieval_unit=self.fact.retrieval_unit,
                )
                pool = [
                    (
                        item.record,
                        item.score,
                        {
                            **original_by_id[item.record.record_id][2],
                            "score": item.score,
                            "semantic": item.semantic,
                            "dense": item.dense,
                            "sparse": item.sparse,
                            "colbert": item.colbert,
                            "feature": item.feature,
                            "salience": self._salience(
                                item.record.record_id, self.hand_counter + 1
                            ),
                            "retrieval_unit": item.retrieval_unit,
                            "matched_decision_index": item.matched_decision_index,
                            "matched_phase": item.matched_phase,
                        },
                    )
                    for item in mirror_ranked
                ]
        else:
            pool = list(edge_pool)
            pool.sort(key=lambda item: (item[0].created_at, item[0].record_id))
        if self.mirror_prob <= 0 or not pool:
            return None, len(pool)
        rng = make_rng(observation.seed, observation.agent_id, observation.hand_id, "mirror")
        if rng.random() >= self.mirror_prob:
            return None, len(pool)
        return rng.choice(pool), len(pool)

    def _recall_for_sweep(self, observation: AgentObservation) -> dict[str, list[object]]:
        """
        功能：为异步 sweep 执行相似、多样性、重要性三路召回。
        参数：
            observation：触发 sweep 的最后观察。
        返回：分组事实。
        副作用：无。
        异常：无。
        设计说明：迁入原版 Path-1/2/3 的巩固证据组织方式。
        """

        active = self._active_records(observation)
        if self.semantics_mode == "paper_faithful_v1":
            similar = [
                item.record
                for item in hybrid_top_k_records(
                    observation,
                    active,
                    PAPER_SWEEP_SIMILAR_K,
                    semantic_weight=self.fact.semantic_weight,
                    feature_weight=self.fact.feature_weight,
                    salience_fn=None,
                    embedding_backend=self.fact.embedding_backend,
                    retrieval_unit=self.fact.retrieval_unit,
                )
            ]
            buckets: dict[tuple[str, str], list[object]] = {}
            for record in active:
                phase = _record_phase(record)
                outcome = (
                    "win"
                    if record.final_reward > 0
                    else "loss"
                    if record.final_reward < 0
                    else "even"
                )
                buckets.setdefault((phase, outcome), []).append(record)
            diverse: list[object] = []
            for bucket_key in sorted(buckets):
                records = sorted(
                    buckets[bucket_key],
                    key=lambda record: (record.created_at, record.record_id),
                )
                bucket_rng = make_rng(
                    observation.seed,
                    observation.agent_id,
                    observation.hand_id,
                    "sweep_diversity",
                    *bucket_key,
                )
                diverse.extend(
                    bucket_rng.sample(
                        records,
                        k=min(PAPER_SWEEP_DIVERSE_PER_BUCKET, len(records)),
                    )
                )
            important = sorted(
                active,
                key=lambda record: (
                    self._salience(record.record_id, self.hand_counter),
                    record.created_at,
                    record.record_id,
                ),
                reverse=True,
            )[:PAPER_SWEEP_IMPORTANT_K]

            merged: dict[str, object] = {}
            for record in [*similar, *diverse, *important]:
                merged.setdefault(record.record_id, record)
            recalled_rng = make_rng(
                observation.seed,
                observation.agent_id,
                observation.hand_id,
                "sweep_union",
            )
            recalled = recalled_rng.sample(
                list(merged.values()),
                k=min(PAPER_SWEEP_TOTAL_K, len(merged)),
            )
            similar_ids = {record.record_id for record in similar}
            diverse_ids = {record.record_id for record in diverse}
            grouped = {
                "Path-1: 相似召回": [],
                "Path-2: 多样性召回": [],
                "Path-3: 重要性召回": [],
            }
            for record in recalled:
                if record.record_id in similar_ids:
                    grouped["Path-1: 相似召回"].append(record)
                elif record.record_id in diverse_ids:
                    grouped["Path-2: 多样性召回"].append(record)
                else:
                    grouped["Path-3: 重要性召回"].append(record)
            return grouped

        similar = [
            item.record
            for item in hybrid_top_k_records(
                observation,
                active,
                self.evidence_k,
                semantic_weight=self.fact.semantic_weight,
                feature_weight=self.fact.feature_weight,
                salience_fn=lambda record_id: self._salience(record_id, self.hand_counter),
                embedding_backend=self.fact.embedding_backend,
                retrieval_unit=self.fact.retrieval_unit,
            )
        ]
        buckets: dict[tuple[str, str], list[object]] = {}
        for record in active:
            phase = str(record.source.get("decisions", [{}])[-1].get("phase", "unknown"))
            outcome = (
                "win" if record.final_reward > 0 else "loss" if record.final_reward < 0 else "even"
            )
            buckets.setdefault((phase, outcome), []).append(record)
        diverse: list[object] = []
        for records in buckets.values():
            records.sort(key=lambda record: (record.created_at, record.record_id), reverse=True)
            diverse.extend(records[:1])
        important = sorted(
            active,
            key=lambda record: (
                self._salience(record.record_id, self.hand_counter),
                abs(record.final_reward),
                record.created_at,
            ),
            reverse=True,
        )[: self.evidence_k]
        return {
            "Path-1: 相似召回": similar,
            "Path-2: 多样性召回": diverse[: self.evidence_k],
            "Path-3: 重要性召回": important,
        }

    def _classify_evidence(
        self,
        evidence: list[object],
        *,
        recent: list[HandTrajectory],
        supporting: list[str],
        contradicting: list[str],
        noise: list[str],
        irrelevant: list[str],
        ambiguous: list[str],
        classification_source: str,
        proposed_stability_adjustments: list[dict[str, object]],
        stability_application_policy: str,
        classification_status: str,
        human_review_required: bool,
    ) -> None:
        """
        功能：记录模型临时分类与模式感知的应用/审核状态。
        参数：
            evidence：事实证据。
            supporting/contradicting/noise/irrelevant：经过 allowed-ID 校验的模型临时标签。
            ambiguous：模型同时放入多类的 ID，不用于调权。
            classification_source：临时分类来源。
        返回：无。
        副作用：追加证据分类审计项。
        异常：无。
        设计说明：quality 模式仅排队调权提案；paper_faithful_v1 明示保留原版临时调权。
        """

        recent_rewards = [item.final_reward for item in recent]
        supporting_set = set(supporting)
        contradicting_set = set(contradicting)
        noise_set = set(noise)
        irrelevant_set = set(irrelevant)
        ambiguous_set = set(ambiguous)
        proposal_by_id = {
            str(item["record_id"]): item for item in proposed_stability_adjustments
        }
        review_batch_index = len(self.sweep_log) + 1
        for record in evidence:
            reward = int(getattr(record, "final_reward", 0))
            record_id = str(getattr(record, "record_id", ""))
            if record_id in ambiguous_set:
                provisional_label: str | None = "ambiguous"
            elif record_id in supporting_set:
                provisional_label = "supporting"
            elif record_id in contradicting_set:
                provisional_label = "contradicting"
            elif record_id in noise_set:
                provisional_label = "noise"
            elif record_id in irrelevant_set:
                provisional_label = "irrelevant"
            else:
                provisional_label = None
            proposal = proposal_by_id.get(record_id)
            self.evidence_review_queue.append(
                {
                    "record_id": record_id,
                    "final_reward_chips": reward,
                    "recent_reward_chips": list(recent_rewards),
                    "phase": _record_phase(record),
                    "action_summary": getattr(record, "action_summary", ""),
                    "suggested_labels": [
                        "supporting",
                        "contradicting",
                        "noise",
                        "irrelevant",
                    ],
                    "provisional_label": provisional_label,
                    "provisional_classification_source": classification_source,
                    "provisional_applied_to_stability": bool(
                        proposal and proposal.get("applied") is True
                    ),
                    "stability_application_policy": stability_application_policy,
                    "stability_before": (
                        proposal.get("stability_before") if proposal else None
                    ),
                    "proposed_stability_multiplier": (
                        proposal.get("multiplier") if proposal else None
                    ),
                    "proposed_stability": (
                        proposal.get("proposed_stability") if proposal else None
                    ),
                    "effective_stability_after_sweep": (
                        proposal.get("effective_stability_after_sweep")
                        if proposal
                        else None
                    ),
                    "human_label": None,
                    "classification_status": classification_status,
                    "human_review_required": human_review_required,
                    "review_batch_index": review_batch_index,
                }
            )
        self.evidence_review_queue = self.evidence_review_queue[-500:]

    def _proposed_stability_adjustments(
        self,
        supporting: list[str],
        contradicting: list[str],
        noise: list[str],
        *,
        applied: bool,
    ) -> list[dict[str, object]]:
        """Compute auditable stability proposals without mutating effective state."""

        labels_and_multipliers = (
            ("supporting", supporting, 1.5),
            ("contradicting", contradicting, 2.0),
            ("noise", noise, 0.3),
        )
        proposals: list[dict[str, object]] = []
        for label, record_ids, multiplier in labels_and_multipliers:
            for record_id in record_ids:
                state = self.fact_state.get(record_id)
                if state is None:
                    continue
                stability_before = float(state.get("stability", self.stability_init))
                proposed = min(
                    max(stability_before * multiplier, self.stability_min),
                    self.stability_max,
                )
                proposals.append(
                    {
                        "record_id": record_id,
                        "provisional_label": label,
                        "stability_before": stability_before,
                        "multiplier": multiplier,
                        "proposed_stability": proposed,
                        "applied": applied,
                        "application_status": (
                            "paper_faithful_v1_original_applied"
                            if applied
                            else "pending_human_review_not_applied"
                        ),
                        "effective_stability_after_sweep": (
                            proposed if applied else stability_before
                        ),
                    }
                )
        return proposals

    def _link_fact_state_revision(
        self,
        record_ids: list[str],
        revision_number: int | None,
    ) -> None:
        """Retain evidence-to-revision provenance without changing stability."""

        if revision_number is None:
            return
        for record_id in dict.fromkeys(record_ids):
            state = self.fact_state.get(record_id)
            if state is None:
                continue
            linked = list(state.get("linked_exp_revs", []))
            if revision_number not in linked:
                linked.append(revision_number)
            state["linked_exp_revs"] = linked

    def _reweight_fact_state(
        self,
        supporting: list[str],
        contradicting: list[str],
        noise: list[str],
        revision_number: int | None,
    ) -> None:
        """
        功能：根据 sweep 判断调整事实稳定性。
        参数：
            supporting：支持事实 ID。
            contradicting：冲突事实 ID。
            noise：噪声事实 ID。
            revision_number：实际创建的经验版本号；未创建时为 None。
        返回：无。
        副作用：修改 fact_state。
        异常：无。
        设计说明：对应原版 supporting/contradicting/noise 对稳定性的重权重。
        """

        supporting_set = set(supporting)
        contradicting_set = set(contradicting)
        noise_set = set(noise)
        for record_id, state in self.fact_state.items():
            stability = float(state.get("stability", self.stability_init))
            if record_id in noise_set:
                state["stability"] = max(stability * 0.3, self.stability_min)
            elif record_id in contradicting_set:
                state["stability"] = min(stability * 2.0, self.stability_max)
            elif record_id in supporting_set:
                state["stability"] = min(stability * 1.5, self.stability_max)
            if (
                revision_number is not None
                and record_id in supporting_set | contradicting_set | noise_set
            ):
                linked = list(state.get("linked_exp_revs", []))
                linked.append(revision_number)
                state["linked_exp_revs"] = linked


def _resolve_async_persistence_paths(
    directory: str | Path,
    reference_root: str | Path,
) -> tuple[Path, Path]:
    """Resolve one proper Agent subdirectory inside the declared run root."""

    resolved_root = Path(reference_root).resolve()
    resolved_directory = Path(directory).resolve()
    try:
        relative = resolved_directory.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Async persistence directory must stay inside run root") from exc
    if relative == Path("."):
        raise ValueError("Async persistence directory must be an Agent subdirectory")
    if resolved_root.exists() and not resolved_root.is_dir():
        raise ValueError("Async persistence reference_root must be a directory")
    if resolved_directory.exists() and not resolved_directory.is_dir():
        raise ValueError("Async persistence directory must be a directory")
    return resolved_directory, resolved_root


def _canonical_json_line(value: dict[str, object]) -> bytes:
    """Serialize one deterministic UTF-8 JSONL row without non-finite values."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + "\n").encode("utf-8")


def _canonical_fact_state_bytes(state: dict[str, dict[str, object]]) -> bytes:
    """Render the complete current state table in stable fact-ID order."""

    return b"".join(
        _canonical_json_line({"id": record_id, **state[record_id]})
        for record_id in sorted(state)
    )


def _canonical_sweep_log_bytes(items: list[dict[str, object]]) -> bytes:
    """Render the complete append-only sweep prefix in original order."""

    return b"".join(_canonical_json_line(_compact_sweep_log_entry(item)) for item in items)


def _assert_regular_exact_file(path: Path, expected: bytes) -> None:
    """Require one non-linked regular file to match its in-memory view exactly."""

    if path.is_symlink():
        raise ValueError(f"{path.name} must not be a symbolic link")
    if not path.is_file():
        raise ValueError(f"{path.name} is missing or is not a regular file")
    if path.read_bytes() != expected:
        raise ValueError(f"{path.name} diverges from Async in-memory state")


def _materialize_new_exact_file(path: Path, payload: bytes) -> None:
    """Materialize a restored view once without replacing an existing path."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            _assert_regular_exact_file(path, payload)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_replace_exact_file(path: Path, payload: bytes) -> None:
    """Replace one current view or append-only prefix through an atomic rename."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validated_sweep_id_list(value: object, field: str) -> list[str]:
    """Validate one ordered, duplicate-free sweep fact-ID list."""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Async sweep_log {field} must be a string list")
    if len(set(value)) != len(value):
        raise ValueError(f"Async sweep_log {field} contains duplicate IDs")
    return list(value)


def _validated_restored_sweep_log(
    raw: list[dict[str, object]],
    *,
    stability_min: float,
    stability_max: float,
    max_experience_version: int,
) -> list[dict[str, object]]:
    """Validate restored sweep IDs, versions, and stability proposal bounds."""

    validated: list[dict[str, object]] = []
    id_fields = (
        "evidence_fact_ids",
        "supporting_fact_ids",
        "contradicting_fact_ids",
        "noise_fact_ids",
        "irrelevant_fact_ids",
        "ambiguous_fact_ids",
    )
    for index, raw_item in enumerate(raw):
        item = _compact_sweep_log_entry(deepcopy(raw_item))
        ids = {
            field: _validated_sweep_id_list(item.get(field, []), field)
            for field in id_fields
        }
        evidence_ids = set(ids["evidence_fact_ids"])
        for field in id_fields[1:]:
            if not set(ids[field]).issubset(evidence_ids):
                raise ValueError(f"Async sweep_log {field} escapes evidence IDs")
        groups = item.get("evidence_groups", {})
        if not isinstance(groups, dict):
            raise ValueError("Async sweep_log evidence_groups must be an object")
        for name, group_ids in groups.items():
            validated_group = _validated_sweep_id_list(
                group_ids,
                f"evidence_groups.{name}",
            )
            if not set(validated_group).issubset(evidence_ids):
                raise ValueError("Async sweep_log evidence group escapes evidence IDs")
        old_version = item.get("old_experience_version")
        new_version = item.get("new_experience_version")
        if old_version is not None or new_version is not None:
            if (
                isinstance(old_version, bool)
                or not isinstance(old_version, int)
                or isinstance(new_version, bool)
                or not isinstance(new_version, int)
                or old_version <= 0
                or new_version not in {old_version, old_version + 1}
                or new_version > max_experience_version
            ):
                raise ValueError(f"Async sweep_log experience version is invalid at {index}")
        proposals = item.get("proposed_stability_adjustments", [])
        if not isinstance(proposals, list):
            raise ValueError("Async sweep_log proposed_stability_adjustments must be a list")
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise ValueError("Async sweep_log stability proposal must be an object")
            record_id = proposal.get("record_id")
            if not isinstance(record_id, str) or record_id not in evidence_ids:
                raise ValueError("Async sweep_log stability proposal has unknown record_id")
            if not isinstance(proposal.get("applied"), bool):
                raise ValueError("Async sweep_log stability proposal applied must be boolean")
            multiplier = proposal.get("multiplier")
            if (
                isinstance(multiplier, bool)
                or not isinstance(multiplier, (int, float))
                or not math.isfinite(float(multiplier))
                or float(multiplier) <= 0.0
            ):
                raise ValueError("Async sweep_log stability multiplier is invalid")
            for field in (
                "stability_before",
                "proposed_stability",
                "effective_stability_after_sweep",
            ):
                value = proposal.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not stability_min <= float(value) <= stability_max
                ):
                    raise ValueError(f"Async sweep_log {field} is out of bounds")
        # Canonical serialization is also the final recursive finite-number and
        # JSON-compatibility check for every retained diagnostic field.
        _canonical_json_line(item)
        validated.append(item)
    return validated


def _validated_restored_fact_state(
    raw: dict[object, object],
    *,
    active_fact_ids: set[str],
    hand_counter: int,
    stability_min: float,
    stability_max: float,
    max_experience_version: int,
) -> dict[str, dict[str, object]]:
    """Validate every persisted Async fact-state row before making it live."""

    state_ids = {str(record_id) for record_id in raw}
    if state_ids != active_fact_ids:
        missing = sorted(active_fact_ids - state_ids)
        orphaned = sorted(state_ids - active_fact_ids)
        raise ValueError(
            f"Async fact_state 与事实库存不一致：missing={missing[:5]}, orphaned={orphaned[:5]}"
        )
    validated: dict[str, dict[str, object]] = {}
    required = {
        "stability",
        "last_accessed_hand",
        "access_count",
        "linked_exp_revs",
    }
    for raw_record_id, raw_state in raw.items():
        record_id = str(raw_record_id)
        if not isinstance(raw_state, dict) or set(raw_state) != required:
            raise ValueError(f"Async fact_state row schema 非法：{record_id}")
        stability = raw_state["stability"]
        last_accessed = raw_state["last_accessed_hand"]
        access_count = raw_state["access_count"]
        linked_revisions = raw_state["linked_exp_revs"]
        if (
            isinstance(stability, bool)
            or not isinstance(stability, (int, float))
            or not math.isfinite(float(stability))
            or not stability_min <= float(stability) <= stability_max
        ):
            raise ValueError(f"Async fact_state stability 非法：{record_id}")
        if (
            isinstance(last_accessed, bool)
            or not isinstance(last_accessed, int)
            or not 0 <= last_accessed <= hand_counter
        ):
            raise ValueError(f"Async fact_state last_accessed_hand 非法：{record_id}")
        if isinstance(access_count, bool) or not isinstance(access_count, int) or access_count < 0:
            raise ValueError(f"Async fact_state access_count 非法：{record_id}")
        if (
            not isinstance(linked_revisions, list)
            or any(
                isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0
                or revision > max_experience_version
                for revision in linked_revisions
            )
            or len(set(linked_revisions)) != len(linked_revisions)
        ):
            raise ValueError(f"Async fact_state linked_exp_revs 非法：{record_id}")
        validated[record_id] = {
            **deepcopy(raw_state),
            "stability": float(stability),
            "last_accessed_hand": last_accessed,
            "access_count": access_count,
            "linked_exp_revs": list(linked_revisions),
        }
    return validated


def _evidence_classification_outcome(
    *,
    has_evidence: bool,
    semantics_mode: str,
    provisional_stability_applied: bool,
) -> tuple[str, bool]:
    """Return the truthful classification status and whether human review is required."""

    if not has_evidence:
        return "not_requested", False
    if semantics_mode == "paper_faithful_v1":
        if provisional_stability_applied:
            return PAPER_FAITHFUL_APPLIED_CLASSIFICATION_STATUS, False
        return NO_VALID_STABILITY_PROPOSAL_CLASSIFICATION_STATUS, False
    return PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS, True


def _normalized_evidence_review_item(
    item: dict[str, object],
    *,
    semantics_mode: str,
) -> dict[str, object]:
    """Migrate legacy review rows without reintroducing a paper-mode human gate."""

    normalized = dict(item)
    status = str(normalized.get("classification_status", ""))
    if semantics_mode == "paper_faithful_v1":
        if not status or status == PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS:
            status = (
                PAPER_FAITHFUL_APPLIED_CLASSIFICATION_STATUS
                if normalized.get("provisional_applied_to_stability") is True
                else NO_VALID_STABILITY_PROPOSAL_CLASSIFICATION_STATUS
            )
        normalized["human_review_required"] = False
    else:
        if not status:
            status = PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS
        normalized["human_review_required"] = (
            status == PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS
        )
    normalized["classification_status"] = status
    return normalized


def _evidence_human_review_required(item: dict[str, object]) -> bool:
    """Interpret explicit status first and retain compatibility with legacy quality rows."""

    explicit = item.get("human_review_required")
    if isinstance(explicit, bool):
        return explicit
    return (
        str(item.get("classification_status", PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS))
        == PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS
    )


def _aggregate_evidence_classification_status(
    items: list[dict[str, object]],
) -> str:
    """Summarize evidence state without calling applied paper rows pending review."""

    if not items:
        return "not_requested"
    if any(_evidence_human_review_required(item) for item in items):
        return PENDING_HUMAN_REVIEW_CLASSIFICATION_STATUS
    statuses = {str(item.get("classification_status", "not_requested")) for item in items}
    if len(statuses) == 1:
        return next(iter(statuses))
    if statuses <= {
        PAPER_FAITHFUL_APPLIED_CLASSIFICATION_STATUS,
        NO_VALID_STABILITY_PROPOSAL_CLASSIFICATION_STATUS,
    }:
        return "paper_faithful_mixed_outcomes"
    return "mixed_non_pending"


def _validate_async_parameters(
    *,
    window_size: int,
    sweep_every: int,
    evidence_k: int,
    salience_threshold: float,
    salience_mirror_threshold: float,
    mirror_prob: float,
    stability_init: float,
    stability_min: float,
    stability_max: float,
) -> None:
    """Fail closed on cadence, probability and stability configuration drift."""

    integer_parameters = {
        "window_size": window_size,
        "sweep_every": sweep_every,
        "evidence_k": evidence_k,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_parameters.values()
    ):
        raise ValueError("Async window_size/sweep_every/evidence_k 必须为整数")
    if window_size <= 0 or sweep_every <= 0 or evidence_k <= 0:
        raise ValueError("Async window_size/sweep_every/evidence_k 必须为正整数")
    if window_size < sweep_every:
        raise ValueError("Async window_size 必须不小于 sweep_every")
    numeric_values = (
        salience_threshold,
        salience_mirror_threshold,
        mirror_prob,
        stability_init,
        stability_min,
        stability_max,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_values
    ) or not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("Async 阈值、概率与稳定性参数必须是有限数")
    if not 0.0 <= salience_threshold <= salience_mirror_threshold <= 1.0:
        raise ValueError("Async salience 阈值必须满足 0<=main<=mirror<=1")
    if not 0.0 <= mirror_prob <= 1.0:
        raise ValueError("Async mirror_prob 必须位于 [0,1]")
    if not 0.0 < stability_min <= stability_init <= stability_max:
        raise ValueError("Async stability 必须满足 0<min<=init<=max")


def _compact_sweep_log_entry(item: dict[str, object]) -> dict[str, object]:
    """Redact legacy raw failure text while retaining stable sweep diagnostics."""

    result = dict(item)
    if result.get("revision_failure") is None:
        return result
    diagnostic = experience_failure_diagnostic(result["revision_failure"])
    result["revision_failure"] = diagnostic["preview"]
    result.setdefault("revision_failure_code", diagnostic["code"])
    result.setdefault("revision_failure_sha256", diagnostic["sha256"])
    result.setdefault("revision_failure_chars", diagnostic["chars"])
    return result


def _record_phase(record: object) -> str:
    source = getattr(record, "source", {})
    decisions = source.get("decisions", []) if isinstance(source, dict) else []
    if decisions and isinstance(decisions[-1], dict):
        return str(decisions[-1].get("phase", "unknown"))
    return "unknown"


def _validated_revision_fact_ids(
    revision: dict[str, object],
    key: str,
    allowed_evidence_ids: set[str],
) -> list[str]:
    """Return unique revision IDs that belong to the current sweep evidence."""

    values = revision.get(key, [])
    if not isinstance(values, list):
        return []
    validated: list[str] = []
    for value in values:
        record_id = str(value)
        if record_id in allowed_evidence_ids and record_id not in validated:
            validated.append(record_id)
    return validated
