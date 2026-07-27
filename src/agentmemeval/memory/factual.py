"""
模块说明：本模块实现 FactAgent 使用的结构化事实记忆。
核心职责：在手牌结束后写入事实记录，决策时按相似特征检索 Top-k。
输入与输出：输入观察与可见轨迹，输出 MemoryContext 和 MemorySnapshot。
依赖边界：不依赖 LLM，不访问环境内部私牌，只消费 HandTrajectory。
不负责：不维护经验文档，不决定动作。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    FactualMemoryRecord,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
)
from agentmemeval.environment.decision_facts import build_decision_facts
from agentmemeval.environment.observation import observation_to_compact_text
from agentmemeval.memory.base import filter_records_by_scope, trajectory_quality
from agentmemeval.memory.rag import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    build_retrieval_query,
    embedding_identity_from_audit,
    embedding_runtime_isolation_from_audit,
    hybrid_top_k_records,
    retrieval_document_template,
    retrieval_position_label,
)
from agentmemeval.memory.retrievers import observation_features, top_k_records


class FactualMemory:
    """
    功能：存储并检索 hand-level 事实记忆。
    参数：
        agent_id：所属 Agent。
        scope：记忆作用域。
        top_k：检索条数。
        max_records：最多保留事实数量；None 或 0 表示不设上限。
    返回：事实记忆机制。
    副作用：内部列表随轨迹更新。
    异常：无。
    设计说明：事实记录只在手牌结束后写入，避免未来信息参与当前决策。
    """

    name = "fact"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        top_k: int = 8,
        max_records: int | None = 500,
        retrieval_backend: str = "hybrid_rag",
        retrieval_unit: str = "hand_terminal_v1",
        semantic_weight: float | None = None,
        feature_weight: float | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        minimum_retrieval_score: float | None = None,
        retrieval_threshold_status: str = "pending_pilot",
        duplicate_window: int | None = None,
        reject_zero_reward_preflop_fold: bool | None = None,
        reject_single_preflop_fold: bool | None = None,
        retrieval_signature_dedup: bool | None = None,
        semantics_mode: str = "quality_filtered_v2",
        admission_log_limit: int = 500,
    ) -> None:
        """
        功能：初始化事实记忆。
        参数：
            agent_id：所属 Agent。
            scope：记忆作用域。
            top_k：检索条数。
            max_records：容量上限；None 或 0 表示不设上限。
        返回：无。
        副作用：创建空事实库。
        异常：无。
        设计说明：容量上限由配置控制，避免长期实验无界增长。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.top_k = _positive_integer(top_k, "top_k")
        self.retrieval_backend = _validated_retrieval_backend(retrieval_backend)
        self.retrieval_unit = _validated_retrieval_unit(retrieval_unit)
        self.embedding_backend = embedding_backend or HashEmbeddingBackend()
        (
            self.minimum_retrieval_score,
            self.retrieval_threshold_status,
        ) = _validated_retrieval_threshold(
            minimum_retrieval_score,
            retrieval_threshold_status,
        )
        if semantics_mode not in {"paper_faithful_v1", "quality_filtered_v2"}:
            raise ValueError(f"未知 fact semantics_mode：{semantics_mode}")
        self.semantics_mode = semantics_mode
        paper_faithful = semantics_mode == "paper_faithful_v1"
        # The paper store is append-only and unbounded.  Wrapper constructors in
        # older snapshots may still pass their historical default (500), so the
        # semantics mode is authoritative here; config validation rejects that
        # misleading value for newly resolved paper-faithful runs.
        self.max_records = None if paper_faithful else _normalize_max_records(max_records)
        self.semantic_weight, self.feature_weight = _resolve_retrieval_weights(
            semantics_mode=semantics_mode,
            semantic_weight=semantic_weight,
            feature_weight=feature_weight,
        )
        for field, value in (
            ("reject_zero_reward_preflop_fold", reject_zero_reward_preflop_fold),
            ("reject_single_preflop_fold", reject_single_preflop_fold),
            ("retrieval_signature_dedup", retrieval_signature_dedup),
        ):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{field} must be a boolean or None")
        if paper_faithful:
            if retrieval_backend != "hybrid_rag":
                raise ValueError(
                    "paper_faithful_v1 要求 retrieval_backend=hybrid_rag 并使用纯 embedding 排序"
                )
            if duplicate_window not in {None, 0}:
                raise ValueError("paper_faithful_v1 要求 duplicate_window=0")
            for field, value in (
                ("reject_zero_reward_preflop_fold", reject_zero_reward_preflop_fold),
                ("reject_single_preflop_fold", reject_single_preflop_fold),
                ("retrieval_signature_dedup", retrieval_signature_dedup),
            ):
                if value is True:
                    raise ValueError(f"paper_faithful_v1 要求 {field}=False")
        # Zero explicitly preserves the original paper behavior: keep every fact.
        self.duplicate_window = _nonnegative_integer(
            0 if paper_faithful else 50 if duplicate_window is None else duplicate_window,
            "duplicate_window",
        )
        self.reject_zero_reward_preflop_fold = (
            not paper_faithful
            if reject_zero_reward_preflop_fold is None
            else reject_zero_reward_preflop_fold
        )
        self.reject_single_preflop_fold = (
            not paper_faithful if reject_single_preflop_fold is None else reject_single_preflop_fold
        )
        self.retrieval_signature_dedup = (
            not paper_faithful if retrieval_signature_dedup is None else retrieval_signature_dedup
        )
        self.admission_log_limit = _positive_integer(admission_log_limit, "admission_log_limit")
        self.records: list[FactualMemoryRecord] = []
        self.next_record_index = 1
        self.last_retrieval: list[str] = []
        self.last_scores: list[dict[str, object]] = []
        self.admission_log: list[dict[str, object]] = []
        self.admission_counts: Counter[str] = Counter()
        self.retrieval_request_count = 0
        self.empty_retrieval_count = 0
        self.retrieval_below_threshold_count = 0
        self.retrieval_duplicate_excluded_count = 0
        self.retrieval_audit_log: list[dict[str, object]] = []
        self.last_admission_status: str | None = None
        self.last_admission_reasons: list[str] = []
        self.last_restore_embedding_runtime_audit: dict[str, object] | None = None
        self.persistence_directory: Path | None = None
        self.persistence_reference_root: Path | None = None
        self.facts_path: Path | None = None
        self.frozen = False

    def set_frozen(self, frozen: bool) -> None:
        """Enable read-only evaluation mode without changing persisted facts."""

        self.frozen = bool(frozen)

    def bind_persistence(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path,
    ) -> None:
        """Bind the canonical append-only ``facts.jsonl`` inside one Agent directory.

        A restored snapshot is materialized exactly once when the destination does
        not exist.  An existing destination is reusable only when its canonical
        bytes already represent the complete in-memory record sequence.
        """

        resolved_directory, resolved_root = _resolve_fact_persistence_paths(
            directory,
            reference_root,
        )
        candidate_path = resolved_directory / "facts.jsonl"
        expected = _canonical_fact_bytes(self.records)
        if candidate_path.is_symlink():
            raise ValueError("facts.jsonl must not be a symbolic link")
        if candidate_path.exists():
            _assert_regular_canonical_fact_file(candidate_path, expected)
        else:
            if self.frozen:
                raise RuntimeError("frozen factual memory cannot materialize facts.jsonl")
            resolved_directory.mkdir(parents=True, exist_ok=True)
            # Resolve again after creation so an existing symlinked parent cannot
            # redirect the final file outside the declared run root.
            resolved_directory, resolved_root = _resolve_fact_persistence_paths(
                resolved_directory,
                resolved_root,
            )
            candidate_path = resolved_directory / "facts.jsonl"
            self._materialize_new_facts_file(candidate_path, expected)

        self.persistence_directory = resolved_directory
        self.persistence_reference_root = resolved_root
        self.facts_path = candidate_path
        self._assert_canonical_facts_current()

    def facts_reference(self) -> dict[str, object]:
        """Return a compact pointer proving the canonical fact sequence in use."""

        canonical = _canonical_fact_bytes(self.records)
        relative_path: str | None = None
        if self.facts_path is not None:
            self._assert_canonical_facts_current()
            if self.persistence_reference_root is None:  # pragma: no cover - invariant guard
                raise RuntimeError("facts persistence root is missing")
            relative_path = self.facts_path.relative_to(
                self.persistence_reference_root
            ).as_posix()
        return {
            "path": relative_path,
            "record_count": len(self.records),
            "sha256": hashlib.sha256(canonical).hexdigest(),
        }

    def _materialize_new_facts_file(self, path: Path, payload: bytes) -> None:
        """Create a new canonical file without replacing any pre-existing path."""

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".facts.",
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
                _assert_regular_canonical_fact_file(path, payload)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _atomic_replace_facts_file(self, payload: bytes) -> None:
        """Atomically extend an already validated canonical fact file."""

        if self.facts_path is None:  # pragma: no cover - guarded by caller
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".facts.",
            suffix=".tmp",
            dir=str(self.facts_path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.facts_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _append_persisted_fact(self, record: FactualMemoryRecord) -> None:
        """Persist one admitted record before the corresponding memory commit."""

        if self.facts_path is None:
            return
        if self.frozen:  # pragma: no cover - public lifecycle returns before admission
            raise RuntimeError("frozen factual memory cannot append facts.jsonl")
        current = self._assert_canonical_facts_current()
        candidate = current + _canonical_fact_line(record)
        self._atomic_replace_facts_file(candidate)

    def _assert_canonical_facts_current(self) -> bytes:
        """Fail closed when the canonical file and active memory diverge."""

        expected = _canonical_fact_bytes(self.records)
        if self.facts_path is None:
            return expected
        _assert_regular_canonical_fact_file(self.facts_path, expected)
        return expected

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：根据当前观察检索相似事实。
        参数：
            observation：合法可见观察。
        返回：包含事实证据的 MemoryContext。
        副作用：记录最近一次检索 ID。
        异常：无。
        设计说明：检索过滤先按作用域执行，再按可解释特征排序。
        """

        self._assert_canonical_facts_current()
        eligible_records = [
            record for record in self.records if record.source.get("memory_eligible", True)
        ]
        candidates = filter_records_by_scope(eligible_records, observation, self.scope)
        retrieval_query = build_retrieval_query(observation)
        active_document_template = (
            "{record.features}"
            if self.retrieval_backend == "feature_jaccard"
            else retrieval_document_template(
                self.embedding_backend,
                self.retrieval_unit,
            )
        )
        self.retrieval_request_count += 1
        below_threshold = 0
        duplicate_excluded = 0
        if self.retrieval_backend == "feature_jaccard":
            scored = top_k_records(observation, candidates, len(candidates))
            candidate_scores = [float(score) for _record, score in scored]
            selected, below_threshold, duplicate_excluded = self._select_diverse_records(
                [(record, score) for record, score in scored]
            )
            retrieved = [record for record, _score in selected]
            self.last_scores = [
                {
                    "record_id": record.record_id,
                    "score": score,
                    "feature": score,
                    "semantic": None,
                    "salience": None,
                }
                for record, score in selected
            ]
        else:
            scored_rag = hybrid_top_k_records(
                observation,
                candidates,
                len(candidates),
                semantic_weight=self.semantic_weight,
                feature_weight=self.feature_weight,
                embedding_backend=self.embedding_backend,
                retrieval_unit=self.retrieval_unit,
            )
            selected_rag, below_threshold, duplicate_excluded = self._select_diverse_records(
                [(item.record, item.score, item) for item in scored_rag]
            )
            candidate_scores = [float(item.score) for item in scored_rag]
            retrieved = [record for record, _score, _item in selected_rag]
            self.last_scores = [
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
                }
                for _record, _score, item in selected_rag
            ]
        self.retrieval_below_threshold_count += below_threshold
        self.retrieval_duplicate_excluded_count += duplicate_excluded
        if not retrieved:
            self.empty_retrieval_count += 1
        score_summary = _score_summary(candidate_scores)
        self.retrieval_audit_log.append(
            {
                "hand_id": observation.hand_id,
                "query": retrieval_query,
                "retrieval_backend": self.retrieval_backend,
                "retrieval_unit": self.retrieval_unit,
                "document_template": active_document_template,
                "candidate_count": len(candidates),
                "returned_count": len(retrieved),
                "below_threshold_count": below_threshold,
                "duplicate_signature_excluded_count": duplicate_excluded,
                "score_summary": score_summary,
            }
        )
        self.retrieval_audit_log = self.retrieval_audit_log[-self.admission_log_limit :]
        self.last_retrieval = [record.record_id for record in retrieved]
        return MemoryContext(
            facts=retrieved,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "retrieval_backend": self.retrieval_backend,
                "retrieval_unit": self.retrieval_unit,
                "query": retrieval_query,
                "document_template": active_document_template,
                "retrieved_fact_ids": list(self.last_retrieval),
                "retrieval_scores": list(self.last_scores),
                "candidate_count": len(candidates),
                "returned_count": len(retrieved),
                "below_threshold_count": below_threshold,
                "duplicate_signature_excluded_count": duplicate_excluded,
                "minimum_retrieval_score": self.minimum_retrieval_score,
                "retrieval_threshold_status": self.retrieval_threshold_status,
                "retrieval_score_weights": {
                    "embedding": self.semantic_weight,
                    "structural_feature": self.feature_weight,
                },
                "retrieval_score_policy": (
                    "embedding_only"
                    if self.feature_weight == 0.0
                    else "embedding_plus_structural_feature"
                ),
                "candidate_score_summary": score_summary,
                "excluded_fallback_fact_count": self.admission_counts.get(
                    "reason:provider_fallback", 0
                ),
                "embedding": self.embedding_backend.audit_metadata(),
            },
        )

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：接收已提交决策事件。
        参数：
            event：决策事件。
        返回：无。
        副作用：无。
        异常：无。
        设计说明：事实记忆只在手牌结束后写入，当前决策不即时入库。
        """

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：把已结束手牌压缩为一条结构化事实。
        参数：
            trajectory：该 Agent 可见轨迹。
        返回：无。
        副作用：追加事实记录并执行容量裁剪。
        异常：无。
        设计说明：事实摘要使用决策点观察和结算回报，不包含对手未公开私牌。
        """

        if self.frozen:
            return
        if not trajectory.decision_events:
            self._record_admission(trajectory, "rejected", ["no_decision_events"])
            return
        last_event = trajectory.decision_events[-1]
        quality = trajectory_quality(trajectory)
        decisions = [_decision_view(event) for event in trajectory.decision_events]
        signature = _structural_signature(trajectory, decisions)
        rejection_reasons = _admission_rejection_reasons(
            trajectory,
            decisions,
            quality,
            reject_zero_reward_preflop_fold=self.reject_zero_reward_preflop_fold,
            reject_single_preflop_fold=self.reject_single_preflop_fold,
        )
        if rejection_reasons:
            self._record_admission(
                trajectory,
                "rejected",
                rejection_reasons,
                structural_signature=signature,
            )
            return
        duplicate = None
        if self.duplicate_window > 0:
            duplicate = next(
                (
                    record
                    for record in reversed(self.records[-self.duplicate_window :])
                    if record.source.get("structural_signature") == signature
                ),
                None,
            )
        if duplicate is not None:
            # An append-only canonical row is immutable once written.  Legacy
            # in-memory-only mode keeps its historical duplicate annotation;
            # file-backed mode records the duplicate solely in admission audit.
            if self.facts_path is None:
                duplicate.source["duplicate_count"] = (
                    int(duplicate.source.get("duplicate_count", 0)) + 1
                )
                duplicate.source["last_duplicate_hand_id"] = trajectory.hand_id
            self._record_admission(
                trajectory,
                "deduplicated",
                ["duplicate_structural_signature"],
                structural_signature=signature,
                canonical_record_id=duplicate.record_id,
            )
            return
        action_summary = "; ".join(
            f"{item['phase']}:{item['action_type']}"
            + (f"({item['amount']})" if item.get("amount") else "")
            for item in decisions
        )
        won_amount, won_amount_status = _resolved_won_amount(trajectory)
        fact_text = _render_fact_text(
            trajectory,
            last_event,
            decisions,
            won_amount=won_amount,
            won_amount_status=won_amount_status,
        )
        record = FactualMemoryRecord(
            record_id=f"{self.agent_id}-fact-{self.next_record_index}",
            agent_id=self.agent_id,
            table_id=trajectory.table_id,
            hand_id=trajectory.hand_id,
            scope=self.scope,
            state_summary=fact_text,
            action_summary=action_summary,
            final_reward=trajectory.final_reward,
            features=observation_features(last_event.observation),
            source={
                "visibility": "agent_observation_plus_declared_intent_and_final_result",
                "quality_policy_version": "graded_fact_admission_v1",
                "quality_label": "admitted_informative",
                "decision_count": len(trajectory.decision_events),
                "showdown_visible_agent_ids": sorted(trajectory.showdown_visible_cards),
                "showdown_visible_cards": trajectory.showdown_visible_cards,
                "decisions": decisions,
                "hand_outcome": _hand_outcome(trajectory.final_reward),
                "hand_net_amount": trajectory.final_reward,
                "hand_won_amount": won_amount,
                "hand_won_amount_status": won_amount_status,
                "retrieval_query": build_retrieval_query(last_event.observation),
                "fact_text": fact_text,
                "compact_state": observation_to_compact_text(last_event.observation),
                "structural_signature": signature,
                "duplicate_count": 0,
                **quality,
            },
        )
        candidate_records = [*self.records, record]
        if self.max_records is not None and len(candidate_records) > self.max_records:
            if self.facts_path is not None:
                raise RuntimeError(
                    "append-only facts.jsonl cannot evict records; use unbounded "
                    "paper_faithful_v1 persistence"
                )
            candidate_records = candidate_records[-self.max_records :]
        admission_state = self._next_admission_state(
            trajectory,
            "admitted",
            [],
            structural_signature=signature,
            canonical_record_id=record.record_id,
        )
        # The disk commit happens first.  The remaining operations are only
        # attribute assignments from fully constructed candidates, so a failed
        # write cannot advance record IDs or admission counters in memory.
        self._append_persisted_fact(record)
        self.next_record_index += 1
        self.records = candidate_records
        self._install_admission_state(admission_state)

    def _select_diverse_records(self, scored: list[tuple[Any, ...]]) -> tuple[list[Any], int, int]:
        """Apply the frozen retrieval order: threshold, signature diversity, then top-k."""

        selected: list[Any] = []
        seen_signatures: set[str] = set()
        below_threshold = 0
        duplicate_excluded = 0
        for item in scored:
            record = item[0]
            score = float(item[1])
            if self.minimum_retrieval_score is not None and score < self.minimum_retrieval_score:
                below_threshold += 1
                continue
            signature = str(record.source.get("structural_signature", record.record_id))
            if self.retrieval_signature_dedup and signature in seen_signatures:
                duplicate_excluded += 1
                continue
            seen_signatures.add(signature)
            if len(selected) < self.top_k:
                selected.append(item)
        return selected, below_threshold, duplicate_excluded

    def _record_admission(
        self,
        trajectory: HandTrajectory,
        status: str,
        reasons: list[str],
        *,
        structural_signature: str | None = None,
        canonical_record_id: str | None = None,
    ) -> None:
        """Keep bounded audit evidence even when a trajectory is not stored as a fact."""

        self._install_admission_state(
            self._next_admission_state(
                trajectory,
                status,
                reasons,
                structural_signature=structural_signature,
                canonical_record_id=canonical_record_id,
            )
        )

    def _next_admission_state(
        self,
        trajectory: HandTrajectory,
        status: str,
        reasons: list[str],
        *,
        structural_signature: str | None = None,
        canonical_record_id: str | None = None,
    ) -> tuple[Counter[str], list[dict[str, object]], str, list[str]]:
        """Build, but do not install, the next bounded admission audit state."""

        counts = Counter(self.admission_counts)
        counts[status] += 1
        for reason in reasons:
            counts[f"reason:{reason}"] += 1
        log = [*self.admission_log]
        log.append(
            {
                "hand_id": trajectory.hand_id,
                "table_id": trajectory.table_id,
                "status": status,
                "quality_policy_version": "graded_fact_admission_v1",
                "reasons": list(reasons),
                "final_reward": trajectory.final_reward,
                "structural_signature": structural_signature,
                "canonical_record_id": canonical_record_id,
            }
        )
        return (
            counts,
            log[-self.admission_log_limit :],
            status,
            list(reasons),
        )

    def _install_admission_state(
        self,
        state: tuple[Counter[str], list[dict[str, object]], str, list[str]],
    ) -> None:
        """Install one already validated admission-audit candidate."""

        counts, log, status, reasons = state
        self.admission_counts = counts
        self.admission_log = log
        self.last_admission_status = status
        self.last_admission_reasons = reasons

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出事实库快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：快照只含 JSON 结构，便于泛化阶段恢复。
        """

        self._assert_canonical_facts_current()
        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "schema_version": 8,
                "semantics_mode": self.semantics_mode,
                "top_k": self.top_k,
                "max_records": self.max_records,
                "capacity_mode": ("unbounded" if self.max_records is None else "bounded"),
                "retrieval_backend": self.retrieval_backend,
                "retrieval_unit": self.retrieval_unit,
                "semantic_weight": self.semantic_weight,
                "feature_weight": self.feature_weight,
                "minimum_retrieval_score": self.minimum_retrieval_score,
                "retrieval_threshold_status": self.retrieval_threshold_status,
                "duplicate_window": self.duplicate_window,
                "reject_zero_reward_preflop_fold": self.reject_zero_reward_preflop_fold,
                "reject_single_preflop_fold": self.reject_single_preflop_fold,
                "retrieval_signature_dedup": self.retrieval_signature_dedup,
                "next_record_index": self.next_record_index,
                "admission_log": deepcopy(self.admission_log),
                "admission_counts": dict(self.admission_counts),
                "last_admission_status": self.last_admission_status,
                "last_admission_reasons": list(self.last_admission_reasons),
                "retrieval_request_count": self.retrieval_request_count,
                "empty_retrieval_count": self.empty_retrieval_count,
                "retrieval_below_threshold_count": self.retrieval_below_threshold_count,
                "retrieval_duplicate_excluded_count": self.retrieval_duplicate_excluded_count,
                "retrieval_audit_log": deepcopy(self.retrieval_audit_log),
                "embedding": deepcopy(self.embedding_backend.audit_metadata()),
                "records": deepcopy([record.to_dict() for record in self.records]),
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：从快照恢复事实库。
        参数：
            snapshot：事实记忆快照。
        返回：无。
        副作用：替换内部事实列表。
        异常：无。
        设计说明：恢复后继续使用当前配置的作用域，快照作用域同步覆盖本对象。
        """

        if snapshot.mechanism != self.name:
            raise ValueError(
                f"Fact snapshot mechanism mismatch: {snapshot.mechanism} != {self.name}"
            )
        if snapshot.agent_id != self.agent_id:
            raise ValueError(
                f"Fact snapshot agent_id mismatch: {snapshot.agent_id} != {self.agent_id}"
            )
        if snapshot.scope != self.scope:
            raise ValueError(f"Fact snapshot scope mismatch: {snapshot.scope} != {self.scope}")
        if not isinstance(snapshot.payload, dict):
            raise ValueError("Fact snapshot payload must be an object")
        payload = snapshot.payload
        raw_schema_version = payload.get("schema_version", 1)
        if isinstance(raw_schema_version, bool) or not isinstance(raw_schema_version, int):
            raise ValueError("Fact snapshot schema_version must be an integer")
        schema_version = raw_schema_version
        if schema_version < 1 or schema_version > 8:
            raise ValueError(f"Unsupported Fact snapshot schema_version: {schema_version}")

        snapshot_semantics_mode = payload.get("semantics_mode", self.semantics_mode)
        if not isinstance(snapshot_semantics_mode, str):
            raise ValueError("Fact snapshot semantics_mode must be a string")
        if snapshot_semantics_mode != self.semantics_mode:
            raise ValueError(
                "Fact snapshot semantics_mode mismatch: "
                f"{snapshot_semantics_mode} != {self.semantics_mode}"
            )

        if schema_version >= 8 and "top_k" not in payload:
            raise ValueError("Fact snapshot top_k is required")
        snapshot_top_k = _positive_integer(payload.get("top_k", self.top_k), "snapshot top_k")
        snapshot_minimum, snapshot_threshold_status = _validated_retrieval_threshold(
            payload.get("minimum_retrieval_score", self.minimum_retrieval_score),
            payload.get("retrieval_threshold_status", self.retrieval_threshold_status),
        )

        try:
            snapshot_embedding_metadata = payload.get("embedding")
            configured_embedding_metadata = self.embedding_backend.audit_metadata()
            snapshot_embedding_identity = embedding_identity_from_audit(snapshot_embedding_metadata)
            configured_embedding_identity = embedding_identity_from_audit(
                configured_embedding_metadata
            )
            snapshot_embedding_runtime = embedding_runtime_isolation_from_audit(
                snapshot_embedding_metadata
            )
            configured_embedding_runtime = embedding_runtime_isolation_from_audit(
                configured_embedding_metadata
            )
        except ValueError as exc:
            raise ValueError(f"Fact snapshot embedding identity is invalid: {exc}") from exc
        if snapshot_embedding_identity != configured_embedding_identity:
            raise ValueError("Fact snapshot embedding identity mismatch")
        restore_runtime_audit = {
            "snapshot": deepcopy(snapshot_embedding_runtime),
            "live": deepcopy(configured_embedding_runtime),
            "runtime_isolation_changed": (
                snapshot_embedding_runtime != configured_embedding_runtime
            ),
        }

        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("Fact snapshot records must be a list")
        restored_records: list[FactualMemoryRecord] = []
        record_ids: set[str] = set()
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                raise ValueError("Fact snapshot record must be an object")
            record = deepcopy(raw_record)
            source = record.get("source", {})
            if not isinstance(source, dict):
                raise ValueError("Fact snapshot record source must be an object")
            source = dict(source)
            if schema_version < 2 and "memory_eligible" not in source:
                source["memory_eligible"] = False
                source["legacy_unverified"] = True
            record["source"] = source
            try:
                restored = FactualMemoryRecord(**record)
            except (TypeError, ValueError) as exc:
                raise ValueError("Fact snapshot record schema is invalid") from exc
            _validate_restored_record(restored, schema_version=schema_version)
            if restored.record_id in record_ids:
                raise ValueError(
                    f"Fact snapshot contains duplicate record_id: {restored.record_id}"
                )
            if restored.agent_id != self.agent_id:
                raise ValueError("Fact snapshot record agent_id mismatch")
            if restored.scope != self.scope:
                raise ValueError("Fact snapshot record scope mismatch")
            record_ids.add(restored.record_id)
            restored_records.append(restored)

        inferred_next_record_index = _next_record_index(restored_records, self.agent_id)
        if schema_version >= 3 and "next_record_index" not in payload:
            raise ValueError("Fact snapshot next_record_index is required")
        raw_next_record_index = payload.get("next_record_index", inferred_next_record_index)
        next_record_index = _positive_integer(raw_next_record_index, "snapshot next_record_index")
        if next_record_index < inferred_next_record_index:
            raise ValueError(
                "Fact snapshot next_record_index would collide with an existing record"
            )

        raw_max_records = payload.get("max_records", self.max_records)
        if self.semantics_mode == "paper_faithful_v1":
            if schema_version >= 8 and raw_max_records not in {None, 0}:
                raise ValueError("paper_faithful_v1 Fact snapshot must remain unbounded")
            snapshot_max_records = None
        else:
            snapshot_max_records = _normalize_max_records(raw_max_records)
        if schema_version >= 8:
            capacity_mode = payload.get("capacity_mode")
            expected_capacity_mode = "unbounded" if snapshot_max_records is None else "bounded"
            if capacity_mode != expected_capacity_mode:
                raise ValueError("Fact snapshot capacity_mode is inconsistent")
        if snapshot_max_records is not None and len(restored_records) > snapshot_max_records:
            raise ValueError("Fact snapshot record count exceeds max_records")

        snapshot_retrieval_backend = _validated_retrieval_backend(
            payload.get("retrieval_backend", self.retrieval_backend)
        )
        snapshot_retrieval_unit = _validated_retrieval_unit(
            payload.get("retrieval_unit", self.retrieval_unit)
        )
        if (
            self.semantics_mode == "paper_faithful_v1"
            and snapshot_retrieval_backend != "hybrid_rag"
        ):
            raise ValueError("paper_faithful_v1 Fact snapshot requires hybrid_rag")
        if self.semantics_mode == "paper_faithful_v1" and schema_version < 8:
            # Schema <=7 could persist the old 0.65/0.35 post-embedding feature
            # blend.  Migrate it to the paper's pure embedding ranking.
            raw_semantic_weight = None
            raw_feature_weight = None
        else:
            raw_semantic_weight = payload.get("semantic_weight", self.semantic_weight)
            raw_feature_weight = payload.get("feature_weight", self.feature_weight)
        snapshot_semantic_weight, snapshot_feature_weight = _resolve_retrieval_weights(
            semantics_mode=self.semantics_mode,
            semantic_weight=raw_semantic_weight,
            feature_weight=raw_feature_weight,
        )
        snapshot_duplicate_window = _nonnegative_integer(
            payload.get("duplicate_window", self.duplicate_window),
            "snapshot duplicate_window",
        )
        snapshot_reject_zero = _strict_boolean(
            payload.get("reject_zero_reward_preflop_fold", self.reject_zero_reward_preflop_fold),
            "snapshot reject_zero_reward_preflop_fold",
        )
        snapshot_reject_single = _strict_boolean(
            payload.get("reject_single_preflop_fold", self.reject_single_preflop_fold),
            "snapshot reject_single_preflop_fold",
        )
        snapshot_signature_dedup = _strict_boolean(
            payload.get("retrieval_signature_dedup", self.retrieval_signature_dedup),
            "snapshot retrieval_signature_dedup",
        )
        if self.semantics_mode == "paper_faithful_v1" and (
            snapshot_duplicate_window != 0
            or snapshot_reject_zero
            or snapshot_reject_single
            or snapshot_signature_dedup
        ):
            raise ValueError("paper_faithful_v1 Fact snapshot enables a quality-filter heuristic")

        snapshot_semantic_config = {
            "top_k": snapshot_top_k,
            "max_records": snapshot_max_records,
            "retrieval_backend": snapshot_retrieval_backend,
            "retrieval_unit": snapshot_retrieval_unit,
            "semantic_weight": snapshot_semantic_weight,
            "feature_weight": snapshot_feature_weight,
            "minimum_retrieval_score": snapshot_minimum,
            "retrieval_threshold_status": snapshot_threshold_status,
            "duplicate_window": snapshot_duplicate_window,
            "reject_zero_reward_preflop_fold": snapshot_reject_zero,
            "reject_single_preflop_fold": snapshot_reject_single,
            "retrieval_signature_dedup": snapshot_signature_dedup,
        }
        live_semantic_config = {
            "top_k": self.top_k,
            "max_records": self.max_records,
            "retrieval_backend": self.retrieval_backend,
            "retrieval_unit": self.retrieval_unit,
            "semantic_weight": self.semantic_weight,
            "feature_weight": self.feature_weight,
            "minimum_retrieval_score": self.minimum_retrieval_score,
            "retrieval_threshold_status": self.retrieval_threshold_status,
            "duplicate_window": self.duplicate_window,
            "reject_zero_reward_preflop_fold": (self.reject_zero_reward_preflop_fold),
            "reject_single_preflop_fold": self.reject_single_preflop_fold,
            "retrieval_signature_dedup": self.retrieval_signature_dedup,
        }
        mismatched_semantic_fields = sorted(
            field
            for field, value in snapshot_semantic_config.items()
            if value != live_semantic_config[field]
        )
        if mismatched_semantic_fields:
            raise ValueError(
                "Fact snapshot semantic config mismatch: " + ", ".join(mismatched_semantic_fields)
            )

        snapshot_admission_log = _validated_object_list(
            payload.get("admission_log", []), "snapshot admission_log"
        )
        snapshot_admission_counts = _validated_nonnegative_counter(
            payload.get("admission_counts", {}), "snapshot admission_counts"
        )
        snapshot_last_status = payload.get("last_admission_status")
        if snapshot_last_status is not None and not isinstance(snapshot_last_status, str):
            raise ValueError("snapshot last_admission_status must be a string or null")
        snapshot_last_reasons = _validated_string_list(
            payload.get("last_admission_reasons", []),
            "snapshot last_admission_reasons",
        )
        snapshot_retrieval_request_count = _nonnegative_integer(
            payload.get("retrieval_request_count", 0),
            "snapshot retrieval_request_count",
        )
        snapshot_empty_retrieval_count = _nonnegative_integer(
            payload.get("empty_retrieval_count", 0),
            "snapshot empty_retrieval_count",
        )
        snapshot_below_threshold_count = _nonnegative_integer(
            payload.get("retrieval_below_threshold_count", 0),
            "snapshot retrieval_below_threshold_count",
        )
        snapshot_duplicate_excluded_count = _nonnegative_integer(
            payload.get("retrieval_duplicate_excluded_count", 0),
            "snapshot retrieval_duplicate_excluded_count",
        )
        snapshot_retrieval_audit_log = _validated_object_list(
            payload.get("retrieval_audit_log", []),
            "snapshot retrieval_audit_log",
        )

        if self.facts_path is not None:
            _assert_regular_canonical_fact_file(
                self.facts_path,
                _canonical_fact_bytes(restored_records),
            )

        # Commit only after every field has passed validation.  A rejected
        # snapshot must leave the live memory byte-for-byte unchanged.
        self.scope = snapshot.scope
        self.records = restored_records
        self.admission_log = snapshot_admission_log
        self.admission_counts = snapshot_admission_counts
        self.last_admission_status = snapshot_last_status
        self.last_admission_reasons = snapshot_last_reasons
        self.retrieval_request_count = snapshot_retrieval_request_count
        self.empty_retrieval_count = snapshot_empty_retrieval_count
        self.retrieval_below_threshold_count = snapshot_below_threshold_count
        self.retrieval_duplicate_excluded_count = snapshot_duplicate_excluded_count
        self.retrieval_audit_log = snapshot_retrieval_audit_log
        self.next_record_index = next_record_index
        self.last_restore_embedding_runtime_audit = restore_runtime_audit

    def metrics(self) -> dict[str, object]:
        """
        功能：返回事实记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：报告可展示事实数量和最近检索命中来源。
        """

        return {
            "mechanism": self.name,
            "frozen": self.frozen,
            "fact_count": len(self.records),
            "eligible_fact_count": sum(
                record.source.get("memory_eligible", True) for record in self.records
            ),
            "excluded_fallback_fact_count": self.admission_counts.get(
                "reason:provider_fallback", 0
            ),
            "experience_updates": 0,
            "last_retrieved_fact_ids": list(self.last_retrieval),
            "retrieval_backend": self.retrieval_backend,
            "retrieval_unit": self.retrieval_unit,
            "max_records": self.max_records,
            "capacity_mode": ("unbounded" if self.max_records is None else "bounded"),
            "retrieval_score_weights": {
                "embedding": self.semantic_weight,
                "structural_feature": self.feature_weight,
            },
            "retrieval_score_policy": (
                "embedding_only"
                if self.feature_weight == 0.0
                else "embedding_plus_structural_feature"
            ),
            "last_retrieval_scores": list(self.last_scores),
            "minimum_retrieval_score": self.minimum_retrieval_score,
            "retrieval_threshold_status": self.retrieval_threshold_status,
            "retrieval_request_count": self.retrieval_request_count,
            "empty_retrieval_count": self.empty_retrieval_count,
            "empty_retrieval_rate": (
                self.empty_retrieval_count / self.retrieval_request_count
                if self.retrieval_request_count
                else 0.0
            ),
            "retrieval_below_threshold_count": self.retrieval_below_threshold_count,
            "retrieval_duplicate_excluded_count": self.retrieval_duplicate_excluded_count,
            "recent_retrieval_audit_log": list(self.retrieval_audit_log),
            "admission_counts": dict(self.admission_counts),
            "last_admission_status": self.last_admission_status,
            "last_admission_reasons": list(self.last_admission_reasons),
            "reject_single_preflop_fold": self.reject_single_preflop_fold,
            "recent_admission_log": list(self.admission_log),
            "max_structural_signature_share": _max_signature_share(self.records),
            "embedding": self.embedding_backend.audit_metadata(),
            "last_restore_embedding_runtime_audit": deepcopy(
                self.last_restore_embedding_runtime_audit
            ),
            "facts_reference": self.facts_reference(),
        }


def _resolve_fact_persistence_paths(
    directory: str | Path,
    reference_root: str | Path,
) -> tuple[Path, Path]:
    """Resolve a proper descendant Agent directory inside its declared run root."""

    resolved_root = Path(reference_root).resolve()
    resolved_directory = Path(directory).resolve()
    try:
        relative = resolved_directory.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("fact persistence directory must stay inside run root") from exc
    if relative == Path("."):
        raise ValueError("fact persistence directory must be an Agent subdirectory")
    if resolved_root.exists() and not resolved_root.is_dir():
        raise ValueError("fact persistence reference_root must be a directory")
    if resolved_directory.exists() and not resolved_directory.is_dir():
        raise ValueError("fact persistence directory must be a directory")
    return resolved_directory, resolved_root


def _canonical_fact_line(record: FactualMemoryRecord) -> bytes:
    """Serialize exactly one record as one deterministic UTF-8 JSONL row."""

    payload = json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if "\n" in payload or "\r" in payload:  # pragma: no cover - json escaping invariant
        raise ValueError("canonical fact JSON must occupy exactly one physical line")
    return (payload + "\n").encode("utf-8")


def _canonical_fact_bytes(records: list[FactualMemoryRecord]) -> bytes:
    """Serialize the complete in-memory record order into canonical JSONL bytes."""

    return b"".join(_canonical_fact_line(record) for record in records)


def _assert_regular_canonical_fact_file(path: Path, expected: bytes) -> None:
    """Require a regular, non-linked file with exact canonical memory bytes."""

    if path.is_symlink():
        raise ValueError("facts.jsonl must not be a symbolic link")
    if not path.is_file():
        raise ValueError("canonical facts.jsonl is missing or is not a regular file")
    observed = path.read_bytes()
    if observed != expected:
        raise ValueError("canonical facts.jsonl diverges from in-memory factual records")


def _admission_rejection_reasons(
    trajectory: HandTrajectory,
    decisions: list[dict[str, object]],
    quality: dict[str, int | bool],
    *,
    reject_zero_reward_preflop_fold: bool,
    reject_single_preflop_fold: bool,
) -> list[str]:
    reasons: list[str] = []
    if int(quality["fallback_count"]) > 0:
        reasons.append("provider_fallback")
    if int(quality["action_mismatch_count"]) > 0:
        reasons.append("guard_action_type_mismatch")
    only = decisions[0] if len(decisions) == 1 else None
    if (
        reject_single_preflop_fold
        and only is not None
        and only.get("phase") == "preflop"
        and only.get("action_type") == "fold"
    ):
        reasons.append("single_preflop_fold_low_information")
    if (
        reject_zero_reward_preflop_fold
        and not reject_single_preflop_fold
        and trajectory.final_reward == 0
        and only is not None
        and only.get("phase") == "preflop"
        and only.get("action_type") == "fold"
        and not trajectory.showdown_visible_cards
    ):
        reasons.append("zero_reward_single_preflop_fold_without_showdown")
    return reasons


def _structural_signature(
    trajectory: HandTrajectory,
    decisions: list[dict[str, object]],
) -> str:
    last = trajectory.decision_events[-1].observation
    position = _structural_position_label(last)
    decision_facts = build_decision_facts(last)
    draw = dict(decision_facts.get("draw", {}))
    hero_draw = dict(draw.get("hero_draw", {}))
    board_threat = dict(draw.get("board_completion_threat", {}))
    call = dict(decision_facts.get("call", {}))
    effective_stack = int(decision_facts["effective_stack"])
    effective_stack_min = int(decision_facts["effective_stack_min"])
    contestable_pot = int(decision_facts["contestable_pot"])
    action_sequence = ">".join(
        f"{item.get('phase')}:{item.get('action_type')}" for item in decisions
    )
    outcome = (
        "win" if trajectory.final_reward > 0 else "loss" if trajectory.final_reward < 0 else "even"
    )
    features = sorted(observation_features(last))
    return "|".join(
        [
            f"phase={last.phase}",
            f"position={position}",
            f"to_call_pot={_ratio_bucket(int(call.get('call_cost', 0)), contestable_pot)}",
            f"effective_stack_pot={_ratio_bucket(effective_stack, contestable_pot)}",
            "effective_stack_range="
            f"{_ratio_bucket(effective_stack_min, contestable_pot)}:"
            f"{_ratio_bucket(effective_stack, contestable_pot)}",
            f"made_hand={decision_facts['made_hand_class']}",
            "draw="
            f"{int(bool(hero_draw.get('flush_draw')))}:"
            f"{int(bool(hero_draw.get('straight_draw')))}",
            "board_threat="
            f"{int(bool(board_threat.get('flush_draw')))}:"
            f"{int(bool(board_threat.get('straight_draw')))}",
            f"features={','.join(features)}",
            f"actions={action_sequence}",
            f"outcome={outcome}",
        ]
    )


def _structural_position_label(observation: AgentObservation) -> str:
    """Return a seat-rotation-invariant button/blind/relative position label."""

    return retrieval_position_label(observation)


def _ratio_bucket(numerator: int, denominator: int) -> str:
    ratio = numerator / max(1, denominator)
    if ratio == 0:
        return "zero"
    if ratio <= 0.25:
        return "low"
    if ratio <= 0.75:
        return "medium"
    return "high"


def _max_signature_share(records: list[FactualMemoryRecord]) -> float:
    if not records:
        return 0.0
    counts = Counter(
        str(record.source.get("structural_signature", record.record_id)) for record in records
    )
    return max(counts.values()) / len(records)


def _score_summary(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "count": len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "mean": sum(scores) / len(scores),
    }


def _next_record_index(records: list[FactualMemoryRecord], agent_id: str) -> int:
    """Infer a collision-free counter when restoring snapshots written before schema v3."""

    prefix = f"{agent_id}-fact-"
    indexes = []
    for record in records:
        suffix = record.record_id.removeprefix(prefix)
        if record.record_id.startswith(prefix) and suffix.isdigit():
            indexes.append(int(suffix))
    return max(indexes, default=0) + 1


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _strict_boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _validated_retrieval_backend(value: object) -> str:
    if not isinstance(value, str) or value not in {
        "feature_jaccard",
        "hybrid_rag",
    }:
        raise ValueError(f"未知 retrieval_backend：{value}")
    return value


def _validated_retrieval_unit(value: object) -> str:
    if not isinstance(value, str) or value not in {
        "hand_terminal_v1",
        "decision_point_max_v1",
    }:
        raise ValueError(f"未知 retrieval_unit：{value}")
    return value


def _validated_retrieval_threshold(
    minimum: object,
    status: object,
) -> tuple[float | None, str]:
    if not isinstance(status, str) or status not in {"pending_pilot", "frozen"}:
        raise ValueError("retrieval_threshold_status must be pending_pilot or frozen")
    if minimum is None:
        if status == "frozen":
            raise ValueError("frozen retrieval threshold requires a numeric value")
        return None, status
    if type(minimum) not in {int, float}:
        raise ValueError("minimum_retrieval_score must be finite")
    parsed_minimum = float(minimum)
    if not math.isfinite(parsed_minimum):
        raise ValueError("minimum_retrieval_score must be finite")
    return parsed_minimum, status


def _normalize_max_records(value: object) -> int | None:
    """Normalize the public capacity contract: ``None``/``0`` means unbounded."""

    if value is None:
        return None
    if type(value) is not int:
        raise ValueError("max_records 必须是非负整数、0 或 None")
    normalized = value
    if normalized < 0:
        raise ValueError("max_records 必须是非负整数、0 或 None")
    return None if normalized == 0 else normalized


def _resolve_retrieval_weights(
    *,
    semantics_mode: str,
    semantic_weight: float | None,
    feature_weight: float | None,
) -> tuple[float, float]:
    """Resolve and validate the embedding-vs-structural score blend."""

    paper_faithful = semantics_mode == "paper_faithful_v1"
    raw_semantic = (
        1.0
        if paper_faithful and semantic_weight is None
        else 0.65
        if semantic_weight is None
        else semantic_weight
    )
    raw_feature = (
        0.0
        if paper_faithful and feature_weight is None
        else 0.35
        if feature_weight is None
        else feature_weight
    )
    if type(raw_semantic) not in {int, float} or type(raw_feature) not in {
        int,
        float,
    }:
        raise ValueError("semantic_weight/feature_weight 必须为有限数值")
    resolved_semantic = float(raw_semantic)
    resolved_feature = float(raw_feature)
    if (
        not math.isfinite(resolved_semantic)
        or not math.isfinite(resolved_feature)
        or resolved_semantic < 0
        or resolved_feature < 0
        or resolved_semantic + resolved_feature <= 0
    ):
        raise ValueError("semantic_weight/feature_weight 必须为有限非负数且总和大于零")
    if paper_faithful and (resolved_semantic != 1.0 or resolved_feature != 0.0):
        raise ValueError(
            "paper_faithful_v1 要求纯 embedding 排序：semantic_weight=1.0, feature_weight=0.0"
        )
    return resolved_semantic, resolved_feature


def _validate_restored_record(
    record: FactualMemoryRecord,
    *,
    schema_version: int,
) -> None:
    for field in (
        "record_id",
        "agent_id",
        "table_id",
        "hand_id",
        "scope",
        "state_summary",
        "action_summary",
        "created_at",
    ):
        if not isinstance(getattr(record, field), str):
            raise ValueError(f"Fact snapshot record {field} must be a string")
    if type(record.final_reward) is not int:
        raise ValueError("Fact snapshot record final_reward must be an integer")
    if not isinstance(record.features, list) or not all(
        isinstance(feature, str) for feature in record.features
    ):
        raise ValueError("Fact snapshot record features must be a string list")
    if not isinstance(record.source, dict):
        raise ValueError("Fact snapshot record source must be an object")
    if schema_version >= 8:
        prefix = f"{record.agent_id}-fact-"
        suffix = record.record_id.removeprefix(prefix)
        if not record.record_id.startswith(prefix) or not suffix.isdigit() or int(suffix) < 1:
            raise ValueError("Fact snapshot record_id violates the schema-v8 identity")


def _validated_object_list(value: object, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of objects")
    return deepcopy(value)


def _validated_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return list(value)


def _validated_nonnegative_counter(value: object, field: str) -> Counter[str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: Counter[str] = Counter()
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise ValueError(f"{field} must map strings to non-negative integers")
        result[key] = count
    return result


def _resolved_won_amount(trajectory: HandTrajectory) -> tuple[int | None, str]:
    """Read gross payout propagated from HandResult.winners; never infer it from net reward."""

    raw_won_amount = getattr(trajectory, "won_amount", None)
    if raw_won_amount is None:
        return None, "unavailable_legacy_or_unpropagated_trajectory"
    try:
        won_amount = int(raw_won_amount)
    except (TypeError, ValueError):
        return None, "unavailable_invalid_hand_result_winners_value"
    if won_amount < 0:
        return None, "unavailable_invalid_negative_hand_result_winners_value"
    return won_amount, "observed_from_hand_result_winners_v1"


def _hand_outcome(final_reward: int) -> str:
    if final_reward > 0:
        return "win"
    if final_reward < 0:
        return "loss"
    return "draw"


def _decision_view(event: DecisionEvent) -> dict[str, object]:
    """
    功能：把一次决策压缩成事实记忆中的可读片段。
    参数：
        event：决策事件。
    返回：字典。
    副作用：无。
    异常：无。
    设计说明：保留可观察状态和已提交动作；显式 intent 单列为模型自述，
    不再把行动理由冒充 intent，也不把任一自述当成客观事实。
    """

    fallback_used = bool(event.llm_metadata.get("fallback_used"))
    declared_intent = None if fallback_used else (event.decision.intent.strip() or None)
    return {
        "decision_state_schema_version": "agent_observation_v2_explicit_intent",
        "observation": event.observation.to_dict(),
        "phase": event.observation.phase,
        "board": list(event.observation.community_cards),
        "hole": list(event.observation.hole_cards),
        "pot_before": event.observation.pot,
        "to_call": event.observation.to_call,
        "action_type": event.committed_action.action_type,
        "raw_action_type": event.decision.action_type,
        "committed_action_type": event.committed_action.action_type,
        "amount": event.committed_action.amount,
        "declared_intent": declared_intent,
        "declared_intent_source": "raw_decision.intent",
        "declared_intent_truth_status": (
            "invalidated_by_action_guard_fallback"
            if fallback_used
            else "model_self_report_not_ground_truth"
        ),
        "guard_repaired": bool(event.llm_metadata.get("guard_repaired")),
        "fallback_used": fallback_used,
        "retrieval_query": build_retrieval_query(event.observation),
        "features": observation_features(event.observation),
    }


def _render_fact_text(
    trajectory: HandTrajectory,
    last_event: DecisionEvent,
    decisions: list[dict[str, object]],
    *,
    won_amount: int | None,
    won_amount_status: str,
) -> str:
    """
    功能：渲染一条接近原版 FactAgent 的 hand-level fact。
    参数：
        trajectory：手牌轨迹。
        last_event：最后一次决策。
        decisions：决策片段。
        won_amount：由 HandResult.winners 累加得到的毛收入；旧轨迹可为 None。
        won_amount_status：毛收入的来源或不可用原因。
    返回：事实文本。
    副作用：无。
    异常：无。
    设计说明：文本只用观察者可见牌面和终局回报，不写入未公开私牌。
    """

    observation = last_event.observation
    board = " ".join(observation.community_cards) if observation.community_cards else "(空)"
    final_board = (
        " ".join(trajectory.final_community_cards) if trajectory.final_community_cards else "(空)"
    )
    hole = " ".join(observation.hole_cards) if observation.hole_cards else "(空)"
    decision_lines = []
    for item in decisions:
        amount = f"({item['amount']})" if item.get("amount") else ""
        declared_intent = item.get("declared_intent") or "(无)"
        decision_lines.append(
            f"  [{item['phase']}] declared_intent={declared_intent!r} "
            "[模型自述，非客观事实] "
            f"observed_action={item['action_type']}{amount}"
        )
    decision_text = "\n".join(decision_lines) or "  (无)"
    outcome = _hand_outcome(trajectory.final_reward)
    won_amount_text = (
        str(won_amount) if won_amount is not None else f"unavailable[{won_amount_status}]"
    )
    visible_showdown_count = len(trajectory.showdown_visible_cards)
    return (
        f"[hand {trajectory.hand_id}, 最后决策阶段={observation.phase}]\n"
        f"最后决策局面 board={board}, hole={hole}, pot={observation.pot}, "
        f"to_call={observation.to_call}\n"
        f"结算时公开 board={final_board}\n"
        f"我的决策序列:\n{decision_text}\n"
        f"hand_outcome: {outcome} "
        f"(净收益 {trajectory.final_reward:+d}, 从底池赢回 {won_amount_text}, "
        f"gross_source={won_amount_status}, 终局筹码 {trajectory.final_stack})\n"
        f"showdown_visible_player_count: {visible_showdown_count}\n"
        f"summary: {trajectory.summary}"
    )
