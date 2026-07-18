"""
模块说明：本模块实现 FactAgent 使用的结构化事实记忆。
核心职责：在手牌结束后写入事实记录，决策时按相似特征检索 Top-k。
输入与输出：输入观察与可见轨迹，输出 MemoryContext 和 MemorySnapshot。
依赖边界：不依赖 LLM，不访问环境内部私牌，只消费 HandTrajectory。
不负责：不维护经验文档，不决定动作。
"""

from __future__ import annotations

from collections import Counter
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
    hybrid_top_k_records,
)
from agentmemeval.memory.retrievers import observation_features, top_k_records


class FactualMemory:
    """
    功能：存储并检索 hand-level 事实记忆。
    参数：
        agent_id：所属 Agent。
        scope：记忆作用域。
        top_k：检索条数。
        max_records：最多保留事实数量。
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
        max_records: int = 500,
        retrieval_backend: str = "hybrid_rag",
        semantic_weight: float = 0.65,
        feature_weight: float = 0.35,
        embedding_backend: EmbeddingBackend | None = None,
        minimum_retrieval_score: float | None = None,
        retrieval_threshold_status: str = "pending_pilot",
        duplicate_window: int = 50,
        reject_zero_reward_preflop_fold: bool = True,
        reject_single_preflop_fold: bool = True,
        retrieval_signature_dedup: bool = True,
        admission_log_limit: int = 500,
    ) -> None:
        """
        功能：初始化事实记忆。
        参数：
            agent_id：所属 Agent。
            scope：记忆作用域。
            top_k：检索条数。
            max_records：容量上限。
        返回：无。
        副作用：创建空事实库。
        异常：无。
        设计说明：容量上限由配置控制，避免长期实验无界增长。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.top_k = top_k
        self.max_records = max_records
        self.retrieval_backend = retrieval_backend
        self.semantic_weight = semantic_weight
        self.feature_weight = feature_weight
        self.embedding_backend = embedding_backend or HashEmbeddingBackend()
        self.minimum_retrieval_score = minimum_retrieval_score
        self.retrieval_threshold_status = retrieval_threshold_status
        # Zero explicitly preserves the original paper behavior: keep every fact.
        self.duplicate_window = max(0, duplicate_window)
        self.reject_zero_reward_preflop_fold = reject_zero_reward_preflop_fold
        self.reject_single_preflop_fold = reject_single_preflop_fold
        self.retrieval_signature_dedup = retrieval_signature_dedup
        self.admission_log_limit = max(1, admission_log_limit)
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

        eligible_records = [
            record for record in self.records if record.source.get("memory_eligible", True)
        ]
        candidates = filter_records_by_scope(eligible_records, observation, self.scope)
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
                "query": build_retrieval_query(observation),
                "retrieved_fact_ids": list(self.last_retrieval),
                "retrieval_scores": list(self.last_scores),
                "candidate_count": len(candidates),
                "returned_count": len(retrieved),
                "below_threshold_count": below_threshold,
                "duplicate_signature_excluded_count": duplicate_excluded,
                "minimum_retrieval_score": self.minimum_retrieval_score,
                "retrieval_threshold_status": self.retrieval_threshold_status,
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
            duplicate.source["duplicate_count"] = int(
                duplicate.source.get("duplicate_count", 0)
            ) + 1
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
        fact_text = _render_fact_text(trajectory, last_event, decisions)
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
                "visibility": "agent_observation_plus_final_reward",
                "quality_policy_version": "graded_fact_admission_v1",
                "quality_label": "admitted_informative",
                "decision_count": len(trajectory.decision_events),
                "showdown_visible_agent_ids": sorted(trajectory.showdown_visible_cards),
                "showdown_visible_cards": trajectory.showdown_visible_cards,
                "decisions": decisions,
                "retrieval_query": build_retrieval_query(last_event.observation),
                "fact_text": fact_text,
                "compact_state": observation_to_compact_text(last_event.observation),
                "structural_signature": signature,
                "duplicate_count": 0,
                **quality,
            },
        )
        self.next_record_index += 1
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
        self._record_admission(
            trajectory,
            "admitted",
            [],
            structural_signature=signature,
            canonical_record_id=record.record_id,
        )

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

        self.admission_counts[status] += 1
        self.last_admission_status = status
        self.last_admission_reasons = list(reasons)
        for reason in reasons:
            self.admission_counts[f"reason:{reason}"] += 1
        self.admission_log.append(
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
        self.admission_log = self.admission_log[-self.admission_log_limit :]

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出事实库快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：快照只含 JSON 结构，便于泛化阶段恢复。
        """

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "schema_version": 5,
                "top_k": self.top_k,
                "max_records": self.max_records,
                "retrieval_backend": self.retrieval_backend,
                "semantic_weight": self.semantic_weight,
                "feature_weight": self.feature_weight,
                "minimum_retrieval_score": self.minimum_retrieval_score,
                "retrieval_threshold_status": self.retrieval_threshold_status,
                "duplicate_window": self.duplicate_window,
                "reject_zero_reward_preflop_fold": self.reject_zero_reward_preflop_fold,
                "reject_single_preflop_fold": self.reject_single_preflop_fold,
                "retrieval_signature_dedup": self.retrieval_signature_dedup,
                "next_record_index": self.next_record_index,
                "admission_log": list(self.admission_log),
                "admission_counts": dict(self.admission_counts),
                "last_admission_status": self.last_admission_status,
                "last_admission_reasons": list(self.last_admission_reasons),
                "retrieval_request_count": self.retrieval_request_count,
                "empty_retrieval_count": self.empty_retrieval_count,
                "retrieval_below_threshold_count": self.retrieval_below_threshold_count,
                "retrieval_duplicate_excluded_count": self.retrieval_duplicate_excluded_count,
                "retrieval_audit_log": list(self.retrieval_audit_log),
                "embedding": self.embedding_backend.audit_metadata(),
                "records": [record.to_dict() for record in self.records],
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

        self.scope = snapshot.scope
        self.top_k = int(snapshot.payload.get("top_k", self.top_k))
        self.max_records = int(snapshot.payload.get("max_records", self.max_records))
        self.retrieval_backend = str(
            snapshot.payload.get("retrieval_backend", self.retrieval_backend)
        )
        self.semantic_weight = float(snapshot.payload.get("semantic_weight", self.semantic_weight))
        self.feature_weight = float(snapshot.payload.get("feature_weight", self.feature_weight))
        raw_minimum = snapshot.payload.get(
            "minimum_retrieval_score", self.minimum_retrieval_score
        )
        self.minimum_retrieval_score = (
            None if raw_minimum is None else float(raw_minimum)
        )
        self.retrieval_threshold_status = str(
            snapshot.payload.get("retrieval_threshold_status", self.retrieval_threshold_status)
        )
        self.duplicate_window = max(
            0, int(snapshot.payload.get("duplicate_window", self.duplicate_window))
        )
        self.reject_zero_reward_preflop_fold = bool(
            snapshot.payload.get(
                "reject_zero_reward_preflop_fold", self.reject_zero_reward_preflop_fold
            )
        )
        self.reject_single_preflop_fold = bool(
            snapshot.payload.get(
                "reject_single_preflop_fold", self.reject_single_preflop_fold
            )
        )
        self.retrieval_signature_dedup = bool(
            snapshot.payload.get(
                "retrieval_signature_dedup", self.retrieval_signature_dedup
            )
        )
        schema_version = int(snapshot.payload.get("schema_version", 1))
        restored_records: list[FactualMemoryRecord] = []
        for raw_record in snapshot.payload.get("records", []):
            record = dict(raw_record)
            source = dict(record.get("source", {}))
            if schema_version < 2 and "memory_eligible" not in source:
                source["memory_eligible"] = False
                source["legacy_unverified"] = True
            record["source"] = source
            restored_records.append(FactualMemoryRecord(**record))
        self.records = restored_records
        self.admission_log = list(snapshot.payload.get("admission_log", []))
        self.admission_counts = Counter(snapshot.payload.get("admission_counts", {}))
        self.last_admission_status = snapshot.payload.get("last_admission_status")
        self.last_admission_reasons = list(
            snapshot.payload.get("last_admission_reasons", [])
        )
        self.retrieval_request_count = int(snapshot.payload.get("retrieval_request_count", 0))
        self.empty_retrieval_count = int(snapshot.payload.get("empty_retrieval_count", 0))
        self.retrieval_below_threshold_count = int(
            snapshot.payload.get("retrieval_below_threshold_count", 0)
        )
        self.retrieval_duplicate_excluded_count = int(
            snapshot.payload.get("retrieval_duplicate_excluded_count", 0)
        )
        self.retrieval_audit_log = list(snapshot.payload.get("retrieval_audit_log", []))
        self.next_record_index = int(
            snapshot.payload.get(
                "next_record_index",
                _next_record_index(self.records, self.agent_id),
            )
        )

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
        }


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
    player_count = max(1, len(last.players))
    position_ratio = last.seat / player_count
    position = "early" if position_ratio < 1 / 3 else "middle" if position_ratio < 2 / 3 else "late"
    decision_facts = build_decision_facts(last)
    draw = dict(decision_facts.get("draw", {}))
    effective_stack = int(decision_facts["effective_stack"])
    action_sequence = ">".join(
        f"{item.get('phase')}:{item.get('action_type')}" for item in decisions
    )
    outcome = (
        "win"
        if trajectory.final_reward > 0
        else "loss"
        if trajectory.final_reward < 0
        else "even"
    )
    features = sorted(observation_features(last))
    return "|".join(
        [
            f"phase={last.phase}",
            f"position={position}",
            f"to_call_pot={_ratio_bucket(last.to_call, last.pot)}",
            f"effective_stack_pot={_ratio_bucket(effective_stack, last.pot)}",
            f"made_hand={decision_facts['made_hand_class']}",
            f"draw={int(bool(draw.get('flush_draw')))}:{int(bool(draw.get('straight_draw')))}",
            f"features={','.join(features)}",
            f"actions={action_sequence}",
            f"outcome={outcome}",
        ]
    )


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
        str(record.source.get("structural_signature", record.record_id))
        for record in records
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


def _decision_view(event: DecisionEvent) -> dict[str, object]:
    """
    功能：把一次决策压缩成事实记忆中的可读片段。
    参数：
        event：决策事件。
    返回：字典。
    副作用：无。
    异常：无。
    设计说明：保留可观察状态和已提交动作，不把模型自述理由升级为历史事实。
    """

    return {
        "phase": event.observation.phase,
        "board": list(event.observation.community_cards),
        "hole": list(event.observation.hole_cards),
        "pot_before": event.observation.pot,
        "to_call": event.observation.to_call,
        "action_type": event.committed_action.action_type,
        "raw_action_type": event.decision.action_type,
        "committed_action_type": event.committed_action.action_type,
        "amount": event.committed_action.amount,
        "guard_repaired": bool(event.llm_metadata.get("guard_repaired")),
        "fallback_used": bool(event.llm_metadata.get("fallback_used")),
    }


def _render_fact_text(
    trajectory: HandTrajectory,
    last_event: DecisionEvent,
    decisions: list[dict[str, object]],
) -> str:
    """
    功能：渲染一条接近原版 FactAgent 的 hand-level fact。
    参数：
        trajectory：手牌轨迹。
        last_event：最后一次决策。
        decisions：决策片段。
    返回：事实文本。
    副作用：无。
    异常：无。
    设计说明：文本只用观察者可见牌面和终局回报，不写入未公开私牌。
    """

    observation = last_event.observation
    board = " ".join(observation.community_cards) if observation.community_cards else "(空)"
    hole = " ".join(observation.hole_cards) if observation.hole_cards else "(空)"
    decision_lines = []
    for item in decisions:
        amount = f"({item['amount']})" if item.get("amount") else ""
        decision_lines.append(
            f"  [{item['phase']}] observed_action={item['action_type']}{amount}"
        )
    decision_text = "\n".join(decision_lines) or "  (无)"
    outcome = (
        "win" if trajectory.final_reward > 0 else "loss" if trajectory.final_reward < 0 else "even"
    )
    visible_showdown = ",".join(sorted(trajectory.showdown_visible_cards)) or "none"
    return (
        f"[hand {trajectory.hand_id}, 终局阶段={observation.phase}]\n"
        f"末态 board={board}, hole={hole}, pot={observation.pot}, to_call={observation.to_call}\n"
        f"我的决策序列:\n{decision_text}\n"
        f"hand_outcome: {outcome} "
        f"(净收益 {trajectory.final_reward:+d}, 终局筹码 {trajectory.final_stack})\n"
        f"showdown_visible_agent_ids: {visible_showdown}\n"
        f"summary: {trajectory.summary}"
    )
