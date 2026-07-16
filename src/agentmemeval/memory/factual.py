"""
模块说明：本模块实现 FactAgent 使用的结构化事实记忆。
核心职责：在手牌结束后写入事实记录，决策时按相似特征检索 Top-k。
输入与输出：输入观察与可见轨迹，输出 MemoryContext 和 MemorySnapshot。
依赖边界：不依赖 LLM，不访问环境内部私牌，只消费 HandTrajectory。
不负责：不维护经验文档，不决定动作。
"""

from __future__ import annotations

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    FactualMemoryRecord,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
)
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
        self.records: list[FactualMemoryRecord] = []
        self.next_record_index = 1
        self.last_retrieval: list[str] = []
        self.last_scores: list[dict[str, object]] = []

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
        if self.retrieval_backend == "feature_jaccard":
            scored = top_k_records(observation, candidates, self.top_k)
            retrieved = [record for record, _score in scored]
            self.last_scores = [
                {
                    "record_id": record.record_id,
                    "score": score,
                    "feature": score,
                    "semantic": None,
                    "salience": None,
                }
                for record, score in scored
            ]
        else:
            scored_rag = hybrid_top_k_records(
                observation,
                candidates,
                self.top_k,
                semantic_weight=self.semantic_weight,
                feature_weight=self.feature_weight,
                embedding_backend=self.embedding_backend,
            )
            retrieved = [item.record for item in scored_rag]
            self.last_scores = [
                {
                    "record_id": item.record.record_id,
                    "score": item.score,
                    "semantic": item.semantic,
                    "feature": item.feature,
                    "salience": item.salience,
                }
                for item in scored_rag
            ]
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
                "excluded_fallback_fact_count": len(self.records) - len(eligible_records),
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
            return
        last_event = trajectory.decision_events[-1]
        quality = trajectory_quality(trajectory)
        decisions = [_decision_view(event) for event in trajectory.decision_events]
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
                "decision_count": len(trajectory.decision_events),
                "showdown_visible_agent_ids": sorted(trajectory.showdown_visible_cards),
                "showdown_visible_cards": trajectory.showdown_visible_cards,
                "decisions": decisions,
                "retrieval_query": build_retrieval_query(last_event.observation),
                "fact_text": fact_text,
                "compact_state": observation_to_compact_text(last_event.observation),
                **quality,
            },
        )
        self.next_record_index += 1
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

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
                "schema_version": 3,
                "top_k": self.top_k,
                "max_records": self.max_records,
                "retrieval_backend": self.retrieval_backend,
                "semantic_weight": self.semantic_weight,
                "feature_weight": self.feature_weight,
                "next_record_index": self.next_record_index,
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
            "excluded_fallback_fact_count": sum(
                not record.source.get("memory_eligible", True) for record in self.records
            ),
            "experience_updates": 0,
            "last_retrieved_fact_ids": list(self.last_retrieval),
            "retrieval_backend": self.retrieval_backend,
            "last_retrieval_scores": list(self.last_scores),
            "embedding": self.embedding_backend.audit_metadata(),
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
    设计说明：保留 intent/reason 但不保存长链路原始回复。
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
        "intent": event.decision.reason_summary or event.committed_action.reason_summary,
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
        intent = item.get("intent") or "(无)"
        decision_lines.append(
            f"  [{item['phase']}] intent={intent} action={item['action_type']}{amount}"
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
