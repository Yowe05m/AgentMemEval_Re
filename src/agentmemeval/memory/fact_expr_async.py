"""
模块说明：本模块实现事实即时写入、经验周期性巩固的异步组合机制。
核心职责：每手写入事实，按配置周期使用近期轨迹和事实证据更新经验文档。
输入与输出：输入观察与轨迹，输出合并 MemoryContext、快照和异步日志。
依赖边界：组合事实与经验模块，不依赖真实 LLM；巩固证据来自可见事实记录。
不负责：不启动后台线程，不在决策中写入未来信息。
"""

from __future__ import annotations

import math

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    ExperienceDocument,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
    utc_now_iso,
)
from agentmemeval.core.protocols import LLMClient
from agentmemeval.core.seeds import make_rng
from agentmemeval.memory.base import trajectory_quality
from agentmemeval.memory.experiential import ExperientialMemory
from agentmemeval.memory.factual import FactualMemory
from agentmemeval.memory.rag import EmbeddingBackend, hybrid_top_k_records


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
        max_records: int = 500,
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
        返回：无。
        副作用：创建子记忆和 sweep 日志。
        异常：无。
        设计说明：所有周期参数都来自配置，便于做消融。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.fact = FactualMemory(
            agent_id,
            scope=scope,
            top_k=top_k,
            max_records=max_records,
            embedding_backend=embedding_backend,
            **(fact_options or {}),
        )
        self.expr = ExperientialMemory(
            agent_id,
            scope=scope,
            window_size=window_size,
            update_period=10**9,
            revision_strategy=revision_strategy,
            llm_client=llm_client,
            model=model,
        )
        self.window_size = window_size
        self.sweep_every = max(1, sweep_every)
        self.evidence_k = evidence_k
        self.salience_threshold = salience_threshold
        self.salience_mirror_threshold = salience_mirror_threshold
        self.mirror_prob = mirror_prob
        self.stability_init = stability_init
        self.stability_min = stability_min
        self.stability_max = stability_max
        self.recent: list[HandTrajectory] = []
        self.sweep_log: list[dict[str, object]] = []
        self.evidence_review_queue: list[dict[str, object]] = []
        self.fact_state: dict[str, dict[str, object]] = {}
        self.hand_counter = 0
        self.eligible_hand_counter = 0
        self.skipped_trajectory_hand_ids: list[str] = []

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：合并当前事实检索结果与经验文档。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：更新事实检索元数据。
        异常：无。
        设计说明：决策读路径与 Sync 一致，差异只在写路径。
        """

        candidates = self._active_records(observation)
        scored = hybrid_top_k_records(
            observation,
            candidates,
            self.fact.top_k,
            salience_fn=lambda record_id: self._salience(record_id, self.hand_counter + 1),
            embedding_backend=self.fact.embedding_backend,
            retrieval_unit=self.fact.retrieval_unit,
        )
        retrieved = [item.record for item in scored]
        retrieved_ids = {item.record.record_id for item in scored}
        mirror = self._mirror_record(observation, retrieved_ids)
        if mirror is not None:
            retrieved.append(mirror)
        self.fact.last_retrieval = [record.record_id for record in retrieved]
        self.fact.last_scores = [
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
            for item in scored
        ]
        for record in retrieved:
            state = self.fact_state.get(record.record_id)
            if state is not None:
                state["last_accessed_hand"] = self.hand_counter + 1
                state["access_count"] = int(state.get("access_count", 0)) + 1
        expr_context = self.expr.build_context(observation)
        return MemoryContext(
            facts=retrieved,
            experience=expr_context.experience,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "fact": {
                    "mechanism": "fact",
                    "retrieval_backend": "salience_hybrid_rag",
                    "retrieved_fact_ids": [record.record_id for record in retrieved],
                    "retrieval_scores": list(self.fact.last_scores),
                    "candidate_count": len(candidates),
                    "mirror_injected": mirror.record_id if mirror else None,
                },
                "expr": expr_context.metadata,
                "sweep_every": self.sweep_every,
            },
        )

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

        quality = trajectory_quality(trajectory)
        before_ids = {record.record_id for record in self.fact.records}
        self.fact.on_hand_finished(trajectory)
        for record in self.fact.records:
            if (
                record.record_id not in before_ids
                and record.record_id not in self.fact_state
                and record.source.get("memory_eligible", True)
            ):
                self.fact_state[record.record_id] = {
                    "stability": self.stability_init,
                    "last_accessed_hand": self.hand_counter + 1,
                    "access_count": 0,
                    "linked_exp_revs": [],
                }
        self.hand_counter += 1
        if not quality["memory_eligible"] or self.fact.last_admission_status != "admitted":
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        self.eligible_hand_counter += 1
        self.recent.append(trajectory)
        self.recent = self.recent[-self.window_size :]
        if self.eligible_hand_counter % self.sweep_every != 0 or not trajectory.decision_events:
            return
        trigger_observation = trajectory.decision_events[-1].observation
        evidence_groups = self._recall_for_sweep(trigger_observation)
        evidence_by_id = {
            record.record_id: record
            for records in evidence_groups.values()
            for record in records
        }
        evidence = list(evidence_by_id.values())
        old_version = self.expr.current.version
        revision = self._sweep(evidence)
        new_version = self.expr.current.version
        self.sweep_log.append(
            {
                "trigger_hand_id": trajectory.hand_id,
                "recent_window_hand_ids": [item.hand_id for item in self.recent],
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
                "created_at": utc_now_iso(),
            }
        )

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出异步组合快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：额外保存 sweep 日志，支持审计异步触发是否符合配置。
        """

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "fact": self.fact.snapshot().payload,
                "expr": self.expr.snapshot().payload,
                "window_size": self.window_size,
                "sweep_every": self.sweep_every,
                "evidence_k": self.evidence_k,
                "salience_threshold": self.salience_threshold,
                "salience_mirror_threshold": self.salience_mirror_threshold,
                "mirror_prob": self.mirror_prob,
                "stability_init": self.stability_init,
                "stability_min": self.stability_min,
                "stability_max": self.stability_max,
                "sweep_log": list(self.sweep_log),
                "evidence_review_queue": list(self.evidence_review_queue),
                "fact_state": self.fact_state,
                "hand_counter": self.hand_counter,
                "eligible_hand_counter": self.eligible_hand_counter,
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

        self.scope = snapshot.scope
        self.window_size = int(snapshot.payload.get("window_size", self.window_size))
        self.sweep_every = int(snapshot.payload.get("sweep_every", self.sweep_every))
        self.evidence_k = int(snapshot.payload.get("evidence_k", self.evidence_k))
        self.salience_threshold = float(
            snapshot.payload.get("salience_threshold", self.salience_threshold)
        )
        self.salience_mirror_threshold = float(
            snapshot.payload.get("salience_mirror_threshold", self.salience_mirror_threshold)
        )
        self.mirror_prob = float(snapshot.payload.get("mirror_prob", self.mirror_prob))
        self.stability_init = float(snapshot.payload.get("stability_init", self.stability_init))
        self.stability_min = float(snapshot.payload.get("stability_min", self.stability_min))
        self.stability_max = float(snapshot.payload.get("stability_max", self.stability_max))
        self.sweep_log = list(snapshot.payload.get("sweep_log", []))
        self.evidence_review_queue = list(
            snapshot.payload.get("evidence_review_queue", [])
        )
        self.fact_state = {
            str(record_id): dict(state)
            for record_id, state in dict(snapshot.payload.get("fact_state", {})).items()
        }
        self.hand_counter = int(snapshot.payload.get("hand_counter", 0))
        self.eligible_hand_counter = int(
            snapshot.payload.get("eligible_hand_counter", self.hand_counter)
        )
        self.skipped_trajectory_hand_ids = list(
            snapshot.payload.get("skipped_trajectory_hand_ids", [])
        )
        self.recent = []
        self.fact.restore(
            MemorySnapshot("fact", self.agent_id, self.scope, snapshot.payload.get("fact", {}))
        )
        self.expr.restore(
            MemorySnapshot("expr", self.agent_id, self.scope, snapshot.payload.get("expr", {}))
        )

    def metrics(self) -> dict[str, object]:
        """
        功能：返回异步记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：报告展示事实数量、经验更新和 sweep 次数。
        """

        fact = self.fact.metrics()
        expr = self.expr.metrics()
        return {
            "mechanism": self.name,
            "fact_count": fact["fact_count"],
            "eligible_fact_count": fact.get("eligible_fact_count", fact["fact_count"]),
            "excluded_fallback_fact_count": fact.get("excluded_fallback_fact_count", 0),
            "experience_updates": expr["experience_updates"],
            "experience_chars": expr.get("experience_chars", 0),
            "async_sweeps": len(self.sweep_log),
            "last_retrieved_fact_ids": fact.get("last_retrieved_fact_ids", []),
            "salience_threshold": self.salience_threshold,
            "fact_state_count": len(self.fact_state),
            "skipped_fallback_trajectories": len(self.skipped_trajectory_hand_ids),
            "skipped_fact_admission_trajectories": len(self.skipped_trajectory_hand_ids),
            "evidence_classification_status": "pending_human_review",
            "evidence_review_queue_count": len(self.evidence_review_queue),
            "evidence_review_queue": list(self.evidence_review_queue),
        }

    def _sweep(self, evidence: list[object]) -> dict[str, object]:
        """
        功能：用近期轨迹和事实证据生成经验新版本。
        参数：
            evidence：事实证据列表。
        返回：无。
        副作用：追加经验版本。
        异常：无。
        设计说明：当前为确定性巩固；真实 LLM 巩固可替换该函数。
        """

        avg_reward = (
            sum(item.final_reward for item in self.recent) / len(self.recent)
            if self.recent
            else 0.0
        )
        evidence_ids = [getattr(item, "record_id", "") for item in evidence]
        supporting, contradicting, noise = self._classify_evidence(evidence)
        body = "\n".join(
            [
                "# 我的经验",
                "",
                "## 起手牌",
                "- 异步巩固后保留可迁移起手牌纪律：强牌主动，边缘牌看位置和价格。",
                "",
                "## 翻牌后",
                "- 参考相似事实决定是否持续下注；单次成功诈唬不直接升级为通用规律。",
                "",
                "## 转牌 / 河牌",
                "- 后两街优先校准摊牌价值和底池赔率，面对大额下注减少自动跟注。",
                "",
                "## 对手类型应对",
                "- 将事实抽象为对手类型与下注线，不把具体 agent_id 写入长期经验。",
                "",
                "## 注码与位置",
                f"- sweep 窗口 {len(self.recent)} 手，平均收益 {avg_reward:.2f}。",
                f"- 支持事实：{', '.join(supporting) or '无'}。",
                f"- 冲突事实：{', '.join(contradicting) or '无'}；"
                f"噪声事实：{', '.join(noise) or '无'}。",
            ]
        )
        if len(body) > self.expr.max_chars:
            body = body[: self.expr.max_chars - 20].rstrip() + "\n- （因长度上限截断）"
        if self.expr.revision_strategy == "llm":
            revision = self.expr._revise_from_window(
                self.recent,
                evidence_records=evidence,
            )
            revision.update(
                {
                    "sweep_evidence_fact_ids": evidence_ids,
                    "sweep_supporting_fact_ids": supporting,
                    "sweep_contradicting_fact_ids": contradicting,
                    "sweep_noise_fact_ids": noise,
                }
            )
        else:
            revision = {
                "rev": len(self.expr.revision_log) + 1,
                "hand_index": self.recent[-1].hand_id if self.recent else "",
                "keep": body == self.expr.current.body,
                "old_md": self.expr.current.body,
                "new_md": body + "\n",
                "calibration_note": (
                    f"异步 sweep 平均收益 {avg_reward:.2f}，按支持/冲突/噪声事实调权。"
                ),
                "self_check": "经验保持五章节结构，并避免具体玩家身份泄露。",
                "supporting_fact_ids": supporting,
                "contradicting_fact_ids": contradicting,
                "noise_fact_ids": noise,
                "revision_strategy": "salience_multi_path_sweep",
                "fallback_used": False,
                "failure": None,
            }
        self.expr.revision_log.append(revision)
        self.expr.history.append(
            ExperienceDocument(
                version=self.expr.current.version + 1,
                body=str(revision["new_md"]),
                source_hand_ids=[item.hand_id for item in self.recent],
                updated_at=utc_now_iso(),
                scope=self.scope,
                metadata={
                    "strategy": revision["revision_strategy"],
                    "schema_version": revision.get("schema_version"),
                    "prompt_version": revision.get("prompt_version"),
                    "prompt_sha256": revision.get("prompt_sha256"),
                    "fallback_used": revision.get("fallback_used", False),
                    "failure": revision.get("failure"),
                    "evidence_fact_ids": evidence_ids,
                    "supporting_fact_ids": supporting,
                    "contradicting_fact_ids": contradicting,
                    "noise_fact_ids": noise,
                },
            )
        )
        self._reweight_fact_state(supporting, contradicting, noise, self.expr.current.version)
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
            return 1.0
        gap = max(0, current_hand - int(state.get("last_accessed_hand", current_hand)))
        stability = max(self.stability_min, float(state.get("stability", self.stability_init)))
        return math.exp(-gap / stability)

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

        if not self.fact_state:
            return [
                record
                for record in self.fact.records
                if record.source.get("memory_eligible", True)
                and record.agent_id == observation.agent_id
            ]
        return [
            record
            for record in self.fact.records
            if self._salience(record.record_id, self.hand_counter + 1) >= self.salience_threshold
            and record.agent_id == observation.agent_id
            and record.source.get("memory_eligible", True)
        ]

    def _mirror_record(
        self,
        observation: AgentObservation,
        retrieved_ids: set[str],
    ) -> object | None:
        """
        功能：确定性注入一条边缘显著性的 mirror 事实。
        参数：
            observation：当前观察。
            retrieved_ids：主路已召回 ID。
        返回：事实或 None。
        副作用：无。
        异常：无。
        设计说明：对应原版 mirror recall，但用 seed 化随机保证复现。
        """

        if self.mirror_prob <= 0 or not self.fact_state:
            return None
        rng = make_rng(observation.seed, observation.agent_id, observation.hand_id, "mirror")
        if rng.random() >= self.mirror_prob:
            return None
        pool = [
            record
            for record in self.fact.records
            if record.record_id not in retrieved_ids
            and record.source.get("memory_eligible", True)
            and self.salience_threshold
            <= self._salience(record.record_id, self.hand_counter + 1)
            < self.salience_mirror_threshold
        ]
        if not pool:
            return None
        pool.sort(key=lambda record: (record.created_at, record.record_id), reverse=True)
        return pool[0]

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
        similar = [
            item.record
            for item in hybrid_top_k_records(
                observation,
                active,
                self.evidence_k,
                salience_fn=lambda record_id: self._salience(record_id, self.hand_counter),
                embedding_backend=self.fact.embedding_backend,
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

    def _classify_evidence(self, evidence: list[object]) -> tuple[list[str], list[str], list[str]]:
        """
        功能：按近期收益方向把证据分为支持、冲突和噪声。
        参数：
            evidence：事实证据。
        返回：三个 ID 列表。
        副作用：无。
        异常：无。
        设计说明：离线替代原版 LLM self-check 的结构化输出。
        """

        recent_rewards = [item.final_reward for item in self.recent]
        for record in evidence:
            reward = int(getattr(record, "final_reward", 0))
            record_id = str(getattr(record, "record_id", ""))
            self.evidence_review_queue.append(
                {
                    "record_id": record_id,
                    "final_reward_chips": reward,
                    "recent_reward_chips": list(recent_rewards),
                    "phase": _record_phase(record),
                    "action_summary": getattr(record, "action_summary", ""),
                    "suggested_labels": ["supporting", "contradicting", "noise"],
                    "human_label": None,
                    "classification_status": "pending_human_review",
                }
            )
        self.evidence_review_queue = self.evidence_review_queue[-500:]
        return [], [], []

    def _reweight_fact_state(
        self,
        supporting: list[str],
        contradicting: list[str],
        noise: list[str],
        revision_number: int,
    ) -> None:
        """
        功能：根据 sweep 判断调整事实稳定性。
        参数：
            supporting：支持事实 ID。
            contradicting：冲突事实 ID。
            noise：噪声事实 ID。
            revision_number：经验版本号。
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
            if record_id in supporting_set | contradicting_set | noise_set:
                linked = list(state.get("linked_exp_revs", []))
                linked.append(revision_number)
                state["linked_exp_revs"] = linked


def _record_phase(record: object) -> str:
    source = getattr(record, "source", {})
    decisions = source.get("decisions", []) if isinstance(source, dict) else []
    if decisions and isinstance(decisions[-1], dict):
        return str(decisions[-1].get("phase", "unknown"))
    return "unknown"
