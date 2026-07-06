"""
模块说明：本模块实现事实记忆与经验记忆同步并行机制。
核心职责：决策时同时提供事实检索和经验文档，手牌结束后分别更新两类记忆。
输入与输出：输入观察与轨迹，输出合并 MemoryContext 和组合快照。
依赖边界：组合 FactualMemory 与 ExperientialMemory，不依赖 LLM 或环境内部状态。
不负责：不让经验更新读取未来不可见信息，不决定动作。
"""

from __future__ import annotations

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
)
from agentmemeval.memory.experiential import ExperientialMemory
from agentmemeval.memory.factual import FactualMemory


class FactExprSyncMemory:
    """
    功能：组合事实和经验两类记忆。
    参数：
        agent_id：所属 Agent。
        scope：记忆作用域。
        top_k：事实检索条数。
        window_size：经验窗口。
    返回：同步组合记忆。
    副作用：手牌结束时更新两个子记忆。
    异常：无。
    设计说明：两个子记忆并行更新，经验只读轨迹，不读取刚写入事实的私有扩展字段。
    """

    name = "fact_expr_sync"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        top_k: int = 8,
        window_size: int = 8,
        max_records: int = 500,
        retrieval_backend: str = "hybrid_rag",
    ) -> None:
        """
        功能：初始化组合记忆。
        参数：
            agent_id：所属 Agent。
            scope：作用域。
            top_k：事实检索条数。
            window_size：经验窗口。
            max_records：事实容量。
        返回：无。
        副作用：创建子记忆。
        异常：无。
        设计说明：组合层只协调生命周期，算法仍在子模块内。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.fact = FactualMemory(
            agent_id,
            scope=scope,
            top_k=top_k,
            max_records=max_records,
            retrieval_backend=retrieval_backend,
        )
        self.expr = ExperientialMemory(agent_id, scope=scope, window_size=window_size)

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
                "fact": fact_context.metadata,
                "expr": expr_context.metadata,
            },
        )

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
        设计说明：两个更新只消费同一可见轨迹，避免同步组合泄露上帝视角。
        """

        self.fact.on_hand_finished(trajectory)
        self.expr.on_hand_finished(trajectory)

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
                "fact": self.fact.snapshot().payload,
                "expr": self.expr.snapshot().payload,
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

        self.scope = snapshot.scope
        self.fact.restore(
            MemorySnapshot("fact", self.agent_id, self.scope, snapshot.payload.get("fact", {}))
        )
        self.expr.restore(
            MemorySnapshot("expr", self.agent_id, self.scope, snapshot.payload.get("expr", {}))
        )

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
            "fact_count": fact["fact_count"],
            "experience_updates": expr["experience_updates"],
            "experience_chars": expr.get("experience_chars", 0),
            "last_retrieved_fact_ids": fact.get("last_retrieved_fact_ids", []),
        }
