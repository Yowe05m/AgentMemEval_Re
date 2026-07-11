"""
模块说明：本模块实现空记忆和记忆基类辅助逻辑。
核心职责：提供 NoMemory 基线所需的 NullMemory，以及作用域过滤工具。
输入与输出：输入观察、事件、轨迹和快照，输出记忆上下文或指标。
依赖边界：依赖核心领域对象，不依赖 LLM Provider。
不负责：不实现事实检索和经验摘要的具体算法。
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


class NullMemory:
    """
    功能：实现完全不读写的 NoMemory 基线。
    参数：
        agent_id：所属 Agent。
        scope：作用域配置，仅用于快照记录。
    返回：空记忆机制。
    副作用：无。
    异常：无。
    设计说明：基线必须真实存在，不能通过 None 分支隐式实现。
    """

    name = "no_memory"

    def __init__(self, agent_id: str, scope: MemoryScope = "per_agent") -> None:
        """
        功能：初始化空记忆。
        参数：
            agent_id：所属 Agent。
            scope：记忆作用域。
        返回：无。
        副作用：保存配置。
        异常：无。
        设计说明：即使不写记忆，也保留 snapshot 以统一实验流程。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：返回空记忆上下文。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：无。
        异常：无。
        设计说明：NoMemory Agent 仍走同一决策管线，确保消融公平。
        """

        return MemoryContext(metadata={"mechanism": self.name, "scope": self.scope})

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：忽略已提交决策。
        参数：
            event：决策事件。
        返回：无。
        副作用：无。
        异常：无。
        设计说明：保留生命周期钩子，避免实验层特殊分支。
        """

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：忽略手牌结束轨迹。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：无。
        异常：无。
        设计说明：NoMemory 不写入任何事实或经验。
        """

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出空记忆快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：泛化流程可以统一保存和恢复快照。
        """

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={},
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：恢复空记忆快照。
        参数：
            snapshot：快照。
        返回：无。
        副作用：更新作用域。
        异常：无。
        设计说明：空记忆只需要接受统一接口，不需要读取 payload。
        """

        self.scope = snapshot.scope

    def metrics(self) -> dict[str, object]:
        """
        功能：返回空记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：聚合报告可以统一读取记忆指标。
        """

        return {"mechanism": self.name, "fact_count": 0, "experience_updates": 0}


def filter_records_by_scope(
    records: list[FactualMemoryRecord],
    observation: AgentObservation,
    scope: MemoryScope,
) -> list[FactualMemoryRecord]:
    """
    功能：按记忆作用域过滤事实记录。
    参数：
        records：候选事实。
        observation：当前观察。
        scope：作用域。
    返回：可见候选事实。
    副作用：无。
    异常：无。
    设计说明：per_table 与 per_agent 用于区分跨桌携带记忆的实验假设。
    """

    if scope == "global":
        return list(records)
    if scope == "per_table":
        return [record for record in records if record.table_id == observation.table_id]
    if scope == "per_agent":
        return [record for record in records if record.agent_id == observation.agent_id]
    return [record for record in records if record.agent_id == observation.agent_id]


def trajectory_quality(trajectory: HandTrajectory) -> dict[str, int | bool]:
    """Summarize guard interventions before a trajectory is used for learning."""

    fallback_count = sum(
        bool(event.llm_metadata.get("fallback_used"))
        for event in trajectory.decision_events
    )
    repaired_count = sum(
        bool(event.llm_metadata.get("guard_repaired"))
        for event in trajectory.decision_events
    )
    action_mismatch_count = sum(
        event.decision.action_type != event.committed_action.action_type
        for event in trajectory.decision_events
    )
    return {
        "fallback_count": fallback_count,
        "repaired_count": repaired_count,
        "action_mismatch_count": action_mismatch_count,
        "memory_eligible": fallback_count == 0 and action_mismatch_count == 0,
    }
