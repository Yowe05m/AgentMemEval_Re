from __future__ import annotations

from agentmemeval.path1_v1.core.domain import (
    AgentObservation,
    DecisionEvent,
    FactualMemoryRecord,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
)


class NullMemory:
    name = "no_memory"

    def __init__(self, agent_id: str, scope: MemoryScope = "per_agent") -> None:

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.frozen = False

    def set_frozen(self, frozen: bool) -> None:
        """Accept the shared frozen lifecycle; NullMemory remains stateless."""

        self.frozen = bool(frozen)

    def build_context(self, observation: AgentObservation) -> MemoryContext:

        return MemoryContext(metadata={"mechanism": self.name, "scope": self.scope})

    def on_decision_committed(self, event: DecisionEvent) -> None:
        pass

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        pass

    def snapshot(self) -> MemorySnapshot:

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={},
        )

    def restore(self, snapshot: MemorySnapshot) -> None:

        self.scope = snapshot.scope

    def metrics(self) -> dict[str, object]:

        return {"mechanism": self.name, "fact_count": 0, "experience_updates": 0}


def filter_records_by_scope(
    records: list[FactualMemoryRecord],
    observation: AgentObservation,
    scope: MemoryScope,
) -> list[FactualMemoryRecord]:

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
        bool(event.llm_metadata.get("fallback_used")) for event in trajectory.decision_events
    )
    repaired_count = sum(
        bool(event.llm_metadata.get("guard_repaired")) for event in trajectory.decision_events
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
