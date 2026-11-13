from __future__ import annotations

from typing import Protocol, TypeVar

from agentmemeval.path1_v1.core.domain import (
    ActionDecision,
    AgentId,
    AgentObservation,
    DecisionEvent,
    ExperimentResult,
    HandResult,
    HandTrajectory,
    LegalActionSet,
    MemoryContext,
    MemorySnapshot,
    StepResult,
    TableSpec,
)

T = TypeVar("T")


class PokerEnvironment(Protocol):
    def reset(self, table_spec: TableSpec, seed: int) -> None:
        pass

    def current_agent_id(self) -> AgentId | None:
        pass

    def current_observation(self, agent_id: AgentId) -> AgentObservation:
        pass

    def legal_actions(self, agent_id: AgentId) -> LegalActionSet:
        pass

    def step(self, agent_id: AgentId, action: ActionDecision) -> StepResult:
        pass

    def is_hand_finished(self) -> bool:
        pass

    def finalize_hand(self) -> HandResult:
        pass


class LLMClient(Protocol):
    def generate_structured(self, request: object, schema: type[T]) -> T:
        pass

    def healthcheck(self) -> dict[str, object]:
        pass


class MemoryMechanism(Protocol):
    name: str

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        pass

    def on_decision_committed(self, event: DecisionEvent) -> None:
        pass

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        pass

    def snapshot(self) -> MemorySnapshot:
        pass

    def restore(self, snapshot: MemorySnapshot) -> None:
        pass

    def metrics(self) -> dict[str, object]:
        pass


class AgentPolicy(Protocol):
    agent_id: AgentId

    def decide(
        self,
        observation: AgentObservation,
    ) -> tuple[ActionDecision, MemoryContext, dict[str, object]]:
        pass

    def observe_decision(self, event: DecisionEvent) -> None:
        pass

    def observe_hand_finished(self, trajectory: HandTrajectory) -> None:
        pass

    def snapshot_memory(self) -> MemorySnapshot:
        pass

    def restore_memory(self, snapshot: MemorySnapshot) -> None:
        pass


class ExperimentScenario(Protocol):
    def run(self, context: object) -> ExperimentResult:
        pass
