from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal

AgentId = str
TableId = str
HandId = str
RunId = str
JsonDict = dict[str, Any]
ActionName = Literal["fold", "check", "call", "raise"]
MemoryScope = Literal["per_agent", "per_table", "global", "per_opponent_cluster"]


def utc_now_iso() -> str:

    return datetime.now(UTC).isoformat()


def to_jsonable(value: Any) -> Any:

    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class Serializable:
    def to_dict(self) -> JsonDict:

        return to_jsonable(self)


@dataclass(slots=True)
class LegalAction(Serializable):
    action_type: ActionName
    min_amount: int | None = None
    max_amount: int | None = None
    reopens: bool = True


@dataclass(slots=True)
class LegalActionSet(Serializable):
    actions: list[LegalAction]

    def types(self) -> set[str]:

        return {action.action_type for action in self.actions}

    def rule_for(self, action_type: str) -> LegalAction | None:

        for action in self.actions:
            if action.action_type == action_type:
                return action
        return None


@dataclass(slots=True)
class ActionDecision(Serializable):
    action_type: str
    amount: int | None = None
    confidence: float = 1.0
    reason_summary: str = ""
    raw_response: JsonDict = field(default_factory=dict)
    intent: str = ""


@dataclass(slots=True)
class PlayerPublicState(Serializable):
    agent_id: AgentId
    seat: int
    stack: int
    current_bet: int
    total_committed: int
    folded: bool
    all_in: bool
    busted: bool = False


@dataclass(slots=True)
class AgentObservation(Serializable):
    agent_id: AgentId
    table_id: TableId
    hand_id: HandId
    phase: str
    seat: int
    hole_cards: list[str]
    community_cards: list[str]
    pot: int
    current_bet: int
    to_call: int
    players: list[PlayerPublicState]
    action_history: list[JsonDict]
    legal_actions: LegalActionSet
    seed: int
    dealer_seat: int | None = None
    small_blind_seat: int | None = None
    big_blind_seat: int | None = None
    small_blind_amount: int | None = None
    big_blind_amount: int | None = None


@dataclass(slots=True)
class DecisionEvent(Serializable):
    agent_id: AgentId
    table_id: TableId
    hand_id: HandId
    observation: AgentObservation
    decision: ActionDecision
    committed_action: ActionDecision
    memory_context: MemoryContext
    llm_metadata: JsonDict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class HandTrajectory(Serializable):
    agent_id: AgentId
    table_id: TableId
    hand_id: HandId
    decision_events: list[DecisionEvent]
    public_actions: list[JsonDict]
    final_reward: int
    final_stack: int
    showdown_visible_cards: dict[AgentId, list[str]]
    summary: str
    won_amount: int | None = None
    final_community_cards: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FactualMemoryRecord(Serializable):
    record_id: str
    agent_id: AgentId
    table_id: TableId
    hand_id: HandId
    scope: MemoryScope
    state_summary: str
    action_summary: str
    final_reward: int
    features: list[str]
    source: JsonDict
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ExperienceDocument(Serializable):
    version: int
    body: str
    source_hand_ids: list[HandId]
    updated_at: str
    scope: MemoryScope
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class MemoryContext(Serializable):
    facts: list[FactualMemoryRecord] = field(default_factory=list)
    experience: ExperienceDocument | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class MemorySnapshot(Serializable):
    mechanism: str
    agent_id: AgentId
    scope: MemoryScope
    payload: JsonDict
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class TableSpec(Serializable):
    table_id: TableId
    agent_ids: list[AgentId]
    starting_stacks: dict[AgentId, int]
    small_blind: int = 1
    big_blind: int = 2
    max_raises_per_street: int = 4
    dealer_index: int = 0
    hand_number: int = 1


@dataclass(slots=True)
class StepResult(Serializable):
    event: JsonDict
    hand_finished: bool


@dataclass(slots=True)
class HandResult(Serializable):
    table_id: TableId
    hand_id: HandId
    rewards: dict[AgentId, int]
    final_stacks: dict[AgentId, int]
    public_actions: list[JsonDict]
    winners: list[JsonDict]
    showdown_visible_cards: dict[AgentId, list[str]]
    showdown_ranks: dict[AgentId, str] = field(default_factory=dict)
    community_cards: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunManifest(Serializable):
    run_id: RunId
    scenario: str
    seed: int
    config_snapshot_path: str
    output_dir: str
    code_version: str
    provider: str
    model: str
    metadata: JsonDict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ExperimentResult(Serializable):
    run_id: RunId
    scenario: str
    metrics: JsonDict
    aggregate_metrics: JsonDict
    artifacts: JsonDict
    notes: list[str] = field(default_factory=list)
