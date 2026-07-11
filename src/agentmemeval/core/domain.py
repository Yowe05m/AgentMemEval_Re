"""
模块说明：本模块定义实验平台的结构化领域对象。
核心职责：描述 Agent、桌面、动作、观察、轨迹、记忆、运行清单和结果。
输入与输出：输入为环境、记忆和实验层传入的数据，输出为可 JSON 化数据对象。
依赖边界：只依赖标准库 dataclass，不依赖具体扑克引擎、LLM SDK 或存储实现。
不负责：不执行动作合法性校验，不推进环境，不计算最终指标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal

AgentId = str
TableId = str
HandId = str
RunId = str
JsonDict = dict[str, Any]
ActionName = Literal["fold", "check", "call", "raise"]
MemoryScope = Literal["per_agent", "per_table", "global", "per_opponent_cluster"]


def utc_now_iso() -> str:
    """
    功能：生成带 UTC 时区的 ISO 时间字符串。
    参数：无。
    返回：ISO-8601 字符串。
    副作用：读取系统时间。
    异常：无。
    设计说明：所有工件用统一时间格式，便于跨平台比较与排序。
    """

    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """
    功能：把 dataclass、列表和字典递归转换为 JSON 可序列化结构。
    参数：
        value：任意对象。
    返回：JSON 友好的对象。
    副作用：无。
    异常：无。
    设计说明：领域对象不绑定具体存储层，统一转换可减少 pickle 依赖。
    """

    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class Serializable:
    """
    功能：为领域 dataclass 提供统一的字典转换入口。
    参数：无。
    返回：可 JSON 化字典。
    副作用：无。
    异常：无。
    设计说明：子类保持轻量，不引入 pydantic 等额外运行时依赖。
    """

    def to_dict(self) -> JsonDict:
        """
        功能：转换当前对象为 JSON 友好字典。
        参数：无。
        返回：字典。
        副作用：无。
        异常：无。
        设计说明：存储层只处理字典，避免认识所有领域类。
        """

        return to_jsonable(self)


@dataclass(slots=True)
class LegalAction(Serializable):
    """
    功能：描述一个当前可执行动作及其金额边界。
    参数：
        action_type：动作类型。
        min_amount：raise 时的最小加注到总额。
        max_amount：raise 时的最大加注到总额。
        reopens：短码 all-in 是否重开行动。
    返回：动作规则对象。
    副作用：无。
    异常：无。
    设计说明：raise 金额采用“加注到的总额”，与旧仓库外显行为保持一致。
    """

    action_type: ActionName
    min_amount: int | None = None
    max_amount: int | None = None
    reopens: bool = True


@dataclass(slots=True)
class LegalActionSet(Serializable):
    """
    功能：封装当前 Agent 的合法动作集合。
    参数：
        actions：合法动作规则列表。
    返回：动作集合对象。
    副作用：无。
    异常：无。
    设计说明：由环境生成，由 ActionGuard 消费，防止 Agent 自行解释规则。
    """

    actions: list[LegalAction]

    def types(self) -> set[str]:
        """
        功能：返回当前允许的动作类型集合。
        参数：无。
        返回：字符串集合。
        副作用：无。
        异常：无。
        设计说明：为动作校验和 mock 策略提供简洁查询。
        """

        return {action.action_type for action in self.actions}

    def rule_for(self, action_type: str) -> LegalAction | None:
        """
        功能：查询某类动作的规则。
        参数：
            action_type：动作类型名称。
        返回：动作规则或 None。
        副作用：无。
        异常：无。
        设计说明：非法动作返回 None，由调用方决定降级或报错。
        """

        for action in self.actions:
            if action.action_type == action_type:
                return action
        return None


@dataclass(slots=True)
class ActionDecision(Serializable):
    """
    功能：表示 Agent 或 LLM 的结构化动作决策。
    参数：
        action_type：动作类型。
        amount：raise 的加注到总额；其他动作通常为 None。
        confidence：提供者声明的置信度或启发式置信度。
        reason_summary：可公开保存的简短理由，不包含长思维链。
        raw_response：可选原始结构化片段，默认只保存安全摘要。
    返回：动作决策对象。
    副作用：无。
    异常：无。
    设计说明：所有动作必须经过该结构进入 ActionGuard。
    """

    action_type: str
    amount: int | None = None
    confidence: float = 1.0
    reason_summary: str = ""
    raw_response: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class PlayerPublicState(Serializable):
    """
    功能：描述某玩家对当前观察者可公开的桌面状态。
    参数：
        agent_id：玩家标识。
        seat：座位号。
        stack：剩余筹码。
        current_bet：本街已投入筹码。
        total_committed：本手累计投入筹码。
        folded：是否已弃牌。
        all_in：是否已全下。
        busted：是否已出局。
    返回：公开玩家状态。
    副作用：无。
    异常：无。
    设计说明：不包含对手私有手牌，避免提示词和记忆泄露。
    """

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
    """
    功能：表示某 Agent 在决策点合法可见的全部信息。
    参数：
        agent_id：观察者。
        table_id：桌号。
        hand_id：手牌编号。
        phase：preflop/flop/turn/river 等阶段。
        seat：观察者座位。
        hole_cards：仅观察者自己的私有手牌。
        community_cards：当前已揭示公共牌。
        pot：当前底池。
        current_bet：本街最高下注线。
        to_call：观察者需要补齐的筹码。
        players：公开玩家状态。
        action_history：已公开行动历史。
        legal_actions：当前合法动作集合。
        seed：本手派生 seed，便于追踪。
    返回：观察对象。
    副作用：无。
    异常：无。
    设计说明：该对象是信息边界，记忆和提示词只能从这里取决策信息。
    """

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


@dataclass(slots=True)
class DecisionEvent(Serializable):
    """
    功能：记录一次决策及其提交到环境前后的必要上下文。
    参数：
        agent_id：决策者。
        table_id：桌号。
        hand_id：手牌编号。
        observation：合法可见观察。
        decision：经结构化解析后的动作。
        committed_action：经 ActionGuard 修正后的动作。
        memory_context：本次注入提示词的记忆摘要。
        llm_metadata：Provider、模型、重试和成本代理信息。
        created_at：UTC 时间戳。
    返回：决策事件。
    副作用：无。
    异常：无。
    设计说明：事件日志保留足够审计信息，但默认不保存长原始回复。
    """

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
    """
    功能：描述一名 Agent 对一手牌可见的轨迹和结算回报。
    参数：
        agent_id：轨迹所属 Agent。
        table_id：桌号。
        hand_id：手牌编号。
        decision_events：该 Agent 在本手中的决策事件。
        public_actions：本手公开行动历史。
        final_reward：该 Agent 本手净筹码变化。
        final_stack：本手结束后的筹码。
        showdown_visible_cards：摊牌后按规则可见的牌，不含未公开私牌。
        summary：结构化短摘要，用于检索和经验更新。
    返回：轨迹对象。
    副作用：无。
    异常：无。
    设计说明：记忆写入仅消费该对象，避免触达环境上帝视角。
    """

    agent_id: AgentId
    table_id: TableId
    hand_id: HandId
    decision_events: list[DecisionEvent]
    public_actions: list[JsonDict]
    final_reward: int
    final_stack: int
    showdown_visible_cards: dict[AgentId, list[str]]
    summary: str


@dataclass(slots=True)
class FactualMemoryRecord(Serializable):
    """
    功能：表示一条可检索的结构化事实记忆。
    参数：
        record_id：事实 ID。
        agent_id：所属 Agent。
        table_id：来源桌号。
        hand_id：来源手牌。
        scope：记忆作用域。
        state_summary：状态摘要。
        action_summary：行动摘要。
        final_reward：终局回报。
        features：可解释检索特征。
        source：来源元数据。
        created_at：写入时间。
    返回：事实记忆记录。
    副作用：无。
    异常：无。
    设计说明：事实记录来自已结束手牌，但只含该 Agent 当时可见与结算后可知信息。
    """

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
    """
    功能：表示持续更新的经验文档版本。
    参数：
        version：经验版本号。
        body：经验正文。
        source_hand_ids：本次更新使用的手牌窗口。
        updated_at：更新时间。
        scope：记忆作用域。
        metadata：更新元数据。
    返回：经验文档。
    副作用：无。
    异常：无。
    设计说明：保留版本历史，支持训练后 snapshot 和泛化阶段 restore。
    """

    version: int
    body: str
    source_hand_ids: list[HandId]
    updated_at: str
    scope: MemoryScope
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class MemoryContext(Serializable):
    """
    功能：封装本次决策要注入提示词的记忆内容。
    参数：
        facts：事实证据列表。
        experience：经验摘要。
        persona：人格相关提示词。
        metadata：检索命中、筛选和作用域信息。
    返回：记忆上下文。
    副作用：无。
    异常：无。
    设计说明：Agent 不直接读取记忆内部结构，只消费该上下文。
    """

    facts: list[FactualMemoryRecord] = field(default_factory=list)
    experience: ExperienceDocument | None = None
    persona: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class MemorySnapshot(Serializable):
    """
    功能：表示可持久化和恢复的记忆快照。
    参数：
        mechanism：机制名称。
        agent_id：所属 Agent。
        scope：作用域。
        payload：机制自定义但 JSON 可序列化的数据。
        created_at：创建时间。
    返回：记忆快照。
    副作用：无。
    异常：无。
    设计说明：快照接口让训练和泛化场景解耦。
    """

    mechanism: str
    agent_id: AgentId
    scope: MemoryScope
    payload: JsonDict
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class TableSpec(Serializable):
    """
    功能：描述一张牌桌的静态配置。
    参数：
        table_id：桌号。
        agent_ids：入座 Agent 列表。
        starting_stacks：每名 Agent 的初始或当前筹码。
        small_blind：小盲。
        big_blind：大盲。
        max_raises_per_street：每街最大有效加注次数。
    返回：牌桌配置对象。
    副作用：无。
    异常：无。
    设计说明：实验层通过该对象实例化环境，避免绑定具体环境类。
    """

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
    """
    功能：表示环境执行一步动作后的结果。
    参数：
        event：公开环境事件。
        hand_finished：本手是否结束。
    返回：步进结果对象。
    副作用：无。
    异常：无。
    设计说明：环境只返回公开事件，完整轨迹由场景层汇总。
    """

    event: JsonDict
    hand_finished: bool


@dataclass(slots=True)
class HandResult(Serializable):
    """
    功能：表示一手牌的结算结果。
    参数：
        table_id：桌号。
        hand_id：手牌编号。
        rewards：每个 Agent 的净筹码变化。
        final_stacks：结束筹码。
        public_actions：公开行动历史。
        winners：赢家摘要。
        showdown_visible_cards：按可见性规则公开的摊牌牌面。
        showdown_ranks：摊牌玩家的牌型名称，未摊牌手为空。
    返回：手牌结果。
    副作用：无。
    异常：无。
    设计说明：离线评估可使用该对象计算指标，Agent 决策不能提前读取。
    """

    table_id: TableId
    hand_id: HandId
    rewards: dict[AgentId, int]
    final_stacks: dict[AgentId, int]
    public_actions: list[JsonDict]
    winners: list[JsonDict]
    showdown_visible_cards: dict[AgentId, list[str]]
    showdown_ranks: dict[AgentId, str] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifest(Serializable):
    """
    功能：描述一次实验运行的不可变清单。
    参数：
        run_id：运行 ID。
        scenario：场景名称。
        seed：根 seed。
        config_snapshot_path：解析后配置路径。
        output_dir：输出目录。
        code_version：代码版本或 unknown。
        provider：Provider 名称。
        model：模型名称。
        created_at：创建时间。
    返回：运行清单。
    副作用：无。
    异常：无。
    设计说明：每次运行都写 manifest，支持复现实验审计。
    """

    run_id: RunId
    scenario: str
    seed: int
    config_snapshot_path: str
    output_dir: str
    code_version: str
    provider: str
    model: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ExperimentResult(Serializable):
    """
    功能：表示场景运行完成后的结果摘要。
    参数：
        run_id：运行 ID。
        scenario：场景名称。
        metrics：主要指标。
        aggregate_metrics：聚合指标。
        artifacts：关键工件路径。
        notes：限制、假设和待验证项。
    返回：实验结果对象。
    副作用：无。
    异常：无。
    设计说明：CLI 只依赖该对象输出最终摘要。
    """

    run_id: RunId
    scenario: str
    metrics: JsonDict
    aggregate_metrics: JsonDict
    artifacts: JsonDict
    notes: list[str] = field(default_factory=list)
