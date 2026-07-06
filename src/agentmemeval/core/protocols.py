"""
模块说明：本模块定义环境、LLM、记忆、Agent 与实验场景的替换协议。
核心职责：约束核心边界，使具体扑克环境、Provider 和记忆机制可以独立替换。
输入与输出：输入为领域对象，输出为领域对象或结构化响应。
依赖边界：只依赖 typing Protocol 与核心领域模型。
不负责：不提供具体实现，不做配置解析，不记录工件。
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from agentmemeval.core.domain import (
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
    """
    功能：描述扑克环境适配器必须提供的最小行为。
    参数：由方法参数给出。
    返回：观察、合法动作、步进结果或结算结果。
    副作用：具体实现会推进内部牌局状态。
    异常：非法动作或状态错误时抛出领域异常。
    设计说明：实验层只认该协议，不绑定具体扑克引擎。
    """

    def reset(self, table_spec: TableSpec, seed: int) -> None:
        """功能：重置牌桌并开始一手牌。"""

    def current_agent_id(self) -> AgentId | None:
        """功能：返回当前行动者，没有行动者时返回 None。"""

    def current_observation(self, agent_id: AgentId) -> AgentObservation:
        """功能：返回指定 Agent 的合法可见观察。"""

    def legal_actions(self, agent_id: AgentId) -> LegalActionSet:
        """功能：返回指定 Agent 当前合法动作集合。"""

    def step(self, agent_id: AgentId, action: ActionDecision) -> StepResult:
        """功能：执行动作并推进环境。"""

    def is_hand_finished(self) -> bool:
        """功能：返回本手是否已经结束。"""

    def finalize_hand(self) -> HandResult:
        """功能：返回本手结算结果。"""


class LLMClient(Protocol):
    """
    功能：描述模型 Provider 的统一结构化生成接口。
    参数：请求对象和目标 schema。
    返回：目标 schema 实例。
    副作用：真实 Provider 可能进行网络调用，mock Provider 无网络副作用。
    异常：Provider 初始化或调用失败时抛出 ProviderError。
    设计说明：上层 Agent 不根据厂商写分支，只消费结构化结果。
    """

    def generate_structured(self, request: object, schema: type[T]) -> T:
        """功能：生成满足 schema 的结构化对象。"""

    def healthcheck(self) -> dict[str, object]:
        """功能：返回 Provider 可用性和配置摘要。"""


class MemoryMechanism(Protocol):
    """
    功能：描述记忆机制读写和快照恢复接口。
    参数：观察、决策事件、轨迹或快照。
    返回：记忆上下文或快照。
    副作用：写入内部记忆状态。
    异常：快照格式错误时可抛出领域异常。
    设计说明：Fact、Expr、Sync、Async 与人格机制共享同一生命周期。
    """

    name: str

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """功能：为当前决策构造记忆上下文。"""

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """功能：接收已提交动作事件。"""

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """功能：接收手牌结束后的可见轨迹并更新记忆。"""

    def snapshot(self) -> MemorySnapshot:
        """功能：导出可恢复的记忆快照。"""

    def restore(self, snapshot: MemorySnapshot) -> None:
        """功能：从快照恢复记忆状态。"""

    def metrics(self) -> dict[str, object]:
        """功能：返回记忆规模、更新次数等指标。"""


class AgentPolicy(Protocol):
    """
    功能：描述 Agent 决策接口。
    参数：合法观察。
    返回：结构化动作和记忆上下文。
    副作用：可能调用 LLM Provider 并记录短期工作缓存。
    异常：Provider 或动作结构错误时抛出领域异常。
    设计说明：实验层不区分 NoMemory、Fact 或人格 Agent 的内部实现。
    """

    agent_id: AgentId

    def decide(
        self,
        observation: AgentObservation,
    ) -> tuple[ActionDecision, MemoryContext, dict[str, object]]:
        """功能：基于观察输出动作、记忆上下文和模型元数据。"""

    def observe_decision(self, event: DecisionEvent) -> None:
        """功能：把已提交动作反馈给 Agent。"""

    def observe_hand_finished(self, trajectory: HandTrajectory) -> None:
        """功能：把手牌轨迹反馈给 Agent。"""

    def snapshot_memory(self) -> MemorySnapshot:
        """功能：导出 Agent 记忆快照。"""

    def restore_memory(self, snapshot: MemorySnapshot) -> None:
        """功能：恢复 Agent 记忆快照。"""


class ExperimentScenario(Protocol):
    """
    功能：描述可运行实验场景。
    参数：实验上下文对象。
    返回：实验结果。
    副作用：写入输出工件。
    异常：配置或运行错误会向上抛出。
    设计说明：固定桌、泛化和换桌都通过同一 runner 调用。
    """

    def run(self, context: object) -> ExperimentResult:
        """功能：执行场景并返回结果摘要。"""
