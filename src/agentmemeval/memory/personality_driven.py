"""
模块说明：本模块实现人格驱动的记忆筛选与提示注入 wrapper。
核心职责：把 persona 配置注入记忆筛选、经验更新说明和动作决策上下文。
输入与输出：输入观察、事件、轨迹，输出带 persona 字段的 MemoryContext。
依赖边界：包装任意 MemoryMechanism，不依赖具体 LLM Provider。
不负责：不把 MBTI 硬编码为唯一选择，不直接决定动作。
"""

from __future__ import annotations

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    FactualMemoryRecord,
    HandTrajectory,
    MemoryContext,
    MemorySnapshot,
)
from agentmemeval.core.protocols import MemoryMechanism

DEFAULT_PERSONAS: dict[str, dict[str, str]] = {
    "INTJ": {
        "description": "重逻辑、重长期规划，偏好经过验证的规律。",
        "decision_prompt": "决策时优先要求证据充分，避免无根据冒险。",
        "filter_prompt": "筛选记忆时优先保留结构清晰、样本支持较强的事实。",
        "update_prompt": "更新经验时强调校准和反例。",
    },
    "ENFP": {
        "description": "好奇、探索、愿意尝试新线。",
        "decision_prompt": "决策时允许更高探索性，但仍服从合法动作。",
        "filter_prompt": "筛选记忆时保留新颖和情境差异大的片段。",
        "update_prompt": "更新经验时记录可能性和灵感，但避免绑定具体玩家。",
    },
    "ISTP": {
        "description": "务实、机会主义、关注即时价格。",
        "decision_prompt": "决策时优先比较成本和直接收益。",
        "filter_prompt": "筛选记忆时保留可操作、和当前局面直接相关的事实。",
        "update_prompt": "更新经验时删除空泛判断。",
    },
    "ESFJ": {
        "description": "稳妥、重视已验证模式和低波动。",
        "decision_prompt": "决策时偏向降低波动和避免边缘投入。",
        "filter_prompt": "筛选记忆时优先保留稳定、重复出现的经验。",
        "update_prompt": "更新经验时强调稳健性。",
    },
}


class PersonalityDrivenMemory:
    """
    功能：为基础记忆机制加入 persona 筛选和提示字段。
    参数：
        wrapped：被包装的记忆机制。
        persona_name：人格名称。
        persona_config：可选人格配置。
    返回：人格驱动记忆机制。
    副作用：构造上下文时可能筛选事实列表。
    异常：无。
    设计说明：人格是可配置 wrapper，不能把 MBTI 集合写死为唯一可选项。
    """

    name = "personality_driven"

    def __init__(
        self,
        wrapped: MemoryMechanism,
        persona_name: str,
        persona_config: dict[str, str] | None = None,
    ) -> None:
        """
        功能：初始化人格 wrapper。
        参数：
            wrapped：底层记忆机制。
            persona_name：人格名称。
            persona_config：人格文本配置。
        返回：无。
        副作用：保存配置。
        异常：无。
        设计说明：配置优先于内置示例，支持任意 persona 注册。
        """

        self.wrapped = wrapped
        self.persona_name = persona_name
        self.persona_config = persona_config or DEFAULT_PERSONAS.get(
            persona_name,
            {
                "description": persona_name,
                "decision_prompt": "按该 persona 的偏好决策。",
                "filter_prompt": "按该 persona 的偏好筛选记忆。",
                "update_prompt": "按该 persona 的偏好更新记忆。",
            },
        )

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：构造带人格信息的记忆上下文。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：根据 persona 对事实列表做确定性筛选。
        异常：无。
        设计说明：mock 测试不依赖真实 LLM，也能验证人格注入阶段是否正确。
        """

        context = self.wrapped.build_context(observation)
        filtered = self._filter_facts(context.facts)
        context.facts = filtered
        context.persona = {
            "name": self.persona_name,
            **self.persona_config,
        }
        context.metadata = {
            **context.metadata,
            "persona_enabled": True,
            "persona_name": self.persona_name,
            "raw_fact_count": len(context.facts),
            "filtered_fact_count": len(filtered),
        }
        return context

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：转发决策事件。
        参数：
            event：决策事件。
        返回：无。
        副作用：底层记忆可能更新短期状态。
        异常：无。
        设计说明：人格 wrapper 不改变生命周期顺序。
        """

        self.wrapped.on_decision_committed(event)

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：转发手牌结束轨迹。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：底层记忆按自身规则更新。
        异常：无。
        设计说明：人格更新提示会通过上下文记录，当前确定性实现不额外改写轨迹。
        """

        self.wrapped.on_hand_finished(trajectory)

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出人格 wrapper 快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：保存底层快照和 persona 配置，泛化阶段可完整恢复。
        """

        inner = self.wrapped.snapshot()
        return MemorySnapshot(
            mechanism=self.name,
            agent_id=inner.agent_id,
            scope=inner.scope,
            payload={
                "persona_name": self.persona_name,
                "persona_config": dict(self.persona_config),
                "wrapped_mechanism": inner.mechanism,
                "wrapped_payload": inner.payload,
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：恢复人格配置和底层记忆。
        参数：
            snapshot：人格快照。
        返回：无。
        副作用：更新 persona 字段并转发底层 restore。
        异常：无。
        设计说明：外层不需要知道底层具体类型，只保留其已有对象。
        """

        self.persona_name = str(snapshot.payload.get("persona_name", self.persona_name))
        self.persona_config = dict(snapshot.payload.get("persona_config", self.persona_config))
        inner = self.wrapped.snapshot()
        inner.payload = dict(snapshot.payload.get("wrapped_payload", inner.payload))
        self.wrapped.restore(inner)

    def metrics(self) -> dict[str, object]:
        """
        功能：返回人格记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：保留底层指标并标记 persona 名称。
        """

        metrics = self.wrapped.metrics()
        metrics["persona_enabled"] = True
        metrics["persona_name"] = self.persona_name
        return metrics

    def _filter_facts(self, facts: list[FactualMemoryRecord]) -> list[FactualMemoryRecord]:
        """
        功能：按 persona 对事实做确定性筛选。
        参数：
            facts：原始事实列表。
        返回：筛选后的事实列表。
        副作用：无。
        异常：无。
        设计说明：这是离线可测的替代逻辑；真实人格 LLM 筛选可替换该函数。
        """

        persona = self.persona_name.upper()
        if persona == "INTJ":
            return sorted(
                facts,
                key=lambda item: (item.final_reward, item.created_at),
                reverse=True,
            )
        if persona == "ENFP":
            return list(reversed(facts))
        if persona == "ESFJ":
            return [fact for fact in facts if fact.final_reward >= 0] or facts[:1]
        return facts
