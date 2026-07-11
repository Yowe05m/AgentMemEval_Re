"""
模块说明：本模块实现通用 LLM Agent 决策管线。
核心职责：连接观察、记忆上下文、提示词、Provider 和 ActionGuard。
输入与输出：输入 AgentObservation，输出合法 ActionDecision 与元数据。
依赖边界：依赖 LLMClient 和 MemoryMechanism 协议，不绑定具体机制或厂商。
不负责：不推进环境，不直接写工件。
"""

from __future__ import annotations

import hashlib
import time

from agentmemeval.core.domain import (
    ActionDecision,
    AgentId,
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    MemoryContext,
    MemorySnapshot,
)
from agentmemeval.core.protocols import LLMClient, MemoryMechanism
from agentmemeval.environment.action_guard import ActionGuard
from agentmemeval.environment.raise_sizing import build_raise_sizing_plan
from agentmemeval.llm.schemas import LLMCallStats, LLMRequest
from agentmemeval.prompts.decision import (
    PROMPT_TEMPLATE_VERSION,
    render_system_prompt,
    render_user_prompt,
)


class LLMDecisionAgent:
    """
    功能：使用 LLMClient 和 MemoryMechanism 进行结构化决策。
    参数：
        agent_id：Agent 标识。
        memory：记忆机制。
        llm_client：Provider 实例。
        model：模型名称。
        guard：动作保护器。
    返回：Agent 实例。
    副作用：调用 decide 时可能访问 Provider，观察反馈时更新记忆。
    异常：Provider 或动作校验失败时向上抛出领域异常。
    设计说明：所有机制共享同一决策流程，确保消融比较公平。
    """

    def __init__(
        self,
        agent_id: AgentId,
        memory: MemoryMechanism,
        llm_client: LLMClient,
        model: str = "mock-deterministic-v1",
        guard: ActionGuard | None = None,
        raise_sizing_policy: str = "native_no_limit",
    ) -> None:
        """
        功能：初始化通用 Agent。
        参数：
            agent_id：Agent 标识。
            memory：记忆机制。
            llm_client：LLM Provider。
            model：模型名称。
            guard：动作保护器。
        返回：无。
        副作用：保存依赖。
        异常：无。
        设计说明：依赖从外部注入，便于测试中替换 mock。
        """

        self.agent_id = agent_id
        self.memory = memory
        self.llm_client = llm_client
        self.model = model
        self.guard = guard or ActionGuard()
        self.raise_sizing_policy = raise_sizing_policy

    def decide(
        self,
        observation: AgentObservation,
    ) -> tuple[ActionDecision, MemoryContext, dict[str, object]]:
        """
        功能：为当前观察生成合法动作。
        参数：
            observation：合法可见观察。
        返回：合法动作、记忆上下文和元数据。
        副作用：调用 Provider。
        异常：Provider 失败或无合法动作时抛出领域异常。
        设计说明：原始动作和修正原因写入元数据，环境只接收合法动作。
        """

        context = self.memory.build_context(observation)
        raise_sizing = build_raise_sizing_plan(observation, self.raise_sizing_policy)
        system_prompt = render_system_prompt(context)
        user_prompt = render_user_prompt(observation, context, raise_sizing=raise_sizing)
        request = LLMRequest(
            observation=observation,
            memory_context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_config={"model": self.model},
            metadata={
                "seed": observation.seed,
                "raise_sizing": raise_sizing.to_dict(),
            },
        )
        started = time.perf_counter()
        raw_decision = self.llm_client.generate_structured(request, ActionDecision)
        elapsed_ms = (time.perf_counter() - started) * 1000
        guard_result = self.guard.guard(
            raw_decision,
            observation.legal_actions,
            strict=False,
            allowed_raise_amounts=raise_sizing.allowed_amounts,
        )
        stats = LLMCallStats(
            provider=getattr(self.llm_client, "provider", "unknown"),
            model=getattr(self.llm_client, "model", self.model),
            elapsed_ms=elapsed_ms,
            prompt_tokens=max(1, len(system_prompt.split()) + len(user_prompt.split())),
            completion_tokens=max(1, len(raw_decision.reason_summary.split()) + 4),
        )
        metadata = {
            "raw_decision": raw_decision.to_dict(),
            "guard_repaired": guard_result.repaired,
            "guard_errors": list(guard_result.errors),
            "fallback_used": guard_result.fallback_used,
            "llm": stats.to_dict(),
            "raise_sizing": raise_sizing.to_dict(),
            "prompt": {
                "template_version": PROMPT_TEMPLATE_VERSION,
                "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "user_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                "system_chars": len(system_prompt),
                "user_chars": len(user_prompt),
            },
        }
        return guard_result.action, context, metadata

    def observe_decision(self, event: DecisionEvent) -> None:
        """
        功能：把已提交决策反馈给记忆机制。
        参数：
            event：决策事件。
        返回：无。
        副作用：记忆机制可更新短期状态。
        异常：由记忆机制向上抛出。
        设计说明：场景层创建事件后统一反馈，避免 Agent 自己访问环境结果。
        """

        self.memory.on_decision_committed(event)

    def observe_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：把手牌结束轨迹反馈给记忆机制。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：记忆机制可能写入事实或经验版本。
        异常：由记忆机制向上抛出。
        设计说明：训练/测试是否更新记忆由场景配置决定是否调用该方法。
        """

        self.memory.on_hand_finished(trajectory)

    def snapshot_memory(self) -> MemorySnapshot:
        """
        功能：导出当前 Agent 记忆快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：固定桌训练后可直接恢复到泛化测试。
        """

        return self.memory.snapshot()

    def restore_memory(self, snapshot: MemorySnapshot) -> None:
        """
        功能：恢复 Agent 记忆快照。
        参数：
            snapshot：记忆快照。
        返回：无。
        副作用：替换记忆状态。
        异常：由记忆机制向上抛出。
        设计说明：恢复只影响记忆，不改变 Provider 或 Agent 标识。
        """

        self.memory.restore(snapshot)

    def memory_metrics(self) -> dict[str, object]:
        """
        功能：返回记忆机制指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：实验报告使用该方法统一采集记忆规模。
        """

        return self.memory.metrics()
