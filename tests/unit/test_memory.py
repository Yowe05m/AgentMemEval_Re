"""
模块说明：本模块测试记忆机制读写和快照。
核心职责：覆盖事实检索、经验版本、异步 sweep 和人格注入。
输入与输出：输入构造的观察和轨迹，输出 pytest 断言结果。
依赖边界：只依赖核心领域对象和 memory 模块。
不负责：不运行真实环境或 Provider。
"""

from agentmemeval.core.domain import (
    ActionDecision,
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    LegalAction,
    LegalActionSet,
    MemoryContext,
    MemorySnapshot,
    PlayerPublicState,
)
from agentmemeval.memory.experiential import ExperientialMemory
from agentmemeval.memory.fact_expr_async import FactExprAsyncMemory
from agentmemeval.memory.factual import FactualMemory
from agentmemeval.memory.personality_driven import DEFAULT_PERSONAS, PersonalityDrivenMemory


def make_observation(agent_id: str = "agent_00") -> AgentObservation:
    """
    功能：构造测试观察。
    参数：
        agent_id：观察者 ID。
    返回：AgentObservation。
    副作用：无。
    异常：无。
    设计说明：测试只关心记忆输入，不需要完整环境。
    """

    return AgentObservation(
        agent_id=agent_id,
        table_id="table_a",
        hand_id="hand_1",
        phase="preflop",
        seat=0,
        hole_cards=["As", "Ah"],
        community_cards=[],
        pot=3,
        current_bet=2,
        to_call=1,
        players=[
            PlayerPublicState(agent_id, 0, 100, 1, 1, False, False),
            PlayerPublicState("agent_01", 1, 100, 2, 2, False, False),
        ],
        action_history=[],
        legal_actions=LegalActionSet([LegalAction("fold"), LegalAction("call")]),
        seed=7,
    )


def make_trajectory(
    agent_id: str = "agent_00",
    reward: int = 5,
    hand_id: str = "hand_1",
    fallback_used: bool = False,
) -> HandTrajectory:
    """
    功能：构造测试轨迹。
    参数：
        agent_id：Agent ID。
        reward：本手奖励。
        hand_id：手牌 ID。
    返回：HandTrajectory。
    副作用：无。
    异常：无。
    设计说明：轨迹只包含合法观察和动作，不含对手私牌。
    """

    observation = make_observation(agent_id)
    observation.hand_id = hand_id
    event = DecisionEvent(
        agent_id=agent_id,
        table_id="table_a",
        hand_id=hand_id,
        observation=observation,
        decision=ActionDecision("call"),
        committed_action=ActionDecision("fold" if fallback_used else "call"),
        memory_context=MemoryContext(),
        llm_metadata={
            "guard_repaired": fallback_used,
            "fallback_used": fallback_used,
        },
    )
    return HandTrajectory(
        agent_id=agent_id,
        table_id="table_a",
        hand_id=hand_id,
        decision_events=[event],
        public_actions=[],
        final_reward=reward,
        final_stack=105,
        showdown_visible_cards={},
        summary="测试轨迹",
    )


def test_factual_memory_snapshot_restore_and_retrieve() -> None:
    """
    功能：验证事实记忆写入、检索和快照恢复。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：FactAgent 的核心语义是已结束手牌入库、决策时检索。
    """

    memory = FactualMemory("agent_00", top_k=3)
    memory.on_hand_finished(make_trajectory())
    context = memory.build_context(make_observation())
    assert len(context.facts) == 1
    snapshot = memory.snapshot()
    restored = FactualMemory("agent_00")
    restored.restore(snapshot)
    assert restored.metrics()["fact_count"] == 1


def test_experiential_memory_versions() -> None:
    """
    功能：验证经验记忆会产生新版本。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：经验文档必须保留版本历史和来源手牌。
    """

    memory = ExperientialMemory("agent_00", window_size=2)
    memory.on_hand_finished(make_trajectory(reward=-3))
    assert memory.current.version == 2
    assert memory.current.source_hand_ids == ["hand_1"]
    assert "## 起手牌" in memory.current.body
    assert "## 对手类型应对" in memory.current.body
    assert memory.metrics()["revision_count"] == 1


def test_fallback_trajectory_is_audited_but_not_learned() -> None:
    """Fallback actions must not become evidence for later model decisions."""

    trajectory = make_trajectory(fallback_used=True)
    factual = FactualMemory("agent_00")
    factual.on_hand_finished(trajectory)
    assert factual.metrics()["fact_count"] == 1
    assert factual.metrics()["eligible_fact_count"] == 0
    assert factual.build_context(make_observation()).facts == []

    experiential = ExperientialMemory("agent_00")
    experiential.on_hand_finished(trajectory)
    assert experiential.current.version == 1
    assert experiential.metrics()["skipped_fallback_trajectories"] == 1


def test_legacy_fact_snapshot_is_not_silently_reused_for_learning() -> None:
    """Old snapshots lack fallback provenance and must be treated as unverified."""

    source = FactualMemory("agent_00")
    source.on_hand_finished(make_trajectory())
    payload = source.snapshot().payload
    payload.pop("schema_version")
    payload["records"][0]["source"].pop("memory_eligible")
    restored = FactualMemory("agent_00")
    restored.restore(MemorySnapshot("fact", "agent_00", "per_agent", payload))
    assert restored.metrics()["eligible_fact_count"] == 0
    assert restored.records[0].source["legacy_unverified"] is True


def test_async_memory_sweep_records_evidence() -> None:
    """
    功能：验证异步机制按周期触发 sweep。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：FactExprAsync 必须记录触发时刻、窗口和证据 ID。
    """

    memory = FactExprAsyncMemory("agent_00", sweep_every=2, evidence_k=2)
    memory.on_hand_finished(make_trajectory(hand_id="hand_1"))
    memory.on_hand_finished(make_trajectory(hand_id="hand_2"))
    assert memory.metrics()["async_sweeps"] == 1
    assert memory.snapshot().payload["sweep_log"][0]["recent_window_hand_ids"] == [
        "hand_1",
        "hand_2",
    ]
    assert memory.snapshot().payload["fact_state"]
    assert memory.snapshot().payload["sweep_log"][0]["evidence_groups"]


def test_personality_context_injection() -> None:
    """
    功能：验证人格 wrapper 注入 persona 字段。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：人格机制需要在离线 mock 下可测。
    """

    wrapped = FactualMemory("agent_00")
    wrapped.on_hand_finished(make_trajectory())
    memory = PersonalityDrivenMemory(wrapped, "INTJ")
    context = memory.build_context(make_observation())
    assert context.persona["name"] == "INTJ"
    assert context.metadata["persona_enabled"] is True
    assert context.metadata["persona_filter_mode"] == "preserve_retrieval_order"
    assert context.metadata["raw_fact_count"] == context.metadata["filtered_fact_count"]
    for persona in DEFAULT_PERSONAS.values():
        assert "不得机械映射成固定动作" in persona["decision_prompt"]
