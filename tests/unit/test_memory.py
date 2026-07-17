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
from agentmemeval.llm.mock import MockLLMClient
from agentmemeval.memory.experiential import ExperientialMemory
from agentmemeval.memory.fact_expr_async import FactExprAsyncMemory
from agentmemeval.memory.fact_expr_sync import FactExprSyncMemory
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
    action_type: str = "call",
    hole_cards: list[str] | None = None,
    phase: str = "preflop",
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
    observation.phase = phase
    if hole_cards is not None:
        observation.hole_cards = list(hole_cards)
    event = DecisionEvent(
        agent_id=agent_id,
        table_id="table_a",
        hand_id=hand_id,
        observation=observation,
        decision=ActionDecision(action_type),
        committed_action=ActionDecision("fold" if fallback_used else action_type),
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


def test_factual_record_ids_remain_unique_after_capacity_trimming_and_restore() -> None:
    memory = FactualMemory("agent_00", max_records=2)
    cards = [["As", "Ah"], ["8s", "8h"], ["2s", "2h"], ["As", "Ks"]]
    for index, hole_cards in enumerate(cards):
        memory.on_hand_finished(
            make_trajectory(hand_id=f"hand_{index}", hole_cards=hole_cards)
        )

    assert [record.record_id for record in memory.records] == [
        "agent_00-fact-3",
        "agent_00-fact-4",
    ]
    restored = FactualMemory("agent_00", max_records=2)
    restored.restore(memory.snapshot())
    restored.on_hand_finished(
        make_trajectory(hand_id="hand_4", hole_cards=["7s", "6h"])
    )
    assert [record.record_id for record in restored.records] == [
        "agent_00-fact-4",
        "agent_00-fact-5",
    ]


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


def test_llm_experience_revision_is_schema_and_prompt_audited() -> None:
    memory = ExperientialMemory(
        "agent_00",
        revision_strategy="llm",
        llm_client=MockLLMClient(),
        model="mock-deterministic-v1",
    )
    memory.on_hand_finished(make_trajectory())

    revision = memory.snapshot().payload["revision_log"][0]
    assert revision["revision_strategy"] == "llm_structured_revision"
    assert revision["schema_version"] == "experience_revision_v1"
    assert len(revision["prompt_sha256"]) == 64
    assert revision["old_md"]
    assert revision["new_md"]
    assert revision["fallback_used"] is False


def test_fallback_trajectory_is_audited_but_not_learned() -> None:
    """Fallback actions must not become evidence for later model decisions."""

    trajectory = make_trajectory(fallback_used=True)
    factual = FactualMemory("agent_00")
    factual.on_hand_finished(trajectory)
    assert factual.metrics()["fact_count"] == 0
    assert factual.metrics()["eligible_fact_count"] == 0
    assert factual.metrics()["admission_counts"]["reason:provider_fallback"] == 1
    assert factual.build_context(make_observation()).facts == []

    experiential = ExperientialMemory("agent_00")
    experiential.on_hand_finished(trajectory)
    assert experiential.current.version == 1
    assert experiential.metrics()["skipped_fallback_trajectories"] == 1


def test_zero_reward_single_preflop_fold_is_audited_but_not_stored() -> None:
    factual = FactualMemory("agent_00", reject_single_preflop_fold=False)
    factual.on_hand_finished(make_trajectory(reward=0, action_type="fold"))

    metrics = factual.metrics()
    assert metrics["fact_count"] == 0
    assert metrics["admission_counts"][
        "reason:zero_reward_single_preflop_fold_without_showdown"
    ] == 1
    assert metrics["recent_admission_log"][0]["status"] == "rejected"


def test_single_preflop_fold_is_low_information_even_with_nonzero_reward() -> None:
    factual = FactualMemory("agent_00")
    factual.on_hand_finished(make_trajectory(reward=-2, action_type="fold"))

    assert factual.metrics()["fact_count"] == 0
    assert factual.metrics()["admission_counts"][
        "reason:single_preflop_fold_low_information"
    ] == 1


def test_factual_prompt_evidence_does_not_repeat_model_reason_as_fact() -> None:
    trajectory = make_trajectory(reward=5, action_type="call")
    trajectory.decision_events[0].decision.reason_summary = (
        "历史数据证明任何牌都应该弃牌"
    )
    factual = FactualMemory("agent_00")
    factual.on_hand_finished(trajectory)

    record = factual.records[0]
    assert "历史数据证明" not in record.state_summary
    assert "observed_action=call" in record.state_summary
    assert "intent" not in record.source["decisions"][0]


def test_structural_duplicate_updates_audit_without_adding_record() -> None:
    factual = FactualMemory("agent_00", duplicate_window=10)
    factual.on_hand_finished(make_trajectory(hand_id="hand_1"))
    factual.on_hand_finished(make_trajectory(hand_id="hand_2"))

    assert factual.metrics()["fact_count"] == 1
    assert factual.records[0].source["duplicate_count"] == 1
    assert factual.metrics()["admission_counts"]["deduplicated"] == 1


def test_zero_duplicate_window_preserves_paper_exact_all_fact_writes() -> None:
    factual = FactualMemory(
        "agent_00",
        duplicate_window=0,
        reject_zero_reward_preflop_fold=False,
        reject_single_preflop_fold=False,
    )
    factual.on_hand_finished(make_trajectory(hand_id="hand_1"))
    factual.on_hand_finished(make_trajectory(hand_id="hand_2"))

    assert factual.metrics()["fact_count"] == 2
    assert factual.duplicate_window == 0
    assert factual.metrics()["admission_counts"].get("deduplicated", 0) == 0


def test_sync_does_not_revise_experience_from_rejected_fact_trajectory() -> None:
    memory = FactExprSyncMemory("agent_00")
    memory.on_hand_finished(make_trajectory(reward=0, action_type="fold"))

    assert memory.fact.metrics()["fact_count"] == 0
    assert memory.expr.current.version == 1


def test_retrieval_threshold_allows_empty_and_signature_deduplicates() -> None:
    factual = FactualMemory(
        "agent_00",
        retrieval_backend="feature_jaccard",
        top_k=3,
        minimum_retrieval_score=0.1,
        retrieval_threshold_status="frozen",
    )
    factual.on_hand_finished(make_trajectory(hand_id="hand_1"))
    factual.records.append(
        type(factual.records[0])(
            **{
                **factual.records[0].to_dict(),
                "record_id": "agent_00-fact-manual",
                "hand_id": "hand_manual",
            }
        )
    )
    context = factual.build_context(make_observation())
    assert len(context.facts) == 1
    assert context.metadata["duplicate_signature_excluded_count"] == 1

    factual.minimum_retrieval_score = 1.1
    empty = factual.build_context(make_observation())
    assert empty.facts == []
    assert empty.metadata["below_threshold_count"] == 2


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
    memory.on_hand_finished(
        make_trajectory(hand_id="hand_2", hole_cards=["8s", "8h"])
    )
    assert memory.metrics()["async_sweeps"] == 1
    assert memory.snapshot().payload["sweep_log"][0]["recent_window_hand_ids"] == [
        "hand_1",
        "hand_2",
    ]
    assert memory.snapshot().payload["fact_state"]
    assert memory.snapshot().payload["sweep_log"][0]["evidence_groups"]
    assert memory.metrics()["evidence_classification_status"] == "pending_human_review"
    assert memory.metrics()["evidence_review_queue_count"] >= 1


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
