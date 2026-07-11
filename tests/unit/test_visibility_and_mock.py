"""
模块说明：本模块测试可见性保护和 mock Provider。
核心职责：确保观察与提示词不泄露对手私牌，并验证 mock 输出确定性。
输入与输出：输入本地环境和 mock 请求，输出 pytest 断言结果。
依赖边界：依赖环境、提示词、mock Provider 和领域对象。
不负责：不测试完整实验场景。
"""

from agentmemeval.core.domain import (
    ActionDecision,
    AgentObservation,
    LegalAction,
    LegalActionSet,
    MemoryContext,
    PlayerPublicState,
    TableSpec,
)
from agentmemeval.environment.holdem_adapter import HoldemEnvironment
from agentmemeval.environment.observation import assert_observation_has_no_private_leak
from agentmemeval.environment.raise_sizing import build_raise_sizing_plan
from agentmemeval.llm.mock import MockLLMClient
from agentmemeval.llm.schemas import LLMRequest
from agentmemeval.prompts.decision import (
    BASE_SYSTEM_PROMPT,
    render_system_prompt,
    render_user_prompt,
)


def test_observation_and_prompt_do_not_leak_private_cards() -> None:
    """
    功能：验证观察和提示词不包含对手私牌。
    参数：无。
    返回：无。
    副作用：创建本地环境。
    异常：断言失败时由 pytest 报告。
    设计说明：信息边界是本任务的硬性要求。
    """

    env = HoldemEnvironment()
    agent_ids = ["agent_00", "agent_01", "agent_02", "agent_03"]
    env.reset(
        TableSpec(
            table_id="visibility",
            agent_ids=agent_ids,
            starting_stacks={agent_id: 100 for agent_id in agent_ids},
        ),
        seed=123,
    )
    observation = env.current_observation("agent_00")
    forbidden = {
        player.agent_id: player.hole_cards
        for player in env.players
        if player.agent_id != "agent_00"
    }
    text = render_user_prompt(observation, MemoryContext())
    assert_observation_has_no_private_leak(observation, forbidden, text)
    assert len(observation.hole_cards) == 2
    assert observation.community_cards == []
    assert "agent_00" not in text
    assert "agent_01" not in text
    assert "- 我:" in text
    assert "下家1" in text


def test_system_prompt_states_ev_and_raise_total_semantics() -> None:
    """本地模型需要收到稳定目标、合法动作和 raise 金额语义。"""

    assert "最大化长期期望筹码收益" in BASE_SYSTEM_PROMPT
    assert "加注到的总额" in BASE_SYSTEM_PROMPT
    assert "必须来自本次给出的合法动作" in BASE_SYSTEM_PROMPT
    assert "机械映射成固定动作" in BASE_SYSTEM_PROMPT


def test_prompt_includes_authoritative_river_hand_and_call_cost() -> None:
    """Small models must not invent draws or use uncapped to_call as the real cost."""

    observation = AgentObservation(
        agent_id="agent_00",
        table_id="risk",
        hand_id="risk-h1",
        phase="river",
        seat=0,
        hole_cards=["3s", "2s"],
        community_cards=["Td", "4c", "Jc", "2c", "9h"],
        pot=2200,
        current_bet=1960,
        to_call=1960,
        players=[
            PlayerPublicState("agent_00", 0, 920, 0, 80, False, False),
            PlayerPublicState("agent_01", 1, 0, 1960, 2000, False, True),
        ],
        action_history=[],
        legal_actions=LegalActionSet([LegalAction("fold"), LegalAction("call")]),
        seed=7,
    )

    text = render_user_prompt(observation, MemoryContext())
    assert "当前已成牌：Pair" in text
    assert "河牌已结束，没有未来补牌" in text
    assert "实际最多投入 920（显示需补齐 1960）" in text
    assert "需要约 29.5% 胜率" in text
    assert "这是全下跟注" in text


def test_prompt_lists_only_local_discrete_raise_candidates() -> None:
    env = HoldemEnvironment()
    agent_ids = ["agent_00", "agent_01", "agent_02", "agent_03"]
    env.reset(
        TableSpec(
            table_id="discrete",
            agent_ids=agent_ids,
            starting_stacks={agent_id: 1000 for agent_id in agent_ids},
        ),
        seed=123,
    )
    agent_id = env.current_agent_id() or "agent_00"
    observation = env.current_observation(agent_id)
    plan = build_raise_sizing_plan(observation, "local_discrete")

    text = render_user_prompt(observation, MemoryContext(), raise_sizing=plan)
    assert "本地离散 raise-to 候选" in text
    assert "选择 raise 时 amount 只能取其中一个" in text


def test_mock_provider_is_deterministic() -> None:
    """
    功能：验证同一请求下 mock Provider 输出稳定。
    参数：无。
    返回：无。
    副作用：创建本地环境。
    异常：断言失败时由 pytest 报告。
    设计说明：离线测试必须可复现。
    """

    env = HoldemEnvironment()
    agent_ids = ["agent_00", "agent_01"]
    env.reset(
        TableSpec(
            table_id="mock",
            agent_ids=agent_ids,
            starting_stacks={agent_id: 100 for agent_id in agent_ids},
        ),
        seed=99,
    )
    agent_id = env.current_agent_id() or "agent_00"
    observation = env.current_observation(agent_id)
    context = MemoryContext()
    request = LLMRequest(
        observation=observation,
        memory_context=context,
        system_prompt=render_system_prompt(context),
        user_prompt=render_user_prompt(observation, context),
        metadata={"seed": 99},
    )
    client = MockLLMClient()
    left = client.generate_structured(request, schema=ActionDecision)
    right = client.generate_structured(request, schema=ActionDecision)
    assert left.to_dict() == right.to_dict()
