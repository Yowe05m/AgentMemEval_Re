"""
模块说明：本模块测试可见性保护和 mock Provider。
核心职责：确保观察与提示词不泄露对手私牌，并验证 mock 输出确定性。
输入与输出：输入本地环境和 mock 请求，输出 pytest 断言结果。
依赖边界：依赖环境、提示词、mock Provider 和领域对象。
不负责：不测试完整实验场景。
"""

from agentmemeval.core.domain import ActionDecision, MemoryContext, TableSpec
from agentmemeval.environment.holdem_adapter import HoldemEnvironment
from agentmemeval.environment.observation import assert_observation_has_no_private_leak
from agentmemeval.llm.mock import MockLLMClient
from agentmemeval.llm.schemas import LLMRequest
from agentmemeval.prompts.decision import render_system_prompt, render_user_prompt


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
