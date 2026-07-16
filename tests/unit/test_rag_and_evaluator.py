"""
模块说明：本模块测试迁移后的 RAG 与扑克评估能力。
核心职责：覆盖离线 hash RAG、牌型名称和 side-pot 分层。
输入与输出：输入构造对象，输出 pytest 断言。
依赖边界：不调用真实 LLM、treys 或网络模型。
不负责：不运行完整实验。
"""

from pathlib import Path
from uuid import uuid4

from agentmemeval.core.domain import (
    FactualMemoryRecord,
    LegalAction,
    LegalActionSet,
)
from agentmemeval.environment.hand_evaluator import evaluate_best, split_side_pots
from agentmemeval.memory.rag import (
    HashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
    hybrid_top_k_records,
)
from tests.unit.test_memory import make_observation


def test_hybrid_rag_prefers_matching_poker_state() -> None:
    """
    功能：验证混合 RAG 优先召回局面相似事实。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：模拟原版 phase/hole/board/pot/to_call query 的离线排序。
    """

    observation = make_observation()
    observation.legal_actions = LegalActionSet([LegalAction("fold"), LegalAction("call")])
    matching = FactualMemoryRecord(
        record_id="fact_matching",
        agent_id="agent_00",
        table_id="table_a",
        hand_id="h1",
        scope="per_agent",
        state_summary="phase=preflop hole=['As', 'Ah'] board=[] pot=3 to_call=1",
        action_summary="preflop:call",
        final_reward=5,
        features=["phase:preflop", "hole_rank:high", "hole_pair", "to_call:small"],
        source={"fact_text": "AA preflop small call won"},
    )
    distant = FactualMemoryRecord(
        record_id="fact_distant",
        agent_id="agent_00",
        table_id="table_a",
        hand_id="h2",
        scope="per_agent",
        state_summary="phase=river board=2c 7d 9h Ts Qc pot=50 to_call=20",
        action_summary="river:fold",
        final_reward=-20,
        features=["phase:river", "board_rank:low", "to_call:large"],
        source={"fact_text": "river weak high card folded to large bet"},
    )
    scored = hybrid_top_k_records(observation, [distant, matching], k=2)
    assert scored[0].record.record_id == "fact_matching"
    assert scored[0].feature > scored[1].feature
    assert HashEmbeddingBackend().audit_metadata()["semantic_model"] is False


def test_versioned_semantic_embedding_cache_batches_missing_texts() -> None:
    class FakeEmbeddingBackend(OpenAICompatibleEmbeddingBackend):
        def _request(self, texts: list[str]) -> list[list[float]]:
            self.request_count += 1
            return [[float(len(text)), 1.0] for text in texts]

    cache_path = Path("tmp") / f"embedding-cache-{uuid4().hex}.json"
    backend = FakeEmbeddingBackend(
        model="semantic-test-model",
        revision="revision-abc",
        cache_path=cache_path,
    )
    first = backend.embed_texts(["same", "other"])
    second = backend.embed_texts(["same"])

    assert first[0] == second[0]
    assert backend.request_count == 1
    assert backend.cache_hit_count == 1
    assert backend.audit_metadata()["revision"] == "revision-abc"
    assert cache_path.exists()


def test_poker_evaluator_names_straight_flush() -> None:
    """
    功能：验证本地 evaluator 输出标准牌型名称。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：替代原版 treys class_to_string 的离线能力。
    """

    rank = evaluate_best(["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"])
    assert rank.class_name == "Straight Flush"
    assert rank.score[0] == 8


def test_side_pot_split_respects_all_in_layers() -> None:
    """
    功能：验证 side-pot 分层派奖。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：迁入原版 controller 的按贡献层分底池语义。
    """

    ranks = {
        "short": evaluate_best(["As", "Ah", "2c", "3d", "4s", "8h", "9c"]),
        "deep": evaluate_best(["Ks", "Kh", "2c", "3d", "4s", "8h", "9c"]),
        "caller": evaluate_best(["Qs", "Qh", "2c", "3d", "4s", "8h", "9c"]),
    }
    payouts = split_side_pots(
        contributions={"short": 10, "deep": 30, "caller": 30},
        folded=set(),
        ranks=ranks,
    )
    by_agent = {}
    for payout in payouts:
        by_agent[payout["agent_id"]] = by_agent.get(payout["agent_id"], 0) + payout["amount"]
    assert by_agent["short"] == 30
    assert by_agent["deep"] == 40
