"""
模块说明：本模块实现完全离线、可复现的 mock LLM Provider。
核心职责：基于合法动作、可见牌面和 persona 偏置生成结构化动作。
输入与输出：输入 LLMRequest，输出 ActionDecision 或简单结构化字典。
依赖边界：不访问网络，不读取密钥，不依赖厂商 SDK。
不负责：不模拟真实模型能力，不生成长思维链，不绕过 ActionGuard。
"""

from __future__ import annotations

from typing import TypeVar

from agentmemeval.core.domain import ActionDecision
from agentmemeval.core.errors import ProviderError
from agentmemeval.core.seeds import make_rng
from agentmemeval.llm.schemas import LLMRequest

T = TypeVar("T")
RANK_VALUE = {rank: index + 2 for index, rank in enumerate("23456789TJQKA")}


class MockLLMClient:
    """
    功能：提供确定性的离线结构化生成。
    参数：
        config：Provider 配置。
    返回：mock Provider 实例。
    副作用：无网络副作用。
    异常：schema 不支持时抛出 ProviderError。
    设计说明：测试默认使用该 Provider，保证无密钥环境也能跑完整闭环。
    """

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """
        功能：初始化 mock Provider。
        参数：
            config：包含 model、style、seed 等可选字段。
        返回：无。
        副作用：保存配置。
        异常：无。
        设计说明：mock 的全部随机性都来自请求 seed 和配置 seed。
        """

        self.config = config or {}
        self.provider = "mock"
        self.model = str(self.config.get("model", "mock-deterministic-v1"))

    def generate_structured(self, request: LLMRequest, schema: type[T]) -> T:
        """
        功能：生成满足 schema 的结构化结果。
        参数：
            request：包含观察、记忆和提示词的请求。
            schema：目标结构类型。
        返回：schema 实例。
        副作用：无。
        异常：schema 不支持时抛出 ProviderError。
        设计说明：当前仅需动作和少量字典响应，后续可扩展经验修订 schema。
        """

        if schema is ActionDecision:
            return self._decide(request)  # type: ignore[return-value]
        if schema is dict:
            return self._generic_dict(request)  # type: ignore[return-value]
        raise ProviderError(f"mock Provider 不支持 schema：{schema!r}")

    def healthcheck(self) -> dict[str, object]:
        """
        功能：返回 mock Provider 状态。
        参数：无。
        返回：健康检查字典。
        副作用：无。
        异常：无。
        设计说明：doctor 命令可在无网络环境下验证核心路径。
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "available": True,
            "offline": True,
            "message": "mock Provider 可用，不需要 API Key。",
        }

    def _decide(self, request: LLMRequest) -> ActionDecision:
        """
        功能：根据可见局面和 persona 偏置选择动作。
        参数：
            request：LLM 请求。
        返回：ActionDecision。
        副作用：无。
        异常：无。
        设计说明：启发式足够稳定，可用于检验动作校验、记忆和实验调度。
        """

        obs = request.observation
        legal = obs.legal_actions
        strength = _hole_strength(obs.hole_cards)
        persona = str(request.memory_context.persona.get("name", "")).upper()
        risk = {"ENFP": 0.18, "ISTP": 0.08, "ESFJ": -0.08, "INTJ": -0.12}.get(persona, 0.0)
        seed = int(request.metadata.get("seed", obs.seed))
        rng = make_rng(seed, obs.agent_id, obs.hand_id, len(obs.action_history), persona)
        jitter = rng.random() * 0.12 - 0.06
        score = strength + risk + jitter
        raise_rule = legal.rule_for("raise")
        if raise_rule and raise_rule.min_amount is not None and score >= 0.72:
            amount = raise_rule.min_amount
            return ActionDecision(
                "raise",
                amount=amount,
                confidence=min(0.99, score),
                reason_summary="mock 根据可见牌力和风险偏置选择最小加注",
            )
        if legal.rule_for("call") and (obs.to_call <= 2 or score >= 0.48):
            return ActionDecision(
                "call",
                confidence=max(0.35, score),
                reason_summary="mock 认为补齐成本可接受",
            )
        if legal.rule_for("check"):
            return ActionDecision(
                "check",
                confidence=0.8,
                reason_summary="mock 在无需补注时选择过牌",
            )
        if legal.rule_for("fold"):
            return ActionDecision(
                "fold",
                confidence=0.75,
                reason_summary="mock 在牌力或价格不足时弃牌",
            )
        if raise_rule and raise_rule.min_amount is not None:
            return ActionDecision(
                "raise",
                amount=raise_rule.min_amount,
                confidence=0.2,
                reason_summary="mock 仅剩 raise 时使用最小金额",
            )
        return ActionDecision("fold", confidence=0.1, reason_summary="mock 无合法偏好时兜底")

    def _generic_dict(self, request: LLMRequest) -> dict[str, object]:
        """
        功能：生成通用结构化字典，供 persona 筛选和经验更新测试使用。
        参数：
            request：LLM 请求。
        返回：字典。
        副作用：无。
        异常：无。
        设计说明：保持确定性，不把自由文本解析作为测试前置条件。
        """

        persona = request.memory_context.persona.get("name", "neutral")
        return {
            "keep": False,
            "curated": f"mock 已按 {persona} 保留最相关的记忆摘要。",
            "new_md": "## 当前经验\n- 观察位置、入池成本和摊牌反馈，优先保留可迁移规律。",
            "supporting_fact_ids": [
                fact.record_id for fact in request.memory_context.facts[:3]
            ],
            "calibration_note": "mock 离线更新，无真实模型自我反思。",
            "self_check": "mock 未发现事实冲突。",
        }


def _hole_strength(cards: list[str]) -> float:
    """
    功能：从两张私牌估计一个可解释牌力分数。
    参数：
        cards：私有手牌。
    返回：0 到 1 附近的分数。
    副作用：无。
    异常：无。
    设计说明：mock 只看当前 Agent 自己能看到的牌，不使用对手私牌。
    """

    if len(cards) < 2:
        return 0.2
    ranks = [RANK_VALUE.get(card[0], 2) for card in cards]
    suited = cards[0][1] == cards[1][1]
    pair = ranks[0] == ranks[1]
    high = max(ranks)
    gap = abs(ranks[0] - ranks[1])
    score = (sum(ranks) - 4) / 24
    if pair:
        score += 0.28
    if suited:
        score += 0.06
    if gap <= 2:
        score += 0.04
    if high >= 13:
        score += 0.06
    return max(0.0, min(1.0, score))
