"""
模块说明：本模块测试结构化动作校验。
核心职责：覆盖非法动作拦截、raise 金额边界和安全回退。
输入与输出：输入测试用动作集合，输出 pytest 断言结果。
依赖边界：只依赖环境 action_guard 与核心领域对象。
不负责：不测试具体扑克环境推进。
"""

import pytest

from agentmemeval.core.domain import ActionDecision, LegalAction, LegalActionSet
from agentmemeval.core.errors import ActionValidationError
from agentmemeval.environment.action_guard import ActionGuard, coerce_decision


def test_action_guard_clamps_invalid_raise_without_fallback() -> None:
    """
    功能：验证方向正确但金额越界的 raise 会裁剪到合法下界。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：保留模型动作意图，避免金额格式问题被扭曲成 fold/check。
    """

    legal = LegalActionSet([LegalAction("fold"), LegalAction("check"), LegalAction("raise", 8, 20)])
    result = ActionGuard().guard(ActionDecision("raise", amount=2), legal)
    assert result.fallback_used is False
    assert result.repaired is True
    assert result.action.action_type == "raise"
    assert result.action.amount == 8
    assert result.errors


def test_guard_repairs_raise_to_nearest_discrete_candidate() -> None:
    guard = ActionGuard()
    legal = LegalActionSet(
        [LegalAction("fold"), LegalAction("call"), LegalAction("raise", 4, 1000)]
    )

    result = guard.guard(
        ActionDecision("raise", amount=1000),
        legal,
        allowed_raise_amounts=(4, 7),
    )

    assert result.repaired is True
    assert result.fallback_used is False
    assert result.action.amount == 7


def test_action_guard_strict_raises_error() -> None:
    """
    功能：验证严格模式下非法动作直接报错。
    参数：无。
    返回：无。
    副作用：无。
    异常：期望 ActionValidationError。
    设计说明：单元测试需要能精确断言动作边界。
    """

    legal = LegalActionSet([LegalAction("fold"), LegalAction("call")])
    with pytest.raises(ActionValidationError):
        ActionGuard().guard(ActionDecision("raise", amount=10), legal, strict=True)


def test_coerce_decision_accepts_legacy_type_key() -> None:
    """
    功能：验证兼容响应中的 type 字段。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：兼容 Provider 可能返回旧字段名，转换后仍统一校验。
    """

    decision = coerce_decision({"type": "call", "amount": None, "reason": "ok"})
    assert decision.action_type == "call"
    assert decision.reason_summary == "ok"


def test_coerce_decision_drops_amount_for_non_raise() -> None:
    """
    功能：验证兼容模型给 call/check/fold 附带 amount 时自动归一。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：LM Studio 本地模型常把 call 的补齐额写入 amount，不能因此回退成 fold。
    """

    decision = coerce_decision({"action_type": "call", "amount": 2, "reason": "call 2"})
    assert decision.action_type == "call"
    assert decision.amount is None


def test_coerce_decision_clamps_confidence() -> None:
    """Provider 自报置信度必须落入公开 schema 的 0 到 1 范围。"""

    assert coerce_decision({"action_type": "check", "confidence": 3}).confidence == 1.0
    assert coerce_decision({"action_type": "check", "confidence": -2}).confidence == 0.0
