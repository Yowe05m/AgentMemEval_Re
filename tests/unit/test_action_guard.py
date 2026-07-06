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


def test_action_guard_fallbacks_to_check() -> None:
    """
    功能：验证非法 raise 会回退到 check。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：解析失败不能变成随机激进动作。
    """

    legal = LegalActionSet([LegalAction("fold"), LegalAction("check"), LegalAction("raise", 8, 20)])
    result = ActionGuard().guard(ActionDecision("raise", amount=2), legal)
    assert result.fallback_used is True
    assert result.action.action_type == "check"
    assert result.errors


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
