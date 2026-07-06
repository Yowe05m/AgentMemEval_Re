"""
模块说明：本模块负责结构化动作的合法性校验与安全回退。
核心职责：把 LLM 或启发式 Agent 的动作约束到环境给出的合法动作集合内。
输入与输出：输入 ActionDecision 与 LegalActionSet，输出可执行 ActionDecision 和校验元数据。
依赖边界：只依赖核心领域对象与领域异常，不依赖具体扑克环境或 Provider。
不负责：不推进环境，不计算下注规则，不修改 Agent 记忆。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.core.domain import ActionDecision, LegalActionSet
from agentmemeval.core.errors import ActionValidationError


@dataclass(slots=True)
class GuardResult:
    """
    功能：表示动作校验后的结果。
    参数：
        action：合法动作。
        repaired：是否发生回退或修正。
        errors：校验过程中发现的问题。
        fallback_used：是否使用安全回退策略。
    返回：校验结果对象。
    副作用：无。
    异常：无。
    设计说明：实验日志需要知道动作是否被修正，便于评估 Provider 稳定性。
    """

    action: ActionDecision
    repaired: bool = False
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False


class ActionGuard:
    """
    功能：校验并修正结构化动作。
    参数：
        fallback_order：非法动作时的偏好顺序。
    返回：动作保护器实例。
    副作用：无。
    异常：严格模式下非法动作会抛出 ActionValidationError。
    设计说明：Provider 输出再像 JSON，也必须经由环境规则校验后才能执行。
    """

    def __init__(self, fallback_order: tuple[str, ...] = ("check", "fold", "call")) -> None:
        """
        功能：初始化动作保护器。
        参数：
            fallback_order：安全回退动作偏好。
        返回：无。
        副作用：保存偏好配置。
        异常：无。
        设计说明：默认优先 check，其次 fold，避免非法输出变成激进动作。
        """

        self.fallback_order = fallback_order

    def guard(
        self,
        decision: ActionDecision,
        legal_actions: LegalActionSet,
        strict: bool = False,
    ) -> GuardResult:
        """
        功能：将候选动作校验为合法动作。
        参数：
            decision：候选决策。
            legal_actions：环境给出的合法动作集合。
            strict：为 True 时非法动作直接抛错。
        返回：GuardResult。
        副作用：无。
        异常：strict=True 且动作非法时抛出 ActionValidationError。
        设计说明：生产实验采用回退保证赛程完成，单元测试可用 strict 断言边界。
        """

        errors = self._validate(decision, legal_actions)
        if not errors:
            return GuardResult(action=decision)
        if strict:
            raise ActionValidationError("；".join(errors))
        fallback = self._fallback(legal_actions)
        return GuardResult(action=fallback, repaired=True, errors=errors, fallback_used=True)

    def _validate(self, decision: ActionDecision, legal_actions: LegalActionSet) -> list[str]:
        """
        功能：收集动作不合法的原因。
        参数：
            decision：候选动作。
            legal_actions：合法动作集合。
        返回：错误说明列表。
        副作用：无。
        异常：无。
        设计说明：返回所有明显问题，日志比只给第一个错误更适合排查 Provider。
        """

        errors: list[str] = []
        rule = legal_actions.rule_for(decision.action_type)
        if rule is None:
            allowed = ", ".join(sorted(legal_actions.types())) or "无"
            return [f"动作 {decision.action_type!r} 不在合法集合内：{allowed}"]
        if decision.action_type == "raise":
            if decision.amount is None:
                errors.append("raise 动作缺少 amount")
            elif rule.min_amount is not None and decision.amount < rule.min_amount:
                errors.append(f"raise amount={decision.amount} 小于最小值 {rule.min_amount}")
            elif rule.max_amount is not None and decision.amount > rule.max_amount:
                errors.append(f"raise amount={decision.amount} 大于最大值 {rule.max_amount}")
        elif decision.amount not in (None, 0):
            errors.append(f"{decision.action_type} 动作不应携带 amount={decision.amount}")
        return errors

    def _fallback(self, legal_actions: LegalActionSet) -> ActionDecision:
        """
        功能：根据配置生成安全回退动作。
        参数：
            legal_actions：当前合法动作集合。
        返回：合法 ActionDecision。
        副作用：无。
        异常：没有合法动作时抛出 ActionValidationError。
        设计说明：回退策略必须保守且可解释，不能把解析失败变成随机加注。
        """

        for action_type in self.fallback_order:
            if legal_actions.rule_for(action_type) is not None:
                return ActionDecision(
                    action_type=action_type,
                    reason_summary="候选动作非法，使用安全回退",
                )
        raise_rule = legal_actions.rule_for("raise")
        if raise_rule is not None and raise_rule.min_amount is not None:
            return ActionDecision(
                action_type="raise",
                amount=raise_rule.min_amount,
                reason_summary="仅有 raise 可用，使用最小合法金额",
            )
        raise ActionValidationError("当前没有可回退的合法动作")


def coerce_decision(payload: object) -> ActionDecision:
    """
    功能：把字典或 ActionDecision 转换为标准动作对象。
    参数：
        payload：Provider 返回的结构化片段。
    返回：ActionDecision。
    副作用：无。
    异常：结构不支持时抛出 ActionValidationError。
    设计说明：Provider 兼容层可以先做宽松转换，再交给 guard 做严格合法性检查。
    """

    if isinstance(payload, ActionDecision):
        return payload
    if not isinstance(payload, dict):
        raise ActionValidationError(f"动作响应不是结构化对象：{payload!r}")
    action_type = payload.get("action_type") or payload.get("type")
    if not isinstance(action_type, str):
        raise ActionValidationError(f"动作响应缺少 action_type：{payload!r}")
    amount = payload.get("amount")
    if amount is not None:
        try:
            amount = int(amount)
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(f"amount 不是整数：{payload!r}") from exc
    confidence = payload.get("confidence", 1.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = payload.get("reason_summary") or payload.get("reason") or ""
    return ActionDecision(
        action_type=action_type,
        amount=amount,
        confidence=confidence,
        reason_summary=str(reason)[:300],
        raw_response={key: value for key, value in payload.items() if key != "chain_of_thought"},
    )
