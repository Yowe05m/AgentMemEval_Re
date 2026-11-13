from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.path1_v1.core.domain import ActionDecision, LegalActionSet
from agentmemeval.path1_v1.core.errors import ActionValidationError


@dataclass(slots=True)
class GuardResult:
    action: ActionDecision
    repaired: bool = False
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False


class ActionGuard:
    def __init__(self, fallback_order: tuple[str, ...] = ("check", "fold", "call")) -> None:

        self.fallback_order = fallback_order

    def guard(
        self,
        decision: ActionDecision,
        legal_actions: LegalActionSet,
        strict: bool = False,
        allowed_raise_amounts: tuple[int, ...] | None = None,
    ) -> GuardResult:

        errors = self._validate(decision, legal_actions, allowed_raise_amounts)
        if not errors:
            return GuardResult(action=decision)
        if strict:
            raise ActionValidationError("; ".join(errors))
        repaired_raise = self._repair_raise(
            decision,
            legal_actions,
            allowed_raise_amounts,
        )
        if repaired_raise is not None:
            return GuardResult(
                action=repaired_raise,
                repaired=True,
                errors=errors,
                fallback_used=False,
            )
        fallback = self._fallback(legal_actions)
        return GuardResult(action=fallback, repaired=True, errors=errors, fallback_used=True)

    def _validate(
        self,
        decision: ActionDecision,
        legal_actions: LegalActionSet,
        allowed_raise_amounts: tuple[int, ...] | None = None,
    ) -> list[str]:

        errors: list[str] = []
        rule = legal_actions.rule_for(decision.action_type)
        if rule is None:
            allowed = ", ".join(sorted(legal_actions.types())) or "none"
            return [f"action {decision.action_type!r} is not legal; allowed: {allowed}"]
        if decision.action_type == "raise":
            if decision.amount is None:
                errors.append("raise action is missing amount")
            elif rule.min_amount is not None and decision.amount < rule.min_amount:
                errors.append(f"raise amount={decision.amount} is below minimum {rule.min_amount}")
            elif rule.max_amount is not None and decision.amount > rule.max_amount:
                errors.append(f"raise amount={decision.amount} exceeds maximum {rule.max_amount}")
            elif allowed_raise_amounts is not None and decision.amount not in allowed_raise_amounts:
                errors.append(
                    f"raise amount={decision.amount} is not in discrete candidates "
                    f"{list(allowed_raise_amounts)}"
                )
        elif decision.amount not in (None, 0):
            errors.append(
                f"{decision.action_type} action must not include amount={decision.amount}"
            )
        return errors

    def _repair_raise(
        self,
        decision: ActionDecision,
        legal_actions: LegalActionSet,
        allowed_raise_amounts: tuple[int, ...] | None = None,
    ) -> ActionDecision | None:

        if decision.action_type != "raise":
            return None
        rule = legal_actions.rule_for("raise")
        if rule is None or rule.min_amount is None:
            return None
        amount = rule.min_amount if decision.amount is None else decision.amount
        amount = max(rule.min_amount, amount)
        if rule.max_amount is not None:
            amount = min(rule.max_amount, amount)
        if allowed_raise_amounts:
            amount = min(
                allowed_raise_amounts,
                key=lambda candidate: (abs(candidate - amount), candidate),
            )
        return ActionDecision(
            action_type="raise",
            amount=amount,
            confidence=decision.confidence,
            reason_summary=decision.reason_summary,
            intent=decision.intent,
            raw_response=dict(decision.raw_response),
        )

    def _fallback(self, legal_actions: LegalActionSet) -> ActionDecision:

        for action_type in self.fallback_order:
            if legal_actions.rule_for(action_type) is not None:
                return ActionDecision(
                    action_type=action_type,
                    reason_summary="invalid candidate action; used safe fallback",
                )
        raise_rule = legal_actions.rule_for("raise")
        if raise_rule is not None and raise_rule.min_amount is not None:
            return ActionDecision(
                action_type="raise",
                amount=raise_rule.min_amount,
                reason_summary="raise was the only legal action; used minimum amount",
            )
        raise ActionValidationError("no legal fallback action is available")


def coerce_decision(payload: object) -> ActionDecision:

    if isinstance(payload, ActionDecision):
        return payload
    if not isinstance(payload, dict):
        raise ActionValidationError(f"action response is not an object: {payload!r}")
    action_type = payload.get("action_type") or payload.get("type")
    if not isinstance(action_type, str):
        raise ActionValidationError(f"action response lacks action_type: {payload!r}")
    amount = payload.get("amount")
    if amount is not None:
        try:
            amount = int(amount)
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(f"amount is not an integer: {payload!r}") from exc
    if action_type != "raise":
        amount = None
    confidence = payload.get("confidence", 1.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    reason = payload.get("reason_summary") or payload.get("reason") or ""
    intent = payload.get("intent") or payload.get("declared_intent") or ""
    safe_raw_fields = {
        "action_type",
        "type",
        "amount",
        "confidence",
        "reason_summary",
        "reason",
        "intent",
        "declared_intent",
        "provider_recovered_from_truncation",
    }
    return ActionDecision(
        action_type=action_type,
        amount=amount,
        confidence=confidence,
        reason_summary=str(reason)[:300],
        intent=str(intent)[:40],
        raw_response={key: value for key, value in payload.items() if key in safe_raw_fields},
    )
