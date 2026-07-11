"""Explicit raise-to sizing policies for native and bounded local-model action spaces."""

from __future__ import annotations

from dataclasses import dataclass

from agentmemeval.core.domain import AgentObservation
from agentmemeval.core.errors import ConfigError

NATIVE_NO_LIMIT = "native_no_limit"
LOCAL_DISCRETE = "local_discrete"
SUPPORTED_RAISE_SIZING_POLICIES = {NATIVE_NO_LIMIT, LOCAL_DISCRETE}


@dataclass(frozen=True, slots=True)
class RaiseSizingPlan:
    """Describe the action-space restriction applied to one decision."""

    policy: str
    allowed_amounts: tuple[int, ...] | None
    pot_after_call: int
    stack_to_pot_ratio: float
    native_min_amount: int | None
    native_max_amount: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "allowed_amounts": (
                list(self.allowed_amounts) if self.allowed_amounts is not None else None
            ),
            "pot_after_call": self.pot_after_call,
            "stack_to_pot_ratio": self.stack_to_pot_ratio,
            "native_min_amount": self.native_min_amount,
            "native_max_amount": self.native_max_amount,
        }


def build_raise_sizing_plan(
    observation: AgentObservation,
    policy: str,
) -> RaiseSizingPlan:
    """Build legal raise-to candidates without changing the environment's native rules."""

    normalized = _normalize_policy(policy)
    self_state = next(
        player for player in observation.players if player.agent_id == observation.agent_id
    )
    call_cost = min(max(0, observation.to_call), max(0, self_state.stack))
    pot_after_call = max(1, observation.pot + call_cost)
    stack_after_call = max(0, self_state.stack - call_cost)
    stack_to_pot_ratio = stack_after_call / pot_after_call
    raise_rule = observation.legal_actions.rule_for("raise")
    native_minimum = raise_rule.min_amount if raise_rule is not None else None
    native_maximum = raise_rule.max_amount if raise_rule is not None else None
    if normalized == NATIVE_NO_LIMIT or raise_rule is None:
        return RaiseSizingPlan(
            policy=normalized,
            allowed_amounts=None,
            pot_after_call=pot_after_call,
            stack_to_pot_ratio=stack_to_pot_ratio,
            native_min_amount=native_minimum,
            native_max_amount=native_maximum,
        )
    if raise_rule.min_amount is None or raise_rule.max_amount is None:
        return RaiseSizingPlan(
            policy=normalized,
            allowed_amounts=None,
            pot_after_call=pot_after_call,
            stack_to_pot_ratio=stack_to_pot_ratio,
            native_min_amount=native_minimum,
            native_max_amount=native_maximum,
        )

    minimum = raise_rule.min_amount
    maximum = raise_rule.max_amount
    current_line = observation.current_bet
    candidates = {
        minimum,
        current_line + max(1, round(pot_after_call * 0.5)),
        current_line + pot_after_call,
    }
    allowed = {
        max(minimum, min(maximum, amount))
        for amount in candidates
    }
    if stack_to_pot_ratio <= 1.5:
        allowed.add(maximum)
    if minimum != maximum and stack_to_pot_ratio > 1.5:
        max_extra_cost = max(0, maximum - self_state.current_bet)
        if max_extra_cost >= self_state.stack * 0.8:
            allowed.discard(maximum)
    if not allowed:
        allowed.add(minimum)
    return RaiseSizingPlan(
        policy=normalized,
        allowed_amounts=tuple(sorted(allowed)),
        pot_after_call=pot_after_call,
        stack_to_pot_ratio=stack_to_pot_ratio,
        native_min_amount=native_minimum,
        native_max_amount=native_maximum,
    )


def _normalize_policy(policy: str) -> str:
    aliases = {
        "native": NATIVE_NO_LIMIT,
        "no_limit": NATIVE_NO_LIMIT,
        "discrete": LOCAL_DISCRETE,
    }
    normalized = aliases.get(str(policy).strip().lower(), str(policy).strip().lower())
    if normalized not in SUPPORTED_RAISE_SIZING_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_RAISE_SIZING_POLICIES))
        raise ConfigError(f"未知 raise_sizing_policy={policy!r}；可选：{supported}")
    return normalized
