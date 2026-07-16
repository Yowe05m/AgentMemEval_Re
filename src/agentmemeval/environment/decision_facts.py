"""Authoritative poker facts shared by prompts and audit logs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from agentmemeval.core.domain import AgentObservation
from agentmemeval.environment.hand_evaluator import RANKS, SUITS, evaluate_best


def build_decision_facts(observation: AgentObservation) -> dict[str, Any]:
    """Compute immutable, visible-information-only facts for one decision."""

    cards = [*observation.hole_cards, *observation.community_cards]
    if len(cards) >= 5:
        rank = evaluate_best(cards)
        made_hand_class = rank.class_name
        best_cards = list(rank.best_cards)
        current_class = rank.score[0]
    else:
        paired = observation.hole_cards[0][0] == observation.hole_cards[1][0]
        made_hand_class = "Pocket Pair" if paired else "Unpaired"
        best_cards = list(observation.hole_cards)
        current_class = -1

    flush_out_cards: list[str] = []
    straight_out_cards: list[str] = []
    if observation.phase in {"flop", "turn"}:
        known = set(cards)
        suit_counts = Counter(card[1] for card in cards)
        flush_suit, flush_known = max(suit_counts.items(), key=lambda item: item[1])
        if flush_known == 4 and current_class < 5:
            flush_out_cards = [
                rank_code + flush_suit
                for rank_code in RANKS
                if rank_code + flush_suit not in known
            ]
        if current_class < 4:
            for rank_code in RANKS:
                rank_is_out = False
                for suit_code in SUITS:
                    candidate = rank_code + suit_code
                    if candidate in known:
                        continue
                    if evaluate_best([*cards, candidate]).score[0] in {4, 8}:
                        rank_is_out = True
                        break
                if rank_is_out:
                    straight_out_cards.extend(
                        rank_code + suit_code
                        for suit_code in SUITS
                        if rank_code + suit_code not in known
                    )

    self_state = next(
        player for player in observation.players if player.agent_id == observation.agent_id
    )
    stack_before = max(0, self_state.stack)
    call_cost = min(max(0, observation.to_call), stack_before)
    required_equity = call_cost / max(1, observation.pot + call_cost)
    stack_fraction = call_cost / max(1, stack_before)
    active_opponents = sum(
        not player.folded and not player.busted and player.agent_id != observation.agent_id
        for player in observation.players
    )
    effective_stack = min(
        [stack_before]
        + [
            max(0, player.stack)
            for player in observation.players
            if player.agent_id != observation.agent_id and not player.folded and not player.busted
        ]
    )
    spr = effective_stack / max(1, observation.pot)
    draw_out_cards = sorted(set(flush_out_cards) | set(straight_out_cards))
    raise_rule = observation.legal_actions.rule_for("raise")
    raise_facts: dict[str, object] = {"available": False}
    if raise_rule and raise_rule.min_amount is not None and raise_rule.max_amount is not None:
        min_cost = min(stack_before, max(0, raise_rule.min_amount - self_state.current_bet))
        max_cost = min(stack_before, max(0, raise_rule.max_amount - self_state.current_bet))
        raise_facts = {
            "available": True,
            "min_amount": raise_rule.min_amount,
            "max_amount": raise_rule.max_amount,
            "min_cost": min_cost,
            "max_cost": max_cost,
            "min_stack_fraction": min_cost / max(1, stack_before),
            "max_stack_fraction": max_cost / max(1, stack_before),
            "max_is_all_in": max_cost >= stack_before,
        }
    return {
        "made_hand_class": made_hand_class,
        "best_cards": best_cards,
        "draw": {
            "flush_draw": bool(flush_out_cards),
            "straight_draw": bool(straight_out_cards),
            "flush_out_cards": flush_out_cards,
            "straight_out_cards": straight_out_cards,
            "unique_out_cards": draw_out_cards,
            "outs": len(draw_out_cards),
            "cards_to_come": (
                0 if observation.phase == "river" else 2 if observation.phase == "flop" else 1
            ),
        },
        "call": {
            "stack_before": stack_before,
            "call_cost": call_cost,
            "required_equity": required_equity,
            "stack_fraction": stack_fraction,
            "is_all_in": call_cost > 0 and call_cost >= stack_before,
            "risk_label": (
                "all_in" if call_cost > 0 and call_cost >= stack_before else
                "high" if stack_fraction >= 0.5 else
                "medium" if stack_fraction >= 0.25 else "low"
            ),
        },
        "raise": raise_facts,
        "spr": spr,
        "effective_stack": effective_stack,
        "active_opponents": active_opponents,
        "multiway_players": active_opponents + 1,
    }
