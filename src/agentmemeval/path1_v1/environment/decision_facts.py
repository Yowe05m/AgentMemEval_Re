"""Authoritative poker facts shared by prompts and audit logs."""

from __future__ import annotations

from typing import Any

from agentmemeval.path1_v1.core.domain import AgentObservation, PlayerPublicState
from agentmemeval.path1_v1.environment.hand_evaluator import RANKS, SUITS, evaluate_best


def chips_to_big_blinds(chips: int, big_blind_amount: int | None) -> float | None:
    """Convert an integer chip amount only when a positive BB amount is known."""

    if type(chips) is not int or type(big_blind_amount) is not int:
        return None
    if big_blind_amount <= 0:
        return None
    return round(chips / big_blind_amount, 4)


def build_decision_facts(observation: AgentObservation) -> dict[str, Any]:
    """Compute immutable, visible-information-only facts for one decision."""

    hole_cards, community_cards = _normalized_visible_cards(observation)
    cards = [*hole_cards, *community_cards]
    if len(cards) >= 5:
        rank = evaluate_best(cards)
        made_hand_class = rank.class_name
        best_cards = list(rank.best_cards)
        current_class = rank.score[0]
    else:
        paired = hole_cards[0][0] == hole_cards[1][0]
        made_hand_class = "Pocket Pair" if paired else "Unpaired"
        best_cards = list(hole_cards)
        current_class = -1

    draw = _draw_facts(
        hole_cards=hole_cards,
        community_cards=community_cards,
        current_class=current_class,
        phase=observation.phase,
    )
    self_state = next(
        player for player in observation.players if player.agent_id == observation.agent_id
    )
    stack_before = max(0, self_state.stack)
    call_cost = min(max(0, observation.to_call), stack_before)
    contestable_pot_before_call, unattributed_pot = _contestable_pot_before_call(
        observation,
        self_state,
        call_cost,
    )
    contestable_pot_after_call = contestable_pot_before_call + call_cost
    required_equity = call_cost / max(1, contestable_pot_after_call)
    stack_fraction = call_cost / max(1, stack_before)

    active_opponent_states = [
        player
        for player in observation.players
        if player.agent_id != observation.agent_id and not player.folded and not player.busted
    ]
    effective_stack_by_opponent = {
        player.agent_id: _contestable_stack_from_now(
            self_state,
            player,
            stack_before=stack_before,
        )
        for player in active_opponent_states
    }
    effective_values = list(effective_stack_by_opponent.values())
    effective_stack_min = min(effective_values, default=0)
    effective_stack_max = max(effective_values, default=0)
    effective_stack = effective_stack_max
    effective_stack_after_call_by_opponent = {
        agent_id: max(0, amount - call_cost)
        for agent_id, amount in effective_stack_by_opponent.items()
    }
    effective_stack_after_call = max(
        effective_stack_after_call_by_opponent.values(),
        default=0,
    )
    spr = effective_stack / max(1, contestable_pot_before_call)
    spr_after_call = effective_stack_after_call / max(1, contestable_pot_after_call)
    small_blind_amount = (
        observation.small_blind_amount
        if type(observation.small_blind_amount) is int and observation.small_blind_amount > 0
        else None
    )
    big_blind_amount = (
        observation.big_blind_amount
        if type(observation.big_blind_amount) is int and observation.big_blind_amount > 0
        else None
    )

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
            "min_amount_chips": raise_rule.min_amount,
            "max_amount_chips": raise_rule.max_amount,
            "min_cost_chips": min_cost,
            "max_cost_chips": max_cost,
            "min_stack_fraction": min_cost / max(1, stack_before),
            "max_stack_fraction": max_cost / max(1, stack_before),
            "max_is_all_in": max_cost >= stack_before,
        }
        if big_blind_amount is not None:
            raise_facts.update(
                {
                    "min_amount_big_blinds": chips_to_big_blinds(
                        raise_rule.min_amount, big_blind_amount
                    ),
                    "max_amount_big_blinds": chips_to_big_blinds(
                        raise_rule.max_amount, big_blind_amount
                    ),
                    "min_cost_big_blinds": chips_to_big_blinds(min_cost, big_blind_amount),
                    "max_cost_big_blinds": chips_to_big_blinds(max_cost, big_blind_amount),
                }
            )
    call_facts: dict[str, object] = {
        "stack_before": stack_before,
        "call_cost": call_cost,
        "pot_before_call": max(0, observation.pot),
        "contestable_pot_before_call": contestable_pot_before_call,
        "contestable_pot_after_call": contestable_pot_after_call,
        "unattributed_pot": unattributed_pot,
        "stack_before_chips": stack_before,
        "call_cost_chips": call_cost,
        "pot_before_call_chips": max(0, observation.pot),
        "contestable_pot_before_call_chips": contestable_pot_before_call,
        "contestable_pot_after_call_chips": contestable_pot_after_call,
        "unattributed_pot_chips": unattributed_pot,
        "required_equity": required_equity,
        "stack_fraction": stack_fraction,
        "is_all_in": call_cost > 0 and call_cost >= stack_before,
        "risk_label": (
            "all_in"
            if call_cost > 0 and call_cost >= stack_before
            else "high"
            if stack_fraction >= 0.5
            else "medium"
            if stack_fraction >= 0.25
            else "low"
        ),
    }
    result: dict[str, Any] = {
        "made_hand_class": made_hand_class,
        "best_cards": best_cards,
        "draw": draw,
        "chip_unit": "chips",
        "small_blind_amount": small_blind_amount,
        "big_blind_amount": big_blind_amount,
        "stack_chips": stack_before,
        "pot_chips": max(0, observation.pot),
        "current_bet_chips": max(0, observation.current_bet),
        "to_call_chips": max(0, observation.to_call),
        "call": call_facts,
        "raise": raise_facts,
        "spr": spr,
        "spr_after_call": spr_after_call,
        "contestable_pot": contestable_pot_before_call,
        "contestable_pot_after_call": contestable_pot_after_call,
        "contestable_pot_chips": contestable_pot_before_call,
        "contestable_pot_after_call_chips": contestable_pot_after_call,
        "effective_stack": effective_stack,
        "effective_stack_min": effective_stack_min,
        "effective_stack_max": effective_stack_max,
        "effective_stack_by_opponent": effective_stack_by_opponent,
        "effective_stack_after_call": effective_stack_after_call,
        "effective_stack_after_call_by_opponent": effective_stack_after_call_by_opponent,
        "effective_stack_chips": effective_stack,
        "effective_stack_min_chips": effective_stack_min,
        "effective_stack_max_chips": effective_stack_max,
        "effective_stack_by_opponent_chips": effective_stack_by_opponent,
        "effective_stack_after_call_chips": effective_stack_after_call,
        "effective_stack_after_call_by_opponent_chips": (effective_stack_after_call_by_opponent),
        "effective_stack_basis": "max_contestable_from_now_across_active_opponents",
        "active_opponents": len(active_opponent_states),
        "multiway_players": len(active_opponent_states) + 1,
    }
    if big_blind_amount is not None:
        call_facts.update(
            {
                "stack_before_big_blinds": chips_to_big_blinds(stack_before, big_blind_amount),
                "call_cost_big_blinds": chips_to_big_blinds(call_cost, big_blind_amount),
                "pot_before_call_big_blinds": chips_to_big_blinds(
                    max(0, observation.pot), big_blind_amount
                ),
                "contestable_pot_before_call_big_blinds": chips_to_big_blinds(
                    contestable_pot_before_call, big_blind_amount
                ),
                "contestable_pot_after_call_big_blinds": chips_to_big_blinds(
                    contestable_pot_after_call, big_blind_amount
                ),
            }
        )
        result.update(
            {
                "stack_big_blinds": chips_to_big_blinds(stack_before, big_blind_amount),
                "pot_big_blinds": chips_to_big_blinds(max(0, observation.pot), big_blind_amount),
                "current_bet_big_blinds": chips_to_big_blinds(
                    max(0, observation.current_bet), big_blind_amount
                ),
                "to_call_big_blinds": chips_to_big_blinds(
                    max(0, observation.to_call), big_blind_amount
                ),
                "contestable_pot_big_blinds": chips_to_big_blinds(
                    contestable_pot_before_call, big_blind_amount
                ),
                "contestable_pot_after_call_big_blinds": chips_to_big_blinds(
                    contestable_pot_after_call, big_blind_amount
                ),
                "effective_stack_big_blinds": chips_to_big_blinds(
                    effective_stack, big_blind_amount
                ),
                "effective_stack_min_big_blinds": chips_to_big_blinds(
                    effective_stack_min, big_blind_amount
                ),
                "effective_stack_max_big_blinds": chips_to_big_blinds(
                    effective_stack_max, big_blind_amount
                ),
                "effective_stack_by_opponent_big_blinds": {
                    agent_id: chips_to_big_blinds(amount, big_blind_amount)
                    for agent_id, amount in effective_stack_by_opponent.items()
                },
                "effective_stack_after_call_big_blinds": chips_to_big_blinds(
                    effective_stack_after_call, big_blind_amount
                ),
                "effective_stack_after_call_by_opponent_big_blinds": {
                    agent_id: chips_to_big_blinds(amount, big_blind_amount)
                    for agent_id, amount in effective_stack_after_call_by_opponent.items()
                },
            }
        )
    return result


def _draw_facts(
    *,
    hole_cards: list[str],
    community_cards: list[str],
    current_class: int,
    phase: str,
) -> dict[str, object]:
    """Separate hero-contributing draws from runouts that only complete the board."""

    applicable = phase in {"flop", "turn"}
    cards_to_come = (
        5 if phase == "preflop" else 2 if phase == "flop" else 1 if phase == "turn" else 0
    )
    hero_flush_outs: set[str] = set()
    hero_straight_outs: set[str] = set()
    board_flush_outs: set[str] = set()
    board_straight_outs: set[str] = set()
    nominal_flush_outs: set[str] = set()
    nominal_straight_outs: set[str] = set()
    if applicable:
        cards = [*hole_cards, *community_cards]
        known = set(cards)
        for rank_code in RANKS:
            for suit_code in SUITS:
                candidate = rank_code + suit_code
                if candidate in known:
                    continue
                all_after = [*cards, candidate]
                board_after = [*community_cards, candidate]
                if current_class < 5:
                    all_flush = _best_flush_score(all_after)
                    board_flush = _best_flush_score(board_after)
                    if all_flush:
                        nominal_flush_outs.add(candidate)
                    if board_flush:
                        board_flush_outs.add(candidate)
                    if all_flush > board_flush:
                        hero_flush_outs.add(candidate)
                if current_class < 4:
                    all_straight = _straight_high(all_after)
                    board_straight = _straight_high(board_after)
                    if all_straight:
                        nominal_straight_outs.add(candidate)
                    if board_straight:
                        board_straight_outs.add(candidate)
                    if all_straight > board_straight:
                        hero_straight_outs.add(candidate)

    hero_outs = hero_flush_outs | hero_straight_outs
    board_outs = board_flush_outs | board_straight_outs
    nominal_outs = nominal_flush_outs | nominal_straight_outs
    hero_draw = _draw_view(hero_flush_outs, hero_straight_outs)
    board_completion_threat = _draw_view(board_flush_outs, board_straight_outs)
    return {
        "applicable": applicable,
        "cards_to_come": cards_to_come,
        "hero_draw": hero_draw,
        "board_completion_threat": board_completion_threat,
        "nominal_visible_completion_out_cards": sorted(nominal_outs),
        "nominal_visible_completion_outs": len(nominal_outs),
        "hole_card_contribution": {
            "flush": bool(hero_flush_outs),
            "straight": bool(hero_straight_outs),
            "any": bool(hero_outs),
        },
        "outs_semantics": "nominal_one_card_category_completion_not_clean_winning_equity",
        # Compatibility aliases now intentionally mean hero-contributing draws only.
        "flush_draw": bool(hero_flush_outs),
        "straight_draw": bool(hero_straight_outs),
        "flush_out_cards": sorted(hero_flush_outs),
        "straight_out_cards": sorted(hero_straight_outs),
        "unique_out_cards": sorted(hero_outs),
        "outs": len(hero_outs),
        "board_only_out_cards": sorted(board_outs - hero_outs),
    }


def _draw_view(flush_outs: set[str], straight_outs: set[str]) -> dict[str, object]:
    unique_outs = flush_outs | straight_outs
    return {
        "flush_draw": bool(flush_outs),
        "straight_draw": bool(straight_outs),
        "flush_out_cards": sorted(flush_outs),
        "straight_out_cards": sorted(straight_outs),
        "unique_out_cards": sorted(unique_outs),
        "outs": len(unique_outs),
    }


def _best_flush_score(cards: list[str]) -> tuple[int, ...]:
    """Return the strongest visible five-card flush ranks, or an empty tuple."""

    best: tuple[int, ...] = ()
    for suit_code in SUITS:
        ranks = sorted(
            (RANKS.index(card[0]) + 2 for card in cards if card[1] == suit_code),
            reverse=True,
        )
        if len(ranks) >= 5:
            best = max(best, tuple(ranks[:5]))
    return best


def _straight_high(cards: list[str]) -> int:
    """Return the highest straight rank visible in cards without hiding it behind a flush."""

    ranks = {RANKS.index(card[0]) + 2 for card in cards}
    if 14 in ranks:
        ranks.add(1)
    for high in range(14, 4, -1):
        if all(rank in ranks for rank in range(high - 4, high + 1)):
            return high
    return 0


def _contestable_pot_before_call(
    observation: AgentObservation,
    self_state: PlayerPublicState,
    call_cost: int,
) -> tuple[int, int]:
    """Cap every contribution at the hero's post-call side-pot eligibility level."""

    pot = max(0, observation.pot)
    contributions = [max(0, player.total_committed) for player in observation.players]
    tracked_pot = sum(contributions)
    unattributed_pot = max(0, pot - tracked_pot)
    hero_cap = max(0, self_state.total_committed) + call_cost
    capped_contributions = sum(min(contribution, hero_cap) for contribution in contributions)
    contestable = min(pot, capped_contributions + unattributed_pot)
    return contestable, unattributed_pot


def _contestable_stack_from_now(
    self_state: PlayerPublicState,
    opponent: PlayerPublicState,
    *,
    stack_before: int,
) -> int:
    """Return how many additional hero chips this opponent can still match from now."""

    unmatched_opponent_commitment = max(
        0,
        max(0, opponent.total_committed) - max(0, self_state.total_committed),
    )
    opponent_exposure = max(0, opponent.stack) + unmatched_opponent_commitment
    return min(stack_before, opponent_exposure)


def _normalized_visible_cards(
    observation: AgentObservation,
) -> tuple[list[str], list[str]]:
    """Normalize visible cards and reject impossible duplicate observations on every street."""

    if len(observation.hole_cards) != 2:
        raise ValueError("decision facts require exactly two hero hole cards")
    hole_cards = [_normalize_card(card) for card in observation.hole_cards]
    community_cards = [_normalize_card(card) for card in observation.community_cards]
    cards = [*hole_cards, *community_cards]
    if len(cards) > 7:
        raise ValueError("decision facts allow at most seven visible cards")
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate visible cards are not allowed: {cards!r}")
    return hole_cards, community_cards


def _normalize_card(card: str) -> str:
    if not isinstance(card, str):
        raise ValueError(f"card code must be a string, got {type(card).__name__}")
    value = card.strip()
    if len(value) != 2:
        raise ValueError(f"invalid card code {card!r}; expected rank+suit such as 'As'")
    rank = value[0].upper()
    suit = value[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"invalid card code {card!r}; valid ranks={RANKS}, suits={SUITS}")
    return rank + suit
