"""Canonical public poker-position facts shared by prompts and memory."""

from __future__ import annotations

from agentmemeval.path1_v1.core.domain import AgentObservation, PlayerPublicState


def active_public_players(observation: AgentObservation) -> list[PlayerPublicState]:
    """Return the current hand's non-busted public roster in natural seat order."""

    return sorted(
        (player for player in observation.players if not player.busted),
        key=lambda player: player.seat,
    )


def public_position_label(observation: AgentObservation) -> str:
    """Return BTN/SB/BB/BTN+N from public anchors and the active sparse roster."""

    dealer_seat = observation.dealer_seat
    small_blind_seat = observation.small_blind_seat
    big_blind_seat = observation.big_blind_seat
    seats = [player.seat for player in active_public_players(observation)]
    if observation.seat not in seats or dealer_seat not in seats:
        return "unknown"
    if observation.seat == dealer_seat:
        return "BTN/SB" if small_blind_seat == dealer_seat else "BTN"
    if observation.seat == small_blind_seat:
        return "SB"
    if observation.seat == big_blind_seat:
        return "BB"
    dealer_index = seats.index(dealer_seat)
    seat_index = seats.index(observation.seat)
    return f"BTN+{(seat_index - dealer_index) % len(seats)}"


def relative_active_seat_offset(observation: AgentObservation, seat: int) -> int:
    """Return clockwise active-roster offset from hero across sparse seat IDs."""

    seats = [player.seat for player in active_public_players(observation)]
    if observation.seat not in seats or seat not in seats:
        return len(seats)
    hero_index = seats.index(observation.seat)
    seat_index = seats.index(seat)
    return (seat_index - hero_index) % len(seats)
