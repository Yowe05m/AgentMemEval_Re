from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
MAX_HOLDEM_CARDS = 7
RANK_CLASS_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "Pair",
    0: "High Card",
}


@dataclass(frozen=True, slots=True)
class PokerHandRank:
    score: tuple[int, tuple[int, ...]]
    class_name: str
    best_cards: tuple[str, ...]


def evaluate_best(cards: list[str]) -> PokerHandRank:

    cards = _validate_cards(cards, min_count=5, max_count=MAX_HOLDEM_CARDS)
    best_score: tuple[int, tuple[int, ...]] = (-1, ())
    best_cards: tuple[str, ...] = ()
    for combo in combinations(cards, 5):
        score = score_five(list(combo))
        if score > best_score:
            best_score = score
            best_cards = tuple(combo)
    return PokerHandRank(
        score=best_score,
        class_name=RANK_CLASS_NAMES[best_score[0]],
        best_cards=best_cards,
    )


def score_five(cards: list[str]) -> tuple[int, tuple[int, ...]]:

    cards = _validate_cards(cards, min_count=5, max_count=5)
    ranks = sorted((RANKS.index(card[0]) + 2 for card in cards), reverse=True)
    suits = [card[1] for card in cards]
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    by_count = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)
    if flush and straight_high:
        return (8, (straight_high,))
    if by_count[0][1] == 4:
        kicker = max(rank for rank in ranks if rank != by_count[0][0])
        return (7, (by_count[0][0], kicker))
    if by_count[0][1] == 3 and len(by_count) > 1 and by_count[1][1] == 2:
        return (6, (by_count[0][0], by_count[1][0]))
    if flush:
        return (5, tuple(ranks))
    if straight_high:
        return (4, (straight_high,))
    if by_count[0][1] == 3:
        kickers = tuple(rank for rank in ranks if rank != by_count[0][0])
        return (3, (by_count[0][0], *kickers))
    if by_count[0][1] == 2 and len(by_count) > 1 and by_count[1][1] == 2:
        pairs = tuple(sorted((by_count[0][0], by_count[1][0]), reverse=True))
        kicker = max(rank for rank in ranks if rank not in pairs)
        return (2, (*pairs, kicker))
    if by_count[0][1] == 2:
        pair = by_count[0][0]
        kickers = tuple(rank for rank in ranks if rank != pair)
        return (1, (pair, *kickers))
    return (0, tuple(ranks))


def _validate_cards(cards: list[str], min_count: int, max_count: int) -> list[str]:
    if not isinstance(cards, list):
        raise ValueError("cards must be a list of two-character card codes")
    if len(cards) < min_count or len(cards) > max_count:
        raise ValueError(f"expected {min_count} to {max_count} cards, got {len(cards)}")
    normalized = [_normalize_card(card) for card in cards]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate cards are not allowed: {cards!r}")
    return normalized


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


def split_side_pots(
    contributions: dict[str, int],
    folded: set[str],
    ranks: dict[str, PokerHandRank],
    odd_chip_order: list[str] | None = None,
) -> list[dict[str, object]]:

    levels = sorted({amount for amount in contributions.values() if amount > 0})
    payouts: list[dict[str, object]] = []
    previous = 0
    for level in levels:
        layer_size = level - previous
        contributors = [agent_id for agent_id, amount in contributions.items() if amount >= level]
        layer_pot = layer_size * len(contributors)
        eligible = [
            agent_id for agent_id in contributors if agent_id not in folded and agent_id in ranks
        ]
        if not contributors or layer_pot <= 0:
            previous = level
            continue
        if not eligible:
            share, remainder = divmod(layer_pot, len(contributors))
            for index, agent_id in enumerate(contributors):
                payouts.append(
                    {
                        "agent_id": agent_id,
                        "amount": share + (1 if index < remainder else 0),
                        "layer_level": level,
                        "hand_rank": None,
                        "reason": "uncontested_refund",
                    }
                )
            previous = level
            continue
        best = max(ranks[agent_id].score for agent_id in eligible)
        winners = [agent_id for agent_id in eligible if ranks[agent_id].score == best]
        if odd_chip_order:
            order_index = {agent_id: index for index, agent_id in enumerate(odd_chip_order)}
            winners.sort(key=lambda agent_id: order_index.get(agent_id, len(order_index)))
        share, remainder = divmod(layer_pot, len(winners))
        for index, agent_id in enumerate(winners):
            payouts.append(
                {
                    "agent_id": agent_id,
                    "amount": share + (1 if index < remainder else 0),
                    "layer_level": level,
                    "hand_rank": ranks[agent_id].class_name,
                    "reason": "showdown",
                }
            )
        previous = level
    return payouts


def _straight_high(ranks: list[int]) -> int:

    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for window in zip(unique, unique[1:], unique[2:], unique[3:], unique[4:], strict=False):
        if tuple(window) == tuple(range(window[0], window[0] - 5, -1)):
            return window[0]
    return 0
