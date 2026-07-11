"""
模块说明：本模块实现德州扑克牌力评估与边池分配。
核心职责：评价最佳五张牌、输出牌型名称，并按 side-pot 分层计算摊牌派奖。
输入与输出：输入牌面、贡献额和弃牌集合，输出可比较牌力与派奖记录。
依赖边界：只依赖标准库；未来可在同一接口下接入 treys。
不负责：不推进牌局街道，不校验动作合法性。
"""

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
    """
    功能：保存一手牌的可比较评估结果。
    参数：
        score：类别与踢脚列表，越大越强。
        class_name：英文牌型名称，与 treys class_to_string 口径接近。
        best_cards：最佳五张牌。
    返回：牌力对象。
    副作用：无。
    异常：无。
    设计说明：使用 tuple 比较，避免外部知道具体类别编码。
    """

    score: tuple[int, tuple[int, ...]]
    class_name: str
    best_cards: tuple[str, ...]


def evaluate_best(cards: list[str]) -> PokerHandRank:
    """
    功能：从最多七张牌中选出最佳五张并给出牌型。
    参数：
        cards：牌面代码，如 As、Td。
    返回：PokerHandRank。
    副作用：无。
    异常：牌少于五张时抛出 ValueError。
    设计说明：作为本地默认 evaluator，保留原版 treys 的牌型名称能力。
    """

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
    """
    功能：评价五张牌。
    参数：
        cards：五张牌。
    返回：类别和踢脚元组，越大越强。
    副作用：无。
    异常：牌数不是五张时抛出 ValueError。
    设计说明：覆盖标准德州扑克九类牌型，含 A2345 轮子顺。
    """

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
) -> list[dict[str, object]]:
    """
    功能：按 side-pot 分层计算派奖。
    参数：
        contributions：每名玩家本手累计投入。
        folded：已弃牌玩家集合。
        ranks：未弃牌玩家牌力。
    返回：派奖记录，每条含 agent_id、amount、layer_level 和 hand_rank。
    副作用：无。
    异常：无。
    设计说明：迁入原版 controller 的分层底池语义，支持短码 all-in。
    """

    levels = sorted({amount for amount in contributions.values() if amount > 0})
    payouts: list[dict[str, object]] = []
    previous = 0
    for level in levels:
        layer_size = level - previous
        contributors = [agent_id for agent_id, amount in contributions.items() if amount >= level]
        layer_pot = layer_size * len(contributors)
        eligible = [
            agent_id
            for agent_id in contributors
            if agent_id not in folded and agent_id in ranks
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
    """
    功能：判断顺子并返回最高张。
    参数：
        ranks：从大到小的 rank 数值列表。
    返回：顺子高张，非顺为 0。
    副作用：无。
    异常：无。
    设计说明：A2345 作为 5-high straight 处理。
    """

    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for window in zip(unique, unique[1:], unique[2:], unique[3:], unique[4:], strict=False):
        if tuple(window) == tuple(range(window[0], window[0] - 5, -1)):
            return window[0]
    return 0
