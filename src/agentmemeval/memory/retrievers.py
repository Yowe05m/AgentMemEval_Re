"""
模块说明：本模块实现可离线测试的解释型事实检索器。
核心职责：从合法可见观察和事实记录中抽取特征，并按相似度排序。
输入与输出：输入观察或事实记录，输出特征列表和排序结果。
依赖边界：不依赖 embedding 服务，不依赖 LLM，不访问网络。
不负责：不写入记忆，不决定动作。
"""

from __future__ import annotations

import math
import re

from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord
from agentmemeval.core.poker_position import public_position_label
from agentmemeval.environment.decision_facts import build_decision_facts

RANK_GROUPS = {
    "low": set("23456"),
    "mid": set("789T"),
    "high": set("JQKA"),
}


def observation_features(observation: AgentObservation) -> list[str]:
    """
    功能：从观察中抽取可解释检索特征。
    参数：
        observation：合法可见观察。
    返回：特征字符串列表。
    副作用：无。
    异常：无。
    设计说明：特征仅来自当前可见信息，避免检索器引入隐藏信息。
    """

    decision_facts = build_decision_facts(observation)
    call_facts = dict(decision_facts["call"])
    draw_facts = dict(decision_facts["draw"])
    hero_draw = dict(draw_facts["hero_draw"])
    board_threat = dict(draw_facts["board_completion_threat"])
    legal_types = sorted(observation.legal_actions.types())
    features = [
        f"phase:{observation.phase}",
        f"players:{len(observation.players)}",
        f"position:{public_position_label(observation)}",
        f"active_opponents:{decision_facts['active_opponents']}",
        (
            "table_shape:heads_up"
            if int(decision_facts["multiway_players"]) == 2
            else "table_shape:multiway"
        ),
        f"effective_stack:{_effective_stack_bucket(int(decision_facts['effective_stack']))}",
        f"spr:{_spr_bucket(float(decision_facts['spr']))}",
        f"pot_odds:{_pot_odds_bucket(float(call_facts['required_equity']))}",
        f"made_hand:{_feature_slug(str(decision_facts['made_hand_class']))}",
        f"hero_draw:{_draw_label(hero_draw, applicable=bool(draw_facts['applicable']))}",
        f"board_threat:{_draw_label(board_threat, applicable=bool(draw_facts['applicable']))}",
        f"facing:{_facing_action(observation)}",
        f"legal_set:{','.join(legal_types) or 'none'}",
    ]
    features.extend(f"legal:{action_type}" for action_type in legal_types)
    features.append(f"to_call:{_bucket(observation.to_call)}")
    features.append(f"pot:{_bucket(observation.pot)}")
    for card in observation.hole_cards:
        features.append(f"hole_rank:{_rank_group(card)}")
        features.append(f"hole_suit:{card[1]}")
    if len(observation.hole_cards) == 2:
        if observation.hole_cards[0][0] == observation.hole_cards[1][0]:
            features.append("hole_pair")
        if observation.hole_cards[0][1] == observation.hole_cards[1][1]:
            features.append("hole_suited")
    for card in observation.community_cards:
        features.append(f"board_rank:{_rank_group(card)}")
    return features


def _draw_label(draw: dict[str, object], *, applicable: bool) -> str:
    """Return a compact hero/board draw category without treating nominal outs as equity."""

    if not applicable:
        return "not_applicable"
    flush_draw = bool(draw.get("flush_draw", False))
    straight_draw = bool(draw.get("straight_draw", False))
    if flush_draw and straight_draw:
        return "combo"
    if flush_draw:
        return "flush"
    if straight_draw:
        return "straight"
    return "none"


def _facing_action(observation: AgentObservation) -> str:
    """Describe the public wager currently faced, without inferring hidden intent."""

    if observation.to_call <= 0:
        return "none"
    for action in reversed(observation.action_history):
        if str(action.get("phase", observation.phase)) != observation.phase:
            continue
        actor = str(action.get("agent_id", action.get("player_id", "")))
        if actor == observation.agent_id:
            continue
        action_type = str(action.get("action_type", action.get("action", ""))).lower()
        if action_type == "raise":
            return "raise"
    return "blind" if observation.phase == "preflop" else "wager"


def _feature_slug(value: str) -> str:
    """Normalize evaluator labels into stable feature tokens."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown"


def _effective_stack_bucket(value: int) -> str:
    """Bucket visible contestable effective chips separately from pot-relative SPR."""

    if value <= 0:
        return "zero"
    if value <= 20:
        return "short"
    if value <= 50:
        return "medium"
    if value <= 100:
        return "deep"
    return "very_deep"


def _spr_bucket(value: float) -> str:
    """Bucket current-street stack-to-contestable-pot ratio."""

    if value <= 0:
        return "zero"
    if value <= 1:
        return "low_le_1"
    if value <= 3:
        return "medium_1_to_3"
    if value <= 6:
        return "high_3_to_6"
    return "very_high_gt_6"


def _pot_odds_bucket(required_equity: float) -> str:
    """Bucket deterministic call break-even equity (the visible pot-odds requirement)."""

    if required_equity <= 0:
        return "free"
    if required_equity <= 0.20:
        return "low_le_20pct"
    if required_equity <= 0.33:
        return "medium_20_to_33pct"
    if required_equity <= 0.50:
        return "high_33_to_50pct"
    return "extreme_gt_50pct"


def jaccard_similarity(left: list[str], right: list[str]) -> float:
    """
    功能：计算两个特征集合的 Jaccard 相似度。
    参数：
        left：左侧特征。
        right：右侧特征。
    返回：0 到 1 的相似度。
    副作用：无。
    异常：无。
    设计说明：简单、可解释、离线可测，后续可替换 embedding 检索器。
    """

    a = set(left)
    b = set(right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_k_records(
    observation: AgentObservation,
    records: list[FactualMemoryRecord],
    k: int,
) -> list[tuple[FactualMemoryRecord, float]]:
    """
    功能：按当前观察检索最相似的事实记录。
    参数：
        observation：查询观察。
        records：候选事实。
        k：返回数量。
    返回：记录与分数列表。
    副作用：无。
    异常：无。
    设计说明：排序稳定地使用 created_at 和 record_id 作为次级键，便于复现。
    """

    query = observation_features(observation)
    scored = [
        (record, jaccard_similarity(query, record.features))
        for record in records
    ]
    scored.sort(key=lambda item: (item[1], item[0].created_at, item[0].record_id), reverse=True)
    return scored[: max(0, k)]


def exposure_entropy(counts: list[int]) -> float:
    """
    功能：计算对手暴露次数的归一化熵。
    参数：
        counts：各对手暴露次数。
    返回：0 到 1 的熵，空输入为 0。
    副作用：无。
    异常：无。
    设计说明：换桌实验用它衡量对手多样性是否集中在少数人身上。
    """

    total = sum(counts)
    positive = [count for count in counts if count > 0]
    if total <= 0 or len(positive) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in positive)
    return entropy / math.log(len(positive))


def _bucket(value: int) -> str:
    """
    功能：把数值压成稳定桶。
    参数：
        value：筹码数。
    返回：桶名。
    副作用：无。
    异常：无。
    设计说明：检索只需要粗粒度相似，避免被绝对筹码噪声主导。
    """

    if value <= 0:
        return "zero"
    if value <= 2:
        return "small"
    if value <= 8:
        return "medium"
    return "large"


def _rank_group(card: str) -> str:
    """
    功能：把牌面 rank 映射到低中高分组。
    参数：
        card：牌面代码。
    返回：分组名。
    副作用：无。
    异常：无。
    设计说明：初版检索器重在可解释，不追求精细牌力 embedding。
    """

    rank = card[0]
    for name, group in RANK_GROUPS.items():
        if rank in group:
            return name
    return "unknown"
