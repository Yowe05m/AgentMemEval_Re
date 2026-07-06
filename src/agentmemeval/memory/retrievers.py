"""
模块说明：本模块实现可离线测试的解释型事实检索器。
核心职责：从合法可见观察和事实记录中抽取特征，并按相似度排序。
输入与输出：输入观察或事实记录，输出特征列表和排序结果。
依赖边界：不依赖 embedding 服务，不依赖 LLM，不访问网络。
不负责：不写入记忆，不决定动作。
"""

from __future__ import annotations

import math

from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord

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

    features = [f"phase:{observation.phase}", f"players:{len(observation.players)}"]
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
