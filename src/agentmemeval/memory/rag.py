"""
模块说明：本模块提供离线可复现的 RAG 检索工具。
核心职责：构造论文风格 retrieval query，生成本地 hash embedding，
并按语义、结构特征和显著性混合排序。
输入与输出：输入观察或事实记录，输出检索查询、向量和排序结果。
依赖边界：只依赖标准库，不强制下载 sentence_transformers 或模型文件。
不负责：不写入记忆，不调用在线 LLM，不决定动作。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord
from agentmemeval.memory.retrievers import jaccard_similarity, observation_features

TOKEN_RE = re.compile(r"[A-Za-z0-9_:\-\[\]\+\.]+|[\u4e00-\u9fff]+")


@dataclass(slots=True)
class RetrievalScore:
    """
    功能：保存一条候选事实的混合检索分数。
    参数：
        record：事实记录。
        score：最终排序分数。
        semantic：hash embedding 余弦相似度。
        feature：可解释特征 Jaccard 相似度。
        salience：显著性权重。
    返回：检索分数对象。
    副作用：无。
    异常：无。
    设计说明：把分数拆开保存，方便报告解释 RAG 命中原因。
    """

    record: FactualMemoryRecord
    score: float
    semantic: float
    feature: float
    salience: float


def build_retrieval_query(observation: AgentObservation) -> str:
    """
    功能：构造接近原版 `build_retrieval_query` 的状态查询短串。
    参数：
        observation：当前合法可见观察。
    返回：检索查询文本。
    副作用：无。
    异常：无。
    设计说明：只包含 phase、hole、board、pot、to_call、seat 等状态特征，不注入策略词。
    """

    return (
        f"phase={observation.phase} "
        f"hole={observation.hole_cards} "
        f"board={observation.community_cards} "
        f"pot={observation.pot} "
        f"to_call={observation.to_call} "
        f"seat={observation.seat} "
        f"players={len(observation.players)}"
    )


def fact_retrieval_text(record: FactualMemoryRecord) -> str:
    """
    功能：把事实记录渲染成 embedding/LLM 友好的文本。
    参数：
        record：事实记录。
    返回：文本。
    副作用：无。
    异常：无。
    设计说明：优先使用 source 中的原始 fact_text，老快照则退回 state/action/reward 字段。
    """

    source_text = str(record.source.get("fact_text", "") or "").strip()
    if source_text:
        return source_text
    return (
        f"{record.state_summary}\n"
        f"我的决策序列: {record.action_summary}\n"
        f"hand_outcome: final_reward={record.final_reward:+d}"
    )


def embed_text(text: str, dimensions: int = 256) -> list[float]:
    """
    功能：使用稳定 hash trick 生成归一化稀疏向量。
    参数：
        text：输入文本。
        dimensions：向量维度。
    返回：归一化向量。
    副作用：无。
    异常：无。
    设计说明：它不是语义模型，但保留 embedding 检索接口，可在无网络环境下验证 RAG 管线。
    """

    dims = max(16, int(dimensions))
    vector = [0.0] * dims
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Iterable[str], dimensions: int = 256) -> list[list[float]]:
    """
    功能：批量生成 hash embedding。
    参数：
        texts：文本序列。
        dimensions：向量维度。
    返回：向量列表。
    副作用：无。
    异常：无。
    设计说明：API 形态与原版 `embed(texts)` 接近，便于后续替换为真实 embedding。
    """

    return [embed_text(text, dimensions=dimensions) for text in texts]


def topk_by_similarity(
    query_text: str,
    candidates: list[FactualMemoryRecord],
    k: int,
    vec_lookup: dict[str, list[float]] | None = None,
    salience_fn: Callable[[str], float] | None = None,
    dimensions: int = 256,
) -> list[RetrievalScore]:
    """
    功能：按 embedding 相似度和可选显著性返回 top-k 事实。
    参数：
        query_text：检索查询文本。
        candidates：候选事实。
        k：返回数量。
        vec_lookup：可选已缓存向量。
        salience_fn：可选显著性函数。
        dimensions：hash embedding 维度。
    返回：RetrievalScore 列表。
    副作用：无。
    异常：无。
    设计说明：对应原版 `topk_by_similarity`，但允许离线 on-the-fly embedding。
    """

    if not candidates or k <= 0:
        return []
    query_vec = embed_text(query_text, dimensions=dimensions)
    scored: list[RetrievalScore] = []
    for record in candidates:
        record_vec = (vec_lookup or {}).get(record.record_id)
        if record_vec is None:
            record_vec = embed_text(fact_retrieval_text(record), dimensions=dimensions)
        semantic = _dot(query_vec, record_vec)
        salience = float(salience_fn(record.record_id) if salience_fn else 1.0)
        score = semantic + math.log(max(salience, 1e-6)) if salience_fn else semantic
        scored.append(RetrievalScore(record, score, semantic, 0.0, salience))
    scored.sort(
        key=lambda item: (item.score, item.record.created_at, item.record.record_id),
        reverse=True,
    )
    return scored[:k]


def hybrid_top_k_records(
    observation: AgentObservation,
    records: list[FactualMemoryRecord],
    k: int,
    semantic_weight: float = 0.65,
    feature_weight: float = 0.35,
    salience_fn: Callable[[str], float] | None = None,
    dimensions: int = 256,
) -> list[RetrievalScore]:
    """
    功能：使用语义 hash embedding + 结构特征 Jaccard 的混合 RAG 排序。
    参数：
        observation：当前观察。
        records：候选事实。
        k：返回数量。
        semantic_weight：语义分数权重。
        feature_weight：结构特征分数权重。
        salience_fn：可选显著性函数。
        dimensions：hash embedding 维度。
    返回：RetrievalScore 列表。
    副作用：无。
    异常：无。
    设计说明：比纯 Jaccard 更接近原版 RAG，同时保持完全离线可测。
    """

    if not records or k <= 0:
        return []
    query_text = build_retrieval_query(observation)
    query_vec = embed_text(query_text, dimensions=dimensions)
    query_features = observation_features(observation)
    scored: list[RetrievalScore] = []
    for record in records:
        semantic = _dot(query_vec, embed_text(fact_retrieval_text(record), dimensions=dimensions))
        feature = jaccard_similarity(query_features, record.features)
        salience = float(salience_fn(record.record_id) if salience_fn else 1.0)
        base = semantic_weight * semantic + feature_weight * feature
        score = base + math.log(max(salience, 1e-6)) if salience_fn else base
        scored.append(RetrievalScore(record, score, semantic, feature, salience))
    scored.sort(
        key=lambda item: (item.score, item.record.created_at, item.record.record_id),
        reverse=True,
    )
    return scored[:k]


def _tokens(text: str) -> list[str]:
    """
    功能：把文本拆为稳定 token。
    参数：
        text：输入文本。
    返回：token 列表。
    副作用：无。
    异常：无。
    设计说明：保留中文连续片段和结构化 key=value 片段，适合本项目日志文本。
    """

    return [item.lower() for item in TOKEN_RE.findall(text) if item.strip()]


def _dot(left: list[float], right: list[float]) -> float:
    """
    功能：计算两个归一化向量的点积。
    参数：
        left：左向量。
        right：右向量。
    返回：相似度。
    副作用：无。
    异常：无。
    设计说明：维度不一致时按较短长度比较，增强老快照兼容性。
    """

    return sum(a * b for a, b in zip(left, right, strict=False))
