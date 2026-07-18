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
import json
import math
import os
import re
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    dense: float | None = None
    sparse: float | None = None
    colbert: float | None = None


@dataclass(slots=True)
class SemanticScore:
    """Model-native semantic score components for one query/document pair."""

    combined: float
    dense: float | None
    sparse: float | None
    colbert: float | None


class EmbeddingBackend(Protocol):
    """Versioned batch embedding boundary used by factual memory."""

    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def audit_metadata(self) -> dict[str, object]: ...


class HashEmbeddingBackend:
    """Deterministic offline ablation backend; it is not a semantic model."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = max(16, int(dimensions))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, dimensions=self.dimensions)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]:
        query_vector = self.embed_query(query)
        return [
            SemanticScore(_dot(query_vector, vector), _dot(query_vector, vector), None, None)
            for vector in self.embed_documents(documents)
        ]

    def audit_metadata(self) -> dict[str, object]:
        return {
            "backend": "hash",
            "dimensions": self.dimensions,
            "semantic_model": False,
        }


class OpenAICompatibleEmbeddingBackend:
    """OpenAI-compatible /embeddings backend with a version-keyed persistent cache."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        base_url_env: str = "EMBEDDING_BASE_URL",
        api_key_env: str = "EMBEDDING_API_KEY",
        api_key_required: bool = False,
        timeout_seconds: float = 60.0,
        cache_path: str | Path | None = None,
        query_instruction: str = "",
    ) -> None:
        if not model or not revision:
            raise ValueError("真实 embedding backend 必须固定 model 和 revision")
        self.model = model
        self.revision = revision
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.api_key_required = api_key_required
        self.timeout_seconds = timeout_seconds
        self.cache_path = Path(cache_path) if cache_path else None
        self.query_instruction = query_instruction.strip()
        self.cache: dict[str, list[float]] = {}
        self.request_count = 0
        self.cache_hit_count = 0
        if self.cache_path and self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.cache = {
                    str(key): [float(value) for value in values]
                    for key, values in raw.items()
                    if isinstance(values, list)
                }

    def embed_query(self, text: str) -> list[float]:
        query = (
            f"Instruct: {self.query_instruction}\nQuery:{text}"
            if self.query_instruction
            else text
        )
        return self._embed_cached([query])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_cached(texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compatibility alias: untyped texts are encoded as documents."""

        return self.embed_documents(texts)

    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]:
        query_vector = self.embed_query(query)
        scores: list[SemanticScore] = []
        for vector in self.embed_documents(documents):
            dense = _dot(query_vector, vector)
            scores.append(SemanticScore(dense, dense, None, None))
        return scores

    def _embed_cached(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        missing = [text for text, key in zip(texts, keys, strict=True) if key not in self.cache]
        self.cache_hit_count += len(texts) - len(missing)
        if missing:
            vectors = self._request(missing)
            for text, vector in zip(missing, vectors, strict=True):
                self.cache[self._key(text)] = _normalize(vector)
            self._persist_cache()
        return [list(self.cache[key]) for key in keys]

    def audit_metadata(self) -> dict[str, object]:
        return {
            "backend": "openai_compatible",
            "model": self.model,
            "revision": self.revision,
            "semantic_model": True,
            "query_instruction": self.query_instruction,
            "query_instruction_sha256": hashlib.sha256(
                self.query_instruction.encode("utf-8")
            ).hexdigest(),
            "query_template": "Instruct: {instruction}\\nQuery:{query}",
            "document_instruction": None,
            "normalization": "l2_client",
            "output_dimensions": len(next(iter(self.cache.values()))) if self.cache else None,
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "cache_entries": len(self.cache),
            "cache_hit_count": self.cache_hit_count,
            "request_count": self.request_count,
        }

    def _request(self, texts: list[str]) -> list[list[float]]:
        base_url = os.environ.get(self.base_url_env)
        api_key = os.environ.get(self.api_key_env, "")
        if not base_url:
            raise RuntimeError(f"embedding backend 缺少环境变量 {self.base_url_env}")
        if self.api_key_required and not api_key:
            raise RuntimeError(f"embedding backend 缺少环境变量 {self.api_key_env}")
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            base_url.rstrip("/") + "/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        ordered = sorted(body["data"], key=lambda item: int(item["index"]))
        vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding 响应数量与输入数量不一致")
        self.request_count += 1
        return vectors

    def _key(self, text: str) -> str:
        payload = f"{self.model}\0{self.revision}\0{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _persist_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.cache_path)


class BgeM3HybridHttpBackend:
    """BGE-M3 native dense+sparse+ColBERT scoring over a dedicated local service."""

    QUERY_POLICY = "raw_symmetric_no_instruction"

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        base_url_env: str = "BGEM3_BASE_URL",
        api_key_env: str = "BGEM3_API_KEY",
        api_key_required: bool = False,
        timeout_seconds: float = 120.0,
        weights: list[float] | tuple[float, float, float] = (0.4, 0.2, 0.4),
    ) -> None:
        if not model or not revision:
            raise ValueError("BGE-M3 hybrid backend 必须固定 model 和 revision")
        parsed_weights = tuple(float(value) for value in weights)
        if len(parsed_weights) != 3 or any(value < 0 for value in parsed_weights):
            raise ValueError("BGE-M3 hybrid weights 必须是三个非负数")
        if sum(parsed_weights) <= 0:
            raise ValueError("BGE-M3 hybrid weights 之和必须大于零")
        self.model = model
        self.revision = revision
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.api_key_required = api_key_required
        self.timeout_seconds = timeout_seconds
        self.weights = parsed_weights
        self.request_count = 0
        self.scored_document_count = 0

    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]:
        if not documents:
            return []
        body = self._request(query, documents)
        if str(body.get("model", "")) != self.model:
            raise RuntimeError("BGE-M3 scoring service model identity mismatch")
        if str(body.get("revision", "")) != self.revision:
            raise RuntimeError("BGE-M3 scoring service revision identity mismatch")
        if str(body.get("query_policy", "")) != self.QUERY_POLICY:
            raise RuntimeError("BGE-M3 scoring service query policy mismatch")
        raw_scores = body.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(documents):
            raise RuntimeError("BGE-M3 scoring service response count mismatch")
        scores: list[SemanticScore] = []
        for item in raw_scores:
            if not isinstance(item, dict):
                raise RuntimeError("BGE-M3 scoring service returned a malformed score")
            scores.append(
                SemanticScore(
                    combined=float(item["combined"]),
                    dense=float(item["dense"]),
                    sparse=float(item["sparse"]),
                    colbert=float(item["colbert"]),
                )
            )
        self.request_count += 1
        self.scored_document_count += len(documents)
        return scores

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("BGE-M3 hybrid backend exposes scores, not dense-only query vectors")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("BGE-M3 hybrid backend exposes scores, not dense-only document vectors")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def audit_metadata(self) -> dict[str, object]:
        return {
            "backend": "bgem3_hybrid_http",
            "model": self.model,
            "revision": self.revision,
            "semantic_model": True,
            "query_instruction": None,
            "query_instruction_sha256": None,
            "query_template": "{query}",
            "document_instruction": None,
            "document_template": "{historical_state_query}\\n{fact_text}",
            "query_policy": self.QUERY_POLICY,
            "retrieval_modes": ["dense", "sparse", "colbert"],
            "hybrid_weights": {
                "dense": self.weights[0],
                "sparse": self.weights[1],
                "colbert": self.weights[2],
            },
            "normalization": "model_native",
            "base_url_env": self.base_url_env,
            "request_count": self.request_count,
            "scored_document_count": self.scored_document_count,
        }

    def _request(self, query: str, documents: list[str]) -> dict[str, object]:
        base_url = os.environ.get(self.base_url_env)
        api_key = os.environ.get(self.api_key_env, "")
        if not base_url:
            raise RuntimeError(f"BGE-M3 hybrid backend 缺少环境变量 {self.base_url_env}")
        if self.api_key_required and not api_key:
            raise RuntimeError(f"BGE-M3 hybrid backend 缺少环境变量 {self.api_key_env}")
        payload = json.dumps(
            {
                "query": query,
                "documents": documents,
                "weights": list(self.weights),
                "query_policy": self.QUERY_POLICY,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            base_url.rstrip("/") + "/score",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise RuntimeError("BGE-M3 scoring service returned a non-object response")
        return body


def build_embedding_backend(config: dict[str, object], agent_id: str) -> EmbeddingBackend:
    """Build a configured embedding backend without silently selecting a real model."""

    backend = str(config.get("embedding_backend", "hash"))
    if backend == "hash":
        return HashEmbeddingBackend(int(config.get("embedding_dimensions", 256)))
    if backend == "bgem3_hybrid_http":
        raw_weights = config.get("embedding_hybrid_weights", [0.4, 0.2, 0.4])
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError("embedding_hybrid_weights 必须是列表")
        return BgeM3HybridHttpBackend(
            model=str(config.get("embedding_model", "")),
            revision=str(config.get("embedding_revision", "")),
            base_url_env=str(config.get("embedding_base_url_env", "BGEM3_BASE_URL")),
            api_key_env=str(config.get("embedding_api_key_env", "BGEM3_API_KEY")),
            api_key_required=bool(config.get("embedding_api_key_required", False)),
            timeout_seconds=float(config.get("embedding_timeout_seconds", 120)),
            weights=[float(value) for value in raw_weights],
        )
    if backend != "openai_compatible":
        raise ValueError(f"未知 embedding backend：{backend}")
    cache_template = str(
        config.get("embedding_cache_path", "outputs/embedding_cache/{agent_id}.json")
    )
    return OpenAICompatibleEmbeddingBackend(
        model=str(config.get("embedding_model", "")),
        revision=str(config.get("embedding_revision", "")),
        base_url_env=str(config.get("embedding_base_url_env", "EMBEDDING_BASE_URL")),
        api_key_env=str(config.get("embedding_api_key_env", "EMBEDDING_API_KEY")),
        api_key_required=bool(config.get("embedding_api_key_required", False)),
        timeout_seconds=float(config.get("embedding_timeout_seconds", 60)),
        cache_path=cache_template.format(agent_id=agent_id),
        query_instruction=str(config.get("embedding_query_instruction", "")),
    )


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


def retrieval_text_for_backend(
    record: FactualMemoryRecord,
    backend: EmbeddingBackend,
) -> str:
    """Use a schema-aligned historical state prefix for BGE-M3 lexical matching."""

    fact_text = fact_retrieval_text(record)
    if isinstance(backend, BgeM3HybridHttpBackend):
        historical_query = str(record.source.get("retrieval_query", "")).strip()
        if historical_query:
            return f"{historical_query}\n{fact_text}"
    return fact_text


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
    embedding_backend: EmbeddingBackend | None = None,
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
    backend = embedding_backend or HashEmbeddingBackend(dimensions)
    documents = [retrieval_text_for_backend(record, backend) for record in candidates]
    semantic_scores = backend.score_documents(query_text, documents)
    scored: list[RetrievalScore] = []
    for index, record in enumerate(candidates):
        component = semantic_scores[index]
        semantic = component.combined
        if vec_lookup and record.record_id in vec_lookup:
            query_vec = backend.embed_query(query_text)
            semantic = _dot(query_vec, vec_lookup[record.record_id])
        salience = float(salience_fn(record.record_id) if salience_fn else 1.0)
        score = semantic + math.log(max(salience, 1e-6)) if salience_fn else semantic
        scored.append(
            RetrievalScore(
                record,
                score,
                semantic,
                0.0,
                salience,
                component.dense,
                component.sparse,
                component.colbert,
            )
        )
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
    embedding_backend: EmbeddingBackend | None = None,
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
    backend = embedding_backend or HashEmbeddingBackend(dimensions)
    semantic_scores = backend.score_documents(
        query_text, [retrieval_text_for_backend(record, backend) for record in records]
    )
    query_features = observation_features(observation)
    scored: list[RetrievalScore] = []
    for index, record in enumerate(records):
        component = semantic_scores[index]
        semantic = component.combined
        feature = jaccard_similarity(query_features, record.features)
        salience = float(salience_fn(record.record_id) if salience_fn else 1.0)
        base = semantic_weight * semantic + feature_weight * feature
        score = base + math.log(max(salience, 1e-6)) if salience_fn else base
        scored.append(
            RetrievalScore(
                record,
                score,
                semantic,
                feature,
                salience,
                component.dense,
                component.sparse,
                component.colbert,
            )
        )
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


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else vector
