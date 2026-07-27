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

from agentmemeval.core.bgem3_contract import (
    BGE_M3_DENSE_NORM_TOLERANCE,
    BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    BGE_M3_NORMALIZATION_IDENTITY,
    output_contract_identity,
    validate_output_contract_metadata,
)
from agentmemeval.core.domain import AgentObservation, FactualMemoryRecord
from agentmemeval.core.poker_position import (
    active_public_players,
    public_position_label,
    relative_active_seat_offset,
)
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
    retrieval_unit: str = "hand_terminal_v1"
    matched_decision_index: int | None = None
    matched_phase: str | None = None


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
        output_dimensions: int | None = None,
    ) -> None:
        if not model or not revision:
            raise ValueError("真实 embedding backend 必须固定 model 和 revision")
        if not base_url_env.strip() or not api_key_env.strip():
            raise ValueError("embedding environment variable names must be non-empty")
        if type(api_key_required) is not bool:
            raise ValueError("embedding api_key_required must be a boolean")
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("embedding timeout_seconds must be a positive finite number")
        self.model = model
        self.revision = revision
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.api_key_required = api_key_required
        self.timeout_seconds = float(timeout_seconds)
        self.cache_path = Path(cache_path) if cache_path else None
        self.query_instruction = query_instruction.strip()
        if output_dimensions is not None and (
            type(output_dimensions) is not int or output_dimensions < 1
        ):
            raise ValueError("embedding_output_dimensions 必须为正整数或 None")
        self.expected_output_dimensions = output_dimensions
        self.observed_output_dimensions: int | None = None
        self.cache: dict[str, list[float]] = {}
        self.request_count = 0
        self.cache_hit_count = 0
        if self.cache_path and self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, values in raw.items():
                    if not isinstance(values, list):
                        raise ValueError("embedding cache vector must be a list")
                    self.cache[str(key)] = self._validated_vector(
                        values,
                        source="cache",
                    )

    def embed_query(self, text: str) -> list[float]:
        query = (
            f"Instruct: {self.query_instruction}\nQuery:{text}" if self.query_instruction else text
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
            "configured_output_dimensions": self.expected_output_dimensions,
            "observed_output_dimensions": self.observed_output_dimensions,
            "output_dimensions": self.observed_output_dimensions,
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
        raw_data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(raw_data, list):
            raise RuntimeError("embedding 响应 data 不是列表")
        if not all(isinstance(item, dict) for item in raw_data):
            raise RuntimeError("embedding 响应项不是对象")
        try:
            ordered = sorted(raw_data, key=lambda item: int(item["index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("embedding 响应 index 非法") from exc
        if [item.get("index") for item in ordered] != list(range(len(texts))):
            raise RuntimeError("embedding 响应 index 不连续或重复")
        vectors = [
            self._validated_vector(item.get("embedding"), source="response") for item in ordered
        ]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding 响应数量与输入数量不一致")
        self.request_count += 1
        return vectors

    def _validated_vector(self, raw: object, *, source: str) -> list[float]:
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"embedding {source} vector must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
            raise RuntimeError(f"embedding {source} vector contains a non-number")
        vector = [float(value) for value in raw]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"embedding {source} vector contains a non-finite value")
        dimension = len(vector)
        expected = self.expected_output_dimensions or self.observed_output_dimensions
        if expected is not None and dimension != expected:
            raise RuntimeError(f"embedding {source} dimension mismatch: {dimension} != {expected}")
        if math.sqrt(sum(value * value for value in vector)) <= 0:
            raise RuntimeError(f"embedding {source} vector has zero norm")
        self.observed_output_dimensions = dimension
        return vector

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
    SERVICE_MAX_DOCUMENTS = 1000
    SCORE_REL_TOLERANCE = 1e-12
    SCORE_ABS_TOLERANCE = 1e-12

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        weights_hash: str,
        tokenizer_revision: str,
        cache_schema_version: str,
        cache_namespace: str,
        base_url_env: str = "BGEM3_BASE_URL",
        api_key_env: str = "BGEM3_API_KEY",
        api_key_required: bool = False,
        timeout_seconds: float = 120.0,
        weights: list[float] | tuple[float, float, float] = (0.4, 0.2, 0.4),
        candidate_depth: int = 1000,
        colbert_rerank_depth: int = 1000,
        final_top_k_policy: str = "agent_roster_top_k",
        service_startup_parameters: dict[str, object] | None = None,
        expected_runtime_identity: str | None = None,
        expected_server_source_sha256: str | None = None,
        expected_dense_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
        normalization_identity: str = BGE_M3_NORMALIZATION_IDENTITY,
        dense_norm_tolerance: float = BGE_M3_DENSE_NORM_TOLERANCE,
    ) -> None:
        if not all(
            value.strip()
            for value in (
                model,
                revision,
                weights_hash,
                tokenizer_revision,
                cache_schema_version,
                cache_namespace,
            )
        ):
            raise ValueError(
                "BGE-M3 hybrid backend 必须固定 model/revision/weights/tokenizer/cache schema"
            )
        if not base_url_env.strip() or not api_key_env.strip():
            raise ValueError("BGE-M3 environment variable names must be non-empty")
        if type(api_key_required) is not bool:
            raise ValueError("BGE-M3 api_key_required must be a boolean")
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("BGE-M3 timeout_seconds must be a positive finite number")
        if len(weights) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in weights
        ):
            raise ValueError("BGE-M3 hybrid weights 必须是三个非负数")
        parsed_weights = tuple(float(value) for value in weights)
        if any(not math.isfinite(value) or value < 0 for value in parsed_weights):
            raise ValueError("BGE-M3 hybrid weights 必须是三个非负数")
        if sum(parsed_weights) <= 0:
            raise ValueError("BGE-M3 hybrid weights 之和必须大于零")
        if parsed_weights != (0.4, 0.2, 0.4):
            raise ValueError("BGE-M3 hybrid weights 必须冻结为 (0.4, 0.2, 0.4)")
        if type(candidate_depth) is not int or type(colbert_rerank_depth) is not int:
            raise ValueError("BGE-M3 candidate/rerank depth 必须为正整数")
        if candidate_depth < 1 or colbert_rerank_depth < 1:
            raise ValueError("BGE-M3 candidate/rerank depth 必须为正整数")
        if candidate_depth > self.SERVICE_MAX_DOCUMENTS:
            raise ValueError("BGE-M3 candidate_depth 超过当前 service 的 1000-document 上限")
        if colbert_rerank_depth > candidate_depth:
            raise ValueError("BGE-M3 colbert_rerank_depth 不能超过 candidate_depth")
        if colbert_rerank_depth != candidate_depth:
            raise ValueError(
                "当前 BGE-M3 service 对全部候选计算 ColBERT，"
                "colbert_rerank_depth 必须等于 candidate_depth"
            )
        if final_top_k_policy != "agent_roster_top_k":
            raise ValueError("当前 BGE-M3 backend 仅实现 agent_roster_top_k")
        parsed_weights_hash = _optional_sha256(weights_hash, "weights_hash")
        if parsed_weights_hash is None:
            raise ValueError("BGE-M3 weights_hash must be a 64-character SHA-256")
        self.model = model
        self.revision = revision
        self.weights_hash = parsed_weights_hash
        self.tokenizer_revision = tokenizer_revision
        self.cache_schema_version = cache_schema_version
        self.cache_namespace = cache_namespace
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.api_key_required = api_key_required
        self.timeout_seconds = float(timeout_seconds)
        self.weights = parsed_weights
        self.candidate_depth = int(candidate_depth)
        self.colbert_rerank_depth = int(colbert_rerank_depth)
        self.final_top_k_policy = final_top_k_policy
        self.expected_runtime_identity = _optional_nonempty_identity(
            expected_runtime_identity,
            "expected_runtime_identity",
        )
        self.expected_server_source_sha256 = _optional_sha256(
            expected_server_source_sha256,
            "expected_server_source_sha256",
        )
        if (
            type(expected_dense_dimensions) is not int
            or expected_dense_dimensions != BGE_M3_EXPECTED_DENSE_DIMENSIONS
        ):
            raise ValueError(
                "BAAI/bge-m3 expected dense dimensions must be frozen at "
                f"{BGE_M3_EXPECTED_DENSE_DIMENSIONS}"
            )
        if normalization_identity != BGE_M3_NORMALIZATION_IDENTITY:
            raise ValueError(
                "BGE-M3 normalization identity must be frozen at "
                f"{BGE_M3_NORMALIZATION_IDENTITY}"
            )
        if (
            isinstance(dense_norm_tolerance, bool)
            or not isinstance(dense_norm_tolerance, (int, float))
            or not math.isfinite(float(dense_norm_tolerance))
            or not math.isclose(
                float(dense_norm_tolerance),
                BGE_M3_DENSE_NORM_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "BGE-M3 dense norm tolerance must be frozen at "
                f"{BGE_M3_DENSE_NORM_TOLERANCE}"
            )
        self.expected_dense_dimensions = expected_dense_dimensions
        self.normalization_identity = normalization_identity
        self.dense_norm_tolerance = float(dense_norm_tolerance)
        self.output_contract = output_contract_identity(
            expected_dense_dimensions=self.expected_dense_dimensions,
            normalization_identity=self.normalization_identity,
            dense_norm_tolerance=self.dense_norm_tolerance,
        )
        self.observed_runtime_identity: str | None = None
        self.observed_runtime_source_root: str | None = None
        self.observed_server_source_sha256: str | None = None
        self.observed_output_contract_identity: dict[str, object] | None = None
        self.observed_dense_norm_range: dict[str, float] | None = None
        self.observed_validated_vector_count: int | None = None
        if service_startup_parameters is not None and not isinstance(
            service_startup_parameters, dict
        ):
            raise ValueError("BGE-M3 service_startup_parameters 必须是映射")
        self.service_startup_parameters = (
            json.loads(
                json.dumps(
                    service_startup_parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if service_startup_parameters is not None
            else None
        )
        startup_json = json.dumps(
            self.service_startup_parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.service_startup_parameters_sha256 = hashlib.sha256(
            startup_json.encode("utf-8")
        ).hexdigest()
        self.request_count = 0
        self.scored_document_count = 0

    def score_documents(self, query: str, documents: list[str]) -> list[SemanticScore]:
        if not documents:
            return []
        if len(documents) > self.candidate_depth:
            raise RuntimeError(
                "BGE-M3 candidate count exceeds frozen candidate_depth "
                f"({len(documents)} > {self.candidate_depth})"
            )
        body = self._request(query, documents)
        if str(body.get("model", "")) != self.model:
            raise RuntimeError("BGE-M3 scoring service model identity mismatch")
        if str(body.get("revision", "")) != self.revision:
            raise RuntimeError("BGE-M3 scoring service revision identity mismatch")
        if str(body.get("weights_hash", "")) != self.weights_hash:
            raise RuntimeError("BGE-M3 scoring service weights identity mismatch")
        if str(body.get("tokenizer_revision", "")) != self.tokenizer_revision:
            raise RuntimeError("BGE-M3 scoring service tokenizer identity mismatch")
        if str(body.get("cache_schema_version", "")) != self.cache_schema_version:
            raise RuntimeError("BGE-M3 scoring service cache schema mismatch")
        if str(body.get("cache_namespace", "")) != self.cache_namespace:
            raise RuntimeError("BGE-M3 scoring service cache namespace mismatch")
        if str(body.get("query_policy", "")) != self.QUERY_POLICY:
            raise RuntimeError("BGE-M3 scoring service query policy mismatch")
        runtime_identity = _required_response_identity(
            body.get("runtime_identity"), "runtime_identity"
        )
        runtime_source_root = _required_response_identity(
            body.get("runtime_source_root"), "runtime_source_root"
        )
        server_source_sha256 = _required_response_sha256(
            body.get("server_source_sha256"), "server_source_sha256"
        )
        if (
            self.expected_runtime_identity is not None
            and runtime_identity != self.expected_runtime_identity
        ):
            raise RuntimeError("BGE-M3 scoring service runtime identity mismatch")
        if (
            self.expected_server_source_sha256 is not None
            and server_source_sha256 != self.expected_server_source_sha256
        ):
            raise RuntimeError("BGE-M3 scoring service source SHA-256 mismatch")
        observed_runtime = (
            self.observed_runtime_identity,
            self.observed_runtime_source_root,
            self.observed_server_source_sha256,
        )
        response_runtime = (
            runtime_identity,
            runtime_source_root,
            server_source_sha256,
        )
        if observed_runtime != (None, None, None) and observed_runtime != response_runtime:
            raise RuntimeError("BGE-M3 scoring service runtime identity drifted")
        try:
            response_output_contract = validate_output_contract_metadata(
                body.get("output_contract"),
                expected_dense_dimensions=self.expected_dense_dimensions,
                normalization_identity=self.normalization_identity,
                dense_norm_tolerance=self.dense_norm_tolerance,
            )
        except ValueError as exc:
            raise RuntimeError("BGE-M3 scoring service output contract is invalid") from exc
        response_output_identity = {
            key: response_output_contract[key]
            for key in (
                "schema_version",
                "expected_dense_dimensions",
                "observed_dense_dimensions",
                "normalization_identity",
                "dense_norm_tolerance",
                "observed_colbert_dimensions",
                "sparse_vector_contract",
                "colbert_vector_contract",
            )
        }
        if (
            self.observed_output_contract_identity is not None
            and response_output_identity != self.observed_output_contract_identity
        ):
            raise RuntimeError("BGE-M3 scoring service output contract identity drifted")
        response_validated_count = int(response_output_contract["validated_vector_count"])
        if (
            self.observed_validated_vector_count is not None
            and response_validated_count < self.observed_validated_vector_count
        ):
            raise RuntimeError("BGE-M3 scoring service output contract counter regressed")
        expected_weights = {
            "dense": self.weights[0],
            "sparse": self.weights[1],
            "colbert": self.weights[2],
        }
        response_weights = body.get("weights")
        if not isinstance(response_weights, dict) or set(response_weights) != set(expected_weights):
            raise RuntimeError("BGE-M3 scoring service hybrid weights mismatch")
        parsed_response_weights: dict[str, float] = {}
        for name, expected in expected_weights.items():
            raw_value = response_weights[name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise RuntimeError("BGE-M3 scoring service hybrid weights mismatch")
            value = float(raw_value)
            if not math.isfinite(value) or not math.isclose(
                value,
                expected,
                rel_tol=self.SCORE_REL_TOLERANCE,
                abs_tol=self.SCORE_ABS_TOLERANCE,
            ):
                raise RuntimeError("BGE-M3 scoring service hybrid weights mismatch")
            parsed_response_weights[name] = value
        raw_scores = body.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(documents):
            raise RuntimeError("BGE-M3 scoring service response count mismatch")
        scores: list[SemanticScore] = []
        for item in raw_scores:
            if not isinstance(item, dict):
                raise RuntimeError("BGE-M3 scoring service returned a malformed score")
            if set(item) != {"combined", "dense", "sparse", "colbert"}:
                raise RuntimeError("BGE-M3 scoring service returned a malformed score")
            values: dict[str, float] = {}
            for name in ("combined", "dense", "sparse", "colbert"):
                raw_value = item[name]
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise RuntimeError("BGE-M3 scoring service returned a malformed score")
                values[name] = float(raw_value)
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError("BGE-M3 scoring service returned a non-finite score")
            weight_sum = sum(parsed_response_weights.values())
            recomputed = (
                parsed_response_weights["dense"] * values["dense"]
                + parsed_response_weights["sparse"] * values["sparse"]
                + parsed_response_weights["colbert"] * values["colbert"]
            ) / weight_sum
            if not math.isfinite(recomputed) or not math.isclose(
                values["combined"],
                recomputed,
                rel_tol=self.SCORE_REL_TOLERANCE,
                abs_tol=self.SCORE_ABS_TOLERANCE,
            ):
                raise RuntimeError("BGE-M3 scoring service combined score drifted from components")
            scores.append(
                SemanticScore(
                    # Never trust the service's aggregate as the ranking value.
                    # The echoed value is only a contract cross-check above.
                    combined=recomputed,
                    dense=values["dense"],
                    sparse=values["sparse"],
                    colbert=values["colbert"],
                )
            )
        self.request_count += 1
        self.scored_document_count += len(documents)
        self.observed_runtime_identity = runtime_identity
        self.observed_runtime_source_root = runtime_source_root
        self.observed_server_source_sha256 = server_source_sha256
        self.observed_output_contract_identity = response_output_identity
        self.observed_dense_norm_range = {
            "min": float(response_output_contract["observed_dense_norm_min"]),
            "max": float(response_output_contract["observed_dense_norm_max"]),
        }
        self.observed_validated_vector_count = response_validated_count
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
            "weights_hash": self.weights_hash,
            "tokenizer_revision": self.tokenizer_revision,
            "semantic_model": True,
            "query_instruction": None,
            "query_instruction_sha256": None,
            "query_template": "{query}",
            "document_instruction": None,
            "document_template": "{historical_state_query}\\n{fact_text}",
            "document_template_scope": "hand_terminal_v1",
            "decision_point_document_template": "{decision_state_query}",
            "document_template_selection": "retrieval_unit",
            "decision_point_batching": "full_coverage_candidate_depth_chunks",
            "query_policy": self.QUERY_POLICY,
            "retrieval_modes": ["dense", "sparse", "colbert"],
            "hybrid_weights": {
                "dense": self.weights[0],
                "sparse": self.weights[1],
                "colbert": self.weights[2],
            },
            "candidate_depth": self.candidate_depth,
            "colbert_rerank_depth": self.colbert_rerank_depth,
            "final_top_k_policy": self.final_top_k_policy,
            "cache_schema_version": self.cache_schema_version,
            "cache_namespace": self.cache_namespace,
            "normalization": self.normalization_identity,
            "output_contract": dict(self.output_contract),
            "base_url_env": self.base_url_env,
            "service_startup_parameters": self.service_startup_parameters,
            "service_startup_parameters_sha256": (self.service_startup_parameters_sha256),
            "expected_runtime_identity": self.expected_runtime_identity,
            "expected_server_source_sha256": self.expected_server_source_sha256,
            "observed_runtime_identity": self.observed_runtime_identity,
            "observed_runtime_source_root": self.observed_runtime_source_root,
            "observed_server_source_sha256": self.observed_server_source_sha256,
            "observed_output_contract_identity": (
                None
                if self.observed_output_contract_identity is None
                else dict(self.observed_output_contract_identity)
            ),
            "observed_dense_norm_range": (
                None
                if self.observed_dense_norm_range is None
                else dict(self.observed_dense_norm_range)
            ),
            "observed_validated_vector_count": self.observed_validated_vector_count,
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
                "cache_namespace": self.cache_namespace,
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


_EMBEDDING_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "hash": (
        "backend",
        "dimensions",
        "semantic_model",
    ),
    "openai_compatible": (
        "backend",
        "model",
        "revision",
        "semantic_model",
        "query_instruction",
        "query_instruction_sha256",
        "query_template",
        "document_instruction",
        "normalization",
        "configured_output_dimensions",
    ),
    "bgem3_hybrid_http": (
        "backend",
        "model",
        "revision",
        "weights_hash",
        "tokenizer_revision",
        "semantic_model",
        "query_instruction",
        "query_instruction_sha256",
        "query_template",
        "document_instruction",
        "document_template",
        "document_template_scope",
        "decision_point_document_template",
        "document_template_selection",
        "decision_point_batching",
        "query_policy",
        "retrieval_modes",
        "hybrid_weights",
        "candidate_depth",
        "colbert_rerank_depth",
        "final_top_k_policy",
        "cache_schema_version",
        "normalization",
        "output_contract",
        "base_url_env",
        "service_startup_parameters",
        "service_startup_parameters_sha256",
        "expected_runtime_identity",
        "expected_server_source_sha256",
    ),
}

_EMBEDDING_RUNTIME_ISOLATION_FIELDS: dict[str, tuple[str, ...]] = {
    "hash": ("backend",),
    "openai_compatible": (
        "backend",
        "cache_path",
        "observed_output_dimensions",
    ),
    "bgem3_hybrid_http": (
        "backend",
        "cache_namespace",
        "base_url_env",
        "observed_runtime_identity",
        "observed_runtime_source_root",
        "observed_server_source_sha256",
        "observed_output_contract_identity",
        "observed_dense_norm_range",
        "observed_validated_vector_count",
    ),
}


def embedding_identity_from_audit(metadata: object) -> dict[str, object]:
    """Extract semantic identity, excluding branch-local cache isolation fields."""

    if not isinstance(metadata, dict):
        raise ValueError("embedding audit metadata must be an object")
    backend = metadata.get("backend")
    if not isinstance(backend, str) or backend not in _EMBEDDING_IDENTITY_FIELDS:
        raise ValueError("embedding audit metadata has an unsupported backend")
    fields = _EMBEDDING_IDENTITY_FIELDS[backend]
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(
            "embedding audit metadata is missing identity fields: " + ", ".join(missing)
        )
    return {field: metadata[field] for field in fields}


def embedding_runtime_isolation_from_audit(metadata: object) -> dict[str, object]:
    """Extract runtime cache/service routing for audit, never semantic equality."""

    if not isinstance(metadata, dict):
        raise ValueError("embedding audit metadata must be an object")
    backend = metadata.get("backend")
    if not isinstance(backend, str) or backend not in _EMBEDDING_RUNTIME_ISOLATION_FIELDS:
        raise ValueError("embedding audit metadata has an unsupported backend")
    fields = _EMBEDDING_RUNTIME_ISOLATION_FIELDS[backend]
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(
            "embedding audit metadata is missing runtime isolation fields: " + ", ".join(missing)
        )
    return {field: metadata[field] for field in fields}


def build_embedding_backend(config: dict[str, object], agent_id: str) -> EmbeddingBackend:
    """Build a configured embedding backend without silently selecting a real model."""

    backend = str(config.get("embedding_backend", "hash"))
    if backend == "hash":
        return HashEmbeddingBackend(int(config.get("embedding_dimensions", 256)))
    if backend == "bgem3_hybrid_http":
        raw_weights = config.get("embedding_hybrid_weights", [0.4, 0.2, 0.4])
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError("embedding_hybrid_weights 必须是列表")
        raw_startup = config.get("embedding_service_startup_parameters")
        startup = dict(raw_startup) if isinstance(raw_startup, dict) else None
        return BgeM3HybridHttpBackend(
            model=str(config.get("embedding_model", "")),
            revision=str(config.get("embedding_revision", "")),
            weights_hash=str(config.get("embedding_weights_hash", "")),
            tokenizer_revision=str(config.get("embedding_tokenizer_revision", "")),
            cache_schema_version=str(config.get("embedding_cache_schema_version", "")),
            cache_namespace=str(config.get("embedding_cache_path", "")).format(agent_id=agent_id),
            base_url_env=str(config.get("embedding_base_url_env", "BGEM3_BASE_URL")),
            api_key_env=str(config.get("embedding_api_key_env", "BGEM3_API_KEY")),
            api_key_required=config.get("embedding_api_key_required", False),  # type: ignore[arg-type]
            timeout_seconds=config.get("embedding_timeout_seconds", 120),  # type: ignore[arg-type]
            weights=list(raw_weights),  # type: ignore[arg-type]
            candidate_depth=config.get("embedding_candidate_depth", 1000),  # type: ignore[arg-type]
            colbert_rerank_depth=config.get("embedding_colbert_rerank_depth", 1000),  # type: ignore[arg-type]
            final_top_k_policy=str(
                config.get("embedding_final_top_k_policy", "agent_roster_top_k")
            ),
            service_startup_parameters=startup,
            expected_runtime_identity=(
                startup.get("runtime_identity") if startup is not None else None
            ),
            expected_server_source_sha256=(
                startup.get("server_source_sha256") if startup is not None else None
            ),
            expected_dense_dimensions=(
                startup.get(
                    "expected_dense_dimensions",
                    BGE_M3_EXPECTED_DENSE_DIMENSIONS,
                )
                if startup is not None
                else BGE_M3_EXPECTED_DENSE_DIMENSIONS
            ),  # type: ignore[arg-type]
            normalization_identity=(
                str(
                    startup.get(
                        "normalization_identity",
                        BGE_M3_NORMALIZATION_IDENTITY,
                    )
                )
                if startup is not None
                else BGE_M3_NORMALIZATION_IDENTITY
            ),
            dense_norm_tolerance=(
                startup.get(
                    "dense_norm_tolerance",
                    BGE_M3_DENSE_NORM_TOLERANCE,
                )
                if startup is not None
                else BGE_M3_DENSE_NORM_TOLERANCE
            ),  # type: ignore[arg-type]
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
        api_key_required=config.get("embedding_api_key_required", False),  # type: ignore[arg-type]
        timeout_seconds=config.get("embedding_timeout_seconds", 60),  # type: ignore[arg-type]
        cache_path=cache_template.format(agent_id=agent_id),
        query_instruction=str(config.get("embedding_query_instruction", "")),
        output_dimensions=config.get("embedding_output_dimensions"),  # type: ignore[arg-type]
    )


def build_retrieval_query(observation: AgentObservation) -> str:
    """
    功能：构造接近原版 `build_retrieval_query` 的状态查询短串。
    参数：
        observation：当前合法可见观察。
    返回：检索查询文本。
    副作用：无。
    异常：无。
    设计说明：使用对称、匿名的完整公开决策状态，不注入策略词或对手隐藏信息。
    """

    active_players = active_public_players(observation)
    ordered_players = sorted(
        active_players,
        key=lambda player: _relative_seat_offset(
            observation,
            player.seat,
        ),
    )
    labels = {
        player.agent_id: (
            "self" if player.agent_id == observation.agent_id else f"opponent_{index}"
        )
        for index, player in enumerate(ordered_players)
    }
    public_players = ";".join(
        f"{labels[player.agent_id]}(stack={player.stack},"
        f"street_bet={player.current_bet},committed={player.total_committed},"
        f"folded={player.folded},all_in={player.all_in})"
        for player in ordered_players
    )
    recent_actions = (
        ";".join(
            "{phase}:{actor}:{action}:{amount}".format(
                phase=action.get("phase", ""),
                actor=labels.get(
                    str(action.get("agent_id", action.get("player_id", ""))),
                    "opponent",
                ),
                action=action.get("action_type", action.get("action", "")),
                amount=action.get("amount", ""),
            )
            for action in observation.action_history[-12:]
        )
        or "none"
    )
    legal_actions = ";".join(
        f"{rule.action_type}(min={rule.min_amount},max={rule.max_amount},reopens={rule.reopens})"
        for rule in observation.legal_actions.actions
    )
    return (
        f"phase={observation.phase} "
        f"position={retrieval_position_label(observation)} "
        f"hole={' '.join(observation.hole_cards) or 'none'} "
        f"board={' '.join(observation.community_cards) or 'none'} "
        f"pot={observation.pot} current_bet={observation.current_bet} "
        f"to_call={observation.to_call} "
        f"public_players=[{public_players}] "
        f"recent_actions=[{recent_actions}] "
        f"legal_actions=[{legal_actions}]"
    )


def retrieval_position_label(observation: AgentObservation) -> str:
    """Backward-compatible name for the canonical public position truth."""

    return public_position_label(observation)


def _relative_seat_offset(
    observation: AgentObservation,
    seat: int,
) -> int:
    """Return clockwise offset from hero across possibly sparse seat numbers."""

    return relative_active_seat_offset(observation, seat)


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


def retrieval_document_template(
    backend: EmbeddingBackend,
    retrieval_unit: str,
) -> str:
    """Return the exact document representation selected by the retrieval unit."""

    if retrieval_unit == "decision_point_max_v1":
        return "{decision_state_query}"
    if retrieval_unit != "hand_terminal_v1":
        raise ValueError(f"未知 retrieval_unit：{retrieval_unit}")
    if isinstance(backend, BgeM3HybridHttpBackend):
        return "{historical_state_query}\\n{fact_text}"
    return "{fact_text}"


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
    retrieval_unit: str = "hand_terminal_v1",
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
    if retrieval_unit not in {"hand_terminal_v1", "decision_point_max_v1"}:
        raise ValueError(f"未知 retrieval_unit：{retrieval_unit}")
    query_text = build_retrieval_query(observation)
    backend = embedding_backend or HashEmbeddingBackend(dimensions)
    query_features = observation_features(observation)
    if retrieval_unit == "decision_point_max_v1":
        views_by_record = [_decision_point_views(record, backend) for record in records]
    else:
        views_by_record = [
            [
                {
                    "text": retrieval_text_for_backend(record, backend),
                    "features": list(record.features),
                    "decision_index": None,
                    "phase": _record_phase(record),
                }
            ]
            for record in records
        ]
    flat_views = [view for views in views_by_record for view in views]
    semantic_scores = _score_decision_documents(
        backend,
        query_text,
        [str(view["text"]) for view in flat_views],
    )
    scored: list[RetrievalScore] = []
    offset = 0
    for record, views in zip(records, views_by_record, strict=True):
        candidates = []
        for view, component in zip(
            views,
            semantic_scores[offset : offset + len(views)],
            strict=True,
        ):
            semantic = component.combined
            feature = jaccard_similarity(query_features, list(view["features"]))
            candidates.append(
                (
                    semantic_weight * semantic + feature_weight * feature,
                    semantic,
                    feature,
                    component,
                    view,
                )
            )
        offset += len(views)
        base, semantic, feature, component, matched_view = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                -1 if item[4]["decision_index"] is None else int(item[4]["decision_index"]),
            ),
        )
        salience = float(salience_fn(record.record_id) if salience_fn else 1.0)
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
                retrieval_unit,
                (
                    None
                    if matched_view["decision_index"] is None
                    else int(matched_view["decision_index"])
                ),
                str(matched_view["phase"]) if matched_view["phase"] else None,
            )
        )
    scored.sort(
        key=lambda item: (item.score, item.record.created_at, item.record.record_id),
        reverse=True,
    )
    return scored[:k]


def _decision_point_views(
    record: FactualMemoryRecord, backend: EmbeddingBackend
) -> list[dict[str, object]]:
    """Return versioned decision-point retrieval views or a legacy terminal fallback."""

    decisions = record.source.get("decisions", [])
    views = []
    if isinstance(decisions, list):
        for index, raw in enumerate(decisions):
            if not isinstance(raw, dict):
                continue
            query = str(raw.get("retrieval_query", "")).strip()
            features = raw.get("features")
            if not query or not isinstance(features, list):
                continue
            views.append(
                {
                    "text": query,
                    "features": [str(value) for value in features],
                    "decision_index": index,
                    "phase": str(raw.get("phase", "")),
                }
            )
    if views:
        return views
    return [
        {
            "text": retrieval_text_for_backend(record, backend),
            "features": list(record.features),
            "decision_index": None,
            "phase": _record_phase(record),
        }
    ]


def _score_decision_documents(
    backend: EmbeddingBackend,
    query: str,
    documents: list[str],
) -> list[SemanticScore]:
    """Score all decision views without bypassing a backend candidate-depth contract."""

    if isinstance(backend, BgeM3HybridHttpBackend) and (len(documents) > backend.candidate_depth):
        scores = []
        for start in range(0, len(documents), backend.candidate_depth):
            scores.extend(
                backend.score_documents(
                    query,
                    documents[start : start + backend.candidate_depth],
                )
            )
    else:
        scores = backend.score_documents(query, documents)
    if len(scores) != len(documents):
        raise RuntimeError("embedding backend score count does not match decision views")
    return scores


def _record_phase(record: FactualMemoryRecord) -> str | None:
    for feature in record.features:
        if str(feature).startswith("phase:"):
            return str(feature).partition(":")[2]
    return None


def _optional_nonempty_identity(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value.strip() == "unspecified":
        raise ValueError(f"BGE-M3 {field} must be a non-empty declared identity")
    return value.strip()


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"BGE-M3 {field} must be a 64-character SHA-256")
    return value.lower()


def _required_response_identity(value: object, field: str) -> str:
    try:
        parsed = _optional_nonempty_identity(value, field)
    except ValueError as exc:
        raise RuntimeError(f"BGE-M3 scoring service {field} is invalid") from exc
    if parsed is None:
        raise RuntimeError(f"BGE-M3 scoring service {field} is missing")
    return parsed


def _required_response_sha256(value: object, field: str) -> str:
    try:
        parsed = _optional_sha256(value, field)
    except ValueError as exc:
        raise RuntimeError(f"BGE-M3 scoring service {field} is invalid") from exc
    if parsed is None:
        raise RuntimeError(f"BGE-M3 scoring service {field} is missing")
    return parsed


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
    设计说明：维度不一致说明模型/缓存身份漂移，必须失败而非静默截断。
    """

    if len(left) != len(right):
        raise RuntimeError(f"embedding vector dimension mismatch: {len(left)} != {len(right)}")
    score = sum(a * b for a, b in zip(left, right, strict=True))
    if not math.isfinite(score):
        raise RuntimeError("embedding dot product is non-finite")
    return score


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else vector
