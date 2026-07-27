"""Serve BGE-M3 dense, sparse lexical, and ColBERT scores from one GPU model."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel, Field

from agentmemeval.core.bgem3_contract import (
    BGE_M3_DENSE_NORM_TOLERANCE,
    BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    BGE_M3_NORMALIZATION_IDENTITY,
    VersionedDocumentCache,
    make_output_contract_metadata,
    validate_colbert_observation,
    validate_dense_observation,
    validate_sparse_weights,
)

QUERY_POLICY = "raw_symmetric_no_instruction"
DEFAULT_CACHE_SCHEMA_VERSION = "bgem3_native_document_repr_v2"


class ScoreRequest(BaseModel):
    """One raw query and its candidate factual-memory documents."""

    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=1000)
    weights: list[float] = Field(min_length=3, max_length=3)
    query_policy: str
    cache_namespace: str = Field(min_length=1, max_length=512)


@dataclass(slots=True)
class EncodedText:
    dense: np.ndarray
    sparse: dict[str, float]
    colbert: np.ndarray


class BgeM3Scorer:
    """Single-process scorer with an LRU document representation cache."""

    def __init__(self) -> None:
        self.model_path = _required_env("BGEM3_MODEL_PATH")
        self.model_id = os.environ.get("BGEM3_MODEL_ID", "BAAI/bge-m3")
        self.revision = _required_env("BGEM3_REVISION")
        self.weights_hash = _required_env("BGEM3_WEIGHTS_HASH")
        self.tokenizer_revision = _required_env("BGEM3_TOKENIZER_REVISION")
        self.cache_schema_version = os.environ.get(
            "BGEM3_CACHE_SCHEMA_VERSION", DEFAULT_CACHE_SCHEMA_VERSION
        ).strip()
        if not self.cache_schema_version:
            raise RuntimeError("BGE-M3 cache schema version must not be empty")
        self.weights = (
            float(os.environ.get("BGEM3_DENSE_WEIGHT", "0.4")),
            float(os.environ.get("BGEM3_SPARSE_WEIGHT", "0.2")),
            float(os.environ.get("BGEM3_COLBERT_WEIGHT", "0.4")),
        )
        if (
            any(not math.isfinite(value) or value < 0 for value in self.weights)
            or not math.isfinite(sum(self.weights))
            or sum(self.weights) <= 0
        ):
            raise RuntimeError(
                "BGE-M3 service weights must be finite, non-negative, and non-zero"
            )
        if self.weights != (0.4, 0.2, 0.4):
            raise RuntimeError(
                "BGE-M3 service weights must be frozen at dense=0.4, "
                "sparse=0.2, colbert=0.4"
            )
        self.batch_size = int(os.environ.get("BGEM3_BATCH_SIZE", "16"))
        self.query_max_length = int(os.environ.get("BGEM3_QUERY_MAX_LENGTH", "256"))
        self.passage_max_length = int(os.environ.get("BGEM3_PASSAGE_MAX_LENGTH", "1024"))
        self.cache_capacity = int(os.environ.get("BGEM3_CACHE_CAPACITY", "4096"))
        if min(
            self.batch_size,
            self.query_max_length,
            self.passage_max_length,
            self.cache_capacity,
        ) < 1:
            raise RuntimeError(
                "BGE-M3 batch/query/passage/cache parameters must be positive integers"
            )
        self.expected_dense_dimensions = int(
            os.environ.get(
                "BGEM3_EXPECTED_DENSE_DIMENSIONS",
                str(BGE_M3_EXPECTED_DENSE_DIMENSIONS),
            )
        )
        if self.expected_dense_dimensions != BGE_M3_EXPECTED_DENSE_DIMENSIONS:
            raise RuntimeError(
                "BAAI/bge-m3 dense dimensions must be frozen at "
                f"{BGE_M3_EXPECTED_DENSE_DIMENSIONS}"
            )
        self.normalization_identity = os.environ.get(
            "BGEM3_NORMALIZATION_IDENTITY",
            BGE_M3_NORMALIZATION_IDENTITY,
        ).strip()
        if self.normalization_identity != BGE_M3_NORMALIZATION_IDENTITY:
            raise RuntimeError(
                "BGE-M3 normalization identity must be frozen at "
                f"{BGE_M3_NORMALIZATION_IDENTITY}"
            )
        self.dense_norm_tolerance = float(
            os.environ.get(
                "BGEM3_DENSE_NORM_TOLERANCE",
                str(BGE_M3_DENSE_NORM_TOLERANCE),
            )
        )
        if (
            not math.isfinite(self.dense_norm_tolerance)
            or not math.isclose(
                self.dense_norm_tolerance,
                BGE_M3_DENSE_NORM_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "BGE-M3 dense norm tolerance must be frozen at "
                f"{BGE_M3_DENSE_NORM_TOLERANCE}"
            )
        self.runtime_identity = os.environ.get(
            "BGEM3_RUNTIME_IDENTITY", "unspecified"
        ).strip()
        self.runtime_source_root = os.environ.get(
            "BGEM3_RUNTIME_SOURCE_ROOT", "unspecified"
        ).strip()
        self.server_source_sha256 = os.environ.get(
            "BGEM3_SERVER_SOURCE_SHA256", "unspecified"
        ).strip()
        if any(
            not value or value == "unspecified"
            for value in (self.runtime_identity, self.runtime_source_root)
        ):
            raise RuntimeError("BGE-M3 runtime identity/source root must be explicitly bound")
        if (
            len(self.server_source_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.server_source_sha256
            )
        ):
            raise RuntimeError("BGE-M3 server source SHA-256 must be explicitly bound")
        self.server_source_sha256 = self.server_source_sha256.lower()
        self.lock = threading.Lock()
        self.cache = VersionedDocumentCache[EncodedText](
            capacity=self.cache_capacity,
            schema_version=self.cache_schema_version,
            model=self.model_id,
            revision=self.revision,
            tokenizer_revision=self.tokenizer_revision,
            passage_max_length=self.passage_max_length,
        )
        self.request_count = 0
        self.scored_document_count = 0
        self.observed_dense_dimensions: int | None = None
        self.observed_colbert_dimensions: int | None = None
        self.observed_dense_norm_min: float | None = None
        self.observed_dense_norm_max: float | None = None
        self.validated_vector_count = 0
        self.model = BGEM3FlagModel(
            self.model_path,
            normalize_embeddings=True,
            use_fp16=True,
            query_instruction_for_retrieval=None,
            devices="cuda:0",
            batch_size=self.batch_size,
            query_max_length=self.query_max_length,
            passage_max_length=self.passage_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            trust_remote_code=False,
        )
        # A successful health response must represent measured model output, not
        # merely echo normalize_embeddings=True from configuration.
        self._encode(["BGE-M3 output contract startup probe"], self.query_max_length)

    def score(self, request: ScoreRequest) -> dict[str, Any]:
        if request.query_policy != QUERY_POLICY:
            raise ValueError(f"query_policy must be {QUERY_POLICY}")
        requested_weights = tuple(float(value) for value in request.weights)
        if any(not math.isfinite(value) or value < 0 for value in requested_weights):
            raise ValueError("request weights must be finite and non-negative")
        if any(abs(left - right) > 1e-12 for left, right in zip(
            requested_weights, self.weights, strict=True
        )):
            raise ValueError("request weights do not match the frozen service weights")
        with self.lock:
            query = self._encode([request.query], self.query_max_length)[0]
            documents = self._documents(request.cache_namespace, request.documents)
            scores = [self._score_pair(query, document) for document in documents]
            self.request_count += 1
            self.scored_document_count += len(documents)
        return {
            "model": self.model_id,
            "revision": self.revision,
            "weights_hash": self.weights_hash,
            "tokenizer_revision": self.tokenizer_revision,
            "cache_schema_version": self.cache_schema_version,
            "runtime_identity": self.runtime_identity,
            "runtime_source_root": self.runtime_source_root,
            "server_source_sha256": self.server_source_sha256,
            "cache_namespace": request.cache_namespace,
            "query_policy": QUERY_POLICY,
            "weights": {
                "dense": self.weights[0],
                "sparse": self.weights[1],
                "colbert": self.weights[2],
            },
            "output_contract": self._output_contract_metadata(),
            "scores": scores,
        }

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "status": "ok",
            "model": self.model_id,
            "revision": self.revision,
            "weights_hash": self.weights_hash,
            "tokenizer_revision": self.tokenizer_revision,
            "runtime_identity": self.runtime_identity,
            "runtime_source_root": self.runtime_source_root,
            "server_source_sha256": self.server_source_sha256,
            "query_policy": QUERY_POLICY,
            "query_instruction": None,
            "retrieval_modes": ["dense", "sparse", "colbert"],
            "weights": {
                "dense": self.weights[0],
                "sparse": self.weights[1],
                "colbert": self.weights[2],
            },
            "batch_size": self.batch_size,
            "query_max_length": self.query_max_length,
            "passage_max_length": self.passage_max_length,
            "request_count": self.request_count,
            "scored_document_count": self.scored_document_count,
            "output_contract": self._output_contract_metadata(),
        }
        metadata.update(self.cache.metadata())
        return metadata

    def _documents(self, namespace: str, texts: list[str]) -> list[EncodedText]:
        return self.cache.resolve(
            namespace,
            texts,
            lambda missing: self._encode(missing, self.passage_max_length),
        )

    def _encode(self, texts: list[str], max_length: int) -> list[EncodedText]:
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        if not isinstance(output, dict):
            raise RuntimeError("BGE-M3 encode output must be a mapping")
        required_fields = ("dense_vecs", "lexical_weights", "colbert_vecs")
        if any(field not in output for field in required_fields):
            raise RuntimeError("BGE-M3 encode output is missing a required representation")
        dense_vectors = output["dense_vecs"]
        sparse_vectors = output["lexical_weights"]
        colbert_vectors = output["colbert_vecs"]
        try:
            response_lengths = (
                len(dense_vectors),
                len(sparse_vectors),
                len(colbert_vectors),
            )
        except TypeError as exc:
            raise RuntimeError("BGE-M3 encode representations must be sized batches") from exc
        if response_lengths != (len(texts), len(texts), len(texts)):
            raise RuntimeError("BGE-M3 encode representation count mismatch")

        encoded: list[EncodedText] = []
        dense_norms: list[float] = []
        observed_dense_dimensions: set[int] = set()
        observed_colbert_dimensions: set[int] = set()
        for index in range(len(texts)):
            try:
                dense = np.asarray(dense_vectors[index])
                colbert = np.asarray(colbert_vectors[index])
                dense_dimension, dense_norm = validate_dense_observation(
                    shape=tuple(int(value) for value in dense.shape),
                    all_finite=bool(np.all(np.isfinite(dense))),
                    l2_norm=float(np.linalg.norm(dense)),
                    expected_dimensions=self.expected_dense_dimensions,
                    norm_tolerance=self.dense_norm_tolerance,
                )
                colbert_dimension = validate_colbert_observation(
                    shape=tuple(int(value) for value in colbert.shape),
                    all_finite=bool(np.all(np.isfinite(colbert))),
                    expected_dimensions=self.expected_dense_dimensions,
                )
                sparse = validate_sparse_weights(sparse_vectors[index])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"BGE-M3 encode output contract failed at batch index {index}: {exc}"
                ) from exc
            observed_dense_dimensions.add(dense_dimension)
            observed_colbert_dimensions.add(colbert_dimension)
            dense_norms.append(dense_norm)
            encoded.append(EncodedText(dense=dense, sparse=sparse, colbert=colbert))

        if self.observed_dense_dimensions is not None:
            observed_dense_dimensions.add(self.observed_dense_dimensions)
        if self.observed_colbert_dimensions is not None:
            observed_colbert_dimensions.add(self.observed_colbert_dimensions)
        if len(observed_dense_dimensions) != 1 or len(observed_colbert_dimensions) != 1:
            raise RuntimeError("BGE-M3 representation dimensions drifted across encode calls")
        self.observed_dense_dimensions = observed_dense_dimensions.pop()
        self.observed_colbert_dimensions = observed_colbert_dimensions.pop()
        batch_norm_min = min(dense_norms)
        batch_norm_max = max(dense_norms)
        self.observed_dense_norm_min = (
            batch_norm_min
            if self.observed_dense_norm_min is None
            else min(self.observed_dense_norm_min, batch_norm_min)
        )
        self.observed_dense_norm_max = (
            batch_norm_max
            if self.observed_dense_norm_max is None
            else max(self.observed_dense_norm_max, batch_norm_max)
        )
        self.validated_vector_count += len(encoded)
        return encoded

    def _output_contract_metadata(self) -> dict[str, object]:
        if (
            self.observed_dense_dimensions is None
            or self.observed_colbert_dimensions is None
            or self.observed_dense_norm_min is None
            or self.observed_dense_norm_max is None
            or self.validated_vector_count < 1
        ):
            raise RuntimeError("BGE-M3 output contract has not been observed")
        return make_output_contract_metadata(
            expected_dense_dimensions=self.expected_dense_dimensions,
            normalization_identity=self.normalization_identity,
            dense_norm_tolerance=self.dense_norm_tolerance,
            observed_dense_dimensions=self.observed_dense_dimensions,
            observed_dense_norm_min=self.observed_dense_norm_min,
            observed_dense_norm_max=self.observed_dense_norm_max,
            observed_colbert_dimensions=self.observed_colbert_dimensions,
            validated_vector_count=self.validated_vector_count,
        )

    def _score_pair(self, query: EncodedText, document: EncodedText) -> dict[str, float]:
        dense = float(np.dot(query.dense, document.dense))
        sparse = _as_float(
            self.model.compute_lexical_matching_score(query.sparse, document.sparse)
        )
        colbert = _as_float(self.model.colbert_score(query.colbert, document.colbert))
        weight_sum = sum(self.weights)
        combined = (
            self.weights[0] * dense
            + self.weights[1] * sparse
            + self.weights[2] * colbert
        ) / weight_sum
        result = {
            "combined": combined,
            "dense": dense,
            "sparse": sparse,
            "colbert": colbert,
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("BGE-M3 produced a non-finite score")
        return result


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


scorer: BgeM3Scorer | None = None
app = FastAPI(title="AgentMemEval BGE-M3 hybrid scorer", version="1")


@app.on_event("startup")
def load_model() -> None:
    global scorer
    scorer = BgeM3Scorer()


@app.get("/health")
def health() -> dict[str, Any]:
    if scorer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return scorer.metadata()


@app.get("/v1/bgem3/metadata")
def metadata() -> dict[str, Any]:
    if scorer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return scorer.metadata()


@app.post("/v1/bgem3/score")
async def score(request: ScoreRequest) -> dict[str, Any]:
    if scorer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        return await run_in_threadpool(scorer.score, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
