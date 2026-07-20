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

from agentmemeval.memory.bgem3_contract import VersionedDocumentCache

QUERY_POLICY = "raw_symmetric_no_instruction"
DEFAULT_CACHE_SCHEMA_VERSION = "bgem3_native_document_repr_v1"


class ScoreRequest(BaseModel):
    """One raw query and its candidate factual-memory documents."""

    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=1000)
    weights: list[float] = Field(min_length=3, max_length=3)
    query_policy: str


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
        if any(value < 0 for value in self.weights) or sum(self.weights) <= 0:
            raise RuntimeError("BGE-M3 service weights must be non-negative and non-zero")
        self.batch_size = int(os.environ.get("BGEM3_BATCH_SIZE", "16"))
        self.query_max_length = int(os.environ.get("BGEM3_QUERY_MAX_LENGTH", "256"))
        self.passage_max_length = int(os.environ.get("BGEM3_PASSAGE_MAX_LENGTH", "1024"))
        self.cache_capacity = int(os.environ.get("BGEM3_CACHE_CAPACITY", "4096"))
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

    def score(self, request: ScoreRequest) -> dict[str, Any]:
        if request.query_policy != QUERY_POLICY:
            raise ValueError(f"query_policy must be {QUERY_POLICY}")
        requested_weights = tuple(float(value) for value in request.weights)
        if any(abs(left - right) > 1e-12 for left, right in zip(
            requested_weights, self.weights, strict=True
        )):
            raise ValueError("request weights do not match the frozen service weights")
        with self.lock:
            query = self._encode([request.query], self.query_max_length)[0]
            documents = self._documents(request.documents)
            scores = [self._score_pair(query, document) for document in documents]
            self.request_count += 1
            self.scored_document_count += len(documents)
        return {
            "model": self.model_id,
            "revision": self.revision,
            "weights_hash": self.weights_hash,
            "tokenizer_revision": self.tokenizer_revision,
            "cache_schema_version": self.cache_schema_version,
            "query_policy": QUERY_POLICY,
            "weights": {
                "dense": self.weights[0],
                "sparse": self.weights[1],
                "colbert": self.weights[2],
            },
            "scores": scores,
        }

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "status": "ok",
            "model": self.model_id,
            "revision": self.revision,
            "weights_hash": self.weights_hash,
            "tokenizer_revision": self.tokenizer_revision,
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
        }
        metadata.update(self.cache.metadata())
        return metadata

    def _documents(self, texts: list[str]) -> list[EncodedText]:
        return self.cache.resolve(
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
        dense_vectors = output["dense_vecs"]
        sparse_vectors = output["lexical_weights"]
        colbert_vectors = output["colbert_vecs"]
        return [
            EncodedText(
                dense=np.asarray(dense_vectors[index]),
                sparse=dict(sparse_vectors[index]),
                colbert=np.asarray(colbert_vectors[index]),
            )
            for index in range(len(texts))
        ]

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
