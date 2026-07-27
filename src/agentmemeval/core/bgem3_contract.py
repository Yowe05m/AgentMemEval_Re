"""Pure BGE-M3 representation and response-metadata contracts."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

T = TypeVar("T")

BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION = "bgem3_hybrid_output_contract_v1"
BGE_M3_EXPECTED_DENSE_DIMENSIONS = 1024
BGE_M3_NORMALIZATION_IDENTITY = "flagembedding_normalize_embeddings_true_l2_v1"
BGE_M3_DENSE_NORM_TOLERANCE = 1e-3
BGE_M3_SPARSE_VECTOR_CONTRACT = (
    "nonempty_token_id_to_finite_nonnegative_weight_v1"
)
BGE_M3_COLBERT_VECTOR_CONTRACT = "nonempty_finite_matrix_dense_width_v1"

_OUTPUT_CONTRACT_FIELDS = {
    "schema_version",
    "expected_dense_dimensions",
    "observed_dense_dimensions",
    "normalization_identity",
    "dense_norm_tolerance",
    "observed_dense_norm_min",
    "observed_dense_norm_max",
    "observed_colbert_dimensions",
    "sparse_vector_contract",
    "colbert_vector_contract",
    "validated_vector_count",
}


def validate_dense_observation(
    *,
    shape: tuple[int, ...],
    all_finite: bool,
    l2_norm: float,
    expected_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    norm_tolerance: float = BGE_M3_DENSE_NORM_TOLERANCE,
) -> tuple[int, float]:
    """Validate the observable dense-vector contract without retaining the vector."""

    _validate_expected_contract(expected_dimensions, norm_tolerance)
    if len(shape) != 1 or shape[0] != expected_dimensions:
        raise ValueError(
            "BGE-M3 dense vector dimension mismatch: "
            f"observed={shape!r}, expected=({expected_dimensions},)"
        )
    if all_finite is not True:
        raise ValueError("BGE-M3 dense vector contains a non-finite value")
    parsed_norm = _finite_float(l2_norm, "dense L2 norm")
    if parsed_norm <= 0:
        raise ValueError("BGE-M3 dense vector must be non-zero")
    if not math.isclose(parsed_norm, 1.0, rel_tol=0.0, abs_tol=norm_tolerance):
        raise ValueError(
            "BGE-M3 dense vector violates normalize_embeddings=True: "
            f"norm={parsed_norm}, tolerance={norm_tolerance}"
        )
    return shape[0], parsed_norm


def validate_colbert_observation(
    *,
    shape: tuple[int, ...],
    all_finite: bool,
    expected_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
) -> int:
    """Validate a non-empty finite ColBERT matrix with the frozen hidden width."""

    if type(expected_dimensions) is not int or expected_dimensions < 1:
        raise ValueError("BGE-M3 expected dense dimensions must be a positive integer")
    if len(shape) != 2 or shape[0] < 1 or shape[1] != expected_dimensions:
        raise ValueError(
            "BGE-M3 ColBERT vector shape mismatch: "
            f"observed={shape!r}, expected=(tokens>0, {expected_dimensions})"
        )
    if all_finite is not True:
        raise ValueError("BGE-M3 ColBERT vector contains a non-finite value")
    return shape[1]


def validate_sparse_weights(raw: object) -> dict[str, float]:
    """Return a canonical, finite, non-empty token-id sparse-weight mapping."""

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("BGE-M3 sparse vector must be a non-empty mapping")
    parsed: dict[str, float] = {}
    has_positive_weight = False
    for raw_token_id, raw_weight in raw.items():
        if isinstance(raw_token_id, bool):
            raise ValueError("BGE-M3 sparse token ids must be non-negative integers")
        token_id = str(raw_token_id).strip()
        if not token_id.isdecimal():
            raise ValueError("BGE-M3 sparse token ids must be non-negative integers")
        canonical_token_id = str(int(token_id))
        if canonical_token_id in parsed:
            raise ValueError("BGE-M3 sparse vector contains duplicate canonical token ids")
        weight = _finite_float(raw_weight, "sparse weight")
        if weight < 0:
            raise ValueError("BGE-M3 sparse weights must be non-negative")
        has_positive_weight = has_positive_weight or weight > 0
        parsed[canonical_token_id] = weight
    if not has_positive_weight:
        raise ValueError("BGE-M3 sparse vector must contain a positive weight")
    return parsed


def make_output_contract_metadata(
    *,
    observed_dense_dimensions: int,
    observed_dense_norm_min: float,
    observed_dense_norm_max: float,
    observed_colbert_dimensions: int,
    validated_vector_count: int,
    expected_dense_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    normalization_identity: str = BGE_M3_NORMALIZATION_IDENTITY,
    dense_norm_tolerance: float = BGE_M3_DENSE_NORM_TOLERANCE,
) -> dict[str, object]:
    """Build the non-vector metadata echoed by BGE-M3 score and health responses."""

    raw = {
        "schema_version": BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION,
        "expected_dense_dimensions": expected_dense_dimensions,
        "observed_dense_dimensions": observed_dense_dimensions,
        "normalization_identity": normalization_identity,
        "dense_norm_tolerance": dense_norm_tolerance,
        "observed_dense_norm_min": observed_dense_norm_min,
        "observed_dense_norm_max": observed_dense_norm_max,
        "observed_colbert_dimensions": observed_colbert_dimensions,
        "sparse_vector_contract": BGE_M3_SPARSE_VECTOR_CONTRACT,
        "colbert_vector_contract": BGE_M3_COLBERT_VECTOR_CONTRACT,
        "validated_vector_count": validated_vector_count,
    }
    return validate_output_contract_metadata(
        raw,
        expected_dense_dimensions=expected_dense_dimensions,
        normalization_identity=normalization_identity,
        dense_norm_tolerance=dense_norm_tolerance,
    )


def validate_output_contract_metadata(
    raw: object,
    *,
    expected_dense_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    normalization_identity: str = BGE_M3_NORMALIZATION_IDENTITY,
    dense_norm_tolerance: float = BGE_M3_DENSE_NORM_TOLERANCE,
) -> dict[str, object]:
    """Fail closed on malformed, incomplete, or incompatible service metadata."""

    _validate_expected_contract(expected_dense_dimensions, dense_norm_tolerance)
    if not normalization_identity.strip():
        raise ValueError("BGE-M3 normalization identity must not be empty")
    if not isinstance(raw, dict) or set(raw) != _OUTPUT_CONTRACT_FIELDS:
        raise ValueError("BGE-M3 output contract metadata fields are malformed")
    if raw["schema_version"] != BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION:
        raise ValueError("BGE-M3 output contract schema mismatch")
    for field in (
        "expected_dense_dimensions",
        "observed_dense_dimensions",
        "observed_colbert_dimensions",
    ):
        value = raw[field]
        if type(value) is not int or value != expected_dense_dimensions:
            raise ValueError(f"BGE-M3 output contract {field} mismatch")
    if raw["normalization_identity"] != normalization_identity:
        raise ValueError("BGE-M3 output contract normalization identity mismatch")
    observed_tolerance = _finite_float(
        raw["dense_norm_tolerance"], "output contract dense norm tolerance"
    )
    if not math.isclose(
        observed_tolerance,
        dense_norm_tolerance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("BGE-M3 output contract dense norm tolerance mismatch")
    if raw["sparse_vector_contract"] != BGE_M3_SPARSE_VECTOR_CONTRACT:
        raise ValueError("BGE-M3 sparse vector contract mismatch")
    if raw["colbert_vector_contract"] != BGE_M3_COLBERT_VECTOR_CONTRACT:
        raise ValueError("BGE-M3 ColBERT vector contract mismatch")
    norm_min = _finite_float(raw["observed_dense_norm_min"], "observed dense norm min")
    norm_max = _finite_float(raw["observed_dense_norm_max"], "observed dense norm max")
    if norm_min > norm_max:
        raise ValueError("BGE-M3 observed dense norm range is inverted")
    for value in (norm_min, norm_max):
        if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=dense_norm_tolerance):
            raise ValueError("BGE-M3 observed dense norm is outside the frozen tolerance")
    validated_vector_count = raw["validated_vector_count"]
    if type(validated_vector_count) is not int or validated_vector_count < 1:
        raise ValueError("BGE-M3 validated vector count must be a positive integer")
    return {
        "schema_version": BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION,
        "expected_dense_dimensions": expected_dense_dimensions,
        "observed_dense_dimensions": expected_dense_dimensions,
        "normalization_identity": normalization_identity,
        "dense_norm_tolerance": dense_norm_tolerance,
        "observed_dense_norm_min": norm_min,
        "observed_dense_norm_max": norm_max,
        "observed_colbert_dimensions": expected_dense_dimensions,
        "sparse_vector_contract": BGE_M3_SPARSE_VECTOR_CONTRACT,
        "colbert_vector_contract": BGE_M3_COLBERT_VECTOR_CONTRACT,
        "validated_vector_count": validated_vector_count,
    }


def output_contract_identity(
    *,
    expected_dense_dimensions: int = BGE_M3_EXPECTED_DENSE_DIMENSIONS,
    normalization_identity: str = BGE_M3_NORMALIZATION_IDENTITY,
    dense_norm_tolerance: float = BGE_M3_DENSE_NORM_TOLERANCE,
) -> dict[str, object]:
    """Return the immutable semantic identity, excluding observations/counters."""

    _validate_expected_contract(expected_dense_dimensions, dense_norm_tolerance)
    if not normalization_identity.strip():
        raise ValueError("BGE-M3 normalization identity must not be empty")
    return {
        "schema_version": BGE_M3_OUTPUT_CONTRACT_SCHEMA_VERSION,
        "expected_dense_dimensions": expected_dense_dimensions,
        "normalization_identity": normalization_identity,
        "dense_norm_tolerance": dense_norm_tolerance,
        "sparse_vector_contract": BGE_M3_SPARSE_VECTOR_CONTRACT,
        "colbert_vector_contract": BGE_M3_COLBERT_VECTOR_CONTRACT,
    }


def _validate_expected_contract(expected_dimensions: int, norm_tolerance: float) -> None:
    if type(expected_dimensions) is not int or expected_dimensions < 1:
        raise ValueError("BGE-M3 expected dense dimensions must be a positive integer")
    tolerance = _finite_float(norm_tolerance, "dense norm tolerance")
    if tolerance <= 0 or tolerance > 0.01:
        raise ValueError("BGE-M3 dense norm tolerance must be in (0, 0.01]")


def _finite_float(raw: object, field: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"BGE-M3 {field} must be a finite number")
    if hasattr(raw, "item"):
        try:
            raw = raw.item()  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BGE-M3 {field} must be a finite scalar") from exc
    if isinstance(raw, bool):
        raise ValueError(f"BGE-M3 {field} must be a finite number")
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BGE-M3 {field} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"BGE-M3 {field} must be finite")
    return value


class VersionedDocumentCache(Generic[T]):
    """Bounded LRU keyed by the full immutable document-encoding identity."""

    KEY_POLICY = (
        "sha256(schema\\0model\\0revision\\0tokenizer\\0passage_max_length"
        "\\0namespace\\0text)"
    )

    def __init__(
        self,
        *,
        capacity: int,
        schema_version: str,
        model: str,
        revision: str,
        tokenizer_revision: str,
        passage_max_length: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        identity = (schema_version, model, revision, tokenizer_revision)
        if not all(value.strip() for value in identity):
            raise ValueError("cache identity fields must not be empty")
        if passage_max_length < 1:
            raise ValueError("passage_max_length must be positive")
        self.capacity = int(capacity)
        self.schema_version = schema_version
        self.model = model
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.passage_max_length = int(passage_max_length)
        self.entries: OrderedDict[str, T] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.replacement_count = 0

    def key(self, namespace: str, text: str) -> str:
        if not namespace.strip():
            raise ValueError("cache namespace must not be empty")
        payload = "\0".join(
            (
                self.schema_version,
                self.model,
                self.revision,
                self.tokenizer_revision,
                str(self.passage_max_length),
                namespace,
                text,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve(
        self,
        namespace: str,
        texts: list[str],
        encoder: Callable[[list[str]], list[T]],
    ) -> list[T]:
        keys = [self.key(namespace, text) for text in texts]
        missing: list[tuple[str, str]] = []
        seen_missing: set[str] = set()
        for text, key in zip(texts, keys, strict=True):
            if key in self.entries:
                self.hit_count += 1
                self.entries.move_to_end(key)
            elif key not in seen_missing:
                missing.append((key, text))
                seen_missing.add(key)
                self.miss_count += 1
        if missing:
            encoded = encoder([text for _key, text in missing])
            if len(encoded) != len(missing):
                raise RuntimeError("BGE-M3 cache encoder response count mismatch")
            for (key, _text), value in zip(missing, encoded, strict=True):
                self.entries[key] = value
                self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
                self.replacement_count += 1
        return [self.entries[key] for key in keys]

    def metadata(self) -> dict[str, int | str]:
        return {
            "cache_capacity": self.capacity,
            "cache_schema_version": self.schema_version,
            "cache_key_policy": self.KEY_POLICY,
            "cache_entries": len(self.entries),
            "cache_hit_count": self.hit_count,
            "cache_miss_count": self.miss_count,
            "cache_replacement_count": self.replacement_count,
        }
