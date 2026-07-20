"""Pure-Python BGE-M3 document-cache contract shared by service and tests."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class VersionedDocumentCache(Generic[T]):
    """Bounded LRU keyed by the full immutable document-encoding identity."""

    KEY_POLICY = (
        "sha256(schema\\0model\\0revision\\0tokenizer\\0passage_max_length\\0text)"
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

    def key(self, text: str) -> str:
        payload = "\0".join(
            (
                self.schema_version,
                self.model,
                self.revision,
                self.tokenizer_revision,
                str(self.passage_max_length),
                text,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve(self, texts: list[str], encoder: Callable[[list[str]], list[T]]) -> list[T]:
        keys = [self.key(text) for text in texts]
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
