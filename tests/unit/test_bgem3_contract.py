from __future__ import annotations

from agentmemeval.memory.bgem3_contract import VersionedDocumentCache


def _cache(capacity: int = 2) -> VersionedDocumentCache[tuple[str, int]]:
    return VersionedDocumentCache(
        capacity=capacity,
        schema_version="bgem3_native_document_repr_v1",
        model="BAAI/bge-m3",
        revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        passage_max_length=1024,
    )


def _encode(texts: list[str]) -> list[tuple[str, int]]:
    return [(text, len(text)) for text in texts]


def test_bgem3_cache_key_is_versioned_and_restart_stable() -> None:
    first = _cache()
    restarted = _cache()
    assert first.key("run-a/agent-1", "same document") == restarted.key(
        "run-a/agent-1", "same document"
    )
    assert first.resolve("run-a/agent-1", ["same document"], _encode) == restarted.resolve(
        "run-a/agent-1",
        ["same document"], _encode
    )
    changed_schema = VersionedDocumentCache(
        capacity=2,
        schema_version="bgem3_native_document_repr_v2",
        model="BAAI/bge-m3",
        revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        passage_max_length=1024,
    )
    assert first.key("run-a/agent-1", "same document") != changed_schema.key(
        "run-a/agent-1", "same document"
    )
    assert first.key("run-a/agent-1", "same document") != first.key(
        "run-b/agent-1", "same document"
    )


def test_bgem3_cache_counters_and_lru_replacement_are_deterministic() -> None:
    cache = _cache(capacity=2)
    assert cache.resolve("run/agent", ["a", "b", "a"], _encode) == [
        ("a", 1),
        ("b", 1),
        ("a", 1),
    ]
    assert cache.metadata()["cache_miss_count"] == 2
    assert cache.metadata()["cache_hit_count"] == 0
    assert cache.resolve("run/agent", ["a"], _encode) == [("a", 1)]
    assert cache.metadata()["cache_hit_count"] == 1
    assert cache.resolve("run/agent", ["c"], _encode) == [("c", 1)]
    metadata = cache.metadata()
    assert metadata["cache_entries"] == 2
    assert metadata["cache_replacement_count"] == 1
    assert metadata["cache_capacity"] == 2
