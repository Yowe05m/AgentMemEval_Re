from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from agentmemeval.memory.fact_expr_async import FactExprAsyncMemory
from tests.unit.test_memory import make_trajectory


def test_async_restore_materializes_verified_empty_persistence_views(
    tmp_path: Path,
) -> None:
    source = FactExprAsyncMemory(
        "agent_00",
        sweep_every=1,
        evidence_k=2,
        mirror_prob=0.0,
    )
    source.on_hand_finished(make_trajectory())
    snapshot = source.snapshot()

    persistence_directory = tmp_path / "agent_00"
    restored = FactExprAsyncMemory(
        "agent_00",
        sweep_every=1,
        evidence_k=2,
        mirror_prob=0.0,
    )
    restored.bind_persistence(
        persistence_directory,
        reference_root=tmp_path,
    )
    assert (persistence_directory / "facts_state.jsonl").read_bytes() == b""
    assert (persistence_directory / "sweep_log.jsonl").read_bytes() == b""

    restored.restore(snapshot)

    assert restored.fact_state == source.fact_state
    assert restored.sweep_log == source.sweep_log
    assert (persistence_directory / "facts_state.jsonl").read_bytes()
    assert (persistence_directory / "sweep_log.jsonl").read_bytes()


def test_async_restore_rejects_nonempty_divergent_persistence_view(
    tmp_path: Path,
) -> None:
    source = FactExprAsyncMemory(
        "agent_00",
        sweep_every=1,
        evidence_k=2,
        mirror_prob=0.0,
    )
    source.on_hand_finished(make_trajectory())
    snapshot = source.snapshot()

    restored = FactExprAsyncMemory(
        "agent_00",
        sweep_every=1,
        evidence_k=2,
        mirror_prob=0.0,
    )
    restored.restore(snapshot)
    persistence_directory = tmp_path / "agent_00"
    restored.bind_persistence(
        persistence_directory,
        reference_root=tmp_path,
    )
    before = (persistence_directory / "facts_state.jsonl").read_bytes()

    divergent = deepcopy(snapshot)
    divergent.payload["fact_state"]["agent_00-fact-1"]["stability"] = 11.0
    with pytest.raises(ValueError, match="facts_state.jsonl diverges"):
        restored.restore(divergent)

    assert (persistence_directory / "facts_state.jsonl").read_bytes() == before
