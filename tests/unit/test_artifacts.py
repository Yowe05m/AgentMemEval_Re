"""Regression tests for non-destructive experiment artifact creation."""

from pathlib import Path
from uuid import uuid4

import pytest

from agentmemeval.storage.artifacts import ArtifactManager


def test_artifact_manager_refuses_nonempty_existing_run_directory() -> None:
    output_root = Path("tmp") / "test_outputs" / f"artifact_collision_{uuid4().hex}"
    config = {
        "provider": {"provider": "mock", "model": "mock-deterministic-v1"},
        "experiment": {"scenario": "fixed_evolving_table", "seed": 1},
    }
    first = ArtifactManager(output_root, "same_run", config)
    first.write_text("sentinel.txt", "preserve me")

    with pytest.raises(FileExistsError, match="拒绝追加或覆盖"):
        ArtifactManager(output_root, "same_run", config)

    assert (first.run_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserve me"
