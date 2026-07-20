from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.campaign import (
    REQUIRED_RUN_ARTIFACTS,
    _valid_completed_attempt,
)


def _row(
    run_dir: Path,
    *,
    attempt: int,
    status: str,
) -> dict[str, str]:
    return {
        "condition_id": "mixed",
        "seed": "1",
        "attempt": str(attempt),
        "status": status,
        "run_dir": str(run_dir),
    }


def _complete_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for name in REQUIRED_RUN_ARTIFACTS:
        (run_dir / name).write_text("x", encoding="utf-8")


def test_valid_completed_attempt_accepts_failed_attempt_then_retry(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "a01"
    completed = tmp_path / "a02"
    _complete_artifacts(completed)
    rows = [
        _row(failed, attempt=1, status="running"),
        _row(failed, attempt=1, status="failed"),
        _row(completed, attempt=2, status="running"),
        _row(completed, attempt=2, status="complete"),
    ]

    selected = _valid_completed_attempt(rows, condition_id="mixed", seed=1)

    assert selected == rows[-1]


def test_valid_completed_attempt_rejects_completed_attempt_superseded_by_failure(
    tmp_path: Path,
) -> None:
    completed = tmp_path / "a01"
    failed = tmp_path / "a02"
    _complete_artifacts(completed)
    rows = [
        _row(completed, attempt=1, status="running"),
        _row(completed, attempt=1, status="complete"),
        _row(failed, attempt=2, status="running"),
        _row(failed, attempt=2, status="failed"),
    ]

    with pytest.raises(ConfigError, match="superseded by a non-complete"):
        _valid_completed_attempt(rows, condition_id="mixed", seed=1)


def test_valid_completed_attempt_rejects_same_attempt_resurrection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "a01"
    _complete_artifacts(run_dir)
    rows = [
        _row(run_dir, attempt=1, status="running"),
        _row(run_dir, attempt=1, status="failed"),
        _row(run_dir, attempt=1, status="complete"),
    ]

    with pytest.raises(ConfigError, match="failed state precedes completion"):
        _valid_completed_attempt(rows, condition_id="mixed", seed=1)


def test_valid_completed_attempt_rejects_incomplete_completed_leaf(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "a01"
    run_dir.mkdir()
    rows = [
        _row(run_dir, attempt=1, status="running"),
        _row(run_dir, attempt=1, status="complete"),
    ]

    with pytest.raises(ConfigError, match="incomplete artifacts"):
        _valid_completed_attempt(rows, condition_id="mixed", seed=1)
