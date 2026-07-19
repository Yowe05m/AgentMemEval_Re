from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.runtime_lock import (
    FORMAL_RUNTIME_LOCK_FIELDS,
    build_formal_runtime_lock_from_manifest,
)
from tests.unit.test_formal_freeze import _runtime_manifest


def _write_manifest(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_runtime_lock_binds_model_embedding_prompt_hardware_and_service(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path, _runtime_manifest())
    result = build_formal_runtime_lock_from_manifest(path)
    lock = result["formal_runtime_lock"]
    assert result["schema_version"] == "task4_formal_runtime_lock_v2"
    assert result["status"] == "verified_from_real_service_run_manifest"
    assert set(lock) == set(FORMAL_RUNTIME_LOCK_FIELDS)
    assert lock["decision_model_revision"] == (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert lock["embedding_model_revision"] == (
        "5cf2132abc99cad020ac570b19d031efec650f2b"
    )
    assert lock["decision_prompt_version"]
    assert len(lock["decision_startup_parameters_sha256"]) == 64
    assert len(lock["embedding_startup_parameters_sha256"]) == 64
    assert result["source_manifest"]["sha256"]


def test_runtime_lock_rejects_dirty_or_unverified_manifest(tmp_path: Path) -> None:
    manifest = _runtime_manifest()
    manifest["metadata"]["code"]["dirty"] = True
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(ConfigError, match="clean recorded worktree"):
        build_formal_runtime_lock_from_manifest(path)


def test_runtime_lock_rejects_missing_prompt_identity(tmp_path: Path) -> None:
    manifest = _runtime_manifest()
    manifest["metadata"]["prompts"]["decision_system_sha256"] = ""
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(ConfigError, match="decision_system_sha256"):
        build_formal_runtime_lock_from_manifest(path)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        (
            "service",
            "service_startup_parameters",
            "decision service startup parameters",
        ),
        (
            "embedding",
            "service_startup_parameters",
            "embedding service startup parameters",
        ),
    ],
)
def test_runtime_lock_rejects_missing_service_startup_parameters(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    manifest = _runtime_manifest()
    del manifest["metadata"][section][field]
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ConfigError, match=message):
        build_formal_runtime_lock_from_manifest(path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("run_id", "run_id"),
        ("code_commit", "code commit"),
    ],
)
def test_runtime_lock_rejects_missing_run_identity(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    manifest = _runtime_manifest()
    if field == "run_id":
        manifest["run_id"] = ""
    else:
        manifest["metadata"]["code"]["commit"] = ""
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ConfigError, match=message):
        build_formal_runtime_lock_from_manifest(path)
