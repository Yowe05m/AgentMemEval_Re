from __future__ import annotations

from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.runtime_lock import runtime_identity_from_metadata
from agentmemeval.experiments.admission import (
    _runtime_lock_blockers,
    assess_run_admission,
)
from tests.unit.test_formal_freeze import _runtime_manifest


def _config(mode: str) -> dict[str, object]:
    return {
        "provider": {"provider": "mock", "model": "mock"},
        "agent": {
            "mechanism": "fact",
            "memory_scope": "per_agent",
            "embedding_backend": "hash",
        },
        "experiment": {
            "scenario": "fixed_evolving_table",
            "seed": 1,
            "run_mode": mode,
        },
    }


def test_smoke_is_explicitly_not_for_paper() -> None:
    audit = assess_run_admission(_config("smoke"), Path.cwd())
    assert audit["not_for_paper"] is True
    assert audit["paper_eligible_at_start"] is False
    assert audit["blockers"] == []


def test_pilot_rejects_unknown_model_and_hash_embedding() -> None:
    with pytest.raises(ConfigError, match="model_revision") as exc_info:
        assess_run_admission(_config("pilot"), Path.cwd())
    assert "openai_compatible embedding" in str(exc_info.value)


def test_pilot_requires_real_service_smokes_before_run_directory() -> None:
    config = _config("pilot")
    config["provider"] = {
        "provider": "openai_compatible",
        "model": "decision",
        "model_revision": "revision",
        "model_weights_hash": "decision-hash",
        "served_model_name": "decision",
        "service_startup_parameters": {"port": 8000},
    }
    config["agent"] = {
        "mechanism": "fact",
        "memory_scope": "per_agent",
        "embedding_backend": "openai_compatible",
        "embedding_model": "embedding",
        "embedding_revision": "revision",
        "embedding_weights_hash": "embedding-hash",
        "embedding_service_startup_parameters": {"port": 8001},
    }
    config["experiment"]["runtime_verification"] = {
        "decision_service_smoke_passed": False,
        "embedding_service_smoke_passed": False,
    }
    with pytest.raises(ConfigError, match="decision_service_smoke_passed") as exc_info:
        assess_run_admission(config, Path.cwd())
    assert "embedding_service_smoke_passed" in str(exc_info.value)

    config["experiment"]["runtime_verification"] = {
        "decision_service_smoke_passed": True,
        "embedding_service_smoke_passed": True,
    }
    audit = assess_run_admission(config, Path.cwd())
    assert audit["blockers"] == []
    assert audit["not_for_paper"] is True


def test_formal_rejects_unfrozen_protocol_before_run_directory() -> None:
    config = _config("formal")
    with pytest.raises(ConfigError, match="protocol_readiness must be ready") as exc_info:
        assess_run_admission(config, Path.cwd())
    assert "statistical_plan_status must be frozen" in str(exc_info.value)


def test_frozen_preflight_reuses_formal_gates_but_is_not_formal() -> None:
    config = _config("pilot")
    config["experiment"]["frozen_config_preflight"] = True
    with pytest.raises(ConfigError, match="protocol_readiness must be ready") as exc_info:
        assess_run_admission(config, Path.cwd())
    assert "statistical_plan_status must be frozen" in str(exc_info.value)


def test_formal_runtime_lock_uses_verified_model_service_environment() -> None:
    runtime = _runtime_manifest()["metadata"]
    experiment = {
        "formal_runtime_lock": runtime_identity_from_metadata(runtime)
    }
    assert _runtime_lock_blockers(experiment, runtime) == []
    runtime["model_service_runtime"]["vllm_version"] = "different"
    assert "runtime vllm_version mismatch" in _runtime_lock_blockers(
        experiment, runtime
    )[0]
