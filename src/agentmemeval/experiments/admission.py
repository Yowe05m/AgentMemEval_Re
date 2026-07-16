"""Fail-closed smoke, pilot, and formal experiment admission checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentmemeval.core.errors import ConfigError
from agentmemeval.storage.artifacts import collect_runtime_metadata

RUN_MODES = {"smoke", "pilot", "formal"}


def assess_run_admission(config: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Validate run-purpose prerequisites before any output directory is created."""

    experiment = dict(config.get("experiment", {}))
    provider = dict(config.get("provider", {}))
    agent = dict(config.get("agent", {}))
    mode = str(experiment.get("run_mode", "smoke"))
    if mode not in RUN_MODES:
        raise ConfigError(f"未知 experiment.run_mode：{mode}")

    audit: dict[str, Any] = {
        "run_mode": mode,
        "paper_eligible_at_start": False,
        "not_for_paper": mode != "formal",
        "blockers": [],
        "checks": {},
    }
    if mode == "smoke":
        audit["checks"] = {"engineering_smoke_only": True}
        return audit

    blockers: list[str] = []
    _require_identity(
        provider,
        (
            "model",
            "model_revision",
            "model_weights_hash",
            "served_model_name",
            "service_startup_parameters",
        ),
        "provider",
        blockers,
    )
    uses_factual_memory = _uses_factual_memory(config)
    if uses_factual_memory:
        if str(agent.get("embedding_backend", "hash")) != "openai_compatible":
            blockers.append("pilot/formal factual memory requires openai_compatible embedding")
        _require_identity(
            agent,
            (
                "embedding_model",
                "embedding_revision",
                "embedding_weights_hash",
                "embedding_service_startup_parameters",
            ),
            "agent",
            blockers,
        )

    verification = dict(experiment.get("runtime_verification", {}))
    if verification.get("decision_service_smoke_passed") is not True:
        blockers.append(
            "experiment.runtime_verification.decision_service_smoke_passed must be true"
        )
    if uses_factual_memory and verification.get("embedding_service_smoke_passed") is not True:
        blockers.append(
            "experiment.runtime_verification.embedding_service_smoke_passed must be true"
        )
    audit["checks"] = {
        "decision_model_identity_complete": not any(
            item.startswith("provider.") for item in blockers
        ),
        "embedding_identity_complete": not any(
            item.startswith("agent.embedding_") for item in blockers
        ),
        "strategy_risk_gate_disabled": True,
        "memory_scope_per_agent_only": True,
        "decision_service_smoke_passed": (
            verification.get("decision_service_smoke_passed") is True
        ),
        "embedding_service_smoke_passed": (
            verification.get("embedding_service_smoke_passed") is True
            if uses_factual_memory
            else "not_required"
        ),
    }
    if mode == "formal":
        readiness = str(experiment.get("protocol_readiness", ""))
        if readiness != "ready":
            blockers.append(
                "experiment.protocol_readiness must be ready, "
                f"got {readiness or 'missing'}"
            )
        required_statuses = {
            "retrieval_threshold_status": agent.get("retrieval_threshold_status"),
            "behavior_threshold_status": experiment.get("behavior_threshold_status"),
            "statistical_plan_status": experiment.get("statistical_plan_status"),
        }
        for field, value in required_statuses.items():
            if str(value) != "frozen":
                blockers.append(f"{field} must be frozen for formal runs")
        if not str(experiment.get("primary_estimand", "")).strip():
            blockers.append("experiment.primary_estimand is required for formal runs")
        if not str(experiment.get("primary_baseline_mechanism", "")).strip():
            blockers.append(
                "experiment.primary_baseline_mechanism is required for formal runs"
            )
        if str(experiment.get("multiple_comparison_method", "")) != "holm":
            blockers.append(
                "experiment.multiple_comparison_method must be holm for formal runs"
            )
        required_seed_pairs = experiment.get("required_seed_pairs")
        if required_seed_pairs is None or int(required_seed_pairs) < 2:
            blockers.append(
                "experiment.required_seed_pairs must be frozen from the independent pilot"
            )
        if verification.get("uniform_hardware_verified") is not True:
            blockers.append(
                "experiment.runtime_verification.uniform_hardware_verified must be true"
            )
        runtime = collect_runtime_metadata(config, cwd)
        code = dict(runtime.get("code", {}))
        if code.get("commit") in {None, "", "unknown"}:
            blockers.append("runtime code commit is unknown")
        if code.get("dirty") is not False:
            blockers.append("formal run requires a clean Git worktree")
        blockers.extend(_runtime_lock_blockers(experiment, runtime))
        audit["runtime_observed"] = runtime

    audit["blockers"] = blockers
    if blockers:
        joined = "; ".join(blockers)
        raise ConfigError(f"{mode} 实验准入失败，尚未创建 run 目录：{joined}")
    audit["paper_eligible_at_start"] = mode == "formal"
    audit["not_for_paper"] = mode != "formal"
    return audit


def _uses_factual_memory(config: dict[str, Any]) -> bool:
    mechanisms = {
        str(dict(config.get(section, {})).get("mechanism", ""))
        for section in ("agent", "opponent_agent", "heldout_agent")
    }
    roster = dict(config.get("experiment", {})).get("agent_roster", [])
    mechanisms.update(
        str(item.get("mechanism", ""))
        for item in roster
        if isinstance(item, dict)
    )
    return bool(mechanisms & {"fact", "fact_expr_sync", "fact_expr_async"})


def _require_identity(
    section: dict[str, Any],
    fields: tuple[str, ...],
    prefix: str,
    blockers: list[str],
) -> None:
    for field in fields:
        value = section.get(field)
        if value is None or value == "" or value == {} or value == []:
            blockers.append(f"{prefix}.{field} is required")


def _runtime_lock_blockers(
    experiment: dict[str, Any], runtime: dict[str, Any]
) -> list[str]:
    lock = experiment.get("formal_runtime_lock")
    if not isinstance(lock, dict):
        return ["experiment.formal_runtime_lock is required"]
    devices = dict(runtime.get("gpu", {})).get("devices", [])
    observed_gpu = devices[0] if isinstance(devices, list) and devices else {}
    service_runtime = dict(runtime.get("model_service_runtime", {}))
    blockers: list[str] = []
    if service_runtime.get("status") != "verified":
        blockers.append("model service runtime probe must be verified")
    observed = {
        "gpu_name": observed_gpu.get("name"),
        "gpu_driver": observed_gpu.get("driver"),
        "service_torch_cuda_version": service_runtime.get("torch_cuda_version"),
        "vllm_version": service_runtime.get("vllm_version"),
    }
    for field, actual in observed.items():
        expected = lock.get(field)
        if expected in {None, ""}:
            blockers.append(f"experiment.formal_runtime_lock.{field} is required")
        elif str(expected) != str(actual):
            blockers.append(f"runtime {field} mismatch: expected {expected}, observed {actual}")
    return blockers
