"""Fail-closed smoke, pilot, and formal experiment admission checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import (
    _effective_secondary_agent_configs,
    _validate_embedding_retrieval_contract,
)
from agentmemeval.core.errors import ConfigError
from agentmemeval.core.memory_mechanism import (
    FACTUAL_MECHANISMS,
    resolve_memory_mechanism,
)
from agentmemeval.evaluation.runtime_lock import (
    FORMAL_RUNTIME_LOCK_FIELDS,
    runtime_identity_from_metadata,
)
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
    frozen_preflight = experiment.get("frozen_config_preflight") is True

    audit: dict[str, Any] = {
        "run_mode": mode,
        "frozen_config_preflight": frozen_preflight,
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
    effective_configs = _effective_agent_configs(config)
    factual_configs = [
        (section, section_config)
        for section, section_config in effective_configs
        if _resolved_memory_mechanism(section, section_config) in FACTUAL_MECHANISMS
    ]
    uses_factual_memory = bool(factual_configs)
    embedding_blocker_start = len(blockers)
    for section, section_config in factual_configs:
        _append_embedding_contract_blockers(
            section,
            section_config,
            blockers,
            cwd=cwd,
        )
    embedding_identity_hashes = {
        section: _embedding_semantic_config_hash(section_config)
        for section, section_config in factual_configs
    }
    if factual_configs:
        primary_embedding_identity_hash = _embedding_semantic_config_hash(agent)
        mismatched_sections = sorted(
            section
            for section, identity_hash in embedding_identity_hashes.items()
            if identity_hash != primary_embedding_identity_hash
        )
        if mismatched_sections:
            blockers.append(
                "effective factual embedding semantic identity must match agent: "
                + ", ".join(mismatched_sections)
            )
        if len(set(embedding_identity_hashes.values())) != 1:
            blockers.append(
                "all effective factual agents must share one embedding semantic identity"
            )
    else:
        primary_embedding_identity_hash = None
    embedding_contract_blockers = blockers[embedding_blocker_start:]

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
        "embedding_identity_complete": not embedding_contract_blockers,
        "embedding_identity_sections": [section for section, _config in factual_configs],
        "embedding_identity_hash_by_section": embedding_identity_hashes,
        "primary_embedding_identity_hash": primary_embedding_identity_hash,
        "embedding_identity_fused": not any(
            "embedding semantic identity" in item for item in embedding_contract_blockers
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
    if mode == "formal" or frozen_preflight:
        readiness = str(experiment.get("protocol_readiness", ""))
        if readiness != "ready":
            blockers.append(
                f"experiment.protocol_readiness must be ready, got {readiness or 'missing'}"
            )
        required_statuses = {
            "retrieval_threshold_status": agent.get("retrieval_threshold_status"),
            "behavior_threshold_status": experiment.get("behavior_threshold_status"),
            "statistical_plan_status": experiment.get("statistical_plan_status"),
        }
        for field, value in required_statuses.items():
            if str(value) != "frozen":
                blockers.append(f"{field} must be frozen for formal runs")
        for section, section_config in factual_configs:
            _append_formal_retrieval_threshold_blockers(
                section,
                section_config,
                blockers,
            )
            _append_formal_retrieval_weight_blockers(
                section,
                section_config,
                blockers,
            )
        if not str(experiment.get("primary_estimand", "")).strip():
            blockers.append("experiment.primary_estimand is required for formal runs")
        if not str(experiment.get("primary_baseline_mechanism", "")).strip():
            blockers.append("experiment.primary_baseline_mechanism is required for formal runs")
        if str(experiment.get("multiple_comparison_method", "")) != "holm":
            blockers.append("experiment.multiple_comparison_method must be holm for formal runs")
        required_seed_pairs = experiment.get("required_seed_pairs")
        if required_seed_pairs is None or int(required_seed_pairs) < 2:
            blockers.append(
                "experiment.required_seed_pairs must be frozen from the independent pilot"
            )
        heterogeneous_timeboxed = (
            str(experiment.get("protocol_variant", ""))
            == "task8b_timeboxed_3seed_qwen_bgem3_v1"
            and verification.get("uniform_hardware_verified") is False
            and verification.get("hardware_uniformity_policy")
            == "recorded_heterogeneous_timeboxed"
        )
        if (
            verification.get("uniform_hardware_verified") is not True
            and not heterogeneous_timeboxed
        ):
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


def _append_formal_retrieval_threshold_blockers(
    section: str,
    config: dict[str, Any],
    blockers: list[str],
) -> None:
    """Require an independently frozen finite cutoff before paper admission."""

    status = str(config.get("retrieval_threshold_status", "")).strip()
    if status != "frozen":
        blockers.append(
            f"{section}.retrieval_threshold_status must be frozen for formal runs"
        )
    minimum = config.get("minimum_retrieval_score")
    if minimum is None:
        blockers.append(
            f"{section}.minimum_retrieval_score must be provided by the frozen "
            "retrieval-threshold approval before formal runs"
        )
        return
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
    ):
        blockers.append(
            f"{section}.minimum_retrieval_score must be a finite numeric frozen value "
            "for formal runs"
        )


def _append_formal_retrieval_weight_blockers(
    section: str,
    config: dict[str, Any],
    blockers: list[str],
) -> None:
    """Reject uncalibrated hybrid-weight heuristics from formal admission."""

    semantic = config.get("semantic_weight", 0.65)
    feature = config.get("feature_weight", 0.35)
    if (
        isinstance(semantic, bool)
        or not isinstance(semantic, (int, float))
        or not math.isfinite(float(semantic))
        or isinstance(feature, bool)
        or not isinstance(feature, (int, float))
        or not math.isfinite(float(feature))
    ):
        blockers.append(f"{section} retrieval weights must be finite numeric values")
        return
    is_original_semantic_only_baseline = float(semantic) == 1.0 and float(feature) == 0.0
    if (
        not is_original_semantic_only_baseline
        and str(config.get("retrieval_weight_status", "")).strip() != "frozen"
    ):
        blockers.append(
            f"{section}.retrieval_weight_status must be frozen before hybrid retrieval "
            "weights can enter formal runs"
        )


def _uses_factual_memory(config: dict[str, Any]) -> bool:
    return any(
        _resolved_memory_mechanism(section, section_config) in FACTUAL_MECHANISMS
        for section, section_config in _effective_agent_configs(config)
    )


def _resolved_memory_mechanism(
    section: str,
    config: dict[str, Any],
) -> str:
    try:
        return resolve_memory_mechanism(config)
    except ValueError as exc:
        raise ConfigError(f"{section}: {exc}") from exc


def _effective_agent_configs(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    primary = config.get("agent", {})
    configs = [("agent", dict(primary) if isinstance(primary, dict) else {})]
    configs.extend(_effective_secondary_agent_configs(config))
    return configs


def _append_embedding_contract_blockers(
    section: str,
    config: dict[str, Any],
    blockers: list[str],
    *,
    cwd: Path,
) -> None:
    """Seal every effective factual agent's complete semantic backend contract."""

    backend = str(config.get("embedding_backend", "hash"))
    if backend not in {"openai_compatible", "bgem3_hybrid_http"}:
        blockers.append(
            f"{section}: pilot/formal factual memory requires openai_compatible "
            "embedding or bgem3_hybrid_http"
        )
        return
    try:
        _validate_embedding_retrieval_contract(section, config)
    except ConfigError as exc:
        blockers.append(str(exc))
        return
    common_fields = (
        "embedding_model",
        "embedding_revision",
        "embedding_weights_hash",
        "embedding_service_startup_parameters",
    )
    _require_identity(config, common_fields, section, blockers)
    weights_hash = config.get("embedding_weights_hash")
    if not _is_sha256(weights_hash):
        blockers.append(f"{section}.embedding_weights_hash must be a SHA-256 hex digest")
    if backend == "openai_compatible":
        return
    startup = config.get("embedding_service_startup_parameters")
    if isinstance(startup, dict) and startup.get("server_source_sha256") is not None:
        service_script = Path(str(startup.get("service_script", "")))
        if not service_script.is_absolute():
            service_script = cwd / service_script
        if not service_script.is_file():
            blockers.append(f"{section}.embedding service_script does not exist")
        else:
            actual_source_sha256 = hashlib.sha256(service_script.read_bytes()).hexdigest()
            if actual_source_sha256 != str(startup["server_source_sha256"]).lower():
                blockers.append(
                    f"{section}.embedding server_source_sha256 does not match service_script"
                )


def _embedding_semantic_config_hash(config: dict[str, Any]) -> str:
    """Hash semantic embedding identity while excluding branch-local cache paths."""

    fields = (
        "embedding_backend",
        "embedding_model",
        "embedding_revision",
        "embedding_weights_hash",
        "embedding_tokenizer_revision",
        "embedding_query_instruction",
        "embedding_query_policy",
        "embedding_hybrid_weights",
        "embedding_candidate_depth",
        "embedding_colbert_rerank_depth",
        "embedding_final_top_k_policy",
        "embedding_cache_schema_version",
        "embedding_base_url_env",
        "embedding_output_dimensions",
        "embedding_service_startup_parameters",
    )
    payload = {field: config.get(field) for field in fields}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


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


def _runtime_lock_blockers(experiment: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    lock = experiment.get("formal_runtime_lock")
    if not isinstance(lock, dict):
        return ["experiment.formal_runtime_lock is required"]
    service_runtime = dict(runtime.get("model_service_runtime", {}))
    blockers: list[str] = []
    if service_runtime.get("status") != "verified":
        blockers.append("model service runtime probe must be verified")
    observed = runtime_identity_from_metadata(runtime)
    driver_policy = lock.get("gpu_driver_policy")
    if driver_policy is not None:
        if driver_policy != "informational_only":
            blockers.append(
                "experiment.formal_runtime_lock.gpu_driver_policy must be informational_only"
            )
        elif observed.get("gpu_driver") in {None, ""}:
            blockers.append("runtime gpu_driver is required for informational recording")
    gpu_name_policy = lock.get("gpu_name_policy")
    if gpu_name_policy not in {None, "informational_only"}:
        blockers.append(
            "experiment.formal_runtime_lock.gpu_name_policy must be informational_only"
        )
    for field in FORMAL_RUNTIME_LOCK_FIELDS:
        if field == "gpu_driver" and driver_policy is not None:
            continue
        if field == "gpu_name" and gpu_name_policy == "informational_only":
            if observed.get("gpu_name") in {None, ""}:
                blockers.append("runtime gpu_name is required for informational recording")
            continue
        actual = observed.get(field)
        expected = lock.get(field)
        if expected in {None, ""}:
            blockers.append(f"experiment.formal_runtime_lock.{field} is required")
        elif str(expected) != str(actual):
            blockers.append(f"runtime {field} mismatch: expected {expected}, observed {actual}")
    return blockers
