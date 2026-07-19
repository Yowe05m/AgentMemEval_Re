"""Fail-closed generation of immutable formal experiment configurations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.config.loader import dump_yaml, load_config, validate_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.aggregation import validate_runtime_homogeneity
from agentmemeval.evaluation.pilot import build_pilot_freeze_proposal_from_paths
from agentmemeval.evaluation.runtime_lock import (
    CONFIG_BOUND_RUNTIME_LOCK_FIELDS,
    FORMAL_RUNTIME_LOCK_FIELDS,
    build_formal_runtime_lock_from_manifest,
    configured_runtime_identity,
)


def generate_formal_freeze_bundle(
    *,
    proposal_path: str | Path,
    runtime_lock_path: str | Path,
    campaign_p_template_path: str | Path,
    campaign_e_template_path: str | Path,
    formal_p_template_path: str | Path,
    formal_e_template_path: str | Path,
    strict_p_template_path: str | Path,
    strict_p_campaign_template_path: str | Path,
    output_dir: str | Path,
    freeze_id: str,
    preflight_seed: int,
    seed_start: int = 2026071801,
) -> dict[str, Any]:
    """Create frozen P/E plus paired strict sensitivity after every gate is valid."""

    inputs = {
        "proposal": Path(proposal_path).resolve(),
        "runtime_lock": Path(runtime_lock_path).resolve(),
        "campaign_p_template": Path(campaign_p_template_path).resolve(),
        "campaign_e_template": Path(campaign_e_template_path).resolve(),
        "formal_p_template": Path(formal_p_template_path).resolve(),
        "formal_e_template": Path(formal_e_template_path).resolve(),
        "strict_p_template": Path(strict_p_template_path).resolve(),
        "strict_p_campaign_template": Path(
            strict_p_campaign_template_path
        ).resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise ConfigError(f"formal freeze input does not exist ({label}): {path}")
    normalized_freeze_id = _validate_freeze_id(freeze_id)
    proposal = _read_json(inputs["proposal"])
    proposal_rebuild_sha256 = _validate_proposal_source_rebuild(proposal)
    runtime_lock = _validate_runtime_lock(
        _read_json(inputs["runtime_lock"]),
        proposal,
    )
    required_seed_pairs, behavior_thresholds, retrieval_score = _validate_proposal(
        proposal
    )
    try:
        first_seed = int(seed_start)
    except (TypeError, ValueError) as exc:
        raise ConfigError("formal seed_start must be an integer") from exc
    if first_seed < 1:
        raise ConfigError("formal seed_start must be positive")
    seeds = [first_seed + offset for offset in range(required_seed_pairs)]
    if len(set(seeds)) != required_seed_pairs:
        raise ConfigError("generated formal seeds are not unique")
    pilot_seeds = _pilot_seeds(proposal)
    try:
        validated_preflight_seed = int(preflight_seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError("preflight_seed must be an integer") from exc
    if validated_preflight_seed < 1:
        raise ConfigError("preflight_seed must be positive")
    if validated_preflight_seed in set(seeds):
        raise ConfigError("preflight_seed must not overlap formal seeds")
    if validated_preflight_seed in pilot_seeds:
        raise ConfigError("preflight_seed must not overlap calibration Pilot seeds")

    source_hashes = {
        label: _sha256(path) for label, path in sorted(inputs.items())
    }
    formal_configs = {
        "p": _build_formal_config(
            inputs["formal_p_template"],
            normalized_freeze_id,
            required_seed_pairs,
            behavior_thresholds,
            retrieval_score,
            runtime_lock,
            seeds[0],
            source_hashes,
        ),
        "e": _build_formal_config(
            inputs["formal_e_template"],
            normalized_freeze_id,
            required_seed_pairs,
            behavior_thresholds,
            retrieval_score,
            runtime_lock,
            seeds[0],
            source_hashes,
        ),
    }
    preflight_configs = {
        label: _build_preflight_config(config, validated_preflight_seed)
        for label, config in formal_configs.items()
    }
    strict_sensitivity_config = _build_strict_sensitivity_config(
        inputs["strict_p_template"],
        normalized_freeze_id,
        required_seed_pairs,
        behavior_thresholds,
        runtime_lock,
        seeds[0],
        source_hashes,
    )
    names = {
        "formal_p": f"task4_campaign_p_robust_formal_{normalized_freeze_id}.yaml",
        "formal_e": f"task4_campaign_e_robust_formal_{normalized_freeze_id}.yaml",
        "campaign_p": f"task4_campaign_p_robust_formal_{normalized_freeze_id}_campaign.yaml",
        "campaign_e": f"task4_campaign_e_robust_formal_{normalized_freeze_id}_campaign.yaml",
        "preflight_p": (
            f"task4_campaign_p_frozen_preflight_{normalized_freeze_id}.yaml"
        ),
        "preflight_e": (
            f"task4_campaign_e_frozen_preflight_{normalized_freeze_id}.yaml"
        ),
        "preflight_campaign_p": (
            f"task4_campaign_p_frozen_preflight_{normalized_freeze_id}_campaign.yaml"
        ),
        "preflight_campaign_e": (
            f"task4_campaign_e_frozen_preflight_{normalized_freeze_id}_campaign.yaml"
        ),
        "strict_p_sensitivity": (
            "task4_campaign_p_strict_model_substituted_sensitivity_"
            f"{normalized_freeze_id}.yaml"
        ),
        "strict_campaign_p": (
            "task4_campaign_p_strict_model_substituted_sensitivity_"
            f"{normalized_freeze_id}_campaign.yaml"
        ),
        "manifest": f"formal_freeze_manifest_{normalized_freeze_id}.json",
    }
    campaigns = {
        "p": _build_campaign(
            inputs["campaign_p_template"],
            expected_design="mixed_table",
            freeze_id=normalized_freeze_id,
            base_name=names["formal_p"],
            seeds=seeds,
            label="p",
        ),
        "e": _build_campaign(
            inputs["campaign_e_template"],
            expected_design="target_vs_seven_no_memory",
            freeze_id=normalized_freeze_id,
            base_name=names["formal_e"],
            seeds=seeds,
            label="e",
        ),
    }
    preflight_campaigns = {
        "p": _build_campaign(
            inputs["campaign_p_template"],
            expected_design="mixed_table",
            freeze_id=normalized_freeze_id,
            base_name=names["preflight_p"],
            seeds=[validated_preflight_seed],
            label="p_preflight",
            protocol_label="frozen_config_preflight_not_for_paper",
            minimum_seeds=1,
        ),
        "e": _build_campaign(
            inputs["campaign_e_template"],
            expected_design="target_vs_seven_no_memory",
            freeze_id=normalized_freeze_id,
            base_name=names["preflight_e"],
            seeds=[validated_preflight_seed],
            label="e_preflight",
            protocol_label="frozen_config_preflight_not_for_paper",
            minimum_seeds=1,
        ),
    }
    strict_sensitivity_campaign = _build_campaign(
        inputs["strict_p_campaign_template"],
        expected_design="mixed_table",
        freeze_id=normalized_freeze_id,
        base_name=names["strict_p_sensitivity"],
        seeds=seeds,
        label="p_strict_sensitivity",
        protocol_label=(
            "strict_paper_replication_model_substituted_sensitivity_"
            "frozen_not_main_table"
        ),
        campaign_id=(
            "task4_campaign_p_strict_model_substituted_sensitivity_"
            f"{normalized_freeze_id}"
        ),
    )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"formal freeze output directory already exists: {destination}")

    manifest = {
        "schema_version": "agentmemeval_formal_freeze_bundle_v3",
        "freeze_id": normalized_freeze_id,
        "status": "immutable_formal_configs_generated",
        "required_seed_pairs": required_seed_pairs,
        "seed_rule": {
            "method": "contiguous_integer_sequence",
            "seed_start": first_seed,
            "count": required_seed_pairs,
            "no_silent_resource_cap": True,
        },
        "seeds": seeds,
        "calibration_pilot_seeds": sorted(pilot_seeds),
        "preflight_seed": validated_preflight_seed,
        "preflight_policy": {
            "paper_eligible": False,
            "uses_formal_admission_checks": True,
            "allowed_config_differences_from_formal": [
                "experiment.seed",
                "experiment.run_mode",
                "experiment.frozen_config_preflight",
            ],
        },
        "strict_sensitivity_policy": {
            "paper_eligible": False,
            "run_mode": "pilot",
            "model_substituted": True,
            "paired_with_robust_formal_seeds": True,
            "retrieval_policy": "paper_exact_unthresholded",
            "main_table_inclusion": "prohibited",
        },
        "runtime_lock": runtime_lock,
        "proposal_source_rebuild": {
            "verified": True,
            "canonical_sha256": proposal_rebuild_sha256,
        },
        "source_sha256": source_hashes,
        "files": names,
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_new_text(destination / names["formal_p"], dump_yaml(formal_configs["p"]))
    _write_new_text(destination / names["formal_e"], dump_yaml(formal_configs["e"]))
    _write_new_text(destination / names["campaign_p"], dump_yaml(campaigns["p"]))
    _write_new_text(destination / names["campaign_e"], dump_yaml(campaigns["e"]))
    _write_new_text(
        destination / names["preflight_p"], dump_yaml(preflight_configs["p"])
    )
    _write_new_text(
        destination / names["preflight_e"], dump_yaml(preflight_configs["e"])
    )
    _write_new_text(
        destination / names["preflight_campaign_p"],
        dump_yaml(preflight_campaigns["p"]),
    )
    _write_new_text(
        destination / names["preflight_campaign_e"],
        dump_yaml(preflight_campaigns["e"]),
    )
    _write_new_text(
        destination / names["strict_p_sensitivity"],
        dump_yaml(strict_sensitivity_config),
    )
    _write_new_text(
        destination / names["strict_campaign_p"],
        dump_yaml(strict_sensitivity_campaign),
    )
    _write_new_text(
        destination / names["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return {"output_dir": str(destination), **manifest}


def _pilot_seeds(proposal: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for label in ("campaign_p_evidence", "campaign_e_evidence"):
        evidence = proposal.get(label)
        if not isinstance(evidence, dict) or not isinstance(
            evidence.get("completed_seeds"), list
        ):
            raise ConfigError(f"pilot proposal is missing {label}.completed_seeds")
        for value in evidence["completed_seeds"]:
            try:
                seeds.add(int(value))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid calibration Pilot seed: {value}") from exc
    if not seeds:
        raise ConfigError("pilot proposal contains no completed calibration seeds")
    return seeds


def _validate_proposal(
    proposal: dict[str, Any],
) -> tuple[int, dict[str, Any], float]:
    if proposal.get("schema_version") != "agentmemeval_pilot_freeze_proposal_v1":
        raise ConfigError("unsupported or missing pilot freeze proposal schema")
    if proposal.get("status") != "ready_to_generate_immutable_formal_configs":
        raise ConfigError("pilot freeze proposal is not ready; formal generation is NO-GO")
    if proposal.get("blockers") != []:
        raise ConfigError("pilot freeze proposal contains blockers")
    power = proposal.get("power_plan")
    if not isinstance(power, dict) or power.get("blockers") != []:
        raise ConfigError("pilot power plan is missing or blocked")
    if power.get("no_silent_resource_cap") is not True:
        raise ConfigError("pilot power plan must prohibit silent resource caps")
    required = proposal.get("required_seed_pairs")
    try:
        required_seed_pairs = int(required)
    except (TypeError, ValueError) as exc:
        raise ConfigError("pilot required_seed_pairs is missing or invalid") from exc
    if required_seed_pairs < 2:
        raise ConfigError("pilot required_seed_pairs must be at least 2")
    if required != power.get("required_seed_pairs_primary_max_across_p_and_e"):
        raise ConfigError("proposal and power-plan seed requirements disagree")

    behavior = proposal.get("behavior_freeze")
    if not isinstance(behavior, dict) or behavior.get("status") != "frozen":
        raise ConfigError("pilot behavior thresholds are not frozen")
    if behavior.get("blockers") != []:
        raise ConfigError("pilot behavior freeze contains blockers")
    thresholds = behavior.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ConfigError("pilot behavior thresholds are empty")

    retrieval = proposal.get("retrieval_freeze")
    if not isinstance(retrieval, dict):
        raise ConfigError("pilot retrieval freeze is missing")
    if retrieval.get("retrieval_threshold_status") != "frozen":
        raise ConfigError("pilot retrieval threshold is not frozen")
    score = retrieval.get("minimum_retrieval_score")
    if score is None:
        raise ConfigError("pilot minimum retrieval score is missing")
    return required_seed_pairs, copy.deepcopy(thresholds), float(score)


def _validate_proposal_source_rebuild(proposal: dict[str, Any]) -> str:
    p_aggregate = _evidence_file(
        proposal,
        "campaign_p_aggregate_evidence",
    )
    e_aggregate = _evidence_file(
        proposal,
        "campaign_e_aggregate_evidence",
    )
    review = _evidence_file(
        proposal,
        "retrieval_review_evidence",
    )
    p_campaign = _evidence_directory(proposal, "campaign_p_evidence")
    e_campaign = _evidence_directory(proposal, "campaign_e_evidence")
    runtime_evidence = proposal.get("runtime_equivalence_evidence")
    runtime_path: Path | None = None
    if runtime_evidence is not None:
        runtime_path = _evidence_file(
            proposal,
            "runtime_equivalence_evidence",
        )
    try:
        rebuilt = build_pilot_freeze_proposal_from_paths(
            p_aggregate,
            e_aggregate,
            p_campaign,
            e_campaign,
            review,
            runtime_path,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(
            "pilot freeze proposal source rebuild failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if rebuilt != proposal:
        raise ConfigError(
            "pilot freeze proposal differs from deterministic source rebuild"
        )
    return _json_sha256(rebuilt)


def _evidence_file(proposal: dict[str, Any], key: str) -> Path:
    evidence = proposal.get(key)
    if not isinstance(evidence, dict):
        raise ConfigError(f"pilot proposal is missing {key}")
    path = Path(str(evidence.get("path", ""))).resolve()
    if not path.is_file():
        raise ConfigError(f"pilot proposal evidence file is missing ({key}): {path}")
    expected_hash = str(evidence.get("sha256", ""))
    if not _is_sha256(expected_hash) or _sha256(path) != expected_hash:
        raise ConfigError(f"pilot proposal evidence hash mismatch: {key}")
    return path


def _evidence_directory(proposal: dict[str, Any], key: str) -> Path:
    evidence = proposal.get(key)
    if not isinstance(evidence, dict):
        raise ConfigError(f"pilot proposal is missing {key}")
    path = Path(str(evidence.get("campaign_dir", ""))).resolve()
    if not path.is_dir():
        raise ConfigError(f"pilot proposal campaign directory is missing ({key}): {path}")
    return path


def _validate_runtime_lock(
    data: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, str]:
    if data.get("schema_version") != "task4_formal_runtime_lock_v2":
        raise ConfigError("formal runtime lock schema is not V2")
    if data.get("status") != "verified_from_real_service_run_manifest":
        raise ConfigError("formal runtime lock is not verified from a real-service run")
    source = data.get("source_manifest")
    if not isinstance(source, dict):
        raise ConfigError("formal runtime lock source manifest evidence is missing")
    source_path = Path(str(source.get("path", ""))).resolve()
    if not source_path.is_file():
        raise ConfigError(f"formal runtime lock source manifest is missing: {source_path}")
    expected_hash = str(source.get("sha256", ""))
    if not _is_sha256(expected_hash) or _sha256(source_path) != expected_hash:
        raise ConfigError("formal runtime lock source manifest hash mismatch")
    rebuilt = build_formal_runtime_lock_from_manifest(source_path)
    if rebuilt != data:
        raise ConfigError("formal runtime lock differs from source manifest rebuild")
    leaf_manifest_paths = {
        (Path(str(item.get("run_dir", ""))) / "manifest.json").resolve()
        for key in ("campaign_p_leaf_evidence", "campaign_e_leaf_evidence")
        for item in proposal.get(key, [])
        if isinstance(item, dict)
    }
    if source_path not in leaf_manifest_paths:
        raise ConfigError(
            "formal runtime lock source is not a completed P/E Pilot leaf manifest"
        )
    source_identity = validate_runtime_homogeneity(
        [_read_json(source_path)]
    ).get("identity")
    if not isinstance(source_identity, dict):
        raise ConfigError("formal runtime lock source identity cannot be reconstructed")
    source_non_code = {
        key: value for key, value in source_identity.items() if key != "code"
    }
    for key in (
        "campaign_p_aggregate_evidence",
        "campaign_e_aggregate_evidence",
    ):
        aggregate = _read_json(_evidence_file(proposal, key))
        identity = dict(aggregate.get("runtime_homogeneity", {})).get("identity")
        if not isinstance(identity, dict):
            raise ConfigError(f"Pilot aggregate runtime identity is missing: {key}")
        non_code = {
            name: value for name, value in identity.items() if name != "code"
        }
        if _json_sha256(non_code) != _json_sha256(source_non_code):
            raise ConfigError(
                "formal runtime lock source identity differs from P/E Pilot runtime"
            )
    candidate = data.get("formal_runtime_lock", data)
    if not isinstance(candidate, dict):
        raise ConfigError("runtime lock must be a JSON object")
    lock: dict[str, str] = {}
    for field in FORMAL_RUNTIME_LOCK_FIELDS:
        value = candidate.get(field)
        if value is None or not str(value).strip():
            raise ConfigError(f"runtime lock field is required: {field}")
        lock[field] = str(value)
    return lock


def _build_formal_config(
    template_path: Path,
    freeze_id: str,
    required_seed_pairs: int,
    behavior_thresholds: dict[str, Any],
    retrieval_score: float,
    runtime_lock: dict[str, str],
    first_seed: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    config = _without_internal_keys(load_config(template_path))
    experiment = dict(config["experiment"])
    agent = dict(config["agent"])
    experiment.update(
        {
            "seed": first_seed,
            "run_mode": "formal",
            "protocol_readiness": "ready",
            "behavior_threshold_status": "frozen",
            "behavior_thresholds": copy.deepcopy(behavior_thresholds),
            "statistical_plan_status": "frozen",
            "required_seed_pairs": required_seed_pairs,
            "runtime_verification": {
                "decision_service_smoke_passed": True,
                "embedding_service_smoke_passed": True,
                "uniform_hardware_verified": True,
            },
            "formal_runtime_lock": copy.deepcopy(runtime_lock),
            "formal_freeze_id": freeze_id,
            "formal_freeze_source_sha256": copy.deepcopy(source_hashes),
        }
    )
    agent.update(
        {
            "retrieval_threshold_status": "frozen",
            "minimum_retrieval_score": retrieval_score,
        }
    )
    config["experiment"] = experiment
    config["agent"] = agent
    configured_identity = configured_runtime_identity(config)
    mismatches = [
        field
        for field in CONFIG_BOUND_RUNTIME_LOCK_FIELDS
        if configured_identity.get(field) != runtime_lock.get(field)
    ]
    if mismatches:
        raise ConfigError(
            "formal template identity differs from runtime lock: "
            f"{sorted(mismatches)}"
        )
    validate_config(config)
    return config


def _build_preflight_config(
    formal_config: dict[str, Any], preflight_seed: int
) -> dict[str, Any]:
    config = copy.deepcopy(formal_config)
    experiment = dict(config["experiment"])
    experiment.update(
        {
            "seed": preflight_seed,
            "run_mode": "pilot",
            "frozen_config_preflight": True,
        }
    )
    config["experiment"] = experiment
    validate_config(config)
    return config


def _build_strict_sensitivity_config(
    template_path: Path,
    freeze_id: str,
    required_seed_pairs: int,
    behavior_thresholds: dict[str, Any],
    runtime_lock: dict[str, str],
    first_seed: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Freeze a paired, model-substituted strict sensitivity without main-table status."""

    config = _without_internal_keys(load_config(template_path))
    experiment = dict(config["experiment"])
    experiment.update(
        {
            "seed": first_seed,
            "run_mode": "pilot",
            "protocol_readiness": (
                "frozen_model_substituted_sensitivity_not_for_main_table"
            ),
            "behavior_threshold_status": "frozen",
            "behavior_thresholds": copy.deepcopy(behavior_thresholds),
            "statistical_plan_status": "frozen",
            "required_seed_pairs": required_seed_pairs,
            "runtime_verification": {
                "decision_service_smoke_passed": True,
                "embedding_service_smoke_passed": True,
                "uniform_hardware_verified": True,
            },
            "formal_runtime_lock": copy.deepcopy(runtime_lock),
            "formal_freeze_id": freeze_id,
            "formal_freeze_source_sha256": copy.deepcopy(source_hashes),
            "strict_sensitivity_pairing": {
                "paired_with": "paper_robust_formal",
                "uses_identical_seed_list": True,
                "paper_eligible": False,
                "model_substituted": True,
            },
        }
    )
    config["experiment"] = experiment
    configured_identity = configured_runtime_identity(config)
    mismatches = [
        field
        for field in CONFIG_BOUND_RUNTIME_LOCK_FIELDS
        if configured_identity.get(field) != runtime_lock.get(field)
    ]
    if mismatches:
        raise ConfigError(
            "strict sensitivity template identity differs from runtime lock: "
            f"{sorted(mismatches)}"
        )
    validate_config(config)
    return config


def _build_campaign(
    template_path: Path,
    *,
    expected_design: str,
    freeze_id: str,
    base_name: str,
    seeds: list[int],
    label: str,
    protocol_label: str = "paper_robust_formal_frozen",
    minimum_seeds: int = 2,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("campaign"), dict):
        raise ConfigError(f"campaign template lacks campaign mapping: {template_path}")
    campaign = copy.deepcopy(raw["campaign"])
    if campaign.get("design") != expected_design:
        raise ConfigError(
            f"campaign {label.upper()} template design must be {expected_design}"
        )
    campaign.update(
        {
            "campaign_id": campaign_id
            or f"task4_campaign_{label}_robust_formal_{freeze_id}",
            "protocol_label": protocol_label,
            "base_experiment_config": base_name,
            "seeds": list(seeds),
        }
    )
    if (
        len(campaign["seeds"]) < minimum_seeds
        or len(set(campaign["seeds"])) != len(seeds)
    ):
        raise ConfigError("generated formal campaign seed matrix is invalid")
    return {"campaign": campaign}


def _validate_freeze_id(value: str) -> str:
    candidate = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", candidate):
        raise ConfigError("freeze_id must be a short filesystem-safe identifier")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON input: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"JSON input must be an object: {path}")
    return data


def _without_internal_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_internal_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_without_internal_keys(item) for item in value]
    return copy.deepcopy(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
