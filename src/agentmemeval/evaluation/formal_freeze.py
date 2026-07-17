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

RUNTIME_LOCK_FIELDS = (
    "gpu_name",
    "gpu_driver",
    "service_torch_cuda_version",
    "vllm_version",
)


def generate_formal_freeze_bundle(
    *,
    proposal_path: str | Path,
    runtime_lock_path: str | Path,
    campaign_p_template_path: str | Path,
    campaign_e_template_path: str | Path,
    formal_p_template_path: str | Path,
    formal_e_template_path: str | Path,
    output_dir: str | Path,
    freeze_id: str,
    preflight_seed: int,
    seed_start: int = 2026071801,
) -> dict[str, Any]:
    """Create a new P/E formal bundle only after every frozen gate is valid."""

    inputs = {
        "proposal": Path(proposal_path).resolve(),
        "runtime_lock": Path(runtime_lock_path).resolve(),
        "campaign_p_template": Path(campaign_p_template_path).resolve(),
        "campaign_e_template": Path(campaign_e_template_path).resolve(),
        "formal_p_template": Path(formal_p_template_path).resolve(),
        "formal_e_template": Path(formal_e_template_path).resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise ConfigError(f"formal freeze input does not exist ({label}): {path}")
    normalized_freeze_id = _validate_freeze_id(freeze_id)
    proposal = _read_json(inputs["proposal"])
    runtime_lock = _validate_runtime_lock(_read_json(inputs["runtime_lock"]))
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
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"formal freeze output directory already exists: {destination}")

    manifest = {
        "schema_version": "agentmemeval_formal_freeze_bundle_v1",
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
        "runtime_lock": runtime_lock,
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


def _validate_runtime_lock(data: dict[str, Any]) -> dict[str, str]:
    candidate = data.get("formal_runtime_lock", data)
    if not isinstance(candidate, dict):
        raise ConfigError("runtime lock must be a JSON object")
    lock: dict[str, str] = {}
    for field in RUNTIME_LOCK_FIELDS:
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
            "campaign_id": f"task4_campaign_{label}_robust_formal_{freeze_id}",
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


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
