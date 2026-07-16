"""Build provenance-rich run maps without mixing Pilot, failed, or formal evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

RUN_MAP_FIELDS = (
    "campaign_id",
    "condition_id",
    "target_mechanism",
    "seed",
    "attempt",
    "state_status",
    "run_id",
    "source_run_dir",
    "resolved_run_dir",
    "run_mode",
    "protocol_variant",
    "run_validity_status",
    "execution_valid",
    "paper_eligible",
    "classification",
    "formal_main_table_eligible",
    "exclusion_reasons",
    "campaign_manifest_sha256",
    "state_tsv_sha256",
)
REQUIRED_LEAF_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "metrics.json",
    "protocol_audit.json",
    "experiment_result.json",
)


def build_run_map(
    campaign_dirs: list[str | Path],
    output_csv: str | Path,
    exclusion_json: str | Path,
) -> dict[str, Any]:
    """Write one latest lifecycle row per attempt plus formal-main exclusion reasons."""

    mapped: list[dict[str, Any]] = []
    for raw_dir in campaign_dirs:
        campaign_dir = Path(raw_dir).resolve()
        manifest_path = campaign_dir / "campaign_manifest.json"
        state_path = campaign_dir / "state.tsv"
        manifest = _read_json(manifest_path)
        campaign_id = str(manifest.get("campaign_id") or campaign_dir.name)
        with state_path.open("r", encoding="utf-8", newline="") as handle:
            states = list(csv.DictReader(handle, delimiter="\t"))
        latest: dict[tuple[str, str, str], dict[str, str]] = {}
        for state in states:
            key = (
                str(state.get("condition_id", "")),
                str(state.get("seed", "")),
                str(state.get("attempt", "")),
            )
            latest[key] = state
        for state in latest.values():
            mapped.append(
                _map_attempt(
                    campaign_dir,
                    campaign_id,
                    state,
                    manifest_sha=_sha256(manifest_path),
                    state_sha=_sha256(state_path),
                )
            )
    mapped.sort(
        key=lambda row: (
            row["campaign_id"],
            row["condition_id"],
            int(row["seed"] or 0),
            int(row["attempt"] or 0),
        )
    )
    output = Path(output_csv).resolve()
    exclusions = Path(exclusion_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusions.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_MAP_FIELDS)
        writer.writeheader()
        writer.writerows(mapped)
    excluded = [row for row in mapped if not row["formal_main_table_eligible"]]
    payload = {
        "schema_version": "task4_formal_main_table_exclusions_v1",
        "run_map": str(output),
        "run_map_sha256": _sha256(output),
        "total_attempts": len(mapped),
        "formal_main_table_candidates": len(mapped) - len(excluded),
        "excluded_attempts": len(excluded),
        "exclusions": excluded,
    }
    with exclusions.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {
        "run_map": str(output),
        "exclusion_list": str(exclusions),
        "total_attempts": len(mapped),
        "formal_main_table_candidates": len(mapped) - len(excluded),
        "excluded_attempts": len(excluded),
        "run_map_sha256": _sha256(output),
        "exclusion_list_sha256": _sha256(exclusions),
    }


def _map_attempt(
    campaign_dir: Path,
    campaign_id: str,
    state: dict[str, str],
    *,
    manifest_sha: str,
    state_sha: str,
) -> dict[str, Any]:
    run_id = str(state.get("run_id", ""))
    source_run_dir = str(state.get("run_dir", ""))
    local = campaign_dir / "runs" / run_id
    run_dir = local if local.is_dir() else Path(source_run_dir)
    reasons: list[str] = []
    status = str(state.get("status", ""))
    run_mode = "unknown"
    protocol_variant = "unknown"
    validity_status = "unavailable"
    execution_valid: bool | None = None
    paper_eligible: bool | None = None
    missing = [name for name in REQUIRED_LEAF_ARTIFACTS if not (run_dir / name).is_file()]
    if status != "complete":
        reasons.append(f"state_status:{status}")
    if missing:
        reasons.append("missing_artifacts:" + ";".join(missing))
    if not missing:
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        experiment = dict(resolved.get("experiment", {}))
        run_mode = str(experiment.get("run_mode", "unknown"))
        protocol_variant = str(experiment.get("protocol_variant", "unknown"))
        metrics = _read_json(run_dir / "metrics.json")
        protocol = _read_json(run_dir / "protocol_audit.json")
        validity = dict(metrics.get("run_validity", {}))
        validity_status = str(validity.get("status", "unknown"))
        paper_eligible = bool(validity.get("paper_eligible"))
        execution_valid = dict(protocol.get("execution_health", {})).get("valid") is True
        if not execution_valid:
            reasons.append("execution_invalid")
        if not paper_eligible:
            reasons.append(f"run_validity:{validity_status}")
        if run_mode != "formal":
            reasons.append(f"run_mode:{run_mode}")
        if "model_substituted" in protocol_variant:
            reasons.append("model_substituted_sensitivity_only")
    eligible = not reasons
    if eligible:
        classification = "formal_main_table_candidate"
    elif status != "complete":
        classification = "partial_or_failed"
    elif run_mode == "pilot":
        classification = "pilot_descriptive_only"
    elif "model_substituted" in protocol_variant:
        classification = "sensitivity_only"
    else:
        classification = "invalid_or_ineligible"
    return {
        "campaign_id": campaign_id,
        "condition_id": state.get("condition_id", ""),
        "target_mechanism": state.get("target_mechanism", ""),
        "seed": state.get("seed", ""),
        "attempt": state.get("attempt", ""),
        "state_status": status,
        "run_id": run_id,
        "source_run_dir": source_run_dir,
        "resolved_run_dir": str(run_dir.resolve()),
        "run_mode": run_mode,
        "protocol_variant": protocol_variant,
        "run_validity_status": validity_status,
        "execution_valid": execution_valid,
        "paper_eligible": paper_eligible,
        "classification": classification,
        "formal_main_table_eligible": eligible,
        "exclusion_reasons": "|".join(reasons),
        "campaign_manifest_sha256": manifest_sha,
        "state_tsv_sha256": state_sha,
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
