"""Build provenance-rich run maps without mixing Pilot, failed, or formal evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from agentmemeval.evaluation.resource_audit import (
    select_latest_completed_state_rows,
)

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
    "leaf_artifacts_sha256",
    "campaign_manifest_sha256",
    "state_tsv_sha256",
)
REQUIRED_LEAF_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)


def build_run_map(
    campaign_dirs: list[str | Path],
    output_csv: str | Path,
    exclusion_json: str | Path,
) -> dict[str, Any]:
    """Write one latest lifecycle row per attempt plus formal-main exclusion reasons."""

    mapped: list[dict[str, Any]] = []
    state_selections: list[dict[str, Any]] = []
    for raw_dir in campaign_dirs:
        campaign_dir = Path(raw_dir).resolve()
        manifest_path = campaign_dir / "campaign_manifest.json"
        state_path = campaign_dir / "state.tsv"
        manifest = _read_json(manifest_path)
        campaign_id = str(manifest.get("campaign_id") or campaign_dir.name)
        nested_campaign = manifest.get("campaign")
        campaign_manifest_identity_invalid = (
            manifest.get("schema_version") != "agentmemeval_campaign_v1"
            or not manifest.get("campaign_id")
            or not isinstance(nested_campaign, dict)
            or nested_campaign.get("campaign_id") != manifest.get("campaign_id")
        )
        with state_path.open("r", encoding="utf-8", newline="") as handle:
            states = list(csv.DictReader(handle, delimiter="\t"))
        latest_completed, state_selection = select_latest_completed_state_rows(states)
        latest_completed_attempts = {
            (
                str(row["condition_id"]),
                str(row["seed"]),
                str(row["attempt"]),
            )
            for row in latest_completed
        }
        state_selections.append({"campaign_id": campaign_id, **state_selection})
        latest: dict[tuple[str, str, str], dict[str, str]] = {}
        lifecycle_identities: dict[
            tuple[str, str, str], set[tuple[str, str, str]]
        ] = {}
        for state in states:
            key = (
                str(state.get("condition_id", "")),
                str(state.get("seed", "")),
                str(state.get("attempt", "")),
            )
            lifecycle_identities.setdefault(key, set()).add(
                (
                    str(state.get("target_mechanism", "")),
                    str(state.get("run_id", "")),
                    str(state.get("run_dir", "")),
                )
            )
            latest[key] = state
        for key, state in latest.items():
            mapped.append(
                _map_attempt(
                    campaign_dir,
                    campaign_id,
                    state,
                    campaign_manifest_identity_invalid=(
                        campaign_manifest_identity_invalid
                    ),
                    lifecycle_identity_conflict=len(lifecycle_identities[key]) != 1,
                    latest_completed_attempt=key in latest_completed_attempts,
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
        "schema_version": "task4_formal_main_table_exclusions_v3",
        "run_map": str(output),
        "run_map_sha256": _sha256(output),
        "total_attempts": len(mapped),
        "formal_main_table_candidates": len(mapped) - len(excluded),
        "excluded_attempts": len(excluded),
        "state_selections": state_selections,
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
        "state_selections": state_selections,
        "run_map_sha256": _sha256(output),
        "exclusion_list_sha256": _sha256(exclusions),
    }


def _map_attempt(
    campaign_dir: Path,
    campaign_id: str,
    state: dict[str, str],
    *,
    campaign_manifest_identity_invalid: bool,
    lifecycle_identity_conflict: bool,
    latest_completed_attempt: bool,
    manifest_sha: str,
    state_sha: str,
) -> dict[str, Any]:
    run_id = str(state.get("run_id", ""))
    source_run_dir = str(state.get("run_dir", ""))
    raw_runs_root = campaign_dir / "runs"
    runs_root = raw_runs_root.resolve()
    run_dir = (campaign_dir / "runs" / run_id).resolve()
    reasons: list[str] = []
    status = str(state.get("status", ""))
    run_mode = "unknown"
    protocol_variant = "unknown"
    validity_status = "unavailable"
    execution_valid: bool | None = None
    paper_eligible: bool | None = None
    canonical_leaf = (
        bool(run_id)
        and not raw_runs_root.is_symlink()
        and run_dir.is_relative_to(runs_root)
        and run_dir.is_dir()
    )
    if not canonical_leaf:
        reasons.append("canonical_archive_leaf_missing_or_unsafe")
    if campaign_manifest_identity_invalid:
        reasons.append("campaign_manifest_identity_invalid")
    if lifecycle_identity_conflict:
        reasons.append("state_lifecycle_identity_conflict")
    if not latest_completed_attempt:
        reasons.append("not_latest_completed_attempt")
    missing = (
        [
            name
            for name in REQUIRED_LEAF_ARTIFACTS
            if not (run_dir / name).is_file()
            or (run_dir / name).stat().st_size < 1
        ]
        if canonical_leaf
        else list(REQUIRED_LEAF_ARTIFACTS)
    )
    if status != "complete":
        reasons.append(f"state_status:{status}")
    if missing:
        reasons.append("missing_artifacts:" + ";".join(missing))
    if not missing:
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if not isinstance(resolved, dict):
            raise ValueError(f"expected YAML object: {run_dir / 'resolved_config.yaml'}")
        experiment = dict(resolved.get("experiment", {}))
        run_mode = str(experiment.get("run_mode", "unknown"))
        protocol_variant = str(experiment.get("protocol_variant", "unknown"))
        run_manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics.json")
        protocol = _read_json(run_dir / "protocol_audit.json")
        experiment_result = _read_json(run_dir / "experiment_result.json")
        reasons.extend(
            _identity_reasons(
                state=state,
                campaign_id=campaign_id,
                source_run_dir=source_run_dir,
                run_manifest=run_manifest,
                experiment=experiment,
                experiment_result=experiment_result,
            )
        )
        validity = dict(metrics.get("run_validity", {}))
        validity_status = str(validity.get("status", "unknown"))
        paper_eligible = validity.get("paper_eligible") is True
        execution = dict(protocol.get("execution_health", {}))
        execution_valid = (
            execution.get("valid") is True
            and execution.get("status") == "passed"
            and all(
                _safe_int(execution.get(key)) == 0
                for key in (
                    "fallback_count",
                    "memory_revision_fallback_count",
                    "reward_conservation_violation_count",
                    "stack_conservation_violation_count",
                )
            )
        )
        if not execution_valid:
            reasons.append("execution_invalid")
        code = dict(dict(run_manifest.get("metadata", {})).get("code", {}))
        if code.get("dirty") is not False:
            reasons.append("dirty_or_unverified_code")
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
    elif "model_substituted" in protocol_variant:
        classification = "sensitivity_only"
    elif run_mode == "pilot":
        classification = "pilot_descriptive_only"
    else:
        classification = "invalid_or_ineligible"
    leaf_hashes = (
        {
            name: _sha256(run_dir / name)
            for name in REQUIRED_LEAF_ARTIFACTS
            if (run_dir / name).is_file()
        }
        if canonical_leaf
        else {}
    )
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
        "leaf_artifacts_sha256": json.dumps(
            leaf_hashes, ensure_ascii=False, sort_keys=True
        ),
        "campaign_manifest_sha256": manifest_sha,
        "state_tsv_sha256": state_sha,
    }


def _identity_reasons(
    *,
    state: dict[str, str],
    campaign_id: str,
    source_run_dir: str,
    run_manifest: dict[str, Any],
    experiment: dict[str, Any],
    experiment_result: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    run_id = str(state.get("run_id", ""))
    seed = _safe_int(state.get("seed"))
    expected_config_path = _source_child_path(
        source_run_dir, "resolved_config.yaml"
    )
    manifest_contract = {
        "run_id": run_id,
        "seed": seed,
        "output_dir": source_run_dir,
        "config_snapshot_path": expected_config_path,
    }
    for key, expected in manifest_contract.items():
        if run_manifest.get(key) != expected:
            reasons.append(f"manifest_identity_mismatch:{key}")
    experiment_contract = {
        "campaign_id": campaign_id,
        "campaign_condition_id": str(state.get("condition_id", "")),
        "seed": seed,
        "run_id": run_id,
    }
    for key, expected in experiment_contract.items():
        if experiment.get(key) != expected:
            reasons.append(f"resolved_config_identity_mismatch:{key}")
    if experiment_result.get("run_id") != run_id:
        reasons.append("experiment_result_identity_mismatch:run_id")
    return reasons


def _source_child_path(source: str, child: str) -> str:
    if not source:
        return ""
    if re.match(r"^[A-Za-z]:[\\/]", source) or source.startswith("\\\\"):
        return str(PureWindowsPath(source) / child)
    return str(PurePosixPath(source) / child)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


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
