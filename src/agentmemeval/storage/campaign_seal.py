"""Fail-closed readiness audit before sealing an immutable campaign snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def audit_campaign_seal_readiness(
    campaign_dir: str | Path,
    *,
    minimum_quiet_seconds: int = 120,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Prove a campaign stopped writing before an append-only snapshot is built."""

    if minimum_quiet_seconds < 0:
        raise ValueError("minimum_quiet_seconds must be non-negative")
    raw_root = Path(campaign_dir)
    root = raw_root.resolve()
    blockers: list[str] = []
    if raw_root.is_symlink():
        blockers.append("campaign root is a symlink")
    manifest_path = root / "campaign_manifest.json"
    state_path = root / "state.tsv"
    for path in (manifest_path, state_path):
        if path.is_symlink():
            blockers.append(f"control file is a symlink: {path.name}")
        if not path.is_file() or path.stat().st_size < 1:
            blockers.append(f"control file is missing or empty: {path.name}")
    if blockers:
        return _result(
            root,
            minimum_quiet_seconds=minimum_quiet_seconds,
            expected_matrix_count=0,
            latest_attempt_count=0,
            complete_latest_attempt_count=0,
            file_count=0,
            total_bytes=0,
            quiet_seconds=None,
            latest_file_mtime_utc=None,
            blockers=blockers,
        )

    manifest = _read_json(manifest_path)
    campaign = manifest.get("campaign")
    if manifest.get("schema_version") != "agentmemeval_campaign_v1":
        blockers.append("campaign manifest schema mismatch")
    if not isinstance(campaign, dict):
        blockers.append("campaign manifest nested campaign is missing")
        campaign = {}
    if campaign.get("campaign_id") != manifest.get("campaign_id"):
        blockers.append("campaign manifest identity mismatch")
    conditions = campaign.get("conditions") or [
        {"condition_id": "mixed_table", "target_mechanism": "mixed"}
    ]
    seeds = campaign.get("seeds", [])
    expected_identities = {
        (str(condition.get("condition_id", "")), int(seed))
        for condition in conditions
        if isinstance(condition, dict)
        for seed in seeds
    }
    if not expected_identities:
        blockers.append("campaign expected matrix is empty")

    with state_path.open("r", encoding="utf-8", newline="") as handle:
        states = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    malformed_state_rows = 0
    for row in states:
        try:
            identity = (str(row.get("condition_id", "")), int(row.get("seed", "")))
            int(row.get("attempt", ""))
        except (TypeError, ValueError):
            malformed_state_rows += 1
            continue
        grouped.setdefault(identity, []).append(row)
    if malformed_state_rows:
        blockers.append(f"malformed state rows: {malformed_state_rows}")
    extras = sorted(set(grouped) - expected_identities)
    missing = sorted(expected_identities - set(grouped))
    if extras:
        blockers.append(f"unexpected matrix identities: {extras}")
    if missing:
        blockers.append(f"missing matrix identities: {missing}")

    latest_rows: dict[tuple[str, int], dict[str, str]] = {}
    for identity, rows in grouped.items():
        max_attempt = max(int(row["attempt"]) for row in rows)
        latest_rows[identity] = [
            row for row in rows if int(row["attempt"]) == max_attempt
        ][-1]
        completed_attempts = {
            int(row["attempt"]) for row in rows if row.get("status") == "complete"
        }
        if len(completed_attempts) > 1:
            blockers.append(
                f"multiple completed attempts for {identity}: "
                f"{sorted(completed_attempts)}"
            )
    incomplete = sorted(
        (identity, row.get("status", ""))
        for identity, row in latest_rows.items()
        if row.get("status") != "complete"
    )
    if incomplete:
        blockers.append(f"latest attempts are not complete: {incomplete}")

    runs_root = (root / "runs").resolve()
    for identity in sorted(expected_identities):
        row = latest_rows.get(identity)
        if row is None or row.get("status") != "complete":
            continue
        run_id = str(row.get("run_id", ""))
        canonical = (root / "runs" / run_id).resolve()
        source = Path(str(row.get("run_dir", ""))).resolve()
        if (
            not run_id
            or not canonical.is_relative_to(runs_root)
            or source != canonical
            or not canonical.is_dir()
        ):
            blockers.append(f"non-canonical completed run for {identity}")
            continue
        missing_artifacts = [
            name
            for name in REQUIRED_LEAF_ARTIFACTS
            if not (canonical / name).is_file()
            or (canonical / name).stat().st_size < 1
        ]
        if missing_artifacts:
            blockers.append(
                f"{run_id} missing required artifacts: {missing_artifacts}"
            )

    files = [path for path in root.rglob("*") if path.is_file()]
    symlinks = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        blockers.append(f"campaign contains symlinks: {symlinks}")
    total_bytes = sum(path.stat().st_size for path in files)
    latest_mtime = max((path.stat().st_mtime for path in files), default=None)
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    quiet_seconds = (
        max(0.0, now.timestamp() - latest_mtime)
        if latest_mtime is not None
        else None
    )
    latest_mtime_utc = (
        datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat()
        if latest_mtime is not None
        else None
    )
    if quiet_seconds is None or quiet_seconds < minimum_quiet_seconds:
        blockers.append(
            "campaign quiet period is insufficient: "
            f"{quiet_seconds}/{minimum_quiet_seconds} seconds"
        )
    return _result(
        root,
        minimum_quiet_seconds=minimum_quiet_seconds,
        expected_matrix_count=len(expected_identities),
        latest_attempt_count=len(latest_rows),
        complete_latest_attempt_count=sum(
            row.get("status") == "complete" for row in latest_rows.values()
        ),
        file_count=len(files),
        total_bytes=total_bytes,
        quiet_seconds=quiet_seconds,
        latest_file_mtime_utc=latest_mtime_utc,
        blockers=blockers,
        campaign_manifest_sha256=_sha256(manifest_path),
        state_tsv_sha256=_sha256(state_path),
    )


def _result(
    root: Path,
    *,
    minimum_quiet_seconds: int,
    expected_matrix_count: int,
    latest_attempt_count: int,
    complete_latest_attempt_count: int,
    file_count: int,
    total_bytes: int,
    quiet_seconds: float | None,
    latest_file_mtime_utc: str | None,
    blockers: list[str],
    campaign_manifest_sha256: str | None = None,
    state_tsv_sha256: str | None = None,
) -> dict[str, Any]:
    blockers = sorted(set(blockers))
    return {
        "schema_version": "task4_campaign_seal_readiness_v1",
        "campaign_dir": str(root),
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "state_tsv_sha256": state_tsv_sha256,
        "minimum_quiet_seconds": minimum_quiet_seconds,
        "observed_quiet_seconds": quiet_seconds,
        "latest_file_mtime_utc": latest_file_mtime_utc,
        "expected_matrix_count": expected_matrix_count,
        "latest_attempt_count": latest_attempt_count,
        "complete_latest_attempt_count": complete_latest_attempt_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "blockers": blockers,
        "status": "ready_to_seal" if not blockers else "not_ready_to_seal",
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
