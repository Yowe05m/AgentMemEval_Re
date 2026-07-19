"""Fail-closed readiness audit before sealing an immutable campaign snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentmemeval.storage.archive import verify_file_manifest
from agentmemeval.storage.snapshot_archive import verify_archive_checksum

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
            condition_id = str(row.get("condition_id", "")).strip()
            identity = (condition_id, int(row.get("seed", "")))
            attempt = int(row.get("attempt", ""))
            if not condition_id or attempt < 1:
                raise ValueError
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
        latest_attempt_rows = [
            row for row in rows if int(row["attempt"]) == max_attempt
        ]
        latest_rows[identity] = latest_attempt_rows[-1]
        completed_attempts = {
            int(row["attempt"]) for row in rows if row.get("status") == "complete"
        }
        if len(completed_attempts) > 1:
            blockers.append(
                f"multiple completed attempts for {identity}: "
                f"{sorted(completed_attempts)}"
            )
        if (
            any(row.get("status") == "failed" for row in latest_attempt_rows)
            and latest_rows[identity].get("status") == "complete"
        ):
            blockers.append(
                f"failed state precedes completion within latest attempt for "
                f"{identity}"
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


def audit_campaign_archive_handoff(
    campaign_dir: str | Path,
    *,
    seal_readiness_path: str | Path,
    snapshot_receipt_path: str | Path,
) -> dict[str, Any]:
    """Reverify a sealed server snapshot before a later campaign may start."""

    root_input = Path(campaign_dir).absolute()
    seal_input = Path(seal_readiness_path).absolute()
    receipt_input = Path(snapshot_receipt_path).absolute()
    root = root_input.resolve()
    seal_path = seal_input.resolve()
    receipt_path = receipt_input.resolve()
    blockers: list[str] = []
    for label, source, path in (
        ("campaign", root_input, root),
        ("seal readiness", seal_input, seal_path),
        ("snapshot receipt", receipt_input, receipt_path),
    ):
        if source.is_symlink():
            blockers.append(f"{label} path is a symlink")
        expected = path.is_dir() if label == "campaign" else path.is_file()
        if not expected:
            blockers.append(f"{label} path is missing")
    if blockers:
        return _archive_handoff_result(
            root,
            seal_path=seal_path,
            receipt_path=receipt_path,
            blockers=blockers,
        )

    seal = _read_json(seal_path)
    receipt = _read_json(receipt_path)
    if seal.get("schema_version") != "task4_campaign_seal_readiness_v1":
        blockers.append("seal readiness schema mismatch")
    if seal.get("status") != "ready_to_seal" or seal.get("blockers") != []:
        blockers.append("seal readiness is not verified ready_to_seal")
    if Path(str(seal.get("campaign_dir", ""))).resolve() != root:
        blockers.append("seal readiness campaign root mismatch")
    if receipt.get("schema_version") != "task4_snapshot_archive_receipt_v1":
        blockers.append("snapshot receipt schema mismatch")
    if receipt.get("status") != "verified":
        blockers.append("snapshot receipt is not verified")
    if Path(str(receipt.get("root", ""))).resolve() != root:
        blockers.append("snapshot receipt campaign root mismatch")
    for key in (
        "source_verification",
        "archive_verification",
        "checksum_verification",
    ):
        nested = receipt.get(key)
        if (
            not isinstance(nested, dict)
            or nested.get("verified") is not True
            or nested.get("status") != "verified"
        ):
            blockers.append(f"snapshot receipt {key} is not verified")

    manifest_input = Path(str(receipt.get("manifest", ""))).absolute()
    archive_input = Path(str(receipt.get("archive", ""))).absolute()
    checksum_input = Path(str(receipt.get("checksum", ""))).absolute()
    manifest_path = manifest_input.resolve()
    archive_path = archive_input.resolve()
    checksum_path = checksum_input.resolve()
    for label, source, path in (
        ("snapshot manifest", manifest_input, manifest_path),
        ("snapshot archive", archive_input, archive_path),
        ("snapshot checksum", checksum_input, checksum_path),
    ):
        if not path.is_file() or source.is_symlink():
            blockers.append(f"{label} is missing or a symlink")

    source_verification: dict[str, Any] = {}
    checksum_verification: dict[str, Any] = {}
    manifest_sha256: str | None = None
    archive_sha256: str | None = None
    if not blockers:
        try:
            source_verification = verify_file_manifest(root, manifest_path)
            checksum_verification = verify_archive_checksum(
                archive_path,
                checksum_path,
            )
            manifest_sha256 = _sha256(manifest_path)
            archive_sha256 = str(checksum_verification.get("observed_sha256"))
        except (OSError, UnicodeError, ValueError) as exc:
            blockers.append(f"snapshot material reverification failed: {type(exc).__name__}")
    if source_verification and source_verification.get("verified") is not True:
        blockers.append("current campaign does not match snapshot manifest")
    if checksum_verification and checksum_verification.get("verified") is not True:
        blockers.append("current snapshot archive checksum is not verified")
    if manifest_sha256 is not None and manifest_sha256 != receipt.get(
        "manifest_sha256"
    ):
        blockers.append("current snapshot manifest hash mismatch")
    if archive_sha256 is not None and archive_sha256 != receipt.get(
        "archive_sha256"
    ):
        blockers.append("current snapshot archive hash mismatch")

    manifest_path_live = root / "campaign_manifest.json"
    state_path_live = root / "state.tsv"
    campaign_manifest_sha256 = (
        _sha256(manifest_path_live) if manifest_path_live.is_file() else None
    )
    state_tsv_sha256 = _sha256(state_path_live) if state_path_live.is_file() else None
    if campaign_manifest_sha256 != seal.get("campaign_manifest_sha256"):
        blockers.append("campaign manifest changed after seal readiness")
    if state_tsv_sha256 != seal.get("state_tsv_sha256"):
        blockers.append("state.tsv changed after seal readiness")

    observed_files = [path for path in root.rglob("*") if path.is_file()]
    observed_file_count = len(observed_files)
    observed_total_bytes = sum(path.stat().st_size for path in observed_files)
    for label, value in (
        ("seal file count", seal.get("file_count")),
        ("snapshot receipt file count", receipt.get("file_count")),
        (
            "current manifest expected file count",
            source_verification.get("expected_file_count"),
        ),
        (
            "current manifest verified file count",
            source_verification.get("verified_file_count"),
        ),
    ):
        if value != observed_file_count:
            blockers.append(f"{label} mismatch: {value}/{observed_file_count}")
    for label, value in (
        ("seal total bytes", seal.get("total_bytes")),
        (
            "snapshot receipt uncompressed bytes",
            receipt.get("total_uncompressed_size_bytes"),
        ),
    ):
        if value != observed_total_bytes:
            blockers.append(f"{label} mismatch: {value}/{observed_total_bytes}")

    return _archive_handoff_result(
        root,
        seal_path=seal_path,
        receipt_path=receipt_path,
        blockers=blockers,
        seal_readiness_sha256=_sha256(seal_path),
        snapshot_receipt_sha256=_sha256(receipt_path),
        manifest_sha256=manifest_sha256,
        archive_sha256=archive_sha256,
        campaign_manifest_sha256=campaign_manifest_sha256,
        state_tsv_sha256=state_tsv_sha256,
        file_count=observed_file_count,
        total_bytes=observed_total_bytes,
        source_verification=source_verification,
        checksum_verification=checksum_verification,
    )


def _archive_handoff_result(
    root: Path,
    *,
    seal_path: Path,
    receipt_path: Path,
    blockers: list[str],
    seal_readiness_sha256: str | None = None,
    snapshot_receipt_sha256: str | None = None,
    manifest_sha256: str | None = None,
    archive_sha256: str | None = None,
    campaign_manifest_sha256: str | None = None,
    state_tsv_sha256: str | None = None,
    file_count: int | None = None,
    total_bytes: int | None = None,
    source_verification: dict[str, Any] | None = None,
    checksum_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = sorted(set(blockers))
    return {
        "schema_version": "task4_campaign_archive_handoff_v1",
        "campaign_dir": str(root),
        "seal_readiness_path": str(seal_path),
        "seal_readiness_sha256": seal_readiness_sha256,
        "snapshot_receipt_path": str(receipt_path),
        "snapshot_receipt_sha256": snapshot_receipt_sha256,
        "manifest_sha256": manifest_sha256,
        "archive_sha256": archive_sha256,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "state_tsv_sha256": state_tsv_sha256,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "source_verification": source_verification or {},
        "checksum_verification": checksum_verification or {},
        "blockers": blockers,
        "status": (
            "verified_campaign_archive_handoff"
            if not blockers
            else "blocked_campaign_archive_handoff"
        ),
    }


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
