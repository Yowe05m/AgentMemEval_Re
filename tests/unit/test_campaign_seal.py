from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentmemeval.storage.campaign_seal import (
    REQUIRED_LEAF_ARTIFACTS,
    audit_campaign_archive_handoff,
    audit_campaign_seal_readiness,
)
from agentmemeval.storage.snapshot_archive import build_snapshot_archive


def test_complete_quiet_canonical_campaign_is_ready_to_seal(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, status="complete")
    old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))

    result = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result["status"] == "ready_to_seal"
    assert result["blockers"] == []
    assert result["expected_matrix_count"] == 1
    assert result["complete_latest_attempt_count"] == 1
    assert result["campaign_manifest_sha256"]
    assert result["state_tsv_sha256"]


def test_running_or_recent_campaign_is_not_ready_to_seal(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, status="running")
    recent = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (recent, recent))

    result = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result["status"] == "not_ready_to_seal"
    assert any("latest attempts are not complete" in item for item in result["blockers"])
    assert any("quiet period is insufficient" in item for item in result["blockers"])


def test_completed_retry_after_failed_attempt_is_allowed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, status="failed")
    state = campaign / "state.tsv"
    run_dir = campaign / "runs" / "mixed_table__s11__a02"
    _leaf(run_dir)
    with state.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event_utc",
                "condition_id",
                "target_mechanism",
                "seed",
                "attempt",
                "status",
                "run_id",
                "run_dir",
                "failure_class",
                "message",
            ),
            delimiter="\t",
        )
        writer.writerow(
            {
                "event_utc": "2026-01-01T00:01:00Z",
                "condition_id": "mixed_table",
                "target_mechanism": "mixed",
                "seed": 11,
                "attempt": 2,
                "status": "complete",
                "run_id": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "failure_class": "",
                "message": "",
            }
        )
    old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))

    result = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result["status"] == "ready_to_seal"


def test_multiple_completed_attempts_are_not_ready_to_seal(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, status="complete")
    state = campaign / "state.tsv"
    second = campaign / "runs" / "mixed_table__s11__a02"
    _leaf(second)
    _append_state(state, second, attempt=2, status="complete")
    old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))

    result = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result["status"] == "not_ready_to_seal"
    assert any("multiple completed attempts" in item for item in result["blockers"])


def test_campaign_symlink_is_not_ready_to_seal(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, status="complete")
    target = campaign / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = campaign / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))

    result = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result["status"] == "not_ready_to_seal"
    assert any("campaign contains symlinks" in item for item in result["blockers"])


def test_archive_handoff_reverifies_seal_snapshot_and_live_campaign(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, status="complete")
    old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in campaign.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))
    seal = audit_campaign_seal_readiness(
        campaign,
        minimum_quiet_seconds=120,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    receipt_path = tmp_path / "snapshot.receipt.json"
    build_snapshot_archive(
        campaign,
        tmp_path / "snapshot.tar.gz",
        tmp_path / "snapshot.files.tsv",
        tmp_path / "snapshot.tar.gz.sha256",
        receipt_path,
    )

    verified = audit_campaign_archive_handoff(
        campaign,
        seal_readiness_path=seal_path,
        snapshot_receipt_path=receipt_path,
    )

    assert verified["status"] == "verified_campaign_archive_handoff"
    assert verified["blockers"] == []
    assert verified["file_count"] == seal["file_count"]
    assert verified["archive_sha256"]
    with (campaign / "state.tsv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    blocked = audit_campaign_archive_handoff(
        campaign,
        seal_readiness_path=seal_path,
        snapshot_receipt_path=receipt_path,
    )

    assert blocked["status"] == "blocked_campaign_archive_handoff"
    assert any(
        "current campaign does not match snapshot manifest" in item
        for item in blocked["blockers"]
    )
    assert any(
        "state.tsv changed after seal readiness" in item
        for item in blocked["blockers"]
    )


def _campaign(tmp_path: Path, *, status: str) -> Path:
    root = tmp_path / "campaign"
    run_dir = root / "runs" / "mixed_table__s11__a01"
    _leaf(run_dir)
    manifest = {
        "schema_version": "agentmemeval_campaign_v1",
        "campaign_id": "campaign",
        "campaign": {
            "campaign_id": "campaign",
            "seeds": [11],
            "conditions": [
                {"condition_id": "mixed_table", "target_mechanism": "mixed"}
            ],
        },
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with (root / "state.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event_utc",
                "condition_id",
                "target_mechanism",
                "seed",
                "attempt",
                "status",
                "run_id",
                "run_dir",
                "failure_class",
                "message",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_utc": "2026-01-01T00:00:00Z",
                "condition_id": "mixed_table",
                "target_mechanism": "mixed",
                "seed": 11,
                "attempt": 1,
                "status": status,
                "run_id": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "failure_class": "",
                "message": "",
            }
        )
    return root


def _append_state(
    state: Path,
    run_dir: Path,
    *,
    attempt: int,
    status: str,
) -> None:
    with state.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event_utc",
                "condition_id",
                "target_mechanism",
                "seed",
                "attempt",
                "status",
                "run_id",
                "run_dir",
                "failure_class",
                "message",
            ),
            delimiter="\t",
        )
        writer.writerow(
            {
                "event_utc": "2026-01-01T00:01:00Z",
                "condition_id": "mixed_table",
                "target_mechanism": "mixed",
                "seed": 11,
                "attempt": attempt,
                "status": status,
                "run_id": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "failure_class": "",
                "message": "",
            }
        )


def _leaf(path: Path) -> None:
    path.mkdir(parents=True)
    for name in REQUIRED_LEAF_ARTIFACTS:
        (path / name).write_text("{}\n", encoding="utf-8")
