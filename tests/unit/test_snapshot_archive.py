from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from agentmemeval.storage.snapshot_archive import (
    build_snapshot_archive,
    verify_archive_checksum,
)


def test_snapshot_archive_builds_and_self_verifies_append_only_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "state.tsv").write_text("status\ncomplete\n", encoding="utf-8")
    run = root / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text('{"run_id":"run-1"}', encoding="utf-8")
    archive = tmp_path / "snapshot.tar.gz"
    manifest = tmp_path / "snapshot.files.tsv"
    checksum = tmp_path / "snapshot.tar.gz.sha256"
    receipt = tmp_path / "snapshot.receipt.json"

    result = build_snapshot_archive(
        root,
        archive,
        manifest,
        checksum,
        receipt,
    )

    assert result["status"] == "verified"
    assert result["file_count"] == 2
    assert result["source_verification"]["verified"] is True
    assert result["archive_verification"]["verified"] is True
    assert result["checksum_verification"]["verified"] is True
    assert verify_archive_checksum(archive, checksum)["verified"] is True
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["archive_sha256"] == result["archive_sha256"]
    with tarfile.open(archive, mode="r:gz") as handle:
        names = {member.name for member in handle.getmembers() if member.isfile()}
    assert names == {
        "campaign/state.tsv",
        "campaign/runs/run-1/manifest.json",
    }
    archive.write_bytes(archive.read_bytes() + b"tamper")
    assert verify_archive_checksum(archive, checksum)["verified"] is False
    with pytest.raises(FileExistsError):
        build_snapshot_archive(root, archive, manifest, checksum, receipt)


def test_snapshot_checksum_rejects_tamper_and_malformed_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "snapshot.tar.gz"
    archive.write_bytes(b"archive")
    checksum = tmp_path / "snapshot.tar.gz.sha256"
    checksum.write_text("00  wrong-name.tar.gz\n", encoding="ascii")

    malformed = verify_archive_checksum(archive, checksum)

    assert malformed["verified"] is False
    assert set(malformed["format_errors"]) == {
        "invalid_sha256",
        "archive_filename_mismatch",
    }

    checksum.write_text(
        f"{'0' * 64}  {archive.name}\n",
        encoding="ascii",
    )
    mismatch = verify_archive_checksum(archive, checksum)
    assert mismatch["verified"] is False
    assert mismatch["format_errors"] == []
    assert mismatch["hash_matches"] is False
