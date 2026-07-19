from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentmemeval.storage.archive import (
    _filesystem_path,
    build_file_manifest,
    verify_file_manifest,
)


def test_file_manifest_round_trip_and_detects_changes(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.json").write_text('{"b": 2}', encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    built = build_file_manifest(root, manifest)
    assert built["file_count"] == 2
    verified = verify_file_manifest(root, manifest)
    assert verified["verified"] is True
    (root / "a.txt").write_text("changed", encoding="utf-8")
    changed = verify_file_manifest(root, manifest)
    assert changed["verified"] is False
    assert changed["size_mismatches"]


def test_file_manifest_detects_missing_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    expected = root / "expected.txt"
    expected.write_text("expected", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    build_file_manifest(root, manifest)
    expected.unlink()
    (root / "extra.txt").write_text("extra", encoding="utf-8")
    result = verify_file_manifest(root, manifest)
    assert result["missing_files"] == ["expected.txt"]
    assert result["extra_files"] == ["extra.txt"]


def test_manifest_rejects_unsafe_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "relative_path\tsize_bytes\tsha256\n../escape\t0\t00\n",
        encoding="utf-8",
    )
    result = verify_file_manifest(root, manifest)
    assert result["verified"] is False
    assert result["unsafe_or_duplicate_paths"] == ["../escape"]


def test_manifest_reports_malformed_header_size_and_hash(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "relative_path\tsize_bytes\twrong_hash_column\n"
        "a.txt\tnot-an-int\tbad\n",
        encoding="utf-8",
    )

    result = verify_file_manifest(root, manifest)

    assert result["verified"] is False
    assert result["schema_version"] == "task4_file_manifest_verification_v2"
    assert result["manifest_format_errors"][0]["kind"] == "header_mismatch"
    assert result["invalid_manifest_rows"][0]["reason"] == "invalid_size_bytes"
    assert result["verified_file_count"] == 0


def test_manifest_reports_invalid_sha256_without_raising(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "relative_path\tsize_bytes\tsha256\n"
        "a.txt\t5\t00\n",
        encoding="utf-8",
    )

    result = verify_file_manifest(root, manifest)

    assert result["verified"] is False
    assert result["invalid_manifest_rows"] == [
        {
            "row_number": 2,
            "relative_path": "a.txt",
            "reason": "invalid_sha256",
            "value": "00",
        }
    ]


def test_manifest_rejects_unlisted_symlink(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    expected = root / "expected.txt"
    expected.write_text("expected", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    build_file_manifest(root, manifest)
    try:
        os.symlink(expected, root / "extra-link.txt")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    result = verify_file_manifest(root, manifest)

    assert result["verified"] is False
    assert result["symlinks"] == ["extra-link.txt"]


def test_file_manifest_round_trip_supports_paths_over_260_characters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    relative = (
        Path("a" * 90)
        / ("b" * 90)
        / ("c" * 60)
        / "async_00_checkpoint_0150.json"
    )
    long_path = root / relative
    filesystem_path = _filesystem_path(long_path)
    filesystem_path.parent.mkdir(parents=True)
    filesystem_path.write_text('{"verified": true}', encoding="utf-8")
    assert len(str(long_path)) > 260

    manifest = tmp_path / "long_manifest.tsv"
    built = build_file_manifest(root, manifest)
    verified = verify_file_manifest(root, manifest)

    assert built["file_count"] == 1
    assert verified["verified"] is True
    assert verified["verified_file_count"] == 1
