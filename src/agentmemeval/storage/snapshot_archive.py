"""Append-only, self-verifying tar.gz snapshots for paper evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from agentmemeval.storage.archive import (
    MANIFEST_FIELDS,
    _filesystem_path,
    build_file_manifest,
    verify_file_manifest,
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def build_snapshot_archive(
    root: str | Path,
    archive: str | Path,
    manifest: str | Path,
    checksum: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Build new archive evidence and write the verified receipt last."""

    source_input = Path(root).absolute()
    if source_input.is_symlink():
        raise NotADirectoryError(
            f"snapshot root must be a real directory: {source_input}"
        )
    source = source_input.resolve()
    archive_path = Path(archive).absolute()
    manifest_path = Path(manifest).absolute()
    checksum_path = Path(checksum).absolute()
    receipt_path = Path(receipt).absolute()
    if not source.is_dir():
        raise NotADirectoryError(f"snapshot root must be a real directory: {source}")
    if not source.name:
        raise ValueError("snapshot root must have a non-empty leaf name")
    targets = [archive_path, manifest_path, checksum_path, receipt_path]
    if len({target.resolve() for target in targets}) != len(targets):
        raise ValueError("snapshot output paths must be distinct")
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        resolved_target = target.resolve()
        if resolved_target == source or source in resolved_target.parents:
            raise ValueError("snapshot outputs must be outside the archived root")
        target.parent.mkdir(parents=True, exist_ok=True)

    manifest_summary = build_file_manifest(source, manifest_path)
    with tarfile.open(archive_path, mode="x:gz") as handle:
        handle.add(source, arcname=source.name, recursive=True)

    source_verification = verify_file_manifest(source, manifest_path)
    archive_verification = _verify_archive_members(
        archive_path,
        manifest_path,
        root_name=source.name,
    )
    if not source_verification["verified"] or not archive_verification["verified"]:
        raise RuntimeError(
            "snapshot verification failed; partial append-only outputs were preserved"
        )

    archive_sha256 = _sha256_path(archive_path)
    with checksum_path.open("x", encoding="ascii", errors="strict") as handle:
        handle.write(f"{archive_sha256}  {archive_path.name}\n")
    checksum_verification = verify_archive_checksum(archive_path, checksum_path)
    if not checksum_verification["verified"]:
        raise RuntimeError("snapshot checksum self-verification failed")

    payload = {
        "schema_version": "task4_snapshot_archive_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "root": str(source),
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "file_count": manifest_summary["file_count"],
        "total_uncompressed_size_bytes": manifest_summary["total_size_bytes"],
        "checksum": str(checksum_path),
        "source_verification": source_verification,
        "archive_verification": archive_verification,
        "checksum_verification": checksum_verification,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"receipt": str(receipt_path), **payload}


def verify_archive_checksum(
    archive: str | Path,
    checksum: str | Path,
) -> dict[str, Any]:
    """Strictly verify a GNU-style one-line SHA-256 checksum file."""

    archive_source = Path(archive).absolute()
    archive_is_symlink = archive_source.is_symlink()
    archive_path = archive_source.resolve()
    checksum_source = Path(checksum).absolute()
    checksum_is_symlink = checksum_source.is_symlink()
    checksum_path = checksum_source.resolve()
    lines = checksum_path.read_text(encoding="ascii", errors="strict").splitlines()
    format_errors: list[str] = []
    expected_hash = ""
    expected_name = ""
    if len(lines) != 1:
        format_errors.append(f"expected_one_line_observed:{len(lines)}")
    elif "  " not in lines[0]:
        format_errors.append("missing_two_space_separator")
    else:
        expected_hash, expected_name = lines[0].split("  ", 1)
        if SHA256_PATTERN.fullmatch(expected_hash) is None:
            format_errors.append("invalid_sha256")
        if expected_name != archive_source.name:
            format_errors.append("archive_filename_mismatch")
    observed_hash = _sha256_path(archive_path)
    hash_matches = not format_errors and observed_hash == expected_hash
    verified = not archive_is_symlink and not checksum_is_symlink and hash_matches
    return {
        "schema_version": "task4_snapshot_archive_checksum_verification_v1",
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "archive_is_symlink": archive_is_symlink,
        "checksum_is_symlink": checksum_is_symlink,
        "expected_sha256": expected_hash or None,
        "observed_sha256": observed_hash,
        "format_errors": format_errors,
        "hash_matches": hash_matches,
        "verified": verified,
        "status": "verified" if verified else "failed",
    }


def extract_snapshot_archive(
    archive: str | Path,
    checksum: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Verify and safely extract one snapshot into a brand-new directory."""

    archive_path = Path(archive).absolute()
    checksum_path = Path(checksum).absolute()
    manifest_path = Path(manifest).absolute()
    output = Path(output_dir).absolute()
    receipt_path = Path(receipt).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    receipt_resolved = receipt_path.resolve()
    output_resolved = output.resolve()
    if (
        receipt_resolved == output_resolved
        or output_resolved in receipt_resolved.parents
    ):
        raise ValueError("extraction receipt must be outside the output directory")
    if manifest_path.is_symlink():
        raise ValueError("snapshot manifest must not be a symlink")

    checksum_verification = verify_archive_checksum(archive_path, checksum_path)
    if not checksum_verification["verified"]:
        raise ValueError("snapshot archive checksum verification failed")
    root_name = _infer_archive_root_name(archive_path)
    archive_verification = _verify_archive_members(
        archive_path,
        manifest_path,
        root_name=root_name,
    )
    if not archive_verification["verified"]:
        raise ValueError("snapshot archive member verification failed")

    filesystem_output = _filesystem_path(output)
    filesystem_output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, mode="r:gz") as handle:
        for member in handle.getmembers():
            pure = PurePosixPath(member.name)
            target = output.joinpath(*pure.parts)
            if not target.resolve().is_relative_to(output_resolved):
                raise ValueError(f"archive member escaped output directory: {member.name}")
            filesystem_target = _filesystem_path(target)
            if member.isdir():
                filesystem_target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name}")
            filesystem_target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"archive member has no data stream: {member.name}")
            with filesystem_target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)

    extracted_root = output / root_name
    extracted_verification = verify_file_manifest(
        extracted_root,
        manifest_path,
    )
    if not extracted_verification["verified"]:
        raise RuntimeError(
            "extracted snapshot verification failed; partial output was preserved"
        )
    payload = {
        "schema_version": "task4_snapshot_extraction_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "archive": str(archive_path.resolve()),
        "checksum": str(checksum_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_path(manifest_path),
        "output_dir": str(output),
        "extracted_root": str(extracted_root),
        "root_name": root_name,
        "checksum_verification": checksum_verification,
        "archive_verification": archive_verification,
        "extracted_verification": extracted_verification,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with _filesystem_path(receipt_path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"receipt": str(receipt_path), **payload}


def _infer_archive_root_name(archive: Path) -> str:
    roots: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            roots.add(pure.parts[0])
    if len(roots) != 1:
        raise ValueError(f"archive must have exactly one top-level root: {sorted(roots)}")
    return next(iter(roots))


def _verify_archive_members(
    archive: Path,
    manifest: Path,
    *,
    root_name: str,
) -> dict[str, Any]:
    expected = _read_manifest(manifest)
    observed: dict[str, dict[str, Any]] = {}
    unsafe_members: list[str] = []
    special_members: list[str] = []
    duplicate_files: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, str]] = []
    verified_file_count = 0
    with tarfile.open(archive, mode="r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                unsafe_members.append(member.name)
                continue
            if pure.parts[0] != root_name:
                unsafe_members.append(member.name)
                continue
            if member.isdir():
                continue
            if not member.isfile():
                special_members.append(member.name)
                continue
            relative = PurePosixPath(*pure.parts[1:]).as_posix()
            if not relative or relative in observed:
                duplicate_files.append(relative)
                continue
            stream = handle.extractfile(member)
            if stream is None:
                special_members.append(member.name)
                continue
            observed[relative] = {
                "size_bytes": member.size,
                "sha256": _sha256_stream(stream),
            }
    missing = sorted(set(expected) - set(observed))
    extras = sorted(set(observed) - set(expected))
    for relative in sorted(set(expected) & set(observed)):
        expected_row = expected[relative]
        observed_row = observed[relative]
        size_matches = observed_row["size_bytes"] == expected_row["size_bytes"]
        hash_matches = observed_row["sha256"] == expected_row["sha256"]
        if not size_matches:
            size_mismatches.append(
                {
                    "relative_path": relative,
                    "expected": expected_row["size_bytes"],
                    "observed": observed_row["size_bytes"],
                }
            )
        if not hash_matches:
            hash_mismatches.append(
                {
                    "relative_path": relative,
                    "expected": expected_row["sha256"],
                    "observed": observed_row["sha256"],
                }
            )
        if size_matches and hash_matches:
            verified_file_count += 1
    verified = not (
        unsafe_members
        or special_members
        or duplicate_files
        or missing
        or extras
        or size_mismatches
        or hash_mismatches
    )
    return {
        "schema_version": "task4_snapshot_archive_member_verification_v1",
        "archive": str(archive),
        "manifest": str(manifest),
        "root_name": root_name,
        "expected_file_count": len(expected),
        "verified_file_count": verified_file_count,
        "unsafe_members": unsafe_members,
        "special_members": special_members,
        "duplicate_files": duplicate_files,
        "missing_files": missing,
        "extra_files": extras,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "verified": verified,
        "status": "verified" if verified else "failed",
    }


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("snapshot manifest header mismatch")
        rows = list(reader)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        relative = str(row["relative_path"])
        if relative in result:
            raise ValueError(f"duplicate snapshot manifest path: {relative}")
        result[relative] = {
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]).lower(),
        }
    return result


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)
