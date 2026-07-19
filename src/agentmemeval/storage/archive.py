"""File-level SHA-256 manifests for portable, non-destructive result archives."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FIELDS = ("relative_path", "size_bytes", "sha256")
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def build_file_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Write an exclusive TSV manifest for every regular file below root."""

    source = Path(root).absolute()
    if source.is_symlink():
        raise ValueError("archive root must not be a symlink")
    directory = source.resolve()
    target = Path(output).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if target == directory or directory in target.parents:
        raise ValueError("manifest output must be outside the archived root")
    filesystem_directory = _filesystem_path(directory)
    rows = []
    paths = sorted(
        filesystem_directory.rglob("*"),
        key=lambda item: item.relative_to(filesystem_directory).as_posix(),
    )
    for filesystem_path in paths:
        relative = filesystem_path.relative_to(filesystem_directory).as_posix()
        if filesystem_path.is_symlink():
            raise ValueError(f"archive root contains a symlink: {relative}")
        if filesystem_path.is_file():
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": filesystem_path.stat().st_size,
                    "sha256": _sha256(filesystem_path),
                }
            )
    filesystem_target = _filesystem_path(target)
    filesystem_target.parent.mkdir(parents=True, exist_ok=True)
    with filesystem_target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": "task4_file_manifest_v1",
        "root": str(directory),
        "manifest": str(target),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_sha256": _sha256(filesystem_target),
    }


def verify_file_manifest(
    root: str | Path, manifest: str | Path, *, reject_extra_files: bool = True
) -> dict[str, Any]:
    """Verify sizes and hashes, rejecting unsafe paths and optionally extra files."""

    source = Path(root).absolute()
    manifest_source = Path(manifest).absolute()
    root_is_symlink = source.is_symlink()
    manifest_is_symlink = manifest_source.is_symlink()
    directory = source.resolve()
    manifest_path = manifest_source.resolve()
    filesystem_manifest = _filesystem_path(manifest_path)
    with filesystem_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    format_errors = []
    if header != MANIFEST_FIELDS:
        format_errors.append(
            {
                "kind": "header_mismatch",
                "observed": list(header),
                "expected": list(MANIFEST_FIELDS),
            }
        )
    seen: set[str] = set()
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, str]] = []
    unsafe: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    verified_file_count = 0
    for row_number, row in enumerate(rows, start=2):
        relative = str(row.get("relative_path", ""))
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in seen
        ):
            unsafe.append(relative)
            continue
        seen.add(relative)
        try:
            expected_size = int(str(row.get("size_bytes", "")))
        except (TypeError, ValueError, OverflowError):
            invalid_rows.append(
                {
                    "row_number": row_number,
                    "relative_path": relative,
                    "reason": "invalid_size_bytes",
                    "value": row.get("size_bytes"),
                }
            )
            continue
        if expected_size < 0:
            invalid_rows.append(
                {
                    "row_number": row_number,
                    "relative_path": relative,
                    "reason": "negative_size_bytes",
                    "value": expected_size,
                }
            )
            continue
        expected_hash = str(row.get("sha256", ""))
        if SHA256_PATTERN.fullmatch(expected_hash) is None:
            invalid_rows.append(
                {
                    "row_number": row_number,
                    "relative_path": relative,
                    "reason": "invalid_sha256",
                    "value": expected_hash,
                }
            )
            continue
        path = directory.joinpath(*pure.parts)
        filesystem_path = _filesystem_path(path)
        if not filesystem_path.is_file() or filesystem_path.is_symlink():
            missing.append(relative)
            continue
        observed_size = filesystem_path.stat().st_size
        if observed_size != expected_size:
            size_mismatches.append(
                {
                    "relative_path": relative,
                    "expected": expected_size,
                    "observed": observed_size,
                }
            )
            continue
        expected_hash = expected_hash.lower()
        observed_hash = _sha256(filesystem_path)
        if observed_hash != expected_hash:
            hash_mismatches.append(
                {
                    "relative_path": relative,
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
            continue
        verified_file_count += 1
    filesystem_directory = _filesystem_path(directory)
    observed_paths = list(filesystem_directory.rglob("*"))
    symlinks = sorted(
        path.relative_to(filesystem_directory).as_posix()
        for path in observed_paths
        if path.is_symlink()
    )
    extras = []
    if reject_extra_files:
        observed = {
            path.relative_to(filesystem_directory).as_posix()
            for path in observed_paths
            if path.is_file() and not path.is_symlink()
        }
        extras = sorted(observed - seen)
    verified = not (
        root_is_symlink
        or manifest_is_symlink
        or format_errors
        or invalid_rows
        or unsafe
        or missing
        or size_mismatches
        or hash_mismatches
        or symlinks
        or extras
    )
    return {
        "schema_version": "task4_file_manifest_verification_v2",
        "root": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(filesystem_manifest),
        "expected_file_count": len(rows),
        "verified_file_count": verified_file_count,
        "root_is_symlink": root_is_symlink,
        "manifest_is_symlink": manifest_is_symlink,
        "manifest_format_errors": format_errors,
        "invalid_manifest_rows": invalid_rows,
        "unsafe_or_duplicate_paths": unsafe,
        "missing_files": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "symlinks": symlinks,
        "extra_files": extras,
        "verified": verified,
        "status": "verified" if verified else "failed",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filesystem_path(path: Path) -> Path:
    """Use the Win32 extended path namespace so 260-char evidence paths remain visible."""

    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)
