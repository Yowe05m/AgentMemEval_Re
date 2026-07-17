"""File-level SHA-256 manifests for portable, non-destructive result archives."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FIELDS = ("relative_path", "size_bytes", "sha256")


def build_file_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Write an exclusive TSV manifest for every regular file below root."""

    directory = Path(root).resolve()
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

    directory = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    filesystem_manifest = _filesystem_path(manifest_path)
    with filesystem_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    seen: set[str] = set()
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, str]] = []
    unsafe: list[str] = []
    for row in rows:
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
        path = directory.joinpath(*pure.parts)
        filesystem_path = _filesystem_path(path)
        if not filesystem_path.is_file() or filesystem_path.is_symlink():
            missing.append(relative)
            continue
        expected_size = int(row["size_bytes"])
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
        expected_hash = str(row["sha256"]).lower()
        observed_hash = _sha256(filesystem_path)
        if observed_hash != expected_hash:
            hash_mismatches.append(
                {
                    "relative_path": relative,
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
    extras = []
    if reject_extra_files:
        filesystem_directory = _filesystem_path(directory)
        observed = {
            path.relative_to(filesystem_directory).as_posix()
            for path in filesystem_directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        extras = sorted(observed - seen)
    verified = not (unsafe or missing or size_mismatches or hash_mismatches or extras)
    return {
        "schema_version": "task4_file_manifest_verification_v1",
        "root": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(filesystem_manifest),
        "expected_file_count": len(rows),
        "verified_file_count": len(rows)
        - len(unsafe)
        - len(missing)
        - len(size_mismatches)
        - len(hash_mismatches),
        "unsafe_or_duplicate_paths": unsafe,
        "missing_files": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
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
