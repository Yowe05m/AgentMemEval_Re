"""Copy a pinned HF cache snapshot into independently verified regular files.

The source cache is read-only. No hard links, overwrite, cleanup, downloads or
model imports occur. A failed destination is retained as incomplete evidence.
This establishes file identity, never GPU loading or experiment admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.path1_v1.operations.contracts import MODEL, MODEL_REVISION

MODEL_SOURCE_SHA256 = "f2f77fd238c64abaa1f610fe5176c59ea812e82c43e10963bc400b8cf91a0a70"


class ModelMaterializationError(ValueError):
    pass


def pinned_model_source() -> dict[str, Any]:
    raw = files("agentmemeval.path1_v1").joinpath("assets/model_source_files.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SOURCE_SHA256:
        raise ModelMaterializationError("packaged model source manifest changed")
    manifest = json.loads(raw)
    if manifest["model"] != MODEL or manifest["revision"] != MODEL_REVISION:
        raise ModelMaterializationError("model source is not the frozen candidate")
    return manifest


def _relative(value: str) -> str:
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/"))):
        raise ModelMaterializationError("unsafe model filename")
    return value


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ModelMaterializationError("model input is not a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _copy_verified(source: Path, destination: Path, expected: dict) -> dict:
    """The private primitive is also exercised with small independent fixtures."""
    expected_size = expected["size"]
    if type(expected_size) is not int or expected_size < 0:
        raise ModelMaterializationError("invalid source size")
    before = _identity(source)
    if before[2] != expected_size:
        raise ModelMaterializationError("source size differs from official metadata")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {expected_size}\0".encode())
    total = 0
    # Copy actual bytes; never share the cache inode by hard-linking it.
    with source.open("rb") as reader, destination.open("xb") as writer:
        opened = os.fstat(reader.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before:
            raise ModelMaterializationError("source identity changed before copying")
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            total += len(chunk)
            if total > expected_size:
                raise ModelMaterializationError("source grew while copying")
            sha256.update(chunk)
            git_blob.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if total != expected_size or _identity(source) != before:
        raise ModelMaterializationError("source changed while copying")
    if "lfs" in expected:
        valid = (expected["lfs"]["size"] == expected_size
                 and sha256.hexdigest() == expected["lfs"]["sha256"])
    else:
        valid = git_blob.hexdigest() == expected["blobId"]
    if not valid:
        raise ModelMaterializationError("source content differs from pinned Hub object")
    recovered = hashlib.sha256()
    with destination.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            recovered.update(chunk)
    if destination.is_symlink() or destination.stat().st_size != total:
        raise ModelMaterializationError("destination is not an independent regular file")
    if recovered.hexdigest() != sha256.hexdigest():
        raise ModelMaterializationError("copied bytes did not verify")
    return {"path": expected["rfilename"], "size": total, "sha256": sha256.hexdigest()}


def _resolve_cached_file(snapshot: Path, blobs: Path, record: dict) -> Path:
    name = _relative(record["rfilename"])
    candidate = snapshot.joinpath(*PurePosixPath(name).parts)
    for part in candidate.parents:
        if part == snapshot:
            break
        if part.is_symlink():
            raise ModelMaterializationError("nested symlink directory in model snapshot")
    resolved = candidate.resolve(strict=True)
    if candidate.is_symlink():
        expected_blob = record.get("lfs", {}).get("sha256", record["blobId"])
        if resolved.parent != blobs or resolved.name != expected_blob:
            raise ModelMaterializationError("cache symlink escapes the expected Hub blob")
    elif resolved.parent != snapshot and not resolved.is_relative_to(snapshot):
        raise ModelMaterializationError("cache input escapes snapshot")
    _identity(resolved)
    return resolved


def materialize_model(snapshot_path: str | Path, destination_path: str | Path) -> dict:
    manifest = pinned_model_source()
    snapshot_input = Path(snapshot_path)
    destination_input = Path(destination_path)
    if not snapshot_input.is_absolute() or not destination_input.is_absolute():
        raise ModelMaterializationError("source and destination must be absolute")
    snapshot = snapshot_input.resolve(strict=True)
    if snapshot.name != MODEL_REVISION or snapshot.parent.name != "snapshots":
        raise ModelMaterializationError("cache snapshot is not the pinned revision layout")
    if snapshot.parent.parent.name != "models--Qwen--Qwen3.5-9B":
        raise ModelMaterializationError("cache snapshot is for another model")
    blobs = (snapshot.parent.parent / "blobs").resolve(strict=True)
    destination = destination_input.parent.resolve(strict=True) / destination_input.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if (destination.is_relative_to(snapshot.parent.parent)
            or destination.is_relative_to(blobs)):
        raise ModelMaterializationError("destination must be outside the shared model cache")
    names = [_relative(row["rfilename"]) for row in manifest["files"]]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ModelMaterializationError("duplicate source filenames")
    observed = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*")
                if p.is_file() or p.is_symlink()}
    if observed != set(names):
        raise ModelMaterializationError("snapshot inventory differs from pinned revision")
    sources = [_resolve_cached_file(snapshot, blobs, row) for row in manifest["files"]]
    destination.mkdir(exist_ok=False)
    copied = []
    for source, row in zip(sources, manifest["files"], strict=True):
        path = destination / row["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        copied.append(_copy_verified(source, path, row))
    inventory = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
                 if p.is_file() or p.is_symlink()}
    if inventory != set(names):
        raise ModelMaterializationError("materialized inventory changed")
    # Earlier files can change while a later multi-gigabyte shard is copied.
    # Re-read the entire completed set, not merely the last copied file.
    if destination.is_symlink():
        raise ModelMaterializationError("materialized root changed to a symlink")
    for row in copied:
        path = destination / row["path"]
        current = path
        while current != destination:
            if current.is_symlink():
                raise ModelMaterializationError("materialized input changed to a symlink")
            current = current.parent
        before = _identity(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if (before != _identity(path) or before[2] != row["size"]
                or digest.hexdigest() != row["sha256"]):
            raise ModelMaterializationError("materialized file changed before completion")
    return {"schema": "path1-materialized-model-v1", "model": MODEL, "revision": MODEL_REVISION,
            "source_manifest_sha256": MODEL_SOURCE_SHA256, "source": str(snapshot),
            "destination": str(destination), "files": sorted(copied, key=lambda r: r["path"]),
            "runtime_admission": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-snapshot", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    receipt = Path(args.receipt)
    destination = Path(args.destination).absolute()
    if not receipt.is_absolute() or receipt.exists() or receipt.is_symlink():
        raise ModelMaterializationError("receipt must be a new absolute file")
    if receipt.resolve().is_relative_to(destination.resolve()):
        raise ModelMaterializationError("receipt must be outside the sealed model directory")
    # Reserve before the long copy. A crash leaves an empty/partial receipt, not
    # a reusable destination or a false success marker.
    with receipt.open("x", encoding="utf-8") as stream:
        result = materialize_model(args.cache_snapshot, args.destination)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
