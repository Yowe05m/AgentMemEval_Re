"""Short-lived, process-local reuse of an actual full model-file verification.

The original contracts._model_identity remains the content-hash producer.
Every reuse checks the complete directory inventory, kernel file identities and
manifest bytes. A diagnostic JSON record is never an authority to reuse hashes.
This detects ordinary filesystem changes; it is not an adversarial filesystem
attestation or evidence that a backend loaded these files into model tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

from agentmemeval.path1_v1.operations.contracts import _model_identity, canonical_json


class ModelVerificationError(ValueError):
    pass


class VerifiedModelFiles:
    """Opaque handle. Only the exact object registered by the producer is valid."""

    __slots__ = ("__weakref__",)

    def __reduce_ex__(self, protocol):
        raise TypeError("model verification cannot be serialized or copied")


@dataclass(frozen=True)
class _VerifiedRecord:
    pid: int
    manifest_path: Path
    model_root: Path
    binding_sha256: str
    identity_bytes: bytes
    state: tuple
    verified_monotonic: float
    max_age_seconds: float
    full_hash_seconds: float


_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelVerificationError(message)


def _canonical(path: Path | str) -> Path:
    path = Path(path)
    _require(path.is_absolute(), "model verification requires absolute paths")
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             "symlink model/manifest paths are unsupported")
    resolved = path.resolve(strict=True)
    _require(path == resolved, "model verification paths must be canonical")
    return resolved


def _stat(path: Path) -> tuple:
    value = path.stat(follow_symlinks=False)
    _require(not stat.S_ISLNK(value.st_mode), "model symlink appeared")
    _require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode),
             "model inventory contains a non-regular input")
    # atime is deliberately excluded: verification and loading may update it.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_uid, value.st_gid, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns)


def _inventory(root: Path) -> tuple:
    _canonical(root)
    nodes = [(".", _stat(root))]
    _require(stat.S_ISDIR(nodes[0][1][2]), "model root is not a directory")
    for path in sorted(root.rglob("*")):
        nodes.append((path.relative_to(root).as_posix(), _stat(path)))
    return tuple(nodes)


def _state(manifest: Path, root: Path) -> tuple:
    _canonical(manifest)
    before = _inventory(root)
    manifest_stat = _stat(manifest)
    _require(stat.S_ISREG(manifest_stat[2]), "model manifest is not a regular file")
    raw = manifest.read_bytes()
    _require(manifest_stat == _stat(manifest), "model manifest changed while reading")
    _canonical(manifest)
    _require(before == _inventory(root), "model inventory changed while reading")
    return before, manifest_stat, raw


def _binding(value: str) -> None:
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value),
             "model verification requires an exact request SHA256 binding")


def verify_model_files(
    manifest_path: Path | str, model_root: Path | str, *, expected_identity: dict,
    binding_sha256: str, max_age_seconds: float = 120.0,
) -> VerifiedModelFiles:
    """Perform the original full content check and register a local reuse handle."""
    _binding(binding_sha256)
    _require(type(max_age_seconds) in (int, float) and math.isfinite(max_age_seconds)
             and 0 < max_age_seconds <= 6600, "invalid model verification lifetime")
    manifest, root = _canonical(manifest_path), _canonical(model_root)
    started = time.monotonic()
    before = _state(manifest, root)
    identity = _model_identity(manifest, root)
    _require(canonical_json(identity) == canonical_json(expected_identity),
             "fully verified model identity differs from the contract")
    _require(before == _state(manifest, root), "model files changed during full hashing")
    _require(identity["manifest_sha256"] == hashlib.sha256(before[2]).hexdigest(),
             "model manifest differs from the verified bytes")
    verified = time.monotonic()
    _require(math.isfinite(verified) and verified >= started, "invalid verification clock")
    capability = VerifiedModelFiles()
    _ISSUED[capability] = _VerifiedRecord(
        os.getpid(), manifest, root, binding_sha256, canonical_json(identity), before,
        verified, float(max_age_seconds), verified - started,
    )
    return capability


def _record(capability: VerifiedModelFiles) -> _VerifiedRecord:
    _require(type(capability) is VerifiedModelFiles and capability in _ISSUED,
             "model verification was not issued as this exact process-local object")
    record = _ISSUED[capability]
    _require(record.pid == os.getpid(), "model verification belongs to another process")
    age = time.monotonic() - record.verified_monotonic
    _require(math.isfinite(age) and 0 <= age <= record.max_age_seconds,
             "model verification expired or monotonic clock moved backwards")
    return record


def reuse_model_files(
    capability: VerifiedModelFiles, manifest_path: Path | str, model_root: Path | str, *,
    binding_sha256: str, expected_identity: dict | None = None,
) -> dict:
    """Recheck every file/manifest identity; never fall back to a slow full scan."""
    _binding(binding_sha256)
    record = _record(capability)
    _require(record.manifest_path == _canonical(manifest_path)
             and record.model_root == _canonical(model_root)
             and record.binding_sha256 == binding_sha256,
             "model verification path/request binding differs")
    _require(record.state == _state(record.manifest_path, record.model_root),
             "model files or manifest changed since full verification")
    if expected_identity is not None:
        _require(record.identity_bytes == canonical_json(expected_identity),
                 "model verification expected identity differs")
    _record(capability)  # A slow metadata read must not outlive the fixed lease.
    return json.loads(record.identity_bytes)


def model_verification_diagnostic(capability: VerifiedModelFiles) -> dict:
    """Describe provenance, not a serializable capability or runtime admission."""
    record = _record(capability)
    return {
        "schema_version": 1, "kind": "process_local_model_file_verification",
        "pid": record.pid, "manifest_path": str(record.manifest_path),
        "model_root": str(record.model_root), "binding_sha256": record.binding_sha256,
        "identity_sha256": hashlib.sha256(record.identity_bytes).hexdigest(),
        "verified_monotonic": record.verified_monotonic,
        "max_age_seconds": record.max_age_seconds, "full_hash_seconds": record.full_hash_seconds,
        "metadata": [{"path": path, "device": values[0], "inode": values[1],
                      "mode": values[2], "nlink": values[3], "uid": values[4],
                      "gid": values[5], "size": values[6], "mtime_ns": values[7],
                      "ctime_ns": values[8]} for path, values in record.state[0]],
        "restores_capability": False, "runtime_admission": False,
    }
