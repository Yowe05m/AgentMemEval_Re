"""Authorized TASK8B task-1 adoption without changing the frozen runner.

Run this file directly with the Python environment used by the frozen checkout.
It deliberately uses only the standard library until the checkout and its Git
identity have been verified.  The only runtime monkeypatch is the known
resolved-config identity canonicalizer in the frozen formal runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FROZEN_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
AUTHORIZATION_SCHEMA = "task8b-same-attempt-recovery-authorization-v1"
ADOPTION_SCHEMA = "task8b-same-attempt-task-adoption-v1"
TASK_RECEIPT_SCHEMA = "task8-worker-task-receipt-v1"
TASK_IDENTITY_SCHEMA = "task8-task-identity-audit-v1"
AUTHORIZED_REASON = "resolved-config-integer-key-canonicalization"
PROTOCOL_AMENDMENT_ID = "TASK8B-VFPR-20260722"
TASK_ID = "isolation_no_memory"
HEALTH_ZERO_FIELDS = (
    "fallback_count",
    "memory_revision_fallback_count",
    "reward_conservation_violation_count",
    "stack_conservation_violation_count",
)
REQUIRED_AUTHORIZATION_FIELDS = {
    "schema_version",
    "authorized",
    "authorization_id",
    "worker_id",
    "task_id",
    "reason",
    "frozen_code_sha",
    "formal_runner_sha256",
    "controller_sha256",
    "worker_manifest_sha256",
    "baseline_manifest_sha256",
    "pre_recovery_archive_sha256",
    "protocol_amendment_id",
    "protocol_amendment_sha256",
    "parent_preunlock_sha256",
    "active_pid_absence_evidence_sha256",
    "verifier_code_sha",
    "verifier_identity_source_sha256",
    "failed_state_row_sha256",
    "attempt_root",
    "active_process_absent_confirmed",
    "same_attempt_recovery_authorized",
    "task1_adoption_only",
}


class RecoveryError(RuntimeError):
    """Fail-closed recovery admission error."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON root must be an object: {path}")
    return value


def canonicalize_resolved_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve integer mapping keys while removing runner-dynamic fields."""

    value = copy.deepcopy(config)
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in (
        "output_root",
        "run_id",
        "initial_memory_snapshots",
        "admission_audit",
    ):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value


def _legacy_json_roundtrip_canonicalizer(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config, ensure_ascii=False))
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in (
        "output_root",
        "run_id",
        "initial_memory_snapshots",
        "admission_audit",
    ):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or not relative
        or ".." in candidate_relative.parts
    ):
        raise RecoveryError(f"unsafe relative path: {relative!r}")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RecoveryError(f"path escapes attempt root: {relative!r}") from exc
    return candidate


def _verify_checkout(checkout: Path, expected_runner_sha256: str) -> Path:
    checkout = checkout.resolve()
    runner_path = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    if not runner_path.is_file() or runner_path.is_symlink():
        raise RecoveryError("frozen formal_runner.py missing or is a symlink")
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError("unable to verify frozen checkout") from exc
    if head != FROZEN_CODE_SHA:
        raise RecoveryError(f"frozen checkout HEAD mismatch: {head}")
    if dirty.strip():
        raise RecoveryError("frozen checkout has tracked changes")
    if _sha256_file(runner_path) != expected_runner_sha256:
        raise RecoveryError("formal runner SHA-256 mismatch")
    return runner_path


def _verify_clean_commit(checkout: Path, expected_commit: str) -> None:
    if len(expected_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in expected_commit
    ):
        raise RecoveryError("verifier_code_sha must be a lowercase 40-hex commit")
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError("unable to verify verifier checkout") from exc
    if head != expected_commit or dirty:
        raise RecoveryError("verifier checkout is not the authorized clean commit")


def _load_frozen_runner(
    checkout: Path, expected_runner_sha256: str
) -> tuple[Any, Callable]:
    _verify_checkout(checkout, expected_runner_sha256)
    if any(
        name == "agentmemeval" or name.startswith("agentmemeval.")
        for name in sys.modules
    ):
        raise RecoveryError(
            "agentmemeval was imported before frozen-checkout verification"
        )
    sys.path.insert(0, str((checkout / "src").resolve()))
    runner = importlib.import_module("agentmemeval.experiments.formal_runner")
    loader = importlib.import_module("agentmemeval.config.loader").load_config
    loaded_path = Path(runner.__file__).resolve()
    expected_path = (
        checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    ).resolve()
    if loaded_path != expected_path or not hasattr(runner, "_semantic_config"):
        raise RecoveryError("did not import the expected a1d1 frozen formal runner")
    return runner, loader


def _validate_authorization(
    authorization: dict[str, Any],
    *,
    authorization_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    attempt_root: Path,
    controller_path: Path,
    protocol_amendment_path: Path,
    parent_preunlock_path: Path,
    archive_path: Path,
    pid_absence_path: Path,
    verifier_checkout: Path,
    protocol_amendment_id: str,
) -> None:
    missing = sorted(REQUIRED_AUTHORIZATION_FIELDS - set(authorization))
    if missing:
        raise RecoveryError(f"authorization missing fields: {', '.join(missing)}")
    checks = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "task_id": TASK_ID,
        "reason": AUTHORIZED_REASON,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "active_process_absent_confirmed": True,
        "same_attempt_recovery_authorized": True,
        "task1_adoption_only": True,
    }
    for field, expected in checks.items():
        if authorization.get(field) != expected:
            raise RecoveryError(f"authorization {field} mismatch")
    if not str(authorization.get("authorization_id", "")).strip():
        raise RecoveryError("authorization_id must be non-empty")
    if (
        protocol_amendment_id != PROTOCOL_AMENDMENT_ID
        or authorization.get("protocol_amendment_id") != PROTOCOL_AMENDMENT_ID
    ):
        raise RecoveryError("protocol amendment id mismatch")
    if Path(str(authorization["attempt_root"])).resolve() != attempt_root.resolve():
        raise RecoveryError("authorization attempt_root mismatch")
    if _sha256_file(controller_path) != authorization["controller_sha256"]:
        raise RecoveryError("controller SHA-256 mismatch")
    if _sha256_file(manifest_path) != authorization["worker_manifest_sha256"]:
        raise RecoveryError("worker manifest SHA-256 mismatch")
    if _sha256_file(baseline_path) != authorization["baseline_manifest_sha256"]:
        raise RecoveryError("baseline manifest SHA-256 mismatch")
    evidence = {
        protocol_amendment_path: "protocol_amendment_sha256",
        parent_preunlock_path: "parent_preunlock_sha256",
        archive_path: "pre_recovery_archive_sha256",
        pid_absence_path: "active_pid_absence_evidence_sha256",
    }
    for evidence_path, field in evidence.items():
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise RecoveryError(f"evidence file missing or symlinked: {field}")
        if _sha256_file(evidence_path) != authorization[field]:
            raise RecoveryError(f"evidence SHA-256 mismatch: {field}")
    pid_evidence = _read_json(pid_absence_path)
    if (
        pid_evidence.get("active_process_absent") is not True
        or int(pid_evidence.get("observed_pid", 0)) <= 0
        or not str(pid_evidence.get("probe_utc", "")).strip()
    ):
        raise RecoveryError("active PID absence evidence is not affirmative")
    _verify_clean_commit(
        verifier_checkout.resolve(), str(authorization["verifier_code_sha"])
    )
    identity_source = (
        verifier_checkout.resolve()
        / "src"
        / "agentmemeval"
        / "experiments"
        / "formal_protocol.py"
    )
    if (
        not identity_source.is_file()
        or _sha256_file(identity_source)
        != authorization["verifier_identity_source_sha256"]
    ):
        raise RecoveryError("verifier identity source SHA-256 mismatch")
    # Ensure the authorization itself is a stable file before any evidence write.
    if not authorization_path.is_file() or authorization_path.is_symlink():
        raise RecoveryError("authorization must be a regular file")


@dataclass(frozen=True)
class _BaselineManifest:
    layout: str
    rows: dict[str, tuple[int, str]]


@dataclass(frozen=True)
class _ArchiveClosure:
    attempt_prefix: str
    attempt_rows: dict[str, tuple[int, str]]


def _safe_posix_relative(value: str, *, label: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or not value.isascii()
        or "\\" in value
        or "\x00" in value
        or candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise RecoveryError(f"{label} path is invalid: {value!r}")
    return value


def _read_baseline(path: Path) -> _BaselineManifest:
    rows: dict[str, tuple[int, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames == ["relative_path", "size", "sha256"]:
                layout = "canonical-one-root"
                size_field = "size"
            elif reader.fieldnames == ["sha256", "bytes", "relative_path"]:
                layout = "legacy-composite"
                size_field = "bytes"
            else:
                raise RecoveryError("baseline manifest header mismatch")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise RecoveryError("baseline manifest row width mismatch")
                relative = _safe_posix_relative(
                    str(row["relative_path"]), label="baseline"
                )
                if relative in rows:
                    raise RecoveryError(f"duplicate baseline path: {relative}")
                raw_size = str(row[size_field])
                size = int(raw_size)
                if size < 0 or str(size) != raw_size:
                    raise RecoveryError(f"invalid baseline size: {relative}")
                sha256 = str(row["sha256"])
                if len(sha256) != 64 or any(
                    char not in "0123456789abcdef" for char in sha256
                ):
                    raise RecoveryError(f"invalid baseline SHA-256: {relative}")
                rows[relative] = (size, sha256)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RecoveryError("invalid baseline manifest") from exc
    if not rows:
        raise RecoveryError("baseline manifest is empty")
    return _BaselineManifest(layout=layout, rows=rows)


def _verify_baseline_subset(
    attempt_root: Path,
    attempt_baseline: dict[str, tuple[int, str]],
    *,
    append_only_paths: set[str] | None = None,
) -> dict[str, tuple[int, str]]:
    append_only_paths = append_only_paths or set()
    for relative, (size, sha256) in attempt_baseline.items():
        path = _inside(attempt_root, relative)
        if not path.is_file() or path.is_symlink():
            raise RecoveryError(f"baseline file missing or symlinked: {relative}")
        if relative in append_only_paths:
            continue
        if path.stat().st_size != size or _sha256_file(path) != sha256:
            raise RecoveryError(f"baseline file integrity mismatch: {relative}")
    return attempt_baseline


def _verify_archive_closure(
    archive_path: Path, baseline: _BaselineManifest, manifest: dict[str, Any]
) -> _ArchiveClosure:
    archived: dict[str, tuple[int, str]] = {}
    archived_directories: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            def is_regular(member: tarfile.TarInfo) -> bool:
                sparse_keys = {
                    key for key in member.pax_headers if key.startswith("GNU.sparse")
                }
                return (
                    member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                    and not member.sparse
                    and not member.linkname
                    and not sparse_keys
                )

            def is_directory(member: tarfile.TarInfo) -> bool:
                return member.type == tarfile.DIRTYPE and not member.linkname

            if any(not (is_directory(member) or is_regular(member)) for member in members):
                raise RecoveryError(
                    "pre-recovery archive permits only directories and regular files"
                )
            parsed_names: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
            seen_member_names: set[str] = set()
            for member in members:
                name = PurePosixPath(member.name)
                normalized_name = name.as_posix()
                raw_name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                if (
                    name.is_absolute()
                    or not name.parts
                    or ".." in name.parts
                    or any(part in {"", "."} for part in name.parts)
                    or not raw_name.isascii()
                    or "\\" in raw_name
                    or "\x00" in raw_name
                    or raw_name != normalized_name
                    or normalized_name in seen_member_names
                ):
                    raise RecoveryError("pre-recovery archive member path is invalid")
                seen_member_names.add(normalized_name)
                parsed_names.append((member, name))
            regular_members = [member for member, _ in parsed_names if is_regular(member)]
            if len(regular_members) != len(baseline.rows) or sum(
                member.size for member in regular_members
            ) != sum(size for size, _ in baseline.rows.values()):
                raise RecoveryError("pre-recovery archive size/count mismatch")
            file_names = [name.as_posix() for member, name in parsed_names if is_regular(member)]
            if baseline.layout == "canonical-one-root":
                roots = {PurePosixPath(name).parts[0] for name in file_names}
                if len(roots) != 1:
                    raise RecoveryError(
                        "pre-recovery archive must have exactly one top-level root"
                    )
                root = next(iter(roots))
                expected_archive_rows = {
                    f"{root}/{relative}": value
                    for relative, value in baseline.rows.items()
                }
            else:
                expected_archive_rows = baseline.rows
            if set(file_names) != set(expected_archive_rows):
                raise RecoveryError(
                    "pre-recovery archive is not closed by the baseline manifest"
                )
            for member, name in parsed_names:
                if is_directory(member):
                    archived_directories.add(name.as_posix())
                    continue
                relative = name.as_posix()
                if relative in archived:
                    raise RecoveryError("pre-recovery archive path set is invalid")
                expected_size, _ = expected_archive_rows[relative]
                if member.size != expected_size:
                    raise RecoveryError("pre-recovery archive member size mismatch")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RecoveryError("unable to read pre-recovery archive member")
                digest = hashlib.sha256()
                size = 0
                while True:
                    block = extracted.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
                archived[relative] = (size, digest.hexdigest())
    except (OSError, tarfile.TarError) as exc:
        raise RecoveryError("invalid pre-recovery archive") from exc
    if baseline.layout == "canonical-one-root":
        roots = {PurePosixPath(name).parts[0] for name in archived}
        if len(roots) != 1:
            raise RecoveryError(
                "pre-recovery archive must have exactly one top-level root"
            )
        stripped = {
            PurePosixPath(*PurePosixPath(relative).parts[1:]).as_posix(): value
            for relative, value in archived.items()
        }
        if stripped != baseline.rows:
            raise RecoveryError(
                "pre-recovery archive is not closed by the baseline manifest"
            )
        return _ArchiveClosure(
            attempt_prefix=next(iter(roots)), attempt_rows=baseline.rows
        )
    if archived != baseline.rows:
        raise RecoveryError(
            "pre-recovery archive is not closed by the baseline manifest"
        )
    worker_id = str(manifest.get("worker_id", ""))
    seed = str(manifest.get("seed_bundle", ""))
    output_path = str(manifest.get("instance_identity", {}).get("output_path", ""))
    expected_output = f"outputs/formal/task8b/{worker_id}/{seed}"
    if output_path != expected_output:
        raise RecoveryError("legacy archive manifest output_path mismatch")
    suffix = f"/{expected_output}/worker_manifest.json"
    candidates = [name for name in archived if name.endswith(suffix)]
    if len(candidates) != 1:
        raise RecoveryError("legacy archive must identify exactly one manifest")
    attempt_prefix = candidates[0][: -len("/worker_manifest.json")]
    attempt_rows = {
        name[len(attempt_prefix) + 1 :]: value
        for name, value in archived.items()
        if name.startswith(attempt_prefix + "/")
    }
    attempt_full_names = {
        attempt_prefix + "/" + relative for relative in attempt_rows
    }
    external = {
        name: value for name, value in archived.items() if name not in attempt_full_names
    }
    required = {
        "worker_manifest.json",
        "state.tsv",
        f"runs/{TASK_ID}/hand_summaries.jsonl",
    }
    nested_manifests = [
        relative
        for relative in attempt_rows
        if relative.endswith("/worker_manifest.json")
    ]
    if (
        len(baseline.rows) != 42
        or len(attempt_rows) != 38
        or len(external) != 4
        or not required.issubset(attempt_rows)
        or nested_manifests
    ):
        raise RecoveryError("legacy archive external evidence count mismatch")
    manifest_row = attempt_rows.get("worker_manifest.json")
    global_manifest_names = [
        name for name in external if name.endswith(f"/manifests/{worker_id}.json")
    ]
    global_manifest = (
        external.get(global_manifest_names[0])
        if len(global_manifest_names) == 1
        else None
    )
    if manifest_row is None or global_manifest != manifest_row:
        raise RecoveryError("legacy archive global manifest mismatch")
    runtime_names = [name for name in external if name not in global_manifest_names]
    runtime_parents = {str(PurePosixPath(name).parent) for name in runtime_names}
    if len(runtime_names) != 3 or len(runtime_parents) != 1:
        raise RecoveryError("legacy archive runtime evidence mismatch")
    runtime_parent = next(iter(runtime_parents))
    runtime_parent_parts = PurePosixPath(runtime_parent).parts
    expected_runtime_names = {
        f"{runtime_parent}/{worker_id}_attempt01.pid",
        f"{runtime_parent}/{worker_id}_attempt01.log",
        f"{runtime_parent}/{worker_id}_attempt01.exitcode",
    }
    if (
        set(runtime_names) != expected_runtime_names
        or len(runtime_parent_parts) < 5
        or runtime_parent_parts[-4] != "runtime_control"
        or runtime_parent_parts[-2:] != (worker_id, "attempt01")
        or PurePosixPath(global_manifest_names[0]).parts[0]
        != runtime_parent_parts[0]
    ):
        raise RecoveryError("legacy archive runtime evidence identity mismatch")
    file_ancestors = {
        PurePosixPath(*PurePosixPath(name).parts[:index]).as_posix()
        for name in archived
        for index in range(1, len(PurePosixPath(name).parts))
    }
    allowed_empty = runtime_parent + f"/{worker_id}_attempt01.launch.lock"
    matplotlib_empty = (
        attempt_prefix + f"/runs/{TASK_ID}/plots/.matplotlib"
    )
    if archived_directories - file_ancestors != {
        allowed_empty,
        matplotlib_empty,
    }:
        raise RecoveryError("legacy archive contains an unexpected empty directory")
    return _ArchiveClosure(attempt_prefix=attempt_prefix, attempt_rows=attempt_rows)


def _current_files(attempt_root: Path) -> set[str]:
    if any(path.is_symlink() for path in attempt_root.rglob("*")):
        raise RecoveryError("attempt tree contains a symlink")
    return {
        path.relative_to(attempt_root).as_posix()
        for path in attempt_root.rglob("*")
        if path.is_file()
    }


def _state_rows(state_path: Path) -> list[dict[str, str]]:
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _verify_state_chain_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RecoveryError("state.tsv is empty")
    previous = "GENESIS"
    for row in rows:
        body = {
            "schema_version": row.get("schema_version"),
            "created_at_utc": row.get("created_at_utc"),
            "status": row.get("status"),
            "detail": row.get("detail"),
            "previous_sha256": row.get("previous_sha256"),
        }
        expected = hashlib.sha256(
            json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if row.get("previous_sha256") != previous or row.get("row_sha256") != expected:
            raise RecoveryError("state.tsv hash chain mismatch")
        previous = expected


def _prepare_task1_config(
    manifest: dict[str, Any],
    raw_task: dict[str, Any],
    run_dir: Path,
    load_config: Callable,
) -> tuple[dict[str, Any], str]:
    config_path = Path(str(raw_task.get("config_path", "")))
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    expected_config_sha256 = str(raw_task.get("config_sha256", ""))
    if not config_path.is_file() or _sha256_file(config_path) != expected_config_sha256:
        raise RecoveryError("task1 config SHA-256 mismatch")
    config = load_config(config_path)
    experiment = config["experiment"]
    experiment["seed"] = int(manifest["seed_bundle"])
    experiment["output_root"] = str(run_dir / "runs")
    experiment["run_id"] = TASK_ID
    experiment["heldout_table_set"] = list(manifest["heldout_table_set"])
    branch_label = str(raw_task.get("memory_mode") or manifest["role"]).lower()
    cache_namespace = (
        f"{manifest['instance_identity']['cache_namespace']}/{TASK_ID}/{branch_label}"
    )
    agent_config = dict(config.get("agent", {}))
    agent_config["embedding_cache_path"] = f"{cache_namespace}/{{agent_id}}.json"
    config["agent"] = agent_config
    if manifest["role"] == "primary" and int(experiment.get("train_hands", 0)) > 0:
        experiment.pop("checkpoint_interval", None)
        experiment["checkpoint_set"] = list(manifest["checkpoint_set"])
    return config, cache_namespace


def _identity_and_health(
    runner: Any,
    manifest: dict[str, Any],
    raw_task: dict[str, Any],
    config: dict[str, Any],
    child_run: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = (
        "manifest.json",
        "schedule_manifest.json",
        "experiment_result.json",
        "protocol_audit.json",
    )
    for name in required:
        path = child_run / name
        if not path.is_file() or path.is_symlink():
            raise RecoveryError(
                f"task1 completion artifact missing or symlinked: {name}"
            )
    metadata = dict(_read_json(child_run / "manifest.json").get("metadata", {}))
    schedule = _read_json(child_run / "schedule_manifest.json")
    actual = {
        "code_sha": dict(metadata.get("code", {})).get("commit"),
        "code_dirty": dict(metadata.get("code", {})).get("dirty"),
        "resolved_config_sha256": runner.sha256_json(runner._semantic_config(config)),
        "prompt_sha256": runner.sha256_json(metadata.get("prompts", {})),
        "model_fingerprint": runner.sha256_json(metadata.get("model", {})),
        "embedding_fingerprint": runner.task8b_embedding_fingerprint(
            metadata.get("embedding", {})
        ),
        "schedule_sha256": schedule.get("schedule_sha256"),
    }
    expected = raw_task.get("expected_identity")
    if not isinstance(expected, dict):
        raise RecoveryError("task1 expected_identity missing")
    for field in runner.REQUIRED_IDENTITY_FIELDS:
        if actual.get(field) != expected.get(field):
            raise RecoveryError(f"task1 identity mismatch: {field}")
    if actual["code_dirty"] is not False:
        raise RecoveryError("task1 code_dirty must be false")
    legacy_hash = runner.sha256_json(_legacy_json_roundtrip_canonicalizer(config))
    corrected_hash = actual["resolved_config_sha256"]
    original_expected_hash = expected.get("resolved_config_sha256")
    semantic_equivalence = _stringify_mapping_keys(
        _legacy_json_roundtrip_canonicalizer(config)
    ) == _stringify_mapping_keys(canonicalize_resolved_config_identity(config))
    if (
        legacy_hash == original_expected_hash
        or corrected_hash != original_expected_hash
        or not semantic_equivalence
    ):
        raise RecoveryError(
            "identity correction is not the unique semantic-equivalent key fix"
        )
    audit = _read_json(child_run / "protocol_audit.json")
    validity = audit.get("run_validity")
    execution = audit.get("execution_health", audit)
    if not isinstance(validity, dict) or validity.get("execution_valid") is not True:
        raise RecoveryError("task1 execution_valid gate failed")
    if validity.get("behavior_valid") is not True:
        raise RecoveryError("task1 behavior_valid gate failed")
    if not isinstance(execution, dict) or execution.get("valid") is not True:
        raise RecoveryError("task1 execution_health.valid gate failed")
    for field in HEALTH_ZERO_FIELDS:
        if int(execution.get(field, -1)) != 0:
            raise RecoveryError(f"task1 health counter is nonzero: {field}")
    health = {
        "execution_valid": True,
        "behavior_valid": True,
        "execution_health_valid": True,
        **{field: 0 for field in HEALTH_ZERO_FIELDS},
    }
    identity_correction = {
        "legacy_json_roundtrip_actual_sha256": legacy_hash,
        "original_expected_sha256": original_expected_hash,
        "corrected_actual_sha256": corrected_hash,
        "semantic_equivalence_after_stringifying_mapping_keys": semantic_equivalence,
        "only_authorized_difference": "integer-versus-string mapping keys",
    }
    return actual, health, identity_correction


def _same_or_atomic_new(runner: Any, path: Path, value: Any) -> None:
    content = _json_bytes(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise RecoveryError(f"refuse to alter existing recovery evidence: {path}")
        return
    runner._write_json_atomic_new(path, value)


def _require_byte_identical(path: Path, value: Any) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.read_bytes() != _json_bytes(value)
    ):
        raise RecoveryError(f"existing recovery evidence is not byte-identical: {path}")


def _count_structural_hands(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError("task1 hand_summaries.jsonl missing or symlinked")
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def execute_recovery(
    *,
    runner: Any,
    load_config: Callable,
    manifest_path: Path,
    receipt_root: Path,
    attempt_root: Path,
    baseline_path: Path,
    authorization_path: Path,
    controller_path: Path,
    protocol_amendment_path: Path,
    parent_preunlock_path: Path,
    archive_path: Path,
    pid_absence_path: Path,
    verifier_checkout: Path,
    protocol_amendment_id: str,
) -> dict[str, Any]:
    """Adopt verified task1, publish its receipt last, then invoke frozen resume."""

    manifest_path = manifest_path.resolve()
    baseline_path = baseline_path.resolve()
    authorization_path = authorization_path.resolve()
    attempt_root = attempt_root.resolve()
    controller_path = controller_path.resolve()
    authorization = _read_json(authorization_path)
    _validate_authorization(
        authorization,
        authorization_path=authorization_path,
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        attempt_root=attempt_root,
        controller_path=controller_path,
        protocol_amendment_path=protocol_amendment_path.resolve(),
        parent_preunlock_path=parent_preunlock_path.resolve(),
        archive_path=archive_path.resolve(),
        pid_absence_path=pid_absence_path.resolve(),
        verifier_checkout=verifier_checkout.resolve(),
        protocol_amendment_id=protocol_amendment_id,
    )
    manifest = _read_json(manifest_path)
    if manifest.get("worker_id") != authorization["worker_id"]:
        raise RecoveryError("authorization worker_id mismatch")
    if (
        manifest.get("role") != "primary"
        or manifest.get("execution_mode") != "experiment_configs"
    ):
        raise RecoveryError(
            "recovery is restricted to formal primary experiment workers"
        )
    tasks = manifest.get("task_configs")
    if not isinstance(tasks, list) or not tasks or tasks[0].get("task_id") != TASK_ID:
        raise RecoveryError("task1 must be isolation_no_memory")
    raw_task = tasks[0]
    preexisting_task_receipt = (
        attempt_root / "task_receipts" / f"{TASK_ID}.json"
    ).exists()
    if (
        attempt_root / "completion_receipt.json"
    ).exists() and not preexisting_task_receipt:
        raise RecoveryError("worker is already complete; recovery is not admissible")
    existing_manifest = attempt_root / "worker_manifest.json"
    if (
        not existing_manifest.is_file()
        or existing_manifest.read_bytes() != manifest_path.read_bytes()
    ):
        raise RecoveryError("attempt worker_manifest is not byte-identical")
    baseline = _read_baseline(baseline_path)
    archive_closure = _verify_archive_closure(
        archive_path.resolve(), baseline, manifest
    )
    receipt_was_present = preexisting_task_receipt
    attempt_baseline = _verify_baseline_subset(
        attempt_root,
        archive_closure.attempt_rows,
        append_only_paths={"state.tsv"} if receipt_was_present else set(),
    )

    state_path = attempt_root / "state.tsv"
    last_status, last_hash = runner._last_state_and_hash(state_path)
    state_rows = _state_rows(state_path)
    _verify_state_chain_rows(state_rows)
    authorized_failed_hash = str(authorization["failed_state_row_sha256"])
    failed_row_present = any(
        row.get("status") == "failed"
        and row.get("row_sha256") == authorized_failed_hash
        for row in state_rows
    )
    if not failed_row_present:
        raise RecoveryError("authorized failed state row is absent")

    child_run = attempt_root / "runs" / TASK_ID
    recovery_root = attempt_root / "recovery_adoptions"
    identity_path = child_run / "task_identity_audit.json"
    canonical_audit_path = recovery_root / f"{TASK_ID}.canonicalization_audit.json"
    attestation_path = recovery_root / f"{TASK_ID}.adoption_attestation.json"
    certificate_path = recovery_root / f"{TASK_ID}.json"
    receipt_path = attempt_root / "task_receipts" / f"{TASK_ID}.json"
    authorization_sha256 = _sha256_file(authorization_path)
    receipt_exists = receipt_path.exists()

    if not receipt_exists:
        if last_status != "failed" or last_hash != authorized_failed_hash:
            raise RecoveryError(
                "pre-adoption state tail is not the authorized failed row"
            )
        allowed_partial = {
            path.relative_to(attempt_root).as_posix()
            for path in (
                identity_path,
                canonical_audit_path,
                attestation_path,
                certificate_path,
            )
        }
        extras = _current_files(attempt_root) - set(attempt_baseline)
        if not extras.issubset(allowed_partial):
            raise RecoveryError(f"unexpected pre-adoption files: {sorted(extras)}")

    # These gates are deliberately common to first publication and every
    # idempotent resume.  An existing receipt never bypasses adoption checks.
    config, cache_namespace = _prepare_task1_config(
        manifest, raw_task, attempt_root, load_config
    )
    planned_hands = int(raw_task.get("planned_hands", -1))
    actual_hands = _count_structural_hands(child_run / "hand_summaries.jsonl")
    if planned_hands != 1350 or actual_hands != 1350:
        raise RecoveryError(
            "task1 planned_hands and structural actual_hands must both be 1350"
        )
    actual, health, identity_correction = _identity_and_health(
        runner, manifest, raw_task, config, child_run
    )
    identity_audit = {
        "schema_version": TASK_IDENTITY_SCHEMA,
        "task_id": TASK_ID,
        "protocol_status": manifest["protocol_status"],
        "actual": actual,
        "status": "verified",
    }
    canonical_audit = {
        "schema_version": "task8b-canonicalization-equivalence-audit-v1",
        "worker_id": manifest["worker_id"],
        "task_id": TASK_ID,
        **identity_correction,
    }
    canonical_audit_sha256 = _sha256_bytes(_json_bytes(canonical_audit))
    phase_f_base = {
        "protocol_amendment_sha256": authorization["protocol_amendment_sha256"],
        "verifier_code_sha": authorization["verifier_code_sha"],
        "pre_recovery_archive_sha256": authorization["pre_recovery_archive_sha256"],
        "pre_recovery_file_manifest_sha256": authorization["baseline_manifest_sha256"],
        "original_terminal_state_sha256": authorized_failed_hash,
        "original_expected_config_sha256": identity_correction[
            "original_expected_sha256"
        ],
        "corrected_config_sha256": identity_correction["corrected_actual_sha256"],
        "canonicalization_equivalence_audit_sha256": canonical_audit_sha256,
    }
    attestation = {
        "schema_version": "task8b-task1-adoption-attestation-v1",
        "worker_id": manifest["worker_id"],
        "task_id": TASK_ID,
        "protocol_amendment_id": PROTOCOL_AMENDMENT_ID,
        "scientific_execution_code_sha": FROZEN_CODE_SHA,
        "identity": actual,
        "health_gates": health,
        "planned_hands": planned_hands,
        "actual_hands": actual_hands,
        "task1_rerun_performed": False,
        "raw_artifact_bytes_modified": False,
        "scientific_outcome_fields_accessed": False,
        "phase_f_evidence": phase_f_base,
    }
    attestation_sha256 = _sha256_bytes(_json_bytes(attestation))
    phase_f_without_certificate = {
        **phase_f_base,
        "task1_adoption_attestation_sha256": attestation_sha256,
    }
    certificate = {
        "schema_version": ADOPTION_SCHEMA,
        "status": "authorized-task1-adoption",
        "worker_id": manifest["worker_id"],
        "task_id": TASK_ID,
        "reason": AUTHORIZED_REASON,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_sha256,
        "controller_sha256": authorization["controller_sha256"],
        "formal_runner_sha256": authorization["formal_runner_sha256"],
        "verifier_identity_source_sha256": authorization[
            "verifier_identity_source_sha256"
        ],
        "protocol_amendment_id": PROTOCOL_AMENDMENT_ID,
        "parent_preunlock_sha256": authorization["parent_preunlock_sha256"],
        "active_pid_absence_evidence_sha256": authorization[
            "active_pid_absence_evidence_sha256"
        ],
        "identity": actual,
        "identity_correction": identity_correction,
        "health_gates": health,
        "planned_hands": planned_hands,
        "actual_hands": actual_hands,
        "effect_fields_read": False,
        "task1_rerun_performed": False,
        "raw_artifact_bytes_modified": False,
        "scientific_outcome_fields_accessed": False,
        "phase_f_evidence": phase_f_without_certificate,
    }
    certificate_sha256 = _sha256_bytes(_json_bytes(certificate))
    phase_f_evidence = {
        **phase_f_without_certificate,
        "recovery_certificate_sha256": certificate_sha256,
    }

    if receipt_exists:
        _require_byte_identical(identity_path, identity_audit)
        _require_byte_identical(canonical_audit_path, canonical_audit)
        _require_byte_identical(attestation_path, attestation)
        _require_byte_identical(certificate_path, certificate)
    else:
        _same_or_atomic_new(runner, identity_path, identity_audit)
        _same_or_atomic_new(runner, canonical_audit_path, canonical_audit)
        _same_or_atomic_new(runner, attestation_path, attestation)
        _same_or_atomic_new(runner, certificate_path, certificate)

    task_row = {
        "task_id": TASK_ID,
        "memory_mode": raw_task.get("memory_mode"),
        "run_dir": f"runs/{TASK_ID}",
        "cache_namespace": cache_namespace,
        "identity_audit": actual,
        "status": "complete",
    }
    receipt = {
        "schema_version": TASK_RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "config_sha256": str(raw_task["config_sha256"]),
        "run_dir": task_row["run_dir"],
        "task_row": task_row,
        "files": runner._directory_file_manifest(child_run),
        "recovery_adoption": {
            "schema_version": ADOPTION_SCHEMA,
            "authorization_sha256": authorization_sha256,
            "authorization_id": authorization["authorization_id"],
            "recovery_certificate_sha256": certificate_sha256,
        },
        "phase_f_evidence": phase_f_evidence,
    }
    if receipt_exists:
        _require_byte_identical(receipt_path, receipt)
        runner._verify_task_receipt(
            marker_path=receipt_path,
            run_dir=attempt_root,
            task_id=TASK_ID,
            config_sha256=str(raw_task["config_sha256"]),
        )
    else:
        # Receipt is intentionally the final recovery-controller publication.
        _same_or_atomic_new(runner, receipt_path, receipt)
    return runner.run_worker_manifest(
        manifest_path, receipt_root=receipt_root, resume_existing=True
    )


def _git_clean_head(checkout: Path) -> str:
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError("unable to read verifier commit") from exc
    _verify_clean_commit(checkout, head)
    return head


def build_authorization_draft(
    *,
    manifest_path: Path,
    attempt_root: Path,
    baseline_path: Path,
    archive_path: Path,
    protocol_amendment_path: Path,
    protocol_amendment_id: str,
    parent_preunlock_path: Path,
    pid_absence_path: Path,
    frozen_checkout: Path,
    verifier_checkout: Path,
    controller_path: Path,
) -> dict[str, Any]:
    """Build deterministic, inactive authorization data without reading effects."""

    paths = (
        manifest_path,
        baseline_path,
        archive_path,
        protocol_amendment_path,
        parent_preunlock_path,
        pid_absence_path,
        controller_path,
    )
    if any(
        not path.resolve().is_file() or path.resolve().is_symlink() for path in paths
    ):
        raise RecoveryError("authorization draft input is missing or symlinked")
    if protocol_amendment_id != PROTOCOL_AMENDMENT_ID:
        raise RecoveryError("protocol amendment id is not the frozen Phase F id")
    manifest = _read_json(manifest_path.resolve())
    baseline = _read_baseline(baseline_path.resolve())
    archive_closure = _verify_archive_closure(
        archive_path.resolve(), baseline, manifest
    )
    attempt_baseline = _verify_baseline_subset(
        attempt_root.resolve(), archive_closure.attempt_rows
    )
    if _current_files(attempt_root.resolve()) != set(attempt_baseline):
        raise RecoveryError(
            "draft requires an exactly closed pre-recovery attempt tree"
        )
    if (
        attempt_root.resolve() / "worker_manifest.json"
    ).read_bytes() != manifest_path.resolve().read_bytes():
        raise RecoveryError(
            "draft worker manifest is not byte-identical to the attempt"
        )
    state_rows = _state_rows(attempt_root.resolve() / "state.tsv")
    _verify_state_chain_rows(state_rows)
    if not state_rows or state_rows[-1].get("status") != "failed":
        raise RecoveryError("draft requires a failed state tail")
    pid_evidence = _read_json(pid_absence_path.resolve())
    if pid_evidence.get("active_process_absent") is not True:
        raise RecoveryError("draft requires affirmative PID absence evidence")
    runner_path = _verify_checkout(
        frozen_checkout.resolve(),
        _sha256_file(
            frozen_checkout.resolve()
            / "src"
            / "agentmemeval"
            / "experiments"
            / "formal_runner.py"
        ),
    )
    verifier_code_sha = _git_clean_head(verifier_checkout.resolve())
    verifier_source = (
        verifier_checkout.resolve()
        / "src"
        / "agentmemeval"
        / "experiments"
        / "formal_protocol.py"
    )
    if not verifier_source.is_file():
        raise RecoveryError("verifier identity source is missing")
    draft: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized": False,
        "worker_id": manifest.get("worker_id"),
        "task_id": TASK_ID,
        "reason": AUTHORIZED_REASON,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "formal_runner_sha256": _sha256_file(runner_path),
        "controller_sha256": _sha256_file(controller_path.resolve()),
        "worker_manifest_sha256": _sha256_file(manifest_path.resolve()),
        "baseline_manifest_sha256": _sha256_file(baseline_path.resolve()),
        "pre_recovery_archive_sha256": _sha256_file(archive_path.resolve()),
        "protocol_amendment_id": protocol_amendment_id,
        "protocol_amendment_sha256": _sha256_file(protocol_amendment_path.resolve()),
        "parent_preunlock_sha256": _sha256_file(parent_preunlock_path.resolve()),
        "active_pid_absence_evidence_sha256": _sha256_file(pid_absence_path.resolve()),
        "verifier_code_sha": verifier_code_sha,
        "verifier_identity_source_sha256": _sha256_file(verifier_source),
        "failed_state_row_sha256": state_rows[-1].get("row_sha256"),
        "attempt_root": str(attempt_root.resolve()),
        "active_process_absent_confirmed": True,
        "same_attempt_recovery_authorized": False,
        "task1_adoption_only": False,
    }
    identity_material = dict(draft)
    identity_material.pop("authorized")
    identity_material.pop("same_attempt_recovery_authorized")
    identity_material.pop("task1_adoption_only")
    draft["authorization_id"] = (
        "task8b-recovery-" + _sha256_bytes(_json_bytes(identity_material))[:24]
    )
    return draft


def activate_authorization_draft(
    *, draft_path: Path, rebuilt_draft: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Activate only a byte-identical, deterministically rebuilt draft."""

    if draft_path.resolve().read_bytes() != _json_bytes(rebuilt_draft):
        raise RecoveryError(
            "authorization draft is not byte-identical to rebuilt inputs"
        )
    activated = copy.deepcopy(rebuilt_draft)
    activated["authorized"] = True
    activated["same_attempt_recovery_authorized"] = True
    activated["task1_adoption_only"] = True
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(_json_bytes(activated))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RecoveryError("refuse to overwrite activated authorization") from exc
    return activated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-checkout", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--build-authorization", type=Path)
    parser.add_argument("--activate-authorization", type=Path)
    parser.add_argument("--activated-authorization-output", type=Path)
    parser.add_argument("--protocol-amendment", required=True, type=Path)
    parser.add_argument("--protocol-amendment-id", required=True)
    parser.add_argument("--parent-preunlock", required=True, type=Path)
    parser.add_argument("--pre-recovery-archive", required=True, type=Path)
    parser.add_argument("--pid-absence-evidence", required=True, type=Path)
    parser.add_argument("--verifier-checkout", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = sum(
        bool(item)
        for item in (
            args.authorization,
            args.build_authorization,
            args.activate_authorization,
        )
    )
    if modes != 1:
        raise RecoveryError("choose exactly one authorization mode")
    if args.build_authorization or args.activate_authorization:
        rebuilt_draft = build_authorization_draft(
            manifest_path=args.manifest,
            attempt_root=args.attempt_root,
            baseline_path=args.baseline_manifest,
            archive_path=args.pre_recovery_archive,
            protocol_amendment_path=args.protocol_amendment,
            protocol_amendment_id=args.protocol_amendment_id,
            parent_preunlock_path=args.parent_preunlock,
            pid_absence_path=args.pid_absence_evidence,
            frozen_checkout=args.frozen_checkout,
            verifier_checkout=args.verifier_checkout,
            controller_path=Path(__file__),
        )
        if args.build_authorization:
            args.build_authorization.parent.mkdir(parents=True, exist_ok=True)
            if args.build_authorization.exists():
                raise RecoveryError("refuse to overwrite authorization draft")
            with args.build_authorization.open("xb") as handle:
                handle.write(_json_bytes(rebuilt_draft))
                handle.flush()
                os.fsync(handle.fileno())
            result = rebuilt_draft
        else:
            if args.activated_authorization_output is None:
                raise RecoveryError(
                    "--activated-authorization-output is required for activation"
                )
            result = activate_authorization_draft(
                draft_path=args.activate_authorization,
                rebuilt_draft=rebuilt_draft,
                output_path=args.activated_authorization_output,
            )
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return 0
    if args.receipt_root is None:
        raise RecoveryError("--receipt-root is required for activated recovery")
    authorization = _read_json(args.authorization.resolve())
    frozen_checkout = args.frozen_checkout.resolve()
    runner, load_config = _load_frozen_runner(
        frozen_checkout,
        str(authorization.get("formal_runner_sha256", "")),
    )
    original_canonicalizer = runner._semantic_config
    original_cwd = Path.cwd()
    runner._semantic_config = canonicalize_resolved_config_identity
    try:
        # The frozen runner's admission layer resolves runtime Git identity from
        # Path.cwd().  Execute from the frozen checkout so a verifier checkout or
        # orchestration shell cannot make the scientific commit appear unknown.
        os.chdir(frozen_checkout)
        result = execute_recovery(
            runner=runner,
            load_config=load_config,
            manifest_path=args.manifest,
            receipt_root=args.receipt_root.resolve(),
            attempt_root=args.attempt_root,
            baseline_path=args.baseline_manifest,
            authorization_path=args.authorization,
            controller_path=Path(__file__),
            protocol_amendment_path=args.protocol_amendment,
            parent_preunlock_path=args.parent_preunlock,
            archive_path=args.pre_recovery_archive,
            pid_absence_path=args.pid_absence_evidence,
            verifier_checkout=args.verifier_checkout,
            protocol_amendment_id=args.protocol_amendment_id,
        )
    finally:
        runner._semantic_config = original_canonicalizer
        os.chdir(original_cwd)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
