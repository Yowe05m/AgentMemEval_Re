"""Fail-closed checkpoint transport and completed-worker archival for TASK8B."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.formal_runner import (
    STATE_SCHEMA_VERSION,
    verify_checkpoint_receipt,
)
from agentmemeval.storage.snapshot_archive import build_snapshot_archive


def transfer_checkpoint_bundle(
    *,
    source_receipt: str | Path,
    source_checkpoint_root: str | Path,
    destination_checkpoint_root: str | Path,
    destination_receipt: str | Path,
    expected_identity: dict[str, Any],
    expected_producer_worker_id: str,
    expected_seed_bundle: int,
    expected_checkpoint_hand: int,
) -> dict[str, Any]:
    """Copy verified checkpoint files first and publish the receipt last."""

    source_root = _real_directory(Path(source_checkpoint_root), "source checkpoint root")
    receipt_source = _real_file(Path(source_receipt), "source receipt")
    final_root = Path(destination_checkpoint_root).absolute()
    receipt_target = Path(destination_receipt).absolute()
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(final_root)
    if receipt_target.exists() or receipt_target.is_symlink():
        raise FileExistsError(receipt_target)
    staging = final_root.with_name(f".{final_root.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    if final_root.resolve() in receipt_target.resolve().parents:
        raise ConfigError("destination receipt 必须位于 checkpoint root 外")
    frozen_receipt_bytes = receipt_source.read_bytes()
    frozen_receipt_sha256 = hashlib.sha256(frozen_receipt_bytes).hexdigest()
    verified = verify_checkpoint_receipt(
        receipt_source,
        source_root,
        expected_identity=expected_identity,
        expected_producer_worker_id=expected_producer_worker_id,
        expected_seed_bundle=expected_seed_bundle,
        expected_checkpoint_hand=expected_checkpoint_hand,
    )
    staging.mkdir(parents=True, exist_ok=False)
    for row in verified["checkpoint_files"]:
        relative = _safe_relative(str(row["relative_path"]))
        source = _inside(source_root, relative)
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        if target.stat().st_size != int(row["size"]) or _sha256(target) != row["sha256"]:
            raise ConfigError(f"staged checkpoint hash 不匹配：{relative.as_posix()}")
    verify_checkpoint_receipt(
        receipt_source,
        staging,
        expected_identity=expected_identity,
        expected_producer_worker_id=expected_producer_worker_id,
        expected_seed_bundle=expected_seed_bundle,
        expected_checkpoint_hand=expected_checkpoint_hand,
    )
    if receipt_source.read_bytes() != frozen_receipt_bytes:
        raise ConfigError("source receipt 在 transport 期间发生变化")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final_root)
    receipt_target.parent.mkdir(parents=True, exist_ok=True)
    receipt_staging = receipt_target.with_name(f".{receipt_target.name}.partial")
    if receipt_staging.exists() or receipt_staging.is_symlink():
        raise FileExistsError(receipt_staging)
    with receipt_staging.open("xb") as output_handle:
        output_handle.write(frozen_receipt_bytes)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if _sha256(receipt_staging) != frozen_receipt_sha256:
        raise ConfigError("staged receipt hash 不匹配")
    verify_checkpoint_receipt(
        receipt_staging,
        final_root,
        expected_identity=expected_identity,
        expected_producer_worker_id=expected_producer_worker_id,
        expected_seed_bundle=expected_seed_bundle,
        expected_checkpoint_hand=expected_checkpoint_hand,
    )
    os.link(receipt_staging, receipt_target)
    receipt_staging.unlink()
    final = verify_checkpoint_receipt(
        receipt_target,
        final_root,
        expected_identity=expected_identity,
        expected_producer_worker_id=expected_producer_worker_id,
        expected_seed_bundle=expected_seed_bundle,
        expected_checkpoint_hand=expected_checkpoint_hand,
    )
    return {
        "schema_version": "task8b-checkpoint-transport-v1",
        "status": "verified",
        "producer_worker_id": final["producer_worker_id"],
        "seed_bundle": int(final["seed_bundle"]),
        "checkpoint_hand": int(final["checkpoint_hand"]),
        "file_count": len(final["checkpoint_files"]),
        "receipt_sha256": _sha256(receipt_target),
    }


def validate_worker_for_archive(run_dir: str | Path) -> dict[str, Any]:
    """Validate completion, manifest hashes, final state, identity, and child health."""

    root = _real_directory(Path(run_dir), "worker run")
    completion_path = _real_file(root / "completion_receipt.json", "completion receipt")
    files_path = _real_file(root / "files.tsv", "worker files manifest")
    state_path = _real_file(root / "state.tsv", "worker state")
    worker_manifest_path = _real_file(root / "worker_manifest.json", "worker manifest")
    completion = _read_json(completion_path)
    worker_manifest = _read_json(worker_manifest_path)
    if (
        completion.get("schema_version") != "task8-worker-completion-v1"
        or completion.get("status") != "complete"
        or completion.get("worker_id") != worker_manifest.get("worker_id")
        or completion.get("files_tsv_sha256") != _sha256(files_path)
    ):
        raise ConfigError("worker completion/identity/hash 门禁失败")
    if _last_state(state_path) != "complete":
        raise ConfigError("worker state 最终状态不是 complete")
    _verify_state_hash_chain(state_path)
    rows = _read_files_tsv(files_path)
    for relative, size, digest in rows:
        path = _inside(root, _safe_relative(relative))
        if not path.is_file() or path.is_symlink():
            raise ConfigError(f"worker artifact 缺失或为 symlink：{relative}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ConfigError(f"worker artifact hash 失败：{relative}")
    listed = {relative for relative, _, _ in rows}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_actual = listed | {"state.tsv", "files.tsv", "completion_receipt.json"}
    if actual != expected_actual:
        raise ConfigError("worker archive 存在 files.tsv 未登记 extra 或缺失文件")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ConfigError("worker archive 禁止 symlink")
    task_results = _read_json(root / "task_results.json")
    task_configs = {
        str(row.get("task_id")): row
        for row in worker_manifest.get("task_configs", [])
        if isinstance(row, dict)
    }
    result_rows = task_results.get("tasks")
    if (
        task_results.get("schema_version") != "task8-worker-task-results-v1"
        or task_results.get("worker_id") != worker_manifest.get("worker_id")
        or not isinstance(result_rows, list)
    ):
        raise ConfigError("worker task_results schema/worker identity 非法")
    authoritative = {str(row.get("task_id")): row for row in result_rows if isinstance(row, dict)}
    if (
        len(authoritative) != len(result_rows)
        or set(authoritative) != set(task_configs)
        or any(row.get("status") != "complete" for row in authoritative.values())
    ):
        raise ConfigError("worker task_results 未精确绑定全部 completed tasks")
    audits: list[Path] = []
    for task_id, result_row in sorted(authoritative.items()):
        config = task_configs[task_id]
        receipt_path = _inside(root, Path("task_receipts") / f"{task_id}.json")
        receipt = _read_json(_real_file(receipt_path, f"task {task_id} receipt"))
        run_relative = _safe_relative(str(result_row.get("run_dir", "")))
        child_run = _inside(root, run_relative)
        if (
            receipt.get("schema_version") != "task8-worker-task-receipt-v1"
            or receipt.get("task_id") != task_id
            or receipt.get("config_sha256") != config.get("config_sha256")
            or receipt.get("run_dir") != run_relative.as_posix()
            or receipt.get("task_row") != result_row
        ):
            raise ConfigError(f"task {task_id} receipt/results identity mismatch")
        receipt_files = receipt.get("files")
        if not isinstance(receipt_files, list) or receipt_files != _directory_file_manifest(
            child_run
        ):
            raise ConfigError(f"task {task_id} receipt files 未绑定 authoritative child")
        path = _real_file(child_run / "protocol_audit.json", f"task {task_id} protocol audit")
        audits.append(path)
        audit = _read_json(path)
        identity_audit = _read_json(path.parent / "task_identity_audit.json")
        expected_identity = config.get("expected_identity")
        if (
            identity_audit.get("status") != "verified"
            or not isinstance(expected_identity, dict)
            or any(
                dict(identity_audit.get("actual", {})).get(field) != value
                for field, value in expected_identity.items()
            )
        ):
            raise ConfigError(f"child task identity 未 verified：{path.parent.name}")
        validity = audit.get("run_validity")
        execution = audit.get("execution_health", audit)
        if not isinstance(validity, dict) or (
            validity.get("execution_valid") is not True
            or validity.get("behavior_valid") is not True
        ):
            raise ConfigError(f"child run health/run_validity 失败：{path.parent.name}")
        if not isinstance(execution, dict) or execution.get("valid") is not True:
            raise ConfigError(f"child execution_health.valid 失败：{path.parent.name}")
        for field in (
            "fallback_count",
            "memory_revision_fallback_count",
            "reward_conservation_violation_count",
            "stack_conservation_violation_count",
        ):
            if int(execution.get(field, -1)) != 0:
                raise ConfigError(f"child run health {field} 非零：{path.parent.name}")
    return {
        "schema_version": "task8b-worker-archive-admission-v1",
        "status": "verified",
        "worker_id": str(worker_manifest["worker_id"]),
        "listed_file_count": len(rows),
        "child_health_audit_count": len(audits),
        "completion_receipt_sha256": _sha256(completion_path),
        "state_sha256": _sha256(state_path),
    }


def archive_completed_worker(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Seal a validated worker using an append-only snapshot with receipt published last."""

    root = Path(run_dir).resolve()
    admission = validate_worker_for_archive(root)
    destination = Path(output_dir).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{admission['worker_id']}_{root.name}"
    final_receipt = destination / f"{stem}.receipt.json"
    if final_receipt.exists() or final_receipt.is_symlink():
        raise FileExistsError(final_receipt)
    snapshot = build_snapshot_archive(
        root,
        destination / f"{stem}.tar.gz",
        destination / f"{stem}.files.csv",
        destination / f"{stem}.sha256",
        destination / f".{stem}.receipt.json.partial",
    )
    partial_receipt = Path(snapshot["receipt"])
    with partial_receipt.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.link(partial_receipt, final_receipt)
    partial_receipt.unlink()
    return {
        "admission": admission,
        "snapshot": {**snapshot, "receipt": str(final_receipt)},
    }


def _directory_file_manifest(root: Path) -> list[dict[str, Any]]:
    directory = _real_directory(root, "authoritative child run")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ConfigError("authoritative child run 禁止 symlink")
    return [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def _read_files_tsv(path: Path) -> list[tuple[str, int, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    seen: set[str] = set()
    result = []
    for row in rows:
        relative = str(row.get("relative_path", ""))
        if relative in seen:
            raise ConfigError(f"files.tsv duplicate path：{relative}")
        seen.add(relative)
        result.append((relative, int(row["size"]), str(row["sha256"])))
    if not result:
        raise ConfigError("files.tsv 为空")
    return result


def _last_state(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ConfigError("worker state 为空")
    return str(rows[-1].get("status", ""))


def _verify_state_hash_chain(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    previous = "GENESIS"
    for row in rows:
        if row.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ConfigError("worker state schema_version 不匹配")
        if row.get("previous_sha256") != previous:
            raise ConfigError("worker state previous hash chain 断裂")
        expected = sha256_json(
            {
                "schema_version": row["schema_version"],
                "created_at_utc": row["created_at_utc"],
                "status": row["status"],
                "detail": row["detail"],
                "previous_sha256": row["previous_sha256"],
            }
        )
        if row.get("row_sha256") != expected:
            raise ConfigError("worker state row hash 不匹配")
        previous = expected


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ConfigError(f"非法相对路径：{value}")
    return path


def _inside(root: Path, relative: Path) -> Path:
    path = root.joinpath(*relative.parts).resolve()
    if root != path and root not in path.parents:
        raise ConfigError(f"路径越界：{relative}")
    return path


def _real_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise ConfigError(f"{label} 必须是真实目录：{absolute}")
    return absolute.resolve()


def _real_file(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise ConfigError(f"{label} 必须是真实文件：{absolute}")
    return absolute.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"JSON 无法读取：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON 顶层必须为对象：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
