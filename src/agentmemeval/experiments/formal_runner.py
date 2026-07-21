"""Fail-closed TASK8 multi-worker manifests, receipts, state, and local mock runner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentmemeval.core.domain import MemorySnapshot
from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import (
    build_clone_audit,
    build_heldout_schedule_manifest,
    clone_memory_branches,
    sha256_json,
)

WORKER_SCHEMA_VERSION = "task8-worker-manifest-v1"
RECEIPT_SCHEMA_VERSION = "task8-checkpoint-receipt-v1"
STATE_SCHEMA_VERSION = "task8-worker-state-v1"
REQUIRED_IDENTITY_FIELDS = (
    "code_sha",
    "resolved_config_sha256",
    "prompt_sha256",
    "model_fingerprint",
    "embedding_fingerprint",
    "schedule_sha256",
)
REQUIRED_RECEIPT_FIELDS = (
    "schema_version",
    "producer_worker_id",
    "seed_bundle",
    "checkpoint_hand",
    "checkpoint_files",
    *REQUIRED_IDENTITY_FIELDS,
    "created_at_utc",
    "status",
)
STATE_TRANSITIONS = {
    None: {"planned", "validating"},
    "planned": {"validating", "failed"},
    "validating": {"waiting_dependency", "running", "failed"},
    "waiting_dependency": {"validating", "running", "failed", "partial"},
    "running": {"finalizing", "failed", "partial"},
    "finalizing": {"complete", "failed"},
    "failed": {"validating"},
    "partial": {"validating"},
    "complete": set(),
}


def generate_worker_manifests(
    *,
    matrix_path: str | Path,
    seeds: list[int],
    common_identity: dict[str, Any],
    output_dir: str | Path,
    output_root: str = "outputs/formal/task8",
    cache_root: str = "task8",
    protocol_status: str = "candidate/not-frozen/not-authorized-to-run",
    execution_mode: str = "formal_candidate",
) -> dict[str, Any]:
    """Generate stable Pxx/Sxx seed pods from one matrix and one ordered seed list."""

    matrix = Path(matrix_path)
    if not matrix.is_file():
        raise ConfigError(f"experiment matrix 不存在：{matrix}")
    _validate_task8_matrix(matrix)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ConfigError("seed list 必须非空且无重复")
    if not protocol_status.startswith("mock/") and len(seeds) != 12:
        raise ConfigError("TASK8 Formal/candidate manifest generator 要求恰好 12 seeds")
    _validate_common_identity(
        common_identity, allow_pending=protocol_status.startswith("candidate/")
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(f"manifest 输出目录非空，拒绝覆盖：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    matrix_sha256 = _sha256_file(matrix)
    manifests: list[dict[str, Any]] = []
    width = max(2, len(str(len(seeds))))
    for index, seed in enumerate(seeds, start=1):
        pod_id = f"pod{index:0{width}d}"
        primary_id = f"P{index:0{width}d}"
        secondary_id = f"S{index:0{width}d}"
        primary = _worker_manifest(
            worker_id=primary_id,
            role="primary",
            seed=seed,
            pod_id=pod_id,
            common_identity=common_identity,
            matrix_sha256=matrix_sha256,
            output_root=output_root,
            cache_root=cache_root,
            protocol_status=protocol_status,
            execution_mode=execution_mode,
        )
        secondary = _worker_manifest(
            worker_id=secondary_id,
            role="secondary",
            seed=seed,
            pod_id=pod_id,
            common_identity=common_identity,
            matrix_sha256=matrix_sha256,
            output_root=output_root,
            cache_root=cache_root,
            protocol_status=protocol_status,
            execution_mode=execution_mode,
            depends_on=primary_id,
        )
        manifests.extend((primary, secondary))
    validate_worker_manifest_set(manifests, expected_seed_count=len(seeds))
    for manifest in manifests:
        _write_json_new(destination / f"{manifest['worker_id']}.json", manifest)
    index_body = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_status": protocol_status,
        "matrix_sha256": matrix_sha256,
        "seed_count": len(seeds),
        "worker_count": len(manifests),
        "workers": [
            {
                "worker_id": item["worker_id"],
                "role": item["role"],
                "seed_bundle": item["seed_bundle"],
                "manifest_sha256": sha256_json(item),
            }
            for item in manifests
        ],
    }
    _write_json_new(destination / "manifest_index.json", index_body)
    return index_body


def _worker_manifest(
    *,
    worker_id: str,
    role: str,
    seed: int,
    pod_id: str,
    common_identity: dict[str, Any],
    matrix_sha256: str,
    output_root: str,
    cache_root: str,
    protocol_status: str,
    execution_mode: str,
    depends_on: str | None = None,
) -> dict[str, Any]:
    checkpoint_set = [1, 3, 5] if execution_mode == "mock_seed_pod" else [30, 75, 150, 300]
    table_set = ["H01", "H02", "H03"]
    output_path = f"{output_root}/{worker_id}/{seed}"
    cache_namespace = f"{cache_root}/{worker_id}/{seed}"
    dependency_output_path = (
        f"{output_root}/{depends_on}/{seed}" if role == "secondary" and depends_on else None
    )
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_status": protocol_status,
        "execution_mode": execution_mode,
        "worker_id": worker_id,
        "role": role,
        "pod_id": pod_id,
        "seed_bundle": int(seed),
        "experiment_families": (
            ["R1-E1-I", "R1-E2", "R1-E3"]
            if role == "primary"
            else ["R1-E1-M", "R1-E4", "R1-E5"]
        ),
        "checkpoint_set": checkpoint_set,
        "heldout_table_set": table_set,
        "memory_modes": ["Frozen"] if role == "primary" else ["Frozen", "Online", "Without"],
        "matrix_sha256": matrix_sha256,
        "common_identity": dict(common_identity),
        "instance_identity": {
            "worker_id": worker_id,
            "cache_namespace": cache_namespace,
            "output_path": output_path,
        },
        "depends_on": depends_on,
        "dependency_output_path": dependency_output_path,
        "receipt_relative_path": f"receipts/{worker_id if role == 'primary' else depends_on}.json",
        "identity_classification": {
            "cross_instance_same": list(REQUIRED_IDENTITY_FIELDS),
            "instance_specific": ["worker_id", "host_id", "started_at_utc"],
            "must_be_unique": ["worker_id", "cache_namespace", "output_path"],
        },
    }


def validate_worker_manifest_set(
    manifests: list[dict[str, Any]], *, expected_seed_count: int = 12
) -> None:
    if len(manifests) != expected_seed_count * 2:
        raise ConfigError("worker manifest 数量必须等于 2 × seed count")
    for manifest in manifests:
        validate_worker_manifest(manifest, allow_candidate=True)
    workers = [str(item["worker_id"]) for item in manifests]
    outputs = [str(item["instance_identity"]["output_path"]) for item in manifests]
    caches = [str(item["instance_identity"]["cache_namespace"]) for item in manifests]
    unique_fields = (
        ("worker_id", workers),
        ("output_path", outputs),
        ("cache_namespace", caches),
    )
    for label, values in unique_fields:
        if len(values) != len(set(values)):
            raise ConfigError(f"worker manifest 存在重复 {label}")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for item in manifests:
        by_seed.setdefault(int(item["seed_bundle"]), []).append(item)
    if len(by_seed) != expected_seed_count:
        raise ConfigError("seed 存在漏配或多配")
    for seed, pod in by_seed.items():
        roles = {str(item["role"]): item for item in pod}
        if set(roles) != {"primary", "secondary"}:
            raise ConfigError(f"seed {seed} 必须恰有 primary 与 secondary")
        if roles["secondary"].get("depends_on") != roles["primary"]["worker_id"]:
            raise ConfigError(f"seed {seed} secondary 依赖不完整")
        if (
            roles["secondary"].get("dependency_output_path")
            != roles["primary"]["instance_identity"]["output_path"]
        ):
            raise ConfigError(f"seed {seed} secondary producer output path 不闭合")
    identities = [sha256_json(item["common_identity"]) for item in manifests]
    if len(set(identities)) != 1:
        raise ConfigError("跨 worker common identity 不一致")
    _reject_dependency_cycles(manifests)


def validate_worker_manifest(
    manifest: dict[str, Any], *, allow_candidate: bool = False
) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_status",
        "execution_mode",
        "worker_id",
        "role",
        "pod_id",
        "seed_bundle",
        "checkpoint_set",
        "heldout_table_set",
        "common_identity",
        "instance_identity",
        "receipt_relative_path",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ConfigError(f"worker manifest 缺字段：{', '.join(missing)}")
    if manifest["schema_version"] != WORKER_SCHEMA_VERSION:
        raise ConfigError("worker manifest schema_version 不受支持")
    if manifest["role"] not in {"primary", "secondary"}:
        raise ConfigError("worker role 必须是 primary 或 secondary")
    expected_checkpoints = (
        [1, 3, 5]
        if str(manifest["protocol_status"]).startswith("mock/")
        else [30, 75, 150, 300]
    )
    if manifest["checkpoint_set"] != expected_checkpoints:
        raise ConfigError(f"TASK8 worker checkpoint_set 必须为 {expected_checkpoints}")
    if manifest["heldout_table_set"] != ["H01", "H02", "H03"]:
        raise ConfigError("TASK8 worker heldout_table_set 必须为 H01/H02/H03")
    status = str(manifest["protocol_status"])
    if status.startswith("candidate/") and not allow_candidate:
        raise ConfigError("candidate/not-frozen manifest 未获运行授权")
    _validate_common_identity(
        dict(manifest["common_identity"]), allow_pending=status.startswith("candidate/")
    )
    instance = manifest["instance_identity"]
    if not isinstance(instance, dict):
        raise ConfigError("instance_identity 必须是映射")
    for field in ("worker_id", "cache_namespace", "output_path"):
        if not str(instance.get(field, "")).strip():
            raise ConfigError(f"instance_identity.{field} 不能为空")
    if str(instance["worker_id"]) != str(manifest["worker_id"]):
        raise ConfigError("instance_identity.worker_id 不一致")
    _safe_relative_path(str(manifest["receipt_relative_path"]))
    if manifest["role"] == "secondary":
        if not manifest.get("depends_on"):
            raise ConfigError("secondary manifest 缺少 depends_on")
        if not str(manifest.get("dependency_output_path", "")).strip():
            raise ConfigError("secondary manifest 缺少 dependency_output_path")
    if manifest["execution_mode"] == "experiment_configs":
        tasks = manifest.get("task_configs")
        if not isinstance(tasks, list) or not tasks:
            raise ConfigError("experiment_configs worker 缺少 task_configs")
    return manifest


def publish_checkpoint_receipt(
    *,
    checkpoint_root: str | Path,
    checkpoint_files: list[str],
    receipt_path: str | Path,
    producer_worker_id: str,
    seed_bundle: int,
    checkpoint_hand: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Hash every closed checkpoint file before exclusively publishing complete receipt."""

    _validate_common_identity(identity, allow_pending=False)
    root = Path(checkpoint_root).resolve()
    if not root.is_dir() or not checkpoint_files:
        raise ConfigError("checkpoint root/files 不完整")
    rows = []
    for relative in checkpoint_files:
        path = _resolve_inside(root, relative)
        if not path.is_file():
            raise ConfigError(f"checkpoint 文件缺失：{relative}")
        first_size = path.stat().st_size
        digest = _sha256_file(path)
        if path.stat().st_size != first_size:
            raise ConfigError(f"checkpoint 文件哈希期间发生变化：{relative}")
        rows.append({"relative_path": relative, "size": first_size, "sha256": digest})
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "producer_worker_id": producer_worker_id,
        "seed_bundle": int(seed_bundle),
        "checkpoint_hand": int(checkpoint_hand),
        "checkpoint_files": rows,
        **{field: identity[field] for field in REQUIRED_IDENTITY_FIELDS},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    _write_json_atomic_new(Path(receipt_path), receipt)
    return receipt


def verify_checkpoint_receipt(
    receipt_path: str | Path,
    checkpoint_root: str | Path,
    *,
    expected_identity: dict[str, Any] | None = None,
    expected_producer_worker_id: str | None = None,
    expected_seed_bundle: int | None = None,
    expected_checkpoint_hand: int | None = None,
) -> dict[str, Any]:
    receipt = _read_json_object(Path(receipt_path))
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in receipt]
    if missing:
        raise ConfigError(f"checkpoint receipt 缺字段：{', '.join(missing)}")
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION or receipt["status"] != "complete":
        raise ConfigError("checkpoint receipt 非受支持 complete receipt")
    if expected_producer_worker_id and receipt["producer_worker_id"] != expected_producer_worker_id:
        raise ConfigError("checkpoint receipt producer_worker_id 不匹配")
    if expected_seed_bundle is not None and int(receipt["seed_bundle"]) != int(
        expected_seed_bundle
    ):
        raise ConfigError("checkpoint receipt seed_bundle 不匹配")
    if expected_checkpoint_hand is not None and int(receipt["checkpoint_hand"]) != int(
        expected_checkpoint_hand
    ):
        raise ConfigError("checkpoint receipt checkpoint_hand 不匹配")
    if expected_identity:
        for field in REQUIRED_IDENTITY_FIELDS:
            if receipt[field] != expected_identity.get(field):
                raise ConfigError(f"checkpoint receipt identity 不匹配：{field}")
    root = Path(checkpoint_root).resolve()
    rows = receipt["checkpoint_files"]
    if not isinstance(rows, list) or not rows:
        raise ConfigError("checkpoint receipt checkpoint_files 为空")
    relative_paths = [
        str(row.get("relative_path", "")) for row in rows if isinstance(row, dict)
    ]
    if len(relative_paths) != len(rows) or len(relative_paths) != len(set(relative_paths)):
        raise ConfigError("checkpoint receipt checkpoint_files 路径重复或非法")
    for row in rows:
        if not isinstance(row, dict) or set(("relative_path", "size", "sha256")) - set(row):
            raise ConfigError("checkpoint receipt file row 不完整")
        path = _resolve_inside(root, str(row["relative_path"]))
        if not path.is_file():
            raise ConfigError(f"checkpoint receipt 文件缺失：{row['relative_path']}")
        if path.stat().st_size != int(row["size"]) or _sha256_file(path) != row["sha256"]:
            raise ConfigError(f"checkpoint receipt 文件哈希不匹配：{row['relative_path']}")
    return receipt


def append_worker_state(path: str | Path, status: str, detail: str = "") -> None:
    if status not in STATE_TRANSITIONS:
        raise ConfigError(f"未知 worker 状态：{status}")
    state_path = Path(path)
    previous, previous_hash = _last_state_and_hash(state_path)
    if status not in STATE_TRANSITIONS[previous]:
        raise ConfigError(f"非法 worker 状态迁移：{previous!r} -> {status!r}")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not state_path.exists()
    with state_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if is_new:
            writer.writerow(
                (
                    "schema_version",
                    "created_at_utc",
                    "status",
                    "detail",
                    "previous_sha256",
                    "row_sha256",
                )
            )
        created_at = datetime.now(timezone.utc).isoformat()
        previous_sha256 = previous_hash or "GENESIS"
        row_sha256 = sha256_json(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "created_at_utc": created_at,
                "status": status,
                "detail": detail,
                "previous_sha256": previous_sha256,
            }
        )
        writer.writerow(
            (
                STATE_SCHEMA_VERSION,
                created_at,
                status,
                detail,
                previous_sha256,
                row_sha256,
            )
        )


def run_worker_manifest(
    manifest_path: str | Path,
    *,
    receipt_root: str | Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Run the bounded local mock seed-pod path; Formal candidates remain non-runnable."""

    manifest = _read_json_object(Path(manifest_path))
    validate_worker_manifest(manifest, allow_candidate=False)
    if manifest["execution_mode"] not in {"mock_seed_pod", "experiment_configs"}:
        raise ConfigError(
            "worker execution_mode 不可执行；Formal candidate 需冻结为 experiment_configs"
        )
    if manifest["execution_mode"] == "experiment_configs":
        _preflight_experiment_tasks(manifest)
    receipt_base = Path(receipt_root).resolve()
    output = Path(str(manifest["instance_identity"]["output_path"]))
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    receipt_path = _resolve_inside(
        receipt_base, str(manifest["receipt_relative_path"]), strict=False
    )
    verified_receipt: dict[str, Any] | None = None
    if manifest["role"] == "secondary":
        producer = str(manifest["depends_on"])
        producer_root = Path(str(manifest.get("dependency_output_path", "")))
        if not producer_root.is_absolute():
            producer_root = (Path.cwd() / producer_root).resolve()
        if not receipt_path.exists():
            run_dir, resumed = _prepare_attempt(output, manifest, resume_existing)
            state_path = run_dir / "state.tsv"
            if not resumed:
                _write_json_new(run_dir / "worker_manifest.json", manifest)
                append_worker_state(state_path, "planned", "manifest admitted")
                append_worker_state(state_path, "validating", "dependency lookup")
            else:
                append_worker_state(state_path, "validating", "resume dependency lookup")
            append_worker_state(
                state_path,
                "waiting_dependency",
                f"complete receipt not present for {producer}",
            )
            return {
                "status": "waiting_dependency",
                "run_dir": str(run_dir),
                "resumed": resumed,
            }
        verified_receipt = verify_checkpoint_receipt(
            receipt_path,
            producer_root,
            expected_identity=dict(manifest["common_identity"]),
            expected_producer_worker_id=producer,
            expected_seed_bundle=int(manifest["seed_bundle"]),
            expected_checkpoint_hand=int(manifest["checkpoint_set"][-1]),
        )
    run_dir, resumed = _prepare_attempt(output, manifest, resume_existing)
    state_path = run_dir / "state.tsv"
    if resumed and (run_dir / "completion_receipt.json").exists():
        last_status = _last_state(state_path)
        if last_status == "finalizing":
            append_worker_state(state_path, "complete", "resume completed final state seal")
        elif last_status != "complete":
            raise ConfigError("completion receipt 与 worker state 不一致")
        return {"status": "complete", "run_dir": str(run_dir), "resumed": True}
    if resumed:
        append_worker_state(state_path, "validating", "resume identity and artifacts verified")
    else:
        _write_json_new(run_dir / "worker_manifest.json", manifest)
        append_worker_state(state_path, "planned", "manifest admitted before directory creation")
        append_worker_state(state_path, "validating", "local mock validation")
    append_worker_state(state_path, "running", str(manifest["execution_mode"]))
    try:
        if manifest["execution_mode"] == "mock_seed_pod":
            _write_mock_common_artifacts(run_dir, manifest)
            if manifest["role"] == "primary":
                checkpoint_files = _write_mock_primary_checkpoint(run_dir, manifest)
                publish_checkpoint_receipt(
                    checkpoint_root=run_dir,
                    checkpoint_files=checkpoint_files,
                    receipt_path=receipt_path,
                    producer_worker_id=str(manifest["worker_id"]),
                    seed_bundle=int(manifest["seed_bundle"]),
                    checkpoint_hand=int(manifest["checkpoint_set"][-1]),
                    identity=dict(manifest["common_identity"]),
                )
            else:
                _write_mock_secondary_branches(run_dir)
        else:
            _run_experiment_tasks(
                run_dir=run_dir,
                manifest=manifest,
                receipt_path=receipt_path,
                receipt_root=receipt_base,
                verified_receipt=verified_receipt,
            )
        append_worker_state(state_path, "finalizing", "sealing worker artifacts")
        _write_files_manifest(run_dir)
        completion = {
            "schema_version": "task8-worker-completion-v1",
            "worker_id": manifest["worker_id"],
            "status": "complete",
            "not_for_paper": str(manifest["protocol_status"]).startswith("mock/"),
            "files_tsv_sha256": _sha256_file(run_dir / "files.tsv"),
        }
        _write_json_new(run_dir / "completion_receipt.json", completion)
        append_worker_state(state_path, "complete", "completion receipt published")
    except Exception as exc:
        if _last_state(state_path) in {"running", "finalizing"}:
            append_worker_state(state_path, "failed", type(exc).__name__)
        raise
    return {"status": "complete", "run_dir": str(run_dir), "resumed": resumed}


def summarize_worker_states(root: str | Path) -> dict[str, Any]:
    rows = []
    for state in sorted(Path(root).rglob("state.tsv")):
        rows.append({"state_path": str(state), "status": _last_state(state)})
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {"worker_count": len(rows), "status_counts": counts, "workers": rows}


def _write_mock_common_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    common = {
        "protocol_status": "mock/not-for-paper/model-substituted",
        "worker_id": manifest["worker_id"],
        "seed_bundle": manifest["seed_bundle"],
    }
    _write_bytes_same_or_new(run_dir / "events.jsonl", (json.dumps(common) + "\n").encode())
    _write_bytes_same_or_new(run_dir / "hand_summaries.jsonl", (json.dumps(common) + "\n").encode())
    for name, body in (
        ("metrics.json", {**common, "mock_hands": 1}),
        ("protocol_audit.json", {**common, "formal_admission": False}),
        ("behavior_audit.json", {**common, "status": "mock_ok"}),
        ("fallback_audit.json", {**common, "fallback_count": 0}),
        ("runtime_identity.json", {**common, **manifest["common_identity"]}),
        ("resolved_config.json", {**common, "checkpoint_set": manifest["checkpoint_set"]}),
    ):
        _write_json_same_or_new(run_dir / name, body)


def _run_experiment_tasks(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    receipt_path: Path,
    receipt_root: Path,
    verified_receipt: dict[str, Any] | None,
) -> None:
    """Execute frozen child configs and release primary receipt at the declared fan-out point."""

    from agentmemeval.config.loader import load_config
    from agentmemeval.experiments.runner import run_resolved_config

    task_rows = []
    receipt_published = receipt_path.exists()
    producer_root = (
        Path(str(manifest.get("dependency_output_path", ""))).resolve()
        if manifest["role"] == "secondary"
        else receipt_root
    )
    allowed_checkpoint_files = {
        str(row["relative_path"])
        for row in (verified_receipt or {}).get("checkpoint_files", [])
        if isinstance(row, dict) and "relative_path" in row
    }
    for index, raw_task in enumerate(manifest["task_configs"], start=1):
        if not isinstance(raw_task, dict):
            raise ConfigError("task_configs 每项必须是映射")
        task_id = str(raw_task.get("task_id", f"task_{index:02d}"))
        _validate_task_id(task_id)
        config_path = Path(str(raw_task.get("config_path", "")))
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        if not config_path.is_file():
            raise ConfigError(f"task {task_id} config 不存在：{config_path}")
        expected_config_sha256 = str(raw_task.get("config_sha256", ""))
        if not expected_config_sha256 or _sha256_file(config_path) != expected_config_sha256:
            raise ConfigError(f"task {task_id} config SHA-256 不匹配")
        config = load_config(config_path)
        experiment = config["experiment"]
        experiment["seed"] = int(manifest["seed_bundle"])
        experiment["output_root"] = str(run_dir / "runs")
        experiment["run_id"] = task_id
        experiment["heldout_table_set"] = list(manifest["heldout_table_set"])
        branch_label = str(raw_task.get("memory_mode") or manifest["role"]).lower()
        task_cache_namespace = (
            f"{manifest['instance_identity']['cache_namespace']}/{task_id}/{branch_label}"
        )
        agent_config = dict(config.get("agent", {}))
        agent_config["embedding_cache_path"] = f"{task_cache_namespace}/{{agent_id}}.json"
        config["agent"] = agent_config
        if manifest["role"] == "primary" and int(experiment.get("train_hands", 0)) > 0:
            experiment.pop("checkpoint_interval", None)
            experiment["checkpoint_set"] = list(manifest["checkpoint_set"])
        if manifest["role"] == "secondary":
            mode = str(raw_task.get("memory_mode", ""))
            bindings = raw_task.get("checkpoint_bindings")
            if mode not in {"Frozen", "Online", "Without"}:
                raise ConfigError(f"secondary task {task_id} memory_mode 非法")
            if not isinstance(bindings, dict) or not bindings:
                raise ConfigError(f"secondary task {task_id} 缺少 checkpoint_bindings")
            experiment["train_hands"] = 0
            experiment.pop("checkpoint_set", None)
            experiment.pop("checkpoint_interval", None)
            experiment["initial_checkpoint_hand"] = int(manifest["checkpoint_set"][-1])
            experiment["memory_mode"] = mode
            experiment["update_memory_test"] = mode == "Online"
            resolved_bindings = {}
            for agent_id, relative in bindings.items():
                relative_text = str(relative)
                if relative_text not in allowed_checkpoint_files:
                    raise ConfigError(
                        f"secondary task {task_id} 引用了 receipt 外 checkpoint：{relative_text}"
                    )
                resolved_bindings[str(agent_id)] = str(
                    _resolve_inside(producer_root, relative_text)
                )
            experiment["initial_memory_snapshots"] = resolved_bindings
        child_root = run_dir / "runs"
        child_run = child_root / task_id
        marker_path = run_dir / "task_receipts" / f"{task_id}.json"
        if marker_path.exists():
            marker = _verify_task_receipt(
                marker_path=marker_path,
                run_dir=run_dir,
                task_id=task_id,
                config_sha256=expected_config_sha256,
            )
            child_run = _resolve_inside(run_dir.resolve(), str(marker["run_dir"]))
            identity_audit = _verify_completed_task_identity(
                manifest=manifest,
                raw_task=raw_task,
                config=config,
                child_run=child_run,
            )
            task_row = dict(marker["task_row"])
            if task_row.get("identity_audit") != identity_audit:
                raise ConfigError(f"task {task_id} resume identity audit mismatch")
        else:
            if child_run.exists() and any(child_run.iterdir()):
                experiment["run_id"] = _next_child_attempt_id(child_root, task_id)
            elif any(child_root.glob(f"{task_id}__attempt_*")):
                experiment["run_id"] = _next_child_attempt_id(child_root, task_id)
            result = run_resolved_config(config)
            child_run = Path(result.artifacts["run_dir"])
            identity_audit = _verify_completed_task_identity(
                manifest=manifest,
                raw_task=raw_task,
                config=config,
                child_run=child_run,
            )
            task_row = {
                "task_id": task_id,
                "memory_mode": raw_task.get("memory_mode"),
                "run_dir": child_run.relative_to(run_dir).as_posix(),
                "cache_namespace": task_cache_namespace,
                "identity_audit": identity_audit,
                "status": "complete",
            }
            _write_json_atomic_new(
                marker_path,
                {
                    "schema_version": "task8-worker-task-receipt-v1",
                    "task_id": task_id,
                    "config_sha256": expected_config_sha256,
                    "run_dir": task_row["run_dir"],
                    "task_row": task_row,
                    "files": _directory_file_manifest(child_run),
                },
            )
        task_rows.append(task_row)
        if (
            manifest["role"] == "primary"
            and bool(raw_task.get("publish_checkpoint_after", False))
            and not receipt_published
        ):
            if not str(manifest["protocol_status"]).startswith("mock/"):
                for field in REQUIRED_IDENTITY_FIELDS:
                    if identity_audit[field] != manifest["common_identity"].get(field):
                        raise ConfigError(
                            f"checkpoint producer 与 common identity 不匹配：{field}"
                        )
            files = _primary_checkpoint_files(run_dir, manifest)
            publish_checkpoint_receipt(
                checkpoint_root=run_dir,
                checkpoint_files=files,
                receipt_path=receipt_path,
                producer_worker_id=str(manifest["worker_id"]),
                seed_bundle=int(manifest["seed_bundle"]),
                checkpoint_hand=int(manifest["checkpoint_set"][-1]),
                identity=dict(manifest["common_identity"]),
            )
            receipt_published = True
    if manifest["role"] == "primary" and not receipt_published:
        files = _primary_checkpoint_files(run_dir, manifest)
        publish_checkpoint_receipt(
            checkpoint_root=run_dir,
            checkpoint_files=files,
            receipt_path=receipt_path,
            producer_worker_id=str(manifest["worker_id"]),
            seed_bundle=int(manifest["seed_bundle"]),
            checkpoint_hand=int(manifest["checkpoint_set"][-1]),
            identity=dict(manifest["common_identity"]),
        )
    _write_json_same_or_new(
        run_dir / "task_results.json",
        {
            "schema_version": "task8-worker-task-results-v1",
            "worker_id": manifest["worker_id"],
            "tasks": task_rows,
        },
    )


def _preflight_experiment_tasks(manifest: dict[str, Any]) -> None:
    """Validate config and schedule identities before any worker run directory is created."""

    from agentmemeval.config.loader import load_config
    from agentmemeval.experiments.admission import assess_run_admission

    for index, raw_task in enumerate(manifest["task_configs"], start=1):
        if not isinstance(raw_task, dict):
            raise ConfigError("task_configs 每项必须是映射")
        task_id = str(raw_task.get("task_id", f"task_{index:02d}"))
        _validate_task_id(task_id)
        config_path = Path(str(raw_task.get("config_path", "")))
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        if not config_path.is_file():
            raise ConfigError(f"task {task_id} config 不存在：{config_path}")
        expected_config_sha256 = str(raw_task.get("config_sha256", ""))
        if not expected_config_sha256 or _sha256_file(config_path) != expected_config_sha256:
            raise ConfigError(f"task {task_id} config SHA-256 不匹配")
        config = load_config(config_path)
        experiment = dict(config["experiment"])
        table = dict(config.get("table", {}))
        if manifest["role"] == "primary" and int(experiment.get("train_hands", 0)) > 0:
            checkpoints = list(manifest["checkpoint_set"])
        else:
            checkpoints = [int(manifest["checkpoint_set"][-1])]
        default_hands = int(
            experiment.get("checkpoint_test_hands", experiment.get("test_hands", 0))
        )
        raw_hands = experiment.get("checkpoint_test_hands_by_checkpoint", {})
        hands_by_checkpoint = {
            point: int(
                raw_hands.get(str(point), raw_hands.get(point, default_hands))
                if isinstance(raw_hands, dict)
                else default_hands
            )
            for point in checkpoints
        }
        table_rosters = experiment.get("heldout_table_rosters")
        if experiment.get("heldout_roster_identity"):
            roster_identity: str | dict[str, str] = str(
                experiment["heldout_roster_identity"]
            )
        elif isinstance(table_rosters, dict):
            roster_identity: str | dict[str, str] = {
                table_id: sha256_json(table_rosters[table_id])
                for table_id in manifest["heldout_table_set"]
            }
        else:
            roster_identity = str(
                experiment.get("heldout_roster_identity")
                or sha256_json(
                    config.get("heldout_agent", config.get("opponent_agent", {}))
                )
            )
        schedule = build_heldout_schedule_manifest(
            root_seed=int(manifest["seed_bundle"]),
            checkpoint_set=checkpoints,
            table_set=list(manifest["heldout_table_set"]),
            hands_by_checkpoint=hands_by_checkpoint,
            table_size=int(experiment.get("table_size", table.get("table_size", 4))),
            roster_identity=roster_identity,
        )
        if str(raw_task.get("schedule_sha256", "")) != schedule["schedule_sha256"]:
            raise ConfigError(f"task {task_id} schedule SHA-256 不匹配")
        if not str(manifest["protocol_status"]).startswith("mock/"):
            config["experiment"]["seed"] = int(manifest["seed_bundle"])
            if manifest["role"] == "primary" and int(experiment.get("train_hands", 0)) > 0:
                config["experiment"].pop("checkpoint_interval", None)
                config["experiment"]["checkpoint_set"] = list(manifest["checkpoint_set"])
            assess_run_admission(config, Path.cwd())
    if (
        manifest["role"] == "primary"
        and not str(manifest["protocol_status"]).startswith("mock/")
        and sum(
            bool(task.get("publish_checkpoint_after", False))
            for task in manifest["task_configs"]
            if isinstance(task, dict)
        )
        != 1
    ):
        raise ConfigError("Formal primary 必须恰有一个 checkpoint receipt 释放点")


def _verify_completed_task_identity(
    *,
    manifest: dict[str, Any],
    raw_task: dict[str, Any],
    config: dict[str, Any],
    child_run: Path,
) -> dict[str, Any]:
    child_manifest = _read_json_object(child_run / "manifest.json")
    metadata = dict(child_manifest.get("metadata", {}))
    schedule = _read_json_object(child_run / "schedule_manifest.json")
    actual = {
        "code_sha": dict(metadata.get("code", {})).get("commit"),
        "code_dirty": dict(metadata.get("code", {})).get("dirty"),
        "resolved_config_sha256": sha256_json(_semantic_config(config)),
        "prompt_sha256": sha256_json(metadata.get("prompts", {})),
        "model_fingerprint": sha256_json(metadata.get("model", {})),
        "embedding_fingerprint": sha256_json(metadata.get("embedding", {})),
        "schedule_sha256": schedule.get("schedule_sha256"),
    }
    if not str(manifest["protocol_status"]).startswith("mock/"):
        expected = raw_task.get("expected_identity")
        if not isinstance(expected, dict):
            raise ConfigError("Formal task 缺少 expected_identity")
        for field in REQUIRED_IDENTITY_FIELDS:
            if actual[field] != expected.get(field):
                raise ConfigError(f"Formal task actual identity 不匹配：{field}")
        if actual["code_dirty"] is not False:
            raise ConfigError("Formal task code dirty state 必须为 false")
    _write_json_same_or_new(
        child_run / "task_identity_audit.json",
        {
            "schema_version": "task8-task-identity-audit-v1",
            "task_id": raw_task.get("task_id"),
            "protocol_status": manifest["protocol_status"],
            "actual": actual,
            "status": (
                "not-for-paper/model-substituted"
                if str(manifest["protocol_status"]).startswith("mock/")
                else "verified"
            ),
        },
    )
    return actual


def _semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config, ensure_ascii=False))
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in ("output_root", "run_id", "initial_memory_snapshots", "admission_audit"):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value


def _primary_checkpoint_files(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    suffix = f"checkpoint_{int(manifest['checkpoint_set'][-1]):04d}.json"
    checkpoint_files = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "runs").rglob(f"*_{suffix}")
        if path.is_file()
    }
    if not checkpoint_files:
        raise ConfigError(f"primary 未生成最终 checkpoint：{suffix}")
    identity_names = {
        "manifest.json",
        "resolved_config.yaml",
        "schedule_manifest.json",
        "task_identity_audit.json",
    }
    identity_files = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "runs").rglob("*")
        if path.is_file() and path.name in identity_names
    }
    return sorted(checkpoint_files | identity_files)


def _write_mock_primary_checkpoint(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    relative_files = []
    for mechanism in ("expr", "fact_expr_async"):
        checkpoint_hand = int(manifest["checkpoint_set"][-1])
        relative = f"snapshots/{mechanism}_checkpoint_{checkpoint_hand:04d}.json"
        snapshot = {
            "mechanism": mechanism,
            "agent_id": f"{mechanism}_target",
            "scope": "per_agent",
            "payload": {"history": [{"version": 1, "body": "mock"}]},
            "not_for_paper": True,
            "seed_bundle": manifest["seed_bundle"],
        }
        _write_json_same_or_new(run_dir / relative, snapshot)
        relative_files.append(relative)
    _write_json_same_or_new(
        run_dir / "checkpoint_index.json",
        {
            "checkpoint_hand": int(manifest["checkpoint_set"][-1]),
            "files": relative_files,
            "status": "complete",
        },
    )
    return relative_files


def _write_mock_secondary_branches(run_dir: Path) -> None:
    parent = MemorySnapshot(
        mechanism="fact_expr_async",
        agent_id="async_target",
        scope="per_agent",
        payload={
            "fact": {"records": [{"record_id": "mock-fact"}]},
            "expr": {"history": [{"version": 1, "body": "mock experience"}]},
            "sweep_log": [{"hand": 1}],
            "evidence_review_queue": [],
            "fact_state": {"mock-fact": {"weight": 1.0}},
            "hand_counter": 1,
            "eligible_hand_counter": 1,
        },
    )
    branches = clone_memory_branches(parent)
    for name, snapshot in branches.items():
        _write_json_same_or_new(run_dir / f"branches/{name}.json", snapshot.to_dict())
    _write_json_same_or_new(
        run_dir / "clone_transform_audit.json", build_clone_audit(parent, branches)
    )


def _write_files_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in {"state.tsv", "files.tsv", "completion_receipt.json"}:
            continue
        rows.append((relative, path.stat().st_size, _sha256_file(path)))
    content = "relative_path\tsize\tsha256\n" + "".join(
        f"{relative}\t{size}\t{digest}\n" for relative, size, digest in rows
    )
    _write_bytes_same_or_new(run_dir / "files.tsv", content.encode("utf-8"))


def _validate_task_id(task_id: str) -> None:
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ConfigError(f"task_id 必须是安全单段名称：{task_id}")


def _next_child_attempt_id(child_root: Path, task_id: str) -> str:
    index = 2
    while (child_root / f"{task_id}__attempt_{index:02d}").exists():
        index += 1
    return f"{task_id}__attempt_{index:02d}"


def _directory_file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _verify_task_receipt(
    *,
    marker_path: Path,
    run_dir: Path,
    task_id: str,
    config_sha256: str,
) -> dict[str, Any]:
    marker = _read_json_object(marker_path)
    if marker.get("schema_version") != "task8-worker-task-receipt-v1":
        raise ConfigError(f"task {task_id} receipt schema_version 不匹配")
    if marker.get("task_id") != task_id or marker.get("config_sha256") != config_sha256:
        raise ConfigError(f"task {task_id} receipt identity mismatch")
    child_run = _resolve_inside(run_dir.resolve(), str(marker.get("run_dir", "")))
    if not (child_run / "experiment_result.json").is_file():
        raise ConfigError(f"task {task_id} completion marker 缺失")
    files = marker.get("files")
    if not isinstance(files, list) or files != _directory_file_manifest(child_run):
        raise ConfigError(f"task {task_id} resume artifact integrity mismatch")
    task_row = marker.get("task_row")
    if not isinstance(task_row, dict) or task_row.get("task_id") != task_id:
        raise ConfigError(f"task {task_id} receipt task_row 不匹配")
    return marker


def _prepare_attempt(
    output: Path, manifest: dict[str, Any], resume_existing: bool
) -> tuple[Path, bool]:
    if not output.exists() or not any(output.iterdir()):
        output.mkdir(parents=True, exist_ok=True)
        return output, False
    if resume_existing:
        existing = _read_json_object(output / "worker_manifest.json")
        if sha256_json(existing) != sha256_json(manifest):
            raise ConfigError("--resume-existing identity mismatch")
        _verify_existing_files_manifest(output)
        _verify_completion_receipt(output, expected_worker_id=str(manifest["worker_id"]))
        _last_state(output / "state.tsv")
        return output, True
    index = 2
    while True:
        candidate = output.with_name(f"{output.name}__attempt_{index:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, False
        index += 1


def _verify_existing_files_manifest(run_dir: Path) -> None:
    path = run_dir / "files.tsv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target = _resolve_inside(run_dir.resolve(), row["relative_path"])
            if not target.is_file() or _sha256_file(target) != row["sha256"]:
                raise ConfigError(f"resume artifact integrity mismatch：{row['relative_path']}")


def _verify_completion_receipt(
    run_dir: Path, *, expected_worker_id: str | None = None
) -> None:
    completion_path = run_dir / "completion_receipt.json"
    if not completion_path.exists():
        return
    completion = _read_json_object(completion_path)
    if completion.get("schema_version") != "task8-worker-completion-v1":
        raise ConfigError("resume completion receipt schema_version 不匹配")
    if completion.get("status") != "complete":
        raise ConfigError("resume completion receipt 非 complete")
    if expected_worker_id and completion.get("worker_id") != expected_worker_id:
        raise ConfigError("resume completion receipt worker_id 不匹配")
    files_path = run_dir / "files.tsv"
    expected = str(completion.get("files_tsv_sha256", ""))
    if not files_path.is_file() or not expected or _sha256_file(files_path) != expected:
        raise ConfigError("resume completion receipt files.tsv SHA-256 不匹配")


def _last_state(path: Path) -> str | None:
    return _last_state_and_hash(path)[0]


def _last_state_and_hash(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ConfigError(f"state.tsv 为空：{path}")
    previous_hash = "GENESIS"
    for row in rows:
        if row.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ConfigError(f"state.tsv schema mismatch：{path}")
        if row.get("previous_sha256") != previous_hash:
            raise ConfigError(f"state.tsv hash chain mismatch：{path}")
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
            raise ConfigError(f"state.tsv row hash mismatch：{path}")
        previous_hash = expected
    return str(rows[-1]["status"]), previous_hash


def _validate_common_identity(identity: dict[str, Any], *, allow_pending: bool) -> None:
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ConfigError(f"common identity 缺字段：{', '.join(missing)}")
    for field in REQUIRED_IDENTITY_FIELDS:
        value = str(identity[field]).strip()
        if not value:
            raise ConfigError(f"common identity.{field} 不能为空")
        if not allow_pending and value.upper().startswith(("TBD", "PENDING")):
            raise ConfigError(f"common identity.{field} 尚未冻结")


def _reject_dependency_cycles(manifests: list[dict[str, Any]]) -> None:
    dependencies = {
        str(item["worker_id"]): str(item["depends_on"])
        for item in manifests
        if item.get("depends_on")
    }
    known = {str(item["worker_id"]) for item in manifests}
    if set(dependencies.values()) - known:
        raise ConfigError("worker dependency 指向未知 worker")
    for worker in known:
        seen = set()
        cursor = worker
        while cursor in dependencies:
            if cursor in seen:
                raise ConfigError("worker manifest 存在依赖环")
            seen.add(cursor)
            cursor = dependencies[cursor]


def _validate_task8_matrix(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"R1-E1-I", "R1-E1-M", "R1-E2", "R1-E3", "R1-E4", "R1-E5"}
    by_id = {str(row.get("experiment_id", "")): row for row in rows}
    missing = sorted(required - set(by_id))
    if missing:
        raise ConfigError(f"experiment matrix 缺 TASK8 required rows：{', '.join(missing)}")
    primary = by_id["R1-E1-I"]
    if str(primary.get("checkpoint_set", "")) != "30|75|150|300":
        raise ConfigError("R1-E1-I checkpoint_set 与 TASK8 freeze 不一致")
    if str(primary.get("heldout_tables", "")) != "3":
        raise ConfigError("R1-E1-I heldout_tables 必须为 3")
    if str(by_id["R1-E4"].get("memory_mode", "")) != "Online":
        raise ConfigError("R1-E4 memory_mode 必须为 Online")
    if str(by_id["R1-E5"].get("memory_mode", "")) != "Without":
        raise ConfigError("R1-E5 memory_mode 必须为 Without")


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ConfigError(f"路径必须是安全相对路径：{value}")
    return path


def _resolve_inside(root: Path, relative: str, *, strict: bool = True) -> Path:
    safe = _safe_relative_path(relative)
    candidate = (root / safe).resolve(strict=strict)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"路径越界：{relative}") from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON 顶层必须是对象：{path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ConfigError(f"拒绝覆盖既有文件：{path}") from exc


def _write_json_atomic_new(path: Path, value: Any) -> None:
    """Publish a fully flushed file through an exclusive same-directory hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ConfigError(f"拒绝覆盖既有文件：{path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_same_or_new(path: Path, value: Any) -> None:
    _write_bytes_same_or_new(path, _json_bytes(value))


def _write_bytes_same_or_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ConfigError(f"resume 拒绝改写既有 evidence：{path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
