"""Independent three-seed TASK8B timeboxed runner for one autonomous GPU host."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import load_config, validate_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.runner import run_resolved_config
from agentmemeval.storage.artifacts import get_code_version

SCHEMA_VERSION = "task8b-timeboxed-3seed-worker-v1"
COMPLETION_SCHEMA_VERSION = "task8b-timeboxed-3seed-completion-v1"
TASK_RECEIPT_SCHEMA_VERSION = "task8b-timeboxed-3seed-task-receipt-v1"
FORMAL_SEEDS = (2026100101, 2026100102, 2026100103)
CHECKPOINT_SET = (30, 90, 150, 270)
HELDOUT_TABLE_SET = ("H01", "H02", "H03")
DEFAULT_OUTPUT_ROOT = "outputs/formal/task8b_timeboxed_3seed_qwen_bgem3_20260727"

TASK_HANDS = {
    "isolation_fact": 720,
    "isolation_expr": 720,
    "expr_online": 180,
    "expr_without": 180,
    "isolation_no_memory": 720,
    "isolation_sync": 720,
    "isolation_async": 720,
    "async_online": 180,
    "async_without": 180,
    "mixed_ecological": 750,
}
SHARD_TASKS = {
    "A": (
        "isolation_fact",
        "isolation_expr",
        "expr_online",
        "expr_without",
        "isolation_no_memory",
    ),
    "B": (
        "isolation_sync",
        "isolation_async",
        "async_online",
        "async_without",
        "mixed_ecological",
    ),
}
SOURCE_TASK = {
    "expr_online": "isolation_expr",
    "expr_without": "isolation_expr",
    "async_online": "isolation_async",
    "async_without": "isolation_async",
}
MECHANISM = {
    "isolation_no_memory": "no_memory",
    "isolation_fact": "fact",
    "isolation_expr": "expr",
    "isolation_sync": "fact_expr_sync",
    "isolation_async": "fact_expr_async",
    "expr_online": "expr",
    "expr_without": "expr",
    "async_online": "fact_expr_async",
    "async_without": "fact_expr_async",
}


def worker_plan(*, seed: int, shard: str) -> list[dict[str, Any]]:
    """Return the frozen ordered task plan for one seed/shard host."""

    seed_value = int(seed)
    shard_value = str(shard).upper()
    if seed_value not in FORMAL_SEEDS:
        raise ConfigError(f"timeboxed seed 必须属于 {list(FORMAL_SEEDS)}")
    if shard_value not in SHARD_TASKS:
        raise ConfigError("timeboxed shard 必须为 A 或 B")
    return [
        {
            "order": index,
            "task_id": task_id,
            "planned_hands": TASK_HANDS[task_id],
            "source_task": SOURCE_TASK.get(task_id),
        }
        for index, task_id in enumerate(SHARD_TASKS[shard_value], start=1)
    ]


def render_worker_manifest(
    *,
    base_config_path: str | Path,
    output_root: str | Path,
    host_id: str,
    seed: int,
    shard: str,
) -> dict[str, Any]:
    """Render a deterministic worker manifest without touching the output tree."""

    host = _safe_component(host_id, "host_id")
    seed_value = int(seed)
    shard_value = str(shard).upper()
    base_path = Path(base_config_path).resolve()
    base = load_config(base_path)
    _assert_local_qwen_bgem3(base)
    plan = worker_plan(seed=seed_value, shard=shard_value)
    worker_id = f"seed_{seed_value}__shard_{shard_value}__{host}"
    worker_root = Path(output_root).resolve() / worker_id
    config_identities = {}
    for row in plan:
        config = build_task_config(
            base,
            task_id=str(row["task_id"]),
            seed=seed_value,
            worker_root=worker_root,
            run_id=str(row["task_id"]),
            source_run_dir=None,
            validate=False,
        )
        config_identities[str(row["task_id"])] = sha256_json(_identity_config(config))
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_status": "timeboxed/descriptive-n3/formal",
        "host_id": host,
        "seed": seed_value,
        "shard": shard_value,
        "worker_id": worker_id,
        "worker_root": str(worker_root),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256_file(base_path),
        "decision_model": str(base["provider"]["model"]),
        "decision_base_url_env": str(base["provider"]["base_url_env"]),
        "embedding_backend": str(base["agent"]["embedding_backend"]),
        "embedding_model": str(base["agent"]["embedding_model"]),
        "embedding_base_url_env": str(base["agent"]["embedding_base_url_env"]),
        "checkpoint_set": list(CHECKPOINT_SET),
        "heldout_table_set": list(HELDOUT_TABLE_SET),
        "planned_hands": sum(int(item["planned_hands"]) for item in plan),
        "task_plan": plan,
        "config_identities": config_identities,
        "code_version": get_code_version(Path.cwd()),
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


def build_task_config(
    base: dict[str, Any],
    *,
    task_id: str,
    seed: int,
    worker_root: Path,
    run_id: str,
    source_run_dir: Path | None,
    validate: bool = True,
) -> dict[str, Any]:
    """Build one resolved task config while preserving only true local dependencies."""

    if task_id not in TASK_HANDS:
        raise ConfigError(f"未知 timeboxed task：{task_id}")
    config = copy.deepcopy(base)
    _force_local_qwen(config)
    experiment = config["experiment"]
    experiment.update(
        {
            "run_mode": "formal",
            "protocol_variant": "task8b_timeboxed_3seed_qwen_bgem3_v1",
            "protocol_readiness": "ready",
            "statistical_plan_status": "frozen",
            "required_seed_pairs": 3,
            "seed": int(seed),
            "run_id": run_id,
            "output_root": str((worker_root / "runs").resolve()),
            "train_hands": 270,
            "test_hands": 60,
            "checkpoint_interval": 0,
            "checkpoint_set": list(CHECKPOINT_SET),
            "checkpoint_test_hands": 60,
            "checkpoint_test_hands_by_checkpoint": {
                30: 30,
                90: 30,
                150: 30,
                270: 60,
            },
            "heldout_table_set": list(HELDOUT_TABLE_SET),
            "target_agent_id": "agent_00",
            "evaluation_target_ids": ["agent_00"],
            "evaluate_all_train_agents": False,
            "agent_roster": [],
            "timeboxed_evidence_scope": "descriptive_n3_not_equivalence_or_significance",
        }
    )
    experiment.pop("initial_memory_snapshots", None)
    experiment.pop("initial_checkpoint_hand", None)
    config["agent"]["embedding_cache_path"] = str(
        (worker_root / "embedding_cache" / task_id / "{agent_id}").resolve()
    )

    if task_id == "mixed_ecological":
        _apply_mixed_config(config)
    else:
        config["agent"]["mechanism"] = MECHANISM[task_id]

    if task_id in SOURCE_TASK:
        if source_run_dir is None:
            if validate:
                raise ConfigError(f"comparison {task_id} 缺少本机 source run")
            source = worker_root / "runs" / SOURCE_TASK[task_id]
        else:
            source = source_run_dir.resolve()
        snapshot = source / "memory_snapshots" / "agent_00_checkpoint_0270.json"
        if validate and not snapshot.is_file():
            raise ConfigError(f"comparison {task_id} 缺少 checkpoint-270：{snapshot}")
        experiment["train_hands"] = 0
        experiment.pop("checkpoint_set", None)
        experiment.pop("checkpoint_test_hands_by_checkpoint", None)
        experiment["initial_checkpoint_hand"] = 270
        experiment["memory_mode"] = "Online" if task_id.endswith("_online") else "Without"
        experiment["update_memory_test"] = task_id.endswith("_online")
        experiment["initial_memory_snapshots"] = {"agent_00": str(snapshot)}
    else:
        experiment["memory_mode"] = "Frozen"
        experiment["update_memory_test"] = False

    experiment.pop("checkpoint_interval", None)
    if validate:
        validate_config(config)
    return config


def preflight_worker(
    *,
    base_config_path: str | Path,
    output_root: str | Path,
    host_id: str,
    seed: int,
    shard: str,
    probe_services: bool = True,
) -> dict[str, Any]:
    """Validate topology/configs and optionally probe the two local model services."""

    manifest = render_worker_manifest(
        base_config_path=base_config_path,
        output_root=output_root,
        host_id=host_id,
        seed=seed,
        shard=shard,
    )
    base = load_config(base_config_path)
    worker_root = Path(str(manifest["worker_root"]))
    for row in manifest["task_plan"]:
        task_config = build_task_config(
            base,
            task_id=str(row["task_id"]),
            seed=int(seed),
            worker_root=worker_root,
            run_id=str(row["task_id"]),
            source_run_dir=(
                worker_root / "runs" / str(row["source_task"]) if row.get("source_task") else None
            ),
            validate=False,
        )
        validate_config(task_config)
    service_status = _probe_services(base) if probe_services else {"status": "not_probed"}
    return {
        "status": "ready",
        "worker_id": manifest["worker_id"],
        "planned_hands": manifest["planned_hands"],
        "decision_model": manifest["decision_model"],
        "embedding_model": manifest["embedding_model"],
        "services": service_status,
    }


def run_worker(
    *,
    base_config_path: str | Path,
    output_root: str | Path,
    host_id: str,
    seed: int,
    shard: str,
    probe_services: bool = True,
) -> dict[str, Any]:
    """Run every ordered task assigned to one host, resuming only sealed tasks."""

    host = _safe_component(host_id, "host_id")
    seed_value = int(seed)
    shard_value = str(shard).upper()
    worker_plan(seed=seed_value, shard=shard_value)
    worker_id = f"seed_{seed_value}__shard_{shard_value}__{host}"
    existing_root = Path(output_root).resolve() / worker_id
    existing_completion = existing_root / "completion.json"
    if existing_completion.is_file():
        return _verify_completion(existing_completion, existing_root)

    manifest = render_worker_manifest(
        base_config_path=base_config_path,
        output_root=output_root,
        host_id=host_id,
        seed=seed,
        shard=shard,
    )
    worker_root = Path(str(manifest["worker_root"]))
    worker_root.mkdir(parents=True, exist_ok=True)
    _write_json_same_or_new(worker_root / "worker_manifest.json", manifest)
    base = load_config(base_config_path)
    if probe_services:
        service_status = _probe_services(base)
        service_path = worker_root / "service_preflight.json"
        if not service_path.exists():
            _write_json_new(service_path, service_status)
    receipts_dir = worker_root / "task_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}

    for row in manifest["task_plan"]:
        task_id = str(row["task_id"])
        receipt_path = receipts_dir / f"{task_id}.json"
        if receipt_path.is_file():
            completed[task_id] = _verify_task_receipt(receipt_path, worker_root, task_id)
            continue
        source_run_dir = None
        source_task = row.get("source_task")
        if source_task:
            source_receipt = completed.get(str(source_task))
            if source_receipt is None:
                source_path = receipts_dir / f"{source_task}.json"
                source_receipt = _verify_task_receipt(source_path, worker_root, str(source_task))
                completed[str(source_task)] = source_receipt
            source_run_dir = _resolve_inside(worker_root, str(source_receipt["run_dir"]))
        run_id = _next_run_id(worker_root / "runs", task_id)
        config = build_task_config(
            base,
            task_id=task_id,
            seed=int(seed),
            worker_root=worker_root,
            run_id=run_id,
            source_run_dir=source_run_dir,
            validate=True,
        )
        result = run_resolved_config(config)
        run_dir = Path(str(result.artifacts["run_dir"])).resolve()
        _ensure_inside(worker_root, run_dir)
        observed_rows = _lightweight_run_check(run_dir)
        receipt = {
            "schema_version": TASK_RECEIPT_SCHEMA_VERSION,
            "task_id": task_id,
            "status": "complete",
            "planned_hands": int(row["planned_hands"]),
            "observed_hand_summary_rows": observed_rows,
            "run_dir": run_dir.relative_to(worker_root).as_posix(),
            "completed_at_utc": _utc_now(),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        _write_json_new(receipt_path, receipt)
        completed[task_id] = receipt

    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "worker_id": manifest["worker_id"],
        "host_id": manifest["host_id"],
        "seed": int(seed),
        "shard": str(shard).upper(),
        "status": "complete",
        "planned_hands": manifest["planned_hands"],
        "task_count": len(completed),
        "tasks": [completed[str(row["task_id"])] for row in manifest["task_plan"]],
        "completed_at_utc": _utc_now(),
    }
    completion["completion_sha256"] = sha256_json(completion)
    _write_json_same_or_new(worker_root / "completion.json", completion)
    return completion


def worker_status(
    *, output_root: str | Path, host_id: str, seed: int, shard: str
) -> dict[str, Any]:
    host = _safe_component(host_id, "host_id")
    shard_value = str(shard).upper()
    worker_id = f"seed_{int(seed)}__shard_{shard_value}__{host}"
    worker_root = Path(output_root).resolve() / worker_id
    completion = worker_root / "completion.json"
    if completion.is_file():
        return _read_json(completion)
    rows = []
    for task in worker_plan(seed=int(seed), shard=shard_value):
        receipt = worker_root / "task_receipts" / f"{task['task_id']}.json"
        rows.append(
            {
                "task_id": task["task_id"],
                "status": "complete" if receipt.is_file() else "pending",
            }
        )
    return {"worker_id": worker_id, "status": "running_or_pending", "tasks": rows}


def _apply_mixed_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    experiment["primary_estimand"] = "same_seed_table_run_mechanism_effect_vs_baseline"
    experiment["primary_baseline_mechanism"] = "fact"
    experiment["within_table_mechanism_aggregation"] = "arithmetic_mean"
    experiment["agent_roster"] = [
        {"agent_id": "fact_00", "mechanism": "fact"},
        {"agent_id": "fact_01", "mechanism": "fact"},
        {"agent_id": "expr_00", "mechanism": "expr"},
        {"agent_id": "expr_01", "mechanism": "expr"},
        {"agent_id": "sync_00", "mechanism": "fact_expr_sync"},
        {"agent_id": "sync_01", "mechanism": "fact_expr_sync"},
        {"agent_id": "async_00", "mechanism": "fact_expr_async"},
        {"agent_id": "async_01", "mechanism": "fact_expr_async"},
    ]
    experiment["evaluate_all_train_agents"] = True
    experiment.pop("target_agent_id", None)
    experiment.pop("evaluation_target_ids", None)
    experiment["checkpoint_set"] = [270]
    experiment["checkpoint_test_hands_by_checkpoint"] = {270: 60}
    experiment["heldout_table_set"] = ["H01"]
    rosters = experiment.get("heldout_table_rosters", {})
    if isinstance(rosters, dict) and "H01" in rosters:
        experiment["heldout_table_rosters"] = {"H01": rosters["H01"]}
    experiment["memory_mode"] = "Frozen"
    experiment["update_memory_test"] = False


def _force_local_qwen(config: dict[str, Any]) -> None:
    provider = config["provider"]
    provider.update(
        {
            "provider": "openai_compatible",
            "model": "qwen/qwen3.5-9b",
            "base_url_env": "LOCAL_LLM_BASE_URL",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "api_key_required": False,
            "response_model_identity_policy": "strict_exact",
            "response_model_aliases": ["qwen/qwen3.5-9b"],
        }
    )
    reviewer = config.get("experience_reviewer_provider")
    if isinstance(reviewer, dict):
        config["experience_reviewer_provider"] = copy.deepcopy(provider)
    _assert_local_qwen_bgem3(config)


def _assert_local_qwen_bgem3(config: dict[str, Any]) -> None:
    provider = config.get("provider", {})
    agent = config.get("agent", {})
    if provider.get("provider") != "openai_compatible":
        raise ConfigError("timeboxed decision provider 必须为 openai_compatible")
    if "qwen" not in str(provider.get("model", "")).lower():
        raise ConfigError("timeboxed decision model 必须为本地 Qwen")
    if provider.get("base_url_env") != "LOCAL_LLM_BASE_URL":
        raise ConfigError("timeboxed Qwen 必须读取 LOCAL_LLM_BASE_URL")
    if agent.get("embedding_backend") != "bgem3_hybrid_http":
        raise ConfigError("timeboxed embedding backend 必须为 bgem3_hybrid_http")
    if str(agent.get("embedding_model", "")) != "BAAI/bge-m3":
        raise ConfigError("timeboxed embedding model 必须为 BAAI/bge-m3")
    if agent.get("embedding_base_url_env") != "BGEM3_BASE_URL":
        raise ConfigError("timeboxed BGE-M3 必须读取 BGEM3_BASE_URL")


def _identity_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    experiment = value.get("experiment", {})
    experiment["output_root"] = "<WORKER_ROOT>/runs"
    experiment["run_id"] = "<TASK_RUN_ID>"
    agent = value.get("agent", {})
    if "embedding_cache_path" in agent:
        agent["embedding_cache_path"] = "<WORKER_ROOT>/embedding_cache/{agent_id}"
    experiment.pop("initial_memory_snapshots", None)
    return value


def _probe_services(config: dict[str, Any]) -> dict[str, Any]:
    provider = config["provider"]
    agent = config["agent"]
    decision_env = str(provider["base_url_env"])
    embedding_env = str(agent["embedding_base_url_env"])
    decision_base = str(os.environ.get(decision_env, "")).strip()
    embedding_base = str(os.environ.get(embedding_env, "")).strip()
    if not decision_base:
        raise ConfigError(f"缺少本地 Qwen 地址环境变量：{decision_env}")
    if not embedding_base:
        raise ConfigError(f"缺少本地 BGE-M3 地址环境变量：{embedding_env}")
    models = _probe_candidates(
        [f"{decision_base.rstrip('/')}/models", f"{decision_base.rstrip('/')}/v1/models"],
        timeout=float(provider.get("timeout_seconds", 120)),
    )
    if "qwen" not in json.dumps(models, ensure_ascii=False).lower():
        raise ConfigError("本地 decision /models 未报告 Qwen")
    bgem3_candidates = [f"{embedding_base.rstrip('/')}/health"]
    if embedding_base.rstrip("/").endswith("/v1"):
        bgem3_candidates.append(f"{embedding_base.rstrip('/')[:-3]}/health")
    health = _probe_candidates(
        bgem3_candidates,
        timeout=float(agent.get("embedding_timeout_seconds", 120)),
    )
    return {
        "schema_version": "task8b-timeboxed-local-service-preflight-v1",
        "status": "healthy",
        "decision_model": str(provider["model"]),
        "decision_base_url_env": decision_env,
        "embedding_model": str(agent["embedding_model"]),
        "embedding_base_url_env": embedding_env,
        "embedding_health": health,
        "checked_at_utc": _utc_now(),
    }


def _probe_candidates(urls: list[str], *, timeout: float) -> Any:
    errors = []
    for url in dict.fromkeys(urls):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read(1024 * 1024).decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {type(exc).__name__}")
    raise ConfigError("本地模型服务探测失败：" + "; ".join(errors))


def _lightweight_run_check(run_dir: Path) -> int:
    required = (
        "manifest.json",
        "resolved_config.yaml",
        "hand_summaries.jsonl",
        "metrics.json",
        "experiment_result.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ConfigError(f"task 输出缺少关键文件：{', '.join(missing)}")
    summaries = (run_dir / "hand_summaries.jsonl").open(encoding="utf-8")
    with summaries:
        rows = sum(1 for line in summaries if line.strip())
    if rows <= 0:
        raise ConfigError("task hand_summaries 为空")
    return rows


def _verify_task_receipt(path: Path, worker_root: Path, task_id: str) -> dict[str, Any]:
    value = _read_json(path)
    digest = str(value.pop("receipt_sha256", ""))
    if (
        value.get("schema_version") != TASK_RECEIPT_SCHEMA_VERSION
        or value.get("task_id") != task_id
        or value.get("status") != "complete"
        or sha256_json(value) != digest
    ):
        raise ConfigError(f"task receipt 非法：{path}")
    value["receipt_sha256"] = digest
    run_dir = _resolve_inside(worker_root, str(value["run_dir"]))
    _lightweight_run_check(run_dir)
    return value


def _verify_completion(path: Path, worker_root: Path) -> dict[str, Any]:
    value = _read_json(path)
    digest = str(value.pop("completion_sha256", ""))
    if (
        value.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or value.get("status") != "complete"
        or sha256_json(value) != digest
    ):
        raise ConfigError(f"worker completion 非法：{path}")
    for task in value.get("tasks", []):
        if not isinstance(task, dict) or not task.get("task_id"):
            raise ConfigError(f"worker completion task 非法：{path}")
        receipt_path = worker_root / "task_receipts" / f"{task['task_id']}.json"
        _verify_task_receipt(receipt_path, worker_root, str(task["task_id"]))
    value["completion_sha256"] = digest
    return value


def _next_run_id(runs_root: Path, task_id: str) -> str:
    direct = runs_root / task_id
    if not direct.exists() or not any(direct.iterdir()):
        return task_id
    attempt = 2
    while (runs_root / f"{task_id}__attempt_{attempt:02d}").exists():
        attempt += 1
    return f"{task_id}__attempt_{attempt:02d}"


def _safe_component(value: str, label: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ConfigError(f"{label} 只能包含字母、数字、下划线和连字符")
    return text


def _ensure_inside(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigError(f"路径越出 worker root：{candidate}") from exc


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root.resolve() / relative).resolve()
    _ensure_inside(root, candidate)
    if not candidate.exists():
        raise ConfigError(f"receipt 路径不存在：{candidate}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON 顶层必须为对象：{path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_json_same_or_new(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ConfigError(f"拒绝覆盖不同内容：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TASK8B three-seed timeboxed local-model runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "run", "status"):
        item = sub.add_parser(name)
        item.add_argument(
            "--base-config",
            default="configs/formal/task8b_timeboxed_3seed_qwen_bgem3.yaml",
        )
        item.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
        item.add_argument("--host-id", required=True)
        item.add_argument("--seed", required=True, type=int)
        item.add_argument("--shard", required=True, choices=("A", "B", "a", "b"))
        if name == "preflight":
            item.add_argument("--no-service-probe", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    common = {
        "output_root": args.output_root,
        "host_id": args.host_id,
        "seed": args.seed,
        "shard": args.shard,
    }
    if args.command == "plan":
        result = render_worker_manifest(base_config_path=args.base_config, **common)
    elif args.command == "preflight":
        result = preflight_worker(
            base_config_path=args.base_config,
            probe_services=not args.no_service_probe,
            **common,
        )
    elif args.command == "run":
        result = run_worker(base_config_path=args.base_config, **common)
    else:
        result = worker_status(**common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
