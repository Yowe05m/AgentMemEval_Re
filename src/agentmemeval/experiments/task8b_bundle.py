"""Deterministic TASK8B executable bundle construction from one frozen base config."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.config.loader import load_config, load_raw_config, validate_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import (
    build_heldout_schedule_manifest,
    canonicalize_resolved_config_identity,
    sha256_json,
)
from agentmemeval.experiments.formal_runner import (
    FLEET_COMMON_IDENTITY_FIELDS,
    TASK8B_EXPEDITED_STATUS,
    TASK8B_FLEET_LOCK_FIELDS,
    TASK8B_FORMAL_SEEDS,
    generate_worker_manifests,
    validate_worker_manifest_set,
)

CHECKPOINT_SET = [30, 75, 150, 300]
HELDOUT_TABLE_SET = ["H01", "H02", "H03"]
MECHANISMS = (
    ("no_memory", "no_memory"),
    ("fact", "fact"),
    ("expr", "expr"),
    ("sync", "fact_expr_sync"),
    ("async", "fact_expr_async"),
)


def build_task8b_executable_bundle(
    *,
    matrix_path: str | Path,
    base_config_path: str | Path,
    fleet_identity_path: str | Path,
    output_dir: str | Path,
    runtime_bundle_root: str,
    canary_seed: int | None = None,
) -> dict[str, Any]:
    """Write configs and executable P/S manifests without overwriting any file."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(f"TASK8B bundle 输出目录非空：{destination}")
    base = load_raw_config(base_config_path)
    base["experiment"].pop("checkpoint_interval", None)
    validate_config(base)
    identity = _read_json(Path(fleet_identity_path))
    _validate_fleet_identity(identity)
    is_canary = canary_seed is not None
    seeds = [int(canary_seed)] if is_canary else list(TASK8B_FORMAL_SEEDS)
    checkpoints = [1, 3, 5] if is_canary else CHECKPOINT_SET
    configs_dir = destination / "configs"
    manifests_dir = destination / "manifests"
    configs = _write_task_configs(base, configs_dir, is_canary=is_canary)
    task_configs_by_worker: dict[str, list[dict[str, Any]]] = {}
    receipt_identities: dict[str, dict[str, Any]] = {}
    pod_identities: dict[int, dict[str, Any]] = {}
    for index, seed in enumerate(seeds, start=1):
        primary_id = f"P{index:02d}"
        secondary_id = f"S{index:02d}"
        primary_tasks = _primary_tasks(
            configs=configs,
            seed=seed,
            checkpoints=checkpoints,
            fleet_identity=identity,
            runtime_bundle_root=runtime_bundle_root,
            is_canary=is_canary,
        )
        secondary_tasks = _secondary_tasks(
            configs=configs,
            seed=seed,
            checkpoints=checkpoints,
            fleet_identity=identity,
            runtime_bundle_root=runtime_bundle_root,
            is_canary=is_canary,
        )
        task_configs_by_worker[primary_id] = primary_tasks
        task_configs_by_worker[secondary_id] = secondary_tasks
        receipt_identities[primary_id] = dict(primary_tasks[-1]["expected_identity"])
        schedule_rows = [
            {
                "worker_role": role,
                "task_id": task["task_id"],
                "schedule_sha256": task["schedule_sha256"],
            }
            for role, tasks in (("primary", primary_tasks), ("secondary", secondary_tasks))
            for task in tasks
        ]
        pod_identities[seed] = {
            "seed_bundle": seed,
            "schedule_sha256": sha256_json(
                {
                    "schema_version": "task8b-seed-pod-schedule-bundle-v1",
                    "seed_bundle": seed,
                    "task_schedules": schedule_rows,
                }
            ),
            "task_schedules": schedule_rows,
        }
    canary_total = None
    status = TASK8B_EXPEDITED_STATUS
    if is_canary:
        status = "canary/not-for-paper"
        canary_total = sum(
            int(task["planned_hands"])
            for tasks in task_configs_by_worker.values()
            for task in tasks
        )
        if canary_total > 100:
            raise ConfigError(f"TASK8B canary 实际预算超过 100 hands：{canary_total}")
    index = generate_worker_manifests(
        matrix_path=matrix_path,
        seeds=seeds,
        common_identity=identity,
        output_dir=manifests_dir,
        output_root="outputs/formal/task8b",
        cache_root="task8b",
        protocol_status=status,
        execution_mode="experiment_configs",
        canary_total_hands=canary_total,
        task_configs_by_worker=task_configs_by_worker,
        receipt_identities_by_worker=receipt_identities,
        seed_pod_identities=pod_identities,
    )
    result = {
        "schema_version": "task8b-executable-bundle-v1",
        "protocol_status": status,
        "seed_count": len(seeds),
        "worker_count": len(seeds) * 2,
        "planned_hands": sum(
            int(task["planned_hands"])
            for tasks in task_configs_by_worker.values()
            for task in tasks
        ),
        "manifest_index_sha256": _sha256(manifests_dir / "manifest_index.json"),
        "manifest_bundle_sha256": index["bundle_sha256"],
        "runtime_bundle_root": runtime_bundle_root,
    }
    _write_json_new(destination / "bundle_manifest.json", result)
    return result


def build_task8b_host_schedule(
    *,
    manifest_dir: str | Path,
    host_ids: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind 24 frozen worker cards to 11 reusable physical hosts in gated waves."""

    hosts = [str(host).strip() for host in host_ids]
    if len(hosts) != 11 or any(not host for host in hosts):
        raise ConfigError("TASK8B 24-worker 调度要求恰好 11 个非空 physical host IDs")
    if len(set(hosts)) != len(hosts):
        raise ConfigError("TASK8B physical host IDs 必须唯一")
    source = Path(manifest_dir)
    index = _read_json(source / "manifest_index.json")
    index_body = dict(index)
    recorded_bundle_sha256 = str(index_body.pop("bundle_sha256", ""))
    if not recorded_bundle_sha256 or sha256_json(index_body) != recorded_bundle_sha256:
        raise ConfigError("TASK8B manifest index bundle SHA-256 不匹配")
    if int(index.get("worker_count", 0)) != 24 or int(index.get("seed_count", 0)) != 12:
        raise ConfigError("TASK8B host schedule 只接受冻结的 12x2 / 24-worker bundle")
    raw_rows = index.get("workers")
    if not isinstance(raw_rows, list) or len(raw_rows) != 24:
        raise ConfigError("TASK8B manifest index worker rows 不完整")
    manifests: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ConfigError("TASK8B manifest index worker row 非法")
        worker_id = str(row.get("worker_id", ""))
        manifest = _read_json(source / f"{worker_id}.json")
        if sha256_json(manifest) != str(row.get("manifest_sha256", "")):
            raise ConfigError(f"TASK8B worker manifest SHA-256 不匹配：{worker_id}")
        if (
            manifest.get("role") != row.get("role")
            or int(manifest.get("seed_bundle", -1)) != int(row.get("seed_bundle", -2))
        ):
            raise ConfigError(f"TASK8B manifest index identity 不匹配：{worker_id}")
        manifests[worker_id] = manifest
    validate_worker_manifest_set(list(manifests.values()), expected_seed_count=12)

    pending = set(manifests)
    completed_primaries: set[str] = set()
    waves: list[dict[str, Any]] = []
    while pending:
        eligible = sorted(
            (
                manifest
                for worker_id, manifest in manifests.items()
                if worker_id in pending
                and (
                    manifest["role"] == "primary"
                    or str(manifest.get("depends_on")) in completed_primaries
                )
            ),
            key=lambda item: (item["role"] != "primary", str(item["worker_id"])),
        )
        if not eligible:
            raise ConfigError("TASK8B host schedule 无可运行 worker；P/S receipt 依赖死锁")
        selected = eligible[: len(hosts)]
        wave_number = len(waves) + 1
        cards = []
        host_slots = hosts[: len(selected)]
        for slot, (host_id, manifest) in enumerate(
            zip(host_slots, selected, strict=True), start=1
        ):
            worker_id = str(manifest["worker_id"])
            card = {
                "slot": slot,
                "host_id": host_id,
                "worker_id": worker_id,
                "role": manifest["role"],
                "seed_bundle": manifest["seed_bundle"],
                "manifest_relative_path": f"{worker_id}.json",
                "manifest_sha256": sha256_json(manifest),
                "cache_namespace": manifest["instance_identity"]["cache_namespace"],
                "output_path": manifest["instance_identity"]["output_path"],
                "depends_on": manifest.get("depends_on"),
                "start_gate": (
                    {"status": "manifest_admitted"}
                    if manifest["role"] == "primary"
                    else {
                        "status": "verified_primary_receipt_required",
                        "producer_worker_id": manifest["depends_on"],
                        "receipt_relative_path": manifest["receipt_relative_path"],
                    }
                ),
            }
            cards.append(card)
            pending.remove(worker_id)
        completed_primaries.update(
            str(manifest["worker_id"])
            for manifest in selected
            if manifest["role"] == "primary"
        )
        waves.append(
            {
                "wave": wave_number,
                "max_active": len(cards),
                "assignments": cards,
            }
        )
    schedule = {
        "schema_version": "task8b-11-host-wave-schedule-v1",
        "manifest_bundle_sha256": recorded_bundle_sha256,
        "physical_host_count": len(hosts),
        "max_active_workers": len(hosts),
        "worker_count": len(manifests),
        "wave_count": len(waves),
        "waves": waves,
    }
    schedule["schedule_sha256"] = sha256_json(schedule)
    _write_json_new(Path(output_path), schedule)
    return schedule


def _write_task_configs(
    base: dict[str, Any], destination: Path, *, is_canary: bool
) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for label, mechanism in MECHANISMS:
        value = copy.deepcopy(base)
        value["agent"]["mechanism"] = mechanism
        experiment = value["experiment"]
        experiment["agent_roster"] = []
        experiment["evaluate_all_train_agents"] = False
        experiment["target_agent_id"] = "agent_00"
        experiment["evaluation_target_ids"] = ["agent_00"]
        _apply_budget(experiment, is_canary=is_canary, mixed=False)
        path = destination / f"isolation_{label}.yaml"
        _write_yaml_new(path, value)
        configs[label] = path
    mixed = copy.deepcopy(base)
    experiment = mixed["experiment"]
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
    _apply_budget(experiment, is_canary=is_canary, mixed=True)
    path = destination / "mixed_ecological.yaml"
    _write_yaml_new(path, mixed)
    configs["mixed"] = path
    return configs


def _apply_budget(experiment: dict[str, Any], *, is_canary: bool, mixed: bool) -> None:
    if is_canary:
        experiment["train_hands"] = 5
        experiment["checkpoint_set"] = [1, 3, 5]
        experiment["checkpoint_test_hands_by_checkpoint"] = {1: 1, 3: 1, 5: 1}
        experiment["checkpoint_test_hands"] = 1
        experiment["test_hands"] = 1
        experiment["run_mode"] = "pilot"
        experiment["not_for_analysis"] = True
        experiment["not_for_paper"] = True
    elif mixed:
        experiment["train_hands"] = 300
        experiment["checkpoint_set"] = [300]
        experiment["checkpoint_test_hands_by_checkpoint"] = {300: 100}
        experiment["checkpoint_test_hands"] = 100
        experiment["test_hands"] = 100
    else:
        experiment["train_hands"] = 300
        experiment["checkpoint_set"] = CHECKPOINT_SET
        experiment["checkpoint_test_hands_by_checkpoint"] = {
            30: 50,
            75: 50,
            150: 50,
            300: 200,
        }
        experiment["checkpoint_test_hands"] = 200
        experiment["test_hands"] = 200
    experiment.pop("checkpoint_interval", None)
    experiment["heldout_table_set"] = HELDOUT_TABLE_SET


def _primary_tasks(
    *,
    configs: dict[str, Path],
    seed: int,
    checkpoints: list[int],
    fleet_identity: dict[str, Any],
    runtime_bundle_root: str,
    is_canary: bool,
) -> list[dict[str, Any]]:
    selected = MECHANISMS if not is_canary else (("async", "fact_expr_async"),)
    tasks = []
    for label, _mechanism in selected:
        config = load_config(configs[label])
        schedule = _schedule(config, seed, checkpoints)
        planned = _planned_hands(config, checkpoints)
        tasks.append(
            _task_row(
                task_id=f"isolation_{label}",
                config_path=configs[label],
                runtime_bundle_root=runtime_bundle_root,
                schedule=schedule,
                expected_identity=_expected_identity(config, seed, schedule, fleet_identity),
                planned_hands=planned,
                covers=["R1-E1-I", "R1-E2", "R1-E3"],
            )
        )
    tasks[-1]["publish_checkpoint_after"] = True
    return tasks


def _secondary_tasks(
    *,
    configs: dict[str, Path],
    seed: int,
    checkpoints: list[int],
    fleet_identity: dict[str, Any],
    runtime_bundle_root: str,
    is_canary: bool,
) -> list[dict[str, Any]]:
    if is_canary:
        branches = (("Frozen", 1), ("Online", 1), ("Without", 1))
        return [
            _checkpoint_task(
                label="async",
                mode=mode,
                planned_hands=hands * 3,
                seed=seed,
                checkpoints=checkpoints,
                config_path=configs["async"],
                fleet_identity=fleet_identity,
                runtime_bundle_root=runtime_bundle_root,
                source_task="isolation_async",
                covers=["R1-E4" if mode == "Online" else "R1-E5"],
            )
            for mode, hands in branches
        ]
    mixed_config = load_config(configs["mixed"])
    mixed_schedule = _schedule(mixed_config, seed, [300])
    tasks = [
        {
            **_task_row(
                task_id="mixed_ecological",
                config_path=configs["mixed"],
                runtime_bundle_root=runtime_bundle_root,
                schedule=mixed_schedule,
                expected_identity=_expected_identity(
                    mixed_config, seed, mixed_schedule, fleet_identity
                ),
                planned_hands=_planned_hands(mixed_config, [300]),
                covers=["R1-E1-M"],
            ),
            "dependency_mode": "standalone",
            "memory_mode": "Frozen",
            "checkpoint_set": [300],
        }
    ]
    for label in ("expr", "async"):
        for mode, family in (("Online", "R1-E4"), ("Without", "R1-E5")):
            tasks.append(
                _checkpoint_task(
                    label=label,
                    mode=mode,
                    planned_hands=600,
                    seed=seed,
                    checkpoints=checkpoints,
                    config_path=configs[label],
                    fleet_identity=fleet_identity,
                    runtime_bundle_root=runtime_bundle_root,
                    source_task=f"isolation_{label}",
                    covers=[family],
                )
            )
    return tasks


def _checkpoint_task(
    *,
    label: str,
    mode: str,
    planned_hands: int,
    seed: int,
    checkpoints: list[int],
    config_path: Path,
    fleet_identity: dict[str, Any],
    runtime_bundle_root: str,
    source_task: str,
    covers: list[str],
) -> dict[str, Any]:
    config = load_config(config_path)
    schedule = _schedule(config, seed, [checkpoints[-1]])
    expected_config = copy.deepcopy(config)
    experiment = expected_config["experiment"]
    experiment["train_hands"] = 0
    experiment.pop("checkpoint_set", None)
    experiment.pop("checkpoint_test_hands_by_checkpoint", None)
    experiment["initial_checkpoint_hand"] = checkpoints[-1]
    experiment["memory_mode"] = mode
    experiment["update_memory_test"] = mode == "Online"
    return {
        **_task_row(
            task_id=f"{label}_{mode.lower()}",
            config_path=config_path,
            runtime_bundle_root=runtime_bundle_root,
            schedule=schedule,
            expected_identity=_expected_identity(expected_config, seed, schedule, fleet_identity),
            planned_hands=planned_hands,
            covers=covers,
        ),
        "dependency_mode": "checkpoint",
        "memory_mode": mode,
        "checkpoint_bindings": {
            "agent_00": (
                f"runs/{source_task}/memory_snapshots/"
                f"agent_00_checkpoint_{checkpoints[-1]:04d}.json"
            )
        },
    }


def _task_row(
    *,
    task_id: str,
    config_path: Path,
    runtime_bundle_root: str,
    schedule: dict[str, Any],
    expected_identity: dict[str, Any],
    planned_hands: int,
    covers: list[str],
) -> dict[str, Any]:
    runtime_path = Path(runtime_bundle_root) / "configs" / config_path.name
    return {
        "task_id": task_id,
        "config_path": runtime_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "expected_identity": expected_identity,
        "planned_hands": planned_hands,
        "covers": covers,
    }


def _schedule(config: dict[str, Any], seed: int, checkpoints: list[int]) -> dict[str, Any]:
    experiment = config["experiment"]
    raw_hands = experiment.get("checkpoint_test_hands_by_checkpoint", {})
    default_hands = int(experiment.get("checkpoint_test_hands", 0))
    hands = {
        point: int(raw_hands.get(point, raw_hands.get(str(point), default_hands)))
        for point in checkpoints
    }
    rosters = experiment.get("heldout_table_rosters")
    if not isinstance(rosters, dict) or set(rosters) != set(HELDOUT_TABLE_SET):
        raise ConfigError("TASK8B base config 必须冻结 H01/H02/H03 heldout_table_rosters")
    roster_identity = {table_id: sha256_json(rosters[table_id]) for table_id in HELDOUT_TABLE_SET}
    return build_heldout_schedule_manifest(
        root_seed=seed,
        checkpoint_set=checkpoints,
        table_set=HELDOUT_TABLE_SET,
        hands_by_checkpoint=hands,
        table_size=int(experiment.get("table_size", config["table"].get("table_size", 8))),
        roster_identity=roster_identity,
    )


def _planned_hands(config: dict[str, Any], checkpoints: list[int]) -> int:
    experiment = config["experiment"]
    raw_hands = experiment.get("checkpoint_test_hands_by_checkpoint", {})
    default_hands = int(experiment.get("checkpoint_test_hands", 0))
    tests = sum(
        int(raw_hands.get(point, raw_hands.get(str(point), default_hands))) for point in checkpoints
    ) * len(HELDOUT_TABLE_SET)
    target_count = (
        len(experiment.get("agent_roster", []))
        if experiment.get("evaluate_all_train_agents")
        else 1
    )
    return int(experiment.get("train_hands", 0)) + tests * target_count


def _expected_identity(
    config: dict[str, Any],
    seed: int,
    schedule: dict[str, Any],
    fleet_identity: dict[str, Any],
) -> dict[str, Any]:
    semantic = canonicalize_resolved_config_identity(config)
    semantic["experiment"]["seed"] = seed
    return {
        **{field: fleet_identity[field] for field in FLEET_COMMON_IDENTITY_FIELDS},
        "resolved_config_sha256": sha256_json(semantic),
        "schedule_sha256": schedule["schedule_sha256"],
    }


def _validate_fleet_identity(identity: dict[str, Any]) -> None:
    missing = [field for field in TASK8B_FLEET_LOCK_FIELDS if not identity.get(field)]
    if missing:
        raise ConfigError(f"TASK8B fleet identity 缺字段：{', '.join(missing)}")
    code_sha = str(identity["code_sha"])
    if len(code_sha) != 40 or any(char not in "0123456789abcdef" for char in code_sha.lower()):
        raise ConfigError("TASK8B fleet identity code_sha 必须是完整 commit SHA")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取 TASK8B JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError("TASK8B identity JSON 顶层必须是对象")
    return value


def _write_yaml_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(value, allow_unicode=True, sort_keys=True)
    try:
        with path.open("x", encoding="utf-8", errors="strict", newline="\n") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ConfigError(f"拒绝覆盖 TASK8B config：{path}") from exc


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ConfigError(f"拒绝覆盖 TASK8B bundle：{path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
