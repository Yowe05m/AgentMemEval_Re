"""Append-only multi-condition campaign execution and aggregation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.config.loader import load_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.aggregation import (
    aggregate_metrics,
    validate_runtime_homogeneity,
)
from agentmemeval.evaluation.statistics import (
    holm_adjust,
    paired_sign_flip_p_value,
    summarize_values,
)
from agentmemeval.experiments.runner import run_resolved_config
from agentmemeval.storage.artifacts import get_code_version

STATE_FIELDS = (
    "event_utc",
    "condition_id",
    "target_mechanism",
    "seed",
    "attempt",
    "status",
    "run_id",
    "run_dir",
    "failure_class",
    "message",
)
REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)
TARGET_VS_NO_MEMORY_ESTIMAND = (
    "same_seed_cross_condition_target_effect_vs_no_memory"
)


def run_campaign(path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Run or resume a frozen campaign without overwriting any prior attempt."""

    config_path = Path(path).resolve()
    raw = _read_campaign_yaml(config_path)
    spec = raw["campaign"]
    base_path = (config_path.parent / str(spec["base_experiment_config"])).resolve()
    base_config = load_config(base_path)
    _validate_campaign_spec(spec, base_config)
    campaign_id = _slug(str(spec["campaign_id"]))
    output_root = Path(str(spec.get("output_root", "outputs/campaigns"))).resolve()
    campaign_dir = output_root / campaign_id
    payload = {
        "campaign": spec,
        "base_config": _without_internal_keys(base_config),
    }
    config_hash = _json_hash(payload)
    manifest_path = campaign_dir / "campaign_manifest.json"
    state_path = campaign_dir / "state.tsv"
    events_path = campaign_dir / "campaign_events.jsonl"
    if campaign_dir.exists():
        if not resume:
            raise FileExistsError(
                f"campaign 目录已存在，必须显式 --resume：{campaign_dir}"
            )
        _verify_existing_manifest(manifest_path, config_hash)
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": "agentmemeval_campaign_v1",
            "campaign_id": campaign_id,
            "created_utc": _utc_now(),
            "config_sha256": config_hash,
            "config_source": str(config_path),
            "base_experiment_config": str(base_path),
            "code_version_at_creation": get_code_version(Path.cwd()),
            **payload,
        }
        _write_new_json(manifest_path, manifest)
    _ensure_state_file(state_path)

    conditions = _conditions(spec)
    seeds = [int(seed) for seed in spec["seeds"]]
    state_rows = _read_state(state_path)
    completed = 0
    failed = 0
    skipped = 0
    pending: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        mechanism = str(condition.get("target_mechanism", "mixed"))
        for seed in seeds:
            valid_complete = _valid_completed_attempt(
                state_rows, condition_id=condition_id, seed=seed
            )
            if valid_complete is not None:
                skipped += 1
                continue
            attempt = _next_attempt(state_rows, condition_id, seed)
            # campaign identity already lives in the immutable parent manifest.
            # Keep the leaf short enough for Windows checkpoint/snapshot paths.
            run_id = f"{_slug(condition_id)}__s{seed}__a{attempt:02d}"
            run_config = _resolve_run_config(
                base_config,
                spec,
                condition,
                seed=seed,
                run_id=run_id,
                campaign_dir=campaign_dir,
            )
            run_dir = (campaign_dir / "runs" / run_id).resolve()
            running = _state_record(
                condition_id,
                mechanism,
                seed,
                attempt,
                "running",
                run_id,
                run_dir,
            )
            pending.append(
                {
                    "condition_id": condition_id,
                    "mechanism": mechanism,
                    "seed": seed,
                    "attempt": attempt,
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "run_config": run_config,
                    "running": running,
                }
            )

    def start_unit(unit: dict[str, Any]) -> None:
        running = unit["running"]
        _append_state(state_path, running)
        _append_jsonl(events_path, {"event": "run_started", **running})
        state_rows.append(running)

    def finish_unit(unit: dict[str, Any], future: Future[bool] | None = None) -> None:
        nonlocal completed, failed
        try:
            paper_eligible = (
                future.result()
                if future is not None
                else _execute_campaign_run(unit["run_config"], unit["run_dir"])
            )
            finished = _state_record(
                unit["condition_id"],
                unit["mechanism"],
                unit["seed"],
                unit["attempt"],
                "complete",
                unit["run_id"],
                unit["run_dir"],
                message=f"paper_eligible={str(paper_eligible).lower()}",
            )
            completed += 1
        except Exception as exc:  # campaign must preserve the rest of the matrix
            failure_class = _classify_failure(exc)
            failure_dir = campaign_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            failure_path = failure_dir / f"{unit['run_id']}.txt"
            if failure_path.exists():
                raise FileExistsError(
                    f"failure evidence path unexpectedly exists: {failure_path}"
                ) from exc
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            finished = _state_record(
                unit["condition_id"],
                unit["mechanism"],
                unit["seed"],
                unit["attempt"],
                "failed",
                unit["run_id"],
                unit["run_dir"],
                failure_class=failure_class,
                message=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
        _append_state(state_path, finished)
        _append_jsonl(events_path, {"event": "run_finished", **finished})
        state_rows.append(finished)

    max_parallel = int(spec.get("max_parallel_runs", 1))
    if max_parallel == 1:
        for unit in pending:
            start_unit(unit)
            finish_unit(unit)
    else:
        queue = iter(pending)
        active: dict[Future[bool], dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            for _ in range(min(max_parallel, len(pending))):
                unit = next(queue)
                start_unit(unit)
                future = executor.submit(
                    _execute_campaign_run, unit["run_config"], unit["run_dir"]
                )
                active[future] = unit
            while active:
                future = next(as_completed(active))
                unit = active.pop(future)
                finish_unit(unit, future)
                next_unit = next(queue, None)
                if next_unit is not None:
                    start_unit(next_unit)
                    next_future = executor.submit(
                        _execute_campaign_run,
                        next_unit["run_config"],
                        next_unit["run_dir"],
                    )
                    active[next_future] = next_unit

    aggregate_result = aggregate_campaign(campaign_dir)
    state_rows = _prefer_campaign_local_run_dirs(campaign_dir, _read_state(state_path))
    completed_runs = _completed_runs(state_rows)
    summary = {
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "state_path": str(state_path),
        "aggregate_path": aggregate_result["aggregate_path"],
        "completed_this_invocation": completed,
        "failed_this_invocation": failed,
        "skipped_valid_completed": skipped,
        "completed_matrix_units": len(completed_runs),
        "expected_matrix_units": len(conditions) * len(seeds),
        "aggregate_status": aggregate_result["status"],
        "max_parallel_runs": max_parallel,
    }
    _append_jsonl(events_path, {"event": "campaign_invocation_finished", **summary})
    return summary


def aggregate_campaign(campaign_dir: str | Path) -> dict[str, Any]:
    """Rebuild a versioned campaign aggregate without rerunning any experiment."""

    root = Path(campaign_dir).resolve()
    manifest_path = root / "campaign_manifest.json"
    state_path = root / "state.tsv"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ConfigError(f"campaign aggregate 缺少 manifest/state：{root}")
    manifest = _read_json(manifest_path)
    spec = manifest.get("campaign")
    base_config = manifest.get("base_config")
    if not isinstance(spec, dict) or not isinstance(base_config, dict):
        raise ConfigError("campaign manifest 缺少冻结的 campaign/base_config")
    state_rows = _prefer_campaign_local_run_dirs(root, _read_state(state_path))
    completed_runs = _completed_runs(state_rows)
    aggregate = _aggregate_campaign(spec, base_config, completed_runs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = root / f"campaign_aggregate_{stamp}.json"
    _write_new_json(output_path, aggregate)
    return {
        "campaign_dir": str(root),
        "aggregate_path": str(output_path),
        "status": aggregate.get("status"),
        "completed_run_count": len(completed_runs),
    }


def _prefer_campaign_local_run_dirs(
    campaign_dir: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Make copied campaign evidence portable without rewriting its source state.tsv."""

    localized: list[dict[str, str]] = []
    for row in rows:
        copy_row = dict(row)
        run_id = str(copy_row.get("run_id", ""))
        local_run = campaign_dir / "runs" / run_id
        if run_id and local_run.is_dir():
            copy_row["run_dir"] = str(local_run.resolve())
        localized.append(copy_row)
    return localized


def _read_campaign_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"campaign 配置不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("campaign"), dict):
        raise ConfigError("campaign 配置必须包含 campaign 映射")
    return data


def _validate_campaign_spec(spec: dict[str, Any], base: dict[str, Any]) -> None:
    for field in ("campaign_id", "design", "base_experiment_config", "seeds"):
        value = spec.get(field)
        if value is None or value == "" or value == []:
            raise ConfigError(f"campaign.{field} 不能为空")
    design = str(spec["design"])
    if design not in {"mixed_table", "target_vs_seven_no_memory"}:
        raise ConfigError(f"未知 campaign.design：{design}")
    seeds = [int(seed) for seed in spec["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ConfigError("campaign.seeds 不能重复")
    try:
        max_parallel = int(spec.get("max_parallel_runs", 1))
    except (TypeError, ValueError) as exc:
        raise ConfigError("campaign.max_parallel_runs 必须是正整数") from exc
    if max_parallel < 1 or max_parallel > 32:
        raise ConfigError("campaign.max_parallel_runs 必须在 1 到 32 之间")
    conditions = _conditions(spec)
    condition_ids = [str(item["condition_id"]) for item in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ConfigError("campaign condition_id 不能重复")
    if design == "target_vs_seven_no_memory":
        mechanisms = [str(item.get("target_mechanism", "")) for item in conditions]
        expected = {
            "no_memory",
            "fact",
            "expr",
            "fact_expr_sync",
            "fact_expr_async",
        }
        if set(mechanisms) != expected or len(mechanisms) != len(expected):
            raise ConfigError("Campaign E 必须恰好包含 NoMemory 与四种 memory target")
        baseline = str(spec.get("baseline_condition_id", "no_memory_target"))
        if baseline not in condition_ids:
            raise ConfigError("Campaign E baseline_condition_id 不在 conditions 中")
    run_mode = str(base.get("experiment", {}).get("run_mode", "smoke"))
    if run_mode == "formal" and len(seeds) < 2:
        raise ConfigError("formal campaign 至少需要 2 个预注册 seed")


def _conditions(spec: dict[str, Any]) -> list[dict[str, Any]]:
    configured = spec.get("conditions")
    if configured is None and str(spec.get("design")) == "mixed_table":
        return [{"condition_id": "mixed_table", "target_mechanism": "mixed"}]
    if not isinstance(configured, list) or not configured:
        raise ConfigError("campaign.conditions 必须是非空列表")
    if not all(isinstance(item, dict) and item.get("condition_id") for item in configured):
        raise ConfigError("每个 campaign condition 必须包含 condition_id")
    return [dict(item) for item in configured]


def _resolve_run_config(
    base: dict[str, Any],
    spec: dict[str, Any],
    condition: dict[str, Any],
    *,
    seed: int,
    run_id: str,
    campaign_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    experiment = config["experiment"]
    experiment.update(
        {
            "seed": int(seed),
            "run_id": run_id,
            "output_root": str((campaign_dir / "runs").resolve()),
            "campaign_id": str(spec["campaign_id"]),
            "campaign_design": str(spec["design"]),
            "campaign_condition_id": str(condition["condition_id"]),
            "protocol_label": str(spec.get("protocol_label", "unspecified")),
            "campaign_max_parallel_runs": int(spec.get("max_parallel_runs", 1)),
        }
    )
    # Embedding caches are mutable acceleration artifacts.  A campaign must not
    # reuse the base config's agent-only path across seeds or conditions.
    run_dir = (campaign_dir / "runs" / run_id).resolve()
    config["agent"] = {
        **dict(config.get("agent", {})),
        "embedding_cache_path": str(
            run_dir / "embedding_cache" / "{agent_id}.json"
        ),
    }
    if str(spec["design"]) == "target_vs_seven_no_memory":
        target_id = str(spec.get("target_agent_id", "target_00"))
        mechanism = str(condition["target_mechanism"])
        target_config = dict(condition.get("target_config", {}))
        experiment.update(
            {
                "table_size": 8,
                "target_agent_id": target_id,
                "evaluate_all_train_agents": False,
                "evaluation_target_ids": [target_id],
                "primary_estimand": TARGET_VS_NO_MEMORY_ESTIMAND,
                "primary_baseline_mechanism": "no_memory",
                "within_table_mechanism_aggregation": "single_target_condition",
                "cross_condition_aggregation": "paired_by_seed",
                "agent_roster": [
                    {
                        "agent_id": target_id,
                        "mechanism": mechanism,
                        **target_config,
                    },
                    *[
                        {
                            "agent_id": f"train_no_memory_{index:02d}",
                            "mechanism": "no_memory",
                        }
                        for index in range(1, 8)
                    ],
                ],
            }
        )
        config["agent"] = {
            **dict(config.get("agent", {})),
            "mechanism": mechanism,
            **target_config,
        }
    return config


def _execute_campaign_run(run_config: dict[str, Any], run_dir: Path) -> bool:
    """Execute one isolated leaf run; safe to call in a worker process."""

    result = run_resolved_config(run_config)
    if not _run_artifacts_valid(Path(run_dir)):
        raise RuntimeError("run returned without the required final artifacts")
    return bool(result.metrics.get("run_validity", {}).get("paper_eligible"))


def _aggregate_campaign(
    spec: dict[str, Any],
    base_config: dict[str, Any],
    completed_runs: list[dict[str, str]],
) -> dict[str, Any]:
    expected = len(_conditions(spec)) * len(spec["seeds"])
    manifests = [_read_json(Path(item["run_dir"]) / "manifest.json") for item in completed_runs]
    homogeneity = validate_runtime_homogeneity(manifests)
    if str(spec["design"]) == "mixed_table":
        metrics = [_read_json(Path(item["run_dir"]) / "metrics.json") for item in completed_runs]
        aggregate = aggregate_metrics(metrics)
        run_mode = str(base_config["experiment"].get("run_mode", "smoke"))
        if len(completed_runs) != expected:
            status = "incomplete_matrix"
        elif run_mode != "formal":
            status = "descriptive_only"
        else:
            status = aggregate.get("inference_status")
        if run_mode == "formal" and not homogeneity.get("formal_aggregation_allowed"):
            status = "blocked_runtime_heterogeneity"
        return {
            "schema_version": "agentmemeval_campaign_aggregate_v1",
            "design": "mixed_table",
            "status": status,
            "completed_run_count": len(completed_runs),
            "expected_run_count": expected,
            "runtime_homogeneity": homogeneity,
            "aggregate_metrics": aggregate,
        }
    return _aggregate_target_vs_no_memory(spec, base_config, completed_runs, homogeneity)


def _aggregate_target_vs_no_memory(
    spec: dict[str, Any],
    base_config: dict[str, Any],
    completed_runs: list[dict[str, str]],
    homogeneity: dict[str, Any],
) -> dict[str, Any]:
    condition_units: dict[str, dict[int, dict[str, Any]]] = {}
    paper_validity: list[bool] = []
    for state in completed_runs:
        run_dir = Path(state["run_dir"])
        checkpoint = _read_json(run_dir / "checkpoint_generalization.json")
        rows = list(checkpoint.get("results", []))
        if not rows:
            continue
        final_checkpoint = max(int(row["checkpoint_hand"]) for row in rows)
        target_id = str(spec.get("target_agent_id", "target_00"))
        matches = [
            row
            for row in rows
            if int(row["checkpoint_hand"]) == final_checkpoint
            and str(row["target_agent_id"]) == target_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Campaign E run must contribute exactly one final target row: {run_dir}"
            )
        row = matches[0]
        condition_units.setdefault(state["condition_id"], {})[int(state["seed"])] = {
            "mechanism": state["target_mechanism"],
            "final_test_bb_per_100": float(row["bb_per_100"]),
            "final_test_chip_per_hand": float(row["test_chip_per_hand"]),
            "train_bb_per_100": float(row["train_bb_per_100"]),
            "train_chip_per_hand": float(row["train_chip_per_hand"]),
            "generalization_gap_bb_per_100": float(
                row["generalization_gap_bb_per_100"]
            ),
        }
        metrics = _read_json(run_dir / "metrics.json")
        paper_validity.append(bool(metrics.get("run_validity", {}).get("paper_eligible")))

    baseline_id = str(spec.get("baseline_condition_id", "no_memory_target"))
    baseline = condition_units.get(baseline_id, {})
    endpoint = str(base_config["experiment"].get("primary_endpoint"))
    metric_names = (
        "final_test_bb_per_100",
        "final_test_chip_per_hand",
        "train_bb_per_100",
        "train_chip_per_hand",
        "generalization_gap_bb_per_100",
    )
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    run_mode = str(base_config["experiment"].get("run_mode", "smoke"))
    required = base_config["experiment"].get("required_seed_pairs")
    statistical_status = str(
        base_config["experiment"].get("statistical_plan_status", "")
    )
    inference_allowed = (
        run_mode == "formal"
        and homogeneity.get("formal_aggregation_allowed") is True
        and bool(paper_validity)
        and all(paper_validity)
        and statistical_status == "frozen"
        and isinstance(required, int)
    )
    for condition_id, values_by_seed in sorted(condition_units.items()):
        if condition_id == baseline_id:
            continue
        matched = sorted(set(values_by_seed) & set(baseline))
        summaries: dict[str, Any] = {}
        for metric in metric_names:
            differences = [
                float(values_by_seed[seed][metric]) - float(baseline[seed][metric])
                for seed in matched
            ]
            summaries[metric] = {
                "definition": "target condition minus NoMemory target on matched seed",
                "matched_seeds": matched,
                "effects": differences,
                "summary": summarize_values(differences),
            }
        comparisons[condition_id] = {
            "target_mechanism": next(iter(values_by_seed.values()))["mechanism"],
            "metrics": summaries,
        }
        primary_differences = summaries[endpoint]["effects"]
        if inference_allowed and primary_differences:
            raw_p[condition_id] = paired_sign_flip_p_value(primary_differences)
    adjusted = holm_adjust(raw_p) if raw_p else {}
    for condition_id, comparison in comparisons.items():
        comparison["primary_raw_p_value"] = raw_p.get(condition_id)
        comparison["primary_holm_adjusted_p_value"] = adjusted.get(condition_id)

    matched_counts = [
        len(comparison["metrics"][endpoint]["matched_seeds"])
        for comparison in comparisons.values()
    ]
    expected = len(_conditions(spec)) * len(spec["seeds"])
    if len(completed_runs) != expected:
        status = "incomplete_matrix"
    elif run_mode != "formal":
        status = "descriptive_only"
    elif not homogeneity.get("formal_aggregation_allowed"):
        status = "blocked_runtime_heterogeneity"
    elif not paper_validity or not all(paper_validity):
        status = "blocked_invalid_or_degenerate_run"
    elif statistical_status != "frozen" or not isinstance(required, int):
        status = "blocked_statistical_plan_not_frozen"
    elif not matched_counts or min(matched_counts) < required:
        status = "insufficient_preregistered_seed_pairs"
    else:
        status = "ready"
    return {
        "schema_version": "agentmemeval_campaign_aggregate_v1",
        "design": "target_vs_seven_no_memory",
        "estimand": TARGET_VS_NO_MEMORY_ESTIMAND,
        "independent_unit": "one target condition run within one seed",
        "baseline_condition_id": baseline_id,
        "primary_endpoint": endpoint,
        "multiple_comparison_method": "holm",
        "required_seed_pairs": required,
        "status": status,
        "completed_run_count": len(completed_runs),
        "expected_run_count": expected,
        "runtime_homogeneity": homogeneity,
        "condition_units": condition_units,
        "paired_comparisons": comparisons,
    }


def _ensure_state_file(path: Path) -> None:
    if path.exists():
        header = path.read_text(encoding="utf-8").splitlines()[:1]
        if not header or header[0].split("\t") != list(STATE_FIELDS):
            raise ConfigError(f"state.tsv schema mismatch: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_FIELDS, delimiter="\t")
        writer.writeheader()


def _append_state(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_FIELDS, delimiter="\t")
        writer.writerow({field: _tsv_safe(record.get(field, "")) for field in STATE_FIELDS})


def _read_state(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _state_record(
    condition_id: str,
    mechanism: str,
    seed: int,
    attempt: int,
    status: str,
    run_id: str,
    run_dir: Path,
    *,
    failure_class: str = "",
    message: str = "",
) -> dict[str, Any]:
    return {
        "event_utc": _utc_now(),
        "condition_id": condition_id,
        "target_mechanism": mechanism,
        "seed": int(seed),
        "attempt": int(attempt),
        "status": status,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "failure_class": failure_class,
        "message": message,
    }


def _next_attempt(rows: list[dict[str, str]], condition_id: str, seed: int) -> int:
    attempts = [
        int(row["attempt"])
        for row in rows
        if row["condition_id"] == condition_id and int(row["seed"]) == int(seed)
    ]
    return max(attempts, default=0) + 1


def _valid_completed_attempt(
    rows: list[dict[str, str]], *, condition_id: str, seed: int
) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row["condition_id"] == condition_id
        and int(row["seed"]) == int(seed)
        and row["status"] == "complete"
    ]
    for row in sorted(candidates, key=lambda item: int(item["attempt"]), reverse=True):
        if _run_artifacts_valid(Path(row["run_dir"])):
            return row
    return None


def _completed_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = sorted({(row["condition_id"], int(row["seed"])) for row in rows})
    completed = []
    for condition_id, seed in keys:
        row = _valid_completed_attempt(rows, condition_id=condition_id, seed=seed)
        if row is not None:
            completed.append(row)
    return completed


def _run_artifacts_valid(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
        for name in REQUIRED_RUN_ARTIFACTS
    )


def _classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, ConfigError):
        return "execution_invalid_config"
    if any(token in text for token in ("timeout", "connection", "oom", "cuda")):
        return "infrastructure_failure"
    return "execution_invalid"


def _verify_existing_manifest(path: Path, expected_hash: str) -> None:
    if not path.is_file():
        raise ConfigError(f"resume campaign 缺少 manifest：{path}")
    manifest = _read_json(path)
    if manifest.get("config_sha256") != expected_hash:
        raise ConfigError("resume campaign 配置哈希与 immutable manifest 不一致")


def _write_new_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖既有 JSON：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_internal_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_internal_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_without_internal_keys(item) for item in value]
    return value


def _json_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        raise ConfigError("campaign/run 标识清洗后为空")
    return cleaned


def _tsv_safe(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")[:1000]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
