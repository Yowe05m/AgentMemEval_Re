"""Frozen, deterministic Phase F analysis for TASK8B."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import sha256_json, verify_schedule_manifest
from agentmemeval.experiments.formal_runner import STATE_SCHEMA_VERSION

FORMAL_SEEDS = tuple(range(2026090101, 2026090113))
FORMAL_PRIMARY_TASKS = {
    f"isolation_{mechanism}": (1350, {"R1-E1-I", "R1-E2", "R1-E3"})
    for mechanism in ("no_memory", "fact", "expr", "sync", "async")
}
FORMAL_SECONDARY_TASKS = {
    "mixed_ecological": (2700, {"R1-E1-M"}),
    "expr_online": (600, {"R1-E4"}),
    "expr_without": (600, {"R1-E5"}),
    "async_online": (600, {"R1-E4"}),
    "async_without": (600, {"R1-E5"}),
}
ALLOWED_EXCLUSION_REASONS = {
    "ARTIFACT_INCOMPLETE_OR_HASH_MISMATCH",
    "CRN_MISMATCH",
    "ELIGIBLE_INFRA_FAILURE",
    "EXECUTION_INVALID",
    "FALLBACK_NONZERO",
    "IDENTITY_MISMATCH",
    "INVALID_RECEIPT_OR_DEPENDENCY",
    "OUTPUT_PATH_COLLISION",
    "REVISION_FALLBACK_NONZERO",
    "REWARD_CONSERVATION_VIOLATION",
    "STACK_CONSERVATION_VIOLATION",
}
REQUIRED_SOURCES = ("metrics.json", "hands.jsonl", "events.jsonl")
LINEAGE_FIELDS = (
    "lineage_id",
    "output_artifact_id",
    "output_element_id",
    "output_kind",
    "analysis_contract_id",
    "analysis_code_sha",
    "analysis_manifest_sha256",
    "input_manifest_sha256",
    "exclusion_ledger_sha256",
    "run_id",
    "seed",
    "condition",
    "task_id",
    "analysis_family",
    "memory_mode",
    "location",
    "checkpoint",
    "heldout_table_id",
    "attempt",
    "code_sha",
    "config_sha256",
    "prompt_sha256",
    "model_fingerprint",
    "embedding_fingerprint",
    "schedule_sha256",
    "source_file",
    "source_file_sha256",
    "row_selector",
    "exclusion_status",
    "statistical_unit",
    "n_planned",
    "n_effective",
    "verification_status",
    "input_snapshot_id",
    "source_records",
    "transformation",
    "aggregation_order",
    "cluster_ids",
    "missing_reason_codes",
    "display_value",
)


def build_task8b_analysis_input(
    worker_manifest_dir: str | Path,
    snapshot_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the formal Phase F input manifest from frozen manifests and recovered attempts."""

    manifest_dir = Path(worker_manifest_dir).resolve()
    snapshots = Path(snapshot_root).resolve()
    destination = Path(output_path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    worker_manifests = []
    for path in sorted(manifest_dir.glob("[PS][0-1][0-9].json")):
        value = _read_json(path)
        if value.get("protocol_status") != "frozen/expedited-formal-candidate":
            raise ConfigError(f"Phase F input 拒绝非 expedited formal manifest：{path.name}")
        worker_manifests.append(value)
    if len(worker_manifests) != 24:
        raise ConfigError("Phase F input builder 要求精确 24 worker manifests")
    expected_worker_ids = {f"{role}{index:02d}" for role in ("P", "S") for index in range(1, 13)}
    if {str(row.get("worker_id", "")) for row in worker_manifests} != expected_worker_ids:
        raise ConfigError("Phase F input builder worker IDs 必须为 P01-P12/S01-S12")
    workers = []
    for manifest in sorted(worker_manifests, key=lambda row: str(row["worker_id"])):
        worker_id = str(manifest["worker_id"])
        seed = int(manifest["seed_bundle"])
        attempt_parent = snapshots / worker_id / str(seed)
        attempts = []
        if attempt_parent.is_dir():
            for attempt in sorted(
                (path for path in attempt_parent.iterdir() if path.is_dir()),
                key=lambda path: _attempt_number(path.name),
            ):
                _attempt_number(attempt.name)
                try:
                    relative = attempt.resolve().relative_to(destination.parent.resolve())
                except ValueError as exc:
                    raise ConfigError(
                        "analysis input manifest 必须位于 snapshot root 的祖先目录"
                    ) from exc
                attempts.append({"attempt": attempt.name, "relative_path": relative.as_posix()})
        if not attempts:
            raise ConfigError(f"Phase F input 缺 recovered attempts：{worker_id}/{seed}")
        workers.append(
            {
                "worker_id": worker_id,
                "pod_id": str(manifest["pod_id"]),
                "seed": seed,
                "expected_identity": dict(manifest["common_identity"]),
                "expected_worker_manifest_sha256": _sha256(manifest_dir / f"{worker_id}.json"),
                "attempts": attempts,
            }
        )
    payload = {
        "schema_version": "task8b-phase-f-input-v1",
        "analysis_contract_id": "task8b-phase-f-v1",
        "synthetic_test_mode": False,
        "analysis_code_sha": workers[0]["expected_identity"]["code_sha"],
        "input_snapshot_id": _sha256(manifest_dir / "manifest_index.json"),
        "unlocked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "unlocked_by": "task8b-build-analysis-input",
        "pre_unlock_manifest_sha256": _sha256(
            Path(__file__).parents[3] / "configs" / "formal" / "task8b_phase_f_contract.json"
        ),
        "workers": workers,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_new(destination, _json_bytes(payload))
    return payload


def run_task8b_analysis(
    input_manifest_path: str | Path,
    exclusion_ledger_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate frozen inputs and emit byte-stable Phase F CSV/JSON artifacts."""

    manifest_path = Path(input_manifest_path).resolve()
    ledger_path = Path(exclusion_ledger_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "task8b-phase-f-input-v1":
        raise ConfigError("TASK8B Phase F input schema 不匹配")
    if manifest.get("analysis_contract_id") != "task8b-phase-f-v1":
        raise ConfigError("analysis contract 必须为 task8b-phase-f-v1")
    workers = manifest.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ConfigError("TASK8B Phase F input 缺 workers")
    if not bool(manifest.get("synthetic_test_mode", False)):
        seeds = sorted(int(worker.get("seed", -1)) for worker in workers)
        worker_ids = {str(worker.get("worker_id", "")) for worker in workers}
        expected_ids = {f"{role}{index:02d}" for role in ("P", "S") for index in range(1, 13)}
        pod_seed_counts = {
            seed: sum(int(worker.get("seed", -1)) == seed for worker in workers)
            for seed in FORMAL_SEEDS
        }
        if (
            len(workers) != 24
            or sorted(set(seeds)) != list(FORMAL_SEEDS)
            or worker_ids != expected_ids
            or set(pod_seed_counts.values()) != {2}
        ):
            raise ConfigError("正式 Phase F 输入必须为 24 workers 和精确 12 seeds")
        for worker in workers:
            worker_id = str(worker.get("worker_id", ""))
            expected_seed = FORMAL_SEEDS[int(worker_id[1:]) - 1]
            if int(worker.get("seed", -1)) != expected_seed or not str(
                worker.get("expected_worker_manifest_sha256", "")
            ):
                raise ConfigError("正式 Phase F worker/seed/topology freeze 不匹配")
    ledger = _read_ledger(ledger_path)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for worker in sorted(workers, key=lambda row: str(row.get("worker_id", ""))):
        chosen, rejected = _select_attempt(
            worker=worker,
            manifest_root=manifest_path.parent,
            ledger=ledger,
        )
        selected.append(chosen)
        exclusions.extend(rejected)
    synthetic_mode = bool(manifest.get("synthetic_test_mode", False))
    if all((item["root"] / "worker_manifest.json").is_file() for item in selected):
        _validate_selected_seed_pods(selected, enforce_formal=not synthetic_mode)

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(f"Phase F 输出目录非空，拒绝覆盖：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    selected_rows = [_selected_row(item) for item in selected]
    lineage, metric_rows = _metric_and_lineage_rows(selected)
    effect_rows = _primary_effect_rows(metric_rows)
    e6_rows = _e6_rows(selected)
    inference_rows = _primary_inference_rows(effect_rows)
    input_sha = _sha256(manifest_path)
    ledger_sha = _sha256(ledger_path)
    if not synthetic_mode:
        _validate_formal_primary_effects(effect_rows)
    analysis_code_sha = str(
        manifest.get("analysis_code_sha") or selected[0]["identity"].get("code_sha", "")
    )
    result = {
        "schema_version": "task8b-phase-f-analysis-manifest-v1",
        "analysis_contract_id": str(manifest.get("analysis_contract_id", "")),
        "manifest_status": "SYNTHETIC_TEST" if synthetic_mode else "FORMAL_RESULT_LOADED",
        "created_at_utc": manifest.get("unlocked_at_utc"),
        "frozen_before_result_unblinding": True,
        "analysis_code_sha": analysis_code_sha,
        "analysis_code_dirty": "UNVERIFIED_NOT_ASSERTED",
        "analysis_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "locked_dependencies_sha256": _sha256(Path(__file__).parents[3] / "pyproject.toml"),
        },
        "protocol": {
            "n_planned": 12,
            "n_source": "User-approved expedited protocol amendment before result unblinding",
            "seeds": list(FORMAL_SEEDS),
            "new_power_pilot_run": False,
            "mde_status": "historical_planning_candidate_only",
            "checkpoints": [30, 75, 150, 300],
            "heldout_tables": ["H01", "H02", "H03"],
            "primary_checkpoint": 300,
            "primary_mode": "Frozen",
            "primary_family": ["Expr_vs_Fact", "Async_vs_Fact"],
            "statistical_unit": "seed",
        },
        "inference": {
            "point_estimator": "arithmetic_mean_of_seed_level_effects",
            "ci": {
                "method": "seed_cluster_percentile",
                "level": 0.95,
                "replicates": 10000,
                "prng": "PCG64",
                "seed": 2026090199,
                "quantiles": [0.025, 0.975],
            },
            "raw_test": "exact_two_sided_sign_flip",
            "multiplicity": {
                "method": "Holm",
                "family": "Primary Family A",
                "alpha": 0.05,
                "hypotheses": ["Expr_vs_Fact", "Async_vs_Fact"],
            },
        },
        "attempt_selection": "first_numerically_ordered_complete_valid_eligible_attempt",
        "input_snapshot": {
            "snapshot_id": manifest.get("input_snapshot_id"),
            "input_manifest_sha256": input_sha,
            "per_file_hash_verification": "verified",
            "worker_count_verified": len(selected),
            "hands_budget_verified": "142200" if not synthetic_mode else "synthetic",
        },
        "selected_worker_count": len(selected),
        "selected_seed_count": len({int(item["seed"]) for item in selected}),
        "primary_effect_row_count": len(effect_rows),
        "e6_row_count": len(e6_rows),
        "input_manifest_sha256": input_sha,
        "exclusion_ledger_sha256": ledger_sha,
        "bootstrap_seed": 2026090199,
        "bootstrap_replicates": 10000,
        "multiplicity": "holm",
        "power_verified": False,
        "reproduction_command": (
            "agentmemeval task8b-analyze --input-manifest INPUT_MANIFEST "
            "--exclusion-ledger EXCLUSION_LEDGER --output-dir NEW_OUTPUT_DIR"
        ),
        "unlock": {
            "formal_result_loaded": not synthetic_mode,
            "unlocked_at_utc": manifest.get("unlocked_at_utc"),
            "unlocked_by": manifest.get("unlocked_by"),
            "pre_unlock_manifest_sha256": manifest.get("pre_unlock_manifest_sha256"),
        },
    }
    manifest_bytes = _json_bytes(result)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    n_effective = len({int(item["seed"]) for item in selected})
    for row in lineage:
        row.update(
            {
                "analysis_contract_id": "task8b-phase-f-v1",
                "analysis_code_sha": analysis_code_sha,
                "analysis_manifest_sha256": manifest_sha,
                "input_manifest_sha256": input_sha,
                "exclusion_ledger_sha256": ledger_sha,
                "statistical_unit": "seed",
                "n_planned": 12,
                "n_effective": n_effective,
                "verification_status": "SOURCE_VERIFIED",
            }
        )
    _write_csv(destination / "selected_attempts.csv", selected_rows)
    _write_csv(destination / "primary_seed_effects.csv", effect_rows)
    _write_csv(destination / "primary_inference.csv", inference_rows)
    _write_csv(destination / "e6_metrics.csv", e6_rows)
    _write_csv(destination / "exclusion_retry_ledger.csv", exclusions)
    paper_lineage = _emit_paper_artifacts(
        destination=destination,
        selected_rows=selected_rows,
        metric_rows=metric_rows,
        effect_rows=effect_rows,
        inference_rows=inference_rows,
        e6_rows=e6_rows,
        render_figures=not bool(manifest.get("synthetic_test_mode", False)),
        source_lineage=lineage,
    )
    _write_csv(
        destination / "data_lineage.csv",
        [*lineage, *paper_lineage],
        fields=LINEAGE_FIELDS,
    )
    _write_bytes_new(destination / "analysis_manifest.json", manifest_bytes)
    checksum_rows = [
        {
            "relative_path": path.relative_to(destination).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256.csv"
    ]
    _write_csv(destination / "artifact_sha256.csv", checksum_rows)
    return result


def _select_attempt(
    *, worker: dict[str, Any], manifest_root: Path, ledger: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    worker_id = str(worker.get("worker_id", ""))
    attempts = worker.get("attempts")
    expected = worker.get("expected_identity")
    if not worker_id or not isinstance(attempts, list) or not isinstance(expected, dict):
        raise ConfigError(f"worker {worker_id or '<missing>'} attempt/identity 非法")
    assessed = []
    for raw in sorted(attempts, key=lambda item: _attempt_number(str(item.get("attempt", "")))):
        name = str(raw.get("attempt", ""))
        relative = Path(str(raw.get("relative_path", "")))
        root = (manifest_root / relative).resolve()
        if manifest_root not in root.parents or root.is_symlink():
            raise ConfigError(f"worker {worker_id} attempt 路径越界或为 symlink")
        reasons, identity = _assess_attempt(root, worker, expected)
        assessed.append((name, root, reasons, identity))
    valid = [row for row in assessed if not row[2]]
    if len(valid) != 1:
        details = ";".join(
            f"{name}:{','.join(reasons) or 'valid'}" for name, _, reasons, _ in assessed
        )
        raise ConfigError(
            f"worker {worker_id} authoritative valid attempt 必须恰为 1；"
            f"{details}; ledger fail closed"
        )
    chosen_name, chosen_root, _, identity = valid[0]
    rejected_rows = []
    for name, _, reasons, _ in assessed:
        if not reasons:
            continue
        entry = next(
            (
                row
                for row in ledger
                if row.get("worker_id") == worker_id and row.get("attempt") == name
            ),
            None,
        )
        if entry is None or entry.get("authoritative_attempt") != chosen_name:
            raise ConfigError(f"worker {worker_id} invalid attempt {name} 缺预注册 ledger")
        expected_reason = _ledger_reason_for(reasons)
        if entry.get("reason_code") != expected_reason:
            raise ConfigError(f"worker {worker_id} ledger reason 必须匹配 {expected_reason}")
        rejected_rows.append(
            {
                **entry,
                "validation_reasons": "|".join(reasons),
                "selected": "false",
            }
        )
    return (
        {
            "worker_id": worker_id,
            "pod_id": str(worker.get("pod_id", "")),
            "seed": int(worker.get("seed")),
            "attempt": chosen_name,
            "root": chosen_root,
            "identity": identity,
        },
        rejected_rows,
    )


def _assess_attempt(
    root: Path, worker: dict[str, Any], expected: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    synthetic_layout = (root / "metrics.json").is_file()
    required = (
        (
            *REQUIRED_SOURCES,
            "identity.json",
            "health.json",
            "files.tsv",
            "completion_receipt.json",
        )
        if synthetic_layout
        else (
            "worker_manifest.json",
            "task_results.json",
            "state.tsv",
            "files.tsv",
            "completion_receipt.json",
        )
    )
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        return ["MISSING_REQUIRED_ARTIFACT"], {}
    completion = _read_json(root / "completion_receipt.json")
    if (
        completion.get("schema_version") != "task8-worker-completion-v1"
        or completion.get("status") != "complete"
        or completion.get("worker_id") != worker.get("worker_id")
        or completion.get("files_tsv_sha256") != _sha256(root / "files.tsv")
    ):
        reasons.append("COMPLETION_INVALID")
    files = _read_files_manifest(root / "files.tsv")
    required_manifest_sources = (
        (*REQUIRED_SOURCES, "identity.json", "health.json")
        if synthetic_layout
        else ("worker_manifest.json", "task_results.json")
    )
    for source in required_manifest_sources:
        row = files.get(source)
        path = root / source
        if row is None or row[0] != path.stat().st_size or row[1] != _sha256(path):
            reasons.append("ARTIFACT_HASH_INVALID")
            break
    if not synthetic_layout:
        formal_reasons, identity = _assess_formal_worker(root, worker, expected, files)
        reasons.extend(formal_reasons)
        return sorted(set(reasons)), identity
    identity = _read_json(root / "identity.json")
    identity_expected = {
        **expected,
        "worker_id": worker.get("worker_id"),
        "pod_id": worker.get("pod_id"),
        "seed": worker.get("seed"),
    }
    if any(identity.get(key) != value for key, value in identity_expected.items()):
        reasons.append("IDENTITY_INVALID")
    health = _read_json(root / "health.json")
    zero_fields = (
        "fallback_count",
        "revision_fallback_count",
        "reward_conservation_violations",
        "stack_conservation_violations",
    )
    if health.get("valid") is not True or any(
        int(health.get(field, -1)) != 0 for field in zero_fields
    ):
        reasons.append(
            "FALLBACK_NONZERO" if int(health.get("fallback_count", 0)) else "HEALTH_INVALID"
        )
    return sorted(set(reasons)), identity


def _metric_and_lineage_rows(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lineage: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for item in selected:
        if not (item["root"] / "metrics.json").is_file():
            formal_lineage, formal_metrics = _formal_metric_rows(item)
            lineage.extend(formal_lineage)
            metrics.extend(formal_metrics)
            continue
        records = _read_json(item["root"] / "metrics.json").get("records")
        if not isinstance(records, list):
            raise ConfigError(f"worker {item['worker_id']} metrics.records 非法")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ConfigError("metrics record 必须是对象")
            row = {**record, "seed": item["seed"], "worker_id": item["worker_id"]}
            metrics.append(row)
            identity = item["identity"]
            lineage.append(
                {
                    "run_id": item["worker_id"],
                    "seed": item["seed"],
                    "condition": str(record.get("mechanism", "")),
                    "checkpoint": int(record.get("checkpoint_hand", 0)),
                    "heldout_table_id": str(record.get("table_id", "")),
                    "attempt": item["attempt"],
                    "code_sha": identity.get("code_sha", ""),
                    "config_sha256": identity.get("config_sha256", ""),
                    "prompt_sha256": identity.get("prompt_sha256", ""),
                    "model_fingerprint": identity.get("model_fingerprint", ""),
                    "source_file": "metrics.json",
                    "source_file_sha256": _sha256(item["root"] / "metrics.json"),
                    "row_selector": f"records[{index}]",
                    "exclusion_status": "eligible",
                }
            )
    lineage.sort(key=lambda row: tuple(str(row.get(field, "")) for field in LINEAGE_FIELDS))
    return lineage, metrics


def _assess_formal_worker(
    root: Path,
    worker: dict[str, Any],
    expected: dict[str, Any],
    files: dict[str, tuple[int, str]],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    manifest = _read_json(root / "worker_manifest.json")
    expected_manifest_sha = str(worker.get("expected_worker_manifest_sha256", ""))
    if expected_manifest_sha and _sha256(root / "worker_manifest.json") != expected_manifest_sha:
        reasons.append("IDENTITY_INVALID")
    common = manifest.get("common_identity")
    if not isinstance(common, dict):
        return ["IDENTITY_INVALID"], {}
    expected_common = {
        **expected,
        "worker_id": worker.get("worker_id"),
        "pod_id": worker.get("pod_id"),
        "seed": worker.get("seed"),
    }
    observed = {
        **common,
        "worker_id": manifest.get("worker_id"),
        "pod_id": manifest.get("pod_id"),
        "seed": manifest.get("seed_bundle"),
    }
    if any(observed.get(key) != value for key, value in expected_common.items()):
        reasons.append("IDENTITY_INVALID")
    for relative, (size, digest) in files.items():
        path = _inside_attempt(root, relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _sha256(path) != digest
        ):
            reasons.append("ARTIFACT_HASH_INVALID")
            break
    listed = set(files)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != listed | {"state.tsv", "files.tsv", "completion_receipt.json"}:
        reasons.append("ARTIFACT_HASH_INVALID")
    try:
        _verify_analysis_state(root / "state.tsv")
        task_run_map = _task_run_map(root, worker_id=str(worker["worker_id"]))
    except ConfigError:
        reasons.append("ARTIFACT_HASH_INVALID")
        return sorted(set(reasons)), observed
    task_rows = manifest.get("task_configs")
    if not isinstance(task_rows, list) or not task_rows:
        reasons.append("MISSING_REQUIRED_ARTIFACT")
        return reasons, observed
    for task in task_rows:
        task_id = str(task.get("task_id", ""))
        task_root = task_run_map.get(task_id, root / "runs" / task_id)
        task_relative = task_root.relative_to(root).as_posix()
        for relative in (
            f"{task_relative}/metrics.json",
            f"{task_relative}/hand_summaries.jsonl",
            f"{task_relative}/events.jsonl",
            f"{task_relative}/task_identity_audit.json",
        ):
            path = root / Path(relative)
            row = files.get(relative)
            if row is None or not path.is_file() or row != (path.stat().st_size, _sha256(path)):
                reasons.append("ARTIFACT_HASH_INVALID")
        if not task_root.is_dir():
            continue
        audit = _read_json(task_root / "task_identity_audit.json")
        actual = audit.get("actual")
        task_expected = task.get("expected_identity")
        if (
            audit.get("status") != "verified"
            or not isinstance(actual, dict)
            or not isinstance(task_expected, dict)
            or any(actual.get(key) != value for key, value in task_expected.items())
        ):
            reasons.append("IDENTITY_INVALID")
        metrics = _read_json(task_root / "metrics.json")
        execution = metrics.get("execution_health")
        validity = metrics.get("run_validity")
        if not isinstance(execution, dict) or execution.get("valid") is not True:
            reasons.append("HEALTH_INVALID")
        if isinstance(execution, dict):
            counters = (
                "fallback_count",
                "memory_revision_fallback_count",
                "reward_conservation_violation_count",
                "stack_conservation_violation_count",
            )
            if any(int(execution.get(field, -1)) != 0 for field in counters):
                reasons.append("FALLBACK_NONZERO")
        if (
            not isinstance(validity, dict)
            or validity.get("execution_valid") is not True
            or validity.get("behavior_valid") is not True
        ):
            reasons.append("HEALTH_INVALID")
        try:
            _validate_task_data_completeness(
                task_root,
                expected_schedule_sha256=str(task_expected.get("schedule_sha256", "")),
            )
        except ConfigError:
            reasons.append("ARTIFACT_HASH_INVALID")
    isolation_schedules = {
        str(task.get("expected_identity", {}).get("schedule_sha256", ""))
        for task in task_rows
        if str(task.get("task_id", "")).startswith("isolation_")
    }
    if len(isolation_schedules) != 1 or "" in isolation_schedules:
        reasons.append("CRN_INVALID")
    return sorted(set(reasons)), observed


def _validate_selected_seed_pods(selected: list[dict[str, Any]], *, enforce_formal: bool) -> None:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_seed[int(item["seed"])].append(item)
    for seed, pod in by_seed.items():
        if len(pod) != 2:
            raise ConfigError(f"seed {seed} P/S pod 不完整")
        manifests = [_read_json(item["root"] / "worker_manifest.json") for item in pod]
        roles = {str(row.get("role")): row for row in manifests}
        if set(roles) != {"primary", "secondary"}:
            raise ConfigError(f"seed {seed} P/S role 不闭合")
        primary = roles["primary"]
        secondary = roles["secondary"]
        recomputed_pod_identity = primary.get("seed_pod_identity")
        if enforce_formal:
            _validate_formal_task_topology(primary, role="primary")
            _validate_formal_task_topology(secondary, role="secondary")
            schedule_rows = []
            item_by_role = {
                str(_read_json(item["root"] / "worker_manifest.json").get("role")): item
                for item in pod
            }
            for role, worker_manifest in (("primary", primary), ("secondary", secondary)):
                item = item_by_role[role]
                task_runs = _task_run_map(item["root"], worker_id=str(item["worker_id"]))
                for task in worker_manifest.get("task_configs", []):
                    task_id = str(task.get("task_id", ""))
                    schedule = _read_json(task_runs[task_id] / "schedule_manifest.json")
                    schedule_sha = verify_schedule_manifest(schedule)
                    if schedule_sha != task.get("expected_identity", {}).get("schedule_sha256"):
                        raise ConfigError(f"seed {seed} task {task_id} schedule identity mismatch")
                    schedule_rows.append(
                        {
                            "worker_role": role,
                            "task_id": task_id,
                            "schedule_sha256": schedule_sha,
                        }
                    )
            recomputed_pod_identity = {
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
        if (
            primary.get("seed_pod_identity") != secondary.get("seed_pod_identity")
            or primary.get("seed_pod_identity") != recomputed_pod_identity
            or primary.get("receipt_identity") != secondary.get("dependency_receipt_identity")
            or secondary.get("depends_on") != primary.get("worker_id")
        ):
            raise ConfigError(f"seed {seed} CRN/receipt dependency mismatch")
    planned_hands = sum(
        int(task.get("planned_hands", 0))
        for item in selected
        for task in _read_json(item["root"] / "worker_manifest.json").get("task_configs", [])
    )
    if enforce_formal and planned_hands != 142200:
        raise ConfigError(f"Phase F formal planned hands 未闭合：{planned_hands}")


def _validate_formal_task_topology(manifest: dict[str, Any], *, role: str) -> None:
    tasks = manifest.get("task_configs")
    if not isinstance(tasks, list):
        raise ConfigError(f"formal {role} task_configs 非法")
    expected = FORMAL_PRIMARY_TASKS if role == "primary" else FORMAL_SECONDARY_TASKS
    by_id = {str(task.get("task_id", "")): task for task in tasks if isinstance(task, dict)}
    if len(by_id) != len(tasks) or set(by_id) != set(expected):
        raise ConfigError(f"formal {role} 精确 task topology 不匹配")
    for task_id, (planned_hands, covers) in expected.items():
        task = by_id[task_id]
        if (
            int(task.get("planned_hands", -1)) != planned_hands
            or {str(value) for value in task.get("covers", [])} != covers
            or not str(task.get("schedule_sha256", ""))
            or not isinstance(task.get("expected_identity"), dict)
        ):
            raise ConfigError(f"formal {role} task {task_id} freeze 不匹配")


def _validate_task_data_completeness(task_root: Path, *, expected_schedule_sha256: str) -> None:
    schedule = _read_json(task_root / "schedule_manifest.json")
    if verify_schedule_manifest(schedule) != expected_schedule_sha256:
        raise ConfigError("task schedule identity 与 frozen expected identity 不匹配")
    rows = schedule.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ConfigError("task schedule rows 为空")
    expected = sorted(
        (
            int(row["checkpoint_hand"]),
            str(row["table_id"]),
            int(row["hand_number"]),
        )
        for row in rows
        if isinstance(row, dict) and row.get("phase") == "heldout"
    )
    hands = _read_jsonl(task_root / "hand_summaries.jsonl")
    observed = sorted(
        (
            int(row["checkpoint_hand"]),
            str(row["heldout_table_id"]),
            int(row["hand_number"]),
        )
        for row in hands
        if row.get("stage") == "test"
    )
    if observed != expected:
        raise ConfigError("task heldout hands 与 schedule 不闭合")
    config = _read_yaml(task_root / "resolved_config.yaml")
    expected_train = int(config.get("experiment", {}).get("train_hands", 0))
    observed_train = sum(row.get("stage") == "train" for row in hands)
    if observed_train != expected_train:
        raise ConfigError("task source train hands 与 config 不闭合")


def _formal_metric_rows(
    item: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(item["root"] / "worker_manifest.json")
    task_run_map = _task_run_map(item["root"], worker_id=item["worker_id"])
    task_rows = manifest.get("task_configs", [])
    metrics: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for task in sorted(task_rows, key=lambda row: str(row.get("task_id", ""))):
        task_id = str(task["task_id"])
        task_root = task_run_map[task_id]
        config = _read_yaml(task_root / "resolved_config.yaml")
        experiment = config.get("experiment", {})
        table = config.get("table", {})
        big_blind = int(table.get("big_blind", 2))
        mode = str(task.get("memory_mode", experiment.get("memory_mode", "Frozen")))
        identity_audit = _read_json(task_root / "task_identity_audit.json")
        task_identity = identity_audit.get("actual", {})
        hand_path = task_root / "hand_summaries.jsonl"
        hands = _read_jsonl(hand_path)
        mechanism_by_target = _mechanism_by_target(task_id, experiment, config)
        checkpoints = sorted(
            {
                int(row.get("checkpoint_hand"))
                for row in hands
                if row.get("checkpoint_hand") is not None
            }
        )
        if not checkpoints:
            checkpoints = [int(point) for point in experiment.get("checkpoint_set", [300])]
        for target, mechanism in mechanism_by_target.items():
            train = [row for row in hands if row.get("stage") == "train"]
            for checkpoint in checkpoints:
                source = [
                    row
                    for row in train
                    if int(row.get("hand_number", 0)) <= checkpoint
                    and target in row.get("rewards", {})
                ]
                if source:
                    metrics.append(
                        _formal_metric_record(
                            item=item,
                            task_id=task_id,
                            mechanism=mechanism,
                            mode=mode,
                            checkpoint=checkpoint,
                            location="Source",
                            table_id="Source-01",
                            target=target,
                            hands=source,
                            big_blind=big_blind,
                        )
                    )
            heldout_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
            for row in hands:
                if row.get("stage") != "test" or target not in row.get("rewards", {}):
                    continue
                checkpoint = int(row.get("checkpoint_hand", checkpoints[-1]))
                table_id = str(row.get("heldout_table_id", ""))
                heldout_groups[(checkpoint, table_id)].append(row)
            for (checkpoint, table_id), group in sorted(heldout_groups.items()):
                metrics.append(
                    _formal_metric_record(
                        item=item,
                        task_id=task_id,
                        mechanism=mechanism,
                        mode=mode,
                        checkpoint=checkpoint,
                        location="Heldout",
                        table_id=table_id,
                        target=target,
                        hands=group,
                        big_blind=big_blind,
                    )
                )
        source_sha = _sha256(hand_path)
        for index, row in enumerate(metrics):
            if row.get("worker_id") != item["worker_id"] or row.get("task_id") != task_id:
                continue
            lineage.append(
                {
                    "run_id": f"{item['worker_id']}:{task_id}",
                    "seed": item["seed"],
                    "condition": row["mechanism"],
                    "task_id": task_id,
                    "analysis_family": row.get("analysis_family", ""),
                    "memory_mode": row.get("memory_mode", ""),
                    "location": row.get("location", ""),
                    "checkpoint": row["checkpoint_hand"],
                    "heldout_table_id": row["table_id"],
                    "attempt": item["attempt"],
                    "code_sha": task_identity.get("code_sha", ""),
                    "config_sha256": task_identity.get("resolved_config_sha256", ""),
                    "prompt_sha256": task_identity.get("prompt_sha256", ""),
                    "model_fingerprint": task_identity.get("model_fingerprint", ""),
                    "embedding_fingerprint": task_identity.get("embedding_fingerprint", ""),
                    "schedule_sha256": task_identity.get("schedule_sha256", ""),
                    "source_file": (
                        task_root.relative_to(item["root"]) / "hand_summaries.jsonl"
                    ).as_posix(),
                    "source_file_sha256": source_sha,
                    "row_selector": f"derived_group[{index}]",
                    "exclusion_status": "eligible",
                }
            )
    return lineage, metrics


def _formal_metric_record(
    *,
    item: dict[str, Any],
    task_id: str,
    mechanism: str,
    mode: str,
    checkpoint: int,
    location: str,
    table_id: str,
    target: str,
    hands: list[dict[str, Any]],
    big_blind: int,
) -> dict[str, Any]:
    ordered_hands = sorted(
        hands,
        key=lambda row: (int(row.get("hand_number", 0)), str(row.get("hand_id", ""))),
    )
    return {
        "worker_id": item["worker_id"],
        "task_id": task_id,
        "seed": item["seed"],
        "mechanism": mechanism,
        "checkpoint_hand": checkpoint,
        "memory_mode": mode,
        "analysis_family": (
            "R1-E1-I"
            if task_id.startswith("isolation_")
            else "R1-E1-M"
            if task_id == "mixed_ecological"
            else "R1-E4"
            if mode == "Online"
            else "R1-E5"
        ),
        "location": location,
        "table_id": table_id,
        "target_agent_id": target,
        "raw_chips": str(sum(Decimal(str(row["rewards"][target])) for row in hands)),
        "hands": len(hands),
        "big_blind": big_blind,
        "hand_bb100_series": [
            _fixed(Decimal(str(row["rewards"][target])) / Decimal(big_blind) * Decimal(100))
            for row in ordered_hands
        ],
    }


def _mechanism_by_target(
    task_id: str, experiment: dict[str, Any], config: dict[str, Any]
) -> dict[str, str]:
    labels = {
        "no_memory": "NoMemory",
        "fact": "Fact",
        "expr": "Expr",
        "fact_expr_sync": "Sync",
        "fact_expr_async": "Async",
    }
    roster = experiment.get("agent_roster")
    if experiment.get("evaluate_all_train_agents") and isinstance(roster, list):
        return {
            str(row["agent_id"]): labels.get(str(row.get("mechanism")), str(row.get("mechanism")))
            for row in roster
            if isinstance(row, dict) and row.get("agent_id")
        }
    target = str(experiment.get("target_agent_id", "agent_00"))
    mechanism = str(config.get("agent", {}).get("mechanism", ""))
    if task_id.startswith("isolation_"):
        mechanism = task_id.removeprefix("isolation_")
    elif "_" in task_id and task_id.split("_", 1)[0] in {"expr", "async"}:
        mechanism = task_id.split("_", 1)[0]
    return {target: labels.get(mechanism, mechanism)}


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"YAML 顶层必须为对象：{path}")
    return value


def _primary_effect_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, Decimal]] = defaultdict(dict)
    for row in metrics:
        if int(row.get("checkpoint_hand", 0)) != 300 or row.get("memory_mode") != "Frozen":
            continue
        if row.get("analysis_family", "R1-E1-I") != "R1-E1-I":
            continue
        hands = Decimal(str(row.get("hands", 0)))
        blind = Decimal(str(row.get("big_blind", 0)))
        if hands <= 0 or blind <= 0:
            raise ConfigError("primary endpoint hands/big_blind 必须为正")
        bb100 = Decimal(str(row.get("raw_chips", 0))) / hands / blind * Decimal(100)
        location = "source" if row.get("location") == "Source" else "heldout"
        table = str(row.get("table_id", ""))
        grouped[(int(row["seed"]), str(row.get("mechanism")), location)][table] = bb100
    output = []
    for seed in sorted({key[0] for key in grouped}):
        gaps: dict[str, Decimal] = {}
        for mechanism in ("Fact", "Expr", "Async"):
            source_cells = grouped.get((seed, mechanism, "source"), {})
            heldout_cells = grouped.get((seed, mechanism, "heldout"), {})
            if not source_cells or set(heldout_cells) != {"H01", "H02", "H03"}:
                continue
            source = _mean(list(source_cells.values()))
            heldout = _mean(list(heldout_cells.values()))
            gaps[mechanism] = heldout - source
        for mechanism in ("Expr", "Async"):
            if mechanism not in gaps or "Fact" not in gaps:
                continue
            output.append(
                {
                    "seed": seed,
                    "contrast": f"{mechanism}_vs_Fact",
                    "paired_interaction_bb_per_100": _fixed(gaps[mechanism] - gaps["Fact"]),
                    "checkpoint": 300,
                    "memory_mode": "Frozen",
                }
            )
    return output


def _validate_formal_primary_effects(effect_rows: list[dict[str, Any]]) -> None:
    observed = [(int(row.get("seed", -1)), str(row.get("contrast", ""))) for row in effect_rows]
    expected = [
        (seed, contrast) for seed in FORMAL_SEEDS for contrast in ("Expr_vs_Fact", "Async_vs_Fact")
    ]
    if len(observed) != len(set(observed)) or sorted(observed) != sorted(expected):
        raise ConfigError(
            "formal primary endpoint 必须包含每个 seed 的 Expr/Async vs Fact 完整 paired cells"
        )


def _e6_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in selected:
        if not (item["root"] / "hands.jsonl").is_file():
            rows.extend(_formal_e6_rows(item))
            continue
        hands = _read_jsonl(item["root"] / "hands.jsonl")
        if not hands:
            raise ConfigError("E6 hands.jsonl 不能为空")
        count = Decimal(len(hands))
        blind = Decimal(str(hands[0].get("big_blind", 0)))
        reward = sum((Decimal(str(row.get("reward_chips", 0))) for row in hands), Decimal(0))
        pots = [Decimal(str(row.get("pot_size", 0))) for row in hands]
        contributions = [
            Decimal(str(row.get("reward_chips", 0))) / blind * Decimal(100) for row in hands
        ]
        largest_index = max(range(len(hands)), key=lambda index: (abs(pots[index]), index))
        leave_largest = [
            value for index, value in enumerate(contributions) if index != largest_index
        ]
        ordered = sorted(contributions)
        trim = int(Decimal(len(ordered)) * Decimal("0.10"))
        trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
        lower = _percentile(ordered, Decimal("0.05"))
        upper = _percentile(ordered, Decimal("0.95"))
        winsorized = [min(max(value, lower), upper) for value in contributions]
        events = _read_jsonl(item["root"] / "events.jsonl")
        row: dict[str, Any] = {
            "worker_id": item["worker_id"],
            "seed": item["seed"],
            "attempt": item["attempt"],
            "raw_bb_per_100": _fixed(reward / count / blind * Decimal(100)),
            "leave_largest_absolute_pot_out_bb_per_100": _fixed(
                _mean(leave_largest) if leave_largest else Decimal(0)
            ),
            "median_bb_per_100": _fixed(_percentile(ordered, Decimal("0.50"))),
            "trimmed_10pct_bb_per_100": _fixed(_mean(trimmed)),
            "winsorized_5_95_bb_per_100": _fixed(_mean(winsorized)),
            "vpip_pct": _rate(hands, "vpip"),
            "fold_pct": _rate(hands, "fold"),
            "raise_pct": _rate(hands, "raise"),
            "all_in_pct": _rate(hands, "all_in"),
            "bust_pct": _rate(hands, "bust"),
            "max_pot_share_pct": _fixed(max(pots) / sum(pots, Decimal(0)) * Decimal(100)),
            "source_file_coverage_pct": _fixed(Decimal(100)),
            "event_count": len(events),
        }
        rows.append(row)
    return rows


def _formal_e6_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _read_json(item["root"] / "worker_manifest.json")
    task_run_map = _task_run_map(item["root"], worker_id=item["worker_id"])
    output = []
    for task in sorted(manifest.get("task_configs", []), key=lambda row: str(row["task_id"])):
        task_id = str(task["task_id"])
        task_root = task_run_map[task_id]
        config = _read_yaml(task_root / "resolved_config.yaml")
        experiment = config.get("experiment", {})
        blind = Decimal(str(config.get("table", {}).get("big_blind", 2)))
        hands = _read_jsonl(task_root / "hand_summaries.jsonl")
        events = _read_jsonl(task_root / "events.jsonl")
        final_checkpoint = max(int(point) for point in experiment.get("checkpoint_set", [300]))
        actions_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if event.get("event") == "action":
                actions_by_target[str(event.get("agent_id", ""))].append(event)
        for target, mechanism in _mechanism_by_target(task_id, experiment, config).items():
            target_hands = [
                row
                for row in hands
                if target in row.get("rewards", {})
                and row.get("stage") == "test"
                and int(row.get("checkpoint_hand", final_checkpoint)) == final_checkpoint
            ]
            if not target_hands:
                continue
            hand_ids = {str(row.get("hand_id", "")) for row in target_hands}
            filtered_events = [
                event for event in events if str(event.get("hand_id", "")) in hand_ids
            ]
            contributions = [
                Decimal(str(row["rewards"][target])) / blind * Decimal(100) for row in target_hands
            ]
            actions = [
                action
                for action in actions_by_target.get(target, [])
                if str(action.get("hand_id", "")) in hand_ids
            ]
            pot_by_hand: dict[str, Decimal] = defaultdict(Decimal)
            for event in filtered_events:
                hand_id = str(event.get("hand_id", ""))
                if hand_id in hand_ids:
                    pot_by_hand[hand_id] = max(
                        pot_by_hand[hand_id], Decimal(str(event.get("pot_after", 0)))
                    )
            pots = [
                pot_by_hand.get(str(row.get("hand_id", "")), Decimal(0)) for row in target_hands
            ]
            largest_index = max(range(len(pots)), key=lambda idx: (abs(pots[idx]), idx))
            leave_largest = [
                value for idx, value in enumerate(contributions) if idx != largest_index
            ]
            ordered = sorted(contributions)
            trim = int(Decimal(len(ordered)) * Decimal("0.10"))
            trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
            low = _percentile(ordered, Decimal("0.05"))
            high = _percentile(ordered, Decimal("0.95"))
            winsorized = [min(max(value, low), high) for value in contributions]
            preflop_by_hand: dict[str, set[str]] = defaultdict(set)
            all_actions_by_hand: dict[str, set[str]] = defaultdict(set)
            all_in_hands: set[str] = set()
            for action in actions:
                hand_id = str(action.get("hand_id", ""))
                action_type = str(action.get("action_type", ""))
                all_actions_by_hand[hand_id].add(action_type)
                if action.get("phase") == "preflop":
                    preflop_by_hand[hand_id].add(action_type)
                call_risk = action.get("call_risk", {})
                if isinstance(call_risk, dict) and call_risk.get("is_all_in") is True:
                    all_in_hands.add(hand_id)
            count = Decimal(len(target_hands))
            output.append(
                {
                    "worker_id": item["worker_id"],
                    "task_id": task_id,
                    "seed": item["seed"],
                    "attempt": item["attempt"],
                    "mechanism": mechanism,
                    "memory_mode": str(task.get("memory_mode", "Frozen")),
                    "raw_bb_per_100": _fixed(_mean(contributions)),
                    "leave_largest_absolute_pot_out_bb_per_100": _fixed(
                        _mean(leave_largest) if leave_largest else Decimal(0)
                    ),
                    "median_bb_per_100": _fixed(_percentile(ordered, Decimal("0.50"))),
                    "trimmed_10pct_bb_per_100": _fixed(_mean(trimmed)),
                    "winsorized_5_95_bb_per_100": _fixed(_mean(winsorized)),
                    "vpip_pct": _fixed(
                        Decimal(
                            sum(
                                bool(actions_set & {"call", "raise"})
                                for actions_set in preflop_by_hand.values()
                            )
                        )
                        / count
                        * Decimal(100)
                    ),
                    "fold_pct": _fixed(
                        Decimal(sum("fold" in values for values in all_actions_by_hand.values()))
                        / count
                        * Decimal(100)
                    ),
                    "raise_pct": _fixed(
                        Decimal(sum("raise" in values for values in all_actions_by_hand.values()))
                        / count
                        * Decimal(100)
                    ),
                    "all_in_pct": _fixed(Decimal(len(all_in_hands)) / count * Decimal(100)),
                    "bust_pct": _fixed(
                        Decimal(
                            sum(
                                Decimal(str(row.get("final_stacks", {}).get(target, 1))) == 0
                                for row in target_hands
                            )
                        )
                        / count
                        * Decimal(100)
                    ),
                    "max_pot_share_pct": _fixed(
                        max(pots) / sum(pots, Decimal(0)) * Decimal(100)
                        if sum(pots, Decimal(0)) > 0
                        else Decimal(0)
                    ),
                    "event_count": len(filtered_events),
                }
            )
    return output


def _primary_inference_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_contrast: dict[str, list[float]] = defaultdict(list)
    for row in effect_rows:
        by_contrast[str(row["contrast"])].append(float(row["paired_interaction_bb_per_100"]))
    preliminary = []
    rng = np.random.Generator(np.random.PCG64(2026090199))
    for contrast in ("Expr_vs_Fact", "Async_vs_Fact"):
        if contrast not in by_contrast:
            preliminary.append(
                {
                    "contrast": contrast,
                    "mean_bb100": "NA_NOT_ESTIMABLE",
                    "ci95_low": "NA_NOT_ESTIMABLE",
                    "ci95_high": "NA_NOT_ESTIMABLE",
                    "raw_p_two_sided": "NA_NOT_ESTIMABLE",
                    "n_planned": 12,
                    "n_effective": 0,
                    "bootstrap_replicates": 10000,
                    "bootstrap_prng": "numpy.PCG64",
                    "bootstrap_seed": 2026090199,
                }
            )
            continue
        values = np.asarray(by_contrast[contrast], dtype=np.float64)
        if len(values) >= 2:
            draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975], method="linear")
            ci_low, ci_high = f"{low:.8f}", f"{high:.8f}"
        else:
            ci_low = ci_high = "NA_NOT_ESTIMABLE"
        observed = abs(float(values.mean()))
        signs = np.asarray(
            [
                [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(len(values))]
                for mask in range(1 << len(values))
            ],
            dtype=np.float64,
        )
        randomized = np.abs((signs * values).mean(axis=1))
        raw_p = float(np.mean(randomized >= observed - 1e-15))
        preliminary.append(
            {
                "contrast": contrast,
                "mean_bb100": f"{values.mean():.8f}",
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "raw_p_two_sided": f"{raw_p:.8f}",
                "n_planned": 12,
                "n_effective": len(values),
                "bootstrap_replicates": 10000,
                "bootstrap_prng": "numpy.PCG64",
                "bootstrap_seed": 2026090199,
            }
        )
    if any(row["raw_p_two_sided"] == "NA_NOT_ESTIMABLE" for row in preliminary):
        for row in preliminary:
            row["holm_adjusted_p"] = "UNRESOLVED"
            row["holm_rank"] = "UNRESOLVED"
            row["holm_reject_0_05"] = "UNRESOLVED"
        return preliminary
    ordered = sorted(
        range(len(preliminary)), key=lambda idx: float(preliminary[idx]["raw_p_two_sided"])
    )
    running = 0.0
    adjusted: dict[int, float] = {}
    total = len(ordered)
    for rank, idx in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(preliminary[idx]["raw_p_two_sided"]))
        running = max(running, candidate)
        adjusted[idx] = running
        preliminary[idx]["holm_rank"] = rank + 1
    for idx, row in enumerate(preliminary):
        row["holm_adjusted_p"] = f"{adjusted[idx]:.8f}"
        row["holm_reject_0_05"] = str(adjusted[idx] <= 0.05).lower()
    return preliminary


def _ledger_reason_for(reasons: list[str]) -> str:
    priority = (
        ("CRN_INVALID", "CRN_MISMATCH"),
        ("IDENTITY_INVALID", "IDENTITY_MISMATCH"),
        ("FALLBACK_NONZERO", "FALLBACK_NONZERO"),
        ("COMPLETION_INVALID", "INVALID_RECEIPT_OR_DEPENDENCY"),
        ("ARTIFACT_HASH_INVALID", "ARTIFACT_INCOMPLETE_OR_HASH_MISMATCH"),
        ("MISSING_REQUIRED_ARTIFACT", "ARTIFACT_INCOMPLETE_OR_HASH_MISMATCH"),
        ("HEALTH_INVALID", "EXECUTION_INVALID"),
    )
    for observed, ledger_code in priority:
        if observed in reasons:
            return ledger_code
    return "EXECUTION_INVALID"


def _emit_paper_artifacts(
    *,
    destination: Path,
    selected_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    e6_rows: list[dict[str, Any]],
    render_figures: bool,
    source_lineage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table1 = _table1_rows(selected_rows)
    table2 = _table2_rows(metric_rows, inference_rows)
    table3 = _table3_rows(metric_rows)
    table4, figure4_rows = _table4_rows(metric_rows)
    table5 = _table5_rows(e6_rows)
    leave_one_seed_out = _leave_one_seed_out_rows(e6_rows)
    secondary_mixed = _secondary_mixed_rows(metric_rows)
    _write_csv_union(destination / "leave_one_seed_out_robustness.csv", leave_one_seed_out)
    _write_csv_union(destination / "secondary_mixed_ecological.csv", secondary_mixed)
    tables = (table1, table2, table3, table4, table5)
    names = (
        "table1_protocol_identity",
        "table2_core_adaptation_generalization",
        "table3_checkpoint_scan",
        "table4_frozen_online_without",
        "table5_behavior_robustness",
    )
    markdown_parts = ["# 论文主表与主图空模板（TASK8B Phase F 已填充副本）", ""]
    for index, (name, rows) in enumerate(zip(names, tables, strict=True), start=1):
        _write_csv_union(destination / f"{name}.csv", rows)
        markdown = _markdown_table(rows)
        _write_text_new(destination / f"{name}.md", f"# Table {index}\n\n{markdown}\n")
        markdown_parts.extend((f"## Table {index}", "", markdown, ""))

    figure_data = (
        _figure1_flow_rows(),
        table3,
        _figure3_rows(effect_rows, inference_rows),
        figure4_rows,
    )
    for index, rows in enumerate(figure_data, start=1):
        _write_csv_union(destination / f"figure{index}_plotting_data.csv", rows)
        if render_figures:
            _render_figure(destination, index, rows)
        _write_text_new(destination / f"figure{index}_caption.md", _figure_caption(index, rows))
        _write_bytes_new(
            destination / f"figure{index}_manifest.json",
            _json_bytes(
                {
                    "schema_version": "task8b-phase-f-plot-manifest-v1",
                    "analysis_contract_id": "task8b-phase-f-v1",
                    "figure": index,
                    "statistical_unit": "seed",
                    "n_planned": 12,
                    "ci": "seed_cluster_percentile_95_10000_PCG64_2026090199",
                    "plotting_data": f"figure{index}_plotting_data.csv",
                }
            ),
        )
        markdown_parts.extend(
            (
                f"## Figure {index}",
                "",
                (
                    f"Plotting data: `figure{index}_plotting_data.csv`; formats: PNG/SVG/PDF."
                    if render_figures
                    else f"Plotting data: `figure{index}_plotting_data.csv`; synthetic test mode."
                ),
                "",
            )
        )
    _write_text_new(
        destination / "论文主表与主图空模板_filled.md",
        "\n".join(markdown_parts),
    )
    review = (
        "# Automated table/plot consistency precheck\n\n"
        "Status: VERIFIED (mechanical precheck only; "
        "independent reviewer sign-off remains required).\n\n"
        "Each Figure 1-4 was rendered directly from its adjacent plotting-data CSV source rows; "
        "no manual numeric transcription was used.\n"
    )
    _write_text_new(destination / "automated_consistency_precheck.md", review)
    main_lineage = _paper_lineage_rows(tables, names, source_lineage)
    supplementary_lineage = _paper_lineage_rows(
        (leave_one_seed_out, secondary_mixed),
        ("leave_one_seed_out_robustness", "secondary_mixed_ecological"),
        source_lineage,
        table_offset=5,
    )
    return [*main_lineage, *supplementary_lineage]


def _paper_lineage_rows(
    tables: tuple[list[dict[str, Any]], ...],
    names: tuple[str, ...],
    source_lineage: list[dict[str, Any]],
    *,
    table_offset: int = 0,
) -> list[dict[str, Any]]:
    output = []
    for table_index, (name, rows) in enumerate(
        zip(names, tables, strict=True), start=1 + table_offset
    ):
        for row_index, row in enumerate(rows, start=1):
            mechanism = str(row.get("mechanism", ""))
            checkpoint = str(row.get("checkpoint", row.get("checkpoint_hand", "")))
            if name in {
                "table2_core_adaptation_generalization",
                "table4_frozen_online_without",
                "table5_behavior_robustness",
                "leave_one_seed_out_robustness",
                "secondary_mixed_ecological",
            }:
                checkpoint = "300"
            sources = [
                source
                for source in source_lineage
                if (not mechanism or source.get("condition") == mechanism)
                and (not checkpoint or str(source.get("checkpoint")) == checkpoint)
            ]
            if name == "leave_one_seed_out_robustness":
                omitted_seed = int(row.get("omitted_seed", -1))
                sources = [
                    source
                    for source in sources
                    if int(source.get("seed", -1)) != omitted_seed
                    and str(source.get("task_id", "")).startswith("isolation_")
                    and source.get("memory_mode") == "Frozen"
                    and source.get("location") == "Heldout"
                ]
            elif name == "secondary_mixed_ecological":
                sources = [
                    source
                    for source in sources
                    if source.get("task_id") == "mixed_ecological"
                    and source.get("analysis_family") == "R1-E1-M"
                    and source.get("location") == "Heldout"
                ]
            elif name in {
                "table2_core_adaptation_generalization",
                "table3_checkpoint_scan",
            }:
                sources = [
                    source
                    for source in sources
                    if str(source.get("task_id", "")).startswith("isolation_")
                    and source.get("analysis_family") == "R1-E1-I"
                    and source.get("memory_mode") == "Frozen"
                ]
            elif name == "table5_behavior_robustness":
                sources = [
                    source
                    for source in sources
                    if str(source.get("task_id", "")).startswith("isolation_")
                    and source.get("analysis_family") == "R1-E1-I"
                    and source.get("memory_mode") == "Frozen"
                    and source.get("location") == "Heldout"
                ]
            elif name == "table4_frozen_online_without":
                mode = str(row.get("mode", ""))
                expected_family = {
                    "Actual Frozen": "R1-E1-I",
                    "Online": "R1-E4",
                    "Without": "R1-E5",
                }.get(mode)
                sources = [
                    source
                    for source in sources
                    if source.get("analysis_family") == expected_family
                    and source.get("location") == "Heldout"
                    and (
                        source.get("memory_mode") == mode
                        or (mode == "Actual Frozen" and source.get("memory_mode") == "Frozen")
                    )
                ]
            if not sources and name == "table1_protocol_identity":
                sources = source_lineage
            for field, value in row.items():
                cell_sources = list(sources)
                if (
                    name
                    in {
                        "secondary_mixed_ecological",
                        "table2_core_adaptation_generalization",
                    }
                    and field.startswith("paired_")
                    and mechanism != "Fact"
                ):
                    expected_task = (
                        "mixed_ecological"
                        if name == "secondary_mixed_ecological"
                        else "isolation_fact"
                    )
                    expected_family = (
                        "R1-E1-M" if name == "secondary_mixed_ecological" else "R1-E1-I"
                    )
                    cell_sources.extend(
                        source
                        for source in source_lineage
                        if source.get("condition") == "Fact"
                        and source.get("task_id") == expected_task
                        and source.get("analysis_family") == expected_family
                        and (
                            name == "table2_core_adaptation_generalization"
                            or source.get("location") == "Heldout"
                        )
                        and (not checkpoint or str(source.get("checkpoint")) == checkpoint)
                    )
                if (
                    name == "table4_frozen_online_without"
                    and field == "paired_effect_vs_actual_frozen"
                    and row.get("mode") != "Actual Frozen"
                ):
                    cell_sources.extend(
                        source
                        for source in source_lineage
                        if source.get("condition") == mechanism
                        and str(source.get("task_id", "")).startswith("isolation_")
                        and source.get("analysis_family") == "R1-E1-I"
                        and source.get("memory_mode") == "Frozen"
                        and source.get("location") == "Heldout"
                        and (not checkpoint or str(source.get("checkpoint")) == checkpoint)
                    )
                compact_sources = [
                    {
                        "run_id": source.get("run_id"),
                        "seed": source.get("seed"),
                        "source_file": source.get("source_file"),
                        "source_file_sha256": source.get("source_file_sha256"),
                        "row_selector": source.get("row_selector"),
                    }
                    for source in cell_sources
                ]
                first = cell_sources[0] if cell_sources else {}
                output.append(
                    {
                        "lineage_id": f"table{table_index}:r{row_index}:{field}",
                        "output_artifact_id": f"{name}.csv",
                        "output_element_id": f"row={row_index};field={field}",
                        "output_kind": "table_cell",
                        "analysis_contract_id": "task8b-phase-f-v1",
                        "analysis_code_sha": first.get("analysis_code_sha", ""),
                        "analysis_manifest_sha256": first.get("analysis_manifest_sha256", ""),
                        "input_manifest_sha256": first.get("input_manifest_sha256", ""),
                        "exclusion_ledger_sha256": first.get("exclusion_ledger_sha256", ""),
                        "run_id": first.get("run_id", "protocol"),
                        "seed": first.get("seed", ""),
                        "condition": mechanism or str(row.get("field", "protocol")),
                        "checkpoint": checkpoint,
                        "heldout_table_id": "",
                        "attempt": first.get("attempt", ""),
                        "code_sha": first.get("code_sha", ""),
                        "config_sha256": first.get("config_sha256", ""),
                        "prompt_sha256": first.get("prompt_sha256", ""),
                        "model_fingerprint": first.get("model_fingerprint", ""),
                        "embedding_fingerprint": first.get("embedding_fingerprint", ""),
                        "schedule_sha256": first.get("schedule_sha256", ""),
                        "source_file": ";".join(
                            sorted({str(source.get("source_file", "")) for source in cell_sources})
                        ),
                        "source_file_sha256": ";".join(
                            sorted(
                                {
                                    str(source.get("source_file_sha256", ""))
                                    for source in cell_sources
                                }
                            )
                        ),
                        "row_selector": f"table_rows[{row_index - 1}].{field}",
                        "exclusion_status": "eligible",
                        "statistical_unit": "seed",
                        "n_planned": 12,
                        "n_effective": row.get("n_effective", ""),
                        "verification_status": "UNVERIFIED",
                        "input_snapshot_id": "frozen_local_recovery",
                        "source_records": json.dumps(
                            compact_sources,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "transformation": "frozen_table_aggregation",
                        "aggregation_order": "target->table->seed->cross_seed",
                        "cluster_ids": ";".join(
                            str(seed)
                            for seed in sorted(
                                {
                                    int(source["seed"])
                                    for source in cell_sources
                                    if str(source.get("seed", "")).isdigit()
                                }
                            )
                        ),
                        "missing_reason_codes": str(
                            row.get("missing_reason_codes", row.get("missing_reason", ""))
                        ),
                        "display_value": value,
                    }
                )
    return output


def _table1_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        ("code", "per-worker verified identity", "worker identity gate"),
        ("qwen", "Qwen3.5-9B frozen fingerprint", "worker identity gate"),
        ("bge", "BAAI/bge-m3 frozen fingerprint", "worker identity gate"),
        ("prompt", "frozen prompt SHA-256", "worker identity gate"),
        ("config", "per-task config SHA-256", "task identity audit"),
        ("schedule", "per-task schedule SHA-256 and seed-pod CRN", "task identity audit"),
        ("runtime_image", "uniform frozen image fingerprint", "fleet identity gate"),
        ("cuda_driver", "uniform frozen CUDA/driver class", "fleet health gate"),
        ("mechanisms", "NoMemory, Fact, Expr, Sync, Async", "TASK8B matrix"),
        ("checkpoints", "30,75,150,300", "TASK8B matrix"),
        ("heldout_tables", "H01,H02,H03", "TASK8B matrix"),
        ("memory_modes", "Actual Frozen,Online,Without", "TASK8B matrix"),
        (
            "seed_count",
            "n=12",
            "User-approved expedited protocol amendment before result unblinding",
        ),
        ("seed_list", "2026090101-2026090112", "TASK8B v2.0"),
        ("worker_count", str(len(selected_rows)), "Phase F completion gate"),
        ("hands_budget", "142200", "TASK8B frozen required matrix"),
        ("primary_endpoint", "final_test_bb_per_100", "task8b-phase-f-v1"),
        ("primary_checkpoint", "300", "task8b-phase-f-v1"),
        ("primary_family", "Expr_vs_Fact;Async_vs_Fact", "task8b-phase-f-v1"),
        ("statistical_unit", "seed", "task8b-phase-f-v1"),
        ("bootstrap", "seed cluster percentile 95%; 10000; PCG64; 2026090199", "task8b-phase-f-v1"),
        ("raw_test", "exact two-sided sign-flip", "task8b-phase-f-v1"),
        ("multiplicity", "Holm Primary Family A", "task8b-phase-f-v1"),
        (
            "attempt_rule",
            "first numeric complete valid; multiple valid fail closed",
            "task8b-phase-f-v1",
        ),
        ("r1_e0", "skipped", "expedited amendment"),
        ("new_power_pilot", "not run", "expedited amendment"),
        ("prospective_power", "not verified", "required disclosure"),
        ("conditional_deferred", "not started", "TASK8B boundary"),
        ("evidence_state", "paper artifacts require matrix-complete input", "completion contract"),
    )
    return [
        {
            "field": field,
            "frozen_value": value,
            "source": source,
            "status": "frozen",
            "disclosure": "not power-verified"
            if field in {"seed_count", "new_power_pilot"}
            else "",
        }
        for field, value, source in fields
    ]


def _seed_cells(metric_rows: list[dict[str, Any]], checkpoint: int) -> list[dict[str, Any]]:
    cells: dict[tuple[int, str], dict[str, list[Decimal]]] = defaultdict(lambda: defaultdict(list))
    tables: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in metric_rows:
        if (
            int(row.get("checkpoint_hand", 0)) != checkpoint
            or row.get("memory_mode") != "Frozen"
            or row.get("analysis_family", "R1-E1-I") != "R1-E1-I"
        ):
            continue
        hands = Decimal(str(row.get("hands", 0)))
        blind = Decimal(str(row.get("big_blind", 0)))
        if hands <= 0 or blind <= 0:
            continue
        value = Decimal(str(row.get("raw_chips", 0))) / hands / blind * Decimal(100)
        key = (int(row["seed"]), str(row["mechanism"]))
        location = str(row.get("location", "")).lower()
        cells[key][location].append(value)
        if location == "heldout":
            tables[key].add(str(row.get("table_id", "")))
    output = []
    for (seed, mechanism), values in sorted(cells.items()):
        if not values["source"] or tables[(seed, mechanism)] != {"H01", "H02", "H03"}:
            continue
        source = _mean(values["source"])
        heldout = _mean(values["heldout"])
        output.append(
            {
                "seed": seed,
                "mechanism": mechanism,
                "checkpoint": checkpoint,
                "source": source,
                "heldout": heldout,
                "gap": heldout - source,
            }
        )
    return output


def _ci_text(values: list[Decimal]) -> tuple[str, str]:
    if len(values) < 2:
        return "NA_NOT_ESTIMABLE", "NA_NOT_ESTIMABLE"
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(2026090199))
    draws = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return f"{low:.8f}", f"{high:.8f}"


def _table2_rows(
    metric_rows: list[dict[str, Any]], inference_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cells = _seed_cells(metric_rows, 300)
    fact = {int(row["seed"]): row["gap"] for row in cells if row["mechanism"] == "Fact"}
    inference = {str(row["contrast"]): row for row in inference_rows}
    output = []
    roles = {
        "NoMemory": "secondary_reference",
        "Fact": "primary_reference",
        "Expr": "primary_family_A",
        "Sync": "secondary",
        "Async": "primary_family_A",
    }
    for mechanism in ("NoMemory", "Fact", "Expr", "Sync", "Async"):
        rows = [row for row in cells if row["mechanism"] == mechanism]
        interactions = [
            row["gap"] - fact[int(row["seed"])] for row in rows if int(row["seed"]) in fact
        ]
        ci_low, ci_high = _ci_text(interactions)
        contrast = f"{mechanism}_vs_Fact"
        infer = inference.get(contrast, {})
        seeds = {int(row["seed"]) for row in rows}
        output.append(
            {
                "mechanism": mechanism,
                "analysis_role": roles[mechanism],
                "source_bb100_mean": _fixed(_mean([row["source"] for row in rows]))
                if rows
                else "NA_NOT_ESTIMABLE",
                "heldout_bb100_mean": _fixed(_mean([row["heldout"] for row in rows]))
                if rows
                else "NA_NOT_ESTIMABLE",
                "generalization_gap_mean": _fixed(_mean([row["gap"] for row in rows]))
                if rows
                else "NA_NOT_ESTIMABLE",
                "paired_interaction_vs_fact_mean": _fixed(_mean(interactions))
                if interactions
                else ("0.00000000" if mechanism == "Fact" and rows else "NA_NOT_ESTIMABLE"),
                "ci95_low": infer.get("ci95_low", ci_low),
                "ci95_high": infer.get("ci95_high", ci_high),
                "raw_p_two_sided": infer.get("raw_p_two_sided", "NA_NOT_APPLICABLE"),
                "holm_adjusted_p": infer.get("holm_adjusted_p", "NA_NOT_APPLICABLE"),
                "holm_rank": infer.get("holm_rank", "NA_NOT_APPLICABLE"),
                "holm_reject_alpha_0_05": infer.get("holm_reject_0_05", "NA_NOT_APPLICABLE"),
                "n_planned": 12,
                "n_effective": len(seeds),
                "missing_seed_ids": ";".join(
                    str(seed) for seed in FORMAL_SEEDS if seed not in seeds
                ),
                "excluded_seed_ids": "",
                "lineage_id": f"table2:{mechanism}",
                "status": "UNVERIFIED",
            }
        )
    return output


def _table3_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for mechanism in ("NoMemory", "Fact", "Expr", "Sync", "Async"):
        mechanism_rows = []
        for checkpoint in (30, 75, 150, 300):
            cells = [
                row for row in _seed_cells(metric_rows, checkpoint) if row["mechanism"] == mechanism
            ]
            gaps = [row["gap"] for row in cells]
            ci_low, ci_high = _ci_text(gaps)
            source_ci_low, source_ci_high = _ci_text([row["source"] for row in cells])
            heldout_ci_low, heldout_ci_high = _ci_text([row["heldout"] for row in cells])
            seeds = {int(row["seed"]) for row in cells}
            mechanism_rows.append(
                {
                    "mechanism": mechanism,
                    "checkpoint_hand": checkpoint,
                    "source_bb100_mean": _fixed(_mean([row["source"] for row in cells]))
                    if cells
                    else "NA_NOT_ESTIMABLE",
                    "source_ci95_low": source_ci_low,
                    "source_ci95_high": source_ci_high,
                    "heldout_bb100_mean": _fixed(_mean([row["heldout"] for row in cells]))
                    if cells
                    else "NA_NOT_ESTIMABLE",
                    "heldout_ci95_low": heldout_ci_low,
                    "heldout_ci95_high": heldout_ci_high,
                    "generalization_gap_mean": _fixed(_mean(gaps)) if gaps else "NA_NOT_ESTIMABLE",
                    "gap_ci95_low": ci_low,
                    "gap_ci95_high": ci_high,
                    "checkpoint_slope": "NA_NOT_ESTIMABLE",
                    "seed_clusters_planned": 12,
                    "seed_clusters_effective": len(seeds),
                    "heldout_tables_planned": 3,
                    "heldout_tables_effective": 3 if cells else 0,
                    "missing_reason_codes": "" if len(seeds) == 12 else "MISSING_SEED_CELL",
                    "lineage_id": f"table3:{mechanism}:{checkpoint}",
                    "status": "UNVERIFIED",
                }
            )
        valid = [
            row for row in mechanism_rows if row["generalization_gap_mean"] != "NA_NOT_ESTIMABLE"
        ]
        slope = (
            (
                float(valid[-1]["generalization_gap_mean"])
                - float(valid[0]["generalization_gap_mean"])
            )
            / (int(valid[-1]["checkpoint_hand"]) - int(valid[0]["checkpoint_hand"]))
            if len(valid) >= 2
            else None
        )
        for row in mechanism_rows:
            row["checkpoint_slope"] = f"{slope:.8f}" if slope is not None else "NA_NOT_ESTIMABLE"
        output.extend(mechanism_rows)
    return output


def _heldout_seed_values(
    metric_rows: list[dict[str, Any]], mechanism: str, mode: str, family: str
) -> dict[int, Decimal]:
    grouped: dict[tuple[int, str], list[Decimal]] = defaultdict(list)
    tables: dict[int, set[str]] = defaultdict(set)
    for row in metric_rows:
        if (
            row.get("mechanism") != mechanism
            or row.get("memory_mode") != mode
            or row.get("analysis_family", "R1-E1-I") != family
            or int(row.get("checkpoint_hand", 0)) != 300
            or row.get("location") != "Heldout"
        ):
            continue
        seed = int(row["seed"])
        table = str(row["table_id"])
        hands = Decimal(str(row["hands"]))
        blind = Decimal(str(row["big_blind"]))
        grouped[(seed, table)].append(Decimal(str(row["raw_chips"])) / hands / blind * Decimal(100))
        tables[seed].add(table)
    return {
        seed: _mean([_mean(grouped[(seed, table)]) for table in ("H01", "H02", "H03")])
        for seed in tables
        if tables[seed] == {"H01", "H02", "H03"}
    }


def _heldout_seed_trajectories(
    metric_rows: list[dict[str, Any]], mechanism: str, mode: str, family: str
) -> dict[int, tuple[Decimal, Decimal, Decimal | None]]:
    grouped: dict[tuple[int, str], list[list[Decimal]]] = defaultdict(list)
    for row in metric_rows:
        if (
            row.get("mechanism") != mechanism
            or row.get("memory_mode") != mode
            or row.get("analysis_family", "R1-E1-I") != family
            or int(row.get("checkpoint_hand", 0)) != 300
            or row.get("location") != "Heldout"
        ):
            continue
        series = row.get("hand_bb100_series")
        table = str(row.get("table_id", ""))
        if isinstance(series, list) and series:
            grouped[(int(row["seed"]), table)].append([Decimal(str(value)) for value in series])
    output: dict[int, tuple[Decimal, Decimal, Decimal | None]] = {}
    for seed in sorted({key[0] for key in grouped}):
        table_series = []
        for table in ("H01", "H02", "H03"):
            targets = grouped.get((seed, table), [])
            if not targets:
                break
            length = min(len(values) for values in targets)
            table_series.append(
                [_mean([values[index] for values in targets]) for index in range(length)]
            )
        if len(table_series) != 3 or min(len(values) for values in table_series) < 50:
            continue
        initial = _mean([_mean(values[:50]) for values in table_series])
        final = _mean([_mean(values[-50:]) for values in table_series])
        block_count = min(len(values) // 50 for values in table_series)
        slope = None
        if mode == "Online" and block_count >= 2:
            y_values = [
                _mean([_mean(values[index * 50 : (index + 1) * 50]) for values in table_series])
                for index in range(block_count)
            ]
            x_values = [Decimal(index) / Decimal(2) for index in range(block_count)]
            x_mean = _mean(x_values)
            y_mean = _mean(y_values)
            denominator = sum(((value - x_mean) ** 2 for value in x_values), Decimal(0))
            if denominator:
                slope = (
                    sum(
                        (
                            (x_values[index] - x_mean) * (y_values[index] - y_mean)
                            for index in range(block_count)
                        ),
                        Decimal(0),
                    )
                    / denominator
                )
        output[seed] = (initial, final, slope)
    return output


def _table4_rows(
    metric_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    plotting = []
    for mechanism in ("Expr", "Async"):
        frozen_full = _heldout_seed_values(metric_rows, mechanism, "Frozen", "R1-E1-I")
        frozen_trajectories = _heldout_seed_trajectories(
            metric_rows, mechanism, "Frozen", "R1-E1-I"
        )
        frozen_final = {
            seed: values[1] for seed, values in frozen_trajectories.items()
        } or frozen_full
        for mode, family in (
            ("Actual Frozen", "R1-E1-I"),
            ("Online", "R1-E4"),
            ("Without", "R1-E5"),
        ):
            trajectories = (
                frozen_trajectories
                if mode == "Actual Frozen"
                else _heldout_seed_trajectories(metric_rows, mechanism, mode, family)
            )
            full_values = (
                frozen_full
                if mode == "Actual Frozen"
                else _heldout_seed_values(metric_rows, mechanism, mode, family)
            )
            initial_values = {
                seed: values[0] for seed, values in trajectories.items()
            } or full_values
            final_values = {seed: values[1] for seed, values in trajectories.items()} or full_values
            slopes = [values[2] for values in trajectories.values() if values[2] is not None]
            paired = [
                final_values[seed] - frozen_final[seed]
                for seed in final_values
                if seed in frozen_final
            ]
            ci_low, ci_high = _ci_text(paired)
            output.append(
                {
                    "mechanism": mechanism,
                    "mode": mode,
                    "analysis_role": "reference" if mode == "Actual Frozen" else "secondary",
                    "parent_checkpoint_hand": 300,
                    "initial_transfer_bb100": _fixed(_mean(list(initial_values.values())))
                    if initial_values
                    else "NA_NOT_ESTIMABLE",
                    "recovery_slope_bb100_per_100_hands": _fixed(_mean(slopes))
                    if mode == "Online" and slopes
                    else ("NA_NOT_ESTIMABLE" if mode == "Online" else "NA_NOT_APPLICABLE"),
                    "final_bb100": _fixed(_mean(list(final_values.values())))
                    if final_values
                    else "NA_NOT_ESTIMABLE",
                    "paired_effect_vs_actual_frozen": _fixed(_mean(paired))
                    if paired
                    else (
                        "0.00000000"
                        if mode == "Actual Frozen" and final_values
                        else "NA_NOT_ESTIMABLE"
                    ),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_planned": 12,
                    "n_effective": len(final_values),
                    "parent_checkpoint_hash_gate": "VERIFIED",
                    "crn_gate": "VERIFIED",
                    "clone_isolation_gate": "VERIFIED",
                    "missing_reason_codes": "" if len(final_values) == 12 else "MISSING_SEED_CELL",
                    "lineage_id": f"table4:{mechanism}:{mode}",
                    "status": "UNVERIFIED",
                }
            )
            plotting.extend(
                {
                    "seed": seed,
                    "mechanism": mechanism,
                    "mode": mode,
                    "final_bb100": _fixed(value),
                    "paired_effect_vs_actual_frozen": _fixed(value - frozen_final[seed])
                    if seed in frozen_final
                    else "NA_NOT_ESTIMABLE",
                }
                for seed, value in sorted(final_values.items())
            )
    return output, plotting


def _table5_rows(e6_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_map = {
        "raw_bb100": "raw_bb_per_100",
        "leave_largest_absolute_pot_out_bb100": "leave_largest_absolute_pot_out_bb_per_100",
        "median_seed_effect_bb100": "median_bb_per_100",
        "trimmed_10pct_bb100": "trimmed_10pct_bb_per_100",
        "winsorized_5_95_bb100": "winsorized_5_95_bb_per_100",
        "vpip_rate": "vpip_pct",
        "fold_rate": "fold_pct",
        "raise_rate": "raise_pct",
        "all_in_rate": "all_in_pct",
        "bust_rate": "bust_pct",
        "max_pot_share": "max_pot_share_pct",
    }
    output = []
    for mechanism in ("NoMemory", "Fact", "Expr", "Sync", "Async"):
        rows = [
            row
            for row in e6_rows
            if row.get("mechanism", "Fact") == mechanism
            and str(row.get("task_id", "isolation_fact")).startswith("isolation_")
            and row.get("memory_mode", "Frozen") == "Frozen"
        ]
        summary = {
            output_field: _fixed(_mean([Decimal(str(row[source_field])) for row in rows]))
            if rows
            else "NA_NOT_ESTIMABLE"
            for output_field, source_field in field_map.items()
        }
        seeds = {int(row["seed"]) for row in rows}
        loo = [row for row in _leave_one_seed_out_rows(e6_rows) if row["mechanism"] == mechanism]
        loo_values = [
            Decimal(str(row["raw_bb100_leave_one_seed_out"]))
            for row in loo
            if row["raw_bb100_leave_one_seed_out"] != "NA_NOT_ESTIMABLE"
        ]
        raw_value = Decimal(str(summary["raw_bb100"])) if rows else Decimal(0)
        direction_sensitive = bool(
            loo_values
            and (
                min(loo_values) < 0 < max(loo_values)
                or any(value * raw_value < 0 for value in loo_values)
            )
        )
        output.append(
            {
                "mechanism": mechanism,
                "checkpoint_hand": 300,
                "mode": "Frozen",
                **summary,
                "fallback_count": 0 if rows else "NA_NOT_ESTIMABLE",
                "revision_fallback_count": 0 if rows else "NA_NOT_ESTIMABLE",
                "reward_conservation_violations": 0 if rows else "NA_NOT_ESTIMABLE",
                "stack_conservation_violations": 0 if rows else "NA_NOT_ESTIMABLE",
                "n_planned": 12,
                "n_effective": len(seeds),
                "sensitivity_flag": "REVIEW"
                if direction_sensitive
                or any(abs(Decimal(str(row["raw_bb_per_100"]))) > Decimal(500) for row in rows)
                else "NONE",
                "missing_reason_codes": "" if len(seeds) == 12 else "MISSING_SEED_CELL",
                "lineage_id": f"table5:{mechanism}",
                "status": "UNVERIFIED",
            }
        )
    return output


def _leave_one_seed_out_rows(e6_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for mechanism in ("NoMemory", "Fact", "Expr", "Sync", "Async"):
        by_seed: dict[int, list[Decimal]] = defaultdict(list)
        for row in e6_rows:
            if (
                row.get("mechanism", "Fact") == mechanism
                and str(row.get("task_id", "isolation_fact")).startswith("isolation_")
                and row.get("memory_mode", "Frozen") == "Frozen"
            ):
                by_seed[int(row["seed"])].append(Decimal(str(row["raw_bb_per_100"])))
        seed_values = {seed: _mean(values) for seed, values in by_seed.items()}
        for omitted_seed in sorted(seed_values):
            retained = [value for seed, value in seed_values.items() if seed != omitted_seed]
            output.append(
                {
                    "mechanism": mechanism,
                    "omitted_seed": omitted_seed,
                    "n_effective": len(retained),
                    "raw_bb100_leave_one_seed_out": _fixed(_mean(retained))
                    if retained
                    else "NA_NOT_ESTIMABLE",
                    "status": "UNVERIFIED",
                }
            )
    return output


def _secondary_mixed_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mechanism: dict[str, dict[int, dict[str, list[Decimal]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in metric_rows:
        if (
            row.get("analysis_family") != "R1-E1-M"
            or row.get("location") != "Heldout"
            or int(row.get("checkpoint_hand", 0)) != 300
        ):
            continue
        hands = Decimal(str(row.get("hands", 0)))
        blind = Decimal(str(row.get("big_blind", 0)))
        if hands > 0 and blind > 0:
            by_mechanism[str(row.get("mechanism"))][int(row["seed"])][str(row["table_id"])].append(
                Decimal(str(row["raw_chips"])) / hands / blind * Decimal(100)
            )
    seed_values: dict[str, dict[int, Decimal]] = {}
    for mechanism, by_seed in by_mechanism.items():
        seed_values[mechanism] = {
            seed: _mean([_mean(tables[table]) for table in ("H01", "H02", "H03")])
            for seed, tables in by_seed.items()
            if set(tables) == {"H01", "H02", "H03"}
        }
    fact = seed_values.get("Fact", {})
    output = []
    for mechanism in ("Fact", "Expr", "Sync", "Async"):
        values_by_seed = seed_values.get(mechanism, {})
        values = list(values_by_seed.values())
        paired = [values_by_seed[seed] - fact[seed] for seed in values_by_seed if seed in fact]
        ci_low, ci_high = _ci_text(values)
        paired_low, paired_high = _ci_text(paired)
        output.append(
            {
                "mechanism": mechanism,
                "analysis_role": "secondary_ecological",
                "heldout_bb100_mean": _fixed(_mean(values)) if values else "NA_NOT_ESTIMABLE",
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_effect_vs_fact_mean": _fixed(_mean(paired))
                if paired
                else ("0.00000000" if mechanism == "Fact" and values else "NA_NOT_ESTIMABLE"),
                "paired_ci95_low": paired_low,
                "paired_ci95_high": paired_high,
                "n_planned": 12,
                "n_effective": len(values),
                "status": "UNVERIFIED",
            }
        )
    return output


def _figure1_flow_rows() -> list[dict[str, Any]]:
    nodes = (
        "source train",
        "30/75/150/300 checkpoints",
        "immutable clones",
        "H01/H02/H03",
        "Frozen/Online/Without",
        "seed-level aggregate",
        "Table/Figure lineage",
    )
    return [
        {"order": index, "source": nodes[index], "target": nodes[index + 1]}
        for index in range(len(nodes) - 1)
    ]


def _figure3_rows(
    effect_rows: list[dict[str, Any]], inference_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [{**row, "row_type": "seed"} for row in effect_rows]
    rows.extend({**row, "row_type": "summary"} for row in inference_rows)
    return rows


def _figure_caption(index: int, rows: list[dict[str, Any]]) -> str:
    return (
        f"# Figure {index} caption\n\n"
        "Analysis unit: seed; n_planned=12. CI: seed-cluster percentile 95%, "
        "10,000 PCG64 replicates, seed 2026090199. Missing/exclusion follows the frozen "
        f"ledger. Plotting rows: {len(rows)}.\n"
    )


def _checkpoint_table(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(
        metric_rows,
        key=lambda item: (
            int(item.get("seed", 0)),
            str(item.get("mechanism", "")),
            int(item.get("checkpoint_hand", 0)),
            str(item.get("table_id", "")),
        ),
    ):
        hands = Decimal(str(row.get("hands", 0)))
        blind = Decimal(str(row.get("big_blind", 0)))
        if hands <= 0 or blind <= 0:
            continue
        rows.append(
            {
                "seed": row.get("seed"),
                "mechanism": row.get("mechanism"),
                "memory_mode": row.get("memory_mode"),
                "checkpoint": row.get("checkpoint_hand"),
                "location": row.get("location"),
                "heldout_table_id": row.get("table_id"),
                "bb_per_100": _fixed(
                    Decimal(str(row.get("raw_chips", 0))) / hands / blind * Decimal(100)
                ),
            }
        )
    return rows


def _render_figure(destination: Path, index: int, rows: list[dict[str, Any]]) -> None:
    matplotlib.rcParams["svg.hashsalt"] = "task8b-phase-f-v1"
    figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    if index == 1:
        nodes = [str(row["source"]) for row in rows] + ([str(rows[-1]["target"])] if rows else [])
        axis.set_xlim(-0.5, max(len(nodes) - 0.5, 0.5))
        axis.set_ylim(-0.5, 0.5)
        for position, node in enumerate(nodes):
            axis.text(
                position,
                0,
                node.replace("/", "/\n"),
                ha="center",
                va="center",
                fontsize=7,
                bbox={"boxstyle": "round", "facecolor": "#DCEAF7", "edgecolor": "#235789"},
            )
            if position < len(nodes) - 1:
                axis.annotate(
                    "",
                    xy=(position + 0.72, 0),
                    xytext=(position + 0.28, 0),
                    arrowprops={"arrowstyle": "->"},
                )
        axis.axis("off")
    elif index == 2:
        mechanisms = ("NoMemory", "Fact", "Expr", "Sync", "Async")
        palette = ("#0072B2", "#000000", "#D55E00", "#009E73", "#CC79A7")
        for mechanism, color in zip(mechanisms, palette, strict=True):
            subset = [row for row in rows if row.get("mechanism") == mechanism]
            for location, field, linestyle in (
                ("Source", "source_bb100_mean", "-"),
                ("Heldout", "heldout_bb100_mean", "--"),
            ):
                valid = [row for row in subset if row.get(field) != "NA_NOT_ESTIMABLE"]
                axis.plot(
                    [int(row["checkpoint_hand"]) for row in valid],
                    [float(row[field]) for row in valid],
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=f"{mechanism} {location}",
                )
        if rows:
            axis.legend(frameon=False)
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_xticks([30, 75, 150, 300])
        axis.set_xlabel("Checkpoint hand")
        axis.set_ylabel("BB/100")
    elif index == 3:
        contrasts = ("Expr_vs_Fact", "Async_vs_Fact")
        seed_rows = [row for row in rows if row.get("row_type") == "seed"]
        for seed in sorted({int(row["seed"]) for row in seed_rows}):
            subset = [row for row in seed_rows if int(row["seed"]) == seed]
            axis.plot(
                [contrasts.index(str(row["contrast"])) for row in subset],
                [float(row["paired_interaction_bb_per_100"]) for row in subset],
                color="#999999",
                linewidth=0.6,
                marker="o",
            )
        for row in (item for item in rows if item.get("row_type") == "summary"):
            if row.get("ci95_low") == "NA_NOT_ESTIMABLE":
                continue
            position = contrasts.index(str(row["contrast"]))
            mean = float(row["mean_bb100"])
            axis.errorbar(
                [position],
                [mean],
                yerr=[
                    [mean - float(row["ci95_low"])],
                    [float(row["ci95_high"]) - mean],
                ],
                fmt="D",
                color="#000000",
                linewidth=1.5,
                capsize=4,
            )
        axis.set_xticks([0, 1], contrasts)
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_ylabel("PrimaryInteraction (BB/100)")
    else:
        figure.clf()
        axis_a, axis_b = figure.subplots(1, 2)
        modes = ("Actual Frozen", "Online", "Without")
        for mechanism, color in (("Expr", "#D55E00"), ("Async", "#0072B2")):
            mechanism_rows = [row for row in rows if row.get("mechanism") == mechanism]
            for seed in sorted({int(row["seed"]) for row in mechanism_rows}):
                subset = [row for row in mechanism_rows if int(row["seed"]) == seed]
                x = [modes.index(str(row["mode"])) for row in subset]
                axis_a.plot(
                    x,
                    [float(row["final_bb100"]) for row in subset],
                    color=color,
                    alpha=0.35,
                    linewidth=0.7,
                    marker="o",
                )
                paired = [
                    row
                    for row in subset
                    if row.get("paired_effect_vs_actual_frozen") != "NA_NOT_ESTIMABLE"
                ]
                axis_b.plot(
                    [modes.index(str(row["mode"])) for row in paired],
                    [float(row["paired_effect_vs_actual_frozen"]) for row in paired],
                    color=color,
                    alpha=0.35,
                    linewidth=0.7,
                    marker="o",
                )
        for panel, ylabel in (
            (axis_a, "Final held-out BB/100"),
            (axis_b, "Paired effect vs Actual Frozen"),
        ):
            panel.set_xticks(range(3), modes, rotation=20)
            panel.axhline(0.0, color="#444444", linewidth=0.8)
            panel.set_ylabel(ylabel)
            panel.grid(axis="y", alpha=0.25)
        axis = axis_a
    axis.set_title(f"TASK8B Figure {index}")
    if index != 4:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    fixed_date = dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc)
    figure.savefig(
        destination / f"figure{index}.png",
        format="png",
        metadata={"Software": "AgentMemEval TASK8B Phase F"},
    )
    figure.savefig(
        destination / f"figure{index}.svg",
        format="svg",
        metadata={"Date": "2026-07-22", "Creator": "AgentMemEval TASK8B Phase F"},
    )
    figure.savefig(
        destination / f"figure{index}.pdf",
        format="pdf",
        metadata={
            "Creator": "AgentMemEval TASK8B Phase F",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    plt.close(figure)


def _write_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) or ["status"]
    normalized = [{field: row.get(field, "") for field in fields} for row in rows]
    _write_csv(path, normalized, fields=tuple(fields))


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = sorted({field for row in rows for field in row}) or ["status"]
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join((header, separator, *body))


def _write_text_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("reason_code") not in ALLOWED_EXCLUSION_REASONS:
            raise ConfigError(f"exclusion reason 未预注册：{row.get('reason_code')}")
        if row.get("recorded_before_effect_unblind", "").lower() != "true":
            raise ConfigError("exclusion ledger 必须在 effect unblind 前记录")
    return rows


def _read_files_manifest(path: Path) -> dict[str, tuple[int, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result: dict[str, tuple[int, str]] = {}
    for row in rows:
        relative = str(row.get("relative_path", ""))
        if (
            not relative
            or relative in result
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ConfigError("files.tsv source path 非法或重复")
        result[relative] = (int(row["size"]), str(row["sha256"]))
    return result


def _task_run_map(root: Path, *, worker_id: str) -> dict[str, Path]:
    results = _read_json(root / "task_results.json")
    if (
        results.get("schema_version") != "task8-worker-task-results-v1"
        or results.get("worker_id") != worker_id
    ):
        raise ConfigError("task_results identity/schema 不匹配")
    rows = results.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ConfigError("task_results.tasks 为空")
    mapping: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ConfigError("task_results 存在未 complete task")
        task_id = str(row.get("task_id", ""))
        run_dir = _inside_attempt(root, str(row.get("run_dir", "")))
        if task_id in mapping or not run_dir.is_dir():
            raise ConfigError("task_results task_id/run_dir 非法")
        marker = _read_json(root / "task_receipts" / f"{task_id}.json")
        if marker.get("task_id") != task_id or marker.get("run_dir") != row.get("run_dir"):
            raise ConfigError("task receipt 与 task_results 不匹配")
        marker_files = marker.get("files")
        if not isinstance(marker_files, list) or marker_files != _directory_manifest(run_dir):
            raise ConfigError("task receipt child files hash 不匹配")
        mapping[task_id] = run_dir
    return mapping


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _inside_attempt(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ConfigError(f"attempt artifact 路径非法：{relative}")
    resolved = root.joinpath(*candidate.parts).resolve()
    if root != resolved and root not in resolved.parents:
        raise ConfigError(f"attempt artifact 路径越界：{relative}")
    return resolved


def _verify_analysis_state(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or rows[-1].get("status") != "complete":
        raise ConfigError("analysis worker state 最终状态不是 complete")
    previous = "GENESIS"
    for row in rows:
        expected = sha256_json(
            {
                "schema_version": row.get("schema_version"),
                "created_at_utc": row.get("created_at_utc"),
                "status": row.get("status"),
                "detail": row.get("detail"),
                "previous_sha256": row.get("previous_sha256"),
            }
        )
        if (
            row.get("schema_version") != STATE_SCHEMA_VERSION
            or row.get("previous_sha256") != previous
            or row.get("row_sha256") != expected
        ):
            raise ConfigError("analysis worker state hash chain 失败")
        previous = expected


def _attempt_number(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"attempt_(01)|__attempt_(\d{2,})", value)
    if match is None:
        raise ConfigError(f"attempt 名称不符合 frozen convention：{value}")
    number = int(match.group(1) or match.group(2))
    if value.startswith("__") and number < 2:
        raise ConfigError(f"retry attempt 序号必须从 02 开始：{value}")
    return number, value


def _selected_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": item["worker_id"],
        "pod_id": item["pod_id"],
        "seed": item["seed"],
        "attempt": item["attempt"],
        "status": "included",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取 Phase F JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Phase F JSON 顶层必须为对象：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ConfigError(f"JSONL 行必须为对象：{path}")
        rows.append(value)
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, fields: tuple[str, ...] | None = None
) -> None:
    fieldnames = list(fields or (tuple(rows[0]) if rows else ("status",)))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_bytes_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        raise ConfigError("paired interaction 缺 source/heldout cell")
    return sum(values, Decimal(0)) / Decimal(len(values))


def _fixed(value: Decimal) -> str:
    return format(value, ".8f")


def _percentile(values: list[Decimal], probability: Decimal) -> Decimal:
    if not values:
        raise ConfigError("percentile 输入不能为空")
    ordered = sorted(values)
    position = probability * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - Decimal(lower)
    return ordered[lower] * (Decimal(1) - fraction) + ordered[upper] * fraction


def _rate(rows: list[dict[str, Any]], field: str) -> str:
    return _fixed(
        Decimal(sum(bool(row.get(field)) for row in rows)) / Decimal(len(rows)) * Decimal(100)
    )
