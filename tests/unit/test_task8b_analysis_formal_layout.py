from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation import task8b_analysis
from agentmemeval.evaluation.task8b_analysis import run_task8b_analysis
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.formal_runner import append_worker_state

WORKER_IDENTITY = {
    "code_sha": "a" * 40,
    "config_sha256": "b" * 64,
    "prompt_sha256": "c" * 64,
    "model_fingerprint": "qwen-frozen-v1",
    "embedding_fingerprint": "bge-m3-frozen-v1",
    "schedule_sha256": "d" * 64,
}
TASK_IDENTITY = {
    "code_sha": "a" * 40,
    "resolved_config_sha256": "e" * 64,
    "prompt_sha256": "c" * 64,
    "model_fingerprint": "qwen-frozen-v1",
    "embedding_fingerprint": "bge-m3-frozen-v1",
    "schedule_sha256": "d" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )


def _formal_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    attempt = tmp_path / "formal-worker" / "P01" / "attempt_01"
    task_id = "isolation_fact"
    task_run_relative = "runs/isolation_fact__attempt_02"
    task = attempt / task_run_relative
    task.mkdir(parents=True)
    schedule_body = {
        "schema_version": "task8-heldout-schedule-v1",
        "source_namespace": "task8/source/v1",
        "heldout_namespace": "task8/heldout/v1",
        "rows": [
            {
                "phase": "heldout",
                "checkpoint_hand": 300,
                "table_id": table_id,
                "hand_number": 1,
            }
            for table_id in ("H01", "H02", "H03")
        ],
    }
    schedule_sha256 = sha256_json(schedule_body)
    task_identity = {**TASK_IDENTITY, "schedule_sha256": schedule_sha256}
    _write_json(
        attempt / "worker_manifest.json",
        {
            "schema_version": "task8-worker-manifest-v1",
            "worker_id": "P01",
            "role": "primary",
            "pod_id": "pod01",
            "seed_bundle": 2026090101,
            "seed_pod_identity": {"seed": 2026090101, "schedule_sha256": "d" * 64},
            "receipt_identity": {"producer": "P01", "checkpoint_hand": 300},
            "execution_mode": "experiment_configs",
            "common_identity": WORKER_IDENTITY,
            "task_configs": [
                {
                    "task_id": task_id,
                    "memory_mode": "Frozen",
                    "schedule_sha256": schedule_sha256,
                    "expected_identity": task_identity,
                }
            ],
        },
    )
    _write_json(
        attempt / "task_results.json",
        {
            "schema_version": "task8-worker-task-results-v1",
            "worker_id": "P01",
            "tasks": [
                {
                    "task_id": task_id,
                    "status": "complete",
                    "run_dir": task_run_relative,
                }
            ],
        },
    )
    for status in ("planned", "validating", "running", "finalizing", "complete"):
        append_worker_state(attempt / "state.tsv", status, "formal analysis fixture")
    (task / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "table": {"big_blind": 2},
                "agent": {"mechanism": "expr"},
                "experiment": {
                    "target_agent_id": "agent_00",
                    "train_hands": 1,
                    "checkpoint_set": [300],
                    "memory_mode": "Frozen",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="",
    )
    _write_json(
        task / "metrics.json",
        {
            "execution_health": {
                "valid": True,
                "fallback_count": 0,
                "memory_revision_fallback_count": 0,
                "reward_conservation_violation_count": 0,
                "stack_conservation_violation_count": 0,
            },
            "run_validity": {"execution_valid": True, "behavior_valid": True},
        },
    )
    hands = [
        {
            "hand_id": "train-1",
            "stage": "train",
            "hand_number": 1,
            "rewards": {"agent_00": 2},
            "final_stacks": {"agent_00": 100},
        }
    ]
    hands.extend(
        {
            "hand_id": f"test-{table_id}",
            "stage": "test",
            "hand_number": 1,
            "checkpoint_hand": 300,
            "heldout_table_id": table_id,
            "rewards": {"agent_00": 1},
            "final_stacks": {"agent_00": 100},
        }
        for table_id in ("H01", "H02", "H03")
    )
    _write_json(
        task / "schedule_manifest.json",
        {**schedule_body, "schedule_sha256": schedule_sha256},
    )
    _write_jsonl(task / "hand_summaries.jsonl", hands)
    _write_jsonl(
        task / "events.jsonl",
        [
            {
                "event": "action",
                "hand_id": "train-1",
                "agent_id": "agent_00",
                "phase": "preflop",
                "action_type": "raise",
                "pot_after": 8,
                "call_risk": {"is_all_in": False},
            },
            {
                "event": "action",
                "hand_id": "test-H01",
                "agent_id": "agent_00",
                "phase": "preflop",
                "action_type": "call",
                "pot_after": 4,
                "call_risk": {"is_all_in": False},
            },
        ],
    )
    _write_json(
        task / "task_identity_audit.json",
        {
            "schema_version": "task8-task-identity-audit-v1",
            "task_id": task_id,
            "status": "verified",
            "actual": task_identity,
        },
    )
    child_files = [
        {
            "relative_path": path.relative_to(task).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in task.rglob("*") if item.is_file())
    ]
    _write_json(
        attempt / "task_receipts" / f"{task_id}.json",
        {
            "schema_version": "task8-worker-task-receipt-v1",
            "task_id": task_id,
            "run_dir": task_run_relative,
            "files": child_files,
        },
    )
    listed = [
        attempt / "worker_manifest.json",
        attempt / "task_results.json",
        attempt / "state.tsv",
        task / "resolved_config.yaml",
        task / "metrics.json",
        task / "hand_summaries.jsonl",
        task / "events.jsonl",
        task / "task_identity_audit.json",
        task / "schedule_manifest.json",
        attempt / "task_receipts" / f"{task_id}.json",
    ]
    with (attempt / "files.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("relative_path", "size", "sha256"))
        for path in listed:
            writer.writerow(
                (path.relative_to(attempt).as_posix(), path.stat().st_size, _sha256(path))
            )
    _write_json(
        attempt / "completion_receipt.json",
        {
            "schema_version": "task8-worker-completion-v1",
            "worker_id": "P01",
            "status": "complete",
            "files_tsv_sha256": _sha256(attempt / "files.tsv"),
        },
    )
    secondary_attempt = tmp_path / "formal-worker" / "S01" / "attempt_01"
    shutil.copytree(attempt, secondary_attempt)
    secondary_manifest_path = secondary_attempt / "worker_manifest.json"
    secondary_manifest = json.loads(secondary_manifest_path.read_text(encoding="utf-8"))
    secondary_manifest.update(
        worker_id="S01",
        role="secondary",
        depends_on="P01",
        dependency_receipt_identity={"producer": "P01", "checkpoint_hand": 300},
    )
    secondary_manifest.pop("receipt_identity", None)
    _write_json(secondary_manifest_path, secondary_manifest)
    secondary_results_path = secondary_attempt / "task_results.json"
    secondary_results = json.loads(secondary_results_path.read_text(encoding="utf-8"))
    secondary_results["worker_id"] = "S01"
    _write_json(secondary_results_path, secondary_results)
    secondary_completion_path = secondary_attempt / "completion_receipt.json"
    secondary_completion = json.loads(
        secondary_completion_path.read_text(encoding="utf-8")
    )
    secondary_completion["worker_id"] = "S01"
    _write_json(secondary_completion_path, secondary_completion)
    _refresh_files(secondary_attempt)
    manifest = tmp_path / "formal_input.json"
    _write_json(
        manifest,
        {
            "schema_version": "task8b-phase-f-input-v1",
            "analysis_contract_id": "task8b-phase-f-v1",
            "synthetic_test_mode": True,
            "workers": [
                {
                    "worker_id": "P01",
                    "pod_id": "pod01",
                    "seed": 2026090101,
                    "expected_identity": WORKER_IDENTITY,
                    "expected_worker_manifest_sha256": _sha256(
                        attempt / "worker_manifest.json"
                    ),
                    "attempts": [
                        {
                            "attempt": "attempt_01",
                            "relative_path": "formal-worker/P01/attempt_01",
                        }
                    ],
                },
                {
                    "worker_id": "S01",
                    "pod_id": "pod01",
                    "seed": 2026090101,
                    "expected_identity": WORKER_IDENTITY,
                    "expected_worker_manifest_sha256": _sha256(
                        secondary_attempt / "worker_manifest.json"
                    ),
                    "attempts": [
                        {
                            "attempt": "attempt_01",
                            "relative_path": "formal-worker/S01/attempt_01",
                        }
                    ],
                },
            ],
        },
    )
    ledger = tmp_path / "exclusion_ledger.csv"
    ledger.write_text(
        "ledger_entry_id,recorded_before_effect_unblind,seed,pod_id,worker_id,attempt,"
        "reason_code,authoritative_attempt\n",
        encoding="utf-8",
        newline="",
    )
    return manifest, ledger, attempt


def _refresh_files(attempt: Path) -> None:
    receipt_path = attempt / "task_receipts" / "isolation_fact.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    child = attempt / receipt["run_dir"]
    receipt["files"] = [
        {
            "relative_path": path.relative_to(child).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in child.rglob("*") if item.is_file())
    ]
    _write_json(receipt_path, receipt)
    files_path = attempt / "files.tsv"
    rows = list(csv.DictReader(files_path.open("r", encoding="utf-8"), delimiter="\t"))
    for row in rows:
        path = attempt / row["relative_path"]
        row["size"] = str(path.stat().st_size)
        row["sha256"] = _sha256(path)
    with files_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completion_path = attempt / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(completion_path, completion)


def test_phase_f_accepts_formal_worker_layout_and_normalizes_raw_hands(
    tmp_path: Path,
) -> None:
    manifest, ledger, _ = _formal_fixture(tmp_path)
    output = tmp_path / "analysis"

    result = run_task8b_analysis(manifest, ledger, output)

    assert result["selected_worker_count"] == 2
    with (output / "table3_checkpoint_scan.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        checkpoint = list(csv.DictReader(handle))
    fact_300 = next(
        row
        for row in checkpoint
        if row["mechanism"] == "Fact" and row["checkpoint_hand"] == "300"
    )
    assert fact_300["source_bb100_mean"] == "100.00000000"
    assert fact_300["heldout_bb100_mean"] == "50.00000000"
    assert fact_300["generalization_gap_mean"] == "-50.00000000"
    e6 = (output / "e6_metrics.csv").read_text(encoding="utf-8")
    assert "50.00000000" in e6
    lineage = (output / "data_lineage.csv").read_text(encoding="utf-8")
    assert "runs/isolation_fact__attempt_02/hand_summaries.jsonl" in lineage


def test_phase_f_formal_layout_rejects_task_identity_tamper(tmp_path: Path) -> None:
    manifest, ledger, attempt = _formal_fixture(tmp_path)
    audit_path = (
        attempt / "runs" / "isolation_fact__attempt_02" / "task_identity_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["actual"]["prompt_sha256"] = "f" * 64
    _write_json(audit_path, audit)
    _refresh_files(attempt)

    with pytest.raises(ConfigError, match="identity|IDENTITY|ledger"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_formal_layout_rejects_child_health_tamper(tmp_path: Path) -> None:
    manifest, ledger, attempt = _formal_fixture(tmp_path)
    metrics_path = attempt / "runs" / "isolation_fact__attempt_02" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["execution_health"]["fallback_count"] = 1
    _write_json(metrics_path, metrics)
    _refresh_files(attempt)

    with pytest.raises(ConfigError, match="health|fallback|FALLBACK|ledger"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_formal_layout_binds_frozen_worker_manifest_sha(tmp_path: Path) -> None:
    manifest, ledger, _ = _formal_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["workers"][0]["expected_worker_manifest_sha256"] = "0" * 64
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="identity|IDENTITY|manifest|ledger"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_seed_pod_identity_is_recomputed_from_authoritative_task_schedule_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentmemeval.evaluation import task8b_analysis

    monkeypatch.setattr(
        task8b_analysis, "FORMAL_PRIMARY_TASKS", {"task": (71100, {"cell"})}
    )
    monkeypatch.setattr(
        task8b_analysis, "FORMAL_SECONDARY_TASKS", {"task": (71100, {"cell"})}
    )
    seed = 2026090101
    schedule_bodies = {
        role: {
            "schema_version": "task8-heldout-schedule-v1",
            "source_namespace": f"task8/source/{role}",
            "heldout_namespace": f"task8/heldout/{role}",
            "rows": [{"table_id": "T01", "schedule_seed": seed}],
        }
        for role in ("primary", "secondary")
    }
    actual_schedule_shas = {
        role: sha256_json(body) for role, body in schedule_bodies.items()
    }
    alternate_primary_body = {
        **schedule_bodies["primary"],
        "rows": [{"table_id": "T02", "schedule_seed": seed}],
    }
    declared_schedule_shas = {
        "primary": sha256_json(alternate_primary_body),
        "secondary": actual_schedule_shas["secondary"],
    }
    declared_rows = [
        {
            "worker_role": role,
            "task_id": "task",
            "schedule_sha256": declared_schedule_shas[role],
        }
        for role in ("primary", "secondary")
    ]
    pod_identity = {
        "seed_bundle": seed,
        "schedule_sha256": task8b_analysis.sha256_json(
            {
                "schema_version": "task8b-seed-pod-schedule-bundle-v1",
                "seed_bundle": seed,
                "task_schedules": declared_rows,
            }
        ),
        "task_schedules": declared_rows,
    }
    selected = []
    for role, worker_id in (
        ("primary", "P01"),
        ("secondary", "S01"),
    ):
        root = tmp_path / worker_id
        child = root / "runs" / "task__attempt_02"
        _write_json(
            child / "schedule_manifest.json",
            {
                **schedule_bodies[role],
                "schedule_sha256": actual_schedule_shas[role],
            },
        )
        task_row = {
            "task_id": "task",
            "status": "complete",
            "run_dir": "runs/task__attempt_02",
        }
        _write_json(
            root / "task_results.json",
            {
                "schema_version": "task8-worker-task-results-v1",
                "worker_id": worker_id,
                "tasks": [task_row],
            },
        )
        _write_json(
            root / "task_receipts" / "task.json",
            {
                "task_id": "task",
                "run_dir": "runs/task__attempt_02",
                "files": [
                    {
                        "relative_path": "schedule_manifest.json",
                        "size": (child / "schedule_manifest.json").stat().st_size,
                        "sha256": _sha256(child / "schedule_manifest.json"),
                    }
                ],
            },
        )
        manifest = {
            "worker_id": worker_id,
            "role": role,
            "depends_on": None if role == "primary" else "P01",
            "seed_pod_identity": pod_identity,
            "receipt_identity": {"bundle": "frozen"} if role == "primary" else None,
            "dependency_receipt_identity": (
                {"bundle": "frozen"} if role == "secondary" else None
            ),
            "task_configs": [
                {
                    "task_id": "task",
                    "planned_hands": 71100,
                    "covers": ["cell"],
                    "schedule_sha256": declared_schedule_shas[role],
                    "expected_identity": {
                        "schedule_sha256": declared_schedule_shas[role]
                    },
                }
            ],
        }
        _write_json(root / "worker_manifest.json", manifest)
        selected.append({"root": root, "seed": seed, "worker_id": worker_id})

    with pytest.raises(ConfigError, match="schedule|CRN|identity"):
        task8b_analysis._validate_selected_seed_pods(selected, enforce_formal=True)


def _six_seed_workers() -> list[dict[str, object]]:
    workers: list[dict[str, object]] = []
    for index in range(7, 13):
        seed = 2026090100 + index
        slots = [f"H{index - 6:02d}", f"H{index:02d}"]
        for role in ("P", "S"):
            worker_id = f"{role}{index:02d}"
            if role == "P":
                shard_role = "primary_isolation_partition"
                partitions = (
                    (
                        "high",
                        [
                            "isolation_no_memory",
                            "isolation_fact",
                            "isolation_expr",
                            "isolation_sync",
                        ],
                        5400,
                    ),
                    ("low", ["isolation_async"], 1350),
                )
            else:
                shard_role = "secondary_mode_partition"
                partitions = (
                    ("low", ["mixed_ecological"], 2700),
                    (
                        "high",
                        [
                            "expr_online",
                            "expr_without",
                            "async_online",
                            "async_without",
                        ],
                        2400,
                    ),
                )
            shards = [
                {
                    "shard_id": f"{worker_id}-{slot}",
                    "physical_slot": slot,
                    "side": side,
                    "shard_role": shard_role,
                    "task_ids": task_ids,
                    "actual_hands": hands,
                    "shard_manifest_sha256": "a" * 64,
                    "completion_receipt_sha256": "b" * 64,
                }
                for slot, (side, task_ids, hands) in zip(
                    slots if role == "S" else reversed(slots),
                    partitions,
                    strict=True,
                )
            ]
            workers.append(
                {
                    "worker_id": worker_id,
                    "pod_id": f"pod{index:02d}",
                    "seed": seed,
                    "expected_worker_manifest_sha256": "c" * 64,
                    "shard_lineage": {
                        "pair_id": f"pair_{index:02d}",
                        "paired_physical_slots": slots,
                        "composition_receipt_path": (
                            f"composition/{worker_id}.json"
                        ),
                        "composition_receipt_sha256": "d" * 64,
                        "shards": shards,
                    },
                }
            )
    return workers


def test_six_seed_contract_freezes_canonical_topology_and_budget() -> None:
    protocol = task8b_analysis._load_six_seed_analysis_protocol()

    assert protocol["seeds"] == tuple(range(2026090107, 2026090113))
    assert protocol["n_planned"] == 6
    assert protocol["planned_hands"] == 71100
    assert protocol["worker_ids"] == tuple(
        f"{role}{index:02d}" for role in ("P", "S") for index in range(7, 13)
    )
    task8b_analysis._validate_analysis_input_topology(
        _six_seed_workers(), protocol
    )


def test_six_seed_contract_requires_pair_and_shard_lineage() -> None:
    protocol = task8b_analysis._load_six_seed_analysis_protocol()
    workers = _six_seed_workers()
    workers[0]["shard_lineage"]["shards"][0].pop("shard_manifest_sha256")  # type: ignore[index]

    with pytest.raises(ConfigError, match="shard lineage"):
        task8b_analysis._validate_analysis_input_topology(workers, protocol)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("primary_endpoint", "posthoc_endpoint"),
        ("primary_checkpoint", 150),
        ("primary_mode", "Online"),
        ("holm_family", ["Expr_vs_Fact"]),
        ("bootstrap", {"cluster": "seed"}),
        ("n_source", "power-verified"),
        (
            "sign_flip_holm_disclosure",
            {
                "alpha": 0.05,
                "minimum_holm_adjusted_p": 0.01,
                "rejection_possible": True,
                "two_sided_exact_minimum_raw_p": 0.01,
            },
        ),
    ),
)
def test_six_seed_contract_loader_rejects_statistical_semantic_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    contract = json.loads(
        task8b_analysis._six_seed_contract_path().read_text(encoding="utf-8")
    )
    contract[field] = value
    path = tmp_path / "contract.json"
    _write_json(path, contract)

    with pytest.raises(ConfigError, match="语义不闭合"):
        task8b_analysis._load_six_seed_analysis_protocol(path)


def test_six_seed_input_rejects_legacy_contract_or_unbound_contract_sha() -> None:
    workers = _six_seed_workers()
    legacy = {
        "analysis_contract_id": task8b_analysis.LEGACY_ANALYSIS_CONTRACT_ID,
    }
    with pytest.raises(ConfigError, match="topology"):
        task8b_analysis._validate_analysis_input_topology(
            workers, task8b_analysis._analysis_protocol_for_manifest(legacy)
        )


def _bind_composition_receipt(
    root: Path, worker: dict[str, object]
) -> None:
    lineage = worker["shard_lineage"]
    assert isinstance(lineage, dict)
    sides = []
    for shard in lineage["shards"]:
        assert isinstance(shard, dict)
        execution_path = (
            root
            / "composition"
            / "execution"
            / f"{worker['worker_id']}-{shard['side']}.json"
        )
        _write_json(execution_path, {"status": "complete"})
        sides.append(
            {
                "side": shard["side"],
                "physical_slot": shard["physical_slot"],
                "task_ids": shard["task_ids"],
                "actual_hands": shard["actual_hands"],
                "execution_receipts": [
                    {
                        "path": f"execution/{execution_path.name}",
                        "sha256": _sha256(execution_path),
                    }
                ],
            }
        )
    composition_path = root / str(lineage["composition_receipt_path"])
    _write_json(
        composition_path,
        {
            "schema_version": "task8b-six-seed-composition-receipt-v1",
            "worker_id": worker["worker_id"],
            "seed": worker["seed"],
            "pair_id": lineage["pair_id"],
            "sides": sides,
            "task_union": [
                task_id for side in sides for task_id in side["task_ids"]
            ],
            "actual_hands": sum(int(side["actual_hands"]) for side in sides),
        },
    )
    lineage["composition_receipt_sha256"] = _sha256(composition_path)


def test_six_seed_composition_receipt_is_read_and_hash_bound(
    tmp_path: Path,
) -> None:
    protocol = task8b_analysis._load_six_seed_analysis_protocol()
    workers = _six_seed_workers()
    worker = workers[0]
    _bind_composition_receipt(tmp_path, worker)

    task8b_analysis._validate_six_seed_shard_lineage(
        worker, protocol, manifest_root=tmp_path
    )
    composition = (
        tmp_path
        / str(worker["shard_lineage"]["composition_receipt_path"])  # type: ignore[index]
    )
    value = json.loads(composition.read_text(encoding="utf-8"))
    value["actual_hands"] = int(value["actual_hands"]) - 1
    _write_json(composition, value)

    with pytest.raises(ConfigError, match="SHA"):
        task8b_analysis._validate_six_seed_shard_lineage(
            worker, protocol, manifest_root=tmp_path
        )


def test_six_seed_table1_and_lineage_use_n_planned_six() -> None:
    protocol = task8b_analysis._load_six_seed_analysis_protocol()
    table1 = task8b_analysis._table1_rows(
        [{"worker_id": "P07"}] * 12, protocol=protocol
    )
    assert (
        next(row for row in table1 if row["field"] == "seed_count")[
            "frozen_value"
        ]
        == "n=6"
    )
    lineage = task8b_analysis._paper_lineage_rows(
        ([{"field": "seed_count", "value": "n=6"}],),
        ("table1_protocol_identity",),
        [],
        n_planned=6,
    )
    assert lineage[0]["n_planned"] == 6


def test_six_seed_preunlock_builder_binds_contract_file_and_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    phase_f = tmp_path / "phase-f"
    for relative in task8b_analysis.PHASE_F_REQUIRED_FILES:
        path = phase_f / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8", newline="")
    for relative in task8b_analysis.PHASE_F_ANALYSIS_CODE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8", newline="")
    source_contract = task8b_analysis._six_seed_contract_path()
    contract = repo / task8b_analysis.SIX_SEED_ANALYSIS_CONTRACT_RELATIVE_PATH
    contract.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_contract, contract)
    lock = repo / "uv.lock"
    lock.write_text("fixture\n", encoding="utf-8", newline="")
    monkeypatch.setattr(task8b_analysis, "_git_clean_head", lambda _repo: "a" * 40)

    payload = task8b_analysis.build_task8b_preunlock_manifest(
        phase_f,
        lock,
        tmp_path / "preunlock.json",
        repository_root=repo,
        frozen_at_utc="2026-07-23T00:00:00+00:00",
        analysis_contract_path=contract,
    )

    assert payload["analysis_contract_id"] == task8b_analysis.SIX_SEED_ANALYSIS_CONTRACT_ID
    assert payload["analysis_contract_sha256"] == _sha256(contract)
    code_paths = {row["relative_path"] for row in payload["analysis_code_files"]}
    assert task8b_analysis.SIX_SEED_ANALYSIS_CONTRACT_RELATIVE_PATH in code_paths


def test_six_seed_preunlock_to_input_selects_contract_workers_from_24(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    phase_f = tmp_path / "phase-f"
    manifests = tmp_path / "worker-manifests"
    snapshots = tmp_path / "snapshots"
    manifests.mkdir()
    _write_json(manifests / "manifest_index.json", {"schema_version": "fixture"})
    for relative in task8b_analysis.PHASE_F_REQUIRED_FILES:
        path = phase_f / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8", newline="")
    for relative in task8b_analysis.PHASE_F_ANALYSIS_CODE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8", newline="")
    contract = repo / task8b_analysis.SIX_SEED_ANALYSIS_CONTRACT_RELATIVE_PATH
    contract.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task8b_analysis._six_seed_contract_path(), contract)
    lock = repo / "uv.lock"
    lock.write_text("fixture\n", encoding="utf-8", newline="")
    monkeypatch.setattr(task8b_analysis, "_git_clean_head", lambda _repo: "a" * 40)
    preunlock = tmp_path / "preunlock.json"
    task8b_analysis.build_task8b_preunlock_manifest(
        phase_f,
        lock,
        preunlock,
        repository_root=repo,
        frozen_at_utc="2026-07-23T00:00:00+00:00",
        analysis_contract_path=contract,
    )
    selected = {
        str(worker["worker_id"]): worker for worker in _six_seed_workers()
    }
    for index in range(1, 13):
        for role in ("P", "S"):
            worker_id = f"{role}{index:02d}"
            value = {
                "protocol_status": "frozen/expedited-formal-candidate",
                "worker_id": worker_id,
                "pod_id": f"pod{index:02d}",
                "seed_bundle": 2026090100 + index,
                "common_identity": WORKER_IDENTITY,
            }
            if worker_id in selected:
                worker = selected[worker_id]
                _bind_composition_receipt(manifests, worker)
                value["phase_f_shard_lineage"] = worker["shard_lineage"]
                (snapshots / worker_id / str(worker["seed"]) / "attempt_01").mkdir(
                    parents=True
                )
            _write_json(manifests / f"{worker_id}.json", value)

    payload = task8b_analysis.build_task8b_analysis_input(
        manifests,
        snapshots,
        tmp_path / "analysis-input.json",
        preunlock,
        repository_root=repo,
        phase_f_dir=phase_f,
        dependency_lock_path=lock,
        analysis_contract_path=contract,
    )

    assert [row["worker_id"] for row in payload["workers"]] == sorted(selected)
    assert payload["analysis_contract_sha256"] == _sha256(contract)

    with pytest.raises(ConfigError, match="拒绝旧合同误跑"):
        task8b_analysis._analysis_protocol_for_manifest(
            {
                "analysis_contract_id": (
                    task8b_analysis.SIX_SEED_ANALYSIS_CONTRACT_ID
                ),
                "analysis_contract_sha256": "0" * 64,
            }
        )
