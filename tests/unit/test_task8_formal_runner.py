from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import (
    build_heldout_schedule_manifest,
    sha256_json,
)
from agentmemeval.experiments.formal_runner import (
    append_worker_state,
    generate_worker_manifests,
    publish_checkpoint_receipt,
    run_worker_manifest,
    summarize_worker_states,
    validate_worker_manifest_set,
    verify_checkpoint_receipt,
)


def _identity() -> dict[str, str]:
    return {
        "code_sha": "a" * 40,
        "resolved_config_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "model_fingerprint": "mock-model-v1",
        "embedding_fingerprint": "mock-embedding-v1",
        "schedule_sha256": "d" * 64,
    }


def _matrix(path: Path) -> Path:
    path.write_text(
        "experiment_id,status,checkpoint_set,heldout_tables,memory_mode\n"
        "R1-E1-I,planned,30|75|150|300,3,Frozen\n"
        "R1-E1-M,planned,300,3,Frozen\n"
        "R1-E2,planned,30|75|150|300,3,Frozen\n"
        "R1-E3,planned,all,3,Frozen\n"
        "R1-E4,planned,300,3,Online\n"
        "R1-E5,planned,300,3,Without\n",
        encoding="utf-8",
    )
    return path


def _load_manifests(directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("[PS][0-9][0-9].json"))
    ]


def test_central_generator_makes_stable_unique_24_worker_manifests(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    seeds = list(range(2026090101, 2026090113))
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = generate_worker_manifests(
        matrix_path=matrix,
        seeds=seeds,
        common_identity=_identity(),
        output_dir=first,
    )
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=seeds,
        common_identity=_identity(),
        output_dir=second,
    )
    assert result["worker_count"] == 24
    assert [path.read_bytes() for path in sorted(first.iterdir())] == [
        path.read_bytes() for path in sorted(second.iterdir())
    ]
    manifests = _load_manifests(first)
    assert len(manifests) == 24
    assert len({m["instance_identity"]["output_path"] for m in manifests}) == 24
    assert len({m["instance_identity"]["cache_namespace"] for m in manifests}) == 24
    assert all(
        "candidate/not-frozen/not-authorized-to-run" == m["protocol_status"]
        for m in manifests
    )


def test_manifest_duplicate_and_dependency_cycle_fail_closed(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    directory = tmp_path / "manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[1, 2],
        common_identity=_identity(),
        output_dir=directory,
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    manifests = _load_manifests(directory)
    duplicate = copy.deepcopy(manifests)
    duplicate[1]["instance_identity"]["output_path"] = duplicate[0]["instance_identity"][
        "output_path"
    ]
    with pytest.raises(ConfigError, match="重复 output_path"):
        validate_worker_manifest_set(duplicate, expected_seed_count=2)
    cycle = copy.deepcopy(manifests)
    by_id = {item["worker_id"]: item for item in cycle}
    by_id["P01"]["depends_on"] = "S01"
    with pytest.raises(ConfigError, match="依赖环"):
        validate_worker_manifest_set(cycle, expected_seed_count=2)


def test_receipt_verifies_hash_identity_and_path_safety(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "snapshot.json").write_text('{"closed":true}\n', encoding="utf-8")
    receipt_path = tmp_path / "receipts" / "P01.json"
    receipt = publish_checkpoint_receipt(
        checkpoint_root=checkpoint,
        checkpoint_files=["snapshot.json"],
        receipt_path=receipt_path,
        producer_worker_id="P01",
        seed_bundle=101,
        checkpoint_hand=300,
        identity=_identity(),
    )
    assert receipt["status"] == "complete"
    verify_checkpoint_receipt(receipt_path, checkpoint, expected_identity=_identity())
    (checkpoint / "snapshot.json").write_text('{"closed":false}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="哈希不匹配"):
        verify_checkpoint_receipt(receipt_path, checkpoint, expected_identity=_identity())
    with pytest.raises(ConfigError, match="安全相对路径"):
        publish_checkpoint_receipt(
            checkpoint_root=checkpoint,
            checkpoint_files=["../escape.json"],
            receipt_path=tmp_path / "escape-receipt.json",
            producer_worker_id="P01",
            seed_bundle=101,
            checkpoint_hand=300,
            identity=_identity(),
        )


def test_append_only_state_rejects_invalid_transition(tmp_path: Path) -> None:
    state = tmp_path / "state.tsv"
    append_worker_state(state, "planned")
    append_worker_state(state, "validating")
    append_worker_state(state, "running")
    append_worker_state(state, "partial")
    append_worker_state(state, "validating")
    with pytest.raises(ConfigError, match="非法 worker 状态迁移"):
        append_worker_state(state, "complete")
    assert state.read_text(encoding="utf-8").count("task8-worker-state-v1") == 5


def test_candidate_admission_fails_before_output_directory_creation(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    manifests_dir = tmp_path / "candidate"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=list(range(1, 13)),
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=str(tmp_path / "must_not_exist"),
    )
    with pytest.raises(ConfigError, match="未获运行授权"):
        run_worker_manifest(
            manifests_dir / "P01.json", receipt_root=tmp_path / "receipts"
        )
    assert not (tmp_path / "must_not_exist").exists()


def test_two_worker_mock_seed_pod_receipt_isolation_and_resume(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    receipt_root = tmp_path / "pod"
    manifests_dir = tmp_path / "mock_manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[101],
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=(receipt_root / "workers").as_posix(),
        cache_root="task8-mock",
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    primary = run_worker_manifest(
        manifests_dir / "P01.json", receipt_root=receipt_root
    )
    secondary = run_worker_manifest(
        manifests_dir / "S01.json", receipt_root=receipt_root
    )
    assert primary["status"] == secondary["status"] == "complete"
    primary_dir = Path(primary["run_dir"])
    secondary_dir = Path(secondary["run_dir"])
    assert (receipt_root / "receipts" / "P01.json").exists()
    assert (secondary_dir / "branches" / "Frozen.json").exists()
    assert (secondary_dir / "branches" / "Online.json").exists()
    without = json.loads(
        (secondary_dir / "branches" / "Without.json").read_text(encoding="utf-8")
    )
    assert without["payload"]["fact"]["records"] == []
    assert without["payload"]["expr"]["history"] == []
    assert json.loads(
        (secondary_dir / "completion_receipt.json").read_text(encoding="utf-8")
    )["not_for_paper"] is True
    resumed = run_worker_manifest(
        manifests_dir / "S01.json",
        receipt_root=receipt_root,
        resume_existing=True,
    )
    assert resumed == {"status": "complete", "run_dir": str(secondary_dir), "resumed": True}
    summary = summarize_worker_states(receipt_root / "workers")
    assert summary["status_counts"] == {"complete": 2}
    assert primary_dir.name == "101"
    retry = run_worker_manifest(manifests_dir / "S01.json", receipt_root=receipt_root)
    assert Path(retry["run_dir"]).name == "101__attempt_02"


def test_secondary_records_waiting_until_complete_receipt_exists(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    receipt_root = tmp_path / "pod"
    manifests_dir = tmp_path / "mock_manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[101],
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=(receipt_root / "workers").as_posix(),
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    waiting = run_worker_manifest(manifests_dir / "S01.json", receipt_root=receipt_root)
    assert waiting["status"] == "waiting_dependency"
    assert summarize_worker_states(receipt_root / "workers")["status_counts"] == {
        "waiting_dependency": 1
    }
    run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    completed = run_worker_manifest(
        manifests_dir / "S01.json", receipt_root=receipt_root, resume_existing=True
    )
    assert completed["status"] == "complete"


def test_resume_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    receipt_root = tmp_path / "pod"
    manifests_dir = tmp_path / "mock_manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[101],
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=(receipt_root / "workers").as_posix(),
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    changed = json.loads((manifests_dir / "P01.json").read_text(encoding="utf-8"))
    changed["common_identity"]["prompt_sha256"] = "e" * 64
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ConfigError, match="identity mismatch"):
        run_worker_manifest(
            changed_path, receipt_root=receipt_root, resume_existing=True
        )


def test_experiment_config_runner_releases_and_consumes_real_snapshots(
    tmp_path: Path,
) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    receipt_root = tmp_path / "pod"
    manifests_dir = tmp_path / "manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[101],
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=(receipt_root / "workers").as_posix(),
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "provider": {"provider": "mock", "model": "mock-deterministic-v1"},
                "table": {
                    "starting_stack": 100,
                    "small_blind": 1,
                    "big_blind": 2,
                    "max_raises_per_street": 3,
                    "lifecycle": "continuous_rebuy",
                },
                "agent": {
                    "mechanism": "expr",
                    "memory_scope": "per_agent",
                    "window_size": 2,
                    "update_period": 1,
                },
                "opponent_agent": {"mechanism": "no_memory", "memory_scope": "per_agent"},
                "heldout_agent": {"mechanism": "no_memory", "memory_scope": "per_agent"},
                "experiment": {
                    "scenario": "fixed_evolving_table",
                    "run_mode": "smoke",
                    "seed": 101,
                    "train_hands": 5,
                    "test_hands": 1,
                    "checkpoint_set": [1, 3, 5],
                    "checkpoint_test_hands": 1,
                    "checkpoint_test_hands_by_checkpoint": {1: 1, 3: 1, 5: 1},
                    "table_size": 3,
                    "target_agent_id": "agent_00",
                    "update_memory_train": True,
                    "update_memory_test": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    primary_path = manifests_dir / "P01.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["execution_mode"] = "experiment_configs"
    roster_identity = sha256_json(
        {"mechanism": "no_memory", "memory_scope": "per_agent"}
    )
    primary_schedule = build_heldout_schedule_manifest(
        root_seed=101,
        checkpoint_set=[1, 3, 5],
        table_set=["H01", "H02", "H03"],
        hands_by_checkpoint={1: 1, 3: 1, 5: 1},
        table_size=3,
        roster_identity=roster_identity,
    )["schedule_sha256"]
    primary["task_configs"] = [
        {
            "task_id": "train",
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "schedule_sha256": primary_schedule,
            "publish_checkpoint_after": True,
        }
    ]
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    primary_result = run_worker_manifest(primary_path, receipt_root=receipt_root)
    assert primary_result["status"] == "complete"

    secondary_path = manifests_dir / "S01.json"
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    secondary["execution_mode"] = "experiment_configs"
    secondary_schedule = build_heldout_schedule_manifest(
        root_seed=101,
        checkpoint_set=[5],
        table_set=["H01", "H02", "H03"],
        hands_by_checkpoint={5: 1},
        table_size=3,
        roster_identity=roster_identity,
    )["schedule_sha256"]
    checkpoint_relative = "runs/train/memory_snapshots/agent_00_checkpoint_0005.json"
    secondary["task_configs"] = [
        {
            "task_id": mode.lower(),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "schedule_sha256": secondary_schedule,
            "memory_mode": mode,
            "checkpoint_bindings": {"agent_00": checkpoint_relative},
        }
        for mode in ("Frozen", "Online", "Without")
    ]
    secondary_path.write_text(json.dumps(secondary), encoding="utf-8")
    secondary_result = run_worker_manifest(secondary_path, receipt_root=receipt_root)
    assert secondary_result["status"] == "complete"
    secondary_dir = Path(secondary_result["run_dir"])
    rows = json.loads((secondary_dir / "task_results.json").read_text(encoding="utf-8"))
    assert [row["memory_mode"] for row in rows["tasks"]] == [
        "Frozen",
        "Online",
        "Without",
    ]
    for mode in ("frozen", "online", "without"):
        audit = json.loads(
            (secondary_dir / "runs" / mode / "clone_transform_audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert audit["memory_mode"].lower() == mode
