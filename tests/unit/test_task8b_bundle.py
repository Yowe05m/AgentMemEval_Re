from __future__ import annotations

import json
from pathlib import Path

from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.task8b_bundle import build_task8b_executable_bundle


def _identity(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "code_sha": "a" * 40,
                "prompt_sha256": "b" * 64,
                "model_fingerprint": "c" * 64,
                "embedding_fingerprint": "d" * 64,
                "protocol_sha256": "e" * 64,
                "runtime_image_fingerprint": "f" * 64,
                "resolved_config_sha256": "legacy",
                "schedule_sha256": "legacy",
            }
        ),
        encoding="utf-8",
    )
    return path


def _matrix(path: Path) -> Path:
    path.write_text(
        "experiment_id,status,checkpoint_set,heldout_tables,memory_mode\n"
        "R1-E1-I,required,30|75|150|300,3,Frozen\n"
        "R1-E1-M,required,,,Frozen\n"
        "R1-E2,required,,,Frozen\n"
        "R1-E3,required,,,Frozen\n"
        "R1-E4,required,,,Online\n"
        "R1-E5,required,,,Without\n"
        "R1-E6,required,,,derived\n",
        encoding="utf-8",
    )
    return path


def test_formal_bundle_recomputes_complete_142200_hand_matrix(tmp_path: Path) -> None:
    result = build_task8b_executable_bundle(
        matrix_path=_matrix(tmp_path / "matrix.csv"),
        base_config_path="configs/formal/task8b_expedited_base.yaml",
        fleet_identity_path=_identity(tmp_path / "identity.json"),
        output_dir=tmp_path / "bundle",
        runtime_bundle_root="/root/autodl-tmp/task8b_bundle_test",
    )

    assert result["planned_hands"] == 142_200
    assert result["worker_count"] == 24
    manifest_dir = tmp_path / "bundle" / "manifests"
    primary = json.loads((manifest_dir / "P01.json").read_text(encoding="utf-8"))
    secondary = json.loads((manifest_dir / "S01.json").read_text(encoding="utf-8"))
    assert sum(task["planned_hands"] for task in primary["task_configs"]) == 6750
    assert sum(task["planned_hands"] for task in secondary["task_configs"]) == 5100
    assert primary["receipt_identity"] == secondary["dependency_receipt_identity"]
    assert primary["seed_pod_identity"] == secondary["seed_pod_identity"]
    assert len(primary["seed_pod_identity"]["task_schedules"]) == 10


def test_canary_bundle_is_real_two_worker_and_under_100_hands(tmp_path: Path) -> None:
    result = build_task8b_executable_bundle(
        matrix_path=_matrix(tmp_path / "matrix.csv"),
        base_config_path="configs/formal/task8b_expedited_base.yaml",
        fleet_identity_path=_identity(tmp_path / "identity.json"),
        output_dir=tmp_path / "canary",
        runtime_bundle_root="/root/autodl-tmp/task8b_canary_test",
        canary_seed=2026090199,
    )

    assert result["protocol_status"] == "canary/not-for-paper"
    assert result["worker_count"] == 2
    assert result["planned_hands"] <= 100
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "canary" / "manifests").glob("[PS]01.json"))
    ]
    assert sum(item["worker_planned_hands"] for item in manifests) == result["planned_hands"]
    assert all(item["checkpoint_set"] == [1, 3, 5] for item in manifests)
    by_role = {item["role"]: item for item in manifests}
    schedule_rows = [
        {
            "worker_role": role,
            "task_id": task["task_id"],
            "schedule_sha256": task["expected_identity"]["schedule_sha256"],
        }
        for role in ("primary", "secondary")
        for task in by_role[role]["task_configs"]
    ]
    expected_pod_identity = {
        "seed_bundle": 2026090199,
        "schedule_sha256": sha256_json(
            {
                "schema_version": "task8b-seed-pod-schedule-bundle-v1",
                "seed_bundle": 2026090199,
                "task_schedules": schedule_rows,
            }
        ),
        "task_schedules": schedule_rows,
    }
    assert by_role["primary"]["seed_pod_identity"] == expected_pod_identity
    assert by_role["secondary"]["seed_pod_identity"] == expected_pod_identity


def test_bundle_is_byte_identical_for_same_frozen_inputs(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path / "matrix.csv")
    identity = _identity(tmp_path / "identity.json")
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        build_task8b_executable_bundle(
            matrix_path=matrix,
            base_config_path="configs/formal/task8b_expedited_base.yaml",
            fleet_identity_path=identity,
            output_dir=root,
            runtime_bundle_root="/root/autodl-tmp/task8b_bundle_test",
        )

    first = {
        path.relative_to(roots[0]).as_posix(): path.read_bytes()
        for path in sorted(roots[0].rglob("*"))
        if path.is_file()
    }
    second = {
        path.relative_to(roots[1]).as_posix(): path.read_bytes()
        for path in sorted(roots[1].rglob("*"))
        if path.is_file()
    }
    assert first == second
