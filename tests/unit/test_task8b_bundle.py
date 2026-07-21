from __future__ import annotations

import json
from pathlib import Path

import yaml

from agentmemeval.evaluation.aggregation import (
    validate_runtime_homogeneity,
    validate_task8b_runtime_homogeneity,
)
from agentmemeval.evaluation.runtime_lock import runtime_identity_from_metadata
from agentmemeval.experiments.admission import _runtime_lock_blockers
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.task8b_bundle import build_task8b_executable_bundle
from tests.unit.test_formal_freeze import _runtime_manifest


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


def test_task8b_runtime_driver_is_informational_but_required_for_health_audit() -> None:
    runtime = _runtime_manifest()["metadata"]
    observed = runtime_identity_from_metadata(runtime)
    lock = dict(observed)
    lock.pop("gpu_driver")
    lock["gpu_driver_policy"] = "informational_only"
    experiment = {"formal_runtime_lock": lock}

    for driver in ("595.58.03", "595.71.05", "580.105.08", "future-driver"):
        runtime["gpu"]["devices"][0]["driver"] = driver
        assert _runtime_lock_blockers(experiment, runtime) == []

    runtime["gpu"]["devices"][0]["driver"] = ""
    assert _runtime_lock_blockers(experiment, runtime) == [
        "runtime gpu_driver is required for informational recording"
    ]

    runtime["gpu"]["devices"][0]["driver"] = "580.105.08"
    runtime["model_service_runtime"]["status"] = "failed"
    assert _runtime_lock_blockers(experiment, runtime)[0] == (
        "model service runtime probe must be verified"
    )
    runtime["model_service_runtime"]["status"] = "verified"
    runtime["model_service_runtime"]["vllm_version"] = "different"
    assert "runtime vllm_version mismatch" in _runtime_lock_blockers(
        experiment, runtime
    )[0]


def test_task8b_base_marks_driver_as_informational_only() -> None:
    base = yaml.safe_load(
        Path("configs/formal/task8b_expedited_base.yaml").read_text(encoding="utf-8")
    )
    runtime_lock = base["experiment"]["formal_runtime_lock"]

    assert runtime_lock["gpu_driver_policy"] == "informational_only"
    assert "gpu_driver" not in runtime_lock


def test_task8b_aggregation_ignores_driver_and_pci_but_not_gpu_or_runtime() -> None:
    def manifest(
        *, gpu: str, driver: str, pci: str, vllm: str = "0.23.1"
    ) -> dict[str, object]:
        return {
            "metadata": {
                "code": {"commit": "same", "dirty": False},
                "gpu": {
                    "devices": [
                        {"name": gpu, "driver": driver, "pci_bus_id": pci}
                    ]
                },
                "model_service_runtime": {
                    "torch_cuda_version": "13.0",
                    "vllm_version": vllm,
                },
                "model": {"name": "qwen", "revision": "r", "weights_hash": "w"},
                "service": {"startup": "same"},
                "embedding": {"name": "bge", "revision": "r"},
                "prompts": {"decision_system_sha256": "p"},
            }
        }

    first = manifest(gpu="RTX 5090", driver="595.58.03", pci="0000:01:00.0")
    second = manifest(gpu="RTX 5090", driver="580.105.08", pci="0000:81:00.0")
    assert validate_task8b_runtime_homogeneity([first, second])["homogeneous"] is True
    assert validate_runtime_homogeneity([first, second])["homogeneous"] is False

    wrong_gpu = manifest(gpu="RTX 4090", driver="580.105.08", pci="0000:81:00.0")
    assert validate_task8b_runtime_homogeneity([first, wrong_gpu])["mismatches"] == {
        "gpu": [("RTX 5090",), ("RTX 4090",)]
    }
    wrong_vllm = manifest(
        gpu="RTX 5090",
        driver="580.105.08",
        pci="0000:81:00.0",
        vllm="different",
    )
    assert "vllm_runtime" in validate_task8b_runtime_homogeneity(
        [first, wrong_vllm]
    )["mismatches"]


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
