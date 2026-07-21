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
    generate_worker_manifests,
    publish_checkpoint_receipt,
    run_worker_manifest,
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


def _mock_seed_pod(tmp_path: Path) -> tuple[Path, Path]:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "experiment_id,status,checkpoint_set,heldout_tables,memory_mode\n"
        "R1-E1-I,planned,30|75|150|300,3,Frozen\n"
        "R1-E1-M,planned,,,Frozen\n"
        "R1-E2,planned,,,Frozen\n"
        "R1-E3,planned,,,Frozen\n"
        "R1-E4,planned,,,Online\n"
        "R1-E5,planned,,,Without\n",
        encoding="utf-8",
    )
    receipt_root = tmp_path / "pod"
    manifests_dir = tmp_path / "manifests"
    generate_worker_manifests(
        matrix_path=matrix,
        seeds=[101],
        common_identity=_identity(),
        output_dir=manifests_dir,
        output_root=(receipt_root / "workers").as_posix(),
        cache_root="task8a-adversarial",
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    return manifests_dir, receipt_root


def _load_manifest_set(directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("[PS][0-9][0-9].json"))
    ]


def _experiment_config(path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    rosters = {
        "H01": {"mechanism": "no_memory", "roster": "natural-a"},
        "H02": {"mechanism": "no_memory", "roster": "natural-b"},
        "H03": {"mechanism": "no_memory", "roster": "natural-c"},
    }
    path.write_text(
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
                "opponent_agent": {
                    "mechanism": "no_memory",
                    "memory_scope": "per_agent",
                },
                "heldout_agent": {
                    "mechanism": "no_memory",
                    "memory_scope": "per_agent",
                },
                "experiment": {
                    "scenario": "fixed_evolving_table",
                    "run_mode": "smoke",
                    "seed": 101,
                    "train_hands": 5,
                    "test_hands": 1,
                    "checkpoint_test_hands": 1,
                    "table_size": 3,
                    "target_agent_id": "agent_00",
                    "update_memory_train": True,
                    "update_memory_test": False,
                    "heldout_table_set": ["H01", "H02", "H03"],
                    "heldout_table_rosters": rosters,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path, rosters


def _set_experiment_tasks(
    manifest_path: Path,
    config_path: Path,
    schedule_sha256: str,
    task_ids: tuple[str, ...],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "experiment_configs"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest["task_configs"] = [
        {
            "task_id": task_id,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "schedule_sha256": schedule_sha256,
            "publish_checkpoint_after": index == 0,
        }
        for index, task_id in enumerate(task_ids)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_receipt_rejects_tampered_seed_bundle_in_two_worker_dependency(
    tmp_path: Path,
) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    receipt_path = receipt_root / "receipts" / "P01.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["seed_bundle"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ConfigError, match="seed_bundle"):
        run_worker_manifest(manifests_dir / "S01.json", receipt_root=receipt_root)

    assert not (receipt_root / "workers" / "S01" / "101").exists()


def test_receipt_rejects_duplicate_checkpoint_file_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "snapshot.json").write_text('{"closed":true}\n', encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    publish_checkpoint_receipt(
        checkpoint_root=checkpoint,
        checkpoint_files=["snapshot.json"],
        receipt_path=receipt_path,
        producer_worker_id="P01",
        seed_bundle=101,
        checkpoint_hand=300,
        identity=_identity(),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoint_files"].append(copy.deepcopy(receipt["checkpoint_files"][0]))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ConfigError, match="checkpoint"):
        verify_checkpoint_receipt(
            receipt_path,
            checkpoint,
            expected_identity=_identity(),
            expected_producer_worker_id="P01",
        )


def test_resume_rejects_tampered_state_even_when_manifest_identity_matches(
    tmp_path: Path,
) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    result = run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    run_dir = Path(result["run_dir"])
    with (run_dir / "state.tsv").open("a", encoding="utf-8") as handle:
        handle.write("task8-worker-state-v1\tforged\tcomplete\tforged\n")

    with pytest.raises(ConfigError, match="state.tsv hash chain mismatch"):
        run_worker_manifest(
            manifests_dir / "P01.json",
            receipt_root=receipt_root,
            resume_existing=True,
        )


def test_resume_rejects_files_manifest_hash_mismatch(tmp_path: Path) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    result = run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    run_dir = Path(result["run_dir"])
    with (run_dir / "files.tsv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ConfigError, match="files.tsv.*SHA-256"):
        run_worker_manifest(
            manifests_dir / "P01.json",
            receipt_root=receipt_root,
            resume_existing=True,
        )


def test_resume_rejects_tampered_persisted_worker_identity(tmp_path: Path) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    result = run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    run_dir = Path(result["run_dir"])
    persisted_path = run_dir / "worker_manifest.json"
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    persisted["common_identity"]["prompt_sha256"] = "e" * 64
    persisted_path.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ConfigError, match="identity mismatch"):
        run_worker_manifest(
            manifests_dir / "P01.json",
            receipt_root=receipt_root,
            resume_existing=True,
        )


@pytest.mark.parametrize("duplicate_field", ["worker_id", "cache_namespace"])
def test_manifest_set_rejects_duplicate_worker_or_cache_identity(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    manifests_dir, _ = _mock_seed_pod(tmp_path)
    manifests = _load_manifest_set(manifests_dir)
    duplicate = copy.deepcopy(manifests)
    if duplicate_field == "worker_id":
        duplicate[1]["worker_id"] = duplicate[0]["worker_id"]
        duplicate[1]["instance_identity"]["worker_id"] = duplicate[0]["worker_id"]
    else:
        duplicate[1]["instance_identity"][duplicate_field] = duplicate[0][
            "instance_identity"
        ][duplicate_field]

    with pytest.raises(ConfigError, match=duplicate_field):
        validate_worker_manifest_set(duplicate, expected_seed_count=1)


def test_secondary_dependency_failure_does_not_create_worker_output(tmp_path: Path) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    receipt_path = receipt_root / "receipts" / "P01.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "partial"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ConfigError, match="complete receipt"):
        run_worker_manifest(manifests_dir / "S01.json", receipt_root=receipt_root)

    assert not (receipt_root / "workers" / "S01" / "101").exists()


def test_per_table_rosters_preflight_schedule_matches_runtime(tmp_path: Path) -> None:
    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    config_path, rosters = _experiment_config(tmp_path / "experiment.yaml")
    schedule = build_heldout_schedule_manifest(
        root_seed=101,
        checkpoint_set=[1, 3, 5],
        table_set=["H01", "H02", "H03"],
        hands_by_checkpoint={1: 1, 3: 1, 5: 1},
        table_size=3,
        roster_identity={
            table_id: sha256_json(roster) for table_id, roster in rosters.items()
        },
    )
    _set_experiment_tasks(
        manifests_dir / "P01.json",
        config_path,
        schedule["schedule_sha256"],
        ("train",),
    )

    result = run_worker_manifest(manifests_dir / "P01.json", receipt_root=receipt_root)
    runtime_schedule = json.loads(
        (Path(result["run_dir"]) / "runs" / "train" / "schedule_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_schedule["schedule_sha256"] == schedule["schedule_sha256"]


def test_secondary_dependency_output_path_closes_same_seed_pod(tmp_path: Path) -> None:
    manifests_dir, _ = _mock_seed_pod(tmp_path)
    manifests = _load_manifest_set(manifests_dir)
    by_role = {str(item["role"]): item for item in manifests}
    primary_output = by_role["primary"]["instance_identity"]["output_path"]
    assert by_role["secondary"]["dependency_output_path"] == primary_output

    mismatched = copy.deepcopy(manifests)
    secondary = next(item for item in mismatched if item["role"] == "secondary")
    secondary["dependency_output_path"] = str(primary_output) + "__wrong_seed"
    with pytest.raises(ConfigError, match="producer output path"):
        validate_worker_manifest_set(mismatched, expected_seed_count=1)


def test_experiment_worker_resume_skips_complete_child_retries_partial_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentmemeval.experiments import runner as experiment_runner

    manifests_dir, receipt_root = _mock_seed_pod(tmp_path)
    config_path, rosters = _experiment_config(tmp_path / "experiment.yaml")
    schedule = build_heldout_schedule_manifest(
        root_seed=101,
        checkpoint_set=[1, 3, 5],
        table_set=["H01", "H02", "H03"],
        hands_by_checkpoint={1: 1, 3: 1, 5: 1},
        table_size=3,
        roster_identity={
            table_id: sha256_json(roster) for table_id, roster in rosters.items()
        },
    )
    manifest_path = manifests_dir / "P01.json"
    _set_experiment_tasks(
        manifest_path,
        config_path,
        schedule["schedule_sha256"],
        ("first", "second"),
    )
    original_run = experiment_runner.run_resolved_config
    calls = 0

    def interrupt_second(config: dict[str, object]):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_run(config)
        experiment = config["experiment"]
        partial = Path(str(experiment["output_root"])) / str(experiment["run_id"])
        partial.mkdir(parents=True)
        (partial / "partial.txt").write_text("interrupted\n", encoding="utf-8")
        raise RuntimeError("synthetic child interruption")

    monkeypatch.setattr(experiment_runner, "run_resolved_config", interrupt_second)
    with pytest.raises(RuntimeError, match="synthetic child interruption"):
        run_worker_manifest(manifest_path, receipt_root=receipt_root)

    run_dir = receipt_root / "workers" / "P01" / "101"
    first_receipt = run_dir / "task_receipts" / "first.json"
    original_receipt = first_receipt.read_bytes()
    tampered = json.loads(original_receipt.decode("utf-8"))
    tampered["config_sha256"] = "0" * 64
    first_receipt.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ConfigError, match="receipt identity mismatch"):
        run_worker_manifest(
            manifest_path,
            receipt_root=receipt_root,
            resume_existing=True,
        )

    first_receipt.write_bytes(original_receipt)
    monkeypatch.setattr(experiment_runner, "run_resolved_config", original_run)
    resumed = run_worker_manifest(
        manifest_path,
        receipt_root=receipt_root,
        resume_existing=True,
    )
    assert resumed["status"] == "complete"
    assert (run_dir / "runs" / "first" / "experiment_result.json").is_file()
    assert (run_dir / "runs" / "second" / "partial.txt").is_file()
    assert (run_dir / "runs" / "second__attempt_02" / "experiment_result.json").is_file()
