from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_runner import (
    append_worker_state,
    publish_checkpoint_receipt,
)
from agentmemeval.experiments.task8b_transport import (
    archive_completed_worker,
    transfer_checkpoint_bundle,
    validate_worker_for_archive,
)

IDENTITY = {
    "code_sha": "a" * 40,
    "resolved_config_sha256": "b" * 64,
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


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "producer-checkpoint"
    (root / "snapshots").mkdir(parents=True)
    (root / "snapshots" / "memory.json").write_text(
        '{"closed":true}\n', encoding="utf-8", newline=""
    )
    (root / "checkpoint_index.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8", newline=""
    )
    receipt = tmp_path / "producer-receipts" / "P01.json"
    publish_checkpoint_receipt(
        checkpoint_root=root,
        checkpoint_files=["snapshots/memory.json", "checkpoint_index.json"],
        receipt_path=receipt,
        producer_worker_id="P01",
        seed_bundle=2026090101,
        checkpoint_hand=300,
        identity=IDENTITY,
    )
    return root, receipt


def _transfer(tmp_path: Path) -> dict[str, object]:
    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    return transfer_checkpoint_bundle(
        source_receipt=source_receipt,
        source_checkpoint_root=source_root,
        destination_checkpoint_root=tmp_path / "consumer" / "checkpoint",
        destination_receipt=tmp_path / "consumer" / "receipts" / "P01.json",
        expected_identity=IDENTITY,
        expected_producer_worker_id="P01",
        expected_seed_bundle=2026090101,
        expected_checkpoint_hand=300,
    )


def _worker_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "worker" / "2026090101"
    child = root / "runs" / "task__attempt_02"
    audit = child / "protocol_audit.json"
    payload = child / "hands.jsonl"
    config_sha256 = "e" * 64
    task_row = {
        "task_id": "task",
        "status": "complete",
        "run_dir": "runs/task__attempt_02",
    }
    _write_json(
        root / "worker_manifest.json",
        {
            "schema_version": "task8-worker-manifest-v1",
            "worker_id": "P01",
            "execution_mode": "experiment_configs",
            "task_configs": [
                {
                    "task_id": "task",
                    "config_sha256": config_sha256,
                    "expected_identity": IDENTITY,
                }
            ],
        },
    )
    _write_json(
        root / "task_results.json",
        {
            "schema_version": "task8-worker-task-results-v1",
            "worker_id": "P01",
            "tasks": [task_row],
        },
    )
    for status in ("planned", "validating", "running", "finalizing", "complete"):
        append_worker_state(root / "state.tsv", status, "archive fixture")
    _write_json(
        audit,
        {
            "status": "verified",
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
    _write_json(
        child / "task_identity_audit.json",
        {
            "schema_version": "task8-task-identity-audit-v1",
            "task_id": "task",
            "status": "verified",
            "actual": IDENTITY,
        },
    )
    payload.write_text('{"hand_id":1}\n', encoding="utf-8", newline="")
    child_files = [
        {
            "relative_path": path.relative_to(child).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in child.rglob("*") if item.is_file())
    ]
    _write_json(
        root / "task_receipts" / "task.json",
        {
            "schema_version": "task8-worker-task-receipt-v1",
            "task_id": "task",
            "config_sha256": config_sha256,
            "run_dir": "runs/task__attempt_02",
            "task_row": task_row,
            "files": child_files,
        },
    )
    listed = [
        root / "worker_manifest.json",
        root / "task_results.json",
        root / "state.tsv",
        audit,
        child / "task_identity_audit.json",
        payload,
        root / "task_receipts" / "task.json",
    ]
    with (root / "files.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("relative_path", "size", "sha256"))
        for path in listed:
            writer.writerow(
                (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
            )
    _write_json(
        root / "completion_receipt.json",
        {
            "schema_version": "task8-worker-completion-v1",
            "worker_id": "P01",
            "status": "complete",
            "files_tsv_sha256": _sha256(root / "files.tsv"),
        },
    )
    return root


def _refresh_worker_manifest(root: Path) -> None:
    files_path = root / "files.tsv"
    rows = list(csv.DictReader(files_path.open("r", encoding="utf-8"), delimiter="\t"))
    for row in rows:
        path = root / row["relative_path"]
        row["size"] = str(path.stat().st_size)
        row["sha256"] = _sha256(path)
    with files_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completion_path = root / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(completion_path, completion)


def _append_listed_files(root: Path, paths: list[Path]) -> None:
    files_path = root / "files.tsv"
    with files_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for path in paths:
            writer.writerow(
                (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
            )
    completion = json.loads((root / "completion_receipt.json").read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(root / "completion_receipt.json", completion)


def _refresh_task_receipt(root: Path, task_id: str = "task") -> None:
    receipt_path = root / "task_receipts" / f"{task_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    child = root / receipt["run_dir"]
    receipt["files"] = [
        {
            "relative_path": path.relative_to(child).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in child.rglob("*") if item.is_file())
    ]
    _write_json(receipt_path, receipt)
    _refresh_worker_manifest(root)


def test_checkpoint_transport_copies_all_files_before_publishing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentmemeval.experiments import task8b_transport

    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    frozen_receipt = source_receipt.read_bytes()
    checkpoint_target = tmp_path / "consumer" / "checkpoint"
    receipt_target = tmp_path / "consumer" / "receipts" / "P01.json"
    links: list[tuple[Path, Path]] = []
    original_link = task8b_transport.os.link

    def record_link(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == receipt_target.absolute()
        assert not destination_path.exists()
        assert source_path.name == ".P01.json.partial"
        assert source_path.read_bytes() == frozen_receipt
        assert (checkpoint_target / "snapshots" / "memory.json").is_file()
        assert (checkpoint_target / "checkpoint_index.json").is_file()
        links.append((source_path, destination_path))
        original_link(source, destination)

    monkeypatch.setattr(task8b_transport.os, "link", record_link)

    result = transfer_checkpoint_bundle(
        source_receipt=source_receipt,
        source_checkpoint_root=source_root,
        destination_checkpoint_root=checkpoint_target,
        destination_receipt=receipt_target,
        expected_identity=IDENTITY,
        expected_producer_worker_id="P01",
        expected_seed_bundle=2026090101,
        expected_checkpoint_hand=300,
    )

    assert result["status"] == "verified"
    assert links == [(receipt_target.with_name(".P01.json.partial"), receipt_target.absolute())]
    assert receipt_target.read_bytes() == frozen_receipt


def test_checkpoint_transport_rejects_source_receipt_mutation_mid_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentmemeval.experiments import task8b_transport

    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    original_receipt = source_receipt.read_bytes()
    original_copy = task8b_transport.shutil.copyfileobj
    copied = 0

    def mutate_after_first_copy(source: object, destination: object, *, length: int) -> None:
        nonlocal copied
        original_copy(source, destination, length=length)
        copied += 1
        if copied == 1:
            source_receipt.write_bytes(original_receipt + b"\n")

    monkeypatch.setattr(task8b_transport.shutil, "copyfileobj", mutate_after_first_copy)

    with pytest.raises(ConfigError, match="receipt.*变化|transport"):
        transfer_checkpoint_bundle(
            source_receipt=source_receipt,
            source_checkpoint_root=source_root,
            destination_checkpoint_root=tmp_path / "consumer" / "checkpoint",
            destination_receipt=tmp_path / "consumer" / "receipts" / "P01.json",
            expected_identity=IDENTITY,
            expected_producer_worker_id="P01",
            expected_seed_bundle=2026090101,
            expected_checkpoint_hand=300,
        )

    assert not (tmp_path / "consumer" / "checkpoint").exists()
    assert not (tmp_path / "consumer" / "receipts" / "P01.json").exists()


def test_checkpoint_transport_rejects_source_tamper_before_destination_publish(
    tmp_path: Path,
) -> None:
    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    (source_root / "snapshots" / "memory.json").write_text(
        '{"closed":false}\n', encoding="utf-8", newline=""
    )

    with pytest.raises(ConfigError, match="hash|哈希"):
        transfer_checkpoint_bundle(
            source_receipt=source_receipt,
            source_checkpoint_root=source_root,
            destination_checkpoint_root=tmp_path / "consumer" / "checkpoint",
            destination_receipt=tmp_path / "consumer" / "receipts" / "P01.json",
            expected_identity=IDENTITY,
            expected_producer_worker_id="P01",
            expected_seed_bundle=2026090101,
            expected_checkpoint_hand=300,
        )

    assert not (tmp_path / "consumer" / "checkpoint").exists()
    assert not (tmp_path / "consumer" / "receipts" / "P01.json").exists()


@pytest.mark.parametrize("existing", ["checkpoint", "receipt"])
def test_checkpoint_transport_refuses_overwrite(tmp_path: Path, existing: str) -> None:
    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    checkpoint = tmp_path / "consumer" / "checkpoint"
    receipt = tmp_path / "consumer" / "receipts" / "P01.json"
    if existing == "checkpoint":
        checkpoint.mkdir(parents=True)
        marker = checkpoint / "keep.txt"
    else:
        receipt.parent.mkdir(parents=True)
        marker = receipt
    marker.write_text("keep\n", encoding="utf-8", newline="")

    with pytest.raises(FileExistsError):
        transfer_checkpoint_bundle(
            source_receipt=source_receipt,
            source_checkpoint_root=source_root,
            destination_checkpoint_root=checkpoint,
            destination_receipt=receipt,
            expected_identity=IDENTITY,
            expected_producer_worker_id="P01",
            expected_seed_bundle=2026090101,
            expected_checkpoint_hand=300,
        )

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_checkpoint_transport_rejects_receipt_path_traversal(tmp_path: Path) -> None:
    source_root, source_receipt = _checkpoint_fixture(tmp_path)
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    receipt["checkpoint_files"][0]["relative_path"] = "../outside.json"
    _write_json(source_receipt, receipt)

    with pytest.raises(ConfigError, match="路径|path|相对"):
        transfer_checkpoint_bundle(
            source_receipt=source_receipt,
            source_checkpoint_root=source_root,
            destination_checkpoint_root=tmp_path / "consumer" / "checkpoint",
            destination_receipt=tmp_path / "consumer" / "receipts" / "P01.json",
            expected_identity=IDENTITY,
            expected_producer_worker_id="P01",
            expected_seed_bundle=2026090101,
            expected_checkpoint_hand=300,
        )


def test_completed_worker_archive_contains_state_and_completion_receipt(
    tmp_path: Path,
) -> None:
    run_dir = _worker_fixture(tmp_path)

    result = archive_completed_worker(run_dir, tmp_path / "archives")

    archive_path = Path(result["snapshot"]["archive"])
    with tarfile.open(archive_path, "r:gz") as handle:
        names = {Path(name).name for name in handle.getnames()}
    assert "state.tsv" in names
    assert "completion_receipt.json" in names
    assert result["admission"]["status"] == "verified"


def test_completed_worker_archive_uses_authoritative_retry_child_only(tmp_path: Path) -> None:
    run_dir = _worker_fixture(tmp_path)
    stale = run_dir / "runs" / "task__attempt_01"
    _write_json(
        stale / "protocol_audit.json",
        {
            "status": "failed",
            "execution_health": {"valid": False, "fallback_count": 99},
            "run_validity": {"execution_valid": False, "behavior_valid": False},
        },
    )
    _write_json(
        stale / "task_identity_audit.json",
        {"schema_version": "invalid-old-attempt", "status": "failed"},
    )
    _append_listed_files(
        run_dir,
        [stale / "protocol_audit.json", stale / "task_identity_audit.json"],
    )

    result = archive_completed_worker(run_dir, tmp_path / "archives")

    assert result["admission"]["status"] == "verified"
    assert result["admission"]["child_health_audit_count"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "task8-worker-task-results-v0"),
        ("worker_id", "S01"),
    ],
)
def test_completed_worker_archive_rejects_task_results_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    run_dir = _worker_fixture(tmp_path)
    results_path = run_dir / "task_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results[field] = value
    _write_json(results_path, results)
    _refresh_worker_manifest(run_dir)

    with pytest.raises(ConfigError, match="task_results|identity|schema"):
        validate_worker_for_archive(run_dir)


def test_completed_worker_archive_rejects_task_receipt_file_binding_tamper(
    tmp_path: Path,
) -> None:
    run_dir = _worker_fixture(tmp_path)
    receipt_path = run_dir / "task_receipts" / "task.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][0]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    _refresh_worker_manifest(run_dir)

    with pytest.raises(ConfigError, match="receipt files|authoritative child|绑定"):
        validate_worker_for_archive(run_dir)


def test_completed_worker_archive_publishes_receipt_via_fsynced_hidden_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentmemeval.experiments import task8b_transport

    run_dir = _worker_fixture(tmp_path)
    destination = tmp_path / "archives"
    stem = "P01_2026090101"
    final_receipt = destination / f"{stem}.receipt.json"
    events: list[str] = []
    original_fsync = task8b_transport.os.fsync
    original_link = task8b_transport.os.link

    def record_fsync(fd: int) -> None:
        events.append("fsync")
        original_fsync(fd)

    def record_link(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert events and events[-1] == "fsync"
        assert source_path == destination / f".{stem}.receipt.json.partial"
        assert target_path == final_receipt
        assert source_path.is_file()
        assert not target_path.exists()
        assert (destination / f"{stem}.tar.gz").is_file()
        assert (destination / f"{stem}.files.csv").is_file()
        assert (destination / f"{stem}.sha256").is_file()
        events.append("link")
        original_link(source, target)

    monkeypatch.setattr(task8b_transport.os, "fsync", record_fsync)
    monkeypatch.setattr(task8b_transport.os, "link", record_link)

    result = archive_completed_worker(run_dir, destination)

    assert events[-2:] == ["fsync", "link"]
    assert Path(result["snapshot"]["receipt"]) == final_receipt
    assert final_receipt.is_file()
    assert not (destination / f".{stem}.receipt.json.partial").exists()


def test_completed_worker_archive_rejects_artifact_hash_failure(tmp_path: Path) -> None:
    run_dir = _worker_fixture(tmp_path)
    with (run_dir / "runs" / "task__attempt_02" / "hands.jsonl").open(
        "a", encoding="utf-8", newline=""
    ) as handle:
        handle.write('{"hand_id":2}\n')

    with pytest.raises(ConfigError, match="hash|哈希"):
        validate_worker_for_archive(run_dir)


def test_completed_worker_archive_rejects_child_health_failure(tmp_path: Path) -> None:
    run_dir = _worker_fixture(tmp_path)
    audit_path = run_dir / "runs" / "task__attempt_02" / "protocol_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["execution_health"]["fallback_count"] = 1
    _write_json(audit_path, audit)
    _refresh_task_receipt(run_dir)

    with pytest.raises(ConfigError, match="health|fallback"):
        archive_completed_worker(run_dir, tmp_path / "archives")

    assert not (tmp_path / "archives").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "execution_invalid", "behavior_invalid"],
)
def test_completed_worker_archive_requires_explicit_valid_run_validity(
    tmp_path: Path, mutation: str
) -> None:
    run_dir = _worker_fixture(tmp_path)
    audit_path = run_dir / "runs" / "task__attempt_02" / "protocol_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        audit.pop("run_validity")
    elif mutation == "execution_invalid":
        audit["run_validity"]["execution_valid"] = False
    else:
        audit["run_validity"]["behavior_valid"] = False
    _write_json(audit_path, audit)
    _refresh_task_receipt(run_dir)

    with pytest.raises(ConfigError, match="run_validity|health"):
        validate_worker_for_archive(run_dir)


@pytest.mark.parametrize("mutation", ["missing", "valid_false"])
def test_completed_worker_archive_requires_explicit_valid_execution_health(
    tmp_path: Path, mutation: str
) -> None:
    run_dir = _worker_fixture(tmp_path)
    audit_path = run_dir / "runs" / "task__attempt_02" / "protocol_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        audit.pop("execution_health")
    else:
        audit["execution_health"]["valid"] = False
    _write_json(audit_path, audit)
    _refresh_task_receipt(run_dir)

    with pytest.raises(ConfigError, match="execution_health|health"):
        validate_worker_for_archive(run_dir)
