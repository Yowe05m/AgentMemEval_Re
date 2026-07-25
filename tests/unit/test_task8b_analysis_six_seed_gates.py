from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation import task8b_analysis


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def _write_state(
    path: Path,
    *,
    detail_prefix: str = "fixture",
) -> list[dict[str, str]]:
    statuses = (
        "planned",
        "validating",
        "running",
        "failed",
        "validating",
        "running",
        "finalizing",
        "complete",
    )
    previous = "GENESIS"
    rows: list[dict[str, str]] = []
    for index, status in enumerate(statuses):
        row = {
            "schema_version": task8b_analysis.STATE_SCHEMA_VERSION,
            "created_at_utc": f"2026-09-01T00:00:{index:02d}+00:00",
            "status": status,
            "detail": f"{detail_prefix}-{index}-{status}",
            "previous_sha256": previous,
        }
        row["row_sha256"] = task8b_analysis.sha256_json(row)
        rows.append(row)
        previous = row["row_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "schema_version",
                "created_at_utc",
                "status",
                "detail",
                "previous_sha256",
                "row_sha256",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _refresh_composition_hashes(
    composition_path: Path,
    worker: dict[str, object],
) -> None:
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    for side in composition["sides"]:
        for row in side["execution_receipts"]:
            path = composition_path.parent / str(row["path"])
            row["sha256"] = _sha256(path)
    _write_json(composition_path, composition)
    lineage = worker["shard_lineage"]
    assert isinstance(lineage, dict)
    lineage["composition_receipt_sha256"] = _sha256(composition_path)


def _normalization_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, Path],
]:
    root = tmp_path / "snapshots" / "S08" / "2026090108" / "attempt_01"
    task_root = root / "runs" / "mixed_ecological"
    task_root.mkdir(parents=True)
    manifest = {
        "worker_id": "S08",
        "role": "secondary",
        "seed_bundle": 2026090108,
    }
    manifest_path = root / "worker_manifest.json"
    _write_json(manifest_path, manifest)
    state_rows = _write_state(root / "state.tsv")
    expected = {
        "code_sha": task8b_analysis.FROZEN_SCIENTIFIC_CODE_SHA,
        "embedding_fingerprint": "2" * 64,
        "model_fingerprint": "3" * 64,
        "prompt_sha256": "4" * 64,
        "resolved_config_sha256": "5" * 64,
        "schedule_sha256": "6" * 64,
    }
    actual = {
        **expected,
        "code_dirty": False,
        "resolved_config_sha256": "7" * 64,
    }
    task_audit = {
        "schema_version": "task8-task-identity-audit-v1",
        "protocol_status": "frozen/expedited-formal-candidate",
        "status": "verified",
        "task_id": "mixed_ecological",
        "actual": actual,
    }
    task_audit_path = task_root / "task_identity_audit.json"
    _write_json(task_audit_path, task_audit)
    recovery_id = "TASK8B-H02-S08-mixed-fixture-v1"
    receipt_path = root / "task_receipts" / "mixed_ecological.json"
    receipt = {
        "schema_version": "task8-worker-task-receipt-v1",
        "task_id": "mixed_ecological",
        "run_dir": "runs/mixed_ecological",
        "files": [
            {
                "relative_path": "task_identity_audit.json",
                "size": task_audit_path.stat().st_size,
                "sha256": _sha256(task_audit_path),
            }
        ],
        "task_row": {
            "cache_namespace": "/root/autodl-tmp/cache/mixed/frozen",
            "identity_audit": actual,
            "memory_mode": "Frozen",
            "recovery_id": recovery_id,
            "run_dir": "runs/mixed_ecological",
            "status": "complete",
            "task_id": "mixed_ecological",
        },
    }
    _write_json(receipt_path, receipt)

    server_attempt = (
        "/root/autodl-tmp/task8b_six_seed_pair_shards/H02/"
        "paired_shards/S08/fixture/output"
    )
    server_evidence = (
        "/root/autodl-tmp/task8b_six_seed_control/H02/" "receipts/recovery/s08-fixture"
    )
    correction = {
        "corrected_actual_sha256": actual["resolved_config_sha256"],
        "only_authorized_difference": (
            "frozen secondary standalone runtime normalization"
        ),
        "original_expected_sha256": expected["resolved_config_sha256"],
        "reason": task8b_analysis.SECONDARY_NORMALIZATION_REASON,
        "runtime_transform": (
            task8b_analysis.SECONDARY_NORMALIZATION_RUNTIME_TRANSFORM
        ),
    }
    correction_audit = {
        "schema_version": "task8b-paired-identity-correction-audit-v1",
        "status": "verified-recovered",
        "worker_id": "S08",
        "seed": 2026090108,
        "task_id": "mixed_ecological",
        "recovery_id": recovery_id,
        "attempt_root": server_attempt,
        "standard_task_identity_audit_path": (
            f"{server_attempt}/runs/mixed_ecological/task_identity_audit.json"
        ),
        "expected": expected,
        "actual": actual,
        "identity_correction": correction,
        "checkpoint_receipt_identity_correction": (
            task8b_analysis.SECONDARY_NORMALIZATION_CHECKPOINT_CORRECTION
        ),
        "effect_fields_read": False,
        "health": task8b_analysis.SECONDARY_NORMALIZATION_HEALTH,
    }
    execution_root = tmp_path / "composition" / "execution" / "s08-normalization"
    correction_audit_path = (
        execution_root / "certificate.identity_correction_audit.json"
    )
    _write_json(correction_audit_path, correction_audit)
    certificate = {
        "schema_version": "task8b-paired-completed-recovery-certificate-v1",
        "status": "execution_complete_recovered",
        "role": "secondary",
        "worker_id": "S08",
        "seed": 2026090108,
        "task_id": "mixed_ecological",
        "recovery_id": recovery_id,
        "attempt_root": server_attempt,
        "reason": task8b_analysis.SECONDARY_NORMALIZATION_REASON,
        "identity_correction": correction,
        "identity_correction_audit_path": (
            f"{server_evidence}/certificate.identity_correction_audit.json"
        ),
        "identity_correction_audit_sha256": _sha256(correction_audit_path),
        "checkpoint_receipt_identity_correction": (
            task8b_analysis.SECONDARY_NORMALIZATION_CHECKPOINT_CORRECTION
        ),
        "effect_fields_read": False,
        "recovered_task_scientific_files_modified": False,
        "failed_state_preserved": True,
        "shard_closed": True,
        "standard_resume_used_frozen_runner": True,
        "standard_resume_task_ids": [],
        "unrecovered_selected_task_ids": [],
        "recovery_continuation_after_frozen_complete": False,
        "planned_hands": 2700,
        "actual_hands": 2700,
        "health": task8b_analysis.SECONDARY_NORMALIZATION_HEALTH,
        "authorization_sha256": "8" * 64,
        "baseline_sha256": "9" * 64,
        "derived_manifest_sha256": _sha256(manifest_path),
        "failed_state_row_sha256": state_rows[3]["row_sha256"],
        "pre_recovery_archive_sha256": "c" * 64,
        "recovery_tool_sha256": "d" * 64,
        "appended_state_row_sha256": [row["row_sha256"] for row in state_rows[4:]],
        "task_identity_audit_sha256": _sha256(task_audit_path),
        "task_receipt_sha256": _sha256(receipt_path),
    }
    certificate_path = execution_root / "certificate.json"
    _write_json(certificate_path, certificate)
    ledger_path = execution_root / "ledger.json"
    _write_json(
        ledger_path,
        {
            "schema_version": "task8b-paired-recovery-ledger-entry-v1",
            "status": "append-only",
            "worker_id": "S08",
            "seed": 2026090108,
            "task_id": "mixed_ecological",
            "recovery_id": recovery_id,
            "attempt_root": server_attempt,
            "certificate_path": f"{server_evidence}/certificate.json",
            "certificate_sha256_pending_at_publication": True,
            "effect_fields_read": False,
            "shard_closed": True,
        },
    )
    shard_seal_path = execution_root / "shard_seal.json"
    _write_json(
        shard_seal_path,
        {
            "schema_version": "task8b-paired-shard-receipt-v1",
            "status": "complete",
            "source_kind": "execution",
            "role": "secondary",
            "worker_id": "S08",
            "seed": 2026090108,
            "pair_id": "pair_08",
            "selected_task_ids": ["mixed_ecological"],
            "shard_role": "low",
            "effect_fields_read": False,
            "attempt_root": server_attempt,
            "derived_manifest_sha256": certificate["derived_manifest_sha256"],
            "canonical_manifest_sha256": "f" * 64,
            "frozen_code_sha": (task8b_analysis.FROZEN_SCIENTIFIC_CODE_SHA),
            "tasks": [
                {
                    "schema_version": "task8b-paired-shard-task-v1",
                    "status": "complete",
                    "task_id": "mixed_ecological",
                    "producer_worker_id": "S08",
                    "run_dir": receipt["run_dir"],
                    "planned_hands": 2700,
                    "actual_hands": 2700,
                    "files": receipt["files"],
                    "task_receipt_path": (
                        f"{server_attempt}/task_receipts/" "mixed_ecological.json"
                    ),
                    "task_receipt_sha256": _sha256(receipt_path),
                }
            ],
        },
    )
    evidence_paths = {
        "certificate": certificate_path,
        "audit": correction_audit_path,
        "ledger": ledger_path,
        "seal": shard_seal_path,
    }
    high_receipt_path = tmp_path / "composition" / "execution" / "s08-high.json"
    _write_json(high_receipt_path, {"status": "complete"})
    composition_path = tmp_path / "composition" / "S08.json"
    _write_json(
        composition_path,
        {
            "schema_version": "task8b-six-seed-composition-receipt-v1",
            "worker_id": "S08",
            "seed": 2026090108,
            "pair_id": "pair_08",
            "sides": [
                {
                    "side": "low",
                    "physical_slot": "H02",
                    "task_ids": ["mixed_ecological"],
                    "actual_hands": 2700,
                    "execution_receipts": [
                        {
                            "path": path.relative_to(
                                composition_path.parent
                            ).as_posix(),
                            "sha256": _sha256(path),
                        }
                        for path in evidence_paths.values()
                    ],
                },
                {
                    "side": "high",
                    "physical_slot": "H08",
                    "task_ids": [
                        "expr_online",
                        "expr_without",
                        "async_online",
                        "async_without",
                    ],
                    "actual_hands": 2400,
                    "execution_receipts": [
                        {
                            "path": high_receipt_path.relative_to(
                                composition_path.parent
                            ).as_posix(),
                            "sha256": _sha256(high_receipt_path),
                        }
                    ],
                },
            ],
            "task_union": list(task8b_analysis.FORMAL_SECONDARY_TASKS),
            "actual_hands": 5100,
        },
    )
    worker: dict[str, object] = {
        "worker_id": "S08",
        "pod_id": "pod08",
        "seed": 2026090108,
        "shard_lineage": {
            "pair_id": "pair_08",
            "composition_receipt_path": "composition/S08.json",
            "composition_receipt_sha256": _sha256(composition_path),
        },
    }
    evidence_paths["composition"] = composition_path
    return (
        root,
        task_root,
        worker,
        manifest,
        expected,
        actual,
        evidence_paths,
    )


def test_isolation_schedule_crn_gate_remains_primary_only() -> None:
    secondary_tasks = [
        {
            "task_id": task_id,
            "expected_identity": {"schedule_sha256": f"{index:064x}"},
        }
        for index, task_id in enumerate(
            task8b_analysis.FORMAL_SECONDARY_TASKS,
            start=1,
        )
    ]
    primary_tasks = [
        {
            "task_id": task_id,
            "expected_identity": {"schedule_sha256": "a" * 64},
        }
        for task_id in task8b_analysis.FORMAL_PRIMARY_TASKS
    ]

    assert (
        task8b_analysis._formal_worker_crn_reasons(
            {"role": "secondary"}, secondary_tasks
        )
        == []
    )
    assert task8b_analysis._formal_worker_crn_reasons(
        {"role": "primary"}, secondary_tasks
    ) == ["CRN_INVALID"]
    assert (
        task8b_analysis._formal_worker_crn_reasons({"role": "primary"}, primary_tasks)
        == []
    )
    primary_tasks[-1]["expected_identity"]["schedule_sha256"] = "b" * 64
    assert task8b_analysis._formal_worker_crn_reasons(
        {"role": "primary"}, primary_tasks
    ) == ["CRN_INVALID"]


def test_secondary_normalization_accepts_only_real_sha_bound_recovery_chain(
    tmp_path: Path,
) -> None:
    root, task_root, worker, manifest, expected, actual, _ = _normalization_fixture(
        tmp_path
    )
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )

    task8b_analysis._validate_secondary_standalone_identity_normalization(
        root=root,
        worker=worker,
        manifest=manifest,
        task_root=task_root,
        task_audit=task_audit,
        expected=expected,
        actual=actual,
    )


def test_secondary_normalization_rejects_non_config_identity_mismatch(
    tmp_path: Path,
) -> None:
    root, task_root, worker, manifest, expected, actual, _ = _normalization_fixture(
        tmp_path
    )
    bad_actual = {**actual, "prompt_sha256": "0" * 64}
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )
    task_audit["actual"] = bad_actual

    with pytest.raises(ConfigError, match="超出 resolved config"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=bad_actual,
        )


def test_secondary_normalization_rejects_semantic_and_sha_tamper(
    tmp_path: Path,
) -> None:
    root, task_root, worker, manifest, expected, actual, evidence = (
        _normalization_fixture(tmp_path)
    )
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )
    certificate = json.loads(evidence["certificate"].read_text(encoding="utf-8"))
    certificate["identity_correction"]["runtime_transform"]["checkpoint_set"] = [150]
    _write_json(evidence["certificate"], certificate)
    _refresh_composition_hashes(evidence["composition"], worker)

    with pytest.raises(ConfigError, match="certificate"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )

    certificate["identity_correction"]["runtime_transform"]["checkpoint_set"] = [300]
    _write_json(evidence["certificate"], certificate)
    _refresh_composition_hashes(evidence["composition"], worker)
    composition = json.loads(evidence["composition"].read_text(encoding="utf-8"))
    low = next(side for side in composition["sides"] if side["side"] == "low")
    certificate_row = next(
        row
        for row in low["execution_receipts"]
        if Path(row["path"]).name == "certificate.json"
    )
    certificate_row["sha256"] = "0" * 64
    _write_json(evidence["composition"], composition)
    lineage = worker["shard_lineage"]
    assert isinstance(lineage, dict)
    lineage["composition_receipt_sha256"] = _sha256(evidence["composition"])

    with pytest.raises(ConfigError, match="SHA"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )


def test_secondary_normalization_rejects_derived_manifest_splice(
    tmp_path: Path,
) -> None:
    root, task_root, worker, manifest, expected, actual, _ = _normalization_fixture(
        tmp_path
    )
    manifest["foreign_provenance"] = True
    _write_json(root / "worker_manifest.json", manifest)
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ConfigError, match="certificate"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "reordered", "truncated"),
)
def test_secondary_normalization_rejects_noncanonical_state_suffix(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, task_root, worker, manifest, expected, actual, evidence = (
        _normalization_fixture(tmp_path)
    )
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )
    certificate = json.loads(evidence["certificate"].read_text(encoding="utf-8"))
    appended = list(certificate["appended_state_row_sha256"])
    if mutation == "duplicate":
        appended[1] = appended[0]
    elif mutation == "reordered":
        appended = list(reversed(appended))
    else:
        appended = appended[:-1]
    certificate["appended_state_row_sha256"] = appended
    _write_json(evidence["certificate"], certificate)
    _refresh_composition_hashes(evidence["composition"], worker)

    with pytest.raises(ConfigError, match="state"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )


@pytest.mark.parametrize(
    "evidence_name",
    ("composition", "certificate"),
)
def test_secondary_normalization_rejects_symlink_evidence(
    tmp_path: Path,
    evidence_name: str,
) -> None:
    root, task_root, worker, manifest, expected, actual, evidence = (
        _normalization_fixture(tmp_path)
    )
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )
    original = evidence[evidence_name]
    target = original.with_name(f"{original.name}.target")
    original.replace(target)
    try:
        original.symlink_to(target.name)
    except (NotImplementedError, OSError):
        if original.is_symlink():
            original.unlink()
        target.replace(original)
        pytest.skip("filesystem does not permit test symlinks")

    with pytest.raises(ConfigError, match="symlink/junction/reparse"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )


def test_secondary_normalization_rejects_junction_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task_root, worker, manifest, expected, actual, evidence = (
        _normalization_fixture(tmp_path)
    )
    task_audit = json.loads(
        (task_root / "task_identity_audit.json").read_text(encoding="utf-8")
    )
    composition_path = evidence["composition"]
    concrete_path_type = type(composition_path)
    original_is_junction = getattr(
        concrete_path_type,
        "is_junction",
        None,
    )

    def fake_is_junction(path: Path) -> bool:
        if path == composition_path:
            return True
        return (
            bool(original_is_junction(path))
            if callable(original_is_junction)
            else False
        )

    monkeypatch.setattr(
        concrete_path_type,
        "is_junction",
        fake_is_junction,
        raising=False,
    )

    with pytest.raises(ConfigError, match="symlink/junction/reparse"):
        task8b_analysis._validate_secondary_standalone_identity_normalization(
            root=root,
            worker=worker,
            manifest=manifest,
            task_root=task_root,
            task_audit=task_audit,
            expected=expected,
            actual=actual,
        )


@pytest.mark.parametrize(
    ("command", "required_args"),
    (
        (
            "task8b-freeze-phase-f",
            [
                "--phase-f-dir",
                "phase-f",
                "--dependency-lock",
                "uv.lock",
                "--output",
                "preunlock.json",
            ],
        ),
        (
            "task8b-build-analysis-input",
            [
                "--worker-manifest-dir",
                "manifests",
                "--snapshot-root",
                "snapshots",
                "--pre-unlock-manifest",
                "preunlock.json",
                "--output",
                "analysis-input.json",
            ],
        ),
    ),
)
def test_phase_f_cli_rejects_implicit_legacy_contract(
    command: str,
    required_args: list[str],
) -> None:
    cli_main = importlib.import_module("agentmemeval.cli.main")

    with pytest.raises(SystemExit):
        cli_main.build_parser().parse_args([command, *required_args])


def test_phase_f_cli_forwards_explicit_six_seed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_main = importlib.import_module("agentmemeval.cli.main")
    captured: list[object] = []

    def fake_freeze(*args: object, **kwargs: object) -> dict[str, object]:
        captured.append(kwargs["analysis_contract_path"])
        return {"status": "fixture"}

    def fake_build(*args: object, **kwargs: object) -> dict[str, object]:
        captured.append(kwargs["analysis_contract_path"])
        return {"status": "fixture"}

    contract = "configs/formal/task8b_six_seed_analysis_contract.json"
    monkeypatch.setattr(
        cli_main,
        "build_task8b_preunlock_manifest",
        fake_freeze,
    )
    monkeypatch.setattr(cli_main, "build_task8b_analysis_input", fake_build)
    assert (
        cli_main.main(
            [
                "task8b-freeze-phase-f",
                "--phase-f-dir",
                "phase-f",
                "--dependency-lock",
                "uv.lock",
                "--output",
                "preunlock.json",
                "--analysis-contract",
                contract,
            ]
        )
        == 0
    )
    assert (
        cli_main.main(
            [
                "task8b-build-analysis-input",
                "--worker-manifest-dir",
                "manifests",
                "--snapshot-root",
                "snapshots",
                "--pre-unlock-manifest",
                "preunlock.json",
                "--output",
                "analysis-input.json",
                "--analysis-contract",
                contract,
            ]
        )
        == 0
    )

    assert captured == [contract, contract]
