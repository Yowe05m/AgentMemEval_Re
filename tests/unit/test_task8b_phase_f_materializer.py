from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation import task8b_analysis
from agentmemeval.experiments import task8b_phase_f_materializer as materializer_module
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.formal_runner import STATE_SCHEMA_VERSION
from agentmemeval.experiments.task8b_phase_f_materializer import (
    ANALYSIS_CONTRACT_ID,
    COMPOSITION_SCHEMA,
    EXPECTED_WORKERS,
    FROZEN_SCIENTIFIC_CODE_SHA,
    METADATA_DRY_RUN_GATE_NAMES,
    PRIMARY_TASKS,
    PROTOCOL_ID,
    RECOVERED_WORKERS,
    RECOVERY_ADOPTION_SCHEMA,
    RECOVERY_ATTESTATION_SCHEMA,
    RECOVERY_CANONICAL_AUDIT_SCHEMA,
    RECOVERY_PROTOCOL_AMENDMENT_ID,
    RECOVERY_REASON,
    SECONDARY_TASKS,
    SOURCE_MAP_SCHEMA,
    _atomic_rename_noreplace,
    _build_metadata_plan,
    _build_state_rows,
    _parse_state_bytes,
    _prepare_materialization,
    _validate_authoritative_real_selections,
    _validate_authoritative_recovery_suffix,
    _validate_pre_materialization_receipt,
    _validate_recovery_state_anchor,
    create_pre_materialization_receipt,
    main,
    materialize_task8b_phase_f,
    metadata_only_dry_run_task8b_phase_f,
)


@pytest.fixture
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Recovery evidence has long fixed filenames; keep each isolated test root short
    # so Windows can exercise the same hash/identity gates as other platforms.
    return tmp_path_factory.mktemp("m")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_ref(path: Path, root: Path, **extra: object) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _member_ref(member: str, payload: bytes, **extra: object) -> dict[str, object]:
    return {
        "archive_id": "formal",
        "member": member,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


def _state_bytes(statuses: list[str], *, worker_id: str) -> tuple[bytes, list[dict[str, str]]]:
    previous = "GENESIS"
    rows = []
    for index, status in enumerate(statuses):
        body = {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_at_utc": f"2026-07-25T0{index}:00:00+00:00",
            "status": status,
            "detail": f"{worker_id}-{status}",
            "previous_sha256": previous,
        }
        row = {**body, "row_sha256": sha256_json(body)}
        rows.append(row)
        previous = row["row_sha256"]
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(
        (
            "schema_version",
            "created_at_utc",
            "status",
            "detail",
            "previous_sha256",
            "row_sha256",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row["schema_version"],
                row["created_at_utc"],
                row["status"],
                row["detail"],
                row["previous_sha256"],
                row["row_sha256"],
            )
        )
    return handle.getvalue().encode("utf-8"), rows


def _recovery_files(
    worker_id: str,
    *,
    original_terminal_state_sha256: str,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, bytes]]:
    expected_config = "4" * 64
    legacy_config = "5" * 64
    authorization_id = f"{worker_id}-recovery"
    support = {
        "protocol_amendment": _json_bytes(
            {
                "amendment_id": RECOVERY_PROTOCOL_AMENDMENT_ID,
                "effect_fields_read": False,
            }
        ),
        "pre_recovery_archive": (f"{worker_id}-immutable-pre-recovery-archive\n".encode()),
        "pre_recovery_manifest": (
            f"sha256\tsize\trelative_path\n{'0' * 64}\t0\t{worker_id}/placeholder\n"
        ).encode(),
        "authorization": _json_bytes(
            {
                "authorization_id": authorization_id,
                "effect_fields_read": False,
            }
        ),
    }
    base = {
        "protocol_amendment_sha256": hashlib.sha256(support["protocol_amendment"]).hexdigest(),
        "verifier_code_sha": "b" * 40,
        "pre_recovery_archive_sha256": hashlib.sha256(support["pre_recovery_archive"]).hexdigest(),
        "pre_recovery_file_manifest_sha256": hashlib.sha256(
            support["pre_recovery_manifest"]
        ).hexdigest(),
        "original_terminal_state_sha256": original_terminal_state_sha256,
        "original_expected_config_sha256": expected_config,
        "corrected_config_sha256": expected_config,
    }
    canonical = {
        "schema_version": RECOVERY_CANONICAL_AUDIT_SCHEMA,
        "worker_id": worker_id,
        "task_id": "isolation_no_memory",
        "original_expected_sha256": expected_config,
        "corrected_actual_sha256": expected_config,
        "legacy_json_roundtrip_actual_sha256": legacy_config,
        "semantic_equivalence_after_stringifying_mapping_keys": True,
        "only_authorized_difference": "integer-versus-string mapping keys",
    }
    canonical_bytes = _json_bytes(canonical)
    base["canonicalization_equivalence_audit_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    attestation = {
        "schema_version": RECOVERY_ATTESTATION_SCHEMA,
        "worker_id": worker_id,
        "task_id": "isolation_no_memory",
        "protocol_amendment_id": RECOVERY_PROTOCOL_AMENDMENT_ID,
        "scientific_execution_code_sha": FROZEN_SCIENTIFIC_CODE_SHA,
        "scientific_outcome_fields_accessed": False,
        "task1_rerun_performed": False,
        "raw_artifact_bytes_modified": False,
        "planned_hands": 1350,
        "actual_hands": 1350,
        "phase_f_evidence": copy.deepcopy(base),
    }
    attestation_bytes = _json_bytes(attestation)
    attestation_sha = hashlib.sha256(attestation_bytes).hexdigest()
    certificate_phase = {
        **base,
        "task1_adoption_attestation_sha256": attestation_sha,
    }
    certificate = {
        "schema_version": RECOVERY_ADOPTION_SCHEMA,
        "status": "authorized-task1-adoption",
        "reason": RECOVERY_REASON,
        "worker_id": worker_id,
        "task_id": "isolation_no_memory",
        "protocol_amendment_id": RECOVERY_PROTOCOL_AMENDMENT_ID,
        "frozen_code_sha": FROZEN_SCIENTIFIC_CODE_SHA,
        "effect_fields_read": False,
        "scientific_outcome_fields_accessed": False,
        "task1_rerun_performed": False,
        "raw_artifact_bytes_modified": False,
        "planned_hands": 1350,
        "actual_hands": 1350,
        "authorization_id": authorization_id,
        "authorization_sha256": hashlib.sha256(support["authorization"]).hexdigest(),
        "identity_correction": {
            "original_expected_sha256": expected_config,
            "corrected_actual_sha256": expected_config,
            "legacy_json_roundtrip_actual_sha256": legacy_config,
            "semantic_equivalence_after_stringifying_mapping_keys": True,
            "only_authorized_difference": "integer-versus-string mapping keys",
        },
        "phase_f_evidence": certificate_phase,
    }
    certificate_bytes = _json_bytes(certificate)
    certificate_sha = hashlib.sha256(certificate_bytes).hexdigest()
    evidence = {
        **base,
        "recovery_certificate_sha256": certificate_sha,
        "task1_adoption_attestation_sha256": attestation_sha,
    }
    adoption = {
        "schema_version": RECOVERY_ADOPTION_SCHEMA,
        "authorization_id": certificate["authorization_id"],
        "authorization_sha256": certificate["authorization_sha256"],
        "recovery_certificate_sha256": certificate_sha,
    }
    return (
        {
            "canonicalization_audit": canonical_bytes,
            "adoption_attestation": attestation_bytes,
            "certificate": certificate_bytes,
        },
        {"recovery_adoption": adoption, "phase_f_evidence": evidence},
        support,
    )


def _task_files(worker_id: str, task_id: str) -> dict[str, bytes]:
    return {
        "events.jsonl": b'{"structural_event":true}\n',
        "experiment_result.json": b'{"status":"complete"}\n',
        "hand_summaries.jsonl": b'{"hand_id":"opaque"}\n',
        "manifest.json": _json_bytes(
            {"schema_version": "fixture-task-manifest-v1", "task_id": task_id}
        ),
        # Deliberately malformed JSON: success proves formal metrics are never parsed.
        "metrics.json": b"FORMAL_EFFECT_PAYLOAD_IS_OPAQUE\n",
        "resolved_config.yaml": f"task_id: {task_id}\n".encode(),
        "schedule_manifest.json": _json_bytes({"task_id": task_id}),
        "task_identity_audit.json": _json_bytes(
            {"task_id": task_id, "status": "verified", "worker_id": worker_id}
        ),
    }


def _make_source_map(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "source"
    root.mkdir()
    payloads: dict[str, bytes] = {
        "payload/unselected/provenance.bin": b"unselected-but-still-sha-verified\n"
    }
    workers: list[dict[str, object]] = []
    base_manifests: dict[str, Path] = {}
    state_paths: dict[str, Path] = {}
    recovery_paths: dict[str, dict[str, Path]] = {}
    receipt_recovery: dict[str, dict[str, object]] = {}

    contract = root / "analysis_contract.json"
    contract.write_bytes(task8b_analysis._six_seed_contract_path().read_bytes())
    preunlock = root / "pre_unlock_manifest.json"
    preunlock.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "task8b_phase_f"
            / "phase_f_pre_unlock_manifest_b382978_six_seed.json"
        ).read_bytes()
    )
    source_receipt = root / "source_receipt.json"
    source_seal = root / "source_seal.json"
    _write_json(
        source_receipt,
        {
            "schema_version": "task8b-paired-shard-cross-host-export-v1",
            "status": "complete",
            "effect_fields_read": False,
        },
    )
    _write_json(
        source_seal,
        {
            "schema_version": "task8b-full-matrix-export-seal-v1",
            "status": "sealed",
            "effect_fields_read": False,
        },
    )

    for worker_id in EXPECTED_WORKERS:
        index = int(worker_id[1:])
        role = "primary" if worker_id.startswith("P") else "secondary"
        tasks = PRIMARY_TASKS if role == "primary" else SECONDARY_TASKS
        state_path = root / "states" / f"{worker_id}.tsv"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if worker_id in RECOVERED_WORKERS:
            state_bytes, state_rows = _state_bytes(["validating", "failed"], worker_id=worker_id)
            recovery, receipt_fields, recovery_support = _recovery_files(
                worker_id,
                original_terminal_state_sha256=state_rows[-1]["row_sha256"],
            )
            recovery_paths[worker_id] = {}
            for key, value in recovery.items():
                path = root / "recovery" / worker_id / f"{key}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
                recovery_paths[worker_id][key] = path
            support_names = {
                "protocol_amendment": "protocol_amendment.json",
                "pre_recovery_archive": "pre_recovery_archive.tar",
                "pre_recovery_manifest": "pre_recovery_manifest.tsv",
                "authorization": "authorization.json",
            }
            for key, value in recovery_support.items():
                path = root / "recovery" / worker_id / support_names[key]
                path.write_bytes(value)
                recovery_paths[worker_id][key] = path
            receipt_recovery[worker_id] = receipt_fields
            state_append = [
                {
                    "created_at_utc": "2026-07-25T10:00:00+00:00",
                    "status": "validating",
                    "detail": "canonical recovery validation",
                },
                {
                    "created_at_utc": "2026-07-25T10:01:00+00:00",
                    "status": "running",
                    "detail": "canonical composition",
                },
                {
                    "created_at_utc": "2026-07-25T10:02:00+00:00",
                    "status": "finalizing",
                    "detail": "canonical sealing",
                },
                {
                    "created_at_utc": "2026-07-25T10:03:00+00:00",
                    "status": "complete",
                    "detail": "canonical completion published",
                },
            ]
        else:
            state_bytes, _state_rows = _state_bytes(
                ["validating", "running", "finalizing", "complete"],
                worker_id=worker_id,
            )
            state_append = []
        state_path.write_bytes(state_bytes)
        state_paths[worker_id] = state_path
        base_manifest = root / "base_manifests" / f"{worker_id}.json"
        common_identity = {
            "code_sha": FROZEN_SCIENTIFIC_CODE_SHA,
            "prompt_sha256": "7" * 64,
            "model_fingerprint": "fixture-model",
            "embedding_fingerprint": "fixture-embedding",
            "protocol_sha256": "a" * 64,
            "runtime_image_fingerprint": "fixture-runtime",
        }
        receipt_identity = {
            **{
                field: common_identity[field]
                for field in (
                    "code_sha",
                    "prompt_sha256",
                    "model_fingerprint",
                    "embedding_fingerprint",
                )
            },
            "resolved_config_sha256": "8" * 64,
            "schedule_sha256": "9" * 64,
        }
        seed = 2026090100 + index
        schedule_rows = [
            {
                "worker_role": worker_role,
                "task_id": task_id,
                "schedule_sha256": "9" * 64,
            }
            for worker_role, worker_tasks in (
                ("primary", PRIMARY_TASKS),
                ("secondary", SECONDARY_TASKS),
            )
            for task_id in worker_tasks
        ]
        seed_pod_identity = {
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
        _write_json(
            base_manifest,
            {
                "schema_version": "task8-worker-manifest-v1",
                "worker_id": worker_id,
                "role": role,
                "seed_bundle": seed,
                "pod_id": f"pod{index:02d}",
                "protocol_status": "frozen/expedited-formal-candidate",
                "execution_mode": "experiment_configs",
                "checkpoint_set": [30, 75, 150, 300],
                "heldout_table_set": ["H01", "H02", "H03"],
                "common_identity": common_identity,
                "seed_pod_identity": seed_pod_identity,
                "instance_identity": {
                    "worker_id": worker_id,
                    "cache_namespace": f"task8b/{worker_id}",
                    "output_path": f"outputs/formal/task8b/{worker_id}",
                },
                "receipt_relative_path": f"receipts/{worker_id}.json",
                **(
                    {"receipt_identity": receipt_identity}
                    if role == "primary"
                    else {
                        "depends_on": f"P{index:02d}",
                        "dependency_output_path": (f"outputs/formal/task8b/P{index:02d}"),
                        "dependency_receipt_identity": receipt_identity,
                    }
                ),
                "task_configs": [
                    {
                        "task_id": task_id,
                        "planned_hands": (
                            2700
                            if task_id == "mixed_ecological"
                            else 600
                            if task_id
                            in {
                                "expr_online",
                                "expr_without",
                                "async_online",
                                "async_without",
                            }
                            else 1350
                        ),
                        "schedule_sha256": "9" * 64,
                        "config_sha256": "c" * 64,
                        "covers": (
                            ["R1-E1-I", "R1-E2", "R1-E3"]
                            if role == "primary"
                            else ["R1-E1-M"]
                            if task_id == "mixed_ecological"
                            else ["R1-E4"]
                            if task_id in {"expr_online", "async_online"}
                            else ["R1-E5"]
                        ),
                        "expected_identity": receipt_identity,
                    }
                    for task_id in tasks
                ],
            },
        )
        base_manifests[worker_id] = base_manifest
        task_specs = []
        for task_id in tasks:
            run_name = (
                f"{task_id}__attempt_02"
                if worker_id == "P12" and task_id == "isolation_sync"
                else task_id
            )
            prefix = f"payload/{worker_id}/runs/{run_name}"
            files = _task_files(worker_id, task_id)
            for relative, content in files.items():
                payloads[f"{prefix}/{relative}"] = content
            file_manifest = [
                {
                    "relative_path": relative,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for relative, content in sorted(files.items())
            ]
            run_dir = f"runs/{run_name}"
            receipt = {
                "schema_version": "task8-worker-task-receipt-v1",
                "task_id": task_id,
                "config_sha256": "c" * 64,
                "run_dir": run_dir,
                "task_row": {
                    "task_id": task_id,
                    "run_dir": run_dir,
                    "status": "complete",
                },
                "files": file_manifest,
                **(
                    receipt_recovery[worker_id]
                    if task_id == "isolation_no_memory" and worker_id in RECOVERED_WORKERS
                    else {}
                ),
            }
            receipt_member = f"payload/{worker_id}/task_receipts/{task_id}.json"
            receipt_bytes = _json_bytes(receipt)
            payloads[receipt_member] = receipt_bytes
            task_specs.append(
                {
                    "task_id": task_id,
                    "source_archive_id": "formal",
                    "source_prefix": prefix,
                    "target_run_dir": run_dir,
                    "receipt": _member_ref(receipt_member, receipt_bytes),
                }
            )
        payloads[f"payload/{worker_id}/files.tsv"] = (
            f"relative_path\tsize\tsha256\n{worker_id}/structural\t0\t{'0' * 64}\n"
        ).encode()
        workers.append(
            {
                "worker_id": worker_id,
                "role": role,
                "seed": 2026090100 + index,
                "pair_id": f"pair_{index:02d}",
                "physical_slots": [f"H{index - 6:02d}", f"H{index:02d}"],
                "base_manifest": _local_ref(base_manifest, root),
                "state_prefix": _local_ref(state_path, root),
                "state_append": state_append,
                "tasks": task_specs,
            }
        )

    archive = root / "formal.tar.gz"
    with tarfile.open(archive, mode="w:gz") as handle:
        for member, content in sorted(payloads.items()):
            info = tarfile.TarInfo(member)
            info.size = len(content)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(content))
    archive_manifest = root / "formal_manifest.tsv"
    manifest_text = "sha256\tsize\trelative_path\n" + "".join(
        f"{hashlib.sha256(content).hexdigest()}\t{len(content)}\t{member}\n"
        for member, content in sorted(payloads.items())
    )
    archive_manifest.write_text(manifest_text, encoding="utf-8", newline="")
    _write_json(
        source_receipt,
        {
            "schema_version": "task8b-paired-shard-cross-host-export-v1",
            "status": "complete",
            "effect_fields_read": False,
            "archive_sha256": _sha256(archive),
            "manifest_sha256": _sha256(archive_manifest),
        },
    )
    _write_json(
        source_seal,
        {
            "schema_version": "task8b-full-matrix-export-seal-v1",
            "status": "sealed",
            "effect_fields_read": False,
            "archive_sha256": _sha256(archive),
            "manifest_sha256": _sha256(archive_manifest),
            "export_receipt_sha256": _sha256(source_receipt),
        },
    )
    manifest_ref = _local_ref(archive_manifest, root)
    for worker in workers:
        worker_id = str(worker["worker_id"])
        role = str(worker["role"])
        index = int(worker_id[1:])
        seed = 2026090100 + index
        pair_id = f"pair_{index:02d}"
        low_slot = f"H{index - 6:02d}"
        high_slot = f"H{index:02d}"
        task_run_dirs = {
            str(task["task_id"]): str(task["target_run_dir"])
            for task in worker["tasks"]  # type: ignore[union-attr]
            if isinstance(task, dict)
        }
        side_specs = (
            (
                ("high", high_slot, PRIMARY_TASKS[:4]),
                ("low", low_slot, PRIMARY_TASKS[4:]),
            )
            if role == "primary"
            else (
                ("low", low_slot, SECONDARY_TASKS[:1]),
                ("high", high_slot, SECONDARY_TASKS[1:]),
            )
        )
        completion_path = root / "completions" / f"{worker_id}.json"
        _write_json(
            completion_path,
            {
                "schema_version": "task8-worker-completion-v1",
                "worker_id": worker_id,
                "status": "complete",
                "effect_fields_read": False,
                "files_tsv_sha256": hashlib.sha256(
                    payloads[f"payload/{worker_id}/files.tsv"]
                ).hexdigest(),
            },
        )
        completion_ref = _local_ref(
            completion_path,
            root,
            json_assertions={
                "status": "complete",
                "effect_fields_read": False,
                "worker_id": worker_id,
            },
        )
        shards = []
        for side, slot, side_tasks in side_specs:
            identity_path = (
                root / "secondary_normalization" / worker_id / side / "shard_seal.json"
                if role == "secondary" and side == "low"
                else root / "shard_receipts" / worker_id / f"{side}.json"
            )
            _write_json(
                identity_path,
                {
                    "schema_version": "task8b-paired-shard-receipt-v1",
                    "status": "complete",
                    "effect_fields_read": False,
                    "worker_id": worker_id,
                    "role": role,
                    "seed": seed,
                    "pair_id": pair_id,
                    "physical_slot": slot,
                    "shard_role": side,
                    "shard_id": f"{worker_id}-{side}-fixture",
                    "source_kind": "execution",
                    "selected_task_ids": list(side_tasks),
                    "tasks": [
                        {
                            "schema_version": "task8b-paired-shard-task-v1",
                            "status": "complete",
                            "producer_worker_id": worker_id,
                            "task_id": task_id,
                            "run_dir": task_run_dirs[task_id],
                            "planned_hands": (
                                2700
                                if task_id == "mixed_ecological"
                                else 600
                                if task_id in SECONDARY_TASKS[1:]
                                else 1350
                            ),
                            "actual_hands": (
                                2700
                                if task_id == "mixed_ecological"
                                else 600
                                if task_id in SECONDARY_TASKS[1:]
                                else 1350
                            ),
                            "task_receipt_sha256": hashlib.sha256(
                                payloads[f"payload/{worker_id}/task_receipts/{task_id}.json"]
                            ).hexdigest(),
                        }
                        for task_id in side_tasks
                    ],
                },
            )
            identity_ref = _local_ref(
                identity_path,
                root,
                json_assertions={
                    "status": "complete",
                    "effect_fields_read": False,
                    "worker_id": worker_id,
                    "pair_id": pair_id,
                    "physical_slot": slot,
                },
            )
            execution_receipts = []
            if role == "secondary" and side == "low":
                schemas = {
                    "certificate.identity_correction_audit.json": (
                        "task8b-paired-identity-correction-audit-v1"
                    ),
                    "certificate.json": ("task8b-paired-completed-recovery-certificate-v1"),
                    "ledger.json": "task8b-paired-recovery-ledger-entry-v1",
                }
                for name, schema in schemas.items():
                    evidence_path = identity_path.parent / name
                    _write_json(
                        evidence_path,
                        {
                            "schema_version": schema,
                            "worker_id": worker_id,
                            "role": role,
                            "seed": seed,
                            "pair_id": pair_id,
                            "physical_slot": slot,
                            "effect_fields_read": False,
                        },
                    )
                    execution_receipts.append(_local_ref(evidence_path, root))
                execution_receipts.append(copy.deepcopy(identity_ref))
            shards.append(
                {
                    "side": side,
                    "shard_id": f"{worker_id}-{side}",
                    "source_archive_ids": ["formal"],
                    "shard_manifest": copy.deepcopy(manifest_ref),
                    "completion_receipt": copy.deepcopy(completion_ref),
                    "identity_receipts": [identity_ref],
                    "execution_receipts": execution_receipts,
                }
            )
        worker["shards"] = shards
        if worker_id in RECOVERED_WORKERS:
            worker["recovery_adoptions"] = {
                key: _local_ref(path, root) for key, path in recovery_paths[worker_id].items()
            }
        assert int(worker["seed"]) == 2026090100 + index
    source_map: dict[str, object] = {
        "schema_version": SOURCE_MAP_SCHEMA,
        "materialization_id": "fixture-six-seed",
        "protocol_id": PROTOCOL_ID,
        "archives": {
            "formal": {
                "path": archive.relative_to(root).as_posix(),
                "size": archive.stat().st_size,
                "sha256": _sha256(archive),
                "manifest": {
                    **manifest_ref,
                    "format": "sha256-size-path-tsv-v1",
                    "has_header": True,
                },
                "receipt": _local_ref(
                    source_receipt,
                    root,
                    json_assertions={
                        "status": "complete",
                        "effect_fields_read": False,
                    },
                ),
                "seal": _local_ref(
                    source_seal,
                    root,
                    json_assertions={
                        "status": "sealed",
                        "effect_fields_read": False,
                    },
                ),
            }
        },
        "analysis": {
            "analysis_contract": _local_ref(contract, root),
            "pre_unlock_manifest": _local_ref(preunlock, root),
            "unlocked_at_utc": "2026-07-25T15:00:00+00:00",
            "unlocked_by": "pytest-materializer",
        },
        "workers": workers,
    }
    source_map_path = root / "source_map.json"
    _write_json(source_map_path, source_map)
    return source_map_path, source_map


def _rebind_formal_archive(
    source_map_path: Path,
    source_map: dict[str, object],
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_format: str,
    has_header: bool,
    profile: str,
) -> None:
    root = source_map_path.parent
    archives = source_map["archives"]
    assert isinstance(archives, dict)
    formal = archives["formal"]
    assert isinstance(formal, dict)
    manifest_ref = _local_ref(manifest_path, root)
    formal.update(
        {
            "path": archive_path.relative_to(root).as_posix(),
            "size": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
            "manifest": {
                **manifest_ref,
                "format": manifest_format,
                "has_header": has_header,
            },
        }
    )
    receipt = root / "source_receipt.json"
    seal = root / "source_seal.json"
    if profile == "local-verification":
        receipt_value = {
            "schema": "task8b-local-export-verification-v1",
            "archive_sha256": _sha256(archive_path),
            "manifest_sha256": _sha256(manifest_path),
            "verified_file_count": 1,
            "archive_members_exact": True,
            "all_member_sha256_verified": True,
            "effect_fields_read": False,
        }
        receipt_assertions = {
            "schema": "task8b-local-export-verification-v1",
            "effect_fields_read": False,
        }
        _write_json(receipt, receipt_value)
        _write_json(
            seal,
            {
                "schema_version": "task8b-full-matrix-export-seal-v1",
                "status": "sealed",
                "archive_sha256": _sha256(archive_path),
                "manifest_sha256": _sha256(manifest_path),
                "export_receipt_sha256": _sha256(receipt),
                "effect_fields_read": False,
            },
        )
        seal_assertions = {
            "status": "sealed",
            "effect_fields_read": False,
        }
    elif profile == "cross-host":
        _write_json(
            receipt,
            {
                "schema_version": "task8b-paired-shard-cross-host-export-v1",
                "status": "complete",
                "archive_sha256": _sha256(archive_path),
                "manifest_sha256": _sha256(manifest_path),
                "effect_fields_read": False,
            },
        )
        receipt_assertions = {
            "status": "complete",
            "effect_fields_read": False,
        }
        _write_json(
            seal,
            {
                "schema_version": "task8b-paired-shard-cross-host-bundle-v1",
                "status": "sealed",
                "effect_fields_read": False,
            },
        )
        seal_assertions = {
            "status": "sealed",
            "effect_fields_read": False,
        }
    else:
        assert profile == "recovery-support"
        _write_json(
            receipt,
            {
                "schema_version": "task8b-recovery-support-export-receipt-v1",
                "status": "source_manifest_complete",
                "archive_sha256": _sha256(archive_path),
                "server_manifest_sha256": _sha256(manifest_path),
                "effect_fields_read": False,
            },
        )
        receipt_assertions = {
            "status": "source_manifest_complete",
            "effect_fields_read": False,
        }
        _write_json(
            seal,
            {
                "schema_version": "task8b-recovery-support-export-seal-v1",
                "status": "sealed",
                "archive_sha256": _sha256(archive_path),
                "server_manifest_sha256": _sha256(manifest_path),
                "export_receipt_sha256": _sha256(receipt),
                "effect_fields_read": False,
            },
        )
        seal_assertions = {
            "status": "sealed",
            "effect_fields_read": False,
        }
    formal["receipt"] = _local_ref(
        receipt,
        root,
        json_assertions=receipt_assertions,
    )
    formal["seal"] = _local_ref(
        seal,
        root,
        json_assertions=seal_assertions,
    )
    workers = source_map["workers"]
    assert isinstance(workers, list)
    for worker in workers:
        assert isinstance(worker, dict)
        shards = worker["shards"]
        assert isinstance(shards, list)
        for shard in shards:
            assert isinstance(shard, dict)
            shard["shard_manifest"] = copy.deepcopy(manifest_ref)


def _rewrite_tar_member_as_directory(
    source: Path,
    target: Path,
    *,
    member_name: str,
) -> None:
    with (
        tarfile.open(source, mode="r:*") as input_archive,
        tarfile.open(target, mode="w:gz") as output_archive,
    ):
        for member in input_archive:
            if member.name == member_name:
                directory = tarfile.TarInfo(member_name)
                directory.type = tarfile.DIRTYPE
                directory.mtime = 0
                output_archive.addfile(directory)
                continue
            extracted = input_archive.extractfile(member)
            if extracted is None:
                output_archive.addfile(member)
            else:
                with extracted:
                    output_archive.addfile(member, extracted)


def _replace_worker_state_prefix(
    source_map_path: Path,
    source_map: dict[str, object],
    *,
    worker_id: str,
    statuses: list[str],
) -> dict[str, object]:
    workers = source_map["workers"]
    assert isinstance(workers, list)
    worker = next(
        row
        for row in workers
        if isinstance(row, dict) and row.get("worker_id") == worker_id
    )
    state_ref = worker["state_prefix"]
    assert isinstance(state_ref, dict)
    state_path = source_map_path.parent / str(state_ref["path"])
    payload, _rows = _state_bytes(statuses, worker_id=worker_id)
    state_path.write_bytes(payload)
    worker["state_prefix"] = _local_ref(state_path, source_map_path.parent)
    return worker


def test_materializer_cli_builds_blind_complete_append_only_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_map, value = _make_source_map(tmp_path)
    output = tmp_path / "canonical"

    assert main(["--source-map", str(source_map), "--output-root", str(output)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["worker_count"] == 12
    assert result["formal_effect_fields_read"] is False
    assert (output / "exclusion_ledger.csv").read_bytes() == (
        materializer_module.EXCLUSION_LEDGER_BYTES
    )
    root_manifest = output / "canonical_file_manifest.tsv"
    root_seal = output / "canonical_root_seal.json"
    assert result["canonical_file_manifest_sha256"] == _sha256(
        root_manifest
    )
    assert result["canonical_root_seal_sha256"] == _sha256(root_seal)
    with root_manifest.open(encoding="utf-8", newline="") as handle:
        root_rows = list(csv.DictReader(handle, delimiter="\t"))
    excluded = {
        root_manifest.relative_to(output).as_posix(),
        root_seal.relative_to(output).as_posix(),
    }
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert {row["relative_path"] for row in root_rows} == (
        actual_paths - excluded
    )
    for row in root_rows:
        path = output.joinpath(*Path(row["relative_path"]).parts)
        assert int(row["size"]) == path.stat().st_size
        assert row["sha256"] == _sha256(path)
    seal = json.loads(root_seal.read_text(encoding="utf-8"))
    assert seal["canonical_file_manifest"]["sha256"] == _sha256(
        root_manifest
    )
    assert seal["published_tree_file_count"] == len(actual_paths)
    assert seal["authoritative_real_profile"] is False
    assert seal["status"] == "sealed_with_declared_limitation"
    assert (
        seal["paper_eligibility"]
        == materializer_module.CANONICAL_PAPER_ELIGIBILITY
    )
    assert seal["limitations"] == materializer_module._declared_limitations()
    analysis_input = json.loads((output / "analysis_input.json").read_text(encoding="utf-8"))
    assert [row["worker_id"] for row in analysis_input["workers"]] == list(EXPECTED_WORKERS)
    assert analysis_input["synthetic_test_mode"] is False
    assert (
        task8b_analysis._analysis_protocol_for_manifest(analysis_input)["analysis_contract_id"]
        == ANALYSIS_CONTRACT_ID
    )
    task8b_analysis._validate_analysis_input_topology(
        analysis_input["workers"],
        task8b_analysis._load_six_seed_analysis_protocol(),
        manifest_root=output,
    )
    p07_attempt = output / "snapshots" / "P07" / "2026090107" / "attempt_01"
    assert (
        p07_attempt / "runs" / "isolation_no_memory" / "metrics.json"
    ).read_bytes() == b"FORMAL_EFFECT_PAYLOAD_IS_OPAQUE\n"
    p12_attempt = output / "snapshots" / "P12" / "2026090112" / "attempt_01"
    p12_sync = p12_attempt / "runs" / "isolation_sync__attempt_02"
    assert p12_sync.is_dir()
    assert not (p12_attempt / "runs" / "isolation_sync").exists()
    p12_receipt_path = p12_attempt / "task_receipts" / "isolation_sync.json"
    p12_receipt = json.loads(p12_receipt_path.read_text(encoding="utf-8"))
    assert p12_receipt["run_dir"] == "runs/isolation_sync__attempt_02"
    p12_task_results = json.loads((p12_attempt / "task_results.json").read_text(encoding="utf-8"))
    p12_sync_result = next(
        row for row in p12_task_results["tasks"] if row["task_id"] == "isolation_sync"
    )
    assert p12_sync_result["run_dir"] == "runs/isolation_sync__attempt_02"
    archive_row = value["archives"]["formal"]  # type: ignore[index]
    archive_path = source_map.parent / str(archive_row["path"])  # type: ignore[index]
    with tarfile.open(archive_path, mode="r:*") as handle:
        source_receipt = handle.extractfile("payload/P12/task_receipts/isolation_sync.json")
        assert source_receipt is not None
        with source_receipt:
            assert p12_receipt_path.read_bytes() == source_receipt.read()
    state = (p07_attempt / "state.tsv").read_text(encoding="utf-8")
    assert "\tfailed\t" in state
    assert state.rstrip().splitlines()[-1].split("\t")[2] == "complete"
    assert {
        path.name for path in (p07_attempt / "recovery_adoptions").iterdir() if path.is_file()
    } == {
        "isolation_no_memory.canonicalization_audit.json",
        "isolation_no_memory.adoption_attestation.json",
        "isolation_no_memory.json",
    }
    composition = json.loads(
        (output / "manifests" / "composition" / "P07.json").read_text(encoding="utf-8")
    )
    assert composition["schema_version"] == COMPOSITION_SCHEMA
    assert all(side["execution_receipts"] for side in composition["sides"])
    assert all(side["source_shards"] for side in composition["sides"])
    s07_composition = json.loads(
        (output / "manifests" / "composition" / "S07.json").read_text(encoding="utf-8")
    )
    s07_low = next(side for side in s07_composition["sides"] if side["side"] == "low")
    assert {
        Path(row["path"]).name
        for row in s07_low["execution_receipts"]
        if Path(row["path"]).name
        in {
            "certificate.json",
            "certificate.identity_correction_audit.json",
            "ledger.json",
            "shard_seal.json",
        }
    } == {
        "certificate.json",
        "certificate.identity_correction_audit.json",
        "ledger.json",
        "shard_seal.json",
    }
    manifest = output / "manifests" / "P07.json"
    attempt_manifest = p07_attempt / "worker_manifest.json"
    assert manifest.read_bytes() == attempt_manifest.read_bytes()
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        == analysis_input["workers"][0]["expected_worker_manifest_sha256"]
    )
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        materialize_task8b_phase_f(source_map, output)


def test_materializer_copies_provenance_for_recovery_only_archive(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p07 = next(
        worker
        for worker in workers
        if isinstance(worker, dict) and worker.get("worker_id") == "P07"
    )
    recovery = p07["recovery_adoptions"]
    assert isinstance(recovery, dict)
    canonical_ref = recovery["canonicalization_audit"]
    assert isinstance(canonical_ref, dict)
    canonical_path = root / str(canonical_ref["path"])
    canonical_payload = canonical_path.read_bytes()
    member = "recovery/P07/canonicalization_audit.json"

    recovery_archive = root / "recovery_H07.tar.gz"
    with tarfile.open(recovery_archive, mode="w:gz") as handle:
        info = tarfile.TarInfo(member)
        info.size = len(canonical_payload)
        info.mtime = 0
        handle.addfile(info, io.BytesIO(canonical_payload))
    recovery_manifest = root / "recovery_H07_server_manifest.tsv"
    recovery_manifest.write_text(
        "sha256\tsize\trelative_path\n"
        f"{hashlib.sha256(canonical_payload).hexdigest()}"
        f"\t{len(canonical_payload)}\t{member}\n",
        encoding="utf-8",
        newline="",
    )
    recovery_receipt = root / "recovery_H07_export_receipt.json"
    _write_json(
        recovery_receipt,
        {
            "schema_version": "task8b-recovery-support-export-receipt-v1",
            "status": "source_manifest_complete",
            "archive_sha256": _sha256(recovery_archive),
            "server_manifest_sha256": _sha256(recovery_manifest),
            "effect_fields_read": False,
        },
    )
    recovery_seal = root / "recovery_H07_export_seal.json"
    _write_json(
        recovery_seal,
        {
            "schema_version": "task8b-recovery-support-export-seal-v1",
            "status": "sealed",
            "archive_sha256": _sha256(recovery_archive),
            "server_manifest_sha256": _sha256(recovery_manifest),
            "export_receipt_sha256": _sha256(recovery_receipt),
            "effect_fields_read": False,
        },
    )

    archives = source_map["archives"]
    assert isinstance(archives, dict)
    archives["recovery_H07"] = {
        "path": recovery_archive.relative_to(root).as_posix(),
        "size": recovery_archive.stat().st_size,
        "sha256": _sha256(recovery_archive),
        "manifest": {
            **_local_ref(recovery_manifest, root),
            "format": "sha256-size-path-tsv-v1",
            "has_header": True,
        },
        "receipt": _local_ref(
            recovery_receipt,
            root,
            json_assertions={
                "status": "source_manifest_complete",
                "effect_fields_read": False,
            },
        ),
        "seal": _local_ref(
            recovery_seal,
            root,
            json_assertions={"status": "sealed", "effect_fields_read": False},
        ),
    }
    recovery["canonicalization_audit"] = {
        "archive_id": "recovery_H07",
        "member": member,
        "size": len(canonical_payload),
        "sha256": hashlib.sha256(canonical_payload).hexdigest(),
    }
    recovery_map = root / "source_map_recovery_archive.json"
    _write_json(recovery_map, source_map)
    output = tmp_path / "recovery-archive-output"

    materialize_task8b_phase_f(recovery_map, output)

    copied = {
        output
        / "provenance"
        / "archive_manifests"
        / "recovery_H07.tsv": recovery_manifest,
        output
        / "provenance"
        / "archive_receipts"
        / "recovery_H07.json": recovery_receipt,
        output
        / "provenance"
        / "archive_seals"
        / "recovery_H07.json": recovery_seal,
    }
    for destination, source in copied.items():
        assert destination.read_bytes() == source.read_bytes()
        assert _sha256(destination) == _sha256(source)
    assert (
        output
        / "snapshots"
        / "P07"
        / "2026090107"
        / "attempt_01"
        / "recovery_adoptions"
        / "isolation_no_memory.canonicalization_audit.json"
    ).read_bytes() == canonical_payload


def test_materializer_accepts_recovery_failed_anchor_with_valid_prefix_suffix(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    p07 = _replace_worker_state_prefix(
        source_map_path,
        source_map,
        worker_id="P07",
        statuses=["validating", "failed", "validating", "running"],
    )
    state_append = p07["state_append"]
    assert isinstance(state_append, list)
    p07["state_append"] = state_append[-2:]
    variant = source_map_path.parent / "recovery_middle_anchor_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "recovery-middle-anchor-output"

    materialize_task8b_phase_f(variant, output)

    state_path = output / "snapshots" / "P07" / "2026090107" / "attempt_01" / "state.tsv"
    statuses = [
        row["status"]
        for row in csv.DictReader(
            io.StringIO(state_path.read_text(encoding="utf-8")),
            delimiter="\t",
        )
    ]
    assert statuses == [
        "validating",
        "failed",
        "validating",
        "running",
        "finalizing",
        "complete",
    ]


def test_materializer_rejects_recovery_state_suffix_outside_authorized_sequence(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    _replace_worker_state_prefix(
        source_map_path,
        source_map,
        worker_id="P07",
        statuses=["validating", "failed", "validating", "waiting_dependency"],
    )
    variant = source_map_path.parent / "bad_recovery_suffix_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "bad-recovery-suffix-output"
    partial = tmp_path / f"bad-recovery-suffix-output.partial-{_sha256(variant)[:12]}"

    with pytest.raises(ConfigError, match="recovery state suffix"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not partial.exists()


def _real_recovery_prefix_rows() -> list[dict[str, str]]:
    payload, _rows = _state_bytes(
        ["planned", "validating", "running", "failed", "validating", "running"],
        worker_id="P07",
    )
    return _parse_state_bytes(payload, label="P07 real recovery prefix")


def test_recovery_state_anchor_accepts_real_six_row_prefix_and_appends_tail() -> None:
    rows = _real_recovery_prefix_rows()
    anchor_sha = rows[3]["row_sha256"]

    anchor_index = _validate_recovery_state_anchor(
        rows,
        original_terminal_state_sha256=anchor_sha,
        label="P07",
    )
    appended = _build_state_rows(
        rows,
        [
            {
                "created_at_utc": "2026-07-25T14:27:00+00:00",
                "status": "finalizing",
                "detail": "phase-f canonical completion sealing",
            },
            {
                "created_at_utc": "2026-07-25T14:28:00+00:00",
                "status": "complete",
                "detail": "phase-f canonical completion published",
            },
        ],
        label="P07",
    )

    assert anchor_index == 3
    assert [row["status"] for row in rows + appended] == [
        "planned",
        "validating",
        "running",
        "failed",
        "validating",
        "running",
        "finalizing",
        "complete",
    ]
    assert appended[0]["previous_sha256"] == rows[-1]["row_sha256"]
    assert appended[1]["previous_sha256"] == appended[0]["row_sha256"]


def test_recovery_state_anchor_rejects_state_without_failed_anchor() -> None:
    payload, _rows = _state_bytes(
        ["planned", "validating", "running"],
        worker_id="P07",
    )
    rows = _parse_state_bytes(payload, label="P07 no failed state")

    with pytest.raises(ConfigError, match="anchor 不唯一或缺失"):
        _validate_recovery_state_anchor(
            rows,
            original_terminal_state_sha256="0" * 64,
            label="P07",
        )


def test_recovery_state_anchor_rejects_duplicate_matching_anchor() -> None:
    rows = _real_recovery_prefix_rows()
    anchor_sha = rows[3]["row_sha256"]
    duplicate_rows = copy.deepcopy(rows)
    duplicate_rows[-1]["row_sha256"] = anchor_sha

    with pytest.raises(ConfigError, match="anchor 不唯一或缺失"):
        _validate_recovery_state_anchor(
            duplicate_rows,
            original_terminal_state_sha256=anchor_sha,
            label="P07",
        )


def test_recovery_state_anchor_rejects_matching_non_failed_row() -> None:
    rows = _real_recovery_prefix_rows()

    with pytest.raises(ConfigError, match="anchor 非 failed"):
        _validate_recovery_state_anchor(
            rows,
            original_terminal_state_sha256=rows[2]["row_sha256"],
            label="P07",
        )


def test_recovery_state_anchor_rejects_certificate_sha_not_in_state() -> None:
    rows = _real_recovery_prefix_rows()

    with pytest.raises(ConfigError, match="anchor 不唯一或缺失"):
        _validate_recovery_state_anchor(
            rows,
            original_terminal_state_sha256="f" * 64,
            label="P07",
        )


def test_recovery_state_parser_rejects_broken_sha_chain() -> None:
    payload, rows = _state_bytes(
        ["planned", "validating", "running", "failed", "validating", "running"],
        worker_id="P07",
    )
    assert payload
    rows[4]["previous_sha256"] = "0" * 64
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0]),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

    with pytest.raises(ConfigError, match="state hash chain"):
        _parse_state_bytes(
            handle.getvalue().encode("utf-8"),
            label="P07 broken state",
        )


def test_materializer_rejects_archive_sha_before_destination(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    source_map["archives"]["formal"]["sha256"] = "0" * 64  # type: ignore[index]
    bad_map = source_map_path.parent / "bad_archive_sha.json"
    _write_json(bad_map, source_map)
    output = tmp_path / "bad-archive-output"

    with pytest.raises(ConfigError, match="size/SHA-256"):
        materialize_task8b_phase_f(bad_map, output)

    assert not output.exists()
    assert not list(tmp_path.glob("bad-archive-output.partial-*"))


def test_materializer_rejects_bad_receipt_before_creating_partial(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p07 = workers[0]
    assert isinstance(p07, dict)
    tasks = p07["tasks"]
    assert isinstance(tasks, list)
    first_task, second_task = tasks[:2]
    assert isinstance(first_task, dict) and isinstance(second_task, dict)
    first_receipt = first_task["receipt"]
    first_task["receipt"] = second_task["receipt"]
    second_task["receipt"] = first_receipt
    bad_map = source_map_path.parent / "bad_receipt_map.json"
    _write_json(bad_map, source_map)
    output = tmp_path / "bad-receipt-output"
    source_sha = _sha256(bad_map)
    partial = tmp_path / f"bad-receipt-output.partial-{source_sha[:12]}"

    with pytest.raises(ConfigError, match="task receipt 不闭合"):
        materialize_task8b_phase_f(bad_map, output)

    assert not output.exists()
    assert not partial.exists()


def test_materializer_rejects_changed_state_prefix_sha(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p07 = workers[0]
    assert isinstance(p07, dict)
    state_ref = p07["state_prefix"]
    assert isinstance(state_ref, dict)
    state_path = source_map_path.parent / str(state_ref["path"])
    state_path.write_bytes(state_path.read_bytes() + b"tamper\n")
    output = tmp_path / "bad-state-output"

    with pytest.raises(ConfigError, match="size/SHA-256"):
        materialize_task8b_phase_f(source_map_path, output)

    assert not output.exists()


def test_materializer_hashes_unselected_archive_members(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    manifest = source_map_path.parent / "formal_manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    rows = [
        (
            f"{'0' * 64}\t{row.split(chr(9), 2)[1]}\t{row.split(chr(9), 2)[2]}"
            if row.endswith("payload/unselected/provenance.bin")
            else row
        )
        for row in rows
    ]
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=source_map_path.parent / "formal.tar.gz",
        manifest_path=manifest,
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="cross-host",
    )
    bad_map = source_map_path.parent / "bad_member_manifest.json"
    _write_json(bad_map, source_map)
    output = tmp_path / "bad-member-output"
    partial = tmp_path / f"bad-member-output.partial-{_sha256(bad_map)[:12]}"

    with pytest.raises(ConfigError, match="stream size/SHA-256"):
        materialize_task8b_phase_f(bad_map, output)

    assert not output.exists()
    assert partial.is_dir()


def test_materializer_ignores_unrequested_regular_export_control_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    source_archive = root / "formal.tar.gz"
    archive = root / "formal_with_unmanifested_control.tar.gz"
    with (
        tarfile.open(source_archive, mode="r:gz") as source,
        tarfile.open(archive, mode="w:gz") as target,
    ):
        for member in source:
            assert member.isreg()
            extracted = source.extractfile(member)
            assert extracted is not None
            target.addfile(member, extracted)
        payload = b"opaque export-control provenance\n"
        info = tarfile.TarInfo("task8b_full_matrix_exports/H01/source_roots.tsv")
        info.size = len(payload)
        target.addfile(info, io.BytesIO(payload))
    manifest = root / "formal_manifest.tsv"
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=archive,
        manifest_path=manifest,
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="recovery-support",
    )
    variant = root / "unmanifested_control_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "unmanifested-control-canonical"
    plan = _prepare_materialization(
        variant,
        tmp_path / "unmanifested-control-plan",
        authoritative_real_profile=False,
    )
    allowed_path = "task8b_full_matrix_exports/H01/source_roots.tsv"
    monkeypatch.setattr(
        materializer_module,
        "AUTHORITATIVE_UNMANIFESTED_REGULAR_MEMBERS",
        {"formal": frozenset({allowed_path})},
    )
    materializer_module._validate_archive_copy_plan(
        requests=plan.requests,
        archives=plan.archives,
        authoritative_real_profile=True,
    )
    monkeypatch.setattr(
        materializer_module,
        "AUTHORITATIVE_UNMANIFESTED_REGULAR_MEMBERS",
        {"formal": frozenset({"unapproved/hidden.bin"})},
    )
    with pytest.raises(ConfigError, match="allowlist 不闭合"):
        materializer_module._validate_archive_copy_plan(
            requests=plan.requests,
            archives=plan.archives,
            authoritative_real_profile=True,
        )

    result = materialize_task8b_phase_f(variant, output)

    assert result["status"] == "complete"
    assert result["formal_effect_fields_read"] is False
    assert not list(output.rglob("source_roots.tsv"))


def test_materializer_still_requires_requested_member_in_manifest(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    manifest = root / "formal_manifest.tsv"
    requested_suffix = "payload/P07/task_receipts/isolation_no_memory.json"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text(
        "\n".join(row for row in rows if not row.endswith(requested_suffix)) + "\n",
        encoding="utf-8",
        newline="",
    )
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=root / "formal.tar.gz",
        manifest_path=manifest,
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="recovery-support",
    )
    variant = root / "requested_member_missing_manifest.json"
    _write_json(variant, source_map)
    output = tmp_path / "requested-member-missing-output"

    with pytest.raises(ConfigError, match="archive manifest 不含 member"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()


def test_materializer_still_hashes_each_requested_member(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    source_archive = root / "formal.tar.gz"
    archive = root / "formal_with_tampered_requested_member.tar.gz"
    target_member = "payload/P07/runs/isolation_no_memory/metrics.json"
    with (
        tarfile.open(source_archive, mode="r:gz") as source,
        tarfile.open(archive, mode="w:gz") as target,
    ):
        for member in source:
            assert member.isreg()
            extracted = source.extractfile(member)
            assert extracted is not None
            payload = extracted.read()
            if member.name == target_member:
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            target.addfile(member, io.BytesIO(payload))
    manifest = root / "formal_manifest.tsv"
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=archive,
        manifest_path=manifest,
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="recovery-support",
    )
    variant = root / "tampered_requested_member_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "tampered-requested-member-output"
    partial = tmp_path / f"tampered-requested-member-output.partial-{_sha256(variant)[:12]}"

    with pytest.raises(ConfigError, match="stream size/SHA-256"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert partial.is_dir()


def test_materializer_rejects_mixed_archive_seal_binding(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    seal = source_map_path.parent / "source_seal.json"
    seal_value = json.loads(seal.read_text(encoding="utf-8"))
    seal_value["archive_sha256"] = "0" * 64
    _write_json(seal, seal_value)
    archive = source_map["archives"]["formal"]  # type: ignore[index]
    assert isinstance(archive, dict)
    archive["seal"] = _local_ref(
        seal,
        source_map_path.parent,
        json_assertions={"status": "sealed", "effect_fields_read": False},
    )
    bad_map = source_map_path.parent / "mixed_seal_map.json"
    _write_json(bad_map, source_map)
    output = tmp_path / "mixed-seal-output"
    partial = tmp_path / f"mixed-seal-output.partial-{_sha256(bad_map)[:12]}"

    with pytest.raises(ConfigError, match="cross-binding"):
        materialize_task8b_phase_f(bad_map, output)

    assert not output.exists()
    assert not partial.exists()


def test_materializer_accepts_no_header_transport_manifest_and_local_verification(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    manifest = root / "server_manifest.tsv"
    rows = (root / "formal_manifest.tsv").read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8", newline="")
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=root / "formal.tar.gz",
        manifest_path=manifest,
        manifest_format="sha256-size-path-tsv-v1",
        has_header=False,
        profile="local-verification",
    )
    variant = root / "no_header_source_map.json"
    _write_json(variant, source_map)

    result = materialize_task8b_phase_f(variant, tmp_path / "no-header-canonical")

    assert result["status"] == "complete"
    assert result["formal_effect_fields_read"] is False


def test_materializer_accepts_files_json_cross_host_summary_and_bundle_profile(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    rows = (root / "formal_manifest.tsv").read_text(encoding="utf-8").splitlines()[1:]
    manifest = root / "cross_host_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "task8b-paired-shard-cross-host-bundle-v1",
            "status": "sealed",
            "effect_fields_read": False,
            "files": [
                {
                    "sha256": row.split("\t", 2)[0],
                    "size": int(row.split("\t", 2)[1]),
                    "relative_path": row.split("\t", 2)[2],
                }
                for row in rows
            ],
        },
    )
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=root / "formal.tar.gz",
        manifest_path=manifest,
        manifest_format="files-json-v1",
        has_header=False,
        profile="cross-host",
    )
    variant = root / "files_json_source_map.json"
    _write_json(variant, source_map)

    result = materialize_task8b_phase_f(variant, tmp_path / "files-json-canonical")

    assert result["status"] == "complete"
    assert result["formal_effect_fields_read"] is False


def test_materializer_accepts_recovery_support_export_profile(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=root / "formal.tar.gz",
        manifest_path=root / "formal_manifest.tsv",
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="recovery-support",
    )
    variant = root / "recovery_support_source_map.json"
    _write_json(variant, source_map)

    result = materialize_task8b_phase_f(
        variant,
        tmp_path / "recovery-support-canonical",
    )

    assert result["status"] == "complete"
    assert result["formal_effect_fields_read"] is False


def test_materializer_rejects_windows_backslash_source_prefix(tmp_path: Path) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    workers = source_map["workers"]
    assert isinstance(workers, list)
    first = workers[0]
    assert isinstance(first, dict)
    tasks = first["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    task["source_prefix"] = str(task["source_prefix"]).replace("/", "\\")
    variant = source_map_path.parent / "backslash_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "backslash-canonical"

    with pytest.raises(ConfigError, match="相对路径非法"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not list(tmp_path.glob("backslash-canonical.partial-*"))


def test_materializer_rejects_attempt_target_not_bound_to_source_prefix(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p12 = next(worker for worker in workers if worker["worker_id"] == "P12")
    assert isinstance(p12, dict)
    tasks = p12["tasks"]
    assert isinstance(tasks, list)
    sync = next(task for task in tasks if task["task_id"] == "isolation_sync")
    assert isinstance(sync, dict)
    sync["target_run_dir"] = "runs/isolation_sync__attempt_03"
    variant = source_map_path.parent / "attempt_target_mismatch_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "attempt-target-mismatch-canonical"

    with pytest.raises(ConfigError, match="source_prefix 必须以 target_run_dir 结尾"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not list(tmp_path.glob("attempt-target-mismatch-canonical.partial-*"))


def test_materializer_rejects_tar_directory_masquerading_as_manifest_file(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    rewritten = root / "directory_masquerade.tar.gz"
    _rewrite_tar_member_as_directory(
        root / "formal.tar.gz",
        rewritten,
        member_name="payload/unselected/provenance.bin",
    )
    _rebind_formal_archive(
        source_map_path,
        source_map,
        archive_path=rewritten,
        manifest_path=root / "formal_manifest.tsv",
        manifest_format="sha256-size-path-tsv-v1",
        has_header=True,
        profile="cross-host",
    )
    variant = root / "directory_masquerade_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "directory-masquerade-canonical"
    partial = tmp_path / (f"directory-masquerade-canonical.partial-{_sha256(variant)[:12]}")

    with pytest.raises(ConfigError, match="exact closure"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not partial.exists()


def test_materializer_rejects_non_authoritative_contract_with_same_id(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    contract = root / "forged_contract.json"
    _write_json(contract, {"analysis_contract_id": ANALYSIS_CONTRACT_ID})
    analysis = source_map["analysis"]
    assert isinstance(analysis, dict)
    analysis["analysis_contract"] = _local_ref(contract, root)
    variant = root / "forged_contract_source_map.json"
    _write_json(variant, source_map)
    output = tmp_path / "forged-contract-canonical"

    with pytest.raises(ConfigError, match="权威 six-seed"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not list(tmp_path.glob("forged-contract-canonical.partial-*"))


def test_atomic_publish_primitive_never_replaces_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complete-staging"
    destination = tmp_path / "canonical"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source\n", encoding="utf-8")
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _atomic_rename_noreplace(source, destination)

    assert (source / "source.txt").read_text(encoding="utf-8") == "source\n"
    assert sentinel.read_text(encoding="utf-8") == "existing\n"


def test_materializer_rejects_source_shard_id_reused_across_sides(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    root = source_map_path.parent
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p07 = workers[0]
    assert isinstance(p07, dict)
    shards = p07["shards"]
    assert isinstance(shards, list)
    high, low = shards
    assert isinstance(high, dict) and isinstance(low, dict)
    high_refs = high["identity_receipts"]
    low_refs = low["identity_receipts"]
    assert isinstance(high_refs, list) and isinstance(low_refs, list)
    high_ref = high_refs[0]
    low_ref = low_refs[0]
    assert isinstance(high_ref, dict) and isinstance(low_ref, dict)
    high_path = root / str(high_ref["path"])
    low_path = root / str(low_ref["path"])
    high_value = json.loads(high_path.read_text(encoding="utf-8"))
    low_value = json.loads(low_path.read_text(encoding="utf-8"))
    low_value["shard_id"] = high_value["shard_id"]
    _write_json(low_path, low_value)
    low["identity_receipts"] = [
        _local_ref(
            low_path,
            root,
            json_assertions=copy.deepcopy(low_ref["json_assertions"]),
        )
    ]
    variant = root / "duplicate_source_shard_id.json"
    _write_json(variant, source_map)
    output = tmp_path / "duplicate-source-shard-canonical"
    partial = tmp_path / (f"duplicate-source-shard-canonical.partial-{_sha256(variant)[:12]}")

    with pytest.raises(ConfigError, match="跨 side 重复"):
        materialize_task8b_phase_f(variant, output)

    assert not output.exists()
    assert not partial.exists()


def test_metadata_dry_run_builds_exact_plan_without_payload_or_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)
    output = tmp_path / "dry-run-canonical"
    report = tmp_path / "metadata-dry-run.json"

    def _payload_copy_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("metadata dry-run invoked the payload copy path")

    monkeypatch.setattr(
        materializer_module,
        "_execute_copy_requests",
        _payload_copy_must_not_run,
    )
    result = metadata_only_dry_run_task8b_phase_f(
        source_map_path,
        output,
        report,
    )

    value = json.loads(report.read_text(encoding="utf-8"))
    partial = tmp_path / (
        f"dry-run-canonical.partial-{result['source_map_sha256'][:12]}"
    )
    assert result["status"] == "pass"
    assert result["worker_count"] == 12
    assert result["task_count"] == 60
    assert result["effect_fields_read"] is False
    assert result["formal_effect_fields_read"] is False
    assert value["gaps"] == []
    assert len(value["gates"]) == 18
    assert len(value["provenance_targets"]) == value["archive_count"] * 3
    assert len(value["file_plan"]) == value["expected_file_count"]
    assert value["large_payload_files_written"] == 0
    assert value["large_payload_bytes_written"] == 0
    assert not output.exists()
    assert not partial.exists()


def test_post_unblind_analysis_supplement_requires_exact_provenance_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    analysis = source_map["analysis"]
    assert isinstance(analysis, dict)
    preunlock_ref = analysis["pre_unlock_manifest"]
    assert isinstance(preunlock_ref, dict)
    fixture_sha = str(preunlock_ref["sha256"])
    monkeypatch.setattr(
        materializer_module,
        "APPROVED_PREUNLOCK_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        materializer_module,
        "APPROVED_POST_UNBLIND_ANALYSIS_FREEZE_SHA256",
        fixture_sha,
    )
    analysis["unlocked_by"] = (
        materializer_module.APPROVED_POST_UNBLIND_UNLOCKED_BY
    )
    accepted = source_map_path.parent / "post_unblind_accepted.json"
    _write_json(accepted, source_map)

    plan = _prepare_materialization(
        accepted,
        tmp_path / "post-unblind-accepted-canonical",
        authoritative_real_profile=False,
    )

    assert plan.analysis["pre_unlock_manifest_sha256"] == fixture_sha

    analysis["unlocked_by"] = "unapproved-post-unblind-marker"
    rejected = source_map_path.parent / "post_unblind_rejected.json"
    _write_json(rejected, source_map)
    with pytest.raises(
        ConfigError,
        match="post-unblind analysis supplement provenance",
    ):
        _prepare_materialization(
            rejected,
            tmp_path / "post-unblind-rejected-canonical",
            authoritative_real_profile=False,
        )


def test_authoritative_materialization_id_rejects_generic_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    source_map["materialization_id"] = (
        materializer_module.AUTHORITATIVE_REAL_MATERIALIZATION_ID
    )
    variant = source_map_path.parent / "authoritative_generic_bypass.json"
    _write_json(variant, source_map)
    monkeypatch.setattr(
        materializer_module,
        "AUTHORITATIVE_REAL_SOURCE_MAP_SHA256",
        _sha256(variant),
    )

    with pytest.raises(
        ConfigError,
        match="禁止 generic materialization",
    ):
        _prepare_materialization(
            variant,
            tmp_path / "forbidden-generic-canonical",
            authoritative_real_profile=False,
        )


def test_authoritative_profile_rejects_nonofficial_id_and_sha(
    tmp_path: Path,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)

    with pytest.raises(
        ConfigError,
        match="必须绑定批准的 v8 ID/SHA",
    ):
        _prepare_materialization(
            source_map_path,
            tmp_path / "forbidden-fake-authoritative-canonical",
            authoritative_real_profile=True,
        )


def test_materializer_git_identity_uses_command_scoped_safe_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    repo_root = Path(materializer_module.__file__).absolute().parents[3]

    def _run(
        args: list[str],
        **_kwargs: object,
    ) -> object:
        calls.append(args)
        tail = args[5:]
        if tail == ["rev-parse", "--show-toplevel"]:
            stdout = str(repo_root)
        elif tail in (
            ["rev-parse", "HEAD"],
            ["rev-parse", "origin/main"],
        ):
            stdout = "a" * 40
        elif tail == ["status", "--porcelain", "--untracked-files=no"]:
            stdout = ""
        elif tail[0:1] == ["rev-parse"]:
            stdout = "b" * 40
        elif tail[0:1] == ["hash-object"]:
            stdout = "b" * 40
        else:
            raise AssertionError(f"unexpected git args: {tail}")
        return materializer_module.subprocess.CompletedProcess(
            args,
            0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(materializer_module.subprocess, "run", _run)

    identity = materializer_module._materializer_git_identity()

    expected_safe = f"safe.directory={repo_root.as_posix()}"
    assert identity["tracked_clean"] is True
    assert calls
    assert all(call[1:3] == ["-c", expected_safe] for call in calls)


def test_active_pid_absence_profiles_require_exact_structural_identity(
    tmp_path: Path,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)
    plan = _prepare_materialization(
        source_map_path,
        tmp_path / "pid-profile-candidate",
        authoritative_real_profile=False,
    )
    workers = {worker.worker_id: worker for worker in plan.workers}
    cases = {
        "P07": {
            "active_process_absent": True,
            "formal_worker_count": 0,
            "host": "H07",
            "observed_pid": 7427,
            "pidfile": (
                "/root/runtime_control/H07/P07/attempt01/"
                "P07_attempt01.pid"
            ),
            "probe_utc": "2026-07-22T16:51:16Z",
            "worker_id": "P07",
        },
        "P08": {
            "active_process_absent": True,
            "observed_pid": 7423,
            "probe_utc": "2026-07-22T16:02:52Z",
        },
        "P09": {
            "active_process_absent": True,
            "host_id": "H09",
            "observed_pid": 7008,
            "pid_file": (
                "/root/runtime_control/H09/P09/attempt01/"
                "P09_attempt01.pid"
            ),
            "probe": "kill -0 returned no such process",
            "probe_utc": "2026-07-22T16:06:52+00:00",
            "schema_version": "task8b-active-pid-absence-v1",
            "worker_id": "P09",
        },
        "P10": {
            "active_process_absent": True,
            "formal_worker_count": 0,
            "host": "H10",
            "observed_pid": 6926,
            "pidfile": (
                "/root/runtime_control/H10/P10/attempt01/"
                "P10_attempt01.pid"
            ),
            "probe_utc": "2026-07-22T16:16:22Z",
            "worker_id": "P10",
        },
        "P11": {
            "active_process_absent": True,
            "formal_worker_count": 0,
            "host": "H11",
            "observed_pid": 6801,
            "probe_utc": "2026-07-22T17:01:49Z",
            "worker_id": "P11",
        },
    }
    for worker_id, value in cases.items():
        profile = str(
            materializer_module.AUTHORITATIVE_ACTIVE_PID_EVIDENCE[
                worker_id
            ]["profile"]
        )
        materializer_module._validate_active_pid_absence_payload(
            workers[worker_id],
            payload=_json_bytes(value),
            profile=profile,
        )
        invalid = {**value, "active_process_absent": False}
        with pytest.raises(ConfigError, match="identity 非法"):
            materializer_module._validate_active_pid_absence_payload(
                workers[worker_id],
                payload=_json_bytes(invalid),
                profile=profile,
            )


def test_metadata_dry_run_rejects_bad_task_receipt_before_report_or_partial(
    tmp_path: Path,
) -> None:
    source_map_path, source_map = _make_source_map(tmp_path)
    workers = source_map["workers"]
    assert isinstance(workers, list)
    p07 = workers[0]
    assert isinstance(p07, dict)
    tasks = p07["tasks"]
    assert isinstance(tasks, list)
    first, second = tasks[:2]
    assert isinstance(first, dict) and isinstance(second, dict)
    first["receipt"], second["receipt"] = second["receipt"], first["receipt"]
    variant = source_map_path.parent / "dry_run_bad_receipt.json"
    _write_json(variant, source_map)
    output = tmp_path / "dry-run-bad-receipt-canonical"
    report = tmp_path / "dry-run-bad-receipt-report.json"
    partial = tmp_path / (
        f"dry-run-bad-receipt-canonical.partial-{_sha256(variant)[:12]}"
    )

    with pytest.raises(ConfigError, match="task receipt"):
        metadata_only_dry_run_task8b_phase_f(variant, output, report)

    assert not report.exists()
    assert not output.exists()
    assert not partial.exists()


def test_metadata_dry_run_fails_closed_when_disk_space_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)
    output = tmp_path / "dry-run-no-space-canonical"
    report = tmp_path / "dry-run-no-space-report.json"

    class _NoSpace:
        free = 0

    monkeypatch.setattr(materializer_module.shutil, "disk_usage", lambda _path: _NoSpace())
    with pytest.raises(ConfigError, match="空间不足"):
        metadata_only_dry_run_task8b_phase_f(source_map_path, output, report)

    assert not report.exists()
    assert not output.exists()


def test_authoritative_real_selection_gate_accepts_only_four_frozen_sources(
    tmp_path: Path,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)
    plan = _prepare_materialization(
        source_map_path,
        tmp_path / "selection-candidate",
        authoritative_real_profile=False,
    )
    formal = plan.archives["formal"]
    archives = {"formal": formal}
    for source_id in (
        "full_H12",
        "full_H02",
        "full_H05",
        "full_H01",
        "extra_01",
        "extra_02",
        "extra_03",
        "extra_04",
        "extra_05",
        "extra_06",
        "extra_07",
        "extra_08",
        "extra_09",
        "extra_10",
        "extra_11",
        "extra_12",
    ):
        archives[source_id] = formal
    by_worker = {worker.worker_id: worker for worker in plan.workers}
    frozen = {
        key: (
            str(value["archive_id"]),
            str(value["source_prefix"]),
            str(value["target_run_dir"]),
        )
        for key, value in (
            materializer_module.AUTHORITATIVE_REAL_SELECTIONS.items()
        )
    }
    for (worker_id, task_id), (archive_id, prefix, target) in frozen.items():
        task = by_worker[worker_id].task_rows[task_id]
        task["source_archive_id"] = archive_id
        task["source_prefix"] = prefix
        task["target_run_dir"] = target

    rows = _validate_authoritative_real_selections(
        archives=archives,
        workers=plan.workers,
    )
    assert len(rows) == 4

    for worker_id, task_id in frozen:
        task = by_worker[worker_id].task_rows[task_id]
        original_prefix = task["source_prefix"]
        task["source_prefix"] = "root/not-an-authoritative-selection"
        with pytest.raises(ConfigError, match="authoritative source selection"):
            _validate_authoritative_real_selections(
                archives=archives,
                workers=plan.workers,
            )
        task["source_prefix"] = original_prefix


def test_authoritative_real_recovery_requires_exact_six_row_prefix_and_append() -> None:
    _payload, rows = _state_bytes(
        ["planned", "validating", "running", "failed", "validating", "running"],
        worker_id="P07",
    )
    append = [{"status": "finalizing"}, {"status": "complete"}]

    _validate_authoritative_recovery_suffix(
        rows,
        original_terminal_state_sha256=rows[3]["row_sha256"],
        state_append=append,
        label="P07",
    )

    with pytest.raises(ConfigError, match="existing suffix"):
        _validate_authoritative_recovery_suffix(
            rows[:-1],
            original_terminal_state_sha256=rows[3]["row_sha256"],
            state_append=append,
            label="P07",
        )
    with pytest.raises(ConfigError, match="authoritative recovery append"):
        _validate_authoritative_recovery_suffix(
            rows,
            original_terminal_state_sha256=rows[3]["row_sha256"],
            state_append=[
                {"status": "validating"},
                {"status": "running"},
                {"status": "finalizing"},
                {"status": "complete"},
            ],
            label="P07",
        )


def test_pre_materialization_receipt_binds_frozen_dry_run_and_file_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map_path, _source_map = _make_source_map(tmp_path)
    output = tmp_path / "receipt-candidate"
    plan = _prepare_materialization(
        source_map_path,
        output,
        authoritative_real_profile=False,
    )
    metadata = _build_metadata_plan(plan, authoritative_real_profile=False)
    authoritative_selections = [
        {
            "worker_id": "P11",
            "task_id": "isolation_async",
            "source_archive_id": "full_H05",
            "source_prefix": (
                "root/pair_11-low-P11-isolation_async-attempt_02/"
                "output/runs/isolation_async"
            ),
            "target_run_dir": "runs/isolation_async",
        }
    ]
    monkeypatch.setattr(
        materializer_module,
        "_validate_authoritative_real_selections",
        lambda **_kwargs: authoritative_selections,
    )
    monkeypatch.setattr(
        materializer_module,
        "_materializer_git_identity",
        lambda: {
            "repo_root": str(Path.cwd()),
            "head_sha": "a" * 40,
            "origin_main_sha": "a" * 40,
            "tracked_clean": True,
            "head_blob_sha": "c" * 40,
            "working_blob_sha": "c" * 40,
        },
    )
    report_path = tmp_path / "authoritative-report.json"
    report = {
        "schema_version": "task8b-phase-f-metadata-dry-run-v1",
        "status": "pass",
        "gaps": [],
        "authoritative_real_profile": True,
        "source_map_path": str(plan.source_path),
        "source_map_sha256": plan.source_sha256,
        "materialization_id": plan.source_map["materialization_id"],
        "candidate_output_root": str(plan.destination),
        "candidate_partial_root": str(plan.partial),
        "analysis_contract_sha256": plan.analysis["analysis_contract_sha256"],
        "pre_unlock_manifest_sha256": plan.analysis["pre_unlock_manifest_sha256"],
        "archive_count": len(plan.archives),
        "worker_count": len(plan.workers),
        "task_count": sum(len(worker.expected_tasks) for worker in plan.workers),
        "pair_count": len({worker.pair_id for worker in plan.workers}),
        "gates": [
            {"gate": name, "status": "pass"} for name in METADATA_DRY_RUN_GATE_NAMES
        ],
        "archive_copy_plan": plan.archive_copy_plan,
        "authoritative_selections": authoritative_selections,
        **metadata,
        "metadata_only": True,
        "large_payload_files_written": 0,
        "large_payload_bytes_written": 0,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "limitations": materializer_module._declared_limitations(),
        "paper_eligibility": (
            materializer_module.CANONICAL_PAPER_ELIGIBILITY
        ),
    }
    _write_json(report_path, report)
    monkeypatch.setattr(
        materializer_module,
        "AUTHORITATIVE_REAL_MATERIALIZATION_ID",
        str(plan.source_map["materialization_id"]),
    )
    monkeypatch.setattr(
        materializer_module,
        "AUTHORITATIVE_REAL_SOURCE_MAP_SHA256",
        _sha256(source_map_path),
    )
    generated_receipt_path = tmp_path / "generated-pre-materialization-receipt.json"
    generated = create_pre_materialization_receipt(
        source_map_path,
        output,
        report_path,
        generated_receipt_path,
    )
    generated_value = json.loads(
        generated_receipt_path.read_text(encoding="utf-8")
    )
    assert generated["status"] == "approved"
    assert generated["receipt_sha256"] == _sha256(generated_receipt_path)
    assert generated_value["dry_run_report_sha256"] == _sha256(report_path)
    assert generated_value["materializer_code_sha"] == "a" * 40
    receipt_path = tmp_path / "pre-materialization-receipt.json"
    receipt = {
        "schema_version": "task8b-phase-f-pre-materialization-receipt-v1",
        "status": "approved",
        "materializer_code_sha": "a" * 40,
        "materializer_file_sha256": _sha256(Path(materializer_module.__file__)),
        "source_map_sha256": plan.source_sha256,
        "analysis_contract_sha256": plan.analysis["analysis_contract_sha256"],
        "pre_unlock_manifest_sha256": plan.analysis["pre_unlock_manifest_sha256"],
        "worker_count": 12,
        "task_count": 60,
        "expected_file_count": metadata["expected_file_count"],
        "expected_bytes": metadata["expected_bytes"],
        "dry_run_report_path": str(report_path),
        "dry_run_report_sha256": _sha256(report_path),
        "authoritative_real_profile": True,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "recovery_code_identity_evidence_status": (
            materializer_module.RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        ),
        "paper_eligibility": (
            materializer_module.CANONICAL_PAPER_ELIGIBILITY
        ),
    }
    _write_json(receipt_path, receipt)

    validated = _validate_pre_materialization_receipt(
        receipt_path,
        plan=plan,
        metadata=metadata,
    )
    assert validated["dry_run_report_sha256"] == _sha256(report_path)
    frozen_root = tmp_path / "frozen-pre-materialization-evidence"
    frozen_root.mkdir()
    frozen_rows = materializer_module._freeze_pre_materialization_evidence(
        receipt_path,
        receipt=validated,
        root=frozen_root,
    )
    assert {row["relative_path"] for row in frozen_rows} == {
        "provenance/pre_materialization/metadata_dry_run_report.json",
        "provenance/pre_materialization/pre_materialization_receipt.json",
    }
    for row in frozen_rows:
        frozen_path = frozen_root.joinpath(
            *Path(str(row["relative_path"])).parts
        )
        assert row["sha256"] == _sha256(frozen_path)

    pristine_report = copy.deepcopy(report)
    report_mutations = {
        "exact gate names": lambda value: value["gates"].__setitem__(
            0,
            {"gate": "gate-00", "status": "pass"},
        ),
        "file plan": lambda value: value.__setitem__("file_plan", []),
        "archive copy plan": lambda value: value.__setitem__("archive_copy_plan", []),
        "provenance targets": lambda value: value.__setitem__("provenance_targets", []),
        "workers": lambda value: value.__setitem__("workers", []),
        "recovery closure": lambda value: value.__setitem__(
            "recovery_closure",
            [{"tampered": True}],
        ),
        "authoritative selections": lambda value: value.__setitem__(
            "authoritative_selections",
            [],
        ),
        "analysis contract": lambda value: value.__setitem__(
            "analysis_contract_sha256",
            "0" * 64,
        ),
        "pre-unlock manifest": lambda value: value.__setitem__(
            "pre_unlock_manifest_sha256",
            "0" * 64,
        ),
        "manifest index": lambda value: value.__setitem__(
            "manifest_index_sha256",
            "0" * 64,
        ),
        "analysis input": lambda value: value.__setitem__(
            "analysis_input_sha256",
            "0" * 64,
        ),
    }
    for label, mutate in report_mutations.items():
        report = copy.deepcopy(pristine_report)
        mutate(report)
        _write_json(report_path, report)
        receipt["dry_run_report_sha256"] = _sha256(report_path)
        _write_json(receipt_path, receipt)
        try:
            _validate_pre_materialization_receipt(
                receipt_path,
                plan=plan,
                metadata=metadata,
            )
        except ConfigError as exc:
            assert "receipt/dry-run report binding" in str(exc)
        else:
            pytest.fail(f"receipt accepted mutated {label}")

    receipt["expected_bytes"] = int(metadata["expected_bytes"]) + 1
    _write_json(report_path, pristine_report)
    receipt["dry_run_report_sha256"] = _sha256(report_path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ConfigError, match="receipt/dry-run report binding"):
        _validate_pre_materialization_receipt(
            receipt_path,
            plan=plan,
            metadata=metadata,
        )

    receipt["expected_bytes"] = metadata["expected_bytes"]
    receipt["materializer_code_sha"] = "b" * 40
    _write_json(receipt_path, receipt)
    with pytest.raises(ConfigError, match="receipt/dry-run report binding"):
        _validate_pre_materialization_receipt(
            receipt_path,
            plan=plan,
            metadata=metadata,
        )
