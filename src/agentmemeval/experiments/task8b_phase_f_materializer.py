"""Append-only TASK8B six-seed Phase F canonical materializer.

The materializer consumes one explicit JSON source map.  It verifies every
source archive, archive manifest, receipt, seal, individual artifact, and SHA
before publishing a canonical Phase F tree.  Formal task payloads are copied as
opaque bytes: metrics, events, hands, reports, and experiment results are never
parsed here.

The supported source-map schema is intentionally narrow and fail-closed.  The
top-level object contains:

* ``archives``: SHA-bound tar archives with an exact member manifest plus a
  blind receipt and seal;
* ``analysis``: SHA-bound six-seed contract and pre-unlock manifest;
* ``workers``: exactly P07-P12 and S07-S12, each with one SHA-bound base
  manifest, five isolated task trees and receipts (including an explicitly
  selected ``__attempt_NN`` run directory when that is the sealed source), a
  real state prefix, two physical shard descriptions with immutable identity
  receipts, and (for P07-P11) the three recovery-adoption files plus four bound
  source artifacts.

Artifact references are either ``{"path", "size", "sha256"}`` or
``{"archive_id", "member", "size", "sha256"}``.  JSON receipts may add
``json_assertions`` containing exact top-level values.  Archive manifests use
either ``files-json-v1`` or ``sha256-size-path-tsv-v1``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime as dt
import errno
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.task8b_analysis import _load_six_seed_analysis_protocol
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.formal_runner import (
    STATE_SCHEMA_VERSION,
    STATE_TRANSITIONS,
    validate_worker_manifest,
    validate_worker_manifest_set,
)

SOURCE_MAP_SCHEMA = "task8b-six-seed-phase-f-source-map-v1"
PROTOCOL_ID = "TASK8B-SIX-SEED-PAIR-SHARD-20260723"
ANALYSIS_CONTRACT_ID = "task8b-phase-f-six-seed-v1"
AUTHORITATIVE_REAL_MATERIALIZATION_ID = (
    "task8b-six-seed-real-a-catalog-20260725-state-tail-p11-controller-v3"
)
AUTHORITATIVE_REAL_SOURCE_MAP_SHA256 = (
    "14c50a5a3088aa22d293f43390911fc2dce169b8ef8cde954e1dad77d9984433"
)
APPROVED_PREUNLOCK_SHA256 = "2da0d149b30b815a747442ebc71661f17f75642aef39b57fd95aad7c2370f300"
APPROVED_POST_UNBLIND_ANALYSIS_FREEZE_SHA256 = (
    "ae9f554fbad03e125fa9ae3a76bccb3ebddebbc67eb5b10d96b4cac9dabd4792"
)
APPROVED_POST_UNBLIND_UNLOCKED_BY = (
    "codex-task8b-phase-f-post-failed-partial-output-"
    "secondary-async-label-supplement-v5"
)
WORKER_MANIFEST_SCHEMA = "task8-worker-manifest-v1"
TASK_RECEIPT_SCHEMA = "task8-worker-task-receipt-v1"
TASK_RESULTS_SCHEMA = "task8-worker-task-results-v1"
COMPLETION_SCHEMA = "task8-worker-completion-v1"
COMPOSITION_SCHEMA = "task8b-six-seed-composition-receipt-v1"
EXPEDITED_STATUS = "frozen/expedited-formal-candidate"
RECOVERED_WORKERS = frozenset(f"P{index:02d}" for index in range(7, 12))
EXPECTED_WORKERS = tuple(f"{role}{index:02d}" for role in ("P", "S") for index in range(7, 13))
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
JSON_ASSERTION_FIELDS = frozenset(
    {
        "active_runner_count",
        "all_member_sha256_verified",
        "archive_members_exact",
        "archive_sha256",
        "effect_fields_read",
        "manifest_sha256",
        "pair_id",
        "physical_slot",
        "role",
        "schema",
        "schema_version",
        "seed",
        "selected_task_ids",
        "shard_id",
        "slot",
        "source_kind",
        "status",
        "structural_hands",
        "task_id",
        "verified_file_count",
        "worker_id",
    }
)
ARCHIVE_EVIDENCE_PROFILES: dict[str, set[str | None]] = {
    "task8b-full-matrix-export-receipt-v1": {"source_manifest_complete"},
    "task8b-full-matrix-export-seal-v1": {"sealed"},
    "task8b-recovery-support-export-receipt-v1": {"source_manifest_complete"},
    "task8b-recovery-support-export-seal-v1": {"sealed"},
    "task8b-local-export-verification-v1": {None},
    "task8b-six-seed-original-evidence-freeze-v1": {None},
    "task8b-paired-shard-cross-host-bundle-v1": {"sealed"},
    "task8b-paired-shard-cross-host-export-v1": {"complete"},
    "task8b-paired-shard-receipt-v1": {"complete"},
}

PRIMARY_TASKS = (
    "isolation_no_memory",
    "isolation_fact",
    "isolation_expr",
    "isolation_sync",
    "isolation_async",
)
SECONDARY_TASKS = (
    "mixed_ecological",
    "expr_online",
    "expr_without",
    "async_online",
    "async_without",
)
PLANNED_HANDS = {
    "isolation_no_memory": 1350,
    "isolation_fact": 1350,
    "isolation_expr": 1350,
    "isolation_sync": 1350,
    "isolation_async": 1350,
    "mixed_ecological": 2700,
    "expr_online": 600,
    "expr_without": 600,
    "async_online": 600,
    "async_without": 600,
}
RECOVERY_TASK_ID = "isolation_no_memory"
RECOVERY_PROTOCOL_AMENDMENT_ID = "TASK8B-VFPR-20260722"
RECOVERY_REASON = "resolved-config-integer-key-canonicalization"
RECOVERY_ADOPTION_SCHEMA = "task8b-same-attempt-task-adoption-v1"
RECOVERY_ATTESTATION_SCHEMA = "task8b-task1-adoption-attestation-v1"
RECOVERY_CANONICAL_AUDIT_SCHEMA = "task8b-canonicalization-equivalence-audit-v1"
FROZEN_SCIENTIFIC_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
AUTHORITATIVE_RECOVERY_VERIFIER_CODE_SHA = (
    "c00da5515af3d874b3499010a02addca2c75fbe4"
)
AUTHORITATIVE_RECOVERY_CONTROLLER_SHA256 = (
    "71f7c7011c79447f32f7e36b5b0379329db67be762f0f140d1de9aae0d46f365"
)
AUTHORITATIVE_RECOVERY_FORMAL_RUNNER_SHA256 = (
    "c4b601ff0de2c27a57ee246efcf91d21f502f27c652d20fd6fa7cfd925a17d5e"
)
AUTHORITATIVE_RECOVERY_VERIFIER_IDENTITY_SOURCE_SHA256 = (
    "6cb3ae7783800debfc3dd50e98e0397a818dca2d71d933f50e6316cae95edeab"
)
RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS = "SELF_ASSERTED_CONSISTENT_UNCLOSED"
CANONICAL_PAPER_ELIGIBILITY = "PENDING_DECLARED_LIMITATION_REVIEW"
RECOVERY_EVIDENCE_FIELDS = (
    "protocol_amendment_sha256",
    "verifier_code_sha",
    "pre_recovery_archive_sha256",
    "pre_recovery_file_manifest_sha256",
    "original_terminal_state_sha256",
    "original_expected_config_sha256",
    "corrected_config_sha256",
    "canonicalization_equivalence_audit_sha256",
    "recovery_certificate_sha256",
    "task1_adoption_attestation_sha256",
)
RECOVERY_TARGETS = {
    "canonicalization_audit": (
        "recovery_adoptions/isolation_no_memory.canonicalization_audit.json"
    ),
    "adoption_attestation": ("recovery_adoptions/isolation_no_memory.adoption_attestation.json"),
    "certificate": "recovery_adoptions/isolation_no_memory.json",
}
RECOVERY_SUPPORT_TARGETS = {
    "protocol_amendment": "protocol_amendment.bin",
    "pre_recovery_archive": "pre_recovery_archive.bin",
    "pre_recovery_manifest": "pre_recovery_manifest.bin",
    "authorization": "authorization.bin",
}
PRE_MATERIALIZATION_REPORT_TARGET = PurePosixPath(
    "provenance/pre_materialization/metadata_dry_run_report.json"
)
PRE_MATERIALIZATION_RECEIPT_TARGET = PurePosixPath(
    "provenance/pre_materialization/pre_materialization_receipt.json"
)
CANONICAL_FILE_MANIFEST_TARGET = PurePosixPath(
    "canonical_file_manifest.tsv"
)
CANONICAL_ROOT_SEAL_TARGET = PurePosixPath("canonical_root_seal.json")
EXCLUSION_LEDGER_TARGET = PurePosixPath("exclusion_ledger.csv")
EXCLUSION_LEDGER_BYTES = (
    b"ledger_entry_id,recorded_at_utc,recorded_before_effect_unblind,seed,"
    b"pod_id,worker_id,run_id,attempt,scope,reason_code,eligible_retry,"
    b"evidence_relative_path,evidence_sha256,retry_attempt,"
    b"authoritative_attempt,excluded_from_confirmatory,"
    b"excluded_from_secondary,decision_rule_version,review_status,reviewer,"
    b"reviewed_at_utc,notes\n"
)
RECOVERY_SOURCE_KEYS = tuple(RECOVERY_TARGETS) + tuple(RECOVERY_SUPPORT_TARGETS)
RECOVERY_STATE_SEQUENCE = ("validating", "running", "finalizing", "complete")
REAL_RECOVERY_EXISTING_SUFFIX = ("validating", "running")
REAL_RECOVERY_APPEND_SUFFIX = ("finalizing", "complete")
AUTHORITATIVE_REAL_SELECTIONS = {
    ("P12", "isolation_sync"): {
        "archive_id": "full_H12",
        "source_prefix": (
            "task8b_six_seed_pair_shard_20260723/staging/paired_shards/"
            "P12/primary-high-expr-sync-rerun-v1/output/runs/"
            "isolation_sync__attempt_02"
        ),
        "target_run_dir": "runs/isolation_sync__attempt_02",
    },
    ("P08", "isolation_async"): {
        "archive_id": "full_H02",
        "source_prefix": (
            "task8b_six_seed_pair_shards/H02/paired_shards/P08/"
            "p08_low_async_attempt03_20260723/output/runs/isolation_async"
        ),
        "target_run_dir": "runs/isolation_async",
    },
    ("P11", "isolation_async"): {
        # The dry-run contract's legacy "H11 Async" gate denotes logical
        # pair_11/P11.  Under the approved paired-shard complement, the sealed
        # physical source for that logical task is full_H05.
        "archive_id": "full_H05",
        "source_prefix": (
            "task8b_six_seed_paired_shards_20260723/paired_shards/P11/"
            "pair_11-low-P11-isolation_async-attempt_02/output/runs/"
            "isolation_async"
        ),
        "target_run_dir": "runs/isolation_async",
    },
    ("S07", "mixed_ecological"): {
        "archive_id": "full_H01",
        "source_prefix": (
            "task8b_six_seed_paired_shards_20260723/paired_shards/S07/"
            "pair_07-low-S07-mixed-e5b732a_mixed_fix_v7_compat/output/"
            "runs/mixed_ecological"
        ),
        "target_run_dir": "runs/mixed_ecological",
    },
}
AUTHORITATIVE_ACTIVE_PID_EVIDENCE = {
    "P07": {
        "archive_id": "recovery_H07",
        "member": (
            "task8b_same_attempt_recovery_v1/"
            "H07_P07_c00da55_20260722T163826Z/"
            "P07_active_pid_absence.json"
        ),
        "sha256": (
            "cd06994330eb2de80cadcddb5804c37377e5bf56ff1f48fac43eb1d0e187d687"
        ),
        "profile": "host-worker-count-pidfile-v0",
    },
    "P08": {
        "archive_id": "recovery_H08",
        "member": (
            "task8b_h08_legacy_composite_recovery_c00da55_v1/"
            "P08/P08.active_pid_absence.json"
        ),
        "sha256": (
            "41a39a33ef8087cef24580f80af82063b5490c7439a61ad249cbbc895ec35fee"
        ),
        "profile": "minimal-legacy-v0",
    },
    "P09": {
        "archive_id": "recovery_H09",
        "member": (
            "task8b_h09_legacy_composite_recovery_c00da55_v1/"
            "P09/active_pid_absence.json"
        ),
        "sha256": (
            "8e6eb0f45d8d6fd3790594f2e18c13f9c11635c93938a4d18275cacedcb7b36f"
        ),
        "profile": "task8b-active-pid-absence-v1",
    },
    "P10": {
        "archive_id": "recovery_H10",
        "member": (
            "task8b_same_attempt_recovery_v1/"
            "H10_P10_c00da55_20260722T155856Z/"
            "P10_active_pid_absence.json"
        ),
        "sha256": (
            "9fc28d3a539787d0870e1d12117b108de8bfc76366b66f8451731556ffaf7564"
        ),
        "profile": "host-worker-count-pidfile-v0",
    },
    "P11": {
        "archive_id": "recovery_H11",
        "member": (
            "task8b_bulk_recovery_c00da55_20260722T1700Z/"
            "P11/P11_active_pid_absence.json"
        ),
        "sha256": (
            "fc74787e1df259f006ba3313f4d8ef61e2da6b32dcdeb1d7eb955c5443d44c87"
        ),
        "profile": "host-worker-count-v0",
    },
}


def _control_members(prefix: str, names: Sequence[str]) -> frozenset[str]:
    return frozenset(f"{prefix}/{name}" for name in names)


def _declared_limitations() -> list[dict[str, str]]:
    return [
        {
            "code": "recovery_code_identity_independent_artifact_unavailable",
            "scope": "P07-P11 recovery controller and verifier identity source",
            "evidence_status": RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS,
            "impact": (
                "canonical bytes remain SHA-closed; paper eligibility requires "
                "explicit review of this declared provenance limitation"
            ),
        }
    ]


AUTHORITATIVE_UNMANIFESTED_REGULAR_MEMBERS = {
    "full_H01": _control_members(
        "task8b_full_matrix_exports/H01_P07_20260725T125212Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H02": _control_members(
        "task8b_full_matrix_exports/H02_P08_20260725T112530Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H03": _control_members(
        "task8b_full_matrix_exports/H03_P09_20260725T102833Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H04": _control_members(
        "task8b_full_matrix_exports/H04_P10_20260725T092121Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H05": _control_members(
        "task8b_full_matrix_exports/H05_P11_20260725T105846Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H06": _control_members(
        "task8b_full_matrix_exports/H06_P12_20260725T111208Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H07": _control_members(
        "task8b_full_matrix_exports/H07_P07_20260725T121033Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H08": _control_members(
        "task8b_full_matrix_exports/H08_P08_20260725T122036Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H09": _control_members(
        "task8b_full_matrix_exports/H09_P09_20260725T122743Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H10": _control_members(
        "task8b_full_matrix_exports/H10_P10_20260725T123406Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H11": _control_members(
        "task8b_full_matrix_exports/H11_P11_20260725T124022Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "full_H12": _control_members(
        "task8b_full_matrix_exports/H12_P12_20260725T124717Z",
        ("export_receipt.json", "server_manifest.tsv", "source_roots.tsv"),
    ),
    "recovery_H07": _control_members(
        "task8b_recovery_support_exports/H07_P07_20260725T132215Z",
        ("server_manifest.tsv", "export_receipt.json"),
    ),
    "recovery_H08": _control_members(
        "task8b_recovery_support_exports/H08_P08_20260725T132431Z",
        ("server_manifest.tsv", "export_receipt.json"),
    ),
    "recovery_H09": _control_members(
        "task8b_recovery_support_exports/H09_P09_20260725T132648Z",
        ("server_manifest.tsv", "export_receipt.json"),
    ),
    "recovery_H10": _control_members(
        "task8b_recovery_support_exports/H10_P10_20260725T132901Z",
        ("server_manifest.tsv", "export_receipt.json"),
    ),
    "recovery_H11": _control_members(
        "task8b_recovery_support_exports/H11_P11_20260725T133429Z",
        ("server_manifest.tsv", "export_receipt.json"),
    ),
}
METADATA_DRY_RUN_GATE_NAMES = (
    "archive_sha_manifest_receipt_seal_binding",
    "worker_and_task_cardinality",
    "unique_task_source_run_attempt_receipt",
    "p12_sync_attempt_02",
    "h02_async_attempt_03",
    "h11_async_attempt_02",
    "s07_mixed_v7_compat",
    "selected_member_manifest_identity_closure",
    "unselected_export_control_exclusion",
    "archive_provenance_copy_plan",
    "destination_collision",
    "shard_completion_receipt_seal_manifest_cross_binding",
    "recovery_state_sha_chain",
    "recovery_authorization_certificate_failure_archive_binding",
    "canonical_output_file_plan",
    "disk_space",
    "append_only_no_overwrite_no_splice",
    "blind_effect_field_access",
)
METADATA_DRY_RUN_PLAN_BINDING_FIELDS = (
    "expected_file_count",
    "expected_bytes",
    "safety_reserve_bytes",
    "copied_target_count",
    "copied_target_bytes",
    "staged_metadata_file_count",
    "staged_metadata_bytes",
    "manifest_index_sha256",
    "analysis_input_sha256",
    "workers",
    "recovery_closure",
    "provenance_targets",
    "file_plan",
)
METADATA_DRY_RUN_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "gaps",
        "authoritative_real_profile",
        "source_map_path",
        "source_map_sha256",
        "materialization_id",
        "candidate_output_root",
        "candidate_partial_root",
        "analysis_contract_sha256",
        "pre_unlock_manifest_sha256",
        "archive_count",
        "worker_count",
        "task_count",
        "pair_count",
        "gates",
        "archive_copy_plan",
        "authoritative_selections",
        "free_bytes",
        "metadata_only",
        "large_payload_files_written",
        "large_payload_bytes_written",
        "effect_fields_read",
        "formal_effect_fields_read",
        "limitations",
        "paper_eligibility",
        *METADATA_DRY_RUN_PLAN_BINDING_FIELDS,
    }
)
RECOVERY_AUTHORIZATION_FIELDS = frozenset(
    {
        "active_pid_absence_evidence_sha256",
        "active_process_absent_confirmed",
        "attempt_root",
        "authorization_id",
        "authorized",
        "baseline_manifest_sha256",
        "controller_sha256",
        "failed_state_row_sha256",
        "formal_runner_sha256",
        "frozen_code_sha",
        "parent_preunlock_sha256",
        "pre_recovery_archive_sha256",
        "protocol_amendment_id",
        "protocol_amendment_sha256",
        "reason",
        "same_attempt_recovery_authorized",
        "schema_version",
        "task1_adoption_only",
        "task_id",
        "verifier_code_sha",
        "verifier_identity_source_sha256",
        "worker_id",
        "worker_manifest_sha256",
    }
)
SECONDARY_NORMALIZATION_FILES = frozenset(
    {
        "certificate.identity_correction_audit.json",
        "certificate.json",
        "ledger.json",
        "shard_seal.json",
    }
)
SECONDARY_NORMALIZATION_SCHEMAS = {
    "certificate.identity_correction_audit.json": ("task8b-paired-identity-correction-audit-v1"),
    "certificate.json": "task8b-paired-completed-recovery-certificate-v1",
    "ledger.json": "task8b-paired-recovery-ledger-entry-v1",
    "shard_seal.json": "task8b-paired-shard-receipt-v1",
}
SHARD_RECEIPT_SCHEMA = "task8b-paired-shard-receipt-v1"
SHARD_TASK_SCHEMA = "task8b-paired-shard-task-v1"
COMPLETION_EVIDENCE_SCHEMAS = frozenset({COMPLETION_SCHEMA, SHARD_RECEIPT_SCHEMA})
FILES_TSV_EXCLUSIONS = frozenset({"state.tsv", "files.tsv", "completion_receipt.json"})
TASK_ROW_FIELDS = frozenset(
    {
        "cache_namespace",
        "identity_audit",
        "memory_mode",
        "recovery_id",
        "run_dir",
        "status",
        "task_id",
    }
)
TASK_RECEIPT_FIELDS = frozenset(
    {
        "config_sha256",
        "files",
        "phase_f_evidence",
        "recovery_adoption",
        "run_dir",
        "schema_version",
        "task_id",
        "task_row",
    }
)
TASK_IDENTITY_FIELDS = frozenset(
    {
        "code_dirty",
        "code_sha",
        "embedding_fingerprint",
        "model_fingerprint",
        "prompt_sha256",
        "resolved_config_sha256",
        "schedule_sha256",
    }
)


@dataclass(frozen=True)
class ManifestEntry:
    member: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArtifactRef:
    size: int
    sha256: str
    path: Path | None = None
    archive_id: str | None = None
    member: str | None = None
    json_assertions: Mapping[str, Any] | None = None

    @property
    def key(self) -> tuple[str, ...]:
        if self.path is not None:
            return ("file", os.path.normcase(str(self.path)))
        assert self.archive_id is not None and self.member is not None
        return ("archive", self.archive_id, self.member)


@dataclass
class ArchiveCatalog:
    source_id: str
    path: Path
    size: int
    sha256: str
    manifest_ref: ArtifactRef
    manifest_format: str
    manifest_has_header: bool
    entries: dict[str, ManifestEntry]
    receipt_ref: ArtifactRef
    seal_ref: ArtifactRef


@dataclass(frozen=True)
class CopyRequest:
    source: ArtifactRef
    targets: tuple[PurePosixPath, ...]


@dataclass
class WorkerPlan:
    raw: dict[str, Any]
    worker_id: str
    role: str
    seed: int
    pair_id: str
    slots: tuple[str, str]
    expected_tasks: tuple[str, ...]
    base_manifest_ref: ArtifactRef
    base_manifest: dict[str, Any]
    state_ref: ArtifactRef
    task_rows: dict[str, dict[str, Any]]
    state_append: list[dict[str, str]]
    recovery_refs: dict[str, ArtifactRef]
    evidence_targets: dict[str, list[tuple[str, ArtifactRef, PurePosixPath]]]
    identity_refs: dict[str, list[ArtifactRef]]


@dataclass
class MaterializationPlan:
    source_path: Path
    source_ref: ArtifactRef
    source_sha256: str
    source_map: dict[str, Any]
    archives: dict[str, ArchiveCatalog]
    analysis: dict[str, Any]
    contract_ref: ArtifactRef
    preunlock_ref: ArtifactRef
    workers: list[WorkerPlan]
    destination: Path
    partial: Path
    requests: list[CopyRequest]
    archive_copy_plan: list[dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materializer_git_identity() -> dict[str, Any]:
    """Bind the running materializer to one pushed, tracked-clean commit."""

    module_path = Path(__file__).absolute()
    repo_root = module_path.parents[3]

    def _git(*args: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.as_posix()}",
                "-C",
                str(repo_root),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ConfigError(
                "materializer git identity 无法读取："
                f"{str(completed.stderr).strip()}"
            )
        return completed.stdout

    top_level = str(_git("rev-parse", "--show-toplevel")).strip()
    head_sha = str(_git("rev-parse", "HEAD")).strip()
    origin_main_sha = str(_git("rev-parse", "origin/main")).strip()
    tracked_status = str(
        _git("status", "--porcelain", "--untracked-files=no")
    ).strip()
    relative_module = module_path.relative_to(repo_root).as_posix()
    head_blob_sha = str(_git("rev-parse", f"HEAD:{relative_module}")).strip()
    working_blob_sha = str(
        _git(
            "hash-object",
            f"--path={relative_module}",
            relative_module,
        )
    ).strip()
    if (
        os.path.normcase(str(Path(top_level).absolute()))
        != os.path.normcase(str(repo_root))
        or SHA1_RE.fullmatch(head_sha) is None
        or SHA1_RE.fullmatch(origin_main_sha) is None
        or SHA1_RE.fullmatch(head_blob_sha) is None
        or SHA1_RE.fullmatch(working_blob_sha) is None
    ):
        raise ConfigError("materializer git repository/commit identity 非法")
    return {
        "repo_root": str(repo_root),
        "head_sha": head_sha,
        "origin_main_sha": origin_main_sha,
        "tracked_clean": tracked_status == "",
        "head_blob_sha": head_blob_sha,
        "working_blob_sha": working_blob_sha,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} 不是有效 UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{label} JSON 顶层必须为对象")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _write_json_new(path: Path, value: Any) -> None:
    _write_new(path, _json_bytes(value))


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ConfigError(
            f"{label} 字段不闭合：expected={sorted(expected)} observed={sorted(value)}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise ConfigError(f"{label} SHA-256 非法")
    return text


def _require_sha1(value: Any, *, label: str) -> str:
    text = str(value)
    if SHA1_RE.fullmatch(text) is None:
        raise ConfigError(f"{label} full code SHA 非法")
    return text


def _safe_identifier(value: Any, *, label: str) -> str:
    text = str(value)
    if SAFE_ID_RE.fullmatch(text) is None:
        raise ConfigError(f"{label} 标识符非法")
    return text


def _task_run_dir(value: Any, *, task_id: str, label: str) -> str:
    """Validate one explicit, receipt-preserving task run selection."""

    path = _safe_relative(value, label=label)
    if len(path.parts) != 2 or path.parts[0] != "runs":
        raise ConfigError(f"{label} 必须为 runs/<task> 或 runs/<task>__attempt_NN")
    name = path.parts[1]
    if (
        name != task_id
        and re.fullmatch(
            rf"{re.escape(task_id)}__attempt_[0-9]{{2}}",
            name,
        )
        is None
    ):
        raise ConfigError(f"{label} 与 task_id 不匹配")
    return path.as_posix()


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or text != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            any(character in part for character in '<>:"|?*')
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
            for part in path.parts
        )
    ):
        raise ConfigError(f"{label} 相对路径非法：{text}")
    return path


def _resolve_regular_file(base: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    path = raw if raw.is_absolute() else base / raw
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"{label} 必须是非 symlink 普通文件：{path}")
    return path


def _validate_file_identity(
    path: Path,
    *,
    size: int,
    sha256: str,
    label: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"{label} 源文件缺失或为 symlink")
    if path.stat().st_size != size or _sha256_file(path) != sha256:
        raise ConfigError(f"{label} size/SHA-256 不匹配")


def _parse_artifact_ref(
    value: Any,
    *,
    base: Path,
    archives: Mapping[str, ArchiveCatalog],
    label: str,
    allow_assertions: bool = True,
) -> ArtifactRef:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} artifact ref 必须为对象")
    allowed = {"path", "archive_id", "member", "size", "sha256"}
    if allow_assertions:
        allowed.add("json_assertions")
    if not set(value).issubset(allowed):
        raise ConfigError(f"{label} artifact ref 含未知字段")
    size = value.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ConfigError(f"{label} size 非法")
    sha256 = _require_sha256(value.get("sha256"), label=label)
    assertions = value.get("json_assertions")
    if assertions is not None and not isinstance(assertions, dict):
        raise ConfigError(f"{label} json_assertions 必须为对象")
    if isinstance(assertions, dict) and not set(assertions).issubset(JSON_ASSERTION_FIELDS):
        raise ConfigError(f"{label} json_assertions 含非盲态/未知字段")
    has_path = "path" in value
    has_archive = "archive_id" in value or "member" in value
    if has_path == has_archive:
        raise ConfigError(f"{label} 必须二选一指定 path 或 archive member")
    if has_path:
        path = _resolve_regular_file(base, value["path"], label=label)
        ref = ArtifactRef(
            path=path,
            size=size,
            sha256=sha256,
            json_assertions=copy.deepcopy(assertions),
        )
        _validate_file_identity(path, size=size, sha256=sha256, label=label)
        return ref
    archive_id = _safe_identifier(value.get("archive_id"), label=f"{label}.archive_id")
    member = _safe_relative(value.get("member"), label=f"{label}.member").as_posix()
    catalog = archives.get(archive_id)
    if catalog is None:
        raise ConfigError(f"{label} 引用未知 archive_id：{archive_id}")
    entry = catalog.entries.get(member)
    if entry is None:
        raise ConfigError(f"{label} archive manifest 不含 member：{member}")
    if entry.size != size or entry.sha256 != sha256:
        raise ConfigError(f"{label} 与 archive manifest size/SHA 不一致")
    return ArtifactRef(
        archive_id=archive_id,
        member=member,
        size=size,
        sha256=sha256,
        json_assertions=copy.deepcopy(assertions),
    )


def _parse_manifest_tsv(
    payload: bytes,
    *,
    has_header: bool,
    label: str,
) -> dict[str, ManifestEntry]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{label} manifest 不是 UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if has_header:
        if not rows:
            raise ConfigError(f"{label} manifest 为空")
        normalized = [item.strip().lower() for item in rows.pop(0)]
        allowed_headers = (
            ["sha256", "bytes", "relative_path"],
            ["sha256", "size", "relative_path"],
            ["sha256", "bytes", "filename"],
            ["sha256", "size", "filename"],
        )
        if normalized not in allowed_headers:
            raise ConfigError(f"{label} manifest header 非法")
    entries: dict[str, ManifestEntry] = {}
    for index, row in enumerate(rows, start=2 if has_header else 1):
        if len(row) != 3:
            raise ConfigError(f"{label} manifest 第 {index} 行列数非法")
        sha256 = _require_sha256(row[0], label=f"{label} row {index}")
        try:
            size = int(row[1])
        except ValueError as exc:
            raise ConfigError(f"{label} manifest 第 {index} 行 size 非法") from exc
        member = _safe_relative(row[2], label=f"{label} manifest row {index}").as_posix()
        if size < 0 or member in entries:
            raise ConfigError(f"{label} manifest 第 {index} 行重复或 size 非法")
        entries[member] = ManifestEntry(member, size, sha256)
    if not entries:
        raise ConfigError(f"{label} manifest 没有文件")
    return entries


def _parse_manifest_json(payload: bytes, *, label: str) -> dict[str, ManifestEntry]:
    value = _read_json_bytes(payload, label=label)
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise ConfigError(f"{label} files-json-v1 缺 files")
    entries: dict[str, ManifestEntry] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigError(f"{label} files[{index}] 非对象")
        member = _safe_relative(
            row.get("relative_path"), label=f"{label} files[{index}]"
        ).as_posix()
        size = row.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ConfigError(f"{label} files[{index}] size 非法")
        sha256 = _require_sha256(row.get("sha256"), label=f"{label} files[{index}]")
        if member in entries:
            raise ConfigError(f"{label} files member 重复：{member}")
        entries[member] = ManifestEntry(member, size, sha256)
    return entries


def _parse_flexible_manifest_tsv(
    payload: bytes,
    *,
    label: str,
) -> dict[str, ManifestEntry]:
    """Parse the two immutable recovery-manifest column layouts in the fleet."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{label} manifest 不是 UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    fields = [str(field or "").strip().lower() for field in (reader.fieldnames or [])]
    if fields not in (
        ["relative_path", "size", "sha256"],
        ["sha256", "bytes", "relative_path"],
        ["sha256", "size", "relative_path"],
    ):
        raise ConfigError(f"{label} manifest header 非法")
    entries: dict[str, ManifestEntry] = {}
    for index, row in enumerate(reader, start=2):
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        member = _safe_relative(
            normalized.get("relative_path"),
            label=f"{label} row {index}",
        ).as_posix()
        size_text = normalized.get("size", normalized.get("bytes"))
        try:
            size = int(str(size_text))
        except ValueError as exc:
            raise ConfigError(f"{label} row {index} size 非法") from exc
        sha256 = _require_sha256(
            normalized.get("sha256"),
            label=f"{label} row {index}",
        )
        if size < 0 or member in entries:
            raise ConfigError(f"{label} row {index} 重复或 size 非法")
        entries[member] = ManifestEntry(member, size, sha256)
    if not entries:
        raise ConfigError(f"{label} manifest 没有文件")
    return entries


def _load_archives(
    value: Any,
    *,
    base: Path,
) -> dict[str, ArchiveCatalog]:
    if not isinstance(value, dict) or not value:
        raise ConfigError("source map archives 必须为非空对象")
    catalogs: dict[str, ArchiveCatalog] = {}
    pending_evidence: dict[str, tuple[Any, Any]] = {}
    for raw_id, raw in value.items():
        source_id = _safe_identifier(raw_id, label="archive_id")
        if source_id in catalogs or not isinstance(raw, dict):
            raise ConfigError(f"archive {source_id} 重复或非法")
        _require_exact_keys(
            raw,
            {"path", "size", "sha256", "manifest", "receipt", "seal"},
            label=f"archive {source_id}",
        )
        path = _resolve_regular_file(base, raw["path"], label=f"archive {source_id}.path")
        size = raw["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ConfigError(f"archive {source_id}.size 非法")
        sha256 = _require_sha256(raw["sha256"], label=f"archive {source_id}")
        _validate_file_identity(path, size=size, sha256=sha256, label=f"archive {source_id}")
        manifest = raw["manifest"]
        if not isinstance(manifest, dict):
            raise ConfigError(f"archive {source_id}.manifest 非对象")
        _require_exact_keys(
            manifest,
            {"path", "size", "sha256", "format", "has_header"},
            label=f"archive {source_id}.manifest",
        )
        manifest_ref = _parse_artifact_ref(
            {
                "path": manifest["path"],
                "size": manifest["size"],
                "sha256": manifest["sha256"],
            },
            base=base,
            archives={},
            label=f"archive {source_id}.manifest",
            allow_assertions=False,
        )
        manifest_format = str(manifest["format"])
        has_header = manifest["has_header"]
        if not isinstance(has_header, bool):
            raise ConfigError(f"archive {source_id}.manifest.has_header 非 bool")
        assert manifest_ref.path is not None
        payload = manifest_ref.path.read_bytes()
        if manifest_format == "sha256-size-path-tsv-v1":
            entries = _parse_manifest_tsv(
                payload,
                has_header=has_header,
                label=f"archive {source_id}",
            )
        elif manifest_format == "files-json-v1":
            if has_header:
                raise ConfigError(f"archive {source_id} JSON manifest 不得有 header")
            entries = _parse_manifest_json(payload, label=f"archive {source_id}")
        else:
            raise ConfigError(f"archive {source_id}.manifest.format 未支持")
        catalogs[source_id] = ArchiveCatalog(
            source_id=source_id,
            path=path,
            size=size,
            sha256=sha256,
            manifest_ref=manifest_ref,
            manifest_format=manifest_format,
            manifest_has_header=has_header,
            entries=entries,
            receipt_ref=ArtifactRef(0, "0" * 64),
            seal_ref=ArtifactRef(0, "0" * 64),
        )
        pending_evidence[source_id] = (raw["receipt"], raw["seal"])
    for source_id, (receipt_raw, seal_raw) in pending_evidence.items():
        catalog = catalogs[source_id]
        receipt = _parse_artifact_ref(
            receipt_raw,
            base=base,
            archives=catalogs,
            label=f"archive {source_id}.receipt",
        )
        seal = _parse_artifact_ref(
            seal_raw,
            base=base,
            archives=catalogs,
            label=f"archive {source_id}.seal",
        )
        for kind, ref in (("receipt", receipt), ("seal", seal)):
            assertions = ref.json_assertions
            if (
                not isinstance(assertions, dict)
                or assertions.get("effect_fields_read") is not False
            ):
                raise ConfigError(f"archive {source_id}.{kind} 必须断言 effect_fields_read=false")
            if not {"status", "schema", "schema_version"}.intersection(assertions):
                raise ConfigError(f"archive {source_id}.{kind} 必须断言 status/schema identity")
        catalog.receipt_ref = receipt
        catalog.seal_ref = seal
    return catalogs


def _read_ref_bytes(
    ref: ArtifactRef,
    *,
    archives: Mapping[str, ArchiveCatalog],
    label: str,
) -> bytes:
    if ref.path is not None:
        _validate_file_identity(ref.path, size=ref.size, sha256=ref.sha256, label=label)
        return ref.path.read_bytes()
    assert ref.archive_id is not None and ref.member is not None
    catalog = archives[ref.archive_id]
    with tarfile.open(catalog.path, mode="r:*") as archive:
        matches = [member for member in archive.getmembers() if member.name == ref.member]
        if len(matches) != 1 or not matches[0].isreg():
            raise ConfigError(f"{label} archive member 缺失、重复或非普通文件")
        extracted = archive.extractfile(matches[0])
        if extracted is None:
            raise ConfigError(f"{label} archive member 无法读取")
        payload = extracted.read()
    if len(payload) != ref.size or hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ConfigError(f"{label} archive member size/SHA 不匹配")
    return payload


def _validate_json_assertions(payload: bytes, ref: ArtifactRef, *, label: str) -> None:
    if ref.json_assertions is None:
        return
    value = _read_json_bytes(payload, label=label)
    for key, expected in ref.json_assertions.items():
        if value.get(key) != expected:
            raise ConfigError(f"{label} assertion 失败：{key}")


def _validate_archive_evidence_bindings(
    catalog: ArchiveCatalog,
    *,
    receipt_payload: bytes,
    seal_payload: bytes,
) -> None:
    receipt = _read_json_bytes(receipt_payload, label=f"archive {catalog.source_id} receipt")
    seal = _read_json_bytes(seal_payload, label=f"archive {catalog.source_id} seal")
    evidence_rows = (
        ("receipt", catalog.receipt_ref, receipt),
        ("seal", catalog.seal_ref, seal),
    )
    if catalog.receipt_ref.key == catalog.seal_ref.key:
        raise ConfigError(f"archive {catalog.source_id} receipt/seal 不得为同一 artifact")
    for kind, _ref, value in evidence_rows:
        profile = str(value.get("schema_version") or value.get("schema") or "")
        status = value.get("status")
        if (
            profile not in ARCHIVE_EVIDENCE_PROFILES
            or status not in ARCHIVE_EVIDENCE_PROFILES[profile]
        ):
            raise ConfigError(f"archive {catalog.source_id} {kind} evidence profile/status 非法")
        if value.get("effect_fields_read") is not False:
            raise ConfigError(f"archive {catalog.source_id} {kind} effect_fields_read 非 false")
        for key in ("archive_sha256",):
            if key in value and value[key] != catalog.sha256:
                raise ConfigError(f"archive {catalog.source_id} {kind}.{key} cross-binding 失败")
        for key in ("manifest_sha256", "server_manifest_sha256"):
            if key in value and value[key] != catalog.manifest_ref.sha256:
                raise ConfigError(f"archive {catalog.source_id} {kind}.{key} cross-binding 失败")
    for key in ("export_receipt_sha256", "receipt_sha256"):
        if key in seal and seal[key] != catalog.receipt_ref.sha256:
            raise ConfigError(f"archive {catalog.source_id} seal.{key} receipt binding 失败")
    outer_bindings = [
        value
        for _kind, ref, value in evidence_rows
        if ref.path is not None
        and value.get("archive_sha256") == catalog.sha256
        and any(
            value.get(key) == catalog.manifest_ref.sha256
            for key in ("manifest_sha256", "server_manifest_sha256")
        )
    ]
    if not outer_bindings:
        raise ConfigError(f"archive {catalog.source_id} 缺 local outer archive/manifest binding")


def _expected_pair(worker_id: str) -> tuple[int, str, tuple[str, str]]:
    try:
        index = int(worker_id[1:])
    except (ValueError, IndexError) as exc:
        raise ConfigError(f"worker_id 非法：{worker_id}") from exc
    if worker_id not in EXPECTED_WORKERS:
        raise ConfigError(f"worker_id 不在 six-seed contract：{worker_id}")
    return (
        2026090100 + index,
        f"pair_{index:02d}",
        (f"H{index - 6:02d}", f"H{index:02d}"),
    )


def _expected_sides(
    role: str, slots: tuple[str, str]
) -> tuple[tuple[str, str, tuple[str, ...], int, str], ...]:
    low_slot, high_slot = slots
    if role == "primary":
        return (
            (
                "high",
                high_slot,
                PRIMARY_TASKS[:4],
                5400,
                "primary_isolation_partition",
            ),
            (
                "low",
                low_slot,
                PRIMARY_TASKS[4:],
                1350,
                "primary_isolation_partition",
            ),
        )
    return (
        (
            "low",
            low_slot,
            SECONDARY_TASKS[:1],
            2700,
            "secondary_mode_partition",
        ),
        (
            "high",
            high_slot,
            SECONDARY_TASKS[1:],
            2400,
            "secondary_mode_partition",
        ),
    )


def _validate_base_manifest(
    manifest: dict[str, Any],
    *,
    worker_id: str,
    role: str,
    seed: int,
    expected_tasks: tuple[str, ...],
) -> None:
    try:
        validate_worker_manifest(copy.deepcopy(manifest))
    except ConfigError as exc:
        raise ConfigError(f"{worker_id} base manifest formal validation 失败") from exc
    if (
        manifest.get("schema_version") != WORKER_MANIFEST_SCHEMA
        or manifest.get("worker_id") != worker_id
        or manifest.get("role") != role
        or int(manifest.get("seed_bundle", -1)) != seed
        or manifest.get("protocol_status") != EXPEDITED_STATUS
        or not str(manifest.get("pod_id", "")).strip()
        or "phase_f_shard_lineage" in manifest
    ):
        raise ConfigError(f"{worker_id} base manifest identity/schema 非法")
    common = manifest.get("common_identity")
    if not isinstance(common, dict):
        raise ConfigError(f"{worker_id} base manifest 缺 common_identity")
    _require_sha1(common.get("code_sha"), label=f"{worker_id} common_identity.code_sha")
    tasks = manifest.get("task_configs")
    if not isinstance(tasks, list) or len(tasks) != len(expected_tasks):
        raise ConfigError(f"{worker_id} base manifest task_configs 数量非法")
    task_ids = [str(row.get("task_id", "")) for row in tasks if isinstance(row, dict)]
    if task_ids != list(expected_tasks):
        raise ConfigError(f"{worker_id} base manifest task 顺序/集合非法")
    for task in tasks:
        assert isinstance(task, dict)
        task_id = str(task["task_id"])
        if int(task.get("planned_hands", -1)) != PLANNED_HANDS[task_id]:
            raise ConfigError(f"{worker_id}/{task_id} planned_hands 非法")


def _parse_analysis(
    value: Any,
    *,
    base: Path,
    archives: Mapping[str, ArchiveCatalog],
) -> tuple[dict[str, Any], ArtifactRef, ArtifactRef]:
    if not isinstance(value, dict):
        raise ConfigError("source map analysis 必须为对象")
    _require_exact_keys(
        value,
        {
            "analysis_contract",
            "pre_unlock_manifest",
            "unlocked_at_utc",
            "unlocked_by",
        },
        label="analysis",
    )
    contract_ref = _parse_artifact_ref(
        value["analysis_contract"],
        base=base,
        archives=archives,
        label="analysis.analysis_contract",
    )
    preunlock_ref = _parse_artifact_ref(
        value["pre_unlock_manifest"],
        base=base,
        archives=archives,
        label="analysis.pre_unlock_manifest",
    )
    contract = _read_json_bytes(
        _read_ref_bytes(contract_ref, archives=archives, label="analysis.analysis_contract"),
        label="analysis.analysis_contract",
    )
    authority = _load_six_seed_analysis_protocol()
    if (
        contract.get("analysis_contract_id") != ANALYSIS_CONTRACT_ID
        or contract_ref.sha256 != authority["contract_sha256"]
    ):
        raise ConfigError("analysis contract 非权威 six-seed Phase F freeze")
    preunlock = _read_json_bytes(
        _read_ref_bytes(preunlock_ref, archives=archives, label="analysis.pre_unlock_manifest"),
        label="analysis.pre_unlock_manifest",
    )
    unlocked_by = str(value["unlocked_by"]).strip()
    if not unlocked_by:
        raise ConfigError("analysis.unlocked_by 为空")
    approved_freezes = {
        APPROVED_PREUNLOCK_SHA256,
        APPROVED_POST_UNBLIND_ANALYSIS_FREEZE_SHA256,
    }
    if preunlock_ref.sha256 not in approved_freezes:
        raise ConfigError("analysis manifest 非批准的 frozen analyzer identity")
    if (
        preunlock_ref.sha256 == APPROVED_POST_UNBLIND_ANALYSIS_FREEZE_SHA256
        and unlocked_by != APPROVED_POST_UNBLIND_UNLOCKED_BY
    ):
        raise ConfigError("post-unblind analysis supplement provenance 非法")
    schema = preunlock.get("schema_version")
    if schema == "task8b-phase-f-pre-unlock-v1":
        contract_id = preunlock.get("analysis_contract_id")
        status = preunlock.get("manifest_status")
        expected_status = "FROZEN_BEFORE_FORMAL_RESULT"
    elif schema == "task8b-phase-f-recovery-supplementary-freeze-v2":
        contract_id = preunlock.get("parent_contract_id")
        status = preunlock.get("status")
        expected_status = "CORRECTED_RECOVERY_LINEAGE_BINDING_BEFORE_FORMAL_RESULT"
    else:
        raise ConfigError("pre-unlock manifest schema 非法或 non-authoritative")
    dependency = preunlock.get("dependency_lock")
    if (
        contract_id != ANALYSIS_CONTRACT_ID
        or status != expected_status
        or preunlock.get("analysis_contract_sha256") != contract_ref.sha256
        or preunlock.get("formal_result_loaded") is not False
        or preunlock.get("analysis_code_dirty") is not False
        or not isinstance(dependency, dict)
        or dependency.get("verified_real_lock") is not True
    ):
        raise ConfigError("pre-unlock manifest 未满足 six-seed 揭盲前冻结门禁")
    analysis_code_sha = _require_sha1(
        preunlock.get("analysis_code_sha"), label="pre-unlock.analysis_code_sha"
    )
    dependency_sha = _require_sha256(dependency.get("sha256"), label="pre-unlock.dependency_lock")
    frozen_at = _parse_utc_timestamp(
        preunlock.get("frozen_at_utc"), label="pre-unlock.frozen_at_utc"
    )
    unlocked_at = _parse_utc_timestamp(value["unlocked_at_utc"], label="analysis.unlocked_at_utc")
    if unlocked_at < frozen_at:
        raise ConfigError("analysis unlocked_at_utc 早于 pre-unlock freeze")
    return (
        {
            "analysis_code_sha": analysis_code_sha,
            "analysis_code_dirty": False,
            "analysis_frozen_at_utc": str(preunlock["frozen_at_utc"]),
            "dependency_lock_sha256": dependency_sha,
            "unlocked_at_utc": str(value["unlocked_at_utc"]),
            "unlocked_by": unlocked_by,
            "pre_unlock_manifest_sha256": preunlock_ref.sha256,
            "analysis_contract_sha256": contract_ref.sha256,
        },
        contract_ref,
        preunlock_ref,
    )


def _parse_utc_timestamp(value: Any, *, label: str) -> dt.datetime:
    text = str(value)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(f"{label} timestamp 非法") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ConfigError(f"{label} 必须为 aware UTC timestamp")
    return parsed


def _parse_workers(
    value: Any,
    *,
    base: Path,
    archives: Mapping[str, ArchiveCatalog],
) -> list[WorkerPlan]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_WORKERS):
        raise ConfigError("source map workers 必须精确包含 12 个 six-seed workers")
    observed = [str(row.get("worker_id", "")) if isinstance(row, dict) else "" for row in value]
    if observed != list(EXPECTED_WORKERS):
        raise ConfigError("source map workers 必须按 P07-P12,S07-S12 精确排序")
    plans: list[WorkerPlan] = []
    for raw in value:
        assert isinstance(raw, dict)
        worker_id = str(raw["worker_id"])
        role = "primary" if worker_id.startswith("P") else "secondary"
        expected_tasks = PRIMARY_TASKS if role == "primary" else SECONDARY_TASKS
        expected_seed, pair_id, slots = _expected_pair(worker_id)
        expected_keys = {
            "worker_id",
            "role",
            "seed",
            "pair_id",
            "physical_slots",
            "base_manifest",
            "state_prefix",
            "state_append",
            "tasks",
            "shards",
        }
        if worker_id in RECOVERED_WORKERS:
            expected_keys.add("recovery_adoptions")
        _require_exact_keys(raw, expected_keys, label=worker_id)
        if (
            raw["role"] != role
            or int(raw["seed"]) != expected_seed
            or raw["pair_id"] != pair_id
            or raw["physical_slots"] != list(slots)
        ):
            raise ConfigError(f"{worker_id} seed/pair/role/slot identity 非法")
        base_ref = _parse_artifact_ref(
            raw["base_manifest"],
            base=base,
            archives=archives,
            label=f"{worker_id}.base_manifest",
        )
        base_manifest = _read_json_bytes(
            _read_ref_bytes(base_ref, archives=archives, label=f"{worker_id}.base_manifest"),
            label=f"{worker_id}.base_manifest",
        )
        _validate_base_manifest(
            base_manifest,
            worker_id=worker_id,
            role=role,
            seed=expected_seed,
            expected_tasks=expected_tasks,
        )
        state_ref = _parse_artifact_ref(
            raw["state_prefix"],
            base=base,
            archives=archives,
            label=f"{worker_id}.state_prefix",
        )
        state_append = _parse_state_append(raw["state_append"], worker_id=worker_id)
        tasks = raw["tasks"]
        if not isinstance(tasks, list) or len(tasks) != 5:
            raise ConfigError(f"{worker_id}.tasks 必须精确包含五个任务")
        task_rows: dict[str, dict[str, Any]] = {}
        for task in tasks:
            if not isinstance(task, dict):
                raise ConfigError(f"{worker_id}.tasks 行非对象")
            _require_exact_keys(
                task,
                {"task_id", "source_archive_id", "source_prefix", "target_run_dir", "receipt"},
                label=f"{worker_id}.task",
            )
            task_id = str(task["task_id"])
            if task_id in task_rows:
                raise ConfigError(f"{worker_id} task_id 重复：{task_id}")
            source_id = _safe_identifier(
                task["source_archive_id"],
                label=f"{worker_id}/{task_id}.source_archive_id",
            )
            catalog = archives.get(source_id)
            if catalog is None:
                raise ConfigError(f"{worker_id}/{task_id} 引用未知 archive")
            prefix = _safe_relative(
                task["source_prefix"],
                label=f"{worker_id}/{task_id}.source_prefix",
            ).as_posix()
            target = _task_run_dir(
                task["target_run_dir"],
                task_id=task_id,
                label=f"{worker_id}/{task_id}.target_run_dir",
            )
            prefix_parts = PurePosixPath(prefix).parts
            target_parts = PurePosixPath(target).parts
            if prefix_parts[-len(target_parts) :] != target_parts:
                raise ConfigError(f"{worker_id}/{task_id} source_prefix 必须以 target_run_dir 结尾")
            entries = [
                entry
                for member, entry in catalog.entries.items()
                if member.startswith(prefix + "/")
            ]
            if not entries:
                raise ConfigError(f"{worker_id}/{task_id} source task root 为空")
            receipt_ref = _parse_artifact_ref(
                task["receipt"],
                base=base,
                archives=archives,
                label=f"{worker_id}/{task_id}.receipt",
            )
            task_rows[task_id] = {
                "source_archive_id": source_id,
                "source_prefix": prefix,
                "target_run_dir": target,
                "receipt_ref": receipt_ref,
                "entries": entries,
            }
        if tuple(task_rows) != expected_tasks:
            raise ConfigError(f"{worker_id}.tasks 顺序/集合非法")
        for left_index, left in enumerate(task_rows.values()):
            left_archive = str(left["source_archive_id"])
            left_prefix = PurePosixPath(str(left["source_prefix"]))
            for right in list(task_rows.values())[left_index + 1 :]:
                if str(right["source_archive_id"]) != left_archive:
                    continue
                right_prefix = PurePosixPath(str(right["source_prefix"]))
                if (
                    left_prefix == right_prefix
                    or left_prefix in right_prefix.parents
                    or right_prefix in left_prefix.parents
                ):
                    raise ConfigError(f"{worker_id} task source prefixes 重复或互为祖先")
        recovery_refs: dict[str, ArtifactRef] = {}
        if worker_id in RECOVERED_WORKERS:
            recovery = raw["recovery_adoptions"]
            if not isinstance(recovery, dict):
                raise ConfigError(f"{worker_id}.recovery_adoptions 非对象")
            _require_exact_keys(
                recovery,
                set(RECOVERY_SOURCE_KEYS),
                label=f"{worker_id}.recovery_adoptions",
            )
            recovery_refs = {
                key: _parse_artifact_ref(
                    recovery[key],
                    base=base,
                    archives=archives,
                    label=f"{worker_id}.recovery_adoptions.{key}",
                )
                for key in RECOVERY_SOURCE_KEYS
            }
        shards = raw["shards"]
        expected_side_rows = _expected_sides(role, slots)
        if not isinstance(shards, list) or len(shards) != 2:
            raise ConfigError(f"{worker_id}.shards 必须精确包含两侧")
        if [str(row.get("side", "")) for row in shards if isinstance(row, dict)] != [
            row[0] for row in expected_side_rows
        ]:
            raise ConfigError(f"{worker_id}.shards side 顺序非法")
        evidence_targets: dict[str, list[tuple[str, ArtifactRef, PurePosixPath]]] = {}
        identity_refs: dict[str, list[ArtifactRef]] = {}
        for shard, expected_side in zip(shards, expected_side_rows, strict=True):
            if not isinstance(shard, dict):
                raise ConfigError(f"{worker_id}.shards 行非对象")
            _require_exact_keys(
                shard,
                {
                    "side",
                    "shard_id",
                    "source_archive_ids",
                    "shard_manifest",
                    "completion_receipt",
                    "identity_receipts",
                    "execution_receipts",
                },
                label=f"{worker_id}.shard",
            )
            side, _slot, side_tasks, _hands, _shard_role = expected_side
            source_ids = shard["source_archive_ids"]
            if not isinstance(source_ids, list) or not source_ids:
                raise ConfigError(f"{worker_id}/{side} source_archive_ids 为空")
            expected_source_ids = list(
                dict.fromkeys(task_rows[task]["source_archive_id"] for task in side_tasks)
            )
            if source_ids != expected_source_ids:
                raise ConfigError(f"{worker_id}/{side} source archives 与 task roots 不闭合")
            shard_id = _safe_identifier(shard["shard_id"], label=f"{worker_id}/{side}.shard_id")
            if shard_id != f"{worker_id}-{side}":
                raise ConfigError(f"{worker_id}/{side}.shard_id 必须为确定性 canonical id")
            shard_manifest = _parse_artifact_ref(
                shard["shard_manifest"],
                base=base,
                archives=archives,
                label=f"{worker_id}/{side}.shard_manifest",
                allow_assertions=False,
            )
            completion = _parse_artifact_ref(
                shard["completion_receipt"],
                base=base,
                archives=archives,
                label=f"{worker_id}/{side}.completion_receipt",
            )
            if (
                not isinstance(completion.json_assertions, dict)
                or completion.json_assertions.get("status") != "complete"
            ):
                raise ConfigError(f"{worker_id}/{side} completion receipt 必须断言 status=complete")
            identity_rows = shard["identity_receipts"]
            if not isinstance(identity_rows, list) or not identity_rows:
                raise ConfigError(f"{worker_id}/{side}.identity_receipts 为空")
            parsed_identity_refs = [
                _parse_artifact_ref(
                    item,
                    base=base,
                    archives=archives,
                    label=f"{worker_id}/{side}.identity_receipts[{index}]",
                )
                for index, item in enumerate(identity_rows, start=1)
            ]
            if len({ref.key for ref in parsed_identity_refs}) != len(parsed_identity_refs):
                raise ConfigError(f"{worker_id}/{side}.identity_receipts 重复")
            identity_refs[side] = parsed_identity_refs
            extras = shard["execution_receipts"]
            if not isinstance(extras, list):
                raise ConfigError(f"{worker_id}/{side}.execution_receipts 非数组")
            extra_refs = [
                _parse_artifact_ref(
                    item,
                    base=base,
                    archives=archives,
                    label=f"{worker_id}/{side}.execution_receipts[{index}]",
                )
                for index, item in enumerate(extras, start=1)
            ]
            extra_names = [
                (
                    PurePosixPath(ref.member).name
                    if ref.member is not None
                    else ref.path.name
                    if ref.path is not None
                    else ""
                )
                for ref in extra_refs
            ]
            if len(extra_names) != len(set(extra_names)):
                raise ConfigError(f"{worker_id}/{side} execution evidence basename 重复")
            if (
                role == "secondary"
                and side == "low"
                and set(extra_names) != SECONDARY_NORMALIZATION_FILES
            ):
                raise ConfigError(f"{worker_id}/{side} secondary normalization evidence 不完整")
            if not (role == "secondary" and side == "low") and extra_refs:
                raise ConfigError(f"{worker_id}/{side} 非 normalization side 不得含 generic extras")
            evidence: list[tuple[str, ArtifactRef]] = [
                ("shard-manifest", shard_manifest),
                ("completion", completion),
            ]
            evidence.extend(
                (f"identity-{index:02d}", ref)
                for index, ref in enumerate(parsed_identity_refs, start=1)
            )
            for source_id in source_ids:
                evidence.extend(
                    (
                        (f"{source_id}-receipt", archives[source_id].receipt_ref),
                        (f"{source_id}-seal", archives[source_id].seal_ref),
                    )
                )
            evidence.extend(
                (
                    f"extra-{index:02d}",
                    ref,
                )
                for index, ref in enumerate(extra_refs, start=1)
            )
            targets: list[tuple[str, ArtifactRef, PurePosixPath]] = []
            for index, (kind, ref) in enumerate(evidence, start=1):
                if kind.startswith("extra-"):
                    filename = (
                        PurePosixPath(ref.member).name
                        if ref.member is not None
                        else ref.path.name
                        if ref.path is not None
                        else ""
                    )
                    _safe_relative(
                        filename,
                        label=f"{worker_id}/{side} execution evidence basename",
                    )
                else:
                    suffix = Path(ref.member or ref.path or "artifact").suffix
                    filename = f"{index:02d}-{kind}{suffix or '.bin'}"
                target = PurePosixPath(
                    f"manifests/composition/execution/{worker_id}/{side}/{filename}"
                )
                targets.append((kind, ref, target))
            evidence_targets[side] = targets
        plans.append(
            WorkerPlan(
                raw=raw,
                worker_id=worker_id,
                role=role,
                seed=expected_seed,
                pair_id=pair_id,
                slots=slots,
                expected_tasks=expected_tasks,
                base_manifest_ref=base_ref,
                base_manifest=base_manifest,
                state_ref=state_ref,
                task_rows=task_rows,
                state_append=state_append,
                recovery_refs=recovery_refs,
                evidence_targets=evidence_targets,
                identity_refs=identity_refs,
            )
        )
    try:
        validate_worker_manifest_set(
            [copy.deepcopy(plan.base_manifest) for plan in plans],
            expected_seed_count=6,
        )
    except ConfigError as exc:
        raise ConfigError(f"six-seed base manifest set 非法：{exc}") from exc
    return plans


def _parse_state_append(value: Any, *, worker_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ConfigError(f"{worker_id}.state_append 必须为数组")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ConfigError(f"{worker_id}.state_append[{index}] 非对象")
        _require_exact_keys(
            row,
            {"created_at_utc", "status", "detail"},
            label=f"{worker_id}.state_append[{index}]",
        )
        status = str(row["status"])
        if status not in STATE_TRANSITIONS:
            raise ConfigError(f"{worker_id}.state_append[{index}] status 非法")
        _parse_utc_timestamp(
            row["created_at_utc"],
            label=f"{worker_id}.state_append[{index}].created_at_utc",
        )
        rows.append(
            {
                "created_at_utc": str(row["created_at_utc"]),
                "status": status,
                "detail": str(row["detail"]),
            }
        )
    return rows


def _add_copy(
    requests: dict[tuple[str, ...], tuple[ArtifactRef, list[PurePosixPath]]],
    targets: dict[PurePosixPath, tuple[str, ...]],
    ref: ArtifactRef,
    target: PurePosixPath,
) -> None:
    _safe_relative(target.as_posix(), label="materialization target")
    existing_key = targets.get(target)
    if existing_key is not None and existing_key != ref.key:
        raise ConfigError(f"materialization target collision：{target}")
    targets[target] = ref.key
    if ref.key in requests:
        existing_ref, existing_targets = requests[ref.key]
        if existing_ref.size != ref.size or existing_ref.sha256 != ref.sha256:
            raise ConfigError("同一 source artifact 声明了冲突 identity")
        existing_assertions = dict(existing_ref.json_assertions or {})
        for key, expected in (ref.json_assertions or {}).items():
            if key in existing_assertions and existing_assertions[key] != expected:
                raise ConfigError("同一 source artifact 声明了冲突 JSON assertions")
            existing_assertions[key] = copy.deepcopy(expected)
        if existing_assertions != dict(existing_ref.json_assertions or {}):
            existing_ref = ArtifactRef(
                path=existing_ref.path,
                archive_id=existing_ref.archive_id,
                member=existing_ref.member,
                size=existing_ref.size,
                sha256=existing_ref.sha256,
                json_assertions=existing_assertions,
            )
            requests[ref.key] = (existing_ref, existing_targets)
        if target not in existing_targets:
            existing_targets.append(target)
    else:
        requests[ref.key] = (ref, [target])


def _task_entry_ref(
    catalog: ArchiveCatalog,
    entry: ManifestEntry,
) -> ArtifactRef:
    return ArtifactRef(
        archive_id=catalog.source_id,
        member=entry.member,
        size=entry.size,
        sha256=entry.sha256,
    )


def _build_copy_requests(
    *,
    source_map_ref: ArtifactRef,
    contract_ref: ArtifactRef,
    preunlock_ref: ArtifactRef,
    archives: Mapping[str, ArchiveCatalog],
    workers: Sequence[WorkerPlan],
) -> list[CopyRequest]:
    requests: dict[tuple[str, ...], tuple[ArtifactRef, list[PurePosixPath]]] = {}
    targets: dict[PurePosixPath, tuple[str, ...]] = {}
    _add_copy(
        requests,
        targets,
        source_map_ref,
        PurePosixPath("provenance/source_map.json"),
    )
    _add_copy(
        requests,
        targets,
        contract_ref,
        PurePosixPath("provenance/analysis_contract.json"),
    )
    _add_copy(
        requests,
        targets,
        preunlock_ref,
        PurePosixPath("provenance/pre_unlock_manifest.json"),
    )
    for source_id, catalog in archives.items():
        suffix = catalog.manifest_ref.path.suffix if catalog.manifest_ref.path else ".manifest"
        _add_copy(
            requests,
            targets,
            catalog.manifest_ref,
            PurePosixPath(f"provenance/archive_manifests/{source_id}{suffix}"),
        )
        receipt_suffix = (
            catalog.receipt_ref.path.suffix
            if catalog.receipt_ref.path is not None
            else PurePosixPath(catalog.receipt_ref.member or "").suffix
        )
        _add_copy(
            requests,
            targets,
            catalog.receipt_ref,
            PurePosixPath(
                f"provenance/archive_receipts/{source_id}{receipt_suffix or '.json'}"
            ),
        )
        seal_suffix = (
            catalog.seal_ref.path.suffix
            if catalog.seal_ref.path is not None
            else PurePosixPath(catalog.seal_ref.member or "").suffix
        )
        _add_copy(
            requests,
            targets,
            catalog.seal_ref,
            PurePosixPath(f"provenance/archive_seals/{source_id}{seal_suffix or '.json'}"),
        )
    for worker in workers:
        attempt = PurePosixPath(f"snapshots/{worker.worker_id}/{worker.seed}/attempt_01")
        _add_copy(
            requests,
            targets,
            worker.base_manifest_ref,
            PurePosixPath(f"provenance/base_manifests/{worker.worker_id}.json"),
        )
        _add_copy(
            requests,
            targets,
            worker.state_ref,
            attempt / "state.tsv",
        )
        for task_id, task in worker.task_rows.items():
            catalog = archives[str(task["source_archive_id"])]
            prefix = str(task["source_prefix"])
            target_root = PurePosixPath(str(task["target_run_dir"]))
            for entry in task["entries"]:
                suffix = PurePosixPath(entry.member).relative_to(PurePosixPath(prefix))
                _add_copy(
                    requests,
                    targets,
                    _task_entry_ref(catalog, entry),
                    attempt / target_root / suffix,
                )
            _add_copy(
                requests,
                targets,
                task["receipt_ref"],
                attempt / "task_receipts" / f"{task_id}.json",
            )
        for key, ref in worker.recovery_refs.items():
            target = (
                attempt / PurePosixPath(RECOVERY_TARGETS[key])
                if key in RECOVERY_TARGETS
                else PurePosixPath(
                    f"provenance/recovery/{worker.worker_id}/{RECOVERY_SUPPORT_TARGETS[key]}"
                )
            )
            _add_copy(
                requests,
                targets,
                ref,
                target,
            )
        for evidence in worker.evidence_targets.values():
            for _kind, ref, target in evidence:
                _add_copy(requests, targets, ref, target)
    return [
        CopyRequest(ref, tuple(sorted(item_targets, key=lambda item: item.as_posix())))
        for ref, item_targets in requests.values()
    ]


def _validate_archive_copy_plan(
    *,
    requests: Sequence[CopyRequest],
    archives: Mapping[str, ArchiveCatalog],
    authoritative_real_profile: bool,
) -> list[dict[str, Any]]:
    requested_by_archive: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        ref = request.source
        if ref.archive_id is not None:
            assert ref.member is not None
            requested_by_archive[ref.archive_id].add(ref.member)
    results = []
    for source_id, catalog in archives.items():
        requested = requested_by_archive[source_id]
        observed: set[str] = set()
        observed_manifest: set[str] = set()
        unmanifested_regular: list[str] = []
        with tarfile.open(catalog.path, mode="r:*") as archive:
            for member in archive:
                raw_name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                name = _safe_relative(
                    raw_name,
                    label=f"archive {source_id} tar member",
                ).as_posix()
                if name in observed:
                    raise ConfigError(f"archive {source_id} tar member 重复：{name}")
                observed.add(name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ConfigError(
                        f"archive {source_id} 含 symlink/hardlink/special member：{name}"
                    )
                manifest_entry = catalog.entries.get(name)
                if manifest_entry is None:
                    if name in requested:
                        raise ConfigError(
                            f"archive {source_id} requested regular member "
                            f"未被 manifest 记录：{name}"
                        )
                    unmanifested_regular.append(name)
                    continue
                if member.size != manifest_entry.size:
                    raise ConfigError(f"archive {source_id} member size 非法：{name}")
                observed_manifest.add(name)
        if observed_manifest != set(catalog.entries) or not requested.issubset(
            observed_manifest
        ):
            missing = sorted(set(catalog.entries) - observed_manifest)
            uncopied = sorted(requested - observed_manifest)
            raise ConfigError(
                f"archive {source_id} metadata exact closure 失败："
                f"missing={missing[:3]} uncopied={uncopied[:3]}"
            )
        if authoritative_real_profile:
            allowed_unmanifested = AUTHORITATIVE_UNMANIFESTED_REGULAR_MEMBERS.get(
                source_id
            )
            if (
                allowed_unmanifested is None
                or set(unmanifested_regular) != allowed_unmanifested
            ):
                raise ConfigError(
                    f"archive {source_id} authoritative export-control "
                    "allowlist 不闭合"
                )
        receipt_payload = _read_ref_bytes(
            catalog.receipt_ref,
            archives=archives,
            label=f"archive {source_id}.receipt preflight",
        )
        seal_payload = _read_ref_bytes(
            catalog.seal_ref,
            archives=archives,
            label=f"archive {source_id}.seal preflight",
        )
        _validate_json_assertions(
            receipt_payload,
            catalog.receipt_ref,
            label=f"archive {source_id}.receipt preflight",
        )
        _validate_json_assertions(
            seal_payload,
            catalog.seal_ref,
            label=f"archive {source_id}.seal preflight",
        )
        _validate_archive_evidence_bindings(
            catalog,
            receipt_payload=receipt_payload,
            seal_payload=seal_payload,
        )
        results.append(
            {
                "archive_id": source_id,
                "manifest_member_count": len(catalog.entries),
                "requested_member_count": len(requested),
                "unmanifested_unrequested_regular_count": len(unmanifested_regular),
                "unmanifested_unrequested_regular": sorted(unmanifested_regular),
                "special_member_count": 0,
                "archive_sha256": catalog.sha256,
                "manifest_sha256": catalog.manifest_ref.sha256,
                "receipt_sha256": catalog.receipt_ref.sha256,
                "seal_sha256": catalog.seal_ref.sha256,
                "archive_receipt_seal_binding_valid": True,
            }
        )
    return results


def _stream_to_targets(
    source: BinaryIO,
    *,
    root: Path,
    targets: Sequence[PurePosixPath],
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    handles: list[BinaryIO] = []
    digest = hashlib.sha256()
    size = 0
    try:
        for relative in targets:
            destination = root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handles.append(destination.open("xb"))
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            for handle in handles:
                handle.write(chunk)
    finally:
        for handle in handles:
            handle.close()
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise ConfigError(f"{label} stream size/SHA-256 不匹配")
    for relative in targets:
        destination = root.joinpath(*relative.parts)
        if (
            destination.stat().st_size != expected_size
            or _sha256_file(destination) != expected_sha256
        ):
            raise ConfigError(f"{label} target post-copy SHA-256 不匹配")


def _execute_copy_requests(
    *,
    requests: Sequence[CopyRequest],
    archives: Mapping[str, ArchiveCatalog],
    root: Path,
) -> None:
    requests_by_key = {request.source.key: request for request in requests}
    local = [request for request in requests if request.source.path is not None]
    archive_requests: dict[str, dict[str, CopyRequest]] = defaultdict(dict)
    for request in requests:
        ref = request.source
        if ref.archive_id is not None:
            assert ref.member is not None
            archive_requests[ref.archive_id][ref.member] = request
    for request in local:
        ref = request.source
        assert ref.path is not None
        _validate_file_identity(
            ref.path,
            size=ref.size,
            sha256=ref.sha256,
            label=f"local source {ref.path}",
        )
        with ref.path.open("rb") as handle:
            _stream_to_targets(
                handle,
                root=root,
                targets=request.targets,
                expected_size=ref.size,
                expected_sha256=ref.sha256,
                label=f"local source {ref.path}",
            )
        _validate_file_identity(
            ref.path,
            size=ref.size,
            sha256=ref.sha256,
            label=f"local source post-copy {ref.path}",
        )
    for source_id, catalog in archives.items():
        requested = archive_requests.get(source_id, {})
        observed: set[str] = set()
        observed_regular: set[str] = set()
        copied: set[str] = set()
        with tarfile.open(catalog.path, mode="r:*") as archive:
            for member in archive:
                raw_name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                name = _safe_relative(raw_name, label=f"archive {source_id} tar member").as_posix()
                if name in observed:
                    raise ConfigError(f"archive {source_id} tar member 重复：{name}")
                observed.add(name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ConfigError(
                        f"archive {source_id} 含 symlink/hardlink/special member：{name}"
                    )
                manifest_entry = catalog.entries.get(name)
                if manifest_entry is None:
                    # Export tooling may append control-plane files (for
                    # example, source_roots.tsv) after freezing the payload
                    # manifest.  The outer archive SHA still binds those
                    # bytes.  They are ineligible as materialization sources:
                    # every requested member was already required to exist in
                    # catalog.entries while parsing the source map.
                    if name in requested:
                        raise ConfigError(
                            f"archive {source_id} requested regular member "
                            f"未被 manifest 记录：{name}"
                        )
                    continue
                observed_regular.add(name)
                if member.size != manifest_entry.size:
                    raise ConfigError(f"archive {source_id} member size 非法：{name}")
                request = requested.get(name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ConfigError(f"archive {source_id} member 无法读取：{name}")
                with extracted:
                    _stream_to_targets(
                        extracted,
                        root=root,
                        targets=request.targets if request is not None else (),
                        expected_size=manifest_entry.size,
                        expected_sha256=manifest_entry.sha256,
                        label=f"archive {source_id}:{name}",
                    )
                if request is not None:
                    copied.add(name)
        if observed_regular != set(catalog.entries) or copied != set(requested):
            missing = sorted(set(catalog.entries) - observed_regular)
            extra = sorted(observed_regular - set(catalog.entries))
            uncopied = sorted(set(requested) - copied)
            raise ConfigError(
                f"archive {source_id} exact closure 失败："
                f"missing={missing[:3]} extra={extra[:3]} uncopied={uncopied[:3]}"
            )
        _validate_file_identity(
            catalog.path,
            size=catalog.size,
            sha256=catalog.sha256,
            label=f"archive {source_id} post-copy",
        )
    for request in requests:
        ref = request.source
        if ref.json_assertions is None:
            continue
        target = root.joinpath(*request.targets[0].parts)
        _validate_json_assertions(
            target.read_bytes(), ref, label=f"copied evidence {request.targets[0]}"
        )
    for catalog in archives.values():
        receipt_request = requests_by_key[catalog.receipt_ref.key]
        seal_request = requests_by_key[catalog.seal_ref.key]
        receipt_path = root.joinpath(*receipt_request.targets[0].parts)
        seal_path = root.joinpath(*seal_request.targets[0].parts)
        _validate_archive_evidence_bindings(
            catalog,
            receipt_payload=receipt_path.read_bytes(),
            seal_payload=seal_path.read_bytes(),
        )


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]


def _planned_task_manifest(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefix = PurePosixPath(str(task["source_prefix"]))
    rows = []
    for entry in task["entries"]:
        relative = PurePosixPath(entry.member).relative_to(prefix).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "size": entry.size,
                "sha256": entry.sha256,
            }
        )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _validate_task_receipt_value(
    worker: WorkerPlan,
    *,
    task_id: str,
    receipt: Mapping[str, Any],
    actual_files: Sequence[Mapping[str, Any]],
    base_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    task = worker.task_rows[task_id]
    configs = {
        str(row["task_id"]): row
        for row in base_manifest["task_configs"]
        if isinstance(row, dict)
    }
    run_dir = str(task["target_run_dir"])
    task_row = receipt.get("task_row")
    config = configs[task_id]
    expected_config_sha = config.get("config_sha256")
    task_row_keys_valid = isinstance(task_row, dict) and set(task_row).issubset(TASK_ROW_FIELDS)
    task_identity = task_row.get("identity_audit") if isinstance(task_row, dict) else None
    task_identity_valid = task_identity is None or (
        isinstance(task_identity, dict) and set(task_identity) == TASK_IDENTITY_FIELDS
    )
    if (
        receipt.get("schema_version") != TASK_RECEIPT_SCHEMA
        or not set(receipt).issubset(TASK_RECEIPT_FIELDS)
        or receipt.get("task_id") != task_id
        or SHA256_RE.fullmatch(str(expected_config_sha or "")) is None
        or receipt.get("config_sha256") != expected_config_sha
        or receipt.get("run_dir") != run_dir
        or not isinstance(task_row, dict)
        or not task_row_keys_valid
        or not task_identity_valid
        or task_row.get("task_id") != task_id
        or task_row.get("run_dir") != run_dir
        or task_row.get("status") != "complete"
        or receipt.get("files") != list(actual_files)
    ):
        raise ConfigError(f"{worker.worker_id}/{task_id} task receipt 不闭合")
    return {
        "task_id": task_id,
        "run_dir": run_dir,
        "status": "complete",
    }


def _validate_task_receipts(worker: WorkerPlan, attempt: Path) -> list[dict[str, Any]]:
    base_manifest = _read_json_bytes(
        (
            attempt.parents[3] / "provenance" / "base_manifests" / f"{worker.worker_id}.json"
        ).read_bytes(),
        label=f"{worker.worker_id} base manifest",
    )
    results: list[dict[str, Any]] = []
    for task_id in worker.expected_tasks:
        task = worker.task_rows[task_id]
        receipt_path = attempt / "task_receipts" / f"{task_id}.json"
        receipt = _read_json_bytes(
            receipt_path.read_bytes(), label=f"{worker.worker_id}/{task_id} receipt"
        )
        run_dir = str(task["target_run_dir"])
        task_root = attempt.joinpath(*PurePosixPath(run_dir).parts)
        results.append(
            _validate_task_receipt_value(
                worker,
                task_id=task_id,
                receipt=receipt,
                actual_files=_directory_manifest(task_root),
                base_manifest=base_manifest,
            )
        )
    return results


def _parse_state_bytes(payload: bytes, *, label: str) -> list[dict[str, str]]:
    if not payload.endswith(b"\n"):
        raise ConfigError(f"{label} state prefix 必须以 LF 结束")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{label} state prefix 非 UTF-8") from exc
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    expected_fields = [
        "schema_version",
        "created_at_utc",
        "status",
        "detail",
        "previous_sha256",
        "row_sha256",
    ]
    if not rows or list(rows[0]) != expected_fields:
        raise ConfigError(f"{label} state header/rows 非法")
    previous = "GENESIS"
    previous_status: str | None = None
    for index, row in enumerate(rows):
        status = str(row.get("status", ""))
        if status not in STATE_TRANSITIONS:
            raise ConfigError(f"{label} state status 非法")
        if status not in STATE_TRANSITIONS[previous_status]:
            raise ConfigError(f"{label} state transition 非法")
        _parse_utc_timestamp(row.get("created_at_utc"), label=f"{label} state row {index}")
        expected_sha = sha256_json(
            {
                "schema_version": row.get("schema_version"),
                "created_at_utc": row.get("created_at_utc"),
                "status": status,
                "detail": row.get("detail"),
                "previous_sha256": row.get("previous_sha256"),
            }
        )
        if (
            row.get("schema_version") != STATE_SCHEMA_VERSION
            or row.get("previous_sha256") != previous
            or row.get("row_sha256") != expected_sha
        ):
            raise ConfigError(f"{label} state hash chain 非法")
        previous = expected_sha
        previous_status = status
    return rows


def _parse_state(path: Path, *, label: str) -> list[dict[str, str]]:
    return _parse_state_bytes(path.read_bytes(), label=label)


def _build_state_rows(
    prefix_rows: Sequence[dict[str, str]],
    append_rows: Sequence[dict[str, str]],
    *,
    label: str,
) -> list[dict[str, str]]:
    previous_status = str(prefix_rows[-1]["status"])
    previous_hash = str(prefix_rows[-1]["row_sha256"])
    built: list[dict[str, str]] = []
    for index, row in enumerate(append_rows):
        status = row["status"]
        if status not in STATE_TRANSITIONS[previous_status]:
            raise ConfigError(
                f"{label} state_append[{index}] transition 非法：{previous_status!r}->{status!r}"
            )
        body = {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_at_utc": row["created_at_utc"],
            "status": status,
            "detail": row["detail"],
            "previous_sha256": previous_hash,
        }
        row_sha = sha256_json(body)
        built.append({**body, "row_sha256": row_sha})
        previous_status = status
        previous_hash = row_sha
    if previous_status != "complete":
        raise ConfigError(f"{label} canonical state 最终状态必须为 complete")
    if built and built[-1]["status"] != "complete":
        raise ConfigError(f"{label} state_append 最后一行必须为 complete")
    return built


def _validate_recovery_state_anchor(
    source_state_rows: Sequence[dict[str, str]],
    *,
    original_terminal_state_sha256: str,
    label: str,
) -> int:
    anchor_indices = [
        index
        for index, row in enumerate(source_state_rows)
        if row["row_sha256"] == original_terminal_state_sha256
    ]
    if len(anchor_indices) != 1:
        raise ConfigError(f"{label} original failed state anchor 不唯一或缺失")
    anchor_index = anchor_indices[0]
    if source_state_rows[anchor_index]["status"] != "failed":
        raise ConfigError(f"{label} original terminal state anchor 非 failed")
    recovery_suffix = tuple(row["status"] for row in source_state_rows[anchor_index + 1 :])
    if recovery_suffix != RECOVERY_STATE_SEQUENCE[: len(recovery_suffix)]:
        raise ConfigError(f"{label} recovery state suffix 非法")
    return anchor_index


def _state_row_bytes(rows: Iterable[dict[str, str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
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
    return handle.getvalue().encode("utf-8")


def _append_bytes(path: Path, payload: bytes) -> None:
    if not payload:
        return
    with path.open("ab") as handle:
        handle.write(payload)


def _validate_recovery(
    worker: WorkerPlan,
    attempt: Path | None,
    *,
    source_state_rows: Sequence[dict[str, str]],
    archives: Mapping[str, ArchiveCatalog] | None = None,
    metadata_only: bool = False,
) -> None:
    if metadata_only:
        if archives is None or attempt is not None:
            raise ConfigError(f"{worker.worker_id} metadata recovery 参数非法")
        observed = set(RECOVERY_TARGETS.values()) if worker.recovery_refs else set()
    else:
        if attempt is None:
            raise ConfigError(f"{worker.worker_id} recovery attempt 缺失")
        recovery_root = attempt / "recovery_adoptions"
        observed = (
            {
                path.relative_to(attempt).as_posix()
                for path in recovery_root.rglob("*")
                if path.is_file()
            }
            if recovery_root.is_dir()
            else set()
        )
    expected = set(RECOVERY_TARGETS.values()) if worker.recovery_refs else set()
    if observed != expected:
        raise ConfigError(f"{worker.worker_id} recovery artifact 集合不闭合")
    if RECOVERY_TASK_ID not in worker.expected_tasks:
        if worker.recovery_refs or observed:
            raise ConfigError(f"{worker.worker_id} secondary worker 不得含 recovery evidence")
        return
    if metadata_only:
        assert archives is not None
        receipt_payload = _read_ref_bytes(
            worker.task_rows[RECOVERY_TASK_ID]["receipt_ref"],
            archives=archives,
            label=f"{worker.worker_id} recovery task receipt",
        )
    else:
        assert attempt is not None
        receipt_payload = (
            attempt / "task_receipts" / f"{RECOVERY_TASK_ID}.json"
        ).read_bytes()
    receipt = _read_json_bytes(
        receipt_payload, label=f"{worker.worker_id} recovery task receipt"
    )
    adoption = receipt.get("recovery_adoption")
    evidence = receipt.get("phase_f_evidence")
    if not worker.recovery_refs:
        if adoption is not None or evidence is not None or observed:
            raise ConfigError(f"{worker.worker_id} 出现 orphan recovery evidence")
        return
    if (
        receipt.get("schema_version") != TASK_RECEIPT_SCHEMA
        or not isinstance(adoption, dict)
        or not isinstance(evidence, dict)
        or set(evidence) != set(RECOVERY_EVIDENCE_FIELDS)
    ):
        raise ConfigError(f"{worker.worker_id} recovery receipt schema/evidence 非法")
    for field in RECOVERY_EVIDENCE_FIELDS:
        pattern = SHA1_RE if field == "verifier_code_sha" else SHA256_RE
        if pattern.fullmatch(str(evidence.get(field, ""))) is None:
            raise ConfigError(f"{worker.worker_id} recovery evidence hash 非法：{field}")
    if metadata_only:
        base_manifest = copy.deepcopy(worker.base_manifest)
    else:
        assert attempt is not None
        base_manifest = _read_json_bytes(
            (
                attempt.parents[3]
                / "provenance"
                / "base_manifests"
                / f"{worker.worker_id}.json"
            ).read_bytes(),
            label=f"{worker.worker_id} recovery base manifest",
        )
    experiment_code_sha = str(base_manifest.get("common_identity", {}).get("code_sha", ""))
    if (
        experiment_code_sha != FROZEN_SCIENTIFIC_CODE_SHA
        or evidence["verifier_code_sha"] == experiment_code_sha
    ):
        raise ConfigError(f"{worker.worker_id} recovery scientific/verifier code SHA 非法")
    if metadata_only:
        assert archives is not None
        recovery_payloads = {
            key: _read_ref_bytes(
                worker.recovery_refs[key],
                archives=archives,
                label=f"{worker.worker_id} recovery {key}",
            )
            for key in RECOVERY_TARGETS
        }
    else:
        assert attempt is not None
        recovery_payloads = {
            key: (attempt / target).read_bytes()
            for key, target in RECOVERY_TARGETS.items()
        }
    if (
        hashlib.sha256(recovery_payloads["canonicalization_audit"]).hexdigest()
        != evidence["canonicalization_equivalence_audit_sha256"]
        or hashlib.sha256(recovery_payloads["adoption_attestation"]).hexdigest()
        != evidence["task1_adoption_attestation_sha256"]
        or hashlib.sha256(recovery_payloads["certificate"]).hexdigest()
        != evidence["recovery_certificate_sha256"]
    ):
        raise ConfigError(f"{worker.worker_id} recovery files SHA 不匹配")
    canonical = _read_json_bytes(
        recovery_payloads["canonicalization_audit"],
        label=f"{worker.worker_id} canonicalization audit",
    )
    attestation = _read_json_bytes(
        recovery_payloads["adoption_attestation"],
        label=f"{worker.worker_id} adoption attestation",
    )
    certificate = _read_json_bytes(
        recovery_payloads["certificate"],
        label=f"{worker.worker_id} recovery certificate",
    )
    phase_f_base = {
        field: evidence[field]
        for field in RECOVERY_EVIDENCE_FIELDS
        if field not in {"task1_adoption_attestation_sha256", "recovery_certificate_sha256"}
    }
    phase_f_without_certificate = {
        **phase_f_base,
        "task1_adoption_attestation_sha256": evidence["task1_adoption_attestation_sha256"],
    }
    if (
        canonical.get("schema_version") != RECOVERY_CANONICAL_AUDIT_SCHEMA
        or canonical.get("worker_id") != worker.worker_id
        or canonical.get("task_id") != RECOVERY_TASK_ID
        or attestation.get("schema_version") != RECOVERY_ATTESTATION_SCHEMA
        or attestation.get("worker_id") != worker.worker_id
        or attestation.get("task_id") != RECOVERY_TASK_ID
        or attestation.get("protocol_amendment_id") != RECOVERY_PROTOCOL_AMENDMENT_ID
        or attestation.get("scientific_execution_code_sha") != FROZEN_SCIENTIFIC_CODE_SHA
        or attestation.get("phase_f_evidence") != phase_f_base
        or certificate.get("schema_version") != RECOVERY_ADOPTION_SCHEMA
        or certificate.get("status") != "authorized-task1-adoption"
        or certificate.get("reason") != RECOVERY_REASON
        or certificate.get("worker_id") != worker.worker_id
        or certificate.get("task_id") != RECOVERY_TASK_ID
        or certificate.get("protocol_amendment_id") != RECOVERY_PROTOCOL_AMENDMENT_ID
        or certificate.get("frozen_code_sha") != FROZEN_SCIENTIFIC_CODE_SHA
        or certificate.get("phase_f_evidence") != phase_f_without_certificate
        or certificate.get("effect_fields_read") is not False
        or certificate.get("scientific_outcome_fields_accessed") is not False
        or attestation.get("scientific_outcome_fields_accessed") is not False
        or certificate.get("task1_rerun_performed") is not False
        or attestation.get("task1_rerun_performed") is not False
        or certificate.get("raw_artifact_bytes_modified") is not False
        or attestation.get("raw_artifact_bytes_modified") is not False
        or certificate.get("planned_hands") != 1350
        or certificate.get("actual_hands") != 1350
        or attestation.get("planned_hands") != 1350
        or attestation.get("actual_hands") != 1350
    ):
        raise ConfigError(f"{worker.worker_id} recovery certificate/attestation identity 非法")
    if (
        adoption.get("schema_version") != RECOVERY_ADOPTION_SCHEMA
        or not str(adoption.get("authorization_id", "")).strip()
        or SHA256_RE.fullmatch(str(adoption.get("authorization_sha256", ""))) is None
        or adoption.get("authorization_id") != certificate.get("authorization_id")
        or adoption.get("authorization_sha256") != certificate.get("authorization_sha256")
        or adoption.get("recovery_certificate_sha256") != evidence["recovery_certificate_sha256"]
    ):
        raise ConfigError(f"{worker.worker_id} recovery adoption receipt 非法")
    if metadata_only:
        support_hashes = {
            key: worker.recovery_refs[key].sha256 for key in RECOVERY_SUPPORT_TARGETS
        }
    else:
        assert attempt is not None
        support_root = attempt.parents[3] / "provenance" / "recovery" / worker.worker_id
        support_paths = {
            key: support_root / filename for key, filename in RECOVERY_SUPPORT_TARGETS.items()
        }
        support_hashes = {key: _sha256_file(path) for key, path in support_paths.items()}
    if (
        support_hashes["protocol_amendment"] != evidence["protocol_amendment_sha256"]
        or support_hashes["pre_recovery_archive"] != evidence["pre_recovery_archive_sha256"]
        or support_hashes["pre_recovery_manifest"]
        != evidence["pre_recovery_file_manifest_sha256"]
        or support_hashes["authorization"] != adoption["authorization_sha256"]
        or adoption["authorization_sha256"] != certificate.get("authorization_sha256")
    ):
        raise ConfigError(f"{worker.worker_id} recovery source evidence SHA 不闭合")
    identity_correction = certificate.get("identity_correction")
    canonical_legacy = canonical.get("legacy_json_roundtrip_actual_sha256")
    if (
        not isinstance(identity_correction, dict)
        or identity_correction.get("original_expected_sha256")
        != evidence["original_expected_config_sha256"]
        or identity_correction.get("corrected_actual_sha256") != evidence["corrected_config_sha256"]
        or canonical.get("original_expected_sha256") != evidence["original_expected_config_sha256"]
        or canonical.get("corrected_actual_sha256") != evidence["corrected_config_sha256"]
        or evidence["corrected_config_sha256"] != evidence["original_expected_config_sha256"]
        or SHA256_RE.fullmatch(str(canonical_legacy or "")) is None
        or identity_correction.get("legacy_json_roundtrip_actual_sha256") != canonical_legacy
        or canonical_legacy == evidence["original_expected_config_sha256"]
        or canonical.get("semantic_equivalence_after_stringifying_mapping_keys") is not True
        or identity_correction.get("semantic_equivalence_after_stringifying_mapping_keys")
        is not True
        or canonical.get("only_authorized_difference") != "integer-versus-string mapping keys"
        or identity_correction.get("only_authorized_difference")
        != "integer-versus-string mapping keys"
    ):
        raise ConfigError(f"{worker.worker_id} recovery canonical identity 非法")
    _validate_recovery_state_anchor(
        source_state_rows,
        original_terminal_state_sha256=str(evidence["original_terminal_state_sha256"]),
        label=worker.worker_id,
    )


def _validate_shard_identity_receipts(
    worker: WorkerPlan,
    *,
    root: Path,
    attempt: Path,
    archives: Mapping[str, ArchiveCatalog],
) -> dict[str, list[dict[str, str]]]:
    """Close each logical side over immutable structural shard receipts."""

    identities_by_side: dict[str, list[dict[str, str]]] = {}
    for side, slot, expected_tasks, expected_hands, _shard_role in _expected_sides(
        worker.role, worker.slots
    ):
        completion_targets = [
            (ref, target)
            for kind, ref, target in worker.evidence_targets[side]
            if kind == "completion"
        ]
        if len(completion_targets) != 1:
            raise ConfigError(f"{worker.worker_id}/{side} completion target 不唯一")
        completion_ref, completion_target = completion_targets[0]
        completion_path = root.joinpath(*completion_target.parts)
        completion = _read_json_bytes(
            completion_path.read_bytes(),
            label=f"{worker.worker_id}/{side} completion receipt",
        )
        if (
            completion.get("schema_version") not in COMPLETION_EVIDENCE_SCHEMAS
            or completion.get("status") != "complete"
            or completion.get("worker_id") != worker.worker_id
            or (
                "effect_fields_read" in completion
                and completion.get("effect_fields_read") is not False
            )
            or completion_path.stat().st_size != completion_ref.size
            or _sha256_file(completion_path) != completion_ref.sha256
        ):
            raise ConfigError(f"{worker.worker_id}/{side} completion receipt schema/identity 非法")
        if completion.get("schema_version") == COMPLETION_SCHEMA:
            files_tsv_sha = _require_sha256(
                completion.get("files_tsv_sha256"),
                label=f"{worker.worker_id}/{side} completion.files_tsv_sha256",
            )
            source_ids = {
                str(task["source_archive_id"])
                for task_id, task in worker.task_rows.items()
                if task_id in expected_tasks
            }
            bound_files = [
                entry
                for source_id in source_ids
                for entry in archives[source_id].entries.values()
                if entry.member.endswith("/files.tsv")
                and f"/{worker.worker_id}/" in f"/{entry.member}"
                and entry.sha256 == files_tsv_sha
            ]
            if len(bound_files) != 1:
                raise ConfigError(f"{worker.worker_id}/{side} completion files.tsv SHA 未唯一绑定")
        identity_targets = [
            (ref, target)
            for kind, ref, target in worker.evidence_targets[side]
            if kind.startswith("identity-")
        ]
        if len(identity_targets) != len(worker.identity_refs[side]):
            raise ConfigError(f"{worker.worker_id}/{side} identity receipt target 不闭合")
        observed_tasks: list[str] = []
        observed_hands = 0
        source_shards: list[dict[str, str]] = []
        for ref, target in identity_targets:
            path = root.joinpath(*target.parts)
            receipt = _read_json_bytes(
                path.read_bytes(),
                label=f"{worker.worker_id}/{side} shard identity receipt",
            )
            selected = receipt.get("selected_task_ids")
            tasks = receipt.get("tasks")
            if (
                receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA
                or receipt.get("status") != "complete"
                or receipt.get("effect_fields_read") is not False
                or receipt.get("worker_id") != worker.worker_id
                or receipt.get("role") != worker.role
                or receipt.get("seed") != worker.seed
                or receipt.get("pair_id") != worker.pair_id
                or receipt.get("physical_slot") != slot
                or receipt.get("shard_role") != side
                or not isinstance(selected, list)
                or not selected
                or not all(isinstance(task_id, str) for task_id in selected)
                or len(selected) != len(set(selected))
                or not isinstance(tasks, list)
                or len(tasks) != len(selected)
            ):
                raise ConfigError(
                    f"{worker.worker_id}/{side} shard identity receipt schema/identity 非法"
                )
            task_ids = [
                str(task.get("task_id", "")) if isinstance(task, dict) else "" for task in tasks
            ]
            if task_ids != selected:
                raise ConfigError(f"{worker.worker_id}/{side} shard identity tasks 顺序/集合非法")
            source_shard_id = _safe_identifier(
                receipt.get("shard_id"),
                label=f"{worker.worker_id}/{side} source shard_id",
            )
            source_kind = _safe_identifier(
                receipt.get("source_kind"),
                label=f"{worker.worker_id}/{side} source_kind",
            )
            for task_id, task in zip(task_ids, tasks, strict=True):
                assert isinstance(task, dict)
                receipt_path = attempt / "task_receipts" / f"{task_id}.json"
                expected_run_dir = str(worker.task_rows.get(task_id, {}).get("target_run_dir", ""))
                if (
                    task_id not in expected_tasks
                    or task.get("schema_version") != SHARD_TASK_SCHEMA
                    or task.get("status") != "complete"
                    or task.get("producer_worker_id") != worker.worker_id
                    or task.get("run_dir") != expected_run_dir
                    or task.get("planned_hands") != PLANNED_HANDS[task_id]
                    or task.get("actual_hands") != PLANNED_HANDS[task_id]
                    or not receipt_path.is_file()
                    or task.get("task_receipt_sha256") != _sha256_file(receipt_path)
                ):
                    raise ConfigError(
                        f"{worker.worker_id}/{side}/{task_id} shard task identity 非法"
                    )
                observed_hands += PLANNED_HANDS[task_id]
            if path.stat().st_size != ref.size or _sha256_file(path) != ref.sha256:
                raise ConfigError(f"{worker.worker_id}/{side} shard identity receipt SHA 非法")
            source_shards.append(
                {
                    "shard_id": source_shard_id,
                    "source_kind": source_kind,
                    "receipt_sha256": ref.sha256,
                }
            )
            observed_tasks.extend(task_ids)
        if (
            len(observed_tasks) != len(set(observed_tasks))
            or tuple(observed_tasks) != expected_tasks
            or observed_hands != expected_hands
            or len({row["shard_id"] for row in source_shards}) != len(source_shards)
        ):
            raise ConfigError(f"{worker.worker_id}/{side} shard identity union/hands 不闭合")
        identities_by_side[side] = source_shards
    all_source_shard_ids = [
        row["shard_id"] for source_shards in identities_by_side.values() for row in source_shards
    ]
    if len(all_source_shard_ids) != len(set(all_source_shard_ids)):
        raise ConfigError(f"{worker.worker_id} source shard_id 跨 side 重复")
    return identities_by_side


def _validate_secondary_normalization_evidence(
    worker: WorkerPlan,
    *,
    root: Path,
) -> None:
    if worker.role != "secondary":
        return
    low_slot = worker.slots[0]
    rows = {
        target.name: root.joinpath(*target.parts)
        for kind, _ref, target in worker.evidence_targets["low"]
        if kind.startswith("extra-")
    }
    if set(rows) != SECONDARY_NORMALIZATION_FILES:
        raise ConfigError(f"{worker.worker_id}/low secondary normalization 集合不闭合")
    for name, expected_schema in SECONDARY_NORMALIZATION_SCHEMAS.items():
        value = _read_json_bytes(
            rows[name].read_bytes(),
            label=f"{worker.worker_id}/low secondary normalization {name}",
        )
        if (
            value.get("schema_version") != expected_schema
            or ("effect_fields_read" in value and value.get("effect_fields_read") is not False)
            or ("worker_id" in value and value.get("worker_id") != worker.worker_id)
            or ("role" in value and value.get("role") != "secondary")
            or ("seed" in value and value.get("seed") != worker.seed)
            or ("pair_id" in value and value.get("pair_id") != worker.pair_id)
            or ("physical_slot" in value and value.get("physical_slot") != low_slot)
        ):
            raise ConfigError(
                f"{worker.worker_id}/low secondary normalization schema/identity 非法"
            )


def _write_files_tsv(attempt: Path) -> None:
    rows = []
    for path in sorted(item for item in attempt.rglob("*") if item.is_file()):
        relative = path.relative_to(attempt).as_posix()
        if relative in FILES_TSV_EXCLUSIONS:
            continue
        rows.append((relative, path.stat().st_size, _sha256_file(path)))
    _write_new(attempt / "files.tsv", _files_tsv_bytes(rows))


def _files_tsv_bytes(rows: Sequence[tuple[str, int, str]]) -> bytes:
    text = "relative_path\tsize\tsha256\n" + "".join(
        f"{relative}\t{size}\t{sha256}\n"
        for relative, size, sha256 in sorted(rows, key=lambda row: row[0])
    )
    return text.encode("utf-8")


def _validate_files_tsv(attempt: Path) -> None:
    with (attempt / "files.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    listed = set()
    for row in rows:
        relative = _safe_relative(row.get("relative_path"), label=f"{attempt.name} files.tsv")
        path = attempt.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row["size"])
            or _sha256_file(path) != row["sha256"]
        ):
            raise ConfigError(f"{attempt} files.tsv artifact SHA 不匹配")
        listed.add(relative.as_posix())
    actual = {path.relative_to(attempt).as_posix() for path in attempt.rglob("*") if path.is_file()}
    if actual != listed | set(FILES_TSV_EXCLUSIONS):
        raise ConfigError(f"{attempt} files.tsv exact closure 失败")


def _build_worker_composition_values(
    worker: WorkerPlan,
    *,
    root: Path,
    source_identities: Mapping[str, list[dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    side_specs = _expected_sides(worker.role, worker.slots)
    composition_sides = []
    lineage_shards = []
    for shard_raw, expected_side in zip(worker.raw["shards"], side_specs, strict=True):
        side, slot, task_ids, hands, shard_role = expected_side
        execution_rows = []
        for kind, ref, target in worker.evidence_targets[side]:
            target_path = root.joinpath(*target.parts)
            if (
                not target_path.is_file()
                or target_path.stat().st_size != ref.size
                or _sha256_file(target_path) != ref.sha256
            ):
                raise ConfigError(f"{worker.worker_id}/{side} execution evidence SHA 非法")
            execution_rows.append(
                {
                    "kind": kind,
                    "path": target.relative_to(PurePosixPath("manifests/composition")).as_posix(),
                    "sha256": ref.sha256,
                }
            )
        composition_sides.append(
            {
                "side": side,
                "physical_slot": slot,
                "task_ids": list(task_ids),
                "actual_hands": hands,
                "source_shards": copy.deepcopy(source_identities[side]),
                "execution_receipts": execution_rows,
            }
        )
        lineage_shards.append(
            {
                "shard_id": str(shard_raw["shard_id"]),
                "physical_slot": slot,
                "side": side,
                "shard_role": shard_role,
                "task_ids": list(task_ids),
                "actual_hands": hands,
                "source_shards": copy.deepcopy(source_identities[side]),
                "shard_manifest_sha256": str(shard_raw["shard_manifest"]["sha256"]),
                "completion_receipt_sha256": str(shard_raw["completion_receipt"]["sha256"]),
            }
        )
    composition = {
        "schema_version": COMPOSITION_SCHEMA,
        "worker_id": worker.worker_id,
        "seed": worker.seed,
        "pair_id": worker.pair_id,
        "sides": composition_sides,
        "task_union": [task_id for side in composition_sides for task_id in side["task_ids"]],
        "actual_hands": sum(side["actual_hands"] for side in composition_sides),
    }
    lineage = {
        "pair_id": worker.pair_id,
        "paired_physical_slots": list(worker.slots),
        "composition_receipt_path": f"composition/{worker.worker_id}.json",
        "composition_receipt_sha256": hashlib.sha256(_json_bytes(composition)).hexdigest(),
        "shards": lineage_shards,
    }
    return composition, lineage


def _materialize_worker(
    worker: WorkerPlan,
    *,
    root: Path,
    archives: Mapping[str, ArchiveCatalog],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    attempt = root / "snapshots" / worker.worker_id / str(worker.seed) / "attempt_01"
    state_path = attempt / "state.tsv"
    prefix_rows = _parse_state(state_path, label=f"{worker.worker_id} state prefix")
    appended = _build_state_rows(prefix_rows, worker.state_append, label=worker.worker_id)
    pre_completion_rows = appended[:-1] if appended else []
    final_rows = appended[-1:] if appended else []
    _append_bytes(state_path, _state_row_bytes(pre_completion_rows))
    task_results = _validate_task_receipts(worker, attempt)
    _validate_recovery(worker, attempt, source_state_rows=prefix_rows)
    source_identities = _validate_shard_identity_receipts(
        worker,
        root=root,
        attempt=attempt,
        archives=archives,
    )
    _validate_secondary_normalization_evidence(worker, root=root)
    composition, lineage = _build_worker_composition_values(
        worker,
        root=root,
        source_identities=source_identities,
    )
    composition_path = root / "manifests" / "composition" / f"{worker.worker_id}.json"
    _write_json_new(composition_path, composition)
    base_manifest = _read_json_bytes(
        _read_ref_bytes(
            worker.base_manifest_ref,
            archives=archives,
            label=f"{worker.worker_id}.base_manifest post-copy",
        ),
        label=f"{worker.worker_id}.base_manifest post-copy",
    )
    augmented = {**base_manifest, "phase_f_shard_lineage": lineage}
    manifest_path = root / "manifests" / f"{worker.worker_id}.json"
    manifest_bytes = _json_bytes(augmented)
    _write_new(manifest_path, manifest_bytes)
    _write_new(attempt / "worker_manifest.json", manifest_bytes)
    _write_json_new(
        attempt / "task_results.json",
        {
            "schema_version": TASK_RESULTS_SCHEMA,
            "worker_id": worker.worker_id,
            "tasks": task_results,
        },
    )
    _write_files_tsv(attempt)
    _write_json_new(
        attempt / "completion_receipt.json",
        {
            "schema_version": COMPLETION_SCHEMA,
            "worker_id": worker.worker_id,
            "status": "complete",
            "not_for_paper": False,
            "files_tsv_sha256": _sha256_file(attempt / "files.tsv"),
        },
    )
    _append_bytes(state_path, _state_row_bytes(final_rows))
    final_state = _parse_state(state_path, label=f"{worker.worker_id} canonical state")
    if final_state[-1]["status"] != "complete":
        raise ConfigError(f"{worker.worker_id} canonical state 非 complete")
    _validate_files_tsv(attempt)
    completion = _read_json_bytes(
        (attempt / "completion_receipt.json").read_bytes(),
        label=f"{worker.worker_id} completion",
    )
    if (
        completion.get("schema_version") != COMPLETION_SCHEMA
        or completion.get("worker_id") != worker.worker_id
        or completion.get("status") != "complete"
        or completion.get("files_tsv_sha256") != _sha256_file(attempt / "files.tsv")
    ):
        raise ConfigError(f"{worker.worker_id} completion receipt 非法")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    return augmented, lineage, manifest_sha


def _build_manifest_index(
    manifests: Sequence[dict[str, Any]],
    manifest_shas: Mapping[str, str],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": WORKER_MANIFEST_SCHEMA,
        "protocol_status": EXPEDITED_STATUS,
        "protocol_id": PROTOCOL_ID,
        "analysis_contract_id": ANALYSIS_CONTRACT_ID,
        "analysis_contract_sha256": contract_sha256,
        "seed_count": 6,
        "worker_count": 12,
        "workers": [
            {
                "worker_id": manifest["worker_id"],
                "role": manifest["role"],
                "seed_bundle": manifest["seed_bundle"],
                "manifest_sha256": manifest_shas[str(manifest["worker_id"])],
            }
            for manifest in manifests
        ],
    }
    return {**body, "bundle_sha256": sha256_json(body)}


def _build_analysis_input(
    *,
    workers: Sequence[WorkerPlan],
    manifests: Mapping[str, dict[str, Any]],
    manifest_shas: Mapping[str, str],
    analysis: Mapping[str, Any],
    manifest_index_sha256: str,
) -> dict[str, Any]:
    code_shas = {
        str(manifests[worker.worker_id]["common_identity"]["code_sha"]) for worker in workers
    }
    if len(code_shas) != 1:
        raise ConfigError("canonical worker experiment code SHA 不唯一")
    experiment_code_sha = _require_sha1(
        next(iter(code_shas)), label="canonical experiment_code_sha"
    )
    rows = []
    for worker in workers:
        manifest = manifests[worker.worker_id]
        lineage = copy.deepcopy(manifest["phase_f_shard_lineage"])
        lineage["composition_receipt_path"] = f"manifests/{lineage['composition_receipt_path']}"
        rows.append(
            {
                "worker_id": worker.worker_id,
                "pod_id": str(manifest["pod_id"]),
                "seed": worker.seed,
                "expected_identity": copy.deepcopy(manifest["common_identity"]),
                "expected_worker_manifest_sha256": manifest_shas[worker.worker_id],
                "attempts": [
                    {
                        "attempt": "attempt_01",
                        "relative_path": (f"snapshots/{worker.worker_id}/{worker.seed}/attempt_01"),
                    }
                ],
                "shard_lineage": lineage,
            }
        )
    return {
        "schema_version": "task8b-phase-f-input-v1",
        "analysis_contract_id": ANALYSIS_CONTRACT_ID,
        "analysis_contract_sha256": analysis["analysis_contract_sha256"],
        "synthetic_test_mode": False,
        "analysis_code_sha": analysis["analysis_code_sha"],
        "analysis_code_dirty": False,
        "analysis_frozen_at_utc": analysis["analysis_frozen_at_utc"],
        "dependency_lock_sha256": analysis["dependency_lock_sha256"],
        "experiment_code_sha": experiment_code_sha,
        "input_snapshot_id": _require_sha256(
            manifest_index_sha256,
            label="manifest_index_sha256",
        ),
        "unlocked_at_utc": analysis["unlocked_at_utc"],
        "unlocked_by": analysis["unlocked_by"],
        "pre_unlock_manifest_sha256": analysis["pre_unlock_manifest_sha256"],
        "workers": rows,
    }


def _validate_task_source_uniqueness(workers: Sequence[WorkerPlan]) -> None:
    sources: list[tuple[str, str]] = []
    receipts: list[tuple[str, ...]] = []
    for worker in workers:
        for task in worker.task_rows.values():
            sources.append(
                (
                    str(task["source_archive_id"]),
                    str(task["source_prefix"]),
                )
            )
            receipts.append(task["receipt_ref"].key)
    if len(sources) != 60 or len(set(sources)) != 60:
        raise ConfigError("six-seed task source 必须精确为 60 个全局唯一选择")
    if len(receipts) != 60 or len(set(receipts)) != 60:
        raise ConfigError("six-seed task receipt 必须精确为 60 个全局唯一来源")


def _validate_authoritative_real_selections(
    *,
    archives: Mapping[str, ArchiveCatalog],
    workers: Sequence[WorkerPlan],
) -> list[dict[str, str]]:
    if len(archives) != 17:
        raise ConfigError("authoritative real source map 必须精确绑定 17 个 archive")
    by_worker = {worker.worker_id: worker for worker in workers}
    results: list[dict[str, str]] = []
    for (worker_id, task_id), expected in AUTHORITATIVE_REAL_SELECTIONS.items():
        task = by_worker[worker_id].task_rows[task_id]
        source_archive_id = str(task["source_archive_id"])
        source_prefix = str(task["source_prefix"])
        target_run_dir = str(task["target_run_dir"])
        if (
            source_archive_id != expected["archive_id"]
            or source_prefix != expected["source_prefix"]
            or target_run_dir != expected["target_run_dir"]
        ):
            raise ConfigError(f"{worker_id}/{task_id} authoritative source selection 非法")
        results.append(
            {
                "worker_id": worker_id,
                "task_id": task_id,
                "source_archive_id": source_archive_id,
                "source_prefix": source_prefix,
                "target_run_dir": target_run_dir,
            }
        )
    for worker in workers:
        for shard, expected_side in zip(
            worker.raw["shards"],
            _expected_sides(worker.role, worker.slots),
            strict=True,
        ):
            side = expected_side[0]
            source_ids = list(shard["source_archive_ids"])
            if len(source_ids) != 1:
                raise ConfigError(
                    f"{worker.worker_id}/{side} authoritative shard 必须唯一绑定一个 archive"
                )
            source_id = str(source_ids[0])
            shard_manifest_sha = str(shard["shard_manifest"]["sha256"])
            if shard_manifest_sha != archives[source_id].manifest_ref.sha256:
                raise ConfigError(
                    f"{worker.worker_id}/{side} shard manifest 与 archive manifest 不一致"
                )
    return results


def _inventory_add(
    inventory: dict[PurePosixPath, tuple[int, str]],
    path: PurePosixPath,
    payload_or_identity: bytes | tuple[int, str],
) -> None:
    _safe_relative(path.as_posix(), label="planned canonical path")
    if isinstance(payload_or_identity, bytes):
        identity = (
            len(payload_or_identity),
            hashlib.sha256(payload_or_identity).hexdigest(),
        )
    else:
        identity = payload_or_identity
    existing = inventory.get(path)
    if existing is not None and existing != identity:
        raise ConfigError(f"planned canonical target collision：{path}")
    inventory[path] = identity


def _copy_target_inventory(
    requests: Sequence[CopyRequest],
) -> dict[PurePosixPath, tuple[int, str]]:
    inventory: dict[PurePosixPath, tuple[int, str]] = {}
    for request in requests:
        for target in request.targets:
            _inventory_add(
                inventory,
                target,
                (request.source.size, request.source.sha256),
            )
    return inventory


def _stage_ref(
    ref: ArtifactRef,
    *,
    archives: Mapping[str, ArchiveCatalog],
    root: Path,
    target: PurePosixPath,
    label: str,
) -> int:
    payload = _read_ref_bytes(ref, archives=archives, label=label)
    _validate_json_assertions(payload, ref, label=label)
    destination = root.joinpath(*target.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new(destination, payload)
    return len(payload)


def _validate_authoritative_recovery_suffix(
    source_state_rows: Sequence[dict[str, str]],
    *,
    original_terminal_state_sha256: str,
    state_append: Sequence[Mapping[str, str]],
    label: str,
) -> None:
    anchor_index = _validate_recovery_state_anchor(
        source_state_rows,
        original_terminal_state_sha256=original_terminal_state_sha256,
        label=label,
    )
    existing_suffix = tuple(
        str(row["status"]) for row in source_state_rows[anchor_index + 1 :]
    )
    append_suffix = tuple(str(row["status"]) for row in state_append)
    if existing_suffix != REAL_RECOVERY_EXISTING_SUFFIX:
        raise ConfigError(
            f"{label} authoritative recovery existing suffix "
            "必须为 validating->running"
        )
    if append_suffix != REAL_RECOVERY_APPEND_SUFFIX:
        raise ConfigError(
            f"{label} authoritative recovery append "
            "必须为 finalizing->complete"
        )


def _validate_authoritative_recovery_state(
    worker: WorkerPlan,
    *,
    archives: Mapping[str, ArchiveCatalog],
    source_state_rows: Sequence[dict[str, str]],
) -> None:
    receipt = _read_json_bytes(
        _read_ref_bytes(
            worker.task_rows[RECOVERY_TASK_ID]["receipt_ref"],
            archives=archives,
            label=f"{worker.worker_id} authoritative recovery receipt",
        ),
        label=f"{worker.worker_id} authoritative recovery receipt",
    )
    evidence = receipt.get("phase_f_evidence")
    if not isinstance(evidence, dict):
        raise ConfigError(f"{worker.worker_id} authoritative recovery evidence 缺失")
    _validate_authoritative_recovery_suffix(
        source_state_rows,
        original_terminal_state_sha256=str(
            evidence.get("original_terminal_state_sha256", "")
        ),
        state_append=worker.state_append,
        label=worker.worker_id,
    )


def _validate_pre_recovery_archive_closure(
    worker: WorkerPlan,
    *,
    archives: Mapping[str, ArchiveCatalog],
    expected_failed_anchor_sha256: str,
) -> dict[str, Any]:
    archive_ref = worker.recovery_refs["pre_recovery_archive"]
    manifest_ref = worker.recovery_refs["pre_recovery_manifest"]
    if archive_ref.path is None:
        raise ConfigError(f"{worker.worker_id} pre-recovery archive 必须为本地不可变文件")
    manifest = _parse_flexible_manifest_tsv(
        _read_ref_bytes(
            manifest_ref,
            archives=archives,
            label=f"{worker.worker_id} pre-recovery manifest",
        ),
        label=f"{worker.worker_id} pre-recovery",
    )
    regular_members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive_ref.path, mode="r:*") as archive:
        for member in archive:
            raw_name = (
                member.name[:-1]
                if member.isdir() and member.name.endswith("/")
                else member.name
            )
            name = _safe_relative(
                raw_name,
                label=f"{worker.worker_id} pre-recovery tar member",
            ).as_posix()
            if member.isdir():
                continue
            if not member.isreg() or name in regular_members:
                raise ConfigError(
                    f"{worker.worker_id} pre-recovery archive 含特殊或重复 member"
                )
            regular_members[name] = member
        candidates: list[str] = []
        manifest_names = set(manifest)
        for prefix in {
            tar_name[: -len(manifest_name)]
            for tar_name in regular_members
            for manifest_name in manifest_names
            if tar_name.endswith(manifest_name)
            and (
                len(tar_name) == len(manifest_name)
                or tar_name[-len(manifest_name) - 1] == "/"
            )
        }:
            if {f"{prefix}{name}" for name in manifest_names} == set(regular_members):
                candidates.append(prefix)
        if len(candidates) != 1:
            raise ConfigError(
                f"{worker.worker_id} pre-recovery archive/manifest exact closure 失败"
            )
        prefix = candidates[0]
        state_payloads: list[bytes] = []
        for relative, entry in manifest.items():
            member = regular_members[f"{prefix}{relative}"]
            if member.size != entry.size:
                raise ConfigError(
                    f"{worker.worker_id} pre-recovery member size 不匹配：{relative}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ConfigError(
                    f"{worker.worker_id} pre-recovery member 无法读取：{relative}"
                )
            digest = hashlib.sha256()
            size = 0
            state_chunks: list[bytes] | None = (
                [] if PurePosixPath(relative).name == "state.tsv" else None
            )
            with extracted:
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    if state_chunks is not None:
                        state_chunks.append(chunk)
            if size != entry.size or digest.hexdigest() != entry.sha256:
                raise ConfigError(
                    f"{worker.worker_id} pre-recovery member SHA 不匹配：{relative}"
                )
            if state_chunks is not None:
                state_payloads.append(b"".join(state_chunks))
    if len(state_payloads) != 1:
        raise ConfigError(f"{worker.worker_id} pre-recovery state.tsv 不唯一")
    failed_rows = _parse_state_bytes(
        state_payloads[0],
        label=f"{worker.worker_id} pre-recovery state",
    )
    if (
        failed_rows[-1]["status"] != "failed"
        or failed_rows[-1]["row_sha256"] != expected_failed_anchor_sha256
    ):
        raise ConfigError(
            f"{worker.worker_id} pre-recovery failed state anchor 未闭合"
        )
    return {
        "worker_id": worker.worker_id,
        "file_count": len(manifest),
        "byte_count": sum(entry.size for entry in manifest.values()),
        "archive_sha256": archive_ref.sha256,
        "manifest_sha256": manifest_ref.sha256,
        "failed_state_sha256": expected_failed_anchor_sha256,
    }


def _positive_pid(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} observed_pid 非正整数")
    return value


def _validate_active_pid_absence_payload(
    worker: WorkerPlan,
    *,
    payload: bytes,
    profile: str,
) -> None:
    value = _read_json_bytes(
        payload,
        label=f"{worker.worker_id} active PID absence evidence",
    )
    common_valid = (
        value.get("active_process_absent") is True
        and _positive_pid(
            value.get("observed_pid"),
            label=f"{worker.worker_id} active PID absence evidence",
        )
        > 0
    )
    _parse_utc_timestamp(
        value.get("probe_utc"),
        label=f"{worker.worker_id} active PID absence evidence.probe_utc",
    )
    expected_host = f"H{int(worker.worker_id[1:]):02d}"
    normalized_pidfile = str(
        value.get("pidfile", value.get("pid_file", ""))
    ).replace("\\", "/")
    expected_pidfile_suffix = (
        f"/runtime_control/{expected_host}/{worker.worker_id}/attempt01/"
        f"{worker.worker_id}_attempt01.pid"
    )
    pidfile_valid = normalized_pidfile.endswith(expected_pidfile_suffix)
    if profile == "host-worker-count-pidfile-v0":
        valid = (
            set(value)
            == {
                "active_process_absent",
                "formal_worker_count",
                "host",
                "observed_pid",
                "pidfile",
                "probe_utc",
                "worker_id",
            }
            and value.get("formal_worker_count") == 0
            and value.get("host") == expected_host
            and value.get("worker_id") == worker.worker_id
            and pidfile_valid
        )
    elif profile == "minimal-legacy-v0":
        valid = set(value) == {
            "active_process_absent",
            "observed_pid",
            "probe_utc",
        }
    elif profile == "task8b-active-pid-absence-v1":
        valid = (
            set(value)
            == {
                "active_process_absent",
                "host_id",
                "observed_pid",
                "pid_file",
                "probe",
                "probe_utc",
                "schema_version",
                "worker_id",
            }
            and value.get("schema_version") == "task8b-active-pid-absence-v1"
            and value.get("host_id") == expected_host
            and value.get("worker_id") == worker.worker_id
            and bool(str(value.get("probe", "")).strip())
            and pidfile_valid
        )
    elif profile == "host-worker-count-v0":
        valid = (
            set(value)
            == {
                "active_process_absent",
                "formal_worker_count",
                "host",
                "observed_pid",
                "probe_utc",
                "worker_id",
            }
            and value.get("formal_worker_count") == 0
            and value.get("host") == expected_host
            and value.get("worker_id") == worker.worker_id
        )
    else:
        raise ConfigError(f"{worker.worker_id} active PID absence profile 未批准")
    if not common_valid or not valid:
        raise ConfigError(
            f"{worker.worker_id} active PID absence evidence identity 非法"
        )


def _bind_authoritative_active_pid_evidence(
    worker: WorkerPlan,
    *,
    archives: Mapping[str, ArchiveCatalog],
) -> ArtifactRef:
    expected = AUTHORITATIVE_ACTIVE_PID_EVIDENCE[worker.worker_id]
    authorization = _read_json_bytes(
        _read_ref_bytes(
            worker.recovery_refs["authorization"],
            archives=archives,
            label=f"{worker.worker_id} recovery authorization PID binding",
        ),
        label=f"{worker.worker_id} recovery authorization PID binding",
    )
    archive_id = str(expected["archive_id"])
    member = str(expected["member"])
    sha256 = str(expected["sha256"])
    catalog = archives.get(archive_id)
    entry = catalog.entries.get(member) if catalog is not None else None
    if (
        authorization.get("active_pid_absence_evidence_sha256") != sha256
        or worker.recovery_refs["authorization"].archive_id != archive_id
        or entry is None
        or entry.sha256 != sha256
    ):
        raise ConfigError(
            f"{worker.worker_id} active PID absence manifest binding 非法"
        )
    ref = ArtifactRef(
        archive_id=archive_id,
        member=member,
        size=entry.size,
        sha256=entry.sha256,
    )
    payload = _read_ref_bytes(
        ref,
        archives=archives,
        label=f"{worker.worker_id} active PID absence evidence",
    )
    _validate_active_pid_absence_payload(
        worker,
        payload=payload,
        profile=str(expected["profile"]),
    )
    worker.evidence_targets["recovery_active_pid"] = [
        (
            "active_pid_absence",
            ref,
            PurePosixPath(
                f"provenance/recovery/{worker.worker_id}/"
                "active_pid_absence.json"
            ),
        )
    ]
    return ref


def _validate_authoritative_recovery_support(
    worker: WorkerPlan,
    *,
    archives: Mapping[str, ArchiveCatalog],
) -> dict[str, Any]:
    receipt = _read_json_bytes(
        _read_ref_bytes(
            worker.task_rows[RECOVERY_TASK_ID]["receipt_ref"],
            archives=archives,
            label=f"{worker.worker_id} authoritative recovery receipt",
        ),
        label=f"{worker.worker_id} authoritative recovery receipt",
    )
    evidence = receipt.get("phase_f_evidence")
    adoption = receipt.get("recovery_adoption")
    if not isinstance(evidence, dict) or not isinstance(adoption, dict):
        raise ConfigError(f"{worker.worker_id} authoritative recovery receipt 非法")
    amendment = _read_json_bytes(
        _read_ref_bytes(
            worker.recovery_refs["protocol_amendment"],
            archives=archives,
            label=f"{worker.worker_id} protocol amendment",
        ),
        label=f"{worker.worker_id} protocol amendment",
    )
    attempt_policy = amendment.get("attempt_policy")
    state_policy = amendment.get("state_policy")
    if (
        amendment.get("schema_version")
        != "task8b-pre-unblind-verifier-recovery-amendment-v1"
        or amendment.get("amendment_id") != RECOVERY_PROTOCOL_AMENDMENT_ID
        or amendment.get("approved_before_effect_unblind") is not True
        or amendment.get("formal_effects_read") is not False
        or amendment.get("scientific_outcome_fields_accessed") is not False
        or amendment.get("activation_fail_closed") is not True
        or not isinstance(attempt_policy, dict)
        or attempt_policy.get("outer_attempt") != "attempt_01"
        or attempt_policy.get("cross_attempt_splicing_allowed") is not False
        or attempt_policy.get("new_outer_attempt_allowed") is not False
        or not isinstance(state_policy, dict)
        or state_policy.get("preserve_original_failed_rows") is not True
        or state_policy.get("rewrite_state_history_allowed") is not False
        or state_policy.get("allowed_append_transition")
        != ["failed", "validating", "running", "finalizing", "complete"]
    ):
        raise ConfigError(f"{worker.worker_id} protocol amendment identity/policy 非法")
    authorization = _read_json_bytes(
        _read_ref_bytes(
            worker.recovery_refs["authorization"],
            archives=archives,
            label=f"{worker.worker_id} recovery authorization",
        ),
        label=f"{worker.worker_id} recovery authorization",
    )
    authorization_ref = worker.recovery_refs["authorization"]
    if authorization_ref.archive_id is None:
        raise ConfigError(
            f"{worker.worker_id} recovery authorization 必须来自 SHA-bound support archive"
        )
    support_catalog = archives[authorization_ref.archive_id]
    active_pid_sha = str(authorization.get("active_pid_absence_evidence_sha256", ""))
    verifier_sha = str(authorization.get("verifier_code_sha", ""))
    expected_attempt_suffix = f"/{worker.worker_id}/{worker.seed}"
    if (
        set(authorization) != RECOVERY_AUTHORIZATION_FIELDS
        or authorization.get("schema_version")
        != "task8b-same-attempt-recovery-authorization-v1"
        or authorization.get("authorization_id") != adoption.get("authorization_id")
        or authorization.get("worker_id") != worker.worker_id
        or authorization.get("task_id") != RECOVERY_TASK_ID
        or authorization.get("reason") != RECOVERY_REASON
        or authorization.get("authorized") is not True
        or authorization.get("same_attempt_recovery_authorized") is not True
        or authorization.get("task1_adoption_only") is not True
        or authorization.get("active_process_absent_confirmed") is not True
        or not str(authorization.get("attempt_root", "")).replace("\\", "/").endswith(
            expected_attempt_suffix
        )
        or authorization.get("frozen_code_sha") != FROZEN_SCIENTIFIC_CODE_SHA
        or authorization.get("worker_manifest_sha256")
        != worker.base_manifest_ref.sha256
        or authorization.get("baseline_manifest_sha256")
        != worker.recovery_refs["pre_recovery_manifest"].sha256
        or authorization.get("pre_recovery_archive_sha256")
        != worker.recovery_refs["pre_recovery_archive"].sha256
        or authorization.get("failed_state_row_sha256")
        != evidence["original_terminal_state_sha256"]
        or authorization.get("protocol_amendment_id")
        != RECOVERY_PROTOCOL_AMENDMENT_ID
        or authorization.get("protocol_amendment_sha256")
        != worker.recovery_refs["protocol_amendment"].sha256
        or authorization.get("parent_preunlock_sha256")
        != amendment.get("parent_pre_unlock_manifest_sha256")
        or verifier_sha != AUTHORITATIVE_RECOVERY_VERIFIER_CODE_SHA
        or verifier_sha != evidence["verifier_code_sha"]
        or authorization.get("verifier_identity_source_sha256")
        != AUTHORITATIVE_RECOVERY_VERIFIER_IDENTITY_SOURCE_SHA256
        or authorization.get("controller_sha256")
        != AUTHORITATIVE_RECOVERY_CONTROLLER_SHA256
        or authorization.get("formal_runner_sha256")
        != AUTHORITATIVE_RECOVERY_FORMAL_RUNNER_SHA256
        or SHA256_RE.fullmatch(active_pid_sha) is None
        or sum(
            entry.sha256 == active_pid_sha for entry in support_catalog.entries.values()
        )
        != 1
        or sum(
            entry.sha256 == worker.recovery_refs["protocol_amendment"].sha256
            for entry in support_catalog.entries.values()
        )
        != 1
    ):
        raise ConfigError(f"{worker.worker_id} recovery authorization identity 非法")
    closure = _validate_pre_recovery_archive_closure(
        worker,
        archives=archives,
        expected_failed_anchor_sha256=str(evidence["original_terminal_state_sha256"]),
    )
    return {
        **closure,
        "protocol_amendment_sha256": worker.recovery_refs[
            "protocol_amendment"
        ].sha256,
        "authorization_sha256": worker.recovery_refs["authorization"].sha256,
        "active_pid_absence_evidence_sha256": active_pid_sha,
        "recovery_code_identity_evidence_status": (
            RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        ),
        "effect_fields_read": False,
    }


def _prepare_materialization(
    source_map_path: str | Path,
    output_root: str | Path,
    *,
    authoritative_real_profile: bool,
) -> MaterializationPlan:
    source_path = Path(source_map_path).absolute()
    if source_path.is_symlink() or not source_path.is_file():
        raise ConfigError("source map 缺失或为 symlink")
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_ref = ArtifactRef(
        path=source_path,
        size=len(source_bytes),
        sha256=source_sha,
    )
    source_map = _read_json_bytes(source_bytes, label="source map")
    _require_exact_keys(
        source_map,
        {
            "schema_version",
            "materialization_id",
            "protocol_id",
            "archives",
            "analysis",
            "workers",
        },
        label="source map",
    )
    if (
        source_map.get("schema_version") != SOURCE_MAP_SCHEMA
        or source_map.get("protocol_id") != PROTOCOL_ID
    ):
        raise ConfigError("source map schema/protocol 不匹配")
    materialization_id = _safe_identifier(
        source_map["materialization_id"],
        label="materialization_id",
    )
    authoritative_identity = (
        materialization_id == AUTHORITATIVE_REAL_MATERIALIZATION_ID
        and source_sha == AUTHORITATIVE_REAL_SOURCE_MAP_SHA256
    )
    if authoritative_real_profile and not authoritative_identity:
        raise ConfigError(
            "authoritative real profile 必须绑定批准的 v8 ID/SHA identity"
        )
    if (
        materialization_id == AUTHORITATIVE_REAL_MATERIALIZATION_ID
        and source_sha != AUTHORITATIVE_REAL_SOURCE_MAP_SHA256
    ):
        raise ConfigError("authoritative real source map SHA 非批准的 v8 identity")
    if authoritative_identity and not authoritative_real_profile:
        raise ConfigError(
            "authoritative real source map 禁止 generic materialization；"
            "必须显式启用 authoritative real profile"
        )
    base = source_path.parent
    archives = _load_archives(source_map["archives"], base=base)
    analysis, contract_ref, preunlock_ref = _parse_analysis(
        source_map["analysis"],
        base=base,
        archives=archives,
    )
    workers = _parse_workers(source_map["workers"], base=base, archives=archives)
    _validate_task_source_uniqueness(workers)
    if authoritative_real_profile:
        _validate_authoritative_real_selections(archives=archives, workers=workers)
        for worker in workers:
            if worker.worker_id in RECOVERED_WORKERS:
                _bind_authoritative_active_pid_evidence(
                    worker,
                    archives=archives,
                )
    destination = Path(output_root).absolute()
    partial = destination.parent / f"{destination.name}.partial-{source_sha[:12]}"
    if destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(
            f"canonical destination/partial 已存在，拒绝覆盖：{destination} / {partial}"
        )
    if not destination.parent.is_dir():
        raise ConfigError(f"canonical destination parent 不存在：{destination.parent}")
    requests = _build_copy_requests(
        source_map_ref=source_ref,
        contract_ref=contract_ref,
        preunlock_ref=preunlock_ref,
        archives=archives,
        workers=workers,
    )
    archive_copy_plan = _validate_archive_copy_plan(
        requests=requests,
        archives=archives,
        authoritative_real_profile=authoritative_real_profile,
    )
    return MaterializationPlan(
        source_path=source_path,
        source_ref=source_ref,
        source_sha256=source_sha,
        source_map=source_map,
        archives=archives,
        analysis=analysis,
        contract_ref=contract_ref,
        preunlock_ref=preunlock_ref,
        workers=workers,
        destination=destination,
        partial=partial,
        requests=requests,
        archive_copy_plan=archive_copy_plan,
    )


def _build_metadata_plan(
    plan: MaterializationPlan,
    *,
    authoritative_real_profile: bool,
) -> dict[str, Any]:
    inventory = _copy_target_inventory(plan.requests)
    _inventory_add(
        inventory,
        EXCLUSION_LEDGER_TARGET,
        EXCLUSION_LEDGER_BYTES,
    )
    manifests: list[dict[str, Any]] = []
    manifest_by_worker: dict[str, dict[str, Any]] = {}
    manifest_shas: dict[str, str] = {}
    worker_reports: list[dict[str, Any]] = []
    recovery_reports: list[dict[str, Any]] = []
    staged_bytes = 0
    staged_files = 0
    with tempfile.TemporaryDirectory(
        prefix=".task8b-metadata-dry-run-",
        dir=plan.destination.parent,
    ) as temporary:
        staging = Path(temporary)
        for worker in plan.workers:
            attempt_relative = PurePosixPath(
                f"snapshots/{worker.worker_id}/{worker.seed}/attempt_01"
            )
            attempt = staging.joinpath(*attempt_relative.parts)
            task_results: list[dict[str, Any]] = []
            for task_id in worker.expected_tasks:
                task = worker.task_rows[task_id]
                receipt_ref = task["receipt_ref"]
                receipt_payload = _read_ref_bytes(
                    receipt_ref,
                    archives=plan.archives,
                    label=f"{worker.worker_id}/{task_id} metadata task receipt",
                )
                receipt = _read_json_bytes(
                    receipt_payload,
                    label=f"{worker.worker_id}/{task_id} metadata task receipt",
                )
                task_results.append(
                    _validate_task_receipt_value(
                        worker,
                        task_id=task_id,
                        receipt=receipt,
                        actual_files=_planned_task_manifest(task),
                        base_manifest=worker.base_manifest,
                    )
                )
                receipt_target = (
                    attempt_relative / "task_receipts" / f"{task_id}.json"
                )
                staged_bytes += _stage_ref(
                    receipt_ref,
                    archives=plan.archives,
                    root=staging,
                    target=receipt_target,
                    label=f"{worker.worker_id}/{task_id} staged task receipt",
                )
                staged_files += 1
            for evidence in worker.evidence_targets.values():
                for kind, ref, target in evidence:
                    staged_bytes += _stage_ref(
                        ref,
                        archives=plan.archives,
                        root=staging,
                        target=target,
                        label=f"{worker.worker_id} staged {kind}",
                    )
                    staged_files += 1
            state_payload = _read_ref_bytes(
                worker.state_ref,
                archives=plan.archives,
                label=f"{worker.worker_id} metadata state prefix",
            )
            prefix_rows = _parse_state_bytes(
                state_payload,
                label=f"{worker.worker_id} metadata state prefix",
            )
            appended_rows = _build_state_rows(
                prefix_rows,
                worker.state_append,
                label=worker.worker_id,
            )
            _validate_recovery(
                worker,
                None,
                source_state_rows=prefix_rows,
                archives=plan.archives,
                metadata_only=True,
            )
            if authoritative_real_profile and worker.worker_id in RECOVERED_WORKERS:
                _validate_authoritative_recovery_state(
                    worker,
                    archives=plan.archives,
                    source_state_rows=prefix_rows,
                )
                recovery_reports.append(
                    _validate_authoritative_recovery_support(
                        worker,
                        archives=plan.archives,
                    )
                )
            source_identities = _validate_shard_identity_receipts(
                worker,
                root=staging,
                attempt=attempt,
                archives=plan.archives,
            )
            _validate_secondary_normalization_evidence(worker, root=staging)
            composition, lineage = _build_worker_composition_values(
                worker,
                root=staging,
                source_identities=source_identities,
            )
            composition_bytes = _json_bytes(composition)
            _inventory_add(
                inventory,
                PurePosixPath(f"manifests/composition/{worker.worker_id}.json"),
                composition_bytes,
            )
            augmented = {
                **copy.deepcopy(worker.base_manifest),
                "phase_f_shard_lineage": lineage,
            }
            manifest_bytes = _json_bytes(augmented)
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            _inventory_add(
                inventory,
                PurePosixPath(f"manifests/{worker.worker_id}.json"),
                manifest_bytes,
            )
            _inventory_add(
                inventory,
                attempt_relative / "worker_manifest.json",
                manifest_bytes,
            )
            task_results_bytes = _json_bytes(
                {
                    "schema_version": TASK_RESULTS_SCHEMA,
                    "worker_id": worker.worker_id,
                    "tasks": task_results,
                }
            )
            _inventory_add(
                inventory,
                attempt_relative / "task_results.json",
                task_results_bytes,
            )
            final_state_bytes = state_payload + _state_row_bytes(appended_rows)
            inventory[attempt_relative / "state.tsv"] = (
                len(final_state_bytes),
                hashlib.sha256(final_state_bytes).hexdigest(),
            )
            attempt_rows: list[tuple[str, int, str]] = []
            for path, (size, sha256) in inventory.items():
                try:
                    relative = path.relative_to(attempt_relative)
                except ValueError:
                    continue
                if relative.as_posix() in FILES_TSV_EXCLUSIONS:
                    continue
                attempt_rows.append((relative.as_posix(), size, sha256))
            files_tsv_bytes = _files_tsv_bytes(attempt_rows)
            _inventory_add(
                inventory,
                attempt_relative / "files.tsv",
                files_tsv_bytes,
            )
            completion_bytes = _json_bytes(
                {
                    "schema_version": COMPLETION_SCHEMA,
                    "worker_id": worker.worker_id,
                    "status": "complete",
                    "not_for_paper": False,
                    "files_tsv_sha256": hashlib.sha256(files_tsv_bytes).hexdigest(),
                }
            )
            _inventory_add(
                inventory,
                attempt_relative / "completion_receipt.json",
                completion_bytes,
            )
            manifests.append(augmented)
            manifest_by_worker[worker.worker_id] = augmented
            manifest_shas[worker.worker_id] = manifest_sha
            worker_reports.append(
                {
                    "worker_id": worker.worker_id,
                    "role": worker.role,
                    "seed": worker.seed,
                    "pair_id": worker.pair_id,
                    "task_count": len(worker.expected_tasks),
                    "actual_hands": sum(
                        PLANNED_HANDS[task_id] for task_id in worker.expected_tasks
                    ),
                    "final_state": "complete",
                    "manifest_sha256": manifest_sha,
                    "composition_receipt_sha256": hashlib.sha256(
                        composition_bytes
                    ).hexdigest(),
                }
            )
        try:
            validate_worker_manifest_set(
                [copy.deepcopy(manifest) for manifest in manifests],
                expected_seed_count=6,
            )
        except ConfigError as exc:
            raise ConfigError(f"metadata six-seed manifest set 非法：{exc}") from exc
        index = _build_manifest_index(
            manifests,
            manifest_shas,
            contract_sha256=str(plan.analysis["analysis_contract_sha256"]),
        )
        index_bytes = _json_bytes(index)
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        _inventory_add(
            inventory,
            PurePosixPath("manifests/manifest_index.json"),
            index_bytes,
        )
        analysis_input = _build_analysis_input(
            workers=plan.workers,
            manifests=manifest_by_worker,
            manifest_shas=manifest_shas,
            analysis=plan.analysis,
            manifest_index_sha256=index_sha,
        )
        analysis_input_bytes = _json_bytes(analysis_input)
        _inventory_add(
            inventory,
            PurePosixPath("analysis_input.json"),
            analysis_input_bytes,
        )
    expected_bytes = sum(size for size, _sha256 in inventory.values())
    expected_file_count = len(inventory)
    if _sha256_file(plan.source_path) != plan.source_sha256:
        raise ConfigError("source map 在 metadata preflight 期间发生变化")
    for source_id, catalog in plan.archives.items():
        _validate_file_identity(
            catalog.path,
            size=catalog.size,
            sha256=catalog.sha256,
            label=f"archive {source_id} post-metadata-preflight",
        )
    validated_local_refs: set[tuple[str, ...]] = set()
    for request in plan.requests:
        ref = request.source
        if ref.path is None or ref.key in validated_local_refs:
            continue
        _validate_file_identity(
            ref.path,
            size=ref.size,
            sha256=ref.sha256,
            label=f"local source post-metadata-preflight {ref.path}",
        )
        validated_local_refs.add(ref.key)
    if (
        plan.destination.exists()
        or plan.destination.is_symlink()
        or plan.partial.exists()
        or plan.partial.is_symlink()
    ):
        raise FileExistsError(
            "canonical destination/partial 在 metadata preflight 期间出现，拒绝继续"
        )
    free_bytes = shutil.disk_usage(plan.destination.parent).free
    safety_reserve = max(2 * 1024**3, expected_bytes // 20)
    if free_bytes < expected_bytes + safety_reserve:
        raise ConfigError(
            "canonical destination volume 空间不足："
            f"free={free_bytes} expected={expected_bytes} reserve={safety_reserve}"
        )
    provenance_targets = [
        target.as_posix()
        for request in plan.requests
        for target in request.targets
        if target.parts[:2]
        in {
            ("provenance", "archive_manifests"),
            ("provenance", "archive_receipts"),
            ("provenance", "archive_seals"),
        }
    ]
    if len(provenance_targets) != len(plan.archives) * 3:
        raise ConfigError("archive manifest/receipt/seal provenance plan 不闭合")
    return {
        "expected_file_count": expected_file_count,
        "expected_bytes": expected_bytes,
        "free_bytes": free_bytes,
        "safety_reserve_bytes": safety_reserve,
        "copied_target_count": sum(len(request.targets) for request in plan.requests),
        "copied_target_bytes": sum(
            request.source.size * len(request.targets) for request in plan.requests
        ),
        "staged_metadata_file_count": staged_files,
        "staged_metadata_bytes": staged_bytes,
        "manifest_index_sha256": index_sha,
        "analysis_input_sha256": hashlib.sha256(analysis_input_bytes).hexdigest(),
        "workers": worker_reports,
        "recovery_closure": recovery_reports,
        "provenance_targets": sorted(provenance_targets),
        "file_plan": [
            {
                "relative_path": path.as_posix(),
                "size": size,
                "sha256": sha256,
            }
            for path, (size, sha256) in sorted(
                inventory.items(),
                key=lambda item: item[0].as_posix(),
            )
        ],
    }


def metadata_only_dry_run_task8b_phase_f(
    source_map_path: str | Path,
    output_root: str | Path,
    report_path: str | Path,
    *,
    authoritative_real_profile: bool = False,
) -> dict[str, Any]:
    """Validate the exact production plan without copying formal task payloads."""

    report = Path(report_path).absolute()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"dry-run report 已存在，拒绝覆盖：{report}")
    if not report.parent.is_dir():
        raise ConfigError(f"dry-run report parent 不存在：{report.parent}")
    plan = _prepare_materialization(
        source_map_path,
        output_root,
        authoritative_real_profile=authoritative_real_profile,
    )
    metadata = _build_metadata_plan(
        plan,
        authoritative_real_profile=authoritative_real_profile,
    )
    selection_rows = (
        _validate_authoritative_real_selections(
            archives=plan.archives,
            workers=plan.workers,
        )
        if authoritative_real_profile
        else []
    )
    value = {
        "schema_version": "task8b-phase-f-metadata-dry-run-v1",
        "status": "pass",
        "gaps": [],
        "authoritative_real_profile": authoritative_real_profile,
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
        "authoritative_selections": selection_rows,
        **metadata,
        "metadata_only": True,
        "large_payload_files_written": 0,
        "large_payload_bytes_written": 0,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "limitations": _declared_limitations(),
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
    }
    _write_json_new(report, value)
    return {
        "status": "pass",
        "report_path": str(report),
        "report_sha256": _sha256_file(report),
        "source_map_sha256": plan.source_sha256,
        "worker_count": len(plan.workers),
        "task_count": 60,
        "expected_file_count": metadata["expected_file_count"],
        "expected_bytes": metadata["expected_bytes"],
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
    }


def create_pre_materialization_receipt(
    source_map_path: str | Path,
    output_root: str | Path,
    dry_run_report_path: str | Path,
    receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Create one append-only authorization receipt from a passed dry-run."""

    source_path = Path(source_map_path).absolute()
    report_path = Path(dry_run_report_path).absolute()
    receipt_path = Path(receipt_output_path).absolute()
    destination = Path(output_root).absolute()
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or report_path.is_symlink()
        or not report_path.is_file()
    ):
        raise ConfigError("pre-materialization source map/report 缺失或为 symlink")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(
            f"pre-materialization receipt 已存在，拒绝覆盖：{receipt_path}"
        )
    if not receipt_path.parent.is_dir():
        raise ConfigError(
            f"pre-materialization receipt parent 不存在：{receipt_path.parent}"
        )
    source_sha = _sha256_file(source_path)
    report_sha = _sha256_file(report_path)
    report = _read_json_bytes(
        report_path.read_bytes(),
        label="metadata dry-run report for receipt",
    )
    expected_gates = [
        {"gate": name, "status": "pass"}
        for name in METADATA_DRY_RUN_GATE_NAMES
    ]
    if (
        set(report) != METADATA_DRY_RUN_REPORT_FIELDS
        or report.get("schema_version")
        != "task8b-phase-f-metadata-dry-run-v1"
        or report.get("status") != "pass"
        or report.get("gaps") != []
        or report.get("authoritative_real_profile") is not True
        or report.get("source_map_path") != str(source_path)
        or report.get("source_map_sha256") != source_sha
        or source_sha != AUTHORITATIVE_REAL_SOURCE_MAP_SHA256
        or report.get("materialization_id")
        != AUTHORITATIVE_REAL_MATERIALIZATION_ID
        or report.get("candidate_output_root") != str(destination)
        or report.get("worker_count") != 12
        or report.get("task_count") != 60
        or report.get("pair_count") != 6
        or report.get("gates") != expected_gates
        or report.get("metadata_only") is not True
        or report.get("large_payload_files_written") != 0
        or report.get("large_payload_bytes_written") != 0
        or report.get("effect_fields_read") is not False
        or report.get("formal_effect_fields_read") is not False
        or report.get("limitations") != _declared_limitations()
        or report.get("paper_eligibility") != CANONICAL_PAPER_ELIGIBILITY
        or isinstance(report.get("expected_file_count"), bool)
        or not isinstance(report.get("expected_file_count"), int)
        or isinstance(report.get("expected_bytes"), bool)
        or not isinstance(report.get("expected_bytes"), int)
    ):
        raise ConfigError("metadata dry-run report 不满足 receipt 生成门禁")
    git_identity = _materializer_git_identity()
    if (
        git_identity["origin_main_sha"] != git_identity["head_sha"]
        or git_identity["tracked_clean"] is not True
        or git_identity["working_blob_sha"] != git_identity["head_blob_sha"]
    ):
        raise ConfigError("materializer 必须 pushed、tracked-clean 且 blob 闭合")
    receipt = {
        "schema_version": "task8b-phase-f-pre-materialization-receipt-v1",
        "status": "approved",
        "materializer_code_sha": git_identity["head_sha"],
        "materializer_file_sha256": _sha256_file(Path(__file__).absolute()),
        "source_map_sha256": source_sha,
        "analysis_contract_sha256": report["analysis_contract_sha256"],
        "pre_unlock_manifest_sha256": report[
            "pre_unlock_manifest_sha256"
        ],
        "worker_count": 12,
        "task_count": 60,
        "expected_file_count": report["expected_file_count"],
        "expected_bytes": report["expected_bytes"],
        "dry_run_report_path": str(report_path),
        "dry_run_report_sha256": report_sha,
        "authoritative_real_profile": True,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "recovery_code_identity_evidence_status": (
            RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        ),
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
    }
    _write_json_new(receipt_path, receipt)
    return {
        "status": "approved",
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "materializer_code_sha": git_identity["head_sha"],
        "source_map_sha256": source_sha,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
    }


def _validate_pre_materialization_receipt(
    receipt_path: str | Path,
    *,
    plan: MaterializationPlan,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(receipt_path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ConfigError("pre-materialization receipt 缺失或为 symlink")
    receipt = _read_json_bytes(path.read_bytes(), label="pre-materialization receipt")
    expected_keys = {
        "schema_version",
        "status",
        "materializer_code_sha",
        "materializer_file_sha256",
        "source_map_sha256",
        "analysis_contract_sha256",
        "pre_unlock_manifest_sha256",
        "worker_count",
        "task_count",
        "expected_file_count",
        "expected_bytes",
        "dry_run_report_path",
        "dry_run_report_sha256",
        "authoritative_real_profile",
        "effect_fields_read",
        "formal_effect_fields_read",
        "recovery_code_identity_evidence_status",
        "paper_eligibility",
    }
    if set(receipt) != expected_keys:
        raise ConfigError("pre-materialization receipt keys 非法")
    report_path = Path(str(receipt["dry_run_report_path"])).absolute()
    if report_path.is_symlink() or not report_path.is_file():
        raise ConfigError("pre-materialization dry-run report 缺失或为 symlink")
    report_sha = _sha256_file(report_path)
    report = _read_json_bytes(report_path.read_bytes(), label="metadata dry-run report")
    materializer_file_sha = _sha256_file(Path(__file__).absolute())
    git_identity = _materializer_git_identity()
    expected_gates = [
        {"gate": name, "status": "pass"} for name in METADATA_DRY_RUN_GATE_NAMES
    ]
    expected_report_bindings = {
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
        "archive_copy_plan": plan.archive_copy_plan,
        "authoritative_selections": _validate_authoritative_real_selections(
            archives=plan.archives,
            workers=plan.workers,
        ),
        **{
            key: metadata[key]
            for key in METADATA_DRY_RUN_PLAN_BINDING_FIELDS
        },
    }
    report_free_bytes = report.get("free_bytes")
    report_reserve_bytes = report.get("safety_reserve_bytes")
    if (
        receipt.get("schema_version")
        != "task8b-phase-f-pre-materialization-receipt-v1"
        or receipt.get("status") != "approved"
        or receipt.get("materializer_code_sha") != git_identity["head_sha"]
        or git_identity["origin_main_sha"] != git_identity["head_sha"]
        or git_identity["tracked_clean"] is not True
        or git_identity["working_blob_sha"] != git_identity["head_blob_sha"]
        or receipt.get("materializer_file_sha256") != materializer_file_sha
        or receipt.get("source_map_sha256") != plan.source_sha256
        or receipt.get("analysis_contract_sha256")
        != plan.analysis["analysis_contract_sha256"]
        or receipt.get("pre_unlock_manifest_sha256")
        != plan.analysis["pre_unlock_manifest_sha256"]
        or receipt.get("worker_count") != 12
        or receipt.get("task_count") != 60
        or receipt.get("expected_file_count") != metadata["expected_file_count"]
        or receipt.get("expected_bytes") != metadata["expected_bytes"]
        or receipt.get("dry_run_report_sha256") != report_sha
        or receipt.get("authoritative_real_profile") is not True
        or receipt.get("effect_fields_read") is not False
        or receipt.get("formal_effect_fields_read") is not False
        or receipt.get("recovery_code_identity_evidence_status")
        != RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        or receipt.get("paper_eligibility")
        != CANONICAL_PAPER_ELIGIBILITY
        or report.get("status") != "pass"
        or report.get("schema_version")
        != "task8b-phase-f-metadata-dry-run-v1"
        or report.get("gaps") != []
        or report.get("authoritative_real_profile") is not True
        or set(report) != METADATA_DRY_RUN_REPORT_FIELDS
        or report.get("gates") != expected_gates
        or any(
            report.get(key) != expected
            for key, expected in expected_report_bindings.items()
        )
        or report.get("metadata_only") is not True
        or report.get("large_payload_files_written") != 0
        or report.get("large_payload_bytes_written") != 0
        or not isinstance(report_free_bytes, int)
        or isinstance(report_free_bytes, bool)
        or not isinstance(report_reserve_bytes, int)
        or isinstance(report_reserve_bytes, bool)
        or report_free_bytes
        < metadata["expected_bytes"] + report_reserve_bytes
        or report.get("effect_fields_read") is not False
        or report.get("formal_effect_fields_read") is not False
        or report.get("limitations") != _declared_limitations()
        or report.get("paper_eligibility") != CANONICAL_PAPER_ELIGIBILITY
    ):
        raise ConfigError("pre-materialization receipt/dry-run report binding 非法")
    return receipt


def _freeze_pre_materialization_evidence(
    receipt_path: str | Path,
    *,
    receipt: Mapping[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    source_receipt = Path(receipt_path).absolute()
    receipt_bytes = source_receipt.read_bytes()
    if (
        _read_json_bytes(
            receipt_bytes,
            label="pre-materialization receipt freeze",
        )
        != dict(receipt)
    ):
        raise ConfigError("pre-materialization receipt 在验证后发生变化")
    source_report = Path(str(receipt["dry_run_report_path"])).absolute()
    report_bytes = source_report.read_bytes()
    if (
        hashlib.sha256(report_bytes).hexdigest()
        != receipt["dry_run_report_sha256"]
    ):
        raise ConfigError("metadata dry-run report 在验证后发生变化")
    targets = (
        (PRE_MATERIALIZATION_REPORT_TARGET, report_bytes),
        (PRE_MATERIALIZATION_RECEIPT_TARGET, receipt_bytes),
    )
    rows = []
    for relative, payload in targets:
        _write_new(root.joinpath(*relative.parts), payload)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _canonical_manifest_tsv_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=["relative_path", "size", "sha256"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "relative_path": row["relative_path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
        )
    return handle.getvalue().encode("utf-8")


def _write_canonical_root_closure(
    plan: MaterializationPlan,
    *,
    root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    materializer_code_sha: str,
    authoritative_real_profile: bool,
) -> dict[str, Any]:
    reserved = {
        CANONICAL_FILE_MANIFEST_TARGET.as_posix(),
        CANONICAL_ROOT_SEAL_TARGET.as_posix(),
    }
    normalized_rows = sorted(
        (dict(row) for row in manifest_rows),
        key=lambda row: str(row["relative_path"]),
    )
    if (
        len({str(row["relative_path"]) for row in normalized_rows})
        != len(normalized_rows)
        or any(str(row["relative_path"]) in reserved for row in normalized_rows)
    ):
        raise ConfigError("canonical root manifest path 集合非法")
    manifest_bytes = _canonical_manifest_tsv_bytes(normalized_rows)
    manifest_path = root.joinpath(*CANONICAL_FILE_MANIFEST_TARGET.parts)
    _write_new(manifest_path, manifest_bytes)
    seal = {
        "schema_version": "task8b-six-seed-canonical-root-seal-v1",
        "status": "sealed_with_declared_limitation",
        "materialization_id": plan.source_map["materialization_id"],
        "source_map_sha256": plan.source_sha256,
        "analysis_contract_sha256": plan.analysis["analysis_contract_sha256"],
        "analysis_freeze_sha256": plan.analysis[
            "pre_unlock_manifest_sha256"
        ],
        "materializer_code_sha": materializer_code_sha,
        "materializer_file_sha256": _sha256_file(Path(__file__).absolute()),
        "authoritative_real_profile": authoritative_real_profile,
        "canonical_file_manifest": {
            "relative_path": CANONICAL_FILE_MANIFEST_TARGET.as_posix(),
            "size": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "manifest_entry_count": len(normalized_rows),
        "manifest_entry_bytes": sum(int(row["size"]) for row in normalized_rows),
        "published_tree_file_count": len(normalized_rows) + 2,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
        "recovery_code_identity_evidence_status": (
            RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        ),
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
        "limitations": _declared_limitations(),
    }
    seal_path = root.joinpath(*CANONICAL_ROOT_SEAL_TARGET.parts)
    _write_json_new(seal_path, seal)
    return {
        "canonical_file_manifest_sha256": _sha256_file(manifest_path),
        "canonical_root_seal_sha256": _sha256_file(seal_path),
        "manifest_entry_count": len(normalized_rows),
        "manifest_entry_bytes": sum(int(row["size"]) for row in normalized_rows),
    }


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one directory while refusing every existing target."""

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
        move_file_ex.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        move_file_ex.restype = ctypes.c_int
        movefile_write_through = 0x00000008
        if move_file_ex(
            str(source),
            str(destination),
            movefile_write_through,
        ):
            return
        error = ctypes.get_last_error()
        if destination.exists() or destination.is_symlink() or error in {80, 183}:
            raise FileExistsError(error, "atomic publish target exists", destination)
        if error in {5, 32, 33}:
            raise PermissionError(error, "atomic publish source is transiently busy", source)
        raise OSError(error, "MoveFileExW no-replace publish failed", source)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOTSUP,
                "renameat2(RENAME_NOREPLACE) unavailable; refusing unsafe publish",
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, "atomic publish target exists", destination)
        raise OSError(error, os.strerror(error), source)
    raise OSError(
        errno.ENOTSUP,
        "platform lacks a configured atomic no-replace directory rename",
    )


def _publish_staging_no_overwrite(staging: Path, destination: Path) -> None:
    """Atomically rename a complete staging tree under an exclusive publish lock."""

    publish_lock = destination.parent / f".{destination.name}.publish.lock"
    with publish_lock.open("xb") as handle:
        handle.write(b"TASK8B Phase F append-only publish lock\n")
    try:
        # The exclusive sibling lock serializes publishers; each immediate
        # existence check makes replacement ineligible.  Windows antivirus can
        # transiently hold a just-written directory, so retry only that exact
        # PermissionError while the destination remains absent.
        delays = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
        for index, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"publish target 已存在，拒绝覆盖：{destination}")
            try:
                # Same-volume whole-tree no-replace rename is atomic, so no
                # partially published canonical directory can be observed.
                _atomic_rename_noreplace(staging, destination)
                break
            except PermissionError:
                if index == len(delays) - 1:
                    raise
    finally:
        publish_lock.unlink()


def materialize_task8b_phase_f(
    source_map_path: str | Path,
    output_root: str | Path,
    *,
    authoritative_real_profile: bool = False,
    pre_materialization_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize one append-only, SHA-closed TASK8B six-seed Phase F tree."""

    plan = _prepare_materialization(
        source_map_path,
        output_root,
        authoritative_real_profile=authoritative_real_profile,
    )
    metadata = _build_metadata_plan(
        plan,
        authoritative_real_profile=authoritative_real_profile,
    )
    validated_receipt: dict[str, Any] | None = None
    if authoritative_real_profile:
        if pre_materialization_receipt_path is None:
            raise ConfigError(
                "authoritative real materialization 必须提供 pre-materialization receipt"
            )
        validated_receipt = _validate_pre_materialization_receipt(
            pre_materialization_receipt_path,
            plan=plan,
            metadata=metadata,
        )
    elif pre_materialization_receipt_path is not None:
        raise ConfigError("generic materialization 不得使用 authoritative receipt")
    plan.partial.mkdir(parents=True, exist_ok=False)
    _execute_copy_requests(
        requests=plan.requests,
        archives=plan.archives,
        root=plan.partial,
    )
    _write_new(
        plan.partial.joinpath(*EXCLUSION_LEDGER_TARGET.parts),
        EXCLUSION_LEDGER_BYTES,
    )
    pre_materialization_rows: list[dict[str, Any]] = []
    if validated_receipt is not None:
        assert pre_materialization_receipt_path is not None
        pre_materialization_rows = _freeze_pre_materialization_evidence(
            pre_materialization_receipt_path,
            receipt=validated_receipt,
            root=plan.partial,
        )
    manifests: list[dict[str, Any]] = []
    manifest_by_worker: dict[str, dict[str, Any]] = {}
    manifest_shas: dict[str, str] = {}
    for worker in plan.workers:
        manifest, _lineage, manifest_sha = _materialize_worker(
            worker,
            root=plan.partial,
            archives=plan.archives,
        )
        manifests.append(manifest)
        manifest_by_worker[worker.worker_id] = manifest
        manifest_shas[worker.worker_id] = manifest_sha
    try:
        validate_worker_manifest_set(
            [copy.deepcopy(manifest) for manifest in manifests],
            expected_seed_count=6,
        )
    except ConfigError as exc:
        raise ConfigError(f"canonical six-seed manifest set 非法：{exc}") from exc
    index = _build_manifest_index(
        manifests,
        manifest_shas,
        contract_sha256=str(plan.analysis["analysis_contract_sha256"]),
    )
    _write_json_new(plan.partial / "manifests" / "manifest_index.json", index)
    manifest_index_sha256 = _sha256_file(
        plan.partial / "manifests" / "manifest_index.json"
    )
    analysis_input = _build_analysis_input(
        workers=plan.workers,
        manifests=manifest_by_worker,
        manifest_shas=manifest_shas,
        analysis=plan.analysis,
        manifest_index_sha256=manifest_index_sha256,
    )
    _write_json_new(plan.partial / "analysis_input.json", analysis_input)
    if _sha256_file(plan.source_path) != plan.source_sha256:
        raise ConfigError("source map 在 materialization 期间发生变化")
    actual_manifest_before_closure = _directory_manifest(plan.partial)
    expected_before_closure = sorted(
        [
            *metadata["file_plan"],
            *pre_materialization_rows,
        ],
        key=lambda row: str(row["relative_path"]),
    )
    if (
        actual_manifest_before_closure != expected_before_closure
    ):
        raise ConfigError("canonical materialization 与 metadata dry-run file plan 不一致")
    materializer_code_sha = "UNBOUND_GENERIC_PROFILE"
    if authoritative_real_profile:
        git_identity = _materializer_git_identity()
        if (
            validated_receipt is None
            or git_identity["head_sha"]
            != validated_receipt["materializer_code_sha"]
            or git_identity["origin_main_sha"] != git_identity["head_sha"]
            or git_identity["tracked_clean"] is not True
            or git_identity["working_blob_sha"] != git_identity["head_blob_sha"]
        ):
            raise ConfigError(
                "materializer git identity 在 materialization 期间发生变化"
            )
        materializer_code_sha = str(git_identity["head_sha"])
    closure = _write_canonical_root_closure(
        plan,
        root=plan.partial,
        manifest_rows=actual_manifest_before_closure,
        materializer_code_sha=materializer_code_sha,
        authoritative_real_profile=authoritative_real_profile,
    )
    actual_manifest = _directory_manifest(plan.partial)
    final_by_path = {
        str(row["relative_path"]): row for row in actual_manifest
    }
    closure_paths = {
        CANONICAL_FILE_MANIFEST_TARGET.as_posix(),
        CANONICAL_ROOT_SEAL_TARGET.as_posix(),
    }
    if (
        len(actual_manifest) != len(actual_manifest_before_closure) + 2
        or closure["manifest_entry_count"]
        != len(actual_manifest_before_closure)
        or closure["manifest_entry_bytes"]
        != sum(int(row["size"]) for row in actual_manifest_before_closure)
        or not closure_paths.issubset(final_by_path)
        or final_by_path[CANONICAL_FILE_MANIFEST_TARGET.as_posix()][
            "sha256"
        ]
        != closure["canonical_file_manifest_sha256"]
        or final_by_path[CANONICAL_ROOT_SEAL_TARGET.as_posix()]["sha256"]
        != closure["canonical_root_seal_sha256"]
    ):
        raise ConfigError("canonical root manifest/seal closure 非法")
    if plan.destination.exists() or plan.destination.is_symlink():
        raise FileExistsError(
            f"canonical destination 在发布前出现，拒绝覆盖：{plan.destination}"
        )
    _publish_staging_no_overwrite(plan.partial, plan.destination)
    return {
        "status": "complete",
        "output_root": str(plan.destination),
        "source_map_sha256": plan.source_sha256,
        "manifest_index_sha256": _sha256_file(
            plan.destination / "manifests" / "manifest_index.json"
        ),
        "analysis_input_sha256": _sha256_file(
            plan.destination / "analysis_input.json"
        ),
        "worker_count": len(plan.workers),
        "task_count": 60,
        "file_count": len(actual_manifest),
        "byte_count": sum(int(row["size"]) for row in actual_manifest),
        "canonical_file_manifest_sha256": closure[
            "canonical_file_manifest_sha256"
        ],
        "canonical_root_seal_sha256": closure[
            "canonical_root_seal_sha256"
        ],
        "canonical_status": "sealed_with_declared_limitation",
        "paper_eligibility": CANONICAL_PAPER_ELIGIBILITY,
        "recovery_code_identity_evidence_status": (
            RECOVERY_CODE_IDENTITY_EVIDENCE_STATUS
        ),
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append-only TASK8B six-seed Phase F canonical materializer"
    )
    parser.add_argument("--source-map", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--metadata-only-dry-run", action="store_true")
    parser.add_argument("--dry-run-report", type=Path)
    parser.add_argument(
        "--create-pre-materialization-receipt",
        action="store_true",
    )
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--authoritative-real-profile", action="store_true")
    parser.add_argument("--pre-materialization-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.create_pre_materialization_receipt:
        if (
            args.metadata_only_dry_run
            or args.dry_run_report is None
            or args.receipt_output is None
            or not args.authoritative_real_profile
            or args.pre_materialization_receipt is not None
        ):
            parser.error(
                "receipt creation requires --dry-run-report, "
                "--receipt-output and --authoritative-real-profile only"
            )
        result = create_pre_materialization_receipt(
            args.source_map,
            args.output_root,
            args.dry_run_report,
            args.receipt_output,
        )
    elif args.metadata_only_dry_run:
        if args.dry_run_report is None:
            parser.error("--metadata-only-dry-run requires --dry-run-report")
        if (
            args.pre_materialization_receipt is not None
            or args.receipt_output is not None
        ):
            parser.error(
                "metadata dry-run does not accept receipt input/output"
            )
        result = metadata_only_dry_run_task8b_phase_f(
            args.source_map,
            args.output_root,
            args.dry_run_report,
            authoritative_real_profile=args.authoritative_real_profile,
        )
    else:
        if args.dry_run_report is not None or args.receipt_output is not None:
            parser.error(
                "--dry-run-report/--receipt-output require a dry-run mode"
            )
        result = materialize_task8b_phase_f(
            args.source_map,
            args.output_root,
            authoritative_real_profile=args.authoritative_real_profile,
            pre_materialization_receipt_path=args.pre_materialization_receipt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
