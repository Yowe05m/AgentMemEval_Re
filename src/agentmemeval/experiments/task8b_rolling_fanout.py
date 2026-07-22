"""Fail-closed, effect-blind rolling TASK8B primary-to-secondary handoff.

The controller has three deliberately separate stages:

1. ``build_release_draft`` runs on the worker host after the producer exits and
   validates completion, the checkpoint receipt, archive receipt, identities,
   runtime health evidence, concurrency evidence, and fresh consumer paths.
2. ``activate_release`` runs only after the archive has been copied locally. It
   verifies the downloaded archive and every archived file by performing a new
   extraction, then publishes a new activated authorization bound to the draft.
3. ``launch_secondary`` runs back on the worker host. It rebuilds the draft,
   requires the activated authorization to match exactly, obtains an exclusive
   append-only claim, and starts the frozen ``formal-worker`` without a shell or
   ``--resume-existing``.

No function reads rewards, effect estimates, p-values, or confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentmemeval.experiments.formal_runner import (
    _receipt_identity,
    validate_worker_manifest,
    verify_checkpoint_receipt,
)
from agentmemeval.experiments.task8b_transport import validate_worker_for_archive
from agentmemeval.storage.snapshot_archive import (
    extract_snapshot_archive,
    verify_archive_checksum,
)

FROZEN_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
FROZEN_RUNNER_SHA256 = "c4b601ff0de2c27a57ee246efcf91d21f502f27c652d20fd6fa7cfd925a17d5e"
FROZEN_CLI_MAIN_SHA256 = "6d2406783862cb740b447801fe9e0ae67d949f1f0ddeee3e690301cb383831d1"
FANOUT_PLAN_SHA256 = "5fb1e429491fdc3c4a5e9e68c55c653bb6f42a8054deb7be498365c991d76f27"
CURRENT_OPS_MANIFEST_SHA256 = (
    "7ce4565fd23eacba386a38ec3d410917bb9386bd2d65b391cb78bc8de21df5de"
)
RELEASE_SCHEMA = "task8b-rolling-fanout-release-v1"
ACTIVATION_SCHEMA = "task8b-rolling-fanout-activation-v1"
CLAIM_SCHEMA = "task8b-rolling-fanout-claim-v1"
LAUNCHED_SCHEMA = "task8b-rolling-fanout-launched-v1"
FAILED_SCHEMA = "task8b-rolling-fanout-failed-v1"
PROCESS_EVIDENCE_SCHEMA = "task8b-producer-process-absence-v1"
HEALTH_EVIDENCE_SCHEMA = "task8b-fanout-runtime-health-v1"
CONCURRENCY_EVIDENCE_SCHEMA = "task8b-fanout-concurrency-v1"
MAX_ACTIVE_PHYSICAL_WORKERS = 12
ACTIVATION_SIGNER_IDENTITY = "task8b-local-auditor"
ACTIVATION_SIGNATURE_NAMESPACE = "task8b-fanout"
ALLOWED_SIGNERS_SHA256 = "575543fab72620d25e4d201d0fd9ccc12353622fe7a79e5173835bfe4471850e"
FROZEN_SECONDARY_LAUNCHER_SHA256 = (
    "cfa048f5d3f818a325cf4cc5eeb03074ca58b82803d3798250093e6b1208c77a"
)


class RollingFanoutError(RuntimeError):
    """Fail-closed rolling handoff error."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_source(path: Path) -> str:
    """Hash UTF-8 source with Git-style LF normalization across Windows/Linux."""

    try:
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RollingFanoutError(f"unable to hash source file: {path}") from exc
    return _sha256_bytes(normalized)


def _sha256_json(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(data)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RollingFanoutError(f"JSON evidence missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RollingFanoutError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RollingFanoutError(f"JSON evidence root must be an object: {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RollingFanoutError(f"refuse to overwrite append-only evidence: {path}") from exc


def _publish_json_atomic_new(path: Path, value: dict[str, Any], staging_root: Path) -> None:
    payload = _json_bytes(value)
    staging = staging_root.resolve() / _sha256_bytes(payload)
    _write_json_new(staging, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staging, path)
    except FileExistsError as exc:
        raise RollingFanoutError(f"refuse to overwrite append-only terminal: {path}") from exc
    except OSError as exc:
        raise RollingFanoutError(f"atomic terminal publication failed: {path}") from exc


def _verify_activation_signature(
    *, activated_path: Path, signature_path: Path, allowed_signers_path: Path
) -> None:
    for label, path in (
        ("activated authorization", activated_path),
        ("activation signature", signature_path),
        ("allowed signers", allowed_signers_path),
    ):
        if not path.is_file() or path.is_symlink():
            raise RollingFanoutError(f"{label} missing or symlinked")
    if _sha256_source(allowed_signers_path) != ALLOWED_SIGNERS_SHA256:
        raise RollingFanoutError("allowed-signers SHA-256 mismatch")
    try:
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_signers_path.resolve()),
                "-I",
                ACTIVATION_SIGNER_IDENTITY,
                "-n",
                ACTIVATION_SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path.resolve()),
            ],
            input=activated_path.read_bytes(),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RollingFanoutError("activation signature verification failed") from exc


def _inside(root: Path, relative: str, *, label: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise RollingFanoutError(f"unsafe {label} relative path: {relative!r}")
    candidate = (root.resolve() / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RollingFanoutError(f"{label} escapes root: {relative!r}") from exc
    return candidate


def _read_plan_row(plan_path: Path, producer_worker_id: str) -> dict[str, str]:
    if not plan_path.is_file() or plan_path.is_symlink():
        raise RollingFanoutError("fanout plan missing or symlinked")
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("producer_worker") == producer_worker_id]
    if len(matches) != 1:
        raise RollingFanoutError("fanout plan must contain exactly one producer row")
    row = matches[0]
    required = {
        "pod_id",
        "seed",
        "physical_slot",
        "producer_worker",
        "consumer_worker",
        "scientific_checkout",
        "producer_manifest_sha256",
        "consumer_manifest_sha256",
        "producer_output_path",
        "producer_receipt_path",
        "consumer_output_path",
        "consumer_cache_namespace",
    }
    missing = sorted(required - set(row))
    if missing or any(not str(row.get(field, "")).strip() for field in required):
        raise RollingFanoutError(f"fanout plan row incomplete: {', '.join(missing)}")
    if row.get("recorded_before_effect_unblind") != "true":
        raise RollingFanoutError("fanout plan is not recorded before effect unblind")
    if row.get("effect_metrics_read") != "false":
        raise RollingFanoutError("fanout plan effect-metrics gate failed")
    return row


def _verify_frozen_checkout(checkout: Path) -> None:
    runner_path = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    cli_main_path = checkout / "src" / "agentmemeval" / "cli" / "main.py"
    for label, path, expected_sha in (
        ("formal runner", runner_path, FROZEN_RUNNER_SHA256),
        ("CLI main", cli_main_path, FROZEN_CLI_MAIN_SHA256),
    ):
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_sha:
            raise RollingFanoutError(f"frozen {label} source SHA-256 mismatch")
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RollingFanoutError("unable to verify frozen scientific checkout") from exc
    if head != FROZEN_CODE_SHA or tracked_dirty:
        raise RollingFanoutError("scientific checkout SHA or tracked-clean gate failed")


def _load_manifest(path: Path, *, worker_id: str, role: str, expected_sha: str) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha:
        raise RollingFanoutError(f"{worker_id} manifest SHA-256 mismatch")
    manifest = validate_worker_manifest(_read_json(path))
    if manifest.get("worker_id") != worker_id or manifest.get("role") != role:
        raise RollingFanoutError(f"{worker_id} manifest identity mismatch")
    return manifest


def receipt_root_for_row(row: dict[str, str], bundle_root: Path) -> Path:
    """Return the exact receipt root, including the P12 bundle-root exception."""

    producer = row["producer_worker"]
    expected = (
        bundle_root / "receipts" / f"{producer}.json"
        if producer == "P12"
        else bundle_root / "receipt_root" / "receipts" / f"{producer}.json"
    ).resolve()
    actual = Path(row["producer_receipt_path"]).resolve()
    if actual != expected:
        raise RollingFanoutError(f"{producer} receipt-root binding mismatch")
    return bundle_root.resolve() if producer == "P12" else (bundle_root / "receipt_root").resolve()


def _authoritative_control_root(bundle_root: Path) -> Path:
    return (bundle_root.resolve() / "control" / "task8b" / "rolling_fanout").resolve()


def build_launch_argv(
    python_path: Path,
    launcher_path: Path,
    checkout: Path,
    consumer_manifest: Path,
    receipt_root: Path,
) -> list[str]:
    """Build the exact fresh-secondary argv; never returns shell text."""

    return [
        str(python_path.resolve()),
        str(launcher_path.resolve()),
        "--checkout",
        str(checkout.resolve()),
        "--manifest",
        str(consumer_manifest.resolve()),
        "--receipt-root",
        str(receipt_root.resolve()),
    ]


def _assert_fresh_consumer_paths(checkout: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    output = _inside(
        checkout,
        str(manifest["instance_identity"]["output_path"]),
        label="consumer output",
    )
    cache = _inside(
        checkout,
        str(manifest["instance_identity"]["cache_namespace"]),
        label="consumer cache",
    )
    for label, path in (("consumer output", output), ("consumer cache", cache)):
        if path.exists() or path.is_symlink():
            raise RollingFanoutError(f"fresh {label} already exists: {path}")
        if path.parent.exists() and list(path.parent.glob(f"{path.name}__attempt_*")):
            raise RollingFanoutError(f"fresh {label} has historical attempt siblings")
    return {"output": output, "cache": cache}


def _validate_process_evidence(path: Path, producer: str) -> dict[str, Any]:
    evidence = _read_json(path)
    checks = {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "producer_worker_id": producer,
        "active_process_absent_confirmed": True,
        "wrapper_exit_code": 0,
        "controller_child_exit_code": 0,
        "recorded_before_effect_unblind": True,
        "effect_metrics_read": False,
    }
    for field, expected in checks.items():
        if evidence.get(field) != expected:
            raise RollingFanoutError(f"process evidence {field} mismatch")
    if producer == "P12" and not str(evidence.get("fresh_controller_terminal_sha256", "")):
        raise RollingFanoutError("P12 fresh-controller terminal evidence missing")
    return evidence


def _validate_health_evidence(path: Path, slot: str) -> dict[str, Any]:
    evidence = _read_json(path)
    checks = {
        "schema_version": HEALTH_EVIDENCE_SCHEMA,
        "physical_slot": slot,
        "gpu_healthy": True,
        "qwen_healthy": True,
        "bge_healthy": True,
        "recorded_before_effect_unblind": True,
        "effect_metrics_read": False,
    }
    for field, expected in checks.items():
        if evidence.get(field) != expected:
            raise RollingFanoutError(f"health evidence {field} mismatch")
    return evidence


def _validate_concurrency_evidence(path: Path, slot: str, producer: str) -> dict[str, Any]:
    evidence = _read_json(path)
    checks = {
        "schema_version": CONCURRENCY_EVIDENCE_SCHEMA,
        "physical_slot": slot,
        "releasing_worker_id": producer,
        "target_slot_process_absent": True,
        "unrelated_instances_248_451_untouched": True,
        "instance_lifecycle_action_requested": False,
        "recorded_before_effect_unblind": True,
        "effect_metrics_read": False,
    }
    for field, expected in checks.items():
        if evidence.get(field) != expected:
            raise RollingFanoutError(f"concurrency evidence {field} mismatch")
    active = evidence.get("active_physical_workers_after_launch")
    if (
        not isinstance(active, int)
        or isinstance(active, bool)
        or not 1 <= active <= MAX_ACTIVE_PHYSICAL_WORKERS
    ):
        raise RollingFanoutError("active physical worker cap would be exceeded")
    return evidence


def _validate_archive_receipt(path: Path, producer_run_dir: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    checks = {
        "schema_version": "task4_snapshot_archive_receipt_v1",
        "status": "verified",
    }
    for field, expected in checks.items():
        if receipt.get(field) != expected:
            raise RollingFanoutError(f"archive receipt {field} mismatch")
    if Path(str(receipt.get("root", ""))).resolve() != producer_run_dir.resolve():
        raise RollingFanoutError("archive receipt root mismatch")
    for field in ("source_verification", "archive_verification", "checksum_verification"):
        value = receipt.get(field)
        if not isinstance(value, dict) or value.get("verified") is not True:
            raise RollingFanoutError(f"archive receipt {field} is not verified")
    for field in ("archive_sha256", "manifest_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RollingFanoutError(f"archive receipt {field} invalid")
    archive_path = Path(str(receipt.get("archive", "")))
    manifest_path = Path(str(receipt.get("manifest", "")))
    checksum_path = Path(str(receipt.get("checksum", "")))
    for label, evidence_path in (
        ("archive", archive_path),
        ("manifest", manifest_path),
        ("checksum", checksum_path),
    ):
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise RollingFanoutError(f"current archive {label} missing or symlinked")
    if _sha256_file(archive_path) != receipt["archive_sha256"]:
        raise RollingFanoutError("current archive SHA-256 mismatch")
    if _sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise RollingFanoutError("current archive manifest SHA-256 mismatch")
    if verify_archive_checksum(archive_path, checksum_path).get("verified") is not True:
        raise RollingFanoutError("current archive checksum verification failed")
    return receipt


def _ledger_root(bundle_root: Path, slot: str, consumer: str) -> Path:
    return _authoritative_control_root(bundle_root) / slot / consumer


def _attempt_paths(ledger_root: Path, attempt: int) -> dict[str, Path]:
    stem = f"attempt_{attempt:04d}"
    return {
        "claim": ledger_root / f"{stem}.claim.json",
        "failed": ledger_root / f"{stem}.failed.json",
        "launched": ledger_root / f"{stem}.launched.json",
    }


def _ledger_admission(ledger_root: Path) -> dict[str, Any]:
    if not ledger_root.exists():
        return {"control_attempt": 1, "previous_terminal_sha256": "GENESIS"}
    if not ledger_root.is_dir() or ledger_root.is_symlink():
        raise RollingFanoutError("fanout ledger root is not a regular directory")
    pattern = re.compile(r"^attempt_(\d{4})\.(claim|failed|launched)\.json$")
    by_kind: dict[str, set[int]] = {"claim": set(), "failed": set(), "launched": set()}
    for path in ledger_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise RollingFanoutError(f"unexpected fanout ledger entry: {path.name}")
        matched = pattern.fullmatch(path.name)
        if matched is None:
            raise RollingFanoutError(f"unexpected fanout ledger file: {path.name}")
        by_kind[matched.group(2)].add(int(matched.group(1)))
    claims = by_kind["claim"]
    failed = by_kind["failed"]
    launched = by_kind["launched"]
    if failed - claims or launched - claims or failed & launched:
        raise RollingFanoutError("fanout ledger terminal/claim topology mismatch")
    if claims and claims != set(range(1, max(claims) + 1)):
        raise RollingFanoutError("fanout ledger attempts are not contiguous")
    if claims - failed - launched:
        raise RollingFanoutError("prior fanout claim has no terminal evidence")
    previous_terminal_sha256 = "GENESIS"
    for attempt in sorted(claims):
        paths = _attempt_paths(ledger_root, attempt)
        claim = _read_json(paths["claim"])
        if claim.get("schema_version") != CLAIM_SCHEMA:
            raise RollingFanoutError("prior fanout claim schema mismatch")
        if claim.get("control_attempt") != attempt:
            raise RollingFanoutError("prior fanout claim attempt mismatch")
        if claim.get("previous_launch_terminal_sha256") != previous_terminal_sha256:
            raise RollingFanoutError("fanout ledger previous-terminal hash mismatch")
        terminal_path = paths["failed"] if attempt in failed else paths["launched"]
        terminal = _read_json(terminal_path)
        expected_schema = FAILED_SCHEMA if attempt in failed else LAUNCHED_SCHEMA
        if terminal.get("schema_version") != expected_schema:
            raise RollingFanoutError("prior fanout terminal schema mismatch")
        if terminal.get("claim_sha256") != _sha256_file(paths["claim"]):
            raise RollingFanoutError("prior fanout terminal claim binding mismatch")
        previous_terminal_sha256 = _sha256_file(terminal_path)
    if launched:
        raise RollingFanoutError("secondary was already launched for this slot")
    if failed:
        last_failed = _read_json(_attempt_paths(ledger_root, max(failed))["failed"])
        if last_failed.get("safe_retry_allowed") is not True:
            raise RollingFanoutError("prior failed fanout requires slot isolation")
    return {
        "control_attempt": len(claims) + 1,
        "previous_terminal_sha256": previous_terminal_sha256,
    }


def build_release_draft(
    *,
    fanout_plan: Path,
    current_ops_manifest: Path,
    producer_worker_id: str,
    producer_manifest: Path,
    consumer_manifest: Path,
    bundle_root: Path,
    producer_run_dir: Path,
    archive_receipt: Path,
    process_evidence: Path,
    health_evidence: Path,
    concurrency_evidence: Path,
    python_path: Path,
    launcher_path: Path,
) -> dict[str, Any]:
    """Build deterministic inactive release data from engineering-only evidence."""

    row = _read_plan_row(fanout_plan.resolve(), producer_worker_id)
    if _sha256_file(fanout_plan.resolve()) != FANOUT_PLAN_SHA256:
        raise RollingFanoutError("authoritative fanout plan SHA-256 mismatch")
    if not current_ops_manifest.is_file() or current_ops_manifest.is_symlink():
        raise RollingFanoutError("current ops manifest missing or symlinked")
    if _sha256_file(current_ops_manifest.resolve()) != CURRENT_OPS_MANIFEST_SHA256:
        raise RollingFanoutError("authoritative current-ops manifest SHA-256 mismatch")
    checkout = Path(row["scientific_checkout"]).resolve()
    _verify_frozen_checkout(checkout)
    if producer_run_dir.resolve() != _inside(
        checkout, row["producer_output_path"], label="producer output"
    ):
        raise RollingFanoutError("producer run-dir binding mismatch")
    producer = _load_manifest(
        producer_manifest.resolve(),
        worker_id=producer_worker_id,
        role="primary",
        expected_sha=row["producer_manifest_sha256"],
    )
    consumer = _load_manifest(
        consumer_manifest.resolve(),
        worker_id=row["consumer_worker"],
        role="secondary",
        expected_sha=row["consumer_manifest_sha256"],
    )
    if int(producer["seed_bundle"]) != int(row["seed"]) or int(
        consumer["seed_bundle"]
    ) != int(row["seed"]):
        raise RollingFanoutError("seed binding mismatch")
    if consumer.get("depends_on") != producer_worker_id:
        raise RollingFanoutError("consumer dependency binding mismatch")
    if consumer.get("dependency_output_path") != row["producer_output_path"]:
        raise RollingFanoutError("consumer dependency-output binding mismatch")
    receipt_root = receipt_root_for_row(row, bundle_root.resolve())
    control_root = _authoritative_control_root(bundle_root.resolve())
    ledger_root = _ledger_root(
        bundle_root.resolve(), row["physical_slot"], row["consumer_worker"]
    )
    ledger = _ledger_admission(ledger_root)
    receipt_path = Path(row["producer_receipt_path"]).resolve()
    validate_worker_for_archive(producer_run_dir.resolve())
    verify_checkpoint_receipt(
        receipt_path,
        producer_run_dir.resolve(),
        expected_identity=_receipt_identity(consumer, consumer=True),
        expected_producer_worker_id=producer_worker_id,
        expected_seed_bundle=int(row["seed"]),
        expected_checkpoint_hand=300,
    )
    archive = _validate_archive_receipt(archive_receipt.resolve(), producer_run_dir.resolve())
    _validate_process_evidence(process_evidence.resolve(), producer_worker_id)
    _validate_health_evidence(health_evidence.resolve(), row["physical_slot"])
    _validate_concurrency_evidence(
        concurrency_evidence.resolve(), row["physical_slot"], producer_worker_id
    )
    consumer_paths = _assert_fresh_consumer_paths(checkout, consumer)
    if consumer_paths["output"] != _inside(
        checkout,
        row["consumer_output_path"],
        label="planned consumer output",
    ):
        raise RollingFanoutError("consumer output binding mismatch")
    if consumer_paths["cache"] != _inside(
        checkout,
        row["consumer_cache_namespace"],
        label="planned consumer cache",
    ):
        raise RollingFanoutError("consumer cache binding mismatch")
    if not python_path.is_file() or python_path.is_symlink():
        raise RollingFanoutError("verified Python interpreter missing or symlinked")
    if not launcher_path.is_file() or launcher_path.is_symlink():
        raise RollingFanoutError("frozen secondary launcher missing or symlinked")
    if _sha256_source(launcher_path.resolve()) != FROZEN_SECONDARY_LAUNCHER_SHA256:
        raise RollingFanoutError("frozen secondary launcher pre-frozen SHA-256 mismatch")
    launch_argv = build_launch_argv(
        python_path, launcher_path, checkout, consumer_manifest, receipt_root
    )
    body: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "active": False,
        "pod_id": row["pod_id"],
        "seed": int(row["seed"]),
        "physical_slot": row["physical_slot"],
        "producer_worker_id": producer_worker_id,
        "consumer_worker_id": row["consumer_worker"],
        "frozen_code_sha": FROZEN_CODE_SHA,
        "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
        "frozen_cli_main_sha256": FROZEN_CLI_MAIN_SHA256,
        "scientific_checkout": str(checkout),
        "fanout_plan_path": str(fanout_plan.resolve()),
        "fanout_plan_sha256": _sha256_file(fanout_plan.resolve()),
        "current_ops_manifest_path": str(current_ops_manifest.resolve()),
        "current_ops_manifest_sha256": _sha256_file(current_ops_manifest.resolve()),
        "producer_manifest_path": str(producer_manifest.resolve()),
        "producer_manifest_sha256": row["producer_manifest_sha256"],
        "consumer_manifest_path": str(consumer_manifest.resolve()),
        "consumer_manifest_sha256": row["consumer_manifest_sha256"],
        "producer_run_dir": str(producer_run_dir.resolve()),
        "producer_completion_receipt_sha256": _sha256_file(
            producer_run_dir.resolve() / "completion_receipt.json"
        ),
        "producer_checkpoint_receipt_path": str(receipt_path),
        "producer_checkpoint_receipt_sha256": _sha256_file(receipt_path),
        "archive_receipt_path": str(archive_receipt.resolve()),
        "archive_receipt_sha256": _sha256_file(archive_receipt.resolve()),
        "archive_sha256": archive["archive_sha256"],
        "archive_manifest_sha256": archive["manifest_sha256"],
        "process_evidence_path": str(process_evidence.resolve()),
        "process_evidence_sha256": _sha256_file(process_evidence.resolve()),
        "health_evidence_path": str(health_evidence.resolve()),
        "health_evidence_sha256": _sha256_file(health_evidence.resolve()),
        "concurrency_evidence_path": str(concurrency_evidence.resolve()),
        "concurrency_evidence_sha256": _sha256_file(concurrency_evidence.resolve()),
        "receipt_root": str(receipt_root),
        "consumer_output_path": str(consumer_paths["output"]),
        "consumer_cache_path": str(consumer_paths["cache"]),
        "python_path": str(python_path.resolve()),
        "python_sha256": _sha256_file(python_path.resolve()),
        "launcher_path": str(launcher_path.resolve()),
        "launcher_sha256": FROZEN_SECONDARY_LAUNCHER_SHA256,
        "launch_argv": launch_argv,
        "control_root": str(control_root),
        "launch_ledger_root": str(ledger_root),
        "control_attempt": ledger["control_attempt"],
        "previous_launch_terminal_sha256": ledger["previous_terminal_sha256"],
        "append_only": True,
        "resume_existing": False,
        "destructive_actions_allowed": False,
        "instance_lifecycle_actions_allowed": False,
        "recorded_before_effect_unblind": True,
        "effect_metrics_read": False,
    }
    body["release_id"] = _sha256_bytes(_json_bytes(body))
    return body


def activate_release(
    *,
    draft_path: Path,
    archive_path: Path,
    checksum_path: Path,
    manifest_path: Path,
    extraction_output: Path,
    extraction_receipt: Path,
    activated_output: Path,
    signing_key: Path,
    allowed_signers: Path,
    signature_output: Path,
) -> dict[str, Any]:
    """Verify the local archive and publish a new activated release."""

    draft = _read_json(draft_path.resolve())
    if draft.get("schema_version") != RELEASE_SCHEMA or draft.get("active") is not False:
        raise RollingFanoutError("release draft schema or active flag mismatch")
    if _sha256_file(archive_path.resolve()) != draft.get("archive_sha256"):
        raise RollingFanoutError("downloaded archive SHA-256 mismatch")
    if _sha256_file(manifest_path.resolve()) != draft.get("archive_manifest_sha256"):
        raise RollingFanoutError("downloaded archive manifest SHA-256 mismatch")
    extraction = extract_snapshot_archive(
        archive_path.resolve(),
        checksum_path.resolve(),
        manifest_path.resolve(),
        extraction_output.resolve(),
        extraction_receipt.resolve(),
    )
    if extraction.get("status") != "verified":
        raise RollingFanoutError("local archive extraction did not verify")
    activated = dict(draft)
    activated["schema_version"] = ACTIVATION_SCHEMA
    activated["active"] = True
    activated["activated_from_sha256"] = _sha256_file(draft_path.resolve())
    activated["local_archive_path"] = str(archive_path.resolve())
    activated["local_archive_sha256"] = _sha256_file(archive_path.resolve())
    activated["local_manifest_path"] = str(manifest_path.resolve())
    activated["local_manifest_sha256"] = _sha256_file(manifest_path.resolve())
    activated["local_checksum_path"] = str(checksum_path.resolve())
    activated["local_checksum_sha256"] = _sha256_file(checksum_path.resolve())
    extraction_object = _read_json(extraction_receipt.resolve())
    activated["local_extraction_receipt_path"] = str(extraction_receipt.resolve())
    activated["local_extraction_receipt_file_sha256"] = _sha256_file(
        extraction_receipt.resolve()
    )
    activated["local_extraction_receipt_sha256"] = _sha256_bytes(
        _json_bytes(extraction_object)
    )
    activated["local_extraction_receipt"] = extraction_object
    activated["activation_controller_sha256"] = _sha256_source(Path(__file__).resolve())
    activated["local_per_file_hash_verified"] = True
    activated["effect_metrics_read"] = False
    _write_json_new(activated_output.resolve(), activated)
    expected_signature = Path(f"{activated_output.resolve()}.sig")
    if signature_output.resolve() != expected_signature:
        raise RollingFanoutError("activation signature output path must be <activated>.sig")
    if signature_output.exists() or signature_output.is_symlink():
        raise RollingFanoutError("refuse to overwrite activation signature")
    if not signing_key.is_file() or signing_key.is_symlink():
        raise RollingFanoutError("activation signing key missing or symlinked")
    try:
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(signing_key.resolve()),
                "-n",
                ACTIVATION_SIGNATURE_NAMESPACE,
                str(activated_output.resolve()),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RollingFanoutError("activation signing failed") from exc
    _verify_activation_signature(
        activated_path=activated_output.resolve(),
        signature_path=signature_output.resolve(),
        allowed_signers_path=allowed_signers.resolve(),
    )
    return activated


def _claim_path(ledger_root: Path, attempt: int) -> Path:
    return _attempt_paths(ledger_root.resolve(), attempt)["claim"]


def _probe_secondary_running(
    process: Any, output_path: Path, timeout_seconds: float = 30.0
) -> tuple[Path, str]:
    deadline = time.monotonic() + timeout_seconds
    state_path = output_path.resolve() / "state.tsv"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RollingFanoutError(
                f"secondary exited before running-state gate: {return_code}"
            )
        if state_path.is_file() and not state_path.is_symlink():
            with state_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if rows and rows[-1].get("status") == "running":
                previous_sha256 = "GENESIS"
                for row in rows:
                    if row.get("schema_version") != "task8-worker-state-v1":
                        raise RollingFanoutError("secondary state schema mismatch")
                    if row.get("previous_sha256") != previous_sha256:
                        raise RollingFanoutError("secondary state previous hash mismatch")
                    expected = _sha256_json(
                        {
                            "schema_version": row["schema_version"],
                            "created_at_utc": row["created_at_utc"],
                            "status": row["status"],
                            "detail": row["detail"],
                            "previous_sha256": row["previous_sha256"],
                        }
                    )
                    if row.get("row_sha256") != expected:
                        raise RollingFanoutError("secondary state row SHA-256 mismatch")
                    previous_sha256 = expected
                return state_path, previous_sha256
        time.sleep(0.25)
    raise RollingFanoutError("secondary did not reach running-state gate in time")


def _publish_failed_handoff(
    *,
    claim_path: Path,
    ledger_root: Path,
    control_attempt: int,
    stage: str,
    error: BaseException,
    consumer_output: Path,
    consumer_cache: Path,
    process_started: bool,
) -> None:
    artifacts_exist = any(
        path.exists() or path.is_symlink() for path in (consumer_output, consumer_cache)
    )
    safe_retry = not process_started and not artifacts_exist
    failed = {
        "schema_version": FAILED_SCHEMA,
        "status": "failed_handoff",
        "control_attempt": control_attempt,
        "claim_sha256": _sha256_file(claim_path),
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:1000],
        "process_started": process_started,
        "consumer_artifacts_exist": artifacts_exist,
        "safe_retry_allowed": safe_retry,
        "isolation_required": not safe_retry,
        "effect_metrics_read": False,
    }
    _write_json_new(_attempt_paths(ledger_root, control_attempt)["failed"], failed)


def launch_secondary(
    *,
    activated_path: Path,
    activation_signature_path: Path,
    allowed_signers_path: Path,
    rebuilt_draft: dict[str, Any],
    stdout_path: Path,
    stderr_path: Path,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Revalidate, claim exclusively, and launch one fresh secondary."""

    activated = _read_json(activated_path.resolve())
    _verify_activation_signature(
        activated_path=activated_path.resolve(),
        signature_path=activation_signature_path.resolve(),
        allowed_signers_path=allowed_signers_path.resolve(),
    )
    expected_base = dict(activated)
    for field in (
        "activated_from_sha256",
        "local_archive_path",
        "local_archive_sha256",
        "local_manifest_path",
        "local_manifest_sha256",
        "local_checksum_path",
        "local_checksum_sha256",
        "local_extraction_receipt_path",
        "local_extraction_receipt_file_sha256",
        "local_extraction_receipt_sha256",
        "local_extraction_receipt",
        "activation_controller_sha256",
        "local_per_file_hash_verified",
    ):
        expected_base.pop(field, None)
    expected_base["schema_version"] = RELEASE_SCHEMA
    expected_base["active"] = False
    if expected_base != rebuilt_draft:
        raise RollingFanoutError("activated release does not match rebuilt current bindings")
    if activated.get("schema_version") != ACTIVATION_SCHEMA or activated.get("active") is not True:
        raise RollingFanoutError("activated release is not active")
    if activated.get("activated_from_sha256") != _sha256_bytes(_json_bytes(rebuilt_draft)):
        raise RollingFanoutError("activated release draft binding mismatch")
    if activated.get("local_archive_sha256") != rebuilt_draft.get("archive_sha256"):
        raise RollingFanoutError("activated release local archive binding mismatch")
    if activated.get("local_manifest_sha256") != rebuilt_draft.get("archive_manifest_sha256"):
        raise RollingFanoutError("activated release local manifest binding mismatch")
    if activated.get("local_per_file_hash_verified") is not True:
        raise RollingFanoutError("activated release lacks local per-file verification")
    if activated.get("activation_controller_sha256") != _sha256_source(Path(__file__).resolve()):
        raise RollingFanoutError("activation controller SHA-256 mismatch")
    extraction = activated.get("local_extraction_receipt")
    if not isinstance(extraction, dict):
        raise RollingFanoutError("local extraction receipt object missing")
    if _sha256_bytes(_json_bytes(extraction)) != activated.get(
        "local_extraction_receipt_sha256"
    ):
        raise RollingFanoutError("local extraction receipt SHA-256 mismatch")
    if extraction.get("schema_version") != "task4_snapshot_extraction_receipt_v1":
        raise RollingFanoutError("local extraction receipt schema mismatch")
    if extraction.get("status") != "verified":
        raise RollingFanoutError("local extraction receipt status mismatch")
    if extraction.get("manifest_sha256") != rebuilt_draft.get("archive_manifest_sha256"):
        raise RollingFanoutError("local extraction manifest binding mismatch")
    for field, activated_field in (
        ("archive", "local_archive_path"),
        ("manifest", "local_manifest_path"),
        ("checksum", "local_checksum_path"),
    ):
        if Path(str(extraction.get(field, ""))).resolve() != Path(
            str(activated.get(activated_field, ""))
        ).resolve():
            raise RollingFanoutError(f"local extraction {field} path binding mismatch")
    for field in ("checksum_verification", "archive_verification", "extracted_verification"):
        value = extraction.get(field)
        if not isinstance(value, dict) or value.get("verified") is not True:
            raise RollingFanoutError(f"local extraction {field} is not verified")
    checksum_verification = extraction["checksum_verification"]
    if checksum_verification.get("observed_sha256") != rebuilt_draft.get("archive_sha256"):
        raise RollingFanoutError("local extraction archive SHA-256 binding mismatch")
    checkout = Path(str(rebuilt_draft["scientific_checkout"]))
    consumer_manifest = _load_manifest(
        Path(str(rebuilt_draft["consumer_manifest_path"])),
        worker_id=str(rebuilt_draft["consumer_worker_id"]),
        role="secondary",
        expected_sha=str(rebuilt_draft["consumer_manifest_sha256"]),
    )
    _assert_fresh_consumer_paths(checkout, consumer_manifest)
    if _sha256_source(Path(str(rebuilt_draft["launcher_path"]))) != (
        FROZEN_SECONDARY_LAUNCHER_SHA256
    ) or rebuilt_draft.get("launcher_sha256") != FROZEN_SECONDARY_LAUNCHER_SHA256:
        raise RollingFanoutError("frozen secondary launcher SHA-256 mismatch")
    if _sha256_file(Path(str(rebuilt_draft["python_path"]))) != rebuilt_draft.get(
        "python_sha256"
    ):
        raise RollingFanoutError("Python interpreter SHA-256 mismatch")
    argv = build_launch_argv(
        Path(str(rebuilt_draft["python_path"])),
        Path(str(rebuilt_draft["launcher_path"])),
        checkout,
        Path(str(rebuilt_draft["consumer_manifest_path"])),
        Path(str(rebuilt_draft["receipt_root"])),
    )
    if argv != rebuilt_draft.get("launch_argv") or "--resume-existing" in argv:
        raise RollingFanoutError("secondary launch argv binding mismatch")
    for path in (stdout_path.resolve(), stderr_path.resolve()):
        if path.exists() or path.is_symlink():
            raise RollingFanoutError(f"launch log already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    expected_ledger_root = _ledger_root(
        Path(str(rebuilt_draft["receipt_root"])).parent
        if rebuilt_draft["producer_worker_id"] != "P12"
        else Path(str(rebuilt_draft["receipt_root"])),
        str(rebuilt_draft["physical_slot"]),
        str(rebuilt_draft["consumer_worker_id"]),
    )
    ledger_root = Path(str(rebuilt_draft["launch_ledger_root"])).resolve()
    if ledger_root != expected_ledger_root:
        raise RollingFanoutError("authoritative launch ledger root mismatch")
    admission = _ledger_admission(ledger_root)
    if admission["control_attempt"] != rebuilt_draft.get("control_attempt"):
        raise RollingFanoutError("launch control-attempt binding mismatch")
    if admission["previous_terminal_sha256"] != rebuilt_draft.get(
        "previous_launch_terminal_sha256"
    ):
        raise RollingFanoutError("launch previous-terminal binding mismatch")
    control_attempt = int(rebuilt_draft["control_attempt"])
    claim_path = _claim_path(ledger_root, control_attempt)
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "status": "claimed",
        "physical_slot": rebuilt_draft["physical_slot"],
        "producer_worker_id": rebuilt_draft["producer_worker_id"],
        "consumer_worker_id": rebuilt_draft["consumer_worker_id"],
        "seed": rebuilt_draft["seed"],
        "control_attempt": control_attempt,
        "release_id": rebuilt_draft["release_id"],
        "activated_release_sha256": _sha256_file(activated_path.resolve()),
        "activation_signature_sha256": _sha256_file(
            activation_signature_path.resolve()
        ),
        "consumer_manifest_sha256": rebuilt_draft["consumer_manifest_sha256"],
        "previous_launch_terminal_sha256": rebuilt_draft[
            "previous_launch_terminal_sha256"
        ],
        "effect_metrics_read": False,
    }
    _write_json_new(claim_path, claim)
    stdout_handle = None
    stderr_handle = None
    process = None
    stage = "claim-acquired"
    try:
        stdout_handle = stdout_path.resolve().open("xb")
        stderr_handle = stderr_path.resolve().open("xb")
        launch_env = dict(os.environ)
        launch_env["PYTHONPATH"] = ""
        launch_env["PYTHONDONTWRITEBYTECODE"] = "1"
        stage = "process-launch"
        process = popen_factory(
            argv,
            cwd=str(checkout.resolve()),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=launch_env,
            shell=False,
            start_new_session=True,
        )
        stdout_handle.close()
        stdout_handle = None
        stderr_handle.close()
        stderr_handle = None
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            raise RollingFanoutError("secondary launcher returned invalid PID")
        stage = "running-state-probe"
        state_path, state_tail_row_sha256 = _probe_secondary_running(
            process, Path(str(rebuilt_draft["consumer_output_path"]))
        )
    except BaseException as exc:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        _publish_failed_handoff(
            claim_path=claim_path,
            ledger_root=ledger_root,
            control_attempt=control_attempt,
            stage=stage,
            error=exc,
            consumer_output=Path(str(rebuilt_draft["consumer_output_path"])),
            consumer_cache=Path(str(rebuilt_draft["consumer_cache_path"])),
            process_started=process is not None,
        )
        raise
    try:
        stage = "running-terminal-publication"
        launched = {
            "schema_version": LAUNCHED_SCHEMA,
            "status": "secondary_running",
            "physical_slot": rebuilt_draft["physical_slot"],
            "producer_worker_id": rebuilt_draft["producer_worker_id"],
            "consumer_worker_id": rebuilt_draft["consumer_worker_id"],
            "seed": rebuilt_draft["seed"],
            "control_attempt": control_attempt,
            "claim_sha256": _sha256_file(claim_path),
            "activated_release_sha256": _sha256_file(activated_path.resolve()),
            "activation_signature_sha256": _sha256_file(
                activation_signature_path.resolve()
            ),
            "pid": pid,
            "state_path": str(state_path),
            "state_tail_row_sha256": state_tail_row_sha256,
            "launch_argv": argv,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "resume_existing": False,
            "effect_metrics_read": False,
        }
        launched_path = _attempt_paths(ledger_root, control_attempt)["launched"]
        staging_root = _authoritative_control_root(
            Path(str(rebuilt_draft["receipt_root"])).parent
            if rebuilt_draft["producer_worker_id"] != "P12"
            else Path(str(rebuilt_draft["receipt_root"]))
        ) / ".s"
        _publish_json_atomic_new(launched_path, launched, staging_root)
    except BaseException as exc:
        _publish_failed_handoff(
            claim_path=claim_path,
            ledger_root=ledger_root,
            control_attempt=control_attempt,
            stage=stage,
            error=exc,
            consumer_output=Path(str(rebuilt_draft["consumer_output_path"])),
            consumer_cache=Path(str(rebuilt_draft["consumer_cache_path"])),
            process_started=True,
        )
        raise
    return launched


def _add_server_binding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fanout-plan", type=Path, required=True)
    parser.add_argument("--current-ops-manifest", type=Path, required=True)
    parser.add_argument("--producer-worker-id", required=True)
    parser.add_argument("--producer-manifest", type=Path, required=True)
    parser.add_argument("--consumer-manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--producer-run-dir", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--process-evidence", type=Path, required=True)
    parser.add_argument("--health-evidence", type=Path, required=True)
    parser.add_argument("--concurrency-evidence", type=Path, required=True)
    parser.add_argument("--python-path", type=Path, required=True)
    parser.add_argument("--launcher-path", type=Path, required=True)


def _server_bindings(args: argparse.Namespace) -> dict[str, Any]:
    return build_release_draft(
        fanout_plan=args.fanout_plan,
        current_ops_manifest=args.current_ops_manifest,
        producer_worker_id=args.producer_worker_id,
        producer_manifest=args.producer_manifest,
        consumer_manifest=args.consumer_manifest,
        bundle_root=args.bundle_root,
        producer_run_dir=args.producer_run_dir,
        archive_receipt=args.archive_receipt,
        process_evidence=args.process_evidence,
        health_evidence=args.health_evidence,
        concurrency_evidence=args.concurrency_evidence,
        python_path=args.python_path,
        launcher_path=args.launcher_path,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-draft")
    _add_server_binding_args(build)
    build.add_argument("--output", type=Path, required=True)
    activate = commands.add_parser("activate")
    activate.add_argument("--draft", type=Path, required=True)
    activate.add_argument("--archive", type=Path, required=True)
    activate.add_argument("--checksum", type=Path, required=True)
    activate.add_argument("--manifest", type=Path, required=True)
    activate.add_argument("--extraction-output", type=Path, required=True)
    activate.add_argument("--extraction-receipt", type=Path, required=True)
    activate.add_argument("--output", type=Path, required=True)
    activate.add_argument("--signing-key", type=Path, required=True)
    activate.add_argument("--allowed-signers", type=Path, required=True)
    activate.add_argument("--signature-output", type=Path, required=True)
    launch = commands.add_parser("launch")
    _add_server_binding_args(launch)
    launch.add_argument("--activated", type=Path, required=True)
    launch.add_argument("--activation-signature", type=Path, required=True)
    launch.add_argument("--allowed-signers", type=Path, required=True)
    launch.add_argument("--stdout", type=Path, required=True)
    launch.add_argument("--stderr", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build-draft":
        result = _server_bindings(args)
        _write_json_new(args.output.resolve(), result)
    elif args.command == "activate":
        result = activate_release(
            draft_path=args.draft,
            archive_path=args.archive,
            checksum_path=args.checksum,
            manifest_path=args.manifest,
            extraction_output=args.extraction_output,
            extraction_receipt=args.extraction_receipt,
            activated_output=args.output,
            signing_key=args.signing_key,
            allowed_signers=args.allowed_signers,
            signature_output=args.signature_output,
        )
    else:
        result = launch_secondary(
            activated_path=args.activated,
            activation_signature_path=args.activation_signature,
            allowed_signers_path=args.allowed_signers,
            rebuilt_draft=_server_bindings(args),
            stdout_path=args.stdout,
            stderr_path=args.stderr,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
