"""Fail-closed P12 fresh-run wrapper for the frozen TASK8B runner.

Run this file directly with the Python environment used by the frozen checkout.
Only standard-library modules are imported before the frozen checkout and its
runner are verified.  This controller never adopts historical output and never
uses ``resume_existing=True``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FROZEN_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
FROZEN_RUNNER_SHA256 = "c4b601ff0de2c27a57ee246efcf91d21f502f27c652d20fd6fa7cfd925a17d5e"
P12_MANIFEST_SHA256 = "af1f7e9373f9a523336550b26beec59a603c0a4edc61221dd13f05ac5268d66c"
RELEASE_SCHEMA = "task8b-p12-fresh-run-release-v1"
PROCESS_ABSENCE_SCHEMA = "task8b-worker-process-absence-v1"
LAUNCH_CLAIM_SCHEMA = "task8b-p12-fresh-launch-claim-v1"
FAILED_LAUNCH_SCHEMA = "task8b-p12-failed-launch-v1"
COMPLETED_LAUNCH_SCHEMA = "task8b-p12-completed-launch-v1"
WORKER_ID = "P12"
ROLE = "primary"
POD_ID = "pod12"
SEED = 2026090112
TASK_IDS = (
    "isolation_no_memory",
    "isolation_fact",
    "isolation_expr",
    "isolation_sync",
    "isolation_async",
)
HANDS_PER_TASK = 1350
TOTAL_HANDS = 6750
OUTPUT_RELATIVE = "outputs/formal/task8b/P12/2026090112"
CACHE_RELATIVE = "task8b/P12/2026090112"
RECEIPT_RELATIVE = "receipts/P12.json"


class FreshRunError(RuntimeError):
    """Fail-closed fresh-run admission error."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreshRunError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FreshRunError(f"JSON root must be an object: {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FreshRunError(f"refuse to overwrite release file: {path}") from exc


def canonicalize_resolved_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Remove runner-dynamic fields without stringifying integer mapping keys."""

    value = copy.deepcopy(config)
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in (
        "output_root",
        "run_id",
        "initial_memory_snapshots",
        "admission_audit",
    ):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value


def _safe_inside(root: Path, relative: str, *, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not relative or ".." in value.parts:
        raise FreshRunError(f"unsafe {label} relative path: {relative!r}")
    candidate = (root.resolve() / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FreshRunError(f"{label} path escapes root: {relative!r}") from exc
    return candidate


def _git_head_and_tracked_status(checkout: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshRunError(f"unable to verify Git checkout: {checkout}") from exc
    return head, dirty


def _require_tracked_file(checkout: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(checkout.resolve()).as_posix()
    except ValueError as exc:
        raise FreshRunError("verifier file escapes checkout") from exc
    try:
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshRunError("controller is not tracked by verifier commit") from exc


def _verify_frozen_checkout(checkout: Path) -> Path:
    checkout = checkout.resolve()
    runner_path = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    if not runner_path.is_file() or runner_path.is_symlink():
        raise FreshRunError("frozen formal_runner.py missing or is a symlink")
    head, dirty = _git_head_and_tracked_status(checkout)
    if head != FROZEN_CODE_SHA:
        raise FreshRunError(f"frozen checkout HEAD mismatch: {head}")
    if dirty:
        raise FreshRunError("frozen checkout has tracked or untracked changes")
    if _sha256_file(runner_path) != FROZEN_RUNNER_SHA256:
        raise FreshRunError("frozen formal runner SHA-256 mismatch")
    return runner_path


def _verifier_identity(checkout: Path, controller_path: Path) -> dict[str, str]:
    checkout = checkout.resolve()
    controller_path = controller_path.resolve()
    expected_controller = (
        checkout / "src" / "agentmemeval" / "experiments" / "task8b_p12_fresh_run.py"
    ).resolve()
    if controller_path != expected_controller:
        raise FreshRunError("controller path is not inside the verifier checkout")
    if not controller_path.is_file() or controller_path.is_symlink():
        raise FreshRunError("controller missing or is a symlink")
    _require_tracked_file(checkout, controller_path)
    head, dirty = _git_head_and_tracked_status(checkout)
    if dirty:
        raise FreshRunError("verifier checkout has tracked or untracked changes")
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise FreshRunError("invalid verifier commit")
    controller_sha256 = _sha256_file(controller_path)
    return {
        "verifier_code_sha": head,
        "controller_sha256": controller_sha256,
        "verifier_identity_source_sha256": controller_sha256,
    }


def _validate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FreshRunError("P12 manifest missing or is a symlink")
    if _sha256_file(manifest_path) != P12_MANIFEST_SHA256:
        raise FreshRunError("P12 manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    exact = {
        "worker_id": WORKER_ID,
        "role": ROLE,
        "pod_id": POD_ID,
        "seed_bundle": SEED,
        "protocol_status": "frozen/expedited-formal-candidate",
        "execution_mode": "experiment_configs",
        "depends_on": None,
        "receipt_relative_path": RECEIPT_RELATIVE,
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise FreshRunError(f"P12 manifest {field} mismatch")
    common = manifest.get("common_identity")
    if not isinstance(common, dict) or common.get("code_sha") != FROZEN_CODE_SHA:
        raise FreshRunError("P12 manifest frozen code identity mismatch")
    instance = manifest.get("instance_identity")
    if not isinstance(instance, dict):
        raise FreshRunError("P12 manifest instance identity missing")
    if instance.get("worker_id") != WORKER_ID:
        raise FreshRunError("P12 manifest instance worker mismatch")
    if instance.get("output_path") != OUTPUT_RELATIVE:
        raise FreshRunError("P12 manifest output path mismatch")
    if instance.get("cache_namespace") != CACHE_RELATIVE:
        raise FreshRunError("P12 manifest cache namespace mismatch")
    tasks = manifest.get("task_configs")
    if not isinstance(tasks, list) or len(tasks) != len(TASK_IDS):
        raise FreshRunError("P12 manifest task count mismatch")
    if tuple(str(task.get("task_id", "")) for task in tasks) != TASK_IDS:
        raise FreshRunError("P12 manifest task order/topology mismatch")
    if any(int(task.get("planned_hands", -1)) != HANDS_PER_TASK for task in tasks):
        raise FreshRunError("P12 manifest per-task hands mismatch")
    if sum(int(task["planned_hands"]) for task in tasks) != TOTAL_HANDS:
        raise FreshRunError("P12 manifest total hands mismatch")
    if sum(bool(task.get("publish_checkpoint_after", False)) for task in tasks) != 1:
        raise FreshRunError("P12 manifest checkpoint publication boundary mismatch")
    return manifest


def _validate_process_absence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FreshRunError("process-absence evidence missing or is a symlink")
    evidence = _read_json(path)
    checks = {
        "schema_version": PROCESS_ABSENCE_SCHEMA,
        "worker_id": WORKER_ID,
        "active_process_absent_confirmed": True,
        "formal_worker_count": 0,
    }
    for field, expected in checks.items():
        if evidence.get(field) != expected:
            raise FreshRunError(f"process-absence evidence {field} mismatch")
    if evidence.get("active_process_absent_confirmed") is not True:
        raise FreshRunError("process-absence evidence confirmation must be boolean true")
    worker_count = evidence.get("formal_worker_count")
    if not isinstance(worker_count, int) or isinstance(worker_count, bool):
        raise FreshRunError("process-absence evidence worker count must be an integer")
    if not str(evidence.get("checked_at_utc", "")).strip():
        raise FreshRunError("process-absence evidence checked_at_utc missing")
    if not str(evidence.get("host_id", "")).strip():
        raise FreshRunError("process-absence evidence host_id missing")
    return evidence


def _bound_paths(
    *, manifest: dict[str, Any], frozen_checkout: Path, receipt_root: Path
) -> dict[str, Path]:
    instance = manifest["instance_identity"]
    return {
        "output_path": _safe_inside(frozen_checkout, str(instance["output_path"]), label="output"),
        "cache_path": _safe_inside(
            frozen_checkout, str(instance["cache_namespace"]), label="cache"
        ),
        "receipt_path": _safe_inside(
            receipt_root, str(manifest["receipt_relative_path"]), label="receipt"
        ),
    }


def _assert_fresh_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if path.exists() or path.is_symlink():
            raise FreshRunError(f"fresh-run {label} already exists: {path}")
    output = paths["output_path"]
    if output.parent.exists():
        siblings = sorted(output.parent.glob(f"{output.name}__attempt_*"))
        if siblings:
            raise FreshRunError("fresh-run historical output attempt sibling exists")
    cache = paths["cache_path"]
    if cache.parent.exists():
        siblings = sorted(cache.parent.glob(f"{cache.name}__attempt_*"))
        if siblings:
            raise FreshRunError("fresh-run historical cache attempt sibling exists")


def _launch_ledger_root(receipt_root: Path) -> Path:
    return _safe_inside(
        receipt_root,
        "control/task8b/P12/fresh_launch",
        label="launch ledger",
    )


def _launch_files(ledger_root: Path, control_attempt: int) -> dict[str, Path]:
    stem = f"attempt_{control_attempt:04d}"
    return {
        "claim": ledger_root / f"{stem}.claim.json",
        "failed": ledger_root / f"{stem}.failed.json",
        "completed": ledger_root / f"{stem}.completed.json",
    }


def _ledger_attempt_sets(ledger_root: Path) -> tuple[set[int], set[int], set[int]]:
    if not ledger_root.exists():
        return set(), set(), set()
    if not ledger_root.is_dir() or ledger_root.is_symlink():
        raise FreshRunError("launch ledger root is not a regular directory")
    pattern = re.compile(r"^attempt_(\d{4})\.(claim|failed|completed)\.json$")
    by_kind = {"claim": set(), "failed": set(), "completed": set()}
    for path in ledger_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise FreshRunError(f"unexpected launch ledger entry: {path.name}")
        matched = pattern.fullmatch(path.name)
        if matched is None:
            raise FreshRunError(f"unexpected launch ledger file: {path.name}")
        attempt = int(matched.group(1))
        if attempt < 1:
            raise FreshRunError("launch ledger attempt must be positive")
        by_kind[matched.group(2)].add(attempt)
    return by_kind["claim"], by_kind["failed"], by_kind["completed"]


def _path_file_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.exists() and not path.is_symlink():
        return []
    if path.is_symlink():
        return [{"relative_path": ".", "kind": "symlink", "label": label}]
    if path.is_file():
        return [
            {
                "relative_path": ".",
                "kind": "file",
                "label": label,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        ]
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            rows.append(
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "kind": "symlink",
                    "label": label,
                }
            )
        elif item.is_file():
            rows.append(
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "kind": "file",
                    "label": label,
                    "size": item.stat().st_size,
                    "sha256": _sha256_file(item),
                }
            )
    return rows


def _artifact_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    rows = []
    for label in ("output_path", "cache_path", "receipt_path"):
        rows.extend(_path_file_rows(paths[label], label=label))
    output = paths["output_path"]
    cache = paths["cache_path"]
    output_siblings = (
        sorted(str(path.resolve()) for path in output.parent.glob(f"{output.name}__attempt_*"))
        if output.parent.exists()
        else []
    )
    cache_siblings = (
        sorted(str(path.resolve()) for path in cache.parent.glob(f"{cache.name}__attempt_*"))
        if cache.parent.exists()
        else []
    )
    receipt_exists = paths["receipt_path"].exists() or paths["receipt_path"].is_symlink()
    safe_retry = not rows and not receipt_exists and not output_siblings and not cache_siblings
    return {
        "artifact_files": rows,
        "output_exists": output.exists() or output.is_symlink(),
        "cache_exists": cache.exists() or cache.is_symlink(),
        "receipt_exists": receipt_exists,
        "output_attempt_siblings": output_siblings,
        "cache_attempt_siblings": cache_siblings,
        "safe_retry_allowed": safe_retry,
    }


def _assert_retryable_paths(paths: dict[str, Path]) -> None:
    snapshot = _artifact_snapshot(paths)
    if not snapshot["safe_retry_allowed"]:
        raise FreshRunError("prior failed launch produced artifacts; worker is isolated")
    for label in ("output_path", "cache_path"):
        path = paths[label]
        if path.exists() and (not path.is_dir() or path.is_symlink()):
            raise FreshRunError(f"retry {label} is not an empty regular directory")


def _launch_admission(
    *, receipt_root: Path, paths: dict[str, Path], control_attempt: int
) -> dict[str, Any]:
    if not isinstance(control_attempt, int) or isinstance(control_attempt, bool):
        raise FreshRunError("control attempt must be an integer")
    if control_attempt < 1 or control_attempt > 9999:
        raise FreshRunError("control attempt must be in 1..9999")
    ledger_root = _launch_ledger_root(receipt_root.resolve())
    claims, failed, completed = _ledger_attempt_sets(ledger_root)
    if failed - claims or completed - claims or failed & completed:
        raise FreshRunError("launch ledger terminal/claim topology mismatch")
    if claims and claims != set(range(1, max(claims) + 1)):
        raise FreshRunError("launch ledger attempts are not contiguous")
    dangling = claims - failed - completed
    if dangling:
        raise FreshRunError("prior launch claim has no terminal evidence")
    previous_terminal_sha256 = "GENESIS"
    prior_failures: dict[int, dict[str, Any]] = {}
    for attempt in sorted(claims):
        files = _launch_files(ledger_root, attempt)
        claim = _read_json(files["claim"])
        if claim.get("schema_version") != LAUNCH_CLAIM_SCHEMA:
            raise FreshRunError("prior launch claim schema mismatch")
        if claim.get("control_attempt") != attempt:
            raise FreshRunError("prior launch claim attempt mismatch")
        if claim.get("previous_launch_terminal_sha256") != previous_terminal_sha256:
            raise FreshRunError("prior launch claim hash chain mismatch")
        if attempt in failed:
            terminal = _read_json(files["failed"])
            if terminal.get("schema_version") != FAILED_LAUNCH_SCHEMA:
                raise FreshRunError("prior failed-launch schema mismatch")
            if terminal.get("control_attempt") != attempt:
                raise FreshRunError("prior failed-launch attempt mismatch")
            if terminal.get("claim_sha256") != _sha256_file(files["claim"]):
                raise FreshRunError("prior failed-launch claim binding mismatch")
            prior_failures[attempt] = terminal
            previous_terminal_sha256 = _sha256_file(files["failed"])
        else:
            terminal = _read_json(files["completed"])
            if terminal.get("schema_version") != COMPLETED_LAUNCH_SCHEMA:
                raise FreshRunError("prior completed-launch schema mismatch")
            if terminal.get("control_attempt") != attempt:
                raise FreshRunError("prior completed-launch attempt mismatch")
            if terminal.get("claim_sha256") != _sha256_file(files["claim"]):
                raise FreshRunError("prior completed-launch claim binding mismatch")
            previous_terminal_sha256 = _sha256_file(files["completed"])
    if completed:
        raise FreshRunError("P12 fresh launch already completed")
    expected_attempt = len(claims) + 1
    if control_attempt != expected_attempt:
        raise FreshRunError(f"control attempt must be next append-only attempt {expected_attempt}")
    if not claims:
        _assert_fresh_paths(paths)
    else:
        previous = max(claims)
        prior_failed = prior_failures[previous]
        if prior_failed.get("safe_retry_allowed") is not True:
            raise FreshRunError("prior failed launch requires isolation")
        _assert_retryable_paths(paths)
    return {
        "ledger_root": ledger_root,
        "control_attempt": control_attempt,
        "previous_terminal_sha256": previous_terminal_sha256,
    }


def _base_release(
    *,
    manifest_path: Path,
    receipt_root: Path,
    frozen_checkout: Path,
    verifier_checkout: Path,
    controller_path: Path,
    process_absence_path: Path,
    control_attempt: int,
) -> dict[str, Any]:
    manifest = _validate_manifest(manifest_path)
    runner_path = _verify_frozen_checkout(frozen_checkout)
    verifier = _verifier_identity(verifier_checkout, controller_path)
    _validate_process_absence(process_absence_path)
    paths = _bound_paths(
        manifest=manifest,
        frozen_checkout=frozen_checkout.resolve(),
        receipt_root=receipt_root.resolve(),
    )
    launch = _launch_admission(
        receipt_root=receipt_root,
        paths=paths,
        control_attempt=control_attempt,
    )
    body: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "active": False,
        "worker_id": WORKER_ID,
        "role": ROLE,
        "pod_id": POD_ID,
        "seed_bundle": SEED,
        "task_ids": list(TASK_IDS),
        "hands_per_task": HANDS_PER_TASK,
        "total_hands": TOTAL_HANDS,
        "manifest_path": str(manifest_path.resolve()),
        "worker_manifest_sha256": _sha256_file(manifest_path.resolve()),
        "frozen_checkout": str(frozen_checkout.resolve()),
        "frozen_code_sha": FROZEN_CODE_SHA,
        "formal_runner_path": str(runner_path.resolve()),
        "formal_runner_sha256": FROZEN_RUNNER_SHA256,
        "verifier_checkout": str(verifier_checkout.resolve()),
        "controller_path": str(controller_path.resolve()),
        **verifier,
        "canonicalizer": "integer-key-preserving-deepcopy-v1",
        "canonicalizer_source_sha256": verifier["controller_sha256"],
        "receipt_publication_policy": "frozen-runner-task-receipt-last",
        "python_bytecode_write_disabled": True,
        "receipt_root": str(receipt_root.resolve()),
        "output_path": str(paths["output_path"]),
        "cache_path": str(paths["cache_path"]),
        "receipt_path": str(paths["receipt_path"]),
        "launch_ledger_root": str(launch["ledger_root"]),
        "control_attempt": control_attempt,
        "previous_launch_terminal_sha256": launch["previous_terminal_sha256"],
        "process_absence_evidence_path": str(process_absence_path.resolve()),
        "process_absence_evidence_sha256": _sha256_file(process_absence_path.resolve()),
        "attempt_policy": "fresh-only",
        "control_attempt_policy": "append-only-sequential",
        "resume_existing": False,
        "historical_adoption_allowed": False,
        "same_attempt_recovery_allowed": False,
        "task_receipt_adoption_allowed": False,
        "effect_fields_read": False,
    }
    body["release_id"] = _sha256_bytes(_json_bytes(body))
    return body


def build_release(**kwargs: Any) -> dict[str, Any]:
    """Build deterministic inactive release data without reading outcomes."""

    return _base_release(**kwargs)


def activate_release(
    *, draft_path: Path, rebuilt_draft: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    draft_path = draft_path.resolve()
    if not draft_path.is_file() or draft_path.is_symlink():
        raise FreshRunError("release draft missing or is a symlink")
    if draft_path.read_bytes() != _json_bytes(rebuilt_draft):
        raise FreshRunError("release draft is not byte-identical to rebuilt draft")
    if rebuilt_draft.get("active") is not False:
        raise FreshRunError("release draft must be inactive")
    activated = dict(rebuilt_draft)
    activated["active"] = True
    activated["activated_from_sha256"] = _sha256_file(draft_path)
    _write_json_new(output_path.resolve(), activated)
    return activated


def _validate_active_release(
    release: dict[str, Any], *, release_path: Path, rebuilt_draft: dict[str, Any]
) -> None:
    expected = dict(rebuilt_draft)
    expected["active"] = True
    expected["activated_from_sha256"] = _sha256_bytes(_json_bytes(rebuilt_draft))
    if release != expected:
        raise FreshRunError("activated release fields do not match current bindings")
    checks = {
        "schema_version": RELEASE_SCHEMA,
        "active": True,
        "worker_id": WORKER_ID,
        "worker_manifest_sha256": P12_MANIFEST_SHA256,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "formal_runner_sha256": FROZEN_RUNNER_SHA256,
        "attempt_policy": "fresh-only",
        "resume_existing": False,
        "historical_adoption_allowed": False,
        "same_attempt_recovery_allowed": False,
        "task_receipt_adoption_allowed": False,
        "receipt_publication_policy": "frozen-runner-task-receipt-last",
    }
    for field, value in checks.items():
        if release.get(field) != value:
            raise FreshRunError(f"activated release {field} mismatch")
    if release.get("active") is not True:
        raise FreshRunError("activated release active must be boolean true")
    if release.get("python_bytecode_write_disabled") is not True:
        raise FreshRunError("activated release bytecode gate must be boolean true")
    for field in (
        "resume_existing",
        "historical_adoption_allowed",
        "same_attempt_recovery_allowed",
        "task_receipt_adoption_allowed",
    ):
        if release.get(field) is not False:
            raise FreshRunError(f"activated release {field} must be boolean false")
    if not release_path.is_file() or release_path.is_symlink():
        raise FreshRunError("activated release missing or is a symlink")


def _load_frozen_runner(checkout: Path) -> Any:
    _verify_frozen_checkout(checkout)
    if any(name == "agentmemeval" or name.startswith("agentmemeval.") for name in sys.modules):
        raise FreshRunError("agentmemeval was imported before frozen-checkout verification")
    # A retryable pre-hand failure must not dirty the frozen checkout merely by
    # importing it.  Keep bytecode writes disabled for the whole launch process.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((checkout / "src").resolve()))
    runner = importlib.import_module("agentmemeval.experiments.formal_runner")
    expected_path = (
        checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    ).resolve()
    if Path(runner.__file__).resolve() != expected_path:
        raise FreshRunError("did not import the expected a1d1 frozen formal runner")
    if not hasattr(runner, "_semantic_config"):
        raise FreshRunError("frozen runner canonicalizer hook missing")
    return runner


def _acquire_launch_claim(
    *, release: dict[str, Any], release_path: Path, paths: dict[str, Path]
) -> tuple[Path, dict[str, Any]]:
    control_attempt = int(release["control_attempt"])
    ledger_root = Path(str(release["launch_ledger_root"])).resolve()
    launch_files = _launch_files(ledger_root, control_attempt)
    claim = {
        "schema_version": LAUNCH_CLAIM_SCHEMA,
        "status": "launching",
        "worker_id": WORKER_ID,
        "seed_bundle": SEED,
        "control_attempt": control_attempt,
        "release_id": release["release_id"],
        "activated_release_sha256": _sha256_file(release_path),
        "worker_manifest_sha256": P12_MANIFEST_SHA256,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "formal_runner_sha256": FROZEN_RUNNER_SHA256,
        "controller_sha256": release["controller_sha256"],
        "process_absence_evidence_sha256": release["process_absence_evidence_sha256"],
        "previous_launch_terminal_sha256": release["previous_launch_terminal_sha256"],
        "output_path": str(paths["output_path"]),
        "cache_path": str(paths["cache_path"]),
        "receipt_path": str(paths["receipt_path"]),
        "resume_existing": False,
        "historical_adoption_allowed": False,
        "same_attempt_recovery_allowed": False,
        "effect_fields_read": False,
    }
    _write_json_new(launch_files["claim"], claim)
    return launch_files["claim"], claim


def _publish_failed_launch(
    *,
    release: dict[str, Any],
    claim_path: Path,
    paths: dict[str, Path],
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    snapshot = _artifact_snapshot(paths)
    failed = {
        "schema_version": FAILED_LAUNCH_SCHEMA,
        "status": "failed_launch",
        "worker_id": WORKER_ID,
        "seed_bundle": SEED,
        "control_attempt": release["control_attempt"],
        "release_id": release["release_id"],
        "claim_sha256": _sha256_file(claim_path),
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:1000],
        "safe_retry_allowed": snapshot["safe_retry_allowed"],
        "isolation_required": not snapshot["safe_retry_allowed"],
        "artifact_snapshot": snapshot,
        "resume_existing": False,
        "effect_fields_read": False,
    }
    failed_path = _launch_files(
        Path(str(release["launch_ledger_root"])),
        int(release["control_attempt"]),
    )["failed"]
    _write_json_new(failed_path, failed)
    return failed


def _publish_completed_launch(
    *, release: dict[str, Any], claim_path: Path, result: dict[str, Any]
) -> None:
    completed = {
        "schema_version": COMPLETED_LAUNCH_SCHEMA,
        "status": "complete",
        "worker_id": WORKER_ID,
        "seed_bundle": SEED,
        "control_attempt": release["control_attempt"],
        "release_id": release["release_id"],
        "claim_sha256": _sha256_file(claim_path),
        "runner_status": result["status"],
        "run_dir": result["run_dir"],
        "resumed": False,
        "effect_fields_read": False,
    }
    completed_path = _launch_files(
        Path(str(release["launch_ledger_root"])),
        int(release["control_attempt"]),
    )["completed"]
    _write_json_new(completed_path, completed)


def execute_fresh_run(
    *,
    runner: Any,
    release_path: Path,
    manifest_path: Path,
    receipt_root: Path,
    frozen_checkout: Path,
    verifier_checkout: Path,
    controller_path: Path,
    process_absence_path: Path,
    control_attempt: int,
) -> dict[str, Any]:
    """Execute P12 once with corrected identity and no adoption/resume path."""

    release_path = release_path.resolve()
    release = _read_json(release_path)
    rebuilt = build_release(
        manifest_path=manifest_path,
        receipt_root=receipt_root,
        frozen_checkout=frozen_checkout,
        verifier_checkout=verifier_checkout,
        controller_path=controller_path,
        process_absence_path=process_absence_path,
        control_attempt=control_attempt,
    )
    _validate_active_release(release, release_path=release_path, rebuilt_draft=rebuilt)
    paths = _bound_paths(
        manifest=_validate_manifest(manifest_path),
        frozen_checkout=frozen_checkout.resolve(),
        receipt_root=receipt_root.resolve(),
    )
    claim_path, _claim = _acquire_launch_claim(
        release=release,
        release_path=release_path,
        paths=paths,
    )
    stage = "claim-acquired"
    original = None
    original_cwd = Path.cwd()
    patched = False
    try:
        # Reserve or reuse only the exact empty primary directory.  The claim
        # prevents concurrent controllers, so the frozen runner cannot race
        # into an automatically numbered attempt directory.
        output = paths["output_path"]
        if output.exists():
            if not output.is_dir() or output.is_symlink() or any(output.iterdir()):
                raise FreshRunError("claimed output is not an empty regular directory")
        else:
            output.mkdir(parents=True, exist_ok=False)
        stage = "output-reserved"
        snapshot = _artifact_snapshot(paths)
        if not snapshot["safe_retry_allowed"]:
            raise FreshRunError("artifact appeared after launch claim acquisition")
        original = runner._semantic_config
        runner._semantic_config = canonicalize_resolved_config_identity
        patched = True
        os.chdir(frozen_checkout.resolve())
        stage = "runner-invoked"
        result = runner.run_worker_manifest(
            manifest_path.resolve(),
            receipt_root=receipt_root.resolve(),
            resume_existing=False,
        )
        stage = "runner-returned-validation"
        if not isinstance(result, dict):
            raise FreshRunError("frozen runner returned a non-object result")
        if result.get("resumed") is not False:
            raise FreshRunError("frozen runner unexpectedly resumed historical output")
        if Path(str(result.get("run_dir", ""))).resolve() != paths["output_path"]:
            raise FreshRunError("frozen runner selected a non-primary attempt directory")
        if result.get("status") != "complete":
            raise FreshRunError("frozen runner did not complete P12")
        _publish_completed_launch(release=release, claim_path=claim_path, result=result)
        return result
    except BaseException as exc:
        _publish_failed_launch(
            release=release,
            claim_path=claim_path,
            paths=paths,
            stage=stage,
            error=exc,
        )
        raise
    finally:
        if patched:
            runner._semantic_config = original
        os.chdir(original_cwd)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-checkout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--verifier-checkout", type=Path, required=True)
    parser.add_argument("--process-absence-evidence", type=Path, required=True)
    parser.add_argument("--control-attempt", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-release", type=Path)
    mode.add_argument("--activate-release", type=Path)
    mode.add_argument("--release", type=Path)
    parser.add_argument("--activated-release-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kwargs = {
        "manifest_path": args.manifest,
        "receipt_root": args.receipt_root,
        "frozen_checkout": args.frozen_checkout,
        "verifier_checkout": args.verifier_checkout,
        "controller_path": Path(__file__),
        "process_absence_path": args.process_absence_evidence,
        "control_attempt": args.control_attempt,
    }
    if args.build_release is not None:
        draft = build_release(**kwargs)
        _write_json_new(args.build_release.resolve(), draft)
        result = draft
    elif args.activate_release is not None:
        if args.activated_release_output is None:
            raise FreshRunError("--activated-release-output is required for activation")
        draft = build_release(**kwargs)
        result = activate_release(
            draft_path=args.activate_release,
            rebuilt_draft=draft,
            output_path=args.activated_release_output,
        )
    else:
        if args.activated_release_output is not None:
            raise FreshRunError("--activated-release-output is only valid with --activate-release")
        release = _read_json(args.release.resolve())
        rebuilt = build_release(**kwargs)
        _validate_active_release(
            release, release_path=args.release.resolve(), rebuilt_draft=rebuilt
        )
        runner = _load_frozen_runner(args.frozen_checkout.resolve())
        result = execute_fresh_run(
            runner=runner,
            release_path=args.release,
            **kwargs,
        )
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
