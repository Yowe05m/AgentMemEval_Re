"""Deterministic task shards for six canonical TASK8B logical seed workers.

The canonical P07--P12/S07--S12 manifests remain authoritative.  Execution
shards are engineering subdivisions of those 12 logical workers, not extra
scientific workers or seeds.  Composition validates structural hand counts,
task receipts, and file hashes without reading scientific metric values.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agentmemeval.config.loader import ConfigError
from agentmemeval.experiments import formal_runner
from agentmemeval.experiments.formal_protocol import (
    sha256_json,
    task8b_embedding_fingerprint,
)

AUTHORIZATION_SCHEMA = "task8b-paired-shard-authorization-v1"
SHARD_RECEIPT_SCHEMA = "task8b-paired-shard-receipt-v1"
COMPOSITION_SCHEMA = "task8b-paired-shard-composition-v1"
PRIMARY_BRIDGE_SCHEMA = "task8b-primary-checkpoint-bridge-v1"
COMPLETED_RECOVERY_BASELINE_SCHEMA = (
    "task8b-paired-completed-recovery-baseline-v1"
)
COMPLETED_RECOVERY_CERTIFICATE_SCHEMA = (
    "task8b-paired-completed-recovery-certificate-v1"
)
COMPLETED_RECOVERY_LEDGER_SCHEMA = "task8b-paired-recovery-ledger-entry-v1"
COMPLETED_RECOVERY_REASON = "resolved-config-integer-key-canonicalization"
RECOVERY_SOURCE_CONTROLLER_SHA256 = (
    "ac1a57480ab8b6366c9f00356cc3a1ec7a512b5716c5c282d51e42ddebed7f8c"
)
RECOVERY_COMPLETION_CONTROLLER_SHA256 = (
    "224cfc525abb97760f53d9694173b63cbe94e1151ad47c411162da6f4535d751"
)
RECOVERY_COMPLETION_CONTROLLER_SHA256S = {
    RECOVERY_COMPLETION_CONTROLLER_SHA256,
    "3fd0b92db3928c45f59a22ca0c68d4fabb9321e7dac01033a2de7d8240c9f110",
}
SEALED_SOURCE_CONTROLLER_SHA256S = {
    RECOVERY_SOURCE_CONTROLLER_SHA256,
    "736f67d95812f872dda4ceff16702301ecad81669e6088484e104587591d9e29",
    "eef9bdbf16643fae40dcb32ef439ef5ad1154765b824c3bc6dfacaad3d830b87",
}
HEALTH_ZERO_FIELDS = (
    "fallback_count",
    "memory_revision_fallback_count",
    "reward_conservation_violation_count",
    "stack_conservation_violation_count",
)
AMENDMENT_ID = "TASK8B-SIX-SEED-PAIR-SHARD-20260723"
EXPECTED_AMENDMENT_SHA256 = (
    "041922a51234bfb9fd83d2e2f0a1cefb7736189fa6693ba7e9c4df39a0b6e2be"
)
FROZEN_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
FROZEN_FORMAL_RUNNER_SHA256 = (
    "c4b601ff0de2c27a57ee246efcf91d21f502f27c652d20fd6fa7cfd925a17d5e"
)
FORMAL_SEEDS = tuple(range(2026090107, 2026090113))
PAIR_MAPPING = {
    seed: (f"P{seed - 2026090100:02d}", f"S{seed - 2026090100:02d}")
    for seed in FORMAL_SEEDS
}
PAIR_IDS = {seed: f"pair_{seed - 2026090100:02d}" for seed in FORMAL_SEEDS}
PHYSICAL_MAPPING = {
    seed: {
        "low": f"H{seed - 2026090106:02d}",
        "high": f"H{seed - 2026090100:02d}",
    }
    for seed in FORMAL_SEEDS
}
EXPECTED_TASKS = {
    "primary": {
        "isolation_no_memory": 1350,
        "isolation_fact": 1350,
        "isolation_expr": 1350,
        "isolation_sync": 1350,
        "isolation_async": 1350,
    },
    "secondary": {
        "mixed_ecological": 2700,
        "expr_online": 600,
        "expr_without": 600,
        "async_online": 600,
        "async_without": 600,
    },
}
SIDE_TASKS = {
    ("primary", "high"): {
        "isolation_no_memory",
        "isolation_fact",
        "isolation_expr",
        "isolation_sync",
    },
    ("primary", "low"): {"isolation_async"},
    ("secondary", "high"): {
        "expr_online",
        "expr_without",
        "async_online",
        "async_without",
    },
    ("secondary", "low"): {"mixed_ecological"},
}
AUTHORIZATION_FIELDS = {
    "schema_version",
    "active",
    "authorization_id",
    "amendment_id",
    "amendment_path",
    "amendment_sha256",
    "scientific_checkout",
    "frozen_code_sha",
    "frozen_formal_runner_sha256",
    "engineering_controller_sha256",
    "approved_staging_root",
    "approved_receipt_root",
    "denied_partial_root",
    "pair_id",
    "physical_slot",
    "shard_role",
    "partition_id",
    "canonical_manifest_sha256",
    "worker_id",
    "seed",
    "shard_id",
    "seal_mode",
    "selected_task_ids",
    "derived_output_path",
    "derived_cache_namespace",
    "derived_receipt_relative_path",
    "primary_bridge_root",
    "primary_bridge_receipt_relative_path",
    "primary_bridge_receipt_sha256",
    "effect_fields_read",
    "scientific_protocol_changed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"paired-shard JSON 缺失或为 symlink：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"paired-shard JSON 非法：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"paired-shard JSON 顶层必须为对象：{path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise ConfigError(f"paired-shard 拒绝 symlink 输出：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ConfigError(f"paired-shard 拒绝覆盖：{path}") from exc


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ConfigError(f"paired-shard task root 非法：{root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ConfigError(f"paired-shard task tree 含 symlink：{path}")
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def _verify_recovery_archive(
    archive: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    expected = {
        str(row["relative_path"]): {
            "size": int(row["size"]),
            "sha256": str(row["sha256"]),
        }
        for row in expected_rows
    }
    if len(expected) != len(expected_rows):
        raise ConfigError("paired recovery baseline file path 重复")
    observed: dict[str, dict[str, Any]] = {}
    prefix_mode: bool | None = None
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            for member in handle.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    not member.name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or "." in pure.parts
                    or "\\" in member.name
                ):
                    raise ConfigError("paired recovery archive member path 非法")
                if member.isdir():
                    continue
                has_output_prefix = pure.parts[0] == "output"
                if prefix_mode is None:
                    prefix_mode = has_output_prefix
                elif prefix_mode != has_output_prefix:
                    raise ConfigError(
                        "paired recovery archive member root prefix 不一致"
                    )
                if has_output_prefix:
                    if len(pure.parts) == 1:
                        raise ConfigError(
                            "paired recovery archive output member path 非法"
                        )
                    normalized_name = PurePosixPath(*pure.parts[1:]).as_posix()
                else:
                    normalized_name = pure.as_posix()
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or normalized_name in observed
                ):
                    raise ConfigError("paired recovery archive member type/duplicate 非法")
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ConfigError("paired recovery archive member 无法读取")
                digest = hashlib.sha256()
                size = 0
                while True:
                    block = extracted.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
                observed[normalized_name] = {
                    "size": size,
                    "sha256": digest.hexdigest(),
                }
    except (OSError, tarfile.TarError) as exc:
        raise ConfigError("paired recovery pre-recovery archive 非法") from exc
    if observed != expected:
        raise ConfigError("paired recovery archive 与 baseline files 不闭合")


def _verify_files_tsv(root: Path) -> list[dict[str, Any]]:
    """Recompute every files.tsv row and require complete manifest coverage."""

    files_tsv = root / "files.tsv"
    if not files_tsv.is_file() or files_tsv.is_symlink():
        raise ConfigError("paired-shard files.tsv 缺失或为 symlink")
    try:
        with files_tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["relative_path", "size", "sha256"]:
                raise ConfigError("paired-shard files.tsv header 非法")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError("paired-shard files.tsv 无法解析") from exc
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if set(raw) != {"relative_path", "size", "sha256"}:
            raise ConfigError("paired-shard files.tsv row 字段非法")
        relative = str(raw["relative_path"])
        if relative in seen:
            raise ConfigError("paired-shard files.tsv 路径重复")
        seen.add(relative)
        path = _strict_relative_target(
            root,
            relative,
            label="files.tsv relative_path",
            must_exist=True,
        )
        if not path.is_file() or path.is_symlink():
            raise ConfigError(f"paired-shard files.tsv 非普通文件：{relative}")
        try:
            size = int(str(raw["size"]))
        except ValueError as exc:
            raise ConfigError("paired-shard files.tsv size 非整数") from exc
        digest = str(raw["sha256"])
        if (
            size < 0
            or len(digest) != 64
            or path.stat().st_size != size
            or _sha256(path) != digest
        ):
            raise ConfigError(f"paired-shard files.tsv 文件不匹配：{relative}")
        rows.append({"relative_path": relative, "size": size, "sha256": digest})
    tree_items = list(root.rglob("*"))
    symlinks = [path for path in tree_items if path.is_symlink()]
    if symlinks:
        raise ConfigError(f"paired-shard files.tsv tree 含 symlink：{symlinks[0]}")
    excluded = {"state.tsv", "files.tsv", "completion_receipt.json"}
    expected = {
        path.relative_to(root).as_posix()
        for path in tree_items
        if path.is_file()
        and path.relative_to(root).as_posix() not in excluded
    }
    if seen != expected:
        raise ConfigError("paired-shard files.tsv coverage 不完整")
    return rows


def _structural_hand_count(root: Path) -> int:
    path = root / "hand_summaries.jsonl"
    if not path.is_file() or path.is_symlink():
        raise ConfigError("paired-shard hand_summaries.jsonl 缺失或为 symlink")
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise ConfigError("paired-shard hand_summaries.jsonl 含空行")
            count += 1
    return count


def _verify_scientific_checkout(checkout: Path) -> Path:
    checkout = Path(os.path.abspath(checkout))
    runner_path = (
        checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    )
    if (
        not checkout.is_dir()
        or checkout.is_symlink()
        or checkout.resolve() != checkout
        or not runner_path.is_file()
        or runner_path.is_symlink()
    ):
        raise ConfigError("paired-shard scientific checkout/runner 缺失")
    commands = (
        (
            ["git", "-c", f"safe.directory={checkout}", "rev-parse", "HEAD"],
            FROZEN_CODE_SHA,
        ),
        (
            [
                "git",
                "-c",
                f"safe.directory={checkout}",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            "",
        ),
    )
    for argv, expected in commands:
        try:
            observed = subprocess.run(
                argv,
                cwd=str(checkout),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ConfigError("paired-shard scientific Git identity probe failed") from exc
        if observed != expected:
            raise ConfigError("paired-shard scientific HEAD/clean gate failed")
    if _sha256(runner_path) != FROZEN_FORMAL_RUNNER_SHA256:
        raise ConfigError("paired-shard frozen formal_runner SHA mismatch")
    return runner_path


_FROZEN_RUNNER_BOOTSTRAP = """\
import json
import pathlib
import sys

checkout = pathlib.Path(sys.argv[1]).resolve()
checkout_src = (checkout / "src").resolve()
sys.path.insert(0, str(checkout_src))
from agentmemeval.experiments import formal_runner

loaded = pathlib.Path(formal_runner.__file__).resolve()
if checkout_src not in loaded.parents:
    raise RuntimeError("formal_runner import escaped frozen scientific checkout")
result = formal_runner.run_worker_manifest(
    pathlib.Path(sys.argv[2]),
    receipt_root=pathlib.Path(sys.argv[3]),
    resume_existing=False,
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
"""

_FROZEN_RECOVERY_RESUME_BOOTSTRAP = """\
import copy
import hashlib
import json
import pathlib
import sys

checkout = pathlib.Path(sys.argv[1]).resolve()
checkout_src = (checkout / "src").resolve()
sys.path.insert(0, str(checkout_src))
from agentmemeval.experiments import formal_runner

loaded = pathlib.Path(formal_runner.__file__).resolve()
if checkout_src not in loaded.parents:
    raise RuntimeError("formal_runner import escaped frozen scientific checkout")

def preserving_mapping_keys(config):
    value = copy.deepcopy(config)
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in ("output_root", "run_id", "initial_memory_snapshots", "admission_audit"):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value

formal_runner._semantic_config = preserving_mapping_keys
original_write_json_same_or_new = formal_runner._write_json_same_or_new
legacy_audit_raw_path = pathlib.Path(sys.argv[4]) if sys.argv[4] else None
legacy_audit_sha256 = sys.argv[5] or None
legacy_receipt_config_sha256 = sys.argv[6] or None
corrected_receipt_config_sha256 = sys.argv[7] or None

def has_symlink_component(path):
    return any(candidate.is_symlink() for candidate in (path, *path.parents))

if legacy_audit_raw_path is not None:
    if (
        not legacy_audit_raw_path.is_file()
        or has_symlink_component(legacy_audit_raw_path)
    ):
        raise RuntimeError("authorized legacy recovery audit path unsafe")
    legacy_audit_path = legacy_audit_raw_path.resolve(strict=True)
else:
    legacy_audit_path = None

def write_json_with_legacy_recovery_audit_bridge(path, value):
    if has_symlink_component(path):
        return original_write_json_same_or_new(path, value)
    resolved = path.resolve(strict=True) if path.exists() else path.resolve()
    if legacy_audit_path is not None and resolved == legacy_audit_path:
        if (
            legacy_audit_raw_path is None
            or not legacy_audit_raw_path.is_file()
            or has_symlink_component(legacy_audit_raw_path)
            or legacy_audit_raw_path.resolve(strict=True) != legacy_audit_path
            or not path.is_file()
        ):
            raise RuntimeError("authorized legacy recovery audit missing")
        observed_sha256 = hashlib.sha256(
            legacy_audit_raw_path.read_bytes()
        ).hexdigest()
        if observed_sha256 != legacy_audit_sha256:
            raise RuntimeError("authorized legacy recovery audit SHA changed")
        existing = json.loads(path.read_text(encoding="utf-8"))
        legacy_fields = {
            "schema_version",
            "task_id",
            "protocol_status",
            "actual",
            "expected",
            "identity_correction",
            "health",
            "status",
            "effect_fields_read",
        }
        if set(existing) == legacy_fields:
            if (
                existing["schema_version"] != value.get("schema_version")
                or existing["task_id"] != value.get("task_id")
                or existing["protocol_status"] != value.get("protocol_status")
                or existing["actual"] != value.get("actual")
                or existing["status"] != "verified-recovered"
                or existing["effect_fields_read"] is not False
            ):
                raise RuntimeError("legacy recovery audit bridge mismatch")
            return
        raise RuntimeError("authorized legacy recovery audit schema mismatch")
    original_write_json_same_or_new(path, value)

formal_runner._write_json_same_or_new = write_json_with_legacy_recovery_audit_bridge
original_read_json_object = formal_runner._read_json_object
original_verify_completed_task_identity = (
    formal_runner._verify_completed_task_identity
)

def recovered_identity(identity):
    if legacy_receipt_config_sha256 is None:
        return identity
    observed = identity.get("resolved_config_sha256")
    if observed == corrected_receipt_config_sha256:
        return identity
    if observed != legacy_receipt_config_sha256:
        return identity
    return {
        **identity,
        "resolved_config_sha256": corrected_receipt_config_sha256,
    }

def read_json_with_recovered_task_receipt_bridge(path):
    value = original_read_json_object(path)
    if (
        legacy_receipt_config_sha256 is None
        or path.parent.name != "task_receipts"
        or value.get("schema_version") != "task8-worker-task-receipt-v1"
    ):
        return value
    task_row = value.get("task_row")
    if not isinstance(task_row, dict):
        return value
    identity_audit = task_row.get("identity_audit")
    if not isinstance(identity_audit, dict):
        return value
    bridged = recovered_identity(identity_audit)
    if bridged is identity_audit:
        return value
    result = copy.deepcopy(value)
    result["task_row"]["identity_audit"] = bridged
    return result

def verify_completed_task_identity_with_recovered_config(**kwargs):
    identity = original_verify_completed_task_identity(**kwargs)
    return recovered_identity(identity)

formal_runner._read_json_object = read_json_with_recovered_task_receipt_bridge
formal_runner._verify_completed_task_identity = (
    verify_completed_task_identity_with_recovered_config
)
original_receipt_identity = formal_runner._receipt_identity

def receipt_identity_with_recovered_config(manifest, *, consumer=False):
    identity = original_receipt_identity(manifest, consumer=consumer)
    if legacy_receipt_config_sha256 is None:
        return identity
    observed = identity.get("resolved_config_sha256")
    if observed == corrected_receipt_config_sha256:
        return identity
    if observed != legacy_receipt_config_sha256:
        raise RuntimeError("legacy checkpoint receipt identity mismatch")
    return {
        **identity,
        "resolved_config_sha256": corrected_receipt_config_sha256,
    }

formal_runner._receipt_identity = receipt_identity_with_recovered_config
result = formal_runner.run_worker_manifest(
    pathlib.Path(sys.argv[2]),
    receipt_root=pathlib.Path(sys.argv[3]),
    resume_existing=True,
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
"""


def _safe_relative(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ConfigError(f"{label} 必须是安全相对路径")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.abspath(left))
    right_text = os.path.normcase(os.path.abspath(right))
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common in {left_text, right_text}


def _approved_root(value: str, *, label: str) -> Path:
    root = Path(value)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ConfigError(f"{label} 必须是既有绝对非-symlink目录")
    absolute = Path(os.path.abspath(root))
    _deny_symlink_components(absolute, label=label)
    return absolute


def _deny_symlink_components(path: Path, *, label: str) -> None:
    """Reject any existing symlink component before calling resolve()."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ConfigError(f"{label} 含 symlink：{current}")


def _inside_approved(root: Path, target: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(target))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{label} 逃逸 approved root") from exc
    _deny_symlink_components(root, label=label)
    _deny_symlink_components(absolute, label=label)
    return absolute


def _strict_relative_target(
    root: Path,
    value: str,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    relative = _safe_relative(value, label=label)
    candidate = _inside_approved(root, root / relative, label=label)
    if must_exist and not candidate.exists():
        raise ConfigError(f"{label} 缺失：{value}")
    return candidate


def _reserve_execution(root: Path, identity: dict[str, Any]) -> Path:
    reservation_root = root / ".reservations"
    reservation_root.mkdir(exist_ok=True)
    if reservation_root.is_symlink():
        raise ConfigError("paired-shard reservation root 为 symlink")
    material = _json_bytes(identity)
    path = reservation_root / f"{hashlib.sha256(material).hexdigest()}.lock"
    try:
        with path.open("xb") as handle:
            handle.write(material)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ConfigError("paired-shard execution 已被原子 reservation") from exc
    return path


def _verify_amendment(path: Path, declared_sha256: str) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or declared_sha256 != EXPECTED_AMENDMENT_SHA256
        or _sha256(path) != EXPECTED_AMENDMENT_SHA256
    ):
        raise ConfigError("paired-shard amendment SHA binding failed")


def _worker_seed_role(manifest: dict[str, Any]) -> tuple[str, int, str]:
    worker_id = str(manifest.get("worker_id", ""))
    seed = int(manifest.get("seed_bundle", -1))
    role = str(manifest.get("role", ""))
    expected_pair = PAIR_MAPPING.get(seed)
    if (
        expected_pair is None
        or not worker_id
        or worker_id not in expected_pair
        or worker_id[0] != ("P" if role == "primary" else "S")
    ):
        raise ConfigError("paired-shard 仅允许 P07-P12/S07-S12 canonical identity")
    formal_runner.validate_worker_manifest(manifest, allow_candidate=False)
    return worker_id, seed, role


def _canonical_identity(manifest: dict[str, Any]) -> tuple[str, int, str]:
    worker_id, seed, role = _worker_seed_role(manifest)
    expected = EXPECTED_TASKS[role]
    tasks = manifest.get("task_configs")
    if not isinstance(tasks, list):
        raise ConfigError("canonical task_configs 非法")
    by_id = {
        str(task.get("task_id", "")): task
        for task in tasks
        if isinstance(task, dict)
    }
    if len(by_id) != len(tasks) or set(by_id) != set(expected):
        raise ConfigError("canonical paired-shard task topology 不匹配")
    for task_id, hands in expected.items():
        if int(by_id[task_id].get("planned_hands", -1)) != hands:
            raise ConfigError(f"canonical task hands 不匹配：{task_id}")
    return worker_id, seed, role


def _authorization_id(
    authorization: dict[str, Any],
    *,
    role: str,
) -> str:
    identity_material = {
        "amendment_sha256": authorization["amendment_sha256"],
        "canonical_manifest_sha256": authorization["canonical_manifest_sha256"],
        "controller_sha256": authorization["engineering_controller_sha256"],
        "pair_id": authorization["pair_id"],
        "physical_slot": authorization["physical_slot"],
        "role": role,
        "seed": authorization["seed"],
        "selected_task_ids": authorization["selected_task_ids"],
        "shard_id": authorization["shard_id"],
        "shard_role": authorization["shard_role"],
        "worker_id": authorization["worker_id"],
    }
    return "auth-" + hashlib.sha256(_json_bytes(identity_material)).hexdigest()


def render_authorization(
    canonical_manifest_path: str | Path,
    amendment_path: str | Path,
    *,
    physical_slot: str,
    shard_role: str,
    selected_task_ids: list[str],
    scientific_checkout: str | Path,
    approved_staging_root: str | Path,
    approved_receipt_root: str | Path,
    shard_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render one deterministic, non-overwriting paired-shard authorization."""

    canonical_path = Path(os.path.abspath(canonical_manifest_path))
    amendment = Path(os.path.abspath(amendment_path))
    checkout = Path(os.path.abspath(scientific_checkout))
    staging_root = _approved_root(
        str(approved_staging_root),
        label="approved_staging_root",
    )
    receipt_root = _approved_root(
        str(approved_receipt_root),
        label="approved_receipt_root",
    )
    canonical = _read_json(canonical_path)
    worker_id, seed, role = _canonical_identity(canonical)
    if role != "primary":
        raise ConfigError(
            "authorization renderer 当前只允许 primary；secondary 必须先绑定 bridge"
        )
    _verify_amendment(amendment, EXPECTED_AMENDMENT_SHA256)
    _verify_scientific_checkout(checkout)
    if shard_role not in {"low", "high"}:
        raise ConfigError("authorization renderer shard_role 必须为 low/high")
    if physical_slot != PHYSICAL_MAPPING[seed][shard_role]:
        raise ConfigError("authorization renderer physical_slot 与 pair 不匹配")
    selected = [str(task_id) for task_id in selected_task_ids]
    canonical_order = [
        str(task["task_id"]) for task in canonical["task_configs"]
    ]
    if (
        not selected
        or len(selected) != len(set(selected))
        or selected
        != [task_id for task_id in canonical_order if task_id in set(selected)]
        or not set(selected).issubset(SIDE_TASKS[(role, shard_role)])
    ):
        raise ConfigError("authorization renderer selected tasks/顺序/side 非法")
    if not shard_id.strip() or Path(shard_id).name != shard_id:
        raise ConfigError("authorization renderer shard_id 非法")
    canonical_output = Path(str(canonical["instance_identity"]["output_path"]))
    if not canonical_output.is_absolute():
        canonical_output = checkout / canonical_output
    derived_base = staging_root / "paired_shards" / worker_id / shard_id
    derived_output = _inside_approved(
        staging_root,
        derived_base / "output",
        label="derived_output_path",
    )
    derived_cache = _inside_approved(
        staging_root,
        derived_base / "cache",
        label="derived_cache_namespace",
    )
    receipt_relative = (
        f"paired_shards/receipts/{worker_id}/{shard_id}.json"
    )
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "active": True,
        "authorization_id": "",
        "amendment_id": AMENDMENT_ID,
        "amendment_path": str(amendment),
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "scientific_checkout": str(checkout),
        "frozen_code_sha": FROZEN_CODE_SHA,
        "frozen_formal_runner_sha256": FROZEN_FORMAL_RUNNER_SHA256,
        "engineering_controller_sha256": _sha256(Path(__file__).resolve()),
        "approved_staging_root": str(staging_root),
        "approved_receipt_root": str(receipt_root),
        "denied_partial_root": str(Path(os.path.abspath(canonical_output))),
        "pair_id": PAIR_IDS[seed],
        "physical_slot": physical_slot,
        "shard_role": shard_role,
        "partition_id": f"{PAIR_IDS[seed]}-{role}-{shard_role}",
        "canonical_manifest_sha256": _sha256(canonical_path),
        "worker_id": worker_id,
        "seed": seed,
        "shard_id": shard_id,
        "seal_mode": "execution",
        "selected_task_ids": selected,
        "derived_output_path": str(derived_output),
        "derived_cache_namespace": str(derived_cache),
        "derived_receipt_relative_path": receipt_relative,
        "primary_bridge_root": None,
        "primary_bridge_receipt_relative_path": None,
        "primary_bridge_receipt_sha256": None,
        "effect_fields_read": False,
        "scientific_protocol_changed": False,
    }
    authorization["authorization_id"] = _authorization_id(
        authorization,
        role=role,
    )
    _write_json_new(Path(os.path.abspath(output_path)), authorization)
    return authorization


def preflight_authorization(
    canonical_manifest_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Validate and derive one shard without publishing files or running hands."""

    derived = derive_authorized_manifest(
        canonical_manifest_path,
        authorization_path,
    )
    shard = dict(derived["paired_shard"])
    _verify_scientific_checkout(Path(str(shard["scientific_checkout"])))
    output = Path(str(derived["instance_identity"]["output_path"]))
    cache = Path(str(derived["instance_identity"]["cache_namespace"]))
    receipt = (
        Path(str(shard["approved_receipt_root"]))
        / str(derived["receipt_relative_path"])
    )
    if output.exists() or output.is_symlink():
        raise ConfigError("preflight 拒绝既有 derived output")
    if cache.exists() or cache.is_symlink():
        raise ConfigError("preflight 拒绝既有 derived cache")
    if derived["role"] == "primary" and (
        receipt.exists() or receipt.is_symlink()
    ):
        raise ConfigError("preflight 拒绝既有 primary shard receipt")
    return {
        "status": "preflight_pass",
        "authorization_sha256": shard["authorization_sha256"],
        "worker_id": derived["worker_id"],
        "seed": derived["seed_bundle"],
        "pair_id": shard["pair_id"],
        "physical_slot": shard["physical_slot"],
        "shard_role": shard["shard_role"],
        "selected_task_ids": shard["selected_task_ids"],
        "derived_output_path": str(output),
        "derived_cache_namespace": str(cache),
        "effect_fields_read": False,
        "hands_started": 0,
    }


def derive_authorized_manifest(
    canonical_manifest_path: str | Path,
    authorization_path: str | Path,
    *,
    _recovery_source_controller_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive one fail-closed staging manifest from an active authorization."""

    canonical_path = Path(os.path.abspath(canonical_manifest_path))
    auth_path = Path(os.path.abspath(authorization_path))
    canonical = _read_json(canonical_path)
    authorization = _read_json(auth_path)
    worker_id, seed, role = _canonical_identity(canonical)
    missing = sorted(AUTHORIZATION_FIELDS - set(authorization))
    if missing:
        raise ConfigError(f"paired-shard authorization 缺字段：{', '.join(missing)}")
    expected_controller_sha256 = _sha256(Path(__file__).resolve())
    if _recovery_source_controller_sha256 is not None:
        if (
            _recovery_source_controller_sha256
            != RECOVERY_SOURCE_CONTROLLER_SHA256
        ):
            raise ConfigError("paired-shard recovery source controller 非法")
        expected_controller_sha256 = RECOVERY_SOURCE_CONTROLLER_SHA256
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("active") is not True
        or authorization.get("amendment_id") != AMENDMENT_ID
        or authorization.get("frozen_code_sha") != FROZEN_CODE_SHA
        or authorization.get("frozen_formal_runner_sha256")
        != FROZEN_FORMAL_RUNNER_SHA256
        or authorization.get("engineering_controller_sha256")
        != expected_controller_sha256
        or authorization.get("pair_id") != PAIR_IDS[seed]
        or authorization.get("worker_id") != worker_id
        or int(authorization.get("seed", -1)) != seed
        or authorization.get("canonical_manifest_sha256") != _sha256(canonical_path)
        or authorization.get("effect_fields_read") is not False
        or authorization.get("scientific_protocol_changed") is not False
        or not str(authorization.get("authorization_id", "")).strip()
        or not str(authorization.get("partition_id", "")).strip()
        or not str(authorization.get("shard_id", "")).strip()
    ):
        raise ConfigError("paired-shard authorization identity/active gate failed")
    scientific_checkout = Path(
        os.path.abspath(str(authorization["scientific_checkout"]))
    )
    if not scientific_checkout.is_dir() or scientific_checkout.is_symlink():
        raise ConfigError("paired-shard scientific_checkout binding failed")
    amendment_path = Path(os.path.abspath(str(authorization["amendment_path"])))
    _verify_amendment(
        amendment_path,
        str(authorization["amendment_sha256"]),
    )
    selected = authorization.get("selected_task_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(str(item) for item in selected))
    ):
        raise ConfigError("paired-shard selected_task_ids 必须非空且无重复")
    canonical_tasks = list(canonical["task_configs"])
    canonical_order = [str(task["task_id"]) for task in canonical_tasks]
    selected_ids = [str(item) for item in selected]
    if set(selected_ids) - set(canonical_order):
        raise ConfigError("paired-shard authorization 含未知 task")
    if selected_ids != [task_id for task_id in canonical_order if task_id in selected_ids]:
        raise ConfigError("paired-shard task 顺序必须与 canonical manifest 一致")
    shard_role = str(authorization.get("shard_role", ""))
    seal_mode = str(authorization.get("seal_mode", ""))
    allowed_side_tasks = SIDE_TASKS.get((role, shard_role))
    if (
        allowed_side_tasks is None
        or not set(selected_ids).issubset(allowed_side_tasks)
        or authorization.get("physical_slot")
        != PHYSICAL_MAPPING[seed].get(shard_role)
        or seal_mode not in {"execution", "historical_adoption"}
    ):
        raise ConfigError("paired-shard high/low task mapping 不匹配")
    if authorization.get("authorization_id") != _authorization_id(
        authorization,
        role=role,
    ):
        raise ConfigError("paired-shard authorization_id identity binding failed")
    if seal_mode == "historical_adoption" and (role != "primary" or shard_role != "high"):
        raise ConfigError("historical adoption 仅允许 primary high 历史任务")

    canonical_instance = dict(canonical["instance_identity"])
    staging_root = _approved_root(
        str(authorization["approved_staging_root"]),
        label="approved_staging_root",
    )
    receipt_root = _approved_root(
        str(authorization["approved_receipt_root"]),
        label="approved_receipt_root",
    )
    output_path = _inside_approved(
        staging_root,
        Path(str(authorization["derived_output_path"])),
        label="derived_output_path",
    )
    cache_path = _inside_approved(
        staging_root,
        Path(str(authorization["derived_cache_namespace"])),
        label="derived_cache_namespace",
    )
    output = str(output_path)
    cache = str(cache_path)
    receipt = str(authorization["derived_receipt_relative_path"])
    _safe_relative(receipt, label="derived_receipt_relative_path")
    _inside_approved(
        receipt_root,
        receipt_root / receipt,
        label="derived_receipt_relative_path",
    )
    canonical_output = Path(str(canonical_instance["output_path"]))
    if not canonical_output.is_absolute():
        canonical_output = scientific_checkout / canonical_output
    denied_partial = Path(str(authorization["denied_partial_root"]))
    if not denied_partial.is_absolute():
        denied_partial = scientific_checkout / denied_partial
    if denied_partial.absolute() != canonical_output.absolute():
        raise ConfigError("paired-shard denied_partial_root 未绑定 canonical output")
    if _paths_overlap(output_path, canonical_output):
        raise ConfigError("paired-shard output 与 canonical output 重叠")
    canonical_cache = Path(str(canonical_instance["cache_namespace"]))
    if not canonical_cache.is_absolute():
        canonical_cache = scientific_checkout / canonical_cache
    if _paths_overlap(cache_path, canonical_cache):
        raise ConfigError("paired-shard cache 与 canonical cache 重叠")
    if _paths_overlap(output_path, cache_path):
        raise ConfigError("paired-shard output/cache 不得重叠")
    if role == "primary" and receipt == str(canonical["receipt_relative_path"]):
        raise ConfigError("primary shard receipt 不得覆盖 canonical receipt")
    bridge_root: Path | None = None
    bridge_receipt: Path | None = None
    if role == "secondary":
        bridge_root_value = authorization.get("primary_bridge_root")
        bridge_receipt_relative = authorization.get(
            "primary_bridge_receipt_relative_path"
        )
        bridge_receipt_sha256 = authorization.get("primary_bridge_receipt_sha256")
        if (
            not isinstance(bridge_root_value, str)
            or not isinstance(bridge_receipt_relative, str)
            or not isinstance(bridge_receipt_sha256, str)
        ):
            raise ConfigError("secondary shard 缺 primary bridge binding")
        bridge_root = _inside_approved(
            staging_root,
            Path(bridge_root_value),
            label="primary_bridge_root",
        )
        if not bridge_root.is_dir() or bridge_root.is_symlink():
            raise ConfigError("secondary primary bridge root 非法")
        bridge_receipt = _strict_relative_target(
            receipt_root,
            bridge_receipt_relative,
            label="primary_bridge_receipt_relative_path",
            must_exist=True,
        )
        if (
            bridge_receipt.is_symlink()
            or _sha256(bridge_receipt) != bridge_receipt_sha256
            or receipt != bridge_receipt_relative
        ):
            raise ConfigError("secondary primary bridge receipt binding failed")
        bridge_manifest = _read_json(bridge_root / "bridge_manifest.json")
        if (
            bridge_manifest.get("schema_version") != PRIMARY_BRIDGE_SCHEMA
            or bridge_manifest.get("status") != "files_sealed_receipt_pending"
            or bridge_manifest.get("worker_id") != canonical["depends_on"]
            or int(bridge_manifest.get("seed", -1)) != seed
            or bridge_manifest.get("pair_id") != PAIR_IDS[seed]
            or bridge_manifest.get("physical_mapping") != PHYSICAL_MAPPING[seed]
            or bridge_manifest.get("amendment_id") != AMENDMENT_ID
            or bridge_manifest.get("amendment_sha256")
            != EXPECTED_AMENDMENT_SHA256
            or bridge_manifest.get("frozen_code_sha") != FROZEN_CODE_SHA
            or bridge_manifest.get("frozen_formal_runner_sha256")
            != FROZEN_FORMAL_RUNNER_SHA256
            or bridge_manifest.get("engineering_controller_sha256")
            not in (
                SEALED_SOURCE_CONTROLLER_SHA256S
                | {_sha256(Path(__file__).resolve())}
            )
            or bridge_manifest.get("task_union")
            != list(EXPECTED_TASKS["primary"])
            or bridge_manifest.get("effect_fields_read") is not False
        ):
            raise ConfigError("secondary primary bridge manifest identity failed")
        verified_bridge_receipt = formal_runner.verify_checkpoint_receipt(
            bridge_receipt,
            bridge_root,
            expected_identity=dict(canonical["dependency_receipt_identity"]),
            expected_producer_worker_id=str(canonical["depends_on"]),
            expected_seed_bundle=seed,
            expected_checkpoint_hand=int(canonical["checkpoint_set"][-1]),
        )
        allowed_checkpoint_files = {
            str(row["relative_path"])
            for row in verified_bridge_receipt["checkpoint_files"]
        }
        for task in canonical["task_configs"]:
            if str(task.get("dependency_mode", "checkpoint")) == "standalone":
                continue
            bindings = task.get("checkpoint_bindings")
            if (
                not isinstance(bindings, dict)
                or not bindings
                or any(
                    str(relative) not in allowed_checkpoint_files
                    for relative in bindings.values()
                )
            ):
                raise ConfigError(
                    "secondary checkpoint binding 未被 primary bridge receipt 覆盖"
                )
    elif any(
        authorization.get(field) is not None
        for field in (
            "primary_bridge_root",
            "primary_bridge_receipt_relative_path",
            "primary_bridge_receipt_sha256",
        )
    ):
        raise ConfigError("primary shard 不得声明 primary bridge")

    derived = copy.deepcopy(canonical)
    derived_tasks = [
        copy.deepcopy(task)
        for task in canonical_tasks
        if str(task["task_id"]) in selected_ids
    ]
    if role == "primary":
        for task in derived_tasks:
            task["publish_checkpoint_after"] = False
        derived_tasks[-1]["publish_checkpoint_after"] = True
    derived["task_configs"] = derived_tasks
    derived["instance_identity"] = {
        **canonical_instance,
        "output_path": output,
        "cache_namespace": cache,
    }
    derived["receipt_relative_path"] = receipt
    if role == "secondary":
        derived["dependency_output_path"] = str(bridge_root)
    derived["paired_shard"] = {
        "schema_version": "task8b-paired-shard-derived-v1",
        "authorization_id": str(authorization["authorization_id"]),
        "authorization_sha256": _sha256(auth_path),
        "authorization_path": str(auth_path),
        "canonical_manifest_path": str(canonical_path),
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": str(authorization["amendment_sha256"]),
        "scientific_checkout": str(scientific_checkout),
        "frozen_code_sha": FROZEN_CODE_SHA,
        "frozen_formal_runner_sha256": FROZEN_FORMAL_RUNNER_SHA256,
        "engineering_controller_sha256": str(
            authorization["engineering_controller_sha256"]
        ),
        "approved_staging_root": str(staging_root),
        "approved_receipt_root": str(receipt_root),
        "denied_partial_root": str(denied_partial.absolute()),
        "pair_id": str(authorization["pair_id"]),
        "physical_slot": str(authorization["physical_slot"]),
        "shard_role": shard_role,
        "partition_id": str(authorization["partition_id"]),
        "shard_id": str(authorization["shard_id"]),
        "seal_mode": seal_mode,
        "canonical_manifest_sha256": _sha256(canonical_path),
        "selected_task_ids": selected_ids,
        "primary_bridge_root": str(bridge_root) if bridge_root else None,
        "primary_bridge_receipt_path": (
            str(bridge_receipt) if bridge_receipt else None
        ),
        "primary_bridge_receipt_sha256": (
            str(authorization["primary_bridge_receipt_sha256"])
            if bridge_receipt
            else None
        ),
        "old_partial_access_authorized": False,
        "effect_fields_read": False,
        "scientific_protocol_changed": False,
    }
    formal_runner.validate_worker_manifest(derived, allow_candidate=False)
    return derived


def run_authorized_shard(
    canonical_manifest_path: str | Path,
    authorization_path: str | Path,
    *,
    derived_manifest_output: str | Path,
    receipt_root: str | Path,
) -> dict[str, Any]:
    """Publish the derived manifest and invoke the original formal runner."""

    output = Path(os.path.abspath(derived_manifest_output))
    derived = derive_authorized_manifest(
        canonical_manifest_path,
        authorization_path,
    )
    reserved_derived = copy.deepcopy(derived)
    shard_identity = dict(derived["paired_shard"])
    if shard_identity.get("seal_mode") != "execution":
        raise ConfigError("run 拒绝 historical adoption authorization")
    scientific_checkout = Path(str(shard_identity["scientific_checkout"]))
    staging_root = Path(str(shard_identity["approved_staging_root"]))
    approved_receipts = Path(str(shard_identity["approved_receipt_root"]))
    manifest_output = _inside_approved(
        staging_root,
        output,
        label="derived_manifest_output",
    )
    derived_worker_output = Path(str(derived["instance_identity"]["output_path"]))
    derived_cache_output = Path(
        str(derived["instance_identity"]["cache_namespace"])
    )
    if _paths_overlap(manifest_output, derived_worker_output) or _paths_overlap(
        manifest_output,
        derived_cache_output,
    ):
        raise ConfigError("paired-shard derived manifest 与 output/cache 重叠")
    if Path(os.path.abspath(receipt_root)) != approved_receipts:
        raise ConfigError("paired-shard receipt_root 未绑定 approved_receipt_root")
    reservation_identity = {
        "authorization_sha256": shard_identity["authorization_sha256"],
        "derived_output_path": derived["instance_identity"]["output_path"],
        "derived_cache_namespace": derived["instance_identity"]["cache_namespace"],
        "derived_receipt_relative_path": derived["receipt_relative_path"],
        "derived_manifest_output": str(manifest_output),
    }
    _reserve_execution(staging_root, reservation_identity)
    # Rebuild every mutable binding after the exclusive reservation.
    derived = derive_authorized_manifest(
        canonical_manifest_path,
        authorization_path,
    )
    if _json_bytes(derived) != _json_bytes(reserved_derived):
        raise ConfigError("paired-shard authorization changed after reservation")
    shard_identity = dict(derived["paired_shard"])
    rebound_checkout = Path(str(shard_identity["scientific_checkout"]))
    rebound_staging = Path(str(shard_identity["approved_staging_root"]))
    rebound_receipts = Path(str(shard_identity["approved_receipt_root"]))
    if (
        rebound_checkout != scientific_checkout
        or rebound_staging != staging_root
        or rebound_receipts != approved_receipts
    ):
        raise ConfigError("paired-shard reserved root binding changed")
    _verify_scientific_checkout(rebound_checkout)
    instance = derived["instance_identity"]
    worker_output = Path(str(instance["output_path"]))
    cache_output = Path(str(instance["cache_namespace"]))
    receipt_base = approved_receipts
    receipt_target = (
        receipt_base / str(derived["receipt_relative_path"])
    ).resolve()
    if worker_output.exists() or worker_output.is_symlink():
        raise ConfigError("paired-shard staging output 已存在，拒绝隐式 retry")
    if cache_output.exists() or cache_output.is_symlink():
        raise ConfigError("paired-shard staging cache 已存在，拒绝复用")
    if derived["role"] == "primary" and (
        receipt_target.exists() or receipt_target.is_symlink()
    ):
        raise ConfigError("paired-shard primary receipt 已存在，拒绝复用")
    _write_json_new(manifest_output, derived)
    bootstrap = staging_root / ".runner" / (
        hashlib.sha256(_json_bytes(reservation_identity)).hexdigest() + ".py"
    )
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    if bootstrap.parent.is_symlink():
        raise ConfigError("paired-shard runner bootstrap parent 为 symlink")
    try:
        with bootstrap.open("xb") as handle:
            handle.write(_FROZEN_RUNNER_BOOTSTRAP.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ConfigError("paired-shard runner bootstrap 已存在") from exc
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(bootstrap),
                str(rebound_checkout),
                str(manifest_output),
                str(rebound_receipts),
            ],
            cwd=str(rebound_checkout),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        diagnostic_root = approved_receipts / "runner_diagnostics"
        material = {
            "authorization_id": shard_identity["authorization_id"],
            "derived_manifest_sha256": _sha256(manifest_output),
            "returncode": int(exc.returncode),
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "effect_fields_read": False,
        }
        diagnostic_path = diagnostic_root / (
            hashlib.sha256(_json_bytes(material)).hexdigest()
            + ".diagnostic.json"
        )
        _write_json_new(diagnostic_path, material)
        innermost = str(exc.stderr or "").strip().splitlines()
        detail = innermost[-1] if innermost else type(exc).__name__
        raise ConfigError(
            "paired-shard frozen subprocess runner failed；"
            f"innermost={detail}；diagnostic={diagnostic_path}"
        ) from exc
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ConfigError("paired-shard frozen subprocess runner failed") from exc
    if not isinstance(result, dict):
        raise ConfigError("paired-shard frozen subprocess result 非对象")
    return result


def _verified_task_rows(
    manifest: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    worker_id, _seed, _role = _worker_seed_role(manifest)
    rows: list[dict[str, Any]] = []
    for task in manifest["task_configs"]:
        task_id = str(task["task_id"])
        marker_path = _strict_relative_target(
            root,
            f"task_receipts/{task_id}.json",
            label="task receipt",
            must_exist=True,
        )
        marker = _read_json(marker_path)
        child = _strict_relative_target(
            root,
            str(marker.get("run_dir", "")),
            label="task run_dir",
            must_exist=True,
        )
        if not child.is_dir() or child.is_symlink():
            raise ConfigError(f"paired-shard task run_dir 非法：{task_id}")
        result_path = child / "experiment_result.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise ConfigError(f"paired-shard task completion marker 缺失：{task_id}")
        files = _directory_manifest(child)
        actual_hands = _structural_hand_count(child)
        identity_audit = marker.get("task_row", {}).get("identity_audit")
        expected_identity = task.get("expected_identity")
        if (
            marker.get("schema_version") != "task8-worker-task-receipt-v1"
            or marker.get("task_id") != task_id
            or marker.get("config_sha256") != task.get("config_sha256")
            or marker.get("files") != files
            or marker.get("task_row", {}).get("task_id") != task_id
            or marker.get("task_row", {}).get("status") != "complete"
            or not isinstance(identity_audit, dict)
            or not isinstance(expected_identity, dict)
            or any(
                identity_audit.get(field) != expected_identity.get(field)
                for field in formal_runner.REQUIRED_IDENTITY_FIELDS
            )
            or actual_hands != int(task.get("planned_hands", -1))
        ):
            raise ConfigError(f"paired-shard task receipt/hash 非法：{task_id}")
        rows.append(
            {
                "schema_version": "task8b-paired-shard-task-v1",
                "status": "complete",
                "task_id": task_id,
                "planned_hands": int(task["planned_hands"]),
                "actual_hands": actual_hands,
                "task_receipt_path": str(marker_path),
                "task_receipt_sha256": _sha256(marker_path),
                "run_dir": str(marker["run_dir"]),
                "files": files,
                "producer_worker_id": worker_id,
            }
        )
    return rows


def _base_shard_receipt(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    root: Path,
    shard: dict[str, Any],
    staging_root: Path,
    approved_receipts: Path,
    rows: list[dict[str, Any]],
    source_kind: str,
) -> dict[str, Any]:
    worker_id, seed, role = _worker_seed_role(manifest)
    return {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "complete",
        "source_kind": source_kind,
        "canonical_scope": {
            "seed_count": 6,
            "logical_worker_count": 12,
            "seeds": list(FORMAL_SEEDS),
            "execution_shards_are_not_workers": True,
        },
        "worker_id": worker_id,
        "role": role,
        "seed": seed,
        "shard_id": shard["shard_id"],
        "partition_id": shard["partition_id"],
        "amendment_id": shard["amendment_id"],
        "amendment_sha256": shard["amendment_sha256"],
        "frozen_code_sha": shard["frozen_code_sha"],
        "frozen_formal_runner_sha256": shard["frozen_formal_runner_sha256"],
        "engineering_controller_sha256": shard["engineering_controller_sha256"],
        "pair_id": shard["pair_id"],
        "physical_slot": shard["physical_slot"],
        "shard_role": shard["shard_role"],
        "approved_staging_root": str(staging_root),
        "approved_receipt_root": str(approved_receipts),
        "denied_partial_root": shard["denied_partial_root"],
        "canonical_manifest_sha256": shard["canonical_manifest_sha256"],
        "derived_manifest_sha256": _sha256(manifest_path),
        "attempt_root": str(root),
        "selected_task_ids": list(shard["selected_task_ids"]),
        "tasks": rows,
        "effect_fields_read": False,
    }


def _canonicalize_preserving_mapping_keys(config: dict[str, Any]) -> dict[str, Any]:
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


def _canonicalize_legacy_json_roundtrip(
    config: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(config, ensure_ascii=False))
    return _canonicalize_preserving_mapping_keys(value)


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stringify_mapping_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _mapping_key_diff(
    corrected: Any,
    legacy: Any,
    *,
    path: str = "$",
) -> list[dict[str, str]]:
    """Prove that JSON round-trip changed only non-bool integer mapping keys."""

    if isinstance(corrected, dict):
        if not isinstance(legacy, dict):
            raise ConfigError("paired recovery canonicalization container mismatch")
        expected_legacy_keys: set[str] = set()
        rows: list[dict[str, str]] = []
        for key, value in corrected.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ConfigError("paired recovery canonicalization 含非法 mapping key")
            legacy_key = str(key)
            if legacy_key in expected_legacy_keys:
                raise ConfigError("paired recovery canonicalization mapping key collision")
            expected_legacy_keys.add(legacy_key)
            if legacy_key not in legacy:
                raise ConfigError("paired recovery canonicalization key 缺失")
            if isinstance(key, int):
                if str(key) in corrected:
                    raise ConfigError(
                        "paired recovery canonicalization int/string key collision"
                    )
                rows.append(
                    {
                        "path": path,
                        "original_key": str(key),
                        "original_key_type": "integer",
                        "legacy_key": legacy_key,
                        "legacy_key_type": "string",
                    }
                )
            child_path = f"{path}.{legacy_key}"
            rows.extend(
                _mapping_key_diff(value, legacy[legacy_key], path=child_path)
            )
        if set(legacy) != expected_legacy_keys or any(
            not isinstance(key, str) for key in legacy
        ):
            raise ConfigError("paired recovery canonicalization key set mismatch")
        return rows
    if isinstance(corrected, list):
        if not isinstance(legacy, list) or len(corrected) != len(legacy):
            raise ConfigError("paired recovery canonicalization list mismatch")
        rows = []
        for index, (left, right) in enumerate(zip(corrected, legacy, strict=True)):
            rows.extend(_mapping_key_diff(left, right, path=f"{path}[{index}]"))
        return rows
    if type(corrected) is not type(legacy) or corrected != legacy:
        raise ConfigError("paired recovery canonicalization scalar mismatch")
    return []


def _read_yaml_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"paired recovery YAML 缺失或为 symlink：{path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"paired recovery YAML 非法：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"paired recovery YAML 顶层必须为对象：{path}")
    return value


def _same_or_new_json(path: Path, value: dict[str, Any]) -> None:
    content = _json_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise ConfigError(f"paired recovery 拒绝修改既有证据：{path}")
        return
    _write_json_new(path, value)


def _same_or_new_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise ConfigError(f"paired recovery 拒绝修改既有证据：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ConfigError(f"paired recovery 拒绝覆盖：{path}") from exc


def _state_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ConfigError("paired recovery state.tsv 缺失或为 symlink")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            expected = [
                "schema_version",
                "created_at_utc",
                "status",
                "detail",
                "previous_sha256",
                "row_sha256",
            ]
            if reader.fieldnames != expected:
                raise ConfigError("paired recovery state.tsv header 非法")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError("paired recovery state.tsv 无法解析") from exc
    previous = "GENESIS"
    for row in rows:
        body = {
            field: row[field]
            for field in (
                "schema_version",
                "created_at_utc",
                "status",
                "detail",
                "previous_sha256",
            )
        }
        digest = sha256_json(body)
        if row["previous_sha256"] != previous or row["row_sha256"] != digest:
            raise ConfigError("paired recovery state.tsv hash chain mismatch")
        previous = digest
    if not rows:
        raise ConfigError("paired recovery state.tsv 为空")
    return rows


def _recovery_state_rows(
    existing: list[dict[str, str]],
    *,
    recovery_id: str,
    created_at_utc: str,
    close_execution: bool,
) -> list[dict[str, str]]:
    statuses = ["recovered"] + (["complete"] if close_execution else [])
    previous = existing[-1]["row_sha256"]
    rows: list[dict[str, str]] = []
    for status in statuses:
        body = {
            "schema_version": existing[-1]["schema_version"],
            "created_at_utc": created_at_utc,
            "status": status,
            "detail": (
                f"{recovery_id}:verifier-only-completed-task"
                if status == "recovered"
                else f"{recovery_id}:all-selected-tasks-complete"
            ),
            "previous_sha256": previous,
        }
        row = {**body, "row_sha256": sha256_json(body)}
        rows.append(row)
        previous = row["row_sha256"]
    return rows


def _write_state_rows_same_or_append(
    path: Path,
    original: list[dict[str, str]],
    appended: list[dict[str, str]],
) -> None:
    observed = _state_rows(path)
    if observed == original + appended:
        return
    if observed != original:
        raise ConfigError("paired recovery state.tsv 已偏离 pre-recovery baseline")
    fieldnames = list(original[0])
    with path.open("r+b") as handle:
        current = handle.read()
        newline = b"\r\n" if current.endswith(b"\r\n") else b"\n"
        handle.seek(0, os.SEEK_END)
        suffix = b"".join(
            "\t".join(row[field] for field in fieldnames).encode("utf-8")
            + newline
            for row in appended
        )
        handle.write(suffix)
        handle.flush()
        os.fsync(handle.fileno())
    if _state_rows(path) != original + appended:
        raise ConfigError("paired recovery state.tsv append 后校验失败")


def _files_tsv_bytes(root: Path) -> bytes:
    rows = _directory_manifest(root)
    excluded = {"state.tsv", "files.tsv", "completion_receipt.json"}
    content = "relative_path\tsize\tsha256\n"
    for row in rows:
        if row["relative_path"] in excluded:
            continue
        content += (
            f"{row['relative_path']}\t{row['size']}\t{row['sha256']}\n"
        )
    return content.encode("utf-8")


def _recovery_identity_and_health(
    manifest: dict[str, Any],
    task: dict[str, Any],
    child: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = (
        "manifest.json",
        "resolved_config.yaml",
        "schedule_manifest.json",
        "experiment_result.json",
        "protocol_audit.json",
        "events.jsonl",
        "hand_summaries.jsonl",
        "metrics.json",
    )
    for name in required:
        path = child / name
        if not path.is_file() or path.is_symlink():
            raise ConfigError(f"paired recovery 必需工件缺失：{name}")
    config = _read_yaml_object(child / "resolved_config.yaml")
    metadata = dict(_read_json(child / "manifest.json").get("metadata", {}))
    schedule = _read_json(child / "schedule_manifest.json")
    corrected_config = _canonicalize_preserving_mapping_keys(config)
    legacy_config = _canonicalize_legacy_json_roundtrip(config)
    key_level_diff = _mapping_key_diff(corrected_config, legacy_config)
    common_identity = manifest.get("common_identity")
    if not isinstance(common_identity, dict):
        raise ConfigError("paired recovery common_identity 缺失")
    actual = {
        "code_sha": dict(metadata.get("code", {})).get("commit"),
        "code_dirty": dict(metadata.get("code", {})).get("dirty"),
        "resolved_config_sha256": sha256_json(corrected_config),
        "prompt_sha256": sha256_json(metadata.get("prompts", {})),
        "model_fingerprint": sha256_json(metadata.get("model", {})),
        "embedding_fingerprint": task8b_embedding_fingerprint(
            metadata.get("embedding", {})
        ),
        "schedule_sha256": schedule.get("schedule_sha256"),
    }
    expected = task.get("expected_identity")
    if not isinstance(expected, dict):
        raise ConfigError("paired recovery expected_identity 缺失")
    expected_fields = set(formal_runner.REQUIRED_IDENTITY_FIELDS)
    if set(expected) != expected_fields:
        raise ConfigError("paired recovery expected_identity 字段集合非法")
    if set(common_identity) != set(formal_runner.TASK8B_FLEET_LOCK_FIELDS):
        raise ConfigError("paired recovery common_identity 字段集合非法")
    for field in formal_runner.FLEET_COMMON_IDENTITY_FIELDS:
        if expected.get(field) != common_identity.get(field):
            raise ConfigError(
                f"paired recovery task/common identity mismatch：{field}"
            )
    for field in expected_fields:
        if actual.get(field) != expected.get(field):
            raise ConfigError(f"paired recovery identity mismatch：{field}")
    if actual["code_dirty"] is not False:
        raise ConfigError("paired recovery code_dirty 必须为 false")
    legacy_hash = sha256_json(legacy_config)
    corrected_hash = actual["resolved_config_sha256"]
    if (
        legacy_hash == corrected_hash
        or corrected_hash != expected["resolved_config_sha256"]
        or not key_level_diff
    ):
        raise ConfigError("paired recovery 非唯一 mapping-key canonicalization 差异")
    audit = _read_json(child / "protocol_audit.json")
    validity = audit.get("run_validity")
    health = audit.get("execution_health")
    validity_fields = {
        "execution_valid",
        "behavior_valid",
        "paper_eligible",
        "run_mode",
        "status",
    }
    if not isinstance(validity, dict) or set(validity) != validity_fields:
        raise ConfigError("paired recovery run_validity exact-key gate failed")
    if (
        validity["execution_valid"] is not True
        or validity["behavior_valid"] is not True
        or validity["paper_eligible"] is not True
        or validity["run_mode"] != "formal"
        or validity["status"] != "valid_for_main_table"
    ):
        raise ConfigError("paired recovery run_validity gate failed")
    health_fields = {
        "valid",
        *HEALTH_ZERO_FIELDS,
        "reward_conservation_violation_hand_ids",
        "stack_conservation_violation_hand_ids",
        "status",
    }
    if not isinstance(health, dict) or set(health) != health_fields:
        raise ConfigError("paired recovery execution_health exact-key gate failed")
    if health["valid"] is not True:
        raise ConfigError("paired recovery execution_health.valid gate failed")
    if (
        health["status"] != "passed"
        or health["reward_conservation_violation_hand_ids"] != []
        or health["stack_conservation_violation_hand_ids"] != []
    ):
        raise ConfigError("paired recovery execution_health detail gate failed")
    for field in HEALTH_ZERO_FIELDS:
        value = health[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ConfigError(f"paired recovery health counter 非零：{field}")
    correction = {
        "reason": COMPLETED_RECOVERY_REASON,
        "legacy_json_roundtrip_actual_sha256": legacy_hash,
        "original_expected_sha256": expected["resolved_config_sha256"],
        "corrected_actual_sha256": corrected_hash,
        "semantic_equivalence_after_stringifying_mapping_keys": True,
        "only_authorized_difference": "integer-versus-string mapping keys",
        "key_level_diff": key_level_diff,
    }
    health_summary = {
        "execution_valid": True,
        "behavior_valid": True,
        "execution_health_valid": True,
        **{field: 0 for field in HEALTH_ZERO_FIELDS},
    }
    return actual, health_summary, correction


def _resume_recovered_execution(
    manifest_path: Path,
    *,
    scientific_checkout: Path,
    receipt_root: Path,
    legacy_audit_path: Path | None = None,
    legacy_audit_sha256: str | None = None,
    legacy_receipt_config_sha256: str | None = None,
    corrected_receipt_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Resume with the frozen runner after publishing the recovered task receipt."""

    _verify_scientific_checkout(scientific_checkout)
    if (legacy_audit_path is None) != (legacy_audit_sha256 is None):
        raise ConfigError("paired recovery legacy audit bridge 参数不闭合")
    if (legacy_receipt_config_sha256 is None) != (
        corrected_receipt_config_sha256 is None
    ):
        raise ConfigError("paired recovery receipt identity bridge 参数不闭合")
    if legacy_receipt_config_sha256 is not None and (
        not _is_sha256(legacy_receipt_config_sha256)
        or not _is_sha256(corrected_receipt_config_sha256)
        or legacy_receipt_config_sha256
        == corrected_receipt_config_sha256
    ):
        raise ConfigError("paired recovery receipt identity bridge SHA 非法")
    if legacy_audit_path is not None:
        if not legacy_audit_path.is_file() or legacy_audit_path.is_symlink():
            raise ConfigError("paired recovery legacy audit bridge 路径非法")
        if _sha256(legacy_audit_path) != legacy_audit_sha256:
            raise ConfigError("paired recovery legacy audit bridge SHA 不匹配")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _FROZEN_RECOVERY_RESUME_BOOTSTRAP,
                str(scientific_checkout),
                str(manifest_path),
                str(receipt_root),
                str(legacy_audit_path) if legacy_audit_path is not None else "",
                legacy_audit_sha256 or "",
                legacy_receipt_config_sha256 or "",
                corrected_receipt_config_sha256 or "",
            ],
            cwd=str(scientific_checkout),
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "")
        stderr_sha256 = hashlib.sha256(stderr.encode("utf-8")).hexdigest()
        exception_lines = [
            line.strip() for line in stderr.splitlines() if line.strip()
        ]
        diagnostic = (
            receipt_root
            / "frozen_resume_diagnostics"
            / (
                f"{manifest_path.stem}-{stderr_sha256[:16]}"
                ".diagnostic.json"
            )
        )
        _same_or_new_json(
            diagnostic,
            {
                "schema_version": (
                    "task8b-paired-frozen-resume-diagnostic-v1"
                ),
                "status": "failed",
                "manifest_path": str(manifest_path),
                "returncode": int(exc.returncode),
                "stderr_sha256": stderr_sha256,
                "innermost_exception": (
                    exception_lines[-1] if exception_lines else None
                ),
                "stderr": stderr,
                "effect_fields_read": False,
            },
        )
        raise ConfigError(
            "paired recovery frozen resume failed；"
            f"innermost={exception_lines[-1] if exception_lines else 'unknown'}；"
            f"diagnostic={diagnostic}"
        ) from exc
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ConfigError("paired recovery frozen resume failed") from exc
    if not isinstance(result, dict):
        raise ConfigError("paired recovery frozen resume result 非对象")
    return result


def recover_completed_execution(
    derived_manifest_path: str | Path,
    run_dir: str | Path,
    baseline_path: str | Path,
    archive_path: str | Path,
    certificate_path: str | Path,
    ledger_entry_path: str | Path,
) -> dict[str, Any]:
    """Recover one scientifically complete task after verifier-only failure."""

    manifest_path = Path(os.path.abspath(derived_manifest_path))
    root = Path(os.path.abspath(run_dir))
    baseline_file = Path(os.path.abspath(baseline_path))
    archive = Path(os.path.abspath(archive_path))
    manifest = _read_json(manifest_path)
    worker_id, seed, role = _worker_seed_role(manifest)
    shard = manifest.get("paired_shard")
    if not isinstance(shard, dict) or shard.get("seal_mode") != "execution":
        raise ConfigError("paired recovery 仅允许 execution derived manifest")
    source_controller_sha256 = str(
        shard.get("engineering_controller_sha256", "")
    )
    current_controller_sha256 = _sha256(Path(__file__).resolve())
    if source_controller_sha256 == RECOVERY_SOURCE_CONTROLLER_SHA256:
        rebuilt = derive_authorized_manifest(
            str(shard.get("canonical_manifest_path", "")),
            str(shard.get("authorization_path", "")),
            _recovery_source_controller_sha256=(
                RECOVERY_SOURCE_CONTROLLER_SHA256
            ),
        )
    elif source_controller_sha256 == current_controller_sha256:
        rebuilt = derive_authorized_manifest(
            str(shard.get("canonical_manifest_path", "")),
            str(shard.get("authorization_path", "")),
        )
    else:
        raise ConfigError("paired recovery source controller SHA 未获授权")
    if _json_bytes(rebuilt) != _json_bytes(manifest):
        raise ConfigError("paired recovery derived manifest 不匹配 authorization")
    staging_root = _approved_root(
        str(shard["approved_staging_root"]),
        label="approved_staging_root",
    )
    approved_receipts = _approved_root(
        str(shard["approved_receipt_root"]),
        label="approved_receipt_root",
    )
    _inside_approved(staging_root, root, label="paired recovery attempt_root")
    _inside_approved(
        approved_receipts,
        baseline_file,
        label="paired recovery baseline",
    )
    certificate = _inside_approved(
        approved_receipts,
        Path(certificate_path),
        label="paired recovery certificate",
    )
    ledger = _inside_approved(
        approved_receipts,
        Path(ledger_entry_path),
        label="paired recovery ledger entry",
    )
    recovery_audit = _inside_approved(
        approved_receipts,
        certificate.with_name(
            f"{certificate.stem}.identity_correction_audit.json"
        ),
        label="paired recovery identity correction audit",
    )
    if (
        _paths_overlap(certificate, root)
        or _paths_overlap(ledger, root)
        or _paths_overlap(recovery_audit, root)
    ):
        raise ConfigError("paired recovery audit outputs 不得写入 attempt_root")
    baseline = _read_json(baseline_file)
    baseline_fields = {
        "schema_version",
        "status",
        "recovery_id",
        "reason",
        "created_at_utc",
        "worker_id",
        "seed",
        "task_id",
        "attempt_root",
        "derived_manifest_sha256",
        "authorization_sha256",
        "recovery_tool_sha256",
        "pre_recovery_files",
        "pre_recovery_archive_path",
        "pre_recovery_archive_sha256",
        "failed_state_row_sha256",
        "effect_fields_read",
    }
    if set(baseline) != baseline_fields:
        raise ConfigError("paired recovery baseline 字段集合非法")
    task_id = str(baseline.get("task_id", ""))
    tasks = {
        str(task["task_id"]): task
        for task in manifest.get("task_configs", [])
        if isinstance(task, dict)
    }
    authorization_path = Path(str(shard["authorization_path"]))
    if (
        baseline.get("schema_version") != COMPLETED_RECOVERY_BASELINE_SCHEMA
        or baseline.get("status") != "activated"
        or baseline.get("reason") != COMPLETED_RECOVERY_REASON
        or baseline.get("worker_id") != worker_id
        or int(baseline.get("seed", -1)) != seed
        or task_id not in tasks
        or task_id not in shard.get("selected_task_ids", [])
        or baseline.get("attempt_root") != str(root)
        or baseline.get("derived_manifest_sha256") != _sha256(manifest_path)
        or baseline.get("authorization_sha256") != _sha256(authorization_path)
        or baseline.get("recovery_tool_sha256")
        != _sha256(Path(__file__).resolve())
        or baseline.get("pre_recovery_archive_path") != str(archive)
        or not archive.is_file()
        or archive.is_symlink()
        or baseline.get("pre_recovery_archive_sha256") != _sha256(archive)
        or baseline.get("effect_fields_read") is not False
        or not isinstance(baseline.get("recovery_id"), str)
        or not baseline["recovery_id"]
        or not isinstance(baseline.get("created_at_utc"), str)
        or not baseline["created_at_utc"]
    ):
        raise ConfigError("paired recovery baseline authorization binding failed")
    pre_files = baseline.get("pre_recovery_files")
    if not isinstance(pre_files, list) or not all(
        isinstance(row, dict)
        and set(row) == {"relative_path", "size", "sha256"}
        for row in pre_files
    ):
        raise ConfigError("paired recovery pre-recovery file manifest 非法")
    _verify_recovery_archive(archive, pre_files)
    if not certificate.exists() and _directory_manifest(root) != pre_files:
        raise ConfigError("paired recovery pre-recovery file manifest 不闭合")
    for row in pre_files:
        if row["relative_path"] == "state.tsv":
            continue
        target = _strict_relative_target(
            root,
            str(row["relative_path"]),
            label="paired recovery baseline file",
            must_exist=True,
        )
        if (
            target.stat().st_size != int(row["size"])
            or _sha256(target) != row["sha256"]
        ):
            raise ConfigError("paired recovery baseline scientific file 已变化")
    observed_state = _state_rows(root / "state.tsv")
    recovery_id = str(baseline["recovery_id"])
    failed_indices = [
        index
        for index, row in enumerate(observed_state)
        if row["row_sha256"] == baseline.get("failed_state_row_sha256")
    ]
    if len(failed_indices) != 1:
        raise ConfigError("paired recovery failed state row 不唯一")
    original_state = observed_state[: failed_indices[0] + 1]
    if (
        not original_state
        or original_state[-1]["status"] != "failed"
        or original_state[-1]["row_sha256"]
        != baseline.get("failed_state_row_sha256")
    ):
        raise ConfigError("paired recovery failed state binding failed")
    child = _strict_relative_target(
        root,
        f"runs/{task_id}",
        label="paired recovery child run",
        must_exist=True,
    )
    if not child.is_dir() or child.is_symlink():
        raise ConfigError("paired recovery child run 非法")
    task = tasks[task_id]
    if (
        int(task.get("planned_hands", -1)) != EXPECTED_TASKS[role][task_id]
        or _structural_hand_count(child) != int(task["planned_hands"])
    ):
        raise ConfigError("paired recovery planned/actual hands 不闭合")
    actual, health, correction = _recovery_identity_and_health(
        manifest,
        task,
        child,
    )
    standard_audit = {
        "schema_version": "task8-task-identity-audit-v1",
        "task_id": task_id,
        "protocol_status": manifest["protocol_status"],
        "actual": actual,
        "status": "verified",
    }
    recovery_audit_body = {
        "schema_version": "task8b-paired-identity-correction-audit-v1",
        "status": "verified-recovered",
        "recovery_id": recovery_id,
        "worker_id": worker_id,
        "seed": seed,
        "task_id": task_id,
        "attempt_root": str(root),
        "standard_task_identity_audit_path": str(
            child / "task_identity_audit.json"
        ),
        "actual": actual,
        "expected": task["expected_identity"],
        "identity_correction": correction,
        "health": health,
        "effect_fields_read": False,
    }
    audit_path = child / "task_identity_audit.json"
    legacy_audit_bridge = False
    if audit_path.exists():
        existing_audit = _read_json(audit_path)
        legacy_fields = {
            "schema_version",
            "task_id",
            "protocol_status",
            "actual",
            "expected",
            "identity_correction",
            "health",
            "status",
            "effect_fields_read",
        }
        if existing_audit != standard_audit and (
            set(existing_audit) != legacy_fields
            or existing_audit["schema_version"]
            != standard_audit["schema_version"]
            or existing_audit["task_id"] != task_id
            or existing_audit["protocol_status"]
            != manifest["protocol_status"]
            or existing_audit["actual"] != actual
            or existing_audit["expected"] != task["expected_identity"]
            or existing_audit["identity_correction"] != correction
            or existing_audit["health"] != health
            or existing_audit["status"] != "verified-recovered"
            or existing_audit["effect_fields_read"] is not False
        ):
            raise ConfigError(
                "paired recovery existing task identity audit 非标准且非精确兼容"
            )
        legacy_audit_bridge = existing_audit != standard_audit
    else:
        _same_or_new_json(audit_path, standard_audit)
    task_row = {
        "task_id": task_id,
        "memory_mode": task.get("memory_mode"),
        "run_dir": f"runs/{task_id}",
        "cache_namespace": (
            f"{manifest['instance_identity']['cache_namespace']}/"
            f"{task_id}/{str(task.get('memory_mode') or role).lower()}"
        ),
        "identity_audit": actual,
        "status": "complete",
        "recovery_id": recovery_id,
    }
    marker = {
        "schema_version": "task8-worker-task-receipt-v1",
        "task_id": task_id,
        "config_sha256": task["config_sha256"],
        "run_dir": task_row["run_dir"],
        "task_row": task_row,
        "files": _directory_manifest(child),
    }
    marker_path = root / "task_receipts" / f"{task_id}.json"
    _same_or_new_json(marker_path, marker)
    receipt_identity = manifest.get("receipt_identity")
    legacy_receipt_config_sha256: str | None = None
    corrected_receipt_config_sha256: str | None = None
    checkpoint_receipt_identity_correction = {
        "applied": False,
        "old_resolved_config_sha256": None,
        "new_resolved_config_sha256": None,
        "only_authorized_field": "resolved_config_sha256",
    }
    if manifest["role"] == "primary":
        if not isinstance(receipt_identity, dict):
            raise ConfigError("paired recovery primary receipt_identity 缺失")
        checkpoint_publishers = [
            candidate
            for candidate in manifest["task_configs"]
            if bool(candidate.get("publish_checkpoint_after", False))
        ]
        if len(checkpoint_publishers) != 1:
            raise ConfigError(
                "paired recovery primary checkpoint publisher 数量非法"
            )
        checkpoint_producer_identity = checkpoint_publishers[0].get(
            "expected_identity"
        )
        if not isinstance(checkpoint_producer_identity, dict):
            raise ConfigError(
                "paired recovery primary checkpoint producer identity 缺失"
            )
        if not all(
            field in checkpoint_producer_identity
            for field in formal_runner.REQUIRED_IDENTITY_FIELDS
        ):
            raise ConfigError(
                "paired recovery checkpoint producer identity 字段集合非法"
            )
        if set(receipt_identity) != set(
            formal_runner.REQUIRED_IDENTITY_FIELDS
        ):
            raise ConfigError(
                "paired recovery primary receipt_identity 字段集合非法"
            )
        for field in formal_runner.REQUIRED_IDENTITY_FIELDS:
            if field == "resolved_config_sha256":
                continue
            if (
                receipt_identity.get(field)
                != checkpoint_producer_identity[field]
            ):
                raise ConfigError(
                    f"paired recovery receipt identity mismatch：{field}"
                )
        observed_receipt_config = receipt_identity.get(
            "resolved_config_sha256"
        )
        checkpoint_producer_config = checkpoint_producer_identity[
            "resolved_config_sha256"
        ]
        if observed_receipt_config != checkpoint_producer_config:
            if not _is_sha256(observed_receipt_config) or not _is_sha256(
                checkpoint_producer_config
            ):
                raise ConfigError(
                    "paired recovery legacy receipt config identity 非法"
                )
            legacy_receipt_config_sha256 = observed_receipt_config
            corrected_receipt_config_sha256 = checkpoint_producer_config
            checkpoint_receipt_identity_correction = {
                "applied": True,
                "old_resolved_config_sha256": (
                    legacy_receipt_config_sha256
                ),
                "new_resolved_config_sha256": (
                    corrected_receipt_config_sha256
                ),
                "only_authorized_field": "resolved_config_sha256",
            }
    recovery_audit_body["checkpoint_receipt_identity_correction"] = (
        checkpoint_receipt_identity_correction
    )
    _same_or_new_json(recovery_audit, recovery_audit_body)
    selected = list(shard["selected_task_ids"])
    pre_recovery_relative_paths = {
        str(row["relative_path"]) for row in pre_files
    }
    standard_resume_task_ids = [
        selected_id
        for selected_id in selected
        if selected_id != task_id
        and f"task_receipts/{selected_id}.json"
        not in pre_recovery_relative_paths
    ]
    _resume_recovered_execution(
        manifest_path,
        scientific_checkout=Path(str(shard["scientific_checkout"])),
        receipt_root=approved_receipts,
        legacy_audit_path=(audit_path if legacy_audit_bridge else None),
        legacy_audit_sha256=(
            _sha256(audit_path) if legacy_audit_bridge else None
        ),
        legacy_receipt_config_sha256=legacy_receipt_config_sha256,
        corrected_receipt_config_sha256=corrected_receipt_config_sha256,
    )
    completion_path = root / "completion_receipt.json"
    files_path = root / "files.tsv"
    close_execution = (
        all(
            (root / "task_receipts" / f"{selected_id}.json").is_file()
            for selected_id in selected
        )
        and completion_path.is_file()
        and files_path.is_file()
    )
    if not close_execution:
        raise ConfigError("paired recovery frozen resume 未闭合 selected tasks")
    _verified_task_rows(manifest, root)
    _verify_files_tsv(root)
    completion = _read_json(completion_path)
    if (
        completion.get("schema_version") != "task8-worker-completion-v1"
        or completion.get("status") != "complete"
        or completion.get("worker_id") != worker_id
        or completion.get("files_tsv_sha256") != _sha256(files_path)
    ):
        raise ConfigError("paired recovery standard completion receipt gate failed")
    final_state = _state_rows(root / "state.tsv")
    if (
        final_state[: len(original_state)] != original_state
        or final_state[-1]["status"] != "complete"
    ):
        raise ConfigError("paired recovery standard state continuation gate failed")
    appended_state = final_state[len(original_state) :]
    if not appended_state:
        raise ConfigError("paired recovery standard state 未 append")
    if close_execution:
        _verify_files_tsv(root)
    certificate_body = {
        "schema_version": COMPLETED_RECOVERY_CERTIFICATE_SCHEMA,
        "status": (
            "execution_complete_recovered"
        ),
        "recovery_id": recovery_id,
        "reason": COMPLETED_RECOVERY_REASON,
        "worker_id": worker_id,
        "seed": seed,
        "role": role,
        "task_id": task_id,
        "planned_hands": int(task["planned_hands"]),
        "actual_hands": _structural_hand_count(child),
        "attempt_root": str(root),
        "failed_state_preserved": True,
        "failed_state_row_sha256": baseline["failed_state_row_sha256"],
        "appended_state_row_sha256": [
            row["row_sha256"] for row in appended_state
        ],
        "derived_manifest_sha256": _sha256(manifest_path),
        "authorization_sha256": _sha256(authorization_path),
        "recovery_tool_sha256": _sha256(Path(__file__).resolve()),
        "baseline_sha256": _sha256(baseline_file),
        "pre_recovery_archive_sha256": _sha256(archive),
        "task_identity_audit_sha256": _sha256(audit_path),
        "identity_correction_audit_path": str(recovery_audit),
        "identity_correction_audit_sha256": _sha256(recovery_audit),
        "task_receipt_sha256": _sha256(marker_path),
        "identity_correction": correction,
        "checkpoint_receipt_identity_correction": (
            checkpoint_receipt_identity_correction
        ),
        "health": health,
        "recovered_task_scientific_files_modified": False,
        "standard_resume_task_ids": standard_resume_task_ids,
        "standard_resume_used_frozen_runner": True,
        "shard_closed": close_execution,
        "unrecovered_selected_task_ids": [
            selected_id
            for selected_id in selected
            if not (root / "task_receipts" / f"{selected_id}.json").is_file()
        ],
        "effect_fields_read": False,
    }
    ledger_body = {
        "schema_version": COMPLETED_RECOVERY_LEDGER_SCHEMA,
        "status": "append-only",
        "recovery_id": recovery_id,
        "worker_id": worker_id,
        "seed": seed,
        "task_id": task_id,
        "attempt_root": str(root),
        "certificate_path": str(certificate),
        "certificate_sha256_pending_at_publication": True,
        "shard_closed": close_execution,
        "effect_fields_read": False,
    }
    _same_or_new_json(certificate, certificate_body)
    _same_or_new_json(ledger, ledger_body)
    return {
        "status": certificate_body["status"],
        "worker_id": worker_id,
        "seed": seed,
        "task_id": task_id,
        "shard_closed": close_execution,
        "certificate_path": str(certificate),
        "certificate_sha256": _sha256(certificate),
        "ledger_entry_path": str(ledger),
        "ledger_entry_sha256": _sha256(ledger),
        "effect_fields_read": False,
    }


def build_shard_receipt(
    derived_manifest_path: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
    recovery_certificate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal a deterministic shard receipt without parsing metric values."""

    manifest_path = Path(os.path.abspath(derived_manifest_path))
    root = Path(os.path.abspath(run_dir))
    manifest = _read_json(manifest_path)
    worker_id, seed, role = _worker_seed_role(manifest)
    shard = manifest.get("paired_shard")
    if not isinstance(shard, dict):
        raise ConfigError("derived manifest 缺 paired_shard identity")
    if shard.get("seal_mode") != "execution":
        raise ConfigError("execution seal 拒绝 historical adoption manifest")
    staging_root = _approved_root(
        str(shard["approved_staging_root"]),
        label="approved_staging_root",
    )
    approved_receipts = _approved_root(
        str(shard["approved_receipt_root"]),
        label="approved_receipt_root",
    )
    source_controller_sha256 = str(
        shard.get("engineering_controller_sha256", "")
    )
    current_controller_sha256 = _sha256(Path(__file__).resolve())
    if source_controller_sha256 == RECOVERY_SOURCE_CONTROLLER_SHA256:
        if recovery_certificate_path is None:
            raise ConfigError(
                "legacy recovered execution seal 必须绑定 recovery certificate"
            )
        certificate_path = _inside_approved(
            approved_receipts,
            Path(recovery_certificate_path),
            label="recovered execution seal certificate",
        )
        certificate = _read_json(certificate_path)
        selected = [str(item) for item in shard.get("selected_task_ids", [])]
        task_id = str(certificate.get("task_id", ""))
        if not selected or task_id not in selected:
            raise ConfigError(
                "legacy recovered execution seal recovery task 不在 shard"
            )
        task_receipt = root / "task_receipts" / f"{task_id}.json"
        identity_audit = root / "runs" / task_id / "task_identity_audit.json"
        correction_audit = Path(
            str(certificate.get("identity_correction_audit_path", ""))
        )
        correction_audit = _inside_approved(
            approved_receipts,
            correction_audit,
            label="recovered execution seal correction audit",
        )
        if (
            not task_receipt.is_file()
            or task_receipt.is_symlink()
            or not identity_audit.is_file()
            or identity_audit.is_symlink()
            or not correction_audit.is_file()
            or correction_audit.is_symlink()
        ):
            raise ConfigError(
                "legacy recovered execution seal evidence 缺失或为 symlink"
            )
        recovery_tool_sha256 = certificate.get("recovery_tool_sha256")
        if (
            certificate.get("schema_version")
            != COMPLETED_RECOVERY_CERTIFICATE_SCHEMA
            or certificate.get("status") != "execution_complete_recovered"
            or certificate.get("worker_id") != worker_id
            or int(certificate.get("seed", -1)) != seed
            or certificate.get("role") != role
            or certificate.get("task_id") != task_id
            or certificate.get("attempt_root") != str(root)
            or certificate.get("derived_manifest_sha256")
            != _sha256(manifest_path)
            or certificate.get("authorization_sha256")
            != _sha256(Path(str(shard["authorization_path"])))
            or recovery_tool_sha256
            not in {
                current_controller_sha256,
                *RECOVERY_COMPLETION_CONTROLLER_SHA256S,
            }
            or certificate.get("task_identity_audit_sha256")
            != _sha256(identity_audit)
            or certificate.get("identity_correction_audit_sha256")
            != _sha256(correction_audit)
            or certificate.get("task_receipt_sha256")
            != _sha256(task_receipt)
            or certificate.get("recovered_task_scientific_files_modified")
            is not False
            or certificate.get("standard_resume_used_frozen_runner")
            is not True
            or certificate.get("shard_closed") is not True
            or certificate.get("unrecovered_selected_task_ids") != []
            or certificate.get("effect_fields_read") is not False
        ):
            raise ConfigError(
                "legacy recovered execution seal certificate binding failed"
            )
        rebuilt = derive_authorized_manifest(
            str(shard.get("canonical_manifest_path", "")),
            str(shard.get("authorization_path", "")),
            _recovery_source_controller_sha256=(
                RECOVERY_SOURCE_CONTROLLER_SHA256
            ),
        )
    elif source_controller_sha256 == current_controller_sha256:
        if recovery_certificate_path is not None:
            raise ConfigError(
                "current-controller execution seal 拒绝 recovery certificate"
            )
        rebuilt = derive_authorized_manifest(
            str(shard.get("canonical_manifest_path", "")),
            str(shard.get("authorization_path", "")),
        )
    else:
        raise ConfigError("execution seal source controller SHA 未获授权")
    if _json_bytes(rebuilt) != _json_bytes(manifest):
        raise ConfigError("derived manifest 不再匹配冻结 authorization")
    _inside_approved(staging_root, root, label="shard attempt_root")
    if root.is_symlink():
        raise ConfigError("paired-shard attempt_root 为 symlink")
    receipt_output = _inside_approved(
        approved_receipts,
        Path(output_path),
        label="shard receipt output",
    )
    manifest_task_ids = [str(task["task_id"]) for task in manifest["task_configs"]]
    if shard.get("selected_task_ids") != manifest_task_ids:
        raise ConfigError("derived manifest task subset 与 shard identity 不一致")
    completion = _read_json(root / "completion_receipt.json")
    files_tsv = root / "files.tsv"
    if (
        completion.get("schema_version") != "task8-worker-completion-v1"
        or completion.get("status") != "complete"
        or completion.get("worker_id") != worker_id
        or not files_tsv.is_file()
        or files_tsv.is_symlink()
        or completion.get("files_tsv_sha256") != _sha256(files_tsv)
    ):
        raise ConfigError("paired-shard completion/files.tsv gate failed")
    first_files_tsv = _verify_files_tsv(root)
    rows = _verified_task_rows(manifest, root)
    if (
        _verify_files_tsv(root) != first_files_tsv
        or _verified_task_rows(manifest, root) != rows
    ):
        raise ConfigError("paired-shard evidence changed during execution seal")
    receipt = _base_shard_receipt(
        manifest_path=manifest_path,
        manifest=manifest,
        root=root,
        shard=shard,
        staging_root=staging_root,
        approved_receipts=approved_receipts,
        rows=rows,
        source_kind="execution",
    )
    _write_json_new(receipt_output, receipt)
    return receipt


def build_historical_adoption_receipt(
    canonical_manifest_path: str | Path,
    authorization_path: str | Path,
    canonical_run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Adopt only fully verified historical tasks without requiring worker closure."""

    canonical_path = Path(os.path.abspath(canonical_manifest_path))
    auth_path = Path(os.path.abspath(authorization_path))
    derived = derive_authorized_manifest(canonical_path, auth_path)
    shard = dict(derived["paired_shard"])
    if shard.get("seal_mode") != "historical_adoption":
        raise ConfigError("historical adoption 需要专用 authorization")
    root = Path(os.path.abspath(canonical_run_dir))
    _deny_symlink_components(root, label="historical canonical run root")
    if (
        not root.is_dir()
        or root.is_symlink()
        or root != Path(str(shard["denied_partial_root"]))
    ):
        raise ConfigError("historical adoption root 未绑定 canonical denied_partial_root")
    staging_root = _approved_root(
        str(shard["approved_staging_root"]),
        label="approved_staging_root",
    )
    approved_receipts = _approved_root(
        str(shard["approved_receipt_root"]),
        label="approved_receipt_root",
    )
    receipt_output = _inside_approved(
        approved_receipts,
        Path(output_path),
        label="historical adoption receipt output",
    )
    if _paths_overlap(receipt_output, root):
        raise ConfigError("historical adoption receipt 不得写入 canonical run")
    rows = _verified_task_rows(derived, root)
    if (
        _json_bytes(
            derive_authorized_manifest(canonical_path, auth_path)
        )
        != _json_bytes(derived)
        or _verified_task_rows(derived, root) != rows
    ):
        raise ConfigError("historical adoption evidence changed during seal")
    receipt = _base_shard_receipt(
        manifest_path=canonical_path,
        manifest=derived,
        root=root,
        shard=shard,
        staging_root=staging_root,
        approved_receipts=approved_receipts,
        rows=rows,
        source_kind="historical_adoption",
    )
    receipt["derived_manifest_sha256"] = hashlib.sha256(
        _json_bytes(derived)
    ).hexdigest()
    receipt["historical_overall_completion_required"] = False
    receipt["historical_source_immutable"] = True
    _write_json_new(receipt_output, receipt)
    return receipt


def compose_primary_checkpoint(
    primary_manifest_path: str | Path,
    shard_receipt_paths: list[str | Path],
    bridge_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Compose exact P-task checkpoint files and publish the standard receipt last."""

    canonical_path = Path(os.path.abspath(primary_manifest_path))
    canonical = _read_json(canonical_path)
    worker_id, seed, role = _canonical_identity(canonical)
    if role != "primary":
        raise ConfigError("primary bridge 需要 primary canonical manifest")
    canonical_by_id = {
        str(task["task_id"]): task for task in canonical["task_configs"]
    }
    task_rows: dict[str, tuple[dict[str, Any], Path]] = {}
    source_receipts: list[dict[str, Any]] = []
    staging_roots: set[Path] = set()
    receipt_roots: set[Path] = set()
    approved_root_pairs: set[tuple[Path, Path]] = set()
    observed_sides: set[str] = set()
    partition_ids: set[str] = set()
    seen_shards: set[str] = set()
    for raw_path in shard_receipt_paths:
        path = Path(os.path.abspath(raw_path))
        receipt = _read_json(path)
        source_kind = str(receipt.get("source_kind", ""))
        shard_role = str(receipt.get("shard_role", ""))
        if (
            receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("role") != "primary"
            or receipt.get("worker_id") != worker_id
            or int(receipt.get("seed", -1)) != seed
            or receipt.get("canonical_manifest_sha256") != _sha256(canonical_path)
            or receipt.get("amendment_id") != AMENDMENT_ID
            or receipt.get("amendment_sha256") != EXPECTED_AMENDMENT_SHA256
            or receipt.get("frozen_code_sha") != FROZEN_CODE_SHA
            or receipt.get("frozen_formal_runner_sha256")
            != FROZEN_FORMAL_RUNNER_SHA256
            or receipt.get("engineering_controller_sha256")
            not in (
                SEALED_SOURCE_CONTROLLER_SHA256S
                | {_sha256(Path(__file__).resolve())}
            )
            or receipt.get("pair_id") != PAIR_IDS[seed]
            or shard_role not in {"high", "low"}
            or receipt.get("physical_slot") != PHYSICAL_MAPPING[seed][shard_role]
            or source_kind not in {"execution", "historical_adoption"}
            or receipt.get("effect_fields_read") is not False
        ):
            raise ConfigError("primary bridge shard receipt identity failed")
        staging = _approved_root(
            str(receipt["approved_staging_root"]),
            label="approved_staging_root",
        )
        approved_receipts = _approved_root(
            str(receipt["approved_receipt_root"]),
            label="approved_receipt_root",
        )
        _inside_approved(approved_receipts, path, label="primary shard receipt")
        staging_roots.add(staging)
        receipt_roots.add(approved_receipts)
        approved_root_pairs.add((staging, approved_receipts))
        observed_sides.add(shard_role)
        partition_id = str(receipt.get("partition_id", ""))
        shard_id = str(receipt.get("shard_id", ""))
        if (
            not partition_id
            or not shard_id
            or shard_id in seen_shards
        ):
            raise ConfigError("primary bridge shard/partition identity 非法")
        partition_ids.add(partition_id)
        seen_shards.add(shard_id)
        attempt = Path(os.path.abspath(str(receipt["attempt_root"])))
        _deny_symlink_components(attempt, label="primary bridge attempt_root")
        if source_kind == "execution":
            _inside_approved(staging, attempt, label="primary bridge attempt_root")
            _verify_files_tsv(attempt)
        elif (
            attempt != Path(str(receipt["denied_partial_root"])).absolute()
            or receipt.get("historical_overall_completion_required") is not False
            or receipt.get("historical_source_immutable") is not True
        ):
            raise ConfigError("primary bridge historical source binding failed")
        tasks = receipt.get("tasks")
        if not isinstance(tasks, list) or receipt.get("selected_task_ids") != [
            row.get("task_id") for row in tasks if isinstance(row, dict)
        ]:
            raise ConfigError("primary bridge task list 非法")
        for row in tasks:
            if not isinstance(row, dict):
                raise ConfigError("primary bridge task row 非法")
            task_id = str(row.get("task_id", ""))
            if task_id in task_rows or task_id not in canonical_by_id:
                raise ConfigError("primary bridge task 重复或未知")
            marker_path = _strict_relative_target(
                attempt,
                f"task_receipts/{task_id}.json",
                label="primary bridge task receipt",
                must_exist=True,
            )
            marker = _read_json(marker_path)
            child = _strict_relative_target(
                attempt,
                str(marker.get("run_dir", "")),
                label="primary bridge task run_dir",
                must_exist=True,
            )
            files = _directory_manifest(child)
            canonical_task = canonical_by_id[task_id]
            identity_audit = marker.get("task_row", {}).get("identity_audit")
            expected_identity = canonical_task.get("expected_identity")
            actual_hands = _structural_hand_count(child)
            if (
                row.get("schema_version") != "task8b-paired-shard-task-v1"
                or row.get("status") != "complete"
                or marker.get("schema_version")
                != "task8-worker-task-receipt-v1"
                or marker.get("task_id") != task_id
                or marker.get("config_sha256") != canonical_task.get("config_sha256")
                or marker.get("files") != files
                or row.get("files") != files
                or row.get("task_receipt_path") != str(marker_path)
                or row.get("task_receipt_sha256") != _sha256(marker_path)
                or not isinstance(identity_audit, dict)
                or not isinstance(expected_identity, dict)
                or any(
                    identity_audit.get(field) != expected_identity.get(field)
                    for field in formal_runner.REQUIRED_IDENTITY_FIELDS
                )
                or actual_hands != int(canonical_task["planned_hands"])
                or int(row.get("actual_hands", -1)) != actual_hands
            ):
                raise ConfigError(f"primary bridge task integrity failed：{task_id}")
            task_rows[task_id] = (row, child)
        source_receipts.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "source_kind": source_kind,
                "shard_role": shard_role,
            }
        )
    if (
        set(task_rows) != set(EXPECTED_TASKS["primary"])
        or observed_sides != {"high", "low"}
        or len(partition_ids) not in {1, len(source_receipts)}
        or not staging_roots
        or not receipt_roots
    ):
        raise ConfigError("primary bridge P task union/high-low 未闭合")
    target_root = Path(os.path.abspath(bridge_root))
    target_receipt = Path(os.path.abspath(receipt_path))
    target_pairs = []
    for staging_root, approved_receipts in approved_root_pairs:
        try:
            target_root.relative_to(staging_root)
            target_receipt.relative_to(approved_receipts)
        except ValueError:
            continue
        target_pairs.append((staging_root, approved_receipts))
    if len(target_pairs) != 1:
        raise ConfigError("primary bridge target 未唯一绑定 source approved root pair")
    staging_root, approved_receipts = target_pairs[0]
    target_root = _inside_approved(
        staging_root,
        target_root,
        label="primary bridge root",
    )
    target_receipt = _inside_approved(
        approved_receipts,
        target_receipt,
        label="primary bridge receipt",
    )
    expected_receipt = _strict_relative_target(
        approved_receipts,
        str(canonical["receipt_relative_path"]),
        label="canonical receipt_relative_path",
        must_exist=False,
    )
    if target_receipt != expected_receipt:
        raise ConfigError("primary bridge receipt 未绑定 canonical relative path")
    if target_root.exists() or target_root.is_symlink():
        raise ConfigError("primary bridge root 已存在，拒绝覆盖")
    if target_receipt.exists() or target_receipt.is_symlink():
        raise ConfigError("primary bridge receipt 已存在，拒绝覆盖")
    target_root.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, Any]] = []
    checkpoint_files: list[str] = []
    checkpoint_suffix = f"checkpoint_{int(canonical['checkpoint_set'][-1]):04d}.json"
    identity_names = {
        "manifest.json",
        "resolved_config.yaml",
        "schedule_manifest.json",
        "task_identity_audit.json",
    }
    for task_id in EXPECTED_TASKS["primary"]:
        row, child = task_rows[task_id]
        for file_row in row["files"]:
            relative = str(file_row["relative_path"])
            name = Path(relative).name
            if not (name.endswith(checkpoint_suffix) or name in identity_names):
                continue
            source = _strict_relative_target(
                child,
                relative,
                label="primary bridge source checkpoint",
                must_exist=True,
            )
            destination_relative = (
                Path(str(row["run_dir"])) / Path(relative)
            ).as_posix()
            destination = _strict_relative_target(
                target_root,
                destination_relative,
                label="primary bridge destination",
                must_exist=False,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ConfigError("primary bridge checkpoint path collision")
            before_size = source.stat().st_size
            before_sha = _sha256(source)
            try:
                with source.open("rb") as source_handle, destination.open(
                    "xb"
                ) as destination_handle:
                    for block in iter(
                        lambda: source_handle.read(1024 * 1024),
                        b"",
                    ):
                        destination_handle.write(block)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
            except FileExistsError as exc:
                raise ConfigError("primary bridge checkpoint 非覆盖复制失败") from exc
            if (
                source.stat().st_size != before_size
                or _sha256(source) != before_sha
                or destination.stat().st_size != before_size
                or _sha256(destination) != before_sha
            ):
                raise ConfigError("primary bridge checkpoint 复制期间发生变化")
            checkpoint_files.append(destination_relative)
            copied.append(
                {
                    "task_id": task_id,
                    "source_path": str(source),
                    "relative_path": destination_relative,
                    "size": before_size,
                    "sha256": before_sha,
                }
            )
    if (
        not checkpoint_files
        or not any(Path(item).name.endswith(checkpoint_suffix) for item in checkpoint_files)
        or len(checkpoint_files) != len(set(checkpoint_files))
    ):
        raise ConfigError("primary bridge 最终 checkpoint 文件不闭合")
    for task_id, (row, child) in task_rows.items():
        if _directory_manifest(child) != row["files"]:
            raise ConfigError(f"primary bridge source changed during copy：{task_id}")
    for source_receipt in source_receipts:
        if _sha256(Path(source_receipt["path"])) != source_receipt["sha256"]:
            raise ConfigError("primary bridge shard receipt changed during copy")
    bridge_manifest = {
        "schema_version": PRIMARY_BRIDGE_SCHEMA,
        "status": "files_sealed_receipt_pending",
        "worker_id": worker_id,
        "seed": seed,
        "pair_id": PAIR_IDS[seed],
        "physical_mapping": PHYSICAL_MAPPING[seed],
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "frozen_code_sha": FROZEN_CODE_SHA,
        "frozen_formal_runner_sha256": FROZEN_FORMAL_RUNNER_SHA256,
        "engineering_controller_sha256": _sha256(Path(__file__).resolve()),
        "canonical_manifest_path": str(canonical_path),
        "canonical_manifest_sha256": _sha256(canonical_path),
        "source_receipts": sorted(source_receipts, key=lambda row: row["path"]),
        "task_union": list(EXPECTED_TASKS["primary"]),
        "checkpoint_files": copied,
        "effect_fields_read": False,
    }
    _write_json_new(target_root / "bridge_manifest.json", bridge_manifest)
    checkpoint_receipt = formal_runner.publish_checkpoint_receipt(
        checkpoint_root=target_root,
        checkpoint_files=sorted(checkpoint_files),
        receipt_path=target_receipt,
        producer_worker_id=worker_id,
        seed_bundle=seed,
        checkpoint_hand=int(canonical["checkpoint_set"][-1]),
        identity=dict(canonical["receipt_identity"]),
    )
    return {
        "bridge_manifest": bridge_manifest,
        "bridge_root": str(target_root),
        "checkpoint_receipt_path": str(target_receipt),
        "checkpoint_receipt_sha256": _sha256(target_receipt),
        "checkpoint_receipt": checkpoint_receipt,
    }


def compose_seed_pair(
    primary_manifest_path: str | Path,
    secondary_manifest_path: str | Path,
    shard_receipt_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate an exact P/S task union and publish its deterministic receipt."""

    canonical_paths = {
        "primary": Path(os.path.abspath(primary_manifest_path)),
        "secondary": Path(os.path.abspath(secondary_manifest_path)),
    }
    canonicals = {role: _read_json(path) for role, path in canonical_paths.items()}
    identities = {
        role: _canonical_identity(manifest)
        for role, manifest in canonicals.items()
    }
    if identities["primary"][1] != identities["secondary"][1]:
        raise ConfigError("paired-shard composition seed 不一致")
    seed = identities["primary"][1]
    if (
        identities["primary"][0],
        identities["secondary"][0],
    ) != PAIR_MAPPING[seed]:
        raise ConfigError("paired-shard composition P/S mapping 不匹配")

    task_rows: dict[str, dict[str, dict[str, Any]]] = {
        "primary": {},
        "secondary": {},
    }
    shard_rows = []
    seen_shards: set[tuple[str, str]] = set()
    side_slots: dict[str, set[str]] = {"high": set(), "low": set()}
    approved_receipt_roots: set[Path] = set()
    for raw_path in shard_receipt_paths:
        path = Path(os.path.abspath(raw_path))
        receipt = _read_json(path)
        role = str(receipt.get("role", ""))
        shard_role = str(receipt.get("shard_role", ""))
        source_kind = str(receipt.get("source_kind", ""))
        if (
            receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA
            or receipt.get("status") != "complete"
            or role not in task_rows
            or int(receipt.get("seed", -1)) != seed
            or receipt.get("worker_id") != identities[role][0]
            or receipt.get("canonical_manifest_sha256")
            != _sha256(canonical_paths[role])
            or receipt.get("amendment_id") != AMENDMENT_ID
            or receipt.get("amendment_sha256") != EXPECTED_AMENDMENT_SHA256
            or receipt.get("frozen_code_sha") != FROZEN_CODE_SHA
            or receipt.get("frozen_formal_runner_sha256")
            != FROZEN_FORMAL_RUNNER_SHA256
            or receipt.get("engineering_controller_sha256")
            not in (
                SEALED_SOURCE_CONTROLLER_SHA256S
                | {_sha256(Path(__file__).resolve())}
            )
            or receipt.get("pair_id") != PAIR_IDS[seed]
            or receipt.get("shard_role") not in {"high", "low"}
            or receipt.get("physical_slot")
            != PHYSICAL_MAPPING[seed][shard_role]
            or receipt.get("effect_fields_read") is not False
            or source_kind not in {"execution", "historical_adoption"}
            or (
                source_kind == "historical_adoption"
                and (role != "primary" or shard_role != "high")
            )
        ):
            raise ConfigError("paired-shard composition receipt identity failed")
        staging_root = _approved_root(
            str(receipt.get("approved_staging_root", "")),
            label="approved_staging_root",
        )
        approved_receipts = _approved_root(
            str(receipt.get("approved_receipt_root", "")),
            label="approved_receipt_root",
        )
        _inside_approved(approved_receipts, path, label="shard receipt")
        approved_receipt_roots.add(approved_receipts)
        shard_key = (str(receipt["worker_id"]), str(receipt["shard_id"]))
        if shard_key in seen_shards:
            raise ConfigError("paired-shard composition shard 重复")
        seen_shards.add(shard_key)
        tasks = receipt.get("tasks")
        if not isinstance(tasks, list) or receipt.get("selected_task_ids") != [
            row.get("task_id") for row in tasks if isinstance(row, dict)
        ]:
            raise ConfigError("paired-shard receipt task list 非法")
        selected_ids = {str(row["task_id"]) for row in tasks}
        if (
            not selected_ids
            or not selected_ids.issubset(SIDE_TASKS[(role, shard_role)])
        ):
            raise ConfigError("paired-shard receipt high/low task mapping failed")
        side_slots[shard_role].add(str(receipt["physical_slot"]))
        attempt_root = Path(os.path.abspath(str(receipt["attempt_root"])))
        denied_partial = Path(str(receipt.get("denied_partial_root", ""))).absolute()
        _deny_symlink_components(attempt_root, label="shard attempt_root")
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ConfigError("paired-shard composition attempt_root 非法")
        if source_kind == "execution":
            _inside_approved(staging_root, attempt_root, label="shard attempt_root")
            if _paths_overlap(attempt_root, denied_partial):
                raise ConfigError("paired-shard attempt 与 denied partial root 重叠")
            completion = _read_json(attempt_root / "completion_receipt.json")
            files_tsv = attempt_root / "files.tsv"
            if (
                completion.get("schema_version") != "task8-worker-completion-v1"
                or completion.get("status") != "complete"
                or completion.get("worker_id") != receipt.get("worker_id")
                or completion.get("files_tsv_sha256") != _sha256(files_tsv)
            ):
                raise ConfigError("paired-shard composition completion/files.tsv failed")
            _verify_files_tsv(attempt_root)
        elif (
            attempt_root != denied_partial
            or receipt.get("historical_overall_completion_required") is not False
            or receipt.get("historical_source_immutable") is not True
        ):
            raise ConfigError("paired-shard historical adoption source binding failed")
        canonical_by_id = {
            str(task["task_id"]): task for task in canonicals[role]["task_configs"]
        }
        for row in tasks:
            if not isinstance(row, dict):
                raise ConfigError("paired-shard receipt task row 非法")
            task_id = str(row.get("task_id", ""))
            if task_id in task_rows[role] or task_id not in canonical_by_id:
                raise ConfigError("paired-shard composition task 重复或未知")
            marker_path = Path(os.path.abspath(str(row.get("task_receipt_path", ""))))
            _deny_symlink_components(marker_path, label="task receipt path")
            marker = _read_json(marker_path)
            expected_marker = _strict_relative_target(
                attempt_root,
                f"task_receipts/{task_id}.json",
                label="task receipt path",
                must_exist=True,
            )
            if marker_path != expected_marker:
                raise ConfigError("paired-shard task receipt path binding failed")
            child = _strict_relative_target(
                attempt_root,
                str(marker.get("run_dir", "")),
                label="composition task run_dir",
                must_exist=True,
            )
            files = _directory_manifest(child)
            actual_hands = _structural_hand_count(child)
            canonical_task = canonical_by_id[task_id]
            identity_audit = marker.get("task_row", {}).get("identity_audit")
            expected_identity = canonical_task.get("expected_identity")
            if (
                row.get("schema_version") != "task8b-paired-shard-task-v1"
                or row.get("status") != "complete"
                or _sha256(marker_path) != row.get("task_receipt_sha256")
                or marker.get("schema_version")
                != "task8-worker-task-receipt-v1"
                or marker.get("task_id") != task_id
                or marker.get("config_sha256") != canonical_task.get("config_sha256")
                or marker.get("run_dir") != row.get("run_dir")
                or marker.get("files") != files
                or row.get("files") != files
                or not isinstance(identity_audit, dict)
                or not isinstance(expected_identity, dict)
                or any(
                    identity_audit.get(field) != expected_identity.get(field)
                    for field in formal_runner.REQUIRED_IDENTITY_FIELDS
                )
                or int(row.get("planned_hands", -1))
                != int(canonical_task.get("planned_hands", -2))
                or int(row.get("actual_hands", -1)) != actual_hands
                or actual_hands != int(canonical_task.get("planned_hands", -2))
            ):
                raise ConfigError(f"paired-shard composition hash/hands failed：{task_id}")
            task_rows[role][task_id] = row
        shard_rows.append(
            {
                "worker_id": receipt["worker_id"],
                "shard_id": receipt["shard_id"],
                "partition_id": receipt["partition_id"],
                "physical_slot": receipt["physical_slot"],
                "shard_role": shard_role,
                "source_kind": source_kind,
                "receipt_path": str(path),
                "receipt_sha256": _sha256(path),
            }
        )

    for role in ("primary", "secondary"):
        role_shards = [
            row for row in shard_rows if str(row["worker_id"]) == identities[role][0]
        ]
        if (
            not role_shards
            or {str(row["shard_role"]) for row in role_shards} != {"high", "low"}
            or len({str(row["partition_id"]) for row in role_shards})
            not in {1, len(role_shards)}
        ):
            raise ConfigError(f"paired-shard {role} high/low execution 不闭合")
    if side_slots["high"] & side_slots["low"]:
        raise ConfigError("paired-shard physical slot 不得同时承担 high/low")
    for role, expected in EXPECTED_TASKS.items():
        if set(task_rows[role]) != set(expected):
            raise ConfigError(f"paired-shard {role} task union 有缺口")
        if sum(
            int(row["planned_hands"]) for row in task_rows[role].values()
        ) != sum(expected.values()):
            raise ConfigError(f"paired-shard {role} planned hands 未闭合")
    secondary_by_side = {
        side: {
            task_id
            for task_id in task_rows["secondary"]
            if task_id in SIDE_TASKS[("secondary", side)]
        }
        for side in ("high", "low")
    }
    if secondary_by_side != {
        "low": {"mixed_ecological"},
        "high": {
            "expr_online",
            "expr_without",
            "async_online",
            "async_without",
        },
    }:
        raise ConfigError("paired-shard secondary 2700/2400 partition 不匹配")
    body = {
        "schema_version": COMPOSITION_SCHEMA,
        "status": "complete",
        "canonical_scope": {
            "seed_count": 6,
            "logical_worker_count": 12,
            "seeds": list(FORMAL_SEEDS),
            "execution_shards_are_not_workers": True,
        },
        "seed": seed,
        "pair_mapping": list(PAIR_MAPPING[seed]),
        "canonical_manifests": {
            role: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for role, path in canonical_paths.items()
        },
        "planned_hands": {
            role: sum(EXPECTED_TASKS[role].values())
            for role in ("primary", "secondary")
        },
        "shards": sorted(
            shard_rows,
            key=lambda row: (str(row["worker_id"]), str(row["shard_id"])),
        ),
        "task_union": {
            role: [
                {
                    "task_id": task_id,
                    "planned_hands": int(task_rows[role][task_id]["planned_hands"]),
                    "task_receipt_sha256": task_rows[role][task_id][
                        "task_receipt_sha256"
                    ],
                }
                for task_id in EXPECTED_TASKS[role]
            ]
            for role in ("primary", "secondary")
        },
        "effect_fields_read": False,
    }
    body["composition_sha256"] = hashlib.sha256(_json_bytes(body)).hexdigest()
    if len(approved_receipt_roots) != 1:
        raise ConfigError("paired-shard composition approved receipt roots 不一致")
    composition_output = _inside_approved(
        next(iter(approved_receipt_roots)),
        Path(output_path),
        label="composition output",
    )
    _write_json_new(composition_output, body)
    return body


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or seal a frozen TASK8B six-seed paired shard."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render-authorization")
    render_parser.add_argument("--canonical-manifest", type=Path, required=True)
    render_parser.add_argument("--amendment", type=Path, required=True)
    render_parser.add_argument("--physical-slot", required=True)
    render_parser.add_argument("--side", choices=("low", "high"), required=True)
    render_parser.add_argument(
        "--selected-task", action="append", required=True
    )
    render_parser.add_argument("--scientific-checkout", type=Path, required=True)
    render_parser.add_argument("--staging-root", type=Path, required=True)
    render_parser.add_argument("--receipt-root", type=Path, required=True)
    render_parser.add_argument("--shard-id", required=True)
    render_parser.add_argument("--output", type=Path, required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--canonical-manifest", type=Path, required=True
    )
    preflight_parser.add_argument("--authorization", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--canonical-manifest", type=Path, required=True)
    run_parser.add_argument("--authorization", type=Path, required=True)
    run_parser.add_argument("--derived-manifest-output", type=Path, required=True)
    run_parser.add_argument("--receipt-root", type=Path, required=True)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--derived-manifest", type=Path, required=True)
    seal_parser.add_argument("--run-dir", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.add_argument("--recovery-certificate", type=Path)

    recovery_parser = subparsers.add_parser("recover-completed-execution")
    recovery_parser.add_argument(
        "--derived-manifest", type=Path, required=True
    )
    recovery_parser.add_argument("--run-dir", type=Path, required=True)
    recovery_parser.add_argument("--baseline", type=Path, required=True)
    recovery_parser.add_argument(
        "--pre-recovery-archive", type=Path, required=True
    )
    recovery_parser.add_argument(
        "--recovery-certificate", type=Path, required=True
    )
    recovery_parser.add_argument(
        "--recovery-ledger-entry", type=Path, required=True
    )

    adopt_parser = subparsers.add_parser("adopt-historical")
    adopt_parser.add_argument("--canonical-manifest", type=Path, required=True)
    adopt_parser.add_argument("--authorization", type=Path, required=True)
    adopt_parser.add_argument("--canonical-run-dir", type=Path, required=True)
    adopt_parser.add_argument("--output", type=Path, required=True)

    bridge_parser = subparsers.add_parser("compose-primary")
    bridge_parser.add_argument("--primary-manifest", type=Path, required=True)
    bridge_parser.add_argument(
        "--shard-receipt", type=Path, action="append", required=True
    )
    bridge_parser.add_argument("--bridge-root", type=Path, required=True)
    bridge_parser.add_argument("--receipt", type=Path, required=True)

    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--primary-manifest", type=Path, required=True)
    compose_parser.add_argument("--secondary-manifest", type=Path, required=True)
    compose_parser.add_argument(
        "--shard-receipt", type=Path, action="append", required=True
    )
    compose_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "render-authorization":
        result = render_authorization(
            args.canonical_manifest,
            args.amendment,
            physical_slot=args.physical_slot,
            shard_role=args.side,
            selected_task_ids=args.selected_task,
            scientific_checkout=args.scientific_checkout,
            approved_staging_root=args.staging_root,
            approved_receipt_root=args.receipt_root,
            shard_id=args.shard_id,
            output_path=args.output,
        )
    elif args.command == "preflight":
        result = preflight_authorization(
            args.canonical_manifest,
            args.authorization,
        )
    elif args.command == "run":
        result = run_authorized_shard(
            args.canonical_manifest,
            args.authorization,
            derived_manifest_output=args.derived_manifest_output,
            receipt_root=args.receipt_root,
        )
    elif args.command == "seal":
        result = build_shard_receipt(
            args.derived_manifest,
            args.run_dir,
            args.output,
            args.recovery_certificate,
        )
    elif args.command == "recover-completed-execution":
        result = recover_completed_execution(
            args.derived_manifest,
            args.run_dir,
            args.baseline,
            args.pre_recovery_archive,
            args.recovery_certificate,
            args.recovery_ledger_entry,
        )
    elif args.command == "adopt-historical":
        result = build_historical_adoption_receipt(
            args.canonical_manifest,
            args.authorization,
            args.canonical_run_dir,
            args.output,
        )
    elif args.command == "compose-primary":
        result = compose_primary_checkpoint(
            args.primary_manifest,
            args.shard_receipt,
            args.bridge_root,
            args.receipt,
        )
    else:
        result = compose_seed_pair(
            args.primary_manifest,
            args.secondary_manifest,
            args.shard_receipt,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
