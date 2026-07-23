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
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import ConfigError
from agentmemeval.experiments import formal_runner

AUTHORIZATION_SCHEMA = "task8b-paired-shard-authorization-v1"
SHARD_RECEIPT_SCHEMA = "task8b-paired-shard-receipt-v1"
COMPOSITION_SCHEMA = "task8b-paired-shard-composition-v1"
PRIMARY_BRIDGE_SCHEMA = "task8b-primary-checkpoint-bridge-v1"
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
    identity_material = {
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "canonical_manifest_sha256": _sha256(canonical_path),
        "controller_sha256": _sha256(Path(__file__).resolve()),
        "pair_id": PAIR_IDS[seed],
        "physical_slot": physical_slot,
        "role": role,
        "seed": seed,
        "selected_task_ids": selected,
        "shard_id": shard_id,
        "shard_role": shard_role,
        "worker_id": worker_id,
    }
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "active": True,
        "authorization_id": (
            "auth-" + hashlib.sha256(_json_bytes(identity_material)).hexdigest()
        ),
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
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("active") is not True
        or authorization.get("amendment_id") != AMENDMENT_ID
        or authorization.get("frozen_code_sha") != FROZEN_CODE_SHA
        or authorization.get("frozen_formal_runner_sha256")
        != FROZEN_FORMAL_RUNNER_SHA256
        or authorization.get("engineering_controller_sha256")
        != _sha256(Path(__file__).resolve())
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
            != _sha256(Path(__file__).resolve())
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
    except (
        OSError,
        subprocess.CalledProcessError,
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


def build_shard_receipt(
    derived_manifest_path: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
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
    rebuilt = derive_authorized_manifest(
        str(shard.get("canonical_manifest_path", "")),
        str(shard.get("authorization_path", "")),
    )
    if _json_bytes(rebuilt) != _json_bytes(manifest):
        raise ConfigError("derived manifest 不再匹配冻结 authorization")
    staging_root = _approved_root(
        str(shard["approved_staging_root"]),
        label="approved_staging_root",
    )
    approved_receipts = _approved_root(
        str(shard["approved_receipt_root"]),
        label="approved_receipt_root",
    )
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
            != _sha256(Path(__file__).resolve())
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
        or len(partition_ids) != 1
        or len(staging_roots) != 1
        or len(receipt_roots) != 1
    ):
        raise ConfigError("primary bridge P task union/high-low 未闭合")
    staging_root = next(iter(staging_roots))
    approved_receipts = next(iter(receipt_roots))
    target_root = _inside_approved(
        staging_root,
        Path(bridge_root),
        label="primary bridge root",
    )
    target_receipt = _inside_approved(
        approved_receipts,
        Path(receipt_path),
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
            != _sha256(Path(__file__).resolve())
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
            or len({str(row["partition_id"]) for row in role_shards}) != 1
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
