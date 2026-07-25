"""Append-only TASK8B Phase F provisional-to-canonical adoption comparator.

The comparator never parses formal task payloads.  It reads only the provisional
SHA manifest and the canonical ``analysis_input.json`` topology metadata, then
rehashes the canonical consumption namespace byte-for-byte.  Files outside the
Phase F consumption namespace (for example canonical provenance receipts) are
deliberately ignored.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.core.errors import ConfigError

EXPECTED_WORKERS = tuple(f"{role}{index:02d}" for role in ("P", "S") for index in range(7, 13))
PROVISIONAL_INPUT_PREFIX = "01_INPUT/"
PROVISIONAL_STATUS = "PROVISIONAL_UNVERIFIED"
PROVISIONAL_PAPER_STATUS = "NOT_YET_PAPER_ELIGIBLE"
CANONICAL_SOURCE_MAP_SHA256 = (
    "14c50a5a3088aa22d293f43390911fc2dce169b8ef8cde954e1dad77d9984433"
)
CANONICAL_ROOT_SEAL_SCHEMA = "task8b-six-seed-canonical-root-seal-v1"
CANONICAL_PAPER_ELIGIBILITY = "PENDING_DECLARED_LIMITATION_REVIEW"
ADOPTED_PAPER_ELIGIBILITY = "FORMAL_ADOPTED_PENDING_DECLARED_LIMITATION_REVIEW"
CANONICAL_FILE_MANIFEST = "canonical_file_manifest.tsv"
CANONICAL_ROOT_SEAL = "canonical_root_seal.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SIZE_RE = re.compile(r"0|[1-9][0-9]*")
CONSUMPTION_ROOT_FILES = ("analysis_input.json", "exclusion_ledger.csv")
CONSUMPTION_RECURSIVE_ROOTS = ("manifests", "snapshots")
OUTPUT_FILES = (
    "canonical_rehash_sha256.tsv",
    "comparison.tsv",
    "materialization_completion_receipt.json",
    "provisional_adoption_receipt.json",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as exc:
        raise ConfigError(f"无法读取文件身份：{path}: {exc}") from exc
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def _require_plain_directory(path: Path, *, label: str) -> None:
    if _is_reparse_or_symlink(path) or not path.is_dir():
        raise ConfigError(f"{label} 缺失、非目录或为 symlink/reparse point：{path}")


def _safe_relative_path(value: Any, *, label: str) -> str:
    text = str(value)
    if not text or "\\" in text:
        raise ConfigError(f"{label} 非安全 POSIX 相对路径：{text!r}")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ConfigError(f"{label} 非安全 POSIX 相对路径：{text!r}")
    return text


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def _read_json_metadata(path: Path, *, label: str) -> dict[str, Any]:
    if _is_reparse_or_symlink(path) or not path.is_file():
        raise ConfigError(f"{label} 缺失、非文件或为 symlink/reparse point：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} 不是有效 UTF-8 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{label} 顶层必须为对象：{path}")
    return value


def _canonical_structure_mapping(canonical_root: Path) -> dict[str, Any]:
    """Derive the consumption mapping from canonical analysis metadata."""

    manifest = _read_json_metadata(
        canonical_root / "analysis_input.json",
        label="canonical analysis_input",
    )
    workers = manifest.get("workers")
    if not isinstance(workers, list) or len(workers) != len(EXPECTED_WORKERS):
        raise ConfigError("canonical analysis_input 必须精确包含 12 workers")
    by_worker: dict[str, dict[str, Any]] = {}
    for raw in workers:
        if not isinstance(raw, dict):
            raise ConfigError("canonical analysis_input worker 非对象")
        worker_id = str(raw.get("worker_id", ""))
        if worker_id in by_worker:
            raise ConfigError(f"canonical analysis_input worker 重复：{worker_id}")
        by_worker[worker_id] = raw
    if tuple(sorted(by_worker)) != tuple(sorted(EXPECTED_WORKERS)):
        raise ConfigError("canonical analysis_input worker 集合非 P07-P12/S07-S12")

    attempt_prefixes: list[str] = []
    composition_paths: list[str] = []
    execution_prefixes: list[str] = []
    worker_manifest_paths: list[str] = []
    for worker_id in EXPECTED_WORKERS:
        worker = by_worker[worker_id]
        attempts = worker.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ConfigError(f"{worker_id} canonical attempts 必须精确为 1")
        attempt = attempts[0]
        if not isinstance(attempt, dict):
            raise ConfigError(f"{worker_id} canonical attempt 非对象")
        attempt_prefix = _safe_relative_path(
            attempt.get("relative_path"),
            label=f"{worker_id}.attempt.relative_path",
        )
        if not attempt_prefix.startswith(f"snapshots/{worker_id}/"):
            raise ConfigError(f"{worker_id} canonical attempt 路径未绑定 worker")
        attempt_prefixes.append(attempt_prefix)

        lineage = worker.get("shard_lineage")
        if not isinstance(lineage, dict):
            raise ConfigError(f"{worker_id} canonical shard_lineage 非对象")
        composition_path = _safe_relative_path(
            lineage.get("composition_receipt_path"),
            label=f"{worker_id}.composition_receipt_path",
        )
        expected_composition = f"manifests/composition/{worker_id}.json"
        if composition_path != expected_composition:
            raise ConfigError(f"{worker_id} canonical composition 路径非法")
        composition_paths.append(composition_path)
        execution_prefixes.append(f"manifests/composition/execution/{worker_id}")
        worker_manifest_paths.append(f"manifests/{worker_id}.json")

    exact_paths = {
        *CONSUMPTION_ROOT_FILES,
        "manifests/manifest_index.json",
        *composition_paths,
        *worker_manifest_paths,
    }
    return {
        "worker_ids": list(EXPECTED_WORKERS),
        "attempt_prefixes": attempt_prefixes,
        "composition_paths": composition_paths,
        "execution_prefixes": execution_prefixes,
        "worker_manifest_paths": worker_manifest_paths,
        "exact_paths": exact_paths,
    }


def _path_is_mapped(path: str, mapping: dict[str, Any]) -> bool:
    if path in mapping["exact_paths"]:
        return True
    return any(
        _is_under(path, prefix)
        for prefix in (
            *mapping["attempt_prefixes"],
            *mapping["execution_prefixes"],
        )
    )


def _read_provisional_manifest(provisional_root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    manifest_path = (
        provisional_root / "03_RECEIPTS" / "phase_f_consumed_input_sha256.tsv"
    )
    if _is_reparse_or_symlink(manifest_path) or not manifest_path.is_file():
        raise ConfigError("provisional consumed-input SHA manifest 缺失或为 link/reparse")
    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_header = [
            "relative_path",
            "bytes",
            "sha256",
            "evidence_status",
            "paper_eligibility",
        ]
        if reader.fieldnames != expected_header:
            raise ConfigError("provisional consumed-input SHA manifest header 非法")
        for line_number, raw in enumerate(reader, start=2):
            source_path = str(raw.get("relative_path", ""))
            if not source_path.startswith(PROVISIONAL_INPUT_PREFIX):
                raise ConfigError(
                    f"provisional manifest 第 {line_number} 行未以 01_INPUT/ 开头"
                )
            relative_path = _safe_relative_path(
                source_path.removeprefix(PROVISIONAL_INPUT_PREFIX),
                label=f"provisional manifest line {line_number}",
            )
            size_text = str(raw.get("bytes", ""))
            sha256 = str(raw.get("sha256", ""))
            if SIZE_RE.fullmatch(size_text) is None:
                raise ConfigError(f"provisional manifest 第 {line_number} 行 bytes 非法")
            if SHA256_RE.fullmatch(sha256) is None:
                raise ConfigError(f"provisional manifest 第 {line_number} 行 SHA-256 非法")
            if (
                raw.get("evidence_status") != PROVISIONAL_STATUS
                or raw.get("paper_eligibility") != PROVISIONAL_PAPER_STATUS
            ):
                raise ConfigError(f"provisional manifest 第 {line_number} 行状态非法")
            if relative_path in rows:
                raise ConfigError(f"provisional manifest 路径重复：{relative_path}")
            rows[relative_path] = {
                "bytes": int(size_text),
                "sha256": sha256,
            }
    if not rows:
        raise ConfigError("provisional consumed-input SHA manifest 为空")
    return manifest_path, rows


def _iter_plain_files(root: Path) -> Iterable[Path]:
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            if _is_reparse_or_symlink(path):
                raise ConfigError(f"canonical consumption tree 含 link/reparse：{path}")
        for name in sorted(file_names):
            path = current_path / name
            if _is_reparse_or_symlink(path) or not path.is_file():
                raise ConfigError(f"canonical consumption tree 含非普通文件：{path}")
            yield path


def _read_canonical_root_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if _is_reparse_or_symlink(path) or not path.is_file():
        raise ConfigError("canonical_file_manifest.tsv 缺失或为 link/reparse")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["relative_path", "size", "sha256"]:
            raise ConfigError("canonical_file_manifest.tsv header 非法")
        for line_number, raw in enumerate(reader, start=2):
            relative_path = _safe_relative_path(
                raw.get("relative_path"),
                label=f"canonical root manifest line {line_number}",
            )
            size_text = str(raw.get("size", ""))
            sha256 = str(raw.get("sha256", ""))
            if relative_path in {CANONICAL_FILE_MANIFEST, CANONICAL_ROOT_SEAL}:
                raise ConfigError("canonical root manifest 不得包含 manifest/seal 自身")
            if SIZE_RE.fullmatch(size_text) is None:
                raise ConfigError(f"canonical root manifest 第 {line_number} 行 size 非法")
            if SHA256_RE.fullmatch(sha256) is None:
                raise ConfigError(f"canonical root manifest 第 {line_number} 行 SHA-256 非法")
            if relative_path in rows:
                raise ConfigError(f"canonical root manifest 路径重复：{relative_path}")
            rows[relative_path] = {"size": int(size_text), "sha256": sha256}
    if not rows:
        raise ConfigError("canonical root manifest 为空")
    return rows


def _validate_canonical_root_closure(canonical_root: Path) -> dict[str, Any]:
    """Independently validate the manifest/seal closure over the full tree."""

    manifest_path = canonical_root / CANONICAL_FILE_MANIFEST
    seal_path = canonical_root / CANONICAL_ROOT_SEAL
    manifest_rows = _read_canonical_root_manifest(manifest_path)
    seal = _read_json_metadata(seal_path, label="canonical root seal")
    limitations = seal.get("limitations")
    if (
        seal.get("schema_version") != CANONICAL_ROOT_SEAL_SCHEMA
        or seal.get("status") != "sealed_with_declared_limitation"
        or seal.get("source_map_sha256") != CANONICAL_SOURCE_MAP_SHA256
        or seal.get("paper_eligibility") != CANONICAL_PAPER_ELIGIBILITY
        or not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(row, dict) or not row for row in limitations)
        or seal.get("effect_fields_read") is not False
        or seal.get("formal_effect_fields_read") is not False
    ):
        raise ConfigError("canonical root seal schema/status/limitation 非法")
    manifest_ref = seal.get("canonical_file_manifest")
    if not isinstance(manifest_ref, dict):
        raise ConfigError("canonical root seal manifest reference 非对象")
    manifest_sha256 = _sha256_file(manifest_path)
    manifest_size = manifest_path.stat().st_size
    manifest_bytes = sum(row["size"] for row in manifest_rows.values())
    if (
        manifest_ref.get("relative_path") != CANONICAL_FILE_MANIFEST
        or manifest_ref.get("size") != manifest_size
        or manifest_ref.get("sha256") != manifest_sha256
        or seal.get("manifest_entry_count") != len(manifest_rows)
        or seal.get("manifest_entry_bytes") != manifest_bytes
        or seal.get("published_tree_file_count") != len(manifest_rows) + 2
    ):
        raise ConfigError("canonical root seal manifest SHA/count/bytes 非法")

    actual_files: dict[str, Path] = {}
    for path in _iter_plain_files(canonical_root):
        relative_path = _safe_relative_path(
            path.relative_to(canonical_root).as_posix(),
            label="canonical full-tree file",
        )
        actual_files[relative_path] = path
    reserved = {CANONICAL_FILE_MANIFEST, CANONICAL_ROOT_SEAL}
    actual_manifest_paths = set(actual_files) - reserved
    if set(manifest_rows) != actual_manifest_paths or not reserved.issubset(actual_files):
        missing = sorted(set(manifest_rows) - actual_manifest_paths)
        extra = sorted(actual_manifest_paths - set(manifest_rows))
        raise ConfigError(
            "canonical root manifest path closure 非法："
            f"missing={len(missing)} extra={len(extra)}"
        )
    for relative_path, expected in manifest_rows.items():
        path = actual_files[relative_path]
        if path.stat().st_size != expected["size"] or _sha256_file(path) != expected["sha256"]:
            raise ConfigError(f"canonical root manifest size/SHA 不闭合：{relative_path}")

    seal_sha256 = _sha256_file(seal_path)
    full_tree_bytes = manifest_bytes + manifest_size + seal_path.stat().st_size
    return {
        "source_map_sha256": CANONICAL_SOURCE_MAP_SHA256,
        "canonical_file_manifest_path": str(manifest_path),
        "canonical_file_manifest_sha256": manifest_sha256,
        "canonical_root_seal_path": str(seal_path),
        "canonical_root_seal_sha256": seal_sha256,
        "manifest_entry_count": len(manifest_rows),
        "manifest_entry_bytes": manifest_bytes,
        "full_tree_file_count": len(actual_files),
        "full_tree_bytes": full_tree_bytes,
        "paper_eligibility": seal["paper_eligibility"],
        "limitations": limitations,
    }


def _canonical_consumption_files(canonical_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in CONSUMPTION_ROOT_FILES:
        path = canonical_root / relative
        if path.exists() or path.is_symlink():
            if _is_reparse_or_symlink(path) or not path.is_file():
                raise ConfigError(f"canonical consumption root file 非普通文件：{path}")
            files[relative] = path
    for relative_root in CONSUMPTION_RECURSIVE_ROOTS:
        root = canonical_root / relative_root
        if not root.exists() and not root.is_symlink():
            continue
        _require_plain_directory(root, label=f"canonical {relative_root}")
        for path in _iter_plain_files(root):
            relative = path.relative_to(canonical_root).as_posix()
            files[_safe_relative_path(relative, label="canonical consumption file")] = path
    return files


def _tsv_bytes(headers: Sequence[str], rows: Iterable[dict[str, Any]]) -> bytes:
    output: list[str] = ["\t".join(headers)]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        if any("\t" in value or "\r" in value or "\n" in value for value in values):
            raise ConfigError("TSV field 含制表符或换行")
        output.append("\t".join(values))
    return ("\n".join(output) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _output_manifest_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in OUTPUT_FILES:
        path = output_dir / name
        rows.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def compare_and_adopt_task8b_phase_f(
    canonical_root: str | Path,
    provisional_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rehash canonical inputs and publish an append-only adoption decision."""

    canonical = Path(canonical_root).absolute()
    provisional = Path(provisional_root).absolute()
    output = Path(output_dir).absolute()
    _require_plain_directory(canonical, label="canonical root")
    _require_plain_directory(provisional, label="provisional root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"adoption output 已存在，拒绝覆盖：{output}")
    if not output.parent.is_dir() or _is_reparse_or_symlink(output.parent):
        raise ConfigError("adoption output parent 缺失或为 link/reparse")
    if canonical == provisional:
        raise ConfigError("canonical root 与 provisional root 不得相同")
    if canonical in output.parents or provisional in output.parents:
        raise ConfigError("adoption output 不得位于 canonical/provisional tree 内")

    root_closure = _validate_canonical_root_closure(canonical)
    mapping = _canonical_structure_mapping(canonical)
    provisional_manifest, provisional_rows = _read_provisional_manifest(provisional)
    unmapped = sorted(path for path in provisional_rows if not _path_is_mapped(path, mapping))
    if unmapped:
        raise ConfigError(
            "provisional consumption path 未被 canonical topology 映射："
            f"{unmapped[0]}"
        )

    canonical_files = _canonical_consumption_files(canonical)
    canonical_rows: dict[str, dict[str, Any]] = {}
    for relative_path, path in sorted(canonical_files.items()):
        canonical_rows[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    comparison_rows: list[dict[str, Any]] = []
    counts = {
        "match": 0,
        "missing_canonical": 0,
        "extra_canonical": 0,
        "size_mismatch": 0,
        "sha256_mismatch": 0,
        "size_and_sha256_mismatch": 0,
    }
    for relative_path in sorted(set(provisional_rows) | set(canonical_rows)):
        provisional_row = provisional_rows.get(relative_path)
        canonical_row = canonical_rows.get(relative_path)
        if provisional_row is None:
            status = "extra_canonical"
        elif canonical_row is None:
            status = "missing_canonical"
        else:
            size_equal = provisional_row["bytes"] == canonical_row["bytes"]
            sha_equal = provisional_row["sha256"] == canonical_row["sha256"]
            if size_equal and sha_equal:
                status = "match"
            elif not size_equal and not sha_equal:
                status = "size_and_sha256_mismatch"
            elif not size_equal:
                status = "size_mismatch"
            else:
                status = "sha256_mismatch"
        counts[status] += 1
        comparison_rows.append(
            {
                "relative_path": relative_path,
                "status": status,
                "provisional_bytes": "" if provisional_row is None else provisional_row["bytes"],
                "canonical_bytes": "" if canonical_row is None else canonical_row["bytes"],
                "provisional_sha256": (
                    "" if provisional_row is None else provisional_row["sha256"]
                ),
                "canonical_sha256": "" if canonical_row is None else canonical_row["sha256"],
            }
        )

    mismatch_count = len(comparison_rows) - counts["match"]
    exact_equality = mismatch_count == 0 and len(canonical_rows) == len(provisional_rows)
    created_at = _utc_now()
    output.mkdir(exist_ok=False)
    rehash_path = output / "canonical_rehash_sha256.tsv"
    comparison_path = output / "comparison.tsv"
    materialization_path = output / "materialization_completion_receipt.json"
    adoption_path = output / "provisional_adoption_receipt.json"
    _write_new(
        rehash_path,
        _tsv_bytes(
            ("relative_path", "bytes", "sha256"),
            (
                {"relative_path": path, **row}
                for path, row in sorted(canonical_rows.items())
            ),
        ),
    )
    _write_new(
        comparison_path,
        _tsv_bytes(
            (
                "relative_path",
                "status",
                "provisional_bytes",
                "canonical_bytes",
                "provisional_sha256",
                "canonical_sha256",
            ),
            comparison_rows,
        ),
    )
    materialization_receipt = {
        "schema_version": "task8b-phase-f-materialization-completion-v1",
        "status": "CANONICAL_FULL_TREE_SHA_CLOSED",
        "created_at_utc": created_at,
        "canonical_root": str(canonical),
        "full_canonical_validation_performed": True,
        "source_map_sha256": root_closure["source_map_sha256"],
        "canonical_file_manifest_path": root_closure[
            "canonical_file_manifest_path"
        ],
        "canonical_file_manifest_sha256": root_closure[
            "canonical_file_manifest_sha256"
        ],
        "canonical_root_seal_path": root_closure["canonical_root_seal_path"],
        "canonical_root_seal_sha256": root_closure[
            "canonical_root_seal_sha256"
        ],
        "canonical_full_tree_file_count": root_closure["full_tree_file_count"],
        "canonical_full_tree_bytes": root_closure["full_tree_bytes"],
        "canonical_manifest_entry_count": root_closure["manifest_entry_count"],
        "canonical_manifest_entry_bytes": root_closure["manifest_entry_bytes"],
        "canonical_paper_eligibility": root_closure["paper_eligibility"],
        "declared_limitations": root_closure["limitations"],
        "canonical_consumption_file_count": len(canonical_rows),
        "canonical_consumption_bytes": sum(row["bytes"] for row in canonical_rows.values()),
        "canonical_rehash_manifest_path": rehash_path.name,
        "canonical_rehash_manifest_sha256": _sha256_file(rehash_path),
        "analysis_input_sha256": canonical_rows.get("analysis_input.json", {}).get("sha256"),
        "consumption_mapping": {
            "metadata_source": "canonical analysis_input.json",
            "root_files": list(CONSUMPTION_ROOT_FILES),
            "recursive_roots": list(CONSUMPTION_RECURSIVE_ROOTS),
            "worker_ids": mapping["worker_ids"],
            "attempt_prefixes": mapping["attempt_prefixes"],
            "ignored_outside_consumption_scope": True,
        },
        "formal_task_payloads_parsed": False,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
    }
    _write_new(materialization_path, _json_bytes(materialization_receipt))
    adoption_receipt = {
        "schema_version": "task8b-phase-f-provisional-adoption-v1",
        "status": "adopted" if exact_equality else "isolated",
        "adoption_status": (
            "FORMAL_ADOPTED" if exact_equality else "ISOLATED_CANONICAL_MISMATCH"
        ),
        "created_at_utc": created_at,
        "canonical_root": str(canonical),
        "provisional_root": str(provisional),
        "provisional_consumed_input_manifest_path": str(provisional_manifest),
        "provisional_consumed_input_manifest_sha256": _sha256_file(provisional_manifest),
        "canonical_rehash_manifest_path": rehash_path.name,
        "canonical_rehash_manifest_sha256": _sha256_file(rehash_path),
        "comparison_path": comparison_path.name,
        "comparison_sha256": _sha256_file(comparison_path),
        "materialization_completion_receipt_path": materialization_path.name,
        "materialization_completion_receipt_sha256": _sha256_file(materialization_path),
        "full_canonical_validation_performed": True,
        "canonical_file_manifest_sha256": root_closure[
            "canonical_file_manifest_sha256"
        ],
        "canonical_root_seal_sha256": root_closure[
            "canonical_root_seal_sha256"
        ],
        "declared_limitations": root_closure["limitations"],
        "provisional_consumed_file_count": len(provisional_rows),
        "canonical_consumption_file_count": len(canonical_rows),
        "comparison_row_count": len(comparison_rows),
        "mismatch_count": mismatch_count,
        "comparison_counts": counts,
        "exact_path_size_sha256_equality": exact_equality,
        "adopted_without_rerun": exact_equality,
        "formal_adopted": exact_equality,
        "evidence_status": (
            "CANONICAL_VERIFIED_ADOPTED"
            if exact_equality
            else "PROVISIONAL_UNVERIFIED_ISOLATED"
        ),
        "paper_eligibility": (
            ADOPTED_PAPER_ELIGIBILITY
            if exact_equality
            else PROVISIONAL_PAPER_STATUS
        ),
        "provisional_tree_modified": False,
        "formal_task_payloads_parsed": False,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
    }
    _write_new(adoption_path, _json_bytes(adoption_receipt))
    package_path = output / "adoption_package_sha256.tsv"
    _write_new(
        package_path,
        _tsv_bytes(
            ("relative_path", "bytes", "sha256"),
            _output_manifest_rows(output),
        ),
    )
    return {
        "status": adoption_receipt["status"],
        "adoption_status": adoption_receipt["adoption_status"],
        "output_dir": str(output),
        "provisional_consumed_file_count": len(provisional_rows),
        "canonical_consumption_file_count": len(canonical_rows),
        "mismatch_count": mismatch_count,
        "adopted_without_rerun": exact_equality,
        "formal_adopted": exact_equality,
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append-only TASK8B Phase F provisional adoption comparator"
    )
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--provisional-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_and_adopt_task8b_phase_f(
        args.canonical_root,
        args.provisional_root,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
