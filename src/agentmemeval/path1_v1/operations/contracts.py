"""Strict, side-effect-free Path1 identity and topology validation.

These contracts do not grant experiment admission, attest a running model, or
replace the worker's launch-time checks. E2/E3 validation re-reads the declared
files; a digest alone is an identity, not evidence of a successful runtime test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_CONTRACT_SHA256 = "dc48cbd357c2b3d02b4825781d38b869b507ff8efbc054adf682b7ee2280e634"
MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ARMS = ["retained", "removed"]
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
             *(f"LPT{i}" for i in range(10))}


class ContractError(ValueError):
    """An ambiguous, incomplete, changed, or unsupported Path1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _json_value(value: Any) -> None:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise ContractError("JSON strings must be valid Unicode") from exc
        return
    if type(value) is float:
        _require(math.isfinite(value), "non-finite JSON number")
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "JSON keys must be strings")
        for key, item in value.items():
            _json_value(key)
            _json_value(item)
        return
    raise ContractError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """UTF-8 JSON; object order is irrelevant, array order and number types are not."""
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _object(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    missing, unknown = fields - value.keys(), value.keys() - fields
    _require(not missing and not unknown,
             f"{name}: missing fields={sorted(missing)}, unknown fields={sorted(unknown)}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be integer >= {minimum}")
    return value


def _sha(value: Any, name: str, length: int = 64) -> str:
    _require(type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
             and value != "0" * length, f"{name} must be a complete lowercase SHA-{length * 4}")
    return value


def _slug(value: Any, name: str) -> str:
    _require(type(value) is str and _SLUG.fullmatch(value) is not None
             and value.upper() not in _RESERVED, f"{name} is not a safe identifier")
    return value


def _relative_path(value: Any) -> str:
    _require(type(value) is str and bool(value), "path must be nonempty relative POSIX text")
    parts = value.split("/")
    _require(not value.startswith("/") and all(parts), "absolute/empty path component")
    for part in parts:
        _require(part.casefold() not in {".", "..", ".git", ".env", ".venv"}
                 and not part.casefold().startswith(".env.") and not part.endswith((" ", "."))
                 and not any(ord(c) < 32 or c in '\\:*?"<>|' for c in part)
                 and part.split(".")[0].upper() not in _RESERVED,
                 "unsafe or platform-ambiguous relative path")
    return value


def _file_at(root: Path, relative: str) -> Path:
    _relative_path(relative)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        _require(not current.is_symlink(), "symlink execution/model/artifact input is unsupported")
    resolved = candidate.resolve(strict=True)
    _require(resolved.is_relative_to(root) and resolved.is_file(), "input escaped root/not a file")
    return resolved


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = _file_at(root, relative)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": relative, "size": size, "sha256": digest.hexdigest()}


def _records(records: Any, name: str, *, nonempty: bool = True) -> None:
    _require(type(records) is list and (bool(records) or not nonempty), f"{name} must be a list")
    names: list[str] = []
    for record in records:
        _object(record, {"path", "size", "sha256"}, name)
        names.append(_relative_path(record["path"]))
        _integer(record["size"], f"{name}.size")
        _sha(record["sha256"], f"{name}.sha256")
    _require(len({name.casefold() for name in names}) == len(names), f"{name}: duplicate paths")
    _require(names == sorted(names), f"{name}: file records must be sorted by path")


def _sealed(value: Mapping[str, Any], digest_key: str) -> None:
    _sha(value[digest_key], digest_key)
    _require(value[digest_key] == content_digest({k: v for k, v in value.items()
                                                if k != digest_key}), f"{digest_key} mismatch")


def _git(root: Path, *args: str) -> bytes:
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    _require(process.returncode == 0, "Git inspection failed (no identity was admitted)")
    return process.stdout


def _read_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
        _json_value(result)
        return result
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid UTF-8 JSON input") from exc


def _model_identity(manifest_path: Path | None, model_root: Path | None) -> Any:
    _require(manifest_path is not None or model_root is None, "model_root needs a model manifest")
    if manifest_path is None:
        return None
    manifest = _read_json(manifest_path)
    _object(manifest, {"schema_version", "model", "revision", "tokenizer_revision",
                       "chat_template_sha256", "files"}, "model_manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
             "unsupported model manifest schema")
    _require(manifest["model"] == MODEL and manifest["revision"] == MODEL_REVISION
             and manifest["tokenizer_revision"] == MODEL_REVISION, "model identity differs")
    _sha(manifest["chat_template_sha256"], "chat_template_sha256")
    _records(manifest["files"], "model_manifest.files")
    if model_root is not None:
        root = model_root.resolve(strict=True)
        actual_paths = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                              if p.is_file() or p.is_symlink())
        _require(actual_paths == [x["path"] for x in manifest["files"]],
                 "model file inventory differs (extra or missing files)")
        _require([_file_record(root, p) for p in actual_paths] == manifest["files"],
                 "model file content differs")
    return {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "manifest": manifest, "files_verified": model_root is not None}


def build_runtime_identity(
    repository_root: Path | str,
    *,
    core_paths: Sequence[str],
    operations_paths: Sequence[str],
    config_path: str,
    dependency_lock_path: str,
    model_manifest_path: Path | str | None = None,
    model_root: Path | str | None = None,
    model_verification: Any = None,
    model_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Read actual source/config/model bytes without importing or running them.

    All tracked src/, prompts/, tools/ and pyproject.toml files must be covered
    for E2/E3. Caller-supplied file lists are never accepted as proof of closure.
    Model verification covers the exact supplied directory, not a running GPU.
    No Git configuration, file, environment variable, or repository is changed.
    """
    root = Path(repository_root).resolve(strict=True)
    _require(not isinstance(core_paths, (str, bytes))
             and not isinstance(operations_paths, (str, bytes)), "file lists must be sequences")
    core, ops = sorted(core_paths), sorted(operations_paths)
    for path in core + ops:
        _relative_path(path)
    _require(bool(core) and bool(ops), "both core and operations inventories are required")
    _require(len({p.casefold() for p in core + ops}) == len(core + ops),
             "duplicate/overlapping core and operations paths")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    _sha(commit, "Git HEAD", 40)
    status_before = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked = set(_git(root, "ls-files", "-z").decode("utf-8").strip("\0").split("\0"))
    dependencies = {p for p in tracked if p.startswith(("src/", "prompts/", "tools/"))
                    or p in {"pyproject.toml", ".gitattributes"}}
    disk_dependencies = {
        p.relative_to(root).as_posix()
        for directory in ("src", "prompts", "tools")
        if (root / directory).exists()
        for p in (root / directory).rglob("*")
        if p.is_file() or p.is_symlink()
    }
    declared = set(core + ops)
    core_files = [_file_record(root, p) for p in core]
    ops_files = [_file_record(root, p) for p in ops]
    config = _file_record(root, config_path)
    lock = _file_record(root, dependency_lock_path)
    head_blobs = {}
    for entry in _git(root, "ls-tree", "-rz", "HEAD").decode("utf-8").split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            _mode, object_type, blob = metadata.split()
            if object_type == "blob":
                head_blobs[path] = blob
    # Git status can deliberately omit assume-unchanged or skip-worktree files.
    # Compare filtered blob identities directly; hash-object does not write objects.
    mismatches = sorted(
        path for path in declared | {config_path, dependency_lock_path}
        if path not in head_blobs or _git(
            root, "hash-object", f"--path={path}", str(_file_at(root, path))
        ).decode().strip() != head_blobs[path]
    )
    manifest_path = Path(model_manifest_path) if model_manifest_path is not None else None
    if model_verification is None:
        _require(model_binding_sha256 is None, "model binding requires live verification")
        model = _model_identity(manifest_path, Path(model_root) if model_root is not None else None)
    else:
        from agentmemeval.path1_v1.operations.model_verification import reuse_model_files

        _require(manifest_path is not None and model_root is not None,
                 "live model verification requires actual model paths")
        model = reuse_model_files(model_verification, manifest_path, model_root,
                                  binding_sha256=model_binding_sha256)
    clean = not status_before
    identity: dict[str, Any] = {
        "schema_version": 1, "kind": "path1_runtime_identity",
        "core_sha": commit, "ops_sha": commit,
        "core_files": core_files, "operations_files": ops_files,
        "core_manifest_sha256": content_digest(core_files),
        "operations_manifest_sha256": content_digest(ops_files),
        "config": config, "dependency_lock": lock, "model": model,
        "dirty": not clean,
        "git_status_sha256": hashlib.sha256(status_before).hexdigest(),
        "untracked_declared_paths": sorted((declared | {config_path, dependency_lock_path})
                                           - tracked),
        "uncovered_execution_paths": sorted(dependencies - declared),
        "untracked_execution_paths": sorted(disk_dependencies - tracked),
        "content_matches_head": not mismatches,
        "head_content_mismatches": mismatches,
        "runtime_attestation": "not_performed_by_static_identity_builder",
    }
    _require(status_before == _git(root, "status", "--porcelain=v1", "-z",
                                   "--untracked-files=all")
             and commit == _git(root, "rev-parse", "HEAD").decode().strip(),
             "repository changed while collecting identity")
    for record in core_files + ops_files + [config, lock]:
        _require(record == _file_record(root, record["path"]),
                 "file changed while collecting identity")
    if manifest_path is not None:
        _require(hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                 == model["manifest_sha256"], "model manifest changed while collecting identity")
    if model_verification is not None:
        reuse_model_files(model_verification, manifest_path, model_root,
                          binding_sha256=model_binding_sha256, expected_identity=model)
    identity["identity_sha256"] = content_digest(identity)
    return identity


def _validate_identity(identity: Any, *, scientific: bool) -> None:
    _object(identity, {"schema_version", "kind", "core_sha", "ops_sha", "core_files",
                       "operations_files", "core_manifest_sha256", "operations_manifest_sha256",
                       "config", "dependency_lock", "model", "dirty", "git_status_sha256",
                       "untracked_declared_paths", "uncovered_execution_paths",
                       "untracked_execution_paths", "head_content_mismatches",
                       "content_matches_head", "runtime_attestation", "identity_sha256"},
            "runtime_identity")
    _require(type(identity["schema_version"]) is int and identity["schema_version"] == 1
             and identity["kind"] == "path1_runtime_identity", "unsupported runtime identity")
    for key in ("core_sha", "ops_sha"):
        _sha(identity[key], key, 40)
    _require(identity["core_sha"] == identity["ops_sha"], "single-repository identity mismatch")
    for name in ("core", "operations"):
        _records(identity[f"{name}_files"], f"{name}_files")
        _require(identity[f"{name}_manifest_sha256"] == content_digest(identity[f"{name}_files"]),
                 f"{name} content manifest mismatch")
    all_paths = [x["path"].casefold()
                 for x in identity["core_files"] + identity["operations_files"]]
    _require(len(all_paths) == len(set(all_paths)), "overlapping source inventories")
    for key in ("config", "dependency_lock"):
        _records([identity[key]], key)
    for key in ("untracked_declared_paths", "uncovered_execution_paths",
                "untracked_execution_paths", "head_content_mismatches"):
        _require(type(identity[key]) is list, f"{key} must be a list")
        for path in identity[key]:
            _relative_path(path)
        _require(identity[key] == sorted(set(identity[key])), f"{key} must be unique/sorted")
    for key in ("dirty", "content_matches_head"):
        _require(type(identity[key]) is bool, f"{key} must be boolean")
    _require(identity["content_matches_head"] == (not identity["head_content_mismatches"]),
             "inconsistent HEAD content state")
    _sha(identity["git_status_sha256"], "git_status_sha256")
    _require(identity["runtime_attestation"] == "not_performed_by_static_identity_builder",
             "static identity must not claim runtime attestation")
    model = identity["model"]
    if model is not None:
        _object(model, {"manifest_sha256", "manifest", "files_verified"}, "model")
        _sha(model["manifest_sha256"], "model.manifest_sha256")
        _require(type(model["files_verified"]) is bool, "files_verified must be boolean")
        manifest = model["manifest"]
        _object(manifest, {"schema_version", "model", "revision", "tokenizer_revision",
                           "chat_template_sha256", "files"}, "model.manifest")
        _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
                 "unsupported model manifest")
        _require(manifest["model"] == MODEL and manifest["revision"] == MODEL_REVISION
                 and manifest["tokenizer_revision"] == MODEL_REVISION, "model identity differs")
        _sha(manifest["chat_template_sha256"], "chat_template_sha256")
        _records(manifest["files"], "model.files")
    _sealed(identity, "identity_sha256")
    if scientific:
        _require(not identity["dirty"] and identity["content_matches_head"],
                 "E2/E3 reject dirty or non-HEAD content")
        _require(not identity["untracked_declared_paths"]
                 and not identity["uncovered_execution_paths"]
                 and not identity["untracked_execution_paths"],
                 "E2/E3 require tracked complete execution dependencies")
        _require(model is not None and model["files_verified"],
                 "E2/E3 require verified model file inventory")


def validate_artifact_receipt(receipt: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Check an immutable formation receipt's schema, self-hash and consumer identity."""
    _object(receipt, {"schema_version", "kind", "campaign_id", "phase", "artifact_id",
                      "formation_intent_id", "seed_namespace", "protocol_sha256",
                      "core_manifest_sha256", "operations_manifest_sha256",
                      "model_manifest_sha256", "formation_config_sha256", "snapshot",
                      "completed_hands", "completed_episodes", "selected_after_hand",
                      "selection_policy", "status", "receipt_sha256"}, "formation_receipt")
    _require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
             and receipt["kind"] == "path1_immutable_formation_receipt",
             "unsupported formation receipt")
    for field in ("campaign_id", "phase", "seed_namespace", "protocol_sha256"):
        _require(receipt[field] == contract[field], f"formation receipt {field} mismatch")
    _require(receipt["artifact_id"] == contract["job"]["artifact_id"], "artifact mismatch")
    _sha(receipt["formation_intent_id"], "formation_intent_id")
    _sha(receipt["formation_config_sha256"], "formation_config_sha256")
    identity = contract["runtime_identity"]
    for key in ("core_manifest_sha256", "operations_manifest_sha256"):
        _require(receipt[key] == identity[key], f"formation receipt {key} mismatch")
    _require(identity["model"] is not None and receipt["model_manifest_sha256"]
             == identity["model"]["manifest_sha256"], "formation receipt model mismatch")
    _records([receipt["snapshot"]], "snapshot")
    for key, expected in (("completed_hands", 300), ("completed_episodes", 6),
                          ("selected_after_hand", 300)):
        _integer(receipt[key], key)
        _require(receipt[key] == expected, f"formation receipt {key} mismatch")
    _require(receipt["selection_policy"] == contract["source_contract"]["formation"]
             ["primary_snapshot"] and receipt["status"] == "complete",
             "receipt must reference completed fixed-terminal formation")
    _sealed(receipt, "receipt_sha256")
    return copy.deepcopy(dict(receipt))


def _validate_contract(contract: Any) -> dict[str, Any]:
    _json_value(contract)
    _object(contract, {"schema_version", "kind", "campaign_id", "phase", "source_contract",
                       "protocol_sha256", "runtime_identity", "seed_namespace", "cohort",
                       "job", "not_for_paper", "execution_admission"}, "campaign_contract")
    _require(type(contract["schema_version"]) is int and contract["schema_version"] == 1
             and contract["kind"] == "path1_campaign_contract", "unsupported campaign schema")
    _slug(contract["campaign_id"], "campaign_id")
    _slug(contract["seed_namespace"], "seed_namespace")
    phase = contract["phase"]
    _require(type(phase) is str and phase in {"E0", "E1", "E2", "E3"}, "unsupported phase")
    _require(content_digest(contract["source_contract"]) == SOURCE_CONTRACT_SHA256,
             "source_contract has missing/unknown fields or differs from frozen 20261113_v1")
    _require(contract["protocol_sha256"] == SOURCE_CONTRACT_SHA256, "protocol hash mismatch")
    _require(type(contract["not_for_paper"]) is bool
             and contract["not_for_paper"] == (phase != "E3"), "phase/not_for_paper mismatch")
    _require(contract["execution_admission"] == "not_granted_by_contract_validation",
             "contract validation cannot claim execution admission")
    _validate_identity(contract["runtime_identity"], scientific=phase in {"E2", "E3"})
    cohort = _object(contract["cohort"], {"formation_count", "episodes_per_artifact_per_arm",
                                         "seed_inventory_sha256", "freeze_receipt_sha256"},
                     "cohort")
    n = _integer(cohort["formation_count"], "formation_count")
    b = _integer(cohort["episodes_per_artifact_per_arm"], "episodes_per_artifact_per_arm")
    if phase == "E0":
        _require((n, b) == (0, 0) and cohort["seed_inventory_sha256"] is None,
                 "E0 is a non-research probe")
    else:
        _sha(cohort["seed_inventory_sha256"], "seed_inventory_sha256")
    if phase == "E1":
        _require((n, b) == (1, 2), "E1 is exactly one technical unit with two pairs")
    if phase == "E2":
        _require(n in {16, 24, 32} and b == 16, "pilot design differs from frozen plan")
    if phase == "E3":
        _require(n >= 40 and n % 8 == 0 and b in {8, 16, 32, 64, 128, 256},
                 "formal sample size/seat balance differs from plan")
        _sha(cohort["freeze_receipt_sha256"], "freeze_receipt_sha256")
    else:
        _require(cohort["freeze_receipt_sha256"] is None, "non-formal phase cannot reuse freeze")
    job = _object(contract["job"], {"kind", "artifact_id", "pair_ids", "arms", "domain",
                                    "attempt", "output_relpath", "formation_receipt",
                                    "retry_evidence"}, "job")
    _require(type(job["kind"]) is str, "job kind must be text")
    _require(job["domain"] == "source", "Path1 only supports the source domain")
    _integer(job["attempt"], "attempt", 1)
    _require(job["attempt"] <= 3, "attempt exceeds initial plus two external retries")
    if job["attempt"] == 1:
        _require(job["retry_evidence"] is None, "initial attempt cannot be a retry")
    else:
        retry = _object(job["retry_evidence"], {"eligibility_receipt_sha256", "prior_intent_id",
                                                "prior_terminal_receipt_sha256", "scope",
                                                "seed_changed"}, "retry_evidence")
        for key in ("eligibility_receipt_sha256", "prior_intent_id",
                    "prior_terminal_receipt_sha256"):
            _sha(retry[key], key)
        _require(retry["scope"] == "complete_unit" and retry["seed_changed"] is False,
                 "retry must retain seeds and rerun the complete unit")
    _relative_path(job["output_relpath"])
    _require(type(job["pair_ids"]) is list, "pair_ids must be a list")
    for pair in job["pair_ids"]:
        _slug(pair, "pair_id")
    _require(len({p.casefold() for p in job["pair_ids"]}) == len(job["pair_ids"]),
             "duplicate pair IDs")
    if phase == "E0":
        _require(job["kind"] in {"resource_probe", "service_probe"}
                 and job["artifact_id"] is None and job["pair_ids"] == []
                 and job["arms"] == [] and job["formation_receipt"] is None,
                 "invalid probe topology")
    else:
        _slug(job["artifact_id"], "artifact_id")
        if phase == "E1":
            _require(job["kind"] == "technical_closed_loop" and len(job["pair_ids"]) == 2
                     and job["arms"] == ARMS and job["formation_receipt"] is None,
                     "E1 must keep the full technical chain and both arms in one job")
        elif job["kind"] == "formation":
            _require(job["pair_ids"] == [] and job["arms"] == []
                     and job["formation_receipt"] is None, "invalid formation topology")
        elif job["kind"] == "evaluation":
            _require(0 < len(job["pair_ids"]) <= b and job["arms"] == ARMS,
                     "evaluation must execute whole retained/removed pairs in one job")
            validate_artifact_receipt(job["formation_receipt"], contract)
        else:
            raise ContractError("scientific jobs must be formation or evaluation")
    return copy.deepcopy(dict(contract))


def canonical_intent_id(contract: Mapping[str, Any]) -> str:
    """Stable logical job ID. This hashes a schema-valid plan; it never admits execution."""
    return content_digest(_validate_contract(contract))


def validate_campaign_contract(
    contract: Mapping[str, Any], *, repository_root: Path | str | None = None,
    model_manifest_path: Path | str | None = None, model_root: Path | str | None = None,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate design/topology and, for E2/E3, freshly verify all declared content.

    A successful return is contract/content validation only. The caller still
    needs independently verified runtime, protocol, Git-sync and recovery gates.
    Existing/legacy run manifests remain readable elsewhere but are not accepted.
    """
    validated = _validate_contract(contract)
    if validated["phase"] in {"E2", "E3"}:
        _require(repository_root is not None and model_manifest_path is not None
                 and model_root is not None,
                 "E2/E3 require fresh source/model content verification")
        old = validated["runtime_identity"]
        observed = build_runtime_identity(
            repository_root, core_paths=[x["path"] for x in old["core_files"]],
            operations_paths=[x["path"] for x in old["operations_files"]],
            config_path=old["config"]["path"], dependency_lock_path=old["dependency_lock"]["path"],
            model_manifest_path=model_manifest_path, model_root=model_root,
        )
        _require(observed == old, "runtime identity/content changed since planning")
        if validated["job"]["kind"] == "evaluation":
            _require(artifact_root is not None, "evaluation requires immutable artifact content")
            snapshot = validated["job"]["formation_receipt"]["snapshot"]
            _require(_file_record(Path(artifact_root), snapshot["path"]) == snapshot,
                     "immutable artifact snapshot content mismatch")
    return validated
