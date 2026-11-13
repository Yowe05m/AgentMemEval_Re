"""Prepare/inspect one service E0; no model execution or scientific admission.

Explicit submit uses the existing durable executor. Preparation preserves the
exact small source inputs and never calls a scheduler or starts a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from agentmemeval.path1_v1.operations.contracts import (
    SOURCE_CONTRACT_SHA256,
    build_runtime_identity,
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.e0_resource_cli import _absolute, _require, verify_environment
from agentmemeval.path1_v1.operations.intent_executor import (
    DurableIntentExecutor,
    E0PreSubmitVerifier,
    ProbeLaunchFiles,
    VerifiedCommand,
    render_probe_script,
)
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.runtime_probe import _strict_json
from agentmemeval.path1_v1.operations.service_profile import (
    validate_service_profile,
    validate_service_submission,
)
from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec, identifier
from agentmemeval.path1_v1.operations.submission_process import file_sha256, write_exclusive_json
from agentmemeval.path1_v1.operations.user_capacity import PinnedUserCommand, UserAssociationReader


def _config_path(repository: Path, relative: str) -> Path:
    _require(
        type(relative) is str
        and relative.startswith("configs/path1_v1/")
        and not any(p in {"", ".", ".."} for p in relative.split("/"))
        and not any(c in relative for c in "\\:\x00\r\n"),
        "invalid Path1 config path",
    )
    return _absolute(repository / relative)


def _config(raw: bytes) -> dict:
    config = _strict_json(raw)
    validate_service_profile(config)
    keys = {
        "schema_version",
        "kind",
        "phase",
        "job_kind",
        "resource_profile_version",
        "walltime_seconds",
        "command_seconds",
        "cleanup_seconds",
        "startup_seconds",
        "gpus",
        "stage_cap",
        "cpus_per_task",
        "memory_mib",
        "queue_policy",
        "qos_policy",
        "service_port",
        "cluster",
        "account",
        "user",
        "partition",
        "control_root",
        "accounting_since",
        "dependency_lock",
        "python_executable",
        "python_sha256",
        "python_prefix",
        "service_python",
        "service_python_sha256",
        "service_python_prefix",
        "model_root",
        "model_manifest_path",
        "model_manifest_sha256",
        "commands",
    }
    _require(set(config) == keys, "service config keys differ")
    for key in ("cluster", "account", "user", "partition"):
        identifier(config[key])
    _require(
        set(config["commands"]) == {"sbatch", "scontrol", "sacctmgr", "squeue", "sacct"},
        "service command pins differ",
    )
    stamp = datetime.fromisoformat(config["accounting_since"])
    _require(stamp.tzinfo is not None, "accounting since must be timezone-aware")
    return config


def _write_raw(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(repository: Path, run_root: Path, campaign_id: str, config_relative: str) -> dict:
    repository = _absolute(repository)
    _absolute(run_root, exists=False)
    if run_root.exists():
        raise FileExistsError("service preparation root already exists")
    identifier(campaign_id)
    config_path = _config_path(repository, config_relative)
    raw = config_path.read_bytes()
    config = _config(raw)
    lock_path = _config_path(repository, config["dependency_lock"])
    lock_raw = lock_path.read_bytes()
    verify_environment(lock_path)
    python = Path(config["python_executable"])
    python_identity = verify_python_interpreter(
        python, config["python_sha256"], expected_prefix=config["python_prefix"]
    )
    service_identity = verify_python_interpreter(
        Path(config["service_python"]),
        config["service_python_sha256"],
        expected_prefix=config["service_python_prefix"],
    )
    model = _absolute(Path(config["model_root"]))
    manifest = _absolute(Path(config["model_manifest_path"]))
    manifest_raw = manifest.read_bytes()
    _require(
        hashlib.sha256(manifest_raw).hexdigest() == config["model_manifest_sha256"],
        "model manifest differs from frozen service config",
    )
    tracked = (
        subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z"], capture_output=True, check=True
        )
        .stdout.decode()
        .split("\0")
    )
    dependencies = [
        p
        for p in tracked
        if p.startswith(("src/", "prompts/", "tools/")) or p in {"pyproject.toml", ".gitattributes"}
    ]
    operations = [p for p in dependencies if p.startswith("src/agentmemeval/path1_v1/operations/")]
    identity = build_runtime_identity(
        repository,
        core_paths=sorted(set(dependencies) - set(operations)),
        operations_paths=sorted(operations),
        config_path=config_relative,
        dependency_lock_path=config["dependency_lock"],
        model_manifest_path=manifest,
        model_root=model,
    )
    _require(
        not identity["dirty"]
        and identity["content_matches_head"]
        and not identity["uncovered_execution_paths"]
        and not identity["untracked_execution_paths"]
        and not identity["untracked_declared_paths"],
        "service source is not clean/complete",
    )
    originals = {
        "site_config.json": (config_path, raw),
        "dependency_lock.json": (lock_path, lock_raw),
        "model_manifest.json": (manifest, manifest_raw),
    }
    expected = {
        "site_config.json": identity["config"]["sha256"],
        "dependency_lock.json": identity["dependency_lock"]["sha256"],
        "model_manifest.json": identity["model"]["manifest_sha256"],
    }
    for name, (path, data) in originals.items():
        _require(
            path.read_bytes() == data and hashlib.sha256(data).hexdigest() == expected[name],
            "service original input changed during preparation",
        )
    source = _strict_json(
        (repository / ("src/agentmemeval/path1_v1/assets/path1_source_contract.json")).read_bytes()
    )
    contract = {
        "schema_version": 1,
        "kind": "path1_campaign_contract",
        "campaign_id": campaign_id,
        "phase": "E0",
        "source_contract": source,
        "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "runtime_identity": identity,
        "seed_namespace": campaign_id + "_technical",
        "cohort": {
            "formation_count": 0,
            "episodes_per_artifact_per_arm": 0,
            "seed_inventory_sha256": None,
            "freeze_receipt_sha256": None,
        },
        "job": {
            "kind": "service_probe",
            "artifact_id": None,
            "pair_ids": [],
            "arms": [],
            "domain": "source",
            "attempt": 1,
            "output_relpath": campaign_id + "/E0/service/attempt_01",
            "formation_receipt": None,
            "retry_evidence": None,
        },
        "not_for_paper": True,
        "execution_admission": "not_granted_by_contract_validation",
    }
    validate_campaign_contract(contract)
    run_root.mkdir(exist_ok=False)
    for name in ("output", "preflight", "inputs"):
        (run_root / name).mkdir()
    inputs = []
    for name, (path, data) in originals.items():
        _write_raw(run_root / "inputs" / name, data)
        inputs.append(
            {
                "source_path": str(path),
                "path": "inputs/" + name,
                "size": len(data),
                "sha256": expected[name],
            }
        )
    contract_path = run_root / "contract.json"
    write_exclusive_json(contract_path, contract)
    launch = {
        key: config[key]
        for key in (
            "python_executable",
            "python_sha256",
            "service_python",
            "service_python_sha256",
            "service_port",
            "command_seconds",
            "cleanup_seconds",
            "startup_seconds",
        )
    }
    launch.update(python_identity=python_identity, service_python_identity=service_identity)
    probe = {
        "schema_version": 2,
        "kind": "path1_e0_probe_request",
        "intent_id": canonical_intent_id(contract),
        "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "repository_root": str(repository),
        "model_root": str(model),
        "model_manifest_path": str(manifest),
        "output_root": str(run_root / "output"),
        "launch": launch,
    }
    probe["request_sha256"] = content_digest(probe)
    request_path = run_root / "probe_request.json"
    write_exclusive_json(request_path, probe)
    script = run_root / "probe.sh"
    _write_raw(
        script,
        render_probe_script(
            python_executable=python, repository_root=repository, probe_request_path=request_path
        ),
    )
    spec = SubmitSpec(
        config["cluster"],
        config["account"],
        config["user"],
        config["partition"],
        str(run_root / "output" / contract["job"]["output_relpath"]),
        str(script),
        file_sha256(script),
        walltime_seconds=7200,
        cpus_per_task=config["cpus_per_task"],
        memory_mib=config["memory_mib"],
    )
    submission = SubmissionRequest(campaign_id, probe["intent_id"], spec)
    profile = validate_service_submission(
        submission, SimpleNamespace(request=probe, contract=contract), config
    )
    descriptor = {
        "schema_version": 1,
        "kind": "path1_prepared_service_submission",
        "created_at": datetime.now(UTC).isoformat(),
        "repository_root": str(repository),
        "config_relative": config_relative,
        "config_sha256": identity["config"]["sha256"],
        "campaign_id": campaign_id,
        "intent_id": probe["intent_id"],
        "spec": spec.payload(),
        "submission_request_sha256": submission.request_sha256,
        "probe_request_sha256": file_sha256(request_path),
        "inputs": inputs,
        "service_profile": profile,
        "implementation_status": "implemented_only",
        "runtime_admission": False,
        "scientific_admission": False,
    }
    path = run_root / "submission.json"
    write_exclusive_json(path, descriptor)
    return {"descriptor_path": str(path), "descriptor_sha256": file_sha256(path), **descriptor}


def read_prepared(run_root: Path, expected_descriptor_sha256: str):
    """Inspect historical bytes without requiring the old environment to be live."""
    _absolute(run_root)
    path = _absolute(run_root / "submission.json")
    _require(file_sha256(path) == expected_descriptor_sha256, "service descriptor changed")
    descriptor = _strict_json(path.read_bytes())
    keys = {
        "schema_version",
        "kind",
        "created_at",
        "repository_root",
        "config_relative",
        "config_sha256",
        "campaign_id",
        "intent_id",
        "spec",
        "submission_request_sha256",
        "probe_request_sha256",
        "inputs",
        "service_profile",
        "implementation_status",
        "runtime_admission",
        "scientific_admission",
    }
    _require(
        set(descriptor) == keys
        and type(descriptor["schema_version"]) is int
        and descriptor["schema_version"] == 1
        and descriptor["kind"] == "path1_prepared_service_submission"
        and descriptor["implementation_status"] == "implemented_only"
        and descriptor["runtime_admission"] is False
        and descriptor["scientific_admission"] is False,
        "invalid service descriptor",
    )
    repository = _absolute(Path(descriptor["repository_root"]))
    inputs = descriptor["inputs"]
    names = ["inputs/site_config.json", "inputs/dependency_lock.json", "inputs/model_manifest.json"]
    _require(
        type(inputs) is list and [row["path"] for row in inputs] == names,
        "service frozen input inventory differs",
    )
    for row in inputs:
        _require(
            set(row) == {"source_path", "path", "size", "sha256"} and type(row["size"]) is int,
            "service frozen input record differs",
        )
        item = _absolute(run_root / row["path"])
        _require(
            item.stat().st_size == row["size"] and file_sha256(item) == row["sha256"],
            "service frozen input bytes changed",
        )
    config = _config((run_root / names[0]).read_bytes())
    request_path = _absolute(run_root / "probe_request.json")
    _require(
        file_sha256(request_path) == descriptor["probe_request_sha256"], "service request changed"
    )
    probe = _strict_json(request_path.read_bytes())
    probe_keys = {
        "schema_version",
        "kind",
        "request_sha256",
        "intent_id",
        "contract_path",
        "contract_sha256",
        "repository_root",
        "model_manifest_path",
        "model_root",
        "output_root",
        "launch",
    }
    _require(
        set(probe) == probe_keys
        and type(probe.get("schema_version")) is int
        and probe["schema_version"] == 2
        and probe["kind"] == "path1_e0_probe_request"
        and content_digest({k: v for k, v in probe.items() if k != "request_sha256"})
        == probe["request_sha256"],
        "service request digest differs",
    )
    contract_path = _absolute(run_root / "contract.json")
    _require(
        probe["contract_path"] == str(contract_path)
        and file_sha256(contract_path) == probe["contract_sha256"],
        "service contract changed",
    )
    contract = validate_campaign_contract(_strict_json(contract_path.read_bytes()))
    identity = contract["runtime_identity"]
    _require(
        contract["phase"] == "E0"
        and contract["job"]["kind"] == "service_probe"
        and contract["job"]["attempt"] == 1
        and canonical_intent_id(contract) == probe["intent_id"] == descriptor["intent_id"]
        and contract["campaign_id"] == descriptor["campaign_id"]
        and probe["repository_root"] == str(repository)
        and probe["output_root"] == str(run_root / "output"),
        "service prepared identity differs",
    )
    expected_sources = [
        str(repository / descriptor["config_relative"]),
        str(repository / config["dependency_lock"]),
        config["model_manifest_path"],
    ]
    expected_hashes = [
        identity["config"]["sha256"],
        identity["dependency_lock"]["sha256"],
        identity["model"]["manifest_sha256"],
    ]
    _require(
        [r["source_path"] for r in inputs] == expected_sources
        and [r["sha256"] for r in inputs] == expected_hashes
        and identity["config"]["path"] == descriptor["config_relative"]
        and identity["dependency_lock"]["path"] == config["dependency_lock"]
        and descriptor["config_sha256"] == expected_hashes[0],
        "service original input identity differs",
    )
    request = SubmissionRequest(
        descriptor["campaign_id"],
        descriptor["intent_id"],
        SubmitSpec.from_payload(descriptor["spec"]),
    )
    script = _absolute(run_root / "probe.sh")
    _require(
        request.request_sha256 == descriptor["submission_request_sha256"]
        and request.spec.script_path == str(script)
        and request.spec.script_sha256 == file_sha256(script)
        and request.spec.work_dir == str(run_root / "output" / contract["job"]["output_relpath"])
        and script.read_bytes()
        == render_probe_script(
            python_executable=Path(probe["launch"]["python_executable"]),
            repository_root=repository,
            probe_request_path=request_path,
        ),
        "service script/spec differs",
    )
    profile = validate_service_submission(
        request, SimpleNamespace(request=probe, contract=contract), config
    )
    _require(
        content_digest(profile) == content_digest(descriptor["service_profile"]),
        "service planned effective parameters differ",
    )
    return descriptor, config, request


def execute(run_root: Path, expected_descriptor_sha256: str, *, lookup: bool) -> dict:
    descriptor, config, request = read_prepared(run_root, expected_descriptor_sha256)
    repository = Path(descriptor["repository_root"])
    if not lookup:
        _require(
            file_sha256(_config_path(repository, descriptor["config_relative"]))
            == descriptor["config_sha256"],
            "current service config changed",
        )
        verify_environment(_config_path(repository, config["dependency_lock"]))
    commands = config["commands"]

    def command(name):
        row = commands[name]
        return VerifiedCommand((row["path"],), ((row["path"], row["sha256"]),))

    reader = UserAssociationReader(
        cluster=config["cluster"],
        account=config["account"],
        user=config["user"],
        commands={
            name: PinnedUserCommand(**commands[name]) for name in ("sacctmgr", "squeue", "sacct")
        },
        since=datetime.fromisoformat(config["accounting_since"]),
    )
    executor = DurableIntentExecutor(
        _absolute(Path(config["control_root"])),
        cluster=config["cluster"],
        account=config["account"],
        command=command("sbatch"),
        launch_files=lambda _: ProbeLaunchFiles(run_root / "probe_request.json"),
        pre_submit_verifier=E0PreSubmitVerifier(
            run_root / "preflight", partition_command=command("scontrol")
        ),
        account_reader=reader,
    )
    receipt = (
        executor.lookup(request, timeout_seconds=60)
        if lookup
        else executor.submit_once(request, timeout_seconds=60)
    )
    return {
        "descriptor_sha256": expected_descriptor_sha256,
        "lookup_only": lookup,
        "receipt": asdict(receipt) if receipt is not None else None,
        "runtime_admission": False,
        "scientific_admission": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repository", type=Path, required=True)
    prep.add_argument("--run-root", type=Path, required=True)
    prep.add_argument("--campaign-id", required=True)
    prep.add_argument("--config", required=True)
    for action in ("inspect", "submit", "lookup"):
        item = sub.add_parser(action)
        item.add_argument("--run-root", type=Path, required=True)
        item.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.repository, args.run_root, args.campaign_id, args.config)
    elif args.action == "inspect":
        result = read_prepared(args.run_root, args.descriptor_sha256)[0]
    else:
        result = execute(args.run_root, args.descriptor_sha256, lookup=args.action == "lookup")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
