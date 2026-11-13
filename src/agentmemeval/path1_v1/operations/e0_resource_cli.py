"""Prepare or submit one bounded E0 resource probe through the durable executor.

Preparation never invokes sbatch. The caller records the returned intent and
descriptor digest in its campaign journal before explicitly choosing submit.
Lookup is read-only and never creates a replacement intent after uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from agentmemeval.path1_v1.operations.contracts import (
    SOURCE_CONTRACT_SHA256,
    build_runtime_identity,
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.intent_executor import (
    DurableIntentExecutor,
    E0PreSubmitVerifier,
    ProbeLaunchFiles,
    VerifiedCommand,
    render_probe_script,
    resource_queue_policy,
)
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.runtime_probe import _strict_json
from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec, identifier
from agentmemeval.path1_v1.operations.submission_process import file_sha256, write_exclusive_json
from agentmemeval.path1_v1.operations.user_capacity import PinnedUserCommand, UserAssociationReader


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute(path: Path, *, exists=True) -> Path:
    _require(path.is_absolute() and path.resolve(strict=exists) == path,
             "use canonical absolute paths")
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink path")
    return path


def installed_versions() -> list[dict]:
    values = [{"name": d.metadata["Name"], "version": d.version}
              for d in importlib.metadata.distributions()]
    _require(all(type(row["name"]) is str and type(row["version"]) is str for row in values),
             "dependency name/version missing")
    values.sort(key=lambda row: (row["name"].casefold(), row["version"]))
    _require(len({row["name"].casefold() for row in values}) == len(values),
             "ambiguous duplicate distribution names")
    return values


def verify_environment(lock_path: Path) -> None:
    lock = _strict_json(lock_path.read_bytes())
    _require(lock["kind"] == "path1_e0_observed_environment_versions"
             and lock["python_version"] == platform.python_version()
             and lock["distributions"] == installed_versions(),
             "current CPU package versions differ from the observed E0 lock")
    # This version inventory deliberately does not claim wheel/loaded-GPU identity.


def _config(repository: Path, relative: str) -> dict:
    path = _absolute(repository / relative)
    _require(path.is_relative_to(repository / "configs/path1_v1"), "config escaped Path1")
    config = _strict_json(path.read_bytes())
    resource_queue_policy(config)
    return config


def prepare(repository: Path, run_root: Path, campaign_id: str, config_relative: str,
            *, attempt: int = 1, retry_evidence: dict | None = None) -> dict:
    _require(type(attempt) is int and 1 <= attempt <= 3, "attempt must be an integer in 1..3")
    repository = _absolute(repository)
    _absolute(run_root, exists=False)
    identifier(campaign_id)
    config = _config(repository, config_relative)
    frozen_config = (repository / config_relative).read_bytes()
    _require(_strict_json(frozen_config) == config, "config changed while preparing")
    lock_relative = config["dependency_lock"]
    lock_path = _absolute(repository / lock_relative)
    _require(lock_path.is_relative_to(repository / "configs/path1_v1"), "lock escaped Path1")
    verify_environment(lock_path)
    python = Path(config["python_executable"])
    proof = verify_python_interpreter(python, config["python_sha256"],
                                     expected_prefix=config["python_prefix"])
    tracked = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"],
                             capture_output=True, check=True).stdout.decode().split("\0")
    dependencies = [p for p in tracked if p.startswith(("src/", "prompts/", "tools/"))
                    or p in {"pyproject.toml", ".gitattributes"}]
    operations = [p for p in dependencies if p.startswith(
        "src/agentmemeval/path1_v1/operations/")]
    identity = build_runtime_identity(
        repository, core_paths=sorted(set(dependencies) - set(operations)),
        operations_paths=sorted(operations), config_path=config_relative,
        dependency_lock_path=lock_relative)
    _require(not identity["dirty"] and identity["content_matches_head"]
             and not identity["uncovered_execution_paths"]
             and not identity["untracked_execution_paths"]
             and not identity["untracked_declared_paths"], "source is not clean and complete")
    _require(hashlib.sha256(frozen_config).hexdigest() == identity["config"]["sha256"],
             "config changed while building source identity")
    source = _strict_json((repository / (
        "src/agentmemeval/path1_v1/assets/path1_source_contract.json")).read_bytes())
    contract = {
        "schema_version": 1, "kind": "path1_campaign_contract", "campaign_id": campaign_id,
        "phase": "E0", "source_contract": source, "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "runtime_identity": identity, "seed_namespace": campaign_id + "_technical",
        "cohort": {"formation_count": 0, "episodes_per_artifact_per_arm": 0,
                   "seed_inventory_sha256": None, "freeze_receipt_sha256": None},
        "job": {"kind": "resource_probe", "artifact_id": None, "pair_ids": [], "arms": [],
                "domain": "source", "attempt": attempt,
                "output_relpath": campaign_id + f"/E0/resource/attempt_{attempt:02d}",
                "formation_receipt": None, "retry_evidence": retry_evidence},
        "not_for_paper": True, "execution_admission": "not_granted_by_contract_validation",
    }
    validate_campaign_contract(contract)
    run_root.mkdir(exist_ok=False)
    (run_root / "output").mkdir()
    (run_root / "preflight").mkdir()
    with (run_root / "site_config.json").open("xb") as stream:
        stream.write(frozen_config)
        stream.flush()
        os.fsync(stream.fileno())
    contract_path = run_root / "contract.json"
    write_exclusive_json(contract_path, contract)
    request = {
        "schema_version": 2, "kind": "path1_e0_probe_request",
        "intent_id": canonical_intent_id(contract), "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path), "repository_root": str(repository),
        "model_root": None, "model_manifest_path": None, "output_root": str(run_root / "output"),
        "launch": {"python_executable": str(python), "python_sha256": config["python_sha256"],
                   "python_identity": proof, "service_python_identity": None,
                   "service_python": None, "service_python_sha256": None, "service_port": None,
                   "command_seconds": 90, "cleanup_seconds": 30, "startup_seconds": 0},
    }
    request["request_sha256"] = content_digest(request)
    request_path = run_root / "probe_request.json"
    write_exclusive_json(request_path, request)
    script = run_root / "probe.sh"
    with script.open("xb") as stream:
        stream.write(render_probe_script(python_executable=python, repository_root=repository,
                                         probe_request_path=request_path))
        stream.flush()
        os.fsync(stream.fileno())
    spec = SubmitSpec(config["cluster"], config["account"], config["user"], config["partition"],
                      str(run_root / "output" / contract["job"]["output_relpath"]), str(script),
                      file_sha256(script), walltime_seconds=120)
    submission = SubmissionRequest(campaign_id, request["intent_id"], spec)
    descriptor = {
        "schema_version": 2, "kind": "path1_prepared_resource_submission",
        "created_at": datetime.now(UTC).isoformat(), "repository_root": str(repository),
        "config_relative": config_relative, "config_sha256": identity["config"]["sha256"],
        "campaign_id": campaign_id, "intent_id": request["intent_id"], "spec": spec.payload(),
        "submission_request_sha256": submission.request_sha256,
        "probe_request_sha256": file_sha256(request_path), "runtime_admission": False,
        "attempt": attempt,
        "queue_start_window_seconds": config["queue_start_window_seconds"],
        "scheduler_deadline_offset_seconds": config["scheduler_deadline_offset_seconds"],
    }
    descriptor_path = run_root / "submission.json"
    write_exclusive_json(descriptor_path, descriptor)
    return {"descriptor_path": str(descriptor_path),
            "descriptor_sha256": file_sha256(descriptor_path), **descriptor}


def read_prepared(run_root: Path, expected_descriptor_sha256: str) -> tuple[dict, dict,
                                                                         SubmissionRequest]:
    """Read bound historical inputs without re-admitting execution or rewriting them.

    v1 did not save site config bytes. Its contract names the immutable Git
    commit containing them; current checkout/environment changes do not rewrite
    that history. Missing old Git objects or byte differences fail closed.
    """
    _absolute(run_root)
    descriptor_path = _absolute(run_root / "submission.json")
    _require(file_sha256(descriptor_path) == expected_descriptor_sha256, "descriptor changed")
    descriptor = _strict_json(descriptor_path.read_bytes())
    version = descriptor.get("schema_version")
    keys = {"schema_version", "kind", "created_at", "repository_root", "config_relative",
            "config_sha256", "campaign_id", "intent_id", "spec", "submission_request_sha256",
            "probe_request_sha256", "runtime_admission"}
    _require(type(version) is int and version in (1, 2)
             and set(descriptor) == keys | ({"attempt", "queue_start_window_seconds",
                 "scheduler_deadline_offset_seconds"} if version == 2 else set())
             and descriptor["kind"] == "path1_prepared_resource_submission"
             and descriptor["runtime_admission"] is False, "invalid prepared descriptor")
    repository = _absolute(Path(descriptor["repository_root"]))
    relative = descriptor["config_relative"]
    _require(type(relative) is str and relative.startswith("configs/path1_v1/")
             and not any(part in {"", ".", ".."} for part in relative.split("/"))
             and not any(char in relative for char in "\\:\x00\r\n"), "invalid config identity")
    request_path = _absolute(run_root / "probe_request.json")
    _require(file_sha256(request_path) == descriptor["probe_request_sha256"], "request changed")
    probe = _strict_json(request_path.read_bytes())
    _require(probe.get("schema_version") == 2 and probe.get("kind") == "path1_e0_probe_request"
             and content_digest({k:v for k,v in probe.items() if k != "request_sha256"})
             == probe.get("request_sha256"), "invalid bound E0 request")
    contract_path = _absolute(run_root / "contract.json")
    _require(probe["contract_path"] == str(contract_path)
             and file_sha256(contract_path) == probe["contract_sha256"], "contract changed")
    contract = validate_campaign_contract(_strict_json(contract_path.read_bytes()))
    _require(contract["phase"] == "E0" and contract["job"]["kind"] == "resource_probe"
             and canonical_intent_id(contract) == descriptor["intent_id"] == probe["intent_id"]
             and contract["campaign_id"] == descriptor["campaign_id"]
             and probe["repository_root"] == str(repository)
             and probe["output_root"] == str(run_root / "output")
             and probe["model_root"] is None and probe["model_manifest_path"] is None,
             "prepared resource identity differs")
    identity = contract["runtime_identity"]
    _require(identity["config"]["path"] == relative
             and identity["config"]["sha256"] == descriptor["config_sha256"],
             "contract config identity differs")
    if version == 2:
        raw_config = _absolute(run_root / "site_config.json").read_bytes()
    else:
        raw_config = subprocess.run(
            ["git", "-C", str(repository), "show", identity["core_sha"] + ":" + relative],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True).stdout
    _require(hashlib.sha256(raw_config).hexdigest() == descriptor["config_sha256"],
             "historical site config bytes differ")
    config = _strict_json(raw_config)
    queue, deadline = resource_queue_policy(config, allow_legacy=True)
    if version == 2:
        _require(type(descriptor["attempt"]) is int
                 and descriptor["attempt"] == contract["job"]["attempt"]
                 and (descriptor["queue_start_window_seconds"],
                      descriptor["scheduler_deadline_offset_seconds"]) == (queue, deadline),
                 "descriptor attempt/queue policy differs")
    request = SubmissionRequest(descriptor["campaign_id"], descriptor["intent_id"],
                                SubmitSpec.from_payload(descriptor["spec"]))
    script = _absolute(run_root / "probe.sh")
    _require(request.request_sha256 == descriptor["submission_request_sha256"]
             and request.spec.script_path == str(script)
             and request.spec.script_sha256 == file_sha256(script)
             and request.spec.work_dir == str(
                 run_root / "output" / contract["job"]["output_relpath"])
             and (request.spec.cluster, request.spec.account, request.spec.user,
                  request.spec.partition) == tuple(config[key] for key in (
                      "cluster", "account", "user", "partition"))
             and request.spec.walltime_seconds == 120 and not request.spec.script_args,
             "submission descriptor differs")
    _require(script.read_bytes() == render_probe_script(
        python_executable=Path(probe["launch"]["python_executable"]),
        repository_root=repository, probe_request_path=request_path), "prepared script differs")
    return descriptor, config, request


def execute(run_root: Path, expected_descriptor_sha256: str, *, lookup: bool) -> dict:
    descriptor, config, request = read_prepared(run_root, expected_descriptor_sha256)
    repository = Path(descriptor["repository_root"])
    request_path = run_root / "probe_request.json"
    if not lookup:
        _require(descriptor["schema_version"] == 2, "legacy descriptor is read-only")
        resource_queue_policy(config)
        _require(_config(repository, descriptor["config_relative"]) == config
                 and file_sha256(repository / descriptor["config_relative"])
                 == descriptor["config_sha256"], "site config changed")
        verify_environment(repository / config["dependency_lock"])
    commands = config["commands"]

    def command(name):
        record = commands[name]
        return VerifiedCommand((record["path"],), ((record["path"], record["sha256"]),))

    reader = UserAssociationReader(
        cluster=config["cluster"], account=config["account"], user=config["user"],
        commands={name: PinnedUserCommand(**commands[name])
                  for name in ("sacctmgr", "squeue", "sacct")},
        since=datetime.fromisoformat(config["accounting_since"]))
    executor = DurableIntentExecutor(
        _absolute(Path(config["control_root"])),
        cluster=config["cluster"], account=config["account"],
        command=command("sbatch"), launch_files=lambda _: ProbeLaunchFiles(request_path),
        pre_submit_verifier=E0PreSubmitVerifier(run_root / "preflight",
                                              partition_command=command("scontrol")),
        account_reader=reader)
    result = (executor.lookup(request, timeout_seconds=60) if lookup
              else executor.submit_once(request, timeout_seconds=60))
    return {"descriptor_sha256": expected_descriptor_sha256, "lookup_only": lookup,
            "receipt": asdict(result) if result is not None else None,
            "runtime_admission": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repository", type=Path, required=True)
    prep.add_argument("--run-root", type=Path, required=True)
    prep.add_argument("--campaign-id", required=True)
    prep.add_argument("--config", required=True)
    prep.add_argument("--attempt", type=int, choices=(1, 2, 3), required=True)
    prep.add_argument("--retry-evidence", type=Path)
    for action in ("submit", "lookup"):
        command = sub.add_parser(action)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        retry = (_strict_json(_absolute(args.retry_evidence).read_bytes())
                 if args.retry_evidence is not None else None)
        result = prepare(args.repository, args.run_root, args.campaign_id, args.config,
                         attempt=args.attempt, retry_evidence=retry)
    else:
        result = execute(args.run_root, args.descriptor_sha256, lookup=args.action == "lookup")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
