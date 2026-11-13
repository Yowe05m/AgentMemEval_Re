"""Prepare or submit one bounded E0 resource probe through the durable executor.

Preparation never invokes sbatch. The caller records the returned intent and
descriptor digest in its campaign journal before explicitly choosing submit.
Lookup is read-only and never creates a replacement intent after uncertainty.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
    _require(config["kind"] == "path1_e0_resource_site_config"
             and config["phase"] == "E0" and config["job_kind"] == "resource_probe"
             and config["walltime_seconds"] == 120 and config["command_seconds"] == 90
             and config["cleanup_seconds"] == 30 and config["gpus"] == 1
             and config["stage_cap"] == 1 and config["queue_start_window_seconds"] == 10
             and config["scheduler_deadline_offset_seconds"] == 130,
             "resource probe parameters differ from the frozen E0 protocol")
    return config


def prepare(repository: Path, run_root: Path, campaign_id: str, config_relative: str) -> dict:
    repository = _absolute(repository)
    _absolute(run_root, exists=False)
    identifier(campaign_id)
    config = _config(repository, config_relative)
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
    source = _strict_json((repository / (
        "src/agentmemeval/path1_v1/assets/path1_source_contract.json")).read_bytes())
    contract = {
        "schema_version": 1, "kind": "path1_campaign_contract", "campaign_id": campaign_id,
        "phase": "E0", "source_contract": source, "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "runtime_identity": identity, "seed_namespace": campaign_id + "_technical",
        "cohort": {"formation_count": 0, "episodes_per_artifact_per_arm": 0,
                   "seed_inventory_sha256": None, "freeze_receipt_sha256": None},
        "job": {"kind": "resource_probe", "artifact_id": None, "pair_ids": [], "arms": [],
                "domain": "source", "attempt": 1,
                "output_relpath": campaign_id + "/E0/resource/attempt_01",
                "formation_receipt": None, "retry_evidence": None},
        "not_for_paper": True, "execution_admission": "not_granted_by_contract_validation",
    }
    validate_campaign_contract(contract)
    run_root.mkdir(exist_ok=False)
    (run_root / "output").mkdir()
    (run_root / "preflight").mkdir()
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
    spec = SubmitSpec(config["cluster"], config["account"], config["user"], config["partition"],
                      str(run_root / "output" / contract["job"]["output_relpath"]), str(script),
                      file_sha256(script), walltime_seconds=120)
    submission = SubmissionRequest(campaign_id, request["intent_id"], spec)
    descriptor = {
        "schema_version": 1, "kind": "path1_prepared_resource_submission",
        "created_at": datetime.now(UTC).isoformat(), "repository_root": str(repository),
        "config_relative": config_relative, "config_sha256": identity["config"]["sha256"],
        "campaign_id": campaign_id, "intent_id": request["intent_id"], "spec": spec.payload(),
        "submission_request_sha256": submission.request_sha256,
        "probe_request_sha256": file_sha256(request_path), "runtime_admission": False,
    }
    descriptor_path = run_root / "submission.json"
    write_exclusive_json(descriptor_path, descriptor)
    return {"descriptor_path": str(descriptor_path),
            "descriptor_sha256": file_sha256(descriptor_path), **descriptor}


def execute(run_root: Path, expected_descriptor_sha256: str, *, lookup: bool) -> dict:
    _absolute(run_root)
    descriptor_path = run_root / "submission.json"
    _require(file_sha256(descriptor_path) == expected_descriptor_sha256, "descriptor changed")
    descriptor = _strict_json(descriptor_path.read_bytes())
    repository = _absolute(Path(descriptor["repository_root"]))
    config = _config(repository, descriptor["config_relative"])
    _require(file_sha256(repository / descriptor["config_relative"])
             == descriptor["config_sha256"], "site config changed")
    verify_environment(repository / config["dependency_lock"])
    request_path = run_root / "probe_request.json"
    _require(file_sha256(request_path) == descriptor["probe_request_sha256"], "request changed")
    request = SubmissionRequest(descriptor["campaign_id"], descriptor["intent_id"],
                                SubmitSpec.from_payload(descriptor["spec"]))
    _require(request.request_sha256 == descriptor["submission_request_sha256"],
             "submission descriptor differs")
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
    for action in ("submit", "lookup"):
        command = sub.add_parser(action)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.repository, args.run_root, args.campaign_id, args.config)
    else:
        result = execute(args.run_root, args.descriptor_sha256, lookup=args.action == "lookup")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
