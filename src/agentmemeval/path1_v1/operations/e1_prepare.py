"""Fixed E1 submission preparation; it supplies no runtime/backend admission."""
from __future__ import annotations

import argparse
import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path

from agentmemeval.path1_v1.operations.contracts import canonical_json, content_digest
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.submission_process import (
    file_sha256,
    read_json,
    write_exclusive_json,
)

RESOURCE_PROFILE = {
    "version": 1, "phase": "E1", "walltime_seconds": 14400, "startup_seconds": 1200,
    "cleanup_seconds": 600, "gpus": 1, "cpus_per_task": 8,
    "memory_mib": None, "memory_policy": "site_default", "stage_cap": 1,
    "runtime_admission": False,
}


@dataclass(frozen=True)
class E1LaunchFiles:
    prepared_path: Path


@dataclass(frozen=True)
class ValidatedE1Prepared:
    descriptor: dict
    prepared_path: Path
    worker: object
    submission: SubmissionRequest
    runtime_plan: dict | None = None

    def __getattr__(self, name):
        return getattr(self.worker, name)


def render_e1_script(*, python_executable: Path, repository_root: Path,
                     prepared_path: Path, native_runtime: dict | None = None) -> bytes:
    values = [str(python_executable), str(repository_root / "src"), str(prepared_path)]
    if any(not Path(value).is_absolute() or any(c in value for c in "\r\n\x00")
           for value in values):
        raise ValueError("E1 launcher paths must be absolute single-line paths")
    if native_runtime is not None:
        from agentmemeval.path1_v1.operations.native_runtime import bootstrap_argv
        argv = bootstrap_argv(python_executable, repository_root, native_runtime,
            module="agentmemeval.path1_v1.operations.e1_launcher",
            arguments=["--prepared", str(prepared_path)])
        return ("#!/bin/sh\nset -eu\nexec " + shlex.join(argv) + "\n").encode("utf-8")
    return ("#!/bin/sh\nset -eu\nexport PYTHONPATH=" + shlex.quote(values[1])
            + "\nexec " + shlex.quote(values[0])
            + " -B -s -m agentmemeval.path1_v1.operations.e1_launcher --prepared "
            + shlex.quote(values[2]) + "\n").encode("utf-8")


def _regular(path: Path) -> Path:
    # Same strict path policy as the executor, except the interpreter entry is
    # checked by the dedicated venv-aware verifier, never through this helper.
    from agentmemeval.path1_v1.operations.intent_executor import _file
    return _file(path)


def _same(left, right):
    return canonical_json(left) == canonical_json(right)


def read_prepared(path: Path, *, model_verification=None) -> ValidatedE1Prepared:
    from agentmemeval.path1_v1.operations.worker import validate_worker_request

    path = _regular(path)
    value = read_json(path)
    keys = {"schema_version", "kind", "worker_request_path", "worker_file_sha256",
            "worker_request_sha256", "python_identity", "resource_profile", "submission",
            "submission_request_sha256", "script_file", "descriptor_sha256"}
    version = value.get("schema_version") if type(value) is dict else None
    if type(version) is int and version == 2:
        keys |= {"runtime_plan_path", "runtime_plan_sha256"}
    if (type(value) is not dict or set(value) != keys
            or type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2)
            or value["kind"] != "path1_e1_prepared"
            or not _same(value["resource_profile"], RESOURCE_PROFILE)
            or value["descriptor_sha256"] != content_digest(
                {k: v for k, v in value.items() if k != "descriptor_sha256"})):
        raise ValueError("invalid E1 prepared descriptor/profile")
    worker_path = _regular(Path(value["worker_request_path"]))
    if file_sha256(worker_path) != value["worker_file_sha256"]:
        raise ValueError("prepared worker bytes changed")
    worker = validate_worker_request(worker_path, model_verification=model_verification)
    if worker.contract["phase"] != "E1" or worker.request_sha256 != value["worker_request_sha256"]:
        raise ValueError("prepared entry is E1 only")
    runtime_plan = None
    if version == 2:
        from agentmemeval.path1_v1.operations.e1_campaign import read_runtime_plan

        plan_path = _regular(Path(value["runtime_plan_path"]))
        if file_sha256(plan_path) != value["runtime_plan_sha256"]:
            raise ValueError("prepared runtime plan bytes changed")
        runtime_plan = read_runtime_plan(plan_path, contract=worker.contract,
                                         request=worker.request)
    proof = value["python_identity"]
    current = verify_python_interpreter(Path(proof["entry_path"]), proof["executable_sha256"],
                                        expected_prefix=proof["prefix"])
    if not _same(current, proof):
        raise ValueError("prepared interpreter/venv changed")
    raw = value["submission"]
    if set(raw) != {"campaign_id", "intent_id", "spec"}:
        raise ValueError("invalid E1 submission envelope")
    request = SubmissionRequest(raw["campaign_id"], raw["intent_id"],
                                SubmitSpec.from_payload(raw["spec"]))
    expected_resources = {k: RESOURCE_PROFILE[k] for k in (
        "walltime_seconds", "gpus", "cpus_per_task", "memory_mib", "memory_policy")}
    if (request.intent_id != worker.intent_id
            or request.campaign_id != worker.contract["campaign_id"]
            or request.request_sha256 != value["submission_request_sha256"]
            or not _same({k: getattr(request.spec, k) for k in expected_resources},
                         expected_resources)
            or request.spec.script_args):
        raise ValueError("prepared E1 submission binding/resources differ")
    script = _regular(Path(value["script_file"]))
    expected = render_e1_script(python_executable=Path(proof["entry_path"]),
        repository_root=Path(worker.request["repository_root"]), prepared_path=path,
        native_runtime=(runtime_plan or {}).get("native_runtime"))
    if script.read_bytes() != expected or file_sha256(script) != request.spec.script_sha256:
        raise ValueError("prepared E1 launcher bytes changed")
    return ValidatedE1Prepared(value, path, worker, request, runtime_plan)


def prepare(*, worker_request_path: Path, prepared_directory: Path,
            python_executable: Path, python_sha256: str, python_prefix: str,
            cluster: str, account: str, user: str, partition: str,
            work_dir: str, script_path: str, runtime_plan_path: Path | None = None) -> Path:
    """Consume an already reserved E1 request; create no claim or scientific seed."""
    from agentmemeval.path1_v1.operations.worker import validate_worker_request

    worker_path = _regular(worker_request_path)
    worker = validate_worker_request(worker_path)
    if worker.contract["phase"] != "E1":
        raise ValueError("prepare requires a real reserved E1 request")
    runtime_plan = None
    if runtime_plan_path is not None:
        from agentmemeval.path1_v1.operations.e1_campaign import read_runtime_plan

        runtime_plan_path = _regular(runtime_plan_path)
        runtime_plan = read_runtime_plan(runtime_plan_path, contract=worker.contract,
                                         request=worker.request)
    proof = verify_python_interpreter(python_executable, python_sha256,
                                      expected_prefix=python_prefix)
    directory = prepared_directory
    if not directory.is_absolute() or directory.exists() or not directory.parent.is_dir():
        raise ValueError("prepared directory must be new with an existing absolute parent")
    # Parent and final paths are checked before any mutation.
    if directory.parent.resolve() != directory.parent:
        raise ValueError("prepared parent must be canonical")
    path = directory / "prepared.json"
    script_file = directory / "e1.sh"
    script = render_e1_script(python_executable=Path(proof["entry_path"]),
        repository_root=Path(worker.request["repository_root"]), prepared_path=path,
        native_runtime=(runtime_plan or {}).get("native_runtime"))
    spec = SubmitSpec(cluster, account, user, partition, work_dir, script_path,
        hashlib.sha256(script).hexdigest(), walltime_seconds=14400, gpus=1,
        cpus_per_task=8, memory_mib=None, memory_policy="site_default")
    request = SubmissionRequest(worker.contract["campaign_id"], worker.intent_id, spec)
    value = {"schema_version": 1, "kind": "path1_e1_prepared",
        "worker_request_path": str(worker_path), "worker_file_sha256": file_sha256(worker_path),
        "worker_request_sha256": worker.request_sha256, "python_identity": proof,
        "resource_profile": RESOURCE_PROFILE,
        "submission": {"campaign_id": request.campaign_id, "intent_id": request.intent_id,
                       "spec": spec.payload()},
        "submission_request_sha256": request.request_sha256, "script_file": str(script_file)}
    if runtime_plan_path is not None:
        value.update(schema_version=2, runtime_plan_path=str(runtime_plan_path),
                     runtime_plan_sha256=file_sha256(runtime_plan_path))
    value["descriptor_sha256"] = content_digest(value)
    directory.mkdir(exist_ok=False)
    with script_file.open("xb") as stream:
        stream.write(script)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    write_exclusive_json(path, value)
    read_prepared(path, model_verification=worker.model_verification)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON containing exactly the explicit prepare keyword arguments")
    args = parser.parse_args(argv)
    values = read_json(_regular(args.input))
    for name in ("worker_request_path", "prepared_directory", "python_executable"):
        values[name] = Path(values[name])
    if values.get("runtime_plan_path") is not None:
        values["runtime_plan_path"] = Path(values["runtime_plan_path"])
    print(prepare(**values))


if __name__ == "__main__":
    main()
