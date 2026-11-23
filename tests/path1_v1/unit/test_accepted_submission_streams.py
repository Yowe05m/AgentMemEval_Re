"""Isolated accepted-reader boundary, not an actual archive or scientific gate.

Upstream request/claim/source validation is represented explicitly by stubs.
The submission JSON/raw files, their hashes, _accepted_inputs, and the shared
raw verifier are real. No scheduler, process-ownership authority, or terminal
seal is created; historical v1 fixtures are not relabelled as new v2 evidence.
"""

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import submission_process as helper
from agentmemeval.path1_v1.operations.contracts import content_digest


def _write(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def boundary(root: Path, *, version=2, missing=None):
    """Anonymous parser fixture below the full source/claim validation boundary."""
    run = root / "run"
    run.mkdir()
    job = run / "job"
    job.mkdir()
    repository = root / "repository"
    repository.mkdir()
    script = b"#!/bin/sh\n# never executed fixture\n"
    (run / "probe.sh").write_bytes(script)
    campaign, intent, request_sha = "stream_fixture", "d" * 64, "e" * 64
    scope = content_digest({"cluster": "cluster", "account": "account"})
    control = root / "control"
    directory = control / "ie1" / scope[:16]
    invocation = directory / "i" / intent[:16]
    invocation.mkdir(parents=True)
    (invocation / "submitted_script.sh").write_bytes(script)
    spec = {"fixture_boundary": True, "script_sha256": _hash(script), "work_dir": str(job)}
    request = SimpleNamespace(
        campaign_id=campaign, intent_id=intent, request_sha256=request_sha,
        spec=SimpleNamespace(cluster="cluster", account="account", work_dir=str(job),
                             script_sha256=_hash(script)),
    )
    source = [
        {"path": name, "sha256": _hash(name.encode())}
        for name in ("src/core.py", "src/ops.py", "configs/site.json", "configs/lock.json")
    ]
    identity = {"core_files": [source[0]], "operations_files": [source[1]],
                "config": source[2], "dependency_lock": source[3]}
    contract = {"runtime_identity": identity}
    descriptor = {"spec": spec, "repository_root": "/offline/fixture_repository"}
    config = {"control_root": str(control)}
    probe = {"request_sha256": "f" * 64}
    manifest = {
        "schema_version": version, "nonce": "a" * 32, "request_sha256": request_sha,
        "argv": ["/offline/sbatch", "--no-requeue"], "cwd": str(job),
        "environment": {"LC_ALL": "C"}, "stdin_path": str(invocation / "submitted_script.sh"),
        "stdin_sha256": _hash(script),
        "input_files": [
            {"path": str(PurePosixPath(descriptor["repository_root"]) / item["path"]),
             "sha256": item["sha256"]} for item in source
        ],
        "wait_seconds": 10, "command_timeout_seconds": 10, "deadline_monotonic": 1000.0,
        "expected_effective_user": None,
    }
    if version == 2:
        manifest["result_schema_version"] = 2
    _write(invocation / "invocation.json", manifest)
    manifest_sha = helper.file_sha256(invocation / "invocation.json")
    process = {"pid": 101, "start_marker": "fixture", "host": "fixture-host",
               "boot_id": "fixture-boot", "effective_uid": 1000}
    ready = {"nonce": manifest["nonce"], "request_sha256": request_sha,
             "manifest_sha256": manifest_sha, "submitter": process, "effective_user": None}
    for name in ("ready.json", "go.json"):
        _write(invocation / name, ready)
    raw = {"stdout": b"551;cluster\n", "stderr": b""}
    result = {
        **copy.deepcopy(ready), "command_started_ns": 100, "command_finished_ns": 200,
        "returncode": 0, "timed_out": False, "stdout": "551;cluster",
        "stdout_sha256": _hash(raw["stdout"]), "stderr_sha256": _hash(raw["stderr"]),
    }
    if version == 2:
        result.update(schema_version=2, command_spawned=True, capture_status="complete",
                      capture_error_type=None, raw_outputs={})
        for name, value in raw.items():
            if name != missing:
                (invocation / (name + ".raw")).write_bytes(value)
            result["raw_outputs"][name] = {
                "path": name + ".raw", "size": len(value), "sha256": _hash(value),
                "limit_bytes": helper.RAW_OUTPUT_LIMIT_BYTES, "complete": True,
                "limit_exceeded": False,
            }
    _write(invocation / "result.json", result)
    cleanup = {
        "binding": {"intent_id": intent, "campaign_id": campaign, "role": "submission_transport",
                    "service_instance_id": manifest["nonce"],
                    "source_sha256": content_digest(identity)},
        "identity": process, "errors": [], "remaining_non_zombie_pids": [],
        "group_cleanup_verified": True, "leader_reaped": True, "returncode": 0,
    }
    _write(invocation / "local_cleanup.json", cleanup)
    record = {
        "status": "accepted", "submitter_finished": True, "intent_id": intent,
        "campaign_id": campaign, "spec": spec, "request_sha256": request_sha,
        "worker_request_sha256": probe["request_sha256"], "invocation_dir": str(invocation),
        "ready": ready, "job_id": "551", "manifest_sha256": manifest_sha,
        "result_sha256": helper.file_sha256(invocation / "result.json"),
    }
    if version == 2:
        record["helper_result_schema_version"] = 2
    store = SimpleNamespace(campaign_id="executor_" + scope[:32], directory=directory)
    claim = SimpleNamespace(intent_id=intent, run_id=intent, run_directory=str(job),
                            metadata={"purpose": "slurm_job_container",
                                      "request_sha256": request_sha})
    return SimpleNamespace(root=root, invocation=invocation, manifest=manifest, result=result,
                           record=record, args=(store, descriptor, request, run, probe,
                                                contract, config, repository, record, claim))


def accept(data):
    return er._accepted_inputs(*data.args, require_empty_output=True)


def reseal_result(data):
    """Fault injection updates only this anonymous fixture's result hash."""
    _write(data.invocation / "result.json", data.result)
    data.record["result_sha256"] = helper.file_sha256(data.invocation / "result.json")


def test_new_v2_accepted_reader_checks_original_bytes_without_granting_admission(tmp_path):
    data = boundary(tmp_path)
    before = {p.name: p.read_bytes() for p in data.invocation.iterdir()}
    result = accept(data)
    assert result["invocation"] == data.invocation
    assert (data.invocation / "stdout.raw").read_bytes() == b"551;cluster\n"
    assert (data.invocation / "stderr.raw").read_bytes() == b""
    assert "runtime_admission" not in result and "scientific_admission" not in result
    assert before == {p.name: p.read_bytes() for p in data.invocation.iterdir()}


def test_complete_v1_boundary_remains_readable_without_raw_files_or_schema_marker(tmp_path):
    data = boundary(tmp_path, version=1)
    assert "schema_version" not in data.result
    assert "helper_result_schema_version" not in data.record
    assert not list(data.invocation.glob("*.raw"))
    assert accept(data)["invocation"] == data.invocation


@pytest.mark.parametrize("status", ["timeout", "error", "output_limit"])
def test_incomplete_capture_is_rejected_even_with_zero_code_and_numeric_stdout(tmp_path, status):
    data = boundary(tmp_path)
    data.result["capture_status"] = status
    data.result["timed_out"] = status == "timeout"
    data.result["capture_error_type"] = "OSError" if status == "error" else None
    data.result["raw_outputs"]["stderr"]["complete"] = False
    if status == "output_limit":
        raw = b"x" * (helper.RAW_OUTPUT_LIMIT_BYTES + 1)
        (data.invocation / "stderr.raw").write_bytes(raw)
        data.result["stderr_sha256"] = _hash(raw)
        data.result["raw_outputs"]["stderr"].update(
            size=len(raw), sha256=_hash(raw), limit_exceeded=True,
        )
    reseal_result(data)
    # The format/raw validator accepts this truthful incomplete diagnostic;
    # the accepted-reader gate must independently reject it.
    assert helper.validate_helper_result(data.invocation, data.result, manifest=data.manifest) == 2
    assert data.result["returncode"] == 0 and data.result["stdout"] == "551;cluster"
    with pytest.raises(er.E0ReconciliationError, match="capture is incomplete"):
        accept(data)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_missing_raw_output_cannot_be_replaced_by_its_json_hash(tmp_path, stream):
    data = boundary(tmp_path, missing=stream)
    with pytest.raises(er.E0ReconciliationError, match="raw submission evidence"):
        accept(data)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_tampered_raw_output_is_rejected_even_if_result_json_is_unchanged(tmp_path, stream):
    data = boundary(tmp_path)
    (data.invocation / (stream + ".raw")).write_bytes(b"tampered")
    with pytest.raises(er.E0ReconciliationError, match="raw submission evidence"):
        accept(data)


@pytest.mark.parametrize("defect", ["remove_result_version", "legacy_result",
                                   "wrong_record_marker", "missing_raw_inventory"])
def test_manifest_two_prevents_downgrade_even_if_record_marker_is_removed(tmp_path, defect):
    data = boundary(tmp_path)
    data.record.pop("helper_result_schema_version")
    if defect == "remove_result_version":
        data.result.pop("schema_version")
    elif defect == "legacy_result":
        for key in ("schema_version", "raw_outputs", "capture_status",
                    "capture_error_type", "command_spawned"):
            data.result.pop(key)
    elif defect == "wrong_record_marker":
        data.record["helper_result_schema_version"] = 1
    else:
        data.result.pop("raw_outputs")
    reseal_result(data)
    assert data.manifest["schema_version"] == data.manifest["result_schema_version"] == 2
    with pytest.raises(er.E0ReconciliationError, match="raw submission evidence"):
        accept(data)


def test_missing_record_marker_does_not_bypass_new_raw_verification(tmp_path):
    data = boundary(tmp_path)
    data.record.pop("helper_result_schema_version")
    assert accept(data)["invocation"] == data.invocation
    data.result["raw_outputs"]["stdout"]["size"] += 1
    reseal_result(data)
    with pytest.raises(er.E0ReconciliationError, match="raw submission evidence"):
        accept(data)


@pytest.mark.parametrize("filename", ["invocation.json", "result.json"])
def test_durable_json_hash_binding_precedes_new_raw_acceptance(tmp_path, filename):
    data = boundary(tmp_path)
    target = data.invocation / filename
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(er.E0ReconciliationError, match="durable invocation/result bytes changed"):
        accept(data)
