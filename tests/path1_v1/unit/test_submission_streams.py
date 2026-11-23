"""Real local helper subprocesses; no scheduler, GPU, or inference service."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import submission_process as helper
from agentmemeval.path1_v1.operations.intent_executor import clean_environment
from agentmemeval.path1_v1.operations.process_ownership import OwnedProcess, ProcessBinding
from agentmemeval.path1_v1.operations.state import StoreIntegrityError
from path1_v1.unit.test_intent_executor import make_executor_fixture


def real_helper(root, mode="success", *, conflict=False):
    python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    sources = {
        "success": "os.write(1, b'551;cluster\\n')",
        "rejected": "os.write(2, 'site policy 拒绝\\n'.encode()); sys.exit(7)",
        "timeout": "os.write(1, b'prefix\\n'); os.write(2, b'before timeout\\n'); time.sleep(2)",
        "stdout_limit": f"os.write(1, b'x' * ({helper.RAW_OUTPUT_LIMIT_BYTES} + 65536))",
        "stderr_limit": f"os.write(2, b'x' * ({helper.RAW_OUTPUT_LIMIT_BYTES} + 65536))",
    }
    child = root / "child.py"
    child.write_text(
        "import os,sys,time\nfrom pathlib import Path\n"
        "Path('executed').open('xb').write(b'once')\n" + sources[mode] + "\n", encoding="utf-8"
    )
    stdin = root / "submitted_script.sh"
    stdin.write_bytes(b"fixture script bytes\n")
    manifest = {
        "schema_version": 2, "result_schema_version": 2, "nonce": "a" * 32,
        "request_sha256": "b" * 64, "argv": [str(python), "-I", str(child)],
        "cwd": str(root), "environment": clean_environment(), "stdin_path": str(stdin),
        "stdin_sha256": helper.file_sha256(stdin),
        "input_files": [{"path": str(python), "sha256": helper.file_sha256(python)},
                        {"path": str(child), "sha256": helper.file_sha256(child)}],
        "wait_seconds": 5, "command_timeout_seconds": 0.3 if mode == "timeout" else 5,
        "deadline_monotonic": time.monotonic() + 12, "expected_effective_user": None,
    }
    invocation = root / "invocation.json"
    helper.write_exclusive_json(invocation, manifest)
    if conflict:
        (root / "stdout.raw").write_bytes(b"unrelated original")
    argv = [str(python), "-I", str(Path(helper.__file__).resolve()),
            str(invocation), helper.file_sha256(invocation)]
    cleanup = None

    def release():
        deadline = time.monotonic() + 8
        while not (root / "ready.json").exists():
            if time.monotonic() > deadline:
                raise AssertionError("real helper never reached ready")
            time.sleep(0.01)
        ready = helper.read_json(root / "ready.json")
        helper.write_exclusive_json(root / "go.json", ready)
        return ready

    with (root / "helper.stderr").open("xb") as errors:
        if sys.platform.startswith("linux"):
            binding = ProcessBinding("fixture", "b" * 64, None, "c" * 64,
                                     "submission_transport", "a" * 32)
            with OwnedProcess.managed(
                argv, binding=binding, cwd=root, env=clean_environment(),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                term_grace_seconds=0.1,
            ) as owner:
                ready = release()
                cleanup = owner.finish(deadline_monotonic=time.monotonic() + 10,
                                       term_grace_seconds=0.1)
                code = cleanup.returncode
            assert cleanup.group_cleanup_verified and cleanup.leader_reaped
        else:
            process = subprocess.Popen(argv, cwd=root, env=clean_environment(),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=errors)
            try:
                ready = release()
                code = process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
    result = helper.read_json(root / "result.json") if (root / "result.json").exists() else None
    return SimpleNamespace(root=root, manifest=manifest, ready=ready, result=result,
                           code=code, cleanup=cleanup)


@pytest.mark.parametrize("mode,status,code", [
    ("success", "complete", 0), ("rejected", "complete", 0),
    ("timeout", "timeout", 72), ("stdout_limit", "output_limit", 73),
    ("stderr_limit", "output_limit", 73),
])
def test_real_helper_keeps_bounded_exact_raw_outputs(tmp_path, mode, status, code):
    data = real_helper(tmp_path, mode)
    assert data.code == code and data.result["capture_status"] == status
    assert data.result["schema_version"] == 2
    assert helper.validate_helper_result(tmp_path, data.result, manifest=data.manifest,
                                         expected_result_schema=2) == 2
    for name in ("stdout", "stderr"):
        raw = (tmp_path / (name + ".raw")).read_bytes()
        record = data.result["raw_outputs"][name]
        assert record["size"] == len(raw) <= helper.RAW_OUTPUT_LIMIT_BYTES + 1
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["limit_exceeded"] == (len(raw) > helper.RAW_OUTPUT_LIMIT_BYTES)
    if mode == "success":
        assert (tmp_path / "stdout.raw").read_bytes() == b"551;cluster\n"
        assert (tmp_path / "stderr.raw").read_bytes() == b""
        assert data.result["stdout"] == "551;cluster" and data.result["returncode"] == 0
    elif mode == "rejected":
        assert (tmp_path / "stderr.raw").read_bytes() == "site policy 拒绝\n".encode()
        assert data.result["returncode"] == 7 and data.result["stdout"] is None
    elif mode == "timeout":
        assert (tmp_path / "stdout.raw").read_bytes() == b"prefix\n"
        assert (tmp_path / "stderr.raw").read_bytes() == b"before timeout\n"
        assert data.result["timed_out"] is True
    else:
        stream = mode.split("_")[0]
        assert data.result["raw_outputs"][stream]["complete"] is False
        assert len((tmp_path / (stream + ".raw")).read_bytes()) == helper.RAW_OUTPUT_LIMIT_BYTES + 1
    assert (tmp_path / "executed").read_bytes() == b"once"


def test_real_helper_never_overwrites_existing_raw_file(tmp_path):
    data = real_helper(tmp_path, conflict=True)
    assert data.code != 0 and data.result is None
    assert (tmp_path / "stdout.raw").read_bytes() == b"unrelated original"
    assert not (tmp_path / "executed").exists()


def test_new_result_requires_raw_bytes_and_cannot_downgrade_when_record_marker_is_missing(tmp_path):
    data = real_helper(tmp_path)
    result = copy.deepcopy(data.result)
    del result["schema_version"]
    with pytest.raises(ValueError, match="downgrade"):
        helper.validate_helper_result(tmp_path, result, manifest=data.manifest)
    with pytest.raises(ValueError, match="schema differs"):
        helper.validate_helper_result(tmp_path, data.result, manifest=data.manifest,
                                      expected_result_schema=1)
    (tmp_path / "stdout.raw").write_bytes(b"552;cluster\n")
    with pytest.raises(ValueError, match="bytes differ"):
        helper.validate_helper_result(tmp_path, data.result, manifest=data.manifest)


def test_raw_validator_rejects_symlink_and_hardlink_aliases(tmp_path):
    data = real_helper(tmp_path)
    raw = tmp_path / "stdout.raw"
    os.link(raw, tmp_path / "alias.raw")
    with pytest.raises(ValueError, match="invalid raw"):
        helper.validate_helper_result(tmp_path, data.result, manifest=data.manifest)


@pytest.mark.parametrize("ancestor", [False, True])
def test_raw_validator_does_not_follow_symlink(tmp_path, ancestor):
    target = tmp_path / "target"
    if ancestor:
        target.mkdir()
        (target / "stdout.raw").write_bytes(b"raw")
    else:
        target.write_bytes(b"raw")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=ancestor)
    except OSError:
        pytest.skip("OS does not permit test symlinks")
    with pytest.raises(ValueError, match="invalid raw"):
        helper._raw_bytes(link / "stdout.raw" if ancestor else link)


def test_legacy_manifest_and_result_remain_readable_without_inventing_raw_files(tmp_path):
    # Anonymous historical schema, not a forged new producer result.
    manifest = {key: None for key in helper._INVOCATION_KEYS}
    manifest["schema_version"] = 1
    result = {"stdout": "551;cluster",
              "stdout_sha256": hashlib.sha256(b"551;cluster\n").hexdigest(),
              "stderr_sha256": hashlib.sha256(b"").hexdigest(), "returncode": 0, "timed_out": False}
    assert helper.validate_helper_result(tmp_path, result, manifest=manifest) == 1
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        helper.validate_helper_result(tmp_path, result, manifest=manifest, expected_result_schema=2)


def test_executor_new_result_binds_raw_files_and_reentry_does_not_submit_again(tmp_path):
    fixture = make_executor_fixture(tmp_path)
    receipt = fixture.executor.submit_once(fixture.request, timeout_seconds=15)
    assert receipt.status == "accepted"
    record = fixture.executor.inspect(fixture.request)
    assert record["helper_result_schema_version"] == 2
    directory = Path(record["invocation_dir"])
    result = helper.read_json(directory / "result.json")
    manifest = helper.read_json(directory / "invocation.json")
    assert manifest["schema_version"] == manifest["result_schema_version"] == 2
    assert helper.validate_helper_result(directory, result, manifest=manifest) == 2
    effect = (tmp_path / "external_effect.json").read_bytes()
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=15) == receipt
    assert (tmp_path / "external_effect.json").read_bytes() == effect
    (directory / "stdout.raw").write_bytes(b"552;cluster\n")
    with pytest.raises(StoreIntegrityError, match="raw output evidence"):
        fixture.executor.lookup(fixture.request, timeout_seconds=15)
    assert (tmp_path / "external_effect.json").read_bytes() == effect


def test_new_executor_result_cannot_downgrade_by_deleting_record_marker(tmp_path):
    fixture = make_executor_fixture(tmp_path)
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=15).status == "accepted"
    record = fixture.executor.inspect(fixture.request)
    del record["helper_result_schema_version"]
    directory = Path(record["invocation_dir"])
    path = directory / "result.json"
    result = helper.read_json(path)
    for key in ("schema_version", "raw_outputs", "capture_status", "capture_error_type",
                "command_spawned"):
        result.pop(key)
    # Explicit fault injection in an isolated fixture, not a historical record.
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(StoreIntegrityError, match="raw output evidence"):
        fixture.executor._result(fixture.request, record)


def test_raw_file_fsync_failure_propagates_without_complete_receipt(tmp_path, monkeypatch):
    capture = helper._RawCapture(tmp_path / "stdout.raw")
    reader, writer = os.pipe()
    os.write(writer, b"known raw bytes")
    os.close(writer)
    original = helper.os.fsync

    def fail_raw(fd):
        if fd == capture.fd:
            raise OSError("injected raw fsync failure")
        return original(fd)

    monkeypatch.setattr(helper.os, "fsync", fail_raw)
    with os.fdopen(reader, "rb") as pipe:
        capture.pump(pipe)
    assert capture.error == "OSError" and not capture.complete
    with pytest.raises(OSError, match="injected raw fsync"):
        capture.freeze()
    assert (tmp_path / "stdout.raw").read_bytes() == b"known raw bytes"
