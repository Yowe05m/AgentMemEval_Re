from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations.state import StoreIntegrityError

UNIT = Path(__file__).resolve().parents[1] / "unit"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(UNIT))
    from path1_v1.unit.test_intent_executor import make_executor_fixture

    return make_executor_fixture(tmp_path)


CHILD = r"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parents[1]))
from path1_v1.unit.test_intent_executor import make_executor_fixture, submit
from agentmemeval.path1_v1.operations.state import LockUnavailable
root, mode = Path(sys.argv[2]), sys.argv[3]
fixture = make_executor_fixture(root, initialize_worker=False)
original = fixture.executor._commit
def interrupted(snapshot, ledger, reason):
    if mode == "after_effect" and reason == "persist completed submitter evidence":
        os._exit(83)
    result = original(snapshot, ledger, reason)
    if mode == "before_process" and reason == "persist unknown before starting submitter":
        os._exit(82)
    return result
fixture.executor._commit = interrupted
if mode == "race":
    (root / ("ready_" + str(os.getpid()))).write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 15
    while not (root / "start").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("barrier not released")
        time.sleep(0.01)
try:
    receipt = submit(fixture)
    print(json.dumps({"status": receipt.status, "job_id": receipt.job_id}))
except LockUnavailable:
    print(json.dumps({"status": "locked"}))
"""


def spawn(root: Path, mode: str):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, str(UNIT), str(root), mode],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def finish(process, expected=0):
    try:
        stdout, stderr = process.communicate(timeout=25)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        raise
    assert process.returncode == expected, (stdout, stderr)
    return stdout


def test_crash_after_durable_unknown_never_starts_a_process_on_retry(fixture):
    finish(spawn(fixture.root, "before_process"), 82)
    assert not (fixture.root / "external_effect.json").exists()
    receipt = fixture.executor.lookup(fixture.request, timeout_seconds=1)
    assert receipt.status == "unknown" and not receipt.submitter_finished
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=2) == receipt
    assert not (fixture.root / "external_effect.json").exists()


def test_crash_after_actual_subprocess_effect_recovers_durable_reply_without_reissue(fixture):
    finish(spawn(fixture.root, "after_effect"), 83)
    original = (fixture.root / "external_effect.json").read_bytes()
    receipt = fixture.executor.lookup(fixture.request, timeout_seconds=1)
    assert receipt.status == "accepted" and receipt.job_id == "551"
    # Parent never durably recorded reaping the helper. Do not fabricate that proof.
    assert not receipt.submitter_finished
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=2) == receipt
    assert (fixture.root / "external_effect.json").read_bytes() == original


def test_two_processes_share_one_durable_claim_and_one_external_effect(fixture):
    processes = [spawn(fixture.root, "race"), spawn(fixture.root, "race")]
    try:
        deadline = time.monotonic() + 15
        while len(list(fixture.root.glob("ready_*"))) != 2:
            assert all(process.poll() is None for process in processes)
            if time.monotonic() > deadline:
                raise TimeoutError("executor subprocesses did not reach barrier")
            time.sleep(0.01)
        (fixture.root / "start").write_text("go", encoding="utf-8")
        results = [json.loads(finish(process)) for process in processes]
        assert any(result["status"] == "accepted" for result in results)
        assert {result["status"] for result in results} <= {"accepted", "locked"}
        assert (fixture.root / "external_effect.json").is_file()
        assert len(fixture.executor.store.read_registry().claims) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_durable_invocation_tamper_cannot_be_used_as_recovery_evidence(fixture):
    finish(spawn(fixture.root, "before_process"), 82)
    record = fixture.executor.inspect(fixture.request)
    manifest = Path(record["invocation_dir"]) / "invocation.json"
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    contents["argv"].append("--array=1-10")
    manifest.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(StoreIntegrityError):
        fixture.executor.lookup(fixture.request, timeout_seconds=1)
    assert not (fixture.root / "external_effect.json").exists()
