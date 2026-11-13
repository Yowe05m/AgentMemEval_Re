from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations.dispatch import NAMESPACE
from agentmemeval.path1_v1.operations.slurm import IntentReceipt

HELPER_PATH = Path(__file__).resolve().parents[1] / "unit/test_operations_dispatch.py"
HELPERS = runpy.run_path(str(HELPER_PATH))
make_fixture, make_request, step = (
    HELPERS[name] for name in ("make_fixture", "make_request", "step")
)


class FileExecutor:
    """Test-only durable external-effect fixture, not a remote executor implementation."""

    def __init__(self, root: Path):
        self.path = root / "simulated_external_acceptance.json"

    def submit_once(self, request, *, timeout_seconds):
        assert 0 < timeout_seconds <= 60
        receipt = IntentReceipt(request.request_sha256, "accepted", "a" * 64, "101", True, 1)
        # Exclusive create makes any duplicate side effect observable as failure.
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(asdict(receipt), stream)
            stream.flush()
            os.fsync(stream.fileno())
        return receipt

    def lookup(self, request, *, timeout_seconds):
        assert 0 < timeout_seconds <= 60
        if not self.path.exists():
            return None
        return IntentReceipt(**json.loads(self.path.read_text(encoding="utf-8")))


CHILD = r"""
import json, os, runpy, sys, time
from pathlib import Path
from agentmemeval.path1_v1.operations.state import CampaignStore
helpers = runpy.run_path(sys.argv[1])
root, mode = Path(sys.argv[2]), sys.argv[3]
fixture = helpers["make_fixture"](root, initialize=False)
fixture.adapter.executor = helpers["FileExecutor"](root)
class CrashStore(CampaignStore):
    def commit_transition(self, **kwargs):
        if mode == "receipt" and kwargs["reason"] == "record executor submission receipt":
            os._exit(73)
        result = super().commit_transition(**kwargs)
        if mode == "unknown" and kwargs["reason"] == "persist submit_unknown before external call":
            os._exit(74)
        if mode == "intent" and kwargs["reason"] == "persist submission intent":
            os._exit(75)
        return result
fixture.store = CrashStore(root / "control", "campaign")
if mode == "race":
    (root / ("ready_" + str(os.getpid()))).write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 10
    while not (root / "start").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("fixture barrier not released")
        time.sleep(0.01)
result = helpers["step"](fixture)
print(json.dumps({"action": result.action, "generation": result.generation}))
"""


def spawn(root: Path, mode: str):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, str(Path(__file__).resolve()), str(root), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def finish(process, expected: int = 0):
    try:
        output, error = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        raise
    assert process.returncode == expected, (output, error)
    return output


def resumed(root: Path):
    fixture = make_fixture(root, initialize=False)
    fixture.adapter.executor = FileExecutor(root)
    return fixture


def test_process_crash_after_persist_unknown_before_rpc_never_submits_on_restart(tmp_path):
    make_fixture(tmp_path)
    finish(spawn(tmp_path, "unknown"), expected=74)
    assert not (tmp_path / "simulated_external_acceptance.json").exists()
    fixture = resumed(tmp_path)
    for _ in range(3):
        assert step(fixture).action == "idle"
    assert not (tmp_path / "simulated_external_acceptance.json").exists()
    record = next(iter(fixture.store.read_registry().registry[NAMESPACE]["intents"].values()))
    assert record["status"] == "submit_unknown"


def test_process_crash_after_external_acceptance_before_local_receipt_reconciles(tmp_path):
    make_fixture(tmp_path)
    finish(spawn(tmp_path, "receipt"), expected=73)
    evidence = tmp_path / "simulated_external_acceptance.json"
    original = evidence.read_bytes()
    fixture = resumed(tmp_path)
    assert step(fixture).action == "idle"
    record = next(iter(fixture.store.read_registry().registry[NAMESPACE]["intents"].values()))
    assert record["status"] == "submitted" and record["job_id"] == "101"
    assert evidence.read_bytes() == original


def test_process_crash_after_safe_intent_before_claim_can_resume_once(tmp_path):
    make_fixture(tmp_path)
    finish(spawn(tmp_path, "intent"), expected=75)
    fixture = resumed(tmp_path)
    assert not fixture.store.read_registry().claims
    assert step(fixture).action == "submitted"
    assert step(fixture).action == "idle"
    assert len(fixture.store.read_registry().claims) == 1


def test_independent_dispatchers_race_with_only_one_external_effect(tmp_path):
    make_fixture(tmp_path)
    processes = [spawn(tmp_path, "race"), spawn(tmp_path, "race")]
    try:
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob("ready_*"))) != 2:
            assert all(process.poll() is None for process in processes)
            if time.monotonic() > deadline:
                raise TimeoutError("dispatcher processes did not reach barrier")
            time.sleep(0.01)
        (tmp_path / "start").write_text("start", encoding="utf-8")
        results = [json.loads(finish(process)) for process in processes]
        assert sum(result["action"] == "submitted" for result in results) == 1
        assert set(result["action"] for result in results) <= {"submitted", "skipped_lock", "idle"}
        assert (tmp_path / "simulated_external_acceptance.json").is_file()
        assert len(resumed(tmp_path).store.read_registry().claims) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_failed_unknown_commit_has_no_side_effect_and_can_resume_safe_intent(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    original = fixture.store.commit_transition

    def reject_unknown(**kwargs):
        if kwargs["reason"] == "persist submit_unknown before external call":
            raise OSError("fixture durable write failed")
        return original(**kwargs)

    monkeypatch.setattr(fixture.store, "commit_transition", reject_unknown)
    with pytest.raises(OSError):
        step(fixture)
    assert not fixture.executor.calls
    assert fixture.store.read_registry().claims[0].status == "started"
    monkeypatch.setattr(fixture.store, "commit_transition", original)
    assert step(fixture).action == "submitted" and len(fixture.executor.calls) == 1


def test_receipt_write_failure_after_side_effect_never_reissues(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    original = fixture.store.commit_transition

    def reject_receipt(**kwargs):
        if kwargs["reason"] == "record executor submission receipt":
            raise OSError("fixture database full")
        return original(**kwargs)

    monkeypatch.setattr(fixture.store, "commit_transition", reject_receipt)
    with pytest.raises(OSError):
        step(fixture)
    assert len(fixture.executor.calls) == 1
    monkeypatch.setattr(fixture.store, "commit_transition", original)
    assert step(fixture).action == "idle" and len(fixture.executor.calls) == 1
