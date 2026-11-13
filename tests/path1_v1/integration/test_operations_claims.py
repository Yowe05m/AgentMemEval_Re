from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations.state import CampaignStore, ClaimConflict, LockUnavailable

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"


def _spawn(code: str, *arguments: str) -> subprocess.Popen[str]:
    environment = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen(
        [sys.executable, "-c", code, *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def _finish(process: subprocess.Popen[str], *, expected: int = 0) -> str:
    try:
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"child timed out: {stdout}\n{stderr}")
    assert process.returncode == expected, (stdout, stderr)
    return stdout.strip()


def _wait_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail(f"child failed before readiness: {stdout}\n{stderr}")
        time.sleep(0.01)


@pytest.fixture
def store(tmp_path: Path) -> CampaignStore:
    result = CampaignStore(tmp_path / "control", "competition")
    with result.lock():
        result.initialize(registry={"old": 1}, state={"phase": "W1"})
    return result


def test_independent_processes_cannot_claim_the_same_run_twice(
    store: CampaignStore,
    tmp_path: Path,
) -> None:
    gate = tmp_path / "go"
    code = """
import sys, time
from pathlib import Path
from agentmemeval.path1_v1.operations.state import CampaignStore, ClaimConflict
store = CampaignStore(sys.argv[1], 'competition')
deadline = time.monotonic() + 10
while not Path(sys.argv[2]).exists():
    if time.monotonic() > deadline: raise RuntimeError('gate timeout')
    time.sleep(.01)
with store.lock(timeout=10):
    snapshot = store.read_registry()
    try:
        store.claim_run(expected_generation=snapshot.generation, run_id='same', intent_id='same',
                        run_directory=Path(sys.argv[3]))
        print('won')
    except ClaimConflict:
        print('already_claimed')
"""
    children = [
        _spawn(code, str(store.directory), str(gate), str(tmp_path / "run")) for _ in range(2)
    ]
    gate.write_text("start", encoding="utf-8")
    try:
        outcomes = [_finish(process) for process in children]
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert sorted(outcomes) == ["already_claimed", "won"]
    assert store.read_registry().generation == 1
    assert len(store.read_registry().claims) == 1


def test_killed_owner_releases_os_lock_without_expiry_or_file_deletion(
    store: CampaignStore,
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    code = """
import sys, time
from pathlib import Path
from agentmemeval.path1_v1.operations.state import CampaignStore
with CampaignStore(sys.argv[1], 'competition').lock():
    Path(sys.argv[2]).write_text('locked', encoding='utf-8')
    time.sleep(30)
"""
    process = _spawn(code, str(store.directory), str(ready))
    try:
        _wait_file(ready, process)
        os.utime(store.lock_path, (1, 1))
        with pytest.raises(LockUnavailable):
            with store.lock():
                pass
        process.kill()
        process.communicate(timeout=10)
        assert store.lock_path.exists()
        with store.lock(timeout=2):
            store.commit_transition(
                expected_generation=0, registry={"recovered": True}, state={}, reason="new owner"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    assert store.read_registry().generation == 1


def test_process_crash_before_commit_rolls_back_payload_and_history(
    store: CampaignStore,
) -> None:
    code = """
import os, sys
from agentmemeval.path1_v1.operations.state import CampaignStore
class CrashStore(CampaignStore):
    def _connect(self, **kwargs):
        connection = super()._connect(**kwargs)
        connection.execute('PRAGMA cache_size=1')
        return connection
    def _commit_transaction(self, connection):
        os._exit(73)
with CrashStore(sys.argv[1], 'competition').lock() as store:
    store.commit_transition(expected_generation=0, registry={'new': 'x' * 1000000},
                            state={'phase': 'formal'}, reason='crash before commit')
"""
    _finish(_spawn(code, str(store.directory)), expected=73)
    with store.lock(timeout=2):
        snapshot = store.read_registry()
        assert snapshot.generation == 0
        assert snapshot.registry == {"old": 1}
        assert snapshot.state == {"phase": "W1"}
        store.commit_transition(
            expected_generation=0, registry={"next": 1}, state={}, reason="recover"
        )


def test_crashed_consumed_claim_cannot_be_started_again(
    store: CampaignStore, tmp_path: Path
) -> None:
    code = """
import os, sys
from pathlib import Path
from agentmemeval.path1_v1.operations.state import CampaignStore
with CampaignStore(sys.argv[1], 'competition').lock() as store:
    claim = store.claim_run(expected_generation=0, run_id='r', intent_id='i',
                            run_directory=Path(sys.argv[2]))
    store.start_claim(expected_generation=1, run_id='r', token=claim.token)
    os._exit(74)
"""
    _finish(_spawn(code, str(store.directory), str(tmp_path / "run")), expected=74)
    with store.lock(timeout=2):
        snapshot = store.read_registry()
        assert snapshot.generation == 2
        claim = snapshot.claims[0]
        assert claim.status == "started"
        with pytest.raises(ClaimConflict):
            store.start_claim(expected_generation=2, run_id="r", token=claim.token)
        with pytest.raises(ClaimConflict):
            store.claim_run(
                expected_generation=2, run_id="r", intent_id="i", run_directory=tmp_path / "run"
            )


def test_old_reader_loses_cas_after_another_process_commit(store: CampaignStore) -> None:
    old = store.read_registry()
    code = """
import sys
from agentmemeval.path1_v1.operations.state import CampaignStore
with CampaignStore(sys.argv[1], 'competition').lock() as store:
    store.commit_transition(expected_generation=0, registry={'writer': 2}, state={}, reason='other')
"""
    _finish(_spawn(code, str(store.directory)))
    from agentmemeval.path1_v1.operations.state import GenerationConflict

    with store.lock(), pytest.raises(GenerationConflict):
        store.commit_transition(
            expected_generation=old.generation,
            registry=old.registry,
            state=old.state,
            reason="stale reader",
        )
    assert store.read_registry().registry == {"writer": 2}


def test_summary_tampering_by_other_process_has_no_authority(
    store: CampaignStore,
    tmp_path: Path,
) -> None:
    target = tmp_path / "STATE.json"
    with store.lock():
        store.export_state_summary(target)
    view = json.loads(target.read_text(encoding="utf-8"))
    view.update(authoritative=True, submission_authority=True, generation=999)
    target.write_text(json.dumps(view), encoding="utf-8")
    code = """
import json, sys
from agentmemeval.path1_v1.operations.state import CampaignStore
snapshot = CampaignStore(sys.argv[1], 'competition').read_registry()
print(json.dumps({'generation': snapshot.generation, 'registry': snapshot.registry}))
"""
    observed = json.loads(_finish(_spawn(code, str(store.directory))))
    assert observed == {"generation": 0, "registry": {"old": 1}}
