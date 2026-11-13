from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import state as state_module
from agentmemeval.path1_v1.operations.state import (
    CampaignStore,
    CampaignStoreError,
    ClaimConflict,
    GenerationConflict,
    LockRequired,
    LockUnavailable,
    StoreIntegrityError,
)


@pytest.fixture
def store(tmp_path: Path) -> CampaignStore:
    result = CampaignStore(tmp_path / "control", "campaign-1")
    with result.lock():
        result.initialize(registry={"runs": {}}, state={"phase": "W1"})
    return result


def test_registry_and_state_are_one_cas_transaction(store: CampaignStore) -> None:
    before = store.read_registry()
    with store.lock():
        after = store.commit_transition(
            expected_generation=0,
            registry={"runs": {"r": "planned"}},
            state={"phase": "E1"},
            reason="advance",
        )
        with pytest.raises(GenerationConflict):
            store.commit_transition(expected_generation=0, registry={}, state={}, reason="stale")
    assert before.generation == 0 and after.generation == 1
    assert after.registry == {"runs": {"r": "planned"}}
    assert after.state == {"phase": "E1"}
    assert store.read_registry() == after


def test_mutations_require_current_process_lock(store: CampaignStore, tmp_path: Path) -> None:
    with pytest.raises(LockRequired):
        store.commit_transition(expected_generation=0, registry={}, state={}, reason="invalid")
    with pytest.raises(LockRequired):
        store.claim_run(
            expected_generation=0, run_id="r", intent_id="i", run_directory=tmp_path / "r"
        )
    with pytest.raises(LockRequired):
        store.export_state_summary(tmp_path / "STATE.json")
    assert store.read_registry().generation == 0


def test_same_instance_lock_is_not_reentrant(store: CampaignStore) -> None:
    with store.lock():
        with pytest.raises(LockUnavailable):
            with store.lock():
                pass
        store.commit_transition(expected_generation=0, registry={}, state={}, reason="still owner")


def test_lock_release_retains_lock_file(store: CampaignStore) -> None:
    with store.lock():
        pass
    assert store.lock_path.is_file()
    with store.lock():
        assert store.read_registry().generation == 0


def test_other_thread_cannot_borrow_owner_lock(store: CampaignStore) -> None:
    errors: list[BaseException] = []

    def mutate() -> None:
        try:
            store.commit_transition(expected_generation=0, registry={}, state={}, reason="borrowed")
        except BaseException as exc:
            errors.append(exc)

    with store.lock():
        thread = threading.Thread(target=mutate)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], LockRequired)


def test_unlock_failure_still_closes_handle(
    store: CampaignStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_unlock = state_module._unlock_fd

    def fail_unlock(fd: int) -> None:
        raise OSError("unlock failure")

    monkeypatch.setattr(state_module, "_unlock_fd", fail_unlock)
    with pytest.raises(OSError, match="unlock failure"), store.lock():
        pass
    monkeypatch.setattr(state_module, "_unlock_fd", original_unlock)
    with CampaignStore(store.directory, store.campaign_id).lock():
        pass


@pytest.mark.parametrize("timeout", [-1, math.inf, math.nan, "1"])
def test_invalid_lock_timeout_rejected(store: CampaignStore, timeout: object) -> None:
    with pytest.raises(ValueError):
        with store.lock(timeout=timeout):  # type: ignore[arg-type]
            pass


def test_initialization_never_replaces_an_existing_database(store: CampaignStore) -> None:
    before = store.database_path.read_bytes()
    with store.lock():
        with pytest.raises(CampaignStoreError, match="already exists"):
            store.initialize(registry={"replacement": True})
    assert store.database_path.read_bytes() == before


def test_campaign_id_is_checked_on_every_read(store: CampaignStore) -> None:
    different = CampaignStore(store.directory, "wrong-campaign")
    with pytest.raises(StoreIntegrityError, match="identity"):
        different.read_registry()


@pytest.mark.parametrize("bad", [{1: "key"}, {"x": math.nan}, {"x": (1, 2)}, {"x": {1, 2}}])
def test_lossy_or_non_json_payloads_are_rejected(store: CampaignStore, bad: dict) -> None:
    with store.lock(), pytest.raises(ValueError):
        store.commit_transition(expected_generation=0, registry=bad, state={}, reason="invalid")
    assert store.read_registry().generation == 0


def test_cycle_rejected_without_mutation(store: CampaignStore) -> None:
    bad: dict = {}
    bad["self"] = bad
    with store.lock(), pytest.raises(ValueError, match="circular"):
        store.commit_transition(expected_generation=0, registry=bad, state={}, reason="cycle")
    assert store.read_registry().generation == 0


def test_returned_payload_changes_do_not_modify_store(store: CampaignStore) -> None:
    snapshot = store.read_registry()
    snapshot.registry["runs"]["fake"] = True
    snapshot.state["phase"] = "formal"
    assert store.read_registry().registry == {"runs": {}}
    assert store.read_registry().state == {"phase": "W1"}


def test_rollback_keeps_payload_and_audit_generation_atomic(
    store: CampaignStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_commit(connection: sqlite3.Connection) -> None:
        raise OSError("simulated durable commit failure")

    before = store.read_registry()
    monkeypatch.setattr(store, "_commit_transaction", fail_commit)
    with store.lock(), pytest.raises(OSError, match="simulated"):
        store.commit_transition(
            expected_generation=0, registry={"new": True}, state={"new": True}, reason="fault"
        )
    assert store.read_registry() == before
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] == 1


def test_claim_is_reserved_consumed_and_finished_once(store: CampaignStore, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    with store.lock():
        claim = store.claim_run(
            expected_generation=0,
            run_id="r1",
            intent_id="intent1",
            run_directory=run_dir,
            metadata={"artifact": "a1"},
        )
        assert claim.status == "reserved" and not run_dir.exists()
        started = store.start_claim(expected_generation=1, run_id="r1", token=claim.token)
        assert started.status == "started"
        with pytest.raises(ClaimConflict, match="consumed"):
            store.start_claim(expected_generation=2, run_id="r1", token=claim.token)
        finished = store.finish_claim(
            expected_generation=2,
            run_id="r1",
            token=claim.token,
            status="completed",
            receipt={"worker_exit_code": 0},
        )
        assert finished.status == "completed"
        with pytest.raises(ClaimConflict, match="terminal"):
            store.finish_claim(
                expected_generation=3,
                run_id="r1",
                token=claim.token,
                status="failed",
                receipt={"error": "cannot rewrite"},
            )
        with pytest.raises(ClaimConflict):
            store.claim_run(
                expected_generation=3, run_id="r1", intent_id="intent1", run_directory=run_dir
            )
    assert store.read_registry().generation == 3
    assert store.read_registry().claims == (finished,)


@pytest.mark.parametrize("duplicate", ["run_id", "intent_id", "path"])
def test_claim_uniqueness_covers_run_intent_and_path(
    store: CampaignStore,
    tmp_path: Path,
    duplicate: str,
) -> None:
    with store.lock():
        store.claim_run(
            expected_generation=0, run_id="r1", intent_id="i1", run_directory=tmp_path / "r1"
        )
        with pytest.raises(ClaimConflict):
            store.claim_run(
                expected_generation=1,
                run_id="r1" if duplicate == "run_id" else "r2",
                intent_id="i1" if duplicate == "intent_id" else "i2",
                run_directory=tmp_path / ("r1" if duplicate == "path" else "r2"),
            )
    assert store.read_registry().generation == 1


def test_empty_existing_directory_also_cannot_be_claimed(
    store: CampaignStore, tmp_path: Path
) -> None:
    path = tmp_path / "existing"
    path.mkdir()
    with store.lock(), pytest.raises(ClaimConflict, match="already exists"):
        store.claim_run(expected_generation=0, run_id="r", intent_id="i", run_directory=path)


def test_output_path_created_after_reservation_prevents_start(
    store: CampaignStore,
    tmp_path: Path,
) -> None:
    path = tmp_path / "output"
    with store.lock():
        claim = store.claim_run(
            expected_generation=0, run_id="r", intent_id="i", run_directory=path
        )
        path.mkdir()
        with pytest.raises(ClaimConflict):
            store.start_claim(expected_generation=1, run_id="r", token=claim.token)
    assert store.read_registry().claims[0].status == "reserved"


def test_claim_cannot_overlap_control_or_use_relative_path(store: CampaignStore) -> None:
    with store.lock():
        for path in (
            store.directory,
            store.directory / "run",
            store.directory.parent,
            Path("relative"),
        ):
            with pytest.raises(ValueError):
                store.claim_run(
                    expected_generation=0, run_id="r", intent_id="i", run_directory=path
                )


def test_wrong_claim_token_and_unstarted_completion_fail(
    store: CampaignStore, tmp_path: Path
) -> None:
    with store.lock():
        claim = store.claim_run(
            expected_generation=0, run_id="r", intent_id="i", run_directory=tmp_path / "r"
        )
        with pytest.raises(ClaimConflict, match="token"):
            store.start_claim(expected_generation=1, run_id="r", token="incorrect")
        with pytest.raises(ClaimConflict, match="unstarted"):
            store.finish_claim(
                expected_generation=1,
                run_id="r",
                token=claim.token,
                status="completed",
                receipt={"exit_code": 0},
            )
        failed = store.finish_claim(
            expected_generation=1,
            run_id="r",
            token=claim.token,
            status="failed",
            receipt={"reason": "never launched"},
        )
        assert failed.status == "failed"
        with pytest.raises(ClaimConflict):
            store.start_claim(expected_generation=2, run_id="r", token=claim.token)


def test_claim_insert_and_generation_roll_back_together(
    store: CampaignStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_commit(connection: sqlite3.Connection) -> None:
        raise OSError("failed commit")

    monkeypatch.setattr(store, "_commit_transaction", fail_commit)
    with store.lock(), pytest.raises(OSError):
        store.claim_run(
            expected_generation=0, run_id="r", intent_id="i", run_directory=tmp_path / "r"
        )
    assert store.read_registry().generation == 0
    assert store.read_registry().claims == ()


def test_corrupted_database_payload_is_not_accepted(store: CampaignStore) -> None:
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("UPDATE campaign SET state_json='{}'")
    with pytest.raises(StoreIntegrityError, match="digest"):
        store.read_registry()


def test_summary_is_non_authoritative_even_when_forged(
    store: CampaignStore, tmp_path: Path
) -> None:
    target = tmp_path / "STATE.json"
    with store.lock():
        view = store.export_state_summary(target)
        assert view["authoritative"] is False and view["submission_authority"] is False
        view.update(generation=999, authoritative=True, state={"phase": "formal"})
        target.write_text(json.dumps(view), encoding="utf-8")
        assert store.read_registry().generation == 0
        assert store.read_registry().state == {"phase": "W1"}
        with pytest.raises(GenerationConflict):
            store.commit_transition(expected_generation=999, registry={}, state={}, reason="forged")
        actual = store.export_state_summary(target)
    assert json.loads(target.read_text(encoding="utf-8")) == actual


def test_summary_cannot_replace_store_or_lock(store: CampaignStore) -> None:
    with store.lock():
        paths = (
            store.database_path,
            store.lock_path,
            *(
                store.directory / (store.database_path.name + suffix)
                for suffix in ("-journal", "-wal", "-shm")
            ),
        )
        for path in paths:
            with pytest.raises(ValueError):
                store.export_state_summary(path)


def test_failed_summary_replace_leaves_authority_and_old_view_unchanged(
    store: CampaignStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "STATE.json"
    target.write_text("previous view", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failure")

    monkeypatch.setattr(state_module.os, "replace", fail_replace)
    with store.lock(), pytest.raises(OSError, match="replace failure"):
        store.export_state_summary(target)
    assert target.read_text(encoding="utf-8") == "previous view"
    assert store.read_registry().generation == 0
    assert len(list(tmp_path.glob("STATE.json.*.partial"))) == 1
