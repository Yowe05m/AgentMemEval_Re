"""Offline service producer/store/reader tests; no Linux or GPU admission evidence."""

from dataclasses import replace

import pytest

from agentmemeval.path1_v1.operations import closure
from agentmemeval.path1_v1.operations.closure_service import (
    PersistedServiceReleaseVerifier,
    ServiceReleaseJournal,
)
from agentmemeval.path1_v1.operations.process_ownership import ProcessSnapshot
from agentmemeval.path1_v1.operations.state import CampaignStore
from path1_v1.unit.test_operations_closure import make_closed_fixture


def service_fixture(tmp_path, monkeypatch):
    data = make_closed_fixture(tmp_path, monkeypatch)
    binding, _, _ = closure._worker_evidence(data["submission"], data["job"], data["paths"])
    store = CampaignStore(tmp_path / "service_control", binding.campaign_id)
    with store.lock():
        store.initialize(registry={}, state={})
    directory = data["output"].parent / "service_release"
    journal = ServiceReleaseJournal.begin(
        store,
        run_id="service_fixture",
        binding=binding,
        directory=directory,
        worker_store=data["store"],
        worker_terminal_path=data["output"] / "worker_terminal.json",
    )
    owner = data["owner"]
    owner.binding, owner.identity = data["cleanup"].binding, data["cleanup"].identity
    snapshot = ProcessSnapshot(
        owner.binding, owner.identity, ("/fixture/python", "-m", "vllm"), "S"
    )
    owner.verify_snapshot = lambda candidate: candidate is snapshot
    reader = PersistedServiceReleaseVerifier(
        service_store=store,
        run_id="service_fixture",
        directory=directory,
        worker_store=data["store"],
        worker_terminal_path=data["output"] / "worker_terminal.json",
    )
    return data, binding, store, journal, reader, snapshot


def test_persisted_producer_facts_close_after_live_capability_is_gone(tmp_path, monkeypatch):
    data, binding, store, journal, reader, snapshot = service_fixture(tmp_path, monkeypatch)
    assert store.read_registry().claims[0].status == "started"
    journal.record_launch(data["owner"], snapshot)
    journal.publish(data["owner"], data["cleanup"])
    assert store.read_registry().claims[0].status == "completed"
    # The historical reader uses no owner object. It cannot signal any process.
    data["owner"].verify_cleanup = lambda receipt: False
    data["owner"].verify_snapshot = lambda value: False
    assert reader.verify(binding).binding == binding
    data["verifier"].service = reader
    receipt = data["verifier"](data["submission"], data["job"])
    assert receipt is not None, data["verifier"].last_failure
    assert not hasattr(reader, "close") and not hasattr(reader, "kill")


@pytest.mark.parametrize(
    "problem",
    [
        "copied_launch",
        "copied_cleanup",
        "worker_changed",
        "incomplete_cleanup",
        "wrong_launch_identity",
    ],
)
def test_producer_requires_real_owner_receipts_and_worker_claim(tmp_path, monkeypatch, problem):
    data, binding, store, journal, reader, snapshot = service_fixture(tmp_path, monkeypatch)
    if problem == "copied_launch":
        with pytest.raises(closure.ClosureError, match="provenance"):
            journal.record_launch(data["owner"], replace(snapshot))
    elif problem == "wrong_launch_identity":
        data["owner"].identity = replace(snapshot.identity, pid=51)
        with pytest.raises(closure.ClosureError, match="differs"):
            journal.record_launch(data["owner"], snapshot)
    else:
        journal.record_launch(data["owner"], snapshot)
        cleanup = data["cleanup"]
        if problem == "copied_cleanup":
            cleanup = replace(cleanup)
        elif problem == "incomplete_cleanup":
            cleanup = replace(cleanup, group_cleanup_verified=False)
            data["owner"].receipt = cleanup
        else:
            (data["output"] / "worker_terminal.json").write_bytes(b'{"verified":true}')
        with pytest.raises(closure.ClosureError):
            journal.publish(data["owner"], cleanup)
    assert store.read_registry().claims[0].status == "started"
    with pytest.raises((closure.ClosureError, FileNotFoundError)):
        reader.verify(binding)


@pytest.mark.parametrize("name", ["launch.json", "cleanup.json", "service_terminal.json"])
def test_persisted_json_flags_or_mutation_do_not_replace_producer_claim(
    tmp_path, monkeypatch, name
):
    data, binding, store, journal, reader, snapshot = service_fixture(tmp_path, monkeypatch)
    journal.record_launch(data["owner"], snapshot)
    journal.publish(data["owner"], data["cleanup"])
    (journal.directory / name).write_bytes(b'{"verified":true}')
    with pytest.raises(closure.ClosureError):
        reader.verify(binding)
    assert store.read_registry().claims[0].status == "completed"


def test_claim_not_completed_cannot_release_and_cannot_be_reused(tmp_path, monkeypatch):
    data, binding, store, journal, reader, snapshot = service_fixture(tmp_path, monkeypatch)
    journal.record_launch(data["owner"], snapshot)
    with pytest.raises((closure.ClosureError, FileNotFoundError)):
        reader.verify(binding)
    with pytest.raises(closure.ClosureError, match="already exists"):
        ServiceReleaseJournal.begin(
            store,
            run_id="service_fixture",
            binding=binding,
            directory=journal.directory,
            worker_store=data["store"],
            worker_terminal_path=data["output"] / "worker_terminal.json",
        )


def test_launch_mutation_before_publish_is_not_sealed_as_genuine(tmp_path, monkeypatch):
    data, binding, store, journal, reader, snapshot = service_fixture(tmp_path, monkeypatch)
    journal.record_launch(data["owner"], snapshot)
    (journal.directory / "launch.json").write_bytes(b'{"verified":true}')
    with pytest.raises(closure.ClosureError, match="hash mismatch"):
        journal.publish(data["owner"], data["cleanup"])
    assert store.read_registry().claims[0].status == "started"
