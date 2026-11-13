"""Retry lineage uses real canonical inputs and SQLite; only terminal issuance is fake.

These are admission boundary tests, not evidence that an actual Slurm intent
has been reconciled. No subprocess scheduler or model is executed here.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile
from agentmemeval.path1_v1.operations.contracts import (
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, SubmissionRequest
from agentmemeval.path1_v1.operations.state import StoreIntegrityError
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e0_executor import make_e0_fixture
from path1_v1.unit.test_operations_contracts import _git


@pytest.fixture
def lineage(tmp_path, monkeypatch):
    original = make_e0_fixture(tmp_path)
    executor = original.executor
    store = executor.store
    with store.lock():
        executor._initialize()
    paths = tmp_path / "lineage_inputs"
    paths.mkdir()
    issued = {}
    calls = []

    def resolver(actual_store, intent):
        assert actual_store is store
        calls.append(intent)
        return issued[intent]

    monkeypatch.setattr(e0_reconcile, "resolve_e0_terminal_intent", resolver)
    return SimpleNamespace(original=original, executor=executor, store=store,
                           paths=paths, issued=issued, calls=calls)


def _contract(case, *, attempt=1, prior=None, campaign=None, kind="resource_probe"):
    value = copy.deepcopy(case.original.contract)
    if campaign is not None:
        value["campaign_id"] = campaign
    value["job"].update(kind=kind, attempt=attempt,
                        output_relpath=f"{value['campaign_id']}/E0/{kind}/attempt_{attempt:02d}",
                        retry_evidence=None if prior is None else _retry(prior))
    return validate_campaign_contract(value)


def _retry(prior):
    return {"prior_intent_id": canonical_intent_id(prior),
            "prior_terminal_receipt_sha256": "a" * 64,
            "eligibility_receipt_sha256": "b" * 64,
            "scope": "complete_unit", "seed_changed": False}


def _worker(contract):
    return SimpleNamespace(contract=validate_campaign_contract(contract),
                           intent_id=canonical_intent_id(contract))


def _record(case, contract):
    """Commit a real durable claimed record with complete probe/contract byte bindings."""
    worker = _worker(contract)
    directory = case.paths / worker.intent_id
    directory.mkdir()
    contract_path = directory / "contract.json"
    contract_path.write_bytes(canonical_json(contract))
    probe = copy.deepcopy(case.original.request_data)
    probe.update(intent_id=worker.intent_id, contract_path=str(contract_path),
                 contract_sha256=file_sha256(contract_path), output_root=str(directory / "output"))
    probe["request_sha256"] = content_digest(
        {key: value for key, value in probe.items() if key != "request_sha256"})
    probe_path = directory / "probe_request.json"
    probe_path.write_bytes(canonical_json(probe))
    request = SubmissionRequest(contract["campaign_id"], worker.intent_id,
                                case.original.request.spec)
    record = {"campaign_id": contract["campaign_id"], "intent_id": worker.intent_id,
              "spec": request.spec.payload(), "request_sha256": request.request_sha256,
              "worker_request_sha256": probe["request_sha256"], "status": "claimed",
              "submitter_finished": False,
              "invocation_dir": str(case.store.directory / "i" / worker.intent_id[:16]),
              "probe_request_path": str(probe_path),
              "probe_request_file_sha256": file_sha256(probe_path)}
    with case.store.lock():
        snapshot = case.store.read_registry()
        ledger = copy.deepcopy(snapshot.registry[NAMESPACE])
        ledger["intents"][worker.intent_id] = record
        case.executor._commit(snapshot, ledger, "offline claimed lineage fixture")
    return record


def _replace_record(case, record):
    with case.store.lock():
        snapshot = case.store.read_registry()
        ledger = copy.deepcopy(snapshot.registry[NAMESPACE])
        ledger["intents"][record["intent_id"]] = record
        case.executor._commit(snapshot, ledger, "offline corruption fixture")


def _issue(case, prior):
    intent = canonical_intent_id(prior)
    validated = []

    def validate(record, *, store):
        assert store is case.store and record["intent_id"] == intent
        validated.append(intent)

    result = SimpleNamespace(intent_id=intent, prior_terminal_receipt_sha256="a" * 64,
                             eligibility_receipt_sha256="b" * 64, validate=validate,
                             classification="unstarted_scheduler_deadline")
    case.issued[intent] = result
    return validated


def _check(case, contract):
    # Production invokes the gate while holding this same lifecycle lock.
    with case.store.lock():
        case.executor._validate_e0_retry(_worker(contract))


def test_executed_resource_terminal_never_grants_unstarted_retry_eligibility(lineage):
    prior = _contract(lineage)
    _record(lineage, prior)
    _issue(lineage, prior)
    issued = lineage.issued[canonical_intent_id(prior)]
    issued.classification = "resource_terminal_observed"
    with pytest.raises(SubmissionDisabled, match="does not grant retry"):
        _check(lineage, _contract(lineage, attempt=2, prior=prior))


def _new_source(case, contract):
    identity = contract["runtime_identity"]
    repository = case.original.repo
    (repository / "src/core.py").write_text("# second source revision\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "alternate source cannot reset attempt lineage")
    contract["runtime_identity"] = build_runtime_identity(
        repository, core_paths=[row["path"] for row in identity["core_files"]],
        operations_paths=[row["path"] for row in identity["operations_files"]],
        config_path=identity["config"]["path"],
        dependency_lock_path=identity["dependency_lock"]["path"])
    return validate_campaign_contract(contract)


def test_first_attempt_has_no_terminal_dependency(lineage):
    _check(lineage, _contract(lineage))
    assert lineage.calls == []


@pytest.mark.parametrize("attempt", [1, 2])
@pytest.mark.parametrize("change", ["source", "run"])
def test_same_attempt_cannot_restart_with_another_source_or_output(lineage, attempt, change):
    prior = _contract(lineage)
    if attempt == 2:
        _record(lineage, prior)
    old = _contract(lineage, attempt=attempt, prior=prior if attempt == 2 else None)
    _record(lineage, old)
    new = copy.deepcopy(old)
    if change == "source":
        new = _new_source(lineage, new)
    else:
        new["job"]["output_relpath"] += "_new_run"
    assert canonical_intent_id(old) != canonical_intent_id(new)
    with pytest.raises(SubmissionDisabled, match="already has another intent"):
        _check(lineage, new)
    assert lineage.calls == []


def test_same_terminal_cannot_fork_a_second_child_with_different_attempt(lineage):
    prior = _contract(lineage)
    _record(lineage, prior)
    _record(lineage, _contract(lineage, attempt=2, prior=prior))
    fork = _contract(lineage, attempt=3, prior=prior)
    with pytest.raises(SubmissionDisabled, match="already has another intent"):
        _check(lineage, fork)
    assert lineage.calls == []


@pytest.mark.parametrize("attempt", [2, 3])
@pytest.mark.parametrize("self_present", [False, True])
def test_one_continuous_child_and_pre_go_self_are_allowed(lineage, attempt, self_present):
    prior = _contract(lineage)
    _record(lineage, prior)
    if attempt == 3:
        prior = _contract(lineage, attempt=2, prior=prior)
        _record(lineage, prior)
    validated = _issue(lineage, prior)
    child = _contract(lineage, attempt=attempt, prior=prior)
    if self_present:
        _record(lineage, child)
    _check(lineage, child)
    assert lineage.calls == validated == [canonical_intent_id(prior)]


@pytest.mark.parametrize("change", ["seed", "campaign", "kind", "skip_attempt"])
def test_retry_must_continue_exact_campaign_kind_seed_and_attempt(lineage, change):
    prior = _contract(lineage)
    _record(lineage, prior)
    _issue(lineage, prior)
    child = _contract(lineage, attempt=2, prior=prior)
    if change == "seed":
        child["seed_namespace"] += "_changed"
    elif change == "campaign":
        child["campaign_id"] = "another_campaign"
    elif change == "kind":
        child["job"]["kind"] = "service_probe"
    else:
        child["job"]["attempt"] = 3
    with pytest.raises(SubmissionDisabled, match="lineage/campaign/attempt"):
        _check(lineage, child)


@pytest.mark.parametrize("key", ["prior_terminal_receipt_sha256", "eligibility_receipt_sha256"])
def test_retry_hashes_must_match_issued_resolution(lineage, key):
    prior = _contract(lineage)
    _record(lineage, prior)
    _issue(lineage, prior)
    child = _contract(lineage, attempt=2, prior=prior)
    child["job"]["retry_evidence"][key] = "c" * 64
    with pytest.raises(SubmissionDisabled, match="differs from durable terminal seal"):
        _check(lineage, child)


@pytest.mark.parametrize("isolation", ["campaign", "kind"])
def test_other_campaign_or_job_kind_does_not_consume_this_initial_attempt(lineage, isolation):
    other = _contract(lineage, campaign="other_campaign" if isolation == "campaign" else None,
                      kind="service_probe" if isolation == "kind" else "resource_probe")
    _record(lineage, other)
    _check(lineage, _contract(lineage))
    assert lineage.calls == []


@pytest.mark.parametrize("change", ["probe", "contract", "canonical_intent"])
def test_stored_probe_and_contract_bytes_cannot_be_rebound(lineage, change):
    prior = _contract(lineage)
    record = _record(lineage, prior)
    probe_path = Path(record["probe_request_path"])
    contract_path = probe_path.with_name("contract.json")
    if change == "probe":
        probe_path.write_bytes(probe_path.read_bytes() + b" ")
    elif change == "contract":
        contract_path.write_bytes(contract_path.read_bytes() + b" ")
    else:
        from agentmemeval.path1_v1.operations.submission_process import read_json

        prior["job"]["output_relpath"] += "_different"
        contract_path.write_bytes(canonical_json(prior))
        probe = read_json(probe_path)
        probe["contract_sha256"] = file_sha256(contract_path)
        probe_path.write_bytes(canonical_json(probe))
        record["probe_request_file_sha256"] = file_sha256(probe_path)
    with pytest.raises(StoreIntegrityError, match="bytes changed|identity changed"):
        lineage.executor._stored_probe_contract(record)


def test_corrupt_record_campaign_cannot_hide_a_same_campaign_attempt(lineage):
    prior = _contract(lineage)
    record = _record(lineage, prior)
    record["campaign_id"] = "wrong_record_campaign"
    changed_request = SubmissionRequest(record["campaign_id"], record["intent_id"],
                                        lineage.original.request.spec)
    record["request_sha256"] = changed_request.request_sha256
    _replace_record(lineage, record)
    new = copy.deepcopy(prior)
    new["job"]["output_relpath"] += "_new_run"
    with pytest.raises(StoreIntegrityError, match="identity changed"):
        _check(lineage, new)


def test_probe_intent_metadata_cannot_disagree_with_bound_contract(lineage):
    from agentmemeval.path1_v1.operations.submission_process import read_json

    record = _record(lineage, _contract(lineage))
    path = Path(record["probe_request_path"])
    probe = read_json(path)
    probe["intent_id"] = "c" * 64
    probe["request_sha256"] = content_digest(
        {key: value for key, value in probe.items() if key != "request_sha256"})
    path.write_bytes(canonical_json(probe))
    record["probe_request_file_sha256"] = file_sha256(path)
    with pytest.raises(StoreIntegrityError, match="identity changed"):
        lineage.executor._stored_probe_contract(record)
