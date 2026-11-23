"""One technical resource-policy replacement; never a scientific retry.

Structural checks use real anonymous Git/model identities. Linux lifecycle
checks reuse actual prepared files and SQLite seals; only scheduler/process
observations come from the explicit rejection fixture. No sbatch is executed.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import e0_service_cli as cli
from agentmemeval.path1_v1.operations import service_supersession as ss
from agentmemeval.path1_v1.operations import submission_rejection as rejection
from agentmemeval.path1_v1.operations.contracts import (
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.intent_executor import (
    NAMESPACE,
    DurableIntentExecutor,
    SubmissionAdmission,
)
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e0_executor import make_e0_fixture
from path1_v1.unit.test_operations_contracts import _git, _seal_identity


def _rebuild(data, identity):
    return build_runtime_identity(
        data.repo,
        core_paths=[row["path"] for row in identity["core_files"]],
        operations_paths=[row["path"] for row in identity["operations_files"]],
        config_path=identity["config"]["path"],
        dependency_lock_path=identity["dependency_lock"]["path"],
        model_manifest_path=Path(data.request_data["model_manifest_path"]),
        model_root=Path(data.request_data["model_root"]),
    )


@pytest.fixture(scope="module")
def transition(tmp_path_factory):
    data = make_e0_fixture(tmp_path_factory.mktemp("policy"), "service_probe")
    prior = copy.deepcopy(data.contract)
    identity = prior["runtime_identity"]
    path = data.repo / identity["config"]["path"]
    old = json.loads(path.read_text(encoding="utf-8"))
    old.update(resource_profile_version=2, language_model_only=True)
    path.write_bytes(canonical_json(old))
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "-qm", "historical text-only v2")
    prior["runtime_identity"] = _rebuild(data, identity)
    new = {
        **old,
        "resource_profile_version": 3,
        "memory_mib": None,
        "memory_policy": "site_default",
    }
    path.write_bytes(canonical_json(new))
    operation = data.repo / identity["operations_files"][0]["path"]
    operation.write_bytes(operation.read_bytes() + b"\n# reviewed operations evolution\n")
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "-qm", "reviewed operations and v3 memory correction")
    current = copy.deepcopy(prior)
    current["runtime_identity"] = _rebuild(data, identity)
    return prior, current, old, new


def test_exact_policy_transition_allows_operations_evolution_without_granting_authority(transition):
    prior, current, old, new = transition
    assert prior["runtime_identity"]["model"] is not None
    assert prior["runtime_identity"]["model"]["files_verified"] is True
    assert prior["runtime_identity"]["core_files"] == current["runtime_identity"]["core_files"]
    assert (
        prior["runtime_identity"]["operations_files"]
        != current["runtime_identity"]["operations_files"]
    )
    assert ss.validate_transition(prior, current, old, new) is None
    assert prior["job"]["attempt"] == current["job"]["attempt"] == 1
    assert current["job"]["retry_evidence"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("service_port", 18235),
        ("model_root", "/other/model"),
        ("cluster", "other_cluster"),
        ("accounting_since", "2001-01-01T00:00:00+00:00"),
        ("startup_seconds", 1000),
        ("memory_mib", 65536),
        ("memory_policy", "automatic"),
        ("language_model_only", False),
    ],
)
def test_no_config_change_beyond_the_three_memory_policy_fields(transition, field, value):
    prior, current, old, new = copy.deepcopy(transition)
    new[field] = value
    with pytest.raises(SubmissionDisabled):
        ss.validate_transition(prior, current, old, new)


@pytest.mark.parametrize("field", ["core_files", "dependency_lock", "model"])
def test_schema_valid_core_model_and_dependency_drift_is_rejected(transition, field):
    prior, current, old, new = copy.deepcopy(transition)
    identity = current["runtime_identity"]
    if field == "core_files":
        identity[field][0]["sha256"] = "a" * 64
        identity["core_manifest_sha256"] = content_digest(identity[field])
    elif field == "dependency_lock":
        identity[field]["sha256"] = "a" * 64
    else:
        identity[field]["manifest_sha256"] = "a" * 64
    _seal_identity(identity)
    validate_campaign_contract(current)
    with pytest.raises(SubmissionDisabled, match="core/model/dependency identity changed"):
        ss.validate_transition(prior, current, old, new)


@pytest.mark.parametrize("change", ["campaign", "seed", "job", "kind", "retry", "science"])
def test_no_scientific_or_unit_identity_reset(transition, change):
    prior, current, old, new = copy.deepcopy(transition)
    if change == "campaign":
        current["campaign_id"] = "different_campaign"
    elif change == "seed":
        current["seed_namespace"] += "_different"
    elif change == "job":
        current["job"]["output_relpath"] += "_different"
    elif change == "kind":
        current["job"]["kind"] = "resource_probe"
    elif change == "retry":
        current["job"]["attempt"] = 2
        current["job"]["retry_evidence"] = {
            "prior_intent_id": canonical_intent_id(prior),
            "prior_terminal_receipt_sha256": "a" * 64,
            "eligibility_receipt_sha256": "b" * 64,
            "scope": "complete_unit",
            "seed_changed": False,
        }
    else:
        current["phase"] = "E1"
        current["cohort"].update(
            formation_count=1, episodes_per_artifact_per_arm=2, seed_inventory_sha256="a" * 64
        )
        current["job"].update(
            kind="technical_closed_loop",
            artifact_id="a",
            pair_ids=["pair1", "pair2"],
            arms=["retained", "removed"],
        )
    validate_campaign_contract(current)
    with pytest.raises(SubmissionDisabled):
        ss.validate_transition(prior, current, old, new)


linux = pytest.mark.skipif(sys.platform != "linux", reason="historical Slurm paths require Linux")


def _prepared_child(data, name="new_v3"):
    config = er._json(data.config_path)
    config.update(resource_profile_version=3, memory_mib=None, memory_policy="site_default")
    data.config_path.write_bytes(canonical_json(config))
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "--allow-empty", "-qm", "v3 resource correction")
    root = data.run.parent / name
    relative = data.config_path.relative_to(data.repo).as_posix()
    prepared = cli.prepare(data.repo, root, data.request.campaign_id, relative)
    _, _, request = cli.read_prepared(root, prepared["descriptor_sha256"])
    contract = er._json(root / "contract.json")
    probe = er._json(root / "probe_request.json")
    worker = SimpleNamespace(
        contract=contract,
        request=probe,
        intent_id=request.intent_id,
        request_sha256=probe["request_sha256"],
    )
    return SimpleNamespace(
        root=root, request=request, contract=contract, probe=probe, worker=worker
    )


def _fixture(tmp_path, monkeypatch, *, sealed=True):
    # This fixture publishes no fake production authority: Git/files/SQLite are
    # real; its documented live scheduler/process callbacks are injected.
    from path1_v1.unit.test_submission_rejection import make_rejection_fixture

    data = make_rejection_fixture(tmp_path, monkeypatch)
    data.initial = data.store.read_registry()
    if sealed:
        data.resolution = rejection.reconcile_rejection(data.store, **data.args)
    data.child = _prepared_child(data)
    # Only the real ledger/lineage/admit methods are exercised, not a submitter.
    executor = object.__new__(DurableIntentExecutor)
    executor.store = data.store
    executor.cluster = data.request.spec.cluster
    executor.account = data.request.spec.account
    executor.fixture_mode = False  # Actual Linux canonical paths; no remote-path substitution.
    executor.clock = lambda: datetime.now(UTC)
    data.executor = executor
    return data


def _save_record(data, record):
    with data.store.lock():
        snapshot = data.store.read_registry()
        registry = copy.deepcopy(snapshot.registry)
        registry[NAMESPACE]["intents"][record["intent_id"]] = record
        data.store.commit_transition(
            expected_generation=snapshot.generation,
            registry=registry,
            state=snapshot.state,
            reason="offline lineage fixture",
        )


def _child_record(data, child, provenance):
    record = {
        "campaign_id": child.contract["campaign_id"],
        "intent_id": child.request.intent_id,
        "spec": child.request.spec.payload(),
        "request_sha256": child.request.request_sha256,
        "worker_request_sha256": child.probe["request_sha256"],
        "status": "claimed",
        "submitter_finished": False,
        "invocation_dir": str(data.store.directory / "i" / child.request.intent_id[:16]),
        "probe_request_path": str(child.root / "probe_request.json"),
        "probe_request_file_sha256": file_sha256(child.root / "probe_request.json"),
    }
    if provenance is not None:
        record["operational_supersession"] = provenance
    return record


def _check(data, child=None):
    with data.store.lock():
        return data.executor._validate_e0_retry((child or data.child).worker)


@linux
def test_real_seal_new_prepare_and_saved_self_preserve_original_record_and_attempt(
    tmp_path, monkeypatch
):
    data = _fixture(tmp_path, monkeypatch)
    before = data.store.read_registry()
    issued = _check(data)
    payload = issued.payload()
    assert payload["prior_intent_id"] == data.request.intent_id
    assert payload["new_intent_id"] == data.child.request.intent_id != data.request.intent_id
    assert payload["original_attempt_unchanged"] is True
    assert payload["retry_eligible"] is payload["scientific_admission"] is False
    assert data.store.read_registry() == before
    record = _child_record(data, data.child, payload)
    _save_record(data, record)
    assert _check(data).payload() == payload
    ss.validate_stored_supersession(record, data.store)
    after = data.store.read_registry()
    assert after.registry[NAMESPACE]["intents"][data.request.intent_id] == data.record
    assert after.claims == before.claims
    assert after.claims[0].status == "failed"
    assert data.child.contract["job"]["attempt"] == 1
    assert data.child.contract["job"]["retry_evidence"] is None


@linux
@pytest.mark.parametrize("defect", ["missing", "seal_hash", "old_intent"])
def test_missing_or_changed_saved_provenance_refuses_pre_go(tmp_path, monkeypatch, defect):
    data = _fixture(tmp_path, monkeypatch)
    payload = _check(data).payload()
    if defect == "missing":
        payload = None
    elif defect == "seal_hash":
        payload["prior_evidence_file_sha256"] = "0" * 64
    else:
        payload["prior_intent_id"] = "0" * 64
    _save_record(data, _child_record(data, data.child, payload))
    before = data.store.read_registry()
    with pytest.raises((SubmissionDisabled, er.E0ReconciliationError)):
        _check(data)
    assert data.store.read_registry() == before


@linux
@pytest.mark.parametrize("state", ["unsealed", "accepted"])
def test_unknown_unsealed_or_accepted_predecessor_does_not_grant_transition(
    tmp_path, monkeypatch, state
):
    data = _fixture(tmp_path, monkeypatch, sealed=False)
    prior = copy.deepcopy(data.record)
    if state == "accepted":
        prior.update(status="accepted", job_id="123")
        _save_record(data, prior)
    before = data.store.read_registry()
    with pytest.raises((SubmissionDisabled, er.E0ReconciliationError)):
        ss.resolve_service_supersession(data.store, data.child.contract, data.child.probe, prior)
    assert data.store.read_registry() == before
    assert before.claims[0].status == "started"


@linux
@pytest.mark.parametrize("first", ["predecessor", "existing_child"])
def test_second_child_cannot_reuse_the_same_sealed_rejection(tmp_path, monkeypatch, first):
    data = _fixture(tmp_path, monkeypatch)
    _save_record(data, _child_record(data, data.child, _check(data).payload()))
    source = data.repo / data.child.contract["runtime_identity"]["operations_files"][0]["path"]
    source.write_bytes(source.read_bytes() + b"\n# different operations revision\n")
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "-qm", "alternate revision cannot create another child")
    second = _prepared_child(data, "second_v3")
    assert second.request.intent_id != data.child.request.intent_id
    before = data.store.read_registry()
    original_ledger = data.executor._ledger
    leading_intent = (
        data.request.intent_id if first == "predecessor" else data.child.request.intent_id
    )
    visited = []

    def ordered_ledger(snapshot):
        # Preserve all real source/seal/SQLite validation, varying only the
        # returned iteration order. Canonical JSON orders the real random
        # intent hashes, so either valid rejection branch can occur.
        ledger = original_ledger(snapshot)
        records = ledger["intents"]
        ledger["intents"] = {
            leading_intent: records[leading_intent],
            **{key: value for key, value in records.items() if key != leading_intent},
        }
        visited.append(next(iter(ledger["intents"])))
        return ledger

    monkeypatch.setattr(data.executor, "_ledger", ordered_ledger)
    with pytest.raises(SubmissionDisabled):
        _check(data, second)
    assert visited == [leading_intent]
    assert data.store.read_registry() == before
    assert second.request.intent_id not in before.registry[NAMESPACE]["intents"]
    assert all(claim.intent_id != second.request.intent_id for claim in before.claims)


@linux
def test_pre_go_admit_rechecks_terminal_bytes_before_consulting_site_verifier(
    tmp_path, monkeypatch
):
    data = _fixture(tmp_path, monkeypatch)
    _save_record(data, _child_record(data, data.child, _check(data).payload()))
    calls = []

    def admit(request, worker):
        calls.append(request.intent_id)
        return SubmissionAdmission(
            request.request_sha256,
            worker.request_sha256,
            worker.request["contract_sha256"],
            "b" * 64,
            1,
            data.executor.clock(),
        )

    data.executor.pre_submit_verifier = admit
    with data.store.lock():
        data.executor._admit(data.child.request, data.child.worker)
    assert calls == [data.child.request.intent_id]
    evidence = rejection._evidence_path(data.store, data.request.intent_id)
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    before = data.store.read_registry()
    with data.store.lock(), pytest.raises(ValueError):
        data.executor._admit(data.child.request, data.child.worker)
    assert calls == [data.child.request.intent_id]
    assert data.store.read_registry() == before
