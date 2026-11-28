"""Explicit offline service retry fixtures; no scheduler submission or GPU access."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import service_retry as retry
from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE, DurableIntentExecutor
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
from path1_v1.unit.test_e0_executor import make_e0_fixture
from path1_v1.unit.test_service_terminal import make_service_fixture, write

DRIVER_LOG = (
    b'  File "gpu_worker.py", line 1, in init_device\n'
    b"    torch._C._cuda_init()\n"
    b"RuntimeError: The NVIDIA driver on your system is too old (found version 12040)\n"
)


@pytest.fixture(scope="module")
def structural(tmp_path_factory):
    f = make_e0_fixture(tmp_path_factory.mktemp("retry"), "service_probe")
    prior = copy.deepcopy(f.contract)
    current = copy.deepcopy(prior)
    current["job"].update(
        attempt=2,
        output_relpath=prior["campaign_id"] + "/E0/service/attempt_02",
        retry_evidence=dict(
            prior_intent_id="a" * 64,
            prior_terminal_receipt_sha256="b" * 64,
            eligibility_receipt_sha256="c" * 64,
            scope="complete_unit",
            seed_changed=False,
        ),
    )
    old = {"resource_profile_version": 3, "cluster": "fixed", "service_python": "/old/python"}
    new = {**old, "service_python": "/new/python"}
    return prior, current, old, new


def test_environment_change_and_consecutive_attempt_preserve_scientific_unit(structural):
    assert retry.validate_transition(*structural) is None
    prior, current, old, new = copy.deepcopy(structural)
    prior["job"]["attempt"] = 2
    prior["job"]["retry_evidence"] = current["job"]["retry_evidence"]
    current["job"].update(
        attempt=3, output_relpath=current["campaign_id"] + "/E0/service/attempt_03"
    )
    retry.validate_transition(prior, current, old, new)


@pytest.mark.parametrize("field", ["campaign_id", "seed_namespace", "source_contract", "cohort"])
def test_unit_changes_rejected(structural, field):
    prior, current, old, new = copy.deepcopy(structural)
    if field in {"campaign_id", "seed_namespace"}:
        current[field] += "_other"
    elif field == "cohort":
        current[field]["formation_count"] = 1
    else:
        current[field]["provider"]["served_model_name"] = "other"
    with pytest.raises((ValueError, SubmissionDisabled)):
        retry.validate_transition(prior, current, old, new)


@pytest.mark.parametrize(
    "field", ["cluster", "control_root", "model_root", "service_port", "memory_mib", "gpus"]
)
def test_non_environment_configuration_changes_rejected(structural, field):
    prior, current, old, new = copy.deepcopy(structural)
    new[field] = "changed"
    with pytest.raises(SubmissionDisabled, match="exceeds"):
        retry.validate_transition(prior, current, old, new)


@pytest.mark.parametrize("attempt", [1, 3, 4, True])
def test_attempt_cannot_reset_skip_or_exceed_two_retries(structural, attempt):
    prior, current, old, new = copy.deepcopy(structural)
    current["job"]["attempt"] = attempt
    with pytest.raises((ValueError, SubmissionDisabled)):
        retry.validate_transition(prior, current, old, new)


def sealed_driver(tmp_path, monkeypatch, *, log=DRIVER_LOG, cleanup_returncode=1,
                  missing_http=True, attempt=1):
    f = make_service_fixture(tmp_path, monkeypatch)
    # Real SQLite/archive/CAS mechanics; inherited binder/scheduler are explicit fixtures.
    f.cleanup["returncode"] = cleanup_returncode
    write(f.root / "service_cleanup.json", f.cleanup)
    (f.root / "service_stderr.log").write_bytes(log)
    if missing_http:
        (f.root / "observations/host_service.json").unlink()
        (f.root / "observations/probe_receipt.json").unlink()
    for row in f.scheduler:
        row[5:7] = ["FAILED", "1:0"]
    f.close()
    actual = er.resolve_e0_terminal_intent(f.store, f.request.intent_id)
    assert actual.claim_status == "failed"
    contract = {"phase": "E0", "job": {"kind": "service_probe", "attempt": attempt}}
    monkeypatch.setattr(DurableIntentExecutor, "_stored_probe_contract", lambda record: contract)
    return f, actual


def test_driver_seal(tmp_path, monkeypatch):
    f, _ = sealed_driver(tmp_path, monkeypatch)
    value = retry.resolve_service_retry(f.store, f.request.intent_id)
    assert value.driver_api_version == 12040
    assert value.driver_log_sha256 == hashlib.sha256(DRIVER_LOG).hexdigest()
    assert value.retry_evidence()["eligibility_receipt_sha256"] == content_digest(value.payload())
    assert value.payload()["scientific_admission"] is False
    assert value.payload() == {
        "kind": "path1_e0_service_driver_retry_eligibility_v1",
        **{k: getattr(value, k) for k in value.__dataclass_fields__ if k != "_issuer"},
        "scope": "complete_unit", "seed_changed": False,
        "runtime_admission": False, "scientific_admission": False,
    }
    (f.root / "service_stderr.log").write_bytes(DRIVER_LOG + b"changed")
    with pytest.raises((SubmissionDisabled, er.E0ReconciliationError)):
        retry.resolve_service_retry(f.store, f.request.intent_id)


NVCC_LOG = (b'File "flashinfer/jit/cpp_ext.py", in get_cuda_path\n'
            b'File "flashinfer/sampling.py", in get_sampling_module\n'
            b'    gen_sampling_module().build_and_load()\n'
            b"RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' "
            b"doesn't exist\n")


@pytest.mark.parametrize("defect", [None, "unsealed_cleanup", "http_present", "other_error",
                                  "wrong_stack", "duplicate"])
def test_exact_sealed_missing_nvcc_can_retry_attempt_two_only(tmp_path, monkeypatch, defect):
    log = NVCC_LOG
    if defect == "other_error":
        log = log.replace(b"Could not find nvcc", b"unsupported GPU")
    if defect == "wrong_stack":
        log = log.replace(b"flashinfer/jit/cpp_ext.py", b"other.py")
    if defect == "duplicate":
        log += NVCC_LOG
    f, _ = sealed_driver(tmp_path, monkeypatch, log=log, attempt=2,
        cleanup_returncode=0 if defect == "unsealed_cleanup" else 1,
        missing_http=defect != "http_present")
    if defect is not None:
        with pytest.raises(SubmissionDisabled):
            retry.resolve_service_retry(f.store, f.request.intent_id)
    else:
        value = retry.resolve_service_retry(f.store, f.request.intent_id)
        payload = value.payload()
        assert payload["kind"] == "path1_e0_service_missing_nvcc_retry_eligibility_v1"
        assert payload["failure_log_sha256"] == hashlib.sha256(NVCC_LOG).hexdigest()
        assert "driver_api_version" not in payload and "driver_log_sha256" not in payload
        assert payload["runtime_admission"] is False and payload["scientific_admission"] is False


def test_unissued_capability_cannot_grant_retry():
    value = retry.ServiceRetryEligibility(
        "c", "/store", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, 12040
    )
    with pytest.raises(SubmissionDisabled, match="untrusted"):
        value.retry_evidence()


@pytest.mark.parametrize("status", ["submit_unknown", "claimed"])
def test_unknown_and_unstarted_records_do_not_resolve(monkeypatch, status):
    store = SimpleNamespace(
        read_registry=lambda: SimpleNamespace(
            registry={
                NAMESPACE: {"intents": {"prior": {"status": status, "submitter_finished": True}}}
            }
        )
    )
    monkeypatch.setattr(er, "resolve_e0_terminal_intent", lambda *a: pytest.fail("not accepted"))
    with pytest.raises(SubmissionDisabled, match="accepted"):
        retry.resolve_service_retry(store, "prior")


def test_executor_rechecks_service_retry_on_each_admission(monkeypatch, structural):
    calls = []
    monkeypatch.setattr(retry, "validate_service_retry", lambda *a: calls.append(a))
    executor = object.__new__(DurableIntentExecutor)
    executor.store = object()
    worker = SimpleNamespace(contract=structural[1], request={})
    executor._validate_e0_retry(worker)
    executor._validate_e0_retry(worker)
    assert len(calls) == 2


def store_record(root, contract, template):
    root.mkdir()
    cpath, ppath = root / "contract.json", root / "probe.json"
    cpath.write_bytes(canonical_json(contract))
    probe = {
        **template,
        "intent_id": canonical_intent_id(contract),
        "contract_path": str(cpath),
        "contract_sha256": er._sha(cpath),
        "output_root": str(root / "output"),
    }
    probe["request_sha256"] = content_digest(
        {k: v for k, v in probe.items() if k != "request_sha256"}
    )
    ppath.write_bytes(canonical_json(probe))
    record = dict(
        intent_id=probe["intent_id"],
        campaign_id=contract["campaign_id"],
        probe_request_path=str(ppath),
        probe_request_file_sha256=er._sha(ppath),
        worker_request_sha256=probe["request_sha256"],
    )
    return record, probe


@pytest.mark.parametrize("defect", [None, "self", "sibling", "same_prior", "proof", "seed"])
def test_lineage(tmp_path, monkeypatch, structural, defect):
    prior, current, old, new = copy.deepcopy(structural)
    template = dict(
        repository_root=str(tmp_path),
        model_root="/fixed/model",
        model_manifest_path="/fixed/model_manifest",
        request_sha256="",
    )
    old_record, _ = store_record(tmp_path / "prior", prior, template)
    eligibility = retry.ServiceRetryEligibility(
        "store",
        str(tmp_path),
        old_record["intent_id"],
        content_digest(prior),
        "b" * 64,
        "d" * 64,
        "e" * 64,
        12040,
        _issuer=retry._ISSUER,
    )
    current["job"]["retry_evidence"] = eligibility.retry_evidence()
    if defect == "seed":
        current["seed_namespace"] += "_changed"
    if defect == "proof":
        current["job"]["retry_evidence"]["eligibility_receipt_sha256"] = "f" * 64
    current_record, probe = store_record(tmp_path / "current", current, template)
    records = {old_record["intent_id"]: old_record}
    if defect == "self":
        records[current_record["intent_id"]] = current_record
    if defect in {"sibling", "same_prior"}:
        sibling = copy.deepcopy(current)
        if defect == "sibling":
            sibling["seed_namespace"] += "_other"
        else:
            sibling["job"].update(
                attempt=3, output_relpath=prior["campaign_id"] + "/E0/service/attempt_03"
            )
        record, _ = store_record(tmp_path / "sibling", sibling, template)
        records[record["intent_id"]] = record
    store = SimpleNamespace(
        read_registry=lambda: SimpleNamespace(registry={NAMESPACE: {"intents": records}})
    )
    monkeypatch.setattr(retry, "resolve_service_retry", lambda *a: eligibility)
    monkeypatch.setattr(
        er,
        "_historical_identity",
        lambda repo, identity, **k: old if identity == prior["runtime_identity"] else new,
    )
    # The fixture contracts have identical source; only the declared environment differs.
    if defect in {None, "self"}:
        assert retry.validate_service_retry(store, current, probe) == eligibility
    else:
        with pytest.raises(SubmissionDisabled):
            retry.validate_service_retry(store, current, probe)


@pytest.mark.skipif(sys.platform != "linux", reason="real prepare uses canonical Linux paths")
def test_prepare_attempt_two(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations import e0_service_cli as cli
    from agentmemeval.path1_v1.operations.state import CampaignStore
    from path1_v1.unit.test_e0_service_cli import fixture

    repo, _, config = fixture(tmp_path, monkeypatch)
    initial = cli.prepare(repo, tmp_path / "first", "campaign", "configs/path1_v1/site.json")
    prior = json.loads((tmp_path / "first/contract.json").read_text())
    probe = json.loads((tmp_path / "first/probe_request.json").read_text())
    record = dict(
        intent_id=initial["intent_id"],
        campaign_id="campaign",
        probe_request_path=str(tmp_path / "first/probe_request.json"),
        probe_request_file_sha256=er._sha(tmp_path / "first/probe_request.json"),
        worker_request_sha256=probe["request_sha256"],
    )
    scope = content_digest({"cluster": config["cluster"], "account": config["account"]})
    from pathlib import Path

    store = CampaignStore(
        Path(config["control_root"]) / "ie1" / scope[:16], "executor_" + scope[:32]
    )
    with store.lock():
        store.initialize(registry={NAMESPACE: {"intents": {initial["intent_id"]: record}}})
    eligibility = retry.ServiceRetryEligibility(
        store.campaign_id,
        str(store.directory),
        initial["intent_id"],
        content_digest(prior),
        "b" * 64,
        "c" * 64,
        "d" * 64,
        12040,
        _issuer=retry._ISSUER,
    )
    monkeypatch.setattr(retry, "resolve_service_retry", lambda *a: eligibility)
    result = cli.prepare(
        repo,
        tmp_path / "second",
        "campaign",
        "configs/path1_v1/site.json",
        attempt=2,
        prior_intent_id=initial["intent_id"],
    )
    cli.read_prepared(tmp_path / "second", result["descriptor_sha256"])
    contract = json.loads((tmp_path / "second/contract.json").read_text())
    assert (
        contract["job"]["attempt"] == 2
        and contract["job"]["retry_evidence"] == eligibility.retry_evidence()
    )
    assert contract["seed_namespace"] == prior["seed_namespace"]
    assert result["intent_id"] != initial["intent_id"]
    assert store.read_registry().registry[NAMESPACE]["intents"] == {initial["intent_id"]: record}


@pytest.mark.skipif(sys.platform != "linux", reason="historical Slurm paths require Linux")
def test_stored_descendants(tmp_path, monkeypatch):
    """Actual old rejection seal/Git/SQLite; driver eligibility separately tested."""
    from agentmemeval.path1_v1.operations import e0_service_cli as cli
    from agentmemeval.path1_v1.operations import service_supersession as ss
    from path1_v1.unit.test_service_supersession import (
        _check,
        _child_record,
        _fixture,
        _save_record,
    )

    data = _fixture(tmp_path, monkeypatch)
    original = data.store.read_registry()
    provenance = _check(data).payload()
    first = _child_record(data, data.child, provenance)
    _save_record(data, first)
    issued = {}

    def eligibility(store, intent):
        assert store.directory == data.store.directory
        return issued[intent]

    monkeypatch.setattr(retry, "resolve_service_retry", eligibility)
    previous = data.child
    children = []
    for attempt in (2, 3):
        issued[previous.request.intent_id] = retry.ServiceRetryEligibility(
            data.store.campaign_id,
            str(data.store.directory),
            previous.request.intent_id,
            content_digest(previous.contract),
            str(attempt) * 64,
            "e" * 64,
            "f" * 64,
            12040,
            _issuer=retry._ISSUER,
        )
        root = data.run.parent / ("retry_" + str(attempt))
        relative = data.config_path.relative_to(data.repo).as_posix()
        prepared = cli.prepare(
            data.repo,
            root,
            data.request.campaign_id,
            relative,
            attempt=attempt,
            prior_intent_id=previous.request.intent_id,
        )
        _, _, request = cli.read_prepared(root, prepared["descriptor_sha256"])
        contract, probe = er._json(root / "contract.json"), er._json(root / "probe_request.json")
        child = SimpleNamespace(root=root, request=request, contract=contract, probe=probe)
        child.worker = SimpleNamespace(
            contract=contract, request=probe, intent_id=request.intent_id
        )
        with data.store.lock():
            assert data.executor._validate_e0_retry(child.worker) is None
        record = _child_record(data, child, None)
        _save_record(data, record)
        with data.store.lock():
            snapshot = data.store.read_registry()
            data.store.claim_run(
                expected_generation=snapshot.generation,
                run_id=request.intent_id,
                intent_id=request.intent_id,
                run_directory=request.spec.work_dir,
                metadata={
                    "purpose": "slurm_job_container",
                    "request_sha256": request.request_sha256,
                },
            )
            # Most important cross-module regression: previous policy provenance
            # stays valid after the real child reservation, with no recursion.
            ledger = data.executor._ledger(data.store.read_registry())
            assert request.intent_id in ledger["intents"]
            assert data.executor._validate_e0_retry(child.worker) is None
        ss.validate_stored_supersession(first, data.store)
        children.append(child)
        previous = child
    assert len(data.store.read_registry().claims) == len(original.claims) + 2
    assert (
        data.store.read_registry().registry[NAMESPACE]["intents"][data.request.intent_id]
        == data.record
    )
    # A different source/seed intent may not consume attempt2 or the same seal.
    second = copy.deepcopy(children[0].contract)
    second["seed_namespace"] += "_different"
    _, probe = store_record(tmp_path / "second_child", second, children[0].probe)
    before = data.store.read_registry()
    with pytest.raises(SubmissionDisabled):
        retry.validate_service_retry(data.store, second, probe)
    assert data.store.read_registry() == before
