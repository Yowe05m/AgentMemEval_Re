"""Offline submission/content boundary; no model is loaded or runtime admitted."""
from __future__ import annotations

import json
import os
import sys

import pytest

from agentmemeval.path1_v1.operations import contracts, model_verification, worker
from path1_v1.unit.test_intent_executor import make_executor_fixture, submit
from path1_v1.unit.test_operations_worker import make_worker_fixture, write_json


def count_full_hashes(monkeypatch):
    calls = []
    original = contracts._model_identity

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(contracts, "_model_identity", counted)
    monkeypatch.setattr(model_verification, "_model_identity", counted)
    return calls


def test_e1_request_reuse_hashes_once_and_preserves_request_claim_and_identity(
    tmp_path, monkeypatch,
):
    data = make_worker_fixture(tmp_path)
    before = data["store"].read_registry()
    calls = count_full_hashes(monkeypatch)
    first = worker.validate_worker_request(data["request_path"])
    second = worker.validate_worker_request(
        data["request_path"], model_verification=first.model_verification,
    )
    assert len(calls) == 1
    assert first == second and first.model_verification is second.model_verification
    assert first.contract["runtime_identity"] == data["identity"]
    assert data["store"].read_registry() == before
    assert not data["output"].exists()
    # Omitting the capability means a fresh caller; there is no process-global cache.
    third = worker.validate_worker_request(data["request_path"])
    assert len(calls) == 2 and third == first
    assert third.model_verification is not first.model_verification


@pytest.mark.parametrize("change", [
    "weights", "extra", "manifest", "source",
    pytest.param("mtime_restored", marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux ctime/change semantics")),
])
def test_reuse_detects_actual_input_changes_without_fallback_hash(tmp_path, monkeypatch, change):
    data = make_worker_fixture(tmp_path)
    first = worker.validate_worker_request(data["request_path"])
    if change in {"weights", "mtime_restored"}:
        path = data["model"] / "weights.bin"
        stat = path.stat()
        raw = path.read_bytes()
        path.write_bytes(b"!" + raw[1:])
        if change == "mtime_restored":
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "extra":
        (data["model"] / "unlisted.bin").write_bytes(b"unlisted")
    elif change == "manifest":
        data["manifest"].write_bytes(data["manifest"].read_bytes() + b"\n")
    else:
        (data["repo"] / "src/core.py").write_bytes(b"# changed source\n")
    calls = count_full_hashes(monkeypatch)
    with pytest.raises((model_verification.ModelVerificationError, contracts.ContractError)):
        worker.validate_worker_request(data["request_path"],
                                       model_verification=first.model_verification)
    assert calls == [] and not data["output"].exists()
    assert data["store"].read_registry().claims[0].status == "reserved"


@pytest.mark.parametrize("change", ["expired", "pid", "request", "forged", "json"])
def test_reuse_rejects_wrong_lifetime_process_request_or_object(tmp_path, monkeypatch, change):
    data = make_worker_fixture(tmp_path)
    first = worker.validate_worker_request(data["request_path"])
    capability = first.model_verification
    if change == "expired":
        issued = model_verification.model_verification_diagnostic(capability)["verified_monotonic"]
        monkeypatch.setattr(model_verification.time, "monotonic", lambda: issued + 120.001)
    elif change == "pid":
        pid = os.getpid()
        monkeypatch.setattr(model_verification.os, "getpid", lambda: pid + 1)
    elif change == "request":
        request = dict(data["request"], artifact_root=str(data["repo"]))
        request["request_sha256"] = contracts.content_digest(
            {key: value for key, value in request.items() if key != "request_sha256"})
        write_json(data["request_path"], request)
    elif change == "forged":
        capability = model_verification.VerifiedModelFiles()
    else:
        capability = model_verification.model_verification_diagnostic(capability)
    calls = count_full_hashes(monkeypatch)
    with pytest.raises(model_verification.ModelVerificationError):
        worker.validate_worker_request(data["request_path"], model_verification=capability)
    assert calls == []
    assert data["store"].read_registry().claims[0].status == "reserved"


@pytest.mark.parametrize("phase", ["E2", "E3"])
def test_scientific_phases_keep_original_full_hash_path_and_reject_capability(
    tmp_path, monkeypatch, phase,
):
    data = make_worker_fixture(tmp_path, phase=phase)
    calls = count_full_hashes(monkeypatch)
    validated = worker.validate_worker_request(data["request_path"])
    assert len(calls) == 2 and validated.model_verification is None
    with pytest.raises(contracts.ContractError, match="E1 only"):
        worker.validate_worker_request(data["request_path"],
                                       model_verification=model_verification.VerifiedModelFiles())
    assert len(calls) == 2


def test_real_submission_handshake_reuses_one_hash_without_runtime_admission(tmp_path, monkeypatch):
    data = make_executor_fixture(tmp_path)
    calls = count_full_hashes(monkeypatch)
    result = submit(data, timeout=20)
    assert result.status == "accepted" and result.job_id == "551"
    assert len(calls) == 1
    assert data.queries == 2 and data.admissions == 2
    assert data.worker["store"].read_registry().claims[0].status == "reserved"
    assert not data.worker["output"].exists()
    effect = json.loads((tmp_path / "external_effect.json").read_bytes())
    assert effect["script_sha256"] == data.request.spec.script_sha256
    before = (tmp_path / "external_effect.json").read_bytes()
    assert submit(data, timeout=20) == result
    assert (tmp_path / "external_effect.json").read_bytes() == before
    assert len(calls) == 1


def test_pre_go_detects_model_change_without_rehash_or_external_effect(tmp_path, monkeypatch):
    data = make_executor_fixture(tmp_path)
    calls = count_full_hashes(monkeypatch)

    def change_after_first_admission(index):
        if index == 1:
            path = data.worker["model"] / "weights.bin"
            path.write_bytes(b"?" + path.read_bytes()[1:])

    data.on_admit = change_after_first_admission
    with pytest.raises(model_verification.ModelVerificationError):
        submit(data, timeout=20)
    record = data.executor.inspect(data.request)
    assert record["status"] == "submit_unknown"
    assert not (tmp_path / "external_effect.json").exists()
    assert len(calls) == 1
    assert submit(data, timeout=20).status == "unknown"
    assert len(calls) == 1 and data.admissions == 1


def test_missing_runtime_verifier_remains_closed_and_allocation_does_fresh_hash(
    tmp_path, monkeypatch,
):
    data = make_worker_fixture(tmp_path)
    first = worker.validate_worker_request(data["request_path"])
    assert first.model_verification is not None
    calls = count_full_hashes(monkeypatch)
    with pytest.raises(worker.RuntimeAdmissionRequired, match="no trusted"):
        worker.run_worker(data["request_path"])
    assert len(calls) == 1
    assert not data["output"].exists()
    assert data["store"].read_registry().claims[0].status == "reserved"
