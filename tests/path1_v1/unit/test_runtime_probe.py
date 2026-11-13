"""Offline contract tests; no test fixture represents Linux/GPU admission."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import runtime_probe as probe
from agentmemeval.path1_v1.operations import worker
from agentmemeval.path1_v1.operations.contracts import canonical_json
from agentmemeval.path1_v1.operations.process_ownership import (
    EndpointObservation,
    ProcessBinding,
    ProcessIdentity,
    ProcessSnapshot,
)


def source():
    return json.loads(files("agentmemeval.path1_v1").joinpath(
        "assets/path1_source_contract.json").read_text())


def action():
    return {"action_type": "fold", "amount": None, "confidence": 0.5,
            "reason_summary": "fixture", "intent": "avoid"}


def response(payload, count):
    if count + payload["max_tokens"] > 65536:
        result = {"error": {"code": "context_length_exceeded"}}
        status = 400
    else:
        content = action() if payload["response_format"]["json_schema"]["name"] == (
            "action_decision") else {"operation": "keep", "patches": [], "fact_assessments": [],
                                     "calibration_note": "", "self_check": ""}
        prompt = payload["messages"][0]["content"]
        if prompt.startswith("Return one valid technical patch: "):
            content = json.loads(prompt.split(": ", 1)[1])
        result = {"model": payload["model"], "usage": {"prompt_tokens": count,
                  "completion_tokens": 1, "total_tokens": count + 1},
                  "choices": [{"finish_reason": "stop", "message": {
                      "content": json.dumps(content)}}]}
        status = 200
    raw = canonical_json(result)
    return {"status": status, "body_utf8": raw.decode(),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "request_sha256": hashlib.sha256(canonical_json(payload)).hexdigest()}


def payload():
    return {"model": "qwen/qwen3.5-9b", "messages": [{"role": "user", "content": "fixture"}],
            "max_tokens": 256, "seed": 2**62 + 19, "temperature": 0.0,
            "response_format": probe._response_format("json_schema")}


@pytest.mark.parametrize("endpoint", ["https://127.0.0.1:8000/v1", "http://localhost:8000/v1",
    "http://example.org:8000/v1", "http://x:y@127.0.0.1:8000/v1", "http://127.0.0.1/v1",
    "http://127.0.0.1:8000/v1?token=x", "http://127.0.0.1:8000/v2"])
def test_endpoint_requires_explicit_local_origin(endpoint):
    with pytest.raises(probe.RuntimeProbeError):
        probe._endpoint(endpoint)


def test_real_loopback_http_preserves_actual_high_seed_and_raw_response(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            raw = response(body, 3)["body_utf8"].encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://invalid.example:9999")
    try:
        requested = payload()
        observed = probe.http_exchange(f"http://127.0.0.1:{server.server_port}/v1", requested)
        result = probe.verify_http_observation(requested, observed, 3)
        assert seen == [requested]
        assert result["client_seed"] == 2**62 + 19
        assert result["sampler_execution_verified"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("change", ["boolean_usage", "missing_usage", "total", "length",
    "wrong_model", "nonfinite", "duplicate", "extra", "budget", "raw_digest"])
def test_http_acceptance_rejects_incomplete_or_forged_raw_evidence(change):
    requested = payload()
    observed = response(requested, 4)
    raw = json.loads(observed["body_utf8"])
    if change == "boolean_usage":
        raw["usage"]["completion_tokens"] = True
    elif change == "missing_usage":
        del raw["usage"]
    elif change == "total":
        raw["usage"]["total_tokens"] += 1
    elif change == "length":
        raw["choices"][0]["finish_reason"] = "length"
    elif change == "wrong_model":
        raw["model"] = "different"
    elif change == "nonfinite":
        raw["choices"][0]["message"]["content"] = '{"confidence": NaN}'
    elif change == "duplicate":
        raw["choices"][0]["message"]["content"] = '{"action_type":"fold","action_type":"call"}'
    elif change == "extra":
        content = action()
        content["verified"] = True
        raw["choices"][0]["message"]["content"] = json.dumps(content)
    elif change == "budget":
        raw["usage"].update(completion_tokens=257, total_tokens=261)
    observed["body_utf8"] = canonical_json(raw).decode()
    observed["body_sha256"] = hashlib.sha256(observed["body_utf8"].encode()).hexdigest()
    if change == "raw_digest":
        observed["body_sha256"] = "0" * 64
    with pytest.raises(probe.RuntimeProbeError):
        probe.verify_http_observation(requested, observed, 4)


def test_local_model_observation_checks_real_bytes_not_revision_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_bytes(b"offline synthetic bytes")
    source_model = source()["model"]
    manifest = {"schema_version": 1, "model": source_model["name"],
                "revision": source_model["revision"],
                "tokenizer_revision": source_model["tokenizer_revision"],
                "chat_template_sha256": "1" * 64,
                "files": [{"path": p.name, "size": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(model.iterdir())]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    identity = probe._model_identity(manifest_path, model)
    request = SimpleNamespace(request={"model_root": str(model),
                                       "model_manifest_path": str(manifest_path)},
                              contract={"runtime_identity": {"model": identity}})
    observed, metadata = probe.observe_model(request)
    assert observed == identity and len(metadata) == 4
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model file content differs"):
        probe.observe_model(request)


class OfflineTokenizer:
    """Only deterministic test sizing; not a deployed tokenizer receipt."""
    def get_chat_template(self):
        return "offline-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False}
        return list(range(len(messages[0]["content"].split()) + 4))


def offline_collector(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    repository = tmp_path / "repo"
    repository.mkdir()
    request = probe.ValidatedProbeRequest(
        request={"model_root": str(model), "repository_root": str(repository)},
        intent_id="fixture", request_sha256="1" * 64, output_directory=tmp_path / "e0",
        contract={"phase": "E0", "campaign_id": "fixture", "source_contract": source(),
                  "runtime_identity": {"operations_manifest_sha256": "2" * 64,
                                       "core_files": [], "operations_files": []}})
    argv = ("python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
            "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "65536",
            "--max-num-seqs", "1", "--no-enable-prefix-caching", "--served-model-name",
            "qwen/qwen3.5-9b")
    snapshot = ProcessSnapshot(ProcessBinding("fixture", "fixture", "123",
        probe.content_digest(request.contract["runtime_identity"]),
        "model_service", "offline"), ProcessIdentity("offline", "boot", 1000, 42, "1234", 42,
        42, ("0::/job_123",), "123", "batch"), argv, "S")
    owner = SimpleNamespace(verify_live=lambda: snapshot)
    monkeypatch.setattr(probe, "observe_host", lambda: {"hostname": "offline", "job_id": "123",
        "euid": 1000, "boot_id": "boot", "allocation_end_utc": (
            datetime.now(UTC) + timedelta(hours=2)).isoformat()})
    monkeypatch.setattr(probe, "observe_gpu", lambda: {"offline_fixture": True})
    monkeypatch.setattr(probe, "observe_dependencies", lambda: {"offline_fixture": True})
    monkeypatch.setattr(probe, "observe_model", lambda _: ({"manifest": {
        "chat_template_sha256": hashlib.sha256(b"offline-template").hexdigest()}}, []))
    monkeypatch.setattr(probe, "_load_tokenizer", lambda _: OfflineTokenizer())
    from agentmemeval.path1_v1.operations import intent_executor

    monkeypatch.setattr(intent_executor, "verify_loaded_package", lambda *_: None)
    calls = []

    def http(_endpoint, requested):
        calls.append(requested)
        count = len(probe._token_ids(OfflineTokenizer(), requested["messages"]))
        return response(requested, count)

    monkeypatch.setattr(probe, "http_exchange", http)
    return request, owner, calls


def test_collector_records_raw_cases_without_backend_admission(tmp_path, monkeypatch):
    request, owner, calls = offline_collector(tmp_path, monkeypatch)
    output = request.output_directory / "observations"
    report = probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1", output)
    assert report["runtime_admission"] is False
    assert report["status"] == "observations_complete_backend_execution_pending"
    assert len(calls) == 6
    assert [x["max_tokens"] for x in calls] == [256, 8192, 256, 4096, 256, 256]
    assert all(type(x["seed"]) is int and x["seed"] >= 2**62 for x in calls)
    assert set(probe.PENDING_EXECUTION_EVIDENCE) <= set(report["pending"])
    assert (output / "over_context_response.json").is_file()
    with pytest.raises(FileExistsError):
        probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1", output)


def test_strict_verifier_never_accepts_successful_observation_as_admission(tmp_path, monkeypatch):
    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    verifier = probe.StrictRuntimeVerifier(owner, "http://127.0.0.1:8000/v1",
                                           request.output_directory / "observations")
    with pytest.raises(probe.RuntimeProbeError, match="backend execution evidence pending"):
        verifier.verify_in_job(request)
    assert not (tmp_path / "payload").exists()


def test_pid_change_stops_before_any_http_and_retains_failure_receipt(tmp_path, monkeypatch):
    request, owner, calls = offline_collector(tmp_path, monkeypatch)
    original = owner.verify_live()
    count = 0

    def live():
        nonlocal count
        count += 1
        return original if count == 1 else replace(original, identity=replace(original.identity,
                                                                              pid=999))

    owner.verify_live = live
    output = request.output_directory / "observations"
    with pytest.raises(probe.RuntimeProbeError, match="service changed"):
        probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1", output)
    assert not calls
    assert json.loads((output / "probe_receipt.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("flag,value", [("--dtype", "float16"), ("--max-num-seqs", "2"),
    ("--max-model-len", "4096"), ("--served-model-name", "different")])
def test_service_settings_must_be_explicit_and_equal_frozen_contract(
    tmp_path, monkeypatch, flag, value,
):
    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    argv = list(owner.verify_live().cmdline)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(probe.RuntimeProbeError, match="service arguments"):
        probe.check_service_arguments(argv, request)


def test_producer_seal_detects_deleted_or_changed_bytes_and_rejects_preexisting_terminal(tmp_path):
    (tmp_path / "raw.jsonl").write_bytes(b"one\n")
    seal = worker._produced_artifacts(tmp_path)
    assert seal == [{"path": "raw.jsonl", "size": 4,
                     "sha256": hashlib.sha256(b"one\n").hexdigest()}]
    (tmp_path / "raw.jsonl").write_bytes(b"two\n")
    assert worker._produced_artifacts(tmp_path) != seal
    (tmp_path / "worker_terminal.json").write_text("{}")
    with pytest.raises(worker.RuntimeAdmissionRequired, match="already exists"):
        worker._produced_artifacts(tmp_path)


def test_producer_seal_rejects_write_during_hash(tmp_path, monkeypatch):
    target = tmp_path / "raw"
    target.write_bytes(b"original")
    original = worker._hash_file

    def changing(path):
        result = original(path)
        path.write_bytes(b"changed length")
        return result

    monkeypatch.setattr(worker, "_hash_file", changing)
    with pytest.raises(worker.RuntimeAdmissionRequired, match="changed during"):
        worker._produced_artifacts(tmp_path)


def test_cli_cannot_select_service_without_explicit_probe(tmp_path):
    assert worker.main(["--request", str(tmp_path / "unused"),
                        "--service-python", str(Path("python").absolute())]) == 2


def test_explicit_e0_cli_success_does_not_call_research_worker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "run_service_probe", lambda *args: calls.append(args)
                        or {"runtime_admission": False})
    monkeypatch.setattr(worker, "run_worker", lambda *_: pytest.fail("research was started"))
    assert worker.main(["--request", str(tmp_path / "e0_request.json"),
                        "--runtime-probe-output", str(tmp_path / "e0"),
                        "--service-python", str(tmp_path / "python"),
                        "--service-port", "8000"]) == 0
    assert len(calls) == 1


def test_worker_terminal_seal_and_identity_are_bound_into_completed_claim(tmp_path, monkeypatch):
    from path1_v1.unit.test_operations_worker import OfflineVerifier, make_worker_fixture

    fixture = make_worker_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(worker, "_execute", lambda *_: {"completed_units": []})
    terminal = worker.run_worker(fixture["request_path"],
                                 runtime_verifier=OfflineVerifier(tmp_path))
    assert terminal["campaign_id"] == fixture["contract"]["campaign_id"]
    assert terminal["slurm_job_id"] == "12345"
    claim = fixture["store"].read_registry().claims[0]
    assert claim.terminal_receipt == terminal
    records = terminal["produced_artifacts"]
    assert records and "worker_terminal.json" not in [row["path"] for row in records]
    for record in records:
        raw = (fixture["output"] / record["path"]).read_bytes()
        assert len(raw) == record["size"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]


def test_service_owner_is_closed_when_probe_fails_before_research(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations.process_ownership import CleanupReceipt, OwnedProcess

    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    request.contract["job"] = {"kind": "service_probe"}
    request.request["schema_version"] = 1
    observed = owner.verify_live()
    closed = []
    owner.interrupt_guard = lambda: nullcontext()
    owner.verify_endpoint = lambda _: None
    cleanup = CleanupReceipt(observed.binding, observed.identity, 0, True, False,
                             True, True, (), ())

    def close(**kwargs):
        if not closed:
            closed.append(kwargs)
        return cleanup

    owner.close = close

    @contextmanager
    def managed(*_, **kwargs):
        try:
            yield owner
        finally:
            receipt = owner.close(term_grace_seconds=kwargs["term_grace_seconds"])
            kwargs["receipt_sink"](owner, receipt)

    monkeypatch.setattr(OwnedProcess, "managed", managed)
    monkeypatch.setattr(probe, "validate_probe_request", lambda _, **kwargs: request)
    monkeypatch.setattr(probe, "build_opener", lambda *_: SimpleNamespace(open=lambda *_, **__: (
        nullcontext(SimpleNamespace(code=200)))))

    def fail(*_):
        raise probe.RuntimeProbeError("offline probe failure")

    monkeypatch.setattr(probe, "collect_e0_probe", fail)
    executable = tmp_path / "offline_python"
    executable.write_bytes(b"not executed")
    with pytest.raises(probe.RuntimeProbeError, match="offline probe failure"):
        probe.run_service_probe(tmp_path / "unused_request", request.output_directory,
                                executable, 8000)
    assert closed == [{"term_grace_seconds": 120}]
    assert (request.output_directory / "service_cleanup.json").is_file()
    assert not (tmp_path / "payload").exists()


def test_probe_rejects_http_when_only_cleanup_time_remains(tmp_path, monkeypatch):
    request, owner, calls = offline_collector(tmp_path, monkeypatch)
    old_host = probe.observe_host

    def late():
        host = old_host()
        host["allocation_end_utc"] = (datetime.now(UTC) + timedelta(seconds=1100)).isoformat()
        return host

    monkeypatch.setattr(probe, "observe_host", late)
    with pytest.raises(probe.RuntimeProbeError, match="insufficient request"):
        probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1",
                               request.output_directory / "observations")
    assert not calls


def test_service_creation_interrupt_sink_persists_before_owner_is_yielded(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations.process_ownership import (
        CleanupReceipt,
        OwnedProcess,
        ProcessInterrupted,
    )

    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    request.contract["job"] = {"kind": "service_probe"}
    request.request["schema_version"] = 1
    observed = owner.verify_live()
    cleanup = CleanupReceipt(observed.binding, observed.identity, -15, True, False,
                             True, True, (), ())

    @contextmanager
    def interrupted(*_, **kwargs):
        kwargs["receipt_sink"](owner, cleanup)
        raise ProcessInterrupted(15)
        yield  # The lifecycle never obtains owner; only the sink has it.

    monkeypatch.setattr(OwnedProcess, "managed", interrupted)
    monkeypatch.setattr(probe, "validate_probe_request", lambda _, **kwargs: request)
    executable = tmp_path / "offline_python"
    executable.write_bytes(b"not executed")
    with pytest.raises(ProcessInterrupted):
        probe.run_service_probe(tmp_path / "unused", request.output_directory, executable, 8000)
    saved = json.loads((request.output_directory / "service_cleanup.json").read_text())
    assert saved["group_cleanup_verified"] and saved["returncode"] == -15
    failure = json.loads((request.output_directory / "service_failure.json").read_text())
    assert failure["failure_type"] == "ProcessInterrupted"
    assert failure["runtime_admission"] is False


def test_process_scheduler_state_can_change_without_changing_identity(tmp_path, monkeypatch):
    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    original = probe._service(owner, request, probe.observe_host())
    changed = dict(original, state="R")
    assert probe._same_service(original, changed)
    changed["identity"] = dict(changed["identity"], pid=999)
    assert not probe._same_service(original, changed)


def test_service_binding_covers_complete_identity_not_only_operations(tmp_path, monkeypatch):
    request, owner, _ = offline_collector(tmp_path, monkeypatch)
    original = owner.verify_live()
    assert original.binding.source_sha256 == probe.content_digest(
        request.contract["runtime_identity"])
    owner.verify_live = lambda: replace(original, binding=replace(original.binding,
        source_sha256=request.contract["runtime_identity"]["operations_manifest_sha256"]))
    with pytest.raises(probe.RuntimeProbeError, match="another contract/job/source"):
        probe._service(owner, request, probe.observe_host())


def test_endpoint_rechecked_after_response_and_loss_keeps_partial_evidence(tmp_path, monkeypatch):
    request, owner, calls = offline_collector(tmp_path, monkeypatch)
    counter = 0

    def endpoint(value):
        nonlocal counter
        counter += 1
        if counter == 3:
            raise probe.RuntimeProbeError("endpoint lost after HTTP")
        return EndpointObservation(owner.verify_live(), value, "net_fixture", (1,), ((42, "1", 1),))

    owner.verify_endpoint = endpoint
    output = request.output_directory / "observations"
    with pytest.raises(probe.RuntimeProbeError, match="endpoint lost"):
        probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1", output)
    assert len(calls) == 1 and counter == 3
    assert (output / "action_response.json").is_file()
    assert json.loads((output / "probe_receipt.json").read_text())["status"] == "incomplete"


def make_probe_fixture(tmp_path):
    from path1_v1.unit.test_operations_contracts import _contract, evidence

    data = evidence.__wrapped__(tmp_path)
    contract = _contract(data, "E0", "service_probe")
    contract["job"]["output_relpath"] = "test_campaign/E0/service/attempt_01"
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(canonical_json(contract))
    root = tmp_path / "output"
    root.mkdir()
    request = {"schema_version": 1, "kind": "path1_e0_probe_request",
               "intent_id": probe.canonical_intent_id(contract),
               "contract_path": str(contract_path),
               "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
               "repository_root": str(data["repo"]), "model_root": str(data["model"]),
               "model_manifest_path": str(data["manifest"]), "output_root": str(root)}
    request["request_sha256"] = probe.content_digest(request)
    path = tmp_path / "probe_request.json"
    path.write_bytes(canonical_json(request))
    return data, contract, request, path


def test_real_e0_git_model_request_reaches_host_gate_without_claim_or_research_seeds(
    tmp_path, monkeypatch,
):
    _, contract, request, path = make_probe_fixture(tmp_path)
    validated = probe.validate_probe_request(path)
    assert contract["phase"] == "E0" and contract["cohort"]["formation_count"] == 0
    assert contract["cohort"]["seed_inventory_sha256"] is None
    assert "claim_path" not in request and "seed_inventory_path" not in request
    touched = []

    def host():
        touched.append(True)
        raise probe.RuntimeProbeError("actual_host_gate")

    monkeypatch.setattr(probe, "observe_host", host)
    with pytest.raises(probe.RuntimeProbeError, match="actual_host_gate"):
        probe.run_service_probe(path, validated.output_directory, Path("unused"), 8000)
    assert touched == [True] and not validated.output_directory.exists()


@pytest.mark.parametrize("change", ["claim", "seed_inventory", "verified", "E1", "model", "source"])
def test_probe_request_rejects_research_impersonation_and_changed_files(tmp_path, change):
    from path1_v1.unit.test_operations_contracts import _contract

    data, _, request, path = make_probe_fixture(tmp_path)
    if change in {"claim", "seed_inventory", "verified"}:
        request[change] = True
    elif change == "E1":
        contract = _contract(data, "E1")
        Path(request["contract_path"]).write_bytes(canonical_json(contract))
        request["contract_sha256"] = hashlib.sha256(canonical_json(contract)).hexdigest()
        request["intent_id"] = probe.canonical_intent_id(contract)
    elif change == "model":
        (data["model"] / "weights.bin").write_bytes(b"changed")
    else:
        (data["repo"] / "src/core.py").write_text("# changed\n")
    request["request_sha256"] = probe.content_digest(
        {key: value for key, value in request.items() if key != "request_sha256"})
    path.write_bytes(canonical_json(request))
    with pytest.raises((probe.RuntimeProbeError, ValueError)):
        probe.validate_probe_request(path)
