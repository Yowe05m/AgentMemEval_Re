"""Offline E1 preparation/real helper lifecycle fixtures, never GPU admission."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e1_launcher, e1_prepare, worker
from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.e1_preflight import E1PreSubmitVerifier
from agentmemeval.path1_v1.operations.intent_executor import VerifiedCommand
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_intent_executor import make_executor_fixture, submit
from path1_v1.unit.test_operations_contracts import _git
from path1_v1.unit.test_operations_worker import OfflineVerifier, write_json


def make_prepared_fixture(root, *, scope=("cluster", "account", "owner")):
    fixture = make_executor_fixture(root)
    data = fixture.worker
    modules = ("e1_prepare", "e1_launcher", "e1_preflight", "interpreters",
               "intent_executor", "submission_process", "user_capacity")
    for module in modules:
        relative = "src/agentmemeval/path1_v1/operations/" + module + ".py"
        (data["repo"] / relative).write_text("# offline source fixture\n", encoding="utf-8")
        data["build_kwargs"]["operations_paths"].append(relative)
    _git(data["repo"], "add", ".")
    _git(data["repo"], "commit", "-qm", "offline complete E1 entry closure")
    contract = data["contract"]
    contract["runtime_identity"] = worker.build_runtime_identity(
        data["repo"], **data["build_kwargs"])
    intent = canonical_intent_id(contract)
    store = CampaignStore(root / "new_worker_control", contract["campaign_id"])
    with store.lock():
        initial = store.initialize(registry={}, state={})
        claim = store.claim_run(expected_generation=initial.generation,
            run_id="prepared_fixture", intent_id=intent, run_directory=data["output"])
    claim_data = dict(data["claim"], intent_id=intent, control_directory=str(store.directory),
                      run_id=claim.run_id, token=claim.token)
    request = data["request"]
    request["intent_id"] = intent
    request["contract_sha256"] = write_json(Path(request["contract_path"]), contract)
    request["claim_sha256"] = write_json(Path(request["claim_path"]), claim_data)
    request["request_sha256"] = content_digest(
        {k: v for k, v in request.items() if k != "request_sha256"})
    write_json(data["request_path"], request)
    data.update(store=store, claim=claim_data)
    python = Path(sys.executable).absolute()
    proof = verify_python_interpreter(python, file_sha256(python))
    prepared = e1_prepare.prepare(worker_request_path=data["request_path"],
        prepared_directory=root / "prepared", python_executable=python,
        python_sha256=file_sha256(python), python_prefix=proof["prefix"],
        cluster=scope[0], account=scope[1], user=scope[2], partition="gpu",
        work_dir="/runs/e1", script_path="/code/e1.sh")
    validated = e1_prepare.read_prepared(prepared)
    fixture.prepared = prepared
    fixture.request = validated.submission
    fixture.script_path = Path(validated.descriptor["script_file"])
    fixture.executor.launch_files = lambda _: e1_prepare.E1LaunchFiles(prepared)
    fixture.executor.fixture_paths = {
        fixture.request.spec.work_dir: data["output"].parent,
        fixture.request.spec.script_path: fixture.script_path}
    partition = root / "scontrol.py"
    partition.write_text("print('PartitionName=gpu State=UP MaxTime=UNLIMITED')\n",
                         encoding="utf-8")
    evidence = root / "preflight"
    evidence.mkdir()
    base_python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    fixture.producer = E1PreSubmitVerifier(evidence,
        partition_command=VerifiedCommand((str(base_python), "-I", str(partition)),
            ((str(base_python), file_sha256(base_python)),
             (str(partition), file_sha256(partition)))), fixture_mode=True,
        clock=lambda: datetime.now(UTC))
    fixture.executor.pre_submit_verifier = fixture.producer
    fixture.executor.clock = lambda: datetime.now(UTC)
    old_reader = fixture.executor.account_reader
    def live_fixture_reader():
        fixture.now = datetime.now(UTC)
        return old_reader()
    fixture.executor.account_reader = live_fixture_reader
    fixture.evidence = evidence
    fixture.partition = partition
    return fixture


def reseal(path, value):
    value["descriptor_sha256"] = content_digest(
        {k: v for k, v in value.items() if k != "descriptor_sha256"})
    path.write_bytes(canonical_json(value))


def test_prepare_exact_resources_and_real_subprocess_two_gates_no_repeat(tmp_path):
    fixture = make_prepared_fixture(tmp_path)
    result = submit(fixture, timeout=45)
    assert result.status == "accepted" and result.job_id == "551" and result.submitter_finished
    effect = json.loads((tmp_path / "external_effect.json").read_text(encoding="utf-8"))
    assert "--cpus-per-task=8" in effect["argv"]
    assert "--gpus=1" in effect["argv"]
    assert not any(arg.startswith(("--mem", "--deadline")) for arg in effect["argv"])
    assert effect["script_sha256"] == fixture.request.spec.script_sha256
    observations = list(fixture.evidence.glob("*.json"))
    assert len(observations) == 2
    for path in observations:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["partition_stdout"] and value["runtime_admission"] is False
        assert value["actual_memory_verified"] is False
    before = fixture.executor.store.read_registry()
    assert submit(fixture, timeout=45) == result
    assert fixture.executor.store.read_registry() == before
    assert fixture.worker["store"].read_registry().claims[0].status == "reserved"


@pytest.mark.parametrize("problem", ["ram", "walltime", "bool", "extra", "venv", "worker"])
def test_prepared_rejects_rebound_bytes_even_when_descriptor_resealed(tmp_path, problem):
    fixture = make_prepared_fixture(tmp_path)
    value = json.loads(fixture.prepared.read_text(encoding="utf-8"))
    if problem == "ram":
        value["resource_profile"]["memory_mib"] = 1
    elif problem == "walltime":
        value["submission"]["spec"]["walltime_seconds"] = 7200
    elif problem == "bool":
        value["schema_version"] = True
    elif problem == "extra":
        value["runtime_admission"] = True
    elif problem == "venv":
        value["python_identity"]["prefix"] += "_wrong"
    else:
        fixture.worker["request_path"].write_bytes(b"{}")
    reseal(fixture.prepared, value)
    with pytest.raises((ValueError, RuntimeError)):
        e1_prepare.read_prepared(fixture.prepared)
    assert not (tmp_path / "external_effect.json").exists()


def test_pre_go_tamper_stays_unknown_without_external_effect_or_retry(tmp_path):
    fixture = make_prepared_fixture(tmp_path)
    original = fixture.producer
    calls = []
    def observe(request, prepared):
        result = original(request, prepared)
        calls.append(result)
        if len(calls) == 1:
            fixture.script_path.write_bytes(b"tampered\n")
        return result
    fixture.executor.pre_submit_verifier = observe
    # The parent's fresh gate rejects changed bytes after the durable ready
    # receipt, and before go; the consumed submission remains unknown.
    with pytest.raises(ValueError, match="launcher bytes changed"):
        submit(fixture, timeout=45)
    assert fixture.executor.lookup(fixture.request, timeout_seconds=45).status == "unknown"
    record = fixture.executor.inspect(fixture.request)
    invocation = Path(record["invocation_dir"])
    assert record["ready"]["submitter"]["start_marker"]
    assert (invocation / "ready.json").is_file()
    assert not (invocation / "go.json").exists()
    assert not (tmp_path / "external_effect.json").exists()
    before = fixture.executor.store.read_registry()
    assert submit(fixture, timeout=45).status == "unknown"
    assert fixture.executor.store.read_registry() == before


def test_executor_user_scope_uses_real_producer_and_helper_without_e0_relabel(
    tmp_path, monkeypatch,
):
    from agentmemeval.path1_v1.operations.intent_executor import DurableIntentExecutor
    from path1_v1.unit.test_user_capacity import make_user_capacity_fixture

    root = tmp_path / "entry"
    root.mkdir()
    fixture = make_prepared_fixture(root, scope=("n16r4", "scze856", "scze856"))
    case = make_user_capacity_fixture(tmp_path, monkeypatch)
    fixture.fake_path.write_text(fixture.fake_path.read_text(encoding="utf-8").replace(
        "551;cluster", "551;n16r4"), encoding="utf-8")
    old = fixture.executor
    command = replace(old.command, files=tuple(
        (name, file_sha256(Path(name))) for name, _ in old.command.files))
    # Typed raw site observations remain fixture-only, while subprocess/helper,
    # double admission and durable occupancy bookkeeping execute unmocked.
    case["reader"].clock = lambda: datetime.now(UTC)
    fixture.executor = DurableIntentExecutor(root / "userctl", cluster="n16r4",
        account="scze856", command=command, launch_files=old.launch_files,
        pre_submit_verifier=fixture.producer, account_reader=case["reader"],
        fixture_mode=True, fixture_paths=old.fixture_paths)
    result = submit(fixture, timeout=45)
    assert result.status == "accepted" and result.job_id == "551"
    record = fixture.executor.inspect(fixture.request)
    assert record["capacity"]["scope"] == "user_association"
    assert record["capacity"]["reserve"] == 2
    assert record["capacity"]["full_account_visibility"] is False
    assert "probe_request_path" not in record
    assert len(case["transport"].calls) == 9  # initial snapshot + two fresh readers
    assert fixture.worker["store"].read_registry().claims[0].status == "reserved"


@pytest.mark.parametrize("line", ["PartitionName=gpu State=DOWN MaxTime=UNLIMITED",
    "PartitionName=gpu State=UP MaxTime=01:00:00",
    "PartitionName=gpu State=UP MaxTime=UNLIMITED MaxCPUsPerNode=4"])
def test_real_partition_producer_negative_observations(tmp_path, line):
    fixture = make_prepared_fixture(tmp_path)
    fixture.partition.write_text("print(" + repr(line) + ")\n", encoding="utf-8")
    command = fixture.producer.command
    fixture.producer.command = replace(command, files=(
        command.files[0], (str(fixture.partition), file_sha256(fixture.partition))))
    with pytest.raises(SubmissionDisabled, match="partition"):
        fixture.producer(fixture.request, e1_prepare.read_prepared(fixture.prepared))
    assert not list(fixture.evidence.glob("*.json"))


def test_launcher_default_no_runtime_cannot_consume_claim(tmp_path, monkeypatch):
    fixture = make_prepared_fixture(tmp_path)
    # Synthetic source identity is not the loaded product: only bypass this
    # separately tested source-location gate, not RuntimeVerifier or reservation.
    monkeypatch.setattr(e1_launcher, "verify_loaded_package", lambda *_: None)
    before = fixture.worker["store"].read_registry()
    with pytest.raises(worker.RuntimeAdmissionRequired, match="bound version-2 runtime plan"):
        e1_launcher.run_prepared(fixture.prepared)
    assert fixture.worker["store"].read_registry() == before
    assert not fixture.worker["output"].exists()


def test_launcher_rejects_mixed_loaded_source_before_claim(tmp_path):
    fixture = make_prepared_fixture(tmp_path)
    before = fixture.worker["store"].read_registry()
    with pytest.raises(RuntimeError, match="another source tree|outside the contract closure"):
        e1_launcher.run_prepared(fixture.prepared)
    assert fixture.worker["store"].read_registry() == before


@pytest.mark.skipif(sys.platform != "linux", reason="actual Linux symlink venv entry")
def test_real_venv_entry_is_preserved_and_other_running_interpreter_rejected(tmp_path):
    fixture = make_prepared_fixture(tmp_path)
    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-I", "-B", "-m", "venv", "--without-pip",
                    str(environment)], check=True, timeout=30)
    entry = environment / "bin/python"
    assert entry.is_symlink()
    proof = verify_python_interpreter(entry, file_sha256(entry))
    second = e1_prepare.prepare(worker_request_path=fixture.worker["request_path"],
        prepared_directory=tmp_path / "other_prepared", python_executable=entry,
        python_sha256=file_sha256(entry), python_prefix=proof["prefix"],
        cluster="cluster", account="account", user="owner", partition="gpu",
        work_dir="/runs/e1", script_path="/code/e1.sh")
    result = e1_prepare.read_prepared(second)
    assert result.descriptor["python_identity"]["entry_path"] == str(entry)
    assert str(entry).encode() in Path(result.descriptor["script_file"]).read_bytes()
    assert result.descriptor["python_identity"]["prefix"] == str(environment)
    before = fixture.worker["store"].read_registry()
    with pytest.raises(ValueError, match="running E1 interpreter/venv"):
        e1_launcher.run_prepared(second)
    assert fixture.worker["store"].read_registry() == before


def test_launcher_offline_verifier_reaches_real_claim_lifecycle_once(tmp_path, monkeypatch):
    fixture = make_prepared_fixture(tmp_path)
    monkeypatch.setattr(e1_launcher, "verify_loaded_package", lambda *_: None)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    calls = []
    def execute(request, factory):
        assert request.contract["phase"] == "E1"
        assert fixture.worker["store"].read_registry().claims[0].status == "started"
        calls.append(request.intent_id)
        return {"completed_units": [{"offline_fixture": True}]}
    monkeypatch.setattr(worker, "_execute", execute)
    receipt = e1_launcher.run_prepared(fixture.prepared, runtime_verifier=OfflineVerifier(tmp_path))
    assert receipt["status"] == "completed"
    assert receipt["primary_analysis_admitted"] is False
    assert fixture.worker["store"].read_registry().claims[0].status == "completed"
    with pytest.raises(ValueError):
        e1_launcher.run_prepared(fixture.prepared, runtime_verifier=OfflineVerifier(tmp_path))
    assert len(calls) == 1


def test_typed_stage_one_user_scope_and_unresolved_occupancy(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations.user_capacity import validate_user_association_capacity
    from path1_v1.unit.test_user_capacity import (
        NOW,
        make_request,
        make_user_capacity_fixture,
        reservation,
    )
    root = tmp_path / "entry"
    root.mkdir()
    fixture = make_prepared_fixture(root, scope=("n16r4", "scze856", "scze856"))
    prepared = e1_prepare.read_prepared(fixture.prepared)
    case = make_user_capacity_fixture(tmp_path, monkeypatch)
    args = (case["snapshot"], fixture.request, case["ledger"], 1, NOW, "E1")
    result = validate_user_association_capacity(*args, validated_e1=prepared)
    assert result["reserve"] == 2 and result["full_account_visibility"] is False
    with pytest.raises(SubmissionDisabled, match="fixed prepared"):
        validate_user_association_capacity(*args)
    with pytest.raises(SubmissionDisabled, match="stage one"):
        validate_user_association_capacity(*args[:3], 2, NOW, "E1", validated_e1=prepared)
    for phase in ("E0", "E2", "E3"):
        with pytest.raises(SubmissionDisabled):
            validate_user_association_capacity(*args[:5], phase, validated_e1=prepared)
    previous = make_request(7, campaign=fixture.request.campaign_id)
    case["ledger"]["intents"][previous.intent_id] = reservation(previous)
    with pytest.raises(SubmissionDisabled, match="capacity"):
        validate_user_association_capacity(*args, validated_e1=prepared)
