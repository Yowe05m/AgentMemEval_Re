from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import intent_executor as executor
from agentmemeval.path1_v1.operations import runtime_probe as probe
from agentmemeval.path1_v1.operations import submission_process
from agentmemeval.path1_v1.operations.contracts import (
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.slurm import (
    AccountSnapshot,
    CapacityEvidence,
    SubmissionDisabled,
    SubmissionRequest,
    SubmitSpec,
)
from agentmemeval.path1_v1.operations.submission_process import file_sha256


def seal_request(fixture):
    request = fixture.request_data
    request["request_sha256"] = content_digest(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    fixture.request_path.write_bytes(canonical_json(request))


def make_e0_fixture(
    root, kind="resource_probe", mode="normal", scope=("cluster", "account", "owner")
):
    from path1_v1.unit.test_intent_executor import FAKE_SBATCH
    from path1_v1.unit.test_operations_contracts import _git
    from path1_v1.unit.test_runtime_probe import make_probe_fixture

    data, contract, request, path = make_probe_fixture(root)
    repo = data["repo"]
    names = (
        "worker",
        "runtime_probe",
        "probe_launcher",
        "resource_probe",
        "intent_executor",
        "process_ownership",
        "submission_process",
    )
    operation_paths = []
    for name in names:
        relative = f"src/agentmemeval/path1_v1/operations/{name}.py"
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# E0 identity fixture; not executed\n", encoding="utf-8")
        operation_paths.append(relative)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "complete E0 fixture closure")
    kwargs = dict(data["build_kwargs"])
    kwargs["operations_paths"] += operation_paths
    if kind == "resource_probe":
        kwargs.update(model_manifest_path=None, model_root=None)
        request.update(model_manifest_path=None, model_root=None)
    contract["runtime_identity"] = build_runtime_identity(repo, **kwargs)
    contract["job"]["kind"] = kind
    contract["job"]["output_relpath"] = "test_campaign/E0/" + kind + "/attempt_01"
    Path(request["contract_path"]).write_bytes(canonical_json(contract))
    request["contract_sha256"] = file_sha256(Path(request["contract_path"]))
    request["intent_id"] = canonical_intent_id(contract)
    python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter

    python_identity = verify_python_interpreter(python, file_sha256(python))
    service = kind == "service_probe"
    request["schema_version"] = 2
    request["launch"] = {
        "python_executable": str(python),
        "python_sha256": file_sha256(python),
        "python_identity": python_identity,
        "service_python_identity": python_identity if service else None,
        "service_python": str(python) if service else None,
        "service_python_sha256": file_sha256(python) if service else None,
        "service_port": 8123 if service else None,
        "command_seconds": 6600 if service else 90,
        "cleanup_seconds": 600 if service else 30,
        "startup_seconds": 1200 if service else 0,
    }
    fixture = SimpleNamespace(
        root=root,
        request_data=request,
        request_path=path,
        contract=contract,
        repo=repo,
        python=python,
    )
    seal_request(fixture)
    script = root / "e0.sh"
    script.write_bytes(
        executor.render_probe_script(
            python_executable=python, repository_root=repo, probe_request_path=path
        )
    )
    fake = root / "sbatch.py"
    fake.write_text(FAKE_SBATCH.replace("551;cluster", "551;" + scope[0]), encoding="utf-8")
    partition = root / "partition.py"
    partition.write_text(
        "print('PartitionName=gpu State=UP MaxTime=UNLIMITED')\n", encoding="utf-8"
    )
    evidence = root / "preflight"
    evidence.mkdir()
    spec = SubmitSpec(
        *scope,
        "gpu",
        "/runs/job",
        "/code/e0.sh",
        file_sha256(script),
        walltime_seconds=7200 if service else 120,
    )
    fixture.request = SubmissionRequest(contract["campaign_id"], request["intent_id"], spec)
    fixture.script = script
    fixture.partition = partition
    fixture.job_dir = Path(request["output_root"]) / contract["job"]["output_relpath"]
    fixture.producer = executor.E0PreSubmitVerifier(
        evidence,
        partition_command=executor.VerifiedCommand(
            (str(python), "-I", str(partition)),
            ((str(python), file_sha256(python)), (str(partition), file_sha256(partition))),
        ),
        fixture_mode=True,
    )
    helper = Path(submission_process.__file__).resolve()
    fixture.executor = executor.DurableIntentExecutor(
        root / "ctl",
        cluster=scope[0],
        account=scope[1],
        command=executor.VerifiedCommand(
            (str(python), "-I", str(fake), str(root), mode, str(helper)),
            (
                (str(python), file_sha256(python)),
                (str(fake), file_sha256(fake)),
                (str(helper), file_sha256(helper)),
            ),
        ),
        launch_files=lambda _: executor.ProbeLaunchFiles(path),
        pre_submit_verifier=fixture.producer,
        account_reader=lambda: AccountSnapshot(
            scope[0],
            scope[1],
            datetime.now(UTC),
            (),
            (),
            "b" * 64,
            CapacityEvidence(scope[0], scope[1], 8, datetime.now(UTC), "c" * 64, True),
        ),
        fixture_mode=True,
        fixture_paths={"/code/e0.sh": script, "/runs/job": fixture.job_dir},
        transport_term_grace_seconds=0.1,
    )
    return fixture


@pytest.mark.parametrize("kind", ["resource_probe", "service_probe"])
def test_real_git_model_preflight_helper_submission_chain_and_reentry(tmp_path, kind):
    fixture = make_e0_fixture(tmp_path, kind)
    receipt = fixture.executor.submit_once(fixture.request, timeout_seconds=30)
    assert receipt.status == "accepted" and receipt.job_id == "551"
    raw = (tmp_path / "external_effect.json").read_bytes()
    effect = json.loads(raw)
    assert ("--deadline=now+130seconds" in effect["argv"]) == (kind == "resource_probe")
    assert not any(arg.startswith("--immediate") for arg in effect["argv"])
    if kind == "service_probe":
        assert not any(arg.startswith("--deadline") for arg in effect["argv"])
    assert ("--time=00:02:00" if kind == "resource_probe" else "--time=02:00:00") in effect["argv"]
    assert effect["script_sha256"] == file_sha256(fixture.script)
    assert (
        "claim_path" not in fixture.request_data
        and "seed_inventory_path" not in fixture.request_data
    )
    assert fixture.contract["cohort"]["formation_count"] == 0
    assert not (fixture.job_dir / "payload").exists()
    preflight = list((tmp_path / "preflight").glob("*.json"))
    assert len(preflight) == 2
    assert all(json.loads(p.read_text())["runtime_admission"] is False for p in preflight)
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=10) == receipt
    assert (tmp_path / "external_effect.json").read_bytes() == raw
    assert len(list((tmp_path / "preflight").glob("*.json"))) == 2


@pytest.mark.parametrize(
    "mutation", ["legacy", "claim", "port", "seconds", "walltime", "script", "source", "python"]
)
def test_e0_rejects_rehashed_launch_drift_before_submission(tmp_path, mutation):
    fixture = make_e0_fixture(tmp_path, "service_probe")
    request = fixture.request_data
    if mutation == "legacy":
        request["schema_version"] = 1
        del request["launch"]
    elif mutation == "claim":
        request["claim_path"] = "not permitted"
    elif mutation == "port":
        request["launch"]["service_port"] = True
    elif mutation == "seconds":
        request["launch"]["cleanup_seconds"] = 120
    elif mutation == "walltime":
        fixture.request = replace(
            fixture.request, spec=replace(fixture.request.spec, walltime_seconds=120)
        )
    elif mutation == "script":
        fixture.script.write_bytes(fixture.script.read_bytes() + b"#SBATCH --array=1-4\n")
        fixture.request = replace(
            fixture.request,
            spec=replace(fixture.request.spec, script_sha256=file_sha256(fixture.script)),
        )
    elif mutation == "source":
        (fixture.repo / "src/core.py").write_text("changed", encoding="utf-8")
    else:
        request["launch"]["python_sha256"] = "f" * 64
    seal_request(fixture)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.executor.submit_once(fixture.request, timeout_seconds=20)
    assert not (tmp_path / "external_effect.json").exists()


def test_e0_default_gate_remains_closed(tmp_path):
    fixture = make_e0_fixture(tmp_path)
    fixture.executor.pre_submit_verifier = None
    with pytest.raises(SubmissionDisabled):
        fixture.executor.submit_once(fixture.request, timeout_seconds=10)
    assert not (tmp_path / "external_effect.json").exists()


@pytest.mark.parametrize("limit", ["00:01:00", "malformed"])
def test_actual_partition_response_cannot_fit_probe_is_rejected(tmp_path, limit):
    fixture = make_e0_fixture(tmp_path)
    fixture.partition.write_text(
        f"print('PartitionName=gpu State=UP MaxTime={limit}')\n", encoding="utf-8"
    )
    fixture.producer.command = replace(
        fixture.producer.command,
        files=(
            (str(fixture.python), file_sha256(fixture.python)),
            (str(fixture.partition), file_sha256(fixture.partition)),
        ),
    )
    with pytest.raises(SubmissionDisabled, match="walltime"):
        fixture.executor.submit_once(fixture.request, timeout_seconds=10)
    assert not (tmp_path / "external_effect.json").exists()


def test_resource_no_model_is_real_static_identity_not_model_bypass(tmp_path):
    fixture = make_e0_fixture(tmp_path)
    validated = probe.validate_probe_request(fixture.request_path)
    assert validated.request["model_root"] is None
    assert validated.contract["runtime_identity"]["model"] is None
    fixture.contract["job"]["kind"] = "service_probe"
    Path(fixture.request_data["contract_path"]).write_bytes(canonical_json(fixture.contract))
    fixture.request_data["contract_sha256"] = file_sha256(
        Path(fixture.request_data["contract_path"])
    )
    fixture.request_data["intent_id"] = canonical_intent_id(fixture.contract)
    seal_request(fixture)
    with pytest.raises(probe.RuntimeProbeError, match="actual model"):
        probe.validate_probe_request(fixture.request_path)


def test_sink_failure_after_actual_command_result_stays_unknown_and_never_resubmits(
    tmp_path,
    monkeypatch,
):
    from agentmemeval.path1_v1.operations.process_ownership import ReceiptSinkError

    fixture = make_e0_fixture(tmp_path)
    original = executor.write_exclusive_json

    def publish(path, payload):
        original(path, payload)
        if path.name == "go.json":
            deadline = time.monotonic() + 10
            while not (path.parent / "result.json").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (path.parent / "result.json").exists()
            # Fault-injected sink exception; real kernel/sink behavior is covered
            # by the separate Linux ownership tests. No fake receipt is admitted.
            raise ReceiptSinkError(None, None, OSError("disk fixture"), None)

    monkeypatch.setattr(executor, "write_exclusive_json", publish)
    receipt = fixture.executor.submit_once(fixture.request, timeout_seconds=30)
    assert receipt.status == "unknown" and not receipt.submitter_finished
    effect = (tmp_path / "external_effect.json").read_bytes()
    record = fixture.executor.inspect(fixture.request)
    assert record["status"] == "submit_unknown" and record["cleanup_persistence_failed"]
    assert fixture.executor.submit_once(fixture.request, timeout_seconds=10) == receipt
    assert fixture.executor.lookup(fixture.request, timeout_seconds=10) == receipt
    assert effect == (tmp_path / "external_effect.json").read_bytes()


def test_accepted_e0_cannot_rebind_mutated_request_at_the_same_path(tmp_path):
    fixture = make_e0_fixture(tmp_path, "service_probe")
    fixture.executor.submit_once(fixture.request, timeout_seconds=30)
    fixture.request_data["launch"]["service_port"] += 1
    seal_request(fixture)
    with pytest.raises(executor.StoreIntegrityError, match="durable E0 request"):
        fixture.executor.submit_once(fixture.request, timeout_seconds=10)


def test_worker_launcher_only_adds_no_bytecode_and_keeps_research_entry(tmp_path):
    raw = executor.render_worker_script(
        python_executable=tmp_path / "python",
        repository_root=tmp_path / "repo",
        worker_request_path=tmp_path / "request.json",
    )
    assert b" -B -s -m agentmemeval.path1_v1.operations.worker --request " in raw
    assert b"probe_launcher" not in raw and b"#SBATCH" not in raw


def test_real_e0_submission_preserves_raw_user_association_scope(tmp_path, monkeypatch):
    from path1_v1.unit.test_user_capacity import NOW, make_user_capacity_fixture

    fixture = make_e0_fixture(tmp_path, scope=("n16r4", "scze856", "scze856"))
    cap_root = tmp_path / "capacity"
    cap_root.mkdir()
    case = make_user_capacity_fixture(cap_root, monkeypatch)
    fixture.executor.account_reader = case["reader"]
    fixture.executor.clock = lambda: NOW
    fixture.producer.clock = lambda: NOW
    receipt = fixture.executor.submit_once(fixture.request, timeout_seconds=30)
    assert receipt.status == "accepted"
    capacity = fixture.executor.inspect(fixture.request)["capacity"]
    assert capacity["scope"] == "user_association"
    assert capacity["full_account_visibility"] is False
    assert capacity["allocation_guaranteed"] is False
    assert capacity["association_id"] == "7193"
    assert content_digest(capacity["raw_observation"]) == capacity["snapshot_sha256"]
    assert fixture.executor.lookup(fixture.request, timeout_seconds=10) == receipt


@pytest.mark.parametrize("phase", [None, "E1", "E2", "E3"])
def test_user_association_capacity_never_admits_research_or_missing_phase(
    tmp_path, monkeypatch, phase
):
    from path1_v1.unit.test_user_capacity import NOW, make_user_capacity_fixture

    fixture = make_e0_fixture(tmp_path, scope=("n16r4", "scze856", "scze856"))
    cap_root = tmp_path / "capacity"
    cap_root.mkdir()
    case = make_user_capacity_fixture(cap_root, monkeypatch)
    fixture.executor.account_reader = case["reader"]
    fixture.executor.clock = lambda: NOW
    with pytest.raises(SubmissionDisabled, match="only for validated E0"):
        fixture.executor._capacity(fixture.request, case["ledger"], 1, validated_phase=phase)
    assert not (tmp_path / "external_effect.json").exists()
