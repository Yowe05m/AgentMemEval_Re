"""Anonymous resource evidence plus real SQLite/archive/restore fault tests.

Historical submission/source validation is an explicit boundary fixture here;
the unchanged implementation is covered by the separate real-archive suite.
Scheduler and kernel observations are fixtures, never actual Slurm admission.
"""

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import resource_terminal as rt
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.state import CampaignStore

NOW = datetime(2026, 11, 14, 1, tzinfo=UTC)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_resource_fixture(tmp_path, monkeypatch):
    run, repository = (tmp_path / name for name in ("run", "repo"))
    for path in (run, repository):
        path.mkdir()
    intent, request_hash = "1" * 64, "2" * 64
    store = CampaignStore(tmp_path / "control", "resource_fixture")
    invocation = store.directory / "i" / intent[:16]
    invocation.mkdir(parents=True)
    (invocation / "invocation.json").write_bytes(b'{"offline_boundary_fixture":true}')
    spec = SubmitSpec(
        cluster="cluster",
        account="account",
        partition="gpu",
        user="person",
        gpus=1,
        walltime_seconds=120,
        script_path="/resource-fixture/probe.sh",
        script_sha256="3" * 64,
        work_dir="/resource-fixture/output",
        script_args=(),
    )
    request = SubmissionRequest("campaign", intent, spec)
    process = {
        "host": "login",
        "boot_id": "boot_login",
        "effective_uid": 17,
        "platform": "linux",
        "pid": 100,
        "start_marker": "1000",
    }
    record = {
        "status": "accepted",
        "intent_id": intent,
        "campaign_id": "campaign",
        "job_id": "42",
        "request_sha256": request.request_sha256,
        "spec": spec.payload(),
        "submitter_finished": True,
        "effective_user": {"uid": 17},
        "created_at": "2026-11-14T00:00:00+00:00",
        "parent_identity": process,
        "ready": {"submitter": dict(process, pid=101, start_marker="1001")},
        "invocation_dir": str(invocation),
        "manifest_sha256": er._sha(invocation / "invocation.json"),
    }
    with store.lock():
        snapshot = store.initialize(
            registry={
                er.NAMESPACE: {
                    "schema_version": 1,
                    "cluster": "cluster",
                    "account": "account",
                    "intents": {intent: record},
                }
            },
            state={},
        )
        claim = store.claim_run(
            expected_generation=snapshot.generation,
            run_id=intent,
            intent_id=intent,
            run_directory=run / "output",
            metadata={"purpose": "slurm_job_container", "request_sha256": request.request_sha256},
        )
        store.start_claim(
            expected_generation=store.read_registry().generation, run_id=intent, token=claim.token
        )
    original_path = er._path

    def mapped_path(value):
        value = str(value)
        if value == "/resource-fixture/output":
            return original_path(run / "output")
        return original_path(Path(value))

    monkeypatch.setattr(er, "_path", mapped_path)
    root = run / "output/e0_probe"
    root.mkdir(parents=True)
    descriptor_path = run / "submission.json"
    write(descriptor_path, {"intent_id": intent, "config_relative": "site.json"})
    (run / "probe.sh").write_bytes(b"offline fixture\n")
    (invocation / "raw.txt").write_bytes(b"accepted fixture transport\n")
    identity = {
        "core_sha": "a" * 40,
        "ops_sha": "a" * 40,
        "model": None,
        "operations_files": [{"path": rt._MODULE_PATH, "sha256": "b" * 64}],
    }
    probe = {
        "request_sha256": request_hash,
        "intent_id": intent,
        "launch": {"python_executable": "/usr/bin/python3"},
    }
    contract = {"campaign_id": "campaign", "runtime_identity": identity}
    config = {
        "commands": {
            name: {"path": "/usr/bin/" + name, "sha256": "b" * 64} for name in ("squeue", "sacct")
        },
        "accounting_since": "2026-11-14T00:00:00+00:00",
    }
    bound = {
        "request": request,
        "config": config,
        "probe": probe,
        "contract": contract,
        "run": run,
        "invocation": invocation,
        "repository": repository,
    }

    def original_submission_boundary(
        store_arg, descriptor_arg, digest_arg, record_arg, claim_arg, *, require_empty_output=True
    ):
        assert store_arg is store and descriptor_arg == descriptor_path and not require_empty_output
        assert record_arg == record and claim_arg.intent_id == intent
        assert digest_arg == er._sha(descriptor_path)
        return bound

    monkeypatch.setattr(er, "_bound_inputs", original_submission_boundary)
    monkeypatch.setattr(rt, "_verifier_identity", lambda repo, cfg: identity)
    monkeypatch.setattr(
        er, "_historical_identity", lambda repo, value: config if value == identity else None
    )

    def gone(rec):
        return {
            "host": "login",
            "boot_id": "boot_login",
            "observations": [
                {"role": role, "identity": value, "actual_start_marker": None}
                for role, value in (
                    ("parent", rec["parent_identity"]),
                    ("submitter", rec["ready"]["submitter"]),
                )
            ],
        }

    monkeypatch.setattr(er, "_processes_gone", gone)
    parent = {
        "hostname": "gpu01",
        "boot_id": "boot_compute",
        "euid": 17,
        "pid": 200,
        "job_id": "42",
        "cgroup": "0::/slurm/uid_17/job_42/step_batch\n",
    }
    child = {
        "host": "gpu01",
        "boot_id": "boot_compute",
        "effective_uid": 17,
        "pid": 201,
        "start_marker": "2001",
        "session_id": 201,
        "pgid": 201,
        "slurm_job_id": "42",
        "step_id": "batch",
        "cgroup": parent["cgroup"].splitlines(),
    }
    binding = {
        "campaign_id": "campaign",
        "intent_id": intent,
        "job_id": "42",
        "role": "resource_probe",
        "service_instance_id": "c" * 32,
        "source_sha256": content_digest(identity),
    }
    launch = {
        "request_sha256": request_hash,
        "runtime_admission": False,
        "binding": binding,
        "host": parent,
        "argv": [
            "/usr/bin/python3",
            "-B",
            "-s",
            "-m",
            "agentmemeval.path1_v1.operations.resource_probe",
            "--collect-request",
            str(run / "probe_request.json"),
        ],
        "command_seconds": 90,
        "cleanup_seconds": 30,
        "term_grace_seconds": 15,
    }
    cleanup = {
        "binding": binding,
        "identity": child,
        "returncode": 0,
        "term_sent": True,
        "kill_sent": False,
        "leader_reaped": True,
        "group_cleanup_verified": True,
        "errors": [],
        "remaining_non_zombie_pids": [],
    }
    write(root / "resource_launch.json", launch)
    write(root / "resource_cleanup.json", cleanup)
    for name in ("resource_stdout.log", "resource_stderr.log"):
        (root / name).write_bytes(b"")
    host = dict(parent, pid=201)
    gpu = {
        "argv": [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        "returncode": 0,
        "stderr": "",
        "stdout": "GPU-fixture, Test GPU, 999, 24000\n",
        "gpus": [
            {
                "uuid": "GPU-fixture",
                "name": "Test GPU",
                "driver_version": "999",
                "memory_mib": "24000",
            }
        ],
        "allocation_gpu_mapping_verified": False,
    }
    write(root / "observations/host.json", host)
    write(root / "observations/gpu.json", gpu)
    write(
        root / "observations/resource_receipt.json",
        {
            "status": "observations_complete",
            "intent_id": intent,
            "request_sha256": request_hash,
            "runtime_admission": False,
            "host": host,
            "gpu": gpu,
            "filesystem": {
                "path": str(root / "observations"),
                "total": 1000,
                "used": 100,
                "free": 900,
            },
        },
    )
    (root / "observations/filesystem_probe.bin").write_bytes(request_hash.encode())
    allocation = [
        "cluster",
        "42",
        "person",
        "account",
        er.job_name("campaign", intent),
        "COMPLETED",
        "0:0",
        "1",
        "2026-11-14T00:00:10",
        "2026-11-14T00:01:00",
        "2026-11-14T00:01:01",
        "gpu01",
        "cpu=2,gres/gpu=1",
        er.job_comment("campaign", intent),
        spec.work_dir,
    ]
    batch = [
        "cluster",
        "42.batch",
        "",
        "",
        "batch",
        "COMPLETED",
        "0:0",
        "1",
        "",
        "2026-11-14T00:01:00",
        "2026-11-14T00:01:01",
        "gpu01",
        "",
        "",
        "",
    ]
    scheduler = [allocation, batch]
    clock = SimpleNamespace(elapsed=0.0)

    def now():
        return NOW + timedelta(seconds=clock.elapsed)

    def sleep(value):
        clock.elapsed += value

    class Reader:
        def read(self, requested, job_id, cfg):
            rows = [
                {
                    "argv": list(argv),
                    "pin": config["commands"][Path(argv[0]).name],
                    "returncode": 0,
                    "stderr": "",
                    "stdout": ""
                    if index < 2
                    else "".join("|".join(row) + "\n" for row in scheduler),
                }
                for index, argv in enumerate(er._read_argv(request, "42", config))
            ]
            return er.DeadlineObservation(now(), now(), 17, tuple(rows))

    kwargs = {
        "descriptor_path": descriptor_path,
        "descriptor_sha256": er._sha(descriptor_path),
        "reader": Reader(),
        "clock": now,
        "monotonic": lambda: clock.elapsed,
        "sleep": sleep,
    }
    return SimpleNamespace(
        store=store,
        record=record,
        bound=bound,
        root=root,
        request=request,
        scheduler=scheduler,
        kwargs=kwargs,
        launch=launch,
        cleanup=cleanup,
        close=lambda: rt.reconcile_resource_terminal(store, **kwargs),
    )


@pytest.fixture
def resource(tmp_path, monkeypatch):
    return make_resource_fixture(tmp_path, monkeypatch)


def test_executed_complete_resource_closes_once_without_research_or_retry(resource):
    before = resource.store.read_registry()
    result = resource.close()
    after = resource.store.read_registry()
    assert after.generation == before.generation + 1 and after.registry == before.registry
    assert result.claim_status == "completed" and result.resource_observation_complete
    assert (
        not result.retry_eligible
        and not result.runtime_admission
        and not result.scientific_admission
    )
    assert result.scientific_samples == 0 and not hasattr(result, "eligibility_receipt_sha256")
    assert result.validate(resource.record, store=resource.store) == result
    assert resource.close() == result and resource.store.read_registry() == after


def test_readonly_check_does_not_close_claim_or_create_archive(resource):
    before = resource.store.read_registry()
    keys = ("descriptor_path", "descriptor_sha256", "reader", "clock")
    result = rt.check_resource_terminal(
        resource.store, **{key: resource.kwargs[key] for key in keys}
    )
    assert result["facts"]["resource_observation_complete"]
    assert not result["technical_claim_closed"] and resource.store.read_registry() == before
    assert not rt._audit_path(resource.store, resource.request.intent_id).parent.exists()


@pytest.mark.parametrize("defect", ["queue", "steps", "stale", "command", "scope"])
def test_scheduler_raw_command_errors_or_live_target_never_create_audit(resource, defect):
    original = resource.kwargs["reader"]

    class Reader:
        def read(self, *args):
            observed = original.read(*args)
            rows = copy.deepcopy(observed.commands)
            if defect == "queue":
                rows[0]["stdout"] = "42|person|account|job|RUNNING|comment|time|work\n"
            elif defect == "steps":
                rows[1]["stdout"] = "42.batch\n"
            elif defect == "command":
                rows[0]["returncode"] = 1
            elif defect == "scope":
                rows[0]["argv"].append("--partition=gpu")
            else:
                return replace(observed, observed_at=observed.observed_at - timedelta(seconds=31))
            return replace(observed, commands=rows)

    resource.kwargs["reader"] = Reader()
    with pytest.raises(ValueError):
        resource.close()
    assert resource.store.read_registry().claims[0].status == "started"
    assert not rt._audit_path(resource.store, resource.request.intent_id).parent.exists()


@pytest.mark.parametrize("action", ["check", "close", "verify"])
def test_cli_has_no_offline_or_admission_bypass(resource, action):
    from agentmemeval.path1_v1.operations.resource_terminal_cli import main

    with pytest.raises(SystemExit) as caught:
        main(
            [
                action,
                "--control-directory",
                str(resource.store.directory),
                "--store-campaign-id",
                resource.store.campaign_id,
                "--verified",
                "true",
            ]
        )
    assert caught.value.code == 2


def test_cli_verify_returns_actual_typed_resolution(resource, capsys, monkeypatch):
    from agentmemeval.path1_v1.operations import resource_terminal_cli as cli

    monkeypatch.setattr(cli, "CampaignStore", lambda directory, campaign: resource.store)
    result = resource.close()
    assert (
        cli.main(
            [
                "verify",
                "--control-directory",
                str(resource.store.directory),
                "--store-campaign-id",
                resource.store.campaign_id,
                "--intent-id",
                resource.request.intent_id,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_receipt_sha256"] == result.terminal_receipt_sha256
    assert payload["runtime_admission"] is False and "_issuer" not in payload


@pytest.mark.parametrize("defect", ["missing", "digest", "origin"])
def test_resource_verifier_requires_its_own_published_source(monkeypatch, tmp_path, defect):
    source = Path(rt.__file__).resolve()
    repo = source.parents[4]
    identity = {"operations_files": [{"path": rt._MODULE_PATH, "sha256": er._sha(source)}]}
    if defect == "missing":
        identity["operations_files"] = []
    elif defect == "digest":
        identity["operations_files"][0]["sha256"] = "f" * 64
    else:
        repo = tmp_path
    monkeypatch.setattr(er, "_verifier_identity", lambda repository, config: identity)
    with pytest.raises(ValueError):
        rt._verifier_identity(repo, {})


@pytest.mark.parametrize(
    "name", ["gpu.json", "host.json", "resource_receipt.json", "filesystem_probe.bin"]
)
def test_missing_observations_are_failed_technical_closure_not_retry_permission(resource, name):
    (resource.root / "observations" / name).unlink()
    result = resource.close()
    assert result.claim_status == "failed" and not result.resource_observation_complete
    assert (
        not result.retry_eligible
        and result.validate(resource.record, store=resource.store) == result
    )


@pytest.mark.parametrize(
    "name", ["resource_launch.json", "resource_cleanup.json", "resource_stdout.log"]
)
def test_missing_core_identity_or_cleanup_keeps_slot(resource, name):
    (resource.root / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        resource.close()
    assert resource.store.read_registry().claims[0].status == "started"


@pytest.mark.parametrize(
    "defect", ["job", "uid", "pid", "cgroup", "group", "remaining", "errors", "source"]
)
def test_cleanup_boolean_does_not_override_identity_or_live_member_conflict(resource, defect):
    value = copy.deepcopy(resource.cleanup)
    if defect == "job":
        value["binding"]["job_id"] = "43"
    elif defect == "uid":
        value["identity"]["effective_uid"] = 18
    elif defect == "pid":
        value["identity"]["pid"] = resource.launch["host"]["pid"]
    elif defect == "cgroup":
        value["identity"]["cgroup"] = ["0::/slurm/job_43/step_batch"]
    elif defect == "group":
        value["group_cleanup_verified"] = False
    elif defect == "remaining":
        value["remaining_non_zombie_pids"] = [205]
    elif defect == "errors":
        value["errors"] = ["uncertain"]
    else:
        value["binding"]["source_sha256"] = "f" * 64
    write(resource.root / "resource_cleanup.json", value)
    with pytest.raises(ValueError):
        resource.close()


def test_parent_launcher_and_child_collector_have_different_valid_pids(resource):
    facts = rt._resource_facts(resource.bound, resource.record)
    assert facts["launcher_pid"] == 200 and facts["resource_pid"] == 201
    assert facts["allocation_gpu_mapping_verified"] is False


@pytest.mark.parametrize("field", ["effective_uid", "session_id", "pgid"])
def test_float_kernel_identity_is_not_equivalent_to_integer(resource, field):
    value = copy.deepcopy(resource.cleanup)
    value["identity"][field] = float(value["identity"][field])
    write(resource.root / "resource_cleanup.json", value)
    with pytest.raises(ValueError, match="identity mismatch"):
        resource.close()


@pytest.mark.parametrize(
    "defect",
    [
        "active",
        "missing_batch",
        "duplicate",
        "wrong_job",
        "wrong_node",
        "missing_tres",
        "missing_start",
        "step_active",
    ],
)
def test_allocation_and_steps_must_jointly_prove_technical_terminal(resource, defect):
    if defect == "active":
        resource.scheduler[0][5] = "RUNNING"
    elif defect == "missing_batch":
        resource.scheduler.pop()
    elif defect == "duplicate":
        resource.scheduler.append(list(resource.scheduler[0]))
    elif defect == "wrong_job":
        resource.scheduler[0][1] = "43"
    elif defect == "wrong_node":
        resource.scheduler[0][11] = "gpu02"
    elif defect == "missing_tres":
        resource.scheduler[0][12] = ""
    elif defect == "missing_start":
        resource.scheduler[0][9] = "None"
    else:
        resource.scheduler[1][5] = "RUNNING"
    with pytest.raises(ValueError):
        resource.close()
    assert resource.store.read_registry().claims[0].status == "started"
    assert not rt._audit_path(resource.store, resource.request.intent_id).parent.exists()


def test_failure_terminal_can_release_only_technical_slot(resource):
    resource.scheduler[0][5:7] = ["FAILED", "1:0"]
    resource.scheduler[1][5:7] = ["FAILED", "1:0"]
    result = resource.close()
    assert result.scheduler_state == "FAILED" and result.claim_status == "failed"
    assert result.resource_observation_complete and not result.retry_eligible


@pytest.mark.parametrize("target", ["original", "archive", "restore", "evidence"])
def test_terminal_resolution_rechecks_all_raw_archive_and_restored_bytes(resource, target):
    result = resource.close()
    audit = rt._audit_path(resource.store, result.intent_id)
    if target == "original":
        path = resource.root / "resource_stdout.log"
    elif target == "archive":
        path = audit.parent / "run/snapshot.tar.gz"
    elif target == "restore":
        path = audit.parent / "run/restored/run/probe.sh"
    else:
        path = audit
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, OSError)):
        result.validate(resource.record, store=resource.store)


def test_forged_typed_resolution_rejected(resource):
    issued = resource.close()
    with pytest.raises(ValueError, match="not issued"):
        replace(issued, _issuer=None).validate(resource.record, store=resource.store)


def test_long_archive_verification_keeps_first_observation_fresh_at_its_own_check(
    resource, monkeypatch
):
    original = rt._archive_sources

    def slow(*args):
        resource.kwargs["sleep"](60)
        return original(*args)

    monkeypatch.setattr(rt, "_archive_sources", slow)
    result = resource.close()
    assert result.claim_status == "completed"
    audit = er._json(rt._audit_path(resource.store, result.intent_id))
    assert (
        datetime.fromisoformat(audit["sealed_at"])
        - datetime.fromisoformat(audit["first_checked_at"])
    ).total_seconds() > 60
    assert result.validate(resource.record, store=resource.store) == result


def test_cas_failure_preserves_original_occupancy_and_audit(resource, monkeypatch):
    before = resource.store.read_registry()

    def fail(**kwargs):
        raise OSError("injected CAS failure")

    monkeypatch.setattr(resource.store, "finish_claim", fail)
    with pytest.raises(OSError, match="CAS failure"):
        resource.close()
    assert resource.store.read_registry() == before
    audit = rt._audit_path(resource.store, resource.request.intent_id)
    assert audit.is_file()
    with pytest.raises(FileExistsError):
        resource.close()


@pytest.mark.parametrize("complete", [True, False])
def test_durable_completed_and_failed_claims_pass_executor_ledger_only_with_intact_audit(
    resource, complete
):
    from agentmemeval.path1_v1.operations.intent_executor import DurableIntentExecutor

    if not complete:
        (resource.root / "observations/gpu.json").unlink()
    result = resource.close()
    executor = object.__new__(DurableIntentExecutor)
    executor.store = resource.store
    executor.cluster, executor.account = "cluster", "account"
    executor._path = lambda value: er._path(value)
    snapshot = resource.store.read_registry()
    assert executor._ledger(snapshot) == snapshot.registry[er.NAMESPACE]
    assert result.claim_status == ("completed" if complete else "failed")
    audit = rt._audit_path(resource.store, result.intent_id)
    audit.write_bytes(audit.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        executor._ledger(snapshot)


@pytest.mark.parametrize("complete", [True, False])
def test_typed_resource_release_removes_one_capacity_slot_but_live_contradiction_denies(
    resource, tmp_path, monkeypatch, complete
):
    from agentmemeval.path1_v1.operations import user_capacity as capacity
    from agentmemeval.path1_v1.operations.intent_executor import DurableIntentExecutor
    from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, job_comment, job_name
    from path1_v1.unit.test_user_capacity import FixtureTransport, make_user_capacity_fixture

    helper = tmp_path / "capacity"
    helper.mkdir()
    pins = make_user_capacity_fixture(helper, monkeypatch)["reader"].commands
    monkeypatch.setattr(capacity, "_effective_identity", lambda user: 17)
    outputs = {
        "sacctmgr": "cluster|account|person|||1|2|normal|||8|||\n",
        "squeue": "",
        "sacct": "",
    }
    reader = capacity.UserAssociationReader(
        cluster="cluster",
        account="account",
        user="person",
        commands=pins,
        since=NOW - timedelta(days=1),
        clock=resource.kwargs["clock"],
        transport=FixtureTransport(outputs),
    )
    request = SubmissionRequest(
        "campaign", "9" * 64, replace(resource.request.spec, work_dir="/other/output")
    )
    executor = object.__new__(DurableIntentExecutor)
    executor.store, executor.account_reader = resource.store, reader.read
    executor.clock = resource.kwargs["clock"]

    def automatic():
        return executor._capacity(
            request, resource.store.read_registry().registry[er.NAMESPACE], 1, validated_phase="E0"
        )

    def check(resolutions=None):
        return capacity.validate_user_association_capacity(
            reader.read(),
            request,
            resource.store.read_registry().registry[er.NAMESPACE],
            1,
            resource.kwargs["clock"](),
            "E0",
            resolved_e0=resolutions,
            store=resource.store,
        )

    with pytest.raises(SubmissionDisabled, match="no admission"):
        check()
    with pytest.raises(SubmissionDisabled, match="no admission"):
        automatic()
    if not complete:
        (resource.root / "observations/gpu.json").unlink()
    resolution = resource.close()
    with pytest.raises(SubmissionDisabled, match="no admission"):
        check()
    assert check({resolution.intent_id: resolution})["account_occupancy"] == 0
    assert automatic()["account_occupancy"] == 0
    outputs["squeue"] = (
        "|".join(
            [
                "42",
                "person",
                "account",
                job_name("campaign", resolution.intent_id),
                "RUNNING",
                job_comment("campaign", resolution.intent_id),
                "2026-11-14T00:00:10",
                resource.request.spec.work_dir,
            ]
        )
        + "\n"
    )
    with pytest.raises(SubmissionDisabled, match="contradicts"):
        check({resolution.intent_id: resolution})
    with pytest.raises(SubmissionDisabled, match="contradicts"):
        automatic()
    outputs["sacct"] = "cluster|" + outputs["squeue"].replace("RUNNING", "COMPLETED")
    outputs["squeue"] = ""
    assert automatic()["account_occupancy"] == 0
    outputs["sacct"] = outputs["sacct"].replace("COMPLETED", "FAILED")
    with pytest.raises(SubmissionDisabled, match="contradicts"):
        automatic()
    outputs["sacct"] = ""
    original_validate = rt.ResourceTerminalResolution.validate
    latency = {"seconds": 29}

    def slow_validation(self, *args, **kwargs):
        result = original_validate(self, *args, **kwargs)
        resource.kwargs["sleep"](latency["seconds"])
        return result

    monkeypatch.setattr(rt.ResourceTerminalResolution, "validate", slow_validation)
    assert automatic()["account_occupancy"] == 0
    latency["seconds"] = 31
    with pytest.raises(SubmissionDisabled, match="outlived snapshot freshness"):
        automatic()


@pytest.mark.parametrize("delay", [29, 31])
def test_full_account_capacity_rechecks_freshness_after_slow_archive_validation(
    resource, monkeypatch, delay
):
    from agentmemeval.path1_v1.operations.intent_executor import DurableIntentExecutor
    from agentmemeval.path1_v1.operations.slurm import (
        AccountSnapshot,
        CapacityEvidence,
        SubmissionDisabled,
    )

    resource.close()
    executor = object.__new__(DurableIntentExecutor)
    executor.store, executor.clock = resource.store, resource.kwargs["clock"]
    executor.cluster, executor.account = "cluster", "account"

    def reader():
        now = executor.clock()
        evidence = CapacityEvidence("cluster", "account", 8, now, "a" * 64, True)
        return AccountSnapshot("cluster", "account", now, (), (), "b" * 64, evidence)

    executor.account_reader = reader
    original = rt.ResourceTerminalResolution.validate

    def slow(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        resource.kwargs["sleep"](delay)
        return result

    monkeypatch.setattr(rt.ResourceTerminalResolution, "validate", slow)
    request = SubmissionRequest(
        "campaign", "9" * 64, replace(resource.request.spec, work_dir="/other/output")
    )

    def check():
        return executor._capacity(
            request, resource.store.read_registry().registry[er.NAMESPACE], 1, validated_phase="E0"
        )

    if delay < 30:
        assert check()["account_occupancy"] == 0
    else:
        with pytest.raises(SubmissionDisabled, match="outlived snapshot freshness"):
            check()
