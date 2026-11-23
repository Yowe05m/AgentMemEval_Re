"""Offline evidence tests; Linux exercises real prepared inputs/Git/SQLite, no scheduler."""

from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import e0_service_cli as service
from agentmemeval.path1_v1.operations import submission_rejection as sr
from agentmemeval.path1_v1.operations import submission_rejection_cli as cli
from agentmemeval.path1_v1.operations import user_capacity as capacity
from agentmemeval.path1_v1.operations.contracts import canonical_json, content_digest
from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.state import CampaignStore
from path1_v1.unit.historical_service_v2_fixture import prepare_historical_service_v2
from path1_v1.unit.test_e0_service_cli import fixture as service_fixture
from path1_v1.unit.test_operations_contracts import _git, _seal_identity, evidence
from path1_v1.unit.test_user_capacity import (
    make_request,
    make_user_capacity_fixture,
)

NOW = datetime(2026, 11, 14, 1, tzinfo=UTC)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def observation_case():
    config = {
        "accounting_since": "2026-11-13T00:00:00+00:00",
        "commands": {
            name: {"path": "/usr/bin/" + name, "sha256": "a" * 64} for name in ("squeue", "sacct")
        },
    }
    request = SubmissionRequest(
        "campaign",
        "1" * 64,
        SubmitSpec("n16r4", "scze856", "scze856", "gpu", "/run/job", "/code/a.sh", "a" * 64),
    )
    bound = {
        "request": request,
        "config": config,
        "result": {"command_finished_ns": int((NOW - timedelta(seconds=1)).timestamp() * 1e9)},
    }
    record = {
        "effective_user": {"uid": 123},
        "created_at": (NOW - timedelta(seconds=60)).isoformat(),
    }
    rows = [
        dict(
            argv=list(argv),
            pin=config["commands"][Path(argv[0]).name],
            returncode=0,
            stdout="",
            stderr="",
        )
        for argv in sr._read_argv(request, config)
    ]
    value = sr.RejectionObservation(NOW, NOW, 123, tuple(rows)).payload()
    return bound, record, value


def test_exact_reconstructed_bytes_satisfy_historical_hash_without_trusting_json(tmp_path):
    candidate, result = tmp_path / "stderr.txt", tmp_path / "result.json"
    candidate.write_bytes(sr.POLICY_STDERR)
    write(result, {"stderr_sha256": sr.POLICY_SHA256})
    assert sr._candidate(candidate, sr.POLICY_SHA256, result) == sr.POLICY_STDERR
    assert sr.POLICY_SHA256 == "4dfd470a52f839c4ad41ee4d7f7cfa153cc0dd371a858bedef7ac24ed90df4fc"


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"rejected",
        b'{"verified":true}',
        sr.POLICY_STDERR.replace(b"55GB", b"64GB"),
        sr.POLICY_STDERR.replace(b"\n", b"\r\n"),
        sr.POLICY_STDERR + b"warning\n",
    ],
)
def test_other_error_even_with_self_consistent_hash_never_releases(tmp_path, body):
    candidate, result = tmp_path / "stderr.txt", tmp_path / "result.json"
    candidate.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    write(result, {"stderr_sha256": digest})
    with pytest.raises(er.E0ReconciliationError, match="unknown rejection body"):
        sr._candidate(candidate, digest, result)


def test_candidate_must_match_the_original_result_commitment(tmp_path):
    candidate, result = tmp_path / "stderr.txt", tmp_path / "result.json"
    candidate.write_bytes(sr.POLICY_STDERR)
    write(result, {"stderr_sha256": "0" * 64})
    with pytest.raises(er.E0ReconciliationError, match="original stderr"):
        sr._candidate(candidate, sr.POLICY_SHA256, result)


def test_model_identity_requires_explicit_historical_source_opt_in(tmp_path):
    data = evidence.__wrapped__(tmp_path)
    # The generic fixture omitted src/operations.py from core+ops closure? It includes both.
    with pytest.raises(er.E0ReconciliationError, match="model run"):
        er._historical_identity(data["repo"], data["identity"])
    assert er._historical_identity(data["repo"], data["identity"], allow_model=True) == {
        "setting": 1
    }


def test_no_target_in_full_observation_is_required_not_a_filtered_empty_query():
    bound, record, value = observation_case()
    sr._validate_observation(value, bound, record, NOW)
    assert not any("--jobs=" in arg for row in value["commands"] for arg in row["argv"])
    assert "--states=all" in value["commands"][0]["argv"]


@pytest.mark.parametrize(
    "defect",
    [
        "uid",
        "stale",
        "future",
        "warning",
        "failed",
        "filtered",
        "missing",
        "float_rc",
        "steps",
        "queue_scope",
        "sacct_scope",
        "unfinished",
    ],
)
def test_observation_refuses_incomplete_or_stale_facts(defect):
    bound, record, value = observation_case()
    if defect == "uid":
        value["effective_uid"] = 456
    elif defect == "stale":
        value["observed_at"] = (NOW - timedelta(seconds=31)).isoformat()
    elif defect == "future":
        value["completed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif defect == "warning":
        value["commands"][0]["stderr"] = "warning"
    elif defect == "failed":
        value["commands"][0]["returncode"] = 1
    elif defect == "float_rc":
        value["commands"][0]["returncode"] = 0.0
    elif defect == "filtered":
        value["commands"][0]["argv"].append("--name=not-target")
    elif defect == "missing":
        value["commands"].pop()
    elif defect == "steps":
        value["commands"][1]["stdout"] = "99.batch\n"
    elif defect == "queue_scope":
        value["commands"][0]["stdout"] = "99|other|scze856|x|RUNNING|||/x\n"
    elif defect == "unfinished":
        bound["result"]["command_finished_ns"] = int((NOW + timedelta(seconds=1)).timestamp() * 1e9)
    else:
        value["commands"][2]["stdout"] = "other|99|scze856|scze856|x|FAILED||x|/x\n"
    with pytest.raises(er.E0ReconciliationError):
        sr._validate_observation(value, bound, record, NOW)


@pytest.mark.parametrize(
    "source,index",
    [
        ("queue", 3),
        ("queue", 5),
        ("queue", 7),
        ("accounting", 4),
        ("accounting", 6),
        ("accounting", 8),
    ],
)
def test_even_partial_target_identity_in_any_scheduler_row_blocks(source, index):
    bound, record, value = observation_case()
    request = bound["request"]
    target = {
        3: sr.job_name(request.campaign_id, request.intent_id),
        5: sr.job_comment(request.campaign_id, request.intent_id),
        7: request.spec.work_dir,
    }
    if source == "queue":
        row = ["99", "scze856", "scze856", "unrelated", "RUNNING", "", "", "/other"]
        row[index] = target[index]
        value["commands"][0]["stdout"] = "|".join(row) + "\n"
    else:
        row = ["n16r4", "99", "scze856", "scze856", "unrelated", "FAILED", "", "x", "/other"]
        row[index] = target[index - 1]
        value["commands"][2]["stdout"] = "|".join(row) + "\n"
    with pytest.raises(er.E0ReconciliationError, match="target identity"):
        sr._validate_observation(value, bound, record, NOW)


def test_cli_has_no_fixture_reader_or_enable_bypass():
    with pytest.raises(SystemExit):
        cli.main(
            [
                "close",
                "--control-directory",
                "/tmp/no",
                "--store-campaign-id",
                "x",
                "--verified",
                "true",
            ]
        )


class Time:
    elapsed = 0.0

    def clock(self):
        return NOW + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


def make_rejection_fixture(tmp_path, monkeypatch):
    """Real anonymous Git/model-manifest/prepared/SQLite; fake live OS observations only."""
    repo, config_path, config = service_fixture(tmp_path, monkeypatch)
    config.pop("memory_policy", None)
    config.update(cluster="n16r4", account="scze856", user="scze856",
                  resource_profile_version=2, memory_mib=65536, language_model_only=True)
    write(config_path, config)
    module = repo / sr._MODULE
    module.write_bytes(Path(sr.__file__).read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "anonymous rejection profile fixture")
    run = tmp_path / "prepared"
    prepared = prepare_historical_service_v2(repo, run, "campaign", "configs/path1_v1/site.json")
    descriptor, config, request = service.read_prepared(run, prepared["descriptor_sha256"])
    contract, probe = er._json(run / "contract.json"), er._json(run / "probe_request.json")
    identity = contract["runtime_identity"]
    scope = content_digest({"cluster": request.spec.cluster, "account": request.spec.account})
    store = CampaignStore(
        Path(config["control_root"]) / "ie1" / scope[:16], "executor_" + scope[:32]
    )
    invocation = store.directory / "i" / request.intent_id[:16]
    invocation.mkdir(parents=True)
    (invocation / "submitted_script.sh").write_bytes((run / "probe.sh").read_bytes())
    user = {"kind": "unix_effective_user", "uid": 123, "name": "scze856"}
    child = dict(
        pid=12345,
        host="fixture",
        platform="linux",
        start_marker="100",
        boot_id="boot",
        effective_uid=123,
    )
    pins = [
        {"path": str(repo / row["path"]), "sha256": row["sha256"]}
        for row in identity["core_files"]
        + identity["operations_files"]
        + [identity["config"], identity["dependency_lock"]]
    ]
    pins += [
        dict(path=probe["model_manifest_path"], sha256=identity["model"]["manifest_sha256"]),
        config["commands"]["sbatch"],
    ]
    manifest = dict(
        schema_version=1,
        nonce="f" * 32,
        request_sha256=request.request_sha256,
        argv=[config["commands"]["sbatch"]["path"], *request.sbatch_argv()[1:-1]],
        cwd=request.spec.work_dir,
        environment=dict(sr.SUBMISSION_ENV),
        stdin_path=str(invocation / "submitted_script.sh"),
        stdin_sha256=request.spec.script_sha256,
        input_files=pins,
        wait_seconds=60,
        command_timeout_seconds=60,
        deadline_monotonic=60,
        expected_effective_user=user,
    )
    write(invocation / "invocation.json", manifest)
    ready = dict(
        nonce=manifest["nonce"],
        request_sha256=request.request_sha256,
        manifest_sha256=er._sha(invocation / "invocation.json"),
        submitter=child,
        effective_user=user,
    )
    write(invocation / "ready.json", ready)
    write(invocation / "go.json", ready)
    result = dict(
        **ready,
        command_started_ns=int((NOW - timedelta(seconds=20)).timestamp() * 1e9),
        command_finished_ns=int((NOW - timedelta(seconds=19)).timestamp() * 1e9),
        returncode=1,
        timed_out=False,
        stdout=None,
        stdout_sha256=sr.EMPTY_SHA256,
        stderr_sha256=sr.POLICY_SHA256,
    )
    write(invocation / "result.json", result)
    cleanup = dict(
        binding=dict(
            campaign_id=request.campaign_id,
            intent_id=request.intent_id,
            job_id=None,
            source_sha256=content_digest(identity),
            role="submission_transport",
            service_instance_id=manifest["nonce"],
        ),
        identity={k: v for k, v in child.items() if k != "platform"},
        returncode=0,
        term_sent=True,
        kill_sent=False,
        leader_reaped=True,
        group_cleanup_verified=True,
        remaining_non_zombie_pids=[],
        errors=[],
    )
    cleanup["identity"].update(
        pgid=child["pid"],
        session_id=child["pid"],
        cgroup=["0::fixture"],
        slurm_job_id=None,
        step_id=None,
    )
    write(invocation / "local_cleanup.json", cleanup)
    record = dict(
        campaign_id=request.campaign_id,
        intent_id=request.intent_id,
        spec=request.spec.payload(),
        request_sha256=request.request_sha256,
        worker_request_sha256=probe["request_sha256"],
        status="submit_unknown",
        created_at=(NOW - timedelta(seconds=60)).isoformat(),
        parent_identity={**child, "pid": 12346},
        effective_user=user,
        invocation_dir=str(invocation),
        ready=ready,
        job_id=None,
        result_sha256=None,
        submitter_finished=True,
        local_cleanup=cleanup,
        manifest_sha256=ready["manifest_sha256"],
        probe_request_path=str(run / "probe_request.json"),
        probe_request_file_sha256=er._sha(run / "probe_request.json"),
    )
    ledger = dict(
        schema_version=1,
        cluster=request.spec.cluster,
        account=request.spec.account,
        intents={request.intent_id: record},
    )
    with store.lock():
        initial = store.initialize(registry={er.NAMESPACE: ledger}, state={})
        claim = store.claim_run(
            expected_generation=initial.generation,
            run_id=request.intent_id,
            intent_id=request.intent_id,
            run_directory=Path(request.spec.work_dir),
            metadata={"purpose": "slurm_job_container", "request_sha256": request.request_sha256},
        )
        store.start_claim(
            expected_generation=claim.updated_generation, run_id=claim.run_id, token=claim.token
        )
    Path(request.spec.work_dir).mkdir(parents=True)
    stderr = tmp_path / "matched_stderr.txt"
    stderr.write_bytes(sr.POLICY_STDERR)
    baseline = tmp_path / "original.files.tsv"
    rows = er._content(er._content_inventory(run, invocation))
    baseline.write_text(
        "relative_path\tsize_bytes\tsha256\n"
        + "".join(f"{row['path']}\t{row['size']}\t{row['sha256']}\n" for row in rows),
        encoding="utf-8",
    )
    timing = Time()
    verifier = copy.deepcopy(identity)
    verifier["model"] = None
    _seal_identity(verifier)
    monkeypatch.setattr(sr, "_verifier_identity", lambda *args: verifier)

    def gone(value):
        return dict(
            host="fixture",
            boot_id="boot",
            observations=[
                dict(
                    role=role,
                    identity=value[key]
                    if key == "parent_identity"
                    else value["ready"]["submitter"],
                    actual_start_marker=None,
                )
                for role, key in (("parent", "parent_identity"), ("submitter", "ready"))
            ],
        )

    monkeypatch.setattr(er, "_processes_gone", gone)

    class Reader:
        def read(self, request, config):
            rows = [
                dict(
                    argv=list(argv),
                    pin=config["commands"][Path(argv[0]).name],
                    returncode=0,
                    stdout="",
                    stderr="",
                )
                for argv in sr._read_argv(request, config)
            ]
            return sr.RejectionObservation(timing.clock(), timing.clock(), 123, tuple(rows))

    arguments = dict(
        descriptor_path=run / "submission.json",
        descriptor_sha256=prepared["descriptor_sha256"],
        original_code_sha=identity["core_sha"],
        baseline_manifest=baseline,
        baseline_manifest_sha256=er._sha(baseline),
        rejection_evidence_path=stderr,
        rejection_evidence_sha256=sr.POLICY_SHA256,
        reader=Reader(),
        clock=timing.clock,
        monotonic=timing.monotonic,
        sleep=timing.sleep,
    )
    return SimpleNamespace(
        store=store,
        args=arguments,
        record=record,
        request=request,
        run=run,
        repo=repo,
        config_path=config_path,
        invocation=invocation,
        timing=timing,
    )


linux = pytest.mark.skipif(
    sys.platform != "linux", reason="prepared service evidence uses Linux canonical Slurm paths"
)


@linux
def test_real_git_model_identity_sqlite_and_seal_release_only_technical_reservation(
    tmp_path, monkeypatch
):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    initial = data.store.read_registry()
    checked = sr.reconcile_rejection(data.store, **data.args, check_only=True)
    assert checked["claim_released"] is False and data.store.read_registry() == initial
    resolution = sr.reconcile_rejection(data.store, **data.args)
    final = data.store.read_registry()
    assert final.registry == initial.registry and final.generation == initial.generation + 1
    assert final.claims[0].status == "failed" and resolution.job_id is None
    assert resolution.retry_eligible is False and resolution.scientific_admission is False
    assert resolution.classification == "site_policy_rejected_before_job"
    assert sr.reconcile_rejection(data.store, **data.args) == resolution
    assert (
        er.validate_e0_terminal_receipt(data.record, final.claims[0], store=data.store)
        == resolution
    )
    assert resolution.validate(data.record, store=data.store) == resolution
    with pytest.raises(er.E0ReconciliationError, match="unissued"):
        replace(resolution, _issuer=None).validate(data.record, store=data.store)
    capacity_root = tmp_path / "capacity"
    capacity_root.mkdir()
    case = make_user_capacity_fixture(capacity_root, monkeypatch)
    request = make_request(2, campaign=data.request.campaign_id)
    ledger = final.registry[er.NAMESPACE]
    args = (case["snapshot"], request, ledger, 1, case["snapshot"].completed_at, "E0")
    with pytest.raises(capacity.SubmissionDisabled, match="no admission slot"):
        capacity.validate_user_association_capacity(*args)
    result = capacity.validate_user_association_capacity(
        *args, resolved_e0={data.request.intent_id: resolution}, store=data.store
    )
    assert result["account_occupancy"] == result["campaign_occupancy"] == 0
    with pytest.raises(capacity.SubmissionDisabled, match="typed durable"):
        capacity.validate_user_association_capacity(
            *args, resolved_e0={data.request.intent_id: {"verified": True}}, store=data.store
        )


@linux
@pytest.mark.parametrize(
    "defect",
    [
        "original_code",
        "stderr",
        "returncode",
        "stdout",
        "cleanup",
        "manifest",
        "manifest_v2",
        "result_v2",
        "output",
        "baseline",
    ],
)
def test_binding_failures_never_change_claim(tmp_path, monkeypatch, defect):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    initial = data.store.read_registry()
    if defect == "original_code":
        data.args["original_code_sha"] = "0" * 40
    elif defect == "stderr":
        data.args["rejection_evidence_path"].write_bytes(b"different")
    elif defect in {"returncode", "stdout"}:
        path = data.invocation / "result.json"
        value = er._json(path)
        value[defect] = 0 if defect == "returncode" else "123;n16r4"
        write(path, value)
    elif defect == "cleanup":
        path = data.invocation / "local_cleanup.json"
        value = er._json(path)
        value["group_cleanup_verified"] = False
        write(path, value)
    elif defect == "manifest":
        path = data.invocation / "invocation.json"
        value = er._json(path)
        value["argv"].append("--test-only")
        write(path, value)
    elif defect in {"manifest_v2", "result_v2"}:
        path = data.invocation / ("invocation.json" if defect == "manifest_v2" else "result.json")
        value = er._json(path)
        value["schema_version"] = 2
        if defect == "result_v2":
            value["capture_status"] = "error"
        write(path, value)
    elif defect == "output":
        (Path(data.request.spec.work_dir) / "slurm-999.out").write_bytes(b"ran")
    else:
        data.args["baseline_manifest_sha256"] = "0" * 64
    with pytest.raises((er.E0ReconciliationError, ValueError)):
        sr.reconcile_rejection(data.store, **data.args)
    assert data.store.read_registry() == initial


@linux
def test_old_source_is_proven_from_git_after_current_config_changes(tmp_path, monkeypatch):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    data.config_path.write_bytes(b"{}\n")
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "-qm", "later configuration, original Git blob retained")
    resolution = sr.reconcile_rejection(data.store, **data.args)
    assert resolution.original_code_sha == data.args["original_code_sha"]


@linux
def test_tampered_seal_refuses_validation_and_cas_crash_keeps_occupied(tmp_path, monkeypatch):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    initial = data.store.read_registry()
    finish = data.store.finish_claim
    monkeypatch.setattr(
        data.store, "finish_claim", lambda **kw: (_ for _ in ()).throw(RuntimeError("CAS crash"))
    )
    with pytest.raises(RuntimeError, match="CAS crash"):
        sr.reconcile_rejection(data.store, **data.args)
    assert data.store.read_registry() == initial
    monkeypatch.setattr(data.store, "finish_claim", finish)
    with pytest.raises(er.E0ReconciliationError, match="pending immutable audit"):
        sr.reconcile_rejection(data.store, **data.args)
    with pytest.raises(er.E0ReconciliationError):
        sr.resolve_rejection(data.store, data.request.intent_id)


@linux
def test_read_query_environment_is_not_the_original_submission_environment(tmp_path, monkeypatch):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    record = copy.deepcopy(data.record)
    manifest_path = data.invocation / "invocation.json"
    manifest = er._json(manifest_path)
    manifest["environment"] = dict(sr.READ_ENV)
    write(manifest_path, manifest)
    record["manifest_sha256"] = er._sha(manifest_path)
    record["ready"]["manifest_sha256"] = record["manifest_sha256"]
    for name in ("ready.json", "go.json", "result.json"):
        value = er._json(data.invocation / name)
        value["manifest_sha256"] = record["manifest_sha256"]
        write(data.invocation / name, value)
    with pytest.raises(er.E0ReconciliationError, match="literal argv/environment/script"):
        sr._bound_inputs(
            data.store,
            data.args["descriptor_path"],
            data.args["descriptor_sha256"],
            data.args["original_code_sha"],
            record,
            data.store.read_registry().claims[0],
        )


@linux
def test_terminal_capability_rechecks_seal_bytes_after_claim_release(tmp_path, monkeypatch):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    resolution = sr.reconcile_rejection(data.store, **data.args)
    evidence_path = sr._evidence_path(data.store, data.request.intent_id)
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    with pytest.raises(er.E0ReconciliationError, match="terminal seal differs"):
        resolution.validate(data.record, store=data.store)


@linux
@pytest.mark.parametrize("delay", [29, 31])
def test_slow_audit_publication_requires_fresh_scheduler_immediately_before_cas(
    tmp_path, monkeypatch, delay
):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    original_write = sr.write_exclusive_json

    def delayed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        data.timing.sleep(delay)

    monkeypatch.setattr(sr, "write_exclusive_json", delayed_write)
    if delay == 29:
        assert sr.reconcile_rejection(data.store, **data.args).retry_eligible is False
        assert data.store.read_registry().claims[0].status == "failed"
    else:
        before = data.store.read_registry()
        with pytest.raises(er.E0ReconciliationError, match="stale/future"):
            sr.reconcile_rejection(data.store, **data.args)
        assert data.store.read_registry() == before
        assert sr._evidence_path(data.store, data.request.intent_id).exists()


@linux
@pytest.mark.parametrize("defect", ["slow_inventory", "live_process"])
def test_readonly_check_revalidates_final_freshness_and_exited_processes(
    tmp_path, monkeypatch, defect
):
    data = make_rejection_fixture(tmp_path, monkeypatch)
    initial = data.store.read_registry()
    if defect == "slow_inventory":
        original = er._content_inventory

        def inventory(*args):
            value = original(*args)
            data.timing.sleep(31)
            return value

        monkeypatch.setattr(er, "_content_inventory", inventory)
    else:
        original = er._processes_gone

        def processes(record):
            value = original(record)
            value["observations"][0]["actual_start_marker"] = record["parent_identity"][
                "start_marker"
            ]
            return value

        monkeypatch.setattr(er, "_processes_gone", processes)
    with pytest.raises(er.E0ReconciliationError):
        sr.reconcile_rejection(data.store, **data.args, check_only=True)
    assert data.store.read_registry() == initial
    assert not sr._evidence_path(data.store, data.request.intent_id).exists()
