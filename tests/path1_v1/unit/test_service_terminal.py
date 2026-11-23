"""Real SQLite/archive/restore mechanics; scheduler and producer data are explicit fixtures."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import service_terminal as st
from agentmemeval.path1_v1.operations import terminal_common as tc
from agentmemeval.path1_v1.operations.contracts import canonical_json, content_digest
from agentmemeval.path1_v1.operations.runtime_probe import (
    RuntimeProbeError,
    verify_http_observation,
)
from agentmemeval.path1_v1.operations.service_profile import planned_service_argv
from path1_v1.unit.test_resource_terminal import make_resource_fixture, write


def make_service_fixture(tmp_path, monkeypatch):
    f = make_resource_fixture(tmp_path, monkeypatch)
    identity = f.bound["contract"]["runtime_identity"]
    identity["model"] = {
        "manifest": {"chat_template_sha256": hashlib.sha256(b"fixture").hexdigest()}
    }
    identity["operations_files"].append({"path": st._MODULE_PATH, "sha256": "b" * 64})
    contract, probe, config = (f.bound[k] for k in ("contract", "probe", "config"))
    contract.update(
        phase="E0",
        job={"kind": "service_probe"},
        source_contract={"provider": {"served_model_name": "fixture-model"}},
    )
    config.update(
        schema_version=1,
        kind="path1_e0_service_site_config",
        phase="E0",
        job_kind="service_probe",
        walltime_seconds=7200,
        command_seconds=6600,
        cleanup_seconds=600,
        startup_seconds=1200,
        gpus=1,
        stage_cap=1,
        cpus_per_task=8,
        memory_mib=65536,
        queue_policy="no_scheduler_deadline",
        qos_policy="site_default",
        resource_profile_version=2,
        language_model_only=True,
        service_port=18001,
        service_python="/fixture/python",
        model_root="/fixture/model",
    )
    probe["launch"]["service_python_identity"] = {
        "resolved_executable": "/fixture/python3",
        "executable_sha256": "c" * 64,
    }
    binding = dict(
        f.cleanup["binding"], role="model_service", source_sha256=content_digest(identity)
    )
    child = f.cleanup["identity"]
    argv = planned_service_argv(config, contract["source_contract"])
    f.launch = {
        "argv": argv,
        "binding": binding,
        "python_resolved": "/fixture/python3",
        "python_sha256": "c" * 64,
    }
    f.cleanup = dict(f.cleanup, binding=binding, returncode=-15)
    write(f.root / "launch.json", f.launch)
    write(f.root / "service_cleanup.json", f.cleanup)
    for name in ("service_stdout.log", "service_stderr.log"):
        (f.root / name).write_bytes(b"")
    out = f.root / "observations"
    host = {
        "hostname": child["host"],
        "boot_id": child["boot_id"],
        "euid": 17,
        "job_id": "42",
        "cgroup": "\n".join(child["cgroup"]),
        "pid": 200,
    }
    live = {"binding": binding, "identity": child, "cmdline": argv, "state": "S"}
    write(out / "host_service.json", {"host": host, "service": live})
    write(
        out / "endpoint.json",
        {
            "process": live,
            "base_url": "http://127.0.0.1:18001/v1",
            "listener_inodes": [123],
            "socket_holders": [[201, "2001", 123]],
            "network_namespace": "net:[123]",
        },
    )
    write(out / "service_arguments.json", {"effective_backend_settings_verified": False})
    write(out / "dependencies.json", {"scope": "explicit_offline_fixture"})
    write(out / "model_files.json", {"identity": identity["model"], "metadata": []})
    write(
        out / "tokenizer.json",
        {
            "chat_template": "fixture",
            "chat_template_sha256": identity["model"]["manifest"]["chat_template_sha256"],
        },
    )
    observed = []
    for index, name in enumerate(st._CASES):
        budget = (256, 8192, 256, 4096, 256, 256)[index]
        payload = {
            "model": "fixture-model",
            "seed": 2**62 + 1729 + index,
            "max_tokens": budget,
            "temperature": 1.0 if name in {"patch", "patch_repair"} else 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [],
            "response_format": {
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "patches": {"type": "array", "items": {"type": "object"}},
                        },
                    }
                }
            },
        }
        content = {"operation": "patch", "patches": [{}]} if name == "patch" else {}
        body = (
            {"error": {"message": "context"}}
            if name == "over_context"
            else {
                "model": "fixture-model",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
            }
        )
        raw = json.dumps(body)
        response = {
            "status": 400 if name == "over_context" else 200,
            "body_utf8": raw,
            "body_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "request_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        }
        write(out / (name + "_request.json"), payload)
        write(out / (name + "_response.json"), response)
        row = verify_http_observation(payload, response, 2, over_limit=name == "over_context")
        row.update(
            name=name,
            local_token_count=2,
            token_ids_sha256="d" * 64,
            output_budget=budget,
            seed_scope="E0_technical_fixture_only",
        )
        observed.append(row)
    f.receipt = {
        "schema_version": 1,
        "kind": "path1_live_e0_observation",
        "intent_id": probe["intent_id"],
        "probe_request_sha256": probe["request_sha256"],
        "campaign_id": "campaign",
        "runtime_admission": False,
        "status": "observations_complete_backend_execution_pending",
        "observations": observed,
    }
    f.refresh = lambda: refresh_inventory(f)
    f.refresh()
    monkeypatch.setattr(st, "_bound_inputs", er._bound_inputs)
    monkeypatch.setattr(st, "_verifier_identity", lambda repository, config: identity)
    f.close = lambda: st.reconcile_service_terminal(f.store, **f.kwargs)
    return f


def refresh_inventory(f):
    out = f.root / "observations"
    f.receipt["files"] = [
        {"path": p.name, "size": p.stat().st_size, "sha256": er._sha(p)}
        for p in sorted(out.iterdir())
        if p.name != "probe_receipt.json" and p.is_file()
    ]
    write(out / "probe_receipt.json", f.receipt)


@pytest.fixture
def service(tmp_path, monkeypatch):
    return make_service_fixture(tmp_path, monkeypatch)


def test_complete_service_term_cleanup_closes_exact_claim_without_retry_or_science(service):
    before = service.store.read_registry()
    value = service.close()
    after = service.store.read_registry()
    assert value.claim_status == "completed" and value.service_observation_complete
    assert after.generation == before.generation + 1 and after.registry == before.registry
    assert value.runtime_admission is value.scientific_admission is value.retry_eligible is False
    assert value.scientific_samples == 0 and not hasattr(value, "eligibility_receipt_sha256")
    assert value.validate(service.record, store=service.store) == value
    assert service.close() == value and service.store.read_registry() == after


def test_readonly_does_not_publish_or_finish(service):
    before = service.store.read_registry()
    result = st.check_service_terminal(
        service.store,
        **{
            k: service.kwargs[k]
            for k in ("descriptor_path", "descriptor_sha256", "reader", "clock")
        },
    )
    assert result["facts"]["service_observation_complete"] and not result["technical_claim_closed"]
    assert before == service.store.read_registry()
    assert not st._audit_path(service.store, service.request.intent_id).parent.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "group",
        "reaped",
        "remaining",
        "error",
        "uid_bool",
        "pid_bool",
        "session",
        "cgroup",
        "job",
        "instance",
        "source",
    ],
)
def test_cleanup_uncertainty_or_identity_conflict_keeps_started(service, defect, monkeypatch):
    cleanup = copy.deepcopy(service.cleanup)
    if defect == "missing":
        original = er._json

        def missing(path):
            if path == service.root / "service_cleanup.json":
                raise FileNotFoundError(path)
            return original(path)

        monkeypatch.setattr(er, "_json", missing)
    elif defect == "group":
        cleanup["group_cleanup_verified"] = False
    elif defect == "reaped":
        cleanup["leader_reaped"] = False
    elif defect == "remaining":
        cleanup["remaining_non_zombie_pids"] = [999]
    elif defect == "error":
        cleanup["errors"] = ["uncertain"]
    elif defect == "uid_bool":
        cleanup["identity"]["effective_uid"] = True
    elif defect == "pid_bool":
        cleanup["identity"]["pid"] = True
    elif defect == "session":
        cleanup["identity"]["session_id"] += 1
    elif defect == "cgroup":
        cleanup["identity"]["cgroup"] = ["0::/job_420/step_batch"]
    elif defect == "job":
        cleanup["identity"]["slurm_job_id"] = "43"
    elif defect == "instance":
        cleanup["binding"]["service_instance_id"] = "f" * 32
    elif defect == "source":
        cleanup["binding"]["source_sha256"] = "f" * 64
    write(service.root / "service_cleanup.json", cleanup)
    with pytest.raises((ValueError, FileNotFoundError)):
        service.close()
    assert service.store.read_registry().claims[0].status == "started"


@pytest.mark.parametrize(
    "defect",
    [
        "argv",
        "python",
        "live_identity",
        "live_binding",
        "host",
        "endpoint",
        "receipt_intent",
        "receipt_science",
        "raw_hash",
        "schema_result",
        "model",
        "tokenizer",
        "inventory",
        "seed",
    ],
)
def test_bound_raw_identity_or_content_cannot_be_relabelled(service, defect):
    out = service.root / "observations"
    if defect in {"argv", "python"}:
        launch = copy.deepcopy(service.launch)
        if defect == "argv":
            launch["argv"].append("--enable-prefix-caching")
        else:
            launch["python_sha256"] = "0" * 64
        write(service.root / "launch.json", launch)
    elif defect.startswith("live_") or defect == "host":
        data = er._json(out / "host_service.json")
        if defect == "live_identity":
            data["service"]["identity"]["start_marker"] = "9000"
        elif defect == "live_binding":
            data["service"]["binding"]["service_instance_id"] = "f" * 32
        else:
            data["host"]["boot_id"] = "another-boot"
        write(out / "host_service.json", data)
    elif defect == "endpoint":
        data = er._json(out / "endpoint.json")
        data["process"]["identity"]["pid"] = 999
        write(out / "endpoint.json", data)
    elif defect == "receipt_intent":
        service.receipt["intent_id"] = "e" * 64
    elif defect == "receipt_science":
        service.receipt["runtime_admission"] = True
    elif defect == "raw_hash":
        data = er._json(out / "action_response.json")
        data["body_sha256"] = "0" * 64
        write(out / "action_response.json", data)
    elif defect == "schema_result":
        service.receipt["observations"][0]["content"] = {"invented": True}
    elif defect == "model":
        write(out / "model_files.json", {"identity": {}})
    elif defect == "tokenizer":
        write(out / "tokenizer.json", {"chat_template": "other", "chat_template_sha256": "0" * 64})
    elif defect == "seed":
        data = er._json(out / "action_request.json")
        data["seed"] += 1
        write(out / "action_request.json", data)
    elif defect == "inventory":
        (out / "unlisted.json").write_bytes(b"{}")
    if defect != "inventory":
        service.refresh()
    with pytest.raises((ValueError, RuntimeProbeError)):
        service.close()
    assert service.store.read_registry().claims[0].status == "started"


@pytest.mark.parametrize(
    "outcome", ["http_failure", "allocation_failed", "kill_cleanup", "incomplete"]
)
def test_known_failure_with_verified_release_closes_failed_without_retry(service, outcome):
    if outcome in {"http_failure", "incomplete"}:
        service.receipt.update(status="incomplete", failure_type="RuntimeProbeError")
        service.refresh()
    if outcome == "http_failure":
        write(
            service.root / "service_failure.json",
            {
                "failure_type": "RuntimeProbeError",
                "request_sha256": service.bound["probe"]["request_sha256"],
                "runtime_admission": False,
            },
        )
    if outcome == "allocation_failed":
        service.scheduler[0][5:7] = ["FAILED", "1:0"]
    if outcome == "kill_cleanup":
        service.cleanup.update(returncode=-9, kill_sent=True)
        write(service.root / "service_cleanup.json", service.cleanup)
    value = service.close()
    assert value.claim_status == "failed" and not value.retry_eligible


@pytest.mark.parametrize("defect", ["active", "nonterminal", "missing_step"])
def test_scheduler_release_required(service, defect):
    if defect == "nonterminal":
        service.scheduler[0][5] = "RUNNING"
    elif defect == "missing_step":
        service.scheduler.pop()
    else:
        original = service.kwargs["reader"]

        class Reader:
            def read(self, *args):
                value = original.read(*args)
                commands = copy.deepcopy(value.commands)
                commands[1]["stdout"] = "42.batch\n"
                return replace(value, commands=commands)

        service.kwargs["reader"] = Reader()
    with pytest.raises(ValueError):
        service.close()
    assert service.store.read_registry().claims[0].status == "started"


def test_crash_between_seal_and_cas_does_not_release_or_overwrite(service, monkeypatch):
    monkeypatch.setattr(
        service.store, "finish_claim", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crash"))
    )
    with pytest.raises(RuntimeError, match="crash"):
        service.close()
    assert service.store.read_registry().claims[0].status == "started"
    before = st._audit_path(service.store, service.request.intent_id).read_bytes()
    with pytest.raises(FileExistsError):
        service.close()
    assert st._audit_path(service.store, service.request.intent_id).read_bytes() == before


@pytest.mark.parametrize("target", ["original", "archive", "restored", "audit"])
def test_replay_rechecks_every_original_archive_restore_and_seal(service, target):
    value = service.close()
    path = st._audit_path(service.store, service.request.intent_id)
    if target == "original":
        dest = service.root / "service_stdout.log"
    elif target == "archive":
        dest = path.parent / "run/snapshot.tar.gz"
    elif target == "restored":
        dest = path.parent / "run/restored/run/probe.sh"
    else:
        dest = path
    dest.write_bytes(dest.read_bytes() + b"x")
    with pytest.raises((ValueError, OSError, KeyError)):
        value.validate(service.record, store=service.store)


def test_constructed_resolution_does_not_authorize_capacity(service):
    value = service.close()
    with pytest.raises(ValueError, match="not issued"):
        replace(value, _issuer=None).validate(service.record, store=service.store)


@pytest.mark.parametrize("defect", ["missing", "digest", "origin"])
def test_new_common_source_is_pinned(monkeypatch, tmp_path, defect):
    from pathlib import Path

    source = Path(tc.__file__).resolve()
    repo = source.parents[4]
    rows = [{"path": tc._MODULE_PATH, "sha256": er._sha(source)}]
    if defect == "missing":
        rows = []
    if defect == "digest":
        rows[0]["sha256"] = "0" * 64
    if defect == "origin":
        repo = tmp_path
    with pytest.raises(ValueError):
        tc.verify_implementation_source(repo, {"operations_files": rows})


def test_service_cli_verify_and_no_verified_bypass(service, monkeypatch, capsys):
    from agentmemeval.path1_v1.operations import service_terminal_cli as cli

    value = service.close()
    monkeypatch.setattr(cli, "CampaignStore", lambda *args: service.store)
    args = [
        "verify",
        "--control-directory",
        str(service.store.directory),
        "--store-campaign-id",
        service.store.campaign_id,
        "--intent-id",
        service.request.intent_id,
    ]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["terminal_receipt_sha256"] == value.terminal_receipt_sha256
    assert "_issuer" not in result and result["runtime_admission"] is False
    with pytest.raises(SystemExit):
        cli.main(args + ["--verified", "true"])


@pytest.mark.parametrize("failed", [False, True])
def test_service_typed_release_capacity_and_post_archive_freshness(
    service, tmp_path, monkeypatch, failed
):
    from datetime import timedelta

    from agentmemeval.path1_v1.operations import user_capacity as capacity
    from agentmemeval.path1_v1.operations.intent_executor import DurableIntentExecutor
    from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, SubmissionRequest
    from path1_v1.unit.test_resource_terminal import NOW
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
        clock=service.kwargs["clock"],
        transport=FixtureTransport(outputs),
    )
    executor = object.__new__(DurableIntentExecutor)
    executor.store, executor.account_reader = service.store, reader.read
    executor.clock = service.kwargs["clock"]
    executor.cluster, executor.account = "cluster", "account"
    executor._path = er._path
    request = SubmissionRequest(
        "campaign", "9" * 64, replace(service.request.spec, work_dir="/other/output")
    )

    def capacity_now():
        snapshot = service.store.read_registry()
        return executor._capacity(request, snapshot.registry[er.NAMESPACE], 1, validated_phase="E0")

    with pytest.raises(SubmissionDisabled, match="no admission"):
        capacity_now()
    if failed:
        service.receipt.update(status="incomplete", failure_type="RuntimeProbeError")
        service.refresh()
    result = service.close()
    assert result.claim_status == ("failed" if failed else "completed")
    snapshot = service.store.read_registry()
    assert executor._ledger(snapshot) == snapshot.registry[er.NAMESPACE]
    assert capacity_now()["account_occupancy"] == 0
    original = st.ServiceTerminalResolution.validate
    delay = {"seconds": 29}

    def slow(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        service.kwargs["sleep"](delay["seconds"])
        return result

    monkeypatch.setattr(st.ServiceTerminalResolution, "validate", slow)
    assert capacity_now()["account_occupancy"] == 0
    delay["seconds"] = 31
    with pytest.raises(SubmissionDisabled, match="outlived snapshot freshness"):
        capacity_now()
    audit = st._audit_path(service.store, service.request.intent_id)
    audit.write_bytes(audit.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        executor._ledger(snapshot)


def test_model_historical_source_scope_is_explicit_and_does_not_rehash_model(tmp_path):
    from path1_v1.unit.test_operations_contracts import evidence

    fixture = evidence.__wrapped__(tmp_path)
    identity = fixture["identity"]
    with pytest.raises(ValueError, match="cannot validate a model"):
        er._historical_identity(fixture["repo"], identity)
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="invalid historical model scope"):
            er._historical_identity(fixture["repo"], identity, allow_model=invalid)
    assert er._historical_identity(fixture["repo"], identity, allow_model=True) == {"setting": 1}
    (fixture["model"] / "weights.bin").write_bytes(b"modified toy weights; never loaded")
    assert er._historical_identity(fixture["repo"], identity, allow_model=True) == {"setting": 1}
    damaged = copy.deepcopy(identity)
    damaged["core_files"][0]["sha256"] = hashlib.sha256(b"different historical fixture").hexdigest()
    with pytest.raises(ValueError, match="content"):
        er._historical_identity(fixture["repo"], damaged, allow_model=True)


@pytest.mark.parametrize("kind", ["service", "resource"])
@pytest.mark.parametrize("stage", ["inventory", "facts", "state", "publish"])
@pytest.mark.parametrize("delay", [29, 31])
def test_terminal_freshness_after_last_io_and_before_cas(tmp_path, monkeypatch, kind, stage, delay):
    from agentmemeval.path1_v1.operations import resource_terminal as rt

    fixture = (make_service_fixture if kind == "service" else make_resource_fixture)(
        tmp_path, monkeypatch
    )
    owner = st if kind == "service" else rt
    calls = {"count": 0}
    if stage == "inventory":
        module, name, when = er, "_content_inventory", 3
    elif stage == "facts":
        module, name, when = owner, "_service_facts" if kind == "service" else "_resource_facts", 2
    elif stage == "state":
        module, name, when = er, "_stored", 2
    else:
        module, name, when = tc, "write_exclusive_json", 1
    original = getattr(module, name)

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == when:
            fixture.kwargs["sleep"](delay)
        return result

    monkeypatch.setattr(module, name, slow)
    before = fixture.store.read_registry()
    if delay == 29:
        assert fixture.close().claim_status == "completed"
    else:
        with pytest.raises(ValueError, match="fresh|stale"):
            fixture.close()
        assert fixture.store.read_registry() == before
        assert fixture.store.read_registry().claims[0].status == "started"


@pytest.mark.parametrize("delay", [29, 31])
def test_readonly_freshness_after_inventory(service, monkeypatch, delay):
    original = er._content_inventory

    def slow(*args):
        result = original(*args)
        service.kwargs["sleep"](delay)
        return result

    monkeypatch.setattr(er, "_content_inventory", slow)
    kwargs = {
        k: service.kwargs[k] for k in ("descriptor_path", "descriptor_sha256", "reader", "clock")
    }
    if delay == 29:
        result = st.check_service_terminal(service.store, **kwargs)
        assert result["checked_at"] == service.kwargs["clock"]().isoformat()
    else:
        with pytest.raises(ValueError, match="fresh|stale"):
            st.check_service_terminal(service.store, **kwargs)
    assert service.store.read_registry().claims[0].status == "started"
