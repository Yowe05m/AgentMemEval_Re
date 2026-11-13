"""Pure schema tests plus opt-in, byte-preserving real E0 archive replay.

PATH1_E0_DEADLINE_ARCHIVE points at the external final-result directory, never
at a Git fixture. Missing archival data skips only the explicit replay cases.
Remote paths, live /proc and current verifier publication are offline adapters;
they are not claims that this test performed a remote closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.contracts import build_runtime_identity, content_digest
from agentmemeval.path1_v1.operations.slurm import CommandResult, SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.state import CampaignStore

ARCHIVE_SHA = "81942e6d9ccf0a9b3389ca03806786e05d20b8fae3f979302777f49eb8bc2ff2"
BASELINE_SHA = "49debe4a290b98cb4e06b7184878e23c4c4bb63783dea6089ef6404d17e29b8b"
NOW = datetime(2026, 11, 13, 17, tzinfo=UTC)


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def now(self):
        return NOW + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def make_archive_fixture(tmp_path, monkeypatch, *, archive_directory=None):
    """Return store/request/record, close(), and explicit offline reader adapters."""
    configured = archive_directory or os.environ.get("PATH1_E0_DEADLINE_ARCHIVE")
    if configured is None:
        pytest.skip("external actual E0 archive not configured; no raw results stored in Git")
    archive_directory = Path(configured)
    archive = archive_directory / "20261114_e0_deadline.tar.gz"
    baseline = archive_directory / "20261114_e0_deadline.files.tsv"
    assert er._sha(archive) == ARCHIVE_SHA
    assert er._sha(baseline) == BASELINE_SHA
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        assert sum(member.isfile() for member in members) == 37
        for member in members:
            assert (member.isfile() or member.isdir()) and not member.name.startswith("/")
            assert ".." not in Path(member.name).parts
            path = extracted / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as stream:
                path.write_bytes(stream.read())
    extracted = extracted / "evidence"
    # The archive has only files; recreate the empty job container from the descriptor.
    run = extracted / "run"
    descriptor_path = run / "submission.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    request = SubmissionRequest(
        descriptor["campaign_id"],
        descriptor["intent_id"],
        SubmitSpec.from_payload(descriptor["spec"]),
    )
    remote_run = str(Path(request.spec.script_path).parent).replace("\\", "/")
    remote_repo = descriptor["repository_root"]
    contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[3]
    # Exact historical config bytes are obtained from the published Git object, not HEAD.
    config_bytes = _git(
        repository,
        "show",
        contract["runtime_identity"]["core_sha"] + ":" + descriptor["config_relative"],
    )
    assert hashlib.sha256(config_bytes).hexdigest() == descriptor["config_sha256"]
    config = json.loads(config_bytes)
    remote_control = config["control_root"]
    scope = content_digest({"cluster": request.spec.cluster, "account": request.spec.account})
    directory = tmp_path / "control" / "ie1" / scope[:16]
    directory.mkdir(parents=True)
    shutil.copyfile(extracted / "executor/campaign.sqlite3", directory / "campaign.sqlite3")
    invocation = directory / "i" / request.intent_id[:16]
    invocation.parent.mkdir()
    shutil.copytree(extracted / "executor/invocation", invocation)
    store = CampaignStore(directory, "executor_" + scope[:32])
    snapshot = store.read_registry()
    record = snapshot.registry[er.NAMESPACE]["intents"][request.intent_id]
    actual_path = er._path

    def map_path(value):
        value = str(value).replace("\\", "/")
        for prefix, local in (
            (remote_run, run),
            (remote_control, tmp_path / "control"),
            (remote_repo, repository),
        ):
            if value == prefix or value.startswith(prefix + "/"):
                return actual_path(local / value[len(prefix) :].lstrip("/"))
        return actual_path(Path(value))

    work_relative = request.spec.work_dir[len(remote_run) :].lstrip("/")
    (run / work_relative).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(er, "_path", map_path)
    historical = er._historical_identity
    # Validate every real historical Git blob once, then cache only that exact identity.
    assert historical(repository, contract["runtime_identity"]) == config
    frozen_identity = copy.deepcopy(contract["runtime_identity"])

    def replay_historical(repo, identity):
        assert repo == repository and identity == frozen_identity
        return copy.deepcopy(config)

    monkeypatch.setattr(er, "_historical_identity", replay_historical)
    monkeypatch.setattr(er, "_verifier_identity", lambda repo, cfg: copy.deepcopy(frozen_identity))
    observation = json.loads(
        (extracted / "observations/closure_observation.json").read_text(encoding="utf-8")
    )
    originals = observation["second_scheduler"]["commands"]
    assert originals[0]["omitted_other_job_rows"] == originals[1]["omitted_other_job_rows"] == 0
    assert originals[0]["stdout"] == originals[1]["stdout"] == ""
    accounting = originals[2]["stdout"].strip().split("|")
    assert len(accounting) == 14 and accounting[7] == "00:00:00"
    # New reader requests ElapsedRaw and Submit. Submit is in preserved initial scontrol.
    initial = (extracted / "observations/initial_remote_observation.json").read_text(
        encoding="utf-8"
    )
    assert "SubmitTime=2026-11-13T16:33:21" in initial
    accounting[7] = "0"
    accounting.insert(8, "2026-11-13T16:33:21")
    clock = Clock()

    class Reader:
        def read(self, requested, job_id, cfg):
            assert requested == request and job_id == record["job_id"] and cfg == config
            commands = []
            for index, argv in enumerate(er._read_argv(request, job_id, config)):
                commands.append(
                    {
                        "argv": list(argv),
                        "pin": config["commands"][Path(argv[0]).name],
                        "returncode": 0,
                        "stderr": "",
                        "stdout": "|".join(accounting) + "\n" if index == 2 else "",
                    }
                )
            return er.DeadlineObservation(
                clock.now(), clock.now(), record["effective_user"]["uid"], tuple(commands)
            )

    processes = {
        "host": record["parent_identity"]["host"],
        "boot_id": record["parent_identity"]["boot_id"],
        "observations": [],
    }
    for row in observation["second_processes"]:
        assert row["proc_exists"] is False
        processes["observations"].append(
            {"role": row["role"], "identity": row["identity"], "actual_start_marker": None}
        )
    monkeypatch.setattr(er, "_processes_gone", lambda rec: copy.deepcopy(processes))
    kwargs = dict(
        descriptor_path=descriptor_path,
        descriptor_sha256=er._sha(descriptor_path),
        baseline_manifest=baseline,
        baseline_manifest_sha256=BASELINE_SHA,
        reader=Reader(),
        clock=clock.now,
    )
    return SimpleNamespace(
        store=store,
        request=request,
        record=record,
        claim=snapshot.claims[0],
        run=run,
        invocation=invocation,
        descriptor=descriptor,
        config=config,
        kwargs=kwargs,
        clock=clock,
        processes=processes,
        accounting=accounting,
        close=lambda: er.reconcile_unstarted_deadline(
            store, **kwargs, monotonic=clock.monotonic, sleep=clock.sleep
        ),
    )


@pytest.fixture
def replay(tmp_path, monkeypatch):
    return make_archive_fixture(tmp_path, monkeypatch)


def test_actual_archive_close_keeps_accepted_history_and_advances_claim_once(replay):
    before = replay.store.read_registry()
    resolution = replay.close()
    after = replay.store.read_registry()
    assert before.generation == 7 and after.generation == 8
    assert after.registry == before.registry
    assert after.claims[0].status == "failed"
    assert resolution.scientific_admission is False
    assert resolution.classification == "unstarted_scheduler_deadline"
    assert resolution.validate(replay.record, store=replay.store) == resolution
    assert replay.close() == resolution
    assert replay.store.read_registry() == after


def test_actual_archive_readonly_check_does_not_release_or_create_audit(replay):
    before = replay.store.read_registry()
    result = er.check_unstarted_deadline(replay.store, **replay.kwargs)
    assert result["claim_released"] is False and result["scientific_admission"] is False
    assert replay.store.read_registry() == before
    assert not er._audit_path(replay.store, replay.request.intent_id).exists()


@pytest.mark.parametrize("filename", ["probe_request.json", "contract.json", "probe.sh"])
def test_actual_input_byte_changes_rejected(replay, filename):
    path = replay.run / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((er.E0ReconciliationError, ValueError)):
        replay.close()
    assert replay.store.read_registry().claims[0].status == "started"


@pytest.mark.parametrize(
    "filename", ["invocation.json", "ready.json", "go.json", "result.json", "local_cleanup.json"]
)
def test_actual_invocation_missing_file_rejected(replay, filename):
    (replay.invocation / filename).unlink()
    with pytest.raises((er.E0ReconciliationError, FileNotFoundError)):
        replay.close()


def test_actual_resource_output_forbids_never_started_classification(replay):
    target = er._path(replay.request.spec.work_dir) / "slurm.out"
    target.write_bytes(b"worker started")
    with pytest.raises(er.E0ReconciliationError, match="never-started"):
        replay.close()


def test_actual_artifacts_are_rechecked_after_closure(replay):
    resolution = replay.close()
    (replay.invocation / "submitted_script.sh").write_bytes(b"changed")
    with pytest.raises(er.E0ReconciliationError):
        resolution.validate(replay.record, store=replay.store)


def test_actual_evidence_tamper_is_not_accepted_from_terminal_boolean(replay):
    resolution = replay.close()
    evidence = er._audit_path(replay.store, replay.request.intent_id)
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(er.E0ReconciliationError, match="audit changed"):
        resolution.validate(replay.record, store=replay.store)


def test_actual_crash_after_audit_before_cas_leaves_slot_occupied(replay, monkeypatch):
    before = replay.store.read_registry()

    def crash(**kwargs):
        raise OSError("injected durable commit failure")

    monkeypatch.setattr(replay.store, "finish_claim", crash)
    with pytest.raises(OSError, match="injected"):
        replay.close()
    path = er._audit_path(replay.store, replay.request.intent_id)
    assert path.is_file()
    original = path.read_bytes()
    assert replay.store.read_registry() == before
    with pytest.raises(FileExistsError):
        replay.close()
    assert path.read_bytes() == original
    with pytest.raises(er.E0ReconciliationError, match="failed claim"):
        er.resolve_e0_terminal_intent(replay.store, replay.request.intent_id)


def test_actual_fabricated_resolution_is_rejected(replay):
    issued = replay.close()
    forged = replace(issued, _issuer=None)
    with pytest.raises(er.E0ReconciliationError, match="not issued"):
        forged.validate(replay.record, store=replay.store)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "submit_unknown"),
        ("submitter_finished", False),
        ("cleanup_persistence_failed", True),
        ("job_id", "999"),
    ],
)
def test_actual_accepted_record_cannot_be_replaced_by_unknown_or_other_job(replay, field, value):
    with replay.store.lock():
        before = replay.store.read_registry()
        registry = copy.deepcopy(before.registry)
        registry[er.NAMESPACE]["intents"][replay.request.intent_id][field] = value
        replay.store.commit_transition(
            expected_generation=before.generation,
            registry=registry,
            state=before.state,
            reason="offline fault injection",
        )
    with pytest.raises(er.E0ReconciliationError):
        replay.close()


def test_actual_writer_identity_present_blocks_closure(replay):
    replay.processes["observations"][0]["actual_start_marker"] = replay.record["parent_identity"][
        "start_marker"
    ]
    with pytest.raises(er.E0ReconciliationError, match="process exit"):
        replay.close()


def test_actual_quiet_period_detects_new_artifact(replay):
    original_sleep = replay.clock.sleep

    def mutate(seconds):
        original_sleep(seconds)
        (replay.run / "unexpected.txt").write_bytes(b"new writer output")

    with pytest.raises(er.E0ReconciliationError, match="not quiescent"):
        er.reconcile_unstarted_deadline(
            replay.store, **replay.kwargs, monotonic=replay.clock.monotonic, sleep=mutate
        )


def test_actual_archive_v2_snapshot_binding_is_checked(replay, tmp_path):
    """Synthetic v2 envelope over the unchanged historical attempt-1 transport."""
    descriptor = dict(
        replay.descriptor,
        schema_version=2,
        attempt=1,
        queue_start_window_seconds=10,
        scheduler_deadline_offset_seconds=130,
    )
    path = replay.kwargs["descriptor_path"]
    _write(path, descriptor)
    config_bytes = _git(
        Path(__file__).resolve().parents[3],
        "show",
        "05279143805355b5999e3cd29f5fa9a54ad8e22e:" + descriptor["config_relative"],
    )
    (replay.run / "site_config.json").write_bytes(config_bytes)
    # This synthetic case has a new explicit trusted baseline; it is not the original archive.
    baseline = tmp_path / "synthetic-v2.tsv"
    rows = er._content(er._content_inventory(replay.run, replay.invocation))
    baseline.write_text(
        "relative_path\tsize_bytes\tsha256\n"
        + "".join(f"{row['path']}\t{row['size']}\t{row['sha256']}\n" for row in rows),
        encoding="utf-8",
    )
    replay.kwargs.update(
        descriptor_sha256=er._sha(path),
        baseline_manifest=baseline,
        baseline_manifest_sha256=er._sha(baseline),
    )
    assert er.check_unstarted_deadline(replay.store, **replay.kwargs)["claim_released"] is False
    (replay.run / "site_config.json").write_bytes(config_bytes + b"\n")
    with pytest.raises(er.E0ReconciliationError, match="v2 frozen"):
        replay.close()


def synthetic_scheduler():
    spec = SimpleNamespace(cluster="test", user="person", account="lab", work_dir="/work/run")
    request = SimpleNamespace(spec=spec, campaign_id="campaign", intent_id="1" * 64)
    record = {
        "job_id": "42",
        "effective_user": {"uid": 17},
        "created_at": "2026-11-13T16:00:00+00:00",
    }
    config = {
        "commands": {
            name: {"path": "/usr/bin/" + name, "sha256": "2" * 64} for name in ("squeue", "sacct")
        },
        "accounting_since": "2026-11-13T00:00:00+00:00",
    }
    accounting = [
        "test",
        "42",
        "person",
        "lab",
        er.job_name("campaign", "1" * 64),
        "DEADLINE",
        "1:0",
        "0",
        "2026-11-13T16:00:10",
        "None",
        "2026-11-13T16:05:00",
        "None assigned",
        "",
        er.job_comment("campaign", "1" * 64),
        "/work/run",
    ]
    rows = [
        {
            "argv": list(argv),
            "pin": config["commands"][Path(argv[0]).name],
            "returncode": 0,
            "stderr": "",
            "stdout": "|".join(accounting) + "\n" if index == 2 else "",
        }
        for index, argv in enumerate(er._read_argv(request, "42", config))
    ]
    observation = er.DeadlineObservation(NOW, NOW, 17, tuple(rows)).payload()
    return request, record, config, observation


def test_anonymous_scheduler_exact_unstarted_evidence():
    request, record, config, observation = synthetic_scheduler()
    er._validate_scheduler(observation, request, record, config, sealed_at=NOW)
    assert "--format=%i" in observation["commands"][1]["argv"]
    assert all("+00:00" not in arg for arg in observation["commands"][2]["argv"])


@pytest.mark.parametrize(
    "column,value",
    [
        (0, "other"),
        (1, "43"),
        (2, "other"),
        (3, "other"),
        (4, "other"),
        (5, "COMPLETED"),
        (6, "bad"),
        (7, "1"),
        (8, "2026-11-13T15:00:00"),
        (9, "2026-11-13T16:01:00"),
        (10, "2026-11-13T18:00:00"),
        (11, "gpu01"),
        (12, "gres/gpu=1"),
        (13, "other"),
        (14, "/other"),
    ],
)
def test_scheduler_refuses_wrong_binding_or_any_start_allocation(column, value):
    request, record, config, observation = synthetic_scheduler()
    fields = observation["commands"][2]["stdout"].strip().split("|")
    fields[column] = value
    observation["commands"][2]["stdout"] = "|".join(fields) + "\n"
    with pytest.raises(er.E0ReconciliationError):
        er._validate_scheduler(observation, request, record, config, sealed_at=NOW)


@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "future",
        "uid",
        "booluid",
        "stderr",
        "rc",
        "filter",
        "queue",
        "step",
        "compressedstep",
        "duplicate",
        "historicalstep",
    ],
)
def test_scheduler_fails_closed_for_observation_defects(defect):
    request, record, config, observation = synthetic_scheduler()
    rows = observation["commands"]
    if defect == "stale":
        observation["observed_at"] = (NOW - timedelta(seconds=31)).isoformat()
    elif defect == "future":
        observation["completed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif defect in {"uid", "booluid"}:
        observation["effective_uid"] = 18 if defect == "uid" else True
    elif defect == "stderr":
        rows[1]["stderr"] = "Invalid job id"
    elif defect == "rc":
        rows[1]["returncode"] = 1
    elif defect == "filter":
        rows[0]["argv"].append("--name=only_my_campaign")
    elif defect == "queue":
        rows[0]["stdout"] = "42|person|lab|name|DEADLINE|comment|time|/work/run\n"
    elif defect == "step":
        rows[1]["stdout"] = "42.extern\n"
    elif defect == "compressedstep":
        rows[1]["stdout"] = "42_[1-4].batch\n"
    elif defect == "duplicate":
        rows[2]["stdout"] *= 2
    else:
        rows[2]["stdout"] += "42.batch|COMPLETED\n"
    with pytest.raises(er.E0ReconciliationError):
        er._validate_scheduler(observation, request, record, config, sealed_at=NOW)


def test_reader_pins_clean_environment_and_rechecks_identity(monkeypatch):
    request, record, config, observation = synthetic_scheduler()
    calls = []
    validations = []
    monkeypatch.setattr(
        er.PinnedUserCommand, "validate", lambda self, name: validations.append(name)
    )
    monkeypatch.setattr(er, "_effective_identity", lambda user: 17)

    class Transport:
        def run(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return CommandResult(0, observation["commands"][len(calls) - 1]["stdout"], "")

    result = er.E0DeadlineReader(transport=Transport(), clock=lambda: NOW).read(
        request, "42", config
    )
    assert result.payload() == observation
    assert validations == ["squeue", "squeue", "squeue", "squeue", "sacct", "sacct"]
    assert all(
        kwargs
        == {"environment": dict(er.READ_ENV), "inherit_environment": False, "timeout_seconds": 8}
        for _, kwargs in calls
    )


@pytest.mark.parametrize("failure", ["stderr", "exit", "timeout", "uid_changed"])
def test_reader_rejects_runtime_observation_failures(monkeypatch, failure):
    request, _, config, _ = synthetic_scheduler()
    identities = iter([17, 18 if failure == "uid_changed" else 17])
    monkeypatch.setattr(er, "_effective_identity", lambda user: next(identities))
    monkeypatch.setattr(er.PinnedUserCommand, "validate", lambda self, name: None)

    class Transport:
        def run(self, argv, **kwargs):
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 8)
            return CommandResult(
                1 if failure == "exit" else 0,
                "",
                "unsupported field" if failure == "stderr" else "",
            )

    with pytest.raises((er.E0ReconciliationError, subprocess.TimeoutExpired)):
        er.E0DeadlineReader(transport=Transport(), clock=lambda: NOW).read(request, "42", config)


@pytest.mark.parametrize("action", ["check", "close"])
def test_cli_requires_pinned_inputs_and_has_no_fixture_bypass(action):
    from agentmemeval.path1_v1.operations.e0_reconcile_cli import main

    with pytest.raises(SystemExit) as caught:
        main([action, "--control-directory", "/does-not-exist", "--store-campaign-id", "campaign"])
    assert caught.value.code == 2


def test_historical_git_content_not_replaced_by_current_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative, payload in {
        "src/core.py": b"VALUE = 1\n",
        "src/ops.py": b"VALUE = 2\n",
        "config.json": b'{"answer": 42}',
        "lock.json": b"{}",
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    identity = build_runtime_identity(
        repo,
        core_paths=["src/core.py"],
        operations_paths=["src/ops.py"],
        config_path="config.json",
        dependency_lock_path="lock.json",
    )
    assert er._historical_identity(repo, identity) == {"answer": 42}
    (repo / "src/core.py").write_bytes(b"VALUE = 9\n")
    (repo / "config.json").write_bytes(b'{"answer": 9}')
    assert er._historical_identity(repo, identity) == {"answer": 42}
    broken = copy.deepcopy(identity)
    broken["core_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        er._historical_identity(repo, broken)
