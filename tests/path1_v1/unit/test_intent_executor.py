from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import submission_process
from agentmemeval.path1_v1.operations.intent_executor import (
    DurableIntentExecutor,
    ExecutorError,
    LaunchFiles,
    LocalCommandTransport,
    SubmissionAdmission,
    VerifiedCommand,
    clean_environment,
    render_worker_script,
    verify_effective_user,
    verify_loaded_package,
)
from agentmemeval.path1_v1.operations.slurm import (
    AccountSnapshot,
    CapacityEvidence,
    JobRecord,
    SubmissionDisabled,
    SubmissionRequest,
    SubmitSpec,
)
from agentmemeval.path1_v1.operations.state import StoreIntegrityError
from agentmemeval.path1_v1.operations.submission_process import file_sha256

FAKE_SBATCH = r"""
import hashlib, json, os, runpy, signal, sys, time
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
identity = runpy.run_path(sys.argv[3])["process_identity"](os.getpid())
script = sys.stdin.buffer.read()
effect = root / "external_effect.json"
with effect.open("x", encoding="utf-8") as stream:
    json.dump({"argv": sys.argv[4:], "script_sha256": hashlib.sha256(script).hexdigest(),
               "command_identity": identity,
               "inherited_secret": "EXECUTOR_TEST_SECRET" in os.environ,
               "inherited_sbatch": "SBATCH_ARRAY_INX" in os.environ}, stream)
    stream.flush()
    os.fsync(stream.fileno())
if mode == "kill_submitter":
    os.kill(os.getppid(), signal.SIGTERM)
    os._exit(0)
if mode == "timeout":
    time.sleep(4)
if mode == "malformed":
    print("unexpected response with a fake credential")
elif mode == "wrong_cluster":
    print("551;different_cluster")
elif mode == "error":
    sys.exit(1)
elif mode == "warning":
    print("private diagnostic", file=sys.stderr)
    print("551;cluster")
else:
    print("551;cluster")
"""


def make_executor_fixture(root: Path, mode: str = "normal", *, initialize_worker: bool = True):
    from path1_v1.unit.test_operations_worker import make_worker_fixture

    worker_root = root / "worker"
    if initialize_worker:
        worker_root.mkdir()
        worker = make_worker_fixture(worker_root)
    else:
        request_path = worker_root / "worker_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        contract = json.loads(Path(request["contract_path"]).read_text(encoding="utf-8"))
        worker = {
            "request_path": request_path,
            "request": request,
            "contract": contract,
            "repo": Path(request["repository_root"]),
            "output": Path(request["output_root"]) / contract["job"]["output_relpath"] / "payload",
        }
    python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    script_path = root / "launcher.sh"
    fake_path = root / "fake_sbatch.py"
    if initialize_worker:
        script_path.write_bytes(
            render_worker_script(
                python_executable=python,
                repository_root=worker["repo"],
                worker_request_path=worker["request_path"],
            )
        )
        fake_path.write_text(FAKE_SBATCH, encoding="utf-8")
    spec = SubmitSpec(
        "cluster",
        "account",
        "owner",
        "gpu",
        "/runs/" + worker["contract"]["job"]["output_relpath"],
        "/code/launcher.sh",
        file_sha256(script_path),
        walltime_seconds=120,
    )
    request = SubmissionRequest(
        worker["contract"]["campaign_id"], worker["request"]["intent_id"], spec
    )
    fixture = SimpleNamespace(
        root=root,
        worker=worker,
        request=request,
        now=datetime.now(UTC),
        limit=8,
        active=[],
        visibility=True,
        queries=0,
        admissions=0,
        stage=1,
        mode=mode,
        script_path=script_path,
        fake_path=fake_path,
        on_admit=None,
        on_query=None,
    )

    def admit(submission, validated):
        fixture.admissions += 1
        if fixture.on_admit is not None:
            fixture.on_admit(fixture.admissions)
        return SubmissionAdmission(
            submission.request_sha256,
            validated.request_sha256,
            validated.request["contract_sha256"],
            "a" * 64,
            fixture.stage,
            fixture.now,
        )

    def account_reader():
        fixture.queries += 1
        if fixture.on_query is not None:
            fixture.on_query(fixture.queries)
        return AccountSnapshot(
            "cluster",
            "account",
            fixture.now,
            tuple(fixture.active),
            (),
            "b" * 64,
            CapacityEvidence(
                "cluster", "account", fixture.limit, fixture.now, "c" * 64, fixture.visibility
            ),
        )

    fixture.executor = DurableIntentExecutor(
        root / "executor_control",
        cluster="cluster",
        account="account",
        command=VerifiedCommand(
            (
                str(python),
                "-I",
                str(fake_path),
                str(root),
                mode,
                str(Path(submission_process.__file__).resolve()),
            ),
            (
                (str(python), file_sha256(python)),
                (str(fake_path), file_sha256(fake_path)),
                (
                    str(Path(submission_process.__file__).resolve()),
                    file_sha256(Path(submission_process.__file__)),
                ),
            ),
        ),
        launch_files=lambda _: LaunchFiles(worker["request_path"], python, file_sha256(python)),
        pre_submit_verifier=admit,
        account_reader=account_reader,
        fixture_mode=True,
        fixture_paths={spec.script_path: script_path, spec.work_dir: worker["output"].parent},
        clock=lambda: fixture.now,
    )
    return fixture


@pytest.fixture
def fixture(tmp_path):
    return make_executor_fixture(tmp_path)


def submit(fixture, timeout: float = 10):
    return fixture.executor.submit_once(fixture.request, timeout_seconds=timeout)


def test_real_subprocess_boundary_uses_pinned_stdin_and_clean_environment(fixture, monkeypatch):
    monkeypatch.setenv("EXECUTOR_TEST_SECRET", "must never reach child")
    monkeypatch.setenv("SBATCH_ARRAY_INX", "1-100")
    result = submit(fixture)
    assert result.status == "accepted" and result.job_id == "551" and result.submitter_finished
    evidence = json.loads((fixture.root / "external_effect.json").read_text(encoding="utf-8"))
    assert not evidence["inherited_secret"] and not evidence["inherited_sbatch"]
    assert evidence["script_sha256"] == fixture.request.spec.script_sha256
    assert evidence["argv"] == list(fixture.request.sbatch_argv()[1:-1])
    assert fixture.queries == 2 and fixture.admissions == 2
    assert fixture.worker["output"].parent.is_dir() and not fixture.worker["output"].exists()
    assert fixture.worker["store"].read_registry().claims[0].status == "reserved"
    record = fixture.executor.inspect(fixture.request)
    assert record["ready"]["submitter"]["start_marker"]
    assert record["process_observation_authorizes_retry"] is False


def test_second_call_and_lookup_return_same_identity_without_another_process(fixture):
    first = submit(fixture)
    before = (fixture.root / "external_effect.json").read_bytes()
    second = submit(fixture)
    assert first == second == fixture.executor.lookup(fixture.request, timeout_seconds=10)
    assert (fixture.root / "external_effect.json").read_bytes() == before
    assert fixture.queries == 2


def test_default_missing_pre_submission_gate_is_closed(fixture):
    fixture.executor.pre_submit_verifier = None
    with pytest.raises(SubmissionDisabled):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()
    assert not fixture.worker["output"].parent.exists()


@pytest.mark.parametrize("gate", [lambda *_: True, lambda *_: {"verified": True}])
def test_metadata_only_gate_cannot_enable_execution(fixture, gate):
    fixture.executor.pre_submit_verifier = gate
    with pytest.raises(SubmissionDisabled, match="typed"):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


def test_production_mode_does_not_use_fixture_paths_or_python_as_sbatch(fixture):
    fixture.executor.fixture_mode = False
    with pytest.raises(SubmissionDisabled, match="Linux sbatch"):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        b"#SBATCH --array=1-4\n",
        b"#SBATCH --gpus=8\n",
        b"touch /tmp/injected\n",
        b"#SBATCH hetjob\n",
        b"exec sbatch other.sh\n",
    ],
)
def test_even_rehashed_script_injections_and_extra_topology_are_rejected(fixture, change):
    fixture.script_path.write_bytes(fixture.script_path.read_bytes() + change)
    fixture.request = replace(
        fixture.request,
        spec=replace(fixture.request.spec, script_sha256=file_sha256(fixture.script_path)),
    )
    with pytest.raises(ExecutorError, match="topology"):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


def test_unapproved_script_args_are_rejected(fixture):
    fixture.request = replace(
        fixture.request, spec=replace(fixture.request.spec, script_args=("--array=9",))
    )
    with pytest.raises(ExecutorError, match="extra script"):
        submit(fixture)


@pytest.mark.parametrize("change", ["script", "command", "source", "request", "claim"])
def test_changed_actual_input_bytes_prevent_the_external_call(fixture, change):
    paths = {
        "script": fixture.script_path,
        "command": fixture.fake_path,
        "source": fixture.worker["repo"] / "src/agentmemeval/path1_v1/operations/worker.py",
        "request": fixture.worker["request_path"],
        "claim": Path(fixture.worker["request"]["claim_path"]),
    }
    path = paths[change]
    path.write_bytes(path.read_bytes() + b"\n# modified\n")
    with pytest.raises((ValueError, RuntimeError)):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


def test_second_capacity_check_can_stop_before_release(fixture):
    fixture.on_query = lambda count: setattr(fixture, "limit", 2) if count == 2 else None
    with pytest.raises(SubmissionDisabled, match="capacity"):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()
    receipt = fixture.executor.lookup(fixture.request, timeout_seconds=1)
    assert receipt.status == "unknown"
    fixture.limit = 8
    assert submit(fixture).status == "unknown"
    assert fixture.queries == 2


def test_source_change_between_final_validation_and_release_is_checked_by_child(fixture):
    fixture.on_admit = lambda count: (
        fixture.script_path.write_bytes(b"# changed after validation\n") if count == 2 else None
    )
    assert submit(fixture).status == "unknown"
    assert not (fixture.root / "external_effect.json").exists()


@pytest.mark.parametrize("count", [6, 8])
def test_other_account_jobs_and_two_reserved_slots_block_submission(fixture, count):
    fixture.active = [
        JobRecord(
            str(900 + index),
            "neighbor",
            "account",
            "another_work",
            "RUNNING",
            "",
            "2026-11-13T01:00:00",
            "/other",
            "squeue",
        )
        for index in range(count)
    ]
    with pytest.raises(SubmissionDisabled, match="capacity"):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


def test_unverified_full_account_visibility_denies_submission(fixture):
    fixture.visibility = False
    with pytest.raises(RuntimeError):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()


def test_existing_job_container_is_not_reused(fixture):
    fixture.worker["output"].parent.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        submit(fixture)
    assert not (fixture.root / "external_effect.json").exists()
    assert submit(fixture).status == "unknown"


@pytest.mark.parametrize(
    "mode", ["malformed", "wrong_cluster", "warning", "error", "kill_submitter"]
)
def test_lost_or_unusable_command_reply_remains_unknown_without_reissue(tmp_path, mode):
    fixture = make_executor_fixture(tmp_path, mode)
    assert submit(fixture).status == "unknown"
    effect = (fixture.root / "external_effect.json").read_bytes()
    assert submit(fixture).status == "unknown"
    assert (fixture.root / "external_effect.json").read_bytes() == effect
    record = fixture.executor.inspect(fixture.request)
    result_path = Path(record["invocation_dir"]) / "result.json"
    if result_path.exists():
        assert "fake credential" not in result_path.read_text(encoding="utf-8")
        assert "private diagnostic" not in result_path.read_text(encoding="utf-8")


def test_pid_reuse_observation_never_authorizes_unknown_retry(fixture, monkeypatch):
    import agentmemeval.path1_v1.operations.intent_executor as executor_module

    original_popen = executor_module.subprocess.Popen
    helper_attempts = []

    def fail_helper_only(argv, *args, **kwargs):
        if any(str(part).endswith("submission_process.py") for part in argv):
            helper_attempts.append(tuple(argv))
            raise OSError("fixture launch error")
        return original_popen(argv, *args, **kwargs)

    monkeypatch.setattr(executor_module.subprocess, "Popen", fail_helper_only)
    if sys.platform.startswith("linux"):
        # Managed creation fails inside the production transport exception gate.
        assert submit(fixture).status == "unknown"
    else:
        with pytest.raises(OSError, match="fixture launch error"):
            submit(fixture)
    assert len(helper_attempts) == 1
    assert not (fixture.root / "external_effect.json").exists()
    before = fixture.executor.lookup(fixture.request, timeout_seconds=1)
    stored_before = fixture.executor.store.read_registry()
    identity = fixture.executor.inspect(fixture.request)["parent_identity"]
    monkeypatch.setattr(
        submission_process, "process_identity", lambda _: {**identity, "start_marker": "different"}
    )
    inspected = fixture.executor.inspect(fixture.request)
    assert inspected["parent_observation"] == "different_identity"
    assert not inspected["process_observation_authorizes_retry"]
    assert fixture.executor.lookup(fixture.request, timeout_seconds=1) == before
    assert not before.submitter_finished and before.status == "unknown"
    assert submit(fixture) == before
    assert len(helper_attempts) == 1
    assert fixture.executor.store.read_registry() == stored_before
    assert not (fixture.root / "external_effect.json").exists()


def test_rebinding_existing_intent_or_tampering_durable_result_fails(fixture):
    submit(fixture)
    changed = replace(fixture.request, spec=replace(fixture.request.spec, walltime_seconds=121))
    with pytest.raises(StoreIntegrityError):
        fixture.executor.lookup(changed, timeout_seconds=1)
    result = Path(fixture.executor.inspect(fixture.request)["invocation_dir"]) / "result.json"
    data = json.loads(result.read_text(encoding="utf-8"))
    data["stdout"] = "999;cluster"
    result.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(StoreIntegrityError):
        fixture.executor.lookup(fixture.request, timeout_seconds=1)


def test_clean_environment_rejects_extra_credentials_and_scheduler_overrides():
    for name in ("API_KEY", "SBATCH_GRES", "PYTHONPATH", "BASH_ENV", "LD_PRELOAD"):
        with pytest.raises(ExecutorError):
            clean_environment({name: "value"})


def test_read_transport_is_real_subprocess_and_refuses_mutating_commands(tmp_path):
    script = tmp_path / "read.py"
    script.write_text("import sys; print(sys.argv[1])\n", encoding="utf-8")
    python = Path(sys.executable).resolve()
    target = VerifiedCommand(
        (str(python), "-I", str(script)),
        ((str(python), file_sha256(python)), (str(script), file_sha256(script))),
    )
    transport = LocalCommandTransport({"squeue": target}, fixture_mode=True)
    result = transport.run(
        ("squeue", "literal;not-a-shell"),
        environment={"TZ": "UTC"},
        inherit_environment=False,
        timeout_seconds=5,
    )
    assert result.returncode == 0 and result.stdout.strip() == "literal;not-a-shell"
    with pytest.raises(ExecutorError):
        LocalCommandTransport({"sbatch": target}, fixture_mode=True)
    with pytest.raises(ExecutorError):
        transport.run(("squeue",), environment={}, inherit_environment=True, timeout_seconds=5)


def test_process_identity_contains_a_kernel_start_marker():
    identity = submission_process.process_identity(os.getpid())
    assert identity["pid"] == os.getpid() and identity["start_marker"] and identity["boot_id"]
    assert submission_process.observe_process(identity) == "same_identity"


def test_control_json_is_published_complete_without_replacing_existing_content(
    tmp_path, monkeypatch
):
    target = tmp_path / "ready.json"
    original_link = submission_process.os.link

    def inspect_publication(source, destination):
        assert not target.exists()
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"ready": True}
        return original_link(source, destination)

    monkeypatch.setattr(submission_process.os, "link", inspect_publication)
    submission_process.write_exclusive_json(target, {"ready": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ready": True}
    with pytest.raises(FileExistsError):
        submission_process.write_exclusive_json(target, {"ready": False})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ready": True}


def test_failed_atomic_publication_keeps_partial_and_no_authoritative_receipt(
    tmp_path, monkeypatch
):
    target = tmp_path / "result.json"

    def refuse_link(*_):
        raise OSError("fixture publication interrupted")

    monkeypatch.setattr(submission_process.os, "link", refuse_link)
    with pytest.raises(OSError):
        submission_process.write_exclusive_json(target, {"job_id": "551"})
    assert not target.exists()
    assert json.loads(target.with_name("result.json.partial").read_text(encoding="utf-8")) == {
        "job_id": "551"
    }


@pytest.mark.parametrize(
    "canonical,named_uid,expected,allowed",
    [
        ("owner", 1001, "owner", True),
        ("other_user", 1001, "owner", False),
        ("owner", 2002, "owner", False),
        ("owner", 1001, "alias", False),
    ],
)
def test_effective_uid_and_os_account_database_are_both_required(
    monkeypatch, canonical, named_uid, expected, allowed
):
    # Inject OS API results, not a trusted/verified receipt. Actual Linux calls
    # remain an E0 check; Windows cannot claim this branch was exercised natively.
    pwd_fixture = SimpleNamespace(
        getpwuid=lambda uid: SimpleNamespace(pw_uid=uid, pw_name=canonical),
        getpwnam=lambda name: SimpleNamespace(pw_uid=named_uid, pw_name=name),
    )
    monkeypatch.setitem(sys.modules, "pwd", pwd_fixture)
    monkeypatch.setattr(os, "geteuid", lambda: 1001, raising=False)
    if allowed:
        assert verify_effective_user(expected) == {
            "kind": "unix_effective_user",
            "uid": 1001,
            "name": "owner",
        }
    else:
        with pytest.raises(SubmissionDisabled, match="effective OS account"):
            verify_effective_user(expected)


def _loaded_source_fixture():
    from agentmemeval.path1_v1.operations import worker

    assert worker.validate_worker_request
    repository = Path(__file__).resolve().parents[3]
    records = {}
    for name, module in tuple(sys.modules.items()):
        if name == "agentmemeval" or name.startswith("agentmemeval."):
            if isinstance(getattr(module, "__file__", None), str):
                path = Path(module.__file__).resolve()
                records[path.relative_to(repository).as_posix()] = file_sha256(path)
    return repository, [{"path": path, "sha256": sha} for path, sha in sorted(records.items())]


def test_current_loaded_package_and_namespace_origins_match_the_source_closure():
    repository, records = _loaded_source_fixture()
    verify_loaded_package(repository, records)


@pytest.mark.parametrize(
    "module_name",
    [
        "agentmemeval.path1_v1.operations.state",
        "agentmemeval.path1_v1.operations.contracts",
        "agentmemeval.path1_v1.operations.slurm",
    ],
)
def test_unchanged_entry_files_cannot_hide_a_dependency_from_another_checkout(
    tmp_path, monkeypatch, module_name
):
    repository, records = _loaded_source_fixture()
    original = sys.modules[module_name]
    foreign_file = tmp_path / "foreign_dependency.py"
    # Origin must match even when the foreign file has identical bytes.
    foreign_file.write_bytes(Path(original.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(module_name, foreign_file)
    foreign = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, foreign)
    with pytest.raises(ExecutorError, match="another source tree"):
        verify_loaded_package(repository, records)


@pytest.mark.parametrize("name", ["state", "contracts", "slurm"])
def test_unchanged_entry_files_cannot_hide_different_dependency_bytes(name):
    repository, records = _loaded_source_fixture()
    target = f"src/agentmemeval/path1_v1/operations/{name}.py"
    for record in records:
        if record["path"] == target:
            record["sha256"] = "a" * 64
    with pytest.raises(ExecutorError, match="outside the contract closure"):
        verify_loaded_package(repository, records)


def test_loaded_package_search_path_cannot_point_at_another_checkout(tmp_path, monkeypatch):
    repository, records = _loaded_source_fixture()
    package = sys.modules["agentmemeval.path1_v1.operations"]
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    with pytest.raises(ExecutorError, match="search path escapes"):
        verify_loaded_package(repository, records)


def test_actual_subprocess_timeout_is_unknown_and_records_exact_descendant_observation(tmp_path):
    fixture = make_executor_fixture(tmp_path, "timeout")
    invocation_dir = fixture.executor.store.directory / "i" / fixture.request.intent_id[:16]
    # The old three-second budget expired before go on slow durable filesystems.
    # Give the unchanged pre-exec gates time, then keep only this fixture command
    # alive beyond the real pinned deadline. No production clock/deadline is mocked.
    fixture.fake_path.write_text(
        """import hashlib, json, os, runpy, sys, time
from pathlib import Path
root = Path(sys.argv[1])
identity = runpy.run_path(sys.argv[3])["process_identity"](os.getpid())
script = sys.stdin.buffer.read()
manifest = json.loads(Path(__PINNED_MANIFEST__).read_text(encoding="utf-8"))
started = time.monotonic()
deadline = manifest["deadline_monotonic"]
assert 0 < deadline - started <= 15
with (root / "external_effect.json").open("x", encoding="utf-8") as stream:
    json.dump({"command_identity": identity, "script_sha256": hashlib.sha256(script).hexdigest(),
               "started_monotonic": started, "deadline_monotonic": deadline}, stream)
    stream.flush()
    os.fsync(stream.fileno())
# Absolute fixture bound: at most 17 seconds after command entry, even if the
# parent disappears; sleep uses the same kernel clock as the actual deadline.
time.sleep(min(17.0, max(0.0, deadline - time.monotonic()) + 2.0))
print("551;cluster")
""".replace("__PINNED_MANIFEST__", repr(str(invocation_dir / "invocation.json"))),
        encoding="utf-8",
    )
    command = fixture.executor.command
    fixture.executor.command = replace(
        command,
        files=tuple(
            (name, file_sha256(fixture.fake_path) if Path(name) == fixture.fake_path else digest)
            for name, digest in command.files
        ),
    )
    started = time.monotonic()
    receipt = submit(fixture, timeout=15)
    assert receipt.status == "unknown" and time.monotonic() - started < 30
    manifest = json.loads((invocation_dir / "invocation.json").read_text(encoding="utf-8"))
    ready = json.loads((invocation_dir / "ready.json").read_text(encoding="utf-8"))
    assert json.loads((invocation_dir / "go.json").read_text(encoding="utf-8")) == ready
    effect_path = tmp_path / "external_effect.json"
    effect = json.loads(effect_path.read_text(encoding="utf-8"))
    assert effect["script_sha256"] == fixture.request.spec.script_sha256
    assert effect["deadline_monotonic"] == manifest["deadline_monotonic"]
    assert effect["started_monotonic"] < manifest["deadline_monotonic"] <= time.monotonic()
    identity = effect["command_identity"]
    assert type(identity["pid"]) is int and identity["pid"] > 0
    assert identity["start_marker"] and identity["platform"] == sys.platform
    initial_observation = submission_process.observe_process(identity)
    # The command ends no later than the pinned deadline + two seconds. Observe
    # its exact kernel identity; do not infer natural exit versus termination.
    deadline = time.monotonic() + 8
    while submission_process.observe_process(identity) == "same_identity":
        if time.monotonic() >= deadline:
            pytest.fail("owned finite-lifetime fixture process did not exit")
        time.sleep(0.05)
    inspected = fixture.executor.inspect(fixture.request)
    stored_before = fixture.executor.store.read_registry()
    observation = {
        "status": receipt.status,
        "command_identity": identity,
        "immediate_command_observation": initial_observation,
        "final_command_observation": submission_process.observe_process(identity),
        "parent_observation": inspected["parent_observation"],
        "helper_observation": inspected["submitter_observation"],
        "cleanup": "test waited without PID kill; completion cause unverified",
        "linux_descendant_cleanup_verified": False,
    }
    (tmp_path / "timeout_observation.json").write_text(json.dumps(observation), encoding="utf-8")
    assert submit(fixture).status == "unknown"
    assert json.loads(effect_path.read_text(encoding="utf-8")) == effect
    assert fixture.executor.store.read_registry() == stored_before
