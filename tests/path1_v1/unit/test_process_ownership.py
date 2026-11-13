from __future__ import annotations

import signal
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import process_ownership as module

REAL_MEMBERS = module._members


def test_cleanup_kernel_identity_never_reads_exiting_address_space(monkeypatch):
    monkeypatch.setattr(module, "_stat", lambda _: (55, 77, 77, "22", "R"))

    def text(path, **_):
        return {
            "status": "Uid:\t1000\t1000\t1000\t1000\n",
            "boot_id": "boot",
            "cgroup": "0::/fixture",
        }[path.name]

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("mm read during cleanup"))
    identity, cmdline, state, _ = module._kernel(77, include_runtime=False)
    assert identity.pid == 77 and identity.start_marker == "22" and state == "R"
    assert cmdline == () and identity.slurm_job_id is None


@pytest.mark.parametrize(
    "next_state,next_marker,accepted",
    [
        ("Z", "22", True),
        ("X", "22", True),
        ("S", "22", False),
        ("Z", "23", False),
    ],
)
def test_exit_environ_permission_race_requires_same_dead_identity(
    monkeypatch, next_state, next_marker, accepted
):
    stats = iter(
        [
            (55, 77, 77, "22", "S"),
            (55, 77, 77, next_marker, next_state),
            (55, 77, 77, next_marker, next_state),
        ]
    )
    monkeypatch.setattr(module, "_stat", lambda _: next(stats))

    def read_text(path, **_):
        if path.name == "status":
            return "Uid:\t1000\t1000\t1000\t1000\n"
        if path.name == "boot_id":
            return "actual-boot"
        if path.name == "cgroup":
            return "0::/actual-group\n"
        raise AssertionError(path)

    def read_bytes(path):
        if path.name == "environ":
            raise PermissionError("controlled transition race")
        assert path.name == "cmdline"
        return b""

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    if accepted:
        identity, _, state, _ = module._kernel(77)
        assert identity.start_marker == "22" and state == next_state
        assert identity.slurm_job_id is None
    else:
        with pytest.raises(PermissionError):
            module._kernel(77)


def test_historical_snapshot_requires_exact_owner_provenance(monkeypatch):
    owner, _, identity = fixture_owner(monkeypatch)
    owner._snapshots = weakref.WeakValueDictionary()
    monkeypatch.setattr(owner, "_anchor", lambda: (identity, ("/python",), "S"))
    monkeypatch.setattr(module, "_kernel", lambda *_, **__: (identity, ("/python",), "S", 55))
    actual = owner.snapshot()
    assert owner.verify_snapshot(actual) is True
    with pytest.raises(module.OwnershipError):
        owner.verify_snapshot(replace(actual))
    other, _, _ = fixture_owner(monkeypatch)
    with pytest.raises(module.OwnershipError):
        other.verify_snapshot(actual)


@pytest.mark.parametrize("field", ["pgid", "session_id"])
def test_group_member_cannot_change_group_between_stat_and_identity(monkeypatch, field):
    owner, _, identity = fixture_owner(monkeypatch)
    # fixture_owner stubs _members; recover its original implementation explicitly.
    monkeypatch.setattr(Path, "iterdir", lambda _: iter([Path("/proc/77")]))
    monkeypatch.setattr(module, "_stat", lambda _: (55, 77, 77, "123", "S"))
    monkeypatch.setattr(
        module,
        "_kernel",
        lambda _, **__: (
            replace(identity, **{field: 88}),
            (),
            "S",
            55,
        ),
    )
    with pytest.raises(module.OwnershipError, match="unverified"):
        REAL_MEMBERS(owner.identity)


@pytest.mark.parametrize("changed", [False, True])
def test_socket_descriptor_holder_is_revalidated_after_fd_lookup(monkeypatch, changed):
    owner, _, identity = fixture_owner(monkeypatch)
    snapshot = module.ProcessSnapshot(owner.binding, identity, ("/python",), "S")
    monkeypatch.setattr(owner, "verify_live", lambda: snapshot)
    monkeypatch.setattr(module, "_members", lambda _: [(identity, "S")])
    monkeypatch.setattr(
        module.os, "readlink", lambda path: "socket:[42]" if Path(path).name == "3" else "net:[1]"
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_, **__: "header\n0: 0100007F:1F90 00000000:0000 0A 0 0 0 1000 0 42\n",
    )
    monkeypatch.setattr(Path, "iterdir", lambda _: iter([Path("/proc/77/fd/3")]))
    monkeypatch.setattr(
        module,
        "_kernel",
        lambda _, **__: (
            replace(identity, start_marker="wrong") if changed else identity,
            (),
            "S",
            55,
        ),
    )
    if changed:
        with pytest.raises(module.OwnershipError, match="socket holder changed"):
            owner.verify_endpoint("http://127.0.0.1:8080/v1")
    else:
        assert owner.verify_endpoint("http://127.0.0.1:8080/v1").socket_holders == (
            (77, "123", 42),
        )


def fixture_owner(monkeypatch):
    binding = module.ProcessBinding("campaign", "intent", None, "a" * 64, "model_service", "svc")
    identity = module.ProcessIdentity(
        "host", "boot", 1000, 77, "123", 77, 77, ("0::/job",), None, None
    )
    events = []
    process = SimpleNamespace(returncode=None)

    def wait(*, timeout):
        events.append("reap")
        process.returncode = 0
        return 0

    process.wait = wait
    owner = module.OwnedProcess(
        module._CONSTRUCTION, process, binding, ["/python"], identity, identity
    )
    monkeypatch.setattr(owner, "_anchor", lambda: events.append("anchor"))
    monkeypatch.setattr(module, "_members", lambda _: [(identity, "Z")])
    monkeypatch.setattr(
        module.os, "killpg", lambda pid, sig: events.append((pid, sig)), raising=False
    )
    # Unit simulation on Windows cannot install POSIX-only signal handlers.
    monkeypatch.setattr(module.threading, "current_thread", lambda: object())
    return owner, events, identity


def test_cleanup_signals_verified_group_before_reaping_and_is_idempotent(monkeypatch):
    owner, events, _ = fixture_owner(monkeypatch)
    receipt = owner.close(term_grace_seconds=0)
    assert receipt.group_cleanup_verified and receipt.leader_reaped
    assert events.index((77, signal.SIGTERM)) < events.index("reap")
    assert owner.close() is receipt
    assert owner.verify_cleanup(receipt) is True
    assert events.count("reap") == 1
    with pytest.raises(module.OwnershipError, match="this live owner"):
        owner.verify_cleanup(replace(receipt))


@pytest.mark.parametrize(
    "error",
    [
        ChildProcessError("reaped"),
        PermissionError("denied"),
        module.OwnershipError("boot/UID/PGID changed"),
    ],
)
def test_uncertain_anchor_does_not_signal_or_reap(monkeypatch, error):
    owner, events, _ = fixture_owner(monkeypatch)

    def invalid():
        raise error

    monkeypatch.setattr(owner, "_anchor", invalid)
    receipt = owner.close(term_grace_seconds=0)
    assert not receipt.group_cleanup_verified and not receipt.leader_reaped
    assert events == []
    assert receipt.errors


def test_signal_permission_error_is_unverified_and_does_not_reap(monkeypatch):
    owner, events, _ = fixture_owner(monkeypatch)

    def denied(*_):
        raise PermissionError("denied")

    monkeypatch.setattr(module.os, "killpg", denied)
    receipt = owner.close(term_grace_seconds=0)
    assert not receipt.group_cleanup_verified and not receipt.leader_reaped
    assert "reap" not in events


def test_term_ignoring_group_is_killed_before_reap(monkeypatch):
    owner, events, identity = fixture_owner(monkeypatch)
    monkeypatch.setattr(module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        module, "_members", lambda _: [(identity, "Z" if (77, 9) in events else "S")]
    )
    receipt = owner.close(term_grace_seconds=0)
    assert receipt.group_cleanup_verified and receipt.kill_sent
    assert events.index((77, signal.SIGTERM)) < events.index((77, 9)) < events.index("reap")


def test_owner_binding_and_kernel_identity_are_read_only(monkeypatch):
    owner, _, identity = fixture_owner(monkeypatch)
    with pytest.raises(AttributeError):
        owner.identity = replace(identity, pid=80)
    with pytest.raises(AttributeError):
        owner.binding = owner.binding
    with pytest.raises(module.OwnershipError, match="only be created"):
        module.OwnedProcess(None, None, None, None, None, None)


@pytest.mark.parametrize("grace", [-1, 121, float("inf"), True])
def test_invalid_cleanup_grace_cannot_signal(monkeypatch, grace):
    owner, events, _ = fixture_owner(monkeypatch)
    with pytest.raises(ValueError):
        owner.close(term_grace_seconds=grace)
    assert events == []


def test_windows_spawn_never_probes_pid_with_os_kill(monkeypatch, tmp_path):
    monkeypatch.setattr(module.sys, "platform", "win32")

    def forbidden(*_, **__):
        raise AssertionError("Windows PID probing/creation is forbidden")

    monkeypatch.setattr(module.os, "kill", forbidden)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(module.OwnershipError, match="Linux"):
        module.OwnedProcess.spawn([], binding=None, cwd=tmp_path, env={})


@pytest.mark.parametrize("during_creation", [True, False])
def test_managed_interrupt_covers_creation_and_body(monkeypatch, during_creation):
    events = []
    handlers = {}
    monkeypatch.setattr(module, "_linux", lambda: None)
    monkeypatch.setattr(module.signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(module.signal, "getsignal", lambda sig: handlers.get(sig, "old"))
    monkeypatch.setattr(
        module.signal, "signal", lambda sig, value: handlers.__setitem__(sig, value)
    )
    owner = SimpleNamespace(close=lambda **_: events.append("closed"))

    def spawn(*_, **__):
        events.append("spawn")
        if during_creation:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            events.append("creation_not_aborted")
        return owner

    monkeypatch.setattr(module.OwnedProcess, "spawn", spawn)
    with pytest.raises(module.ProcessInterrupted):
        with module.OwnedProcess.managed(["/python"], term_grace_seconds=0):
            assert not during_creation
            handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert events[-1] == "closed"
    assert all(value == "old" for value in handlers.values())
    assert ("creation_not_aborted" in events) is during_creation


@pytest.mark.parametrize("lost_child", [False, True])
def test_failed_spawn_only_cleans_waitable_direct_child_and_records_unknown(
    monkeypatch, tmp_path, lost_child
):
    binding = module.ProcessBinding("c", "i", None, "a" * 64, "model_service", "s")
    creator = module.ProcessIdentity("host", "boot", 1000, 55, "11", 55, 55, (), None, None)
    events = []
    process = SimpleNamespace(pid=77, returncode=None)
    process.wait = lambda **_: events.append("reap")
    monkeypatch.setattr(module, "_linux", lambda: None)
    monkeypatch.setattr(module.signal, "SIGCHLD", 17, raising=False)
    monkeypatch.setattr(module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(module.signal, "getsignal", lambda _: signal.SIG_DFL)
    monkeypatch.setattr(module.os, "getpid", lambda: 55)
    monkeypatch.setattr(module.os, "P_PID", 1, raising=False)
    for name in ("WEXITED", "WNOHANG", "WNOWAIT"):
        monkeypatch.setattr(module.os, name, 1, raising=False)

    def kernel(pid, **_):
        if pid == 77:
            raise PermissionError("identity unreadable")
        return creator, (), "S", 1

    def waitid(*_):
        if lost_child:
            raise ChildProcessError("already reaped")
        events.append("child_waitable")

    monkeypatch.setattr(module, "_kernel", kernel)
    monkeypatch.setattr(module, "_stat", lambda _: (55, 77, 77, "22", "S"))
    monkeypatch.setattr(module.os, "waitid", waitid, raising=False)
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(
        module.os, "killpg", lambda *_: pytest.fail("group signal forbidden"), raising=False
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(module, "_UNVERIFIED_CHILDREN", [])
    with pytest.raises(module.SpawnOwnershipError) as raised:
        module.OwnedProcess.spawn([str(tmp_path / "python")], binding=binding, cwd=tmp_path, env={})
    receipt = raised.value.receipt
    assert not receipt.group_cleanup_verified
    assert receipt.direct_child_reaped is (not lost_child)
    assert receipt.direct_child_killed is (not lost_child)
    assert bool(module._UNVERIFIED_CHILDREN) is lost_child
    assert ((77, 9) in events) is (not lost_child)
