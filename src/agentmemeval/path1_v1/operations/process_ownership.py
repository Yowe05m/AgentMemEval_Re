"""Linux live ownership of a newly created session, never authority from a PID file.

Only this owner may reap its direct leader. WNOWAIT retains that leader while
the group is inspected/signalled, including when the leader has already exited.
Escaped sessions, hard-killed controllers and Slurm allocation release require
separate scheduler/cgroup evidence; a local cleanup receipt does not prove them.
"""

from __future__ import annotations

import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class OwnershipError(RuntimeError):
    pass


class ProcessInterrupted(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"owned process interrupted by signal {signum}")


@dataclass(frozen=True)
class ProcessBinding:
    campaign_id: str
    intent_id: str
    job_id: str | None
    source_sha256: str
    role: str
    service_instance_id: str

    def __post_init__(self):
        for value in (self.campaign_id, self.intent_id, self.role, self.service_instance_id):
            if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value):
                raise ValueError("invalid process binding")
        if self.job_id is not None and not re.fullmatch(r"[1-9][0-9]*", self.job_id):
            raise ValueError("invalid Slurm allocation ID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("process binding requires a complete source digest")


@dataclass(frozen=True)
class ProcessIdentity:
    host: str
    boot_id: str
    effective_uid: int
    pid: int
    start_marker: str
    session_id: int
    pgid: int
    cgroup: tuple[str, ...]
    slurm_job_id: str | None
    step_id: str | None


@dataclass(frozen=True)
class ProcessSnapshot:
    binding: ProcessBinding
    identity: ProcessIdentity
    cmdline: tuple[str, ...]
    state: str


@dataclass(frozen=True)
class CleanupReceipt:
    binding: ProcessBinding
    identity: ProcessIdentity
    returncode: int | None
    term_sent: bool
    kill_sent: bool
    leader_reaped: bool
    group_cleanup_verified: bool
    remaining_non_zombie_pids: tuple[int, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class SpawnFailureReceipt:
    binding: ProcessBinding
    pid: int
    direct_child_killed: bool
    direct_child_reaped: bool
    group_cleanup_verified: bool
    errors: tuple[str, ...]


class SpawnOwnershipError(OwnershipError):
    def __init__(self, receipt: SpawnFailureReceipt):
        self.receipt = receipt
        super().__init__("new child ownership failed; group cleanup remains unverified")


class ReceiptSinkError(OwnershipError):
    """Cleanup ran, but its receipt was not durably acknowledged by the sink.

    The attached live objects are diagnostic capabilities, never a serializable
    permission to kill/retry. A verified group exit does not imply persistence.
    """

    def __init__(self, receipt, cleanup_owner, sink_error, original_failure):
        self.receipt = receipt
        self.cleanup_owner = cleanup_owner
        self.sink_error_type = type(sink_error).__name__
        self.original_failure_type = (
            type(original_failure).__name__ if original_failure is not None else None
        )
        self.interrupted_signal = getattr(original_failure, "signum", None)
        self.receipt_persisted = False
        super().__init__("process cleanup receipt sink failed; persistence is unverified")


@dataclass(frozen=True)
class EndpointObservation:
    process: ProcessSnapshot
    base_url: str
    network_namespace: str
    listener_inodes: tuple[int, ...]
    socket_holders: tuple[tuple[int, str, int], ...]  # PID, creation marker, inode


def _linux() -> None:
    if not sys.platform.startswith("linux") or not hasattr(os, "waitid"):
        raise OwnershipError("live process ownership requires Linux waitid/WNOWAIT")


def _stat(pid: int) -> tuple[int, int, int, str, str]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2 :].split()
    return int(fields[1]), int(fields[2]), int(fields[3]), fields[19], fields[0]


def _kernel(
    pid: int, *, include_runtime: bool = True
) -> tuple[ProcessIdentity, tuple[str, ...], str, int]:
    ppid, pgid, session_id, marker, state = _stat(pid)
    root = Path(f"/proc/{pid}")
    status = (root / "status").read_text(encoding="utf-8")
    uid = int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[2])
    # Read only these public allocation keys; never retain the complete environment.
    wanted = {b"SLURM_JOB_ID": None, b"SLURM_STEP_ID": None}
    if include_runtime and state != "Z":
        try:
            environment = (root / "environ").read_bytes()
        except PermissionError:
            # Exit can revoke environ after our first stat read. Accept only
            # the same creation marker now explicitly dead, never a live UID or
            # permission mismatch. Other kernel identity reads remain mandatory.
            _, _, _, current_marker, current_state = _stat(pid)
            if current_marker != marker or current_state not in {"Z", "X"}:
                raise
            state = current_state
            environment = b""
        for item in environment.split(b"\0"):
            key, _, value = item.partition(b"=")
            if key in wanted:
                wanted[key] = value.decode("ascii", errors="strict")
    identity = ProcessIdentity(
        socket.gethostname(),
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
        uid,
        pid,
        marker,
        session_id,
        pgid,
        tuple((root / "cgroup").read_text(encoding="utf-8").splitlines()),
        wanted[b"SLURM_JOB_ID"],
        wanted[b"SLURM_STEP_ID"],
    )
    cmdline = (
        tuple(os.fsdecode(value) for value in (root / "cmdline").read_bytes().split(b"\0") if value)
        if include_runtime
        else ()
    )
    if _stat(pid)[3] != marker:
        raise OwnershipError("process changed while reading kernel identity")
    return identity, cmdline, state, ppid


def _anchor_key(identity: ProcessIdentity) -> tuple:
    # A zombie no longer exposes environ; allocation fields remain in the initial identity.
    return (
        identity.host,
        identity.boot_id,
        identity.effective_uid,
        identity.pid,
        identity.start_marker,
        identity.session_id,
        identity.pgid,
        identity.cgroup,
    )


def _members(identity: ProcessIdentity) -> list[tuple[ProcessIdentity, str]]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            _, pgid, session_id, _, _ = _stat(int(entry.name))
            if pgid != identity.pgid:
                continue
            member, _, state, _ = _kernel(int(entry.name), include_runtime=False)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (
            session_id != identity.session_id
            or member.session_id != identity.session_id
            or member.pgid != identity.pgid
            or member.effective_uid != identity.effective_uid
            or member.boot_id != identity.boot_id
            or member.host != identity.host
        ):
            raise OwnershipError("group contains an unverified process identity")
        members.append((member, state))
    return members


_CONSTRUCTION = object()
# Retain unverified handles so Popen.__del__ cannot silently reap the PID anchor.
# These are diagnostics, never authority for a later kill-by-record operation.
_UNVERIFIED_CHILDREN: list[tuple[subprocess.Popen, SpawnFailureReceipt]] = []


class OwnedProcess:
    """A live creator capability. No attach/from_json/kill-by-record API exists.

    Do not expose the Popen object to another reaper. Child programs must remain
    in the created group; group completion is not proof about escaped children.
    Stdout/stderr must be files or DEVNULL, not PIPE (which can deadlock a waiter).
    """

    def __init__(self, token, process, binding, argv, identity, creator):
        if token is not _CONSTRUCTION:
            raise OwnershipError("ownership can only be created by spawn")
        self._process = process
        self._binding = binding
        self._identity = identity
        self._argv = tuple(argv)
        self._creator = creator
        self._lock = threading.RLock()
        self._receipt: CleanupReceipt | None = None
        self._snapshots = weakref.WeakValueDictionary()

    @property
    def binding(self) -> ProcessBinding:
        return self._binding

    @property
    def identity(self) -> ProcessIdentity:
        return self._identity

    @classmethod
    def spawn(
        cls,
        argv,
        *,
        binding: ProcessBinding,
        cwd: Path,
        env: Mapping[str, str],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        _linux()
        if not isinstance(binding, ProcessBinding):
            raise OwnershipError("a typed process binding is required")
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise OwnershipError("exclusive child reaping requires default SIGCHLD handling")
        if (
            not isinstance(argv, (list, tuple))
            or not argv
            or not Path(argv[0]).is_absolute()
            or any(type(arg) is not str or "\0" in arg for arg in argv)
            or stdin == subprocess.PIPE
            or stdout == subprocess.PIPE
            or stderr == subprocess.PIPE
        ):
            raise OwnershipError("absolute literal argv and non-pipe streams are required")
        creator, _, _, _ = _kernel(os.getpid(), include_runtime=False)
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
        )
        # Popen retains a direct child even if exec finishes before this read.
        # If identity cannot be established, do not signal an unverified group.
        try:
            identity, _, state, ppid = _kernel(process.pid)
            if (
                identity.pgid != process.pid
                or identity.session_id != process.pid
                or ppid != creator.pid
                or identity.effective_uid != creator.effective_uid
                or identity.boot_id != creator.boot_id
                or (
                    binding.job_id is not None
                    and (
                        env.get("SLURM_JOB_ID") != binding.job_id
                        or (state != "Z" and identity.slurm_job_id != binding.job_id)
                    )
                )
            ):
                raise OwnershipError("new child did not establish the requested owned session")
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            killed = reaped = False
            try:
                # Only the new, still-waitable direct child is eligible. No group
                # signal is authorized by an incomplete kernel identity read.
                current_creator, _, _, _ = _kernel(os.getpid(), include_runtime=False)
                if _anchor_key(current_creator) != _anchor_key(creator):
                    raise OwnershipError("creator changed during failed spawn")
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if _stat(process.pid)[0] != creator.pid or process.returncode is not None:
                    raise OwnershipError("failed spawn lost its unreaped direct child")
                os.kill(process.pid, signal.SIGKILL)
                killed = True
                process.wait(timeout=5)
                reaped = True
            except (
                OSError,
                ValueError,
                OwnershipError,
                subprocess.TimeoutExpired,
            ) as cleanup_error:
                errors.append(f"{type(cleanup_error).__name__}: {cleanup_error}")
            receipt = SpawnFailureReceipt(
                binding, process.pid, killed, reaped, False, tuple(errors)
            )
            if not reaped:
                _UNVERIFIED_CHILDREN.append((process, receipt))
            raise SpawnOwnershipError(receipt) from exc
        return cls(_CONSTRUCTION, process, binding, argv, identity, creator)

    @classmethod
    @contextmanager
    def managed(
        cls,
        argv,
        *,
        term_grace_seconds: float = 120,
        receipt_sink: Callable[[OwnedProcess | None, CleanupReceipt | SpawnFailureReceipt], None]
        | None = None,
        **kwargs,
    ):
        """Guard ordinary interrupts before creation, and always close before restoring handlers.

        During Popen/identity establishment signals are deferred, not blocked in
        the OS (children must not inherit blocked TERM). Only after an owner is
        assigned may the handler raise. SIGKILL and foreign reapers are excluded.
        A sink is invoked exactly once after cleanup, even before the first yield
        or after finish() already closed the owner. Production callers must use a
        durable sink; its successful return is the persistence acknowledgement.
        Sink failure cannot interrupt cleanup, and cannot be treated as closure.
        """
        _linux()
        if threading.current_thread() is not threading.main_thread():
            raise OwnershipError("managed creation must run on the main thread")
        if receipt_sink is not None and not callable(receipt_sink):
            raise ValueError("receipt sink must be callable")
        if (
            type(term_grace_seconds) not in (int, float)
            or not math.isfinite(term_grace_seconds)
            or not 0 <= term_grace_seconds <= 120
        ):
            raise ValueError("TERM grace must be between 0 and 120 seconds")
        previous = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        }
        pending = []
        armed = False
        owner = None
        receipt = None

        def interrupted(signum, _frame):
            if armed:
                raise ProcessInterrupted(signum)
            pending.append(signum)

        try:
            for sig in previous:
                signal.signal(sig, interrupted)
            try:
                owner = cls.spawn(argv, **kwargs)
            except SpawnOwnershipError as exc:
                receipt = exc.receipt
                raise
            armed = True
            if pending:
                raise ProcessInterrupted(pending[0])
            yield owner
        finally:
            armed = False
            original_failure = sys.exc_info()[1]
            try:
                if owner is not None:
                    receipt = owner.close(term_grace_seconds=term_grace_seconds)
                if receipt is not None and receipt_sink is not None:
                    try:
                        receipt_sink(owner, receipt)
                    except BaseException as sink_error:
                        raise ReceiptSinkError(
                            receipt, owner, sink_error, original_failure
                        ) from sink_error
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    def _anchor(self):
        _linux()
        if self._receipt is not None:
            raise OwnershipError("ownership has already been closed")
        creator, _, _, _ = _kernel(os.getpid(), include_runtime=False)
        if _anchor_key(creator) != _anchor_key(self._creator):
            raise OwnershipError("only the original live creator may manage this process")
        if self._process.pid != self.identity.pid or self._process.returncode is not None:
            raise OwnershipError("leader was already reaped; group authority is lost")
        # Unlike poll()/wait(), WNOWAIT cannot free the leader's PID/PGID anchor.
        os.waitid(os.P_PID, self.identity.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        observed, cmdline, state, ppid = _kernel(self.identity.pid, include_runtime=False)
        if _anchor_key(observed) != _anchor_key(self.identity) or ppid != self._creator.pid:
            raise OwnershipError("leader kernel identity no longer matches ownership")
        return observed, cmdline, state

    def poll_exit(self) -> bool:
        with self._lock:
            self._anchor()
            value = os.waitid(os.P_PID, self.identity.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            return value is not None

    def snapshot(self) -> ProcessSnapshot:
        with self._lock:
            anchored, _, _ = self._anchor()
            observed, cmdline, state, _ = _kernel(self.identity.pid)
            if _anchor_key(observed) != _anchor_key(anchored):
                raise OwnershipError("identity changed while reading live process details")
            value = ProcessSnapshot(self.binding, observed, cmdline, state)
            self._snapshots[id(value)] = value
            return value

    def verify_snapshot(self, snapshot: ProcessSnapshot) -> bool:
        """Authenticate this owner's historical live observation, never current liveness."""
        with self._lock:
            if (
                self._snapshots.get(id(snapshot)) is not snapshot
                or snapshot.binding != self.binding
                or snapshot.identity != self.identity
                or snapshot.state in {"Z", "X"}
                or snapshot.cmdline != self._argv
            ):
                raise OwnershipError("snapshot is not a live observation from this owner")
            return True

    def verify_live(self) -> ProcessSnapshot:
        value = self.snapshot()
        if (
            value.state in {"Z", "X"}
            or value.cmdline != self._argv
            or value.identity.slurm_job_id != self.identity.slurm_job_id
            or value.identity.step_id != self.identity.step_id
        ):
            raise OwnershipError("service is not the original live command/allocation")
        return value

    @contextmanager
    def interrupt_guard(self):
        """Turn TERM/HUP into unwind so callers' finally blocks can close this owner."""
        if threading.current_thread() is not threading.main_thread():
            raise OwnershipError("signal cleanup guard must run on the main thread")
        previous = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        }

        def interrupted(signum, _frame):
            raise ProcessInterrupted(signum)

        try:
            for sig in previous:
                signal.signal(sig, interrupted)
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def finish(
        self, *, deadline_monotonic: float, term_grace_seconds: float = 120
    ) -> CleanupReceipt:
        if not math.isfinite(deadline_monotonic):
            raise ValueError("finite process deadline required")
        try:
            while not self.poll_exit() and time.monotonic() < deadline_monotonic:
                time.sleep(0.01)
        finally:
            receipt = self.close(term_grace_seconds=term_grace_seconds)
        return receipt

    def close(self, *, term_grace_seconds: float = 120) -> CleanupReceipt:
        if (
            type(term_grace_seconds) not in (int, float)
            or not math.isfinite(term_grace_seconds)
            or not 0 <= term_grace_seconds <= 120
        ):
            raise ValueError("TERM grace must be between 0 and 120 seconds")
        with self._lock:
            if self._receipt is not None:
                return self._receipt
            term_sent = kill_sent = reaped = verified = False
            remaining: tuple[int, ...] = ()
            errors: list[str] = []
            # Repeated TERM must not interrupt the bounded cleanup itself.
            signals = {}
            if threading.current_thread() is threading.main_thread():
                signals = {
                    sig: signal.getsignal(sig)
                    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
                }
                for sig in signals:
                    signal.signal(sig, signal.SIG_IGN)
            try:
                self._anchor()
                _members(self.identity)
                self._anchor()
                os.killpg(self.identity.pgid, signal.SIGTERM)
                term_sent = True
                deadline = time.monotonic() + term_grace_seconds
                while True:
                    self._anchor()
                    remaining = tuple(
                        sorted(
                            member.pid
                            for member, state in _members(self.identity)
                            if state not in {"Z", "X"}
                        )
                    )
                    if not remaining:
                        break
                    if time.monotonic() >= deadline:
                        self._anchor()
                        os.killpg(self.identity.pgid, signal.SIGKILL)
                        kill_sent = True
                        break
                    time.sleep(0.01)
                kill_deadline = time.monotonic() + 5
                while remaining and time.monotonic() < kill_deadline:
                    self._anchor()
                    remaining = tuple(
                        sorted(
                            member.pid
                            for member, state in _members(self.identity)
                            if state not in {"Z", "X"}
                        )
                    )
                    if remaining:
                        time.sleep(0.01)
                if remaining:
                    raise OwnershipError("owned group did not become quiescent after cleanup")
                self._anchor()
                # Only now may this owner release the leader PID/PGID anchor.
                self._process.wait(timeout=5)
                reaped = verified = True
            except (OSError, ValueError, OwnershipError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                for sig, handler in signals.items():
                    signal.signal(sig, handler)
            self._receipt = CleanupReceipt(
                self.binding,
                self.identity,
                self._process.returncode,
                term_sent,
                kill_sent,
                reaped,
                verified,
                remaining,
                tuple(errors),
            )
            return self._receipt

    def verify_cleanup(self, receipt: CleanupReceipt) -> bool:
        if (
            receipt is not self._receipt
            or not isinstance(receipt, CleanupReceipt)
            or not receipt.group_cleanup_verified
            or not receipt.leader_reaped
        ):
            raise OwnershipError("cleanup is not a verified receipt from this live owner")
        return True

    def verify_endpoint(self, base_url: str) -> EndpointObservation:
        """Bind a local listening TCP socket to actual owned-group file descriptors."""
        with self._lock:
            process = self.verify_live()
            url = urlsplit(base_url)
            if (
                url.scheme != "http"
                or url.hostname not in {"127.0.0.1", "::1"}
                or url.username
                or url.password
                or url.query
                or url.fragment
                or not url.port
            ):
                raise OwnershipError("endpoint must be an explicit local HTTP loopback port")
            own_net = os.readlink("/proc/self/ns/net")
            if os.readlink(f"/proc/{self.identity.pid}/ns/net") != own_net:
                raise OwnershipError("service uses a different network namespace")
            family = "tcp" if url.hostname == "127.0.0.1" else "tcp6"
            allowed = (
                {"00000000", "0100007F"}
                if family == "tcp"
                else {
                    "0" * 32,
                    "00000000000000000000000001000000",
                }
            )
            inodes = set()
            for row in Path(f"/proc/net/{family}").read_text(encoding="ascii").splitlines()[1:]:
                fields = row.split()
                address, port = fields[1].split(":")
                if fields[3] == "0A" and address in allowed and int(port, 16) == url.port:
                    if int(fields[7]) != self.identity.effective_uid:
                        raise OwnershipError("endpoint has an unowned listener UID")
                    inodes.add(int(fields[9]))
            holders = []
            for member, state in _members(self.identity):
                if state in {"Z", "X"}:
                    continue
                for fd in Path(f"/proc/{member.pid}/fd").iterdir():
                    try:
                        target = os.readlink(fd)
                    except FileNotFoundError:
                        continue
                    match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
                    if match and int(match[1]) in inodes:
                        holders.append((member.pid, member.start_marker, int(match[1])))
                current, _, current_state, _ = _kernel(member.pid, include_runtime=False)
                if _anchor_key(current) != _anchor_key(member) or current_state in {"Z", "X"}:
                    raise OwnershipError("socket holder changed during descriptor inspection")
            if not inodes or {item[2] for item in holders} != inodes:
                raise OwnershipError("listener socket is not bound to the owned process group")
            after = self.verify_live()
            if after.identity != process.identity or after.cmdline != process.cmdline:
                raise OwnershipError("service identity changed during endpoint verification")
            return EndpointObservation(
                process, base_url, own_net, tuple(sorted(inodes)), tuple(sorted(set(holders)))
            )
