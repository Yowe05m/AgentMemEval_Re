"""Run with stdlib Python on Linux: SCRIPT REPOSITORY NEW_OUTPUT_DIRECTORY.

Creates only finite fixture processes, and only their live owners clean them.
No Slurm, SSH, GPU or model program is called. This is not a site release probe.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

SAFE_ENV = {
    "PATH": os.defpath,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
}

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
MODULE = REPO / "src/agentmemeval/path1_v1/operations/process_ownership.py"
spec = importlib.util.spec_from_file_location("smoke_ownership", MODULE)
ownership = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ownership
spec.loader.exec_module(ownership)
results = []


def binding(name):
    return ownership.ProcessBinding("linux_smoke", name, None, "a" * 64, "model_service", name)


def spawn(name, source):
    return ownership.OwnedProcess.spawn(
        [sys.executable, "-c", source],
        binding=binding(name),
        cwd=OUT,
        env=SAFE_ENV,
    )


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"fixture did not create {path.name}")
        time.sleep(0.01)
    return path.read_text()


def gone_or_zombie(pid):
    try:
        return ownership._stat(pid)[4] in {"Z", "X"}
    except FileNotFoundError:
        return True


def record(name, receipt):
    assert receipt.group_cleanup_verified and receipt.leader_reaped, asdict(receipt)
    results.append({"case": name, "receipt": asdict(receipt)})


def descendants_source(pidfile, *, leader_exit=False, ignore=False):
    child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    return (
        "import os,sys,time,signal,subprocess,pathlib;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore else "")
        + f"child=subprocess.Popen([sys.executable,'-c',{child!r}]);"
        + f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid));"
        + ("time.sleep(.1)" if leader_exit else "time.sleep(30)")
    )


sentinel = spawn("unrelated", "import time;time.sleep(30)")
try:
    for name, exits, ignore in (("normal_leader_exit", True, False), ("term_ignored", False, True)):
        pidfile = OUT / (name + ".pid")
        owner = spawn(name, descendants_source(pidfile, leader_exit=exits, ignore=ignore))
        try:
            child_pid = int(wait_file(pidfile))
            time.sleep(0.15)  # allow the fixture to install its deliberate TERM handler
            receipt = owner.finish(
                deadline_monotonic=time.monotonic() + 0.2, term_grace_seconds=0.15
            )
            record(name, receipt)
            assert receipt.kill_sent and gone_or_zombie(child_pid)
            assert sentinel.verify_live().identity.pid == sentinel.identity.pid
        finally:
            owner.close(term_grace_seconds=0.1)

    owner = spawn("parent_interrupted", "import time;time.sleep(30)")
    try:
        with owner.interrupt_guard():
            timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            try:
                owner.finish(deadline_monotonic=time.monotonic() + 10, term_grace_seconds=0.1)
            except ownership.ProcessInterrupted:
                pass
            else:
                raise AssertionError("TERM did not interrupt the controller")
            timer.join()
        record("parent_interrupted", owner.close(term_grace_seconds=0.1))
    finally:
        owner.close(term_grace_seconds=0.1)

    owner = spawn("identity_negative", "import time;time.sleep(30)")
    original = owner.identity
    try:
        for field, value in (
            ("boot_id", "wrong"),
            ("effective_uid", original.effective_uid + 1),
            ("start_marker", "0"),
            ("pgid", original.pgid + 1),
            ("session_id", original.session_id + 1),
            ("pid", sentinel.identity.pid),
        ):
            owner._identity = replace(original, **{field: value})
            try:
                owner.snapshot()
            except (ownership.OwnershipError, ChildProcessError):
                results.append({"case": "identity_rejected_" + field})
            else:
                raise AssertionError("incorrect identity was accepted")
            finally:
                owner._identity = original
        sentinel.verify_live()
    finally:
        record("identity_negative_cleanup", owner.close(term_grace_seconds=0.1))

    # Send TERM at the exact post-Popen/pre-owner boundary. managed() must defer
    # unwind until it can clean the newly created group, and restore handlers.
    original_kernel = ownership._kernel
    created = []

    def interrupt_at_kernel(pid, **kwargs):
        result = original_kernel(pid, **kwargs)
        if pid != os.getpid() and not created:
            created.append(pid)
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    ownership._kernel = interrupt_at_kernel
    try:
        try:
            with ownership.OwnedProcess.managed(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                binding=binding("creation_interrupt"),
                cwd=OUT,
                env=SAFE_ENV,
                term_grace_seconds=0.1,
            ):
                raise AssertionError("deferred TERM should unwind before body")
        except ownership.ProcessInterrupted:
            pass
        assert created and gone_or_zombie(created[0])
        results.append({"case": "term_during_creation_cleaned", "pid": created[0]})
    finally:
        ownership._kernel = original_kernel

    failed_pid = []

    def fail_first_child_kernel(pid, **kwargs):
        if pid != os.getpid() and not failed_pid:
            failed_pid.append(pid)
            raise PermissionError("controlled identity-read fault")
        return original_kernel(pid, **kwargs)

    ownership._kernel = fail_first_child_kernel
    try:
        try:
            with ownership.OwnedProcess.managed(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                binding=binding("creation_fault"),
                cwd=OUT,
                env=SAFE_ENV,
                term_grace_seconds=0.1,
            ):
                raise AssertionError("failed identity cannot return an owner")
        except ownership.SpawnOwnershipError as exc:
            assert exc.receipt.direct_child_reaped and not exc.receipt.group_cleanup_verified
            results.append(
                {"case": "failed_creation_direct_child_only", "receipt": asdict(exc.receipt)}
            )
        assert failed_pid and gone_or_zombie(failed_pid[0])
        sentinel.verify_live()
    finally:
        ownership._kernel = original_kernel

    owner = spawn("externally_reaped", "pass")
    owner._process.wait(timeout=5)  # deliberate foreign reaper fault, never production API
    receipt = owner.close(term_grace_seconds=0.1)
    assert not receipt.group_cleanup_verified and not receipt.term_sent
    results.append({"case": "externally_reaped_refused", "receipt": asdict(receipt)})

    for name in ("listener", "foreign_listener"):
        file = OUT / (name + ".port")
        source = (
            "import socket,pathlib,time; s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
            f"pathlib.Path({str(file)!r}).write_text(str(s.getsockname()[1]));time.sleep(30)"
        )
        listener = spawn(name, source)
        try:
            port = int(wait_file(file))
            endpoint = listener.verify_endpoint(f"http://127.0.0.1:{port}/v1")
            assert endpoint.listener_inodes and endpoint.socket_holders
            try:
                sentinel.verify_endpoint(f"http://127.0.0.1:{port}/v1")
            except ownership.OwnershipError:
                pass
            else:
                raise AssertionError("a foreign process's same-user socket was admitted")
            results.append({"case": name + "_socket_binding", "endpoint": asdict(endpoint)})
        finally:
            record(name + "_cleanup", listener.close(term_grace_seconds=0.1))

    # Actual standalone submission helper, with a harmless command replacing sbatch.
    helper = REPO / "src/agentmemeval/path1_v1/operations/submission_process.py"
    for mode in ("timeout", "success"):
        directory = OUT / ("helper_" + mode)
        directory.mkdir()
        pidfile = directory / "descendant.pid"
        command = directory / "command.py"
        child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
        command.write_text(
            "import pathlib,subprocess,sys,time\n"
            f"p=subprocess.Popen([sys.executable,'-c',{child!r}],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
            "time.sleep(.15)\nprint('12345;fixture',flush=True)\n"
            + ("time.sleep(30)\n" if mode == "timeout" else ""),
            encoding="utf-8",
        )
        stdin = directory / "script.sh"
        stdin.write_bytes(b"#!/bin/sh\nexit 0\n")
        manifest = {
            "schema_version": 1,
            "nonce": "smoke",
            "request_sha256": "a" * 64,
            "argv": [sys.executable, str(command)],
            "cwd": str(directory),
            "environment": SAFE_ENV,
            "stdin_path": str(stdin),
            "stdin_sha256": hashlib.sha256(stdin.read_bytes()).hexdigest(),
            "input_files": [],
            "wait_seconds": 2,
            "command_timeout_seconds": 0.7,
            "deadline_monotonic": time.monotonic() + 3,
            "expected_effective_user": None,
        }
        path = directory / "invocation.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        owner = ownership.OwnedProcess.spawn(
            [
                sys.executable,
                "-I",
                str(helper),
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ],
            binding=binding("helper_" + mode),
            cwd=directory,
            env=SAFE_ENV,
        )
        try:
            ready = json.loads(wait_file(directory / "ready.json"))
            (directory / "go.json").write_text(json.dumps(ready), encoding="utf-8")
            child_pid = int(wait_file(pidfile))
            receipt = owner.finish(deadline_monotonic=time.monotonic() + 2, term_grace_seconds=0.15)
            record("helper_" + mode, receipt)
            assert gone_or_zombie(child_pid)
            if mode == "timeout":
                assert not (directory / "result.json").exists()
            else:
                result = json.loads((directory / "result.json").read_text())
                assert result["stdout"] == "12345;fixture" and not result["timed_out"]
        finally:
            owner.close(term_grace_seconds=0.1)
finally:
    record("unrelated_sentinel_survived_then_owned_cleanup", sentinel.close(term_grace_seconds=0.1))

report = {
    "platform": sys.platform,
    "python": sys.version,
    "cases": results,
    "scope": "real Linux local kernel only; no Slurm/GPU/site cgroup proof",
}
(OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"case_count": len(results), "status": "passed", "output": str(OUT)}))
