"""Literal-argv local subprocess boundary and handshake submitter.

This file can run under ``python -I`` without importing the package. The helper
announces its OS process identity, then waits for a separately persisted release
token before starting the command. A timeout or missing receipt never proves a
remote submission did not happen. It does not kill any scheduler job or service.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_identity(pid: int) -> dict[str, Any]:
    """Read a kernel creation marker; a PID alone is never an identity."""
    if type(pid) is not int or pid < 1:
        raise ValueError("invalid PID")
    extra: dict[str, Any] = {}
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        )
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            raise OSError(ctypes.get_last_error(), "process identity cannot be inspected")
        try:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                raise OSError(ctypes.get_last_error(), "process creation time unavailable")
            marker = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
        boot_id = "windows-filetime"
    elif sys.platform.startswith("linux"):
        # /proc stat's command may contain spaces and parentheses. Fields after
        # the final ')' start at field 3; starttime is field 22 (index 19 here).
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        marker = stat[stat.rfind(")") + 2 :].split()[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        extra["effective_uid"] = int(uid_line.split()[2])
    else:
        raise OSError("strong process identity is unsupported on this platform")
    return {
        "pid": pid,
        "host": socket.gethostname(),
        "platform": sys.platform,
        "start_marker": marker,
        "boot_id": boot_id,
        **extra,
    }


def current_effective_user() -> dict[str, Any]:
    """Read the current Unix credential and OS passwd database, not a receipt."""
    if not hasattr(os, "geteuid"):
        raise OSError("effective Unix credentials are unavailable")
    import pwd

    uid = os.geteuid()
    user = pwd.getpwuid(uid)
    return {"kind": "unix_effective_user", "uid": uid, "name": user.pw_name}


def observe_process(expected: dict[str, Any]) -> str:
    """Diagnostics only; neither absence nor PID reuse grants a retry/negative receipt."""
    try:
        actual = process_identity(expected["pid"])
    except (OSError, KeyError, ValueError):
        return "unavailable"
    return "same_identity" if actual == expected else "different_identity"


def write_exclusive_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic no-replace publication; readers never see a half-written ready/go/
    # result. Retain the partial inode as evidence, including on publication failure.
    os.link(partial, path)
    if os.name == "posix":
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("pinned invocation input changed")


def helper_main(manifest_path: Path, expected_sha256: str) -> int:
    _strict_file(manifest_path, expected_sha256)
    manifest = read_json(manifest_path)
    if (
        set(manifest)
        != {
            "schema_version",
            "nonce",
            "request_sha256",
            "argv",
            "cwd",
            "environment",
            "stdin_path",
            "stdin_sha256",
            "input_files",
            "wait_seconds",
            "command_timeout_seconds",
            "deadline_monotonic",
            "expected_effective_user",
        }
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported invocation manifest")
    directory = manifest_path.parent
    if sys.platform.startswith("linux") and (
        os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid()
    ):
        raise ValueError("Linux submitter must be its owner's isolated session leader")
    identity = process_identity(os.getpid())
    expected_user = manifest["expected_effective_user"]
    if expected_user is not None and current_effective_user() != expected_user:
        raise ValueError("submitter did not inherit the verified effective user")
    binding = {
        "nonce": manifest["nonce"],
        "request_sha256": manifest["request_sha256"],
        "manifest_sha256": expected_sha256,
        "submitter": identity,
        "effective_user": expected_user,
    }
    write_exclusive_json(directory / "ready.json", binding)
    wait_seconds, timeout = manifest["wait_seconds"], manifest["command_timeout_seconds"]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 60
        for value in (wait_seconds, timeout)
    ):
        raise ValueError("invalid process deadlines")
    deadline = time.monotonic() + wait_seconds
    while not (directory / "go.json").exists():
        if time.monotonic() >= deadline:
            return 70  # No release, no child command. Parent still retains unknown.
        time.sleep(0.01)
    if read_json(directory / "go.json") != binding:
        raise ValueError("release token does not match this process identity")
    if expected_user is not None and current_effective_user() != expected_user:
        raise ValueError("submitter effective user changed before external execution")
    _strict_file(manifest_path, expected_sha256)
    for record in manifest["input_files"]:
        _strict_file(Path(record["path"]), record["sha256"])
    stdin_path = Path(manifest["stdin_path"])
    _strict_file(stdin_path, manifest["stdin_sha256"])
    argv = manifest["argv"]
    if (
        type(argv) is not list
        or not argv
        or not Path(argv[0]).is_absolute()
        or any(type(arg) is not str or "\x00" in arg for arg in argv)
    ):
        raise ValueError("invalid literal argv")
    timeout = min(timeout, manifest["deadline_monotonic"] - time.monotonic())
    if timeout <= 0:
        return 71
    # No shell, no inherited SBATCH_* / SQUEUE_* variables, no retry. Pinned script
    # bytes are passed on stdin so sbatch cannot reread a changed script path.
    with stdin_path.open("rb") as stdin_stream:
        started = time.time_ns()
        process = subprocess.Popen(
            argv,
            cwd=manifest["cwd"],
            env=manifest["environment"],
            stdin=stdin_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform.startswith("linux"):
                # The parent retains this helper's unreaped session-leader PID
                # until it has TERM/KILLed the whole local group. Do not race its
                # cleanup by killing only sbatch, or waiting on inherited pipes.
                # Exiting here leaves the helper waitable as the PGID anchor.
                return 72
            # Only this exact Popen child is stopped, never a Slurm allocation.
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
    output = stdout.decode("utf-8", errors="replace").strip()
    safe_output = output if re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", output) else None
    result = {
        **binding,
        "command_started_ns": started,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": safe_output,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "command_finished_ns": time.time_ns(),
    }
    write_exclusive_json(directory / "result.json", result)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: submission_process.py MANIFEST SHA256")
    raise SystemExit(helper_main(Path(sys.argv[1]), sys.argv[2]))
