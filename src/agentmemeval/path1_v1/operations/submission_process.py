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
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

RAW_OUTPUT_LIMIT_BYTES = 1024 * 1024
_INVOCATION_KEYS = {
    "schema_version", "nonce", "request_sha256", "argv", "cwd", "environment",
    "stdin_path", "stdin_sha256", "input_files", "wait_seconds",
    "command_timeout_seconds", "deadline_monotonic", "expected_effective_user",
}


def invocation_result_schema(manifest: dict) -> int:
    """Read a hash-bound invocation version; never infer it from result JSON."""
    if type(manifest) is not dict:
        raise ValueError("invocation manifest must be an object")
    version = manifest.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported invocation manifest")
    expected = _INVOCATION_KEYS | ({"result_schema_version"} if version == 2 else set())
    if set(manifest) != expected:
        raise ValueError("invocation manifest fields differ")
    if version == 2 and (
        type(manifest["result_schema_version"]) is not int
        or manifest["result_schema_version"] != 2
    ):
        raise ValueError("invocation result schema differs")
    return version


def _signature(value) -> tuple:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _raw_bytes(path: Path) -> bytes:
    """Bounded regular-file read with stable inode/path checks, no symlink following."""
    if not path.is_absolute() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("invalid raw output parent path")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size > RAW_OUTPUT_LIMIT_BYTES + 1):
        raise ValueError("invalid raw command output file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if _signature(os.fstat(fd)) != _signature(before):
            raise ValueError("raw output identity changed before read")
        parts, remaining = [], RAW_OUTPUT_LIMIT_BYTES + 2
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                break
            parts.append(block)
            remaining -= len(block)
        data = b"".join(parts)
        if (len(data) != before.st_size
                or _signature(os.fstat(fd)) != _signature(before)
                or _signature(path.lstat()) != _signature(before)):
            raise ValueError("raw command output changed during read")
        return data
    finally:
        os.close(fd)


def validate_helper_result(
    directory: Path, result: dict, *, manifest: dict, expected_result_schema: int | None = None,
) -> int:
    """Validate raw evidence, not acceptance/rejection or any retry permission.

    The caller must first bind manifest/result bytes to its durable record.
    Historical manifest 1 results remain readable without invented raw files.
    """
    version = invocation_result_schema(manifest)
    if type(result) is not dict:
        raise ValueError("helper result must be an object")
    if expected_result_schema is not None and (
        type(expected_result_schema) is not int or expected_result_schema != version
    ):
        raise ValueError("durable helper result schema differs from invocation")
    actual = result.get("schema_version", 1)
    if type(actual) is not int or actual != version:
        raise ValueError("helper result schema downgrade or mismatch")
    if version == 1:
        if "raw_outputs" in result or "capture_status" in result:
            raise ValueError("legacy helper result cannot contain new evidence fields")
        return 1
    status = result.get("capture_status")
    if status not in {"complete", "timeout", "output_limit", "error"}:
        raise ValueError("invalid command capture status")
    if (type(result.get("timed_out")) is not bool
            or result["timed_out"] != (status == "timeout")
            or type(result.get("command_spawned")) is not bool
            or (result.get("returncode") is not None and type(result["returncode"]) is not int)
            or (status == "complete" and (
                not result["command_spawned"] or type(result["returncode"]) is not int))):
        raise ValueError("invalid command completion evidence")
    error = result.get("capture_error_type")
    if (status == "error" and (type(error) is not str or not error)
            or status != "error" and error is not None):
        raise ValueError("invalid command capture failure evidence")
    outputs = result.get("raw_outputs")
    if type(outputs) is not dict or set(outputs) != {"stdout", "stderr"}:
        raise ValueError("raw output inventory differs")
    payloads = {}
    for name in ("stdout", "stderr"):
        item = outputs[name]
        if (type(item) is not dict
                or set(item) != {"path", "size", "sha256", "limit_bytes", "complete",
                                 "limit_exceeded"}
                or item["path"] != name + ".raw"
                or type(item["size"]) is not int
                or not 0 <= item["size"] <= RAW_OUTPUT_LIMIT_BYTES + 1
                or type(item["limit_bytes"]) is not int
                or item["limit_bytes"] != RAW_OUTPUT_LIMIT_BYTES
                or type(item["complete"]) is not bool
                or type(item["limit_exceeded"]) is not bool
                or item["limit_exceeded"] != (item["size"] > RAW_OUTPUT_LIMIT_BYTES)
                or item["limit_exceeded"] and item["complete"]):
            raise ValueError("invalid raw output descriptor")
        raw = _raw_bytes(directory / item["path"])
        if (len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]
                or result.get(name + "_sha256") != item["sha256"]):
            raise ValueError("raw output bytes differ from result")
        payloads[name] = raw
    exceeded = any(x["limit_exceeded"] for x in outputs.values())
    if (status == "output_limit") != exceeded:
        raise ValueError("output limit status differs from raw evidence")
    if status == "complete" and not all(x["complete"] for x in outputs.values()):
        raise ValueError("complete result has incomplete streams")
    display = payloads["stdout"].decode("utf-8", errors="replace").strip()
    safe = display if re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", display) else None
    if result.get("stdout") != safe:
        raise ValueError("display stdout differs from raw bytes")
    return 2


def _sync_directory(directory: Path) -> None:
    if os.name == "posix":
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class _RawCapture:
    def __init__(self, path: Path):
        self.path = path
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        self.fd = os.open(path, flags, 0o600)
        self.identity = os.fstat(self.fd)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.complete = False
        self.exceeded = False
        self.error = None
        self.frozen = False
        self.size = 0

    def pump(self, pipe) -> None:
        try:
            while True:
                block = os.read(pipe.fileno(), min(65536, RAW_OUTPUT_LIMIT_BYTES + 1 - self.size))
                with self.lock:
                    if self.frozen:
                        return
                    if not block:
                        self.complete = True
                        return
                    remaining = memoryview(block)
                    while remaining:
                        count = os.write(self.fd, remaining)
                        if count <= 0:
                            raise OSError("raw output write made no progress")
                        self.size += count
                        remaining = remaining[count:]
                    # Prefixes survive timeout/termination even without a final result.
                    os.fsync(self.fd)
                    if self.size > RAW_OUTPUT_LIMIT_BYTES:
                        self.exceeded = True
                        return
        except Exception as exc:
            with self.lock:
                self.error = type(exc).__name__
        finally:
            self.done.set()

    def freeze(self) -> dict:
        with self.lock:
            self.frozen = True
            try:
                os.fsync(self.fd)
                current = os.fstat(self.fd)
                if ((current.st_dev, current.st_ino) !=
                        (self.identity.st_dev, self.identity.st_ino)
                        or _signature(current) != _signature(self.path.lstat())):
                    raise ValueError("raw output path no longer names its created inode")
            finally:
                os.close(self.fd)
            raw = _raw_bytes(self.path)
            return {"path": self.path.name, "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "limit_bytes": RAW_OUTPUT_LIMIT_BYTES, "complete": self.complete,
                    "limit_exceeded": self.exceeded}


def _capture_command(argv, manifest: dict, directory: Path, stdin_stream, timeout: float) -> dict:
    outputs = {name: _RawCapture(directory / (name + ".raw")) for name in ("stdout", "stderr")}
    _sync_directory(directory)
    process, status, failure = None, "error", None
    started = time.time_ns()
    try:
        process = subprocess.Popen(
            argv, cwd=manifest["cwd"], env=manifest["environment"], stdin=stdin_stream,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
        )
        for name, capture in outputs.items():
            threading.Thread(target=capture.pump, args=(getattr(process, name),),
                             daemon=True).start()
        deadline = time.monotonic() + timeout
        while True:
            if any(x.exceeded for x in outputs.values()):
                status = "output_limit"
                break
            if any(x.error for x in outputs.values()):
                failure = next(x.error for x in outputs.values() if x.error)
                break
            if all(x.done.is_set() for x in outputs.values()) and process.poll() is not None:
                status = "complete"
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                break
            time.sleep(0.005)
    except Exception as exc:
        failure = type(exc).__name__
    if process is not None and status != "complete" and not sys.platform.startswith("linux"):
        # Only the exact Windows fixture child. Linux group cleanup remains with
        # the parent owner, whose unreaped helper leader anchors the PGID.
        process.kill()
        process.wait(timeout=5)
    records = {name: capture.freeze() for name, capture in outputs.items()}
    # Creation names, as well as the file content, must be durable before result.
    _sync_directory(directory)
    raw = _raw_bytes(directory / "stdout.raw")
    display = raw.decode("utf-8", errors="replace").strip()
    safe = display if re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", display) else None
    return {
        "schema_version": 2, "command_started_ns": started,
        "command_spawned": process is not None,
        "returncode": process.returncode if process is not None else None,
        "timed_out": status == "timeout", "capture_status": status,
        "capture_error_type": failure, "raw_outputs": records, "stdout": safe,
        "stdout_sha256": records["stdout"]["sha256"],
        "stderr_sha256": records["stderr"]["sha256"], "command_finished_ns": time.time_ns(),
    }


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
    if invocation_result_schema(manifest) != 2:
        raise ValueError("new helper execution requires version 2 invocation")
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
        result = {**binding, **_capture_command(argv, manifest, directory, stdin_stream, timeout)}
    validate_helper_result(directory, result, manifest=manifest, expected_result_schema=2)
    write_exclusive_json(directory / "result.json", result)
    return {"complete": 0, "timeout": 72, "output_limit": 73, "error": 74}[result["capture_status"]]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: submission_process.py MANIFEST SHA256")
    raise SystemExit(helper_main(Path(sys.argv[1]), sys.argv[2]))
