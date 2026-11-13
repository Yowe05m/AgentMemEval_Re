"""Content-bound Python entry points that preserve an actual virtual environment.

A venv commonly symlinks its Python to a different environment. Resolving that
entry before execution changes sys.prefix. This verifier preserves the entry
argv, pins the resolved executable and pyvenv.cfg, and observes the real prefix
with an isolated, bounded stdlib subprocess. It does not attest GPU libraries.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations.contracts import content_digest

PROBE = (
    "import json,os,sys;print(json.dumps({"
    "'executable':sys.executable,'resolved_executable':os.path.realpath(sys.executable),"
    "'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
    "'version':list(sys.version_info[:3]),'implementation':sys.implementation.name"
    "},sort_keys=True))"
)


class InterpreterError(ValueError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise InterpreterError(message)


def _canonical_parent(path: Path) -> None:
    _require(path.is_absolute(), "Python entry must be absolute")
    _require(not any(char in str(path) for char in "\x00\r\n"), "invalid Python path")
    _require(path.parent.resolve(strict=True) == path.parent,
             "Python entry ancestors must use their canonical paths")
    current = path.parent
    while current != current.parent:
        _require(not current.is_symlink(), "Python entry ancestor is a symlink")
        current = current.parent


def _signature(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _regular_record(path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    _canonical_parent(path)
    before = _signature(path)
    _require(stat.S_ISREG(before[2]), "interpreter pin must be a regular file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    _require(before == _signature(path) and size == before[3], "interpreter pin changed")
    return {"path": str(path), "size": size, "sha256": digest.hexdigest()}, before


def _snapshot(entry: Path) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    _canonical_parent(entry)
    chain, signatures, seen = [], {}, set()
    current = entry
    for _ in range(16):
        _require(str(current) not in seen, "Python entry symlink cycle")
        seen.add(str(current))
        signatures[str(current)] = _signature(current)
        if not current.is_symlink():
            break
        raw_target = os.readlink(current)
        target_path = raw_target
        # Windows readlink exposes the native extended filesystem namespace;
        # normalize only its ordinary drive/UNC spellings, not device paths.
        if os.name == "nt" and target_path.startswith("\\\\?\\UNC\\"):
            target_path = "\\\\" + target_path[8:]
        elif (os.name == "nt" and target_path.startswith("\\\\?\\")
              and len(target_path) >= 7 and target_path[4].isalpha()
              and target_path[5:7] == ":\\"):
            target_path = target_path[4:]
        candidate = Path(target_path)
        if not candidate.is_absolute():
            candidate = current.parent / candidate
        # Alias ancestors in an existing link target are resolved and recorded;
        # the executable argv itself always retains its canonical venv entry.
        candidate = candidate.parent.resolve(strict=True) / candidate.name
        chain.append({"path": str(current), "target": raw_target,
                      "next_path": str(candidate)})
        current = candidate
    else:
        raise InterpreterError("Python entry symlink chain is too long")
    resolved = entry.resolve(strict=True)
    _require(current.resolve(strict=True) == resolved and resolved.is_file(),
             "Python entry is not a file")
    executable, signature = _regular_record(resolved)
    signatures[str(resolved)] = signature
    pins = [executable]
    config = None
    venv_root = None
    if entry.parent.name.casefold() in {"bin", "scripts"}:
        config_path = entry.parent.parent / "pyvenv.cfg"
        if os.path.lexists(config_path):
            config, signature = _regular_record(config_path)
            signatures[str(config_path)] = signature
            pins.append(config)
            venv_root = str(entry.parent.parent)
    return {"entry_path": str(entry), "resolved_executable": str(resolved),
            "executable_sha256": executable["sha256"], "symlink_chain": chain,
            "files": sorted(pins, key=lambda row: row["path"]),
            "venv_config": config, "venv_root": venv_root}, signatures


def verify_python_interpreter(
    executable: Path,
    expected_executable_sha256: str,
    *,
    expected_prefix: Path | str | None = None,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Return a stable JSON proof; compare proof_sha256 at every use boundary.

    ``files`` contains only resolved regular files, suitable for the strict
    submission helper's input pins. Link/prefix evidence must also be rechecked
    in the launched process before it performs any probe or scientific work.
    """
    _require(isinstance(executable, Path), "Python entry must be a Path")
    _require(type(expected_executable_sha256) is str and re.fullmatch(
        r"[0-9a-f]{64}", expected_executable_sha256) is not None,
        "expected executable SHA-256 is invalid")
    _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 30, "interpreter probe timeout must be within 0..30")
    try:
        before, signatures = _snapshot(executable)
        _require(before["executable_sha256"] == expected_executable_sha256,
                 "Python executable content differs")
        environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
        if os.name == "nt" and os.environ.get("SystemRoot"):
            environment["SystemRoot"] = os.environ["SystemRoot"]
        result = subprocess.run(
            [str(executable), "-I", "-B", "-c", PROBE], shell=False,
            env=environment, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout_seconds, check=False,
        )
        _require(result.returncode == 0 and not result.stderr.strip()
                 and 0 < len(result.stdout) <= 32768,
                 "isolated interpreter observation failed")
        observed = json.loads(result.stdout.decode("utf-8"))
        _require(type(observed) is dict and set(observed) == {
            "executable", "resolved_executable", "prefix", "base_prefix",
            "version", "implementation"}, "invalid interpreter observation")
        for key in ("executable", "resolved_executable", "prefix", "base_prefix"):
            _require(type(observed[key]) is str and Path(observed[key]).is_absolute(),
                     "interpreter reported an invalid path")
        _require(os.path.normcase(os.path.normpath(observed["executable"]))
                 == os.path.normcase(os.path.normpath(str(executable))),
                 "interpreter observation did not preserve the entry argv")
        _require(Path(observed["resolved_executable"]).resolve(strict=True)
                 == Path(before["resolved_executable"]), "interpreter target changed")
        _require(observed["implementation"] == "cpython"
                 and type(observed["version"]) is list and len(observed["version"]) == 3
                 and all(type(value) is int and value >= 0 for value in observed["version"])
                 and observed["version"][0] == 3, "unsupported Python implementation/version")
        prefix = str(Path(observed["prefix"]).resolve(strict=True))
        base_prefix = str(Path(observed["base_prefix"]).resolve(strict=True))
        if before["venv_root"] is not None:
            _require(prefix == before["venv_root"], "venv entry did not select its own prefix")
        if expected_prefix is not None:
            _require(prefix == str(Path(expected_prefix).resolve(strict=True)),
                     "Python environment prefix differs")
        after, after_signatures = _snapshot(executable)
        _require(before == after and signatures == after_signatures,
                 "interpreter entry/config/executable changed during observation")
        proof = {"schema_version": 1, "kind": "path1_python_interpreter_identity",
                 **before, "prefix": prefix, "base_prefix": base_prefix,
                 "version": observed["version"], "implementation": observed["implementation"],
                 "runtime_admission": False}
        proof["proof_sha256"] = content_digest(proof)
        return proof
    except InterpreterError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        raise InterpreterError("Python entry could not be verified") from exc
