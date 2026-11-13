from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import interpreters as module
from agentmemeval.path1_v1.operations.contracts import content_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, symlink=False):
    base = tmp_path / "base"
    base.mkdir()
    binary = base / "python"
    binary.write_bytes(b"pinned interpreter bytes")
    venv = tmp_path / "venv"
    bindir = venv / "bin"
    bindir.mkdir(parents=True)
    entry = bindir / "python"
    if symlink:
        try:
            entry.symlink_to(binary)
        except OSError:
            pytest.skip("this host does not permit creating symlinks")
    else:
        entry.write_bytes(binary.read_bytes())
        binary = entry
    config = venv / "pyvenv.cfg"
    config.write_text("home = " + str(base) + "\n", encoding="utf-8")
    observed = {"executable":str(entry), "resolved_executable":str(binary),
                "prefix":str(venv), "base_prefix":str(base),
                "version":[3, 11, 16], "implementation":"cpython"}
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(observed).encode(), b"")
    monkeypatch.setattr(module.subprocess, "run", run)
    return entry, binary, config, observed, calls, run


def test_real_current_interpreter_has_stable_content_and_prefix_proof():
    # Actual interpreter process: no fixture or model dependency is imported by it.
    entry = Path(sys.executable).absolute()
    proof = module.verify_python_interpreter(entry, sha(entry), expected_prefix=sys.prefix)
    again = module.verify_python_interpreter(entry, sha(entry), expected_prefix=sys.prefix)
    assert proof == again
    assert proof["prefix"] == str(Path(sys.prefix).resolve())
    assert proof["version"] == list(sys.version_info[:3])
    assert proof["runtime_admission"] is False
    assert proof["proof_sha256"] == content_digest(
        {k: v for k, v in proof.items() if k != "proof_sha256"}
    )


def test_venv_symlink_preserves_entry_and_pins_only_real_files(tmp_path, monkeypatch):
    entry, binary, config, observed, calls, _ = fixture(tmp_path, monkeypatch, symlink=True)
    proof = module.verify_python_interpreter(entry, sha(binary), expected_prefix=observed["prefix"])
    assert calls[0][0] == [str(entry), "-I", "-B", "-c", module.PROBE]
    assert calls[0][1]["shell"] is False
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert proof["entry_path"] == str(entry)
    assert proof["resolved_executable"] == str(binary)
    assert proof["prefix"] != proof["base_prefix"]
    assert proof["symlink_chain"]
    assert {row["path"] for row in proof["files"]} == {str(binary), str(config)}
    assert all(not Path(row["path"]).is_symlink() for row in proof["files"])


def test_different_venv_config_has_different_stable_proof(tmp_path, monkeypatch):
    entry, binary, config, _, _, _ = fixture(tmp_path, monkeypatch)
    first = module.verify_python_interpreter(entry, sha(binary))
    config.write_text("home = changed\n", encoding="utf-8")
    second = module.verify_python_interpreter(entry, sha(binary))
    assert first["executable_sha256"] == second["executable_sha256"]
    assert first["proof_sha256"] != second["proof_sha256"]


@pytest.mark.parametrize("part", ["binary", "config", "entry"])
def test_observation_rejects_identity_changes(tmp_path, monkeypatch, part):
    entry, binary, config, _, _, run = fixture(tmp_path, monkeypatch, symlink=part == "entry")
    expected = sha(binary)
    def changing(argv, **kwargs):
        result = run(argv, **kwargs)
        if part == "binary":
            binary.write_bytes(b"changed executable")
        elif part == "config":
            config.write_bytes(b"changed configuration")
        else:
            other = binary.with_name("other")
            other.write_bytes(binary.read_bytes())
            entry.unlink()  # only this test's freshly created link
            entry.symlink_to(other)
        return result
    monkeypatch.setattr(module.subprocess, "run", changing)
    with pytest.raises(module.InterpreterError, match="changed"):
        module.verify_python_interpreter(entry, expected)


@pytest.mark.parametrize(
    "case", ["wrong_hash", "wrong_prefix", "lost_venv", "wrong_entry", "wrong_target"]
)
def test_mismatched_execution_environment_is_rejected(tmp_path, monkeypatch, case):
    entry, binary, _, observed, calls, _ = fixture(tmp_path, monkeypatch)
    expected = sha(binary)
    kwargs = {}
    if case == "wrong_hash":
        expected = "a" * 64
    elif case == "wrong_prefix":
        kwargs["expected_prefix"] = observed["base_prefix"]
    elif case == "lost_venv":
        observed["prefix"] = observed["base_prefix"]
    elif case == "wrong_entry":
        observed["executable"] = str(Path(observed["base_prefix"]) / "python")
    else:
        observed["resolved_executable"] = str(Path(observed["base_prefix"]) / "python")
    with pytest.raises(module.InterpreterError):
        module.verify_python_interpreter(entry, expected, **kwargs)
    if case == "wrong_hash":
        assert not calls


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "stderr", "invalid_json", "oversized"])
def test_failed_observation_never_returns_proof(tmp_path, monkeypatch, failure):
    entry, binary, _, _, _, _ = fixture(tmp_path, monkeypatch)
    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, int(failure == "nonzero"),
            b"x" * 32769 if failure == "oversized" else b"not JSON",
            b"warning" if failure == "stderr" else b"")
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module.InterpreterError):
        module.verify_python_interpreter(entry, sha(binary))


def test_symlink_ancestor_cannot_select_an_unpinned_entry(tmp_path, monkeypatch):
    entry, binary, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(entry.parent, target_is_directory=True)
    except OSError:
        pytest.skip("this host does not permit creating symlinks")
    with pytest.raises(module.InterpreterError, match="canonical|ancestor"):
        module.verify_python_interpreter(alias / entry.name, sha(binary))
    assert not calls
