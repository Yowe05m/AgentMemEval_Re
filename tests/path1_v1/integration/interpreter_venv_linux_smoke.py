"""Actual Linux venv identity smoke; no pytest, model, Slurm or inherited-env log."""

from __future__ import annotations

import hashlib
import json
import sys
import venv
from pathlib import Path


def main() -> None:
    if sys.platform != "linux":
        raise RuntimeError("this smoke requires actual Linux")
    repository, output = (Path(value).resolve() for value in sys.argv[1:])
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(repository / "src"))
    from agentmemeval.path1_v1.operations.interpreters import (
        InterpreterError,
        verify_python_interpreter,
    )

    environment = output / "venv_a"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    entry = environment / "bin/python"
    expected = hashlib.sha256(entry.read_bytes()).hexdigest()
    first = verify_python_interpreter(entry, expected, expected_prefix=environment)
    second = verify_python_interpreter(entry, expected, expected_prefix=environment)
    assert first == second and first["symlink_chain"]
    assert first["entry_path"] == str(entry)
    assert first["prefix"] != first["base_prefix"]
    rejected = False
    try:
        verify_python_interpreter(entry.resolve(), expected, expected_prefix=environment)
    except InterpreterError:
        rejected = True
    assert rejected, "resolving away the venv entry must not preserve its admission"
    another = output / "venv_b"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(another)
    third = verify_python_interpreter(another / "bin/python", expected, expected_prefix=another)
    assert first["executable_sha256"] == third["executable_sha256"]
    assert first["proof_sha256"] != third["proof_sha256"]
    assert not first["runtime_admission"] and not third["runtime_admission"]
    receipt = {"status":"passed", "cases":["actual_venv_entry_preserves_prefix",
               "repeated_proof_is_stable", "resolved_base_loses_venv_and_is_rejected",
               "same_binary_different_venvs_have_different_proofs"],
               "first_proof":first, "second_environment_proof":third,
               "python":sys.version, "model_imports":False, "scientific_admission":False}
    with (output / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status":"passed", "cases":len(receipt["cases"])}, sort_keys=True))


if __name__ == "__main__":
    main()
