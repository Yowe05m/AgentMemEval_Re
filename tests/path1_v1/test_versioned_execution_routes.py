"""The optional worker stays versioned without narrowing origin attestation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations.intent_executor import (
    ExecutorError,
    render_worker_script,
    verify_loaded_package,
)

from .unit.test_intent_executor import _loaded_source_fixture, make_executor_fixture, submit


@pytest.mark.parametrize("module_name", ["agentmemeval", "agentmemeval.core.domain"])
def test_loaded_parent_and_main_sibling_require_declared_bytes(module_name):
    module = importlib.import_module(module_name)
    repository, records = _loaded_source_fixture()
    relative = Path(module.__file__).resolve().relative_to(repository).as_posix()
    records = [record for record in records if record["path"] != relative]
    with pytest.raises(
        ExecutorError, match="outside the contract closure|escapes the source closure",
    ):
        verify_loaded_package(repository, records)


@pytest.mark.parametrize("module_name", ["agentmemeval", "agentmemeval.core.domain"])
def test_loaded_parent_and_main_sibling_reject_same_bytes_from_other_tree(
    tmp_path, monkeypatch, module_name,
):
    original = importlib.import_module(module_name)
    repository, records = _loaded_source_fixture()
    foreign_file = tmp_path / "foreign.py"
    foreign_file.write_bytes(Path(original.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(module_name, foreign_file)
    foreign = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, foreign)
    with pytest.raises(ExecutorError, match="another source tree"):
        verify_loaded_package(repository, records)


def test_launcher_routes_only_to_versioned_worker(tmp_path):
    script = render_worker_script(
        python_executable=Path(sys.executable).resolve(),
        repository_root=tmp_path, worker_request_path=tmp_path / "request.json",
    )
    assert b" -s -m agentmemeval.path1_v1.operations.worker --request " in script
    assert b" -m agentmemeval.operations.worker " not in script


def test_legacy_launcher_is_rejected_even_with_its_matching_content_digest(tmp_path):
    fixture = make_executor_fixture(tmp_path)
    script = Path(fixture.request.spec.script_path)
    # The fixture uses a portable logical remote path; its verified local
    # mapping is the launcher written at this root.
    if not script.is_file():
        script = tmp_path / "launcher.sh"
    original = script.read_bytes()
    altered = original.replace(b"agentmemeval.path1_v1.operations.worker",
                               b"agentmemeval.operations.worker")
    assert altered != original
    script.write_bytes(altered)
    fixture.request = replace(fixture.request, spec=replace(
        fixture.request.spec, script_sha256=hashlib.sha256(altered).hexdigest(),
    ))
    with pytest.raises(ExecutorError, match="launcher bytes/topology"):
        submit(fixture)


def test_versioned_module_cli_is_importable_and_missing_request_fails_closed(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-s", "-m", "agentmemeval.path1_v1.operations.worker",
         "--request", str(tmp_path / "missing.json")],
        cwd=tmp_path, env=env, capture_output=True, timeout=20,
    )
    assert result.returncode == 1, result.stderr.decode(errors="replace")
    assert b"FileNotFoundError" in result.stderr
    assert not list(tmp_path.iterdir())
