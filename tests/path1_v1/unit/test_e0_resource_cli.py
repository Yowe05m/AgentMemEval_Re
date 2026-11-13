"""CLI preparation is non-submitting and binds the later explicit action."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e0_resource_cli as cli
from agentmemeval.path1_v1.operations.contracts import SOURCE_CONTRACT_SHA256, content_digest
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_operations_contracts import SOURCE, _git, evidence


def fixture(tmp_path):
    data = evidence.__wrapped__(tmp_path)
    repo = data["repo"]
    path = repo / "configs/path1_v1"
    path.mkdir(parents=True)
    source = repo / "src/agentmemeval/path1_v1/assets/path1_source_contract.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE.read_bytes())
    operations = repo / "src/agentmemeval/path1_v1/operations"
    operations.mkdir()
    (operations / "fixture.py").write_text("# source inventory fixture\n", encoding="utf-8")
    python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    proof = cli.verify_python_interpreter(python, file_sha256(python))
    lock = {"kind": "path1_e0_observed_environment_versions",
            "python_version": platform.python_version(), "distributions": cli.installed_versions()}
    (path / "environment.json").write_text(json.dumps(lock), encoding="utf-8")
    config = {"kind": "path1_e0_resource_site_config", "phase": "E0",
              "job_kind": "resource_probe", "walltime_seconds": 120, "command_seconds": 90,
              "cleanup_seconds": 30, "gpus": 1, "stage_cap": 1,
              "queue_start_window_seconds": 10, "scheduler_deadline_offset_seconds": 130,
              "dependency_lock": "configs/path1_v1/environment.json",
              "python_executable": str(python), "python_sha256": file_sha256(python),
              "python_prefix": proof["prefix"], "cluster": "cluster", "account": "account",
              "user": "owner", "partition": "gpu"}
    (path / "site.json").write_text(json.dumps(config), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "CLI fixture source config and observed environment")
    return repo


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
def test_prepare_never_submits_and_pins_contract_request_script(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    monkeypatch.setattr(cli.DurableIntentExecutor, "submit_once",
                        lambda *a, **k: pytest.fail("prepare invoked scheduler"))
    root = tmp_path / "prepared"
    descriptor = cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    contract = json.loads((root / "contract.json").read_text())
    request = json.loads((root / "probe_request.json").read_text())
    assert content_digest(contract["source_contract"]) == SOURCE_CONTRACT_SHA256
    assert contract["phase"] == "E0" and contract["cohort"]["formation_count"] == 0
    assert request["model_root"] is None
    assert request["launch"]["service_python"] is None
    assert descriptor["probe_request_sha256"] == file_sha256(root / "probe_request.json")
    assert descriptor["spec"]["script_sha256"] == file_sha256(root / "probe.sh")
    assert descriptor["descriptor_sha256"] == file_sha256(root / "submission.json")
    assert descriptor["runtime_admission"] is False
    assert not Path(descriptor["spec"]["work_dir"]).exists()
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("action", [False, True])
def test_submit_and_lookup_reject_unregistered_descriptor_before_any_scheduler(tmp_path, action):
    root = tmp_path / "run"
    root.mkdir()
    (root / "submission.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor changed"):
        cli.execute(root, "a" * 64, lookup=action)


def test_environment_drift_blocks_prepare_without_creating_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    monkeypatch.setattr(cli, "installed_versions", lambda: [])
    root = tmp_path / "not_created"
    with pytest.raises(ValueError, match="versions differ"):
        cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    assert not root.exists()
