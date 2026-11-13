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


def fixture(tmp_path, *, queue_policy=(3600, 3720)):
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
              "queue_start_window_seconds": queue_policy[0],
              "scheduler_deadline_offset_seconds": queue_policy[1],
              "dependency_lock": "configs/path1_v1/environment.json",
              "python_executable": str(python), "python_sha256": file_sha256(python),
              "python_prefix": proof["prefix"], "cluster": "cluster", "account": "account",
              "user": "owner", "partition": "gpu",
              "control_root": str(tmp_path / "control"),
              "accounting_since": "2026-11-13T00:00:00+00:00",
              "commands": {name: {"path": str(tmp_path / name), "sha256": "a" * 64}
                           for name in ("sbatch", "scontrol", "sacctmgr", "squeue", "sacct")}}
    (tmp_path / "control").mkdir()
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
    assert descriptor["schema_version"] == 2 and descriptor["attempt"] == 1
    assert descriptor["queue_start_window_seconds"] == 3600
    assert descriptor["scheduler_deadline_offset_seconds"] == 3720
    assert (root / "site_config.json").read_bytes() == (
        repo / "configs/path1_v1/site.json").read_bytes()
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


@pytest.mark.parametrize("attempt", [0, 4, True, "2", 2.0, None])
def test_attempt_requires_explicit_integer_within_frozen_retry_bound(tmp_path, attempt):
    with pytest.raises(ValueError, match="attempt"):
        cli.prepare(tmp_path, tmp_path / "unused", "campaign", "unused", attempt=attempt)
    assert not (tmp_path / "unused").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
def test_explicit_retry_keeps_seed_namespace_and_creates_distinct_intent(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    first_root, second_root = tmp_path / "attempt01", tmp_path / "attempt02"
    first = cli.prepare(repo, first_root, "campaign", "configs/path1_v1/site.json", attempt=1)
    before = {p.name: p.read_bytes() for p in first_root.iterdir() if p.is_file()}
    retry = {"prior_intent_id": first["intent_id"], "prior_terminal_receipt_sha256": "a" * 64,
             "eligibility_receipt_sha256": "b" * 64,
             "scope": "complete_unit", "seed_changed": False}
    second = cli.prepare(repo, second_root, "campaign", "configs/path1_v1/site.json",
                         attempt=2, retry_evidence=retry)
    contracts = [json.loads((root / "contract.json").read_text())
                 for root in (first_root, second_root)]
    assert second["attempt"] == 2 and second["intent_id"] != first["intent_id"]
    assert contracts[0]["seed_namespace"] == contracts[1]["seed_namespace"]
    assert contracts[1]["job"]["retry_evidence"] == retry
    assert contracts[1]["job"]["output_relpath"].endswith("/attempt_02")
    assert before == {p.name: p.read_bytes() for p in first_root.iterdir() if p.is_file()}
    # Preparation binds these hashes; it does not grant scheduler retry admission.
    assert second["runtime_admission"] is False
    assert not (second_root / "output" / contracts[1]["job"]["output_relpath"]).exists()


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
def test_new_root_alone_does_not_create_a_new_logical_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    first = cli.prepare(repo, tmp_path / "a", "campaign", "configs/path1_v1/site.json", attempt=1)
    second = cli.prepare(repo, tmp_path / "b", "campaign", "configs/path1_v1/site.json", attempt=1)
    assert first["intent_id"] == second["intent_id"]
    assert first["submission_request_sha256"] != second["submission_request_sha256"]
    # The durable executor rejects rebinding this intent to different submission bytes.


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
@pytest.mark.parametrize("retry", [None, {"prior_intent_id": "a" * 64,
    "prior_terminal_receipt_sha256": "b" * 64, "eligibility_receipt_sha256": "c" * 64,
    "scope": "complete_unit", "seed_changed": True}])
def test_retry_evidence_must_satisfy_the_existing_contract(tmp_path, monkeypatch, retry):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    with pytest.raises(ValueError, match="retry|object"):
        cli.prepare(repo, tmp_path / "not_created", "campaign", "configs/path1_v1/site.json",
                    attempt=2, retry_evidence=retry)
    assert not (tmp_path / "not_created").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
def test_legacy_descriptor_survives_checkout_and_environment_change_only_for_lookup(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path, queue_policy=(10, 130))
    root = tmp_path / "old_run"
    current_policy = cli.resource_queue_policy
    # Generate an offline historical fixture, not a production bypass.
    with monkeypatch.context() as historical:
        historical.setattr(cli, "resource_queue_policy",
                           lambda config: current_policy(config, allow_legacy=True))
        cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json", attempt=1)
    path = root / "submission.json"
    legacy = json.loads(path.read_text())
    legacy["schema_version"] = 1
    for key in ("attempt", "queue_start_window_seconds", "scheduler_deadline_offset_seconds"):
        del legacy[key]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    expected = file_sha256(path)
    original = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    site = repo / "configs/path1_v1/site.json"
    config = json.loads(site.read_text())
    config.update(queue_start_window_seconds=3600, scheduler_deadline_offset_seconds=3720)
    site.write_text(json.dumps(config), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "new queue policy, preserve old immutable commit")
    monkeypatch.setattr(cli, "verify_environment", lambda *a: pytest.fail("lookup read live env"))
    calls = []

    class ReadOnlyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def lookup(self, request, **kwargs):
            calls.append(request.intent_id)
            return None

        def submit_once(self, *args, **kwargs):
            pytest.fail("historical lookup submitted")

    monkeypatch.setattr(cli, "DurableIntentExecutor", ReadOnlyExecutor)
    result = cli.execute(root, expected, lookup=True)
    assert result["lookup_only"] is True and calls == [legacy["intent_id"]]
    assert cli.read_prepared(root, expected)[1]["queue_start_window_seconds"] == 10
    assert original == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="read-only"):
        cli.execute(root, expected, lookup=False)


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm paths require Linux")
def test_v2_frozen_config_cannot_be_replaced_after_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "installed_versions", lambda: [{"name": "fixture", "version": "1"}])
    repo = fixture(tmp_path)
    root = tmp_path / "prepared"
    descriptor = cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    assert cli.read_prepared(root, descriptor["descriptor_sha256"])[0]["attempt"] == 1
    (root / "site_config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="historical site config"):
        cli.execute(root, descriptor["descriptor_sha256"], lookup=True)
