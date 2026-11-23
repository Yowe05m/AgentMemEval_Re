from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e0_resource_cli as resource
from agentmemeval.path1_v1.operations import e0_service_cli as service
from agentmemeval.path1_v1.operations.contracts import canonical_json
from agentmemeval.path1_v1.operations.intent_executor import ExecutorError, VerifiedCommand
from agentmemeval.path1_v1.operations.runtime_probe import validate_probe_request
from agentmemeval.path1_v1.operations.service_profile import planned_service_argv
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e0_executor import make_e0_fixture
from path1_v1.unit.test_e0_resource_cli import fixture as resource_fixture
from path1_v1.unit.test_operations_contracts import _git


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resource, "installed_versions", lambda: [{"name": "fixture", "version": "1"}]
    )
    repo = resource_fixture(tmp_path)
    path = repo / "configs/path1_v1/site.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("queue_start_window_seconds", "scheduler_deadline_offset_seconds"):
        del config[key]
    config.update(
        schema_version=1,
        kind="path1_e0_service_site_config",
        job_kind="service_probe",
        resource_profile_version=3,
        language_model_only=True,
        walltime_seconds=7200,
        command_seconds=6600,
        cleanup_seconds=600,
        startup_seconds=1200,
        cpus_per_task=8,
        memory_mib=None,
        memory_policy="site_default",
        queue_policy="no_scheduler_deadline",
        qos_policy="site_default",
        service_port=18231,
        service_python=config["python_executable"],
        service_python_sha256=config["python_sha256"],
        service_python_prefix=config["python_prefix"],
        model_root=str(tmp_path / "model"),
        model_manifest_path=str(tmp_path / "model_manifest.json"),
        model_manifest_sha256=file_sha256(tmp_path / "model_manifest.json"),
    )
    path.write_bytes(canonical_json(config))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "version 3 site-default text-only service fixture")
    return repo, path, config


@pytest.mark.skipif(sys.platform != "linux", reason="Slurm canonical paths require local Linux")
def test_prepare_real_git_model_and_inspect_without_scheduler_or_model(tmp_path, monkeypatch):
    repo, path, config = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        service.DurableIntentExecutor,
        "submit_once",
        lambda *a, **k: pytest.fail("prepare submitted a job"),
    )
    root = tmp_path / "prepared"
    result = service.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    descriptor, frozen, request = service.read_prepared(root, result["descriptor_sha256"])
    assert frozen == config
    assert request.spec.cpus_per_task == 8 and request.spec.memory_mib is None
    assert request.spec.memory_policy == "site_default"
    assert "--cpus-per-task=8" in request.sbatch_argv()
    assert not any(arg.startswith("--mem") for arg in request.sbatch_argv())
    assert result["implementation_status"] == "implemented_only"
    assert result["runtime_admission"] is False and result["scientific_admission"] is False
    assert result["service_profile"]["allocation_verified"] is False
    assert "--no-enable-prefix-caching" in result["service_profile"]["service_argv"]
    assert result["service_profile"]["service_argv"].count("--language-model-only") == 1
    assert result["service_profile"]["profile_version"] == 3
    assert result["service_profile"]["allocated_memory_verified"] is False
    assert result["service_profile"]["language_model_only"] is True
    assert result["service_profile"]["effective_multimodal_state_verified"] is False
    assert (root / "inputs/site_config.json").read_bytes() == path.read_bytes()
    assert (root / "inputs/model_manifest.json").read_bytes() == (
        tmp_path / "model_manifest.json"
    ).read_bytes()
    probe = validate_probe_request(root / "probe_request.json")
    assert probe.contract["cohort"]["formation_count"] == 0
    assert probe.contract["cohort"]["seed_inventory_sha256"] is None
    assert probe.request["launch"]["startup_seconds"] == 1200
    assert probe.request["launch"]["language_model_only"] is True
    assert not probe.output_directory.exists()
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError):
        service.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    path.write_bytes(b"{}")
    monkeypatch.setattr(
        service, "verify_environment", lambda *a: pytest.fail("inspect read live env")
    )
    assert service.read_prepared(root, result["descriptor_sha256"])[0] == descriptor
    with pytest.raises(ValueError, match="current service config"):
        service.execute(root, result["descriptor_sha256"], lookup=False)
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpus_per_task", 1),
        ("memory_mib", 15550),
        ("memory_mib", 65536.0),
        ("startup_seconds", 600),
        ("command_seconds", 7200),
        ("walltime_seconds", 120),
        ("queue_policy", "resource"),
        ("qos_policy", "gpugpu"),
        ("service_port", True),
        ("queue_start_window_seconds", 3600),
        ("scheduler_deadline_offset_seconds", 3720),
    ],
)
def test_service_profile_rejects_resource_defaults_and_silent_variants(
    tmp_path, monkeypatch, field, value
):
    repo, path, config = fixture(tmp_path, monkeypatch)
    config[field] = value
    path.write_bytes(canonical_json(config))
    with pytest.raises(SubmissionDisabled):
        service.prepare(repo, tmp_path / "not_created", "campaign", "configs/path1_v1/site.json")
    assert not (tmp_path / "not_created").exists()


def test_model_manifest_change_rejected_before_creating_output(tmp_path, monkeypatch):
    repo, _, _ = fixture(tmp_path, monkeypatch)
    (tmp_path / "model_manifest.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="model manifest differs"):
        service.prepare(repo, tmp_path / "not_created", "campaign", "configs/path1_v1/site.json")
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("field,value", [("cpus_per_task", 4), ("memory_mib", 32768)])
def test_executor_checks_service_resources_before_submit_and_pre_go(tmp_path, field, value):
    data = make_e0_fixture(tmp_path, "service_probe")
    bad = replace(data.request, spec=replace(data.request.spec, **{field: value}))
    with pytest.raises(SubmissionDisabled, match="resources differ"):
        data.executor._validate_launch(bad)
    probe = validate_probe_request(data.request_path)
    with pytest.raises(SubmissionDisabled, match="resources differ"):
        data.producer(bad, probe)


def test_service_preflight_records_requested_not_allocated_resources(tmp_path):
    data = make_e0_fixture(tmp_path, "service_probe")
    validated = data.executor._validate_launch(data.request)[0]
    admission = data.producer(data.request, validated)
    files = list((tmp_path / "preflight").glob("*.json"))
    assert len(files) == 1 and admission.evidence_sha256 == file_sha256(files[0])
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["service_profile"]["requested_resources"] == {
        "cpus_per_task": 8,
        "memory_mib": 65536,
        "gpus": 1,
        "walltime_seconds": 7200,
    }
    assert record["service_profile"]["allocation_verified"] is False
    assert record["runtime_admission"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Slurm canonical paths require local Linux")
@pytest.mark.parametrize(
    "name", ["site_config.json", "dependency_lock.json", "model_manifest.json"]
)
def test_frozen_original_bytes_cannot_be_replaced(tmp_path, monkeypatch, name):
    repo, _, _ = fixture(tmp_path, monkeypatch)
    root = tmp_path / "prepared"
    result = service.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    (root / "inputs" / name).write_bytes(b"{}")
    with pytest.raises(ValueError, match="input bytes changed"):
        service.read_prepared(root, result["descriptor_sha256"])


def test_resource_v2_does_not_accept_service_resources(tmp_path):
    data = make_e0_fixture(tmp_path)
    bad = replace(data.request, spec=replace(data.request.spec, cpus_per_task=8, memory_mib=65536))
    with pytest.raises(ExecutorError, match="historical resources"):
        data.executor._validate_launch(bad)


@pytest.mark.parametrize("limits", ["MaxCPUsPerNode=7", "MaxMemPerCPU=4096"])
def test_actual_partition_limits_reject_service_request(tmp_path, limits):
    data = make_e0_fixture(tmp_path, "service_probe")
    validated = data.executor._validate_launch(data.request)[0]
    data.partition.write_text(
        "print('PartitionName=gpu State=UP MaxTime=UNLIMITED " + limits + "')\n", encoding="utf-8"
    )
    data.producer.command = VerifiedCommand(
        (str(data.python), "-I", str(data.partition)),
        (
            (str(data.python), file_sha256(data.python)),
            (str(data.partition), file_sha256(data.partition)),
        ),
    )
    with pytest.raises(SubmissionDisabled, match="partition cannot accommodate service"):
        data.producer(data.request, validated)
    assert not list((tmp_path / "preflight").glob("*.json"))


@pytest.mark.parametrize("profile_version", [1, 2])
def test_planned_argv_matches_real_launcher_before_any_process_creation(
    tmp_path, monkeypatch, profile_version
):
    from agentmemeval.path1_v1.operations import runtime_probe
    from agentmemeval.path1_v1.operations.process_ownership import OwnedProcess
    from path1_v1.unit.test_runtime_probe import offline_collector

    request, _, _ = offline_collector(tmp_path, monkeypatch)
    request.contract["job"] = {"kind": "service_probe"}
    request.request["schema_version"] = 1
    executable = tmp_path / "offline_python"
    executable.write_bytes(b"never executed")
    config = json.loads(
        (Path(service.__file__).parents[4] / "configs/path1_v1/e0_service_scze856.json").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        service_python=str(executable), model_root=request.request["model_root"], service_port=8000
    )
    if profile_version == 1:
        config["resource_profile_version"] = 1
        del config["language_model_only"]
    else:
        request.request["schema_version"] = 2
        request.request["launch"] = {"service_python": str(executable), "service_port": 8000,
                                     "language_model_only": True}
    actual = []

    def capture(argv, **kwargs):
        actual.append(argv)
        raise RuntimeError("offline capture before process creation")

    monkeypatch.setattr(OwnedProcess, "managed", capture)
    monkeypatch.setattr(runtime_probe, "validate_probe_request", lambda *a, **k: request)
    with pytest.raises(RuntimeError, match="offline capture"):
        runtime_probe.run_service_probe(
            tmp_path / "unused", request.output_directory, executable, 8000
        )
    assert actual == [planned_service_argv(config, request.contract["source_contract"])]
    launch = json.loads((request.output_directory / "launch.json").read_text(encoding="utf-8"))
    if profile_version == 2:
        assert launch["requested_service_policy"]["language_model_only"] is True
        assert launch["requested_service_policy"]["effective_multimodal_state_verified"] is False
    else:
        assert "requested_service_policy" not in launch
    assert not (tmp_path / "payload").exists()
