"""Site-default RAM is a requested policy, not observed allocation capacity."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e0_service_cli as cli
from agentmemeval.path1_v1.operations.contracts import (
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.intent_executor import ExecutorError, VerifiedCommand
from agentmemeval.path1_v1.operations.service_profile import validate_service_profile
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e0_executor import make_e0_fixture, seal_request
from path1_v1.unit.test_e0_service_cli import fixture
from path1_v1.unit.test_intent_executor import make_executor_fixture
from path1_v1.unit.test_operations_contracts import _git
from path1_v1.unit.test_service_resources import legacy_spec


def new_config():
    path = Path(cli.__file__).parents[4] / "configs/path1_v1/e0_service_scze856_v3.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_site_default_request_serializes_and_renders_exactly():
    spec = replace(legacy_spec(), cpus_per_task=8, memory_policy="site_default",
                   walltime_seconds=7200)
    request = SubmissionRequest("campaign", "b" * 64, spec)
    payload = spec.payload()
    assert payload["memory_policy"] == "site_default" and payload["memory_mib"] is None
    assert payload["cpus_per_task"] == 8
    restored = SubmitSpec.from_payload(json.loads(json.dumps(payload)))
    assert restored == spec and restored.payload() == payload
    replay = SubmissionRequest("campaign", "b" * 64, restored)
    assert replay.request_sha256 == request.request_sha256
    assert replay.sbatch_argv() == request.sbatch_argv()
    assert "--cpus-per-task=8" in request.sbatch_argv()
    assert "--gpus=1" in request.sbatch_argv() and "--time=02:00:00" in request.sbatch_argv()
    assert not any(x.startswith(("--mem", "--deadline", "--qos")) for x in request.sbatch_argv())
    explicit = replace(spec, memory_policy=None, memory_mib=65536)
    assert (
        SubmissionRequest("campaign", "b" * 64, explicit).request_sha256 != request.request_sha256
    )


def test_legacy_explicit_resources_keep_frozen_digest():
    spec = replace(legacy_spec(), cpus_per_task=8, memory_mib=65536, walltime_seconds=7200)
    # Independently executed immutable 765d1509 slurm.py; no new code self-golden.
    assert SubmissionRequest("campaign", "b" * 64, spec).request_sha256 == (
        "1575491e651d322be3aa5bc5bfc8d744aad2903d7ade57621f3a3dfbda1fa453"
    )
    assert "memory_policy" not in spec.payload()
    assert SubmitSpec.from_payload(spec.payload()).payload() == spec.payload()


@pytest.mark.skipif(sys.platform != "linux", reason="prepared Slurm paths require Linux")
def test_rehashed_prepared_descriptor_still_rejects_memory_policy_drift(tmp_path, monkeypatch):
    repo, _, _ = fixture(tmp_path, monkeypatch)
    root = tmp_path / "prepared"
    result = cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    original, _, request = cli.read_prepared(root, result["descriptor_sha256"])
    assert request.spec.memory_policy == "site_default"
    for mutation in ("missing", "false", "explicit_ram"):
        descriptor = copy.deepcopy(original)
        spec = descriptor["spec"]
        if mutation == "missing":
            spec.pop("memory_policy")
        elif mutation == "false":
            spec["memory_policy"] = "auto"
        else:
            spec.pop("memory_policy")
            spec["memory_mib"] = 65536
        descriptor["submission_request_sha256"] = content_digest({
            "campaign_id": descriptor["campaign_id"], "intent_id": descriptor["intent_id"],
            "spec": spec,
        })
        (root / "submission.json").write_bytes(canonical_json(descriptor))
        with pytest.raises((ValueError, SubmissionDisabled)):
            cli.read_prepared(root, file_sha256(root / "submission.json"))
    assert not (root / "output" / "campaign").exists()


@pytest.mark.parametrize("changes", [
    {"memory_policy": None}, {"memory_policy": "default"}, {"memory_policy": True},
    {"memory_mib": 65536}, {"cpus_per_task": None}, {"cpus_per_task": True},
    {"cpus_per_task": 0}, {"cpus_per_task": 257},
])
def test_site_default_cannot_silently_supply_missing_or_false_policy(changes):
    values = dict(cpus_per_task=8, memory_mib=None, memory_policy="site_default")
    values.update(changes)
    with pytest.raises(ValueError):
        replace(legacy_spec(), **values)
    with pytest.raises(ValueError):
        SubmitSpec.from_payload(legacy_spec().payload() | values)


@pytest.mark.parametrize("missing", ["memory_policy", "memory_mib", "cpus_per_task"])
def test_site_default_payload_requires_all_three_explicit_fields(missing):
    value = legacy_spec().payload() | {
        "cpus_per_task": 8, "memory_mib": None, "memory_policy": "site_default"
    }
    del value[missing]
    with pytest.raises(ValueError):
        SubmitSpec.from_payload(value)


@pytest.mark.parametrize("changes", [
    {"memory_mib": 51200}, {"memory_mib": 65536}, {"memory_policy": None},
    {"memory_policy": "auto"}, {"memory_policy": True},
    {"resource_profile_version": 2}, {"resource_profile_version": 1},
    {"language_model_only": False}, {"phase": "E1"}, {"job_kind": "resource_probe"},
    {"cpus_per_task": 1},
])
def test_version_three_profile_has_one_exact_memory_policy(changes):
    config = new_config()
    validate_service_profile(config)
    config.update(changes)
    with pytest.raises(SubmissionDisabled):
        validate_service_profile(config)


@pytest.mark.parametrize("version", [1, 2])
def test_old_config_is_read_only_and_cannot_borrow_memory_policy(tmp_path, monkeypatch, version):
    repo, path, value = fixture(tmp_path, monkeypatch)
    value.update(resource_profile_version=version, memory_mib=65536)
    value.pop("memory_policy")
    if version == 1:
        value.pop("language_model_only")
    raw = canonical_json(value)
    assert cli._config(raw) == value
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="new service preparation"):
        cli.prepare(repo, tmp_path / "absent", "campaign", "configs/path1_v1/site.json")
    assert not (tmp_path / "absent").exists()
    value["memory_policy"] = "site_default"
    with pytest.raises(SubmissionDisabled, match="legacy"):
        validate_service_profile(value)


def service_v3_fixture(tmp_path):
    data = make_e0_fixture(tmp_path, "service_probe")
    identity = data.contract["runtime_identity"]
    path = data.repo / identity["config"]["path"]
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(resource_profile_version=3, memory_mib=None, memory_policy="site_default",
                  language_model_only=True)
    path.write_bytes(canonical_json(config))
    _git(data.repo, "add", ".")
    _git(data.repo, "commit", "-qm", "bind site-default v3 fixture")
    data.contract["runtime_identity"] = build_runtime_identity(
        data.repo, core_paths=[r["path"] for r in identity["core_files"]],
        operations_paths=[r["path"] for r in identity["operations_files"]],
        config_path=identity["config"]["path"],
        dependency_lock_path=identity["dependency_lock"]["path"],
        model_manifest_path=Path(data.request_data["model_manifest_path"]),
        model_root=Path(data.request_data["model_root"]),
    )
    contract_path = Path(data.request_data["contract_path"])
    contract_path.write_bytes(canonical_json(data.contract))
    data.request_data.update(contract_sha256=file_sha256(contract_path),
                             intent_id=canonical_intent_id(data.contract))
    data.request_data["launch"]["language_model_only"] = True
    seal_request(data)
    data.request = replace(
        data.request, intent_id=data.request_data["intent_id"],
        spec=replace(data.request.spec, memory_mib=None, memory_policy="site_default"),
    )
    return data


def partition_limits(data, text):
    data.partition.write_text(
        "print('PartitionName=gpu State=UP MaxTime=UNLIMITED " + text + "')\n",
        encoding="utf-8",
    )
    data.producer.command = VerifiedCommand(
        (str(data.python), "-I", str(data.partition)),
        ((str(data.python), file_sha256(data.python)),
         (str(data.partition), file_sha256(data.partition))),
    )


def test_v3_preflight_preserves_cpu_limit_without_claiming_allocated_ram(tmp_path):
    data = service_v3_fixture(tmp_path)
    partition_limits(data, "MaxCPUsPerNode=8 MaxMemPerCPU=1")
    probe = data.executor._validate_launch(data.request)[0]
    admission = data.producer(data.request, probe)
    records = list((tmp_path / "preflight").glob("*.json"))
    assert len(records) == 1 and admission.evidence_sha256 == file_sha256(records[0])
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["service_profile"]["requested_resources"] == {
        "cpus_per_task": 8, "memory_mib": None, "memory_policy": "site_default",
        "gpus": 1, "walltime_seconds": 7200,
    }
    assert record["service_profile"]["allocated_memory_verified"] is False
    assert record["service_profile"]["allocation_verified"] is False
    assert record["runtime_admission"] is False
    partition_limits(data, "MaxCPUsPerNode=7 MaxMemPerCPU=UNLIMITED")
    with pytest.raises(SubmissionDisabled, match="MaxCPUsPerNode"):
        data.producer(data.request, probe)
    assert len(list((tmp_path / "preflight").glob("*.json"))) == 1


def test_v3_wrong_explicit_ram_rejected_on_launch_and_pre_go(tmp_path):
    data = service_v3_fixture(tmp_path)
    probe = data.executor._validate_launch(data.request)[0]
    wrong = replace(data.request,
                    spec=replace(data.request.spec, memory_policy=None, memory_mib=65536))
    with pytest.raises(SubmissionDisabled, match="resources differ"):
        data.executor._validate_launch(wrong)
    with pytest.raises(SubmissionDisabled, match="resources differ"):
        data.producer(wrong, probe)
    assert not list((tmp_path / "preflight").glob("*.json"))


@pytest.mark.parametrize("research", [False, True])
def test_site_default_cannot_enter_resource_or_research_launcher(tmp_path, research):
    data = make_executor_fixture(tmp_path) if research else make_e0_fixture(tmp_path)
    request = replace(data.request, spec=replace(
        data.request.spec, cpus_per_task=8, memory_mib=None, memory_policy="site_default"
    ))
    with pytest.raises(ExecutorError, match="service E0 only|historical resources"):
        data.executor._validate_launch(request)
