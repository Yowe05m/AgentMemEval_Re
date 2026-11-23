"""The text-only startup choice is explicit; backend execution remains unverified."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e0_service_cli as cli
from agentmemeval.path1_v1.operations import runtime_probe as probe
from agentmemeval.path1_v1.operations.contracts import (
    ContractError,
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.service_profile import (
    planned_service_argv,
    validate_service_profile,
    validate_service_submission,
)
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, SubmissionRequest, SubmitSpec
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e0_executor import make_e0_fixture
from path1_v1.unit.test_e0_service_cli import fixture
from path1_v1.unit.test_operations_contracts import _git


def config():
    return json.loads(
        (Path(cli.__file__).parents[4] / "configs/path1_v1/e0_service_scze856.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.parametrize("value", [False, 1, 0, "true", None])
def test_new_config_requires_literal_true(value):
    value_config = config()
    value_config["language_model_only"] = value
    with pytest.raises(SubmissionDisabled, match="requires language_model_only"):
        validate_service_profile(value_config)


def test_legacy_config_can_be_read_but_cannot_prepare_new_job(tmp_path, monkeypatch):
    repo, path, value = fixture(tmp_path, monkeypatch)
    value["resource_profile_version"] = 1
    value["memory_mib"] = 65536
    value.pop("memory_policy")
    value.pop("language_model_only")
    raw = canonical_json(value)
    assert cli._config(raw) == value
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="new service preparation"):
        cli.prepare(repo, tmp_path / "absent", "campaign", "configs/path1_v1/site.json")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("value", [False, True])
def test_legacy_profile_cannot_silently_add_new_field(value):
    value_config = config()
    value_config.update(resource_profile_version=1, language_model_only=value)
    with pytest.raises(SubmissionDisabled, match="legacy service profile"):
        validate_service_profile(value_config)


@pytest.mark.parametrize("value", [False, True, 1, None])
def test_resource_launch_rejects_text_only_field(tmp_path, value):
    data = make_e0_fixture(tmp_path)
    launch = copy.deepcopy(data.request_data["launch"])
    launch["language_model_only"] = value
    with pytest.raises(ContractError, match="unknown fields=.*language_model_only"):
        probe.validate_probe_launch(launch, "resource_probe")


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_service_launch_requires_actual_boolean(tmp_path, value):
    data = make_e0_fixture(tmp_path, "service_probe")
    launch = copy.deepcopy(data.request_data["launch"])
    launch["language_model_only"] = value
    with pytest.raises(probe.RuntimeProbeError, match="explicit boolean"):
        probe.validate_probe_launch(launch, "service_probe")


def argument_case(tmp_path, *, language_model_only):
    value_config = config()
    model = tmp_path / "model"
    model.mkdir()
    value_config["model_root"] = str(model)
    if not language_model_only:
        value_config["resource_profile_version"] = 1
        value_config.pop("language_model_only")
    source = {"provider": {"served_model_name": "fixture"}}
    launch = {"language_model_only": True} if language_model_only else {}
    request = SimpleNamespace(
        request={"schema_version": 2, "model_root": str(model), "launch": launch},
        contract={"source_contract": source},
    )
    return request, planned_service_argv(value_config, source)


@pytest.mark.parametrize("defect", ["missing", "false", "negated", "duplicate"])
def test_kernel_argv_must_match_new_bound_text_policy(tmp_path, defect):
    request, argv = argument_case(tmp_path, language_model_only=True)
    if defect == "missing":
        argv.remove("--language-model-only")
    elif defect == "false":
        argv[argv.index("--language-model-only")] = "--language-model-only=false"
    elif defect == "negated":
        argv.append("--no-language-model-only")
    else:
        argv.append("--language-model-only")
    with pytest.raises(probe.RuntimeProbeError):
        probe.check_service_arguments(argv, request)


def test_requested_policy_record_does_not_claim_effective_tower_or_loader(tmp_path):
    request, argv = argument_case(tmp_path, language_model_only=True)
    result = probe.check_service_arguments(argv, request)
    assert result["language_model_only"] is True and result["requested_modality_limit"] == 0
    assert result["effective_multimodal_state_verified"] is False
    assert result["effective_backend_settings_verified"] is False


def test_legacy_kernel_argv_remains_without_new_flag(tmp_path):
    request, argv = argument_case(tmp_path, language_model_only=False)
    result = probe.check_service_arguments(argv, request)
    assert "language_model_only" not in result and "--language-model-only" not in argv
    with pytest.raises(probe.RuntimeProbeError, match="startup policy"):
        probe.check_service_arguments([*argv, "--language-model-only"], request)


@pytest.mark.parametrize("value", [False, True, None])
def test_legacy_service_submission_cannot_claim_a_new_launch_policy(tmp_path, value):
    data = make_e0_fixture(tmp_path, "service_probe")
    validated = probe.validate_probe_request(data.request_path)
    old_config = json.loads(
        (data.repo / validated.contract["runtime_identity"]["config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    validated.request["launch"]["language_model_only"] = value
    with pytest.raises(SubmissionDisabled, match="text-only profile"):
        validate_service_submission(data.request, validated, old_config)


@pytest.mark.skipif(sys.platform != "linux", reason="historical prepared Slurm paths require Linux")
@pytest.mark.parametrize("historical_version", [1, 2])
def test_legacy_descriptor_replay_remains_exact_after_new_profile_is_available(
    tmp_path, monkeypatch, historical_version
):
    """Anonymous historical fixture; not a copy of any actual run artifact."""
    repo, path, value = fixture(tmp_path, monkeypatch)
    root = tmp_path / "prepared"
    cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    value["resource_profile_version"] = historical_version
    value["memory_mib"] = 65536
    value.pop("memory_policy")
    if historical_version == 1:
        value.pop("language_model_only")
    raw = canonical_json(value)
    path.write_bytes(raw)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "anonymous legacy profile fixture")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    identity = contract["runtime_identity"]
    contract["runtime_identity"] = build_runtime_identity(
        repo,
        core_paths=[row["path"] for row in identity["core_files"]],
        operations_paths=[row["path"] for row in identity["operations_files"]],
        config_path="configs/path1_v1/site.json",
        dependency_lock_path=value["dependency_lock"],
        model_manifest_path=Path(value["model_manifest_path"]),
        model_root=Path(value["model_root"]),
    )
    (root / "contract.json").write_bytes(canonical_json(contract))
    request_data = json.loads((root / "probe_request.json").read_text(encoding="utf-8"))
    if historical_version == 1:
        request_data["launch"].pop("language_model_only")
    request_data.update(
        intent_id=canonical_intent_id(contract), contract_sha256=file_sha256(root / "contract.json")
    )
    request_data["request_sha256"] = content_digest(
        {k: v for k, v in request_data.items() if k != "request_sha256"}
    )
    (root / "probe_request.json").write_bytes(canonical_json(request_data))
    (root / "inputs/site_config.json").write_bytes(raw)
    descriptor = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    descriptor["spec"].pop("memory_policy")
    descriptor["spec"]["memory_mib"] = 65536
    request = SubmissionRequest(
        descriptor["campaign_id"],
        request_data["intent_id"],
        SubmitSpec.from_payload(descriptor["spec"]),
    )
    descriptor.update(
        intent_id=request.intent_id,
        submission_request_sha256=request.request_sha256,
        config_sha256=file_sha256(path),
        probe_request_sha256=file_sha256(root / "probe_request.json"),
        service_profile=validate_service_submission(
            request, SimpleNamespace(request=request_data, contract=contract), value
        ),
    )
    descriptor["inputs"][0].update(size=len(raw), sha256=file_sha256(path))
    (root / "submission.json").write_bytes(canonical_json(descriptor))
    digest = file_sha256(root / "submission.json")
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    observed, frozen, historical_request = cli.read_prepared(root, digest)
    assert observed == descriptor and frozen == value and historical_request == request
    assert observed["service_profile"]["profile_version"] == historical_version
    assert ("language_model_only" in observed["service_profile"]) == (historical_version == 2)
    assert ("--language-model-only" in observed["service_profile"]["service_argv"]) == (
        historical_version == 2
    )
    assert "memory_policy" not in historical_request.spec.payload()
    assert "--mem=65536" in historical_request.sbatch_argv()
    assert SubmitSpec.from_payload(historical_request.spec.payload()) == historical_request.spec
    with pytest.raises(ValueError, match="lookup only"):
        cli.execute(root, digest, lookup=False)
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
