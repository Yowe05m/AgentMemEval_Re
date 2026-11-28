"""Exact toolchain binding and E0/E1 continuity; no real scheduler or GPU."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import cuda_toolkit as toolkit
from agentmemeval.path1_v1.operations.contracts import canonical_json


def fixture(tmp_path):
    (tmp_path / "bin").mkdir()
    nvcc = tmp_path / "bin/nvcc"
    nvcc.write_bytes(b"#!/bin/sh\nexit 0\n")
    nvcc.chmod(0o755)
    return {"cuda_home": str(tmp_path),
            "nvcc_sha256": hashlib.sha256(nvcc.read_bytes()).hexdigest()}


def test_exact_compiler_overrides_hidden_launchers_without_library_path_change(tmp_path):
    pin = fixture(tmp_path)
    base = {"PATH": "/usr/bin", "CUDA_HOME": "/wrong", "CUDA_PATH": "/wrong",
            "FLASHINFER_NVCC": "/wrong", "FLASHINFER_NVCC_LAUNCHER": "evil",
            "FLASHINFER_CXX_LAUNCHER": "evil", "LD_PRELOAD": "keep"}
    env = toolkit.environment(pin, base)
    assert base["FLASHINFER_NVCC"] == "/wrong"
    assert env["CUDA_HOME"] == env["CUDA_PATH"] == str(tmp_path)
    assert env["FLASHINFER_NVCC"] == str(tmp_path / "bin/nvcc")
    assert env["PATH"] == str(tmp_path / "bin") + os.pathsep + "/usr/bin"
    assert env["LD_PRELOAD"] == "keep" and "LD_LIBRARY_PATH" not in env
    assert "FLASHINFER_NVCC_LAUNCHER" not in env and "FLASHINFER_CXX_LAUNCHER" not in env
    (tmp_path / "bin/nvcc").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        toolkit.environment(pin, base)


@pytest.mark.parametrize("change", [lambda p: p.update(extra="bad"),
    lambda p: p.update(cuda_home="relative"), lambda p: p.update(nvcc_sha256="bad")])
def test_invalid_binding_rejected(tmp_path, change):
    pin = fixture(tmp_path)
    change(pin)
    with pytest.raises(ValueError):
        toolkit.validate_binding(pin)


@pytest.mark.parametrize("version", [1, 2])
def test_shared_profile_rejects_toolkit_on_legacy_policy(tmp_path, version):
    from agentmemeval.path1_v1.operations.service_profile import validate_service_profile
    from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
    repo = Path(toolkit.__file__).parents[4]
    config = json.loads((repo / "configs/path1_v1/e0_service_scze856_v3.json").read_bytes())
    config.update(resource_profile_version=version, cuda_toolkit=fixture(tmp_path))
    with pytest.raises(SubmissionDisabled, match="CUDA toolkit requires service profile 3"):
        validate_service_profile(config)


@pytest.mark.parametrize("mode", ["matching", "missing", "different"])
def test_e1_toolkit_plan_must_match_typed_prior_e0(tmp_path, monkeypatch, mode):
    from agentmemeval.path1_v1.operations import e1_campaign as campaign
    pin = fixture(tmp_path)
    nvcc = tmp_path / "bin/nvcc"
    proof = {"entry_path": str(tmp_path / "python"), "executable_sha256": "b" * 64,
             "prefix": str(tmp_path)}
    pins = [{"module_name": name, "path": str(nvcc), "sha256": pin["nvcc_sha256"]}
            for name in sorted(campaign.BACKEND_MODULES)]
    value = {"schema_version": 1, "kind": "path1_e1_runtime_plan", "service_port": 18231,
             "service_python_identity": proof, "source_pins": pins,
             "startup_seconds": 1200, "cleanup_seconds": 600,
             "backend_probe_seeds": [2**62 + 9001, 2**61 + 9001],
             "prior_e0": {}, "language_model_only": True, "cuda_toolkit": pin}
    monkeypatch.setattr(campaign, "verify_python_interpreter", lambda *a, **k: proof)
    monkeypatch.setattr(campaign, "_e0_observations", lambda *a: {"cuda_toolkit": pin})
    if mode == "missing":
        value.pop("cuda_toolkit")
    elif mode == "different":
        value["cuda_toolkit"] = pin | {"nvcc_sha256": "c" * 64}
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json(value))
    if mode == "matching":
        assert campaign.read_runtime_plan(path, contract={"phase": "E1"}) == value
    else:
        with pytest.raises(ValueError, match="CUDA toolkit must match"):
            campaign.read_runtime_plan(path, contract={"phase": "E1"})


@pytest.mark.skipif(sys.platform != "linux", reason="real Slurm canonical paths require Linux")
def test_e0_preparation_binds_toolkit_config_request_and_profile(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations import e0_service_cli as cli
    from path1_v1.unit.test_e0_service_cli import fixture as e0_fixture
    from path1_v1.unit.test_operations_contracts import _git
    repo, path, config = e0_fixture(tmp_path, monkeypatch)
    home = tmp_path / "toolkit"
    home.mkdir()
    pin = fixture(home)
    config["cuda_toolkit"] = pin
    path.write_bytes(canonical_json(config))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pin CUDA toolkit fixture")
    root = tmp_path / "prepared"
    result = cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    descriptor, frozen, _ = cli.read_prepared(root, result["descriptor_sha256"])
    assert frozen["cuda_toolkit"] == pin and descriptor["service_profile"]["cuda_toolkit"] == pin
    probe = json.loads((root / "probe_request.json").read_bytes())
    assert probe["launch"]["cuda_toolkit"] == pin
