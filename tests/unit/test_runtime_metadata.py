from __future__ import annotations

import json
from types import SimpleNamespace

from agentmemeval.storage.artifacts import _service_runtime_metadata


def test_service_runtime_probe_uses_actual_model_environment(monkeypatch) -> None:
    payload = {
        "python": "3.12.13",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "cuda_available": True,
        "cuda_device_count": 1,
        "vllm_version": "0.23.1",
    }

    def fake_run(args, **kwargs):
        assert args[0] == "/envs/vllm/bin/python"
        assert args[1] == "-c"
        assert kwargs["timeout"] == 15
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr("agentmemeval.storage.artifacts.subprocess.run", fake_run)
    metadata = _service_runtime_metadata(
        {"runtime_probe_python": "/envs/vllm/bin/python"}
    )
    assert metadata == {
        "status": "verified",
        "python_executable": "/envs/vllm/bin/python",
        **payload,
    }


def test_service_runtime_probe_is_explicit_when_not_configured() -> None:
    assert _service_runtime_metadata({}) == {"status": "not_configured"}
