"""Build a paper-grade formal runtime lock from one verified real-service run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentmemeval.core.errors import ConfigError
from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT

FORMAL_RUNTIME_LOCK_FIELDS = (
    "gpu_name",
    "gpu_driver",
    "service_torch_cuda_version",
    "vllm_version",
    "decision_model_name",
    "decision_model_revision",
    "decision_weights_hash",
    "decision_served_name",
    "decision_startup_parameters_sha256",
    "embedding_backend",
    "embedding_model_name",
    "embedding_model_revision",
    "embedding_weights_hash",
    "embedding_startup_parameters_sha256",
    "decision_prompt_version",
    "decision_system_sha256",
    "experience_update_sha256",
)
CONFIG_BOUND_RUNTIME_LOCK_FIELDS = FORMAL_RUNTIME_LOCK_FIELDS[4:]


def build_formal_runtime_lock_from_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Bind model, embedding, prompt, hardware, and service identity from a real run."""

    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ConfigError("run manifest metadata is missing")
    protocol = metadata.get("protocol")
    admission = (
        protocol.get("admission_audit")
        if isinstance(protocol, dict)
        else None
    )
    checks = admission.get("checks") if isinstance(admission, dict) else None
    if not isinstance(checks, dict):
        raise ConfigError("run manifest admission checks are missing")
    for field in (
        "decision_model_identity_complete",
        "embedding_identity_complete",
        "decision_service_smoke_passed",
        "embedding_service_smoke_passed",
    ):
        if checks.get(field) is not True:
            raise ConfigError(f"run manifest admission check is not verified: {field}")
    code = metadata.get("code")
    if not isinstance(code, dict) or code.get("dirty") is not False:
        raise ConfigError("runtime lock source must have a clean recorded worktree")
    code_commit = code.get("commit")
    if not isinstance(code_commit, str) or not code_commit.strip():
        raise ConfigError("runtime lock source code commit is missing")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ConfigError("runtime lock source run_id is missing")
    model = _mapping(metadata, "model")
    embedding = _mapping(metadata, "embedding")
    service = _mapping(metadata, "service")
    service_runtime = _mapping(metadata, "model_service_runtime")
    prompts = _mapping(metadata, "prompts")
    gpu = _mapping(metadata, "gpu")
    devices = gpu.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ConfigError("runtime lock source must record exactly one GPU device")
    device = devices[0]
    if not isinstance(device, dict):
        raise ConfigError("runtime lock GPU record is invalid")
    if service_runtime.get("status") != "verified":
        raise ConfigError("runtime lock source model-service probe is not verified")
    decision_startup = service.get("service_startup_parameters")
    embedding_startup = embedding.get("service_startup_parameters")
    if not isinstance(decision_startup, dict) or not decision_startup:
        raise ConfigError(
            "runtime lock source decision service startup parameters are missing"
        )
    if not isinstance(embedding_startup, dict) or not embedding_startup:
        raise ConfigError(
            "runtime lock source embedding service startup parameters are missing"
        )
    lock = {
        "gpu_name": device.get("name"),
        "gpu_driver": device.get("driver"),
        "service_torch_cuda_version": service_runtime.get("torch_cuda_version"),
        "vllm_version": service_runtime.get("vllm_version"),
        "decision_model_name": model.get("name"),
        "decision_model_revision": model.get("revision"),
        "decision_weights_hash": model.get("weights_hash"),
        "decision_served_name": model.get("served_model_name"),
        "decision_startup_parameters_sha256": _json_sha256(decision_startup),
        "embedding_backend": embedding.get("backend"),
        "embedding_model_name": embedding.get("name"),
        "embedding_model_revision": embedding.get("revision"),
        "embedding_weights_hash": embedding.get("weights_hash"),
        "embedding_startup_parameters_sha256": _json_sha256(embedding_startup),
        "decision_prompt_version": prompts.get("decision_version"),
        "decision_system_sha256": prompts.get("decision_system_sha256"),
        "experience_update_sha256": prompts.get("experience_update_sha256"),
    }
    for field in FORMAL_RUNTIME_LOCK_FIELDS:
        value = lock.get(field)
        if value is None or not str(value).strip():
            raise ConfigError(f"runtime lock source field is missing: {field}")
    for field in (
        "decision_weights_hash",
        "decision_startup_parameters_sha256",
        "embedding_weights_hash",
        "embedding_startup_parameters_sha256",
        "decision_system_sha256",
        "experience_update_sha256",
    ):
        if not _is_sha256(lock[field]):
            raise ConfigError(f"runtime lock source hash is invalid: {field}")
    return {
        "schema_version": "task4_formal_runtime_lock_v2",
        "status": "verified_from_real_service_run_manifest",
        "formal_runtime_lock": {
            field: str(lock[field]) for field in FORMAL_RUNTIME_LOCK_FIELDS
        },
        "source_manifest": {
            "path": str(path),
            "sha256": _sha256(path),
            "run_id": run_id,
            "code_commit": code_commit,
        },
    }


def runtime_identity_from_metadata(metadata: dict[str, Any]) -> dict[str, str | None]:
    """Project current runtime metadata onto the immutable formal lock schema."""

    model = metadata.get("model")
    embedding = metadata.get("embedding")
    service = metadata.get("service")
    service_runtime = metadata.get("model_service_runtime")
    prompts = metadata.get("prompts")
    gpu = metadata.get("gpu")
    model = model if isinstance(model, dict) else {}
    embedding = embedding if isinstance(embedding, dict) else {}
    service = service if isinstance(service, dict) else {}
    service_runtime = service_runtime if isinstance(service_runtime, dict) else {}
    prompts = prompts if isinstance(prompts, dict) else {}
    gpu = gpu if isinstance(gpu, dict) else {}
    devices = gpu.get("devices")
    device = (
        devices[0]
        if isinstance(devices, list)
        and len(devices) == 1
        and isinstance(devices[0], dict)
        else {}
    )
    return {
        "gpu_name": _optional_text(device.get("name")),
        "gpu_driver": _optional_text(device.get("driver")),
        "service_torch_cuda_version": _optional_text(
            service_runtime.get("torch_cuda_version")
        ),
        "vllm_version": _optional_text(service_runtime.get("vllm_version")),
        "decision_model_name": _optional_text(model.get("name")),
        "decision_model_revision": _optional_text(model.get("revision")),
        "decision_weights_hash": _optional_text(model.get("weights_hash")),
        "decision_served_name": _optional_text(model.get("served_model_name")),
        "decision_startup_parameters_sha256": _json_sha256(
            service.get("service_startup_parameters")
        ),
        "embedding_backend": _optional_text(embedding.get("backend")),
        "embedding_model_name": _optional_text(embedding.get("name")),
        "embedding_model_revision": _optional_text(embedding.get("revision")),
        "embedding_weights_hash": _optional_text(embedding.get("weights_hash")),
        "embedding_startup_parameters_sha256": _json_sha256(
            embedding.get("service_startup_parameters")
        ),
        "decision_prompt_version": _optional_text(prompts.get("decision_version")),
        "decision_system_sha256": _optional_text(
            prompts.get("decision_system_sha256")
        ),
        "experience_update_sha256": _optional_text(
            prompts.get("experience_update_sha256")
        ),
    }


def configured_runtime_identity(config: dict[str, Any]) -> dict[str, str]:
    """Project a resolved config and current prompts onto config-bound lock fields."""

    provider = config.get("provider")
    agent = config.get("agent")
    provider = provider if isinstance(provider, dict) else {}
    agent = agent if isinstance(agent, dict) else {}
    return {
        "decision_model_name": str(provider.get("model", "")),
        "decision_model_revision": str(provider.get("model_revision", "")),
        "decision_weights_hash": str(provider.get("model_weights_hash", "")),
        "decision_served_name": str(provider.get("served_model_name", "")),
        "decision_startup_parameters_sha256": _json_sha256(
            provider.get("service_startup_parameters")
        ),
        "embedding_backend": str(agent.get("embedding_backend", "")),
        "embedding_model_name": str(agent.get("embedding_model", "")),
        "embedding_model_revision": str(agent.get("embedding_revision", "")),
        "embedding_weights_hash": str(agent.get("embedding_weights_hash", "")),
        "embedding_startup_parameters_sha256": _json_sha256(
            agent.get("embedding_service_startup_parameters")
        ),
        "decision_prompt_version": PROMPT_TEMPLATE_VERSION,
        "decision_system_sha256": hashlib.sha256(
            BASE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "experience_update_sha256": hashlib.sha256(
            EXPERIENCE_UPDATE_PROMPT.encode("utf-8")
        ).hexdigest(),
    }


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ConfigError(f"run manifest metadata.{key} is missing")
    return item


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read runtime-lock manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError("runtime-lock manifest must be a JSON object")
    return data


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)
