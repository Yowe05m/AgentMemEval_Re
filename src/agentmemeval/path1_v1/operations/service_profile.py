"""Versioned service E0 launch resources; no backend or scientific admission."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled


def validate_service_profile(config: dict) -> None:
    expected = {
        "schema_version": 1,
        "kind": "path1_e0_service_site_config",
        "phase": "E0",
        "job_kind": "service_probe",
        "resource_profile_version": 1,
        "walltime_seconds": 7200,
        "command_seconds": 6600,
        "cleanup_seconds": 600,
        "startup_seconds": 1200,
        "gpus": 1,
        "stage_cap": 1,
        "cpus_per_task": 8,
        "memory_mib": 65536,
        "queue_policy": "no_scheduler_deadline",
        "qos_policy": "site_default",
    }
    if type(config) is not dict or any(
        type(config.get(key)) is not type(value) or config.get(key) != value
        for key, value in expected.items()
    ):
        raise SubmissionDisabled("service E0 profile differs from version 1")
    if any(
        key in config
        for key in (
            "queue_start_window_seconds",
            "scheduler_deadline_offset_seconds",
            "qos",
        )
    ):
        raise SubmissionDisabled("service E0 cannot borrow resource deadlines or hidden QOS")
    port = config.get("service_port")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise SubmissionDisabled("service E0 requires a fixed dedicated port")


def planned_service_argv(config: dict, source: dict) -> list[str]:
    """Reviewable expected argv of the already versioned runtime_probe launcher."""
    validate_service_profile(config)
    model = config["model_root"]
    return [
        config["service_python"],
        "-B",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--tokenizer",
        model,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "1",
        "--no-enable-prefix-caching",
        "--served-model-name",
        source["provider"]["served_model_name"],
        "--host",
        "127.0.0.1",
        "--port",
        str(config["service_port"]),
    ]


def validate_service_submission(request, probe, config: dict) -> dict:
    """Compare submitted resources and bound launch with the frozen site config."""
    validate_service_profile(config)
    if (
        probe.contract["phase"] != "E0"
        or probe.contract["job"]["kind"] != "service_probe"
        or request.spec.cpus_per_task != config["cpus_per_task"]
        or request.spec.memory_mib != config["memory_mib"]
        or request.spec.walltime_seconds != config["walltime_seconds"]
        or request.spec.gpus != config["gpus"]
        or request.spec.script_args
    ):
        raise SubmissionDisabled("service submitted resources differ from frozen profile")
    launch = probe.request["launch"]
    for key in (
        "python_executable",
        "python_sha256",
        "service_python",
        "service_python_sha256",
        "service_port",
        "command_seconds",
        "cleanup_seconds",
        "startup_seconds",
    ):
        if type(launch[key]) is not type(config[key]) or launch[key] != config[key]:
            raise SubmissionDisabled("service launch differs from site config: " + key)
    if (
        launch["python_identity"]["prefix"] != config["python_prefix"]
        or launch["service_python_identity"]["prefix"] != config["service_python_prefix"]
        or probe.request["model_root"] != config["model_root"]
        or probe.request["model_manifest_path"] != config["model_manifest_path"]
        or probe.contract["runtime_identity"]["model"]["manifest_sha256"]
        != config["model_manifest_sha256"]
    ):
        raise SubmissionDisabled("service model/interpreter differs from site config")
    if (request.spec.cluster, request.spec.account, request.spec.user, request.spec.partition) != (
        config["cluster"],
        config["account"],
        config["user"],
        config["partition"],
    ):
        raise SubmissionDisabled("service scheduler scope differs from site config")
    resources = {
        key: getattr(request.spec, key)
        for key in ("cpus_per_task", "memory_mib", "gpus", "walltime_seconds")
    }
    return {
        "profile_version": 1,
        "requested_resources": resources,
        "queue_policy": config["queue_policy"],
        "qos_policy": config["qos_policy"],
        "service_argv": planned_service_argv(config, probe.contract["source_contract"]),
        "config_semantic_sha256": content_digest(config),
        "allocation_verified": False,
        "runtime_admission": False,
    }


def read_service_config(probe) -> dict:
    from agentmemeval.path1_v1.operations.runtime_probe import _strict_json

    identity = probe.contract["runtime_identity"]["config"]
    path = Path(probe.request["repository_root"]) / identity["path"]
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise SubmissionDisabled("bound service config changed")
    value = _strict_json(raw)
    validate_service_profile(value)
    return value
