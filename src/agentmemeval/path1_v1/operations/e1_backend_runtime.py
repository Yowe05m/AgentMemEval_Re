"""E1-only live backend observations, read beside the owning service process.

The plugin records actual runner objects; the parent independently checks kernel
membership and the challenge event chain. Serialized reports grant no capability.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from agentmemeval.path1_v1.operations.backend_challenges import canonical, sha
from agentmemeval.path1_v1.operations.backend_observer import ObserverError
from agentmemeval.path1_v1.operations.process_ownership import _kernel


def engine_observation(runner, trace) -> None:
    """Called only inside an actually selected challenge's sampler invocation."""
    import torch
    import transformers
    import vllm

    model = runner.model_config
    device = runner.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ObserverError("E1 requires an actual CUDA runner")
    parameter = next(runner.model.parameters())
    properties = torch.cuda.get_device_properties(device)
    identity, _, state, _ = _kernel(os.getpid())
    if state in {"Z", "X"} or parameter.device.type != "cuda":
        raise ObserverError("model parameters are not live on CUDA")
    trace.write(
        "engine_runtime",
        {
            "identity": asdict(identity),
            "model_path": str(model.model),
            "tokenizer_path": str(model.tokenizer),
            "model_class": type(runner.model).__module__ + "." + type(runner.model).__qualname__,
            "dtype": str(model.dtype),
            "parameter_dtype": str(parameter.dtype),
            "parameter_device": str(parameter.device),
            "device": str(device),
            "max_model_len": model.max_model_len,
            "max_num_seqs": runner.scheduler_config.max_num_seqs,
            "prefix_caching": runner.cache_config.enable_prefix_caching,
            "tensor_parallel_size": runner.parallel_config.tensor_parallel_size,
            "quantization": model.quantization,
            # The official cu129 wheel has distribution 0.23.0+cu129 while
            # vllm.__version__ is 0.23.0. Preserve both exact observations;
            # dependency-lock comparison uses the distribution metadata.
            "vllm_version": metadata.version("vllm"),
            "torch_version": metadata.version("torch"),
            "transformers_version": metadata.version("transformers"),
            "module_versions": {
                "vllm": vllm.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "compiled_cuda": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_uuid": str(getattr(properties, "uuid", "")),
            "gpu_total_memory": properties.total_memory,
            "scope": "observed_loaded_runner_and_parameter_not_weight_tensor_hash",
        },
    )


def read_traces(directory: Path, *, service_instance: str, source_sha256: str) -> list[dict]:
    result = []
    for path in sorted(directory.glob("sampler-*.jsonl")):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ObserverError("invalid backend trace file")
        raw = path.read_bytes()
        # A writer may be appending a later ordinary diagnostic. Never accept a
        # partial record; a completed challenge must be durably newline-delimited.
        if not raw.endswith(b"\n"):
            raise ObserverError("partial backend trace")
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows:
            raise ObserverError("empty backend trace")
        for index, row in enumerate(rows):
            if (
                row.get("kind") != "backend_observation"
                or type(row.get("sequence")) is not int
                or row["sequence"] != index
                or row.get("service_instance") != service_instance
                or row.get("source_sha256") != source_sha256
                or row.get("pid") != rows[0]["pid"]
                or row.get("process_nonce") != rows[0]["process_nonce"]
            ):
                raise ObserverError("backend event identity/sequence mismatch")
        result.extend(rows)
    return result


def verify_process_member(owner, pid: int) -> dict:
    anchor = owner.verify_live().identity
    if type(pid) is not int or pid <= 0:
        raise ObserverError("invalid observed process")
    identity, _, state, _ = _kernel(pid)
    fields = ("host", "boot_id", "effective_uid", "session_id", "pgid", "cgroup", "slurm_job_id")
    if state in {"Z", "X"} or any(getattr(identity, key) != getattr(anchor, key) for key in fields):
        raise ObserverError("backend process is outside the owned allocation/service group")
    return asdict(identity)


def observe_engine_gpus(engine_pids: set[int], visible_gpus: list[dict]) -> dict:
    argv = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(argv, capture_output=True, timeout=30, check=False)
    raw = result.stdout.decode("utf-8", errors="strict")
    if result.returncode != 0 or result.stderr.strip():
        raise ObserverError("GPU process inventory failed")
    pairs = set()
    for line in raw.splitlines():
        columns = [item.strip() for item in line.split(",")]
        if len(columns) != 2 or not columns[0].startswith("GPU-") or not columns[1].isdigit():
            raise ObserverError("malformed GPU process inventory")
        pairs.add((columns[0], int(columns[1])))
    known = {row["uuid"] for row in visible_gpus}
    selected = {(uuid, pid) for uuid, pid in pairs if pid in engine_pids}
    if (
        {pid for _, pid in selected} != engine_pids
        or len({uuid for uuid, _ in selected}) != 1
        or not {uuid for uuid, _ in selected} <= known
    ):
        raise ObserverError("owned engine PIDs do not map to one observed CUDA GPU")
    return {
        "argv": argv,
        "stdout": raw,
        "returncode": result.returncode,
        "engine_gpu_pairs": [{"gpu_uuid": uuid, "pid": pid} for uuid, pid in sorted(selected)],
        "scope": "owned_allocation_processes_on_one_GPU_not_scheduler_exclusivity",
    }


def verify_challenge_events(
    events, challenges, *, owner, model_root: Path, source_pins, expected_versions: dict
) -> dict:
    """Require API tokenization -> seeded q consumption -> completed sample.

    This verifies the two technical challenges, not every later research request.
    E2/E3 cannot obtain admission through this E1 verifier.
    """
    expected_pins = {item["module_name"]: item["sha256"] for item in source_pins}
    if set(expected_versions) != {"torch", "vllm", "transformers"} or any(
        type(value) is not str or not value for value in expected_versions.values()
    ):
        raise ObserverError("exact observed backend dependency versions required")
    matched, identities = [], {}
    for challenge in challenges:
        api = [
            row
            for row in events
            if row["event"] == "api_challenge_registered"
            and row["payload"].get("external_request_id") == challenge.external_request_id
        ]
        if len(api) != 1:
            raise ObserverError("missing/duplicate exact API challenge registration")
        registration = api[0]["payload"]
        api_installs = [
            row
            for row in events
            if row["event"] == "api_adapter_install"
            and row["pid"] == api[0]["pid"]
            and row["sequence"] < api[0]["sequence"]
        ]
        if (
            len(api_installs) != 1
            or api_installs[0]["payload"].get("source_sha256")
            != expected_pins["vllm.v1.engine.async_llm"]
        ):
            raise ObserverError("actual API source adapter evidence missing")
        actual = registration["actual"]
        for key in (
            "seed",
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "prompt_token_ids_sha256",
        ):
            if canonical(actual.get(key)) != canonical(getattr(challenge, key)):
                raise ObserverError("API challenge inputs differ")
        if (
            actual.get("n") != 1
            or registration.get("prompt_text_sha256") != challenge.prompt_text_sha256
        ):
            raise ObserverError("API challenge prompt/sampling differs")
        internal = registration["internal_request_id"]
        dispatched = [
            row
            for row in events
            if row["event"] == "api_dispatch_completed"
            and row["pid"] == api[0]["pid"]
            and row["payload"].get("internal_request_id") == internal
            and row["payload"].get("external_request_id") == challenge.external_request_id
        ]
        if len(dispatched) != 1:
            raise ObserverError("challenge API dispatch did not complete")
        begin = [
            row
            for row in events
            if row["event"] == "sample_begin"
            and any(x.get("internal_request_id") == internal for x in row["payload"]["rows"])
        ]
        if not begin:
            raise ObserverError("challenge has no executed sampler observation")
        first = begin[0]
        stream = [
            row
            for row in events
            if row["pid"] == first["pid"] and row["process_nonce"] == first["process_nonce"]
        ]
        installs = [row for row in stream if row["event"] == "adapter_install"]
        if len(installs) != 1 or installs[0]["payload"]["pins"] != {
            key: value for key, value in expected_pins.items() if key != "vllm.v1.engine.async_llm"
        }:
            raise ObserverError("actual sampler source closure differs")
        engine = [
            row
            for row in stream
            if row["event"] == "engine_runtime" and row["sequence"] < first["sequence"]
        ]
        if not engine:
            raise ObserverError("actual loaded runner evidence missing")
        observed = engine[-1]["payload"]
        if (
            observed.get("model_path") != str(model_root)
            or observed.get("tokenizer_path") != str(model_root)
            or observed.get("dtype") != "torch.bfloat16"
            or observed.get("parameter_dtype") != "torch.bfloat16"
            or not observed.get("parameter_device", "").startswith("cuda")
            or observed.get("max_model_len") != 65536
            or observed.get("max_num_seqs") != 1
            or observed.get("prefix_caching") is not False
            or observed.get("tensor_parallel_size") != 1
            or observed.get("quantization") is not None
            or any(
                observed.get(name + "_version") != version
                for name, version in expected_versions.items()
            )
        ):
            raise ObserverError("actual engine settings differ from frozen E1 policy")
        live = verify_process_member(owner, first["pid"])
        if canonical(live) != canonical(observed["identity"]):
            raise ObserverError("engine process identity changed")
        identities[first["pid"]] = live
        identities[api[0]["pid"]] = verify_process_member(owner, api[0]["pid"])
        sequence = first["sequence"]
        next_begin = next(
            (
                row["sequence"]
                for row in stream
                if row["sequence"] > sequence and row["event"] == "sample_begin"
            ),
            float("inf"),
        )
        segment = [row for row in stream if sequence < row["sequence"] < next_begin]
        noise = [row for row in segment if row["event"] == "exponential_noise_before_consumption"]
        ends = [row for row in segment if row["event"] == "sample_end"]
        if len(noise) != 1 or len(ends) != 1 or noise[0]["sequence"] >= ends[0]["sequence"]:
            raise ObserverError("challenge RNG consumption was not completed exactly once")
        initial = next(
            row for row in first["payload"]["rows"] if row["internal_request_id"] == internal
        )
        final = next(
            row for row in ends[0]["payload"]["rows"] if row["internal_request_id"] == internal
        )
        q = noise[0]["payload"]["q_before_original"]
        if (
            initial["seed"] != challenge.seed
            or initial["generator_before"]["initial_seed"] != challenge.seed
            or final["generator_after"]["initial_seed"] != challenge.seed
            or initial["generator_before"]["state_sha256"]
            == final["generator_after"]["state_sha256"]
            or initial["prompt_token_ids_sha256"] != challenge.prompt_token_ids_sha256
            or internal not in noise[0]["payload"]["rows"]
            or not q["device"].startswith("cuda")
            or q["sampled_values_finite"] is not True
        ):
            raise ObserverError("challenge CUDA RNG/seed/tokenizer binding failed")
        matched.append(
            {
                "external_request_id": challenge.external_request_id,
                "internal_request_id": internal,
                "seed": challenge.seed,
                "api_pid": api[0]["pid"],
                "engine_pid": first["pid"],
                "sampled_token_id": final["sampled_token_id"],
                "q_prefix_sha256": sha(canonical(q)),
                "generator_after_sha256": final["generator_after"]["state_sha256"],
                "engine": observed,
            }
        )
    if len(challenges) == 2:
        if (
            challenges[0].seed == challenges[1].seed
            or challenges[0].seed % 2**32 != challenges[1].seed % 2**32
            or matched[0]["q_prefix_sha256"] == matched[1]["q_prefix_sha256"]
            or matched[0]["generator_after_sha256"] == matched[1]["generator_after_sha256"]
        ):
            raise ObserverError("high seed bits did not produce distinct observed CUDA RNG streams")
    return {
        "kind": "path1_e1_backend_execution",
        "challenges": matched,
        "kernel_identities": list(identities.values()),
        "scientific_admission": False,
        "scope": "two_seeded_technical_requests_same_live_E1_service",
    }
