"""Live E0 observations and a closed admission gate for uninstrumented backends.

This module reads real kernel/files/HTTP data. A report is evidence, never a
bearer token. No JSON boolean, response text repetition, version string, or
model-directory name can establish sampler or loaded-model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from agentmemeval.path1_v1.experiments.path1 import _new_json
from agentmemeval.path1_v1.llm.providers.openai_compatible import (
    _experience_patch_response_format,
    _remove_json_schema_keywords,
    _response_format,
)
from agentmemeval.path1_v1.operations.contracts import (
    _model_identity,
    _object,
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.worker import (
    RuntimeAdmissionRequired,
    ValidatedWorkerRequest,
    _absolute,
    _bound_json,
    _hash_file,
)

PENDING_EXECUTION_EVIDENCE = (
    "sampler_seed_63bit_reaches_executed_generator",
    "server_tokenizer_chat_template_matches_local_token_ids",
    "backend_loader_reads_bound_weight_manifest_and_effective_configuration",
)


class RuntimeProbeError(RuntimeAdmissionRequired):
    pass


@dataclass(frozen=True)
class ValidatedProbeRequest:
    request: dict
    contract: dict
    intent_id: str
    request_sha256: str
    output_directory: Path


def validate_probe_request(request_path: Path | str) -> ValidatedProbeRequest:
    """Independent E0 topology; never borrows a research claim or seed inventory."""
    raw = _absolute(str(request_path)).read_bytes()
    request = _strict_json(raw)
    _object(request, {"schema_version", "kind", "request_sha256", "intent_id",
                      "contract_path", "contract_sha256", "repository_root",
                      "model_manifest_path", "model_root", "output_root"}, "E0 probe request")
    _require(type(request["schema_version"]) is int and request["schema_version"] == 1
             and request["kind"] == "path1_e0_probe_request", "invalid E0 request schema")
    _require(content_digest({k: v for k, v in request.items() if k != "request_sha256"})
             == request["request_sha256"], "E0 request digest mismatch")
    contract = validate_campaign_contract(_bound_json(request["contract_path"],
                                                       request["contract_sha256"]))
    _require(contract["phase"] == "E0" and contract["job"]["kind"] == "service_probe",
             "probe requires a dedicated E0 service_probe contract")
    repository = _absolute(request["repository_root"])
    model_root = _absolute(request["model_root"])
    manifest = _absolute(request["model_manifest_path"])
    identity = contract["runtime_identity"]
    observed = build_runtime_identity(repository,
        core_paths=[x["path"] for x in identity["core_files"]],
        operations_paths=[x["path"] for x in identity["operations_files"]],
        config_path=identity["config"]["path"],
        dependency_lock_path=identity["dependency_lock"]["path"],
        model_manifest_path=manifest, model_root=model_root)
    _require(observed == identity, "E0 source/model identity changed")
    _require(not observed["dirty"] and observed["content_matches_head"]
             and not observed["untracked_declared_paths"]
             and not observed["untracked_execution_paths"]
             and not observed["uncovered_execution_paths"], "E0 source closure is not frozen")
    intent = canonical_intent_id(contract)
    _require(request["intent_id"] == intent, "E0 intent mismatch")
    root = _absolute(request["output_root"])
    output = root.joinpath(*contract["job"]["output_relpath"].split("/"), "e0_probe")
    _absolute(str(output), exists=False)
    _require(not output.exists(), "E0 output already exists")
    _require(all(output != path and not output.is_relative_to(path)
                 for path in (repository, model_root)), "E0 output overlaps source/model")
    _require(_absolute(str(request_path)).read_bytes() == raw, "E0 request changed while reading")
    return ValidatedProbeRequest(request, contract, intent, request["request_sha256"], output)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeProbeError(message)


def _strict_json(raw: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise RuntimeProbeError("nonfinite JSON value: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _endpoint(value: str) -> str:
    parsed = urlsplit(value)
    _require(
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
        and parsed.username is None and parsed.password is None
        and parsed.port is not None and 0 < parsed.port < 65536
        and parsed.path.rstrip("/") == "/v1" and not parsed.query and not parsed.fragment,
        "probe requires an explicit numeric loopback /v1 endpoint",
    )
    return value.rstrip("/")


def _metadata(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "model symlinks are not immutable evidence")
        if path.is_file():
            stat = path.stat()
            result.append({"path": path.relative_to(root).as_posix(), "size": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                           "device": stat.st_dev, "inode": stat.st_ino})
    return result


def observe_model(request: ValidatedProbeRequest) -> tuple[dict, list[dict]]:
    root = _absolute(request.request["model_root"])
    manifest = _absolute(request.request["model_manifest_path"])
    before = _metadata(root)
    identity = _model_identity(manifest, root)
    _require(identity == request.contract["runtime_identity"]["model"],
             "actual model files differ from the contract")
    paths = {record["path"] for record in before}
    _require("config.json" in paths and "tokenizer_config.json" in paths,
             "model/config/tokenizer files must be present")
    _require("tokenizer.json" in paths or "tokenizer.model" in paths,
             "tokenizer vocabulary bytes missing")
    _require(any(path.endswith(".safetensors") for path in paths), "weight bytes missing")
    after = _metadata(root)
    _require(before == after, "model files changed during hashing")
    return identity, after


def observe_host() -> dict:
    _require(sys.platform == "linux" and hasattr(os, "geteuid"),
             "live E0 producer requires a Linux allocation")
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and int(job) > 0, "not inside a Slurm job")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    _require(any(f"job_{job}" in part.split("/") for part in cgroup.splitlines()),
             "current kernel cgroup does not bind the advertised Slurm job")
    allocation = subprocess.run(["scontrol", "show", "job", "--oneliner", job],
                                capture_output=True, timeout=30, check=False)
    raw = allocation.stdout.decode("utf-8", errors="strict")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", raw))
    _require(allocation.returncode == 0 and fields.get("JobId") == job
             and fields.get("JobState") == "RUNNING"
             and fields.get("UserId", "").endswith(f"({os.geteuid()})"),
             "actual Slurm allocation/owner is not running")
    try:
        end = datetime.fromisoformat(fields["EndTime"]).astimezone(UTC)
    except (KeyError, ValueError) as exc:
        raise RuntimeProbeError("actual Slurm deadline is missing/unsupported") from exc
    _require(end > datetime.now(UTC) + timedelta(seconds=600),
             "allocation has no reserved cleanup interval")
    return {"hostname": socket.gethostname(), "boot_id": boot, "euid": os.geteuid(),
            "pid": os.getpid(), "job_id": job, "cgroup": cgroup,
            "scontrol_stdout": raw,
            "allocation_end_utc": end.isoformat(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version()}


def observe_gpu() -> dict:
    args = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits"]
    completed = subprocess.run(args, capture_output=True, timeout=30, check=False)
    raw = completed.stdout.decode("utf-8", errors="strict")
    _require(completed.returncode == 0 and bool(raw.strip()), "nvidia-smi did not report GPU data")
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4 and parts[0].startswith("GPU-")
                 and parts[3].isdigit(), "malformed GPU observation")
        rows.append(dict(zip(("uuid", "name", "driver_version", "memory_mib"), parts,
                             strict=True)))
    _require(len({row["uuid"] for row in rows}) == len(rows), "duplicate GPU UUID")
    return {"argv": args, "returncode": completed.returncode, "stdout": raw,
            "stderr": completed.stderr.decode("utf-8", errors="strict"), "gpus": rows,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "allocation_gpu_mapping_verified": False}


def observe_dependencies() -> dict:
    result = {}
    for name in ("torch", "transformers", "vllm"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        result[name] = {"version": str(module.__version__), "module_file": str(path),
                        "module_sha256": _hash_file(path)}
        if name == "torch":
            result[name]["compiled_cuda"] = module.version.cuda
    # This is the probe interpreter's actual import, not proof of server imports.
    return {"scope": "probe_interpreter_only", "modules": result}


def _service(owner, request: ValidatedProbeRequest, host: dict) -> dict:
    snapshot = owner.verify_live()
    data = asdict(snapshot)
    binding = data["binding"]
    identity = data["identity"]
    _require(binding["campaign_id"] == request.contract["campaign_id"]
             and binding["intent_id"] == request.intent_id
             and binding["job_id"] == host["job_id"]
             and binding["source_sha256"]
             == content_digest(request.contract["runtime_identity"])
             and binding["role"] == "model_service" and bool(binding["service_instance_id"]),
             "live service belongs to another contract/job/source")
    _require(identity["effective_uid"] == host["euid"]
             and identity["host"] == host["hostname"] and identity["boot_id"] == host["boot_id"]
             and identity["slurm_job_id"] == host["job_id"], "service kernel identity mismatch")
    argv = data["cmdline"]
    _require(isinstance(argv, (tuple, list)) and all(type(arg) is str for arg in argv),
             "service argv must come from the live kernel")
    _require(not any("api-key" in arg or "token=" in arg for arg in argv),
             "secret-bearing service argv is not admissible to raw evidence")
    return data


def _same_service(left: dict, right: dict) -> bool:
    # Scheduling state R/S is observational, not a process identity change.
    return all(left[key] == right[key] for key in ("binding", "identity", "cmdline"))


def _verify_loaded_source(request: ValidatedProbeRequest) -> None:
    from agentmemeval.path1_v1.operations.intent_executor import verify_loaded_package

    identity = request.contract["runtime_identity"]
    verify_loaded_package(Path(request.request["repository_root"]),
                          identity["core_files"] + identity["operations_files"])


def check_service_arguments(argv: list[str] | tuple[str, ...], request) -> dict:
    """Check explicit startup arguments; effective backend state stays pending."""
    flags = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg.split("=", 1)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                key, value = arg, argv[index + 1]
                index += 1
            else:
                key, value = arg, True
            _require(key not in flags, "duplicate service flag")
            flags[key] = value
        index += 1
    expected = {"--dtype": "bfloat16", "--tensor-parallel-size": "1",
                "--max-model-len": "65536", "--max-num-seqs": "1",
                "--served-model-name": request.contract["source_contract"]["provider"]
                ["served_model_name"]}
    _require(all(flags.get(key) == value for key, value in expected.items()),
             "explicit service arguments differ from frozen settings")
    _require(flags.get("--no-enable-prefix-caching") is True
             and "--enable-prefix-caching" not in flags and "--quantization" not in flags,
             "cache/quantization startup policy mismatch")
    model = str(_absolute(request.request["model_root"]))
    _require(flags.get("--model") == model or ("serve" in argv and
             list(argv[argv.index("serve") + 1:argv.index("serve") + 2]) == [model]),
             "service did not select the actual bound model path")
    return {"explicit_flags": flags, "effective_backend_settings_verified": False}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeProbeError("runtime probe refuses HTTP redirects")


def http_exchange(base_url: str, payload: dict) -> dict:
    data = canonical_json(payload)
    request = Request(_endpoint(base_url) + "/chat/completions", data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
    key = os.environ.get("PATH1_MODEL_API_KEY")
    if key:
        request.add_header("Authorization", "Bearer " + key)
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        response = opener.open(request, timeout=600)
    except HTTPError as exc:
        response = exc
    with response:
        raw = response.read(16 * 1024 * 1024 + 1)
        _require(len(raw) <= 16 * 1024 * 1024, "probe response too large")
        return {"status": response.code, "body_utf8": raw.decode("utf-8", errors="strict"),
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "request_sha256": hashlib.sha256(data).hexdigest()}


def _validate_schema(value, schema) -> None:
    _require(set(schema) <= {"anyOf", "type", "enum", "required", "properties",
                            "additionalProperties", "minItems", "maxItems", "items",
                            "minLength", "maxLength", "minimum", "maximum"},
             "probe validator does not implement this schema keyword")
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate_schema(value, branch)
                return
            except RuntimeProbeError:
                pass
        raise RuntimeProbeError("schema anyOf did not match")
    kind = schema.get("type")
    matches = {"object": type(value) is dict, "array": type(value) is list,
               "string": type(value) is str, "integer": type(value) is int,
               "number": type(value) in {float, int}, "null": value is None,
               "boolean": type(value) is bool}
    _require(kind in matches and matches[kind], "schema type mismatch")
    if "enum" in schema:
        _require(any(type(value) is type(x) and value == x for x in schema["enum"]),
                 "schema enum mismatch")
    if kind == "object":
        _require(set(schema.get("required", ())) <= value.keys(),
                 "required response fields missing")
        props = schema.get("properties", {})
        _require(schema.get("additionalProperties") is not False or value.keys() <= props.keys(),
                 "extra response fields")
        for key, item in value.items():
            _require(key in props, "unknown schema property")
            _validate_schema(item, props[key])
    elif kind in {"array", "string"}:
        suffix = "Items" if kind == "array" else "Length"
        _require(schema.get("min" + suffix, 0) <= len(value)
                 <= schema.get("max" + suffix, math.inf), "schema length outside range")
        if kind == "array":
            for item in value:
                _validate_schema(item, schema["items"])
    elif kind in {"number", "integer"}:
        _require(math.isfinite(value) and schema.get("minimum", -math.inf) <= value
                 <= schema.get("maximum", math.inf), "schema number outside range")


def verify_http_observation(payload: dict, observed: dict, local_tokens: int,
                            *, over_limit: bool = False) -> dict:
    _require(type(observed["status"]) is int, "HTTP status must be integral")
    raw = observed["body_utf8"].encode("utf-8")
    _require(hashlib.sha256(raw).hexdigest() == observed["body_sha256"]
             and hashlib.sha256(canonical_json(payload)).hexdigest()
             == observed["request_sha256"], "HTTP raw evidence digest mismatch")
    response = _strict_json(raw)
    if over_limit:
        _require(observed["status"] == 400 and type(response) is dict
                 and type(response.get("error")) is dict, "over-limit request was not rejected")
        return {"observed_http_400": True, "context_rejection_reason": response["error"],
                "server_tokenizer_execution_verified": False}
    _require(observed["status"] == 200 and type(response) is dict, "probe request failed")
    _require(response.get("model") == payload["model"], "response model alias differs")
    usage = response.get("usage")
    _require(type(usage) is dict, "response token usage missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _require(type(usage.get(key)) is int and usage[key] >= 0, "invalid usage count")
    _require(usage["prompt_tokens"] == local_tokens
             and 0 < usage["completion_tokens"] <= payload["max_tokens"]
             and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
             and usage["total_tokens"] <= 65536, "token accounting/context mismatch")
    choices = response.get("choices")
    _require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
             and choices[0].get("finish_reason") == "stop", "truncated or ambiguous response")
    message = choices[0].get("message")
    _require(type(message) is dict and type(message.get("content")) is str,
             "response message missing")
    content = _strict_json(message["content"])
    _validate_schema(content, payload["response_format"]["json_schema"]["schema"])
    return {"observed_response_schema_match": True, "usage": usage,
            "client_seed": payload["seed"], "sampler_execution_verified": False,
            "content": content}


def _load_tokenizer(model_root: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_root), local_files_only=True,
                                         trust_remote_code=False)


def _token_ids(tokenizer, messages: list[dict]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    _require(type(ids) is list and all(type(x) is int and x >= 0 for x in ids),
             "tokenizer must return actual integral token IDs")
    return ids


def _capacity_messages(tokenizer, target: int) -> list[dict]:
    low, high = 0, target * 4
    best = None
    while low <= high:
        middle = (low + high) // 2
        messages = [{"role": "user", "content": "capacity " * middle + "Return a legal fold."}]
        size = len(_token_ids(tokenizer, messages))
        if size <= target:
            best = messages
            low = middle + 1
        else:
            high = middle - 1
    _require(best is not None and target - len(_token_ids(tokenizer, best)) <= 16,
             "could not construct an actual near-boundary tokenizer probe")
    return best


def collect_e0_probe(request: ValidatedProbeRequest, owner, base_url: str,
                     output_directory: Path) -> dict:
    """Read real evidence. Requires a live owner capability, never a PID JSON list.

    Even all successful observations leave backend execution pending. The owner
    must be created in this process by the service lifecycle implementation.
    """
    _require(type(request) is ValidatedProbeRequest and request.contract["phase"] == "E0",
             "collector requires an independent E0 request")
    output = _absolute(str(output_directory), exists=False)
    endpoint = _endpoint(base_url)
    _require(output == request.output_directory / "observations",
             "E0 observations must use the bound output path")
    for name in ("repository_root", "model_root"):
        root = _absolute(request.request[name])
        _require(output != root and not output.is_relative_to(root),
                 "E0 evidence must be outside source/model roots")
    output.mkdir(parents=True, exist_ok=False)
    evidence = {"schema_version": 1, "kind": "path1_live_e0_observation",
                "intent_id": request.intent_id, "probe_request_sha256": request.request_sha256,
                "campaign_id": request.contract["campaign_id"],
                "issued_at_utc": datetime.now(UTC).isoformat(),
                "runtime_admission": False, "pending": [*PENDING_EXECUTION_EVIDENCE,
                    "GPU_UUIDs_mapped_to_allocation_and_owned_engine_processes",
                    "server_interpreter_loaded_dependency_closure"],
                "observations": []}
    try:
        host = observe_host()
        service = _service(owner, request, host)
        _new_json(output / "host_service.json", {"host": host, "service": service})
        arguments = check_service_arguments(service["cmdline"], request)
        _new_json(output / "service_arguments.json", arguments)
        _new_json(output / "gpu.json", observe_gpu())
        _new_json(output / "dependencies.json", observe_dependencies())
        model, metadata = observe_model(request)
        _new_json(output / "model_files.json", {"identity": model, "metadata": metadata})
        _verify_loaded_source(request)
        tokenizer = _load_tokenizer(Path(request.request["model_root"]))
        template = tokenizer.get_chat_template()
        _require(type(template) is str and hashlib.sha256(template.encode()).hexdigest()
                 == model["manifest"]["chat_template_sha256"], "actual chat template differs")
        _new_json(output / "tokenizer.json", {"class": type(tokenizer).__name__,
                  "chat_template": template, "chat_template_sha256": hashlib.sha256(
                      template.encode()).hexdigest(), "server_equivalence_verified": False})
        if hasattr(owner, "verify_endpoint"):
            _new_json(output / "endpoint.json", asdict(owner.verify_endpoint(endpoint)))
        else:
            evidence["pending"].append("endpoint_socket_bound_to_owned_process")
        action_schema = _response_format("json_schema")
        patch_schema = _experience_patch_response_format(SimpleNamespace(metadata={}))
        patch_schema = _remove_json_schema_keywords(patch_schema, {"uniqueItems"})
        section = patch_schema["json_schema"]["schema"]["properties"]["patches"]["items"][
            "anyOf"][0]["properties"]["section"]["enum"][0]
        patch_example = {"operation": "patch", "patches": [{
            "section": section, "op": "add", "rule_id": "", "when": ["technical fixture"],
            "guidance": "Evaluate legal actions before acting.", "invalidators": ["fixture ends"],
            "basis": "model_prior", "status": "tentative", "supporting_trajectory_ids": [],
            "supporting_fact_ids": []}], "fact_assessments": [], "calibration_note": "E0 only",
            "self_check": "technical fixture, never an automatically formed research artifact"}
        short = [{"role": "user", "content": "Return a legal fold with a brief explanation."}]
        cases = [
            ("action", short, action_schema, 256, 0.0, False),
            ("patch", [{"role": "user",
                        "content": "Return one valid technical patch: "
                        + canonical_json(patch_example).decode()}],
             patch_schema, 8192, 1.0, False),
            ("repair", [{"role": "user",
                         "content": "Previous reply was not JSON. Return a legal fold."}],
             action_schema, 256, 0.0, False),
            ("patch_repair", [{"role": "user",
                               "content": "Previous patch was invalid. Return keep."}],
             patch_schema, 4096, 1.0, False),
            ("near_context", _capacity_messages(tokenizer, 65536 - 256),
             action_schema, 256, 0.0, False),
            ("over_context", _capacity_messages(tokenizer, 65536 + 32),
             action_schema, 256, 0.0, True),
        ]
        for index, (name, messages, schema, budget, temperature, over) in enumerate(cases):
            _require(datetime.fromisoformat(host["allocation_end_utc"]) > datetime.now(UTC)
                     + timedelta(seconds=1200), "insufficient request and cleanup interval")
            _require(_same_service(_service(owner, request, host), service),
                     "service changed during E0")
            if hasattr(owner, "verify_endpoint"):
                owner.verify_endpoint(endpoint)
            ids = _token_ids(tokenizer, messages)
            model_name = request.contract["source_contract"]["provider"]["served_model_name"]
            payload = {"model": model_name,
                       "messages": messages, "max_tokens": budget, "temperature": temperature,
                       "seed": 2**62 + 1729 + index, "response_format": schema,
                       "chat_template_kwargs": {"enable_thinking": False}}
            _new_json(output / (name + "_request.json"), payload)
            observed = http_exchange(endpoint, payload)
            _new_json(output / (name + "_response.json"), observed)
            result = verify_http_observation(payload, observed, len(ids), over_limit=over)
            if name == "patch":
                _require(result["content"]["operation"] == "patch"
                         and bool(result["content"]["patches"]),
                         "patch probe did not exercise the patch branch")
            _require(_same_service(_service(owner, request, host), service),
                     "service changed after request")
            if hasattr(owner, "verify_endpoint"):
                owner.verify_endpoint(endpoint)
            result.update(name=name, local_token_count=len(ids),
                          token_ids_sha256=content_digest(ids), output_budget=budget,
                          seed_scope="E0_technical_fixture_only")
            evidence["observations"].append(result)
        _require(metadata == _metadata(Path(request.request["model_root"])),
                 "model files changed during E0")
        evidence["status"] = "observations_complete_backend_execution_pending"
    except BaseException as exc:
        evidence.update(status="incomplete", failure_type=type(exc).__name__)
        _new_json(output / "probe_receipt.json", evidence)
        raise
    evidence["files"] = [{"path": path.name, "size": path.stat().st_size,
                          "sha256": _hash_file(path)} for path in sorted(output.iterdir())
                         if path.is_file()]
    _new_json(output / "probe_receipt.json", evidence)
    return evidence


class StrictRuntimeVerifier:
    """Live producer-backed verifier; uninstrumented execution cannot be admitted."""

    def __init__(self, owner, base_url: str, output_directory: Path) -> None:
        self.owner = owner
        self.base_url = _endpoint(base_url)
        self.output_directory = Path(output_directory)

    def verify_in_job(self, request: ValidatedWorkerRequest):
        _require(type(request) is ValidatedProbeRequest,
                 "no live backend verifier for research; collect independent E0 evidence first")
        report = collect_e0_probe(request, self.owner, self.base_url, self.output_directory)
        # Intentionally no receipt loader or user-provided backend evidence flag.
        # A reviewed service-internal instrumentation implementation must replace
        # this missing capability, with raw challenges tied to this live process.
        raise RuntimeProbeError("backend execution evidence pending: "
                                + ", ".join(report["pending"]))


def run_service_probe(request_path: Path, output_directory: Path,
                      service_python: Path, port: int) -> dict:
    """The explicit CLI's real service lifecycle; never starts a research unit."""
    request = validate_probe_request(request_path)
    host = observe_host()  # Refuse login nodes/Windows before creating a service.
    _require(datetime.fromisoformat(host["allocation_end_utc"]) > datetime.now(UTC)
             + timedelta(seconds=1800), "insufficient service startup and cleanup interval")
    _require(type(port) is int and 1024 <= port <= 65535, "invalid dedicated service port")
    executable = Path(service_python).absolute()
    _require(executable.is_file(), "service interpreter missing")
    output = _absolute(str(output_directory), exists=False)
    _require(output == request.output_directory, "CLI E0 output differs from bound contract")
    from agentmemeval.path1_v1.operations.process_ownership import (
        OwnedProcess,
        ProcessBinding,
        SpawnOwnershipError,
    )

    _verify_loaded_source(request)  # Before model service creation, then again before probes.
    output.mkdir(parents=True, exist_ok=False)
    model = request.request["model_root"]
    argv = [str(executable), "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--tokenizer", model, "--dtype", "bfloat16", "--tensor-parallel-size", "1",
            "--max-model-len", "65536", "--max-num-seqs", "1", "--no-enable-prefix-caching",
            "--served-model-name", request.contract["source_contract"]["provider"]
            ["served_model_name"], "--host", "127.0.0.1", "--port", str(port)]
    check_service_arguments(argv, request)
    binding = ProcessBinding(campaign_id=request.contract["campaign_id"],
                             intent_id=request.intent_id, job_id=host["job_id"],
                             source_sha256=content_digest(request.contract["runtime_identity"]),
                             role="model_service",
                             service_instance_id=uuid4().hex)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    endpoint = f"http://127.0.0.1:{port}/v1"
    _new_json(output / "launch.json", {"argv": argv, "binding": asdict(binding),
              "python_resolved": str(executable.resolve()),
              "python_sha256": _hash_file(executable.resolve())})
    with (output / "service_stdout.log").open("xb") as stdout, (
        output / "service_stderr.log"
    ).open("xb") as stderr:
        owner = None
        try:
            with OwnedProcess.managed(
                argv, binding=binding, cwd=Path(request.request["repository_root"]), env=env,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                term_grace_seconds=120,
            ) as owner:
                start_deadline = time.monotonic() + 1200
                while True:
                    owner.verify_live()
                    try:
                        owner.verify_endpoint(endpoint)
                        opener = build_opener(ProxyHandler({}), _NoRedirect())
                        with opener.open(endpoint.removesuffix("/v1") + "/health",
                                         timeout=5) as response:
                            ready = response.code == 200
                    except (OSError, RuntimeError):
                        ready = False
                    if ready:
                        break
                    _require(time.monotonic() < start_deadline,
                             "service startup exceeded 20 minutes")
                    time.sleep(1)
                return collect_e0_probe(request, owner, endpoint, output / "observations")
        except SpawnOwnershipError as exc:
            _new_json(output / "service_spawn_failure.json", asdict(exc.receipt))
            raise
        finally:
            if owner is not None:
                # managed() already closed; repeat access returns its exact receipt.
                cleanup = owner.close(term_grace_seconds=120)
                _new_json(output / "service_cleanup.json", asdict(cleanup))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    validate_probe_request(args.request)
    # A standalone process cannot reconstruct the service owner's capability.
    raise RuntimeProbeError("invoke the live service lifecycle; JSON cannot restore ownership")


if __name__ == "__main__":
    raise SystemExit(main())
