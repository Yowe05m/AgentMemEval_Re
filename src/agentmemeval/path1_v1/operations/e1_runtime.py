"""Owned, instrumented E1 service lifecycle. Never admits E2/E3 or a paper endpoint."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from agentmemeval.path1_v1.operations import runtime_probe as probe
from agentmemeval.path1_v1.operations.backend_challenges import Challenge, canonical, sha
from agentmemeval.path1_v1.operations.backend_plugin import PLUGIN_NAME, create_overlay
from agentmemeval.path1_v1.operations.closure import ClosureBinding
from agentmemeval.path1_v1.operations.closure_service import ServiceReleaseJournal
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.e1_backend_runtime import (
    observe_engine_gpus,
    read_traces,
    verify_challenge_events,
    verify_process_member,
)
from agentmemeval.path1_v1.operations.process_ownership import OwnedProcess, ProcessBinding
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_process import file_sha256, write_exclusive_json
from agentmemeval.path1_v1.operations.worker import (
    RuntimeAdmission,
    RuntimeAdmissionRequired,
    RuntimeProbeFile,
    run_worker,
)


def _require(condition, message):
    if not condition:
        raise RuntimeAdmissionRequired(message)


def _write(path, value):
    write_exclusive_json(path, json.loads(json.dumps(value, allow_nan=False)))


def challenge_plan(tokenizer, seeds):
    messages = [{"role": "user", "content": "Reply with one ordinary English word."}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = probe._token_ids(tokenizer, messages)
    challenges = tuple(
        Challenge(
            "chatcmpl-path1-" + uuid4().hex,
            sha(prompt.encode()),
            sha(canonical(ids)),
            seed,
            1.0,
            1.0,
            -1,
            8,
        )
        for seed in seeds
    )
    return messages, challenges


def _schemas():
    action = probe._response_format("json_schema")
    patch = probe._experience_patch_response_format(SimpleNamespace(metadata={}))
    patch = probe._remove_json_schema_keywords(patch, {"uniqueItems"})
    section = patch["json_schema"]["schema"]["properties"]["patches"]["items"]["anyOf"][0][
        "properties"
    ]["section"]["enum"][0]
    example = {
        "operation": "patch",
        "patches": [
            {
                "section": section,
                "op": "add",
                "rule_id": "",
                "when": ["technical fixture"],
                "guidance": "Evaluate legal actions before acting.",
                "invalidators": ["fixture ends"],
                "basis": "model_prior",
                "status": "tentative",
                "supporting_trajectory_ids": [],
                "supporting_fact_ids": [],
            }
        ],
        "fact_assessments": [],
        "calibration_note": "E1 technical check",
        "self_check": "Not an automatically formed research artifact",
    }
    return action, patch, example


def collect_http_checks(request, owner, endpoint, tokenizer, output, live):
    action, patch, example = _schemas()
    short = [{"role": "user", "content": "Return a legal fold with a brief explanation."}]
    cases = [
        ("action", short, action, 256, 0.0, False),
        (
            "patch",
            [
                {
                    "role": "user",
                    "content": "Return this technical patch: " + canonical(example).decode(),
                }
            ],
            patch,
            8192,
            1.0,
            False,
        ),
        (
            "repair",
            [{"role": "user", "content": "Previous output invalid. Return legal fold."}],
            action,
            256,
            0.0,
            False,
        ),
        (
            "patch_repair",
            [{"role": "user", "content": "Previous patch invalid. Return keep."}],
            patch,
            4096,
            1.0,
            False,
        ),
        ("near_context", probe._capacity_messages(tokenizer, 65536 - 256), action, 256, 0.0, False),
        ("over_context", probe._capacity_messages(tokenizer, 65536 + 32), action, 256, 0.0, True),
    ]
    observations = []
    for index, (name, messages, schema, budget, temperature, over) in enumerate(cases):
        live()
        ids = probe._token_ids(tokenizer, messages)
        payload = {
            "model": request.contract["source_contract"]["provider"]["served_model_name"],
            "messages": messages,
            "max_tokens": budget,
            "temperature": temperature,
            "seed": 2**62 + 1000 + index,
            "response_format": schema,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        _write(output / (name + "_request.json"), payload)
        response = probe.http_exchange(endpoint, payload)
        _write(output / (name + "_response.json"), response)
        result = probe.verify_http_observation(payload, response, len(ids), over_limit=over)
        if name == "patch":
            _require(
                result["content"]["operation"] == "patch" and bool(result["content"]["patches"]),
                "patch branch was not exercised",
            )
        live()
        observations.append(
            {
                "name": name,
                "token_ids_sha256": content_digest(ids),
                "local_token_count": len(ids),
                "request_file_sha256": file_sha256(output / (name + "_request.json")),
                "response_file_sha256": file_sha256(output / (name + "_response.json")),
                **result,
            }
        )
    return observations


class E1RuntimeVerifier:
    """Issued only beside a live owner; not constructible from a JSON success flag."""

    def __init__(
        self,
        *,
        owner,
        request,
        endpoint,
        directory,
        tokenizer,
        challenges,
        challenge_messages,
        plan,
        host,
    ):
        _require(request.contract["phase"] == "E1", "runtime is restricted to technical E1")
        self.owner, self.request, self.endpoint = owner, request, endpoint
        self.directory, self.tokenizer = directory, tokenizer
        self.challenges, self.challenge_messages = challenges, challenge_messages
        self.plan, self.host = plan, host
        self.snapshot = owner.verify_live()
        self.deadline = datetime.fromisoformat(host["allocation_end_utc"]) - timedelta(
            seconds=plan["cleanup_seconds"]
        )
        self._admission = None
        self._model_metadata = None
        self._backend_identities = ()

    def _live(self):
        _require(
            datetime.now(UTC) + timedelta(seconds=600) < self.deadline,
            "allocation has insufficient request/cleanup time",
        )
        current = self.owner.verify_live()
        _require(
            current.identity == self.snapshot.identity
            and current.binding == self.snapshot.binding
            and current.cmdline == self.snapshot.cmdline,
            "E1 service identity changed",
        )
        self.owner.verify_endpoint(self.endpoint)
        for pin in self.plan["source_pins"]:
            _require(file_sha256(Path(pin["path"])) == pin["sha256"], "backend source pin changed")

        if self._model_metadata is not None:
            _require(
                self._model_metadata == probe._metadata(Path(self.request.request["model_root"])),
                "model file identity changed while service is running",
            )
        for identity in self._backend_identities:
            _require(
                canonical(verify_process_member(self.owner, identity["pid"]))
                == canonical(identity),
                "backend engine/API process identity changed",
            )

    def verify_live(self, request, admission):
        _require(
            admission is self._admission
            and request.intent_id == self.request.intent_id
            and request.request_sha256 == self.request.request_sha256,
            "runtime capability belongs to another worker",
        )
        self._live()

    def verify_in_job(self, request):
        _require(
            self._admission is None
            and request.contract["phase"] == "E1"
            and request.intent_id == self.request.intent_id
            and request.request_sha256 == self.request.request_sha256,
            "E1 runtime cannot be reused or transferred",
        )
        self._live()
        identity = request.contract["runtime_identity"]
        model, metadata = probe.observe_model(request)
        template = self.tokenizer.get_chat_template()
        _require(
            sha(template.encode()) == model["manifest"]["chat_template_sha256"],
            "local actual tokenizer template differs from model manifest",
        )
        probe._verify_loaded_source(request)
        observations = collect_http_checks(
            request, self.owner, self.endpoint, self.tokenizer, self.directory, self._live
        )
        for index, challenge in enumerate(self.challenges):
            self._live()
            payload = {
                "model": request.contract["source_contract"]["provider"]["served_model_name"],
                "messages": self.challenge_messages,
                "request_id": challenge.external_request_id.removeprefix("chatcmpl-"),
                "seed": challenge.seed,
                "temperature": challenge.temperature,
                "top_p": challenge.top_p,
                "top_k": challenge.top_k,
                "max_tokens": challenge.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            _write(self.directory / f"seed_{index}_request.json", payload)
            response = probe.http_exchange(self.endpoint, payload)
            _write(self.directory / f"seed_{index}_response.json", response)
            body = probe._strict_json(response["body_utf8"])
            _require(
                response["status"] == 200
                and body.get("id") == challenge.external_request_id
                and body.get("model") == payload["model"],
                "exact seed challenge HTTP failed",
            )
        self._live()
        events = read_traces(
            self.directory / "traces",
            service_instance=self.snapshot.binding.service_instance_id,
            source_sha256=content_digest(identity),
        )
        lock = identity["dependency_lock"]
        lock_path = Path(request.request["repository_root"]) / lock["path"]
        _require(file_sha256(lock_path) == lock["sha256"], "observed environment lock changed")
        distribution_rows = json.loads(lock_path.read_bytes())["distributions"]
        versions = {
            row["name"].lower().replace("_", "-"): row["version"] for row in distribution_rows
        }
        _require(len(versions) == len(distribution_rows), "duplicate observed distribution")
        expected_versions = {name: versions[name] for name in ("torch", "vllm", "transformers")}
        _require(
            expected_versions["vllm"] in {"0.23.0", "0.23.0+cu129"},
            "backend adapter has not reviewed this exact vLLM build",
        )
        backend = verify_challenge_events(
            events,
            self.challenges,
            owner=self.owner,
            model_root=Path(request.request["model_root"]),
            source_pins=self.plan["source_pins"],
            expected_versions=expected_versions,
        )
        gpu = probe.observe_gpu()
        backend["gpu_processes"] = observe_engine_gpus(
            {row["engine_pid"] for row in backend["challenges"]}, gpu["gpus"]
        )
        self._backend_identities = tuple(backend["kernel_identities"])
        _write(self.directory / "gpu.json", gpu)
        # Freeze the observed event prefix, not files that the live plugin may append.
        _write(self.directory / "backend_events.json", {"events": events})
        backend["raw_events_sha256"] = file_sha256(self.directory / "backend_events.json")
        _write(self.directory / "backend_execution.json", backend)
        _require(
            metadata == probe._metadata(Path(request.request["model_root"])),
            "model files changed during runtime verification",
        )
        self._model_metadata = metadata
        _write(
            self.directory / "tokenizer_context.json",
            {
                "observations": observations,
                "chat_template_sha256": sha(template.encode()),
                "server_token_ids_observed_in_seed_challenges": True,
            },
        )
        _write(self.directory / "structured_schema.json", {"observations": observations})
        _write(
            self.directory / "model_source.json",
            {
                "model": model,
                "metadata": metadata,
                "runtime_identity": identity,
                "host": self.host,
                "service": asdict(self.snapshot),
                "backend": backend,
            },
        )
        files = (
            ("sampler_seed_execution_63bit", "backend_execution.json"),
            ("tokenizer_total_context", "tokenizer_context.json"),
            ("structured_schema", "structured_schema.json"),
            ("loaded_model_and_source_identity", "model_source.json"),
        )
        probes = tuple(
            RuntimeProbeFile(
                purpose,
                str(self.directory / name),
                (self.directory / name).stat().st_size,
                file_sha256(self.directory / name),
            )
            for purpose, name in files
        )
        self._live()
        self._admission = RuntimeAdmission(
            request.intent_id,
            request.request_sha256,
            identity["core_manifest_sha256"],
            identity["operations_manifest_sha256"],
            identity["config"]["sha256"],
            identity["model"]["manifest_sha256"],
            self.host["job_id"],
            self.snapshot.binding.service_instance_id,
            datetime.now(UTC),
            self.deadline,
            probes,
        )
        return self._admission


def run_e1_service(prepared):
    """Actual allocation path. Every exit runs the exact owned process cleanup."""
    request, plan = prepared.worker, prepared.runtime_plan
    _require(
        prepared.descriptor["schema_version"] == 2 and type(plan) is dict,
        "live E1 requires a bound version-2 runtime plan",
    )
    _require(request.contract["phase"] == "E1", "only E1 technical execution is supported")
    proof = plan["service_python_identity"]
    _require(proof == prepared.descriptor["python_identity"], "service interpreter differs")
    host = probe.observe_host(cleanup_seconds=plan["cleanup_seconds"])
    _require(
        datetime.fromisoformat(host["allocation_end_utc"])
        > datetime.now(UTC)
        + timedelta(seconds=plan["startup_seconds"] + plan["cleanup_seconds"] + 600),
        "allocation too short for startup/request/cleanup",
    )
    output = request.output_directory.parent / "service_runtime"
    _require(not output.exists(), "E1 service evidence output already exists")
    output.mkdir(parents=True, exist_ok=False)
    trace, registrations = output / "traces", output / "registrations"
    trace.mkdir()
    registrations.mkdir()
    tokenizer = probe._load_tokenizer(Path(request.request["model_root"]))
    messages, challenges = challenge_plan(tokenizer, plan["backend_probe_seeds"])
    technical_seeds = set(plan["backend_probe_seeds"]) | set(range(2**62 + 1000, 2**62 + 1006))
    _require(
        technical_seeds <= set(request.seed_inventory["forbidden_sampler_seeds"]),
        "technical challenge seeds must be excluded from worker sampler streams",
    )
    overlay = output / "plugin_overlay"
    overlay_files = create_overlay(overlay)
    instance = uuid4().hex
    identity = request.contract["runtime_identity"]
    config = {
        "schema_version": 2,
        "service_instance": instance,
        "source_sha256": content_digest(identity),
        "trace_directory": str(trace),
        "registration_directory": str(registrations),
        "challenges": [asdict(item) for item in challenges],
        "source_pins": plan["source_pins"],
    }
    config_path = output / "backend_plugin.json"
    _write(config_path, config)
    model = request.request["model_root"]
    endpoint = f"http://127.0.0.1:{plan['service_port']}/v1"
    argv = [
        proof["entry_path"],
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
        "--language-model-only",
        "--served-model-name",
        request.contract["source_contract"]["provider"]["served_model_name"],
        "--host",
        "127.0.0.1",
        "--port",
        str(plan["service_port"]),
    ]
    binding = ProcessBinding(
        request.contract["campaign_id"],
        request.intent_id,
        host["job_id"],
        content_digest(identity),
        "model_service",
        instance,
    )
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            (str(Path(request.request["repository_root"]) / "src"), str(overlay))
        ),
        VLLM_PLUGINS=PLUGIN_NAME,
        PATH1_BACKEND_PLUGIN_CONFIG=str(config_path),
        PATH1_BACKEND_PLUGIN_CONFIG_SHA256=file_sha256(config_path),
    )
    native = plan.get("native_runtime")
    if native is not None:
        from agentmemeval.path1_v1.operations.native_runtime import environment as native_env
        from agentmemeval.path1_v1.operations.native_runtime import verify_loaded
        verify_loaded(native)
        environment = native_env(native, environment)
    if "cuda_toolkit" in plan:
        from agentmemeval.path1_v1.operations.cuda_toolkit import environment as toolkit_env
        environment = toolkit_env(plan["cuda_toolkit"], environment)
    _write(
        output / "launch.json",
        {
            "argv": argv,
            "binding": asdict(binding),
            "runtime_plan_sha256": content_digest(plan),
            "overlay_files": overlay_files,
            "plugin_config_sha256": file_sha256(config_path),
            "scientific_admission": False,
        },
    )
    claim = request.claim
    store = CampaignStore(claim["control_directory"], claim["campaign_id"])
    service_control = (
        Path(claim["control_directory"]).parent.parent / "e1_service" / claim["campaign_id"]
    )
    service_store = CampaignStore(service_control, claim["campaign_id"])
    with service_store.lock():
        if not (service_control / "campaign.sqlite3").exists():
            service_store.initialize(registry={}, state={})
        else:
            service_store.read_registry()
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", host["scontrol_stdout"]))
    closure_binding = ClosureBinding(
        request.contract["campaign_id"],
        request.intent_id,
        prepared.submission.request_sha256,
        request.request_sha256,
        host["job_id"],
        fields["SubmitTime"],
        claim["run_id"],
        instance,
        content_digest(identity),
        identity["core_manifest_sha256"],
        identity["operations_manifest_sha256"],
        identity["config"]["sha256"],
        identity["model"]["manifest_sha256"],
        request.contract["protocol_sha256"],
    )
    journal = ServiceReleaseJournal.begin(
        service_store,
        run_id=claim["run_id"] + "_service",
        binding=closure_binding,
        directory=request.output_directory.parent / "service_release",
        worker_store=store,
        worker_terminal_path=request.output_directory / "worker_terminal.json",
    )
    completed = None

    def persist_cleanup(owner, receipt):
        _write(
            output / ("service_cleanup.json" if owner else "spawn_failure.json"), asdict(receipt)
        )
        if owner is not None and completed is not None:
            journal.publish(owner, receipt)

    previous_url = os.environ.get("PATH1_MODEL_BASE_URL")
    try:
        with (
            (output / "service_stdout.log").open("xb") as stdout,
            (output / "service_stderr.log").open("xb") as stderr,
        ):
            with OwnedProcess.managed(
                argv,
                binding=binding,
                cwd=Path(request.request["repository_root"]),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                term_grace_seconds=120,
                receipt_sink=persist_cleanup,
            ) as owner:
                journal.record_launch(owner, owner.verify_live())
                start_deadline = time.monotonic() + plan["startup_seconds"]
                while True:
                    owner.verify_live()
                    try:
                        owner.verify_endpoint(endpoint)
                        opener = build_opener(ProxyHandler({}), probe._NoRedirect())
                        with opener.open(
                            endpoint.removesuffix("/v1") + "/health", timeout=5
                        ) as response:
                            ready = response.code == 200
                    except (OSError, RuntimeError):
                        ready = False
                    if ready:
                        break
                    _require(time.monotonic() < start_deadline, "E1 service startup timed out")
                    time.sleep(1)
                if native is not None:
                    _write(output / "native_runtime.json", verify_loaded(
                        native, pid=owner.verify_live().identity.pid))
                verifier = E1RuntimeVerifier(
                    owner=owner,
                    request=request,
                    endpoint=endpoint,
                    directory=output,
                    tokenizer=tokenizer,
                    challenges=challenges,
                    challenge_messages=messages,
                    plan=plan,
                    host=host,
                )
                os.environ["PATH1_MODEL_BASE_URL"] = endpoint
                completed = run_worker(
                    Path(prepared.descriptor["worker_request_path"]), runtime_verifier=verifier
                )
        return completed
    except BaseException as exc:
        _write(
            output / "service_failure.json",
            {
                "failure_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "intent_id": request.intent_id,
                "scientific_admission": False,
                "retry_eligibility": False,
            },
        )
        raise
    finally:
        if previous_url is None:
            os.environ.pop("PATH1_MODEL_BASE_URL", None)
        else:
            os.environ["PATH1_MODEL_BASE_URL"] = previous_url
