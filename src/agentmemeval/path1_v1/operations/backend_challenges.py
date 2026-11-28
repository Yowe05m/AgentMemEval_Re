"""Diagnostic-only exact request registration and opt-in seeded coverage.

Registration is an observation of the pinned API method, never a RuntimeAdmission
or a reconstructed live capability. Ordinary requests retain the original call.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations.backend_observer import (
    ObserverError,
    ProcessTrace,
    SamplerObserver,
    SourcePin,
)
from agentmemeval.path1_v1.operations.submission_process import write_exclusive_json

API = "vllm.v1.engine.async_llm"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identifier(value: Any) -> str:
    if type(value) is not str or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
        raise ObserverError("invalid exact request identifier")
    return value


def _hash(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ObserverError("invalid challenge digest")
    return value


@dataclass(frozen=True)
class Challenge:
    external_request_id: str
    prompt_text_sha256: str
    prompt_token_ids_sha256: str
    seed: int
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int

    def __post_init__(self) -> None:
        _identifier(self.external_request_id)
        _hash(self.prompt_text_sha256)
        _hash(self.prompt_token_ids_sha256)
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ObserverError("challenge seed must be an exact 63-bit integer")
        for value, upper in ((self.temperature, 2.0), (self.top_p, 1.0)):
            if type(value) is not float or not math.isfinite(value) or not 0 < value <= upper:
                raise ObserverError("challenge temperature/top_p must be bounded positive floats")
        if type(self.top_k) is not int or self.top_k not in {-1, 0} and self.top_k < 1:
            raise ObserverError("invalid challenge top_k")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 64:
            raise ObserverError("challenge output bound must be 1..64")


def projection(params: Any, tokens: Any) -> dict[str, Any]:
    if (
        type(tokens) is not list
        or not 1 <= len(tokens) <= 8192
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ObserverError("actual bounded text prompt token IDs required")
    return {
        "n": params.n,
        "seed": params.seed,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "max_tokens": params.max_tokens,
        "prompt_token_ids_sha256": sha(canonical(tokens)),
        "prompt_token_count": len(tokens),
    }


class ChallengeRegistry:
    """Small immutable API-produced registration files shared with worker processes."""

    def __init__(self, directory: Path, challenges: tuple[Challenge, ...], trace: ProcessTrace):
        if (
            not directory.is_absolute()
            or directory.resolve() != directory
            or not directory.is_dir()
        ):
            raise ObserverError("registration directory must be an existing canonical directory")
        if directory.is_symlink() or not 1 <= len(challenges) <= 16:
            raise ObserverError("bounded challenge set required")
        self.directory = directory
        self.challenges = {item.external_request_id: item for item in challenges}
        if len(self.challenges) != len(challenges):
            raise ObserverError("duplicate external challenge identifier")
        self.trace = trace
        self.plan_sha256 = sha(canonical([asdict(item) for item in challenges]))
        self._pid = os.getpid()
        self._seen: set[tuple[str, tuple[str, ...]]] = set()
        self.diagnostic_failure: str | None = None

    def event(self, reason: str, ids: list[str], **details: Any) -> None:
        """Diagnostics cannot make an ordinary/greedy request fail."""
        if self._pid != os.getpid():
            self._pid, self._seen, self.diagnostic_failure = os.getpid(), set(), None
        key = (reason, tuple(ids))
        if key in self._seen:
            return
        # Bound auxiliary diagnostics even when a normal service sees many requests.
        if len(self._seen) >= 256:
            return
        self._seen.add(key)
        try:
            self.trace.write(
                "coverage_not_covered", {"reason": reason, "request_ids": ids, **details}
            )
        except Exception as exc:
            self.diagnostic_failure = type(exc).__name__

    def path(self, external: str) -> Path:
        return self.directory / (sha(external.encode()) + ".json")

    def expected(self, challenge: Challenge, count: int) -> dict[str, Any]:
        value = asdict(challenge)
        return {
            key: value[key]
            for key in (
                "seed",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "prompt_token_ids_sha256",
            )
        } | {"n": 1, "prompt_token_count": count}

    def register(self, request: Any, prompt: Any, parent: Any, index: Any) -> dict | None:
        external = request.external_req_id
        challenge = self.challenges.get(external)
        if challenge is None:
            return None
        internal = _identifier(request.request_id)
        if parent is not None or type(index) is not int or index != 0 or type(prompt) is not str:
            raise ObserverError("challenge is not a single ordinary text API request")
        if (
            getattr(request, "resumable", False)
            or getattr(request, "prompt_embeds", None) is not None
        ):
            raise ObserverError("streaming/embedded prompt is outside challenge coverage")
        actual = projection(request.sampling_params, request.prompt_token_ids)
        if sha(prompt.encode()) != challenge.prompt_text_sha256 or canonical(actual) != canonical(
            self.expected(challenge, actual["prompt_token_count"])
        ):
            raise ObserverError("actual API inputs differ from the locked challenge")
        value = {
            "schema_version": 1,
            "kind": "path1_api_challenge_registration",
            "phase": "before_engine_dispatch",
            "scientific_admission": False,
            "service_instance": self.trace.service_instance,
            "source_sha256": self.trace.source_sha256,
            "plan_sha256": self.plan_sha256,
            "api_pid": os.getpid(),
            "external_request_id": external,
            "internal_request_id": internal,
            "prompt_text_sha256": challenge.prompt_text_sha256,
            "actual": actual,
        }
        # Exclusive final/partial publication is intentional. No replacement, retry,
        # prefix match or modification of upstream internal IDs is permitted.
        # The API observation must exist before any worker can see registration.
        # A failed diagnostic write must not leave a newly covered challenge.
        self.trace.write("api_challenge_registered", value)
        write_exclusive_json(self.path(external), value)
        return value

    def lookup(self, internal: str) -> dict | None:
        matched = []
        for external, challenge in self.challenges.items():
            path = self.path(external)
            try:
                before = path.lstat()
            except FileNotFoundError:
                continue
            if path.is_symlink() or not path.is_file() or before.st_size > 64 * 1024:
                raise ObserverError("invalid registration file")
            raw = path.read_bytes()
            after = path.lstat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in fields):
                raise ObserverError("registration changed during read")
            value = json.loads(raw)
            expected = {
                "schema_version": 1,
                "kind": "path1_api_challenge_registration",
                "phase": "before_engine_dispatch",
                "scientific_admission": False,
                "service_instance": self.trace.service_instance,
                "source_sha256": self.trace.source_sha256,
                "plan_sha256": self.plan_sha256,
                "api_pid": value["api_pid"],
                "external_request_id": external,
                "internal_request_id": value["internal_request_id"],
                "prompt_text_sha256": challenge.prompt_text_sha256,
                "actual": self.expected(challenge, value["actual"]["prompt_token_count"]),
            }
            if (
                type(value["api_pid"]) is not int
                or value["api_pid"] <= 0
                or type(value["actual"]["prompt_token_count"]) is not int
                or not 1 <= value["actual"]["prompt_token_count"] <= 8192
                or canonical(value) != canonical(expected)
            ):
                raise ObserverError("registration identity/content differs")
            _identifier(value["internal_request_id"])
            if value["internal_request_id"] == internal:
                matched.append(value)
        if len(matched) > 1:
            raise ObserverError("ambiguous internal request ID")
        return matched[0] if matched else None

    def batch_reason(self, runner: Any, spec: Any) -> str | None:
        if spec is not None or runner.speculative_config is not None:
            return "speculative"
        if runner.vllm_config.use_v2_model_runner:
            return "v2"
        batch = runner.input_batch
        ids = list(batch.req_ids)
        if not 1 <= len(ids) <= 32 or batch.req_id_to_index != dict(
            zip(ids, range(len(ids)), strict=True)
        ):
            return "unsupported_rows"
        metadata = batch.sampling_metadata
        if metadata.all_greedy:
            return "greedy"
        if not metadata.all_random:
            return "mixed_sampling"
        for internal in ids:
            registration = self.lookup(internal)
            if registration is None:
                return "unregistered_or_mixed_batch"
            request = runner.requests[internal]
            actual = projection(request.sampling_params, request.prompt_token_ids)
            if canonical(actual) != canonical(registration["actual"]):
                return "worker_inputs_differ"
        return None


class SelectedSamplerObserver(SamplerObserver):
    def __init__(self, registry: ChallengeRegistry, pins: tuple[SourcePin, ...]):
        super().__init__(registry.trace, pins)
        self.registry = registry
        self.runtime_recorder = None
        self._runtime_pid = None

    def _runner_call(self, runner: Any, logits: Any, spec: Any) -> Any:
        try:
            reason = self.registry.batch_reason(runner, spec)
        except Exception as exc:
            reason = "registration_or_coverage_error:" + type(exc).__name__
        if reason is None:
            if self.runtime_recorder is not None and self._runtime_pid != os.getpid():
                self.runtime_recorder(runner, self.trace)
                self._runtime_pid = os.getpid()
            # Existing strict observer checks source code, generators and exact RNG
            # path; failure never manufactures sample_end or retries the original.
            return super()._runner_call(runner, logits, spec)
        self.registry.event(reason, list(getattr(runner.input_batch, "req_ids", [])))
        assert self._originals is not None
        return self._originals[0](runner, logits, spec)


def install_api(registry: ChallengeRegistry, pin: SourcePin) -> Any:
    if pin.module_name != API:
        raise ObserverError("exact AsyncLLM source pin required")
    module = pin.verify()
    original = module.AsyncLLM._add_request
    pin.verify_function(original, "AsyncLLM._add_request")

    async def observed(api, request, prompt, parent_req, index, queue):
        registration = None
        try:
            if request.external_req_id in registry.challenges:
                pin.verify_function(original, "AsyncLLM._add_request")
                registration = registry.register(request, prompt, parent_req, index)
        except Exception as exc:
            registry.event("api_registration_error:" + type(exc).__name__, [request.request_id])
        result = await original(api, request, prompt, parent_req, index, queue)
        if registration is not None:
            try:
                registry.trace.write(
                    "api_dispatch_completed",
                    {
                        "external_request_id": registration["external_request_id"],
                        "internal_request_id": registration["internal_request_id"],
                    },
                )
            except Exception as exc:
                registry.diagnostic_failure = type(exc).__name__
        return result

    registry.trace.write(
        "api_adapter_install", {"source_sha256": pin.sha256, "execution_verified": False}
    )
    module.AsyncLLM._add_request = observed
    return observed
