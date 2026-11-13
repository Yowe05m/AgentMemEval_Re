"""Explicit, source-pinned observations of the vLLM 0.23 V1 seeded sampler.

This module imports neither vLLM nor torch, installs no plugin automatically,
and issues no RuntimeAdmission. An owner must install it after the selected
modules have loaded. Offline adapter tests are not backend/GPU validation.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNNER = "vllm.v1.worker.gpu_model_runner"
SAMPLER = "vllm.v1.sample.ops.topk_topp_sampler"
DEPENDENCIES = ("vllm.v1.sample.sampler", "vllm.v1.worker.gpu_input_batch")


class ObserverError(RuntimeError):
    """The observation is incomplete; no execution evidence is certified."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _code_shape(code: types.CodeType) -> tuple:
    """Compare executing code with compilation of its pinned source bytes."""
    constants = tuple(
        _code_shape(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts
    )
    return (
        code.co_code,
        constants,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_flags,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_firstlineno,
    )


@dataclass(frozen=True)
class SourcePin:
    module_name: str
    path: Path
    sha256: str

    def verify(self) -> types.ModuleType:
        if not self.path.is_absolute() or self.path.is_symlink():
            raise ObserverError("source pin must identify an absolute regular source")
        if self.path.resolve() != self.path or not self.path.is_file():
            raise ObserverError("source pin has a symlink component or missing file")
        module = sys.modules.get(self.module_name)
        if not isinstance(module, types.ModuleType):
            raise ObserverError(f"required module is not loaded: {self.module_name}")
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if getattr(module, "__file__", None) != str(self.path) or origin != str(self.path):
            raise ObserverError("loaded module does not come from the pinned source")
        if _sha(self.path.read_bytes()) != self.sha256:
            raise ObserverError("source bytes changed")
        return module

    def verify_function(self, function: Any, qualified_name: str) -> None:
        module = self.verify()
        if (
            not inspect.isfunction(function)
            or function.__globals__ is not vars(module)
            or function.__module__ != self.module_name
            or function.__code__.co_filename != str(self.path)
            or function.__closure__
        ):
            raise ObserverError("function is not the pinned module's ordinary Python function")
        compiled = compile(self.path.read_bytes(), str(self.path), "exec", dont_inherit=True)
        for part in qualified_name.split("."):
            matches = [
                value
                for value in compiled.co_consts
                if isinstance(value, types.CodeType) and value.co_name == part
            ]
            if len(matches) != 1:
                raise ObserverError("pinned function is absent or ambiguous")
            compiled = matches[0]
        if _code_shape(function.__code__) != _code_shape(compiled):
            raise ObserverError("executing function code differs from pinned source")


class ProcessTrace:
    """Exclusive per-PID JSONL. Fork creates a new stream, never reuses an fd.

    Events are diagnostics, not authority for admission, submission or cleanup.
    A failed write permanently poisons this PID's stream.
    """

    def __init__(self, directory: Path, *, service_instance: str, source_sha256: str):
        directory = Path(directory)
        if not directory.is_absolute() or directory.resolve() != directory:
            raise ObserverError("trace directory must be an absolute real path")
        if not directory.is_dir() or directory.is_symlink():
            raise ObserverError("trace directory must already exist")
        if not service_instance or len(service_instance) > 256:
            raise ObserverError("service instance is required")
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise ObserverError("source identity must be a SHA-256")
        self.directory = directory
        self.service_instance = service_instance
        self.source_sha256 = source_sha256
        self._pid = os.getpid()
        self._sequence = 0
        self._failed = False
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._file_identity: tuple[int, int] | None = None
        self._bytes_written = 0
        self._nonce = uuid.uuid4().hex
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self) -> None:
        self._pid = os.getpid()
        self._sequence = 0
        self._failed = False
        self._lock = threading.Lock()
        self._path = None
        self._file_identity = None
        self._bytes_written = 0
        self._nonce = uuid.uuid4().hex

    @property
    def path(self) -> Path | None:
        return self._path

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if self._pid != os.getpid():
            self._after_fork()
        with self._lock:
            if self._failed:
                raise ObserverError("trace is poisoned by an earlier write failure")
            try:
                record = {
                    "schema_version": 1,
                    "kind": "backend_observation",
                    "scientific_admission": False,
                    "pid": self._pid,
                    "process_nonce": self._nonce,
                    "sequence": self._sequence,
                    "service_instance": self.service_instance,
                    "source_sha256": self.source_sha256,
                    "event": event,
                    "payload": payload,
                }
                data = _json(record) + b"\n"
                if len(data) > 256 * 1024:
                    raise ObserverError("event exceeds bounded trace size")
                flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                if self._path is None:
                    self._path = self.directory / f"sampler-{self._pid}-{self._nonce}.jsonl"
                    flags |= os.O_CREAT | os.O_EXCL
                else:
                    flags |= os.O_APPEND
                fd = os.open(self._path, flags, 0o600)
                try:
                    with os.fdopen(fd, "ab", closefd=True) as stream:
                        info = os.fstat(stream.fileno())
                        identity = (info.st_dev, info.st_ino)
                        if self._file_identity is None:
                            self._file_identity = identity
                        if identity != self._file_identity or info.st_size != self._bytes_written:
                            raise ObserverError("trace file was replaced or concurrently changed")
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    # fdopen owns fd after construction; no fallback output file.
                    raise
                self._sequence += 1
                self._bytes_written += len(data)
            except BaseException:
                self._failed = True
                raise


def _generator_state(generator: Any) -> dict[str, Any]:
    state = generator.get_state()
    # get_state is observational; no seed/reset or extra random operation.
    values = state.cpu().tolist()
    return {"initial_seed": generator.initial_seed(), "state_sha256": _sha(bytes(values))}


def _tensor_sample(tensor: Any, rows: int, *, limit: int = 8) -> dict[str, Any]:
    shape = tuple(tensor.shape)
    if len(shape) != 2 or shape[0] != rows or shape[1] < 1:
        raise ObserverError("unexpected sampler tensor shape")
    columns = list(range(min(limit, shape[1])))
    # Small deterministic prefix only. No RNG, mutation, full tensor hash or RAM scan.
    values = tensor.detach()[:, : len(columns)].cpu().tolist()
    finite = all(math.isfinite(value) for row in values for value in row)
    return {
        "shape": list(shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "columns": columns,
        "values": values if finite else None,
        "sampled_values_finite": finite,
        "scope": "bounded_prefix_not_full_tensor",
    }


@dataclass
class _Frame:
    pid: int
    rows: list[dict[str, Any]]
    generators: dict[int, Any]
    random_calls: int = 0
    consume_calls: int = 0
    in_random: bool = False


class SamplerObserver:
    """V1/non-speculative/all-seeded, non-greedy branch only.

    Installation does not establish a functioning general plugin, API-to-engine
    linkage, loaded weights, GPU hardware identity, or scientific admission.
    """

    def __init__(self, trace: ProcessTrace, pins: tuple[SourcePin, ...]):
        self.trace = trace
        self.pins = {pin.module_name: pin for pin in pins}
        if len(self.pins) != len(pins) or set(self.pins) != {RUNNER, SAMPLER, *DEPENDENCIES}:
            raise ObserverError("exact sampler source dependency pins are required")
        self._frame: contextvars.ContextVar[_Frame | None] = contextvars.ContextVar(
            f"sampler_observation_{id(self)}", default=None
        )
        self._installed = False
        self._originals: tuple[Any, Any, Any] | None = None

    def install(self) -> None:
        if self._installed:
            raise ObserverError("observer already installed")
        modules = {name: pin.verify() for name, pin in self.pins.items()}
        runner_class = modules[RUNNER].GPUModelRunner
        sampler = modules[SAMPLER]
        originals = (
            runner_class._sample,
            sampler.random_sample,
            sampler.sample_with_exponential_noise,
        )
        self.pins[RUNNER].verify_function(originals[0], "GPUModelRunner._sample")
        self.pins[SAMPLER].verify_function(originals[1], "random_sample")
        self.pins[SAMPLER].verify_function(originals[2], "sample_with_exponential_noise")
        self._originals = originals
        observer = self

        def runner_wrapper(runner, logits, spec_decode_metadata):
            return observer._runner_call(runner, logits, spec_decode_metadata)

        def random_wrapper(probs, generators, use_fp64_gumbel=False):
            return observer._random_call(probs, generators, use_fp64_gumbel)

        def consume_wrapper(probs, q):
            return observer._consume_call(probs, q)

        # All checks and durable readiness output precede any mutation.
        self.trace.write(
            "adapter_install",
            {
                "adapter_target_version": "0.23.0",
                "runner": "V1",
                "pins": {name: pin.sha256 for name, pin in self.pins.items()},
                "gpu_execution_verified": False,
            },
        )
        runner_class._sample = runner_wrapper
        sampler.random_sample = random_wrapper
        sampler.sample_with_exponential_noise = consume_wrapper
        self._wrappers = (runner_wrapper, random_wrapper, consume_wrapper)
        self._installed = True

    def _verify(self) -> None:
        for pin in self.pins.values():
            pin.verify()
        runner = sys.modules[RUNNER].GPUModelRunner
        sampler = sys.modules[SAMPLER]
        actual = (runner._sample, sampler.random_sample, sampler.sample_with_exponential_noise)
        if actual != self._wrappers:
            raise ObserverError("installed observer function was replaced")

    def _runner_call(self, runner: Any, logits: Any, spec: Any) -> Any:
        self._verify()
        if self._frame.get() is not None:
            raise ObserverError("nested runner sample is unsupported")
        if spec is not None or runner.speculative_config is not None:
            raise ObserverError("speculative sampling is outside this adapter")
        if runner.vllm_config.use_v2_model_runner:
            raise ObserverError("V2 runner is outside this adapter")
        batch = runner.input_batch
        ids = list(batch.req_ids)
        if not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids):
            raise ObserverError("invalid or oversized request batch")
        if batch.req_id_to_index != {req_id: index for index, req_id in enumerate(ids)}:
            raise ObserverError("request rows do not match the current batch")
        metadata = batch.sampling_metadata
        generators = dict(metadata.generators)
        if (
            not metadata.all_random
            or metadata.all_greedy
            or set(generators) != set(range(len(ids)))
        ):
            raise ObserverError("only fully seeded random batches are supported")
        rows = []
        for index, req_id in enumerate(ids):
            request = runner.requests[req_id]
            params = request.sampling_params
            seed = params.seed
            if (
                type(seed) is not int
                or not 0 <= seed < 2**63
                or params.n != 1
                or params.temperature <= 0
                or request.generator is not generators[index]
                or generators[index].initial_seed() != seed
            ):
                raise ObserverError("request/generator/seed binding mismatch")
            token_ids = request.prompt_token_ids
            if not isinstance(token_ids, list) or any(
                type(x) is not int or x < 0 for x in token_ids
            ):
                raise ObserverError("only actual text prompt token IDs are supported")
            rows.append(
                {
                    "row": index,
                    "internal_request_id": req_id,
                    "seed": seed,
                    "prompt_token_count": len(token_ids),
                    "prompt_token_ids_sha256": _sha(_json(token_ids)),
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "top_k": params.top_k,
                    "generator_before": _generator_state(generators[index]),
                }
            )
        frame = _Frame(os.getpid(), rows, generators)
        token = self._frame.set(frame)
        try:
            self.trace.write("sample_begin", {"rows": rows})
            assert self._originals is not None
            result = self._originals[0](runner, logits, spec)
            if frame.random_calls != 1 or frame.consume_calls != 1:
                raise ObserverError("actual sampler bypassed the observed random/q branch")
            sampled = result.sampled_token_ids.detach().cpu().tolist()
            if len(sampled) != len(rows) or any(
                len(row) != 1 or type(row[0]) is not int or row[0] < 0 for row in sampled
            ):
                raise ObserverError("unexpected sampled token IDs shape or dtype")
            self.trace.write(
                "sample_end",
                {
                    "rows": [
                        {
                            "internal_request_id": row["internal_request_id"],
                            "sampled_token_id": sampled[i][0],
                            "generator_after": _generator_state(generators[i]),
                        }
                        for i, row in enumerate(rows)
                    ],
                },
            )
            return result
        finally:
            self._frame.reset(token)

    def _current(self) -> _Frame | None:
        frame = self._frame.get()
        if frame is not None and frame.pid != os.getpid():
            raise ObserverError("fork inherited an active sampler frame")
        return frame

    def _random_call(self, probs: Any, generators: Any, fp64: bool) -> Any:
        frame = self._current()
        assert self._originals is not None
        if frame is None:
            # Warmup calls do not have a request mapping and are not claimed.
            return self._originals[1](probs, generators, fp64)
        if (
            frame.in_random
            or generators.keys() != frame.generators.keys()
            or any(generators[key] is not value for key, value in frame.generators.items())
        ):
            raise ObserverError("sampler replaced request generators")
        frame.random_calls += 1
        frame.in_random = True
        try:
            return self._originals[1](probs, generators, fp64)
        finally:
            frame.in_random = False

    def _consume_call(self, probs: Any, q: Any) -> Any:
        frame = self._current()
        assert self._originals is not None
        if frame is not None:
            if not frame.in_random:
                raise ObserverError("q consumption is outside the observed random_sample")
            frame.consume_calls += 1
            self.trace.write(
                "exponential_noise_before_consumption",
                {
                    "rows": [row["internal_request_id"] for row in frame.rows],
                    "q_before_original": _tensor_sample(q, len(frame.rows)),
                },
            )
        # The original consumes exactly the same q and probs objects, once.
        return self._originals[2](probs, q)

    def uninstall(self) -> None:
        if not self._installed or self._frame.get() is not None:
            raise ObserverError("cannot uninstall an absent or active observer")
        self._verify()
        assert self._originals is not None
        runner = sys.modules[RUNNER].GPUModelRunner
        sampler = sys.modules[SAMPLER]
        runner._sample, sampler.random_sample, sampler.sample_with_exponential_noise = (
            self._originals
        )
        self._installed = False
