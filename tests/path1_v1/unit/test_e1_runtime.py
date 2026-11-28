"""CPU-only evidence/capability failures. No test substitutes for actual E0/E1 GPU proof."""

import copy
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import e1_backend_runtime as backend
from agentmemeval.path1_v1.operations import e1_runtime as runtime
from agentmemeval.path1_v1.operations.backend_challenges import Challenge
from agentmemeval.path1_v1.operations.backend_observer import DEPENDENCIES, RUNNER, SAMPLER
from agentmemeval.path1_v1.operations.worker import RuntimeAdmissionRequired


@pytest.fixture
def chain(tmp_path, monkeypatch):
    challenge = Challenge("chatcmpl-test", "a" * 64, "b" * 64, 2**62 + 4, 1.0, 1.0, -1, 8)
    pins = [
        {"module_name": name, "path": str(tmp_path / "pin.py"), "sha256": "c" * 64}
        for name in ("vllm.v1.engine.async_llm", RUNNER, SAMPLER, *DEPENDENCIES)
    ]
    identity = {"pid": 10}
    monkeypatch.setattr(backend, "verify_process_member", lambda owner, pid: identity)
    engine = {
        "identity": identity,
        "model_path": str(tmp_path),
        "tokenizer_path": str(tmp_path),
        "dtype": "torch.bfloat16",
        "parameter_dtype": "torch.bfloat16",
        "parameter_device": "cuda:0",
        "max_model_len": 65536,
        "max_num_seqs": 1,
        "prefix_caching": False,
        "tensor_parallel_size": 1,
        "quantization": None,
        "vllm_version": "0.23.0",
        "torch_version": "2.11.0+cu129",
        "transformers_version": "5.16.1",
    }
    actual = {
        key: value
        for key, value in asdict(challenge).items()
        if key not in {"external_request_id", "prompt_text_sha256"}
    }
    actual.update(n=1, prompt_token_count=4)
    payloads = [
        ("adapter_install", {"pins": {x["module_name"]: x["sha256"] for x in pins[1:]}}),
        (
            "api_challenge_registered",
            {
                "external_request_id": challenge.external_request_id,
                "internal_request_id": "internal1",
                "prompt_text_sha256": challenge.prompt_text_sha256,
                "actual": actual,
            },
        ),
        ("engine_runtime", engine),
        (
            "sample_begin",
            {
                "rows": [
                    {
                        "internal_request_id": "internal1",
                        "seed": challenge.seed,
                        "prompt_token_ids_sha256": challenge.prompt_token_ids_sha256,
                        "generator_before": {
                            "initial_seed": challenge.seed,
                            "state_sha256": "d" * 64,
                        },
                    }
                ]
            },
        ),
        (
            "exponential_noise_before_consumption",
            {
                "rows": ["internal1"],
                "q_before_original": {"device": "cuda:0", "sampled_values_finite": True},
            },
        ),
        (
            "sample_end",
            {
                "rows": [
                    {
                        "internal_request_id": "internal1",
                        "sampled_token_id": 7,
                        "generator_after": {
                            "initial_seed": challenge.seed,
                            "state_sha256": "e" * 64,
                        },
                    }
                ]
            },
        ),
    ]
    events = [
        {"pid": 10, "process_nonce": "nonce", "sequence": i, "event": event, "payload": payload}
        for i, (event, payload) in enumerate(payloads)
    ]
    for row in events:
        row["sequence"] += 1
    events.append(
        {
            "pid": 10,
            "process_nonce": "nonce",
            "sequence": 0,
            "event": "api_adapter_install",
            "payload": {"source_sha256": "c" * 64},
        }
    )
    events.append(
        {
            "pid": 10,
            "process_nonce": "nonce",
            "sequence": 7,
            "event": "api_dispatch_completed",
            "payload": {
                "external_request_id": challenge.external_request_id,
                "internal_request_id": "internal1",
            },
        }
    )
    return SimpleNamespace(events=events, challenge=challenge, pins=pins, root=tmp_path)


def verify(case):
    return backend.verify_challenge_events(
        case.events,
        (case.challenge,),
        owner=object(),
        model_root=case.root,
        source_pins=case.pins,
        expected_versions={"vllm": "0.23.0", "torch": "2.11.0+cu129", "transformers": "5.16.1"},
    )


def test_exact_complete_challenge_chain_is_technical_only(chain):
    result = verify(chain)
    assert result["scientific_admission"] is False
    assert result["challenges"][0]["seed"] == 2**62 + 4
    assert result["challenges"][0]["sampled_token_id"] == 7


@pytest.mark.parametrize(
    "event",
    [
        "adapter_install",
        "api_challenge_registered",
        "engine_runtime",
        "sample_begin",
        "exponential_noise_before_consumption",
        "sample_end",
        "api_adapter_install",
        "api_dispatch_completed",
    ],
)
def test_missing_chain_link_cannot_issue_backend_proof(chain, event):
    chain.events = [row for row in chain.events if row["event"] != event]
    with pytest.raises(backend.ObserverError):
        verify(chain)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_path", "/different"),
        ("dtype", "torch.float16"),
        ("parameter_device", "cpu"),
        ("prefix_caching", True),
        ("max_model_len", 32768),
        ("max_num_seqs", 2),
        ("tensor_parallel_size", 2),
        ("quantization", "int8"),
        ("vllm_version", "0.24.0"),
    ],
)
def test_actual_engine_mismatch_rejected(chain, key, value):
    chain.events[2]["payload"][key] = value
    with pytest.raises(backend.ObserverError):
        verify(chain)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_seed", "unchanged_rng", "cpu_q", "duplicate_api", "wrong_source", "changed_tokenizer"],
)
def test_critical_execution_mutations_rejected(chain, mutation):
    if mutation == "wrong_seed":
        chain.events[3]["payload"]["rows"][0]["generator_before"]["initial_seed"] = 1
    elif mutation == "unchanged_rng":
        chain.events[5]["payload"]["rows"][0]["generator_after"]["state_sha256"] = "d" * 64
    elif mutation == "cpu_q":
        chain.events[4]["payload"]["q_before_original"]["device"] = "cpu"
    elif mutation == "duplicate_api":
        chain.events.append(copy.deepcopy(chain.events[1]))
    elif mutation == "wrong_source":
        chain.events[0]["payload"]["pins"][RUNNER] = "f" * 64
    else:
        chain.events[1]["payload"]["actual"]["prompt_token_ids_sha256"] = "f" * 64
    with pytest.raises(backend.ObserverError):
        verify(chain)


def test_process_replacement_rejected(chain, monkeypatch):
    monkeypatch.setattr(backend, "verify_process_member", lambda owner, pid: {"pid": 11})
    with pytest.raises(backend.ObserverError, match="identity changed"):
        verify(chain)


@pytest.mark.parametrize("bad", ["sequence", "service", "source", "partial"])
def test_trace_stream_binding_and_partial_rejection(tmp_path, bad):
    row = {
        "kind": "backend_observation",
        "sequence": 0,
        "pid": 10,
        "process_nonce": "a",
        "service_instance": "service",
        "source_sha256": "b" * 64,
    }
    if bad == "sequence":
        row["sequence"] = 1
    elif bad == "service":
        row["service_instance"] = "other"
    elif bad == "source":
        row["source_sha256"] = "c" * 64
    (tmp_path / "sampler-10-a.jsonl").write_text(
        json.dumps(row) + ("" if bad == "partial" else "\n"), encoding="utf-8"
    )
    with pytest.raises(backend.ObserverError):
        backend.read_traces(tmp_path, service_instance="service", source_sha256="b" * 64)


def test_live_e1_requires_version2_bound_plan_before_any_service_side_effect():
    with pytest.raises(RuntimeAdmissionRequired, match="version-2"):
        runtime.run_e1_service(
            SimpleNamespace(worker=object(), runtime_plan=None, descriptor={"schema_version": 1})
        )


def test_verifier_cannot_be_constructed_for_science_phases():
    for phase in ("E0", "E2", "E3"):
        with pytest.raises(RuntimeAdmissionRequired, match="technical E1"):
            runtime.E1RuntimeVerifier(
                owner=None,
                request=SimpleNamespace(contract={"phase": phase}),
                endpoint="http://127.0.0.1:8000/v1",
                directory=Path("/unused"),
                tokenizer=None,
                challenges=(),
                challenge_messages=[],
                plan={},
                host={},
            )


def test_live_capability_cannot_transfer_or_survive_owner_change():
    old = SimpleNamespace(identity="original", binding="binding", cmdline=("server",))
    owner = SimpleNamespace(verify_live=lambda: old, verify_endpoint=lambda endpoint: None)
    request = SimpleNamespace(contract={"phase": "E1"}, intent_id="intent", request_sha256="sha")
    verifier = runtime.E1RuntimeVerifier(
        owner=owner,
        request=request,
        endpoint="http://localhost",
        directory=Path("/unused"),
        tokenizer=None,
        challenges=(),
        challenge_messages=[],
        plan={"cleanup_seconds": 600, "source_pins": []},
        host={"allocation_end_utc": (datetime.now(UTC) + timedelta(hours=2)).isoformat()},
    )
    token = object()
    verifier._admission = token
    verifier.verify_live(request, token)
    with pytest.raises(RuntimeAdmissionRequired, match="another worker"):
        verifier.verify_live(request, object())
    owner.verify_live = lambda: SimpleNamespace(
        identity="replacement", binding="binding", cmdline=("server",)
    )
    with pytest.raises(RuntimeAdmissionRequired, match="identity changed"):
        verifier.verify_live(request, token)


def test_deadline_preserves_request_and_cleanup_budget():
    request = SimpleNamespace(contract={"phase": "E1"}, intent_id="i", request_sha256="s")
    owner = SimpleNamespace(verify_live=lambda: None)
    verifier = runtime.E1RuntimeVerifier(
        owner=owner,
        request=request,
        endpoint="http://localhost",
        directory=Path("/unused"),
        tokenizer=None,
        challenges=(),
        challenge_messages=[],
        plan={"cleanup_seconds": 600},
        host={"allocation_end_utc": (datetime.now(UTC) + timedelta(seconds=900)).isoformat()},
    )
    with pytest.raises(RuntimeAdmissionRequired, match="cleanup time"):
        verifier._live()


@pytest.mark.parametrize("raw", [b"GPU-a, 10\n", b"GPU-a, 10\nGPU-z, 99\n"])
def test_gpu_mapping_is_owned_pid_specific(monkeypatch, raw):
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=raw, stderr=b""),
    )
    result = backend.observe_engine_gpus({10}, [{"uuid": "GPU-a"}])
    assert result["engine_gpu_pairs"] == [{"gpu_uuid": "GPU-a", "pid": 10}]


@pytest.mark.parametrize(
    "raw", [b"", b"GPU-a, 11\n", b"GPU-b, 10\n", b"GPU-a, 10\nGPU-b, 10\n", b"GPU-a, invalid\n"]
)
def test_missing_ambiguous_wrong_gpu_mapping_rejected(monkeypatch, raw):
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=raw, stderr=b""),
    )
    with pytest.raises(backend.ObserverError):
        backend.observe_engine_gpus({10}, [{"uuid": "GPU-a"}])


def test_cuda_local_version_must_match_exact_observed_lock(chain):
    chain.events[2]["payload"]["vllm_version"] = "0.23.0+cu129"
    chain.events[2]["payload"]["module_versions"] = {"vllm": "0.23.0"}
    with pytest.raises(backend.ObserverError):
        verify(chain)
    result = backend.verify_challenge_events(
        chain.events,
        (chain.challenge,),
        owner=object(),
        model_root=chain.root,
        source_pins=chain.pins,
        expected_versions={
            "vllm": "0.23.0+cu129",
            "torch": "2.11.0+cu129",
            "transformers": "5.16.1",
        },
    )
    assert result["challenges"][0]["engine"]["vllm_version"] == "0.23.0+cu129"
    assert result["challenges"][0]["engine"]["module_versions"]["vllm"] == "0.23.0"


def test_engine_producer_preserves_distribution_and_module_versions_separately(monkeypatch):
    @dataclass
    class Identity:
        pid: int

    device = SimpleNamespace(type="cuda")
    parameter = SimpleNamespace(device=device, dtype="torch.bfloat16")
    torch = SimpleNamespace(
        __version__="2.11.0+cu129",
        version=SimpleNamespace(cuda="12.9"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="fixture", total_memory=1),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.23.0"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.16.1"))
    versions = {"vllm": "0.23.0+cu129", "torch": "2.11.0+cu129", "transformers": "5.16.1"}
    monkeypatch.setattr(backend.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(backend, "_kernel", lambda _: (Identity(10), None, "R", None))
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/model", tokenizer="/model", dtype="torch.bfloat16",
            max_model_len=65536, quantization=None,
        ),
        model=SimpleNamespace(parameters=lambda: iter([parameter])),
        device=device,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    events = []
    backend.engine_observation(runner, SimpleNamespace(write=lambda *row: events.append(row)))
    assert events[0][0] == "engine_runtime"
    assert events[0][1]["vllm_version"] == "0.23.0+cu129"
    assert events[0][1]["module_versions"]["vllm"] == "0.23.0"


def test_api_leader_surviving_does_not_allow_engine_replacement(monkeypatch):
    snapshot = SimpleNamespace(identity="leader", binding="binding", cmdline=("server",))
    owner = SimpleNamespace(verify_live=lambda: snapshot, verify_endpoint=lambda *_: None)
    request = SimpleNamespace(contract={"phase": "E1"}, intent_id="i", request_sha256="s")
    verifier = runtime.E1RuntimeVerifier(
        owner=owner,
        request=request,
        endpoint="http://localhost",
        directory=Path("/unused"),
        tokenizer=None,
        challenges=(),
        challenge_messages=[],
        plan={"cleanup_seconds": 600, "source_pins": []},
        host={"allocation_end_utc": (datetime.now(UTC) + timedelta(hours=2)).isoformat()},
    )
    verifier._backend_identities = ({"pid": 100, "start_marker": "original"},)
    monkeypatch.setattr(
        runtime, "verify_process_member", lambda *_: {"pid": 100, "start_marker": "original"}
    )
    verifier._live()
    monkeypatch.setattr(
        runtime, "verify_process_member", lambda *_: {"pid": 100, "start_marker": "replacement"}
    )
    with pytest.raises(RuntimeAdmissionRequired, match="engine/API process identity changed"):
        verifier._live()
