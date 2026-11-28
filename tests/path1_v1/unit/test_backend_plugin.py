"""CPU fixtures only: no installed vLLM, model, GPU or RuntimeAdmission proof."""

import asyncio
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import backend_plugin as plugin
from agentmemeval.path1_v1.operations.backend_challenges import (
    API,
    Challenge,
    ChallengeRegistry,
    SelectedSamplerObserver,
    canonical,
    install_api,
    sha,
)
from agentmemeval.path1_v1.operations.backend_observer import RUNNER, ObserverError
from tests.path1_v1.unit.test_backend_observer import case as original_case
from tests.path1_v1.unit.test_backend_observer import events, load_module

case = original_case

API_SOURCE = """class AsyncLLM:
    async def _add_request(self, request, prompt, parent_req, index, queue):
        self.calls.append((request, prompt, parent_req, index, queue))
        if self.failure is not None:
            raise self.failure
        return self.result
"""


@pytest.fixture
def selected(case, tmp_path, monkeypatch):
    challenges, requests = [], []
    for internal, request in case.runner.requests.items():
        request.sampling_params.max_tokens = 8
        external = "external-" + internal
        prompt = "prompt-" + internal
        challenges.append(
            Challenge(
                external,
                sha(prompt.encode()),
                sha(canonical(request.prompt_token_ids)),
                request.sampling_params.seed,
                0.8,
                0.9,
                50,
                8,
            )
        )
        requests.append(
            (
                SimpleNamespace(
                    external_req_id=external,
                    request_id=internal,
                    sampling_params=request.sampling_params,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_embeds=None,
                    resumable=False,
                ),
                prompt,
            )
        )
    directory = tmp_path / "registration"
    directory.mkdir()
    registry = ChallengeRegistry(directory, tuple(challenges), case.trace)
    observer = SelectedSamplerObserver(registry, tuple(case.pins))
    api_module, pin = load_module(monkeypatch, API, tmp_path / "api.py", API_SOURCE)
    api = api_module.AsyncLLM()
    api.calls, api.failure, api.result = [], None, object()
    case.registry, case.selected, case.requests = registry, observer, requests
    case.api, case.api_pin = api, pin
    return case


def register_all(case):
    install_api(case.registry, case.api_pin)
    for request, prompt in case.requests:
        assert asyncio.run(case.api._add_request(request, prompt, None, 0, None)) is case.api.result


def test_registered_exact_ids_have_single_original_rng_call(selected):
    register_all(selected)
    selected.selected.install()
    assert selected.runner._sample(None, None) is selected.sampler.result
    assert [item.draws for item in selected.generators.values()] == [1, 1]
    assert len(selected.sampler.calls) == 1
    assert [row["event"] for row in events(selected)].count("sample_end") == 1
    first, _ = selected.requests[0]
    assert (
        selected.registry.lookup(first.request_id)["external_request_id"] == first.external_req_id
    )
    assert selected.registry.lookup(first.request_id + "-prefix-guess") is None
    assert selected.api.calls[0][0] is first


@pytest.mark.parametrize("branch", ["greedy", "ordinary", "mixed_batch", "mixed_sampling", "spec"])
def test_uncovered_call_result_and_rng_count_remain_original(selected, branch):
    if branch == "mixed_batch":
        request, prompt = selected.requests[0]
        selected.registry.register(request, prompt, None, 0)
    elif branch == "greedy":
        selected.runner.input_batch.sampling_metadata.all_greedy = True
    elif branch == "mixed_sampling":
        selected.runner.input_batch.sampling_metadata.all_random = False
    elif branch == "spec":
        selected.runner.speculative_config = object()
    selected.selected.install()
    assert selected.runner._sample(None, None) is selected.sampler.result
    assert [item.draws for item in selected.generators.values()] == [1, 1]
    assert len(selected.sampler.calls) == 1
    assert "sample_begin" not in [row["event"] for row in events(selected)]
    assert "sample_end" not in [row["event"] for row in events(selected)]


def test_malformed_registry_cannot_upgrade_ordinary_request(selected):
    register_all(selected)
    selected.registry.path(selected.requests[0][0].external_req_id).write_bytes(b"[]")
    selected.selected.install()
    selected.runner._sample(None, None)
    assert len(selected.sampler.calls) == 1
    assert "sample_end" not in [row["event"] for row in events(selected)]


def test_noncovered_trace_failure_does_not_change_original_result(selected, monkeypatch):
    selected.selected.install()
    monkeypatch.setattr(
        selected.trace, "write", lambda *args: (_ for _ in ()).throw(OSError("trace full"))
    )
    assert selected.runner._sample(None, None) is selected.sampler.result
    assert selected.registry.diagnostic_failure == "OSError"
    assert len(selected.sampler.calls) == 1


def test_registration_trace_failure_never_publishes_coverage(selected, monkeypatch):
    install_api(selected.registry, selected.api_pin)
    monkeypatch.setattr(
        selected.trace, "write", lambda *args: (_ for _ in ()).throw(OSError("full"))
    )
    request, prompt = selected.requests[0]
    assert (
        asyncio.run(selected.api._add_request(request, prompt, None, 0, None))
        is selected.api.result
    )
    assert not selected.registry.path(request.external_req_id).exists()
    assert len(selected.api.calls) == 1


def test_greedy_original_returns_without_any_random_operation(selected, monkeypatch, tmp_path):
    source = """class GPUModelRunner:
    def _sample(self, logits, spec_decode_metadata):
        if self.input_batch.sampling_metadata.all_greedy:
            self.greedy_calls += 1
            return self.greedy_result
        return sampler.random_sample(logits, self.input_batch.sampling_metadata.generators, False)
"""
    module, pin = load_module(monkeypatch, RUNNER, tmp_path / "greedy.py", source)
    runner = module.GPUModelRunner()
    runner.__dict__.update(selected.runner.__dict__)
    runner.input_batch.sampling_metadata.all_greedy = True
    runner.input_batch.sampling_metadata.all_random = False
    runner.greedy_calls, runner.greedy_result = 0, object()
    observer = SelectedSamplerObserver(selected.registry, (pin, *selected.pins[1:]))
    observer.install()
    assert runner._sample(None, None) is runner.greedy_result
    assert runner.greedy_calls == 1
    assert not selected.sampler.calls
    assert [item.draws for item in selected.generators.values()] == [0, 0]
    assert not any(row["event"].startswith("sample_") for row in events(selected))


def test_api_mismatch_is_not_registered_and_original_receives_exact_objects(selected):
    install_api(selected.registry, selected.api_pin)
    request, prompt = selected.requests[0]
    request.sampling_params.seed += 1
    queue = object()
    assert (
        asyncio.run(selected.api._add_request(request, prompt, None, 0, queue))
        is selected.api.result
    )
    assert selected.api.calls == [(request, prompt, None, 0, queue)]
    assert not selected.registry.path(request.external_req_id).exists()


def test_api_exception_is_propagated_once_without_completion_event(selected):
    install_api(selected.registry, selected.api_pin)
    request, prompt = selected.requests[0]
    selected.api.failure = ValueError("upstream")
    with pytest.raises(ValueError, match="upstream"):
        asyncio.run(selected.api._add_request(request, prompt, None, 0, None))
    assert len(selected.api.calls) == 1
    assert "api_dispatch_completed" not in [row["event"] for row in events(selected)]


def test_duplicate_external_id_never_overwrites_original_registration(selected):
    register_all(selected)
    request, prompt = selected.requests[0]
    path = selected.registry.path(request.external_req_id)
    before = path.read_bytes()
    request.request_id += "-actual-new-upstream-id"
    asyncio.run(selected.api._add_request(request, prompt, None, 0, None))
    assert path.read_bytes() == before
    assert selected.registry.lookup(request.request_id) is None
    assert len(selected.api.calls) == 3


@pytest.mark.parametrize("change", ["seed", "max_tokens", "tokens"])
def test_worker_projection_must_match_api_registration(selected, change):
    register_all(selected)
    request = selected.runner.requests["internal-A"]
    if change == "tokens":
        request.prompt_token_ids = [99]
    else:
        setattr(request.sampling_params, change, getattr(request.sampling_params, change) + 1)
    selected.selected.install()
    selected.runner._sample(None, None)
    assert len(selected.sampler.calls) == 1
    assert "sample_end" not in [row["event"] for row in events(selected)]


def test_already_loaded_install_uses_same_checks_and_is_idempotent_callback(selected, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    deferred = plugin.DeferredObserver(selected.registry, (*selected.pins, selected.api_pin))
    deferred.start()
    assert deferred.installed == {API, RUNNER}
    installed = selected.modules[RUNNER].GPUModelRunner._sample
    deferred.after_loaded(RUNNER)
    assert selected.modules[RUNNER].GPUModelRunner._sample is installed
    with pytest.raises(ObserverError, match="already started"):
        deferred.start()


def test_early_plugin_defers_runner_exec_and_preserves_spec(selected, monkeypatch):
    source_pin = selected.pins[0]
    monkeypatch.delitem(sys.modules, RUNNER)
    original_spec = importlib.util.spec_from_file_location(RUNNER, source_pin.path)

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            return original_spec if fullname == RUNNER else None

    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    deferred = plugin.DeferredObserver(selected.registry, (*selected.pins, selected.api_pin))
    deferred.start()
    assert RUNNER not in sys.modules
    spec = deferred.find_spec(RUNNER)
    assert spec is original_spec
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, RUNNER, module)
    spec.loader.exec_module(module)
    module.sampler = selected.sampler
    assert RUNNER in deferred.installed
    assert deferred.observer._installed


def test_custom_source_loader_is_not_wrapped(selected, monkeypatch):
    class CustomLoader(importlib.machinery.SourceFileLoader):
        pass

    pin = selected.pins[0]
    loader = CustomLoader(RUNNER, str(pin.path))
    spec = importlib.util.spec_from_loader(RUNNER, loader)
    finder = SimpleNamespace(find_spec=lambda *args: spec)
    monkeypatch.setattr(sys, "meta_path", [finder])
    deferred = plugin.DeferredObserver(selected.registry, (*selected.pins, selected.api_pin))
    assert deferred.find_spec(RUNNER) is spec
    assert spec.loader is loader
    assert deferred.installed == set()


def test_source_pin_failure_does_not_replace_runner_or_claim_install(selected, monkeypatch):
    original = selected.modules[RUNNER].GPUModelRunner._sample
    selected.pins[0].path.write_bytes(selected.pins[0].path.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    deferred = plugin.DeferredObserver(selected.registry, (*selected.pins, selected.api_pin))
    deferred.start()
    assert RUNNER not in deferred.installed
    assert selected.modules[RUNNER].GPUModelRunner._sample is original


def test_real_spawn_entrypoint_discovery_is_disabled_without_config(tmp_path):
    overlay = tmp_path / "overlay"
    hashes = plugin.create_overlay(overlay)
    assert len(hashes) == 2
    with pytest.raises(FileExistsError):
        plugin.create_overlay(overlay)
    source_root = Path(plugin.__file__).resolve().parents[3]
    code = """import importlib.metadata,json,os,sys
points=[p for p in importlib.metadata.entry_points(group="vllm.general_plugins")
        if p.name=="path1_seeded_observer"]
assert len(points)==1
assert points[0].load()() is None
assert not any(n=="vllm" or n.startswith("vllm.") or n=="torch" for n in sys.modules)
print(json.dumps({"pid":os.getpid(),"disabled":True,"gpu_execution":False}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(overlay), str(source_root)))
    environment.pop("PATH1_BACKEND_PLUGIN_CONFIG", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["pid"] != os.getpid()


def configured(selected, tmp_path):
    config = {
        "schema_version": 1,
        "service_instance": selected.trace.service_instance,
        "source_sha256": selected.trace.source_sha256,
        "trace_directory": str(selected.trace.directory),
        "registration_directory": str(selected.registry.directory),
        "challenges": [asdict(item) for item in selected.registry.challenges.values()],
        "source_pins": [
            {"module_name": pin.module_name, "path": str(pin.path), "sha256": pin.sha256}
            for pin in (*selected.pins, selected.api_pin)
        ],
    }
    path = tmp_path / "config.json"
    path.write_bytes(canonical(config))
    return path, sha(path.read_bytes())


def test_real_spawn_enabled_plugin_defers_all_gpu_imports(selected, tmp_path):
    overlay = tmp_path / "overlay"
    plugin.create_overlay(overlay)
    path, digest = configured(selected, tmp_path)
    code = """import importlib.metadata,json,os,sys
points=[p for p in importlib.metadata.entry_points(group="vllm.general_plugins")
        if p.name=="path1_seeded_observer"]
activate=points[0].load()
observer=activate()
assert observer is activate()
assert observer.installed==set()
assert observer in sys.meta_path
assert not any(n=="vllm" or n.startswith("vllm.") or n=="torch" for n in sys.modules)
print(json.dumps({"pid":os.getpid(),"deferred":True,"gpu_execution":False}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(overlay), str(Path(plugin.__file__).parents[3]))),
            "PATH1_BACKEND_PLUGIN_CONFIG": str(path),
            "PATH1_BACKEND_PLUGIN_CONFIG_SHA256": digest,
            "VLLM_PLUGINS": plugin.PLUGIN_NAME,
        }
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["pid"] != os.getpid()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="real Linux fork boundary")
def test_real_fork_inherits_installed_wrappers_without_double_draw(selected, tmp_path, monkeypatch):
    path, digest = configured(selected, tmp_path)
    monkeypatch.setenv("PATH1_BACKEND_PLUGIN_CONFIG", str(path))
    monkeypatch.setenv("PATH1_BACKEND_PLUGIN_CONFIG_SHA256", digest)
    monkeypatch.setenv("VLLM_PLUGINS", plugin.PLUGIN_NAME)
    monkeypatch.setattr(plugin, "_active", None)
    monkeypatch.setattr(plugin, "_configuration_sha256", None)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    deferred = plugin.activate()
    for request, prompt in selected.requests:
        asyncio.run(selected.api._add_request(request, prompt, None, 0, None))
    original_wrapper = selected.modules[RUNNER].GPUModelRunner._sample
    parent_path = deferred.registry.trace.path
    parent_bytes = parent_path.read_bytes()
    pid = os.fork()
    if pid == 0:
        try:
            assert plugin.activate() is deferred
            assert selected.modules[RUNNER].GPUModelRunner._sample is original_wrapper
            assert selected.runner._sample(None, None) is selected.sampler.result
            assert [item.draws for item in selected.generators.values()] == [1, 1]
            assert len(selected.sampler.calls) == 1
            assert deferred.registry.trace.path != parent_path
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert status == 0
    assert parent_path.read_bytes() == parent_bytes
    assert [item.draws for item in selected.generators.values()] == [0, 0]
    rows = [
        json.loads(line)
        for file in selected.trace.directory.glob("*.jsonl")
        for line in file.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["pid"] for row in rows if row["event"] == "sample_end"] == [pid]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="real Linux fork boundary")
def test_real_fork_coverage_trace_is_independent(selected):
    selected.registry.event("ordinary", ["same-request"])
    parent_path = selected.trace.path
    before = parent_path.read_bytes()
    pid = os.fork()
    if pid == 0:
        try:
            selected.registry.event("ordinary", ["same-request"])
            assert selected.trace.path != parent_path
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert status == 0
    assert parent_path.read_bytes() == before
    rows = [
        json.loads(line)
        for path in selected.trace.directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {row["pid"] for row in rows} == {os.getpid(), pid}
    assert len({row["process_nonce"] for row in rows}) == 2


def test_plugin_bad_configuration_never_loads_vllm(tmp_path, monkeypatch):
    path = tmp_path / "plugin.json"
    path.write_bytes(b"{}")
    monkeypatch.setenv("PATH1_BACKEND_PLUGIN_CONFIG", str(path))
    monkeypatch.setenv("PATH1_BACKEND_PLUGIN_CONFIG_SHA256", hashlib.sha256(b"{}").hexdigest())
    monkeypatch.setenv("VLLM_PLUGINS", plugin.PLUGIN_NAME)
    monkeypatch.setattr(plugin, "_active", None)
    with pytest.raises(ObserverError, match="schema"):
        plugin.activate()
