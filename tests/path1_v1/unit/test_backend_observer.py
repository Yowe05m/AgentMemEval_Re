"""Offline tests of adapter binding; these do not execute vLLM or a GPU."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations.backend_observer import (
    DEPENDENCIES,
    RUNNER,
    SAMPLER,
    ObserverError,
    ProcessTrace,
    SamplerObserver,
    SourcePin,
)

RUNNER_SOURCE = """class GPUModelRunner:
    def _sample(self, logits, spec_decode_metadata):
        return sampler.random_sample(logits, self.input_batch.sampling_metadata.generators, False)
"""
SAMPLER_SOURCE = """def random_sample(probs, generators, use_fp64_gumbel=False):
    for generator in generators.values():
        generator.draws += 1
    q = make_q(len(generators))
    return sample_with_exponential_noise(probs, q)

def sample_with_exponential_noise(probs, q):
    calls.append((probs, q))
    q.values[0][0] = -99.0
    return result
"""


class Tensor:
    dtype = "torch.float32"
    device = "cuda:0"

    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [list(row) for row in self.values]

    def __getitem__(self, indices):
        rows, columns = indices
        return Tensor([row[columns] for row in self.values[rows]])


class Generator:
    def __init__(self, seed):
        self.seed = seed
        self.draws = 0

    def initial_seed(self):
        return self.seed

    def get_state(self):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [self.draws]))


def load_module(monkeypatch, name, path, source):
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    # Compile exact fixture bytes without writing a pycache.
    exec(compile(path.read_bytes(), str(path), "exec"), vars(module))
    return module, SourcePin(name, path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def case(tmp_path, monkeypatch):
    modules = {}
    pins = []
    for index, name in enumerate((RUNNER, SAMPLER, *DEPENDENCIES)):
        source = RUNNER_SOURCE if name == RUNNER else SAMPLER_SOURCE if name == SAMPLER else ""
        modules[name], pin = load_module(monkeypatch, name, tmp_path / f"module{index}.py", source)
        pins.append(pin)
    sampler = modules[SAMPLER]
    sampler.calls = []
    sampler.result = SimpleNamespace(sampled_token_ids=Tensor([[2], [1]]))
    sampler.make_q = lambda rows: Tensor([[0.25, 0.75, 1.25] for _ in range(rows)])
    modules[RUNNER].sampler = sampler
    runner = modules[RUNNER].GPUModelRunner()
    runner.speculative_config = None
    runner.vllm_config = SimpleNamespace(use_v2_model_runner=False)
    generators = {0: Generator(2**62 + 13), 1: Generator(19)}
    runner.input_batch = SimpleNamespace(
        req_ids=["internal-A", "internal-B"],
        req_id_to_index={"internal-A": 0, "internal-B": 1},
        sampling_metadata=SimpleNamespace(generators=generators, all_random=True, all_greedy=False),
    )
    runner.requests = {
        req_id: SimpleNamespace(
            generator=generators[index],
            prompt_token_ids=[1, 2, index + 3],
            sampling_params=SimpleNamespace(
                seed=generators[index].seed,
                n=1,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
            ),
        )
        for index, req_id in enumerate(runner.input_batch.req_ids)
    }
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trace = ProcessTrace(trace_dir, service_instance="offline-fixture", source_sha256="a" * 64)
    observer = SamplerObserver(trace, tuple(pins))
    return SimpleNamespace(
        modules=modules,
        pins=pins,
        runner=runner,
        sampler=sampler,
        observer=observer,
        trace=trace,
        generators=generators,
    )


def events(case):
    return [json.loads(line) for line in case.trace.path.read_text(encoding="utf-8").splitlines()]


def test_q_is_observed_before_original_mutation_without_extra_draw_or_replacement(case):
    case.observer.install()
    probs = Tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    assert case.runner._sample(probs, None) is case.sampler.result
    assert len(case.sampler.calls) == 1
    passed_probs, passed_q = case.sampler.calls[0]
    assert passed_probs is probs and passed_q.values[0][0] == -99
    assert [gen.draws for gen in case.generators.values()] == [1, 1]
    records = events(case)
    assert [record["event"] for record in records] == [
        "adapter_install",
        "sample_begin",
        "exponential_noise_before_consumption",
        "sample_end",
    ]
    observed = records[2]["payload"]
    assert observed["q_before_original"]["values"][0][0] == 0.25
    assert observed["q_before_original"]["scope"] == "bounded_prefix_not_full_tensor"
    assert observed["rows"] == ["internal-A", "internal-B"]
    assert records[1]["payload"]["rows"][0]["seed"] == 2**62 + 13
    assert all(record["scientific_admission"] is False for record in records)


def test_row_mapping_is_taken_again_after_batch_compaction(case):
    case.observer.install()
    case.runner._sample(None, None)
    batch = case.runner.input_batch
    batch.req_ids[:] = ["internal-B"]
    batch.req_id_to_index = {"internal-B": 0}
    batch.sampling_metadata.generators = {0: case.generators[1]}
    case.sampler.result.sampled_token_ids = Tensor([[1]])
    case.runner._sample(None, None)
    assert events(case)[-2]["payload"]["rows"] == ["internal-B"]
    assert events(case)[-3]["payload"]["rows"][0]["seed"] == 19


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_rows",
        "missing_generator",
        "wrong_seed",
        "wrong_generator",
        "unseeded",
        "greedy",
        "v2",
        "spec",
        "n2",
        "temperature_zero",
        "prompt_embeds",
    ],
)
def test_unsupported_or_unbound_request_fails_before_random_draw(case, mutation):
    case.observer.install()
    runner = case.runner
    first = runner.requests["internal-A"]
    if mutation == "bad_rows":
        runner.input_batch.req_id_to_index["internal-A"] = 1
    elif mutation == "missing_generator":
        runner.input_batch.sampling_metadata.generators.pop(1)
    elif mutation == "wrong_seed":
        first.sampling_params.seed = 23
    elif mutation == "wrong_generator":
        first.generator = Generator(first.sampling_params.seed)
    elif mutation == "unseeded":
        first.sampling_params.seed = None
    elif mutation == "greedy":
        runner.input_batch.sampling_metadata.all_random = False
    elif mutation == "v2":
        runner.vllm_config.use_v2_model_runner = True
    elif mutation == "spec":
        runner.speculative_config = object()
    elif mutation == "n2":
        first.sampling_params.n = 2
    elif mutation == "temperature_zero":
        first.sampling_params.temperature = 0
    else:
        first.prompt_token_ids = None
    with pytest.raises(ObserverError):
        runner._sample(None, None)
    assert all(gen.draws == 0 for gen in case.generators.values())
    assert case.sampler.calls == []


def test_installed_bytes_change_is_rejected_before_draw(case):
    case.observer.install()
    case.pins[0].path.write_bytes(case.pins[0].path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ObserverError, match="source bytes"):
        case.runner._sample(None, None)
    assert case.sampler.calls == []


def test_forged_same_filename_function_code_does_not_pass_source_pin(case):
    module = case.modules[SAMPLER]
    forged = compile(
        "def random_sample(probs, generators, use_fp64_gumbel=False):\n return None\n",
        str(case.pins[1].path),
        "exec",
    )
    exec(forged, vars(module))
    with pytest.raises(ObserverError, match="code differs"):
        case.observer.install()


def test_same_bytes_from_other_module_path_do_not_pass_origin(case, tmp_path):
    path = tmp_path / "another.py"
    path.write_bytes(case.pins[0].path.read_bytes())
    case.modules[RUNNER].__file__ = str(path)
    with pytest.raises(ObserverError, match="pinned source"):
        case.observer.install()


def test_uninstalled_or_replaced_hook_is_not_silently_accepted(case):
    original = case.runner._sample.__func__
    case.observer.install()
    with pytest.raises(ObserverError, match="already"):
        case.observer.install()
    case.observer.uninstall()
    assert case.runner._sample.__func__ is original
    case.observer.install()
    case.sampler.sample_with_exponential_noise = lambda probs, q: None
    with pytest.raises(ObserverError, match="replaced"):
        case.runner._sample(None, None)


def test_warmup_has_no_request_claim_and_preserves_return(case):
    case.observer.install()
    assert case.sampler.random_sample(None, {0: Generator(7)}, False) is case.sampler.result
    assert len(events(case)) == 1


def test_original_failure_resets_context_and_has_no_success_event(case):
    case.observer.install()

    def fail(_rows):
        raise ValueError("fixture backend failed")

    case.sampler.make_q = fail
    with pytest.raises(ValueError, match="backend failed"):
        case.runner._sample(None, None)
    assert case.observer._frame.get() is None
    assert "sample_end" not in [row["event"] for row in events(case)]


def test_backend_branch_bypass_has_no_success_record(case):
    case.observer.install()
    original_runner, random, consume = case.observer._originals
    case.observer._originals = (lambda *args: case.sampler.result, random, consume)
    with pytest.raises(ObserverError, match="bypassed"):
        case.runner._sample(None, None)
    assert "sample_end" not in [row["event"] for row in events(case)]


def test_nonfinite_prefix_is_explicit_and_does_not_change_backend_values(case):
    case.sampler.make_q = lambda rows: Tensor([[float("nan"), 1.0] for _ in range(rows)])
    case.observer.install()
    case.runner._sample(None, None)
    q_record = events(case)[2]["payload"]["q_before_original"]
    assert q_record["sampled_values_finite"] is False
    assert q_record["values"] is None


def test_append_to_changed_trace_is_rejected(case):
    case.trace.write("first", {})
    with case.trace.path.open("ab") as stream:
        stream.write(b"injected\n")
    with pytest.raises(ObserverError, match="concurrently changed"):
        case.trace.write("second", {})


def test_trace_failure_blocks_consumption_and_poison_is_sticky(case, monkeypatch):
    case.observer.install()

    def fail(_fd):
        raise OSError("fixture fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failure"):
        case.runner._sample(None, None)
    assert case.sampler.calls == []
    with pytest.raises(ObserverError, match="poisoned"):
        case.runner._sample(None, None)


def test_fork_pid_reset_uses_distinct_stream_and_sequence(case, monkeypatch):
    case.trace.write("parent", {})
    parent = case.trace.path
    monkeypatch.setattr(os, "getpid", lambda: 987654321)
    case.trace.write("child", {})
    assert case.trace.path != parent
    assert events(case)[0]["sequence"] == 0
    assert events(case)[0]["pid"] == 987654321
    assert json.loads(parent.read_text())["event"] == "parent"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual Linux fork")
def test_actual_fork_writes_independent_trace(case):
    case.trace.write("parent_before", {})
    parent = case.trace.path
    pid = os.fork()
    if pid == 0:
        try:
            case.trace.write("child", {})
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    case.trace.write("parent_after", {})
    paths = list(case.trace.directory.glob("*.jsonl"))
    assert len(paths) == 2
    child = next(path for path in paths if path != parent)
    child_record = json.loads(child.read_text())
    assert child_record["pid"] == pid and child_record["sequence"] == 0
    assert [json.loads(line)["sequence"] for line in parent.read_text().splitlines()] == [0, 1]


def test_module_import_has_no_implicit_gpu_dependency():
    # The adapter deliberately uses only the standard library.
    import ast

    import agentmemeval.path1_v1.operations.backend_observer as observer

    tree = ast.parse(Path(observer.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(item.name.split(".")[0] for item in node.names)
        if isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert not roots.intersection({"torch", "vllm", "transformers"})
