"""Actual tiny model bytes exercise reuse; fixtures never admit a real backend."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import contracts
from agentmemeval.path1_v1.operations import model_verification as verification
from agentmemeval.path1_v1.operations import runtime_probe as probe

BINDING = "a" * 64


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (root / name).write_bytes(b"verified fixture bytes")
    manifest = {
        "schema_version": 1, "model": contracts.MODEL, "revision": contracts.MODEL_REVISION,
        "tokenizer_revision": contracts.MODEL_REVISION, "chat_template_sha256": "1" * 64,
        "files": [{"path": p.name, "size": p.stat().st_size,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(root.iterdir())],
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(contracts.canonical_json(manifest))
    return path, root, contracts._model_identity(path, root)


def issue(model, **kwargs):
    manifest, root, identity = model
    return verification.verify_model_files(manifest, root, expected_identity=identity,
                                           binding_sha256=BINDING, **kwargs)


def reuse(model, capability, **kwargs):
    manifest, root, identity = model
    return verification.reuse_model_files(capability, manifest, root,
        expected_identity=identity, binding_sha256=kwargs.pop("binding_sha256", BINDING), **kwargs)


def test_exact_full_hash_once_then_reuse_with_no_weight_reads(model, monkeypatch):
    original = verification._model_identity
    calls = []
    monkeypatch.setattr(verification, "_model_identity", lambda *args: (
        calls.append(args), original(*args))[1])
    capability = issue(model)
    original_open = Path.open

    def no_weight_reads(path, *args, **kwargs):
        assert path.suffix != ".safetensors", "reuse must not stream weight bytes again"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_weight_reads)
    for _ in range(4):
        assert reuse(model, capability) == model[2]
    assert len(calls) == 1
    diagnostic = verification.model_verification_diagnostic(capability)
    assert diagnostic["runtime_admission"] is False and not diagnostic["restores_capability"]
    assert len(diagnostic["metadata"]) == 5  # Root plus four files.
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("change", ["content", "replace", "extra", "missing", "manifest",
                                     "root_replace", "mtime", "hardlink"])
def test_every_inventory_and_file_identity_change_rejects_without_rehash(
    model, tmp_path, monkeypatch, change,
):
    capability = issue(model)
    path, root, _ = model
    weights = root / "model.safetensors"
    if change == "content":
        weights.write_bytes(b"different fixture bytes")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(weights.read_bytes())
        replacement.replace(weights)
    elif change == "extra":
        (root / "extra").write_bytes(b"extra")
    elif change == "missing":
        weights.unlink()
    elif change == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "root_replace":
        root.rename(tmp_path / "old_model")
        root.mkdir()
        for old in (tmp_path / "old_model").iterdir():
            (root / old.name).write_bytes(old.read_bytes())
    elif change == "mtime":
        original = weights.stat()
        os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns + 10_000_000))
    else:
        os.link(weights, tmp_path / "outside_link")
    monkeypatch.setattr(verification, "_model_identity", lambda *_: pytest.fail("no fallback"))
    with pytest.raises((verification.ModelVerificationError, FileNotFoundError)):
        reuse(model, capability)


def test_modification_during_full_hash_does_not_issue_capability(model, monkeypatch):
    original = verification._model_identity

    def mutate(*args):
        identity = original(*args)
        (model[1] / "config.json").write_bytes(b"mutated after it was hashed")
        return identity

    monkeypatch.setattr(verification, "_model_identity", mutate)
    with pytest.raises(verification.ModelVerificationError, match="during full hashing"):
        issue(model)


@pytest.mark.parametrize("forged", ["new_object", "json", "identity"])
def test_json_and_constructed_objects_cannot_restore_capability(model, forged):
    capability = issue(model)
    value = (verification.VerifiedModelFiles() if forged == "new_object" else
             json.loads(json.dumps(verification.model_verification_diagnostic(capability)))
             if forged == "json" else model[2])
    with pytest.raises(verification.ModelVerificationError, match="exact process-local"):
        reuse(model, value)


def test_capability_cannot_be_copied_or_pickled(model):
    capability = issue(model)
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="serialized or copied"):
            operation(capability)


def test_binding_and_expected_identity_cannot_be_reassigned(model):
    capability = issue(model)
    with pytest.raises(verification.ModelVerificationError, match="path/request"):
        reuse(model, capability, binding_sha256="b" * 64)
    different = dict(model[2], files_verified=1)
    with pytest.raises(verification.ModelVerificationError, match="expected identity"):
        reuse((model[0], model[1], different), capability)


def test_lifetime_starts_after_full_hash_and_cannot_be_extended(model, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(verification.time, "monotonic", lambda: clock[0])
    original = verification._model_identity

    def slow(*args):
        clock[0] += 47.0
        return original(*args)

    monkeypatch.setattr(verification, "_model_identity", slow)
    capability = issue(model, max_age_seconds=120)
    diagnostic = verification.model_verification_diagnostic(capability)
    assert diagnostic["verified_monotonic"] == 147 and diagnostic["full_hash_seconds"] == 47
    clock[0] = 267
    assert reuse(model, capability) == model[2]
    clock[0] += 0.001
    with pytest.raises(verification.ModelVerificationError, match="expired"):
        reuse(model, capability)
    clock[0] = 146
    with pytest.raises(verification.ModelVerificationError, match="backwards"):
        reuse(model, capability)


def test_metadata_scan_cannot_run_past_lease_deadline(model, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(verification.time, "monotonic", lambda: clock[0])
    capability = issue(model, max_age_seconds=1)
    original = verification._state

    def slow(*args):
        value = original(*args)
        clock[0] = 102.0
        return value

    monkeypatch.setattr(verification, "_state", slow)
    with pytest.raises(verification.ModelVerificationError, match="expired"):
        reuse(model, capability)


def test_changed_pid_cannot_reuse_parent_capability(model, monkeypatch):
    capability = issue(model)
    pid = os.getpid()
    monkeypatch.setattr(verification.os, "getpid", lambda: pid + 1)
    with pytest.raises(verification.ModelVerificationError, match="another process"):
        reuse(model, capability)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="actual Linux fork boundary")
def test_real_fork_rejects_parent_capability_but_can_perform_new_full_hash(model):
    capability = issue(model)
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(reader)
        try:
            try:
                reuse(model, capability)
            except verification.ModelVerificationError:
                fresh = issue(model)
                assert reuse(model, fresh) == model[2]
                os.write(writer, b"rejected_parent_and_verified_child")
                os._exit(0)
            os._exit(1)
        except BaseException:
            os._exit(2)
    os.close(writer)
    observed = os.read(reader, 200)
    os.close(reader)
    _, status = os.waitpid(child, 0)
    assert status == 0 and observed == b"rejected_parent_and_verified_child"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux ctime/change semantics")
def test_same_size_inplace_write_with_restored_mtime_is_detected(model):
    capability = issue(model)
    weight = model[1] / "model.safetensors"
    old = weight.stat()
    weight.write_bytes(b"X" * old.st_size)
    os.utime(weight, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(verification.ModelVerificationError, match="changed"):
        reuse(model, capability)


@pytest.mark.skipif(sys.platform != "linux", reason="actual symlink no-follow boundary")
def test_replaced_file_symlink_rejected_even_when_target_bytes_match(model, tmp_path):
    capability = issue(model)
    weight = model[1] / "model.safetensors"
    target = tmp_path / "target"
    weight.rename(target)
    weight.symlink_to(target)
    with pytest.raises(verification.ModelVerificationError, match="symlink"):
        reuse(model, capability)


def test_real_git_probe_revalidation_reuses_model_but_still_checks_source(tmp_path, monkeypatch):
    from path1_v1.unit.test_runtime_probe import make_probe_fixture

    data, _, _, path = make_probe_fixture(tmp_path)
    original = verification._model_identity
    calls = []
    monkeypatch.setattr(verification, "_model_identity", lambda *args: (
        calls.append(args), original(*args))[1])
    first = probe.validate_probe_request(path)
    second = probe.validate_probe_request(path, model_verification=first.model_verification)
    assert first == second and first.model_verification is second.model_verification
    assert len(calls) == 1
    (data["repo"] / "src/core.py").write_text("# modified source\n", encoding="utf-8")
    with pytest.raises(probe.RuntimeProbeError, match="source/model identity changed"):
        probe.validate_probe_request(path, model_verification=first.model_verification)
    assert len(calls) == 1


def test_build_identity_reuse_exactly_preserves_original_hash_semantics(tmp_path, monkeypatch):
    from path1_v1.unit.test_operations_contracts import evidence

    data = evidence.__wrapped__(tmp_path)
    original = contracts.build_runtime_identity(data["repo"], **data["build_kwargs"])
    capability = verification.verify_model_files(data["manifest"], data["model"],
        expected_identity=original["model"], binding_sha256=BINDING)
    monkeypatch.setattr(contracts, "_model_identity", lambda *_: pytest.fail("unneeded rehash"))
    observed = contracts.build_runtime_identity(data["repo"], **data["build_kwargs"],
        model_verification=capability, model_binding_sha256=BINDING)
    assert contracts.canonical_json(observed) == contracts.canonical_json(original)


def test_observe_model_reuses_actual_verified_files(model, monkeypatch):
    capability = issue(model)
    request = probe.ValidatedProbeRequest(
        {"model_root": str(model[1]), "model_manifest_path": str(model[0])},
        {"runtime_identity": {"model": model[2]}}, "fixture", BINDING,
        model[1].parent / "output", capability)
    monkeypatch.setattr(probe, "_model_identity", lambda *_: pytest.fail("unneeded fullhash"))
    observed, metadata = probe.observe_model(request)
    assert observed == model[2] and len(metadata) == 4
    (model[1] / "config.json").write_bytes(b"changed")
    with pytest.raises(verification.ModelVerificationError, match="changed"):
        probe.observe_model(request)


def test_fixed_launcher_service_revalidates_real_request_without_second_full_hash(
    tmp_path, monkeypatch,
):
    from agentmemeval.path1_v1.operations import probe_launcher
    from path1_v1.unit.test_e0_executor import make_e0_fixture

    fixture = make_e0_fixture(tmp_path, kind="service_probe")
    original = verification._model_identity
    calls = []
    monkeypatch.setattr(verification, "_model_identity", lambda *args: (
        calls.append(args), original(*args))[1])
    monkeypatch.setattr(sys, "executable", str(fixture.python))
    monkeypatch.setattr(sys, "prefix", fixture.request_data["launch"]["python_identity"]["prefix"])

    def host():
        raise probe.RuntimeProbeError("real_validator_reached_host_gate")

    monkeypatch.setattr(probe, "observe_host", host)
    with pytest.raises(probe.RuntimeProbeError, match="real_validator_reached_host_gate"):
        probe_launcher.main(["--request", str(fixture.request_path)])
    assert len(calls) == 1 and not (fixture.job_dir / "e0_probe").exists()


def test_http_response_saved_but_probe_incomplete_if_model_changes_during_http(
    model, tmp_path, monkeypatch,
):
    from path1_v1.unit.test_runtime_probe import offline_collector

    fixture_root = tmp_path / "collector"
    fixture_root.mkdir()
    request, owner, calls = offline_collector(fixture_root, monkeypatch)
    old_root = request.request["model_root"]
    request.request.update(model_root=str(model[1]), model_manifest_path=str(model[0]))
    request.contract["runtime_identity"]["model"] = model[2]
    capability = verification.verify_model_files(model[0], model[1], expected_identity=model[2],
        binding_sha256=request.request_sha256)
    request = replace(request, model_verification=capability)
    snapshot = owner.verify_live()
    snapshot = replace(snapshot, binding=replace(snapshot.binding,
        source_sha256=contracts.content_digest(request.contract["runtime_identity"])),
        cmdline=tuple(str(model[1]) if item == old_root else item for item in snapshot.cmdline))
    owner.verify_live = lambda: snapshot
    original = probe.http_exchange

    def changed(*args):
        observed = original(*args)
        (model[1] / "config.json").write_bytes(b"modified while HTTP was in flight")
        return observed

    monkeypatch.setattr(probe, "http_exchange", changed)
    output = request.output_directory / "observations"
    with pytest.raises(verification.ModelVerificationError, match="changed"):
        probe.collect_e0_probe(request, owner, "http://127.0.0.1:8000/v1", output)
    assert len(calls) == 1 and (output / "action_response.json").is_file()
    receipt = json.loads((output / "probe_receipt.json").read_bytes())
    assert receipt["status"] == "incomplete" and receipt["runtime_admission"] is False


def test_changed_request_cannot_reuse_capability_or_extend_its_lifetime(tmp_path, monkeypatch):
    from path1_v1.unit.test_runtime_probe import make_probe_fixture

    _, _, request, path = make_probe_fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(verification.time, "monotonic", lambda: clock[0])
    validated = probe.validate_probe_request(path)
    clock[0] = 221.0
    with pytest.raises(verification.ModelVerificationError, match="expired"):
        probe.validate_probe_request(path, model_verification=validated.model_verification,
                                     model_max_age_seconds=6600.0)
    clock[0] = 110.0
    new_output = tmp_path / "other_output"
    new_output.mkdir()
    request["output_root"] = str(new_output)
    request["request_sha256"] = contracts.content_digest(
        {key: value for key, value in request.items() if key != "request_sha256"})
    path.write_bytes(contracts.canonical_json(request))
    with pytest.raises(verification.ModelVerificationError, match="path/request"):
        probe.validate_probe_request(path, model_verification=validated.model_verification)
