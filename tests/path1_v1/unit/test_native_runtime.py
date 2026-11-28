"""Single-library launch boundaries; Linux fixtures do not attest GPU execution."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import native_runtime as native
from agentmemeval.path1_v1.operations.contracts import canonical_json
from agentmemeval.path1_v1.operations.e0_service_cli import _config
from agentmemeval.path1_v1.operations.e1_prepare import render_e1_script
from agentmemeval.path1_v1.operations.intent_executor import render_probe_script


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("bad", [None, {}, {"path": "/x", "sha256": True},
    {"path": "relative", "sha256": "a" * 64},
    {"path": "/a:b", "sha256": "a" * 64, "verified": True},
    {"path": "/a b", "sha256": "a" * 64},
    {"path": "/x\n", "sha256": "a" * 64}])
def test_binding_rejects_unbound_environment(bad):
    with pytest.raises(ValueError):
        native.validate_binding(bad)


def test_exact_file_environment_and_tamper(tmp_path):
    path = tmp_path / "lib.so"
    path.write_bytes(b"original")
    pin = binding(path)
    assert native.verify_file(pin) == pin | {"size": 8}
    before = {"PATH": "fixture", "LD_PRELOAD": "/other",
              "LD_LIBRARY_PATH": "/other", "LD_AUDIT": "x"}
    after = native.environment(pin, before)
    assert after == {"PATH": "fixture", "LD_PRELOAD": str(path)}
    assert before["LD_PRELOAD"] == "/other"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash differs"):
        native.environment(pin, {})


def test_legacy_renderers_unchanged_and_native_explicit(tmp_path):
    python, repo, request = (tmp_path / name for name in ("python", "repo", "request.json"))
    pin = {"path": str(tmp_path / "lib.so"), "sha256": "a" * 64}
    legacy = render_probe_script(python_executable=python, repository_root=repo,
                                probe_request_path=request)
    assert b"native_runtime" not in legacy and b" -B -s -m " in legacy
    for renderer, key, module in ((render_probe_script, "probe_request_path", "probe_launcher"),
                                 (render_e1_script, "prepared_path", "e1_launcher")):
        raw = renderer(python_executable=python, repository_root=repo, **{key: request},
                       native_runtime=pin)
        argv = shlex.split(raw.decode().split("exec ", 1)[1])
        assert argv == native.bootstrap_argv(python, repo, pin,
            module="agentmemeval.path1_v1.operations." + module,
            arguments=["--request" if key == "probe_request_path" else "--prepared", str(request)])


def test_config_native_requires_profile3_and_exact_object(tmp_path):
    repo = Path(native.__file__).parents[4]
    value = json.loads((repo / "configs/path1_v1/e0_service_scze856_v3.json").read_bytes())
    value["native_runtime"] = {"path": str(tmp_path / "library"), "sha256": "a" * 64}
    assert _config(canonical_json(value))["native_runtime"] == value["native_runtime"]
    value["native_runtime"] = None
    with pytest.raises(ValueError):
        _config(canonical_json(value))


@pytest.mark.parametrize("version", [1, 2])
def test_shared_profile_validator_rejects_legacy_native_policy(tmp_path, version):
    from agentmemeval.path1_v1.operations.service_profile import validate_service_profile
    from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled

    repo = Path(native.__file__).parents[4]
    value = json.loads((repo / "configs/path1_v1/e0_service_scze856_v3.json").read_bytes())
    value.update(resource_profile_version=version, memory_mib=65536,
                 native_runtime={"path": str(tmp_path / "library"), "sha256": "a" * 64})
    value.pop("memory_policy")
    if version == 1:
        value.pop("language_model_only")
    with pytest.raises(SubmissionDisabled, match="native runtime requires service profile 3"):
        validate_service_profile(value)


@pytest.mark.parametrize("mode", ["matching", "missing", "different"])
def test_e1_native_plan_must_match_prior_e0(tmp_path, monkeypatch, mode):
    from agentmemeval.path1_v1.operations import e1_campaign as campaign

    library = tmp_path / "lib.so"
    library.write_bytes(b"fixture native bytes")
    pin = binding(library)
    proof = {"entry_path": str(tmp_path / "python"), "executable_sha256": "b" * 64,
             "prefix": str(tmp_path)}
    pins = [{"module_name": name, "path": str(library), "sha256": pin["sha256"]}
            for name in sorted(campaign.BACKEND_MODULES)]
    value = {"schema_version": 1, "kind": "path1_e1_runtime_plan", "service_port": 18231,
             "service_python_identity": proof, "source_pins": pins,
             "startup_seconds": 1200, "cleanup_seconds": 600,
             "backend_probe_seeds": [2**62 + 9001, 2**61 + 9001],
             "prior_e0": {}, "language_model_only": True, "native_runtime": pin}
    monkeypatch.setattr(campaign, "verify_python_interpreter", lambda *a, **k: proof)
    monkeypatch.setattr(campaign, "_e0_observations", lambda *a: {"native_runtime": pin})
    if mode == "missing":
        value.pop("native_runtime")
    elif mode == "different":
        value["native_runtime"] = pin | {"sha256": "c" * 64}
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json(value))
    if mode == "matching":
        assert campaign.read_runtime_plan(path, contract={"phase": "E1"}) == value
    else:
        with pytest.raises(ValueError, match="native runtime must match"):
            campaign.read_runtime_plan(path, contract={"phase": "E1"})


def linux_library():
    # Use a real installed ordinary shared library, never manufacture a loaded flag.
    for path in (Path("/usr/lib/x86_64-linux-gnu/libm.so.6"),
                 Path("/lib/x86_64-linux-gnu/libm.so.6")):
        if path.exists():
            return path.resolve()
    pytest.fail("real Linux shared library fixture missing")


@pytest.mark.skipif(sys.platform != "linux", reason="actual ELF preload and /proc require Linux")
@pytest.mark.parametrize("mode", ["stdin", "module"])
def test_real_bootstrap_reexec_preserves_stdin_arguments_and_loaded_identity(tmp_path, mode):
    pin = binding(linux_library())
    repo = Path(native.__file__).parents[4]
    entry = Path(sys.executable)
    if mode == "stdin":
        venv.EnvBuilder(with_pip=False).create(tmp_path / "venv")
        entry = tmp_path / "venv/bin/python"
        code = ("import json,sys;from agentmemeval.path1_v1.operations.native_runtime "
                "import verify_loaded;print(json.dumps({'argv':sys.argv,'prefix':sys.prefix,"
                "'loaded':verify_loaded("
                + repr(pin) + ")}))")
        argv = native.bootstrap_argv(entry, repo, pin,
                                     arguments=["hello", "two words"])
    else:
        code = '{"value": 3}'
        argv = native.bootstrap_argv(Path(sys.executable), repo, pin, module="json.tool")
    env = dict(os.environ)
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"):
        env.pop(key, None)
    completed = subprocess.run(argv, input=code, text=True, capture_output=True,
                               env=env, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    result = json.loads(completed.stdout)
    if mode == "stdin":
        assert result["argv"] == ["-", "hello", "two words"]
        assert result["prefix"] == str(tmp_path / "venv")
        assert result["loaded"]["sha256"] == pin["sha256"]
        assert result["loaded"]["executable_mapping"] is True
        assert result["loaded"]["runtime_admission"] is False
    else:
        assert result == {"value": 3}


@pytest.mark.skipif(sys.platform != "linux", reason="actual ELF preload requires Linux")
def test_bootstrap_bad_hash_never_consumes_or_executes_stdin(tmp_path):
    pin = binding(linux_library())
    pin["sha256"] = "0" * 64
    effect = tmp_path / "effect"
    argv = native.bootstrap_argv(Path(sys.executable), Path(native.__file__).parents[4], pin)
    result = subprocess.run(argv, input="from pathlib import Path;Path(" + repr(str(effect))
                            + ").write_text('bad')", text=True, capture_output=True, timeout=15)
    assert result.returncode != 0 and not effect.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="actual child ELF mappings require Linux")
def test_live_child_mapping_is_bound_to_exact_file_inode(tmp_path):
    library = tmp_path / "pinned_libm.so"
    library.write_bytes(linux_library().read_bytes())
    pin = binding(library)
    clean = dict(os.environ)
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"):
        clean.pop(key, None)
    for loaded in (True, False):
        env = native.environment(pin, clean) if loaded else clean
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c",
                                  "import time;print('ready',flush=True);time.sleep(15)"],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        try:
            assert child.stdout.readline() == "ready\n"
            if loaded:
                assert native.verify_loaded(pin, pid=child.pid)["pid"] == child.pid
                with pytest.raises(ValueError, match="hash differs"):
                    native.verify_loaded(pin | {"sha256": "0" * 64}, pid=child.pid)
            else:
                with pytest.raises(ValueError, match="not actually loaded"):
                    native.verify_loaded(pin, pid=child.pid)
        finally:
            child.terminate()
            child.communicate(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Slurm canonical paths require Linux")
def test_prepared_e0_binds_native_config_request_script_and_profile(tmp_path, monkeypatch):
    from agentmemeval.path1_v1.operations import e0_service_cli as cli
    from path1_v1.unit.test_e0_service_cli import fixture
    from path1_v1.unit.test_operations_contracts import _git

    repo, path, config = fixture(tmp_path, monkeypatch)
    pin = binding(linux_library())
    config["native_runtime"] = pin
    path.write_bytes(canonical_json(config))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pin real native library fixture")
    # This boundary isolates preparation from current pytest process startup.
    # The actual reexec+mapping is exercised separately above without this mock.
    monkeypatch.setattr(native, "verify_loaded", lambda value: native.verify_file(value))
    root = tmp_path / "prepared"
    result = cli.prepare(repo, root, "campaign", "configs/path1_v1/site.json")
    descriptor, frozen, _ = cli.read_prepared(root, result["descriptor_sha256"])
    assert frozen["native_runtime"] == pin
    assert descriptor["service_profile"]["native_runtime"] == pin
    probe = json.loads((root / "probe_request.json").read_bytes())
    assert probe["launch"]["native_runtime"] == pin
    assert pin["sha256"].encode() in (root / "probe.sh").read_bytes()
