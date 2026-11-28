"""Opt-in general-plugin entrypoint with exact-module, post-exec installation.

No vLLM/torch eager import; no automatic runtime-launch integration or admission.
The owner must separately pin this overlay/config in a prepared service request.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations.backend_challenges import (
    API,
    Challenge,
    ChallengeRegistry,
    SelectedSamplerObserver,
    install_api,
    sha,
)
from agentmemeval.path1_v1.operations.backend_observer import (
    DEPENDENCIES,
    RUNNER,
    SAMPLER,
    ObserverError,
    ProcessTrace,
    SourcePin,
)

PLUGIN_NAME = "path1_seeded_observer"
ENTRYPOINT = "agentmemeval.path1_v1.operations.backend_plugin:activate"


class _AfterExec:
    def __init__(self, original: Any, callback: Any):
        self.original = original
        self.callback = callback

    def create_module(self, spec: Any) -> Any:
        return self.original.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.original.exec_module(module)
        self.callback(module.__name__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class DeferredObserver(importlib.abc.MetaPathFinder):
    """Delegate the preexisting finder chain; wrap only exact ordinary source loaders."""

    def __init__(self, registry: ChallengeRegistry, pins: tuple[SourcePin, ...], *, runtime=False):
        self.registry = registry
        self.pins = {pin.module_name: pin for pin in pins}
        if len(self.pins) != len(pins) or set(self.pins) != {API, RUNNER, SAMPLER, *DEPENDENCIES}:
            raise ObserverError("exact API and sampler source pins required")
        self.targets = frozenset({API, RUNNER})
        self.installed: set[str] = set()
        self.observer: SelectedSamplerObserver | None = None
        self.api_wrapper: Any = None
        self.runtime = runtime

    def start(self) -> None:
        if self in sys.meta_path:
            raise ObserverError("deferred observer already started")
        sys.meta_path.insert(0, self)
        for target in (API, RUNNER):
            module = sys.modules.get(target)
            if module is not None and not getattr(
                getattr(module, "__spec__", None), "_initializing", False
            ):
                self.after_loaded(target)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in self.targets or fullname in self.installed:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                # Do not alter legacy/custom import semantics by bypassing it.
                self.registry.event("unsupported_legacy_finder", [fullname])
                return None
            spec = find(fullname, path, target)
            if spec is None:
                continue
            if type(spec.loader) is not importlib.machinery.SourceFileLoader:
                self.registry.event("unsupported_loader", [fullname])
                return spec
            if spec.origin != str(self.pins[fullname].path):
                self.registry.event("unmatched_module_origin", [fullname])
                return spec
            # Preserve the actual ModuleSpec and delegate original module execution
            # exactly once, then install only after its dependencies are initialized.
            spec.loader = _AfterExec(spec.loader, self.after_loaded)
            return spec
        return None

    def after_loaded(self, fullname: str) -> None:
        if fullname in self.installed:
            return
        try:
            if fullname == API:
                self.api_wrapper = install_api(self.registry, self.pins[API])
            elif fullname == RUNNER:
                pins = tuple(pin for name, pin in self.pins.items() if name != API)
                self.observer = SelectedSamplerObserver(self.registry, pins)
                if self.runtime:
                    from agentmemeval.path1_v1.operations.e1_backend_runtime import (
                        engine_observation,
                    )

                    self.observer.runtime_recorder = engine_observation
                self.observer.install()
            else:
                raise ObserverError("unexpected module callback")
            self.installed.add(fullname)
        except Exception as exc:
            # Basic E0 may still run without coverage. Never claim a ready/GPU event
            # from this failure and do not turn an import into a model-service crash.
            self.registry.event("installation_failed:" + type(exc).__name__, [fullname])


_active: DeferredObserver | None = None
_configuration_sha256: str | None = None


def activate() -> DeferredObserver | None:
    """No configuration means disabled, including ordinary installations of the package."""
    global _active, _configuration_sha256
    raw_path = os.environ.get("PATH1_BACKEND_PLUGIN_CONFIG")
    if raw_path is None:
        return None
    expected = os.environ.get("PATH1_BACKEND_PLUGIN_CONFIG_SHA256")
    if os.environ.get("VLLM_PLUGINS") != PLUGIN_NAME:
        raise ObserverError("explicit single-plugin allowlist required")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or path.stat().st_size > 128 * 1024
    ):
        raise ObserverError("invalid bounded plugin configuration path")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ObserverError("plugin configuration bytes differ")
    if _active is not None:
        if _configuration_sha256 != expected:
            raise ObserverError("cannot replace an active observer configuration")
        # Fork inherits wrappers and trace objects. ProcessTrace rebuilds its PID,
        # lock and nonce; do not reinstall wrappers or reset vLLM.plugins_loaded.
        return _active
    config = json.loads(raw)
    if (
        set(config)
        != {
            "schema_version",
            "service_instance",
            "source_sha256",
            "trace_directory",
            "registration_directory",
            "challenges",
            "source_pins",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] not in {1, 2}
    ):
        raise ObserverError("unsupported plugin configuration schema")
    trace = ProcessTrace(
        Path(config["trace_directory"]),
        service_instance=config["service_instance"],
        source_sha256=config["source_sha256"],
    )
    registry = ChallengeRegistry(
        Path(config["registration_directory"]),
        tuple(Challenge(**item) for item in config["challenges"]),
        trace,
    )
    pins = tuple(
        SourcePin(item["module_name"], Path(item["path"]), item["sha256"])
        for item in config["source_pins"]
    )
    observer = DeferredObserver(registry, pins, runtime=config["schema_version"] == 2)
    observer.start()
    _active, _configuration_sha256 = observer, expected
    return observer


def create_overlay(destination: Path) -> dict[str, str]:
    """Exclusive metadata-only overlay; caller must pin it before runtime launch.

    This function does not edit PYTHONPATH, install a package, or start vLLM.
    """
    if not destination.is_absolute() or destination.parent.resolve() != destination.parent:
        raise ObserverError("overlay needs a canonical absolute parent")
    destination.mkdir(exist_ok=False)
    metadata = destination / "path1_backend_observer-0.1.dist-info"
    metadata.mkdir()
    files = {
        "METADATA": "Metadata-Version: 2.1\nName: path1-backend-observer\nVersion: 0.1\n",
        "entry_points.txt": f"[vllm.general_plugins]\n{PLUGIN_NAME} = {ENTRYPOINT}\n",
    }
    hashes = {}
    for name, text in files.items():
        data = text.encode()
        with (metadata / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[(metadata / name).relative_to(destination).as_posix()] = sha(data)
    return hashes
