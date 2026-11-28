"""Exact CUDA compiler binding for service JIT; no backend or GPU admission."""
from __future__ import annotations

import os
from pathlib import Path

from agentmemeval.path1_v1.operations import native_runtime


def validate_binding(value: dict) -> dict:
    required = {"cuda_home", "nvcc_sha256"}
    if type(value) is not dict or set(value) not in (required, required | {"ninja"}):
        raise ValueError("CUDA toolkit requires cuda_home, nvcc_sha256 and optional exact ninja")
    if type(value["cuda_home"]) is not str or str(Path(value["cuda_home"])) != value["cuda_home"]:
        raise ValueError("CUDA toolkit root must be a normalized absolute path")
    native_runtime.validate_binding({"path": value["cuda_home"], "sha256": value["nvcc_sha256"]})
    result = dict(value)
    if "ninja" in value:
        result["ninja"] = native_runtime.validate_binding(value["ninja"])
        if Path(result["ninja"]["path"]).name != "ninja":
            raise ValueError("bound Ninja executable must be named ninja")
    return result


def input_pin(binding: dict) -> dict:
    binding = validate_binding(binding)
    return {"path": str(Path(binding["cuda_home"]) / "bin/nvcc"), "sha256": binding["nvcc_sha256"]}


def input_pins(binding: dict) -> list[dict]:
    binding = validate_binding(binding)
    return [input_pin(binding)] + ([dict(binding["ninja"])] if "ninja" in binding else [])


def verify_file(binding: dict) -> dict:
    binding = validate_binding(binding)
    pin = input_pin(binding)
    observed = native_runtime.verify_file(pin)
    if not os.access(pin["path"], os.X_OK):
        raise ValueError("bound nvcc is not executable")
    result = {**binding, "nvcc_path": pin["path"], "nvcc_size": observed["size"]}
    if "ninja" in binding:
        ninja = native_runtime.verify_file(binding["ninja"])
        if not os.access(binding["ninja"]["path"], os.X_OK):
            raise ValueError("bound ninja is not executable")
        result["ninja_size"] = ninja["size"]
    return result


def environment(binding: dict, base: dict[str, str]) -> dict[str, str]:
    observed = verify_file(binding)
    result = dict(base)
    result.pop("FLASHINFER_NVCC_LAUNCHER", None)
    result.pop("FLASHINFER_CXX_LAUNCHER", None)
    result.update(CUDA_HOME=binding["cuda_home"], CUDA_PATH=binding["cuda_home"],
                  FLASHINFER_NVCC=observed["nvcc_path"])
    directories = [str(Path(binding["cuda_home"]) / "bin")]
    if "ninja" in observed:
        directories.insert(0, str(Path(observed["ninja"]["path"]).parent))
    result["PATH"] = os.pathsep.join([*directories, base.get("PATH", os.defpath)])
    return result
