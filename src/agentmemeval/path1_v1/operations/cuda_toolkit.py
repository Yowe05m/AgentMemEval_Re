"""Exact CUDA compiler binding for service JIT; no backend or GPU admission."""
from __future__ import annotations

import os
from pathlib import Path

from agentmemeval.path1_v1.operations import native_runtime


def validate_binding(value: dict) -> dict:
    if type(value) is not dict or set(value) != {"cuda_home", "nvcc_sha256"}:
        raise ValueError("CUDA toolkit requires exact cuda_home and nvcc_sha256")
    if type(value["cuda_home"]) is not str or str(Path(value["cuda_home"])) != value["cuda_home"]:
        raise ValueError("CUDA toolkit root must be a normalized absolute path")
    native_runtime.validate_binding({"path": value["cuda_home"], "sha256": value["nvcc_sha256"]})
    return dict(value)


def input_pin(binding: dict) -> dict:
    binding = validate_binding(binding)
    return {"path": str(Path(binding["cuda_home"]) / "bin/nvcc"), "sha256": binding["nvcc_sha256"]}


def verify_file(binding: dict) -> dict:
    pin = input_pin(binding)
    observed = native_runtime.verify_file(pin)
    if not os.access(pin["path"], os.X_OK):
        raise ValueError("bound nvcc is not executable")
    return {**binding, "nvcc_path": pin["path"], "nvcc_size": observed["size"]}


def environment(binding: dict, base: dict[str, str]) -> dict[str, str]:
    observed = verify_file(binding)
    result = dict(base)
    result.pop("FLASHINFER_NVCC_LAUNCHER", None)
    result.pop("FLASHINFER_CXX_LAUNCHER", None)
    result.update(CUDA_HOME=binding["cuda_home"], CUDA_PATH=binding["cuda_home"],
                  FLASHINFER_NVCC=observed["nvcc_path"])
    result["PATH"] = (str(Path(binding["cuda_home"]) / "bin") + os.pathsep
                      + base.get("PATH", os.defpath))
    return result
