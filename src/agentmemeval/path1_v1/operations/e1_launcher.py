"""E1 fixed allocation entry; runtime admission must come from a live verifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentmemeval.path1_v1.operations.e1_prepare import read_prepared
from agentmemeval.path1_v1.operations.intent_executor import verify_loaded_package
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from agentmemeval.path1_v1.operations.worker import run_worker


def run_prepared(path: Path, *, runtime_verifier=None):
    # Fresh process validation includes an actual full model hash. A prepared
    # descriptor is never a RuntimeAdmission. run_worker revalidates independently
    # and rejects its default None verifier before claiming or issuing requests.
    prepared = read_prepared(path)
    if (prepared.runtime_plan or {}).get("native_runtime") is not None:
        from agentmemeval.path1_v1.operations.native_runtime import verify_loaded
        verify_loaded(prepared.runtime_plan["native_runtime"])
    proof = prepared.descriptor["python_identity"]
    if (str(Path(sys.executable).absolute()) != proof["entry_path"]
            or str(Path(sys.prefix).resolve()) != proof["prefix"]
            or file_sha256(Path(sys.executable)) != proof["executable_sha256"]):
        raise ValueError("running E1 interpreter/venv differs from the prepared entry")
    identity = prepared.contract["runtime_identity"]
    verify_loaded_package(Path(prepared.request["repository_root"]),
                          identity["core_files"] + identity["operations_files"])
    if runtime_verifier is not None:
        # Library-only test integration; the CLI never accepts a verifier flag.
        return run_worker(Path(prepared.descriptor["worker_request_path"]),
                          runtime_verifier=runtime_verifier)
    from agentmemeval.path1_v1.operations.e1_runtime import run_e1_service

    return run_e1_service(prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args(argv)
    run_prepared(args.prepared)


if __name__ == "__main__":
    main()
