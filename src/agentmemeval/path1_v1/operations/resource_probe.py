"""Finite E0 resource collection under a 90 second owned process-group deadline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentmemeval.path1_v1.experiments.path1 import _new_json
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.process_ownership import OwnedProcess, ProcessBinding
from agentmemeval.path1_v1.operations.runtime_probe import (
    _require,
    _verify_loaded_source,
    observe_gpu,
    observe_host,
    validate_probe_request,
)
from agentmemeval.path1_v1.operations.submission_process import write_exclusive_json


def collect_resource_probe(request_path: Path) -> dict:
    request = validate_probe_request(request_path, _collecting_resource=True)
    _verify_loaded_source(request)
    output = request.output_directory / "observations"
    output.mkdir(exist_ok=False)
    receipt = {
        "schema_version": 1,
        "kind": "path1_e0_resource_observations",
        "request_sha256": request.request_sha256,
        "intent_id": request.intent_id,
        "runtime_admission": False,
        "status": "incomplete",
    }
    try:
        host = observe_host(cleanup_seconds=30)
        _new_json(output / "host.json", host)
        gpu = observe_gpu()
        _new_json(output / "gpu.json", gpu)
        disk = shutil.disk_usage(output)
        probe_file = output / "filesystem_probe.bin"
        payload = request.request_sha256.encode("ascii")
        with probe_file.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _require(probe_file.read_bytes() == payload, "resource filesystem roundtrip failed")
        receipt.update(
            status="observations_complete",
            host=host,
            gpu=gpu,
            filesystem={
                "path": str(output),
                "total": disk.total,
                "free": disk.free,
                "used": disk.used,
            },
            completed_at=datetime.now(UTC).isoformat(),
        )
    except BaseException as exc:
        receipt["failure_type"] = type(exc).__name__
        _new_json(output / "resource_receipt.json", receipt)
        raise
    _new_json(output / "resource_receipt.json", receipt)
    return receipt


def run_resource_probe(request_path: Path) -> dict:
    request = validate_probe_request(request_path)
    _require(
        request.request["schema_version"] == 2
        and request.contract["job"]["kind"] == "resource_probe",
        "not a v2 resource probe",
    )
    # Host admission is in the bounded child. It cannot load or run a model.
    _verify_loaded_source(request)
    host = observe_host(cleanup_seconds=30)
    command_deadline = min(
        time.monotonic() + 90,
        time.monotonic()
        + (datetime.fromisoformat(host["allocation_end_utc"]) - datetime.now(UTC)).total_seconds()
        - 30,
    )
    _require(command_deadline > time.monotonic(), "no resource command interval remains")
    output = request.output_directory
    output.mkdir(parents=True, exist_ok=False)
    launch = request.request["launch"]
    binding = ProcessBinding(
        request.contract["campaign_id"],
        request.intent_id,
        host["job_id"],
        content_digest(request.contract["runtime_identity"]),
        "resource_probe",
        uuid4().hex,
    )
    argv = [
        launch["python_executable"],
        "-B",
        "-s",
        "-m",
        "agentmemeval.path1_v1.operations.resource_probe",
        "--collect-request",
        str(request_path),
    ]
    _new_json(
        output / "resource_launch.json",
        {
            "request_sha256": request.request_sha256,
            "argv": argv,
            "binding": asdict(binding),
            "command_seconds": 90,
            "cleanup_seconds": 30,
            "term_grace_seconds": 15,
            "host": host,
            "runtime_admission": False,
        },
    )

    def persist_cleanup(owner, receipt):
        name = "resource_cleanup.json" if owner is not None else "resource_spawn_failure.json"
        write_exclusive_json(output / name, asdict(receipt))

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(request.request["repository_root"]) / "src")
    with (
        (output / "resource_stdout.log").open("xb") as stdout,
        (output / "resource_stderr.log").open("xb") as stderr,
    ):
        try:
            deadline = command_deadline
            with OwnedProcess.managed(
                argv,
                binding=binding,
                cwd=Path(request.request["repository_root"]),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                term_grace_seconds=15,
                receipt_sink=persist_cleanup,
            ) as owner:
                while not owner.poll_exit() and time.monotonic() < deadline:
                    time.sleep(0.01)
                completed_before_deadline = owner.poll_exit() and time.monotonic() < deadline
                cleanup = owner.close(term_grace_seconds=15)
                _require(
                    completed_before_deadline
                    and cleanup.returncode == 0
                    and cleanup.group_cleanup_verified,
                    "resource command did not complete",
                )
            from agentmemeval.path1_v1.operations.runtime_probe import _strict_json

            receipt = _strict_json((output / "observations/resource_receipt.json").read_bytes())
            _require(
                receipt.get("status") == "observations_complete"
                and receipt.get("request_sha256") == request.request_sha256
                and receipt.get("runtime_admission") is False,
                "resource evidence incomplete",
            )
            return receipt
        except BaseException as exc:
            _new_json(
                output / "resource_failure.json",
                {
                    "failure_type": type(exc).__name__,
                    "request_sha256": request.request_sha256,
                    "runtime_admission": False,
                },
            )
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-request", type=Path, required=True)
    args = parser.parse_args(argv)
    collect_resource_probe(args.collect_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
