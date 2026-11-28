"""Actual host/site pre-submit observations for the fixed E1 prepared entry."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.e1_prepare import ValidatedE1Prepared, read_prepared
from agentmemeval.path1_v1.operations.intent_executor import SubmissionAdmission, clean_environment
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled, utc
from agentmemeval.path1_v1.operations.submission_process import (
    file_sha256,
    read_json,
    write_exclusive_json,
)


class E1PreSubmitVerifier:
    def __init__(self, evidence_root: Path, *, partition_command,
                 min_free_bytes: int = 1024**3, fixture_mode=False,
                 clock=lambda: datetime.now(UTC)):
        if (not evidence_root.is_absolute() or evidence_root.resolve() != evidence_root
                or not evidence_root.is_dir()
                or type(min_free_bytes) is not int or min_free_bytes < 1):
            raise ValueError("invalid E1 evidence root/disk reserve")
        self.root = evidence_root
        self.command = partition_command
        self.min_free_bytes = min_free_bytes
        self.fixture_mode = fixture_mode
        self.clock = clock

    def __call__(self, request, prepared):
        if type(prepared) is not ValidatedE1Prepared or prepared.contract["phase"] != "E1":
            raise SubmissionDisabled("E1 producer requires a validated E1 prepared entry")
        current = read_prepared(prepared.prepared_path,
                                model_verification=prepared.model_verification)
        if (current.descriptor != prepared.descriptor
                or current.submission != request):
            raise SubmissionDisabled("E1 descriptor/submission changed before preflight")
        pins = self.command.validate(fixture_mode=self.fixture_mode)
        if not self.fixture_mode and Path(self.command.argv_prefix[0]).name != "scontrol":
            raise SubmissionDisabled("E1 requires the pinned site scontrol")
        observed_at = utc(self.clock())
        argv = (*self.command.argv_prefix, f"--clusters={request.spec.cluster}",
                "show", "partition", request.spec.partition, "--oneliner")
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                env=clean_environment(), timeout=20, check=False)
        if self.command.validate(fixture_mode=self.fixture_mode) != pins:
            raise SubmissionDisabled("partition command changed during observation")
        raw = result.stdout.decode("utf-8", errors="strict")
        pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", raw)
        fields = dict(pairs)
        if (result.returncode != 0 or result.stderr.strip() or len(pairs) != len(fields)
                or fields.get("PartitionName") != request.spec.partition
                or fields.get("State") != "UP"):
            raise SubmissionDisabled("E1 partition is unavailable/ambiguous")
        limit = fields.get("MaxTime", "")
        match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", limit)
        if limit == "UNLIMITED":
            seconds = 86400 * 365
        elif match and int(match[3]) < 60 and int(match[4]) < 60:
            seconds = (int(match[1] or 0) * 86400 + int(match[2]) * 3600
                       + int(match[3]) * 60 + int(match[4]))
        else:
            raise SubmissionDisabled("E1 partition walltime is unreadable")
        cpus = fields.get("MaxCPUsPerNode", "UNLIMITED")
        if (seconds < 14400 or cpus != "UNLIMITED"
                and (not cpus.isdigit() or 0 < int(cpus) < 8)):
            raise SubmissionDisabled("E1 partition cannot accommodate fixed CPU/walltime")
        output_root = Path(prepared.request["output_root"])
        if (prepared.output_directory.exists() or not output_root.is_dir()
                or not os.access(output_root, os.W_OK | os.X_OK)):
            raise SubmissionDisabled("E1 output is unavailable")
        disk = shutil.disk_usage(output_root)
        if disk.free < self.min_free_bytes:
            raise SubmissionDisabled("E1 disk reserve is insufficient")
        health_path = output_root / ("e1_preflight_" + uuid4().hex + ".json")
        health = {"request_sha256": request.request_sha256,
                  "worker_request_sha256": prepared.request_sha256,
                  "runtime_admission": False}
        write_exclusive_json(health_path, health)
        if read_json(health_path) != health:
            raise SubmissionDisabled("E1 filesystem readback differs")
        evidence = {"schema_version": 1, "kind": "path1_e1_presubmit_observation",
            "request_sha256": request.request_sha256,
            "worker_request_sha256": prepared.request_sha256,
            "contract_sha256": prepared.request["contract_sha256"],
            "runtime_identity_sha256": content_digest(prepared.contract["runtime_identity"]),
            "prepared": {"path": str(prepared.prepared_path),
                         "sha256": file_sha256(prepared.prepared_path)},
            "observed_at": observed_at.isoformat(), "argv": list(argv),
            "command_files": pins, "partition_stdout": raw,
            "disk": {"root": str(output_root), "free": disk.free,
                     "minimum_free_bytes": self.min_free_bytes},
            "filesystem_probe": {"path": str(health_path), "sha256": file_sha256(health_path)},
            "resource_profile": prepared.descriptor["resource_profile"],
            "python_identity": prepared.descriptor["python_identity"],
            "stage_cap": 1, "gpu_allocation_verified": False,
            "actual_memory_verified": False, "runtime_admission": False}
        destination = self.root / (uuid4().hex + ".json")
        write_exclusive_json(destination, evidence)
        if read_json(destination) != evidence:
            raise SubmissionDisabled("E1 preflight durable readback differs")
        admission = SubmissionAdmission(request.request_sha256, prepared.request_sha256,
            prepared.request["contract_sha256"], file_sha256(destination), 1, observed_at)
        admission.validate(request, prepared, self.clock())
        return admission
