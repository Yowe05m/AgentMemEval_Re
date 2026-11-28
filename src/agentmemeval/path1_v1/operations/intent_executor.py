"""Durable intent executor for a local Slurm submission host.

The same code is tested with literal-argv subprocess fixtures on Windows. Live
Slurm execution requires Linux, verified command bytes, a real worker request,
fresh source/contract checks, trusted pre-submission admission and account-wide
capacity. All write gates default closed. No SSH or service deployment is done.

Local durability means at-most-one attempt to call sbatch, not exactly-once job
execution. A crash/timeout after submit_unknown never permits reissue. Kernel
process creation markers are diagnostic; PID reuse/absence is not negative proof.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from agentmemeval.path1_v1.operations.contracts import (
    build_runtime_identity,
    content_digest,
)
from agentmemeval.path1_v1.operations.process_ownership import (
    OwnedProcess,
    OwnershipError,
    ProcessBinding,
    ReceiptSinkError,
)
from agentmemeval.path1_v1.operations.slurm import (
    AccountSnapshot,
    CommandResult,
    IntentReceipt,
    SlurmError,
    SubmissionDisabled,
    SubmissionRequest,
    SubmitSpec,
    digest,
    utc,
)
from agentmemeval.path1_v1.operations.state import (
    CampaignStore,
    RegistrySnapshot,
    StoreIntegrityError,
)
from agentmemeval.path1_v1.operations.submission_process import (
    current_effective_user,
    file_sha256,
    observe_process,
    process_identity,
    read_json,
    write_exclusive_json,
)

NAMESPACE = "durable_intent_executor_v1"


class ExecutorError(SlurmError):
    pass


def verify_effective_user(expected_name: str) -> dict[str, Any]:
    """Bind SubmitSpec.user to effective kernel credentials and the OS account DB."""
    try:
        import pwd

        observed = current_effective_user()
        named = pwd.getpwnam(expected_name)
        if (
            type(observed["uid"]) is not int
            or observed["name"] != expected_name
            or named.pw_uid != observed["uid"]
            or named.pw_name != expected_name
        ):
            raise SubmissionDisabled("effective OS account does not match SubmitSpec.user")
        return observed
    except (ImportError, OSError, KeyError) as exc:
        raise SubmissionDisabled("effective submission account cannot be verified") from exc


def verify_loaded_package(repository_root: Path, source_records: list[dict]) -> None:
    """Check every currently loaded project module's actual origin and file bytes.

    This closes mixed-checkout imports; it is not runtime-memory attestation or
    a claim about future imports. Future project files remain covered by the
    complete manifest and fixed source path, with in-job checks still required.
    """
    repository = repository_root.resolve(strict=True)
    package_root = repository / "src/agentmemeval"
    declared = {item["path"]: item["sha256"] for item in source_records}
    required = {
        "agentmemeval.path1_v1.operations.contracts",
        "agentmemeval.path1_v1.operations.state",
        "agentmemeval.path1_v1.operations.slurm",
        "agentmemeval.path1_v1.operations.worker",
        "agentmemeval.path1_v1.operations.intent_executor",
        "agentmemeval.path1_v1.operations.submission_process",
    }
    modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "agentmemeval" or name.startswith("agentmemeval.")
    }
    if not required <= modules.keys():
        raise ExecutorError("required execution dependency is not loaded")
    for name, module in modules.items():
        raw_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if (
            raw_file is None
            and origin is None
            and isinstance(getattr(spec, "loader", None), importlib.machinery.NamespaceLoader)
        ):
            expected = repository / "src" / Path(*name.split("."))
            locations = list(getattr(module, "__path__", ()))
            prefix = expected.relative_to(repository).as_posix() + "/"
            if (
                len(locations) != 1
                or Path(locations[0]).resolve(strict=True) != expected
                or not any(path.startswith(prefix) for path in declared)
            ):
                raise ExecutorError(f"namespace package escapes the source closure: {name}")
            continue
        if type(raw_file) is not str or type(origin) is not str:
            raise ExecutorError(f"loaded module has no concrete source origin: {name}")
        source = _file(Path(raw_file))
        if not source.is_relative_to(package_root) or _file(Path(origin)) != source:
            raise ExecutorError(f"loaded dependency comes from another source tree: {name}")
        relative = source.relative_to(repository).as_posix()
        if declared.get(relative) != file_sha256(source):
            raise ExecutorError(f"loaded dependency bytes are outside the contract closure: {name}")
        for location in getattr(module, "__path__", ()):
            if Path(location).resolve(strict=True) != source.parent:
                raise ExecutorError(f"package search path escapes the declared source: {name}")


def _file(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ExecutorError("executor file paths must be absolute")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ExecutorError("symlink executor paths are unsupported")
        current = current.parent
    if not path.is_file():
        raise ExecutorError("required executor input is not a file")
    return path.resolve()


def clean_environment(additional: Mapping[str, str] | None = None) -> dict[str, str]:
    """No credentials, SBATCH overrides, shell hooks, user Python path or startup."""
    environment = {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"}
    if os.name == "nt" and os.environ.get("SystemRoot"):
        environment["SystemRoot"] = os.environ["SystemRoot"]
    allowed = {"LC_ALL", "LANG", "TZ", "SLURM_TIME_FORMAT", "SLURM_CONF"}
    for key, value in (additional or {}).items():
        if key not in allowed or type(value) is not str or "\x00" in value:
            raise ExecutorError("unapproved child environment variable")
        environment[key] = value
    return environment


@dataclass(frozen=True)
class VerifiedCommand:
    """Explicit pinned executable and (fixture-only) interpreter/script prefix."""

    argv_prefix: tuple[str, ...]
    files: tuple[tuple[str, str], ...]

    def validate(self, *, fixture_mode: bool = False) -> list[dict[str, str]]:
        if (
            type(self.argv_prefix) is not tuple
            or not self.argv_prefix
            or any(type(x) is not str or "\x00" in x for x in self.argv_prefix)
        ):
            raise ExecutorError("invalid command prefix")
        executable = _file(Path(self.argv_prefix[0]))
        if not fixture_mode and len(self.argv_prefix) != 1:
            raise ExecutorError("production commands do not accept hidden prefix arguments")
        records = []
        seen = set()
        for name, expected in self.files:
            path = _file(Path(name))
            digest(expected)
            if path in seen or file_sha256(path) != expected:
                raise ExecutorError("command executable/script content changed")
            seen.add(path)
            records.append({"path": str(path), "sha256": expected})
        if executable not in seen:
            raise ExecutorError("command executable is not content-pinned")
        for argument in self.argv_prefix[1:]:
            if (
                Path(argument).is_absolute()
                and Path(argument).is_file()
                and Path(argument).resolve() not in seen
            ):
                raise ExecutorError("fixture script is not content-pinned")
        return records


class LocalCommandTransport:
    """Real subprocess reads for SlurmAdapter; sbatch/scancel are not exposed here."""

    def __init__(self, commands: Mapping[str, VerifiedCommand], *, fixture_mode: bool = False):
        if not set(commands) <= {"squeue", "sacct", "sacctmgr", "sinfo"}:
            raise ExecutorError("read transport cannot execute mutating commands")
        self.commands, self.fixture_mode = dict(commands), fixture_mode

    def run(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        inherit_environment: bool,
        timeout_seconds: float,
    ) -> CommandResult:
        if inherit_environment is not False or argv[0] not in self.commands:
            raise ExecutorError("unknown command or inherited environment")
        _timeout(timeout_seconds)
        target = self.commands[argv[0]]
        target.validate(fixture_mode=self.fixture_mode)
        result = subprocess.run(
            (*target.argv_prefix, *argv[1:]),
            shell=False,
            env=clean_environment(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            result.returncode,
            result.stdout.decode("utf-8", errors="strict"),
            result.stderr.decode("utf-8", errors="replace"),
        )


def _timeout(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 60:
        raise ValueError("timeout_seconds must be finite and in (0, 60]")


def render_worker_script(
    *, python_executable: Path, repository_root: Path, worker_request_path: Path
) -> bytes:
    """The only accepted topology: one fixed worker invocation; no SBATCH directives."""
    for path in (python_executable, repository_root, worker_request_path):
        if not path.is_absolute() or any(c in str(path) for c in "\x00\r\n"):
            raise ExecutorError("worker launcher paths must be absolute and single-line")
    text = (
        "#!/bin/sh\nset -eu\nexport PYTHONPATH="
        + shlex.quote(str(repository_root / "src"))
        + "\nexec "
        + shlex.quote(str(python_executable))
        + " -B -s -m agentmemeval.path1_v1.operations.worker --request "
        + shlex.quote(str(worker_request_path))
        + "\n"
    )
    return text.encode("utf-8")


def render_probe_script(
    *, python_executable: Path, repository_root: Path, probe_request_path: Path,
    native_runtime: dict | None = None,
) -> bytes:
    """Separate exact E0 topology: request v2 carries every execution parameter."""
    for path in (python_executable, repository_root, probe_request_path):
        if not path.is_absolute() or any(c in str(path) for c in "\x00\r\n"):
            raise ExecutorError("probe launcher paths must be absolute and single-line")
    if native_runtime is not None:
        from agentmemeval.path1_v1.operations.native_runtime import bootstrap_argv
        argv = bootstrap_argv(python_executable, repository_root, native_runtime,
            module="agentmemeval.path1_v1.operations.probe_launcher",
            arguments=["--request", str(probe_request_path)])
        return ("#!/bin/sh\nset -eu\nexec " + shlex.join(argv) + "\n").encode("utf-8")
    return ("#!/bin/sh\nset -eu\nexport PYTHONPATH="
            + shlex.quote(str(repository_root / "src")) + "\nexec "
            + shlex.quote(str(python_executable))
            + " -B -s -m agentmemeval.path1_v1.operations.probe_launcher --request "
            + shlex.quote(str(probe_request_path)) + "\n").encode("utf-8")


@dataclass(frozen=True)
class LaunchFiles:
    worker_request_path: Path
    python_executable: Path
    python_sha256: str


@dataclass(frozen=True)
class ProbeLaunchFiles:
    probe_request_path: Path


RESOURCE_QUEUE_START_SECONDS = 3600
RESOURCE_DEADLINE_OFFSET_SECONDS = RESOURCE_QUEUE_START_SECONDS + 120


def resource_queue_policy(config: dict, *, allow_legacy: bool = False) -> tuple[int, int]:
    """Current submit policy; the former 10/130 policy is read-only history."""
    expected = {"kind": "path1_e0_resource_site_config", "phase": "E0",
                "job_kind": "resource_probe", "walltime_seconds": 120,
                "command_seconds": 90, "cleanup_seconds": 30, "gpus": 1, "stage_cap": 1}
    if type(config) is not dict or any(
        type(config.get(key)) is not type(value) or config.get(key) != value
        for key, value in expected.items()
    ):
        raise ExecutorError("resource probe parameters differ from the fixed E0 protocol")
    pair = (config.get("queue_start_window_seconds"),
            config.get("scheduler_deadline_offset_seconds"))
    allowed = {(RESOURCE_QUEUE_START_SECONDS, RESOURCE_DEADLINE_OFFSET_SECONDS)}
    if allow_legacy:
        allowed.add((10, 130))
    if any(type(value) is not int for value in pair) or pair not in allowed:
        raise ExecutorError("resource queue policy is not admissible for this operation")
    return pair


@dataclass(frozen=True)
class SubmissionAdmission:
    """Verified pre-allocation evidence, NEVER a proof about an unallocated GPU."""

    request_sha256: str
    worker_request_sha256: str
    contract_sha256: str
    evidence_sha256: str
    stage_cap: int
    observed_at: datetime

    def validate(self, request: SubmissionRequest, worker: Any, now: datetime) -> None:
        for value in (
            self.request_sha256,
            self.worker_request_sha256,
            self.contract_sha256,
            self.evidence_sha256,
        ):
            digest(value)
        if (
            self.request_sha256 != request.request_sha256
            or self.worker_request_sha256 != worker.request_sha256
            or self.contract_sha256 != worker.request["contract_sha256"]
            or type(self.stage_cap) is not int
            or self.stage_cap not in (1, 2, 4, 6)
        ):
            raise SubmissionDisabled("pre-submit admission is not bound to this request")
        age = (utc(now) - utc(self.observed_at)).total_seconds()
        if age < 0 or age > 30:
            raise SubmissionDisabled("pre-submit admission is stale")


class E0PreSubmitVerifier:
    """Produce E0 launch admission from current files, disk and a real site query.

    This does not allocate a GPU, establish backend execution or replace the
    executor's independent account-wide capacity reader. Every observation is
    persisted exclusively; no external verified JSON is consumed.
    """

    def __init__(self, evidence_root: Path, *, partition_command: VerifiedCommand,
                 min_free_bytes: int = 1024 ** 3, fixture_mode: bool = False,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        if not evidence_root.is_absolute() or not evidence_root.is_dir():
            raise ExecutorError("E0 preflight evidence root must be an existing absolute directory")
        if type(min_free_bytes) is not int or min_free_bytes <= 0:
            raise ExecutorError("positive explicit disk budget required")
        self.root, self.command = evidence_root, partition_command
        self.min_free_bytes, self.fixture_mode, self.clock = min_free_bytes, fixture_mode, clock

    def __call__(self, request: SubmissionRequest, probe: Any) -> SubmissionAdmission:
        from agentmemeval.path1_v1.operations.runtime_probe import (
            ValidatedProbeRequest,
            validate_probe_launch,
        )

        if type(probe) is not ValidatedProbeRequest or probe.request["schema_version"] != 2:
            raise SubmissionDisabled("E0 preflight cannot admit research or legacy requests")
        if (probe.intent_id != request.intent_id
                or probe.contract["campaign_id"] != request.campaign_id):
            raise SubmissionDisabled("E0 preflight intent mismatch")
        validate_probe_launch(probe.request["launch"], probe.contract["job"]["kind"])
        service_profile = None
        if probe.contract["job"]["kind"] == "service_probe":
            from agentmemeval.path1_v1.operations.service_profile import (
                read_service_config,
                validate_service_submission,
            )

            service_profile = validate_service_submission(
                request, probe, read_service_config(probe))
        pins = self.command.validate(fixture_mode=self.fixture_mode)
        if not self.fixture_mode and Path(self.command.argv_prefix[0]).name != "scontrol":
            raise SubmissionDisabled("E0 preflight requires pinned scontrol")
        argv = (*self.command.argv_prefix, f"--clusters={request.spec.cluster}", "show",
                "partition", request.spec.partition, "--oneliner")
        observed_at = utc(self.clock())
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                env=clean_environment(), timeout=20, check=False)
        raw = result.stdout.decode("utf-8", errors="strict")
        pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", raw)
        fields = dict(pairs)
        if (result.returncode != 0 or result.stderr.strip()
                or len(fields) != len(pairs)
                or fields.get("PartitionName") != request.spec.partition
                or fields.get("State") != "UP"):
            raise SubmissionDisabled("actual partition is unavailable or ambiguous")
        limit = fields.get("MaxTime", "")
        match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", limit)
        if limit == "UNLIMITED":
            seconds = 86400 * 365
        elif match and int(match[3]) < 60 and int(match[4]) < 60:
            seconds = (int(match[1] or 0) * 86400 + int(match[2]) * 3600
                       + int(match[3]) * 60 + int(match[4]))
        else:
            raise SubmissionDisabled("partition walltime could not be read")
        if seconds < request.spec.walltime_seconds:
            raise SubmissionDisabled("partition cannot accommodate the frozen probe walltime")
        if service_profile is not None:
            limits = [("MaxCPUsPerNode", request.spec.cpus_per_task)]
            if request.spec.memory_policy == "site_default":
                if (service_profile["profile_version"] != 3
                        or request.spec.memory_mib is not None
                        or type(request.spec.cpus_per_task) is not int):
                    raise SubmissionDisabled("invalid site-default service memory policy")
                # The allocation must later report actual RAM. No numeric memory
                # amount was requested, so MaxMemPerCPU cannot validate one here.
            elif (request.spec.memory_policy is None
                  and type(request.spec.memory_mib) is int
                  and type(request.spec.cpus_per_task) is int):
                limits.append(
                    ("MaxMemPerCPU", request.spec.memory_mib / request.spec.cpus_per_task)
                )
            else:
                raise SubmissionDisabled("service memory is unspecified without site_default")
            for field, requested in limits:
                value = fields.get(field)
                if value is not None and value != "UNLIMITED" and (
                    not value.isdigit() or (int(value) != 0 and requested > int(value))
                ):
                    raise SubmissionDisabled("partition cannot accommodate service " + field)
        output_root = Path(probe.request["output_root"])
        if (probe.output_directory.exists() or not output_root.is_dir()
                or not os.access(output_root, os.W_OK | os.X_OK)):
            raise SubmissionDisabled("E0 output is not exclusively available/writable")
        disk = shutil.disk_usage(output_root)
        if disk.free < self.min_free_bytes:
            raise SubmissionDisabled("E0 output has insufficient declared disk reserve")
        filesystem_record = output_root / ("e0_preflight_" + uuid4().hex + ".json")
        health = {"request_sha256": request.request_sha256,
                  "probe_request_sha256": probe.request_sha256,
                  "purpose": "E0 exclusive output filesystem roundtrip", "runtime_admission": False}
        write_exclusive_json(filesystem_record, health)
        if read_json(filesystem_record) != health:
            raise SubmissionDisabled("E0 output filesystem roundtrip differs")
        evidence = {"schema_version": 1, "kind": "path1_e0_presubmit_observation",
                    "request_sha256": request.request_sha256,
                    "probe_request_sha256": probe.request_sha256,
                    "runtime_identity_sha256": content_digest(probe.contract["runtime_identity"]),
                    "contract_sha256": probe.request["contract_sha256"],
                    "observed_at": observed_at.isoformat(), "argv": list(argv),
                    "command_files": pins, "partition_stdout": raw,
                    "disk": {"root": str(output_root), "total": disk.total,
                             "free": disk.free, "minimum_free_bytes": self.min_free_bytes},
                    "filesystem_probe": {"path": str(filesystem_record),
                                         "sha256": file_sha256(filesystem_record)},
                    "launch": probe.request["launch"], "stage_cap": 1,
                    "runtime_admission": False}
        if service_profile is not None:
            evidence["service_profile"] = service_profile
        # The write/fsync is also an actual evidence-filesystem health check.
        destination = self.root / (uuid4().hex + ".json")
        write_exclusive_json(destination, evidence)
        if read_json(destination) != evidence:
            raise SubmissionDisabled("E0 preflight durable readback differs")
        admission = SubmissionAdmission(request.request_sha256, probe.request_sha256,
                        probe.request["contract_sha256"], file_sha256(destination), 1, observed_at)
        admission.validate(request, probe, self.clock())
        return admission


class DurableIntentExecutor:
    def __init__(
        self,
        control_root: Path,
        *,
        cluster: str,
        account: str,
        command: VerifiedCommand,
        launch_files: Callable[[SubmissionRequest], LaunchFiles | ProbeLaunchFiles],
        pre_submit_verifier: Callable[[SubmissionRequest, Any], SubmissionAdmission] | None = None,
        account_reader: Callable[[], Any] | None = None,
        fixture_mode: bool = False,
        fixture_paths: Mapping[str, Path] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        transport_term_grace_seconds: float = 120,
    ) -> None:
        if not control_root.is_absolute():
            raise ExecutorError("control root must be absolute")
        scope = content_digest({"cluster": cluster, "account": account})
        # Fixed independent namespace: never opens the caller's existing campaign DB.
        # Short filesystem labels keep Windows fixture paths usable. Full scope
        # and intent identities remain in SQLite; any prefix collision fails the
        # scope check/exclusive directory creation, never aliases a claimed run.
        self.store = CampaignStore(control_root / "ie1" / scope[:16], "executor_" + scope[:32])
        self.cluster, self.account, self.command = cluster, account, command
        self.launch_files, self.pre_submit_verifier = launch_files, pre_submit_verifier
        self.account_reader, self.fixture_mode, self.clock = account_reader, fixture_mode, clock
        if (
            type(transport_term_grace_seconds) not in (int, float)
            or not math.isfinite(transport_term_grace_seconds)
            or not 0 <= transport_term_grace_seconds <= 120
        ):
            raise ExecutorError("transport TERM grace must be within 0..120 seconds")
        self.transport_term_grace_seconds = transport_term_grace_seconds
        if fixture_paths and not fixture_mode:
            raise ExecutorError("path remapping is restricted to explicit offline fixtures")
        self.fixture_paths = dict(fixture_paths or {})

    def _path(self, remote: str) -> Path:
        if self.fixture_mode:
            if remote not in self.fixture_paths:
                raise ExecutorError("offline fixture has no exact path mapping")
            path = self.fixture_paths[remote]
            if not path.is_absolute():
                raise ExecutorError("fixture path must be absolute")
            return path
        if not sys.platform.startswith("linux"):
            raise SubmissionDisabled("production submission requires a validated Linux host")
        return Path(remote)

    def _initialize(self) -> RegistrySnapshot:
        if not self.store.database_path.exists():
            return self.store.initialize(
                registry={
                    NAMESPACE: {
                        "schema_version": 1,
                        "cluster": self.cluster,
                        "account": self.account,
                        "intents": {},
                    }
                }
            )
        return self.store.read_registry()

    def _ledger(self, snapshot: RegistrySnapshot) -> dict:
        value = copy.deepcopy(snapshot.registry.get(NAMESPACE))
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "cluster", "account", "intents"}
            or value["schema_version"] != 1
            or value["cluster"] != self.cluster
            or value["account"] != self.account
            or type(value["intents"]) is not dict
        ):
            raise StoreIntegrityError("invalid executor scope/ledger")
        for intent, record in value["intents"].items():
            request = self._stored_request(record)
            if (
                intent != request.intent_id
                or record["request_sha256"] != request.request_sha256
                or record["status"] not in {"claimed", "submit_unknown", "accepted"}
            ):
                raise StoreIntegrityError("executor request identity/status changed")
            expected_dir = self.store.directory / "i" / intent[:16]
            if Path(record["invocation_dir"]) != expected_dir:
                raise StoreIntegrityError("invocation directory escaped its intent namespace")
            if record["status"] != "claimed":
                claims = [claim for claim in snapshot.claims if claim.intent_id == intent]
                if (
                    len(claims) != 1
                    or claims[0].status not in {"started", "failed", "completed"}
                    or claims[0].metadata
                    != {"purpose": "slurm_job_container", "request_sha256": request.request_sha256}
                    or claims[0].run_directory != str(self._path(request.spec.work_dir).resolve())
                ):
                    raise StoreIntegrityError("attempted submission has no matching consumed claim")
                if claims[0].status in {"failed", "completed"}:
                    from agentmemeval.path1_v1.operations.e0_reconcile import (
                        validate_e0_terminal_receipt,
                    )

                    # A scheduler row alone cannot close a consumed claim. Only
                    # independently sealed E0 terminal paths are accepted.
                    validate_e0_terminal_receipt(record, claims[0], store=self.store)
                digest(record["manifest_sha256"])
                manifest_path = expected_dir / "invocation.json"
                if file_sha256(_file(manifest_path)) != record["manifest_sha256"]:
                    raise StoreIntegrityError("durable invocation bytes changed")
            if type(record["submitter_finished"]) is not bool:
                raise StoreIntegrityError("invalid submitter completion evidence")
            if record.get("probe_request_path") is not None:
                path = _file(Path(record["probe_request_path"]))
                if file_sha256(path) != record.get("probe_request_file_sha256"):
                    raise StoreIntegrityError("durable E0 request bytes changed")
            if "operational_supersession" in record:
                from agentmemeval.path1_v1.operations.service_supersession import (
                    validate_stored_supersession,
                )

                validate_stored_supersession(record, self.store)
        return value

    @staticmethod
    def _stored_request(record: dict) -> SubmissionRequest:
        try:
            return SubmissionRequest(
                record["campaign_id"], record["intent_id"], SubmitSpec.from_payload(record["spec"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreIntegrityError("invalid persisted submission request") from exc

    def _commit(self, snapshot: RegistrySnapshot, ledger: dict, reason: str) -> RegistrySnapshot:
        return self.store.commit_transition(
            expected_generation=snapshot.generation,
            registry={NAMESPACE: ledger},
            state=snapshot.state,
            reason=reason,
        )

    def _validate_launch(
        self, request: SubmissionRequest, *, e0_model_verification=None,
        e1_model_verification=None,
    ) -> tuple[Any, LaunchFiles, bytes, list[dict], dict | None]:
        # Lazy import avoids making package import an execution/admission action.
        from agentmemeval.path1_v1.operations.e1_prepare import (
            E1LaunchFiles,
            read_prepared,
            render_e1_script,
        )
        from agentmemeval.path1_v1.operations.worker import validate_worker_request

        if (request.spec.cluster, request.spec.account) != (self.cluster, self.account):
            raise ExecutorError("submission targets another executor scope")
        effective_user = None if self.fixture_mode else verify_effective_user(request.spec.user)
        if request.spec.script_args:
            raise ExecutorError("worker launcher takes no extra script arguments")
        files = self.launch_files(request)
        if type(files) is ProbeLaunchFiles:
            if e1_model_verification is not None:
                raise ExecutorError("E0 launcher cannot borrow E1 model verification")
            return self._validate_probe_launch(
                request, files, effective_user, model_verification=e0_model_verification
            )
        if type(files) not in (LaunchFiles, E1LaunchFiles):
            raise ExecutorError("launcher must explicitly select research or E0 topology")
        if e0_model_verification is not None:
            raise ExecutorError("research launcher cannot borrow E0 model verification")
        prepared = None
        if type(files) is E1LaunchFiles:
            prepared = read_prepared(files.prepared_path,
                                     model_verification=e1_model_verification)
            if prepared.submission != request:
                raise ExecutorError("E1 prepared submission differs from exact descriptor")
            proof = prepared.descriptor["python_identity"]
            python = Path(proof["entry_path"])
            worker_path = _file(Path(prepared.descriptor["worker_request_path"]))
            worker = prepared
        elif request.spec.cpus_per_task is not None or request.spec.memory_mib is not None:
            raise ExecutorError("explicit host resources are currently service E0 only")
        else:
            python = _file(files.python_executable)
            digest(files.python_sha256)
            if file_sha256(python) != files.python_sha256:
                raise ExecutorError("worker interpreter content changed")
            worker_path = _file(files.worker_request_path)
            worker = validate_worker_request(worker_path, model_verification=e1_model_verification)
        if (
            worker.intent_id != request.intent_id
            or worker.contract["campaign_id"] != request.campaign_id
        ):
            raise ExecutorError("worker request is bound to a different campaign/intent")
        repository = Path(worker.request["repository_root"]).resolve(strict=True)
        identity = worker.contract["runtime_identity"]
        model_arguments = {}
        if worker.contract["phase"] == "E1":
            model_arguments = {
                "model_verification": worker.model_verification,
                "model_binding_sha256": worker.request_sha256,
            }
        observed = build_runtime_identity(
            repository,
            core_paths=[x["path"] for x in identity["core_files"]],
            operations_paths=[x["path"] for x in identity["operations_files"]],
            config_path=identity["config"]["path"],
            dependency_lock_path=identity["dependency_lock"]["path"],
            model_manifest_path=worker.request["model_manifest_path"],
            model_root=worker.request["model_root"],
            **model_arguments,
        )
        if observed != identity:
            raise ExecutorError("current source/model identity differs from planned worker")
        source_records = identity["core_files"] + identity["operations_files"]
        paths = {item["path"] for item in source_records}
        if prepared is not None and not {
            "src/agentmemeval/path1_v1/operations/" + name + ".py"
            for name in ("e1_prepare", "e1_launcher", "e1_preflight", "interpreters",
                         "intent_executor", "submission_process", "worker", "user_capacity")
        } <= paths:
            raise ExecutorError("E1 prepared execution module closure is incomplete")
        if (
            "src/agentmemeval/path1_v1/operations/worker.py" not in paths
            or identity["dirty"]
            or identity["untracked_declared_paths"]
            or identity["uncovered_execution_paths"]
            or identity["untracked_execution_paths"]
            or not identity["content_matches_head"]
        ):
            raise ExecutorError(
                "submission requires a clean complete tracked worker source identity"
            )
        if not self.fixture_mode:
            verify_loaded_package(repository, source_records)
        job_dir = Path(worker.request["output_root"]) / Path(
            *PurePosixPath(worker.contract["job"]["output_relpath"]).parts
        )
        if self._path(request.spec.work_dir).resolve() != job_dir.resolve():
            raise ExecutorError("Slurm working directory is not the worker job container")
        expected_script = (
            render_e1_script(python_executable=python, repository_root=repository,
                             prepared_path=prepared.prepared_path,
                             native_runtime=(prepared.runtime_plan or {}).get("native_runtime"))
            if prepared is not None else render_worker_script(
                python_executable=python, repository_root=repository,
                worker_request_path=worker_path))
        script_path = _file(self._path(request.spec.script_path))
        if prepared is not None and script_path != Path(prepared.descriptor["script_file"]):
            raise ExecutorError("E1 script path differs from prepared artifact")
        script = script_path.read_bytes()
        if (
            hashlib.sha256(script).hexdigest() != request.spec.script_sha256
            or script != expected_script
        ):
            raise ExecutorError("launcher bytes/topology differ from the exact one-worker template")
        pinned = [
            {"path": str(repository / item["path"]), "sha256": item["sha256"]}
            for item in source_records + [identity["config"], identity["dependency_lock"]]
        ]
        pinned += [
            {"path": str(worker_path), "sha256": file_sha256(worker_path)},
            {"path": str(script_path), "sha256": request.spec.script_sha256},
            {"path": worker.request["contract_path"], "sha256": worker.request["contract_sha256"]},
            {"path": worker.request["claim_path"], "sha256": worker.request["claim_sha256"]},
            {
                "path": worker.request["seed_inventory_path"],
                "sha256": worker.request["seed_inventory_sha256"],
            },
        ]
        if prepared is None:
            pinned.append({"path": str(python), "sha256": files.python_sha256})
        else:
            if (prepared.runtime_plan or {}).get("native_runtime") is not None:
                pinned.append(dict(prepared.runtime_plan["native_runtime"]))
            if (prepared.runtime_plan or {}).get("cuda_toolkit") is not None:
                from agentmemeval.path1_v1.operations.cuda_toolkit import input_pin
                pinned.append(input_pin(prepared.runtime_plan["cuda_toolkit"]))
            pinned += [{"path": row["path"], "sha256": row["sha256"]}
                       for row in proof["files"]]
            pinned.append({"path": str(prepared.prepared_path),
                           "sha256": file_sha256(prepared.prepared_path)})
        for path_key, sha_key in (("runtime_gate_path", "runtime_gate_sha256"),):
            if worker.request[path_key] is not None:
                pinned.append({"path": worker.request[path_key], "sha256": worker.request[sha_key]})
        if worker.request["model_manifest_path"] is not None:
            pinned.append(
                {
                    "path": worker.request["model_manifest_path"],
                    "sha256": identity["model"]["manifest_sha256"],
                }
            )
        if worker.contract["job"]["kind"] == "evaluation":
            artifact = worker.contract["job"]["formation_receipt"]["snapshot"]
            pinned.append(
                {
                    "path": str(Path(worker.request["artifact_root"]) / artifact["path"]),
                    "sha256": artifact["sha256"],
                }
            )
        return worker, files, script, pinned, effective_user

    def _validate_probe_launch(self, request, files, effective_user, *, model_verification=None):
        from agentmemeval.path1_v1.operations.runtime_probe import (
            _strict_json,
            validate_probe_request,
        )

        path = _file(files.probe_request_path)
        probe = validate_probe_request(path, model_verification=model_verification)
        if probe.request["schema_version"] != 2:
            raise ExecutorError("automatic E0 submission requires request v2")
        kind = probe.contract["job"]["kind"]
        expected_walltime = {"resource_probe": 120, "service_probe": 7200}[kind]
        if (probe.intent_id != request.intent_id
                or probe.contract["campaign_id"] != request.campaign_id
                or request.spec.walltime_seconds != expected_walltime):
            raise ExecutorError("E0 intent/campaign/fixed walltime differs")
        launch = probe.request["launch"]
        repository = Path(probe.request["repository_root"])
        identity = probe.contract["runtime_identity"]
        if kind == "resource_probe":
            resource_queue_policy(_strict_json(
                _file(repository / identity["config"]["path"]).read_bytes()))
            if request.spec.cpus_per_task is not None or request.spec.memory_mib is not None:
                raise ExecutorError("v2 resource probe cannot change its historical resources")
        else:
            from agentmemeval.path1_v1.operations.service_profile import (
                read_service_config,
                validate_service_submission,
            )

            validate_service_submission(request, probe, read_service_config(probe))
        source_records = identity["core_files"] + identity["operations_files"]
        required = {"src/agentmemeval/path1_v1/operations/" + name + ".py"
                    for name in ("worker", "runtime_probe", "probe_launcher", "resource_probe",
                                 "intent_executor", "process_ownership", "submission_process")}
        if not required <= {row["path"] for row in source_records}:
            raise ExecutorError("E0 execution module closure is incomplete")
        if not self.fixture_mode:
            verify_loaded_package(repository, source_records)
        job_dir = probe.output_directory.parent
        if self._path(request.spec.work_dir).resolve() != job_dir.resolve():
            raise ExecutorError("Slurm working directory is not the E0 container")
        script_path = _file(self._path(request.spec.script_path))
        script = script_path.read_bytes()
        if (script != render_probe_script(python_executable=Path(launch["python_executable"]),
                repository_root=repository, probe_request_path=path,
                native_runtime=launch.get("native_runtime"))
                or file_sha256(script_path) != request.spec.script_sha256):
            raise ExecutorError("E0 launcher bytes/topology differ from the fixed template")
        pinned = [{"path": str(repository / item["path"]), "sha256": item["sha256"]}
                  for item in source_records + [identity["config"], identity["dependency_lock"]]]
        pinned += [{"path": str(path), "sha256": file_sha256(path)},
                   {"path": str(script_path), "sha256": request.spec.script_sha256},
                   {"path": probe.request["contract_path"],
                    "sha256": probe.request["contract_sha256"]}]
        pinned += [{"path": row["path"], "sha256": row["sha256"]}
                   for row in launch["python_identity"]["files"]]
        if "native_runtime" in launch:
            pinned.append(dict(launch["native_runtime"]))
        if "cuda_toolkit" in launch:
            from agentmemeval.path1_v1.operations.cuda_toolkit import input_pin
            pinned.append(input_pin(launch["cuda_toolkit"]))
        if kind == "service_probe":
            pinned += [{"path": row["path"], "sha256": row["sha256"]}
                       for row in launch["service_python_identity"]["files"]]
        if probe.request["model_manifest_path"] is not None:
            pinned.append({"path": probe.request["model_manifest_path"],
                           "sha256": identity["model"]["manifest_sha256"]})
        return probe, files, script, pinned, effective_user

    def _admit(self, request: SubmissionRequest, worker: Any) -> SubmissionAdmission:
        from agentmemeval.path1_v1.operations.e1_prepare import ValidatedE1Prepared

        self._validate_e0_retry(worker)
        if self.pre_submit_verifier is None:
            raise SubmissionDisabled("no trusted pre-submit admission verifier is installed")
        admission = self.pre_submit_verifier(request, worker)
        if not isinstance(admission, SubmissionAdmission):
            raise SubmissionDisabled("pre-submit verifier returned no typed evidence")
        admission.validate(request, worker, self.clock())
        if type(worker) is ValidatedE1Prepared and admission.stage_cap != 1:
            raise SubmissionDisabled("prepared E1 submission is restricted to stage one")
        return admission

    def _validate_e0_retry(self, worker: Any):
        contract = worker.contract
        if contract["phase"] != "E0":
            return
        if contract["job"]["kind"] == "service_probe" and contract["job"]["attempt"] > 1:
            from agentmemeval.path1_v1.operations.service_retry import validate_service_retry

            if type(getattr(worker, "request", None)) is not dict:
                raise SubmissionDisabled(
                    "E0 retry lineage/campaign/attempt differs: missing service request")
            validate_service_retry(self.store, contract, worker.request)
            return
        from agentmemeval.path1_v1.operations.contracts import canonical_intent_id
        from agentmemeval.path1_v1.operations.e0_reconcile import resolve_e0_terminal_intent

        ledger = self._ledger(self.store.read_registry())
        retry = contract["job"]["retry_evidence"]
        supersession = None
        for intent, record in ledger["intents"].items():
            if intent == worker.intent_id or record.get("probe_request_path") is None:
                continue
            stored = self._stored_probe_contract(record)
            if (stored["campaign_id"] != contract["campaign_id"]
                    or stored["job"]["kind"] != contract["job"]["kind"]):
                continue
            old_retry = stored["job"]["retry_evidence"]
            if (contract["job"]["kind"] == "service_probe"
                    and contract["job"]["attempt"] == stored["job"]["attempt"] == 1
                    and retry is None and old_retry is None):
                from agentmemeval.path1_v1.operations.service_supersession import (
                    resolve_service_supersession,
                )

                if supersession is not None:
                    raise SubmissionDisabled("another service transition consumes this unit")
                supersession = resolve_service_supersession(
                    self.store, contract, worker.request, record)
                continue
            if (stored["job"]["attempt"] == contract["job"]["attempt"]
                    or (retry is not None and old_retry is not None
                        and retry["prior_intent_id"] == old_retry["prior_intent_id"])):
                raise SubmissionDisabled("E0 attempt or prior terminal already has another intent")
        if contract["job"]["attempt"] == 1:
            current = ledger["intents"].get(worker.intent_id)
            expected = supersession.payload() if supersession is not None else None
            if current is not None and (
                ("operational_supersession" in current) != (expected is not None)
                or current.get("operational_supersession") != expected
            ):
                raise SubmissionDisabled("saved service transition differs from live evidence")
            return supersession
        resolution = resolve_e0_terminal_intent(self.store, retry["prior_intent_id"])
        if getattr(resolution, "classification", None) != "unstarted_scheduler_deadline":
            raise SubmissionDisabled("executed resource terminal does not grant retry eligibility")
        if retry != {
            "prior_intent_id": resolution.intent_id,
            "prior_terminal_receipt_sha256": resolution.prior_terminal_receipt_sha256,
            "eligibility_receipt_sha256": resolution.eligibility_receipt_sha256,
            "scope": "complete_unit",
            "seed_changed": False,
        }:
            raise SubmissionDisabled("E0 retry evidence differs from durable terminal seal")
        previous = ledger["intents"][resolution.intent_id]
        resolution.validate(previous, store=self.store)
        prior = self._stored_probe_contract(previous)
        if (canonical_intent_id(prior) != resolution.intent_id
                or prior["campaign_id"] != contract["campaign_id"]
                or prior["phase"] != "E0"
                or prior["job"]["kind"] != "resource_probe"
                or contract["job"]["kind"] != "resource_probe"
                or prior["job"]["attempt"] + 1 != contract["job"]["attempt"]
                or prior["seed_namespace"] != contract["seed_namespace"]):
            raise SubmissionDisabled("E0 retry lineage/campaign/attempt differs")

    @staticmethod
    def _stored_probe_contract(previous: dict) -> dict:
        from agentmemeval.path1_v1.operations.contracts import (
            canonical_intent_id,
            validate_campaign_contract,
        )

        probe_path = _file(Path(previous["probe_request_path"]))
        if file_sha256(probe_path) != previous["probe_request_file_sha256"]:
            raise StoreIntegrityError("prior E0 probe bytes changed")
        probe = read_json(probe_path)
        if (probe["intent_id"] != previous["intent_id"]
                or probe["request_sha256"] != previous["worker_request_sha256"]
                or probe["request_sha256"] != content_digest({
                    key: value for key, value in probe.items() if key != "request_sha256"
                })):
            raise StoreIntegrityError("stored E0 probe request identity changed")
        contract_path = _file(Path(probe["contract_path"]))
        if file_sha256(contract_path) != probe["contract_sha256"]:
            raise StoreIntegrityError("prior E0 contract bytes changed")
        prior = validate_campaign_contract(read_json(contract_path))
        if (canonical_intent_id(prior) != previous["intent_id"] or prior["phase"] != "E0"
                or prior["campaign_id"] != previous["campaign_id"]):
            raise StoreIntegrityError("stored E0 probe contract identity changed")
        return prior

    def _capacity(self, request: SubmissionRequest, ledger: dict, stage_cap: int,
                  *, validated_phase: str | None = None, validated_e1=None) -> dict:
        if self.account_reader is None:
            raise SubmissionDisabled("no live account capacity reader is installed")
        resolved_e0 = {}
        if validated_phase == "E0" or validated_e1 is not None:
            from agentmemeval.path1_v1.operations.e0_reconcile import (
                validate_e0_terminal_receipt,
            )

            snapshot = self.store.read_registry()
            for claim in snapshot.claims:
                if claim.status in {"failed", "completed"} and claim.intent_id in ledger["intents"]:
                    resolved_e0[claim.intent_id] = validate_e0_terminal_receipt(
                        ledger["intents"][claim.intent_id], claim, store=self.store
                    )
        observed = self.account_reader()
        if not isinstance(observed, AccountSnapshot):
            from agentmemeval.path1_v1.operations.user_capacity import (
                UserAssociationSnapshot,
                validate_user_association_capacity,
            )

            if type(observed) is UserAssociationSnapshot:
                result = validate_user_association_capacity(
                    observed, request, ledger, stage_cap, self.clock(), validated_phase,
                    resolved_e0=resolved_e0, store=self.store, validated_e1=validated_e1)
                # Raw archive/source verification can be slow. Capacity must
                # still be fresh after that I/O, immediately before return.
                finished_at = utc(self.clock())
                if not (
                    utc(observed.observed_at) <= utc(observed.completed_at) <= finished_at
                    and 0 <= (finished_at - utc(observed.observed_at)).total_seconds() <= 30
                ):
                    raise SubmissionDisabled("capacity verification outlived snapshot freshness")
                return result
        if (
            not isinstance(observed, AccountSnapshot)
            or observed.capacity is None
            or (observed.cluster, observed.account) != (self.cluster, self.account)
        ):
            raise SubmissionDisabled("account capacity is not independently available")
        observed.capacity.validate(
            cluster=self.cluster, account=self.account, now=self.clock(), max_age=30
        )
        age = (utc(self.clock()) - utc(observed.observed_at)).total_seconds()
        if not 0 <= age <= 30:
            raise SubmissionDisabled("account snapshot is stale")
        jobs = observed.allocations()
        if any(job.matches(request) for job in jobs):
            raise SubmissionDisabled("this intent already appears in scheduler evidence")
        active = {job.key: job for job in jobs if job.active}
        campaign = {
            job.key
            for job in active.values()
            if job.comment.startswith("ame-v1:" + request.campaign_id + ":")
        }
        unknown_account = unknown_campaign = 0
        strict_occupancy = validated_phase == "E0" or validated_e1 is not None
        for intent, record in ledger["intents"].items():
            if intent == request.intent_id or (
                record["status"] == "claimed" and not strict_occupancy
            ):
                continue
            previous = self._stored_request(record)
            represented = [job for job in jobs if job.matches(previous)]
            if intent in resolved_e0:
                resolved_e0[intent].validate(record, store=self.store)
                expected_state = getattr(resolved_e0[intent], "scheduler_state", "DEADLINE")
                if any(job.state != expected_state or job.job_id != resolved_e0[intent].job_id
                       for job in represented):
                    raise SubmissionDisabled("scheduler contradicts sealed E0 terminal")
                continue
            if not represented or (strict_occupancy and not any(
                job.active for job in represented
            )):
                unknown_account += 1
                unknown_campaign += previous.campaign_id == request.campaign_id
        total = len(active) + unknown_account
        own = len(campaign) + unknown_campaign
        available = min(stage_cap - own, min(observed.capacity.max_submit_jobs, 8) - total - 2)
        if available < 1:
            raise SubmissionDisabled("fresh account/stage capacity has no admission slot")
        finished_at = utc(self.clock())
        if not 0 <= (finished_at - utc(observed.observed_at)).total_seconds() <= 30:
            raise SubmissionDisabled("capacity verification outlived snapshot freshness")
        observed.capacity.validate(
            cluster=self.cluster, account=self.account, now=finished_at, max_age=30
        )
        return {
            "snapshot_sha256": observed.evidence_sha256,
            "limit_evidence_sha256": observed.capacity.evidence_sha256,
            "observed_at": utc(observed.observed_at).isoformat(),
            "account_occupancy": total,
            "campaign_occupancy": own,
            "stage_cap": stage_cap,
            "limit": min(observed.capacity.max_submit_jobs, 8),
            "reserve": 2,
        }

    def _result(self, request: SubmissionRequest, record: dict) -> tuple[str | None, str | None]:
        from agentmemeval.path1_v1.operations.submission_process import validate_helper_result

        path = Path(record["invocation_dir"]) / "result.json"
        if not path.exists():
            return None, None
        result = read_json(_file(path))
        binding = record.get("ready")
        if binding is None or any(result.get(key) != value for key, value in binding.items()):
            raise StoreIntegrityError("submitter result identity differs from persisted release")
        manifest_path = _file(Path(record["invocation_dir"]) / "invocation.json")
        if file_sha256(manifest_path) != record["manifest_sha256"]:
            raise StoreIntegrityError("submitter invocation bytes changed")
        try:
            version = validate_helper_result(
                path.parent, result, manifest=read_json(manifest_path),
                expected_result_schema=record.get("helper_result_schema_version"),
            )
        except (ValueError, OSError) as exc:
            raise StoreIntegrityError("submitter raw output evidence is invalid") from exc
        if (
            (version == 2 and result["capture_status"] != "complete")
            or result.get("timed_out") is not False
            or result.get("returncode") != 0
            or result.get("stderr_sha256") != hashlib.sha256(b"").hexdigest()
            or type(result.get("stdout")) is not str
        ):
            return None, file_sha256(path)
        match = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?", result["stdout"])
        if match is None or match[2] not in (None, request.spec.cluster):
            return None, file_sha256(path)
        return match[1], file_sha256(path)

    def _receipt(self, request: SubmissionRequest, record: dict) -> IntentReceipt:
        if record.get("cleanup_persistence_failed") is True:
            return IntentReceipt(request.request_sha256, "unknown", content_digest(record),
                                 None, False, None).validate(request)
        job_id, result_sha = (
            self._result(request, record) if record.get("invocation_dir") else (None, None)
        )
        if record["status"] == "accepted" and (
            job_id != record["job_id"] or result_sha != record["result_sha256"]
        ):
            raise StoreIntegrityError("accepted receipt lost its durable command result")
        evidence = content_digest({"record": record, "result_sha256": result_sha})
        return IntentReceipt(
            request.request_sha256,
            "accepted" if job_id else "unknown",
            evidence,
            job_id,
            bool(record.get("submitter_finished", False)),
            1 if job_id else None,
        ).validate(request)

    def lookup(self, request: SubmissionRequest, *, timeout_seconds: float) -> IntentReceipt | None:
        _timeout(timeout_seconds)
        if not self.store.database_path.exists():
            return None
        ledger = self._ledger(self.store.read_registry())
        record = ledger["intents"].get(request.intent_id)
        if record is None:
            return None
        if record["request_sha256"] != request.request_sha256:
            raise StoreIntegrityError("intent cannot be rebound to different submission bytes")
        return self._receipt(request, record)

    def inspect(self, request: SubmissionRequest) -> dict | None:
        """Read-only provenance, with process state explicitly non-authoritative."""
        if not self.store.database_path.exists():
            return None
        record = self._ledger(self.store.read_registry())["intents"].get(request.intent_id)
        if record is None:
            return None
        if record["request_sha256"] != request.request_sha256:
            raise StoreIntegrityError("inspection request differs")
        result = copy.deepcopy(record)
        result["parent_observation"] = observe_process(record["parent_identity"])
        result["submitter_observation"] = (
            observe_process(record["ready"]["submitter"])
            if record["ready"] is not None
            else "not_recorded"
        )
        result["process_observation_authorizes_retry"] = False
        return result

    def submit_once(self, request: SubmissionRequest, *, timeout_seconds: float) -> IntentReceipt:
        from agentmemeval.path1_v1.operations.e1_prepare import ValidatedE1Prepared

        _timeout(timeout_seconds)
        with self.store.lock():
            snapshot = self._initialize()
            ledger = self._ledger(snapshot)
            existing = ledger["intents"].get(request.intent_id)
            if existing is not None:
                if existing["request_sha256"] != request.request_sha256:
                    raise StoreIntegrityError("intent was already claimed for different bytes")
                # Even an incomplete 'claimed' attempt is never automatically re-executed.
                return self._receipt(request, existing)
            if self.pre_submit_verifier is None or self.account_reader is None:
                raise SubmissionDisabled("pre-submit admission and live capacity are required")
            if not self.fixture_mode and (
                not sys.platform.startswith("linux")
                or Path(self.command.argv_prefix[0]).name != "sbatch"
            ):
                raise SubmissionDisabled("production requires the pinned Linux sbatch binary")
            command_files = self.command.validate(fixture_mode=self.fixture_mode)
            worker, _files, script, pinned, effective_user = self._validate_launch(request)
            admission = self._admit(request, worker)
            capacity = self._capacity(request, ledger, admission.stage_cap,
                validated_phase=worker.contract["phase"],
                validated_e1=worker if type(worker) is ValidatedE1Prepared else None)
            job_dir = self._path(request.spec.work_dir).resolve()
            invocation_dir = self.store.directory / "i" / request.intent_id[:16]
            record = {
                "campaign_id": request.campaign_id,
                "intent_id": request.intent_id,
                "spec": request.spec.payload(),
                "request_sha256": request.request_sha256,
                "worker_request_sha256": worker.request_sha256,
                "status": "claimed",
                "created_at": utc(self.clock()).isoformat(),
                "parent_identity": process_identity(os.getpid()),
                "effective_user": effective_user,
                "invocation_dir": str(invocation_dir),
                "ready": None,
                "job_id": None,
                "result_sha256": None,
                "submitter_finished": False,
                "admission_sha256": admission.evidence_sha256,
                "capacity": capacity,
            }
            if type(_files) is ProbeLaunchFiles:
                record.update(probe_request_path=str(_files.probe_request_path),
                              probe_request_file_sha256=file_sha256(_files.probe_request_path))
                supersession = self._validate_e0_retry(worker)
                if supersession is not None:
                    record["operational_supersession"] = supersession.payload()
            ledger["intents"][request.intent_id] = record
            snapshot = self._commit(snapshot, ledger, "reserve remote submission intent")
            claim = self.store.claim_run(
                expected_generation=snapshot.generation,
                run_id=request.intent_id,
                intent_id=request.intent_id,
                run_directory=job_dir,
                metadata={
                    "purpose": "slurm_job_container",
                    "request_sha256": request.request_sha256,
                },
            )
            self.store.start_claim(
                expected_generation=claim.updated_generation, run_id=claim.run_id, token=claim.token
            )
            snapshot = self.store.read_registry()
            job_dir.mkdir(parents=True, exist_ok=False)
            invocation_dir.mkdir(parents=True, exist_ok=False)
            script_path = invocation_dir / "submitted_script.sh"
            with script_path.open("xb") as stream:
                stream.write(script)
                stream.flush()
                os.fsync(stream.fileno())
            deadline = time.monotonic() + timeout_seconds
            helper = _file(Path(__file__).with_name("submission_process.py"))
            # Windows venv python.exe is a redirector with an extra PID. The
            # stdlib-only handshake runs the actual base interpreter directly so
            # the Popen handle and kernel identity refer to the same submitter.
            # This helper is stdlib-only: canonicalize its base ELF explicitly.
            # Worker/service argv retain their verified venv entry and prefix.
            helper_python = _file(
                Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True))
            # Remove the original script filename; sbatch consumes pinned stdin bytes.
            argv = (*self.command.argv_prefix, *request.sbatch_argv()[1:-1])
            if (worker.contract["phase"] == "E0"
                    and worker.contract["job"]["kind"] == "resource_probe"):
                # sbatch has no --immediate option on the deployed Slurm.
                # 120s walltime + a 3600s planned start opportunity: scheduling
                # feasibility deadline, not a precise queue-deletion timer.
                argv = (*argv, f"--deadline=now+{RESOURCE_DEADLINE_OFFSET_SECONDS}seconds")
            manifest = {
                "schema_version": 2,
                "result_schema_version": 2,
                "nonce": uuid4().hex,
                "request_sha256": request.request_sha256,
                "argv": list(argv),
                "cwd": str(job_dir),
                "environment": clean_environment(),
                "stdin_path": str(script_path),
                "stdin_sha256": request.spec.script_sha256,
                "input_files": pinned + command_files,
                "wait_seconds": timeout_seconds,
                "command_timeout_seconds": timeout_seconds,
                "deadline_monotonic": deadline,
                "expected_effective_user": effective_user,
            }
            manifest_path = invocation_dir / "invocation.json"
            write_exclusive_json(manifest_path, manifest)
            manifest_sha = file_sha256(manifest_path)
            record.update(
                status="submit_unknown",
                manifest_sha256=manifest_sha,
                helper_sha256=file_sha256(helper),
                helper_python_sha256=file_sha256(helper_python),
                helper_result_schema_version=2,
            )
            snapshot = self._commit(snapshot, ledger, "persist unknown before starting submitter")
            helper_argv = [str(helper_python), "-I", str(helper), str(manifest_path), manifest_sha]
            owner = None
            def persist_cleanup(cleanup_owner, cleanup_receipt):
                name = "local_cleanup.json" if cleanup_owner is not None else "spawn_failure.json"
                write_exclusive_json(invocation_dir / name, asdict(cleanup_receipt))

            if sys.platform.startswith("linux"):
                guard = OwnedProcess.managed(
                    helper_argv,
                    cwd=invocation_dir,
                    env=clean_environment(),
                    term_grace_seconds=self.transport_term_grace_seconds,
                    receipt_sink=persist_cleanup,
                    binding=ProcessBinding(
                        request.campaign_id,
                        request.intent_id,
                        None,
                        content_digest(worker.contract["runtime_identity"]),
                        "submission_transport",
                        manifest["nonce"],
                    ),
                )
                process = None
            else:
                # Windows remains an explicit offline fixture, not a production
                # process-tree release producer. Use the exact Popen handle only.
                process = subprocess.Popen(
                    helper_argv,
                    env=clean_environment(),
                    cwd=invocation_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                guard = nullcontext()
            try:
                with guard as owner:
                    ready_path = invocation_dir / "ready.json"
                    while not ready_path.exists():
                        exited = (
                            owner.poll_exit() if owner is not None else process.poll() is not None
                        )
                        if exited or time.monotonic() >= deadline:
                            return self._receipt(request, record)
                        time.sleep(0.01)
                    ready = read_json(ready_path)
                    helper_pid = owner.identity.pid if owner is not None else process.pid
                    expected_ready = {
                        "nonce": manifest["nonce"],
                        "request_sha256": request.request_sha256,
                        "manifest_sha256": manifest_sha,
                        "submitter": process_identity(helper_pid),
                        "effective_user": effective_user,
                    }
                    if ready != expected_ready:
                        raise StoreIntegrityError("submitter handshake identity mismatch")
                    record["ready"] = ready
                    snapshot = self._commit(
                        snapshot, ledger, "persist submitter identity before release"
                    )
                    current_worker, _, current_script, _, current_user = self._validate_launch(
                        request,
                        e0_model_verification=(getattr(worker, "model_verification", None)
                                               if worker.contract["phase"] == "E0" else None),
                        e1_model_verification=(worker.model_verification
                                               if worker.contract["phase"] == "E1" else None),
                    )
                    if (
                        current_worker.request_sha256 != worker.request_sha256
                        or current_script != script
                        or current_user != effective_user
                    ):
                        raise ExecutorError("launch changed during submitter handshake")
                    current_admission = self._admit(request, current_worker)
                    record["capacity"] = self._capacity(
                        request, ledger, current_admission.stage_cap,
                        validated_phase=current_worker.contract["phase"],
                        validated_e1=(current_worker
                                      if type(current_worker) is ValidatedE1Prepared else None),
                    )
                    record["admission_sha256"] = current_admission.evidence_sha256
                    snapshot = self._commit(snapshot, ledger, "persist fresh pre-exec gates")
                    if time.monotonic() >= deadline:
                        return self._receipt(request, record)
                    write_exclusive_json(invocation_dir / "go.json", ready)
                    if owner is not None:
                        cleanup = owner.finish(
                            deadline_monotonic=deadline,
                            term_grace_seconds=self.transport_term_grace_seconds,
                        )
                        # Dataclass tuple fields are not CampaignStore JSON
                        # values. This serializes diagnostics only; the typed
                        # owner-issued receipt remains the cleanup authority.
                        record["local_cleanup"] = json.loads(
                            json.dumps(asdict(cleanup), allow_nan=False))
                        record["submitter_finished"] = cleanup.leader_reaped
                    else:
                        process.wait(timeout=max(0.01, deadline - time.monotonic()))
                        record["submitter_finished"] = True
                # managed must persist cleanup successfully before acceptance is
                # committed. A scheduler result alone cannot close local transport.
                job_id, result_sha = self._result(request, record)
                if job_id:
                    record.update(status="accepted", job_id=job_id, result_sha256=result_sha)
                snapshot = self._commit(snapshot, ledger, "persist completed submitter evidence")
                return self._receipt(request, record)
            except (subprocess.TimeoutExpired, OSError, ExecutorError, OwnershipError) as exc:
                if isinstance(exc, ReceiptSinkError):
                    record.update(status="submit_unknown", job_id=None, result_sha256=None,
                                  submitter_finished=False, cleanup_persistence_failed=True,
                                  cleanup_sink_failure_type=exc.sink_error_type)
                    snapshot = self._commit(
                        snapshot, ledger, "retain unknown after cleanup sink failure")
                # Unknown remains durable. Do not convert transport/process failure
                # into not_submitted or a retry; do not cancel any allocation.
                return self._receipt(request, record)
            finally:
                if owner is None and process is not None and process.poll() is None:
                    process.kill()  # Only this exact internal handshake child.
                    process.wait(timeout=5)
