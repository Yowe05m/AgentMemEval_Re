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
    *, python_executable: Path, repository_root: Path, probe_request_path: Path
) -> bytes:
    """Separate exact E0 topology: request v2 carries every execution parameter."""
    for path in (python_executable, repository_root, probe_request_path):
        if not path.is_absolute() or any(c in str(path) for c in "\x00\r\n"):
            raise ExecutorError("probe launcher paths must be absolute and single-line")
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
                    or claims[0].status != "started"
                    or claims[0].metadata
                    != {"purpose": "slurm_job_container", "request_sha256": request.request_sha256}
                    or claims[0].run_directory != str(self._path(request.spec.work_dir).resolve())
                ):
                    raise StoreIntegrityError("attempted submission has no matching consumed claim")
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
        self, request: SubmissionRequest
    ) -> tuple[Any, LaunchFiles, bytes, list[dict], dict | None]:
        # Lazy import avoids making package import an execution/admission action.
        from agentmemeval.path1_v1.operations.worker import validate_worker_request

        if (request.spec.cluster, request.spec.account) != (self.cluster, self.account):
            raise ExecutorError("submission targets another executor scope")
        effective_user = None if self.fixture_mode else verify_effective_user(request.spec.user)
        if request.spec.script_args:
            raise ExecutorError("worker launcher takes no extra script arguments")
        files = self.launch_files(request)
        if type(files) is ProbeLaunchFiles:
            return self._validate_probe_launch(request, files, effective_user)
        if type(files) is not LaunchFiles:
            raise ExecutorError("launcher must explicitly select research or E0 topology")
        python = _file(files.python_executable)
        digest(files.python_sha256)
        if file_sha256(python) != files.python_sha256:
            raise ExecutorError("worker interpreter content changed")
        worker_path = _file(files.worker_request_path)
        worker = validate_worker_request(worker_path)
        if (
            worker.intent_id != request.intent_id
            or worker.contract["campaign_id"] != request.campaign_id
        ):
            raise ExecutorError("worker request is bound to a different campaign/intent")
        repository = Path(worker.request["repository_root"]).resolve(strict=True)
        identity = worker.contract["runtime_identity"]
        observed = build_runtime_identity(
            repository,
            core_paths=[x["path"] for x in identity["core_files"]],
            operations_paths=[x["path"] for x in identity["operations_files"]],
            config_path=identity["config"]["path"],
            dependency_lock_path=identity["dependency_lock"]["path"],
            model_manifest_path=worker.request["model_manifest_path"],
            model_root=worker.request["model_root"],
        )
        if observed != identity:
            raise ExecutorError("current source/model identity differs from planned worker")
        source_records = identity["core_files"] + identity["operations_files"]
        paths = {item["path"] for item in source_records}
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
        expected_script = render_worker_script(
            python_executable=python, repository_root=repository, worker_request_path=worker_path
        )
        script_path = _file(self._path(request.spec.script_path))
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
            {"path": str(python), "sha256": files.python_sha256},
            {"path": str(worker_path), "sha256": file_sha256(worker_path)},
            {"path": str(script_path), "sha256": request.spec.script_sha256},
            {"path": worker.request["contract_path"], "sha256": worker.request["contract_sha256"]},
            {"path": worker.request["claim_path"], "sha256": worker.request["claim_sha256"]},
            {
                "path": worker.request["seed_inventory_path"],
                "sha256": worker.request["seed_inventory_sha256"],
            },
        ]
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

    def _validate_probe_launch(self, request, files, effective_user):
        from agentmemeval.path1_v1.operations.runtime_probe import validate_probe_request

        path = _file(files.probe_request_path)
        probe = validate_probe_request(path)
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
                repository_root=repository, probe_request_path=path)
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
        if kind == "service_probe":
            pinned += [{"path": row["path"], "sha256": row["sha256"]}
                       for row in launch["service_python_identity"]["files"]]
        if probe.request["model_manifest_path"] is not None:
            pinned.append({"path": probe.request["model_manifest_path"],
                           "sha256": identity["model"]["manifest_sha256"]})
        return probe, files, script, pinned, effective_user

    def _admit(self, request: SubmissionRequest, worker: Any) -> SubmissionAdmission:
        if self.pre_submit_verifier is None:
            raise SubmissionDisabled("no trusted pre-submit admission verifier is installed")
        admission = self.pre_submit_verifier(request, worker)
        if not isinstance(admission, SubmissionAdmission):
            raise SubmissionDisabled("pre-submit verifier returned no typed evidence")
        admission.validate(request, worker, self.clock())
        return admission

    def _capacity(self, request: SubmissionRequest, ledger: dict, stage_cap: int,
                  *, validated_phase: str | None = None) -> dict:
        if self.account_reader is None:
            raise SubmissionDisabled("no live account capacity reader is installed")
        observed = self.account_reader()
        if not isinstance(observed, AccountSnapshot):
            from agentmemeval.path1_v1.operations.user_capacity import (
                UserAssociationSnapshot,
                validate_user_association_capacity,
            )

            if type(observed) is UserAssociationSnapshot:
                return validate_user_association_capacity(
                    observed, request, ledger, stage_cap, self.clock(), validated_phase)
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
        for intent, record in ledger["intents"].items():
            if intent == request.intent_id or record["status"] == "claimed":
                continue
            previous = self._stored_request(record)
            represented = [job for job in jobs if job.matches(previous)]
            if not represented:
                unknown_account += 1
                unknown_campaign += previous.campaign_id == request.campaign_id
        total = len(active) + unknown_account
        own = len(campaign) + unknown_campaign
        available = min(stage_cap - own, min(observed.capacity.max_submit_jobs, 8) - total - 2)
        if available < 1:
            raise SubmissionDisabled("fresh account/stage capacity has no admission slot")
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
        path = Path(record["invocation_dir"]) / "result.json"
        if not path.exists():
            return None, None
        result = read_json(_file(path))
        binding = record.get("ready")
        if binding is None or any(result.get(key) != value for key, value in binding.items()):
            raise StoreIntegrityError("submitter result identity differs from persisted release")
        if (
            result.get("timed_out") is not False
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
                                      validated_phase=worker.contract["phase"])
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
                # 120s walltime + a 10s planned start opportunity: scheduling
                # feasibility deadline, not a precise queue-deletion timer.
                argv = (*argv, "--deadline=now+130seconds")
            manifest = {
                "schema_version": 1,
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
                        request
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
