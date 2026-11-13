"""Fail-closed Slurm boundaries; no SSH transport or remote executor is installed.

Reads use explicit argv and a sanitized environment, never a shell. Submission
requires an injected durable IntentExecutor: a bare ``sbatch`` transport is not
an executor. Its implementation must persist the request/submitter identity
before sbatch, serialize by intent remotely, verify worker bytes, and persist the
receipt. This module neither implements nor claims those remote guarantees.

Accounting absence is never proof that submission did not happen. Query errors,
unsupported allocation formats and unverified account visibility deny capacity.
Slurm completion is scheduler evidence, not scientific acceptance or release.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from agentmemeval.path1_v1.operations.contracts import content_digest

ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SPECIAL_EXIT",
    }
)
TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
READ_ENV = {"LC_ALL": "C", "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"}


class SlurmError(RuntimeError):
    """Uncertain or invalid scheduler evidence; no additional work is authorized."""


class SubmissionDisabled(SlurmError):
    pass


class OwnershipError(SlurmError):
    pass


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("an aware timestamp is required")
    return value.astimezone(UTC)


def identifier(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) is None:
        raise ValueError("invalid Slurm identity")
    return value


def digest(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise ValueError("expected a SHA-256 identity")
    return value


def remote_path(value: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value == "/"
        or any(c in value for c in "\x00\r\n|\\")
        or any(part in {"", ".", ".."} for part in value.split("/")[1:])
    ):
        raise ValueError("expected a normalized absolute remote POSIX path")
    return value


def job_name(campaign_id: str, intent_id: str) -> str:
    """Full digest in the name; never use a truncated label as a unique claim."""
    identifier(campaign_id)
    digest(intent_id)
    return "ame-" + intent_id


def job_comment(campaign_id: str, intent_id: str) -> str:
    return "ame-v1:" + identifier(campaign_id) + ":" + digest(intent_id)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandTransport(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        inherit_environment: bool,
        timeout_seconds: float,
    ) -> CommandResult:
        """Execute literal argv with bounded duration, no shell or hidden retries.

        inherit_environment=False excludes SQUEUE_*/SACCT_*/SCANCEL_* filters.
        A real transport must supply its separately verified site configuration.
        Network/auth configuration belongs to transport, not these argv/logs.
        """
        ...


@dataclass(frozen=True)
class SubmitSpec:
    cluster: str
    account: str
    user: str
    partition: str
    work_dir: str
    script_path: str
    script_sha256: str
    script_args: tuple[str, ...] = ()
    walltime_seconds: int = 120
    gpus: int = 1
    cpus_per_task: int | None = None
    memory_mib: int | None = None

    def __post_init__(self) -> None:
        for value in (self.cluster, self.account, self.user, self.partition):
            identifier(value)
        remote_path(self.work_dir)
        remote_path(self.script_path)
        digest(self.script_sha256)
        if (
            type(self.walltime_seconds) is not int
            or not 1 <= self.walltime_seconds <= 86400
            or type(self.gpus) is not int
            or self.gpus != 1
        ):
            raise ValueError("dispatch supports bounded single-GPU jobs only")
        if type(self.script_args) is not tuple or any(
            type(x) is not str or any(c in x for c in "\x00\r\n") for x in self.script_args
        ):
            raise ValueError("script arguments must be literal single-line strings")
        if (self.cpus_per_task is None) != (self.memory_mib is None):
            raise ValueError("explicit CPU and memory must be supplied together")
        if self.cpus_per_task is not None and (
            type(self.cpus_per_task) is not int or not 1 <= self.cpus_per_task <= 256
            or type(self.memory_mib) is not int or not 1 <= self.memory_mib <= 1048576
        ):
            raise ValueError("explicit CPU/MiB resources must be bounded integers")

    def payload(self) -> dict:
        value = asdict(self)
        value["script_args"] = list(self.script_args)
        if self.cpus_per_task is None:
            # Preserve every historical payload and SubmissionRequest digest.
            del value["cpus_per_task"]
            del value["memory_mib"]
        return value

    @classmethod
    def from_payload(cls, value: dict) -> SubmitSpec:
        copied = dict(value)
        explicit = {"cpus_per_task", "memory_mib"} & copied.keys()
        if explicit and (len(explicit) != 2 or any(copied[key] is None for key in explicit)):
            raise ValueError("explicit resource payload cannot contain partial/null fields")
        copied["script_args"] = tuple(copied["script_args"])
        return cls(**copied)


@dataclass(frozen=True)
class SubmissionRequest:
    campaign_id: str
    intent_id: str
    spec: SubmitSpec

    def __post_init__(self) -> None:
        identifier(self.campaign_id)
        digest(self.intent_id)

    @property
    def request_sha256(self) -> str:
        return content_digest(
            {
                "campaign_id": self.campaign_id,
                "intent_id": self.intent_id,
                "spec": self.spec.payload(),
            }
        )

    def sbatch_argv(self) -> tuple[str, ...]:
        """Executor input, NOT permission to execute sbatch directly.

        A remote executor must verify script_sha256 and current runtime gates,
        reject hidden SBATCH environment overrides, and persist its receipt.
        CLI arguments enforce bounded resources, no requeue, and owned logs.
        The reviewed script must not request arrays/heterogeneous jobs or hide
        additional resource requests; absence of an --array flag cannot override
        an array directive inside an unchecked script.
        """
        seconds = self.spec.walltime_seconds
        walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        return (
            "sbatch",
            "--parsable",
            f"--clusters={self.spec.cluster}",
            f"--account={self.spec.account}",
            f"--partition={self.spec.partition}",
            f"--job-name={job_name(self.campaign_id, self.intent_id)}",
            f"--comment={job_comment(self.campaign_id, self.intent_id)}",
            f"--chdir={self.spec.work_dir}",
            "--nodes=1",
            "--ntasks=1",
            "--gpus=1",
            *((f"--cpus-per-task={self.spec.cpus_per_task}",
               f"--mem={self.spec.memory_mib}") if self.spec.cpus_per_task is not None else ()),
            "--no-requeue",
            f"--time={walltime}",
            f"--output={self.spec.work_dir}/slurm-%j.out",
            f"--error={self.spec.work_dir}/slurm-%j.err",
            "--export=NONE",
            self.spec.script_path,
            *self.spec.script_args,
        )


@dataclass(frozen=True)
class IntentReceipt:
    """Evidence returned by a trusted executor, not a verified remote fact here."""

    request_sha256: str
    status: str  # accepted / unknown / not_submitted
    evidence_sha256: str
    job_id: str | None = None
    submitter_finished: bool = False
    submission_count: int | None = None

    def validate(self, request: SubmissionRequest) -> IntentReceipt:
        digest(self.request_sha256)
        digest(self.evidence_sha256)
        if self.request_sha256 != request.request_sha256:
            raise SlurmError("executor receipt is bound to a different request")
        if type(self.submitter_finished) is not bool:
            raise SlurmError("invalid submitter state")
        if self.submission_count is not None and (
            type(self.submission_count) is not int or self.submission_count not in (0, 1)
        ):
            raise SlurmError("executor reports duplicate or invalid submission count")
        if self.status == "accepted":
            if (
                type(self.job_id) is not str
                or not re.fullmatch(r"[1-9][0-9]*", self.job_id)
                or self.submission_count != 1
            ):
                raise SlurmError("accepted receipt requires one non-array allocation")
        elif self.status == "not_submitted":
            if not self.submitter_finished or self.submission_count != 0 or self.job_id is not None:
                raise SlurmError("negative receipt lacks ended submitter/zero-submission proof")
        elif self.status == "unknown":
            if self.job_id is not None:
                raise SlurmError("an unknown receipt must not assert a job ID")
        else:
            raise SlurmError("unsupported executor receipt")
        return self


class IntentExecutor(Protocol):
    """Remote implementation deliberately absent; mocks only prove local behavior.

    submit_once must atomically claim request_sha256 under intent_id remotely,
    verify exact bytes/runtime, persist submitter identity BEFORE sbatch, and
    durably save the result. Re-entry MUST NOT execute a second sbatch. A missing
    lookup result proves nothing, including after a submitter crash or timeout.
    """

    def submit_once(
        self, request: SubmissionRequest, *, timeout_seconds: float
    ) -> IntentReceipt: ...

    def lookup(
        self, request: SubmissionRequest, *, timeout_seconds: float
    ) -> IntentReceipt | None: ...


@dataclass(frozen=True)
class CapacityEvidence:
    cluster: str
    account: str
    max_submit_jobs: int
    observed_at: datetime
    evidence_sha256: str
    full_account_visibility: bool
    counting_unit: str = "expanded_allocation"

    def validate(self, *, cluster: str, account: str, now: datetime, max_age: float) -> None:
        digest(self.evidence_sha256)
        if (
            self.cluster != cluster
            or self.account != account
            or type(self.max_submit_jobs) is not int
            or self.max_submit_jobs < 0
            or self.full_account_visibility is not True
            or self.counting_unit != "expanded_allocation"
        ):
            raise SlurmError("capacity scope, effective limit or visibility is unverified")
        age = (utc(now) - utc(self.observed_at)).total_seconds()
        if age < 0 or age > max_age:
            raise SlurmError("capacity evidence is stale or from the future")


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    user: str
    account: str
    name: str
    state: str
    comment: str
    submitted_at: str
    work_dir: str
    source: str

    @property
    def key(self) -> tuple[str, str]:
        return self.job_id, self.submitted_at

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def matches(self, request: SubmissionRequest) -> bool:
        return (
            self.user == request.spec.user
            and self.account == request.spec.account
            and self.name == job_name(request.campaign_id, request.intent_id)
            and self.comment == job_comment(request.campaign_id, request.intent_id)
            and self.work_dir == request.spec.work_dir
            and re.fullmatch(r"[1-9][0-9]*", self.job_id) is not None
        )


@dataclass(frozen=True)
class AccountSnapshot:
    cluster: str
    account: str
    observed_at: datetime
    queue: tuple[JobRecord, ...]
    accounting: tuple[JobRecord, ...]
    evidence_sha256: str
    capacity: CapacityEvidence | None

    def allocations(self) -> tuple[JobRecord, ...]:
        """Queue wins over delayed accounting for the same allocation identity."""
        rows = {job.key: job for job in self.accounting}
        rows.update({job.key: job for job in self.queue})
        return tuple(rows.values())


def _parse_rows(text: str, *, source: str, account: str) -> tuple[JobRecord, ...]:
    if type(text) is not str:
        raise SlurmError("scheduler output is not text")
    found: dict[tuple[str, str], JobRecord] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split("|")]
        if len(fields) != 8:
            raise SlurmError("unsupported/truncated scheduler row")
        job_id, user, owner_account, name, state, comment, submitted, directory = fields
        # Explicit --array/--allocations: compressed ranges/steps are not countable here.
        if not re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?", job_id):
            raise SlurmError("unsupported array, heterogeneous or job-step identity")
        if owner_account != account or not user or not name:
            raise SlurmError("scheduler account/owner scope mismatch")
        if source == "sacct" and re.fullmatch(r"CANCELLED by [0-9]+", state):
            state = "CANCELLED"
        if state not in ACTIVE_STATES | TERMINAL_STATES:
            raise SlurmError("unknown or truncated scheduler state")
        try:
            parsed = datetime.strptime(submitted, "%Y-%m-%dT%H:%M:%S")
        except ValueError as exc:
            raise SlurmError("missing or ambiguous submission timestamp") from exc
        if parsed.strftime("%Y-%m-%dT%H:%M:%S") != submitted:
            raise SlurmError("noncanonical submission timestamp")
        record = JobRecord(
            job_id, user, owner_account, name, state, comment, submitted, directory, source
        )
        if record.key in found and found[record.key] != record:
            raise SlurmError("conflicting duplicate scheduler allocation")
        found[record.key] = record
    return tuple(found.values())


class SlurmAdapter:
    def __init__(
        self,
        transport: CommandTransport,
        *,
        cluster: str,
        account: str,
        user: str,
        capacity_probe: Callable[[], CapacityEvidence] | None = None,
        executor: IntentExecutor | None = None,
        allow_cancel: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 20.0,
        max_snapshot_age_seconds: float = 30.0,
    ) -> None:
        self.transport = transport
        self.cluster, self.account, self.user = map(identifier, (cluster, account, user))
        self.capacity_probe, self.executor, self.allow_cancel = (
            capacity_probe,
            executor,
            allow_cancel,
        )
        self.clock = clock
        if not 0 < timeout_seconds <= 60 or not 0 < max_snapshot_age_seconds <= 60:
            raise ValueError("bounded transport/snapshot deadlines are required")
        self.timeout_seconds, self.max_snapshot_age_seconds = (
            timeout_seconds,
            max_snapshot_age_seconds,
        )

    def _run(self, argv: tuple[str, ...]) -> str:
        try:
            result = self.transport.run(
                argv,
                environment=READ_ENV,
                inherit_environment=False,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            raise SlurmError("scheduler transport failed; outcome is uncertain") from exc
        if (
            not isinstance(result, CommandResult)
            or type(result.returncode) is not int
            or result.returncode != 0
            or type(result.stderr) is not str
            or result.stderr.strip()
        ):
            raise SlurmError("scheduler command did not produce a clean complete result")
        return result.stdout

    def _queue(self) -> tuple[tuple[JobRecord, ...], str]:
        output = self._run(
            (
                "squeue",
                "--noheader",
                "--all",
                "--array",
                "--local",
                f"--clusters={self.cluster}",
                f"--account={self.account}",
                "--states=all",
                "--format=%i|%u|%a|%j|%T|%k|%V|%Z",
            )
        )
        return _parse_rows(output, source="squeue", account=self.account), output

    def query(self, *, since: datetime) -> AccountSnapshot:
        """Read every visible account job, never filter to this campaign/user."""
        began = utc(self.clock())
        queue, queue_raw = self._queue()
        start = (utc(since) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
        output = self._run(
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--allusers",
                "--array",
                "--allocations",
                "--duplicates",
                f"--clusters={self.cluster}",
                f"--accounts={self.account}",
                f"--starttime={start}",
                "--format=JobID,User,Account,JobName%200,State%40,Comment%200,Submit,WorkDir%4096",
            )
        )
        accounting = _parse_rows(output, source="sacct", account=self.account)
        return AccountSnapshot(
            self.cluster,
            self.account,
            began,
            queue,
            accounting,
            hashlib.sha256((queue_raw + "\0" + output).encode()).hexdigest(),
            None,
        )

    def account_snapshot(self, *, since: datetime) -> AccountSnapshot:
        snapshot = self.query(since=since)
        if self.capacity_probe is None:
            return snapshot
        try:
            capacity = self.capacity_probe()
            capacity.validate(
                cluster=self.cluster,
                account=self.account,
                now=self.clock(),
                max_age=self.max_snapshot_age_seconds,
            )
        except Exception:
            # Broken quota evidence must not prevent old-job reconciliation.
            # It denies ONLY new submission; query evidence remains usable.
            capacity = None
        return AccountSnapshot(
            snapshot.cluster,
            snapshot.account,
            snapshot.observed_at,
            snapshot.queue,
            snapshot.accounting,
            snapshot.evidence_sha256,
            capacity,
        )

    def check_request(self, request: SubmissionRequest) -> None:
        if (request.spec.cluster, request.spec.account, request.spec.user) != (
            self.cluster,
            self.account,
            self.user,
        ):
            raise SlurmError("request targets a different scheduler identity")

    def submit(self, request: SubmissionRequest) -> IntentReceipt:
        self.check_request(request)
        if self.executor is None:
            raise SubmissionDisabled("a durable remote intent executor is not installed")
        return self.executor.submit_once(request, timeout_seconds=self.timeout_seconds).validate(
            request
        )

    def lookup(self, request: SubmissionRequest) -> IntentReceipt | None:
        self.check_request(request)
        if self.executor is None:
            return None
        receipt = self.executor.lookup(request, timeout_seconds=self.timeout_seconds)
        return None if receipt is None else receipt.validate(request)

    def cancel_owned(self, request: SubmissionRequest, owned: JobRecord) -> dict:
        """Fresh identity + directory check, then exactly one constrained job ID.

        The caller supplies its persisted owned record. No broad/pattern cancel,
        PID killing, deletion, automatic timeout cleanup or release claim exists.
        A successful scancel means request accepted; release needs later evidence.
        """
        self.check_request(request)
        if not self.allow_cancel:
            raise SubmissionDisabled("cancellation is explicitly disabled")
        if not owned.matches(request):
            raise OwnershipError("persisted job identity is not owned by this request")
        queue, _raw = self._queue()
        matches = [job for job in queue if job.job_id == owned.job_id]
        if not matches:
            return {"status": "not_visible", "job_id": owned.job_id, "release_verified": False}
        if (
            len(matches) != 1
            or not matches[0].matches(request)
            or matches[0].key != owned.key
            or not matches[0].active
        ):
            raise OwnershipError("live job identity changed; cancellation refused")
        self._run(
            (
                "scancel",
                f"--clusters={self.cluster}",
                f"--account={self.account}",
                f"--user={self.user}",
                f"--name={owned.name}",
                owned.job_id,
            )
        )
        return {"status": "cancel_requested", "job_id": owned.job_id, "release_verified": False}
