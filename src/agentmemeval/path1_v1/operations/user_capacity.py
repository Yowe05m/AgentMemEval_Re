"""E0-only capacity from the effective user's unpartitioned Slurm association.

This is an observed user-association upper bound, not full-account visibility,
an allocation guarantee, or scientific admission. Hidden parent limits and
external concurrent submissions remain scheduler-enforced. The executor must
hold its existing cluster/account lock across reading, ledger counting and
submission. A snapshot is evidence from a trusted producer, not a bearer token:
the default reader executes pinned programs; injected transports are trusted
test/integration boundaries, never a JSON ``verified`` bypass.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.slurm import (
    ACTIVE_STATES,
    READ_ENV,
    CommandResult,
    CommandTransport,
    JobRecord,
    SubmissionDisabled,
    SubmissionRequest,
    SubmitSpec,
    _parse_rows,
    digest,
    identifier,
    utc,
)

ASSOCIATION_FIELDS = (
    "Cluster,Account,User,Partition,ParentName,ID,ParentID,QOS,DefaultQOS,"
    "MaxJobs,MaxSubmitJobs,GrpJobs,GrpSubmitJobs,GrpTRES"
)
COMMAND_NAMES = ("sacctmgr", "squeue", "sacct")


def _effective_identity(expected_user: str) -> int:
    try:
        if not sys.platform.startswith("linux"):
            raise SubmissionDisabled("user association reads require Linux effective credentials")
        import pwd

        uid = os.geteuid()
        by_id, by_name = pwd.getpwuid(uid), pwd.getpwnam(expected_user)
        if (
            type(uid) is not int
            or uid < 0
            or by_id.pw_name != expected_user
            or by_name.pw_name != expected_user
            or by_id.pw_uid != uid
            or by_name.pw_uid != uid
        ):
            raise SubmissionDisabled("user association differs from effective OS credentials")
        return uid
    except (ImportError, OSError, KeyError) as exc:
        raise SubmissionDisabled("effective user identity is unavailable") from exc


@dataclass(frozen=True)
class PinnedUserCommand:
    path: str
    sha256: str

    def validate(self, name: str) -> None:
        digest(self.sha256)
        if type(self.path) is not str:
            raise SubmissionDisabled("pinned command path must be text")
        path = Path(self.path)
        try:
            if (
                not path.is_absolute()
                or path.name != name
                or any(part.is_symlink() for part in (path, *path.parents))
                or path.resolve(strict=True) != path
                or not stat.S_ISREG(path.stat().st_mode)
                or hashlib.sha256(path.read_bytes()).hexdigest() != self.sha256
            ):
                raise SubmissionDisabled("user capacity command bytes/path are not pinned")
        except OSError as exc:
            raise SubmissionDisabled("user capacity command is unavailable") from exc


@dataclass(frozen=True)
class UserCommandObservation:
    name: str
    command: PinnedUserCommand
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def payload(self) -> dict:
        value = asdict(self)
        value["argv"] = list(self.argv)
        return value


@dataclass(frozen=True)
class UserAssociationSnapshot:
    cluster: str
    account: str
    user: str
    effective_uid: int
    observed_at: datetime
    completed_at: datetime
    since: datetime
    commands: tuple[UserCommandObservation, ...]

    def payload(self) -> dict:
        return {
            "scope": "user_association",
            "cluster": self.cluster,
            "account": self.account,
            "user": self.user,
            "effective_uid": self.effective_uid,
            "observed_at": utc(self.observed_at).isoformat(),
            "completed_at": utc(self.completed_at).isoformat(),
            "since": utc(self.since).isoformat(),
            "commands": [command.payload() for command in self.commands],
        }

    @property
    def evidence_sha256(self) -> str:
        return content_digest(self.payload())


def _argv(name: str, command: PinnedUserCommand, snapshot: UserAssociationSnapshot) -> tuple:
    cluster, account, user = snapshot.cluster, snapshot.account, snapshot.user
    if name == "sacctmgr":
        return (
            command.path,
            "show",
            "assoc",
            "where",
            f"cluster={cluster}",
            f"account={account}",
            f"user={user}",
            f"format={ASSOCIATION_FIELDS}",
            "--noheader",
            "--parsable2",
        )
    if name == "squeue":
        return (
            command.path,
            "--noheader",
            "--all",
            "--array",
            "--local",
            "--states=all",
            f"--user={user}",
            f"--account={account}",
            f"--clusters={cluster}",
            "--format=%i|%u|%a|%j|%T|%k|%V|%Z",
        )
    start = (utc(snapshot.since) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    return (
        command.path,
        "--noheader",
        "--parsable2",
        "--array",
        "--allocations",
        "--duplicates",
        f"--user={user}",
        f"--accounts={account}",
        f"--clusters={cluster}",
        f"--starttime={start}",
        "--format=Cluster,JobID,User,Account,JobName%200,State%40,Comment%200,Submit,WorkDir%4096",
    )


def _parsed(snapshot: UserAssociationSnapshot) -> tuple[str, int, tuple[JobRecord, ...]]:
    """Reparse raw outputs, including scope and command shape, at the decision boundary."""
    if type(snapshot.commands) is not tuple or len(snapshot.commands) != 3:
        raise SubmissionDisabled("complete user association observations are required")
    for name, observation in zip(COMMAND_NAMES, snapshot.commands, strict=True):
        if (
            type(observation) is not UserCommandObservation
            or type(observation.command) is not PinnedUserCommand
            or observation.name != name
            or type(observation.argv) is not tuple
            or observation.argv != _argv(name, observation.command, snapshot)
            or type(observation.returncode) is not int
            or observation.returncode != 0
            or type(observation.stdout) is not str
            or type(observation.stderr) is not str
            or observation.stderr.strip()
        ):
            raise SubmissionDisabled("incomplete, filtered or failed user association observation")
        observation.command.validate(name)
    assoc, queue, accounting = snapshot.commands
    rows = [line.split("|") for line in assoc.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 14:
        raise SubmissionDisabled("exactly one complete user association is required")
    fields = rows[0]
    if (
        fields[:4] != [snapshot.cluster, snapshot.account, snapshot.user, ""]
        or re.fullmatch(r"[1-9][0-9]*", fields[5]) is None
        or re.fullmatch(r"[1-9][0-9]*", fields[6]) is None
        or re.fullmatch(r"0|[1-9][0-9]*", fields[10]) is None
    ):
        raise SubmissionDisabled("unpartitioned association scope/MaxSubmitJobs is unavailable")
    queue_lines = queue.stdout.splitlines()
    if queue_lines and queue_lines[0] == "CLUSTER: " + snapshot.cluster:
        queue_lines = queue_lines[1:]
    queue_jobs = _parse_rows("\n".join(queue_lines), source="squeue", account=snapshot.account)
    accounting_lines = []
    for line in accounting.stdout.splitlines():
        if not line.strip():
            continue
        columns = line.split("|")
        if len(columns) != 9 or columns[0] != snapshot.cluster:
            raise SubmissionDisabled("accounting contains a different/unknown cluster")
        accounting_lines.append("|".join(columns[1:]))
    accounting_jobs = _parse_rows(
        "\n".join(accounting_lines), source="sacct", account=snapshot.account
    )
    jobs = {}
    for job in (*accounting_jobs, *queue_jobs):
        if job.user != snapshot.user:
            raise SubmissionDisabled("scheduler output contains a different user association")
        previous = jobs.get(job.key)
        if previous and (
            previous.user,
            previous.account,
            previous.name,
            previous.comment,
            previous.work_dir,
        ) != (job.user, job.account, job.name, job.comment, job.work_dir):
            raise SubmissionDisabled("allocation identity conflicts across scheduler observations")
        # Accounting can lag behind the live queue even when queried after it.
        jobs[job.key] = job
    return fields[5], int(fields[10]), tuple(jobs.values())


class UserAssociationReader:
    def __init__(
        self,
        *,
        cluster: str,
        account: str,
        user: str,
        commands: Mapping[str, PinnedUserCommand],
        since: datetime,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        transport: CommandTransport | None = None,
    ) -> None:
        self.cluster, self.account, self.user = map(identifier, (cluster, account, user))
        if set(commands) != set(COMMAND_NAMES) or any(
            type(command) is not PinnedUserCommand for command in commands.values()
        ):
            raise ValueError("exact pinned sacctmgr/squeue/sacct commands are required")
        self.commands, self.since = dict(commands), utc(since)
        self.clock, self.transport = clock, transport

    def read(self) -> UserAssociationSnapshot:
        began, uid = utc(self.clock()), _effective_identity(self.user)
        if self.since > began:
            raise SubmissionDisabled("accounting start time is in the future")
        shape = UserAssociationSnapshot(
            self.cluster, self.account, self.user, uid, began, began, self.since, ()
        )
        observations = []
        for name in COMMAND_NAMES:
            command = self.commands[name]
            command.validate(name)
            argv = _argv(name, command, shape)
            try:
                if self.transport is None:
                    result = subprocess.run(
                        argv,
                        shell=False,
                        env=dict(READ_ENV),
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        timeout=8,
                        check=False,
                    )
                    outcome = CommandResult(
                        result.returncode,
                        result.stdout.decode("utf-8", errors="strict"),
                        result.stderr.decode("utf-8", errors="strict"),
                    )
                else:
                    outcome = self.transport.run(
                        argv,
                        environment=dict(READ_ENV),
                        inherit_environment=False,
                        timeout_seconds=8,
                    )
            except Exception as exc:
                raise SubmissionDisabled(
                    "user association read failed; no capacity established"
                ) from exc
            if type(outcome) is not CommandResult:
                raise SubmissionDisabled("user association transport returned no complete result")
            observations.append(
                UserCommandObservation(
                    name, command, argv, outcome.returncode, outcome.stdout, outcome.stderr
                )
            )
            command.validate(name)
        completed = utc(self.clock())
        if not 0 <= (completed - began).total_seconds() <= 30:
            raise SubmissionDisabled("user association observation exceeded freshness bound")
        if _effective_identity(self.user) != uid:
            raise SubmissionDisabled("effective user changed during association observation")
        snapshot = UserAssociationSnapshot(
            self.cluster,
            self.account,
            self.user,
            uid,
            began,
            completed,
            self.since,
            tuple(observations),
        )
        _parsed(snapshot)
        return snapshot

    __call__ = read


def validate_user_association_capacity(
    snapshot: UserAssociationSnapshot,
    request: SubmissionRequest,
    ledger: dict,
    stage_cap: int,
    now: datetime,
    validated_phase: str,
    *,
    resolved_e0: dict | None = None,
    store=None,
    validated_e1=None,
) -> dict:
    """Validate E0 or fixed E1 stage-one capacity under the account ledger lock."""
    from agentmemeval.path1_v1.operations.e1_prepare import ValidatedE1Prepared

    if validated_phase == "E1":
        if (type(validated_e1) is not ValidatedE1Prepared
                or validated_e1.contract["phase"] != "E1"
                or validated_e1.submission != request or type(stage_cap) is not int
                or stage_cap != 1):
            raise SubmissionDisabled(
                "without fixed prepared evidence and stage one, capacity is enabled "
                "only for validated E0")
    elif validated_phase != "E0" or validated_e1 is not None:
        raise SubmissionDisabled("user association capacity is enabled only for validated E0")
    if type(snapshot) is not UserAssociationSnapshot or type(request) is not SubmissionRequest:
        raise SubmissionDisabled("typed user association snapshot/request required")
    if type(stage_cap) is not int or stage_cap not in (1, 2, 4, 6):
        raise SubmissionDisabled("stage cap must be one of 1, 2, 4, 6")
    if (snapshot.cluster, snapshot.account, snapshot.user) != (
        request.spec.cluster,
        request.spec.account,
        request.spec.user,
    ):
        raise SubmissionDisabled("user association snapshot scope differs from request")
    if type(snapshot.effective_uid) is not int or snapshot.effective_uid != _effective_identity(
        request.spec.user
    ):
        raise SubmissionDisabled("snapshot does not belong to the effective user")
    began, completed, current = map(utc, (snapshot.observed_at, snapshot.completed_at, now))
    if not began <= completed <= current or not 0 <= (current - began).total_seconds() <= 30:
        raise SubmissionDisabled("user association evidence is stale or from the future")
    if utc(snapshot.since) > began:
        raise SubmissionDisabled("invalid accounting observation interval")
    if (
        type(ledger) is not dict
        or ledger.get("cluster") != snapshot.cluster
        or ledger.get("account") != snapshot.account
        or type(ledger.get("schema_version")) is not int
        or ledger.get("schema_version") != 1
        or type(ledger.get("intents")) is not dict
    ):
        raise SubmissionDisabled("capacity requires the matching account-scope ledger")
    association_id, limit, jobs = _parsed(snapshot)
    resolutions = {} if resolved_e0 is None else resolved_e0
    if type(resolutions) is not dict or any(key not in ledger["intents"] for key in resolutions):
        raise SubmissionDisabled("invalid E0 terminal resolution scope")
    if resolutions:
        from agentmemeval.path1_v1.operations.e0_reconcile import E0TerminalResolution
        from agentmemeval.path1_v1.operations.resource_terminal import ResourceTerminalResolution
        from agentmemeval.path1_v1.operations.service_terminal import ServiceTerminalResolution
        from agentmemeval.path1_v1.operations.submission_rejection import (
            RejectionTerminalResolution,
        )

        if store is None or any(
            type(value) not in {
                E0TerminalResolution, ResourceTerminalResolution, ServiceTerminalResolution,
                RejectionTerminalResolution,
            }
            for value in resolutions.values()
        ):
            raise SubmissionDisabled("typed durable E0 terminal resolutions required")
        for intent, resolution in resolutions.items():
            if resolution.intent_id != intent:
                raise SubmissionDisabled("E0 terminal resolution intent differs")
            resolution.validate(ledger["intents"][intent], store=store)
    if any(job.matches(request) for job in jobs):
        raise SubmissionDisabled("this intent already exists in scheduler observations")
    active = {job.key: job for job in jobs if job.state in ACTIVE_STATES}
    campaign = {
        job.key
        for job in active.values()
        if job.comment.startswith("ame-v1:" + request.campaign_id + ":")
    }
    unknown_user = unknown_campaign = 0
    for intent, record in ledger["intents"].items():
        try:
            previous = SubmissionRequest(
                record["campaign_id"], record["intent_id"], SubmitSpec.from_payload(record["spec"])
            )
            if (
                intent != previous.intent_id
                or record["request_sha256"] != previous.request_sha256
                or record["status"] not in {"claimed", "submit_unknown", "accepted"}
                or (previous.spec.cluster, previous.spec.account)
                != (snapshot.cluster, snapshot.account)
            ):
                raise ValueError("ledger request identity/status differs")
            known_id = record.get("job_id")
            if known_id is not None and (
                type(known_id) is not str or re.fullmatch(r"[1-9][0-9]*", known_id) is None
            ):
                raise ValueError("ledger job ID is invalid")
            cleanup_failed = record.get("cleanup_persistence_failed", False)
            if type(cleanup_failed) is not bool:
                raise ValueError("ledger cleanup persistence status is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            raise SubmissionDisabled("invalid durable capacity reservation") from exc
        if intent == request.intent_id:
            if previous.request_sha256 != request.request_sha256:
                raise SubmissionDisabled("current intent reservation contains different bytes")
            continue  # The executor's fresh pre-exec check includes its own reserved intent.
        if previous.spec.user != snapshot.user:
            continue  # A different user's association cannot consume this per-user limit.
        represented = [
            job
            for job in jobs
            if job.matches(previous) and (known_id is None or known_id == job.job_id)
        ]
        if len(represented) > 1:
            raise SubmissionDisabled("one durable intent matches multiple scheduler allocations")
        if intent in resolutions:
            expected_state = getattr(resolutions[intent], "scheduler_state", "DEADLINE")
            if cleanup_failed or any(
                job.state != expected_state or job.job_id != resolutions[intent].job_id
                for job in represented
            ):
                raise SubmissionDisabled("scheduler/cleanup contradicts sealed E0 terminal")
            continue
        if (
            not represented
            or represented[0].state not in ACTIVE_STATES
        ):
            # Terminal accounting proves scheduler status, not completion of
            # the durable submission/cleanup protocol. Keep the reservation
            # until the bound E0 reconciliation above verifies its release.
            unknown_user += 1
            unknown_campaign += previous.campaign_id == request.campaign_id
    total, own = len(active) + unknown_user, len(campaign) + unknown_campaign
    effective_limit = min(limit, 8)
    if min(stage_cap - own, effective_limit - total - 2) < 1:
        raise SubmissionDisabled("fresh user association/stage capacity has no admission slot")
    return {
        "snapshot_sha256": snapshot.evidence_sha256,
        "raw_observation": snapshot.payload(),
        "limit_evidence_sha256": content_digest(snapshot.commands[0].payload()),
        "observed_at": began.isoformat(),
        "account_occupancy": total,  # Legacy key; scope below explicitly narrows its meaning.
        "campaign_occupancy": own,
        "stage_cap": stage_cap,
        "limit": effective_limit,
        "reserve": 2,
        "scope": "user_association",
        "user": snapshot.user,
        "effective_uid": snapshot.effective_uid,
        "association_id": association_id,
        "partition": "",
        "full_account_visibility": False,
        "allocation_guaranteed": False,
    }
