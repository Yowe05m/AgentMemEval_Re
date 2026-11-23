"""Seal one hash-proven site-policy rejection, without resubmission authority.

The original helper discarded stderr bytes. An exact reconstructed byte string
can satisfy its recorded SHA-256 commitment; a similar error or dry-run alone
cannot. Fresh full user-association observations, immutable original inputs,
writer exit, transport cleanup and the authoritative claim are still required.
No model is loaded or claimed to have been verified by this reconciliation.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.e0_service_cli import read_prepared
from agentmemeval.path1_v1.operations.slurm import (
    READ_ENV,
    CommandResult,
    CommandTransport,
    job_comment,
    job_name,
    utc,
)
from agentmemeval.path1_v1.operations.state import CampaignStore, RunClaim
from agentmemeval.path1_v1.operations.submission_process import write_exclusive_json
from agentmemeval.path1_v1.operations.user_capacity import PinnedUserCommand, _effective_identity

KIND = "path1_submission_rejection_terminal_v1"
EVIDENCE_KIND = "path1_submission_rejection_evidence_v1"
CLASSIFICATION = "site_policy_rejected_before_job"
POLICY = "slurm_lua_n16r4_memory_message_v1"
POLICY_STDERR = (
    "sbatch: error: slurm_job_submit_lua, N16R4区每卡默认分配 55GB 内存, "
    "不允许超额申请内存。可申请更多卡数获取更多内存\n"
    "sbatch: error: Batch job submission failed: Unspecified error\n"
).encode()
POLICY_SHA256 = hashlib.sha256(POLICY_STDERR).hexdigest()
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
# Original executor._clean_environment() submission defaults, not READ_ENV.
SUBMISSION_ENV = {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"}
_ISSUER = object()
_MODULE = "src/agentmemeval/path1_v1/operations/submission_rejection.py"


def _require(value: bool, reason: str) -> None:
    er._require(value, "submission rejection: " + reason)


def _evidence_path(store: CampaignStore, intent: str) -> Path:
    er.digest(intent)
    return store.directory / "submission_rejections" / intent / "evidence.json"


def _verifier_identity(repository: Path, config: dict) -> dict:
    identity = er._verifier_identity(repository, config)
    own = Path(__file__).resolve()
    _require(own == repository / _MODULE, "verifier imported from another checkout")
    _require(
        any(
            row["path"] == _MODULE and row["sha256"] == er._sha(own)
            for row in identity["operations_files"]
        ),
        "verifier is not published",
    )
    return identity


def _candidate(path: Path, expected_sha: str, result_path: Path) -> bytes:
    _require(er._sha(er._path(path)) == er.digest(expected_sha), "candidate evidence changed")
    value, result = path.read_bytes(), er._json(result_path)
    _require(
        value == POLICY_STDERR,
        "unknown rejection body; keep reservation occupied",
    )
    _require(
        hashlib.sha256(value).hexdigest() == result["stderr_sha256"] == POLICY_SHA256,
        "candidate does not reproduce the original stderr commitment",
    )
    return value


def _bound_inputs(store, descriptor_path, descriptor_sha256, original_code_sha, record, claim):
    descriptor_path = er._path(descriptor_path)
    _require(descriptor_path.name == "submission.json", "descriptor filename differs")
    run = descriptor_path.parent
    descriptor, config, request = read_prepared(run, descriptor_sha256)
    probe, contract = er._json(run / "probe_request.json"), er._json(run / "contract.json")
    identity = contract["runtime_identity"]
    repository = er._path(descriptor["repository_root"])
    _require(
        re.fullmatch("[0-9a-f]{40}", original_code_sha) is not None
        and identity["core_sha"] == identity["ops_sha"] == original_code_sha,
        "original source revision differs",
    )
    _require(
        er._historical_identity(repository, identity, allow_model=True) == config,
        "prepared configuration differs from historical Git",
    )
    _require(
        config["resource_profile_version"] == 2
        and config["language_model_only"] is True
        and config["memory_mib"] == 65536
        and config["cpus_per_task"] == 8
        and request.spec.gpus == 1
        and request.spec.walltime_seconds == 7200
        and (request.spec.cluster, request.spec.account, request.spec.user, request.spec.partition)
        == ("n16r4", "scze856", "scze856", "gpu")
        and contract["phase"] == "E0"
        and contract["job"]["kind"] == "service_probe"
        and contract["job"]["attempt"] == 1
        and contract["job"]["retry_evidence"] is None,
        "unsupported original site/service profile",
    )
    scope = content_digest({"cluster": request.spec.cluster, "account": request.spec.account})
    _require(
        store.campaign_id == "executor_" + scope[:32]
        and store.directory == er._path(config["control_root"]) / "ie1" / scope[:16],
        "not the original authoritative store",
    )
    invocation = store.directory / "i" / request.intent_id[:16]
    _require(
        record["status"] == "submit_unknown"
        and record["job_id"] is None
        and record["result_sha256"] is None
        and record["submitter_finished"] is True
        and record.get("cleanup_persistence_failed", False) is False
        and record["intent_id"] == request.intent_id == claim.intent_id == claim.run_id
        and record["campaign_id"] == request.campaign_id
        and record["spec"] == request.spec.payload()
        and record["request_sha256"] == request.request_sha256
        and record["worker_request_sha256"] == probe["request_sha256"]
        and record["probe_request_file_sha256"] == er._sha(run / "probe_request.json")
        and er._path(record["probe_request_path"]) == run / "probe_request.json"
        and er._path(record["invocation_dir"]) == invocation
        and er._path(claim.run_directory) == er._path(request.spec.work_dir)
        and claim.metadata
        == {"purpose": "slurm_job_container", "request_sha256": request.request_sha256},
        "original unknown intent/claim differs",
    )
    _require(
        list(er._path(request.spec.work_dir).iterdir()) == [],
        "rejected job must have no Slurm/worker/service output",
    )
    manifest = er._json(invocation / "invocation.json")
    ready, go, result = [
        er._json(invocation / name) for name in ("ready.json", "go.json", "result.json")
    ]
    cleanup = er._json(invocation / "local_cleanup.json")
    # This seal is for the historical v1 helper only. Newer stream-capture
    # protocols need their own validated complete result, never silent fallback.
    er._keys(
        manifest,
        {
            "schema_version",
            "nonce",
            "request_sha256",
            "argv",
            "cwd",
            "environment",
            "stdin_path",
            "stdin_sha256",
            "input_files",
            "wait_seconds",
            "command_timeout_seconds",
            "deadline_monotonic",
            "expected_effective_user",
        },
        "original legacy helper manifest",
    )
    er._keys(
        ready,
        {"nonce", "request_sha256", "manifest_sha256", "submitter", "effective_user"},
        "original legacy helper ready",
    )
    er._keys(
        result,
        set(ready)
        | {
            "command_started_ns",
            "command_finished_ns",
            "returncode",
            "timed_out",
            "stdout",
            "stdout_sha256",
            "stderr_sha256",
        },
        "original legacy helper result",
    )
    _require(
        type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
        "only original helper v1 is supported",
    )
    _require(
        er._sha(invocation / "invocation.json") == record["manifest_sha256"]
        and ready == go == record["ready"]
        and all(result.get(key) == value for key, value in ready.items())
        and result["timed_out"] is False
        and type(result["returncode"]) is int
        and result["returncode"] == 1
        and result["stdout"] is None
        and result["stdout_sha256"] == EMPTY_SHA256
        and result["stderr_sha256"] == POLICY_SHA256,
        "one completed exact policy rejection is not proven",
    )
    _require(
        type(result["command_started_ns"]) is int
        and type(result["command_finished_ns"]) is int
        and 0 < result["command_started_ns"] <= result["command_finished_ns"]
        and utc(datetime.fromisoformat(record["created_at"])).timestamp() * 1e9
        <= result["command_started_ns"],
        "command timestamps differ",
    )
    expected_argv = [config["commands"]["sbatch"]["path"], *request.sbatch_argv()[1:-1]]
    _require(
        manifest["argv"] == expected_argv
        and manifest["cwd"] == request.spec.work_dir
        and manifest["environment"] == SUBMISSION_ENV
        and manifest["request_sha256"] == request.request_sha256
        and manifest["nonce"] == ready["nonce"]
        and manifest["expected_effective_user"]
        == ready["effective_user"]
        == record["effective_user"]
        and er._path(manifest["stdin_path"]) == invocation / "submitted_script.sh"
        and er._sha(invocation / "submitted_script.sh")
        == er._sha(run / "probe.sh")
        == manifest["stdin_sha256"]
        == request.spec.script_sha256,
        "original literal argv/environment/script differs",
    )
    pins = {}
    for row in manifest["input_files"]:
        er._keys(row, {"path", "sha256"}, "original input pin")
        er.digest(row["sha256"])
        _require(
            row["path"] not in pins or pins[row["path"]] == row["sha256"],
            "conflicting original input pins",
        )
        pins[row["path"]] = row["sha256"]
    for row in (
        identity["core_files"]
        + identity["operations_files"]
        + [identity["config"], identity["dependency_lock"]]
    ):
        original = str(PurePosixPath(descriptor["repository_root"]) / row["path"])
        _require(pins.get(original) == row["sha256"], "historical submission pin differs")
    _require(
        pins.get(probe["model_manifest_path"]) == identity["model"]["manifest_sha256"]
        and pins.get(config["commands"]["sbatch"]["path"])
        == config["commands"]["sbatch"]["sha256"],
        "model/command pin differs",
    )
    _require(
        cleanup == record["local_cleanup"]
        and cleanup["binding"]
        == {
            "campaign_id": request.campaign_id,
            "intent_id": request.intent_id,
            "job_id": None,
            "source_sha256": content_digest(identity),
            "role": "submission_transport",
            "service_instance_id": manifest["nonce"],
        }
        and cleanup["errors"] == []
        and cleanup["remaining_non_zombie_pids"] == []
        and cleanup["group_cleanup_verified"] is True
        and cleanup["leader_reaped"] is True
        and type(cleanup["returncode"]) is int
        and cleanup["returncode"] == 0,
        "transport cleanup is incomplete or differs",
    )
    child = cleanup["identity"]
    _require(
        child["pid"] == child["pgid"] == child["session_id"], "transport process group differs"
    )
    for key in ("pid", "start_marker", "host", "boot_id", "effective_uid"):
        _require(child[key] == ready["submitter"][key], "transport identity differs")
    return dict(
        request=request,
        descriptor=descriptor,
        probe=probe,
        contract=contract,
        config=config,
        run=run,
        invocation=invocation,
        repository=repository,
        result=result,
    )


def _read_argv(request, config):
    scope = (f"--clusters={request.spec.cluster}", f"--user={request.spec.user}")
    return (
        (
            config["commands"]["squeue"]["path"],
            "--noheader",
            "--all",
            "--array",
            "--local",
            "--states=all",
            *scope,
            f"--account={request.spec.account}",
            "--format=%i|%u|%a|%j|%T|%k|%V|%Z",
        ),
        (
            config["commands"]["squeue"]["path"],
            "--steps",
            "--all",
            "--noheader",
            *scope,
            "--format=%i",
        ),
        (
            config["commands"]["sacct"]["path"],
            "--noheader",
            "--parsable2",
            "--array",
            "--allocations",
            "--duplicates",
            *scope,
            f"--accounts={request.spec.account}",
            "--starttime="
            + utc(datetime.fromisoformat(config["accounting_since"])).strftime("%Y-%m-%dT%H:%M:%S"),
            "--format=Cluster,JobID,User,Account,JobName%200,State%40,Comment%200,Submit,WorkDir%4096",
        ),
    )


@dataclass(frozen=True)
class RejectionObservation:
    observed_at: datetime
    completed_at: datetime
    effective_uid: int
    commands: tuple[dict, ...]

    def payload(self):
        return {
            "observed_at": utc(self.observed_at).isoformat(),
            "completed_at": utc(self.completed_at).isoformat(),
            "effective_uid": self.effective_uid,
            "commands": list(self.commands),
        }


class RejectionReader:
    def __init__(
        self,
        *,
        transport: CommandTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.transport, self.clock = transport, clock

    def read(self, request, config):
        began, uid = utc(self.clock()), _effective_identity(request.spec.user)
        commands = []
        for argv in _read_argv(request, config):
            name = Path(argv[0]).name
            pin = PinnedUserCommand(**config["commands"][name])
            pin.validate(name)
            if self.transport is None:
                process = subprocess.run(
                    argv,
                    env=dict(READ_ENV),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    shell=False,
                    check=False,
                    timeout=8,
                )
                result = CommandResult(
                    process.returncode,
                    process.stdout.decode("utf-8"),
                    process.stderr.decode("utf-8"),
                )
            else:
                result = self.transport.run(
                    argv, environment=dict(READ_ENV), inherit_environment=False, timeout_seconds=8
                )
            _require(
                type(result) is CommandResult
                and type(result.returncode) is int
                and result.returncode == 0
                and type(result.stdout) is str
                and type(result.stderr) is str
                and not result.stderr.strip(),
                "scheduler read failed/warned",
            )
            pin.validate(name)
            commands.append(
                dict(
                    argv=list(argv),
                    pin=asdict(pin),
                    returncode=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
        _require(_effective_identity(request.spec.user) == uid, "effective user changed")
        return RejectionObservation(began, utc(self.clock()), uid, tuple(commands))


def _validate_observation(value, bound, record, checked_at):
    er._keys(
        value,
        {"observed_at", "completed_at", "effective_uid", "commands"},
        "rejection scheduler evidence",
    )
    began = utc(datetime.fromisoformat(value["observed_at"]))
    ended = utc(datetime.fromisoformat(value["completed_at"]))
    _require(
        began <= ended <= checked_at and (checked_at - began).total_seconds() <= 30,
        "scheduler observation stale/future",
    )
    _require(
        type(value["effective_uid"]) is int
        and value["effective_uid"] == record["effective_user"]["uid"],
        "wrong scheduler UID",
    )
    request, config = bound["request"], bound["config"]
    _require(
        type(value["commands"]) is list and len(value["commands"]) == 3,
        "complete full-scope scheduler reads required",
    )
    rows = value["commands"]
    for row, argv in zip(rows, _read_argv(request, config), strict=True):
        er._keys(row, {"argv", "pin", "returncode", "stdout", "stderr"}, "scheduler command")
        _require(
            row["argv"] == list(argv)
            and row["pin"] == config["commands"][Path(argv[0]).name]
            and type(row["returncode"]) is int
            and row["returncode"] == 0
            and type(row["stdout"]) is str
            and type(row["stderr"]) is str
            and not row["stderr"].strip(),
            "filtered or unclean scheduler observation",
        )
    expected = (
        job_name(request.campaign_id, request.intent_id),
        job_comment(request.campaign_id, request.intent_id),
        request.spec.work_dir,
    )
    for line in rows[0]["stdout"].splitlines():
        fields = line.split("|")
        _require(
            len(fields) == 8
            and fields[1:3] == [request.spec.user, request.spec.account]
            and re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?", fields[0]),
            "queue scope/format differs",
        )
        _require(
            not any(
                actual == target
                for actual, target in zip((fields[3], fields[5], fields[7]), expected, strict=True)
            ),
            "target identity appears in queue",
        )
    # No job ID exists to safely associate an ambiguous active step. Be conservative.
    _require(rows[1]["stdout"].strip() == "", "active user steps prevent rejection seal")
    for line in rows[2]["stdout"].splitlines():
        fields = line.split("|")
        _require(
            len(fields) == 9
            and fields[0] == request.spec.cluster
            and fields[2:4] == [request.spec.user, request.spec.account]
            and re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?", fields[1]),
            "accounting scope/format differs",
        )
        _require(
            not any(
                actual == target
                for actual, target in zip((fields[4], fields[6], fields[8]), expected, strict=True)
            ),
            "target identity appears in accounting",
        )
    since = utc(datetime.fromisoformat(config["accounting_since"]))
    _require(
        since <= utc(datetime.fromisoformat(record["created_at"])) <= began,
        "accounting interval omits original submission",
    )
    _require(
        bound["result"]["command_finished_ns"] <= int(began.timestamp() * 1e9),
        "scheduler read precedes command completion",
    )


def _binding(store, bound, record, claim, descriptor_sha):
    return dict(
        campaign_id=record["campaign_id"],
        intent_id=claim.intent_id,
        job_id=None,
        submission_request_sha256=record["request_sha256"],
        descriptor_sha256=descriptor_sha,
        store_campaign_id=store.campaign_id,
        claim_token_sha256=hashlib.sha256(claim.token.encode()).hexdigest(),
        original_source_identity_sha256=content_digest(bound["contract"]["runtime_identity"]),
        command_invocation_count=1,
        accepted_submission_count=0,
    )


def _terminal(path, evidence):
    return {
        "schema_version": 1,
        "kind": KIND,
        **evidence["binding"],
        "evidence_path": str(path),
        "evidence_file_sha256": er._sha(path),
        "phase": "E0",
        "classification": CLASSIFICATION,
        "policy": POLICY,
        "retry_eligible": False,
        "runtime_admission": False,
        "scientific_admission": False,
    }


@dataclass(frozen=True)
class RejectionTerminalResolution:
    intent_id: str
    campaign_id: str
    submission_request_sha256: str
    descriptor_sha256: str
    prior_terminal_receipt_sha256: str
    evidence_file_sha256: str
    original_code_sha: str
    verifier_code_sha: str
    job_id: None = None
    scheduler_state: None = None
    phase: str = "E0"
    classification: str = CLASSIFICATION
    retry_eligible: bool = False
    scientific_admission: bool = False
    _issuer: Any = field(default=None, compare=False, repr=False)

    def validate(self, record, *, store):
        _require(self._issuer is _ISSUER, "unissued terminal capability")
        current = resolve_rejection(store, self.intent_id)
        _require(
            current == self and current.submission_request_sha256 == record["request_sha256"],
            "terminal capability differs",
        )
        return current


def validate_rejection_terminal(record: dict, claim: RunClaim, *, store: CampaignStore):
    snapshot, actual_record, actual_claim = er._stored(store, claim.intent_id)
    _require(
        record == actual_record and claim == actual_claim and claim.status == "failed",
        "not the current authoritative failed claim",
    )
    path = _evidence_path(store, claim.intent_id)
    terminal = claim.terminal_receipt
    _require(
        type(terminal) is dict
        and terminal.get("kind") == KIND
        and terminal.get("evidence_path") == str(path)
        and terminal.get("evidence_file_sha256") == er._sha(er._path(path)),
        "terminal seal differs",
    )
    evidence = er._json(path)
    er._keys(
        evidence,
        {
            "schema_version",
            "kind",
            "binding",
            "descriptor_path",
            "original_code_sha",
            "baseline_manifest",
            "baseline_manifest_sha256",
            "rejection_evidence_path",
            "rejection_evidence_sha256",
            "original_record_sha256",
            "claim_before",
            "generation_before",
            "sealed_at",
            "first_checked_at",
            "original_inventory",
            "first_inventory",
            "second_inventory",
            "quiet_seconds",
            "scheduler_before",
            "scheduler_after",
            "processes_before",
            "processes_after",
            "verifier_runtime_identity",
        },
        "rejection seal",
    )
    _require(
        type(evidence["schema_version"]) is int
        and evidence["schema_version"] == 1
        and evidence["kind"] == EVIDENCE_KIND,
        "unsupported rejection seal",
    )
    bound = _bound_inputs(
        store,
        er._path(evidence["descriptor_path"]),
        terminal["descriptor_sha256"],
        evidence["original_code_sha"],
        record,
        claim,
    )
    _candidate(
        er._path(evidence["rejection_evidence_path"]),
        evidence["rejection_evidence_sha256"],
        bound["invocation"] / "result.json",
    )
    _require(
        terminal == _terminal(path, evidence)
        and evidence["binding"]
        == _binding(store, bound, record, claim, terminal["descriptor_sha256"])
        and evidence["original_record_sha256"] == content_digest(record),
        "original bindings changed",
    )
    old = evidence["claim_before"]
    ignored = {"status", "terminal_receipt", "updated_generation"}
    _require(
        old["status"] == "started"
        and old["terminal_receipt"] is None
        and {k: v for k, v in old.items() if k not in ignored}
        == {k: v for k, v in asdict(claim).items() if k not in ignored}
        and type(evidence["generation_before"]) is int
        and claim.updated_generation == evidence["generation_before"] + 1
        and snapshot.generation >= claim.updated_generation,
        "original claim transition differs",
    )
    baseline = er._baseline(
        er._path(evidence["baseline_manifest"]), evidence["baseline_manifest_sha256"]
    )
    inventory = er._content_inventory(bound["run"], bound["invocation"])
    _require(
        evidence["original_inventory"] == baseline == er._content(inventory)
        and evidence["first_inventory"] == evidence["second_inventory"] == inventory
        and type(evidence["quiet_seconds"]) in (int, float)
        and math.isfinite(evidence["quiet_seconds"])
        and evidence["quiet_seconds"] >= 10,
        "original artifacts/quiescence changed",
    )
    first = utc(datetime.fromisoformat(evidence["first_checked_at"]))
    sealed = utc(datetime.fromisoformat(evidence["sealed_at"]))
    _require(
        first <= utc(datetime.fromisoformat(evidence["scheduler_after"]["observed_at"])) <= sealed
        and (sealed - first).total_seconds() >= 10,
        "observation/quiet sequence differs",
    )
    _validate_observation(evidence["scheduler_before"], bound, record, first)
    _validate_observation(evidence["scheduler_after"], bound, record, sealed)
    for key in ("processes_before", "processes_after"):
        er._validate_processes(evidence[key], record)
    verifier = evidence["verifier_runtime_identity"]
    er._historical_identity(bound["repository"], verifier)
    _require(
        any(row["path"] == _MODULE for row in verifier["operations_files"]),
        "seal verifier module absent from historical source",
    )
    return RejectionTerminalResolution(
        claim.intent_id,
        record["campaign_id"],
        record["request_sha256"],
        terminal["descriptor_sha256"],
        content_digest(terminal),
        terminal["evidence_file_sha256"],
        evidence["original_code_sha"],
        verifier["core_sha"],
        _issuer=_ISSUER,
    )


def resolve_rejection(store, intent):
    _, record, claim = er._stored(store, er.digest(intent))
    return validate_rejection_terminal(record, claim, store=store)


def reconcile_rejection(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    original_code_sha: str,
    baseline_manifest: Path,
    baseline_manifest_sha256: str,
    rejection_evidence_path: Path,
    rejection_evidence_sha256: str,
    reader: RejectionReader | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    check_only: bool = False,
):
    """Default live readers only in CLI; library injection is an explicit test boundary."""
    descriptor = er._json(descriptor_path)
    reader = RejectionReader(clock=clock) if reader is None else reader
    with store.lock():
        snapshot, record, claim = er._stored(store, descriptor["intent_id"])
        path = _evidence_path(store, claim.intent_id)
        if claim.status == "failed":
            result = validate_rejection_terminal(record, claim, store=store)
            prior = er._json(path)
            _require(
                result.descriptor_sha256 == descriptor_sha256
                and prior["original_code_sha"] == original_code_sha
                and prior["descriptor_path"] == str(descriptor_path)
                and prior["baseline_manifest"] == str(baseline_manifest)
                and prior["baseline_manifest_sha256"] == baseline_manifest_sha256
                and prior["rejection_evidence_path"] == str(rejection_evidence_path)
                and prior["rejection_evidence_sha256"] == rejection_evidence_sha256,
                "repeat supplied different original evidence",
            )
            return result
        _require(
            claim.status == "started" and claim.terminal_receipt is None,
            "only original started claim may close",
        )
        _require(not path.parent.exists(), "pending immutable audit exists; no overwrite/retry")
        bound = _bound_inputs(
            store, descriptor_path, descriptor_sha256, original_code_sha, record, claim
        )
        _candidate(
            rejection_evidence_path, rejection_evidence_sha256, bound["invocation"] / "result.json"
        )
        verifier = _verifier_identity(
            bound["repository"],
            {**bound["config"], "config_relative": descriptor["config_relative"]},
        )
        baseline = er._baseline(baseline_manifest, baseline_manifest_sha256)
        before = reader.read(bound["request"], bound["config"])
        _require(type(before) is RejectionObservation, "typed live observation required")
        first_checked = utc(clock())
        _validate_observation(before.payload(), bound, record, first_checked)
        before_processes = er._processes_gone(record)
        first_inventory = er._content_inventory(bound["run"], bound["invocation"])
        _require(er._content(first_inventory) == baseline, "original artifact baseline differs")
        if check_only:
            _require(
                er._stored(store, claim.intent_id) == (snapshot, record, claim),
                "authoritative state changed",
            )
            er._validate_processes(before_processes, record)
            _validate_observation(before.payload(), bound, record, utc(clock()))
            return dict(
                kind="path1_submission_rejection_preflight",
                classification=CLASSIFICATION,
                intent_id=claim.intent_id,
                claim_released=False,
                retry_eligible=False,
                runtime_admission=False,
                scientific_admission=False,
            )
        began = monotonic()
        sleep(10.1)
        second_inventory = er._content_inventory(bound["run"], bound["invocation"])
        elapsed = monotonic() - began
        _require(
            math.isfinite(elapsed) and elapsed >= 10 and first_inventory == second_inventory,
            "original artifacts are not quiescent",
        )
        after_processes = er._processes_gone(record)
        after = reader.read(bound["request"], bound["config"])
        _require(type(after) is RejectionObservation, "typed live observation required")
        sealed = utc(clock())
        _validate_observation(after.payload(), bound, record, sealed)
        for processes in (before_processes, after_processes):
            er._validate_processes(processes, record)
        _require(
            er._stored(store, claim.intent_id) == (snapshot, record, claim),
            "authoritative state changed before seal",
        )
        evidence = dict(
            schema_version=1,
            kind=EVIDENCE_KIND,
            binding=_binding(store, bound, record, claim, descriptor_sha256),
            descriptor_path=str(descriptor_path),
            original_code_sha=original_code_sha,
            baseline_manifest=str(baseline_manifest),
            baseline_manifest_sha256=baseline_manifest_sha256,
            rejection_evidence_path=str(rejection_evidence_path),
            rejection_evidence_sha256=rejection_evidence_sha256,
            original_record_sha256=content_digest(record),
            claim_before=asdict(claim),
            generation_before=snapshot.generation,
            sealed_at=sealed.isoformat(),
            first_checked_at=first_checked.isoformat(),
            original_inventory=baseline,
            first_inventory=first_inventory,
            second_inventory=second_inventory,
            quiet_seconds=elapsed,
            scheduler_before=before.payload(),
            scheduler_after=after.payload(),
            processes_before=before_processes,
            processes_after=after_processes,
            verifier_runtime_identity=verifier,
        )
        path.parent.mkdir(parents=True, exist_ok=False)
        write_exclusive_json(path, evidence)
        _require(
            er._json(path) == evidence
            and er._content_inventory(bound["run"], bound["invocation"]) == second_inventory,
            "audit/original artifacts changed before CAS",
        )
        terminal = _terminal(path, evidence)
        # Publishing/hash scans may be slow. Recheck with a new clock only after
        # every pre-CAS filesystem operation, never reuse the earlier seal time.
        _validate_observation(after.payload(), bound, record, utc(clock()))
        store.finish_claim(
            expected_generation=snapshot.generation,
            run_id=claim.run_id,
            token=claim.token,
            status="failed",
            receipt=terminal,
        )
        return resolve_rejection(store, claim.intent_id)
