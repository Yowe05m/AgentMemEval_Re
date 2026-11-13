"""Close one accepted E0 resource intent that Slurm demonstrably never started.

Historical source is verified against Git objects, not today's working tree.
The current verifier is separately content-bound. Closing a claim records a
technical failure, never worker success, GPU availability or research admission.
Only the default live readers are exposed by the CLI; injected clocks/readers
are explicit library trust boundaries for offline fault tests.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.path1_v1.operations.closure import inventory_tree
from agentmemeval.path1_v1.operations.contracts import (
    _read_json,
    _validate_identity,
    build_runtime_identity,
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.slurm import (
    READ_ENV,
    CommandResult,
    CommandTransport,
    SubmissionRequest,
    SubmitSpec,
    digest,
    job_comment,
    job_name,
    utc,
)
from agentmemeval.path1_v1.operations.state import CampaignStore, RunClaim
from agentmemeval.path1_v1.operations.submission_process import write_exclusive_json
from agentmemeval.path1_v1.operations.user_capacity import PinnedUserCommand, _effective_identity

NAMESPACE = "durable_intent_executor_v1"
KIND = "path1_e0_unstarted_deadline_terminal_v1"
_ISSUER = object()


class E0ReconciliationError(ValueError):
    pass


def _require(value: bool, reason: str) -> None:
    if not value:
        raise E0ReconciliationError(reason)


def _path(value: str | Path) -> Path:
    """Tests may map archived remote locations; production has no remapping option."""
    path = Path(value)
    _require(path.is_absolute(), "absolute evidence path required")
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink evidence path")
    _require(path.resolve(strict=True) == path, "canonical existing evidence path required")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = _read_json(_path(path))
    _require(type(value) is dict, "evidence JSON must be an object")
    return value


def _keys(value: dict, names: set[str], description: str) -> None:
    _require(type(value) is dict and set(value) == names, "invalid " + description + " fields")


def _git(repository: Path, *arguments: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=data,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(result.returncode == 0, "historical Git evidence unavailable")
    return result.stdout


def _historical_identity(repository: Path, identity: dict) -> dict:
    """Hash every declared historical blob in one cat-file batch; never import it."""
    _validate_identity(identity, scientific=False)
    commit = identity["core_sha"]
    _require(identity["ops_sha"] == commit, "mixed historical source revisions unsupported")
    _require(not identity["dirty"] and identity["content_matches_head"], "historical dirty source")
    for name in (
        "untracked_declared_paths",
        "uncovered_execution_paths",
        "untracked_execution_paths",
        "head_content_mismatches",
    ):
        _require(identity[name] == [], "historical source closure is incomplete")
    _require(identity["model"] is None, "resource reconciliation cannot validate a model run")
    tree = {}
    for entry in _git(repository, "ls-tree", "-rz", commit).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            tree[name.decode("utf-8")] = (mode, kind, oid)
    source = identity["core_files"] + identity["operations_files"]
    expected_dependencies = {
        name
        for name in tree
        if name.startswith(("src/", "prompts/", "tools/"))
        or name in {"pyproject.toml", ".gitattributes"}
    }
    _require(
        {row["path"] for row in source} == expected_dependencies,
        "historical execution dependency closure differs",
    )
    records = source + [identity["config"], identity["dependency_lock"]]
    objects = []
    for row in records:
        entry = tree.get(row["path"])
        _require(
            entry is not None and entry[0] in {"100644", "100755"} and entry[1] == "blob",
            "historical input is not a regular tracked blob",
        )
        objects.append(entry[2])
    stream = io.BytesIO(
        _git(repository, "cat-file", "--batch", data=("\n".join(objects) + "\n").encode())
    )
    config = None
    for row, oid in zip(records, objects, strict=True):
        header = stream.readline().decode().strip().split()
        _require(len(header) == 3 and header[:2] == [oid, "blob"], "invalid Git batch response")
        payload = stream.read(int(header[2]))
        _require(
            stream.read(1) == b"\n"
            and len(payload) == row["size"]
            and hashlib.sha256(payload).hexdigest() == row["sha256"],
            "historical content does not match declared source",
        )
        if row == identity["config"]:
            config = json.loads(payload)
    _require(stream.read() == b"" and type(config) is dict, "historical config unavailable")
    return config


def _verifier_identity(repository: Path, config: dict) -> dict:
    paths = _git(repository, "ls-files", "-z").decode().strip("\0").split("\0")
    dependencies = {
        p
        for p in paths
        if p.startswith(("src/", "prompts/", "tools/")) or p in {"pyproject.toml", ".gitattributes"}
    }
    operations = {p for p in dependencies if p.startswith("src/agentmemeval/path1_v1/operations/")}
    own = Path(__file__).resolve()
    _require(
        own == repository / "src/agentmemeval/path1_v1/operations/e0_reconcile.py",
        "verifier loaded from a different checkout",
    )
    _require(own.relative_to(repository).as_posix() in operations, "verifier is not published")
    identity = build_runtime_identity(
        repository,
        core_paths=sorted(dependencies - operations),
        operations_paths=sorted(operations),
        config_path=config["config_relative"],
        dependency_lock_path=config["dependency_lock"],
    )
    _historical_identity(repository, identity)
    return identity


def _stored(store: CampaignStore, intent: str) -> tuple[Any, dict, RunClaim]:
    snapshot = store.read_registry()
    ledger = snapshot.registry.get(NAMESPACE)
    _require(
        type(ledger) is dict and type(ledger.get("intents")) is dict,
        "authoritative executor ledger unavailable",
    )
    record = ledger["intents"].get(intent)
    claims = [claim for claim in snapshot.claims if claim.intent_id == intent]
    _require(type(record) is dict and len(claims) == 1, "unique durable intent/claim required")
    return snapshot, record, claims[0]


def _content_inventory(run: Path, invocation: Path) -> list[dict]:
    rows = []
    for prefix, root in (("run", run), ("executor/invocation", invocation)):
        rows.extend({**row, "path": prefix + "/" + row["path"]} for row in inventory_tree(root))
    return sorted(rows, key=lambda row: row["path"])


def _content(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("path", "size", "sha256")} for row in rows]


def _baseline(path: Path, expected_sha256: str) -> list[dict]:
    digest(expected_sha256)
    _require(_sha(_path(path)) == expected_sha256, "pinned original artifact manifest changed")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(
            reader.fieldnames == ["relative_path", "size_bytes", "sha256"],
            "unsupported original artifact manifest",
        )
        rows = []
        for row in reader:
            relative = row["relative_path"]
            parts = PurePosixPath(relative)
            _require(
                not parts.is_absolute()
                and ".." not in parts.parts
                and str(parts) == relative
                and "\\" not in relative,
                "unsafe baseline path",
            )
            if relative.startswith(("run/", "executor/invocation/")):
                _require(
                    re.fullmatch(r"0|[1-9][0-9]*", row["size_bytes"]) is not None,
                    "invalid baseline size",
                )
                rows.append(
                    {
                        "path": relative,
                        "size": int(row["size_bytes"]),
                        "sha256": digest(row["sha256"]),
                    }
                )
    _require(
        rows and len({row["path"].casefold() for row in rows}) == len(rows),
        "baseline has duplicate/empty content inventory",
    )
    return sorted(rows, key=lambda row: row["path"])


def _bound_inputs(
    store: CampaignStore, descriptor_path: Path, expected_sha: str, record: dict, claim: RunClaim,
    *, require_empty_output: bool = True,
) -> dict:
    descriptor_path = _path(descriptor_path)
    _require(_sha(descriptor_path) == digest(expected_sha), "original descriptor changed")
    descriptor = _json(descriptor_path)
    version = descriptor.get("schema_version")
    _require(type(version) is int and version in {1, 2}, "unsupported descriptor version")
    extra = (
        {"attempt", "queue_start_window_seconds", "scheduler_deadline_offset_seconds"}
        if version == 2
        else set()
    )
    _keys(
        descriptor,
        {
            "schema_version",
            "kind",
            "created_at",
            "repository_root",
            "config_relative",
            "config_sha256",
            "campaign_id",
            "intent_id",
            "spec",
            "submission_request_sha256",
            "probe_request_sha256",
            "runtime_admission",
        }
        | extra,
        "original descriptor",
    )
    _require(
        descriptor["kind"] == "path1_prepared_resource_submission"
        and descriptor["runtime_admission"] is False,
        "not an E0 prepared descriptor",
    )
    request = SubmissionRequest(
        descriptor["campaign_id"],
        descriptor["intent_id"],
        SubmitSpec.from_payload(descriptor["spec"]),
    )
    run = descriptor_path.parent
    _require(
        descriptor_path.name == "submission.json"
        and _path(request.spec.script_path) == run / "probe.sh",
        "descriptor root differs",
    )
    probe = _json(run / "probe_request.json")
    contract_path = run / "contract.json"
    contract = validate_campaign_contract(_json(contract_path))
    _require(
        contract["phase"] == "E0"
        and contract["job"]["kind"] == "resource_probe"
        and contract["runtime_identity"]["model"] is None,
        "only E0 resource jobs can reconcile",
    )
    _require(
        canonical_intent_id(contract) == request.intent_id == probe["intent_id"]
        and request.request_sha256 == descriptor["submission_request_sha256"]
        and _sha(run / "probe_request.json") == descriptor["probe_request_sha256"]
        and _sha(contract_path) == probe["contract_sha256"]
        and _path(probe["contract_path"]) == contract_path
        and probe["request_sha256"]
        == content_digest({key: value for key, value in probe.items() if key != "request_sha256"}),
        "original contract/request/descriptor binding differs",
    )
    repository = _path(descriptor["repository_root"])
    config = _historical_identity(repository, contract["runtime_identity"])
    from agentmemeval.path1_v1.operations.intent_executor import resource_queue_policy

    queue, deadline = resource_queue_policy(config, allow_legacy=True)
    if version == 2:
        _require(
            _sha(_path(run / "site_config.json")) == descriptor["config_sha256"]
            and _json(run / "site_config.json") == config
            and type(descriptor["attempt"]) is int
            and descriptor["attempt"] == contract["job"]["attempt"]
            and type(descriptor["queue_start_window_seconds"]) is int
            and type(descriptor["scheduler_deadline_offset_seconds"]) is int
            and (
                descriptor["queue_start_window_seconds"],
                descriptor["scheduler_deadline_offset_seconds"],
            )
            == (queue, deadline),
            "v2 frozen site config/attempt/queue binding differs",
        )
    _require(
        descriptor["config_relative"] == contract["runtime_identity"]["config"]["path"]
        and descriptor["config_sha256"] == contract["runtime_identity"]["config"]["sha256"],
        "historical site configuration differs",
    )
    scope = content_digest({"cluster": request.spec.cluster, "account": request.spec.account})
    _require(
        store.campaign_id == "executor_" + scope[:32]
        and store.directory == _path(config["control_root"]) / "ie1" / scope[:16],
        "not the original authoritative account store",
    )
    invocation = store.directory / "i" / request.intent_id[:16]
    _require(
        record["status"] == "accepted"
        and record["submitter_finished"] is True
        and record.get("cleanup_persistence_failed", False) is False
        and record["intent_id"] == claim.intent_id == claim.run_id == request.intent_id
        and record["campaign_id"] == request.campaign_id
        and record["spec"] == descriptor["spec"]
        and record["request_sha256"] == request.request_sha256
        and record["worker_request_sha256"] == probe["request_sha256"]
        and _path(record["invocation_dir"]) == invocation
        and _path(claim.run_directory) == _path(request.spec.work_dir)
        and claim.metadata
        == {"purpose": "slurm_job_container", "request_sha256": request.request_sha256},
        "durable accepted intent/claim differs",
    )
    if require_empty_output:
        _require(
            list(_path(request.spec.work_dir).iterdir()) == [],
            "a never-started job cannot contain worker or Slurm output",
        )
    manifest_path = invocation / "invocation.json"
    manifest = _json(manifest_path)
    ready, go, result = (
        _json(invocation / name) for name in ("ready.json", "go.json", "result.json")
    )
    cleanup = _json(invocation / "local_cleanup.json")
    _require(
        _sha(manifest_path) == record["manifest_sha256"]
        and _sha(invocation / "result.json") == record["result_sha256"]
        and ready == go == record["ready"]
        and all(result.get(key) == value for key, value in ready.items())
        and result["timed_out"] is False
        and type(result["returncode"]) is int
        and result["returncode"] == 0
        and result["stderr_sha256"] == hashlib.sha256(b"").hexdigest()
        and result["stdout"] == record["job_id"] + ";" + request.spec.cluster
        # The producer stores stripped display text, but hashes raw sbatch bytes.
        and result["stdout_sha256"]
        in {
            hashlib.sha256((result["stdout"] + ending).encode()).hexdigest()
            for ending in ("", "\n")
        }
        and result["manifest_sha256"] == record["manifest_sha256"]
        and result["request_sha256"] == request.request_sha256,
        "one completed sbatch invocation is not proven",
    )
    _require(
        re.fullmatch(r"[1-9][0-9]*", record["job_id"]) is not None, "non-scalar accepted Slurm job"
    )
    _require(
        manifest["request_sha256"] == request.request_sha256
        and manifest["nonce"] == ready["nonce"]
        and manifest["stdin_sha256"] == request.spec.script_sha256
        and _path(manifest["stdin_path"]) == invocation / "submitted_script.sh"
        and _sha(invocation / "submitted_script.sh")
        == _sha(run / "probe.sh")
        == request.spec.script_sha256
        and "--no-requeue" in manifest["argv"],
        "submitted script/identity differs",
    )
    original_pins = {row["path"]: row["sha256"] for row in manifest["input_files"]}
    for source in (
        contract["runtime_identity"]["core_files"]
        + contract["runtime_identity"]["operations_files"]
        + [contract["runtime_identity"]["config"], contract["runtime_identity"]["dependency_lock"]]
    ):
        original = str(PurePosixPath(descriptor["repository_root"]) / source["path"])
        _require(original_pins.get(original) == source["sha256"], "submission source pin differs")
    _require(
        cleanup["binding"]["intent_id"] == request.intent_id
        and cleanup["binding"]["campaign_id"] == request.campaign_id
        and cleanup["binding"]["role"] == "submission_transport"
        and cleanup["binding"]["service_instance_id"] == manifest["nonce"]
        and cleanup["binding"]["source_sha256"] == content_digest(contract["runtime_identity"])
        and cleanup["errors"] == []
        and cleanup["remaining_non_zombie_pids"] == []
        and cleanup["group_cleanup_verified"] is True
        and cleanup["leader_reaped"] is True
        and type(cleanup["returncode"]) is int
        and cleanup["returncode"] == 0,
        "submission transport cleanup is incomplete",
    )
    for key in ("pid", "start_marker", "host", "boot_id", "effective_uid"):
        _require(
            cleanup["identity"][key] == ready["submitter"][key], "cleanup process identity differs"
        )
    return {
        "request": request,
        "descriptor": descriptor,
        "probe": probe,
        "contract": contract,
        "config": config,
        "run": run,
        "invocation": invocation,
        "repository": repository,
    }


@dataclass(frozen=True)
class DeadlineObservation:
    observed_at: datetime
    completed_at: datetime
    effective_uid: int
    commands: tuple[dict, ...]

    def payload(self) -> dict:
        return {
            "observed_at": utc(self.observed_at).isoformat(),
            "completed_at": utc(self.completed_at).isoformat(),
            "effective_uid": self.effective_uid,
            "commands": list(self.commands),
        }


def _read_argv(
    request: SubmissionRequest, job_id: str, config: dict
) -> tuple[tuple[str, ...], ...]:
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
        # %T is not a supported step format in the observed Slurm 23.11.10 site.
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
            "--duplicates",
            *scope,
            f"--accounts={request.spec.account}",
            f"--jobs={job_id}",
            "--starttime="
            + utc(datetime.fromisoformat(config["accounting_since"])).strftime("%Y-%m-%dT%H:%M:%S"),
            "--format=Cluster,JobID,User,Account,JobName%200,State%40,ExitCode,ElapsedRaw,"
            "Submit,Start,End,NodeList%200,AllocTRES%300,Comment%200,WorkDir%4096",
        ),
    )


class E0DeadlineReader:
    """Pinned real subprocess reads by default; no JSON observation loader grants access."""

    def __init__(
        self,
        *,
        transport: CommandTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.transport, self.clock = transport, clock

    def read(self, request: SubmissionRequest, job_id: str, config: dict) -> DeadlineObservation:
        began, uid = utc(self.clock()), _effective_identity(request.spec.user)
        commands = []
        for argv in _read_argv(request, job_id, config):
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
                and type(result.stderr) is str
                and not result.stderr.strip()
                and type(result.stdout) is str,
                "live scheduler read failed or warned",
            )
            pin.validate(name)
            commands.append(
                {
                    "argv": list(argv),
                    "pin": asdict(pin),
                    "returncode": 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        _require(_effective_identity(request.spec.user) == uid, "effective user changed")
        return DeadlineObservation(began, utc(self.clock()), uid, tuple(commands))


def _validate_scheduler(
    value: dict, request: SubmissionRequest, record: dict, config: dict, *, sealed_at: datetime
) -> None:
    _keys(value, {"observed_at", "completed_at", "effective_uid", "commands"}, "scheduler evidence")
    began = utc(datetime.fromisoformat(value["observed_at"]))
    end = utc(datetime.fromisoformat(value["completed_at"]))
    _require(
        began <= end <= sealed_at and (sealed_at - began).total_seconds() <= 30,
        "scheduler evidence is stale/future or unordered",
    )
    _require(
        type(value["effective_uid"]) is int
        and value["effective_uid"] == record["effective_user"]["uid"],
        "scheduler uid differs",
    )
    rows = value["commands"]
    _require(type(rows) is list and len(rows) == 3, "complete scheduler observations required")
    expected = _read_argv(request, record["job_id"], config)
    for row, argv in zip(rows, expected, strict=True):
        _keys(row, {"argv", "pin", "returncode", "stdout", "stderr"}, "scheduler command")
        _require(
            row["argv"] == list(argv)
            and row["pin"] == config["commands"][Path(argv[0]).name]
            and type(row["returncode"]) is int
            and row["returncode"] == 0
            and type(row["stderr"]) is str
            and not row["stderr"].strip()
            and type(row["stdout"]) is str,
            "filtered or unclean scheduler evidence",
        )
    # These are full user association reads. Never infer an inactive step from a failed query.
    for line in rows[0]["stdout"].splitlines():
        fields = line.split("|")
        _require(
            len(fields) == 8 and fields[1:3] == [request.spec.user, request.spec.account],
            "queue observation scope/format differs",
        )
        _require(
            fields[0] != record["job_id"] and not fields[0].startswith(record["job_id"] + "."),
            "target still appears in the live queue",
        )
    for line in rows[1]["stdout"].splitlines():
        _require(
            re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?\.[A-Za-z0-9_-]+", line) is not None,
            "unsupported or compressed step identity",
        )
        _require(line.split(".", 1)[0] != record["job_id"], "target still has a live step")
    accounting = [line.split("|") for line in rows[2]["stdout"].splitlines() if line]
    _require(
        len(accounting) == 1 and len(accounting[0]) == 15,
        "one allocation and no historical steps required for unstarted reconciliation",
    )
    row = accounting[0]
    _require(
        row[:5]
        == [
            request.spec.cluster,
            record["job_id"],
            request.spec.user,
            request.spec.account,
            job_name(request.campaign_id, request.intent_id),
        ]
        and row[5] == "DEADLINE"
        and row[7] == "0"
        and row[9] == "None"
        and row[11] in {"", "None assigned", "None"}
        and row[12] == ""
        and row[13] == job_comment(request.campaign_id, request.intent_id)
        and row[14] == request.spec.work_dir
        and re.fullmatch(r"[0-9]+:[0-9]+", row[6]) is not None,
        "scheduler does not prove this exact job was never started/allocated",
    )
    submitted, ended = (
        datetime.strptime(row[index], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC) for index in (8, 10)
    )
    created = utc(datetime.fromisoformat(record["created_at"]))
    _require(
        created.replace(microsecond=0) <= submitted <= ended <= end,
        "scheduler times precede the durable intent or are inconsistent",
    )


def _processes_gone(record: dict) -> dict:
    _require(os.name == "posix" and hasattr(os, "geteuid"), "live process checks require Linux")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    host = socket.gethostname()
    observations = []
    for role, identity in (
        ("parent", record["parent_identity"]),
        ("submitter", record["ready"]["submitter"]),
    ):
        _require(
            identity["host"] == host
            and identity["boot_id"] == boot
            and identity["effective_uid"] == os.geteuid(),
            "process host/boot/user unavailable",
        )
        path = Path("/proc") / str(identity["pid"])
        observed = None
        if path.exists():
            raw = (path / "stat").read_text()
            observed = raw[raw.rfind(")") + 2 :].split()[19]
            _require(
                observed != identity["start_marker"], "original writer process is still present"
            )
        observations.append({"role": role, "identity": identity, "actual_start_marker": observed})
    return {"host": host, "boot_id": boot, "observations": observations}


def _validate_processes(value: dict, record: dict) -> None:
    _keys(value, {"host", "boot_id", "observations"}, "writer exit evidence")
    _require(
        type(value["observations"]) is list and len(value["observations"]) == 2,
        "both writer identities are required",
    )
    for row, (role, expected) in zip(
        value["observations"],
        (("parent", record["parent_identity"]), ("submitter", record["ready"]["submitter"])),
        strict=True,
    ):
        _keys(row, {"role", "identity", "actual_start_marker"}, "process observation")
        _require(
            row["role"] == role
            and row["identity"] == expected
            and value["host"] == expected["host"]
            and value["boot_id"] == expected["boot_id"]
            and (
                row["actual_start_marker"] is None
                or (
                    type(row["actual_start_marker"]) is str
                    and row["actual_start_marker"].isdigit()
                    and row["actual_start_marker"] != expected["start_marker"]
                )
            ),
            "original process exit identity is unproven",
        )


@dataclass(frozen=True)
class E0TerminalResolution:
    intent_id: str
    job_id: str
    campaign_id: str
    submission_request_sha256: str
    descriptor_sha256: str
    prior_terminal_receipt_sha256: str
    eligibility_receipt_sha256: str
    evidence_file_sha256: str
    original_code_sha: str
    verifier_code_sha: str
    phase: str = "E0"
    classification: str = "unstarted_scheduler_deadline"
    scientific_admission: bool = False
    _issuer: Any = field(default=None, repr=False, compare=False)

    def validate(self, record: dict, *, store: CampaignStore) -> E0TerminalResolution:
        _require(
            self._issuer is _ISSUER, "resolution was not issued by the reconciliation verifier"
        )
        current = resolve_e0_terminal_intent(store, self.intent_id)
        _require(
            current == self and current.submission_request_sha256 == record["request_sha256"],
            "terminal resolution changed or targets another record",
        )
        return current


def _terminal_payload(evidence_path: Path, evidence: dict, evidence_sha: str) -> dict:
    binding = evidence["binding"]
    return {
        "schema_version": 1,
        "kind": KIND,
        **binding,
        "evidence_path": str(evidence_path),
        "evidence_file_sha256": evidence_sha,
        "eligibility_receipt_sha256": content_digest(evidence["eligibility"]),
        "classification": "unstarted_scheduler_deadline",
        "phase": "E0",
        "scientific_admission": False,
        "resource_probe_success": False,
    }


def _audit_path(store: CampaignStore, intent: str) -> Path:
    digest(intent)
    return store.directory / "e0_reconciliations" / intent / "evidence.json"


def validate_e0_terminal_receipt(
    record: dict, claim: RunClaim, *, store: CampaignStore
) -> E0TerminalResolution:
    """Re-read authoritative SQLite plus original/sealed bytes; no fresh Slurm RPC is needed."""
    if (type(claim.terminal_receipt) is dict
            and claim.terminal_receipt.get("kind") == "path1_e0_resource_terminal_v1"):
        from agentmemeval.path1_v1.operations.resource_terminal import validate_resource_terminal

        return validate_resource_terminal(record, claim, store=store)
    snapshot, current_record, current_claim = _stored(store, claim.intent_id)
    _require(
        record == current_record and claim == current_claim and claim.status == "failed",
        "failed claim and record are not the current authoritative state",
    )
    terminal = claim.terminal_receipt
    _require(
        type(terminal) is dict and terminal.get("kind") == KIND,
        "not a reviewed E0 unstarted terminal claim",
    )
    path = _audit_path(store, claim.intent_id)
    _require(
        terminal.get("evidence_path") == str(path)
        and _sha(_path(path)) == terminal.get("evidence_file_sha256"),
        "terminal audit changed",
    )
    evidence = _json(path)
    _keys(
        evidence,
        {
            "schema_version",
            "kind",
            "binding",
            "descriptor_path",
            "baseline_manifest",
            "baseline_manifest_sha256",
            "original_record_sha256",
            "claim_before",
            "generation_before",
            "sealed_at",
            "original_inventory",
            "first_inventory",
            "second_inventory",
            "quiet_seconds",
            "scheduler_before",
            "scheduler_after",
            "processes_before",
            "processes_after",
            "verifier_runtime_identity",
            "eligibility",
        },
        "sealed reconciliation",
    )
    _require(
        evidence["kind"] == "path1_e0_unstarted_deadline_evidence_v1"
        and type(evidence["schema_version"]) is int
        and evidence["schema_version"] == 1,
        "unsupported reconciliation schema",
    )
    bound = _bound_inputs(
        store, _path(evidence["descriptor_path"]), terminal["descriptor_sha256"], record, claim
    )
    _require(terminal == _terminal_payload(path, evidence, _sha(path)), "terminal payload differs")
    binding = evidence["binding"]
    expected = _binding(store, bound, record, claim, terminal["descriptor_sha256"])
    _require(
        binding == expected and evidence["original_record_sha256"] == content_digest(record),
        "original record or source binding changed",
    )
    old_claim = evidence["claim_before"]
    _require(
        old_claim["status"] == "started"
        and old_claim["terminal_receipt"] is None
        and {
            key: value
            for key, value in old_claim.items()
            if key not in {"status", "terminal_receipt", "updated_generation"}
        }
        == {
            key: value
            for key, value in asdict(claim).items()
            if key not in {"status", "terminal_receipt", "updated_generation"}
        }
        and type(evidence["generation_before"]) is int
        and claim.updated_generation == evidence["generation_before"] + 1
        and snapshot.generation >= claim.updated_generation,
        "claim transition does not continue the original started reservation",
    )
    baseline = _baseline(_path(evidence["baseline_manifest"]), evidence["baseline_manifest_sha256"])
    now_inventory = _content_inventory(bound["run"], bound["invocation"])
    _require(
        evidence["original_inventory"] == baseline == _content(now_inventory)
        and evidence["first_inventory"] == evidence["second_inventory"] == now_inventory
        and type(evidence["quiet_seconds"]) in {int, float}
        and evidence["quiet_seconds"] >= 10,
        "original artifacts or quiescence evidence changed",
    )
    sealed = utc(datetime.fromisoformat(evidence["sealed_at"]))
    for key in ("scheduler_before", "scheduler_after"):
        _validate_scheduler(
            evidence[key], bound["request"], record, bound["config"], sealed_at=sealed
        )
    for key in ("processes_before", "processes_after"):
        _validate_processes(evidence[key], record)
    verifier = evidence["verifier_runtime_identity"]
    _historical_identity(bound["repository"], verifier)
    _require(evidence["eligibility"] == _eligibility(binding), "E0 retry eligibility changed")
    return E0TerminalResolution(
        intent_id=claim.intent_id,
        job_id=record["job_id"],
        campaign_id=record["campaign_id"],
        submission_request_sha256=record["request_sha256"],
        descriptor_sha256=terminal["descriptor_sha256"],
        prior_terminal_receipt_sha256=content_digest(terminal),
        eligibility_receipt_sha256=terminal["eligibility_receipt_sha256"],
        evidence_file_sha256=terminal["evidence_file_sha256"],
        original_code_sha=bound["contract"]["runtime_identity"]["core_sha"],
        verifier_code_sha=verifier["core_sha"],
        _issuer=_ISSUER,
    )


def resolve_e0_terminal_intent(store: CampaignStore, prior_intent_id: str) -> E0TerminalResolution:
    digest(prior_intent_id)
    _, record, claim = _stored(store, prior_intent_id)
    return validate_e0_terminal_receipt(record, claim, store=store)


def _binding(
    store: CampaignStore, bound: dict, record: dict, claim: RunClaim, descriptor_sha: str
) -> dict:
    return {
        "campaign_id": record["campaign_id"],
        "intent_id": claim.intent_id,
        "job_id": record["job_id"],
        "submission_request_sha256": record["request_sha256"],
        "descriptor_sha256": descriptor_sha,
        "store_campaign_id": store.campaign_id,
        "claim_token_sha256": hashlib.sha256(claim.token.encode()).hexdigest(),
        "original_source_identity_sha256": content_digest(bound["contract"]["runtime_identity"]),
        "submission_count": 1,
    }


def _eligibility(binding: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": "path1_e0_unstarted_technical_retry_eligibility",
        "prior_intent_id": binding["intent_id"],
        "job_id": binding["job_id"],
        "submission_request_sha256": binding["submission_request_sha256"],
        "scope": "complete_unit",
        "seed_changed": False,
        "phase": "E0",
        "job_kind": "resource_probe",
        "scientific_retry_eligible": False,
        "reason": "accepted_once_but_scheduler_never_started_or_allocated",
    }


def reconcile_unstarted_deadline(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    baseline_manifest: Path,
    baseline_manifest_sha256: str,
    reader: E0DeadlineReader | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> E0TerminalResolution:
    """Append evidence first, then one CAS failed-claim transition; never submit/cancel."""
    descriptor = _json(descriptor_path)
    _require(_sha(descriptor_path) == digest(descriptor_sha256), "original descriptor changed")
    intent = descriptor["intent_id"]
    reader = E0DeadlineReader(clock=clock) if reader is None else reader
    with store.lock():
        snapshot, record, claim = _stored(store, intent)
        if claim.status == "failed":
            resolution = validate_e0_terminal_receipt(record, claim, store=store)
            prior = _json(_audit_path(store, intent))
            _require(
                resolution.descriptor_sha256 == descriptor_sha256
                and prior["baseline_manifest"] == str(baseline_manifest)
                and prior["baseline_manifest_sha256"] == baseline_manifest_sha256,
                "repeated closure supplied a different trusted input",
            )
            return resolution
        _require(
            claim.status == "started" and claim.terminal_receipt is None,
            "only the original started submission claim can be closed",
        )
        if _audit_path(store, intent).parent.exists():
            raise FileExistsError(
                "immutable pending audit already exists; unresolved claim stays occupied and "
                "requires separately reviewed recovery, not overwrite or automatic retry"
            )
        bound = _bound_inputs(store, descriptor_path, descriptor_sha256, record, claim)
        config = {**bound["config"], "config_relative": descriptor["config_relative"]}
        verifier = _verifier_identity(bound["repository"], config)
        baseline = _baseline(baseline_manifest, baseline_manifest_sha256)
        before_scheduler = reader.read(bound["request"], record["job_id"], bound["config"])
        _require(
            type(before_scheduler) is DeadlineObservation,
            "typed live scheduler observation required",
        )
        before_processes = _processes_gone(record)
        first = _content_inventory(bound["run"], bound["invocation"])
        _require(_content(first) == baseline, "original artifacts differ from recovered baseline")
        started = monotonic()
        sleep(10.1)
        second = _content_inventory(bound["run"], bound["invocation"])
        elapsed = monotonic() - started
        _require(elapsed >= 10 and first == second, "original artifacts are not quiescent")
        after_processes = _processes_gone(record)
        after_scheduler = reader.read(bound["request"], record["job_id"], bound["config"])
        _require(
            type(after_scheduler) is DeadlineObservation,
            "typed live scheduler observation required",
        )
        sealed = utc(clock())
        for observation in (before_scheduler, after_scheduler):
            _validate_scheduler(
                observation.payload(), bound["request"], record, bound["config"], sealed_at=sealed
            )
        for processes in (before_processes, after_processes):
            _validate_processes(processes, record)
        current, current_record, current_claim = _stored(store, intent)
        _require(
            current == snapshot and current_record == record and current_claim == claim,
            "authoritative state changed before closure",
        )
        binding = _binding(store, bound, record, claim, descriptor_sha256)
        evidence = {
            "schema_version": 1,
            "kind": "path1_e0_unstarted_deadline_evidence_v1",
            "binding": binding,
            "descriptor_path": str(descriptor_path),
            "baseline_manifest": str(baseline_manifest),
            "baseline_manifest_sha256": baseline_manifest_sha256,
            "original_record_sha256": content_digest(record),
            "claim_before": asdict(claim),
            "generation_before": snapshot.generation,
            "sealed_at": sealed.isoformat(),
            "original_inventory": baseline,
            "first_inventory": first,
            "second_inventory": second,
            "quiet_seconds": elapsed,
            "scheduler_before": before_scheduler.payload(),
            "scheduler_after": after_scheduler.payload(),
            "processes_before": before_processes,
            "processes_after": after_processes,
            "verifier_runtime_identity": verifier,
            "eligibility": _eligibility(binding),
        }
        path = _audit_path(store, intent)
        path.parent.mkdir(parents=True, exist_ok=False)
        write_exclusive_json(path, evidence)
        _require(
            _json(path) == evidence
            and _content_inventory(bound["run"], bound["invocation"]) == second,
            "audit publication or original artifacts changed before commit",
        )
        terminal = _terminal_payload(path, evidence, _sha(path))
        store.finish_claim(
            expected_generation=snapshot.generation,
            run_id=claim.run_id,
            token=claim.token,
            status="failed",
            receipt=terminal,
        )
        return resolve_e0_terminal_intent(store, intent)


def check_unstarted_deadline(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    baseline_manifest: Path,
    baseline_manifest_sha256: str,
    reader: E0DeadlineReader | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Read-only preflight. This does not establish quiescence or release a claim."""
    descriptor = _json(descriptor_path)
    snapshot, record, claim = _stored(store, descriptor["intent_id"])
    _require(
        claim.status == "started" and claim.terminal_receipt is None,
        "preflight requires the original unresolved claim",
    )
    bound = _bound_inputs(store, descriptor_path, descriptor_sha256, record, claim)
    verifier = _verifier_identity(
        bound["repository"], {**bound["config"], "config_relative": descriptor["config_relative"]}
    )
    baseline = _baseline(baseline_manifest, baseline_manifest_sha256)
    _require(
        _content(_content_inventory(bound["run"], bound["invocation"])) == baseline,
        "original artifacts differ from recovered baseline",
    )
    observation = (E0DeadlineReader(clock=clock) if reader is None else reader).read(
        bound["request"], record["job_id"], bound["config"]
    )
    _require(type(observation) is DeadlineObservation, "typed live scheduler observation required")
    _validate_scheduler(
        observation.payload(), bound["request"], record, bound["config"], sealed_at=utc(clock())
    )
    processes = _processes_gone(record)
    _validate_processes(processes, record)
    _require(
        _stored(store, descriptor["intent_id"]) == (snapshot, record, claim),
        "authoritative state changed during preflight",
    )
    return {
        "kind": "path1_e0_unstarted_deadline_preflight",
        "schema_version": 1,
        "intent_id": claim.intent_id,
        "job_id": record["job_id"],
        "generation": snapshot.generation,
        "claim_status": claim.status,
        "descriptor_sha256": descriptor_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "original_code_sha": bound["contract"]["runtime_identity"]["core_sha"],
        "verifier_code_sha": verifier["core_sha"],
        "scheduler": observation.payload(),
        "processes": processes,
        "claim_released": False,
        "scientific_admission": False,
    }
