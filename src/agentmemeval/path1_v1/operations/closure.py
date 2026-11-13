"""Verify worker/release/quiet/archive/recovery evidence, never scientific success.

No default scheduler transport, service owner, or launch-record resolver exists.
Those explicitly injected capabilities are the trust roots. JSON flags, a queue
absence, and a deserialized process receipt cannot supply them. Local files and
copied archives are always read and hashed again, including on repeated calls.
This module neither submits jobs nor signals processes nor implements SSH/scp.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agentmemeval.path1_v1.operations.contracts import (
    _read_json,
    canonical_intent_id,
    content_digest,
    validate_artifact_receipt,
)
from agentmemeval.path1_v1.operations.dispatch import ClosureReceipt
from agentmemeval.path1_v1.operations.slurm import (
    READ_ENV,
    TERMINAL_STATES,
    CommandTransport,
    JobRecord,
    SubmissionRequest,
    digest,
    utc,
)
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.worker import validate_seed_inventory
from agentmemeval.path1_v1.storage.archive import MANIFEST_FIELDS, _safe_relative
from agentmemeval.path1_v1.storage.snapshot_archive import (
    _verify_archive_members,
    build_snapshot_archive,
    extract_snapshot_archive,
    verify_archive_checksum,
)


class ClosureError(ValueError):
    """Missing, conflicting, partial or changed evidence; closure remains pending."""


def _require(value: bool, reason: str) -> None:
    if not value:
        raise ClosureError(reason)


def _path(value: Path, *, directory: bool = False, exists: bool = True) -> Path:
    _require(isinstance(value, Path) and value.is_absolute(), "absolute evidence path required")
    _require(not any(p.is_symlink() for p in (value, *value.parents)), "symlink evidence path")
    resolved = value.resolve(strict=exists)
    if exists:
        _require(resolved.is_dir() if directory else resolved.is_file(), "evidence path type")
    return resolved


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path) -> dict:
    _path(path)
    value = _read_json(path)
    _require(type(value) is dict, "evidence JSON must be an object")
    return value


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fingerprint(value: os.stat_result) -> tuple:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def inventory_tree(root: Path) -> list[dict]:
    """Read all regular files with before/after identity checks; reject special entries."""
    root = _path(root, directory=True)
    rows, names = [], set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        _require(
            _safe_relative(relative) and relative.casefold() not in names,
            "unsafe or aliased artifact path",
        )
        names.add(relative.casefold())
        before = path.lstat()
        _require(not stat.S_ISLNK(before.st_mode), "artifact symlink")
        if stat.S_ISDIR(before.st_mode):
            continue
        _require(stat.S_ISREG(before.st_mode), "non-regular artifact")
        checksum = _sha(path)
        after = path.lstat()
        _require(_fingerprint(before) == _fingerprint(after), "artifact changed while hashing")
        rows.append(
            {
                "path": relative,
                "size": before.st_size,
                "sha256": checksum,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
            }
        )
    return rows


def _content(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("path", "size", "sha256")} for row in rows]


@dataclass(frozen=True)
class ClosurePaths:
    """Paths and immutable worker ID supplied by the trusted launch-record resolver.

    source_root is the job container (payload plus Slurm logs). control_directory
    is the worker's authoritative SQLite store, not an exported JSON summary.
    Archive and recovery directories must be distinct, outside the source tree.
    """

    worker_request_path: Path
    worker_request_sha256: str
    control_directory: Path
    source_root: Path
    bundle_directory: Path
    local_directory: Path


@dataclass(frozen=True)
class ClosureBinding:
    campaign_id: str
    intent_id: str
    request_sha256: str
    worker_request_sha256: str
    job_id: str
    submitted_at: str
    run_id: str
    service_instance_id: str
    source_sha256: str
    core_manifest_sha256: str
    operations_manifest_sha256: str
    config_sha256: str
    model_manifest_sha256: str
    protocol_sha256: str


@dataclass(frozen=True)
class SchedulerClosureEvidence:
    binding: ClosureBinding
    observed_at: datetime
    allocation_state: str
    ended_at: str
    steps: tuple[tuple[str, str], ...]
    commands: tuple[dict, ...]


class SchedulerClosureReader:
    """Actually read allocation AND steps from an explicitly trusted site transport."""

    def __init__(
        self,
        transport: CommandTransport,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.transport, self.clock = transport, clock

    def read(self, request: SubmissionRequest, binding: ClosureBinding) -> SchedulerClosureEvidence:
        commands = []
        formats = "JobIDRaw,User,Account,JobName,State,Comment,Submit,WorkDir,ExitCode,End"
        calls = (
            (
                "sacct",
                "--noheader",
                "--parsable2",
                f"--clusters={request.spec.cluster}",
                f"--jobs={binding.job_id}",
                f"--format={formats}",
            ),
            (
                "squeue",
                "--noheader",
                f"--clusters={request.spec.cluster}",
                f"--jobs={binding.job_id}",
                "--format=%i|%T",
            ),
            (
                "squeue",
                "--noheader",
                "--steps",
                f"--clusters={request.spec.cluster}",
                f"--jobs={binding.job_id}",
                "--format=%i|%T",
            ),
        )
        for argv in calls:
            result = self.transport.run(
                argv, environment=READ_ENV, inherit_environment=False, timeout_seconds=15
            )
            _require(
                type(result.returncode) is int
                and result.returncode == 0
                and not result.stderr.strip(),
                "scheduler read failed or warned",
            )
            _require(type(result.stdout) is str, "scheduler output is not text")
            commands.append(
                {
                    "argv": list(argv),
                    "stdout": result.stdout,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
            )
        _require(
            not commands[1]["stdout"].strip() and not commands[2]["stdout"].strip(),
            "allocation or job step remains queued/active",
        )
        allocation, steps, seen = None, [], set()
        for line in commands[0]["stdout"].splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            _require(len(fields) == 10, "truncated or unsupported accounting row")
            job, user, account, name, state, comment, submitted, directory, code, end = fields
            _require(job not in seen, "duplicate accounting identity")
            seen.add(job)
            state = re.sub(r"^CANCELLED by [0-9]+$", "CANCELLED", state)
            _require(state in TERMINAL_STATES, "non-terminal or unknown accounting state")
            _require(re.fullmatch(r"[0-9]+:[0-9]+", code) is not None, "missing exit code")
            try:
                parsed_end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
            except ValueError as exc:
                raise ClosureError("missing accounting end time") from exc
            _require(parsed_end.strftime("%Y-%m-%dT%H:%M:%S") == end, "ambiguous end time")
            if job == binding.job_id:
                allocation = JobRecord(
                    job, user, account, name, state, comment, submitted, directory, "sacct"
                )
                _require(
                    allocation.matches(request) and submitted == binding.submitted_at,
                    "accounting allocation differs from owned submission",
                )
                ended_at = end
            else:
                _require(
                    re.fullmatch(re.escape(binding.job_id) + r"\.(?:batch|extern|[0-9]+)", job)
                    is not None,
                    "unowned or unsupported accounting step",
                )
                _require(
                    user in {"", request.spec.user} and account in {"", request.spec.account},
                    "step owner differs",
                )
                steps.append((job, state))
        _require(allocation is not None, "terminal allocation accounting is absent")
        _require(
            any(job == binding.job_id + ".batch" for job, _ in steps),
            "batch step terminal accounting is absent",
        )
        return SchedulerClosureEvidence(
            binding,
            utc(self.clock()),
            allocation.state,
            ended_at,
            tuple(sorted(steps)),
            tuple(commands),
        )


@dataclass(frozen=True)
class ServiceReleaseEvidence:
    binding: ClosureBinding
    cleanup: dict


class ServiceReleaseVerifier(Protocol):
    def verify(self, binding: ClosureBinding) -> ServiceReleaseEvidence:
        """Use a trusted producer capability; return concrete identity/exit facts."""
        ...


class OwnedServiceReleaseVerifier:
    """Use the still-held live owner's exact, internally generated cleanup receipt.

    A copied dataclass/JSON cannot reconstitute this capability. After a process
    restart a separate trusted persisted-evidence verifier is needed; absent one,
    closure stays pending. Group exit is combined with scheduler step evidence.
    """

    def __init__(self, owner: Any, receipt: Any) -> None:
        self.owner, self.receipt = owner, receipt

    def verify(self, binding: ClosureBinding) -> ServiceReleaseEvidence:
        _require(
            self.owner.verify_cleanup(self.receipt) is True,
            "service owner cannot verify this exact cleanup receipt",
        )
        return _cleanup_evidence(binding, asdict(self.receipt))


def _cleanup_evidence(binding: ClosureBinding, value: dict) -> ServiceReleaseEvidence:
    """Validate concrete facts; callers must separately establish producer provenance."""
    expected = {
        "campaign_id": binding.campaign_id,
        "intent_id": binding.intent_id,
        "job_id": binding.job_id,
        "source_sha256": binding.source_sha256,
        "role": "model_service",
        "service_instance_id": binding.service_instance_id,
    }
    _require(value.get("binding") == expected, "service cleanup binding mismatch")
    identity = value.get("identity", {})
    for key in ("host", "boot_id", "start_marker", "slurm_job_id", "step_id"):
        _require(
            type(identity.get(key)) is str and bool(identity[key]),
            "service process identity is incomplete",
        )
    _require(identity["slurm_job_id"] == binding.job_id, "service process job mismatch")
    _require(
        re.fullmatch(r"(?:batch|extern|[0-9]+)", identity["step_id"]) is not None,
        "service step mismatch",
    )
    for key in ("effective_uid", "pid", "session_id", "pgid"):
        _require(
            type(identity.get(key)) is int
            and identity[key] >= (0 if key == "effective_uid" else 1),
            "service process kernel identity is incomplete",
        )
    _require(bool(identity.get("cgroup")), "service cgroup identity missing")
    _require(
        value.get("group_cleanup_verified") is True
        and value.get("leader_reaped") is True
        and value.get("remaining_non_zombie_pids") in ([], ())
        and value.get("errors") in ([], ())
        and type(value.get("returncode")) is int,
        "service cleanup is incomplete",
    )
    return ServiceReleaseEvidence(binding, value)


def _bound_json(path: Path, checksum: str) -> dict:
    digest(checksum)
    _path(path)
    _require(_sha(path) == checksum, "bound input hash mismatch")
    value = _json(path)
    _require(_sha(path) == checksum, "bound input changed during read")
    return value


def _worker_evidence(
    request: SubmissionRequest, scheduler_job: Mapping[str, Any], paths: ClosurePaths
) -> tuple[ClosureBinding, dict, str]:
    """Read immutable inputs + SQLite terminal; never call startup-only validation."""
    source = _path(paths.source_root, directory=True)
    worker = _json(paths.worker_request_path)
    _require(
        worker.get("kind") == "path1_worker_request"
        and type(worker.get("schema_version")) is int
        and worker["schema_version"] == 1,
        "unsupported worker request",
    )
    worker_sha = content_digest({k: v for k, v in worker.items() if k != "request_sha256"})
    _require(
        worker_sha == worker.get("request_sha256") == paths.worker_request_sha256,
        "worker request is not the launch-bound content",
    )
    contract = _bound_json(Path(worker["contract_path"]), worker["contract_sha256"])
    _require(
        canonical_intent_id(contract) == worker.get("intent_id") == request.intent_id
        and contract["campaign_id"] == request.campaign_id,
        "worker contract binding mismatch",
    )
    seeds = _bound_json(Path(worker["seed_inventory_path"]), worker["seed_inventory_sha256"])
    validate_seed_inventory(seeds, contract)
    expected_root = Path(worker["output_root"]).joinpath(
        *contract["job"]["output_relpath"].split("/")
    )
    _require(_path(expected_root, directory=True) == source, "worker output mapping differs")
    job = JobRecord(**dict(scheduler_job))
    _require(
        job.matches(request) and job.state in TERMINAL_STATES and job.source == "sacct",
        "dispatch has no owned terminal accounting record",
    )
    terminal_path = source / "payload/worker_terminal.json"
    terminal = _json(terminal_path)
    claim = _bound_json(Path(worker["claim_path"]), worker["claim_sha256"])
    control = _path(paths.control_directory, directory=True)
    _require(
        _path(Path(claim["control_directory"]), directory=True) == control,
        "worker control store differs from launch record",
    )
    store = CampaignStore(control, request.campaign_id)
    _path(store.database_path)
    snapshot = store.read_registry()
    matches = [item for item in snapshot.claims if item.run_id == claim["run_id"]]
    _require(len(matches) == 1, "worker claim is missing or ambiguous")
    owned = matches[0]
    _require(
        owned.intent_id == request.intent_id
        and owned.token == claim["token"]
        and _path(Path(owned.run_directory), directory=True) == source / "payload"
        and owned.status == "completed"
        and content_digest(owned.terminal_receipt) == content_digest(terminal),
        "worker terminal is not the completed authoritative claim",
    )
    _require(
        terminal.get("status") == "completed"
        and terminal.get("kind") == "path1_worker_terminal"
        and terminal.get("primary_analysis_admitted") is False,
        "missing/incomplete worker terminal cannot grant full closure",
    )
    started = _json(source / "payload/worker_started.json")
    runtime = _json(source / "payload/runtime_admission.json")
    identity = contract["runtime_identity"]
    binding = ClosureBinding(
        request.campaign_id,
        request.intent_id,
        request.request_sha256,
        worker_sha,
        job.job_id,
        job.submitted_at,
        claim["run_id"],
        started["service_instance_id"],
        content_digest(identity),
        identity["core_manifest_sha256"],
        identity["operations_manifest_sha256"],
        identity["config"]["sha256"],
        identity["model"]["manifest_sha256"],
        contract["protocol_sha256"],
    )
    for key in (
        "campaign_id",
        "intent_id",
        "worker_request_sha256",
        "run_id",
        "service_instance_id",
        "core_manifest_sha256",
        "operations_manifest_sha256",
        "config_sha256",
        "model_manifest_sha256",
        "protocol_sha256",
    ):
        _require(terminal.get(key) == getattr(binding, key), "worker terminal identity mismatch")
    _require(
        terminal.get("slurm_job_id") == binding.job_id
        and terminal.get("phase") == contract["phase"],
        "worker terminal job/phase mismatch",
    )
    _require(
        started.get("intent_id") == binding.intent_id
        and started.get("request_sha256") == worker_sha
        and started.get("slurm_job_id") == binding.job_id,
        "worker start binding mismatch",
    )
    _require(runtime.get("request_sha256") == worker_sha, "runtime worker request mismatch")
    for key in (
        "intent_id",
        "slurm_job_id",
        "service_instance_id",
        "core_manifest_sha256",
        "operations_manifest_sha256",
        "config_sha256",
        "model_manifest_sha256",
    ):
        expected = binding.job_id if key == "slurm_job_id" else getattr(binding, key)
        _require(runtime.get(key) == expected, "runtime receipt identity mismatch")
    seal = terminal.get("produced_artifacts")
    _require(type(seal) is list and bool(seal), "worker has no produced-artifact seal")
    for row in seal:
        _require(
            type(row) is dict and set(row) == {"path", "size", "sha256"},
            "malformed produced-artifact row",
        )
        _require(
            type(row["path"]) is str
            and _safe_relative(row["path"])
            and type(row["size"]) is int
            and row["size"] >= 0,
            "invalid produced-artifact path/size",
        )
        digest(row["sha256"])
    payload_inventory = _content(inventory_tree(source / "payload"))
    publication = [
        row
        for row in payload_inventory
        if re.fullmatch(
            r"_publication_staging/worker_terminal\.json\.[0-9a-f]{32}\.pending", row["path"]
        )
    ]
    terminal_hash = _sha(terminal_path)
    _require(
        len(publication) == 1
        and publication[0]["sha256"] == terminal_hash
        and publication[0]["size"] == terminal_path.stat().st_size
        and (source / "payload" / publication[0]["path"]).samefile(terminal_path),
        "terminal publication copy missing/ambiguous/changed",
    )
    actual = [
        row
        for row in payload_inventory
        if row["path"] not in {"worker_terminal.json", publication[0]["path"]}
    ]
    _require(
        content_digest(seal) == content_digest(actual), "produced artifacts lost/changed/added"
    )
    _required_artifacts(source / "payload", terminal, contract, seeds)
    return binding, terminal, terminal_hash


def _raw_rows(path: Path) -> list[dict]:
    def pairs(values):
        result = {}
        for key, value in values:
            _require(key not in result, "duplicate raw JSON key")
            result[key] = value
        return result

    def invalid(_value):
        raise ClosureError("non-finite raw JSON")

    rows = [
        json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)
        for line in _path(path).read_text(encoding="utf-8").splitlines()
    ]
    _require(all(type(row) is dict for row in rows), "raw row is not an object")
    return rows


def _required_artifacts(payload: Path, terminal: dict, contract: dict, seeds: dict) -> None:
    """Independently require raw episode evidence, not only a self-declared file seal."""
    job, source = contract["job"], contract["source_contract"]
    if job["kind"] == "formation":
        immutable = validate_artifact_receipt(
            _json(payload / "immutable_formation_receipt.json"), contract
        )
        snapshot = immutable["snapshot"]
        path = _path(payload / snapshot["path"])
        _require(
            path.stat().st_size == snapshot["size"] and _sha(path) == snapshot["sha256"],
            "immutable formation snapshot bytes differ",
        )
    technical = contract["phase"] == "E1"
    formation_count = source["technical"]["formation_episodes"] if technical else 6
    formation_hands = source["technical"]["formation_episode_hands"] if technical else 50
    evaluation_hands = source["technical"]["evaluation_episode_hands"] if technical else 50
    units = terminal.get("completed_units")
    _require(
        type(units) is list and all(type(row) is dict for row in units), "completed units absent"
    )
    artifact = next(row for row in seeds["artifacts"] if row["artifact_id"] == job["artifact_id"])
    pair_indices = {row["pair_id"]: row["episode_index"] for row in artifact["pairs"]}
    sampler = _json(payload / "sampler_seed_manifest.json")
    _require(
        sampler.get("algorithm") == seeds["sampler_algorithm"]
        and type(sampler.get("streams")) is dict,
        "sampler raw manifest missing",
    )
    expected = set(job["pair_ids"])
    if job["kind"] in {"formation", "technical_closed_loop"}:
        expected.add("formation")
    _require(
        len(units) == len(expected) and {row.get("unit") for row in units} == expected,
        "worker unit topology differs from contract",
    )
    for unit in units:
        root = payload / unit["unit"]
        saved = _json(root / "terminal_receipt.json")
        _require(
            content_digest(saved) == content_digest(unit["receipt"])
            and saved.get("status") == "complete",
            "unit terminal mismatch/incomplete",
        )
        _json(root / "path1_manifest.json")
        _json(root / "seed_manifest.json")
        if unit["unit"] == "formation":
            _json(root / "artifact_snapshot.json")
            episodes = [
                (i, "formation", "formation", formation_hands) for i in range(formation_count)
            ]
        else:
            _require(
                type(unit.get("episode_index")) is int
                and unit["episode_index"] == pair_indices[unit["unit"]],
                "pair episode index missing",
            )
            episodes = [
                (unit["episode_index"], arm, "evaluation", evaluation_hands)
                for arm in ("retained", "removed")
            ]
        _require(
            saved.get("completed_episodes") == len(episodes)
            and saved.get("settled_hands") == sum(row[3] for row in episodes),
            "unit raw completeness count differs",
        )
        _require(
            len(_raw_rows(root / "settled_hands.jsonl")) == sum(row[3] for row in episodes),
            "settled raw hand rows missing",
        )
        for index, arm, stage, hands in episodes:
            episode = root / f"{stage}_{index:04d}_{arm}"
            receipt = _json(episode / "episode_receipt.json")
            _require(
                receipt.get("status") == "complete"
                and receipt.get("hands") == hands
                and receipt.get("episode_index") == index
                and receipt.get("arm") == arm,
                "episode completion receipt mismatch",
            )
            _require(
                receipt.get("campaign_id") == contract["campaign_id"]
                and receipt.get("phase") == contract["phase"]
                and receipt.get("artifact_id") == job["artifact_id"]
                and receipt.get("stage") == stage,
                "episode identity mismatch",
            )
            for filename in ("initial_snapshot.json", "final_snapshot.json"):
                _json(episode / filename)
            for filename in ("resolved_config.yaml", "events.jsonl", "provider_exchanges.jsonl"):
                _require(
                    _path(episode / filename).stat().st_size > 0, "required raw artifact empty"
                )
            _require(len(_raw_rows(episode / "hand_summaries.jsonl")) == hands, "raw hands missing")


def _manifest_content(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, "archive manifest schema")
        rows = list(reader)
    return [
        {"path": row["relative_path"], "size": int(row["size_bytes"]), "sha256": row["sha256"]}
        for row in rows
    ]


def _supporting_inputs(paths: ClosurePaths) -> list[tuple[str, Path, str]]:
    """Keep immutable launch inputs and actual probe bytes with the recovered archive."""
    worker = _json(paths.worker_request_path)
    rows = [("worker_request.json", paths.worker_request_path, _sha(paths.worker_request_path))]
    for name in ("contract", "seed_inventory", "claim", "runtime_gate"):
        if worker[name + "_path"] is not None:
            path, checksum = Path(worker[name + "_path"]), worker[name + "_sha256"]
            _bound_json(path, checksum)
            rows.append((name + ".json", path, checksum))
    if worker["artifact_root"] is not None:
        contract = _bound_json(Path(worker["contract_path"]), worker["contract_sha256"])
        snapshot = contract["job"]["formation_receipt"]["snapshot"]
        path = _path(Path(worker["artifact_root"]) / snapshot["path"])
        _require(
            path.stat().st_size == snapshot["size"] and _sha(path) == snapshot["sha256"],
            "input formation snapshot lost/changed",
        )
        rows.append(("input_formation_snapshot.json", path, snapshot["sha256"]))
    runtime = _json(paths.source_root / "payload/runtime_admission.json")
    probes = runtime.get("probes")
    purposes = {
        "sampler_seed_execution_63bit",
        "tokenizer_total_context",
        "structured_schema",
        "loaded_model_and_source_identity",
    }
    _require(
        type(probes) is list
        and len(probes) == len(purposes)
        and {row.get("purpose") for row in probes} == purposes,
        "runtime raw probes absent",
    )
    for probe in probes:
        path = _path(Path(probe["path"]))
        digest(probe["sha256"])
        _require(
            type(probe["size"]) is int
            and path.stat().st_size == probe["size"]
            and _sha(path) == probe["sha256"],
            "runtime raw probe lost/changed",
        )
        rows.append(("probe_" + probe["purpose"] + ".json", path, probe["sha256"]))
    return rows


def _copy_exclusive(original: Path, target: Path, checksum: str) -> None:
    _path(original)
    _require(_sha(original) == checksum, "copy source changed")
    with original.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    _require(_sha(original) == checksum == _sha(target), "copy bytes changed")


class FileClosureVerifier:
    """Concrete dispatch ClosureVerifier using current files and explicit capabilities."""

    def __init__(
        self,
        resolve_paths: Callable[[SubmissionRequest], ClosurePaths],
        *,
        scheduler: SchedulerClosureReader | None = None,
        service: ServiceReleaseVerifier | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.resolve_paths, self.scheduler, self.service = resolve_paths, scheduler, service
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.last_failure: str | None = None

    def __call__(
        self, request: SubmissionRequest, scheduler_job: Mapping[str, Any]
    ) -> ClosureReceipt | None:
        self.last_failure = None
        try:
            return self.verify(request, scheduler_job)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            self.last_failure = f"{type(exc).__name__}: {exc}"
            return None

    def _release(self, request: SubmissionRequest, binding: ClosureBinding) -> dict:
        _require(
            self.scheduler is not None and self.service is not None,
            "trusted scheduler/service evidence producers are not installed",
        )
        scheduler = self.scheduler.read(request, binding)
        _require(
            type(scheduler) is SchedulerClosureEvidence and scheduler.binding == binding,
            "scheduler evidence is not bound",
        )
        age = (utc(self.clock()) - utc(scheduler.observed_at)).total_seconds()
        _require(0 <= age <= 30, "scheduler release evidence stale")
        service = self.service.verify(binding)
        _require(
            type(service) is ServiceReleaseEvidence and service.binding == binding,
            "service evidence absent or unbound",
        )
        _cleanup_evidence(binding, service.cleanup)
        _require(
            any(
                job == binding.job_id + "." + service.cleanup["identity"]["step_id"]
                for job, _ in scheduler.steps
            ),
            "owned service step lacks terminal accounting",
        )
        return {
            "binding": asdict(binding),
            "scheduler": {**asdict(scheduler), "observed_at": scheduler.observed_at.isoformat()},
            "service": asdict(service),
            "primary_analysis_admitted": False,
        }

    def verify(
        self, request: SubmissionRequest, scheduler_job: Mapping[str, Any]
    ) -> ClosureReceipt:
        _require(
            self.scheduler is not None and self.service is not None,
            "trusted scheduler/service evidence producers are not installed",
        )
        paths = self.resolve_paths(request)
        _require(type(paths) is ClosurePaths, "trusted resolver must return ClosurePaths")
        binding, terminal, worker_sha = _worker_evidence(request, scheduler_job, paths)
        source = _path(paths.source_root, directory=True)
        bundle = _path(paths.bundle_directory, directory=True, exists=False)
        local = _path(paths.local_directory, directory=True, exists=False)
        roots = (source, bundle, local)
        _require(
            all(a != b and not a.is_relative_to(b) for a in roots for b in roots if a != b)
            and len(set(roots)) == 3,
            "closure output directories overlap",
        )
        first_release = self._release(request, binding)
        first = inventory_tree(source)
        started = self.monotonic()
        self.sleep(10.0)
        elapsed = self.monotonic() - started
        _require(elapsed >= 10.0, "quiet interval shorter than ten seconds")
        second = inventory_tree(source)
        _require(first == second, "source changed during quiet interval")
        content = _content(second)
        inputs = _supporting_inputs(paths)
        inputs_content = [
            {"path": name, "size": path.stat().st_size, "sha256": checksum}
            for name, path, checksum in sorted(inputs)
        ]
        archive, manifest, checksum = (
            bundle / name for name in ("snapshot.tar.gz", "snapshot.files.tsv", "snapshot.sha256")
        )
        if bundle.exists() or local.exists():
            _require(
                (bundle / "closure_receipt.json").is_file(),
                "partial closure outputs preserved; automatic reuse forbidden",
            )
        else:
            bundle.mkdir(parents=True, exist_ok=False)
            _write(bundle / "release_receipt.json", first_release)
            _write(
                bundle / "quiet_receipt.json",
                {
                    "binding": asdict(binding),
                    "elapsed_monotonic_seconds": elapsed,
                    "first": first,
                    "second": second,
                    "primary_analysis_admitted": False,
                },
            )
            (bundle / "evidence_inputs").mkdir()
            for name, path, sha in inputs:
                _copy_exclusive(path, bundle / "evidence_inputs" / name, sha)
            build_snapshot_archive(
                source, archive, manifest, checksum, bundle / "archive_receipt.json"
            )
            _require(
                _manifest_content(manifest) == content, "archive manifest differs from quiet tree"
            )
            _require(inventory_tree(source) == second, "source changed during archive creation")
            local.mkdir(parents=True, exist_ok=False)
            (local / "evidence_inputs").mkdir()
            for name, _original_path, sha in inputs:
                _copy_exclusive(
                    bundle / "evidence_inputs" / name, local / "evidence_inputs" / name, sha
                )
            for original in (archive, manifest, checksum):
                _copy_exclusive(original, local / original.name, _sha(original))
            extract_snapshot_archive(
                local / archive.name,
                local / checksum.name,
                local / manifest.name,
                local / "extracted",
                local / "extraction_receipt.json",
            )
            recovered = local / "extracted" / source.name
            _require(
                _content(inventory_tree(recovered)) == content, "recovered file inventory differs"
            )
            final_release = self._release(request, binding)
            _require(inventory_tree(source) == second, "source changed during recovery")
            again, _, again_sha = _worker_evidence(request, scheduler_job, paths)
            _require(
                again == binding and again_sha == worker_sha,
                "worker binding changed during closure",
            )
            _write(
                bundle / "recovery_receipt.json",
                {
                    "binding": asdict(binding),
                    "worker_terminal_sha256": worker_sha,
                    "source_inventory": content,
                    "archive_sha256": _sha(archive),
                    "manifest_sha256": _sha(manifest),
                    "checksum_sha256": _sha(checksum),
                    "supporting_inputs": inputs_content,
                    "quiet_receipt_sha256": _sha(bundle / "quiet_receipt.json"),
                    "archive_receipt_sha256": _sha(bundle / "archive_receipt.json"),
                    "extraction_receipt_sha256": _sha(local / "extraction_receipt.json"),
                    "final_release": final_release,
                    "primary_analysis_admitted": False,
                    "scientific_status": "not_assessed",
                },
            )
            closure = ClosureReceipt(
                request.intent_id,
                request.request_sha256,
                worker_sha,
                _sha(bundle / "release_receipt.json"),
                _sha(bundle / "recovery_receipt.json"),
            )
            _write(bundle / "closure_receipt.json", asdict(closure))
            for name in (
                "release_receipt.json",
                "quiet_receipt.json",
                "archive_receipt.json",
                "recovery_receipt.json",
                "closure_receipt.json",
            ):
                _copy_exclusive(bundle / name, local / name, _sha(bundle / name))
        # Existing receipt flags are never proof: reread all archive and extracted bytes.
        recovery = _json(bundle / "recovery_receipt.json")
        _require(
            recovery.get("binding") == asdict(binding)
            and recovery.get("source_inventory") == content
            and recovery.get("worker_terminal_sha256") == worker_sha,
            "stored recovery binding differs",
        )
        _require(
            recovery.get("supporting_inputs")
            == inputs_content
            == _content(inventory_tree(bundle / "evidence_inputs"))
            == _content(inventory_tree(local / "evidence_inputs")),
            "supporting evidence recovery changed",
        )
        for name in ("quiet_receipt", "archive_receipt", "extraction_receipt"):
            root = local if name == "extraction_receipt" else bundle
            _require(
                _sha(_path(root / (name + ".json"))) == recovery[name + "_sha256"],
                "closure supporting receipt changed",
            )
        quiet = _json(bundle / "quiet_receipt.json")
        _require(
            quiet.get("binding") == asdict(binding)
            and quiet["first"] == quiet["second"]
            and _content(quiet["second"]) == content
            and type(quiet["elapsed_monotonic_seconds"]) in (float, int)
            and quiet["elapsed_monotonic_seconds"] >= 10,
            "stored quiet evidence differs",
        )
        for path, key in (
            (archive, "archive_sha256"),
            (manifest, "manifest_sha256"),
            (checksum, "checksum_sha256"),
        ):
            _path(path)
            _require(
                _sha(path) == recovery.get(key) == _sha(_path(local / path.name)),
                "archive/recovery transport bytes changed",
            )
        _require(
            verify_archive_checksum(archive, checksum)["verified"] is True,
            "archive checksum failed",
        )
        _require(
            _verify_archive_members(archive, manifest, root_name=source.name)["verified"] is True
            and _manifest_content(manifest) == content,
            "archive members differ",
        )
        _require(
            _content(inventory_tree(local / "extracted" / source.name)) == content,
            "recovered artifacts lost/changed/added",
        )
        closure = ClosureReceipt(**_json(bundle / "closure_receipt.json"))
        closure.validate(request)
        _require(
            closure.worker_terminal_sha256 == worker_sha
            and closure.release_receipt_sha256 == _sha(bundle / "release_receipt.json")
            and closure.recovery_receipt_sha256 == _sha(bundle / "recovery_receipt.json"),
            "closure receipt bytes changed",
        )
        _require(
            _json(bundle / "release_receipt.json").get("binding") == asdict(binding),
            "release receipt binding differs",
        )
        for name in (
            "release_receipt.json",
            "quiet_receipt.json",
            "archive_receipt.json",
            "recovery_receipt.json",
            "closure_receipt.json",
        ):
            _require(
                _sha(_path(local / name)) == _sha(bundle / name),
                "recovered control receipt changed",
            )
        self._release(request, binding)
        again, _, again_sha = _worker_evidence(request, scheduler_job, paths)
        _require(
            again == binding and again_sha == worker_sha, "worker changed before final closure"
        )
        _require(inventory_tree(source) == second, "source changed before final closure")
        return closure
