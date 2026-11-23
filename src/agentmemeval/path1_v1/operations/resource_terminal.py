"""Technical closure for an executed, model-free E0 resource probe.

This verifies historical producer files and fresh scheduler facts. It cannot
restore process-control capabilities, authorize retries or admit research.
Archive restoration is verified on the verifier host; delivery to another host
requires its own transfer verification and is never implied here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.closure import inventory_tree
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.slurm import TERMINAL_STATES, job_comment, job_name, utc
from agentmemeval.path1_v1.operations.state import CampaignStore, RunClaim
from agentmemeval.path1_v1.storage.archive import verify_file_manifest
from agentmemeval.path1_v1.storage.snapshot_archive import (
    _verify_archive_members,
    build_snapshot_archive,
    extract_snapshot_archive,
    verify_archive_checksum,
)

KIND = "path1_e0_resource_terminal_v1"
_ISSUER = object()
_MODULE_PATH = "src/agentmemeval/path1_v1/operations/resource_terminal.py"


def _assert(value: bool, message: str) -> None:
    er._require(value, "resource terminal: " + message)


def _audit_path(store: CampaignStore, intent: str) -> Path:
    er.digest(intent)
    return store.directory / "resource_terminals" / intent / "evidence.json"


def _verifier_identity(repository: Path, config: dict) -> dict:
    identity = er._verifier_identity(repository, config)
    own = Path(__file__).resolve()
    _assert(own == repository / _MODULE_PATH, "resource verifier loaded from another checkout")
    _assert(
        any(
            row["path"] == _MODULE_PATH and row["sha256"] == er._sha(own)
            for row in identity["operations_files"]
        ),
        "resource verifier is not published",
    )
    from agentmemeval.path1_v1.operations.terminal_common import verify_implementation_source

    verify_implementation_source(repository, identity)
    return identity


def _timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    _assert(parsed.strftime("%Y-%m-%dT%H:%M:%S") == value, "noncanonical scheduler time")
    return parsed


def _state(value: str) -> str:
    # Slurm may render CANCELLED by <uid>; a trailing '+' means truncation.
    match = re.fullmatch(r"([A-Z_]+)(?: by ([0-9]+))?", value)
    _assert(match is not None, "truncated/unknown scheduler state")
    state = match.group(1)
    _assert(
        state in TERMINAL_STATES and (match.group(2) is None or state == "CANCELLED"),
        "allocation or step is not terminal",
    )
    return state


def validate_resource_scheduler(
    value: dict, bound: dict, record: dict, facts: dict, *, sealed_at: datetime
) -> dict:
    """Read actual command output, including every returned accounting step."""
    request, config = bound["request"], bound["config"]
    er._keys(value, {"observed_at", "completed_at", "effective_uid", "commands"}, "scheduler")
    began, ended = (
        utc(datetime.fromisoformat(value[key])) for key in ("observed_at", "completed_at")
    )
    _assert(
        began <= ended <= sealed_at and (sealed_at - began).total_seconds() <= 30,
        "scheduler observation is not fresh",
    )
    _assert(
        type(value["effective_uid"]) is int
        and value["effective_uid"] == record["effective_user"]["uid"],
        "scheduler user differs",
    )
    rows = value["commands"]
    _assert(type(rows) is list and len(rows) == 3, "missing scheduler command")
    for row, argv in zip(rows, er._read_argv(request, record["job_id"], config), strict=True):
        er._keys(row, {"argv", "pin", "returncode", "stdout", "stderr"}, "scheduler command")
        _assert(
            row["argv"] == list(argv)
            and row["pin"] == config["commands"][Path(argv[0]).name]
            and type(row["returncode"]) is int
            and row["returncode"] == 0
            and type(row["stdout"]) is str
            and type(row["stderr"]) is str
            and not row["stderr"].strip(),
            "unclean or filtered scheduler query",
        )
    job_id = record["job_id"]
    for line in rows[0]["stdout"].splitlines():
        columns = line.split("|")
        _assert(
            len(columns) == 8 and columns[1:3] == [request.spec.user, request.spec.account],
            "queue format/scope differs",
        )
        _assert(
            re.match(re.escape(job_id) + r"(?:$|[._+])", columns[0]) is None,
            "target still appears in live queue",
        )
    for line in rows[1]["stdout"].splitlines():
        _assert(
            re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?\.[A-Za-z0-9_-]+", line) is not None,
            "step list has unsupported identity",
        )
        _assert(re.match(re.escape(job_id) + r"(?:$|[._+])", line) is None, "target has live steps")
    accounting = [line.split("|") for line in rows[2]["stdout"].splitlines() if line]
    _assert(accounting and all(len(row) == 15 for row in accounting), "accounting format differs")
    allocation = [row for row in accounting if row[1] == job_id]
    _assert(len(allocation) == 1, "unique allocation required")
    allocation = allocation[0]
    _assert(
        allocation[:5]
        == [
            request.spec.cluster,
            job_id,
            request.spec.user,
            request.spec.account,
            job_name(request.campaign_id, request.intent_id),
        ]
        and allocation[13:]
        == [job_comment(request.campaign_id, request.intent_id), request.spec.work_dir],
        "allocation identity differs",
    )
    state = _state(allocation[5])
    submitted, started, finished = (_timestamp(allocation[index]) for index in (8, 9, 10))
    _assert(
        utc(datetime.fromisoformat(record["created_at"])).replace(microsecond=0)
        <= submitted
        <= started
        <= finished
        <= ended,
        "allocation times inconsistent with original submission",
    )
    _assert(
        re.fullmatch(r"[0-9]+", allocation[7]) is not None
        and re.fullmatch(r"[0-9]+:[0-9]+", allocation[6]) is not None,
        "allocation elapsed/exit code invalid",
    )
    _assert(
        allocation[11] == facts["hostname"]
        and allocation[12]
        and any(
            re.fullmatch(r"gres/gpu(?:\:[^=,]+)?=1", entry) for entry in allocation[12].split(",")
        ),
        "allocation node/GPU resource record differs",
    )
    steps, step_details = set(), []
    for row in accounting:
        if row is allocation:
            continue
        match = re.fullmatch(re.escape(job_id) + r"\.([A-Za-z0-9_-]+)", row[1])
        _assert(
            match is not None and match.group(1) not in steps, "unknown/duplicate accounting step"
        )
        name = match.group(1)
        steps.add(name)
        _assert(
            row[0] in {"", request.spec.cluster}
            and row[2] in {"", request.spec.user}
            and row[3] in {"", request.spec.account}
            and row[11] in {"", allocation[11]}
            and row[13] in {"", allocation[13]}
            and row[14] in {"", allocation[14]},
            "step belongs to another scope/node",
        )
        step_state = _state(row[5])
        _assert(
            re.fullmatch(r"[0-9]+:[0-9]+", row[6]) is not None
            and re.fullmatch(r"[0-9]+", row[7]) is not None,
            "invalid step exit/elapsed",
        )
        if row[9] == "None":
            _assert(
                row[7] == "0" and name not in facts["required_steps"],
                "required execution step never started",
            )
        else:
            _assert(
                started <= _timestamp(row[9]) <= _timestamp(row[10]) <= finished,
                "step execution interval escapes allocation",
            )
        step_details.append({"name": name, "state": step_state, "exit_code": row[6]})
    _assert(set(facts["required_steps"]) <= steps, "required historical execution step missing")
    return {
        "job_id": job_id,
        "state": state,
        "exit_code": allocation[6],
        "started_at": started.isoformat(),
        "ended_at": finished.isoformat(),
        "hostname": allocation[11],
        "allocated_tres": allocation[12],
        "steps": sorted(step_details, key=lambda row: row["name"]),
    }


def _resource_facts(bound: dict, record: dict) -> dict:
    probe, contract = bound["probe"], bound["contract"]
    root = er._path(bound["request"].spec.work_dir) / "e0_probe"
    launch, cleanup = (
        er._json(root / "resource_launch.json"),
        er._json(root / "resource_cleanup.json"),
    )
    binding = launch["binding"]
    expected_binding = {
        "campaign_id": contract["campaign_id"],
        "intent_id": record["intent_id"],
        "job_id": record["job_id"],
        "source_sha256": content_digest(contract["runtime_identity"]),
        "role": "resource_probe",
        "service_instance_id": binding["service_instance_id"],
    }
    _assert(
        re.fullmatch(r"[0-9a-f]{32}", binding["service_instance_id"]) is not None
        and binding == cleanup["binding"] == expected_binding,
        "resource launch/cleanup binding differs",
    )
    _assert(
        launch["request_sha256"] == probe["request_sha256"]
        and launch["runtime_admission"] is False
        and launch["argv"]
        == [
            probe["launch"]["python_executable"],
            "-B",
            "-s",
            "-m",
            "agentmemeval.path1_v1.operations.resource_probe",
            "--collect-request",
            str(bound["run"] / "probe_request.json"),
        ]
        and (launch["command_seconds"], launch["cleanup_seconds"], launch["term_grace_seconds"])
        == (90, 30, 15),
        "resource launcher contract differs",
    )
    er._keys(
        cleanup,
        {
            "binding",
            "identity",
            "returncode",
            "term_sent",
            "kill_sent",
            "leader_reaped",
            "group_cleanup_verified",
            "remaining_non_zombie_pids",
            "errors",
        },
        "resource cleanup",
    )
    _assert(
        cleanup["leader_reaped"] is True
        and cleanup["group_cleanup_verified"] is True
        and cleanup["remaining_non_zombie_pids"] == []
        and cleanup["errors"] == []
        and type(cleanup["returncode"]) is int
        and type(cleanup["term_sent"]) is bool
        and type(cleanup["kill_sent"]) is bool,
        "resource cleanup did not close its owned group",
    )
    child, parent = cleanup["identity"], launch["host"]
    er._keys(
        child,
        {
            "host",
            "boot_id",
            "effective_uid",
            "pid",
            "start_marker",
            "session_id",
            "pgid",
            "cgroup",
            "slurm_job_id",
            "step_id",
        },
        "resource process identity",
    )
    _assert(
        type(child["pid"]) is int
        and child["pid"] > 0
        and type(child["pgid"]) is int
        and type(child["session_id"]) is int
        and child["pid"] == child["pgid"] == child["session_id"]
        and type(child["start_marker"]) is str
        and child["start_marker"].isdigit()
        and int(child["start_marker"]) > 0
        and type(child["host"]) is str
        and bool(child["host"])
        and type(child["boot_id"]) is str
        and bool(child["boot_id"])
        and child["host"] == parent["hostname"]
        and child["boot_id"] == parent["boot_id"]
        and type(child["effective_uid"]) is int
        and type(parent["euid"]) is int
        and child["effective_uid"] == parent["euid"] == record["effective_user"]["uid"]
        and child["slurm_job_id"] == parent["job_id"] == record["job_id"]
        and type(parent["pid"]) is int
        and parent["pid"] > 0
        and parent["pid"] != child["pid"],
        "resource parent/child kernel identity mismatch",
    )
    _assert(
        type(child["cgroup"]) is list
        and child["cgroup"]
        and all(type(value) is str for value in child["cgroup"])
        and child["cgroup"] == parent["cgroup"].splitlines()
        and any("job_" + record["job_id"] in value.split("/") for value in child["cgroup"]),
        "resource kernel cgroup does not bind this job",
    )
    required = {"batch"}
    if child["step_id"] is not None:
        _assert(
            type(child["step_id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]+", child["step_id"]),
            "invalid resource step identity",
        )
        required.add(child["step_id"])
    for line in child["cgroup"]:
        required.update(
            part.removeprefix("step_") for part in line.split("/") if part.startswith("step_")
        )
    for name in ("resource_stdout.log", "resource_stderr.log"):
        _assert(er._path(root / name).is_file(), "resource command log missing")
    facts = {
        "output_directory": str(root),
        "hostname": child["host"],
        "boot_id": child["boot_id"],
        "resource_pid": child["pid"],
        "launcher_pid": parent["pid"],
        "cleanup_returncode": cleanup["returncode"],
        "required_steps": sorted(required),
        "launch_sha256": er._sha(root / "resource_launch.json"),
        "cleanup_sha256": er._sha(root / "resource_cleanup.json"),
    }
    facts.update(_observations(root, probe, child, parent))
    return facts


def _observations(root: Path, probe: dict, child: dict, parent: dict) -> dict:
    """Missing/malformed observational content is incomplete; identity conflicts refuse closure."""
    observed, issues = {}, []
    for name in ("host.json", "gpu.json", "resource_receipt.json"):
        path = root / "observations" / name
        if not path.exists():
            issues.append("missing:" + name)
            continue
        try:
            observed[name] = er._json(path)
        except (ValueError, UnicodeError):
            issues.append("invalid_json:" + name)
    host = observed.get("host.json")
    if host is not None:
        _assert(
            all(
                host.get(key) == expected
                for key, expected in {
                    "hostname": child["host"],
                    "boot_id": child["boot_id"],
                    "euid": child["effective_uid"],
                    "pid": child["pid"],
                    "job_id": child["slurm_job_id"],
                    "cgroup": parent["cgroup"],
                }.items()
            ),
            "collector host identity conflicts with owned child",
        )
    receipt = observed.get("resource_receipt.json")
    if receipt is not None:
        _assert(
            receipt.get("request_sha256") == probe["request_sha256"]
            and receipt.get("intent_id") == probe["intent_id"]
            and receipt.get("runtime_admission") is False,
            "collector receipt identity conflicts",
        )
        if (
            receipt.get("status") != "observations_complete"
            or receipt.get("host") != host
            or receipt.get("gpu") != observed.get("gpu.json")
        ):
            issues.append("incomplete_or_inconsistent_collector_receipt")
    gpu = observed.get("gpu.json")
    try:
        _assert(
            gpu is not None
            and gpu["returncode"] == 0
            and type(gpu["returncode"]) is int
            and gpu["argv"]
            == [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
            and not gpu["stderr"].strip(),
            "GPU observation incomplete",
        )
        parsed = []
        for line in gpu["stdout"].splitlines():
            values = [part.strip() for part in line.split(",")]
            _assert(
                len(values) == 4 and values[0].startswith("GPU-") and values[3].isdigit(),
                "GPU raw output malformed",
            )
            parsed.append(
                dict(zip(("uuid", "name", "driver_version", "memory_mib"), values, strict=True))
            )
        _assert(
            parsed
            and parsed == gpu["gpus"]
            and len({row["uuid"] for row in parsed}) == len(parsed),
            "GPU records do not reproduce raw output",
        )
    except (er.E0ReconciliationError, KeyError, TypeError, AttributeError):
        issues.append("invalid_or_missing_gpu_observation")
    fs = root / "observations/filesystem_probe.bin"
    if not fs.is_file() or er._path(fs).read_bytes() != probe["request_sha256"].encode("ascii"):
        issues.append("filesystem_roundtrip_missing_or_mismatched")
    if receipt is not None:
        filesystem = receipt.get("filesystem", {})
        if filesystem.get("path") != str(root / "observations") or any(
            type(filesystem.get(key)) is not int or filesystem[key] < 0
            for key in ("total", "used", "free")
        ):
            issues.append("filesystem_capacity_observation_incomplete")
    return {
        "resource_observation_complete": not issues,
        "observation_issues": sorted(set(issues)),
        "allocation_gpu_mapping_verified": False,
        "preinventory_unknown_artifact_completeness_proven": False,
    }


def _archive_sources(path: Path, bound: dict) -> list[dict]:
    result = []
    for label, source in (("run", bound["run"]), ("invocation", bound["invocation"])):
        bundle = path.parent / label
        bundle.mkdir()
        archive, manifest, checksum = (
            bundle / name for name in ("snapshot.tar.gz", "files.tsv", "snapshot.sha256")
        )
        build_snapshot_archive(source, archive, manifest, checksum, bundle / "archive_receipt.json")
        extract_snapshot_archive(
            archive, checksum, manifest, bundle / "restored", bundle / "restore_receipt.json"
        )
        result.append(
            {
                "label": label,
                "source": str(source),
                "bundle": str(bundle),
                "files": [
                    {"name": name, "sha256": er._sha(bundle / name)}
                    for name in (
                        "snapshot.tar.gz",
                        "files.tsv",
                        "snapshot.sha256",
                        "archive_receipt.json",
                        "restore_receipt.json",
                    )
                ],
                "content": er._content(inventory_tree(source)),
            }
        )
    return result


def _verify_archives(path: Path, bound: dict, records: list[dict]) -> None:
    _assert(type(records) is list and len(records) == 2, "two original evidence archives required")
    for row, (label, source) in zip(
        records, (("run", bound["run"]), ("invocation", bound["invocation"])), strict=True
    ):
        er._keys(row, {"label", "source", "bundle", "files", "content"}, "archive binding")
        bundle = er._path(path.parent / label)
        _assert(
            row["label"] == label and row["source"] == str(source) and row["bundle"] == str(bundle),
            "archive source/bundle differs",
        )
        names = [
            "snapshot.tar.gz",
            "files.tsv",
            "snapshot.sha256",
            "archive_receipt.json",
            "restore_receipt.json",
        ]
        _assert(
            row["files"]
            == [{"name": name, "sha256": er._sha(er._path(bundle / name))} for name in names],
            "immutable archive file changed",
        )
        archive, manifest, checksum = (bundle / name for name in names[:3])
        _assert(
            verify_archive_checksum(archive, checksum)["verified"]
            and _verify_archive_members(archive, manifest, root_name=source.name)["verified"]
            and verify_file_manifest(source, manifest)["verified"],
            "archive checksum/member/source mismatch",
        )
        restored = er._path(bundle / "restored" / source.name)
        _assert(
            verify_file_manifest(restored, manifest)["verified"]
            and row["content"]
            == er._content(inventory_tree(source))
            == er._content(inventory_tree(restored)),
            "restored archive differs from original source",
        )


@dataclass(frozen=True)
class ResourceTerminalResolution:
    intent_id: str
    job_id: str
    campaign_id: str
    submission_request_sha256: str
    descriptor_sha256: str
    evidence_file_sha256: str
    terminal_receipt_sha256: str
    original_code_sha: str
    verifier_code_sha: str
    scheduler_state: str
    claim_status: str
    resource_observation_complete: bool
    classification: str = "resource_terminal_observed"
    phase: str = "E0"
    retry_eligible: bool = False
    runtime_admission: bool = False
    scientific_admission: bool = False
    primary_analysis_admitted: bool = False
    scientific_samples: int = 0
    _issuer: Any = field(default=None, repr=False, compare=False)

    def validate(self, record: dict, *, store: CampaignStore) -> ResourceTerminalResolution:
        _assert(self._issuer is _ISSUER, "resolution was not issued by verifier")
        current = er.resolve_e0_terminal_intent(store, self.intent_id)
        _assert(
            type(current) is ResourceTerminalResolution
            and current == self
            and record["request_sha256"] == self.submission_request_sha256,
            "resolution changed",
        )
        return current


def _claim_status(facts: dict, scheduler: dict) -> str:
    success = (
        facts["resource_observation_complete"]
        and facts["cleanup_returncode"] == 0
        and scheduler["state"] == "COMPLETED"
        and scheduler["exit_code"] == "0:0"
        and all(
            row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in scheduler["steps"]
        )
    )
    return "completed" if success else "failed"


def _policy():
    from agentmemeval.path1_v1.operations.terminal_common import TerminalPolicy

    return TerminalPolicy(
        kind=KIND,
        evidence_kind="path1_e0_resource_terminal_evidence_v1",
        readonly_kind="path1_e0_resource_readonly_check_v1",
        classification="resource_terminal_observed",
        observation_key="resource_observation_complete",
        module_path=_MODULE_PATH,
        issuer=_ISSUER,
        resolution_type=ResourceTerminalResolution,
        audit_path=_audit_path,
        bound_inputs=er._bound_inputs,
        verifier_identity=_verifier_identity,
        facts=_resource_facts,
        scheduler=validate_resource_scheduler,
        archive=_archive_sources,
        verify_archives=_verify_archives,
        claim_status=_claim_status,
    )


def _terminal(path: Path, evidence: dict) -> dict:
    from agentmemeval.path1_v1.operations.terminal_common import _terminal as terminal

    return terminal(_policy(), path, evidence)


def validate_resource_terminal(
    record: dict, claim: RunClaim, *, store: CampaignStore
) -> ResourceTerminalResolution:
    from agentmemeval.path1_v1.operations.terminal_common import validate_terminal

    return validate_terminal(_policy(), record, claim, store=store)


def check_resource_terminal(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
) -> dict:
    from agentmemeval.path1_v1.operations.terminal_common import check_terminal

    return check_terminal(
        _policy(),
        store,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        reader=reader,
        clock=clock,
    )


def reconcile_resource_terminal(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> ResourceTerminalResolution:
    from agentmemeval.path1_v1.operations.terminal_common import reconcile_terminal

    return reconcile_terminal(
        _policy(),
        store,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        reader=reader,
        clock=clock,
        monotonic=monotonic,
        sleep=sleep,
    )
