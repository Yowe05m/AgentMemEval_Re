"""Shared E0 terminal sealing mechanics; policies are code, never JSON capabilities.

Preserves the original accepted intent, verifies immutable evidence and archives,
and changes only its authoritative claim under the CampaignStore lock.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.slurm import utc
from agentmemeval.path1_v1.operations.state import CampaignStore, RunClaim
from agentmemeval.path1_v1.operations.submission_process import write_exclusive_json

_MODULE_PATH = "src/agentmemeval/path1_v1/operations/terminal_common.py"
_assert = er._require


@dataclass(frozen=True)
class TerminalPolicy:
    kind: str
    evidence_kind: str
    readonly_kind: str
    classification: str
    observation_key: str
    module_path: str
    issuer: Any
    resolution_type: Any
    audit_path: Any
    bound_inputs: Any
    verifier_identity: Any
    facts: Any
    scheduler: Any
    archive: Any
    verify_archives: Any
    claim_status: Any


def verify_implementation_source(repository: Path, identity: dict) -> None:
    own = Path(__file__).resolve()
    _assert(own == repository / _MODULE_PATH, "terminal common loaded from another checkout")
    _assert(
        any(
            row["path"] == _MODULE_PATH and row["sha256"] == er._sha(own)
            for row in identity["operations_files"]
        ),
        "terminal common implementation is not published",
    )


def _terminal(policy: TerminalPolicy, path: Path, evidence: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": policy.kind,
        "classification": policy.classification,
        "binding": evidence["binding"],
        "evidence_path": str(path),
        "evidence_file_sha256": er._sha(path),
        "claim_status": evidence["claim_status"],
        "scheduler_state": evidence["scheduler"]["state"],
        policy.observation_key: evidence["facts"][policy.observation_key],
        "retry_eligible": False,
        "runtime_admission": False,
        "scientific_admission": False,
        "primary_analysis_admitted": False,
        "scientific_samples": 0,
    }


def validate_terminal(
    policy: TerminalPolicy, record: dict, claim: RunClaim, *, store: CampaignStore
):
    snapshot, current_record, current_claim = er._stored(store, claim.intent_id)
    _assert(
        record == current_record
        and claim == current_claim
        and claim.status in {"completed", "failed"},
        "current authoritative terminal claim required",
    )
    path = policy.audit_path(store, claim.intent_id)
    terminal = claim.terminal_receipt
    _assert(
        type(terminal) is dict
        and terminal.get("kind") == policy.kind
        and terminal.get("evidence_path") == str(path)
        and terminal.get("evidence_file_sha256") == er._sha(er._path(path)),
        "terminal evidence changed",
    )
    evidence = er._json(path)
    er._keys(
        evidence,
        {
            "schema_version",
            "kind",
            "binding",
            "descriptor_path",
            "original_record_sha256",
            "claim_before",
            "generation_before",
            "sealed_at",
            "first_inventory",
            "second_inventory",
            "quiet_seconds",
            "scheduler_before",
            "scheduler_after",
            "processes_before",
            "processes_after",
            "verifier_runtime_identity",
            "facts",
            "scheduler",
            "archives",
            "claim_status",
            "first_checked_at",
        },
        "resource evidence",
    )
    _assert(
        evidence["schema_version"] == 1
        and type(evidence["schema_version"]) is int
        and evidence["kind"] == policy.evidence_kind,
        "evidence schema differs",
    )
    bound = policy.bound_inputs(
        store,
        er._path(evidence["descriptor_path"]),
        evidence["binding"]["descriptor_sha256"],
        record,
        claim,
        require_empty_output=False,
    )
    _assert(
        evidence["binding"]
        == er._binding(store, bound, record, claim, evidence["binding"]["descriptor_sha256"])
        and evidence["original_record_sha256"] == content_digest(record)
        and terminal == _terminal(policy, path, evidence),
        "terminal source/submission binding differs",
    )
    old = evidence["claim_before"]
    _assert(
        old["status"] == "started"
        and old["terminal_receipt"] is None
        and {
            k: v
            for k, v in old.items()
            if k not in {"status", "terminal_receipt", "updated_generation"}
        }
        == {
            k: v
            for k, v in asdict(claim).items()
            if k not in {"status", "terminal_receipt", "updated_generation"}
        }
        and type(evidence["generation_before"]) is int
        and claim.updated_generation == evidence["generation_before"] + 1
        and snapshot.generation >= claim.updated_generation,
        "terminal claim did not continue original reservation",
    )
    _assert(
        evidence["first_inventory"]
        == evidence["second_inventory"]
        == er._content_inventory(bound["run"], bound["invocation"])
        and type(evidence["quiet_seconds"]) in {int, float}
        and evidence["quiet_seconds"] >= 10,
        "original resource artifacts or quiescence evidence changed",
    )
    facts = policy.facts(bound, record)
    _assert(evidence["facts"] == facts, "resource evidence interpretation changed")
    sealed = utc(datetime.fromisoformat(evidence["sealed_at"]))
    first_checked = utc(datetime.fromisoformat(evidence["first_checked_at"]))
    _assert(
        first_checked <= sealed
        and (
            utc(datetime.fromisoformat(evidence["scheduler_after"]["observed_at"])) - first_checked
        ).total_seconds()
        >= 10,
        "scheduler/quiet sequence is inconsistent",
    )
    for key, checked_at in (("scheduler_before", first_checked), ("scheduler_after", sealed)):
        _assert(
            policy.scheduler(evidence[key], bound, record, facts, sealed_at=checked_at)
            == evidence["scheduler"],
            "scheduler observations differ",
        )
    for key in ("processes_before", "processes_after"):
        er._validate_processes(evidence[key], record)
    policy.verify_archives(path, bound, evidence["archives"])
    verifier = evidence["verifier_runtime_identity"]
    er._historical_identity(bound["repository"], verifier)
    _assert(
        any(row["path"] == policy.module_path for row in verifier["operations_files"]),
        "sealed verifier does not include resource terminal implementation",
    )
    _assert(
        claim.status
        == evidence["claim_status"]
        == policy.claim_status(facts, evidence["scheduler"]),
        "technical outcome/claim status differs",
    )
    return policy.resolution_type(
        intent_id=claim.intent_id,
        job_id=record["job_id"],
        campaign_id=record["campaign_id"],
        submission_request_sha256=record["request_sha256"],
        descriptor_sha256=evidence["binding"]["descriptor_sha256"],
        evidence_file_sha256=er._sha(path),
        terminal_receipt_sha256=content_digest(terminal),
        original_code_sha=bound["contract"]["runtime_identity"]["core_sha"],
        verifier_code_sha=verifier["core_sha"],
        scheduler_state=evidence["scheduler"]["state"],
        claim_status=claim.status,
        **{policy.observation_key: facts[policy.observation_key]},
        _issuer=policy.issuer,
    )


def check_terminal(
    policy: TerminalPolicy,
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
) -> dict:
    """Read current core facts without publishing evidence or releasing occupancy."""
    descriptor_path = er._path(descriptor_path)
    _assert(er._sha(descriptor_path) == er.digest(descriptor_sha256), "descriptor changed")
    descriptor = er._json(descriptor_path)
    reader = er.E0DeadlineReader() if reader is None else reader
    with store.lock():
        snapshot, record, claim = er._stored(store, descriptor["intent_id"])
        _assert(
            claim.status == "started" and claim.terminal_receipt is None,
            "read-only precheck requires the original started claim",
        )
        bound = policy.bound_inputs(
            store, descriptor_path, descriptor_sha256, record, claim, require_empty_output=False
        )
        verifier = policy.verifier_identity(
            bound["repository"],
            {**bound["config"], "config_relative": descriptor["config_relative"]},
        )
        facts = policy.facts(bound, record)
        observed = reader.read(bound["request"], record["job_id"], bound["config"])
        _assert(type(observed) is er.DeadlineObservation, "typed live observation required")
        checked_at = utc(clock())
        scheduler = policy.scheduler(observed.payload(), bound, record, facts, sealed_at=checked_at)
        processes = er._processes_gone(record)
        er._validate_processes(processes, record)
        inventory = er._content_inventory(bound["run"], bound["invocation"])
        inventory_sha256 = content_digest(inventory)
        checked_at = utc(clock())
        _assert(
            scheduler
            == policy.scheduler(observed.payload(), bound, record, facts, sealed_at=checked_at),
            "scheduler changed during read-only evidence verification",
        )
        return {
            "kind": policy.readonly_kind,
            "generation": snapshot.generation,
            "intent_id": claim.intent_id,
            "job_id": record["job_id"],
            "checked_at": checked_at.isoformat(),
            "scheduler": scheduler,
            "facts": facts,
            "inventory_sha256": inventory_sha256,
            "verifier_runtime_identity": verifier,
            "technical_claim_closed": False,
            "retry_eligible": False,
            "runtime_admission": False,
            "scientific_admission": False,
        }


def reconcile_terminal(
    policy: TerminalPolicy,
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Create verified archives and seal the exact original claim; never submit or signal."""
    descriptor = er._json(descriptor_path)
    _assert(er._sha(descriptor_path) == er.digest(descriptor_sha256), "descriptor changed")
    reader = er.E0DeadlineReader(clock=clock) if reader is None else reader
    with store.lock():
        snapshot, record, claim = er._stored(store, descriptor["intent_id"])
        if claim.status in {"completed", "failed"}:
            result = validate_terminal(policy, record, claim, store=store)
            _assert(result.descriptor_sha256 == descriptor_sha256, "repeat descriptor differs")
            return result
        _assert(
            claim.status == "started" and claim.terminal_receipt is None, "started claim required"
        )
        path = policy.audit_path(store, claim.intent_id)
        if path.parent.exists():
            raise FileExistsError(
                "pending resource audit exists; preserve occupancy and review recovery"
            )
        bound = policy.bound_inputs(
            store, descriptor_path, descriptor_sha256, record, claim, require_empty_output=False
        )
        verifier = policy.verifier_identity(
            bound["repository"],
            {**bound["config"], "config_relative": descriptor["config_relative"]},
        )
        facts = policy.facts(bound, record)
        before = reader.read(bound["request"], record["job_id"], bound["config"])
        _assert(type(before) is er.DeadlineObservation, "typed live observation required")
        first_checked = utc(clock())
        policy.scheduler(before.payload(), bound, record, facts, sealed_at=first_checked)
        before_process = er._processes_gone(record)
        er._validate_processes(before_process, record)
        first = er._content_inventory(bound["run"], bound["invocation"])
        start = monotonic()
        sleep(10.1)
        second = er._content_inventory(bound["run"], bound["invocation"])
        elapsed = monotonic() - start
        _assert(first == second and elapsed >= 10, "resource source is not quiescent")
        path.parent.mkdir(parents=True, exist_ok=False)
        archives = policy.archive(path, bound)
        policy.verify_archives(path, bound, archives)
        after_process = er._processes_gone(record)
        after = reader.read(bound["request"], record["job_id"], bound["config"])
        _assert(type(after) is er.DeadlineObservation, "typed live observation required")
        sealed = utc(clock())
        scheduler = policy.scheduler(
            before.payload(), bound, record, facts, sealed_at=first_checked
        )
        _assert(
            scheduler == policy.scheduler(after.payload(), bound, record, facts, sealed_at=sealed),
            "terminal scheduler evidence changed",
        )
        for value in (before_process, after_process):
            er._validate_processes(value, record)
        _assert(
            second == er._content_inventory(bound["run"], bound["invocation"])
            and facts == policy.facts(bound, record)
            and er._stored(store, claim.intent_id) == (snapshot, record, claim),
            "original evidence/state changed before sealing",
        )
        sealed = utc(clock())
        _assert(
            scheduler == policy.scheduler(after.payload(), bound, record, facts, sealed_at=sealed),
            "scheduler evidence expired during final source verification",
        )
        evidence = {
            "schema_version": 1,
            "kind": policy.evidence_kind,
            "binding": er._binding(store, bound, record, claim, descriptor_sha256),
            "descriptor_path": str(descriptor_path),
            "original_record_sha256": content_digest(record),
            "claim_before": asdict(claim),
            "generation_before": snapshot.generation,
            "sealed_at": sealed.isoformat(),
            "first_checked_at": first_checked.isoformat(),
            "first_inventory": first,
            "second_inventory": second,
            "quiet_seconds": elapsed,
            "scheduler_before": before.payload(),
            "scheduler_after": after.payload(),
            "processes_before": before_process,
            "processes_after": after_process,
            "verifier_runtime_identity": verifier,
            "facts": facts,
            "scheduler": scheduler,
            "archives": archives,
            "claim_status": policy.claim_status(facts, scheduler),
        }
        write_exclusive_json(path, evidence)
        _assert(er._json(path) == evidence, "published evidence changed")
        terminal = _terminal(policy, path, evidence)
        _assert(
            scheduler
            == policy.scheduler(after.payload(), bound, record, facts, sealed_at=utc(clock())),
            "scheduler evidence expired while publishing the seal; occupancy retained",
        )
        store.finish_claim(
            expected_generation=snapshot.generation,
            run_id=claim.run_id,
            token=claim.token,
            status=evidence["claim_status"],
            receipt=terminal,
        )
        _, current_record, current_claim = er._stored(store, claim.intent_id)
        return validate_terminal(policy, current_record, current_claim, store=store)
