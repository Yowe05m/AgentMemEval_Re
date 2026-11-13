"""One bounded, fail-closed campaign step; scheduling the next tick is external.

SQLite is the sole authority. The sequence is lock -> reread -> reconcile ->
gates/capacity -> persist intent/claim -> persist submit_unknown -> submit.
There is deliberately no automatic retry from submit_unknown, even if Slurm
and the executor both report nothing for days. A terminal scheduler observation
does not finish a worker claim, reclaim resources, recover files or accept data.

Real execution needs a verified gate callback, account visibility/effective-limit
probe, and durable remote IntentExecutor, none of which is fabricated here.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.slurm import (
    TERMINAL_STATES,
    AccountSnapshot,
    IntentReceipt,
    JobRecord,
    SlurmAdapter,
    SlurmError,
    SubmissionRequest,
    SubmitSpec,
    digest,
    job_comment,
    job_name,
    utc,
)
from agentmemeval.path1_v1.operations.state import (
    CampaignStore,
    LockUnavailable,
    RegistrySnapshot,
    StoreIntegrityError,
)

NAMESPACE = "dispatch_v1"
STAGES = (1, 2, 4, 6)
ATTEMPTED = frozenset({"submit_unknown", "submitted", "running"})
FINISHED = frozenset({"terminal_observed", "terminal_closed", "submission_rejected"})


class DispatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateDecision:
    granted: bool
    evidence_sha256: str | None = None
    reason: str = "execution admission not established"

    def validate(self) -> None:
        if type(self.granted) is not bool:
            raise DispatchError("gate must return an explicit boolean decision")
        if self.granted:
            digest(self.evidence_sha256)


@dataclass(frozen=True)
class DispatchRequest:
    contract: dict[str, Any]
    spec: SubmitSpec
    run_id: str
    local_run_directory: Path
    validation_context: Mapping[str, Any] | None = None

    def validate(self, campaign_id: str) -> tuple[str, dict]:
        contract = validate_campaign_contract(self.contract, **dict(self.validation_context or {}))
        if contract["campaign_id"] != campaign_id:
            raise DispatchError("contract belongs to a different campaign")
        if (
            not isinstance(self.local_run_directory, Path)
            or not self.local_run_directory.is_absolute()
        ):
            raise DispatchError("local claim directory must be an absolute Path")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise DispatchError("run ID must be explicit")
        relative = PurePosixPath(contract["job"]["output_relpath"]).parts
        if PurePosixPath(self.spec.work_dir).parts[-len(relative) :] != relative:
            raise DispatchError("remote work directory must match the contract output path")
        return canonical_intent_id(contract), contract


class AdmissionVerifier(Protocol):
    def __call__(self, request: DispatchRequest) -> GateDecision:
        """Verify runtime/protocol/Git/recovery gates. Contract validity is not admission."""
        ...


@dataclass(frozen=True)
class StageApproval:
    from_cap: int
    to_cap: int
    evidence_sha256: str


@dataclass(frozen=True)
class ClosureReceipt:
    """Three separately verified receipts; closure never grants scientific acceptance."""

    intent_id: str
    request_sha256: str
    worker_terminal_sha256: str
    release_receipt_sha256: str
    recovery_receipt_sha256: str

    def validate(self, request: SubmissionRequest) -> None:
        for value in asdict(self).values():
            digest(value)
        if self.intent_id != request.intent_id or self.request_sha256 != request.request_sha256:
            raise DispatchError("terminal closure is bound to a different submission")


class ClosureVerifier(Protocol):
    def __call__(
        self, request: SubmissionRequest, scheduler_job: Mapping[str, Any]
    ) -> ClosureReceipt | None:
        """Verify actual worker terminal, exact resource release, and recovered file hashes.

        Return None until all three receipts are verified; scheduler exit alone
        cannot synthesize any of them. Partial recovery remains scientifically
        ineligible. The real OP04/OP06 verifier is deliberately not provided here.
        """
        ...


@dataclass(frozen=True)
class CapacityDecision:
    stage_cap: int
    effective_limit: int
    account_occupancy: int
    campaign_occupancy: int
    unresolved_reservations: int
    available: int
    reserve_slots: int = 2


@dataclass(frozen=True)
class StepResult:
    action: str
    reason: str
    generation: int | None
    intent_id: str | None = None
    job_id: str | None = None
    capacity: CapacityDecision | None = None


def _request(campaign_id: str, intent_id: str, record: dict) -> SubmissionRequest:
    return SubmissionRequest(campaign_id, intent_id, SubmitSpec.from_payload(record["spec"]))


def _ledger(snapshot: RegistrySnapshot) -> dict:
    if NAMESPACE not in snapshot.registry:
        return {"schema_version": 1, "stage_cap": 1, "stage_evidence": [], "intents": {}}
    ledger = copy.deepcopy(snapshot.registry[NAMESPACE])
    if (
        type(ledger) is not dict
        or set(ledger) != {"schema_version", "stage_cap", "stage_evidence", "intents"}
        or type(ledger["schema_version"]) is not int
        or ledger["schema_version"] != 1
        or type(ledger["stage_cap"]) is not int
        or ledger["stage_cap"] not in STAGES
        or type(ledger["stage_evidence"]) is not list
        or type(ledger["intents"]) is not dict
    ):
        raise StoreIntegrityError("unsupported dispatch ledger")
    stage = 1
    try:
        for evidence in ledger["stage_evidence"]:
            if (
                set(evidence) != {"from_cap", "to_cap", "evidence_sha256"}
                or stage == 6
                or type(evidence["from_cap"]) is not int
                or type(evidence["to_cap"]) is not int
                or evidence["from_cap"] != stage
                or evidence["to_cap"] != STAGES[STAGES.index(stage) + 1]
            ):
                raise ValueError("broken stage chain")
            digest(evidence["evidence_sha256"])
            stage = evidence["to_cap"]
        if stage != ledger["stage_cap"]:
            raise ValueError("stage has no chain of expansion receipts")
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreIntegrityError("invalid persisted concurrency admission") from exc
    for intent_id, record in ledger["intents"].items():
        try:
            if set(record) != {
                "contract",
                "spec",
                "request_sha256",
                "run_id",
                "run_directory",
                "created_at",
                "status",
                "job_id",
                "owned_job",
                "executor_receipt",
                "admission_sha256",
                "conflict",
                "closure_receipt",
            }:
                raise ValueError("unknown record fields")
            if canonical_intent_id(record["contract"]) != intent_id:
                raise ValueError("intent changed")
            if record["contract"]["campaign_id"] != snapshot.campaign_id:
                raise ValueError("campaign changed")
            request = _request(snapshot.campaign_id, intent_id, record)
            if request.request_sha256 != record["request_sha256"]:
                raise ValueError("submission binding changed")
            utc(datetime.fromisoformat(record["created_at"]))
            if record["status"] not in ATTEMPTED | FINISHED | {"intent", "ownership_conflict"}:
                raise ValueError("unknown dispatch state")
            digest(record["admission_sha256"])
            if record["executor_receipt"] is not None:
                IntentReceipt(**record["executor_receipt"]).validate(request)
            job = None
            if record["owned_job"] is not None:
                job = JobRecord(**record["owned_job"])
                if not job.matches(request) or job.job_id != record["job_id"]:
                    raise ValueError("owned scheduler identity is inconsistent")
            if record["status"] in {"terminal_observed", "terminal_closed"} and (
                job is None or job.state not in TERMINAL_STATES
            ):
                raise ValueError("terminal status requires an explicitly terminal owned job")
            if record["status"] == "terminal_closed":
                ClosureReceipt(**record["closure_receipt"]).validate(request)
            if record["status"] == "intent" and any(
                record[key] is not None
                for key in ("executor_receipt", "job_id", "owned_job", "closure_receipt")
            ):
                raise ValueError("unattempted intent contains submission evidence")
            if record["status"] != "intent":
                claims = [claim for claim in snapshot.claims if claim.intent_id == intent_id]
                if (
                    len(claims) != 1
                    or claims[0].status == "reserved"
                    or claims[0].run_id != record["run_id"]
                    or claims[0].run_directory != record["run_directory"]
                    or claims[0].metadata
                    != {"namespace": NAMESPACE, "request_sha256": record["request_sha256"]}
                ):
                    raise ValueError("attempted intent has no matching consumed claim")
        except (KeyError, TypeError, ValueError, SlurmError, DispatchError) as exc:
            raise StoreIntegrityError("invalid persisted dispatch identity") from exc
    return ledger


def _commit(
    store: CampaignStore, snapshot: RegistrySnapshot, ledger: dict, reason: str
) -> RegistrySnapshot:
    if snapshot.registry.get(NAMESPACE) == ledger:
        return snapshot
    registry = copy.deepcopy(snapshot.registry)
    registry[NAMESPACE] = ledger
    return store.commit_transition(
        expected_generation=snapshot.generation,
        registry=registry,
        state=snapshot.state,
        reason=reason,
    )


def _conflict(record: dict, reason: str) -> None:
    record["status"], record["conflict"] = "ownership_conflict", reason


def _reconcile(
    ledger: dict, campaign_id: str, snapshot: AccountSnapshot, adapter: SlurmAdapter
) -> bool:
    """Record only changed facts. Empty reads and query timestamps do not churn generations."""
    for intent_id, record in ledger["intents"].items():
        if record["status"] in {"intent", "ownership_conflict"}:
            continue
        request = _request(campaign_id, intent_id, record)
        jobs = [job for job in snapshot.allocations() if job.matches(request)]
        suspicious = [
            job
            for job in snapshot.allocations()
            if (
                job.name == job_name(campaign_id, intent_id)
                or job.comment == job_comment(campaign_id, intent_id)
                or (record["job_id"] is not None and job.job_id == record["job_id"])
            )
            and not job.matches(request)
        ]
        if suspicious or len(jobs) > 1:
            _conflict(record, "job identity reused, mismatched, or duplicate intent allocations")
            continue
        if jobs:
            job = jobs[0]
            old_job = record["owned_job"]
            if (
                (record["job_id"] is not None and record["job_id"] != job.job_id)
                or (old_job is not None and old_job["submitted_at"] != job.submitted_at)
                or (record["status"] == "terminal_closed" and old_job["state"] != job.state)
                or (record["status"] in {"terminal_observed", "terminal_closed"} and job.active)
                or record["status"] == "submission_rejected"
            ):
                _conflict(
                    record, "scheduler evidence contradicts the prior identity/terminal state"
                )
                continue
            record["job_id"], record["owned_job"] = job.job_id, asdict(job)
            if record["status"] != "terminal_closed":
                record["status"] = (
                    "terminal_observed"
                    if not job.active
                    else "running"
                    if job.state == "RUNNING"
                    else "submitted"
                )
            continue
        if record["status"] in FINISHED:
            continue
        # Lookup errors cannot authorize reissue, nor erase a previously saved receipt.
        try:
            receipt = adapter.lookup(request)
        except Exception:
            receipt = None
        if receipt is None:
            continue
        if (
            record["job_id"] is not None
            and receipt.job_id is not None
            and record["job_id"] != receipt.job_id
        ):
            _conflict(record, "executor receipt changed its Slurm job identity")
            continue
        if receipt.status == "not_submitted" and record["job_id"] is not None:
            _conflict(record, "negative executor proof contradicts accepted submission")
            continue
        record["executor_receipt"] = asdict(receipt)
        if receipt.status == "accepted":
            record["job_id"], record["status"] = receipt.job_id, "submitted"
        elif receipt.status == "not_submitted":
            # Explicit negative proof resolves this attempt; it NEVER retries it.
            record["status"] = "submission_rejected"
    return any(record["status"] == "ownership_conflict" for record in ledger["intents"].values())


def capacity_decision(
    ledger: dict, campaign_id: str, snapshot: AccountSnapshot, *, now: datetime, max_age: float
) -> CapacityDecision:
    if snapshot.capacity is None:
        raise SlurmError("effective limit/account visibility evidence is missing")
    snapshot.capacity.validate(
        cluster=snapshot.cluster, account=snapshot.account, now=now, max_age=max_age
    )
    age = (utc(now) - utc(snapshot.observed_at)).total_seconds()
    if age < 0 or age > max_age:
        raise SlurmError("queue/accounting snapshot expired before submission")
    active = {job.key: job for job in snapshot.allocations() if job.active}
    campaign_jobs = {
        job.key for job in active.values() if job.comment.startswith("ame-v1:" + campaign_id + ":")
    }
    missing = 0
    for intent_id, record in ledger["intents"].items():
        if record["status"] not in ATTEMPTED:
            continue
        request = _request(campaign_id, intent_id, record)
        represented = [job for job in active.values() if job.matches(request)]
        if represented:
            campaign_jobs.update(job.key for job in represented)
        else:
            # Unobserved accepted IDs and unresolved no-ID intents each reserve one.
            missing += 1
    stage, limit = ledger["stage_cap"], min(snapshot.capacity.max_submit_jobs, 8)
    account_count, campaign_count = len(active) + missing, len(campaign_jobs) + missing
    available = max(0, min(stage - campaign_count, limit - account_count - 2))
    return CapacityDecision(stage, limit, account_count, campaign_count, missing, available)


def _claim(
    store: CampaignStore, snapshot: RegistrySnapshot, intent_id: str, record: dict
) -> RegistrySnapshot:
    existing = [claim for claim in snapshot.claims if claim.intent_id == intent_id]
    if not existing:
        store.claim_run(
            expected_generation=snapshot.generation,
            run_id=record["run_id"],
            intent_id=intent_id,
            run_directory=record["run_directory"],
            metadata={"namespace": NAMESPACE, "request_sha256": record["request_sha256"]},
        )
        snapshot = store.read_registry()
        existing = [claim for claim in snapshot.claims if claim.intent_id == intent_id]
    claim = existing[0]
    if (
        claim.run_id != record["run_id"]
        or claim.run_directory != record["run_directory"]
        or claim.metadata != {"namespace": NAMESPACE, "request_sha256": record["request_sha256"]}
        or claim.status not in {"reserved", "started"}
    ):
        raise DispatchError("local claim conflicts with persisted submission intent")
    if claim.status == "reserved":
        store.start_claim(
            expected_generation=snapshot.generation, run_id=claim.run_id, token=claim.token
        )
        snapshot = store.read_registry()
    # A crash here can safely resume: no external call precedes submit_unknown.
    return snapshot


def advance_campaign_once(
    store: CampaignStore,
    adapter: SlurmAdapter,
    *,
    requests: Sequence[DispatchRequest] = (),
    admission_verifier: AdmissionVerifier | None = None,
    closure_verifier: ClosureVerifier | None = None,
    stage_approval: StageApproval | None = None,
    stage_verifier: Callable[[str, StageApproval], bool] | None = None,
) -> StepResult:
    """Reconcile all known intents, submit at most ONE new intent, and return.

    Invariant: an attempted intent never calls submit again, even after a crash
    before the actual RPC. Conservative loss of liveness requires diagnosis, not
    guessing. No cleanup/cancel/retry is performed by this function. Local claims
    reserve the local evidence directory; remote workers still need their own
    durable claim/start mechanism in the remote executor.
    """
    try:
        with store.lock():
            return _advance_locked(
                store,
                adapter,
                requests=requests,
                admission_verifier=admission_verifier,
                closure_verifier=closure_verifier,
                stage_approval=stage_approval,
                stage_verifier=stage_verifier,
            )
    except LockUnavailable:
        return StepResult("skipped_lock", "another process holds the campaign writer lock", None)


def _advance_locked(
    store: CampaignStore,
    adapter: SlurmAdapter,
    *,
    requests: Sequence[DispatchRequest],
    admission_verifier: AdmissionVerifier | None,
    stage_approval: StageApproval | None,
    closure_verifier: ClosureVerifier | None,
    stage_verifier: Callable[[str, StageApproval], bool] | None,
) -> StepResult:
    snapshot = store.read_registry()  # Never use a pre-lock snapshot or derived STATE.
    ledger = _ledger(snapshot)
    now = utc(adapter.clock())
    since = min(
        [now] + [datetime.fromisoformat(x["created_at"]) for x in ledger["intents"].values()]
    )
    try:
        observed = adapter.account_snapshot(since=since)
    except SlurmError:
        return StepResult("blocked", "scheduler/capacity query is incomplete", snapshot.generation)
    conflict = _reconcile(ledger, store.campaign_id, observed, adapter)
    snapshot = _commit(store, snapshot, ledger, "reconcile dispatch evidence")
    if conflict:
        return StepResult(
            "blocked", "owned identity conflict requires diagnosis", snapshot.generation
        )
    for intent_id, record in ledger["intents"].items():
        if record["status"] != "terminal_observed" or closure_verifier is None:
            continue
        request = _request(store.campaign_id, intent_id, record)
        try:
            closure = closure_verifier(request, copy.deepcopy(record["owned_job"]))
            if closure is not None:
                if not isinstance(closure, ClosureReceipt):
                    raise DispatchError("closure verifier did not return a ClosureReceipt")
                closure.validate(request)
                record["closure_receipt"], record["status"] = asdict(closure), "terminal_closed"
        except Exception:
            return StepResult(
                "blocked",
                "worker/release/recovery closure could not be verified",
                snapshot.generation,
                intent_id,
            )
    snapshot = _commit(store, snapshot, ledger, "record independently verified terminal closure")
    if any(record["status"] == "terminal_observed" for record in ledger["intents"].values()):
        return StepResult(
            "blocked", "worker/release/recovery closure is pending", snapshot.generation
        )
    if stage_approval is not None and stage_approval.to_cap != ledger["stage_cap"]:
        current = ledger["stage_cap"]
        if (
            type(stage_approval.from_cap) is not int
            or type(stage_approval.to_cap) is not int
            or stage_approval.from_cap != current
            or current == 6
            or stage_approval.to_cap != STAGES[STAGES.index(current) + 1]
            or stage_verifier is None
        ):
            return StepResult(
                "blocked", "stage expansion requires the next verified gate", snapshot.generation
            )
        digest(stage_approval.evidence_sha256)
        if stage_verifier(store.campaign_id, stage_approval) is not True:
            return StepResult(
                "blocked", "stage expansion gate was not verified", snapshot.generation
            )
        ledger["stage_cap"] = stage_approval.to_cap
        ledger["stage_evidence"].append(asdict(stage_approval))
        snapshot = _commit(store, snapshot, ledger, "advance verified concurrency stage")
    try:
        capacity = capacity_decision(
            ledger,
            store.campaign_id,
            observed,
            now=adapter.clock(),
            max_age=adapter.max_snapshot_age_seconds,
        )
    except SlurmError:
        return StepResult("blocked", "no current verifiable account capacity", snapshot.generation)
    if capacity.available == 0:
        return StepResult(
            "idle", "stage or account capacity is exhausted", snapshot.generation, capacity=capacity
        )
    for candidate in requests:
        intent_id, contract = candidate.validate(store.campaign_id)
        request = SubmissionRequest(store.campaign_id, intent_id, candidate.spec)
        adapter.check_request(request)
        existing = ledger["intents"].get(intent_id)
        run_directory = str(candidate.local_run_directory.resolve())
        if any(
            other_id != intent_id
            and other["spec"]["cluster"] == candidate.spec.cluster
            and other["spec"]["work_dir"] == candidate.spec.work_dir
            for other_id, other in ledger["intents"].items()
        ):
            raise DispatchError("remote output directory was already assigned to another intent")
        if existing is not None:
            if (
                existing["request_sha256"] != request.request_sha256
                or existing["run_id"] != candidate.run_id
                or existing["run_directory"] != run_directory
            ):
                raise DispatchError("same intent cannot change its execution spec or local claim")
            if existing["status"] != "intent":
                continue
        if admission_verifier is None:
            return StepResult(
                "blocked",
                "execution admission verifier is absent",
                snapshot.generation,
                intent_id,
                capacity=capacity,
            )
        gate = admission_verifier(candidate)
        if not isinstance(gate, GateDecision):
            raise DispatchError("admission verifier did not return a GateDecision")
        gate.validate()
        if not gate.granted:
            return StepResult(
                "blocked",
                "execution admission denied",
                snapshot.generation,
                intent_id,
                capacity=capacity,
            )
        if adapter.executor is None:
            return StepResult(
                "blocked",
                "durable remote intent executor is absent",
                snapshot.generation,
                intent_id,
                capacity=capacity,
            )
        if existing is None:
            ledger["intents"][intent_id] = {
                "contract": contract,
                "spec": candidate.spec.payload(),
                "request_sha256": request.request_sha256,
                "run_id": candidate.run_id,
                "run_directory": run_directory,
                "created_at": now.isoformat(),
                "status": "intent",
                "job_id": None,
                "owned_job": None,
                "executor_receipt": None,
                "admission_sha256": gate.evidence_sha256,
                "conflict": None,
                "closure_receipt": None,
            }
            snapshot = _commit(store, snapshot, ledger, "persist submission intent")
        record = ledger["intents"][intent_id]
        snapshot = _claim(store, snapshot, intent_id, record)
        # Gates/claim IO may be slow; expired snapshots must not authorize submission.
        try:
            capacity_decision(
                ledger,
                store.campaign_id,
                observed,
                now=adapter.clock(),
                max_age=adapter.max_snapshot_age_seconds,
            )
        except SlurmError:
            return StepResult(
                "blocked",
                "capacity expired during admission",
                snapshot.generation,
                intent_id,
                capacity=capacity,
            )
        record["status"] = "submit_unknown"
        record["admission_sha256"] = gate.evidence_sha256
        snapshot = _commit(store, snapshot, ledger, "persist submit_unknown before external call")
        try:
            receipt = adapter.submit(request)
        except Exception:
            return StepResult(
                "submit_unknown",
                "submit response unavailable; reconcile only",
                snapshot.generation,
                intent_id,
                capacity=capacity,
            )
        # A failure committing the receipt propagates. The prior durable unknown
        # remains authoritative and a later tick will NOT call submit again.
        record["executor_receipt"] = asdict(receipt)
        if receipt.status == "accepted":
            record["job_id"], record["status"] = receipt.job_id, "submitted"
        elif receipt.status == "not_submitted":
            record["status"] = "submission_rejected"
        snapshot = _commit(store, snapshot, ledger, "record executor submission receipt")
        return StepResult(
            record["status"],
            "executor result recorded; worker outcome unverified",
            snapshot.generation,
            intent_id,
            record["job_id"],
            capacity,
        )
    return StepResult(
        "idle", "no unattempted eligible intent", snapshot.generation, capacity=capacity
    )
