from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations.contracts import (
    SOURCE_CONTRACT_SHA256,
    ContractError,
    canonical_intent_id,
    content_digest,
)
from agentmemeval.path1_v1.operations.dispatch import (
    NAMESPACE,
    ClosureReceipt,
    DispatchError,
    DispatchRequest,
    GateDecision,
    StageApproval,
    advance_campaign_once,
)
from agentmemeval.path1_v1.operations.slurm import (
    CapacityEvidence,
    CommandResult,
    IntentReceipt,
    SlurmAdapter,
    SubmitSpec,
    job_comment,
    job_name,
)
from agentmemeval.path1_v1.operations.state import CampaignStore, ClaimConflict, StoreIntegrityError

NOW = datetime(2026, 11, 13, 8, 0, tzinfo=UTC)
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/agentmemeval/path1_v1/assets/path1_source_contract.json"
)


def make_contract(number: int = 1) -> dict:
    """Synthetic E0 identity, never a scientific or runtime attestation."""
    core = [{"path": "src/core.py", "size": 1, "sha256": "a" * 64}]
    ops = [{"path": "src/ops.py", "size": 1, "sha256": "b" * 64}]
    identity = {
        "schema_version": 1,
        "kind": "path1_runtime_identity",
        "core_sha": "c" * 40,
        "ops_sha": "c" * 40,
        "core_files": core,
        "operations_files": ops,
        "core_manifest_sha256": content_digest(core),
        "operations_manifest_sha256": content_digest(ops),
        "config": {"path": "config.json", "size": 1, "sha256": "d" * 64},
        "dependency_lock": {"path": "requirements.lock", "size": 1, "sha256": "e" * 64},
        "model": None,
        "dirty": True,
        "git_status_sha256": "f" * 64,
        "untracked_declared_paths": [],
        "uncovered_execution_paths": [],
        "untracked_execution_paths": [],
        "content_matches_head": True,
        "head_content_mismatches": [],
        "runtime_attestation": "not_performed_by_static_identity_builder",
    }
    identity["identity_sha256"] = content_digest(identity)
    return {
        "schema_version": 1,
        "kind": "path1_campaign_contract",
        "campaign_id": "campaign",
        "phase": "E0",
        "source_contract": json.loads(SOURCE.read_text(encoding="utf-8")),
        "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "runtime_identity": identity,
        "seed_namespace": "probe_namespace",
        "cohort": {
            "formation_count": 0,
            "episodes_per_artifact_per_arm": 0,
            "seed_inventory_sha256": None,
            "freeze_receipt_sha256": None,
        },
        "job": {
            "kind": "resource_probe",
            "artifact_id": None,
            "pair_ids": [],
            "arms": [],
            "domain": "source",
            "attempt": 1,
            "output_relpath": f"campaign/E0/probe_{number}/attempt_01",
            "formation_receipt": None,
            "retry_evidence": None,
        },
        "not_for_paper": True,
        "execution_admission": "not_granted_by_contract_validation",
    }


def make_request(root: Path, number: int = 1) -> DispatchRequest:
    contract = make_contract(number)
    return DispatchRequest(
        contract,
        SubmitSpec(
            "cluster",
            "account",
            "owner",
            "gpu",
            "/runs/" + contract["job"]["output_relpath"],
            "/code/worker.sh",
            "a" * 64,
        ),
        f"run_{number}",
        root / f"run_{number}",
    )


def job_row(
    request: DispatchRequest,
    *,
    job_id: str = "101",
    state: str = "PENDING",
    submitted: str = "2026-11-13T08:00:00",
) -> str:
    intent = canonical_intent_id(request.contract)
    return (
        "|".join(
            (
                job_id,
                "owner",
                "account",
                job_name("campaign", intent),
                state,
                job_comment("campaign", intent),
                submitted,
                request.spec.work_dir,
            )
        )
        + "\n"
    )


def other_rows(count: int) -> str:
    return "".join(
        f"{201 + x}|neighbor|account|unrelated_{x}|RUNNING||2026-11-13T07:00:00|/unrelated\n"
        for x in range(count)
    )


class FakeTransport:
    def __init__(self):
        self.queue = ""
        self.accounting = ""
        self.calls = []
        self.failure = False

    def run(self, argv, **kwargs):
        assert kwargs["inherit_environment"] is False
        self.calls.append(argv)
        if self.failure:
            raise TimeoutError("fixture offline")
        assert argv[0] in {"squeue", "sacct"}, "dispatch must never directly sbatch/scancel"
        return CommandResult(0, self.queue if argv[0] == "squeue" else self.accounting)


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.receipts = {}
        self.lose_reply = False
        self.hide_lookup = False
        self.before_submit = None
        self.negative = False

    def submit_once(self, request, *, timeout_seconds):
        assert 0 < timeout_seconds <= 60
        if self.before_submit is not None:
            self.before_submit(request)
        self.calls.append(request)
        assert request.intent_id not in self.receipts, "second submission of same intent"
        receipt = IntentReceipt(
            request.request_sha256,
            "not_submitted" if self.negative else "accepted",
            "f" * 64,
            None if self.negative else str(100 + len(self.calls)),
            True,
            0 if self.negative else 1,
        )
        self.receipts[request.intent_id] = receipt
        if self.lose_reply:
            raise TimeoutError("fixture lost reply after Slurm accepted")
        return receipt

    def lookup(self, request, *, timeout_seconds):
        assert 0 < timeout_seconds <= 60
        return None if self.hide_lookup else self.receipts.get(request.intent_id)


def make_fixture(root: Path, *, initialize: bool = True):
    store = CampaignStore(root / "control", "campaign")
    if initialize:
        with store.lock():
            store.initialize(registry={"unrelated": {"preserve": True}}, state={"summary": "old"})
    transport, executor = FakeTransport(), FakeExecutor()
    fixture = SimpleNamespace(
        store=store,
        transport=transport,
        executor=executor,
        now=NOW,
        limit=8,
        visibility=True,
        root=root,
    )
    fixture.adapter = SlurmAdapter(
        transport,
        cluster="cluster",
        account="account",
        user="owner",
        executor=executor,
        capacity_probe=lambda: CapacityEvidence(
            "cluster", "account", fixture.limit, fixture.now, "e" * 64, fixture.visibility
        ),
        clock=lambda: fixture.now,
    )
    return fixture


def step(fixture, request: DispatchRequest | None = None, **kwargs):
    return advance_campaign_once(
        fixture.store,
        fixture.adapter,
        requests=[request or make_request(fixture.root)],
        admission_verifier=kwargs.pop("admission_verifier", lambda _: GateDecision(True, "a" * 64)),
        **kwargs,
    )


def ledger(fixture) -> dict:
    return fixture.store.read_registry().registry[NAMESPACE]


@pytest.fixture
def harness(tmp_path):
    return make_fixture(tmp_path)


def test_two_ticks_submit_once_without_unnecessary_state_writes(harness) -> None:
    first = step(harness)
    after = harness.store.read_registry()
    second = step(harness)
    assert first.action == "submitted" and second.action == "idle"
    assert len(harness.executor.calls) == 1
    assert harness.store.read_registry().generation == after.generation
    assert after.registry["unrelated"] == {"preserve": True} and after.state == {"summary": "old"}
    assert len(after.claims) == 1 and after.claims[0].status == "started"


def test_persisted_unknown_and_claim_precede_the_external_side_effect(harness) -> None:
    def check(request):
        snapshot = harness.store.read_registry()
        assert (
            snapshot.registry[NAMESPACE]["intents"][request.intent_id]["status"] == "submit_unknown"
        )
        assert snapshot.claims[0].status == "started"

    harness.executor.before_submit = check
    assert step(harness).action == "submitted"


def test_contract_validation_alone_and_missing_executor_do_not_grant_admission(harness) -> None:
    assert step(harness, admission_verifier=None).action == "blocked"
    harness.adapter.executor = None
    assert step(harness).action == "blocked"
    assert not harness.executor.calls and not harness.store.read_registry().claims


def test_lost_reply_and_days_of_absence_never_resubmit(harness) -> None:
    harness.executor.lose_reply, harness.executor.hide_lookup = True, True
    assert step(harness).action == "submit_unknown"
    original = harness.store.read_registry().generation
    for _ in range(4):
        harness.now += timedelta(days=1)
        assert step(harness).action == "idle"
    assert len(harness.executor.calls) == 1
    assert harness.store.read_registry().generation == original
    assert next(iter(ledger(harness)["intents"].values()))["status"] == "submit_unknown"


def test_delayed_executor_receipt_resolves_unknown_without_a_new_submit(harness) -> None:
    harness.executor.lose_reply = True
    first = step(harness)
    harness.now += timedelta(minutes=10)
    second = step(harness)
    record = ledger(harness)["intents"][first.intent_id]
    assert second.action == "idle" and record["status"] == "submitted"
    assert record["job_id"] == "101" and len(harness.executor.calls) == 1


def test_visible_unknown_is_deduplicated_with_account_job(harness) -> None:
    harness.executor.lose_reply, harness.executor.hide_lookup = True, True
    first = step(harness)
    harness.transport.queue = job_row(make_request(harness.root), state="RUNNING")
    second = step(harness)
    assert second.capacity.account_occupancy == second.capacity.campaign_occupancy == 1
    assert second.capacity.unresolved_reservations == 0
    assert ledger(harness)["intents"][first.intent_id]["status"] == "running"
    assert len(harness.executor.calls) == 1


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"])
def test_scheduler_terminal_is_not_scientific_acceptance_or_claim_completion(
    harness, state
) -> None:
    first = step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state=state)
    second = step(harness)
    record = ledger(harness)["intents"][first.intent_id]
    assert record["status"] == "terminal_observed" and record["owned_job"]["state"] == state
    assert second.action == "blocked"
    assert harness.store.read_registry().claims[0].status == "started"
    assert "accepted" not in record and "failure_classification" not in record
    assert len(harness.executor.calls) == 1


def test_terminal_job_blocks_new_admitted_job_until_three_verified_closures(harness) -> None:
    first = step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state="COMPLETED")
    second_request = make_request(harness.root, 2)
    blocked = step(harness, second_request)
    assert blocked.action == "blocked" and "closure" in blocked.reason
    assert len(harness.executor.calls) == 1

    def verified_closure(request, scheduler_job):
        assert request.intent_id == first.intent_id and scheduler_job["state"] == "COMPLETED"
        return ClosureReceipt(
            request.intent_id, request.request_sha256, "a" * 64, "b" * 64, "c" * 64
        )

    result = step(harness, second_request, closure_verifier=verified_closure)
    assert result.action == "submitted" and len(harness.executor.calls) == 2
    record = ledger(harness)["intents"][first.intent_id]
    assert record["status"] == "terminal_closed" and "accepted" not in record


def test_wrong_intent_closure_cannot_release_terminal_gate(harness) -> None:
    step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state="COMPLETED")
    result = step(
        harness,
        make_request(harness.root, 2),
        closure_verifier=lambda request, _: ClosureReceipt(
            "d" * 64, request.request_sha256, "a" * 64, "b" * 64, "c" * 64
        ),
    )
    assert result.action == "blocked" and len(harness.executor.calls) == 1


@pytest.mark.parametrize("others,expected", [(5, "submitted"), (6, "idle"), (8, "idle")])
def test_all_other_account_users_consume_capacity_and_two_slots_stay_reserved(
    harness, others, expected
):
    harness.transport.queue = other_rows(others)
    result = step(harness)
    assert result.action == expected and result.capacity.account_occupancy == others
    assert result.capacity.reserve_slots == 2


def test_dynamic_limit_drop_does_not_cancel_or_restart_any_job(harness) -> None:
    step(harness)
    harness.transport.queue = job_row(make_request(harness.root)) + other_rows(2)
    harness.limit = 2
    result = step(harness, make_request(harness.root, 2))
    assert result.action == "idle" and result.capacity.effective_limit == 2
    assert len(harness.executor.calls) == 1
    assert all(call[0] in {"squeue", "sacct"} for call in harness.transport.calls)


def test_expansion_only_accepts_one_verified_stage_at_a_time(harness) -> None:
    bad = step(
        harness, stage_approval=StageApproval(1, 4, "a" * 64), stage_verifier=lambda *_: True
    )
    assert bad.action == "blocked" and not harness.executor.calls
    bad = step(harness, stage_approval=StageApproval(1, 2, "a" * 64))
    assert bad.action == "blocked"
    for previous, current in ((1, 2), (2, 4), (4, 6)):
        result = step(
            harness,
            stage_approval=StageApproval(previous, current, "a" * 64),
            stage_verifier=lambda *_: True,
        )
        assert result.capacity.stage_cap == current
    assert [entry["to_cap"] for entry in ledger(harness)["stage_evidence"]] == [2, 4, 6]
    assert len(harness.executor.calls) == 1


def test_limit_is_conservatively_capped_at_eight(harness) -> None:
    harness.limit = 100
    result = step(harness)
    assert result.capacity.effective_limit == 8 and result.capacity.stage_cap == 1


@pytest.mark.parametrize("visibility", [False, None])
def test_unverified_visibility_or_missing_limit_denies_submission(harness, visibility) -> None:
    harness.visibility = visibility
    assert step(harness).action == "blocked"
    assert not harness.executor.calls


def test_query_failure_is_not_an_empty_queue(harness) -> None:
    harness.transport.failure = True
    assert step(harness).action == "blocked"
    assert not harness.executor.calls and harness.store.read_registry().generation == 0


def test_failed_limit_probe_still_reconciles_terminal_job(harness) -> None:
    first = step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state="COMPLETED")
    harness.adapter.capacity_probe = lambda: (_ for _ in ()).throw(
        RuntimeError("no limit evidence")
    )
    assert step(harness).action == "blocked"
    assert ledger(harness)["intents"][first.intent_id]["status"] == "terminal_observed"


def test_expired_capacity_after_gate_does_not_submit_and_can_resume_safe_intent(harness) -> None:
    def slow_gate(_):
        harness.now += timedelta(seconds=31)
        return GateDecision(True, "b" * 64)

    assert step(harness, admission_verifier=slow_gate).action == "blocked"
    assert not harness.executor.calls
    assert next(iter(ledger(harness)["intents"].values()))["status"] == "intent"
    assert step(harness).action == "submitted"


def test_duplicate_owned_allocations_block_without_automatic_cancellation(harness) -> None:
    first = step(harness)
    request = make_request(harness.root)
    harness.transport.queue = job_row(request) + job_row(request, job_id="102")
    assert step(harness, make_request(harness.root, 2)).action == "blocked"
    assert ledger(harness)["intents"][first.intent_id]["status"] == "ownership_conflict"
    assert len(harness.executor.calls) == 1


def test_job_id_reuse_or_wrong_output_directory_blocks_ownership(harness) -> None:
    first = step(harness)
    harness.transport.queue = job_row(make_request(harness.root)).replace("/runs/", "/wrong/")
    assert step(harness).action == "blocked"
    assert ledger(harness)["intents"][first.intent_id]["owned_job"] is None


def test_same_intent_cannot_change_submission_spec(harness) -> None:
    request = make_request(harness.root)
    step(harness, request)
    changed = replace(request, spec=replace(request.spec, walltime_seconds=121))
    with pytest.raises(DispatchError, match="same intent"):
        step(
            harness,
            changed,
            stage_approval=StageApproval(1, 2, "a" * 64),
            stage_verifier=lambda *_: True,
        )
    assert len(harness.executor.calls) == 1


def test_definitive_negative_receipt_resolves_but_never_retries_same_attempt(harness) -> None:
    harness.executor.negative = True
    first = step(harness)
    second = step(harness)
    assert first.action == "submission_rejected" and second.action == "idle"
    assert len(harness.executor.calls) == 1


def test_existing_local_directory_prevents_any_external_call(harness) -> None:
    request = make_request(harness.root)
    request.local_run_directory.mkdir()
    with pytest.raises(ClaimConflict):
        step(harness, request)
    assert not harness.executor.calls


def test_invalid_contract_or_output_binding_is_rejected(harness) -> None:
    request = make_request(harness.root)
    invalid = copy.deepcopy(request.contract)
    invalid["not_for_paper"] = False
    with pytest.raises(ContractError):
        step(harness, replace(request, contract=invalid))
    with pytest.raises(DispatchError, match="remote work directory"):
        step(harness, replace(request, spec=replace(request.spec, work_dir="/wrong/path")))
    assert not harness.executor.calls


def test_missing_submission_process_exit_proof_remains_unknown(harness) -> None:
    harness.executor.lose_reply = True
    first = step(harness)
    request = harness.executor.calls[0]
    harness.executor.receipts[first.intent_id] = IntentReceipt(
        request.request_sha256, "not_submitted", "e" * 64, None, False, 0
    )
    assert step(harness).action == "idle"
    assert ledger(harness)["intents"][first.intent_id]["status"] == "submit_unknown"
    assert len(harness.executor.calls) == 1


def test_other_controller_lock_is_respected(harness) -> None:
    other_store = CampaignStore(harness.store.directory, "campaign")
    with other_store.lock():
        result = step(harness)
    assert result.action == "skipped_lock" and not harness.transport.calls


def test_unknown_plus_five_neighbors_uses_six_slots_not_five(harness) -> None:
    harness.executor.lose_reply, harness.executor.hide_lookup = True, True
    step(harness)
    harness.transport.queue = other_rows(5)
    result = step(
        harness,
        make_request(harness.root, 2),
        stage_approval=StageApproval(1, 2, "a" * 64),
        stage_verifier=lambda *_: True,
    )
    assert result.capacity.account_occupancy == 6
    assert result.capacity.unresolved_reservations == 1 and result.action == "idle"
    assert len(harness.executor.calls) == 1


def test_two_stage_allows_two_distinct_intents_but_not_a_third(harness) -> None:
    assert step(harness).action == "submitted"
    result = step(
        harness,
        make_request(harness.root, 2),
        stage_approval=StageApproval(1, 2, "a" * 64),
        stage_verifier=lambda *_: True,
    )
    assert result.action == "submitted"
    third = step(harness, make_request(harness.root, 3))
    assert third.action == "idle" and third.capacity.campaign_occupancy == 2
    assert len(harness.executor.calls) == 2


def test_stored_stage_without_expansion_receipts_fails_closed(harness) -> None:
    with harness.store.lock():
        snapshot = harness.store.read_registry()
        harness.store.commit_transition(
            expected_generation=snapshot.generation,
            registry={
                NAMESPACE: {
                    "schema_version": 1,
                    "stage_cap": 6,
                    "stage_evidence": [],
                    "intents": {},
                }
            },
            state={},
            reason="fixture malformed ledger",
        )
    with pytest.raises(StoreIntegrityError, match="concurrency"):
        step(harness)
    assert not harness.executor.calls


def test_terminal_observation_cannot_expand_stage_before_closure(harness) -> None:
    step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state="COMPLETED")
    result = step(
        harness,
        make_request(harness.root, 2),
        stage_approval=StageApproval(1, 2, "a" * 64),
        stage_verifier=lambda *_: True,
    )
    assert result.action == "blocked" and ledger(harness)["stage_cap"] == 1


def test_different_intents_cannot_reuse_a_remote_output_directory(harness) -> None:
    request = make_request(harness.root)
    step(harness, request)
    different = copy.deepcopy(request.contract)
    different["seed_namespace"] = "different_namespace"
    candidate = replace(
        request,
        contract=different,
        run_id="other_run",
        local_run_directory=harness.root / "other_run",
    )
    with pytest.raises(DispatchError, match="remote output directory"):
        step(
            harness,
            candidate,
            stage_approval=StageApproval(1, 2, "a" * 64),
            stage_verifier=lambda *_: True,
        )
    assert len(harness.executor.calls) == 1


def test_persisted_closed_status_without_three_receipts_fails_closed(harness) -> None:
    first = step(harness)
    harness.transport.accounting = job_row(make_request(harness.root), state="COMPLETED")
    step(harness)
    with harness.store.lock():
        snapshot = harness.store.read_registry()
        registry = copy.deepcopy(snapshot.registry)
        registry[NAMESPACE]["intents"][first.intent_id]["status"] = "terminal_closed"
        harness.store.commit_transition(
            expected_generation=snapshot.generation,
            registry=registry,
            state=snapshot.state,
            reason="fixture incomplete closure",
        )
    with pytest.raises(StoreIntegrityError):
        step(harness, make_request(harness.root, 2))
    assert len(harness.executor.calls) == 1


@pytest.mark.parametrize("status", ["terminal_observed", "terminal_closed"])
@pytest.mark.parametrize(
    "corruption", ["missing", "empty", "running", "pending", "unknown", "wrong_identity"]
)
def test_persisted_terminal_requires_an_explicit_terminal_owned_job(
    harness, status, corruption
) -> None:
    first = step(harness)
    harness.transport.queue = job_row(make_request(harness.root), state="RUNNING")
    step(harness)
    with harness.store.lock():
        snapshot = harness.store.read_registry()
        registry = copy.deepcopy(snapshot.registry)
        record = registry[NAMESPACE]["intents"][first.intent_id]
        record["status"] = status
        if corruption == "missing":
            record["owned_job"] = None
        elif corruption == "empty":
            record["owned_job"] = {}
        elif corruption == "wrong_identity":
            record["owned_job"]["state"] = "COMPLETED"
            record["owned_job"]["comment"] = "ame-v1:another_campaign:" + first.intent_id
        else:
            record["owned_job"]["state"] = {
                "running": "RUNNING",
                "pending": "PENDING",
                "unknown": "UNRECOGNIZED",
            }[corruption]
        if status == "terminal_closed":
            record["closure_receipt"] = {
                "intent_id": first.intent_id,
                "request_sha256": record["request_sha256"],
                "worker_terminal_sha256": "a" * 64,
                "release_receipt_sha256": "b" * 64,
                "recovery_receipt_sha256": "c" * 64,
            }
        harness.store.commit_transition(
            expected_generation=snapshot.generation,
            registry=registry,
            state=snapshot.state,
            reason="fixture invalid terminal job evidence",
        )
    # No current job can repair the inconsistent persisted record. Even valid
    # new-job admission/closure callbacks must not let it bypass ledger checks.
    harness.transport.queue = harness.transport.accounting = ""
    before = harness.store.read_registry().generation
    before_queries = len(harness.transport.calls)
    with pytest.raises(StoreIntegrityError, match="persisted dispatch identity"):
        step(
            harness,
            make_request(harness.root, 2),
            closure_verifier=lambda request, _: ClosureReceipt(
                request.intent_id, request.request_sha256, "a" * 64, "b" * 64, "c" * 64
            ),
        )
    assert len(harness.executor.calls) == 1
    assert len(harness.transport.calls) == before_queries
    assert harness.store.read_registry().generation == before
