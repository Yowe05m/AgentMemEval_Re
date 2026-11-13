from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agentmemeval.path1_v1.operations.slurm import (
    CapacityEvidence,
    CommandResult,
    IntentReceipt,
    OwnershipError,
    SlurmAdapter,
    SlurmError,
    SubmissionDisabled,
    SubmissionRequest,
    SubmitSpec,
    job_comment,
    job_name,
)

NOW = datetime(2026, 11, 13, 8, 0, tzinfo=UTC)
INTENT = "a" * 64


def request() -> SubmissionRequest:
    return SubmissionRequest(
        "campaign",
        INTENT,
        SubmitSpec(
            "cluster", "account", "owner", "gpu", "/runs/campaign/one", "/code/worker.sh", "b" * 64
        ),
    )


def row(
    job_id: str = "101",
    *,
    own: bool = True,
    state: str = "RUNNING",
    submitted: str = "2026-11-13T07:59:00",
) -> str:
    return (
        "|".join(
            (
                job_id,
                "owner" if own else "neighbor",
                "account",
                job_name("campaign", INTENT) if own else "other-work",
                state,
                job_comment("campaign", INTENT) if own else "",
                submitted,
                "/runs/campaign/one" if own else "/unrelated",
            )
        )
        + "\n"
    )


class Transport:
    def __init__(self, queue: str = "", accounting: str = ""):
        self.queue, self.accounting = queue, accounting
        self.calls: list[tuple[str, ...]] = []
        self.result: CommandResult | Exception | None = None

    def run(self, argv, *, environment, inherit_environment, timeout_seconds):
        assert environment["TZ"] == "UTC" and environment["LC_ALL"] == "C"
        assert inherit_environment is False and 0 < timeout_seconds <= 60
        self.calls.append(argv)
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is not None:
            return self.result
        return CommandResult(
            0, {"squeue": self.queue, "sacct": self.accounting, "scancel": ""}[argv[0]]
        )


def adapter(transport: Transport, **kwargs) -> SlurmAdapter:
    return SlurmAdapter(
        transport, cluster="cluster", account="account", user="owner", clock=lambda: NOW, **kwargs
    )


def test_queries_cover_other_users_expanded_arrays_and_explicit_history() -> None:
    transport = Transport(row() + row("202_4", own=False), row(state="PENDING"))
    snapshot = adapter(transport).query(since=NOW - timedelta(days=3))
    assert len(snapshot.allocations()) == 2
    assert next(job for job in snapshot.allocations() if job.job_id == "101").state == "RUNNING"
    assert snapshot.capacity is None
    queue, accounting = transport.calls
    assert "--array" in queue and "--all" in queue and "--states=all" in queue
    assert not any("--user=" in arg for args in transport.calls for arg in args)
    assert "--allusers" in accounting and "--allocations" in accounting
    assert "--starttime=2026-11-10T07:55:00" in accounting
    assert "--duplicates" in accounting


@pytest.mark.parametrize(
    "text",
    [
        "bad truncated row",
        row("202_[1-5]", own=False),
        row("101.batch"),
        row(state="COMPLETED+"),
        row(state="NEW_UNKNOWN_STATE"),
        row(submitted="Unknown"),
        row().replace("|account|", "|another_account|"),
        row().replace("/runs/campaign/one", "/invalid|path"),
        row() + row(state="PENDING"),
    ],
)
def test_uninterpretable_rows_fail_closed(text: str) -> None:
    with pytest.raises(SlurmError):
        adapter(Transport(text)).query(since=NOW)


@pytest.mark.parametrize(
    "failure",
    [
        CommandResult(1, "partial"),
        CommandResult(0, "", "permission warning"),
        TimeoutError("connection lost"),
    ],
)
def test_failed_or_partial_query_is_not_empty_capacity(failure) -> None:
    transport = Transport()
    transport.result = failure
    with pytest.raises(SlurmError):
        adapter(transport).account_snapshot(since=NOW)


def test_accounting_cancelled_uid_and_duplicate_identical_rows() -> None:
    snapshot = adapter(Transport("", row(state="CANCELLED by 123") * 2)).query(since=NOW)
    assert len(snapshot.accounting) == 1 and snapshot.accounting[0].state == "CANCELLED"


@pytest.mark.parametrize(
    "override",
    [
        {"observed_at": NOW - timedelta(seconds=31)},
        {"observed_at": NOW + timedelta(seconds=1)},
        {"full_account_visibility": False},
        {"max_submit_jobs": True},
        {"account": "wrong"},
        {"counting_unit": "compressed_array_parent"},
    ],
)
def test_bad_capacity_keeps_read_evidence_but_grants_no_capacity(override: dict) -> None:
    evidence = CapacityEvidence("cluster", "account", 8, NOW, "c" * 64, True)
    snapshot = adapter(
        Transport(row()), capacity_probe=lambda: replace(evidence, **override)
    ).account_snapshot(since=NOW)
    assert len(snapshot.queue) == 1 and snapshot.capacity is None


def test_good_capacity_is_current_and_scoped() -> None:
    evidence = CapacityEvidence("cluster", "account", 8, NOW, "c" * 64, True)
    assert (
        adapter(Transport(), capacity_probe=lambda: evidence).account_snapshot(since=NOW).capacity
        == evidence
    )


def test_default_submission_is_disabled_without_running_sbatch() -> None:
    transport = Transport()
    with pytest.raises(SubmissionDisabled):
        adapter(transport).submit(request())
    assert transport.calls == []


def test_sbatch_envelope_has_full_identity_owned_logs_and_bounded_single_gpu() -> None:
    args = request().sbatch_argv()
    assert "--parsable" in args and "--no-requeue" in args and "--gpus=1" in args
    assert "--time=00:02:00" in args and "--nodes=1" in args and "--ntasks=1" in args
    assert f"--job-name=ame-{INTENT}" in args
    assert f"--comment=ame-v1:campaign:{INTENT}" in args
    assert "--output=/runs/campaign/one/slurm-%j.out" in args
    assert "--export=NONE" in args
    assert not any("array" in arg for arg in args)


@pytest.mark.parametrize(
    "override",
    [
        {"gpus": 2},
        {"gpus": True},
        {"walltime_seconds": 0},
        {"walltime_seconds": 86401},
        {"work_dir": "/runs/../escape"},
        {"script_args": ("bad\nargument",)},
        {"script_path": "relative.sh"},
        {"account": "one,two"},
    ],
)
def test_submit_spec_rejects_unsafe_or_unsupported_configuration(override: dict) -> None:
    with pytest.raises(ValueError):
        replace(request().spec, **override)


@pytest.mark.parametrize(
    "receipt",
    [
        IntentReceipt("d" * 64, "accepted", "e" * 64, "101", True, 1),
        IntentReceipt(request().request_sha256, "accepted", "e" * 64, "101_1", True, 1),
        IntentReceipt(request().request_sha256, "accepted", "e" * 64, "101", True, 2),
        IntentReceipt(request().request_sha256, "not_submitted", "e" * 64, None, False, 0),
        IntentReceipt(request().request_sha256, "not_submitted", "e" * 64, "101", True, 0),
        IntentReceipt(request().request_sha256, "unknown", "e" * 64, "101"),
    ],
)
def test_receipts_cannot_fabricate_one_submission_or_definitive_negative(receipt) -> None:
    with pytest.raises(SlurmError):
        receipt.validate(request())


def test_cancel_checks_live_identity_and_only_names_one_owned_job() -> None:
    transport = Transport(row() + row("102", own=False))
    target = adapter(transport, allow_cancel=True)
    owned = target.query(since=NOW).queue[0]
    result = target.cancel_owned(request(), owned)
    assert result == {"status": "cancel_requested", "job_id": "101", "release_verified": False}
    command = transport.calls[-1]
    assert command == (
        "scancel",
        "--clusters=cluster",
        "--account=account",
        "--user=owner",
        f"--name=ame-{INTENT}",
        "101",
    )
    assert "102" not in command and "--signal=KILL" not in command


@pytest.mark.parametrize(
    "new_row",
    [
        row(own=False),
        row().replace("/runs/campaign/one", "/different/output"),
        row(submitted="2026-11-13T08:00:00"),
        row().replace("ame-v1:campaign:", "ame-v1:neighbor:"),
    ],
)
def test_reused_or_mismatched_job_id_is_never_cancelled(new_row: str) -> None:
    transport = Transport(row())
    target = adapter(transport, allow_cancel=True)
    owned = target.query(since=NOW).queue[0]
    transport.queue = new_row
    with pytest.raises(OwnershipError):
        target.cancel_owned(request(), owned)
    assert all(command[0] != "scancel" for command in transport.calls)


def test_queue_absence_is_not_release_and_cancel_is_disabled_by_default() -> None:
    transport = Transport(row())
    target = adapter(transport)
    owned = target.query(since=NOW).queue[0]
    with pytest.raises(SubmissionDisabled):
        target.cancel_owned(request(), owned)
    target.allow_cancel = True
    transport.queue = ""
    assert target.cancel_owned(request(), owned)["status"] == "not_visible"
    assert all(command[0] != "scancel" for command in transport.calls)
