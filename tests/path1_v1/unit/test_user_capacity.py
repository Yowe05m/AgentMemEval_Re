from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import user_capacity as capacity
from agentmemeval.path1_v1.operations.slurm import (
    READ_ENV,
    CommandResult,
    SlurmError,
    SubmissionRequest,
    SubmitSpec,
    job_comment,
    job_name,
)

NOW = datetime(2026, 11, 13, 16, tzinfo=UTC)
ASSOC = "n16r4|scze856|scze856|||7193|7192|gpugpu,normal|||8|||gres/gpu=8\n"


class FixtureTransport:
    def __init__(self, outputs=None):
        self.outputs = outputs or {"sacctmgr": ASSOC, "squeue": "", "sacct": ""}
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        output = self.outputs[Path(argv[0]).name]
        return output if type(output) is CommandResult else CommandResult(0, output, "")


def make_request(index=1, *, campaign="campaign", user="scze856"):
    return SubmissionRequest(
        campaign,
        f"{index:064x}",
        SubmitSpec(
            "n16r4",
            "scze856",
            user,
            "gpu",
            f"/runs/{campaign}/{index}",
            f"/launch/{campaign}/{index}.sh",
            "a" * 64,
        ),
    )


def reservation(request, *, status="submit_unknown", job_id=None):
    return {
        "campaign_id": request.campaign_id,
        "intent_id": request.intent_id,
        "spec": request.spec.payload(),
        "request_sha256": request.request_sha256,
        "status": status,
        "job_id": job_id,
    }


def scheduler_row(request, job_id="123", state="RUNNING", *, source="squeue", **changes):
    fields = {
        "job_id": job_id,
        "user": request.spec.user,
        "account": request.spec.account,
        "name": job_name(request.campaign_id, request.intent_id),
        "state": state,
        "comment": job_comment(request.campaign_id, request.intent_id),
        "submitted": "2026-11-13T15:00:00",
        "work_dir": request.spec.work_dir,
    }
    fields.update(changes)
    return ("n16r4|" if source == "sacct" else "") + "|".join(fields.values()) + "\n"


def make_user_capacity_fixture(tmp_path, monkeypatch):
    """Offline producer fixture; never evidence of actual Slurm or Linux credentials."""
    monkeypatch.setattr(capacity, "_effective_identity", lambda user: 4781)
    commands = {}
    for name in capacity.COMMAND_NAMES:
        path = tmp_path / name
        path.write_bytes(("fixture-" + name).encode())
        commands[name] = capacity.PinnedUserCommand(
            str(path), hashlib.sha256(path.read_bytes()).hexdigest()
        )
    transport = FixtureTransport()
    reader = capacity.UserAssociationReader(
        cluster="n16r4",
        account="scze856",
        user="scze856",
        commands=commands,
        since=NOW - timedelta(days=1),
        clock=lambda: NOW,
        transport=transport,
    )
    return {
        "reader": reader,
        "transport": transport,
        "snapshot": reader.read(),
        "request": make_request(),
        "ledger": {"schema_version": 1, "cluster": "n16r4", "account": "scze856", "intents": {}},
    }


@pytest.fixture
def case(tmp_path, monkeypatch):
    return make_user_capacity_fixture(tmp_path, monkeypatch)


def validate(case, *, snapshot=None, stage_cap=1, now=NOW, phase="E0"):
    return capacity.validate_user_association_capacity(
        case["snapshot"] if snapshot is None else snapshot,
        case["request"],
        case["ledger"],
        stage_cap,
        now,
        phase,
    )


def raw(case, *, association=None, queue=None, accounting=None):
    commands = list(case["snapshot"].commands)
    for index, value in enumerate((association, queue, accounting)):
        if value is not None:
            commands[index] = replace(commands[index], stdout=value)
    return replace(case["snapshot"], commands=tuple(commands))


def test_live_reader_shape_and_scope_are_explicit(case):
    result = validate(case)
    assert result["scope"] == "user_association"
    assert result["full_account_visibility"] is False
    assert result["allocation_guaranteed"] is False
    assert result["association_id"] == "7193"
    assert result["partition"] == ""
    assert (result["limit"], result["reserve"], result["account_occupancy"]) == (8, 2, 0)
    assert result["effective_uid"] == 4781
    assert capacity.content_digest(result["raw_observation"]) == result["snapshot_sha256"]
    for argv, kwargs in case["transport"].calls:
        assert Path(argv[0]).is_absolute()
        assert kwargs == {
            "environment": READ_ENV,
            "inherit_environment": False,
            "timeout_seconds": 8,
        }
        assert not any(arg.startswith(("--name", "--partition", "--jobs")) for arg in argv)
    assoc, queue, accounting = [entry[0] for entry in case["transport"].calls]
    assert {"cluster=n16r4", "account=scze856", "user=scze856"} <= set(assoc)
    assert {
        "--all",
        "--array",
        "--states=all",
        "--user=scze856",
        "--account=scze856",
        "--clusters=n16r4",
    } <= set(queue)
    assert {
        "--user=scze856",
        "--accounts=scze856",
        "--clusters=n16r4",
        "--allocations",
        "--duplicates",
        "--array",
    } <= set(accounting)


@pytest.mark.parametrize("phase", ["E1", "E2", "E3", "e0", "", None, True])
def test_research_and_unvalidated_phases_refused(case, phase):
    with pytest.raises(SlurmError, match="only.*E0"):
        validate(case, phase=phase)


@pytest.mark.parametrize("cap", [0, 3, 5, 7, 8, True, 1.0])
def test_invalid_stage_cap_refused(case, cap):
    with pytest.raises(SlurmError, match="stage cap"):
        validate(case, stage_cap=cap)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster", "elsewhere"),
        ("account", "other"),
        ("user", "other"),
        ("effective_uid", 999),
        ("effective_uid", True),
    ],
)
def test_snapshot_identity_does_not_grant_other_scope(case, field, value):
    with pytest.raises(SlurmError):
        validate(case, snapshot=replace(case["snapshot"], **{field: value}))


@pytest.mark.parametrize("age", [-1, 30.0001, 60])
def test_future_or_stale_snapshot_refused(case, age):
    with pytest.raises(SlurmError, match="stale|future"):
        validate(case, now=NOW + timedelta(seconds=age))


def test_thirty_second_boundary_is_inclusive(case):
    assert validate(case, now=NOW + timedelta(seconds=30))["reserve"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_at": NOW + timedelta(seconds=1)},
        {"completed_at": NOW - timedelta(seconds=1)},
        {"since": NOW + timedelta(seconds=1)},
    ],
)
def test_time_order_is_verified(case, changes):
    with pytest.raises(SlurmError):
        validate(case, snapshot=replace(case["snapshot"], **changes))


@pytest.mark.parametrize(
    "association",
    [
        "",
        ASSOC + ASSOC,
        ASSOC.replace("scze856|scze856", "else|scze856"),
        ASSOC.replace("n16r4", "other"),
        ASSOC.replace("scze856|||", "scze856|gpu||"),
        ASSOC.replace("|||8|||", "||||||"),
        ASSOC.replace("|||8|||", "|||UNLIMITED|||"),
        ASSOC.replace("|||8|||", "|||8.0|||"),
        ASSOC.replace("|7193|", "|unknown|"),
        ASSOC.rstrip() + "|extra\n",
    ],
)
def test_association_row_must_prove_exact_scope_and_numeric_max_submit(case, association):
    with pytest.raises(SlurmError):
        validate(case, snapshot=raw(case, association=association))


def test_gpu_or_maxjobs_is_never_used_as_submit_limit(case):
    fields = ASSOC.rstrip().split("|")
    fields[9], fields[10], fields[13] = "999", "3", "gres/gpu=999"
    result = validate(case, snapshot=raw(case, association="|".join(fields) + "\n"))
    assert result["limit"] == 3
    fields[10] = "2"
    with pytest.raises(SlurmError, match="no admission"):
        validate(case, snapshot=raw(case, association="|".join(fields) + "\n"))


@pytest.mark.parametrize(
    "row",
    [
        "123|wrong|scze856|job|RUNNING||2026-11-13T15:00:00|/dir\n",
        "123|scze856|other|job|RUNNING||2026-11-13T15:00:00|/dir\n",
        "123|scze856|scze856|job|UNKNOWN||2026-11-13T15:00:00|/dir\n",
        "123_[1-4]|scze856|scze856|job|RUNNING||2026-11-13T15:00:00|/dir\n",
        "123.batch|scze856|scze856|job|RUNNING||2026-11-13T15:00:00|/dir\n",
        "123|scze856|scze856|job|RUNNING||Unknown|/dir\n",
        "CLUSTER: other\n",
        "partial row\n",
    ],
)
def test_mixed_or_uncountable_queue_refused(case, row):
    with pytest.raises(SlurmError):
        validate(case, snapshot=raw(case, queue=row))


def test_sacct_cluster_is_parsed_not_assumed(case):
    row = scheduler_row(make_request(2), state="COMPLETED", source="sacct")
    with pytest.raises(SlurmError, match="cluster"):
        validate(case, snapshot=raw(case, accounting=row.replace("n16r4", "other")))


def test_unknown_reservations_include_other_campaigns_and_claimed(case):
    a, b = make_request(2), make_request(3, campaign="unrelated")
    case["ledger"]["intents"] = {
        a.intent_id: reservation(a),
        b.intent_id: reservation(b, status="claimed"),
    }
    result = validate(case, stage_cap=2)
    assert (result["account_occupancy"], result["campaign_occupancy"]) == (2, 1)
    with pytest.raises(SlurmError, match="no admission"):
        validate(case)


@pytest.mark.parametrize(
    "state", ["RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED by 4781"]
)
def test_matching_active_or_explicit_terminal_removes_unknown_double_count(case, state):
    old = make_request(2)
    case["ledger"]["intents"][old.intent_id] = reservation(old, job_id="123")
    row = scheduler_row(old, state=state, source="sacct")
    result = validate(case, snapshot=raw(case, accounting=row), stage_cap=2)
    assert result["account_occupancy"] == int(state in {"RUNNING", "PENDING"})


@pytest.mark.parametrize(
    "mismatch",
    [{"job_id": "456"}, {"work_dir": "/wrong"}, {"comment": "unrelated"}, {"name": "different"}],
)
def test_wrong_identity_terminal_cannot_release_unknown_reservation(case, mismatch):
    old = make_request(2)
    case["ledger"]["intents"][old.intent_id] = reservation(old, job_id="123")
    row = scheduler_row(old, state="COMPLETED", source="sacct", **mismatch)
    assert validate(case, snapshot=raw(case, accounting=row), stage_cap=2)["account_occupancy"] == 1


def test_queue_wins_lagged_accounting_without_doublecount(case):
    old = make_request(2)
    case["ledger"]["intents"][old.intent_id] = reservation(old)
    snap = raw(
        case,
        queue=scheduler_row(old),
        accounting=scheduler_row(old, state="COMPLETED", source="sacct"),
    )
    assert validate(case, snapshot=snap, stage_cap=2)["account_occupancy"] == 1
    bad = raw(
        case, queue=scheduler_row(old), accounting=scheduler_row(old, source="sacct", name="wrong")
    )
    with pytest.raises(SlurmError, match="conflicts"):
        validate(case, snapshot=bad, stage_cap=2)


def test_multiple_allocations_for_same_intent_fail_closed(case):
    old = make_request(2)
    case["ledger"]["intents"][old.intent_id] = reservation(old)
    with pytest.raises(SlurmError, match="multiple"):
        validate(
            case,
            snapshot=raw(case, queue=scheduler_row(old) + scheduler_row(old, "124")),
            stage_cap=6,
        )


@pytest.mark.parametrize("used,allowed", [(5, True), (6, False)])
def test_six_slot_cap_preserves_two_slots_for_all_user_campaigns(case, used, allowed):
    rows = "".join(
        scheduler_row(make_request(i + 2, campaign="external"), str(i + 100)) for i in range(used)
    )
    if allowed:
        assert validate(case, snapshot=raw(case, queue=rows), stage_cap=6)["account_occupancy"] == 5
    else:
        with pytest.raises(SlurmError, match="no admission"):
            validate(case, snapshot=raw(case, queue=rows), stage_cap=6)


@pytest.mark.parametrize("stage", [1, 2, 4, 6])
def test_stage_cap_is_independent_of_user_limit(case, stage):
    rows = "".join(scheduler_row(make_request(i + 2), str(i + 100)) for i in range(stage))
    with pytest.raises(SlurmError, match="no admission"):
        validate(case, snapshot=raw(case, queue=rows), stage_cap=stage)


def test_current_intent_reserved_before_go_is_not_counted_twice(case):
    current = case["request"]
    case["ledger"]["intents"][current.intent_id] = reservation(current)
    assert validate(case)["account_occupancy"] == 0
    with pytest.raises(SlurmError, match="already exists"):
        validate(case, snapshot=raw(case, queue=scheduler_row(current)))


def test_other_user_ledger_does_not_consume_this_per_user_limit(case):
    other = make_request(2, user="otheruser")
    case["ledger"]["intents"][other.intent_id] = reservation(other)
    assert validate(case)["account_occupancy"] == 0


@pytest.mark.parametrize("state", ["COMPLETED", "RUNNING"])
def test_cleanup_persistence_failure_keeps_slot_after_scheduler_terminal(case, state):
    old = make_request(2)
    entry = reservation(old, job_id="123")
    entry["cleanup_persistence_failed"] = True
    case["ledger"]["intents"][old.intent_id] = entry
    snap = raw(case, accounting=scheduler_row(old, state=state, source="sacct"))
    assert validate(case, snapshot=snap, stage_cap=2)["account_occupancy"] == 1
    with pytest.raises(SlurmError, match="no admission"):
        validate(case, snapshot=snap)


@pytest.mark.parametrize(
    "change",
    [{"schema_version": True}, {"cluster": "elsewhere"}, {"account": "other"}, {"intents": []}],
)
def test_invalid_ledger_scope_refused(case, change):
    case["ledger"].update(change)
    with pytest.raises(SlurmError, match="ledger"):
        validate(case)


def test_corrupt_reservation_refused(case):
    old = make_request(2)
    entry = reservation(old)
    entry["request_sha256"] = "f" * 64
    case["ledger"]["intents"][old.intent_id] = entry
    with pytest.raises(SlurmError, match="reservation"):
        validate(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 1),
        ("returncode", False),
        ("stderr", "permission denied"),
        ("stdout", None),
        ("argv", ("squeue", "--partition=gpu")),
    ],
)
def test_raw_command_shape_or_failure_cannot_grant_capacity(case, field, value):
    commands = list(case["snapshot"].commands)
    commands[1] = replace(commands[1], **{field: value})
    with pytest.raises(SlurmError):
        validate(case, snapshot=replace(case["snapshot"], commands=tuple(commands)))


def test_command_mutation_after_observation_denies_capacity(case):
    Path(case["reader"].commands["squeue"].path).write_text("changed", encoding="utf-8")
    with pytest.raises(SlurmError, match="pinned"):
        validate(case)


def test_reader_does_not_retry_failure(case):
    case["transport"].calls.clear()
    case["transport"].outputs["squeue"] = CommandResult(1, "", "denied")
    with pytest.raises(SlurmError):
        case["reader"].read()
    assert sum(Path(argv[0]).name == "squeue" for argv, _ in case["transport"].calls) == 1


def test_default_reader_uses_real_subprocess_boundary_with_no_inherited_environment(
    case, monkeypatch
):
    observed = []

    def fake_run(argv, **kwargs):
        observed.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(ASSOC if Path(argv[0]).name == "sacctmgr" else "").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(capacity.subprocess, "run", fake_run)
    case["reader"].transport = None
    case["reader"].read()
    assert len(observed) == 3
    assert all(kwargs["env"] == READ_ENV and kwargs["shell"] is False for _, kwargs in observed)


def test_real_identity_code_requires_bidirectional_pwd_and_euid(monkeypatch):
    # Injected OS observations exercise comparisons; this is not a Linux syscall test.
    monkeypatch.setattr(capacity.sys, "platform", "linux")
    monkeypatch.setattr(capacity.os, "geteuid", lambda: 4781, raising=False)
    user = SimpleNamespace(pw_uid=4781, pw_name="scze856")
    fake_pwd = SimpleNamespace(getpwuid=lambda uid: user, getpwnam=lambda name: user)
    monkeypatch.setitem(sys.modules, "pwd", fake_pwd)
    assert capacity._effective_identity("scze856") == 4781
    fake_pwd.getpwnam = lambda name: SimpleNamespace(pw_uid=999, pw_name="scze856")
    with pytest.raises(SlurmError, match="credentials"):
        capacity._effective_identity("scze856")


def test_reader_refuses_elapsed_or_changed_identity(case, monkeypatch):
    moments = iter((NOW, NOW + timedelta(seconds=31)))
    case["reader"].clock = lambda: next(moments)
    with pytest.raises(SlurmError, match="freshness"):
        case["reader"].read()
    case["reader"].clock = lambda: NOW
    users = iter((4781, 999))
    monkeypatch.setattr(capacity, "_effective_identity", lambda user: next(users))
    with pytest.raises(SlurmError, match="changed"):
        case["reader"].read()
