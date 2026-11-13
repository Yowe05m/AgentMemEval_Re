"""Offline closure faults; synthetic raw records do not attest Linux, GPU or science."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from agentmemeval.path1_v1.operations import closure, worker
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.process_ownership import (
    CleanupReceipt,
    ProcessBinding,
    ProcessIdentity,
)
from agentmemeval.path1_v1.operations.slurm import (
    CommandResult,
    JobRecord,
    SubmissionRequest,
    SubmitSpec,
    job_comment,
    job_name,
)
from path1_v1.unit.test_operations_worker import OfflineVerifier, make_worker_fixture, write_json


def _raw(path, count):
    path.write_text('{"offline_fixture":true}\n' * count, encoding="utf-8")


class FakeOwner:
    def __init__(self, receipt):
        self.receipt = receipt

    def verify_cleanup(self, receipt):
        return receipt is self.receipt and receipt.group_cleanup_verified and receipt.leader_reaped


class SiteFixture:
    def __init__(self, request, job, now):
        self.request, self.job, self.now = request, job, now
        self.queue = ""
        self.steps_queue = ""
        self.state = "COMPLETED"
        self.batch = True
        self.warn = ""
        self.extra = ""
        self.calls = []

    def run(self, argv, *, environment, inherit_environment, timeout_seconds):
        assert environment == closure.READ_ENV and not inherit_environment and timeout_seconds == 15
        self.calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(0, self.steps_queue if "--steps" in argv else self.queue)
        assert argv[0] == "sacct" and "--jobs=12345" in argv
        job = self.job
        end = (self.now - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
        row = "|".join(
            (
                job.job_id,
                job.user,
                job.account,
                job.name,
                self.state,
                job.comment,
                job.submitted_at,
                job.work_dir,
                "0:0",
                end,
            )
        )
        if self.batch:
            row += "\n" + "|".join(
                (
                    "12345.batch",
                    job.user,
                    job.account,
                    "batch",
                    self.state,
                    "",
                    job.submitted_at,
                    "",
                    "0:0",
                    end,
                )
            )
        return CommandResult(0, row + self.extra, self.warn)


def make_closed_fixture(tmp_path, monkeypatch, *, before_seal=None):
    data = make_worker_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    contract = data["contract"]

    def offline_execute(request, factory):
        del factory
        payload = request.output_directory
        source, job = contract["source_contract"], contract["job"]
        count = source["technical"]["formation_episodes"]
        formation_hands = source["technical"]["formation_episode_hands"]
        evaluation_hands = source["technical"]["evaluation_episode_hands"]
        units = []
        for unit_name in ("formation", *job["pair_ids"]):
            root = payload / unit_name
            root.mkdir()
            if unit_name == "formation":
                episodes = [(i, "formation", "formation", formation_hands) for i in range(count)]
                write_json(root / "artifact_snapshot.json", {"offline_fixture": True})
            else:
                index = job["pair_ids"].index(unit_name)
                episodes = [
                    (index, arm, "evaluation", evaluation_hands) for arm in ("retained", "removed")
                ]
            receipt = {
                "status": "complete",
                "completed_episodes": len(episodes),
                "settled_hands": sum(row[3] for row in episodes),
            }
            write_json(root / "terminal_receipt.json", receipt)
            write_json(root / "path1_manifest.json", {"offline_fixture": True})
            write_json(root / "seed_manifest.json", {"offline_fixture": True})
            _raw(root / "settled_hands.jsonl", receipt["settled_hands"])
            for index, arm, stage, hands in episodes:
                episode = root / f"{stage}_{index:04d}_{arm}"
                episode.mkdir()
                write_json(
                    episode / "episode_receipt.json",
                    {
                        "status": "complete",
                        "hands": hands,
                        "episode_index": index,
                        "arm": arm,
                        "stage": stage,
                        "campaign_id": contract["campaign_id"],
                        "phase": "E1",
                        "artifact_id": job["artifact_id"],
                    },
                )
                for name in ("initial_snapshot.json", "final_snapshot.json"):
                    write_json(episode / name, {"offline_fixture": True})
                (episode / "resolved_config.yaml").write_text(
                    "offline_fixture: true\n", encoding="utf-8"
                )
                for name in ("events.jsonl", "provider_exchanges.jsonl"):
                    _raw(episode / name, 1)
                _raw(episode / "hand_summaries.jsonl", hands)
            item = {"unit": unit_name, "receipt": receipt}
            if unit_name != "formation":
                item["episode_index"] = job["pair_ids"].index(unit_name)
            units.append(item)
        if before_seal:
            before_seal(payload, units)
        return {"completed_units": units}

    monkeypatch.setattr(worker, "_execute", offline_execute)
    terminal = worker.run_worker(data["request_path"], runtime_verifier=OfflineVerifier(tmp_path))
    submission = SubmissionRequest(
        contract["campaign_id"],
        data["request"]["intent_id"],
        SubmitSpec(
            "fixture", "account", "user", "gpu", "/jobs/fixture", "/launch/worker.sh", "a" * 64
        ),
    )
    now = datetime.now(UTC)
    job = JobRecord(
        "12345",
        "user",
        "account",
        job_name(submission.campaign_id, submission.intent_id),
        "COMPLETED",
        job_comment(submission.campaign_id, submission.intent_id),
        (now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        submission.spec.work_dir,
        "sacct",
    )
    paths = closure.ClosurePaths(
        data["request_path"],
        data["request"]["request_sha256"],
        data["store"].directory,
        data["output"].parent,
        tmp_path / "bundle",
        tmp_path / "local",
    )
    process_binding = ProcessBinding(
        submission.campaign_id,
        submission.intent_id,
        "12345",
        content_digest(contract["runtime_identity"]),
        "model_service",
        "offline-test-service",
    )
    cleanup = CleanupReceipt(
        process_binding,
        ProcessIdentity(
            "fixture-host",
            "fixture-boot",
            1000,
            50,
            "123",
            50,
            50,
            ("0::/slurm/job_12345",),
            "12345",
            "batch",
        ),
        0,
        False,
        False,
        True,
        True,
        (),
        (),
    )
    owner = FakeOwner(cleanup)
    site = SiteFixture(submission, job, now)
    ticks = [0.0]

    def sleep(seconds):
        ticks[0] += seconds

    verifier = closure.FileClosureVerifier(
        lambda request: paths,
        scheduler=closure.SchedulerClosureReader(site, clock=lambda: now),
        service=closure.OwnedServiceReleaseVerifier(owner, cleanup),
        clock=lambda: now,
        monotonic=lambda: ticks[0],
        sleep=sleep,
    )
    return {
        **data,
        "terminal": terminal,
        "submission": submission,
        "job": asdict(job),
        "paths": paths,
        "site": site,
        "owner": owner,
        "cleanup": cleanup,
        "verifier": verifier,
        "ticks": ticks,
    }


def test_real_files_archive_extract_and_repeat_full_hash_validation(tmp_path, monkeypatch):
    data = make_closed_fixture(tmp_path, monkeypatch)
    verifier = data["verifier"]
    receipt = verifier(data["submission"], data["job"])
    assert receipt is not None, verifier.last_failure
    receipt.validate(data["submission"])
    recovery = json.loads((data["paths"].local_directory / "recovery_receipt.json").read_text())
    assert recovery["scientific_status"] == "not_assessed"
    assert recovery["primary_analysis_admitted"] is False
    assert len(recovery["supporting_inputs"]) == 8
    assert verifier(data["submission"], data["job"]) == receipt, verifier.last_failure
    assert data["ticks"][0] == 20.0
    assert all(call[0] in {"sacct", "squeue"} for call in data["site"].calls)


def test_default_no_producers_does_not_even_resolve_or_write(tmp_path):
    def forbidden(_request):
        raise AssertionError("default gate must fail before inspecting paths")

    verifier = closure.FileClosureVerifier(forbidden)
    assert verifier(None, {}) is None
    assert "producers" in verifier.last_failure
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "problem",
    [
        "raw_deleted",
        "raw_changed",
        "raw_added",
        "terminal_changed",
        "probe_deleted",
        "seed_changed",
        "incomplete_claim",
    ],
)
def test_changed_or_missing_evidence_never_closes(tmp_path, monkeypatch, problem):
    data = make_closed_fixture(tmp_path, monkeypatch)
    path = data["output"] / "formation/formation_0000_formation/hand_summaries.jsonl"
    if problem == "raw_deleted":
        path.unlink()
    elif problem == "raw_changed":
        path.write_text("changed", encoding="utf-8")
    elif problem == "raw_added":
        (data["output"] / "unexpected.txt").write_text("added", encoding="utf-8")
    elif problem == "terminal_changed":
        value = data["terminal"] | {"status": "incomplete"}
        write_json(data["output"] / "worker_terminal.json", value)
    elif problem == "probe_deleted":
        (tmp_path / "fake_probe_0.json").unlink()
    elif problem == "seed_changed":
        (tmp_path / "seeds.json").write_text("{}", encoding="utf-8")
    else:
        snapshot = data["store"].read_registry()
        monkeypatch.setattr(
            data["store"].__class__,
            "read_registry",
            lambda self: replace(snapshot, claims=(replace(snapshot.claims[0], status="started"),)),
        )
    assert data["verifier"](data["submission"], data["job"]) is None


@pytest.mark.parametrize("problem", ["missing_raw", "wrong_pair_index", "wrong_episode_id"])
def test_sealed_but_structurally_incomplete_worker_is_rejected(tmp_path, monkeypatch, problem):
    def before_seal(payload, units):
        if problem == "missing_raw":
            (payload / "formation/formation_0000_formation/hand_summaries.jsonl").unlink()
        elif problem == "wrong_pair_index":
            units[1]["episode_index"] = 1
        else:
            path = payload / "formation/formation_0000_formation/episode_receipt.json"
            value = json.loads(path.read_text())
            value["artifact_id"] = "wrong"
            write_json(path, value)

    data = make_closed_fixture(tmp_path, monkeypatch, before_seal=before_seal)
    assert data["verifier"](data["submission"], data["job"]) is None
    assert not data["paths"].bundle_directory.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "queue",
        "step_queue",
        "nonterminal",
        "missing_batch",
        "warning",
        "foreign_step",
        "copied_cleanup",
        "json_true",
        "wrong_service",
    ],
)
def test_release_requires_all_three_exact_evidence_sides(tmp_path, monkeypatch, problem):
    data = make_closed_fixture(tmp_path, monkeypatch)
    site, verifier = data["site"], data["verifier"]
    if problem == "queue":
        site.queue = "12345|RUNNING\n"
    elif problem == "step_queue":
        site.steps_queue = "12345.0|RUNNING\n"
    elif problem == "nonterminal":
        site.state = "RUNNING"
    elif problem == "missing_batch":
        site.batch = False
    elif problem == "warning":
        site.warn = "accounting unavailable"
    elif problem == "foreign_step":
        site.extra = "\n99999.batch|user|account|batch|COMPLETED|||/|0:0|2026-11-13T00:00:00"
    elif problem == "copied_cleanup":
        verifier.service.receipt = replace(data["cleanup"])
    elif problem == "wrong_service":
        bad = replace(
            data["cleanup"], binding=replace(data["cleanup"].binding, service_instance_id="other")
        )
        data["owner"].receipt = bad
        verifier.service.receipt = bad
    else:
        monkeypatch.setattr(verifier.service, "verify", lambda binding: {"verified": True})
    assert verifier(data["submission"], data["job"]) is None
    assert not data["paths"].bundle_directory.exists()


@pytest.mark.parametrize("problem", ["too_short", "changed_during_wait", "partial"])
def test_quiet_and_partial_output_gate(tmp_path, monkeypatch, problem):
    data = make_closed_fixture(tmp_path, monkeypatch)
    verifier = data["verifier"]
    if problem == "too_short":
        verifier.sleep = lambda seconds: None
    elif problem == "partial":
        data["paths"].bundle_directory.mkdir()
        (data["paths"].bundle_directory / "partial.txt").write_text("partial", encoding="utf-8")
    else:

        def changed(seconds):
            data["ticks"][0] += seconds
            (data["output"].parent / "slurm-12345.out").write_text("late log", encoding="utf-8")

        verifier.sleep = changed
    assert verifier(data["submission"], data["job"]) is None


@pytest.mark.parametrize(
    "relative",
    [
        "snapshot.tar.gz",
        "snapshot.files.tsv",
        "quiet_receipt.json",
        "evidence_inputs/contract.json",
        "recovery_receipt.json",
    ],
)
def test_existing_receipts_do_not_bypass_recovery_hashes(tmp_path, monkeypatch, relative):
    data = make_closed_fixture(tmp_path, monkeypatch)
    verifier = data["verifier"]
    assert verifier(data["submission"], data["job"]) is not None, verifier.last_failure
    (data["paths"].local_directory / relative).write_bytes(b"tampered")
    assert verifier(data["submission"], data["job"]) is None


def test_recovered_raw_loss_and_source_unchanged_are_distinct(tmp_path, monkeypatch):
    data = make_closed_fixture(tmp_path, monkeypatch)
    verifier = data["verifier"]
    assert verifier(data["submission"], data["job"]) is not None, verifier.last_failure
    extracted = data["paths"].local_directory / "extracted" / data["paths"].source_root.name
    (extracted / "payload/formation/formation_0000_formation/hand_summaries.jsonl").unlink()
    assert verifier(data["submission"], data["job"]) is None
    assert "recovered artifacts" in verifier.last_failure


@pytest.mark.parametrize("problem", ["duplicate", "copy_not_same_inode"])
def test_terminal_publication_exception_is_narrow(tmp_path, monkeypatch, problem):
    data = make_closed_fixture(tmp_path, monkeypatch)
    staging = data["output"] / "_publication_staging"
    original = next(staging.glob("worker_terminal.json.*.pending"))
    if problem == "duplicate":
        shutil.copyfile(original, staging / ("worker_terminal.json." + "f" * 32 + ".pending"))
    else:
        original.unlink()
        shutil.copyfile(data["output"] / "worker_terminal.json", original)
    assert data["verifier"](data["submission"], data["job"]) is None
    assert "publication copy" in data["verifier"].last_failure
