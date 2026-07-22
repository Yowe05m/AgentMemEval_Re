from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from agentmemeval.experiments import task8b_same_attempt_recovery as recovery


def _json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _sha_json(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRunner:
    REQUIRED_IDENTITY_FIELDS = (
        "code_sha",
        "prompt_sha256",
        "model_fingerprint",
        "embedding_fingerprint",
        "resolved_config_sha256",
        "schedule_sha256",
    )

    def __init__(self, failed_hash: str):
        self.failed_hash = failed_hash
        self._semantic_config = recovery.canonicalize_resolved_config_identity
        self.writes = []
        self.resume_calls = []

    sha256_json = staticmethod(_sha_json)
    task8b_embedding_fingerprint = staticmethod(_sha_json)

    def _last_state_and_hash(self, _path):
        return "failed", self.failed_hash

    def _write_json_atomic_new(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(value))
        self.writes.append(path.name)

    def _directory_file_manifest(self, root):
        return [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in sorted(item for item in root.rglob("*") if item.is_file())
        ]

    def _verify_task_receipt(self, *, marker_path, run_dir, task_id, config_sha256):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["task_id"] == task_id
        assert marker["config_sha256"] == config_sha256
        child = run_dir / marker["run_dir"]
        if marker["files"] != self._directory_file_manifest(child):
            raise recovery.RecoveryError("receipt files mismatch")
        return marker

    def run_worker_manifest(self, manifest_path, *, receipt_root, resume_existing):
        self.resume_calls.append((manifest_path, receipt_root, resume_existing))
        return {"status": "continued", "resumed": True}


def _state_row():
    body = {
        "schema_version": "task8-worker-state-v1",
        "created_at_utc": "2026-07-22T00:00:00Z",
        "status": "failed",
        "detail": "identity mismatch",
        "previous_sha256": "GENESIS",
    }
    return {**body, "row_sha256": _sha_json(body)}


def _write_baseline_and_archive(
    attempt: Path, baseline: Path, archive: Path, *, add_fifo: bool = False
):
    rows = []
    for path in sorted(item for item in attempt.rglob("*") if item.is_file()):
        rows.append(
            (path.relative_to(attempt).as_posix(), path.stat().st_size, _sha(path))
        )
    with baseline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("relative_path", "size", "sha256"))
        writer.writerows(rows)
    with tarfile.open(archive, "w:gz") as handle:
        for relative, _, _ in rows:
            content = (attempt / relative).read_bytes()
            info = tarfile.TarInfo(f"attempt/{relative}")
            info.size = len(content)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(content))
        if add_fifo:
            special = tarfile.TarInfo("attempt/unlisted_fifo")
            special.type = tarfile.FIFOTYPE
            handle.addfile(special)


def _fixture(tmp_path: Path, monkeypatch):
    attempt = tmp_path / "attempt"
    child = attempt / "runs" / recovery.TASK_ID
    child.mkdir(parents=True)
    controller = tmp_path / "controller.py"
    controller.write_text("# controller\n", encoding="utf-8")
    config_path = tmp_path / "task.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    config = {
        "experiment": {
            "train_hands": 1350,
            "checkpoint_interval": 30,
            "checkpoint_test_hands_by_checkpoint": {30: 1, 75: 1, 150: 1, 300: 1},
        },
        "agent": {},
    }
    dynamic = json.loads(json.dumps(config))
    # Restore the key type that YAML loading preserves and the frozen bundle hashed.
    dynamic["experiment"]["checkpoint_test_hands_by_checkpoint"] = {
        30: 1,
        75: 1,
        150: 1,
        300: 1,
    }
    expected_input = {
        "experiment": {
            **dynamic["experiment"],
            "seed": 2026090103,
            "heldout_table_set": ["T01"],
            "checkpoint_set": [30, 75, 150, 300],
        },
        "agent": {
            "embedding_cache_path": "task8b/P03/isolation_no_memory/without/{agent_id}.json"
        },
    }
    expected_input["experiment"].pop("checkpoint_interval", None)
    expected_config = recovery.canonicalize_resolved_config_identity(expected_input)
    prompts = {"system": "fixed"}
    model = {"model": "fixed"}
    embedding = {"model": "fixed-embedding"}
    schedule_sha = "a" * 64
    expected_identity = {
        "code_sha": recovery.FROZEN_CODE_SHA,
        "prompt_sha256": _sha_json(prompts),
        "model_fingerprint": _sha_json(model),
        "embedding_fingerprint": _sha_json(embedding),
        "resolved_config_sha256": _sha_json(expected_config),
        "schedule_sha256": schedule_sha,
    }
    manifest = {
        "worker_id": "P03",
        "role": "primary",
        "execution_mode": "experiment_configs",
        "protocol_status": "frozen/expedited-formal-candidate",
        "seed_bundle": 2026090103,
        "heldout_table_set": ["T01"],
        "checkpoint_set": [30, 75, 150, 300],
        "instance_identity": {"cache_namespace": "task8b/P03"},
        "task_configs": [
            {
                "task_id": recovery.TASK_ID,
                "planned_hands": 1350,
                "config_path": str(config_path),
                "config_sha256": _sha(config_path),
                "memory_mode": "Without",
                "expected_identity": expected_identity,
            }
        ],
    }
    manifest_path = tmp_path / "P03.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    (attempt / "worker_manifest.json").write_bytes(manifest_path.read_bytes())
    row = _state_row()
    with (attempt / "state.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    (child / "manifest.json").write_bytes(
        _json_bytes(
            {
                "metadata": {
                    "code": {"commit": recovery.FROZEN_CODE_SHA, "dirty": False},
                    "prompts": prompts,
                    "model": model,
                    "embedding": embedding,
                }
            }
        )
    )
    (child / "schedule_manifest.json").write_bytes(
        _json_bytes({"schedule_sha256": schedule_sha})
    )
    (child / "experiment_result.json").write_bytes(_json_bytes({"status": "complete"}))
    (child / "protocol_audit.json").write_bytes(
        _json_bytes(
            {
                "run_validity": {"execution_valid": True, "behavior_valid": True},
                "execution_health": {
                    "valid": True,
                    **{field: 0 for field in recovery.HEALTH_ZERO_FIELDS},
                },
            }
        )
    )
    (child / "hand_summaries.jsonl").write_bytes(b"{}\n" * 1350)
    baseline = tmp_path / "P03.files.tsv"
    archive = tmp_path / "P03.tar.gz"
    _write_baseline_and_archive(attempt, baseline, archive)
    amendment = tmp_path / "amendment.md"
    amendment.write_text("authorized amendment\n", encoding="utf-8")
    preunlock = tmp_path / "preunlock.json"
    preunlock.write_bytes(_json_bytes({"status": "locked"}))
    pid = tmp_path / "pid.json"
    pid.write_bytes(
        _json_bytes(
            {
                "active_process_absent": True,
                "observed_pid": 1234,
                "probe_utc": "2026-07-22T01:00:00Z",
            }
        )
    )
    verifier = tmp_path / "verifier"
    source = verifier / "src" / "agentmemeval" / "experiments" / "formal_protocol.py"
    source.parent.mkdir(parents=True)
    source.write_text("# verifier\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "_verify_clean_commit", lambda *_: None)
    authorization = {
        "schema_version": recovery.AUTHORIZATION_SCHEMA,
        "authorized": True,
        "authorization_id": "auth-P03",
        "worker_id": "P03",
        "task_id": recovery.TASK_ID,
        "reason": recovery.AUTHORIZED_REASON,
        "frozen_code_sha": recovery.FROZEN_CODE_SHA,
        "formal_runner_sha256": "1" * 64,
        "controller_sha256": _sha(controller),
        "worker_manifest_sha256": _sha(manifest_path),
        "baseline_manifest_sha256": _sha(baseline),
        "pre_recovery_archive_sha256": _sha(archive),
        "protocol_amendment_id": recovery.PROTOCOL_AMENDMENT_ID,
        "protocol_amendment_sha256": _sha(amendment),
        "parent_preunlock_sha256": _sha(preunlock),
        "active_pid_absence_evidence_sha256": _sha(pid),
        "verifier_code_sha": "2" * 40,
        "verifier_identity_source_sha256": _sha(source),
        "failed_state_row_sha256": row["row_sha256"],
        "attempt_root": str(attempt.resolve()),
        "active_process_absent_confirmed": True,
        "same_attempt_recovery_authorized": True,
        "task1_adoption_only": True,
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_bytes(_json_bytes(authorization))
    runner = FakeRunner(row["row_sha256"])
    kwargs = {
        "runner": runner,
        "load_config": lambda _path: json.loads(json.dumps(dynamic))
        | {"experiment": {**dynamic["experiment"]}},
        "manifest_path": manifest_path,
        "receipt_root": tmp_path / "receipts",
        "attempt_root": attempt,
        "baseline_path": baseline,
        "authorization_path": authorization_path,
        "controller_path": controller,
        "protocol_amendment_path": amendment,
        "parent_preunlock_path": preunlock,
        "archive_path": archive,
        "pid_absence_path": pid,
        "verifier_checkout": verifier,
        "protocol_amendment_id": recovery.PROTOCOL_AMENDMENT_ID,
    }
    return kwargs, runner, child


def test_adoption_is_receipt_last_and_idempotent(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    assert recovery.execute_recovery(**kwargs) == {
        "status": "continued",
        "resumed": True,
    }
    assert runner.writes[-1] == f"{recovery.TASK_ID}.json"
    assert runner.writes == [
        "task_identity_audit.json",
        f"{recovery.TASK_ID}.canonicalization_audit.json",
        f"{recovery.TASK_ID}.adoption_attestation.json",
        f"{recovery.TASK_ID}.json",
        f"{recovery.TASK_ID}.json",
    ]
    receipt_path = kwargs["attempt_root"] / "task_receipts" / f"{recovery.TASK_ID}.json"
    adoption_path = (
        kwargs["attempt_root"] / "recovery_adoptions" / f"{recovery.TASK_ID}.json"
    )
    receipt_before = receipt_path.read_bytes()
    adoption_before = adoption_path.read_bytes()
    certificate = json.loads(adoption_before)
    assert certificate["planned_hands"] == certificate["actual_hands"] == 1350
    correction = certificate["identity_correction"]
    assert (
        correction["legacy_json_roundtrip_actual_sha256"]
        != correction["original_expected_sha256"]
    )
    assert (
        correction["corrected_actual_sha256"] == correction["original_expected_sha256"]
    )
    assert correction["semantic_equivalence_after_stringifying_mapping_keys"] is True
    assert certificate["effect_fields_read"] is False
    assert certificate["task1_rerun_performed"] is False
    assert certificate["raw_artifact_bytes_modified"] is False
    assert certificate["scientific_outcome_fields_accessed"] is False
    receipt = json.loads(receipt_before)
    expected_phase_f = {
        "protocol_amendment_sha256",
        "verifier_code_sha",
        "pre_recovery_archive_sha256",
        "pre_recovery_file_manifest_sha256",
        "original_terminal_state_sha256",
        "original_expected_config_sha256",
        "corrected_config_sha256",
        "canonicalization_equivalence_audit_sha256",
        "recovery_certificate_sha256",
        "task1_adoption_attestation_sha256",
    }
    assert set(receipt["phase_f_evidence"]) == expected_phase_f

    state_path = kwargs["attempt_root"] / "state.tsv"
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    appended_body = {
        "schema_version": "task8-worker-state-v1",
        "created_at_utc": "2026-07-22T00:01:00Z",
        "status": "validating",
        "detail": "post-receipt resume",
        "previous_sha256": rows[-1]["row_sha256"],
    }
    appended = {**appended_body, "row_sha256": _sha_json(appended_body)}
    with state_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(appended), delimiter="\t", lineterminator="\n"
        )
        writer.writerow(appended)

    recovery.execute_recovery(**kwargs)
    assert receipt_path.read_bytes() == receipt_before
    assert adoption_path.read_bytes() == adoption_before
    assert len(runner.resume_calls) == 2
    assert (child / "task_identity_audit.json").is_file()


def test_rejects_baseline_tampering_before_any_publication(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    (child / "experiment_result.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(
        recovery.RecoveryError, match="baseline file integrity mismatch"
    ):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []
    assert not (kwargs["attempt_root"] / "task_receipts").exists()


def test_rejects_health_failure(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    # Re-seal archive/baseline so the failure is attributed to the health gate.
    audit = json.loads((child / "protocol_audit.json").read_text(encoding="utf-8"))
    audit["execution_health"]["fallback_count"] = 1
    (child / "protocol_audit.json").write_bytes(_json_bytes(audit))
    _write_baseline_and_archive(
        kwargs["attempt_root"], kwargs["baseline_path"], kwargs["archive_path"]
    )
    auth = json.loads(kwargs["authorization_path"].read_text(encoding="utf-8"))
    auth["baseline_manifest_sha256"] = _sha(kwargs["baseline_path"])
    auth["pre_recovery_archive_sha256"] = _sha(kwargs["archive_path"])
    kwargs["authorization_path"].write_bytes(_json_bytes(auth))
    with pytest.raises(recovery.RecoveryError, match="health counter is nonzero"):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []


def test_rejects_wrong_structural_hand_count(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    (child / "hand_summaries.jsonl").write_bytes(b"{}\n" * 1349)
    _write_baseline_and_archive(
        kwargs["attempt_root"], kwargs["baseline_path"], kwargs["archive_path"]
    )
    auth = json.loads(kwargs["authorization_path"].read_text(encoding="utf-8"))
    auth["baseline_manifest_sha256"] = _sha(kwargs["baseline_path"])
    auth["pre_recovery_archive_sha256"] = _sha(kwargs["archive_path"])
    kwargs["authorization_path"].write_bytes(_json_bytes(auth))
    with pytest.raises(recovery.RecoveryError, match="must both be 1350"):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []


def test_rejects_unarchived_extra_and_pid_evidence_tamper(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    (kwargs["attempt_root"] / "unarchived.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="unexpected pre-adoption files"):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []

    (kwargs["attempt_root"] / "unarchived.txt").unlink()
    kwargs["pid_absence_path"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="evidence SHA-256 mismatch"):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []


def test_rejects_special_tar_member(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    _write_baseline_and_archive(
        kwargs["attempt_root"],
        kwargs["baseline_path"],
        kwargs["archive_path"],
        add_fifo=True,
    )
    auth = json.loads(kwargs["authorization_path"].read_text(encoding="utf-8"))
    auth["pre_recovery_archive_sha256"] = _sha(kwargs["archive_path"])
    kwargs["authorization_path"].write_bytes(_json_bytes(auth))
    with pytest.raises(
        recovery.RecoveryError, match="only directories and regular files"
    ):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []


def test_existing_receipt_reexecutes_health_gate(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    recovery.execute_recovery(**kwargs)
    audit = json.loads((child / "protocol_audit.json").read_text(encoding="utf-8"))
    audit["execution_health"]["fallback_count"] = 1
    (child / "protocol_audit.json").write_bytes(_json_bytes(audit))
    _write_baseline_and_archive(
        kwargs["attempt_root"], kwargs["baseline_path"], kwargs["archive_path"]
    )
    auth = json.loads(kwargs["authorization_path"].read_text(encoding="utf-8"))
    auth["baseline_manifest_sha256"] = _sha(kwargs["baseline_path"])
    auth["pre_recovery_archive_sha256"] = _sha(kwargs["archive_path"])
    kwargs["authorization_path"].write_bytes(_json_bytes(auth))
    with pytest.raises(recovery.RecoveryError, match="health counter is nonzero"):
        recovery.execute_recovery(**kwargs)
    assert len(runner.resume_calls) == 1


def test_existing_receipt_reexecutes_hand_count_gate(tmp_path, monkeypatch):
    kwargs, runner, child = _fixture(tmp_path, monkeypatch)
    recovery.execute_recovery(**kwargs)
    (child / "hand_summaries.jsonl").write_bytes(b"{}\n" * 1349)
    _write_baseline_and_archive(
        kwargs["attempt_root"], kwargs["baseline_path"], kwargs["archive_path"]
    )
    auth = json.loads(kwargs["authorization_path"].read_text(encoding="utf-8"))
    auth["baseline_manifest_sha256"] = _sha(kwargs["baseline_path"])
    auth["pre_recovery_archive_sha256"] = _sha(kwargs["archive_path"])
    kwargs["authorization_path"].write_bytes(_json_bytes(auth))
    with pytest.raises(recovery.RecoveryError, match="must both be 1350"):
        recovery.execute_recovery(**kwargs)
    assert len(runner.resume_calls) == 1


def test_existing_receipt_byte_verifies_certificate(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    recovery.execute_recovery(**kwargs)
    certificate = (
        kwargs["attempt_root"] / "recovery_adoptions" / f"{recovery.TASK_ID}.json"
    )
    value = json.loads(certificate.read_text(encoding="utf-8"))
    value["actual_hands"] = 999
    certificate.write_bytes(_json_bytes(value))
    with pytest.raises(recovery.RecoveryError, match="not byte-identical"):
        recovery.execute_recovery(**kwargs)
    assert len(runner.resume_calls) == 1


def test_completed_worker_repeat_is_evidence_idempotent(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    recovery.execute_recovery(**kwargs)
    recovery_files = {
        path: path.read_bytes()
        for path in (
            kwargs["attempt_root"] / "recovery_adoptions"
        ).glob("*")
    }
    receipt_path = (
        kwargs["attempt_root"]
        / "task_receipts"
        / f"{recovery.TASK_ID}.json"
    )
    receipt_bytes = receipt_path.read_bytes()
    (kwargs["attempt_root"] / "completion_receipt.json").write_bytes(
        _json_bytes({"schema_version": "task8-worker-completion-v1", "status": "complete"})
    )
    recovery.execute_recovery(**kwargs)
    assert receipt_path.read_bytes() == receipt_bytes
    assert {path: path.read_bytes() for path in recovery_files} == recovery_files
    assert len(runner.resume_calls) == 2


def test_rejects_nonfrozen_protocol_amendment_id(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    kwargs["protocol_amendment_id"] = "WRONG"
    with pytest.raises(recovery.RecoveryError, match="protocol amendment id mismatch"):
        recovery.execute_recovery(**kwargs)
    assert runner.writes == []


def test_deterministic_draft_activate_execute_chain(tmp_path, monkeypatch):
    kwargs, runner, _child = _fixture(tmp_path, monkeypatch)
    frozen = tmp_path / "frozen"
    frozen_runner = frozen / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    frozen_runner.parent.mkdir(parents=True)
    frozen_runner.write_text("# frozen runner\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "_verify_checkout", lambda *_: frozen_runner)
    monkeypatch.setattr(recovery, "_git_clean_head", lambda *_: "2" * 40)
    draft = recovery.build_authorization_draft(
        manifest_path=kwargs["manifest_path"],
        attempt_root=kwargs["attempt_root"],
        baseline_path=kwargs["baseline_path"],
        archive_path=kwargs["archive_path"],
        protocol_amendment_path=kwargs["protocol_amendment_path"],
        protocol_amendment_id=kwargs["protocol_amendment_id"],
        parent_preunlock_path=kwargs["parent_preunlock_path"],
        pid_absence_path=kwargs["pid_absence_path"],
        frozen_checkout=frozen,
        verifier_checkout=kwargs["verifier_checkout"],
        controller_path=kwargs["controller_path"],
    )
    draft_path = tmp_path / "draft.json"
    draft_path.write_bytes(_json_bytes(draft))
    activated_path = tmp_path / "activated.json"
    activated = recovery.activate_authorization_draft(
        draft_path=draft_path,
        rebuilt_draft=draft,
        output_path=activated_path,
    )
    assert activated["authorized"] is True
    assert activated["same_attempt_recovery_authorized"] is True
    assert activated["task1_adoption_only"] is True
    kwargs["authorization_path"] = activated_path
    assert recovery.execute_recovery(**kwargs)["status"] == "continued"
    assert len(runner.resume_calls) == 1


def test_main_executes_frozen_runner_from_frozen_checkout_and_restores_cwd(
    tmp_path, monkeypatch, capsys
):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(_json_bytes({"formal_runner_sha256": "1" * 64}))
    original_cwd = Path.cwd()
    observed = {}
    runner = type("Runner", (), {"_semantic_config": staticmethod(lambda value: value)})()
    original_canonicalizer = runner._semantic_config

    monkeypatch.setattr(
        recovery,
        "_parse_args",
        lambda _argv: Namespace(
            frozen_checkout=frozen,
            manifest=tmp_path / "manifest.json",
            receipt_root=tmp_path / "receipts",
            attempt_root=tmp_path / "attempt",
            baseline_manifest=tmp_path / "baseline.tsv",
            authorization=authorization,
            build_authorization=None,
            activate_authorization=None,
            activated_authorization_output=None,
            protocol_amendment=tmp_path / "amendment.json",
            protocol_amendment_id=recovery.PROTOCOL_AMENDMENT_ID,
            parent_preunlock=tmp_path / "preunlock.json",
            pre_recovery_archive=tmp_path / "archive.tar.gz",
            pid_absence_evidence=tmp_path / "pid.json",
            verifier_checkout=tmp_path / "verifier",
        ),
    )
    monkeypatch.setattr(recovery, "_load_frozen_runner", lambda *_: (runner, object()))

    def fake_execute_recovery(**_kwargs):
        observed["cwd"] = Path.cwd()
        return {"status": "continued"}

    monkeypatch.setattr(recovery, "execute_recovery", fake_execute_recovery)

    assert recovery.main([]) == 0
    assert observed["cwd"] == frozen.resolve()
    assert Path.cwd() == original_cwd
    assert runner._semantic_config == original_canonicalizer
    assert json.loads(capsys.readouterr().out) == {"status": "continued"}
