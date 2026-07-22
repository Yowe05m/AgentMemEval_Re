from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agentmemeval.experiments import task8b_p12_fresh_run as fresh
from agentmemeval.experiments.task8b_same_attempt_recovery import (
    canonicalize_resolved_config_identity as audited_recovery_canonicalizer,
)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    tasks = []
    for index, task_id in enumerate(fresh.TASK_IDS):
        tasks.append(
            {
                "task_id": task_id,
                "planned_hands": fresh.HANDS_PER_TASK,
                "publish_checkpoint_after": index == len(fresh.TASK_IDS) - 1,
            }
        )
    return {
        "schema_version": "task8-worker-manifest-v1",
        "protocol_status": "frozen/expedited-formal-candidate",
        "execution_mode": "experiment_configs",
        "worker_id": fresh.WORKER_ID,
        "role": fresh.ROLE,
        "pod_id": fresh.POD_ID,
        "seed_bundle": fresh.SEED,
        "depends_on": None,
        "receipt_relative_path": fresh.RECEIPT_RELATIVE,
        "common_identity": {"code_sha": fresh.FROZEN_CODE_SHA},
        "instance_identity": {
            "worker_id": fresh.WORKER_ID,
            "output_path": fresh.OUTPUT_RELATIVE,
            "cache_namespace": fresh.CACHE_RELATIVE,
        },
        "task_configs": tasks,
    }


def _fixture(tmp_path, monkeypatch):
    manifest_path = tmp_path / "P12.json"
    manifest_path.write_bytes(_json_bytes(_manifest()))
    monkeypatch.setattr(fresh, "P12_MANIFEST_SHA256", _sha(manifest_path))

    frozen = tmp_path / "frozen"
    runner_path = frozen / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# frozen runner fixture\n", encoding="utf-8")
    monkeypatch.setattr(fresh, "_verify_frozen_checkout", lambda _path: runner_path)

    verifier = Path(fresh.__file__).resolve().parents[3]
    monkeypatch.setattr(
        fresh,
        "_git_head_and_tracked_status",
        lambda _path: ("2" * 40, ""),
    )
    monkeypatch.setattr(fresh, "_require_tracked_file", lambda *_args: None)

    absence = tmp_path / "process_absence.json"
    absence.write_bytes(
        _json_bytes(
            {
                "schema_version": fresh.PROCESS_ABSENCE_SCHEMA,
                "worker_id": fresh.WORKER_ID,
                "active_process_absent_confirmed": True,
                "formal_worker_count": 0,
                "checked_at_utc": "2026-07-23T00:00:00Z",
                "host_id": "AEM12",
            }
        )
    )
    kwargs = {
        "manifest_path": manifest_path,
        "receipt_root": tmp_path / "control",
        "frozen_checkout": frozen,
        "verifier_checkout": verifier,
        "controller_path": Path(fresh.__file__).resolve(),
        "process_absence_path": absence,
        "control_attempt": 1,
    }
    return kwargs


def _activated(tmp_path, kwargs):
    draft = fresh.build_release(**kwargs)
    suffix = f"{int(kwargs['control_attempt']):04d}"
    draft_path = tmp_path / f"draft_{suffix}.json"
    draft_path.write_bytes(_json_bytes(draft))
    active_path = tmp_path / f"active_{suffix}.json"
    active = fresh.activate_release(
        draft_path=draft_path,
        rebuilt_draft=draft,
        output_path=active_path,
    )
    return draft, active, active_path


def test_deterministic_release_chain_binds_fresh_only_policy(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    first = fresh.build_release(**kwargs)
    second = fresh.build_release(**kwargs)
    assert _json_bytes(first) == _json_bytes(second)
    assert first["active"] is False
    assert first["worker_manifest_sha256"] == fresh.P12_MANIFEST_SHA256
    assert first["formal_runner_sha256"] == fresh.FROZEN_RUNNER_SHA256
    assert first["resume_existing"] is False
    assert first["historical_adoption_allowed"] is False
    assert first["same_attempt_recovery_allowed"] is False
    assert first["task_receipt_adoption_allowed"] is False
    assert first["receipt_publication_policy"] == "frozen-runner-task-receipt-last"
    assert first["control_attempt_policy"] == "append-only-sequential"
    assert first["control_attempt"] == 1
    assert first["previous_launch_terminal_sha256"] == "GENESIS"
    assert first["task_ids"] == list(fresh.TASK_IDS)
    assert first["total_hands"] == fresh.TOTAL_HANDS

    draft, active, _active_path = _activated(tmp_path, kwargs)
    assert active["active"] is True
    assert active["release_id"] == draft["release_id"]
    assert active["activated_from_sha256"] == hashlib.sha256(_json_bytes(draft)).hexdigest()


def test_embedded_canonicalizer_matches_audited_recovery_identity():
    config = {
        "_config_path": "dynamic",
        "experiment": {
            "run_id": "dynamic",
            "output_root": "dynamic",
            "checkpoint_test_hands_by_checkpoint": {30: 1, 75: 2},
        },
        "agent": {"embedding_cache_path": "dynamic", "nested": {1: "kept"}},
    }
    actual = fresh.canonicalize_resolved_config_identity(config)
    assert actual == audited_recovery_canonicalizer(config)
    assert 30 in actual["experiment"]["checkpoint_test_hands_by_checkpoint"]
    assert 1 in actual["agent"]["nested"]


@pytest.mark.parametrize("mutation", ["task_order", "hands", "worker", "output"])
def test_rejects_non_p12_manifest_topology(tmp_path, monkeypatch, mutation):
    kwargs = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(kwargs["manifest_path"].read_text(encoding="utf-8"))
    if mutation == "task_order":
        manifest["task_configs"][0], manifest["task_configs"][1] = (
            manifest["task_configs"][1],
            manifest["task_configs"][0],
        )
    elif mutation == "hands":
        manifest["task_configs"][0]["planned_hands"] = 1349
    elif mutation == "worker":
        manifest["worker_id"] = "P11"
    else:
        manifest["instance_identity"]["output_path"] += "_wrong"
    kwargs["manifest_path"].write_bytes(_json_bytes(manifest))
    monkeypatch.setattr(fresh, "P12_MANIFEST_SHA256", _sha(kwargs["manifest_path"]))
    with pytest.raises(fresh.FreshRunError, match="P12 manifest"):
        fresh.build_release(**kwargs)


@pytest.mark.parametrize("collision", ["output", "cache", "receipt", "attempt"])
def test_rejects_any_historical_path_collision(tmp_path, monkeypatch, collision):
    kwargs = _fixture(tmp_path, monkeypatch)
    manifest = _manifest()
    paths = fresh._bound_paths(
        manifest=manifest,
        frozen_checkout=kwargs["frozen_checkout"],
        receipt_root=kwargs["receipt_root"],
    )
    if collision == "attempt":
        target = paths["output_path"].with_name(f"{paths['output_path'].name}__attempt_02")
    else:
        target = paths[f"{collision}_path"]
    target.mkdir(parents=True)
    with pytest.raises(fresh.FreshRunError, match="fresh-run"):
        fresh.build_release(**kwargs)


def test_activation_rejects_nonidentical_draft(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    draft = fresh.build_release(**kwargs)
    draft_path = tmp_path / "draft.json"
    tampered = dict(draft)
    tampered["historical_adoption_allowed"] = True
    draft_path.write_bytes(_json_bytes(tampered))
    with pytest.raises(fresh.FreshRunError, match="byte-identical"):
        fresh.activate_release(
            draft_path=draft_path,
            rebuilt_draft=draft,
            output_path=tmp_path / "active.json",
        )


def test_active_release_tamper_fails_before_output_reservation(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft, active, active_path = _activated(tmp_path, kwargs)
    active["same_attempt_recovery_allowed"] = True
    active_path.write_bytes(_json_bytes(active))
    runner = type("Runner", (), {"_semantic_config": staticmethod(lambda value: value)})()
    with pytest.raises(fresh.FreshRunError, match="activated release"):
        fresh.execute_fresh_run(runner=runner, release_path=active_path, **kwargs)
    output = kwargs["frozen_checkout"] / fresh.OUTPUT_RELATIVE
    assert not output.exists()


def test_execute_uses_corrected_identity_fresh_attempt_and_restores_state(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft, _active, active_path = _activated(tmp_path, kwargs)
    output = (kwargs["frozen_checkout"] / fresh.OUTPUT_RELATIVE).resolve()
    cache = (kwargs["frozen_checkout"] / fresh.CACHE_RELATIVE).resolve()
    receipt = (kwargs["receipt_root"] / fresh.RECEIPT_RELATIVE).resolve()
    original_cwd = Path.cwd()
    observed = {}

    def legacy(value):
        return value

    class Runner:
        _semantic_config = staticmethod(legacy)

        def run_worker_manifest(self, manifest_path, *, receipt_root, resume_existing):
            observed["cwd"] = Path.cwd()
            observed["manifest"] = manifest_path
            observed["receipt_root"] = receipt_root
            observed["resume_existing"] = resume_existing
            observed["canonicalizer"] = self._semantic_config
            assert output.is_dir() and not any(output.iterdir())
            assert not cache.exists()
            assert not receipt.exists()
            canonical = self._semantic_config(
                {
                    "experiment": {"checkpoint_test_hands_by_checkpoint": {30: 1}},
                    "agent": {},
                }
            )
            assert 30 in canonical["experiment"]["checkpoint_test_hands_by_checkpoint"]
            return {"status": "complete", "run_dir": str(output), "resumed": False}

    runner = Runner()
    result = fresh.execute_fresh_run(runner=runner, release_path=active_path, **kwargs)
    assert result["status"] == "complete"
    assert observed["cwd"] == kwargs["frozen_checkout"].resolve()
    assert observed["manifest"] == kwargs["manifest_path"].resolve()
    assert observed["receipt_root"] == kwargs["receipt_root"].resolve()
    assert observed["resume_existing"] is False
    assert observed["canonicalizer"] == fresh.canonicalize_resolved_config_identity
    assert runner._semantic_config == legacy
    assert Path.cwd() == original_cwd
    ledger = Path(_active["launch_ledger_root"])
    claim = fresh._launch_files(ledger, 1)["claim"]
    completed = fresh._launch_files(ledger, 1)["completed"]
    assert claim.is_file()
    assert completed.is_file()
    assert not fresh._launch_files(ledger, 1)["failed"].exists()
    assert json.loads(completed.read_text(encoding="utf-8"))["claim_sha256"] == _sha(claim)


@pytest.mark.parametrize(
    "field,value",
    [
        ("formal_worker_count", 1),
        ("active_process_absent_confirmed", False),
        ("worker_id", "P11"),
    ],
)
def test_rejects_process_absence_evidence_mismatch(tmp_path, monkeypatch, field, value):
    kwargs = _fixture(tmp_path, monkeypatch)
    evidence = json.loads(kwargs["process_absence_path"].read_text(encoding="utf-8"))
    evidence[field] = value
    kwargs["process_absence_path"].write_bytes(_json_bytes(evidence))
    with pytest.raises(fresh.FreshRunError, match="process-absence evidence"):
        fresh.build_release(**kwargs)


def test_git_status_reports_real_untracked_file(tmp_path):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    commands = (
        ("init",),
        ("config", "user.email", "p12-test@example.invalid"),
        ("config", "user.name", "P12 Test"),
    )
    for command in commands:
        subprocess.run(["git", *command], cwd=checkout, check=True, capture_output=True)
    tracked = checkout / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    (checkout / "rogue.txt").write_text("untracked\n", encoding="utf-8")
    _head, dirty = fresh._git_head_and_tracked_status(checkout)
    assert "?? rogue.txt" in dirty


def test_frozen_checkout_rejects_simulated_untracked_file(tmp_path, monkeypatch):
    checkout = tmp_path / "frozen"
    runner = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# runner\n", encoding="utf-8")
    monkeypatch.setattr(fresh, "FROZEN_RUNNER_SHA256", _sha(runner))
    monkeypatch.setattr(
        fresh,
        "_git_head_and_tracked_status",
        lambda _path: (fresh.FROZEN_CODE_SHA, "?? rogue.txt"),
    )
    with pytest.raises(fresh.FreshRunError, match="untracked"):
        fresh._verify_frozen_checkout(checkout)


def test_verifier_checkout_rejects_simulated_untracked_file(monkeypatch):
    controller = Path(fresh.__file__).resolve()
    checkout = controller.parents[3]
    monkeypatch.setattr(fresh, "_require_tracked_file", lambda *_args: None)
    monkeypatch.setattr(
        fresh,
        "_git_head_and_tracked_status",
        lambda _path: ("2" * 40, "?? rogue.txt"),
    )
    with pytest.raises(fresh.FreshRunError, match="untracked"):
        fresh._verifier_identity(checkout, controller)


def test_empty_failed_launch_can_retry_with_new_control_attempt(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft1, active1, active_path1 = _activated(tmp_path, kwargs)
    output = Path(active1["output_path"])

    class FailingRunner:
        _semantic_config = staticmethod(lambda value: value)

        def run_worker_manifest(self, *_args, **kwargs):
            assert kwargs["resume_existing"] is False
            raise RuntimeError("fixture launch failure")

    with pytest.raises(RuntimeError, match="fixture launch failure"):
        fresh.execute_fresh_run(runner=FailingRunner(), release_path=active_path1, **kwargs)
    ledger = Path(active1["launch_ledger_root"])
    claim1 = fresh._launch_files(ledger, 1)["claim"]
    failed1 = fresh._launch_files(ledger, 1)["failed"]
    assert claim1.is_file() and failed1.is_file()
    failed_body = json.loads(failed1.read_text(encoding="utf-8"))
    assert failed_body["safe_retry_allowed"] is True
    assert failed_body["isolation_required"] is False
    assert failed_body["claim_sha256"] == _sha(claim1)
    assert output.is_dir() and not any(output.iterdir())

    retry_kwargs = dict(kwargs)
    retry_kwargs["control_attempt"] = 2
    _draft2, active2, active_path2 = _activated(tmp_path, retry_kwargs)
    assert active2["previous_launch_terminal_sha256"] == _sha(failed1)

    class SuccessfulRunner:
        _semantic_config = staticmethod(lambda value: value)

        def run_worker_manifest(self, *_args, **kwargs):
            assert kwargs["resume_existing"] is False
            assert output.is_dir() and not any(output.iterdir())
            return {"status": "complete", "run_dir": str(output), "resumed": False}

    result = fresh.execute_fresh_run(
        runner=SuccessfulRunner(), release_path=active_path2, **retry_kwargs
    )
    assert result["status"] == "complete"
    assert fresh._launch_files(ledger, 2)["claim"].is_file()
    assert fresh._launch_files(ledger, 2)["completed"].is_file()
    assert not output.with_name(f"{output.name}__attempt_02").exists()


def test_failed_launch_with_any_scientific_file_is_isolated(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft, active, active_path = _activated(tmp_path, kwargs)
    output = Path(active["output_path"])

    class ArtifactFailingRunner:
        _semantic_config = staticmethod(lambda value: value)

        def run_worker_manifest(self, *_args, **_kwargs):
            (output / "worker_manifest.json").write_text("{}\n", encoding="utf-8")
            raise RuntimeError("failure after scientific artifact")

    with pytest.raises(RuntimeError, match="after scientific artifact"):
        fresh.execute_fresh_run(runner=ArtifactFailingRunner(), release_path=active_path, **kwargs)
    ledger = Path(active["launch_ledger_root"])
    failed = json.loads(fresh._launch_files(ledger, 1)["failed"].read_text(encoding="utf-8"))
    assert failed["safe_retry_allowed"] is False
    assert failed["isolation_required"] is True
    assert failed["artifact_snapshot"]["artifact_files"]
    retry_kwargs = dict(kwargs)
    retry_kwargs["control_attempt"] = 2
    with pytest.raises(fresh.FreshRunError, match="isolation"):
        fresh.build_release(**retry_kwargs)
    assert not output.with_name(f"{output.name}__attempt_02").exists()


def test_failure_before_runner_invocation_publishes_retryable_evidence(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft, active, active_path = _activated(tmp_path, kwargs)
    runner_without_hook = object()
    with pytest.raises(AttributeError, match="_semantic_config"):
        fresh.execute_fresh_run(
            runner=runner_without_hook,
            release_path=active_path,
            **kwargs,
        )
    ledger = Path(active["launch_ledger_root"])
    failed_path = fresh._launch_files(ledger, 1)["failed"]
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed["failure_stage"] == "output-reserved"
    assert failed["safe_retry_allowed"] is True
    assert failed["artifact_snapshot"]["artifact_files"] == []


def test_existing_claim_blocks_concurrent_controller(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    _draft, active, active_path = _activated(tmp_path, kwargs)
    paths = fresh._bound_paths(
        manifest=_manifest(),
        frozen_checkout=kwargs["frozen_checkout"],
        receipt_root=kwargs["receipt_root"],
    )
    fresh._acquire_launch_claim(
        release=active,
        release_path=active_path,
        paths=paths,
    )
    runner = type("Runner", (), {"_semantic_config": staticmethod(lambda value: value)})()
    with pytest.raises(fresh.FreshRunError, match="no terminal evidence"):
        fresh.execute_fresh_run(runner=runner, release_path=active_path, **kwargs)
    assert not paths["output_path"].exists()
