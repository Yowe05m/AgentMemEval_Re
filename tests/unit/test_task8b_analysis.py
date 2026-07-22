from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation import task8b_analysis as task8b_analysis_module
from agentmemeval.evaluation.task8b_analysis import (
    PHASE_F_REQUIRED_FILES,
    build_task8b_analysis_input,
    build_task8b_preunlock_manifest,
    run_task8b_analysis,
)

IDENTITY = {
    "code_sha": "a" * 40,
    "config_sha256": "b" * 64,
    "prompt_sha256": "c" * 64,
    "model_fingerprint": "qwen-frozen-v1",
    "embedding_fingerprint": "bge-m3-frozen-v1",
    "schedule_sha256": "d" * 64,
}
FROZEN_EXCLUSION_REASONS = (
    "IDENTITY_MISMATCH",
    "CRN_MISMATCH",
    "FALLBACK_NONZERO",
    "REVISION_FALLBACK_NONZERO",
    "REWARD_CONSERVATION_VIOLATION",
    "STACK_CONSERVATION_VIOLATION",
    "ARTIFACT_INCOMPLETE_OR_HASH_MISMATCH",
    "OUTPUT_PATH_COLLISION",
    "INVALID_RECEIPT_OR_DEPENDENCY",
    "EXECUTION_INVALID",
    "ELIGIBLE_INFRA_FAILURE",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def test_phase_f_input_builder_explicitly_rejects_canary_protocol(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    _write_json(
        manifests / "P01.json",
        {
            "protocol_status": "canary/not-for-paper",
            "worker_id": "P01",
            "seed_bundle": 2026090101,
        },
    )

    with pytest.raises(ConfigError, match="拒绝非 expedited formal manifest"):
        build_task8b_analysis_input(
            manifests,
            tmp_path / "snapshots",
            tmp_path / "phase-f-input.json",
            tmp_path / "pre-unlock.json",
        )


def _write_phase_f_files(root: Path) -> None:
    for relative_path in PHASE_F_REQUIRED_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen:{relative_path}\n", encoding="utf-8", newline="")


def test_phase_f_preunlock_manifest_rejects_pyproject_as_dependency_lock(
    tmp_path: Path,
) -> None:
    phase_f = tmp_path / "phase_f"
    _write_phase_f_files(phase_f)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname='not-a-lock'\n", encoding="utf-8", newline="")

    with pytest.raises(ConfigError, match="真实依赖锁文件"):
        build_task8b_preunlock_manifest(
            phase_f,
            pyproject,
            tmp_path / "pre-unlock.json",
            repository_root=tmp_path / "repo",
            frozen_at_utc="2026-07-22T00:00:00+00:00",
        )


def test_phase_f_preunlock_manifest_hashes_files_code_and_real_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_f = tmp_path / "phase_f"
    repo = tmp_path / "repo"
    _write_phase_f_files(phase_f)
    for relative_path in task8b_analysis_module.PHASE_F_ANALYSIS_CODE_FILES:
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"code:{relative_path}\n", encoding="utf-8", newline="")
    lock = repo / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8", newline="")
    monkeypatch.setattr(
        task8b_analysis_module,
        "_git_clean_head",
        lambda _repo: "a" * 40,
    )

    preunlock_path = tmp_path / "pre-unlock.json"
    payload = build_task8b_preunlock_manifest(
        phase_f,
        lock,
        preunlock_path,
        repository_root=repo,
        frozen_at_utc="2026-07-22T00:00:00+00:00",
    )

    assert payload["formal_result_loaded"] is False
    assert payload["analysis_code_dirty"] is False
    assert payload["analysis_code_sha"] == "a" * 40
    assert payload["dependency_lock"]["verified_real_lock"] is True
    assert payload["dependency_lock"]["sha256"] == _sha256(lock)
    assert {row["relative_path"] for row in payload["phase_f_files"]} == set(
        PHASE_F_REQUIRED_FILES
    )
    assert {row["relative_path"] for row in payload["analysis_code_files"]} == set(
        task8b_analysis_module.PHASE_F_ANALYSIS_CODE_FILES
    )
    assert task8b_analysis_module._validate_preunlock_manifest(preunlock_path) == payload

    payload["formal_result_loaded"] = True
    tampered = tmp_path / "pre-unlock-tampered.json"
    _write_json(tampered, payload)
    with pytest.raises(ConfigError, match="揭盲前冻结门禁"):
        task8b_analysis_module._validate_preunlock_manifest(tampered)


def _write_input_builder_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    phase_f = tmp_path / "phase_f"
    repo = tmp_path / "repo"
    manifests = tmp_path / "manifests"
    snapshots = tmp_path / "snapshots"
    _write_phase_f_files(phase_f)
    for relative_path in task8b_analysis_module.PHASE_F_ANALYSIS_CODE_FILES:
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"code:{relative_path}\n", encoding="utf-8", newline="")
    lock = repo / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8", newline="")
    monkeypatch.setattr(task8b_analysis_module, "_git_clean_head", lambda _repo: "e" * 40)
    preunlock = tmp_path / "pre-unlock.json"
    build_task8b_preunlock_manifest(
        phase_f,
        lock,
        preunlock,
        repository_root=repo,
        frozen_at_utc="2026-07-22T00:00:00+00:00",
    )
    manifest_index = manifests / "manifest_index.json"
    _write_json(manifest_index, {"schema_version": "fixture"})
    for index, seed in enumerate(range(2026090101, 2026090113), start=1):
        for role in ("P", "S"):
            worker_id = f"{role}{index:02d}"
            _write_json(
                manifests / f"{worker_id}.json",
                {
                    "protocol_status": "frozen/expedited-formal-candidate",
                    "worker_id": worker_id,
                    "pod_id": f"pod{index:02d}",
                    "seed_bundle": seed,
                    "common_identity": IDENTITY,
                },
            )
            (snapshots / worker_id / str(seed) / "attempt_01").mkdir(parents=True)
    return phase_f, repo, lock, manifests, snapshots


def test_phase_f_input_separates_analysis_and_experiment_code_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_f, repo, lock, manifests, snapshots = _write_input_builder_fixture(
        tmp_path, monkeypatch
    )
    payload = build_task8b_analysis_input(
        manifests,
        snapshots,
        tmp_path / "phase-f-input.json",
        tmp_path / "pre-unlock.json",
        repository_root=repo,
        phase_f_dir=phase_f,
        dependency_lock_path=lock,
    )

    assert payload["analysis_code_sha"] == "e" * 40
    assert payload["analysis_code_dirty"] is False
    assert payload["analysis_frozen_at_utc"] == "2026-07-22T00:00:00+00:00"
    assert payload["dependency_lock_sha256"] == _sha256(lock)
    assert payload["experiment_code_sha"] == IDENTITY["code_sha"]
    assert payload["analysis_code_sha"] != payload["experiment_code_sha"]


@pytest.mark.parametrize(
    ("relative_path", "message"),
    (
        ("src/agentmemeval/evaluation/task8b_analysis.py", "analysis code 文件 hash 不匹配"),
        ("analysis_contract.md", "Phase F contract 文件 hash 不匹配"),
    ),
)
def test_phase_f_input_rejects_tampered_preunlock_frozen_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    message: str,
) -> None:
    phase_f, repo, lock, manifests, snapshots = _write_input_builder_fixture(
        tmp_path, monkeypatch
    )
    target = repo / relative_path if relative_path.startswith("src/") else phase_f / relative_path
    target.write_text("tampered after pre-unlock freeze\n", encoding="utf-8", newline="")

    with pytest.raises(ConfigError, match=message):
        build_task8b_analysis_input(
            manifests,
            snapshots,
            tmp_path / "phase-f-input.json",
            tmp_path / "pre-unlock.json",
            repository_root=repo,
            phase_f_dir=phase_f,
            dependency_lock_path=lock,
        )


def test_phase_f_input_recomputes_preunlock_dependency_lock_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_f, repo, lock, manifests, snapshots = _write_input_builder_fixture(
        tmp_path, monkeypatch
    )
    lock.write_text("tampered lock after pre-unlock freeze\n", encoding="utf-8", newline="")

    with pytest.raises(ConfigError, match="dependency lock hash 不匹配"):
        build_task8b_analysis_input(
            manifests,
            snapshots,
            tmp_path / "phase-f-input.json",
            tmp_path / "pre-unlock.json",
            repository_root=repo,
            phase_f_dir=phase_f,
            dependency_lock_path=lock,
        )


def _metric_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    heldout_raw = {"Fact": "1.0", "Expr": "0.2", "Async": "0.6"}
    for mechanism in ("Fact", "Expr", "Async"):
        records.append(
            {
                "mechanism": mechanism,
                "checkpoint_hand": 300,
                "memory_mode": "Frozen",
                "location": "Source",
                "table_id": "Source-01",
                "target_agent_id": "agent_00",
                "raw_chips": "2.0",
                "hands": 10,
                "big_blind": 2,
            }
        )
        for table_id in ("H01", "H02", "H03"):
            records.append(
                {
                    "mechanism": mechanism,
                    "checkpoint_hand": 300,
                    "memory_mode": "Frozen",
                    "location": "Heldout",
                    "table_id": table_id,
                    "target_agent_id": "agent_00",
                    "raw_chips": heldout_raw[mechanism],
                    "hands": 10,
                    "big_blind": 2,
                }
            )
    return records


def _write_attempt(
    root: Path,
    attempt: str,
    *,
    health_valid: bool,
    worker_id: str = "P01",
    pod_id: str = "pod01",
    seed: int = 2026090101,
) -> Path:
    attempt_root = root / "snapshot" / worker_id / attempt
    attempt_root.mkdir(parents=True)
    _write_json(attempt_root / "metrics.json", {"records": _metric_records()})
    hands = [
        {
            "hand_id": 1,
            "agent_id": "agent_00",
            "mechanism": "Fact",
            "reward_chips": "2.0",
            "pot_size": "10.0",
            "big_blind": 2,
            "vpip": True,
            "fold": False,
            "raise": True,
            "all_in": False,
            "bust": False,
        },
        {
            "hand_id": 2,
            "agent_id": "agent_00",
            "mechanism": "Fact",
            "reward_chips": "-1.0",
            "pot_size": "20.0",
            "big_blind": 2,
            "vpip": False,
            "fold": True,
            "raise": False,
            "all_in": True,
            "bust": True,
        },
    ]
    (attempt_root / "hands.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in hands),
        encoding="utf-8",
        newline="",
    )
    (attempt_root / "events.jsonl").write_text(
        json.dumps({"mechanism": "Fact", "event": "decision"}, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    _write_json(
        attempt_root / "identity.json",
        {
            **IDENTITY,
            "worker_id": worker_id,
            "pod_id": pod_id,
            "seed": seed,
        },
    )
    _write_json(
        attempt_root / "health.json",
        {
            "valid": health_valid,
            "fallback_count": 0,
            "revision_fallback_count": 0,
            "reward_conservation_violations": 0,
            "stack_conservation_violations": 0,
        },
    )
    artifacts = [
        "metrics.json",
        "hands.jsonl",
        "events.jsonl",
        "identity.json",
        "health.json",
    ]
    with (attempt_root / "files.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("relative_path", "size", "sha256"))
        for relative in artifacts:
            path = attempt_root / relative
            writer.writerow((relative, path.stat().st_size, _sha256(path)))
    _write_json(
        attempt_root / "completion_receipt.json",
        {
            "schema_version": "task8-worker-completion-v1",
            "worker_id": worker_id,
            "status": "complete",
            "files_tsv_sha256": _sha256(attempt_root / "files.tsv"),
        },
    )
    return attempt_root


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    first = _write_attempt(tmp_path, "attempt_01", health_valid=False)
    second = _write_attempt(tmp_path, "__attempt_02", health_valid=True)
    manifest = {
        "schema_version": "task8b-phase-f-input-v1",
        "analysis_contract_id": "task8b-phase-f-v1",
        "synthetic_test_mode": True,
        "workers": [
            {
                "worker_id": "P01",
                "pod_id": "pod01",
                "seed": 2026090101,
                "expected_identity": IDENTITY,
                "attempts": [
                    {"attempt": "__attempt_02", "relative_path": "snapshot/P01/__attempt_02"},
                    {"attempt": "attempt_01", "relative_path": "snapshot/P01/attempt_01"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "input_manifest.json"
    _write_json(manifest_path, manifest)
    ledger_path = tmp_path / "exclusion_ledger.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "ledger_entry_id",
            "recorded_before_effect_unblind",
            "seed",
            "pod_id",
            "worker_id",
            "attempt",
            "reason_code",
            "authoritative_attempt",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "ledger_entry_id": "ledger-001",
                "recorded_before_effect_unblind": "true",
                "seed": 2026090101,
                "pod_id": "pod01",
                "worker_id": "P01",
                "attempt": "attempt_01",
                "reason_code": "EXECUTION_INVALID",
                "authoritative_attempt": "__attempt_02",
            }
        )
    return manifest_path, ledger_path, first, second


def _refresh_attempt_hashes(attempt_root: Path) -> None:
    files_path = attempt_root / "files.tsv"
    rows = list(csv.DictReader(files_path.open("r", encoding="utf-8"), delimiter="\t"))
    for row in rows:
        path = attempt_root / row["relative_path"]
        row["size"] = str(path.stat().st_size)
        row["sha256"] = _sha256(path)
    with files_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completion_path = attempt_root / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(completion_path, completion)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_phase_f_analysis_is_byte_identical_and_uses_numeric_attempt_order(
    tmp_path: Path,
) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    first_output = tmp_path / "analysis-a"
    second_output = tmp_path / "analysis-b"

    first = run_task8b_analysis(manifest, ledger, first_output)
    second = run_task8b_analysis(manifest, ledger, second_output)

    assert first == second
    assert _tree_bytes(first_output) == _tree_bytes(second_output)
    selected = (first_output / "selected_attempts.csv").read_text(encoding="utf-8")
    assert "__attempt_02" in selected
    assert "attempt_01" not in selected
    assert all(b"\r\n" not in content for content in _tree_bytes(first_output).values())


def test_phase_f_emits_fixed_precision_primary_and_raw_e6_outputs(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "analysis"

    run_task8b_analysis(manifest, ledger, output)

    effects = (output / "primary_seed_effects.csv").read_text(encoding="utf-8")
    assert "Expr_vs_Fact" in effects
    assert "Async_vs_Fact" in effects
    assert ".00000000" in effects
    e6 = (output / "e6_metrics.csv").read_text(encoding="utf-8")
    assert "25.00000000" in e6
    assert "100.00000000" in e6
    assert (output / "figure2_plotting_data.csv").is_file()
    assert (output / "data_lineage.csv").is_file()
    assert (output / "exclusion_retry_ledger.csv").is_file()
    assert (output / "analysis_manifest.json").is_file()


@pytest.mark.parametrize(
    ("target", "mutate", "match"),
    [
        ("identity.json", lambda value: value.update(code_sha="f" * 40), "identity|ledger"),
        (
            "health.json",
            lambda value: value.update(fallback_count=1),
            "FALLBACK_NONZERO|ledger",
        ),
        (
            "completion_receipt.json",
            lambda value: value.update(status="partial"),
            "receipt|ledger",
        ),
    ],
)
def test_phase_f_fails_closed_on_identity_health_or_completion(
    tmp_path: Path,
    target: str,
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    manifest, ledger, _, valid = _write_fixture(tmp_path)
    path = valid / target
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)

    with pytest.raises(ConfigError, match=match):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_fails_closed_on_artifact_hash_change(tmp_path: Path) -> None:
    manifest, ledger, _, valid = _write_fixture(tmp_path)
    with (valid / "hands.jsonl").open("a", encoding="utf-8", newline="") as handle:
        handle.write("{}\n")

    with pytest.raises(ConfigError, match="hash|artifact|ledger|ARTIFACT"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_rejects_multiple_valid_attempts_instead_of_selecting_latest(
    tmp_path: Path,
) -> None:
    manifest, ledger, invalid, _ = _write_fixture(tmp_path)
    health_path = invalid / "health.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    health["valid"] = True
    _write_json(health_path, health)
    files_path = invalid / "files.tsv"
    rows = list(csv.DictReader(files_path.open("r", encoding="utf-8"), delimiter="\t"))
    for row in rows:
        if row["relative_path"] == "health.json":
            row["size"] = str(health_path.stat().st_size)
            row["sha256"] = _sha256(health_path)
    with files_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completion_path = invalid / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(completion_path, completion)

    with pytest.raises(ConfigError, match="authoritative|valid attempt|multiple|恰为 1"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_rejects_non_preregistered_exclusion_reason(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    text = ledger.read_text(encoding="utf-8").replace("EXECUTION_INVALID", "BAD_SCIENCE_RESULT")
    ledger.write_text(text, encoding="utf-8", newline="")

    with pytest.raises(ConfigError, match="reason|预注册|exclusion"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_non_synthetic_input_requires_exact_24_workers_and_12_seeds(
    tmp_path: Path,
) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["synthetic_test_mode"] = False
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="24|12|seed|worker"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_files_manifest_must_cover_every_analysis_source(tmp_path: Path) -> None:
    manifest, ledger, _, valid = _write_fixture(tmp_path)
    files_path = valid / "files.tsv"
    rows = list(csv.DictReader(files_path.open("r", encoding="utf-8"), delimiter="\t"))
    rows = [row for row in rows if row["relative_path"] != "events.jsonl"]
    with files_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completion_path = valid / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_tsv_sha256"] = _sha256(files_path)
    _write_json(completion_path, completion)

    with pytest.raises(ConfigError, match="files|source|artifact|ledger|ARTIFACT"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


@pytest.mark.parametrize("reason_code", FROZEN_EXCLUSION_REASONS)
def test_phase_f_accepts_every_and_only_frozen_exclusion_reason(
    tmp_path: Path, reason_code: str
) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    extra = dict(rows[0])
    extra.update(
        ledger_entry_id="allowlist-probe",
        worker_id="P99",
        attempt="attempt_01",
        reason_code=reason_code,
        authoritative_attempt="__attempt_02",
    )
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows([*rows, extra])

    run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_ledger_reason_must_match_observed_engineering_failure(
    tmp_path: Path,
) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    text = ledger.read_text(encoding="utf-8").replace(
        "EXECUTION_INVALID", "IDENTITY_MISMATCH"
    )
    ledger.write_text(text, encoding="utf-8", newline="")

    with pytest.raises(ConfigError, match="reason|ledger|EXECUTION_INVALID"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_rejects_wrong_analysis_contract_id(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["analysis_contract_id"] = "task8b-phase-f-v999"
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="contract|合同|task8b-phase-f-v1"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_attempt_name_must_use_frozen_attempt_convention(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["workers"][0]["attempts"][0]["attempt"] = "retry_02"
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="attempt|名称|convention"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_incomplete_heldout_table_cell_does_not_enter_primary_effect(
    tmp_path: Path,
) -> None:
    manifest, ledger, _, valid = _write_fixture(tmp_path)
    metrics_path = valid / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["records"] = [
        row
        for row in metrics["records"]
        if not (row["mechanism"] == "Expr" and row.get("table_id") == "H03")
    ]
    _write_json(metrics_path, metrics)
    _refresh_attempt_hashes(valid)

    run_task8b_analysis(manifest, ledger, tmp_path / "analysis")

    effects = (tmp_path / "analysis" / "primary_seed_effects.csv").read_text(
        encoding="utf-8"
    )
    assert "Expr_vs_Fact" not in effects


def test_phase_f_lineage_and_inference_outputs_cover_frozen_schema(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "analysis"

    run_task8b_analysis(manifest, ledger, output)

    with (output / "data_lineage.csv").open("r", encoding="utf-8", newline="") as handle:
        lineage_rows = list(csv.DictReader(handle))
    lineage_fields = set(lineage_rows[0])
    assert {
        "analysis_contract_id",
        "analysis_code_sha",
        "analysis_manifest_sha256",
        "input_manifest_sha256",
        "source_file_sha256",
        "task_id",
        "analysis_family",
        "memory_mode",
        "location",
        "embedding_fingerprint",
        "schedule_sha256",
        "statistical_unit",
        "n_planned",
        "n_effective",
        "exclusion_ledger_sha256",
        "verification_status",
    }.issubset(lineage_fields)
    lineage_statuses = {row["verification_status"] for row in lineage_rows}
    assert lineage_statuses <= {"UNVERIFIED", "VERIFIED", "REJECTED"}
    required_nonempty = {
        "lineage_id",
        "output_artifact_id",
        "output_element_id",
        "output_kind",
        "analysis_contract_id",
        "analysis_code_sha",
        "analysis_manifest_sha256",
        "input_manifest_sha256",
        "input_snapshot_id",
        "exclusion_ledger_sha256",
        "row_selector",
        "transformation",
        "aggregation_order",
        "cluster_ids",
        "statistical_unit",
        "n_planned",
        "n_effective",
        "verification_status",
        "source_records",
    }
    assert all(
        all(str(row[field]).strip() for field in required_nonempty)
        for row in lineage_rows
    )
    required_source_fields = {
        "run_id",
        "worker_id",
        "pod_id",
        "seed",
        "condition",
        "task_id",
        "analysis_family",
        "mechanism",
        "memory_mode",
        "location",
        "checkpoint_hand",
        "heldout_table_id",
        "attempt",
        "code_sha",
        "config_sha256",
        "prompt_sha256",
        "model_fingerprint",
        "embedding_fingerprint",
        "schedule_sha256",
        "source_file_relative_path",
        "source_file_sha256",
        "exclusion_status",
    }
    for row in lineage_rows:
        source_records = json.loads(row["source_records"])
        assert source_records
        assert all(required_source_fields <= set(record) for record in source_records)
        assert json.loads(row["aggregation_order"])
        assert json.loads(row["cluster_ids"])
        assert row["output_kind"] in {
            "table_cell",
            "figure_point",
            "figure_line",
            "figure_interval",
            "caption_statistic",
        }
    assert all(row["output_kind"] != "raw_source_provenance" for row in lineage_rows)
    assert (output / "raw_source_provenance.csv").is_file()
    with (output / "primary_inference.csv").open("r", encoding="utf-8", newline="") as handle:
        inference_fields = set(next(csv.reader(handle)))
    assert {
        "contrast",
        "mean_bb100",
        "ci95_low",
        "ci95_high",
        "raw_p_two_sided",
        "holm_adjusted_p",
        "n_planned",
        "n_effective",
        "bootstrap_replicates",
        "bootstrap_prng",
        "bootstrap_seed",
    }.issubset(inference_fields)


def test_phase_f_e6_contains_all_frozen_raw_robustness_fields(tmp_path: Path) -> None:
    manifest, ledger, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "analysis"

    run_task8b_analysis(manifest, ledger, output)

    with (output / "e6_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        fields = set(next(csv.reader(handle)))
    assert {
        "raw_bb_per_100",
        "leave_largest_absolute_pot_out_bb_per_100",
        "median_bb_per_100",
        "trimmed_10pct_bb_per_100",
        "winsorized_5_95_bb_per_100",
        "vpip_pct",
        "fold_pct",
        "raise_pct",
        "all_in_pct",
        "bust_pct",
        "max_pot_share_pct",
        "event_count",
    }.issubset(fields)


def test_primary_holm_family_is_unresolved_when_one_contrast_is_missing() -> None:
    rows = task8b_analysis_module._primary_inference_rows(
        [
            {
                "contrast": "Expr_vs_Fact",
                "paired_interaction_bb_per_100": "1.00000000",
            }
        ]
    )

    assert [row["contrast"] for row in rows] == ["Expr_vs_Fact", "Async_vs_Fact"]
    assert {row["holm_adjusted_p"] for row in rows} == {"UNRESOLVED"}
    assert {row["holm_rank"] for row in rows} == {"UNRESOLVED"}
    assert {row["holm_reject_0_05"] for row in rows} == {"UNRESOLVED"}
    assert next(row for row in rows if row["contrast"] == "Async_vs_Fact")[
        "n_effective"
    ] == 0


@pytest.mark.parametrize(
    ("counter", "expected_reason"),
    (
        ("fallback_count", "FALLBACK_NONZERO"),
        ("memory_revision_fallback_count", "REVISION_FALLBACK_NONZERO"),
        (
            "reward_conservation_violation_count",
            "REWARD_CONSERVATION_VIOLATION",
        ),
        (
            "stack_conservation_violation_count",
            "STACK_CONSERVATION_VIOLATION",
        ),
    ),
)
def test_formal_health_counters_keep_distinct_frozen_exclusion_codes(
    counter: str,
    expected_reason: str,
) -> None:
    execution = {
        "fallback_count": 0,
        "memory_revision_fallback_count": 0,
        "reward_conservation_violation_count": 0,
        "stack_conservation_violation_count": 0,
    }
    execution[counter] = 1

    reasons = task8b_analysis_module._formal_execution_health_reasons(execution)

    assert reasons == [expected_reason]
    assert task8b_analysis_module._ledger_reason_for(reasons) == expected_reason


def test_formal_health_counter_ledger_priority_is_explicit() -> None:
    reasons = task8b_analysis_module._formal_execution_health_reasons(
        {
            "fallback_count": 1,
            "memory_revision_fallback_count": 1,
            "reward_conservation_violation_count": 1,
            "stack_conservation_violation_count": 1,
        }
    )

    assert task8b_analysis_module._ledger_reason_for(reasons) == (
        "REVISION_FALLBACK_NONZERO"
    )


def test_table3_lineage_uses_cell_specific_n_effective() -> None:
    source_rows = [
        {
            "condition": "Fact",
            "mechanism": "Fact",
            "seed": seed,
            "checkpoint": 30,
            "task_id": "isolation_fact",
            "analysis_family": "R1-E1-I",
            "memory_mode": "Frozen",
            "location": "Heldout",
            "run_id": f"P{seed}",
            "source_file": "metrics.json",
            "source_file_sha256": "a" * 64,
            "row_selector": "records[0]",
        }
        for seed in task8b_analysis_module.FORMAL_SEEDS[:7]
    ]
    rows = task8b_analysis_module._paper_lineage_rows(
        (
            [
                {
                    "mechanism": "Fact",
                    "checkpoint_hand": 30,
                    "seed_clusters_effective": 7,
                    "generalization_gap_mean": "1.00000000",
                }
            ],
        ),
        ("table3_checkpoint_scan",),
        source_rows,
    )

    assert rows
    assert {int(row["n_effective"]) for row in rows} == {7}
    assert {row["cluster_ids"] for row in rows} == {
        ";".join(str(seed) for seed in task8b_analysis_module.FORMAL_SEEDS[:7])
    }


def test_figure_lineage_has_one_mark_per_plotting_row() -> None:
    source = {
        "condition": "Fact",
        "mechanism": "Fact",
        "seed": task8b_analysis_module.FORMAL_SEEDS[0],
        "checkpoint": 300,
        "task_id": "isolation_fact",
        "analysis_family": "R1-E1-I",
        "memory_mode": "Frozen",
        "location": "Heldout",
        "run_id": "P01:isolation_fact",
        "source_file": "metrics.json",
        "source_file_sha256": "a" * 64,
        "row_selector": "records[0]",
    }
    figure_data = (
        [{"order": 0, "source": "source train", "target": "checkpoint"}],
        [{"mechanism": "Fact", "checkpoint_hand": 300, "n_effective": 1}],
        [
            {
                "seed": task8b_analysis_module.FORMAL_SEEDS[0],
                "contrast": "Expr_vs_Fact",
                "paired_interaction_bb_per_100": "1.00000000",
                "row_type": "seed",
            }
        ],
        [
            {
                "seed": task8b_analysis_module.FORMAL_SEEDS[0],
                "mechanism": "Fact",
                "mode": "Actual Frozen",
                "final_bb100": "1.00000000",
            }
        ],
    )

    rows = task8b_analysis_module._figure_lineage_rows(figure_data, [source])

    assert len(rows) == sum(len(figure) for figure in figure_data)
    assert {row["output_artifact_id"] for row in rows} == {
        f"figure{index}_plotting_data.csv" for index in range(1, 5)
    }
    assert {row["output_kind"] for row in rows} <= {
        "figure_point",
        "figure_line",
        "figure_interval",
    }


def _write_non_synthetic_24_worker_fixture(tmp_path: Path) -> tuple[Path, Path]:
    workers = []
    for index, seed in enumerate(range(2026090101, 2026090113), start=1):
        pod_id = f"pod{index:02d}"
        for prefix in ("P", "S"):
            worker_id = f"{prefix}{index:02d}"
            _write_attempt(
                tmp_path,
                "attempt_01",
                health_valid=True,
                worker_id=worker_id,
                pod_id=pod_id,
                seed=seed,
            )
            workers.append(
                {
                    "worker_id": worker_id,
                    "pod_id": pod_id,
                    "seed": seed,
                    "expected_identity": IDENTITY,
                    "expected_worker_manifest_sha256": "f" * 64,
                    "attempts": [
                        {
                            "attempt": "attempt_01",
                            "relative_path": f"snapshot/{worker_id}/attempt_01",
                        }
                    ],
                }
            )
    manifest_path = tmp_path / "formal_input_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "task8b-phase-f-input-v1",
            "analysis_contract_id": "task8b-phase-f-v1",
            "synthetic_test_mode": False,
            "analysis_code_sha": "e" * 40,
            "analysis_code_dirty": False,
            "analysis_frozen_at_utc": "2026-07-22T00:00:00+00:00",
            "dependency_lock_sha256": "9" * 64,
            "experiment_code_sha": IDENTITY["code_sha"],
            "input_snapshot_id": "synthetic-24-worker-render-fixture",
            "workers": workers,
        },
    )
    ledger_path = tmp_path / "formal_exclusion_ledger.csv"
    ledger_path.write_text(
        "ledger_entry_id,recorded_before_effect_unblind,seed,pod_id,worker_id,attempt,"
        "reason_code,authoritative_attempt\n",
        encoding="utf-8",
        newline="",
    )
    return manifest_path, ledger_path


def test_phase_f_formal_input_requires_preunlock_dependency_lock_sha(
    tmp_path: Path,
) -> None:
    manifest, ledger = _write_non_synthetic_24_worker_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value.pop("dependency_lock_sha256")
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="dependency lock SHA-256"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_rejects_same_budget_wrong_worker_seed_topology(tmp_path: Path) -> None:
    manifest, ledger = _write_non_synthetic_24_worker_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    by_id = {row["worker_id"]: row for row in value["workers"]}
    for prefix in ("P", "S"):
        by_id[f"{prefix}01"]["seed"], by_id[f"{prefix}02"]["seed"] = (
            by_id[f"{prefix}02"]["seed"],
            by_id[f"{prefix}01"]["seed"],
        )
    _write_json(manifest, value)

    with pytest.raises(ConfigError, match="topology|seed|freeze"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_phase_f_formal_primary_cell_requires_exact_12_seeds(tmp_path: Path) -> None:
    manifest, ledger = _write_non_synthetic_24_worker_fixture(tmp_path)
    for worker_id in ("P01", "S01"):
        attempt = tmp_path / "snapshot" / worker_id / "attempt_01"
        metrics_path = attempt / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["records"] = [
            row
            for row in metrics["records"]
            if not (
                row["mechanism"] == "Expr"
                and row["location"] == "Heldout"
                and row["table_id"] == "H03"
            )
        ]
        _write_json(metrics_path, metrics)
        _refresh_attempt_hashes(attempt)

    with pytest.raises(ConfigError, match="12|seed|cell|primary|contrast"):
        run_task8b_analysis(manifest, ledger, tmp_path / "analysis")


def test_table4_uses_fixed_50_hand_windows_and_50_hand_block_ols() -> None:
    from agentmemeval.evaluation import task8b_analysis

    metric_rows: list[dict[str, object]] = []
    for table_id in ("H01", "H02", "H03"):
        metric_rows.extend(
            [
                {
                    "seed": 2026090101,
                    "mechanism": "Expr",
                    "memory_mode": "Frozen",
                    "analysis_family": "R1-E1-I",
                    "checkpoint_hand": 300,
                    "location": "Heldout",
                    "table_id": table_id,
                    "hands": 120,
                    "big_blind": 2,
                    "raw_chips": 24,
                    "hand_bb100_series": [10] * 120,
                },
                {
                    "seed": 2026090101,
                    "mechanism": "Expr",
                    "memory_mode": "Online",
                    "analysis_family": "R1-E4",
                    "checkpoint_hand": 300,
                    "location": "Heldout",
                    "table_id": table_id,
                    "hands": 120,
                    "big_blind": 2,
                    "raw_chips": 142.8,
                    "hand_bb100_series": list(range(120)),
                },
            ]
        )

    rows, _ = task8b_analysis._table4_rows(metric_rows)
    expr = {(row["mode"]): row for row in rows if row["mechanism"] == "Expr"}

    assert expr["Actual Frozen"]["initial_transfer_bb100"] == "10.00000000"
    assert expr["Actual Frozen"]["final_bb100"] == "10.00000000"
    assert (
        expr["Actual Frozen"]["recovery_slope_bb100_per_100_hands"]
        == "NA_NOT_APPLICABLE"
    )
    assert expr["Online"]["initial_transfer_bb100"] == "24.50000000"
    assert expr["Online"]["final_bb100"] == "94.50000000"
    assert expr["Online"]["recovery_slope_bb100_per_100_hands"] == "100.00000000"
    assert expr["Online"]["paired_effect_vs_actual_frozen"] == "84.50000000"
    assert expr["Without"]["recovery_slope_bb100_per_100_hands"] == (
        "NA_NOT_APPLICABLE"
    )


def test_table1_marks_gpu_driver_as_observed_non_gating() -> None:
    from agentmemeval.evaluation import task8b_analysis

    rows = {row["field"]: row for row in task8b_analysis._table1_rows([])}

    assert rows["gpu_driver"] == {
        "field": "gpu_driver",
        "frozen_value": "observed per worker; informational only; non-gating",
        "source": "runtime health audit",
        "status": "frozen",
        "disclosure": "",
    }
    assert rows["cuda_runtime"]["frozen_value"] == "uniform frozen CUDA runtime class"


def test_phase_f_paper_renderer_emits_all_frozen_tables_figures_and_hashes(
    tmp_path: Path,
) -> None:
    manifest, ledger = _write_non_synthetic_24_worker_fixture(tmp_path)
    output = tmp_path / "paper-artifacts"

    run_task8b_analysis(manifest, ledger, output)

    expected = {
        "论文主表与主图空模板_filled.md",
        "analysis_manifest.json",
        "artifact_sha256.csv",
        "leave_one_seed_out_robustness.csv",
        "secondary_mixed_ecological.csv",
    }
    for index in range(1, 6):
        stem = (
            "table1_protocol_identity"
            if index == 1
            else "table2_core_adaptation_generalization"
            if index == 2
            else "table3_checkpoint_scan"
            if index == 3
            else "table4_frozen_online_without"
            if index == 4
            else "table5_behavior_robustness"
        )
        expected.update({f"{stem}.csv", f"{stem}.md"})
    for index in range(1, 5):
        expected.update(
            {
                f"figure{index}.png",
                f"figure{index}.svg",
                f"figure{index}.pdf",
                f"figure{index}_plotting_data.csv",
            }
        )
    assert expected.issubset({path.name for path in output.iterdir() if path.is_file()})
    assert all((output / name).stat().st_size > 0 for name in expected)

    with (output / "artifact_sha256.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        checksum_rows = list(csv.DictReader(handle))
    by_path = {row["relative_path"]: row for row in checksum_rows}
    assert "artifact_sha256.csv" not in by_path
    for relative, row in by_path.items():
        artifact = output / relative
        assert artifact.is_file()
        assert int(row["size"]) == artifact.stat().st_size
        assert row["sha256"] == _sha256(artifact)
    assert expected - {"artifact_sha256.csv"} <= set(by_path)

    filled = (output / "论文主表与主图空模板_filled.md").read_text(encoding="utf-8")
    assert all(f"## Table {index}" in filled for index in range(1, 6))
    assert all(f"## Figure {index}" in filled for index in range(1, 5))

    expected_headers = {
        "table2_core_adaptation_generalization.csv": [
            "mechanism",
            "analysis_role",
            "source_bb100_mean",
            "heldout_bb100_mean",
            "generalization_gap_mean",
            "paired_interaction_vs_fact_mean",
            "ci95_low",
            "ci95_high",
            "raw_p_two_sided",
            "holm_adjusted_p",
            "holm_rank",
            "holm_reject_alpha_0_05",
            "n_planned",
            "n_effective",
            "missing_seed_ids",
            "excluded_seed_ids",
            "lineage_id",
            "status",
        ],
        "table3_checkpoint_scan.csv": [
            "mechanism",
            "checkpoint_hand",
            "source_bb100_mean",
            "source_ci95_low",
            "source_ci95_high",
            "heldout_bb100_mean",
            "heldout_ci95_low",
            "heldout_ci95_high",
            "generalization_gap_mean",
            "gap_ci95_low",
            "gap_ci95_high",
            "checkpoint_slope",
            "seed_clusters_planned",
            "seed_clusters_effective",
            "heldout_tables_planned",
            "heldout_tables_effective",
            "missing_reason_codes",
            "lineage_id",
            "status",
        ],
        "table4_frozen_online_without.csv": [
            "mechanism",
            "mode",
            "analysis_role",
            "parent_checkpoint_hand",
            "initial_transfer_bb100",
            "recovery_slope_bb100_per_100_hands",
            "final_bb100",
            "paired_effect_vs_actual_frozen",
            "ci95_low",
            "ci95_high",
            "n_planned",
            "n_effective",
            "parent_checkpoint_hash_gate",
            "crn_gate",
            "clone_isolation_gate",
            "missing_reason_codes",
            "lineage_id",
            "status",
        ],
        "table5_behavior_robustness.csv": [
            "mechanism",
            "checkpoint_hand",
            "mode",
            "raw_bb100",
            "leave_largest_absolute_pot_out_bb100",
            "median_seed_effect_bb100",
            "trimmed_10pct_bb100",
            "winsorized_5_95_bb100",
            "vpip_rate",
            "fold_rate",
            "raise_rate",
            "all_in_rate",
            "bust_rate",
            "max_pot_share",
            "fallback_count",
            "revision_fallback_count",
            "reward_conservation_violations",
            "stack_conservation_violations",
            "n_planned",
            "n_effective",
            "sensitivity_flag",
            "missing_reason_codes",
            "lineage_id",
            "status",
        ],
    }
    for name, fields in expected_headers.items():
        with (output / name).open("r", encoding="utf-8", newline="") as handle:
            actual = csv.DictReader(handle).fieldnames
        assert actual is not None
        assert set(actual) == set(fields)
        assert len(actual) == len(fields)

    with (output / "leave_one_seed_out_robustness.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        leave_one_rows = list(csv.DictReader(handle))
    assert leave_one_rows
    assert set(leave_one_rows[0]) == {
        "mechanism",
        "omitted_seed",
        "n_effective",
        "raw_bb100_leave_one_seed_out",
        "status",
    }
    with (output / "secondary_mixed_ecological.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mixed_rows = list(csv.DictReader(handle))
    assert [row["mechanism"] for row in mixed_rows] == [
        "Fact",
        "Expr",
        "Sync",
        "Async",
    ]
    assert set(mixed_rows[0]) == {
        "mechanism",
        "analysis_role",
        "heldout_bb100_mean",
        "ci95_low",
        "ci95_high",
        "paired_effect_vs_fact_mean",
        "paired_ci95_low",
        "paired_ci95_high",
        "n_planned",
        "n_effective",
        "status",
    }

    with (output / "exclusion_retry_ledger.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        exclusion_fields = next(csv.reader(handle))
    assert exclusion_fields == list(task8b_analysis_module.EXCLUSION_LEDGER_FIELDS)

    analysis_manifest = json.loads(
        (output / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert analysis_manifest["analysis_code_sha"] == "e" * 40
    assert analysis_manifest["experiment_code_sha"] == IDENTITY["code_sha"]
    assert analysis_manifest["analysis_code_dirty"] is False
    assert analysis_manifest["analysis_frozen_at_utc"] == "2026-07-22T00:00:00+00:00"
    assert analysis_manifest["analysis_environment"]["locked_dependencies_sha256"] == "9" * 64
    assert analysis_manifest["input_artifacts"]
    assert analysis_manifest["exclusion_ledger"]["relative_path"]
    assert analysis_manifest["exclusion_ledger"]["sha256"] == _sha256(
        output / "exclusion_retry_ledger.csv"
    )
    assert len(analysis_manifest["planned_outputs"]) >= 11

    with (output / "data_lineage.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        lineage_rows = list(csv.DictReader(handle))
    figure_rows = [
        row for row in lineage_rows if row["output_artifact_id"].startswith("figure")
    ]
    plotting_row_count = 0
    for index in range(1, 5):
        with (output / f"figure{index}_plotting_data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            plotting_row_count += sum(1 for _ in csv.DictReader(handle))
    assert len(figure_rows) == plotting_row_count
    assert all(row["analysis_code_sha"] == "e" * 40 for row in figure_rows)
    assert all(row["experiment_code_sha"] == IDENTITY["code_sha"] for row in figure_rows)


def test_supplementary_lineage_uses_only_authoritative_family_and_mode_sources() -> None:
    from agentmemeval.evaluation import task8b_analysis

    sources = [
        {
            "condition": "Fact",
            "seed": seed,
            "task_id": task_id,
            "memory_mode": mode,
            "analysis_family": family,
            "checkpoint": 300,
            "location": "Heldout",
            "run_id": f"run-{seed}",
            "source_file": f"source-{seed}.jsonl",
            "source_file_sha256": str(seed) * 8,
            "row_selector": f"records[{seed}]",
        }
        for seed, task_id, mode, family in (
            (1, "isolation_fact", "Frozen", "R1-E1-I"),
            (2, "isolation_fact", "Frozen", "R1-E1-I"),
            (3, "isolation_fact", "Online", "R1-E4"),
            (4, "mixed_ecological", "Frozen", "R1-E1-M"),
            (5, "mixed_ecological", "Frozen", "WRONG_FAMILY"),
        )
    ]
    rows = task8b_analysis._paper_lineage_rows(
        (
            [
                {
                    "mechanism": "Fact",
                    "omitted_seed": 1,
                    "n_effective": 1,
                    "raw_bb100_leave_one_seed_out": "1.00000000",
                }
            ],
            [
                {
                    "mechanism": "Fact",
                    "n_effective": 1,
                    "heldout_bb100_mean": "2.00000000",
                }
            ],
        ),
        ("leave_one_seed_out_robustness", "secondary_mixed_ecological"),
        sources,
        table_offset=5,
    )
    leave_one = next(
        row
        for row in rows
        if row["output_artifact_id"] == "leave_one_seed_out_robustness.csv"
    )
    mixed = next(
        row
        for row in rows
        if row["output_artifact_id"] == "secondary_mixed_ecological.csv"
    )

    assert {item["seed"] for item in json.loads(leave_one["source_records"])} == {2}
    assert {item["seed"] for item in json.loads(mixed["source_records"])} == {4}


def test_table4_lineage_does_not_fallback_to_wrong_location_sources() -> None:
    from agentmemeval.evaluation import task8b_analysis

    rows = task8b_analysis._paper_lineage_rows(
        (
            [
                {
                    "mechanism": "Expr",
                    "mode": "Online",
                    "n_effective": 0,
                    "final_bb100": "NA_NOT_ESTIMABLE",
                }
            ],
        ),
        ("table4_frozen_online_without",),
        [
            {
                "condition": "Expr",
                "checkpoint": 300,
                "task_id": "expr_online",
                "analysis_family": "R1-E4",
                "memory_mode": "Online",
                "location": "Source",
                "seed": 2026090101,
                "run_id": "wrong-location",
                "source_file": "source.jsonl",
                "source_file_sha256": "a" * 64,
                "row_selector": "records[0]",
            }
        ],
        table_offset=3,
    )

    assert all(json.loads(row["source_records"]) == [] for row in rows)
