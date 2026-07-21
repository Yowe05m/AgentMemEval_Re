from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.task8b_analysis import (
    build_task8b_analysis_input,
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
        lineage_fields = set(next(csv.reader(handle)))
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
