from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from agentmemeval.evaluation.study_reporting import build_task4_study_report


def test_study_report_is_ready_only_when_all_evidence_is_verified(
    tmp_path: Path,
) -> None:
    p_analysis = _analysis_bundle(tmp_path / "p", "mixed_table", "ready")
    e_analysis = _analysis_bundle(
        tmp_path / "e", "target_vs_seven_no_memory", "ready"
    )
    p_run_map = _run_map(
        tmp_path / "p_run_map.csv", eligible=True, add_excluded_attempt=True
    )
    e_run_map = _run_map(tmp_path / "e_run_map.csv", eligible=True)
    p_resource = _resource_audit(tmp_path / "p_resource.json")
    e_resource = _resource_audit(tmp_path / "e_resource.json")
    runtime_lock = _json(
        tmp_path / "runtime_lock.json",
        {"status": "verified_from_real_service_run_manifest"},
    )
    receipt = _json(tmp_path / "receipt.json", {"status": "verified"})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# verified", encoding="utf-8")
    spec = _json(
        tmp_path / "study_spec.json",
        {
            "schema_version": "task4_study_report_spec_v1",
            "campaign_p_analysis_manifest": str(p_analysis),
            "campaign_e_analysis_manifest": str(e_analysis),
            "campaign_p_run_map": str(p_run_map),
            "campaign_e_run_map": str(e_run_map),
            "campaign_p_resource_audit": str(p_resource),
            "campaign_e_resource_audit": str(e_resource),
            "formal_runtime_lock": str(runtime_lock),
            "archive_receipts": [str(receipt)],
            "protocol_evidence": [
                {"label": "protocol", "path": str(protocol), "status": "verified"}
            ],
            "limitations": ["Pilot 与 formal seed 不重叠。"],
        },
    )

    result = build_task4_study_report(spec, tmp_path / "report")

    assert result["paper_inference_eligible"] is True
    assert result["paper_conclusion_prohibited"] is False
    assert result["classification"] == "paper_inference_ready"
    assert result["blockers"] == []
    report = (tmp_path / "report" / "task4_paper_report_zh.md").read_text(
        encoding="utf-8"
    )
    assert "Campaign P：混合桌结果" in report
    assert "Campaign E：训练、泛化与 Generalization Gap" in report
    assert "论文推断资格：`True`" in report
    with (tmp_path / "report" / "study_effects.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["campaign"] for row in rows} == {"campaign_p", "campaign_e"}


def test_study_report_keeps_pilot_and_bad_run_map_out_of_paper(
    tmp_path: Path,
) -> None:
    p_analysis = _analysis_bundle(
        tmp_path / "p", "mixed_table", "descriptive_only"
    )
    e_analysis = _analysis_bundle(
        tmp_path / "e", "target_vs_seven_no_memory", "ready"
    )
    p_run_map = _run_map(tmp_path / "p_run_map.csv", eligible=False)
    e_run_map = _run_map(tmp_path / "e_run_map.csv", eligible=True)
    p_resource = _resource_audit(tmp_path / "p_resource.json")
    e_resource = _resource_audit(tmp_path / "e_resource.json")
    runtime_lock = _json(
        tmp_path / "runtime_lock.json",
        {"status": "verified_from_real_service_run_manifest"},
    )
    receipt = _json(tmp_path / "receipt.json", {"status": "verified"})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# verified", encoding="utf-8")
    spec = _json(
        tmp_path / "study_spec.json",
        {
            "schema_version": "task4_study_report_spec_v1",
            "campaign_p_analysis_manifest": str(p_analysis),
            "campaign_e_analysis_manifest": str(e_analysis),
            "campaign_p_run_map": str(p_run_map),
            "campaign_e_run_map": str(e_run_map),
            "campaign_p_resource_audit": str(p_resource),
            "campaign_e_resource_audit": str(e_resource),
            "formal_runtime_lock": str(runtime_lock),
            "archive_receipts": [str(receipt)],
            "protocol_evidence": [
                {"label": "protocol", "path": str(protocol), "status": "verified"}
            ],
        },
    )

    result = build_task4_study_report(spec, tmp_path / "report")

    assert result["paper_inference_eligible"] is False
    assert result["paper_conclusion_prohibited"] is True
    assert result["classification"] == "interim_or_blocked_no_paper_conclusion"
    assert "campaign_p analysis is not formal inference eligible" in result["blockers"]
    assert (
        "campaign_p run map does not cover the formal analysis matrix"
        in result["blockers"]
    )
    report = (tmp_path / "report" / "task4_paper_report_zh.md").read_text(
        encoding="utf-8"
    )
    assert "当前禁止把结果写成正式论文结论" in report


def _analysis_bundle(root: Path, design: str, status: str) -> Path:
    root.mkdir()
    aggregate = root / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            {"design": design, "status": status, "expected_run_count": 1}
        ),
        encoding="utf-8",
    )
    table = root / "main_table.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "campaign_status",
                "paper_inference_eligible",
                "analysis_classification",
                "design",
                "contrast",
                "endpoint",
                "baseline",
                "n_seed_pairs",
                "mean_effect",
                "median_effect",
                "std_effect",
                "ci95_low",
                "ci95_high",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "raw_p_value",
                "holm_adjusted_p_value",
            ),
        )
        writer.writeheader()
        contrasts = (
            ("expr_vs_fact", "fact_expr_sync_vs_fact", "fact_expr_async_vs_fact")
            if design == "mixed_table"
            else ("fact_target", "expr_target", "sync_target", "async_target")
        )
        endpoints = (
            "final_test_bb_per_100",
            "final_test_chip_per_hand",
            "train_bb_per_100",
            "train_chip_per_hand",
            "generalization_gap_bb_per_100",
        )
        for contrast in contrasts:
            for endpoint in endpoints:
                writer.writerow(
                    {
                        "campaign_status": status,
                        "paper_inference_eligible": status == "ready",
                        "analysis_classification": (
                            "formal_inference_ready"
                            if status == "ready"
                            else "pilot_descriptive_only"
                        ),
                        "design": design,
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "baseline": "baseline",
                        "n_seed_pairs": 8,
                        "mean_effect": 1.5,
                        "median_effect": 1.0,
                        "std_effect": 2.0,
                        "ci95_low": -0.5,
                        "ci95_high": 3.5,
                        "bootstrap_ci95_low": -0.25,
                        "bootstrap_ci95_high": 3.25,
                        "raw_p_value": (
                            0.1 if endpoint == "final_test_bb_per_100" else None
                        ),
                        "holm_adjusted_p_value": (
                            0.4 if endpoint == "final_test_bb_per_100" else None
                        ),
                    }
                )
    report = root / "campaign_analysis_report.md"
    report.write_text("# report", encoding="utf-8")
    plot_data = root / "primary_effects_plot_data.csv"
    plot_data.write_text("x\n1\n", encoding="utf-8")
    plot = root / "primary_effects_plot.png"
    plot.write_bytes(b"png")
    paired = root / "paired_effects.csv"
    paired.write_text("seed,effect\n1,1\n", encoding="utf-8")
    outputs = [table, report, plot_data, plot, paired]
    manifest = root / "analysis_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "task4_campaign_analysis_bundle_v3",
                "source_aggregate": str(aggregate),
                "source_aggregate_sha256": _sha256(aggregate),
                "campaign_status": status,
                "analysis_classification": (
                    "formal_inference_ready"
                    if status == "ready"
                    else "pilot_descriptive_only"
                ),
                "paper_inference_eligible": status == "ready",
                "outputs": {
                    path.name: {"path": str(path), "sha256": _sha256(path)}
                    for path in outputs
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _run_map(
    path: Path,
    *,
    eligible: bool,
    add_excluded_attempt: bool = False,
) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("run_id", "formal_main_table_eligible")
        )
        writer.writeheader()
        writer.writerow(
            {"run_id": "run_1", "formal_main_table_eligible": eligible}
        )
        if add_excluded_attempt:
            writer.writerow(
                {"run_id": "run_1_failed", "formal_main_table_eligible": False}
            )
    return path


def _resource_audit(path: Path) -> Path:
    return _json(
        path,
        {
            "schema_version": "task4_campaign_resource_audit_v1",
            "completed_leaf_count": 8,
            "campaign_wall_hours": 10.0,
            "action_request_count": 100,
            "action_fallback_count": 0,
            "experience_revision_fallback_count": 0,
            "gpu_identities": [
                {
                    "name": "NVIDIA GeForce RTX 4090",
                    "driver": "580.00",
                    "pci_bus_id": "0000:01:00.0",
                }
            ],
            "token_accounting": {
                "status": "heuristic_estimate_not_provider_usage",
                "estimated_total_tokens": 1234,
            },
        },
    )


def _json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
