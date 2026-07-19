from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agentmemeval.evaluation.campaign_reporting import build_campaign_analysis
from agentmemeval.evaluation.statistics import summarize_values


def test_build_mixed_campaign_analysis_bundle(tmp_path: Path) -> None:
    effects = [1.0, 3.0, 5.0]
    summary = summarize_values(effects, bootstrap_samples=100)
    aggregate = {
        "design": "mixed_table",
        "status": "descriptive_only",
        "completed_run_count": 3,
        "expected_run_count": 3,
        "aggregate_metrics": {
            "paired_estimand_descriptive": {
                "endpoint": "final_test_bb_per_100",
                "baseline_mechanism": "fact",
                "matched_seeds": [11, 12, 13],
                "effects_by_mechanism": {"expr": effects},
                "metrics": {"expr": summary},
            }
        },
    }
    source = tmp_path / "aggregate.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")
    output = tmp_path / "analysis"
    result = build_campaign_analysis(source, output)
    assert result["analysis_is_descriptive_only"] is True
    assert result["paper_inference_eligible"] is False
    assert result["paper_conclusion_prohibited"] is True
    assert result["analysis_classification"] == "pilot_descriptive_only"
    assert result["paired_effect_row_count"] == 3
    assert (output / "primary_effects_plot.png").stat().st_size > 0
    with (output / "main_table.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["contrast"] == "expr_vs_fact"
    assert rows[0]["n_seed_pairs"] == "3"
    assert rows[0]["paper_inference_eligible"] == "False"
    with pytest.raises(FileExistsError):
        build_campaign_analysis(source, output)


def test_build_campaign_e_analysis_uses_matched_seed_effects(tmp_path: Path) -> None:
    effects = [2.0, -1.0]
    summary = summarize_values(effects, bootstrap_samples=100)
    train_effects = [4.0, 2.0]
    train_summary = summarize_values(train_effects, bootstrap_samples=100)
    aggregate = {
        "design": "target_vs_seven_no_memory",
        "status": "ready",
        "completed_run_count": 10,
        "expected_run_count": 10,
        "primary_endpoint": "final_test_bb_per_100",
        "baseline_condition_id": "no_memory_target",
        "paired_comparisons": {
            "fact_target": {
                "metrics": {
                    "final_test_bb_per_100": {
                        "matched_seeds": [21, 22],
                        "effects": effects,
                        "summary": summary,
                    },
                    "train_bb_per_100": {
                        "matched_seeds": [21, 22],
                        "effects": train_effects,
                        "summary": train_summary,
                    },
                },
                "primary_raw_p_value": 0.5,
                "primary_holm_adjusted_p_value": 1.0,
            }
        },
    }
    source = tmp_path / "aggregate-e.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")
    output = tmp_path / "analysis-e"
    result = build_campaign_analysis(source, output)
    assert result["analysis_is_descriptive_only"] is False
    assert result["paper_inference_eligible"] is True
    assert result["paper_conclusion_prohibited"] is False
    assert result["analysis_classification"] == "formal_inference_ready"
    assert result["table_row_count"] == 2
    assert result["primary_table_row_count"] == 1
    assert result["schema_version"] == "task4_campaign_analysis_bundle_v3"
    with (output / "paired_effects.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (row["endpoint"], row["seed"], row["effect"]) for row in rows
    ] == [
        ("final_test_bb_per_100", "21", "2.0"),
        ("final_test_bb_per_100", "22", "-1.0"),
        ("train_bb_per_100", "21", "4.0"),
        ("train_bb_per_100", "22", "2.0"),
    ]
    assert all(row["paper_inference_eligible"] == "True" for row in rows)
    with (output / "primary_effects_plot_data.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        plot_rows = list(csv.DictReader(handle))
    assert len(plot_rows) == 1
    report = (output / "campaign_analysis_report.md").read_text(
        encoding="utf-8"
    )
    assert "训练、泛化与 Gap 次要指标" in report
    assert "train_bb_per_100" in report


def test_blocked_campaign_analysis_is_explicitly_not_paper_inference(
    tmp_path: Path,
) -> None:
    effects = [2.0, -1.0]
    summary = summarize_values(effects, bootstrap_samples=100)
    aggregate = {
        "design": "mixed_table",
        "status": "insufficient_preregistered_seed_pairs",
        "completed_run_count": 2,
        "expected_run_count": 2,
        "aggregate_metrics": {
            "main_table": {
                "endpoint": "final_test_bb_per_100",
                "baseline_mechanism": "fact",
                "matched_seeds": [31, 32],
                "effects_by_mechanism": {"expr": effects},
                "metrics": {"expr": summary},
            }
        },
    }
    source = tmp_path / "aggregate-blocked.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")
    output = tmp_path / "analysis-blocked"

    result = build_campaign_analysis(source, output)

    assert result["paper_inference_eligible"] is False
    assert result["paper_conclusion_prohibited"] is True
    assert result["analysis_classification"] == "blocked_or_underpowered"
    report = (output / "campaign_analysis_report.md").read_text(
        encoding="utf-8"
    )
    assert "禁止形成论文推断结论" in report
    with (output / "primary_effects_plot_data.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["campaign_status"] == (
        "insufficient_preregistered_seed_pairs"
    )
    assert rows[0]["paper_inference_eligible"] == "False"
