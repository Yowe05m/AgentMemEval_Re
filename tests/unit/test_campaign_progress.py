from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from agentmemeval.evaluation.campaign_progress import build_campaign_progress


def test_campaign_progress_accounts_for_all_checkpoint_targets(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run_dir = campaign / "runs" / "mixed__s1__a01"
    run_dir.mkdir(parents=True)
    config = {
        "experiment": {
            "train_hands": 150,
            "checkpoint_interval": 150,
            "checkpoint_test_hands": 50,
            "evaluate_all_train_agents": True,
            "agent_roster": [
                {"agent_id": f"agent_{index:02d}", "mechanism": "fact"}
                for index in range(8)
            ],
        }
    }
    manifest = {
        "schema_version": "agentmemeval_campaign_v1",
        "campaign": {
            "campaign_id": "progress-test",
            "design": "mixed_table",
            "seeds": [1, 2],
        },
        "base_config": config,
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with (campaign / "state.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event_utc",
                "condition_id",
                "target_mechanism",
                "seed",
                "attempt",
                "status",
                "run_id",
                "run_dir",
                "failure_class",
                "message",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_utc": "t",
                "condition_id": "mixed_table",
                "target_mechanism": "mixed",
                "seed": 1,
                "attempt": 1,
                "status": "running",
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "failure_class": "",
                "message": "",
            }
        )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )
    (run_dir / "hand_summaries.jsonl").write_text(
        "{}\n" * 158,
        encoding="utf-8",
    )

    progress = build_campaign_progress(campaign)

    assert progress["status"] == "consistent"
    assert progress["expected_matrix_units"] == 2
    assert progress["observed_hand_summaries_total"] == 158
    assert progress["expected_hand_summaries_total"] == 1100
    assert progress["progress_fraction"] == round(158 / 1100, 6)
    assert progress["default_budget"]["total_hand_summaries_per_run"] == 550
    assert progress["default_budget"]["checkpoint_cost_budget"][
        "evaluation_target_count"
    ] == 8
    assert progress["units"][0]["stage"] == "checkpoint_generalization"
    assert progress["units"][1]["stage"] == "pending_or_initializing"
    assert progress["paper_eligibility_not_assessed"] is True


def test_campaign_progress_flags_complete_count_and_artifact_mismatch(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run_dir = campaign / "runs" / "target__s1__a01"
    run_dir.mkdir(parents=True)
    config = {
        "experiment": {
            "train_hands": 2,
            "checkpoint_interval": 2,
            "checkpoint_test_hands": 1,
            "evaluation_target_ids": ["target_00"],
        }
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentmemeval_campaign_v1",
                "campaign": {
                    "campaign_id": "bad-progress-test",
                    "design": "target_vs_seven_no_memory",
                    "seeds": [1],
                    "conditions": [
                        {
                            "condition_id": "target",
                            "target_mechanism": "fact",
                        }
                    ],
                },
                "base_config": config,
            }
        ),
        encoding="utf-8",
    )
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"t\ttarget\tfact\t1\t1\tcomplete\t{run_dir.name}\t{run_dir}\t\t\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )
    (run_dir / "hand_summaries.jsonl").write_text("{}\n", encoding="utf-8")

    progress = build_campaign_progress(campaign)

    assert progress["status"] == "inconsistent"
    assert any("hand count mismatch" in item for item in progress["anomalies"])
    assert any("missing artifacts" in item for item in progress["anomalies"])
