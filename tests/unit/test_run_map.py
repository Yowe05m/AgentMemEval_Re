from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from agentmemeval.storage.run_map import build_run_map


def _leaf(root: Path, run_id: str, *, run_mode: str, paper_eligible: bool) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "run_mode": run_mode,
                    "protocol_variant": "paper_robust_extension",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "experiment_result.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_validity": {
                    "status": "paper_eligible" if paper_eligible else "blocked",
                    "paper_eligible": paper_eligible,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "protocol_audit.json").write_text(
        json.dumps({"execution_health": {"valid": True}}), encoding="utf-8"
    )


def test_run_map_separates_formal_pilot_and_failed_attempts(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "campaign-x"}), encoding="utf-8"
    )
    _leaf(campaign, "formal__s1__a01", run_mode="formal", paper_eligible=True)
    _leaf(campaign, "pilot__s2__a01", run_mode="pilot", paper_eligible=False)
    header = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
    )
    states = [
        "t\tformal\tfact\t1\t1\trunning\tformal__s1__a01\t/missing\t\t",
        "t\tformal\tfact\t1\t1\tcomplete\tformal__s1__a01\t/missing\t\t",
        "t\tpilot\tfact\t2\t1\tcomplete\tpilot__s2__a01\t/missing\t\t",
        "t\tfailed\tfact\t3\t1\tfailed\tfailed__s3__a01\t/missing\terror\tx",
    ]
    (campaign / "state.tsv").write_text(
        header + "\n".join(states) + "\n", encoding="utf-8"
    )
    output = tmp_path / "server_run_map.csv"
    exclusions = tmp_path / "exclusions.json"
    result = build_run_map([campaign], output, exclusions)
    assert result["total_attempts"] == 3
    assert result["formal_main_table_candidates"] == 1
    assert result["excluded_attempts"] == 2
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    assert rows["formal__s1__a01"]["formal_main_table_eligible"] == "True"
    assert rows["pilot__s2__a01"]["classification"] == "pilot_descriptive_only"
    assert rows["failed__s3__a01"]["classification"] == "partial_or_failed"
    payload = json.loads(exclusions.read_text(encoding="utf-8"))
    assert payload["excluded_attempts"] == 2
