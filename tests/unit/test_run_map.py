from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from agentmemeval.storage.run_map import (
    REQUIRED_LEAF_ARTIFACTS,
    _source_child_path,
    build_run_map,
)


def _leaf(
    root: Path,
    run_id: str,
    *,
    condition_id: str,
    seed: int,
    source_run_dir: str,
    run_mode: str,
    paper_eligible: bool,
    protocol_variant: str = "paper_robust_extension",
) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "campaign_id": "campaign-x",
                    "campaign_condition_id": condition_id,
                    "seed": seed,
                    "run_id": run_id,
                    "run_mode": run_mode,
                    "protocol_variant": protocol_variant,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": seed,
                "output_dir": source_run_dir,
                "config_snapshot_path": f"{source_run_dir}/resolved_config.yaml",
                "metadata": {"code": {"dirty": False}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "experiment_result.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )
    (run_dir / "hand_summaries.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "checkpoint_generalization.json").write_text(
        json.dumps({"results": []}), encoding="utf-8"
    )
    (run_dir / "report.md").write_text("complete\n", encoding="utf-8")
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
        json.dumps(
            {
                "execution_health": {
                    "valid": True,
                    "status": "passed",
                    "fallback_count": 0,
                    "memory_revision_fallback_count": 0,
                    "reward_conservation_violation_count": 0,
                    "stack_conservation_violation_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_run_map_separates_formal_pilot_and_failed_attempts(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentmemeval_campaign_v1",
                "campaign_id": "campaign-x",
                "campaign": {"campaign_id": "campaign-x"},
            }
        ),
        encoding="utf-8",
    )
    formal_source = "/server/campaign/runs/formal__s1__a01"
    pilot_source = "/server/campaign/runs/pilot__s2__a01"
    _leaf(
        campaign,
        "formal__s1__a01",
        condition_id="formal",
        seed=1,
        source_run_dir=formal_source,
        run_mode="formal",
        paper_eligible=True,
    )
    _leaf(
        campaign,
        "pilot__s2__a01",
        condition_id="pilot",
        seed=2,
        source_run_dir=pilot_source,
        run_mode="pilot",
        paper_eligible=False,
    )
    header = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
    )
    states = [
        f"t\tformal\tfact\t1\t1\trunning\tformal__s1__a01\t{formal_source}\t\t",
        f"t\tformal\tfact\t1\t1\tcomplete\tformal__s1__a01\t{formal_source}\t\t",
        f"t\tpilot\tfact\t2\t1\tcomplete\tpilot__s2__a01\t{pilot_source}\t\t",
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
    formal_hashes = json.loads(
        rows["formal__s1__a01"]["leaf_artifacts_sha256"]
    )
    assert set(formal_hashes) == set(REQUIRED_LEAF_ARTIFACTS)
    assert rows["pilot__s2__a01"]["classification"] == "pilot_descriptive_only"
    assert rows["failed__s3__a01"]["classification"] == "partial_or_failed"
    payload = json.loads(exclusions.read_text(encoding="utf-8"))
    assert payload["excluded_attempts"] == 2


def test_run_map_never_falls_back_to_state_source_directory(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentmemeval_campaign_v1",
                "campaign_id": "campaign-x",
                "campaign": {"campaign_id": "campaign-x"},
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "external"
    _leaf(
        external,
        "formal__s1__a01",
        condition_id="formal",
        seed=1,
        source_run_dir=str(external / "runs" / "formal__s1__a01"),
        run_mode="formal",
        paper_eligible=True,
    )
    state = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"t\tformal\tfact\t1\t1\tcomplete\tformal__s1__a01\t"
        f"{external / 'runs' / 'formal__s1__a01'}\t\t\n"
    )
    (campaign / "state.tsv").write_text(state, encoding="utf-8")

    output = tmp_path / "run_map.csv"
    exclusions = tmp_path / "exclusions.json"
    result = build_run_map([campaign], output, exclusions)

    assert result["formal_main_table_candidates"] == 0
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "canonical_archive_leaf_missing_or_unsafe" in row["exclusion_reasons"]


def test_run_map_rejects_leaf_identity_mismatch(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentmemeval_campaign_v1",
                "campaign_id": "campaign-x",
                "campaign": {"campaign_id": "campaign-x"},
            }
        ),
        encoding="utf-8",
    )
    source = "/server/campaign/runs/formal__s1__a01"
    _leaf(
        campaign,
        "formal__s1__a01",
        condition_id="formal",
        seed=1,
        source_run_dir=source,
        run_mode="formal",
        paper_eligible=True,
    )
    manifest_path = campaign / "runs/formal__s1__a01/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    campaign_manifest_path = campaign / "campaign_manifest.json"
    campaign_manifest = json.loads(
        campaign_manifest_path.read_text(encoding="utf-8")
    )
    campaign_manifest["campaign"]["campaign_id"] = "other-campaign"
    campaign_manifest_path.write_text(
        json.dumps(campaign_manifest), encoding="utf-8"
    )
    state = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"t\tformal\tfact\t1\t1\tcomplete\tformal__s1__a01\t{source}\t\t\n"
    )
    (campaign / "state.tsv").write_text(state, encoding="utf-8")

    output = tmp_path / "run_map.csv"
    exclusions = tmp_path / "exclusions.json"
    result = build_run_map([campaign], output, exclusions)

    assert result["formal_main_table_candidates"] == 0
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "campaign_manifest_identity_invalid" in row["exclusion_reasons"]
    assert "manifest_identity_mismatch:seed" in row["exclusion_reasons"]


def test_run_map_classifies_model_substituted_pilot_as_sensitivity(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentmemeval_campaign_v1",
                "campaign_id": "campaign-x",
                "campaign": {"campaign_id": "campaign-x"},
            }
        ),
        encoding="utf-8",
    )
    source = "/server/campaign/runs/strict__s1__a01"
    _leaf(
        campaign,
        "strict__s1__a01",
        condition_id="strict",
        seed=1,
        source_run_dir=source,
        run_mode="pilot",
        paper_eligible=False,
        protocol_variant="strict_paper_replication_model_substituted",
    )
    state = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"t\tstrict\tmixed\t1\t1\tcomplete\tstrict__s1__a01\t{source}\t\t\n"
    )
    (campaign / "state.tsv").write_text(state, encoding="utf-8")

    output = tmp_path / "run_map.csv"
    exclusions = tmp_path / "exclusions.json"
    build_run_map([campaign], output, exclusions)

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["classification"] == "sensitivity_only"
    payload = json.loads(exclusions.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "task4_formal_main_table_exclusions_v2"


def test_run_map_preserves_source_platform_path_style() -> None:
    assert _source_child_path(
        "/root/campaign/runs/run-1", "resolved_config.yaml"
    ) == "/root/campaign/runs/run-1/resolved_config.yaml"
    assert _source_child_path(
        r"C:\campaign\runs\run-1", "resolved_config.yaml"
    ) == r"C:\campaign\runs\run-1\resolved_config.yaml"
