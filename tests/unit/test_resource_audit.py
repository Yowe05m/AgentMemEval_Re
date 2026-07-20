from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmemeval.evaluation.resource_audit import build_campaign_resource_audit


def test_resource_audit_separates_measured_and_estimated_fields(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    run_id = "mixed__s1__a01"
    run_dir = campaign / "runs" / run_id
    run_dir.mkdir(parents=True)
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"2026-01-01T00:00:00Z\tm\tmixed\t1\t1\trunning\t{run_id}\t/x\t\t\n"
        f"2026-01-01T00:01:00Z\tm\tmixed\t1\t1\tcomplete\t{run_id}\t/x\t\t\n",
        encoding="utf-8",
    )
    event = {
        "agent_id": "a",
        "fallback_used": False,
        "llm": {
            "elapsed_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 4,
        },
        "memory_context": {
            "experience": {
                "version": 1,
                "metadata": {"prompt_sha256": "abc", "fallback_used": False},
            }
        },
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "gpu": {
                        "devices": [
                            {"name": "GPU", "driver": "1", "pci_bus_id": "0"}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    audit = build_campaign_resource_audit(campaign)
    assert audit["campaign_wall_seconds"] == 60.0
    assert audit["action_latency_ms"]["mean"] == 100.0
    assert audit["action_requests_per_wall_second"] == 1 / 60
    assert audit["token_accounting"]["estimated_total_tokens"] == 14
    assert audit["token_accounting"]["status"] == (
        "heuristic_estimate_not_provider_usage"
    )
    assert audit["experience_revision_count"] == 1
    assert audit["gpu_identities"][0]["name"] == "GPU"
    assert audit["state_selection"] == {
        "matrix_unit_count": 1,
        "complete_latest_attempt_count": 1,
        "incomplete_latest_attempt_count": 0,
        "superseded_failed_state_rows": 0,
    }


def test_resource_audit_excludes_superseded_complete_attempt(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    run_id = "mixed__s1__a01"
    run_dir = campaign / "runs" / run_id
    run_dir.mkdir(parents=True)
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"2026-01-01T00:00:00Z\tm\tmixed\t1\t1\trunning\t{run_id}\t/x\t\t\n"
        f"2026-01-01T00:01:00Z\tm\tmixed\t1\t1\tcomplete\t{run_id}\t/x\t\t\n"
        "2026-01-01T00:02:00Z\tm\tmixed\t1\t2\trunning\t"
        "mixed__s1__a02\t/y\t\t\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    audit = build_campaign_resource_audit(campaign)

    assert audit["completed_leaf_count"] == 0
    assert audit["leaf_sources"] == []
    assert audit["state_selection"]["incomplete_latest_attempt_count"] == 1


@pytest.mark.parametrize(
    "extra_rows, message",
    [
        (
            "2026-01-01T00:02:00Z\tm\tmixed\t1\t2\trunning\tr2\t/y\t\t\n"
            "2026-01-01T00:03:00Z\tm\tmixed\t1\t2\tcomplete\tr2\t/y\t\t\n",
            "multiple completed attempts",
        ),
        (
            "2026-01-01T00:02:00Z\tm\tmixed\t1\t1\tfailed\tr1\t/x\texecution\t\n"
            "2026-01-01T00:03:00Z\tm\tmixed\t1\t1\tcomplete\tr1\t/x\t\t\n",
            "failed state precedes completion",
        ),
    ],
)
def test_resource_audit_rejects_ambiguous_completion_history(
    tmp_path: Path,
    extra_rows: str,
    message: str,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        "2026-01-01T00:00:00Z\tm\tmixed\t1\t1\trunning\tr1\t/x\t\t\n"
        "2026-01-01T00:01:00Z\tm\tmixed\t1\t1\tcomplete\tr1\t/x\t\t\n"
        + extra_rows,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        build_campaign_resource_audit(campaign)
