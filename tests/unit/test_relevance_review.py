from __future__ import annotations

import json
from pathlib import Path

from agentmemeval.evaluation.relevance_review import (
    audit_relevance_labels,
    build_relevance_review_pack,
)


def _campaign(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    run_id = "mixed__s1__a01"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (root / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "pilot"}), encoding="utf-8"
    )
    (root / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"now\tmixed\tmixed\t1\t1\tcomplete\t{run_id}\t{run_dir}\t\t\n",
        encoding="utf-8",
    )
    events = []
    for index, score in enumerate((0.1, 0.3, 0.6, 0.9), 1):
        record_id = f"fact_{index}"
        events.append(
            {
                "agent_id": "agent_01",
                "stage": "train",
                "phase": "flop",
                "memory_context": {
                    "metadata": {
                        "mechanism": "fact",
                        "query": f"query {index}",
                        "retrieval_scores": [
                            {
                                "record_id": record_id,
                                "score": score,
                                "semantic": score,
                                "feature": 0.0,
                                "salience": 1.0,
                            }
                        ],
                    },
                    "facts": [
                        {
                            "record_id": record_id,
                            "state_summary": "visible state",
                            "action_summary": "call",
                            "features": ["phase:flop"],
                            "final_reward": 999,
                        }
                    ],
                },
            }
        )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    return root


def test_build_relevance_review_pack_is_deterministic_and_score_blind(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    first = build_relevance_review_pack([campaign], sample_size=3, sample_seed=7)
    second = build_relevance_review_pack([campaign], sample_size=3, sample_seed=7)
    assert first == second
    assert first["sampled_row_count"] == 3
    assert first["population_query_count"] == 4
    assert "score" not in first["blind_rows"][0]
    assert "final_reward" not in first["blind_rows"][0]["record"]
    assert "score" in first["keyed_rows"][0]


def test_audit_relevance_labels_freezes_lowest_eligible_threshold() -> None:
    rows = [
        {"row_id": f"RR{index:04d}", "score": 0.8}
        for index in range(1, 201)
    ]
    pack = {
        "keyed_rows": rows,
        "query_max_scores": [0.8] * 200,
        "candidate_thresholds": [0.0, 0.8, 0.9],
    }
    labels = [
        {
            "row_id": row["row_id"],
            "label": "relevant",
            "reviewer_id": "reviewer-a",
            "reviewer_type": "human",
            "comment": "",
        }
        for row in rows
    ]
    audit = audit_relevance_labels(pack, labels)
    assert audit["retrieval_threshold_status"] == "frozen"
    assert audit["minimum_retrieval_score"] == 0.0
    assert audit["blockers"] == []


def test_audit_relevance_labels_rejects_model_labels() -> None:
    rows = [
        {"row_id": f"RR{index:04d}", "score": 0.8}
        for index in range(1, 201)
    ]
    pack = {
        "keyed_rows": rows,
        "query_max_scores": [0.8] * 200,
        "candidate_thresholds": [0.0],
    }
    labels = [
        {
            "row_id": row["row_id"],
            "label": "relevant",
            "reviewer_id": "judge-model",
            "reviewer_type": "model",
            "comment": "",
        }
        for row in rows
    ]
    audit = audit_relevance_labels(pack, labels)
    assert audit["retrieval_threshold_status"] == "blocked"
    assert any("not declared as human-reviewed" in item for item in audit["blockers"])
