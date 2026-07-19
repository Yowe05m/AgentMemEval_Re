from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmemeval.evaluation.relevance_review import (
    REVIEW_POLICY,
    audit_relevance_labels,
    build_relevance_review_pack,
)


def _campaign(
    tmp_path: Path,
    name: str,
    design: str,
    scores: tuple[float, ...] = (0.1, 0.3, 0.6, 0.9),
) -> Path:
    root = tmp_path / name
    run_id = "mixed__s1__a01"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": name,
                "campaign": {
                    "campaign_id": name,
                    "design": design,
                    "seeds": [1],
                    "conditions": [
                        {
                            "condition_id": "mixed",
                            "target_mechanism": "mixed",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        f"now\tmixed\tmixed\t1\t1\tcomplete\t{run_id}\t{run_dir}\t\t\n",
        encoding="utf-8",
    )
    events = []
    for index, score in enumerate(scores, 1):
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
                            "state_summary": (
                                "visible state\n"
                                "hand_outcome: win (净收益 +999)\n"
                                "showdown_visible_agent_ids: agent_01\n"
                                "summary: 最终净收益 +999"
                            ),
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
    campaigns = [
        _campaign(tmp_path, "pilot-p", "mixed_table"),
        _campaign(tmp_path, "pilot-e", "target_vs_seven_no_memory"),
    ]
    first = build_relevance_review_pack(campaigns, sample_size=3, sample_seed=7)
    second = build_relevance_review_pack(campaigns, sample_size=3, sample_seed=7)
    assert first == second
    assert first["sampled_row_count"] == 3
    assert first["population_query_count"] == 8
    assert "score" not in first["blind_rows"][0]
    assert "final_reward" not in first["blind_rows"][0]["record"]
    blind_state = first["blind_rows"][0]["record"]["state_summary"]
    assert "hand_outcome" not in blind_state
    assert "净收益" not in blind_state
    assert "showdown_visible_agent_ids" not in blind_state
    assert first["keyed_rows"][0]["record"]["state_summary"].find("hand_outcome") >= 0
    assert "score" in first["keyed_rows"][0]


def _audit_pack(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "task4_retrieval_relevance_review_pack_v1",
        "status": "pending_independent_human_labels",
        "policy": REVIEW_POLICY,
        "keyed_rows": rows,
        "query_max_scores": [0.8] * 200,
        "candidate_thresholds": [0.0, 0.8, 0.9],
        "sources": [
            {
                "campaign_id": "p",
                "design": "mixed_table",
                "matrix_complete": True,
                "expected_state_rows": 1,
                "completed_state_rows": 1,
                "campaign_manifest_sha256": "a" * 64,
                "state_tsv_sha256": "b" * 64,
                "event_sources": [
                    {"run_id": "p__s1__a01", "events_sha256": "c" * 64}
                ],
            },
            {
                "campaign_id": "e",
                "design": "target_vs_seven_no_memory",
                "matrix_complete": True,
                "expected_state_rows": 1,
                "completed_state_rows": 1,
                "campaign_manifest_sha256": "d" * 64,
                "state_tsv_sha256": "e" * 64,
                "event_sources": [
                    {"run_id": "e__s1__a01", "events_sha256": "f" * 64}
                ],
            },
        ],
    }


def test_audit_relevance_labels_freezes_lowest_eligible_threshold(
    tmp_path: Path,
) -> None:
    campaigns = [
        _campaign(
            tmp_path,
            "pilot-p",
            "mixed_table",
            scores=(0.0, 0.8) * 50,
        ),
        _campaign(
            tmp_path,
            "pilot-e",
            "target_vs_seven_no_memory",
            scores=(0.0, 0.8) * 50,
        ),
    ]
    pack = build_relevance_review_pack(campaigns, sample_size=200, sample_seed=7)
    rows = pack["keyed_rows"]
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
    assert audit["schema_version"] == "task4_retrieval_relevance_audit_v2"
    assert audit["source_designs"] == [
        "mixed_table",
        "target_vs_seven_no_memory",
    ]
    assert audit["source_rebuild_verified"] is True
    assert (
        audit["source_rebuild_content_sha256"]
        == audit["review_pack_content_sha256"]
    )


def test_audit_relevance_labels_rejects_model_labels() -> None:
    rows = [
        {"row_id": f"RR{index:04d}", "score": 0.8}
        for index in range(1, 201)
    ]
    pack = _audit_pack(rows)
    pack["candidate_thresholds"] = [0.0]
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


def test_build_relevance_review_pack_rejects_incomplete_source_matrix(
    tmp_path: Path,
) -> None:
    campaign_p = _campaign(tmp_path, "pilot-p", "mixed_table")
    manifest_path = campaign_p / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign"]["seeds"] = [1, 2]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    campaign_e = _campaign(
        tmp_path,
        "pilot-e",
        "target_vs_seven_no_memory",
    )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        build_relevance_review_pack([campaign_p, campaign_e])


def test_build_relevance_review_pack_requires_exactly_one_p_and_e_campaign(
    tmp_path: Path,
) -> None:
    campaigns = [
        _campaign(tmp_path, "pilot-p-1", "mixed_table"),
        _campaign(tmp_path, "pilot-p-2", "mixed_table"),
        _campaign(tmp_path, "pilot-e", "target_vs_seven_no_memory"),
    ]
    with pytest.raises(ValueError, match="exactly one complete Campaign P and E"):
        build_relevance_review_pack(campaigns)


def test_audit_relevance_labels_rejects_blind_projection_tampering(
    tmp_path: Path,
) -> None:
    campaigns = [
        _campaign(tmp_path, "pilot-p", "mixed_table"),
        _campaign(tmp_path, "pilot-e", "target_vs_seven_no_memory"),
    ]
    pack = build_relevance_review_pack(campaigns, sample_size=8, sample_seed=7)
    pack["blind_rows"][0]["query"] = "tampered query"
    audit = audit_relevance_labels(pack, [])
    assert any(
        "blind review rows do not match" in item for item in audit["blockers"]
    )


def test_audit_relevance_labels_rejects_deterministic_source_rebuild_mismatch(
    tmp_path: Path,
) -> None:
    campaigns = [
        _campaign(tmp_path, "pilot-p", "mixed_table"),
        _campaign(tmp_path, "pilot-e", "target_vs_seven_no_memory"),
    ]
    pack = build_relevance_review_pack(campaigns, sample_size=8, sample_seed=7)
    pack["keyed_rows"][0]["score"] = 0.123456
    audit = audit_relevance_labels(pack, [])
    assert audit["source_rebuild_verified"] is False
    assert any(
        "differs from deterministic source rebuild" in item
        for item in audit["blockers"]
    )


def test_audit_relevance_labels_rejects_invalid_source_hash() -> None:
    rows = [
        {"row_id": f"RR{index:04d}", "score": 0.8}
        for index in range(1, 201)
    ]
    pack = _audit_pack(rows)
    pack["sources"][0]["event_sources"][0]["events_sha256"] = "not-a-hash"
    labels = [
        {
            "row_id": row["row_id"],
            "label": "relevant",
            "reviewer_id": "human-1",
            "reviewer_type": "human",
        }
        for row in rows
    ]
    audit = audit_relevance_labels(pack, labels)
    assert audit["retrieval_threshold_status"] == "blocked"
    assert any("event evidence is incomplete" in item for item in audit["blockers"])
