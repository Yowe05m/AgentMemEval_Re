from __future__ import annotations

import json
from pathlib import Path

from agentmemeval.evaluation.pilot import (
    build_pilot_freeze_proposal,
    build_pilot_freeze_proposal_from_paths,
    build_pilot_power_plan,
    calibrate_behavior_thresholds,
)


def _p() -> dict[str, object]:
    return {
        "status": "descriptive_only",
        "completed_run_count": 3,
        "expected_run_count": 3,
        "runtime_homogeneity": {
            "homogeneous": True,
            "identity": {"code": [["commit", "same"], ["dirty", False]]},
        },
        "aggregate_metrics": {
            "paired_estimand_descriptive": {
                "effects_by_mechanism": {
                    "expr": [1.0, 3.0, 5.0],
                    "fact_expr_sync": [2.0, 4.0, 6.0],
                }
            }
        },
    }


def _e() -> dict[str, object]:
    return {
        "status": "descriptive_only",
        "completed_run_count": 6,
        "expected_run_count": 6,
        "runtime_homogeneity": {
            "homogeneous": True,
            "identity": {"code": [["commit", "same"], ["dirty", False]]},
        },
        "primary_endpoint": "final_test_bb_per_100",
        "paired_comparisons": {
            "fact_target": {
                "metrics": {
                    "final_test_bb_per_100": {"effects": [2.0, 5.0, 8.0]}
                }
            }
        },
    }


def test_pilot_power_plan_uses_max_requirement_without_capping() -> None:
    plan = build_pilot_power_plan(_p(), _e())
    assert plan["status"] == (
        "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
    )
    assert plan["primary_mde_bb_per_100"] == 5.0
    assert plan["sensitivity_mdes_bb_per_100"] == [3.0, 5.0, 10.0]
    requirements = [
        item["sensitivity_by_mde_bb_per_100"]["5.0"][
            "required_seed_pairs_normal_approximation"
        ]
        for item in plan["contrasts"].values()
    ]
    assert plan["required_seed_pairs_primary_max_across_p_and_e"] == max(requirements)
    assert plan["no_silent_resource_cap"] is True


def test_pilot_power_plan_blocks_incomplete_matrix() -> None:
    campaign_e = _e()
    campaign_e["completed_run_count"] = 5
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert plan["required_seed_pairs_primary_max_across_p_and_e"] is None
    assert "campaign_e matrix is incomplete: 5/6" in plan["blockers"]


def test_pilot_power_plan_blocks_cross_campaign_runtime_mismatch() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "different"], ["dirty", False]]
    }
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert "campaign P/E runtime identities differ" in plan["blockers"]


def _metrics(
    *,
    fold_rate: float = 0.40,
    heldout_fold_rate: float | None = None,
) -> dict[str, object]:
    values = {
        "vpip": 0.30,
        "fold_rate": fold_rate,
        "voluntary_participation_hands": 10,
        "all_in_hand_rate": 0.05,
        "bust_hand_rate": 0.01,
        "hand_reward_sensitivity": {"share_of_absolute_reward_activity": 0.30},
        "memory": {
            "empty_retrieval_rate": 0.20,
            "max_structural_signature_share": 0.15,
        },
    }
    test = {"fact_00": dict(values)}
    if heldout_fold_rate is not None:
        heldout = dict(values)
        heldout["fold_rate"] = heldout_fold_rate
        test["heldout_fact_00_00"] = heldout
    return {
        "primary_metrics": {
            "stage_per_agent": {
                "train": {"fact_00": dict(values)},
                "test": test,
            }
        }
    }


def _review() -> dict[str, object]:
    return {
        "schema_version": "task4_retrieval_relevance_audit_v1",
        "review_status": "human_labels_verified",
        "retrieval_threshold_status": "frozen",
        "minimum_retrieval_score": 0.42,
        "blockers": [],
    }


def test_behavior_freeze_uses_quantiles_and_domain_caps() -> None:
    freeze = calibrate_behavior_thresholds([_metrics(), _metrics(), _metrics()])
    assert freeze["status"] == "frozen"
    assert freeze["thresholds"]["min_vpip"] == 0.28
    assert freeze["thresholds"]["max_fold_rate"] == 0.42
    assert freeze["thresholds"]["min_voluntary_participation_hands"] == 1

    degenerate = calibrate_behavior_thresholds(
        [_metrics(fold_rate=1.0), _metrics(fold_rate=1.0)]
    )
    assert degenerate["thresholds"]["max_fold_rate"] == 0.98
    assert degenerate["status"] == "blocked_pilot_behavior_degenerate"


def test_pilot_is_judged_by_hard_domain_gate_not_its_frozen_quantile() -> None:
    freeze = calibrate_behavior_thresholds(
        [*[_metrics(fold_rate=0.40) for _ in range(20)], _metrics(fold_rate=0.95)]
    )
    assert freeze["thresholds"]["max_fold_rate"] < 0.95
    assert freeze["pilot_domain_gate_thresholds"]["max_fold_rate"] == 0.98
    assert freeze["status"] == "frozen"


def test_behavior_freeze_ignores_heldout_opponents() -> None:
    freeze = calibrate_behavior_thresholds(
        [
            _metrics(heldout_fold_rate=1.0),
            _metrics(heldout_fold_rate=1.0),
        ],
        [["fact_00"], ["fact_00"]],
    )
    assert freeze["status"] == "frozen"
    assert freeze["sample_counts"]["fold_rate"] == 4
    assert freeze["evaluated_agent_ids_by_run"] == [["fact_00"], ["fact_00"]]


def test_freeze_proposal_requires_power_behavior_and_execution() -> None:
    metrics = [_metrics(), _metrics(), _metrics()]
    p_audits = [
        {
            "evaluation_target_ids": ["fact_00"],
            "execution_health": {"valid": True},
        }
        for _ in metrics
    ]
    e_audits = [{"execution_health": {"valid": True}} for _ in range(6)]
    proposal = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 2, e_audits, _review()
    )
    assert proposal["status"] == "ready_to_generate_immutable_formal_configs"
    assert proposal["retrieval_freeze"] == {
        "retrieval_threshold_status": "frozen",
        "minimum_retrieval_score": 0.42,
        "reason": "independent outcome-blind human relevance review",
        "review_audit_schema_version": "task4_retrieval_relevance_audit_v1",
    }
    p_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": {"valid": False},
    }
    blocked = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 2, e_audits, _review()
    )
    assert blocked["status"] == "no_go_pilot_freeze_blocked"
    assert blocked["execution_blockers"]
    p_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": {"valid": True},
    }
    e_audits[0] = {"execution_health": {"valid": False}}
    blocked_e = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 2, e_audits, _review()
    )
    assert "campaign_e run 0 execution health is not valid" in blocked_e[
        "execution_blockers"
    ]
    fallback_metrics = [_metrics() for _ in range(6)]
    fallback_metrics[0]["primary_metrics"]["stage_per_agent"]["train"]["fact_00"][  # type: ignore[index]
        "memory"
    ]["revision_fallback_count"] = 1
    revision_blocked = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, fallback_metrics, e_audits, _review()
    )
    assert (
        "campaign_e run 0 used deterministic experience revision fallback"
        in revision_blocked["execution_blockers"]
    )


def test_freeze_path_loader_ignores_noncomplete_state_rows(tmp_path: Path) -> None:
    p_path = tmp_path / "p.json"
    e_path = tmp_path / "e.json"
    p_dir = tmp_path / "campaign-p"
    e_dir = tmp_path / "campaign-e"
    p_dir.mkdir()
    e_dir.mkdir()
    p_path.write_text(json.dumps(_p()), encoding="utf-8")
    e_path.write_text(json.dumps(_e()), encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(_review()), encoding="utf-8")
    state_lines = [
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage"
    ]
    for index in range(3):
        run_dir = tmp_path / f"p-run-{index}"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(
            json.dumps(_metrics()), encoding="utf-8"
        )
        (run_dir / "protocol_audit.json").write_text(
            json.dumps(
                {
                    "evaluation_target_ids": ["fact_00"],
                    "execution_health": {"valid": True},
                }
            ),
            encoding="utf-8",
        )
        state_lines.append(
            f"t\tmixed\tmixed\t{index}\t1\tcomplete\tr{index}\t{run_dir}\t\t"
        )
    state_lines.append(
        f"t\tmixed\tmixed\t99\t1\tinterrupted\tpartial\t{tmp_path / 'partial'}"
        "\tsuperseded\tpartial"
    )
    (p_dir / "state.tsv").write_text(
        "\n".join(state_lines) + "\n", encoding="utf-8"
    )
    e_state_lines = [state_lines[0]]
    for index in range(6):
        run_dir = tmp_path / f"e-run-{index}"
        run_dir.mkdir()
        (run_dir / "protocol_audit.json").write_text(
            json.dumps({"execution_health": {"valid": True}}), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(_metrics()), encoding="utf-8"
        )
        e_state_lines.append(
            f"t\tcondition-{index}\ttarget\t{index}\t1\tcomplete\te{index}"
            f"\t{run_dir}\t\t"
        )
    (e_dir / "state.tsv").write_text(
        "\n".join(e_state_lines) + "\n", encoding="utf-8"
    )
    proposal = build_pilot_freeze_proposal_from_paths(
        p_path, e_path, p_dir, e_dir, review_path
    )
    assert proposal["status"] == "ready_to_generate_immutable_formal_configs"
    assert proposal["campaign_p_evidence"]["completed_state_rows"] == 3
    assert proposal["campaign_p_evidence"]["ignored_noncomplete_state_rows"] == 1
    assert proposal["campaign_e_evidence"]["completed_state_rows"] == 6


def test_freeze_path_loader_prefers_relocated_campaign_leaf(tmp_path: Path) -> None:
    p_dir = tmp_path / "campaign-p"
    e_dir = tmp_path / "campaign-e"
    p_dir.mkdir()
    e_dir.mkdir()
    header = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
    )
    for campaign_dir, prefix, count in ((p_dir, "p", 3), (e_dir, "e", 6)):
        rows = [header.rstrip("\n")]
        for index in range(count):
            run_id = f"{prefix}{index}"
            run_dir = campaign_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(_metrics()), encoding="utf-8"
            )
            (run_dir / "protocol_audit.json").write_text(
                json.dumps(
                    {
                        "evaluation_target_ids": ["fact_00"],
                        "execution_health": {"valid": True},
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                f"t\tc{index}\ttarget\t{index}\t1\tcomplete\t{run_id}\t"
                f"/missing/server/{run_id}\t\t"
            )
        (campaign_dir / "state.tsv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    p_path = tmp_path / "p.json"
    e_path = tmp_path / "e.json"
    review_path = tmp_path / "review.json"
    p_path.write_text(json.dumps(_p()), encoding="utf-8")
    e_path.write_text(json.dumps(_e()), encoding="utf-8")
    review_path.write_text(json.dumps(_review()), encoding="utf-8")
    proposal = build_pilot_freeze_proposal_from_paths(
        p_path, e_path, p_dir, e_dir, review_path
    )
    assert proposal["status"] == "ready_to_generate_immutable_formal_configs"
