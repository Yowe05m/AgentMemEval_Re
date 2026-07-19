from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentmemeval.evaluation.pilot import (
    PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256,
    _completed_state_rows,
    build_pilot_freeze_proposal,
    build_pilot_freeze_proposal_from_paths,
    build_pilot_power_plan,
    build_pilot_prelaunch_code_audit,
    build_pilot_runtime_equivalence_audit,
    calibrate_behavior_thresholds,
)
from agentmemeval.evaluation.relevance_review import REVIEW_POLICY


def _p() -> dict[str, object]:
    return {
        "design": "mixed_table",
        "status": "descriptive_only",
        "completed_run_count": 3,
        "expected_run_count": 3,
        "runtime_homogeneity": {
            "homogeneous": True,
            "identity": {"code": [["commit", "same"], ["dirty", False]]},
        },
        "aggregate_metrics": {
            "paired_estimand_descriptive": {
                "status": "descriptive_only",
                "independent_seed_count": 3,
                "matched_seeds": [1, 2, 3],
                "design": "A7-R_same_seed_table_run_paired_mechanism_effect",
                "endpoint": "final_test_bb_per_100",
                "baseline_mechanism": "fact",
                "multiple_comparison_method": "holm",
                "effects_by_mechanism": {
                    "expr": [1.0, 3.0, 5.0],
                    "fact_expr_async": [3.0, 2.0, 1.0],
                    "fact_expr_sync": [2.0, 4.0, 6.0],
                }
            }
        },
    }


def _e() -> dict[str, object]:
    return {
        "design": "target_vs_seven_no_memory",
        "estimand": "same_seed_cross_condition_target_effect_vs_no_memory",
        "baseline_condition_id": "no_memory_target",
        "multiple_comparison_method": "holm",
        "status": "descriptive_only",
        "completed_run_count": 15,
        "expected_run_count": 15,
        "runtime_homogeneity": {
            "homogeneous": True,
            "identity": {"code": [["commit", "same"], ["dirty", False]]},
        },
        "primary_endpoint": "final_test_bb_per_100",
        "paired_comparisons": {
            "fact_target": {
                "metrics": {
                    "final_test_bb_per_100": {
                        "matched_seeds": [1, 2, 3],
                        "effects": [2.0, 5.0, 8.0],
                    }
                }
            },
            "expr_target": {
                "metrics": {
                    "final_test_bb_per_100": {
                        "matched_seeds": [1, 2, 3],
                        "effects": [1.0, 4.0, 7.0],
                    }
                }
            },
            "sync_target": {
                "metrics": {
                    "final_test_bb_per_100": {
                        "matched_seeds": [1, 2, 3],
                        "effects": [3.0, 6.0, 9.0],
                    }
                }
            },
            "async_target": {
                "metrics": {
                    "final_test_bb_per_100": {
                        "matched_seeds": [1, 2, 3],
                        "effects": [4.0, 2.0, 5.0],
                    }
                }
            },
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
    assert plan["runtime_identity_mode"] == "exact_identity"
    assert plan["formal_homogeneity_not_granted"] is False


def test_pilot_power_plan_blocks_incomplete_matrix() -> None:
    campaign_e = _e()
    campaign_e["completed_run_count"] = 14
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert plan["required_seed_pairs_primary_max_across_p_and_e"] is None
    assert "campaign_e matrix is incomplete: 14/15" in plan["blockers"]


def test_pilot_power_plan_blocks_cross_campaign_runtime_mismatch() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "different"], ["dirty", False]]
    }
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert "campaign P/E runtime identities differ" in plan["blockers"]


def test_pilot_power_plan_blocks_missing_contrast_or_seed_mismatch() -> None:
    campaign_e = _e()
    del campaign_e["paired_comparisons"]["async_target"]  # type: ignore[index]
    campaign_e["paired_comparisons"]["fact_target"]["metrics"][  # type: ignore[index]
        "final_test_bb_per_100"
    ]["matched_seeds"] = [1, 3, 2]
    plan = build_pilot_power_plan(_p(), campaign_e)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert any("campaign_e contrasts mismatch" in item for item in plan["blockers"])
    assert any("matched seeds mismatch" in item for item in plan["blockers"])


def test_pilot_power_plan_accepts_verified_orchestration_only_equivalence() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "later"], ["dirty", False]]
    }
    changed_paths = [
        "tools/task4/gate_campaign_p_before_e.py",
        "configs/campaigns/"
        "task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml",
    ]
    audit = build_pilot_runtime_equivalence_audit(
        _p(),
        campaign_e,
        changed_paths,
    )
    assert audit["status"] == (
        "verified_execution_runtime_equivalent_for_pilot_power_only"
    )
    assert audit["formal_homogeneity_not_granted"] is True
    plan = build_pilot_power_plan(_p(), campaign_e, audit)
    assert plan["status"] == (
        "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
    )
    assert plan["blockers"] == []
    assert plan["runtime_identity_mode"] == (
        "pilot_only_verified_execution_equivalence"
    )
    assert plan["formal_homogeneity_not_granted"] is True


def test_pilot_runtime_equivalence_accepts_registered_post_p_v7_changes() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "later"], ["dirty", False]]
    }
    changed_paths = [
        "README.md",
        "configs/campaigns/"
        "task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml",
        "configs/campaigns/task4_campaign_p_strict_model_substituted.yaml",
        "src/agentmemeval/cli/main.py",
        "src/agentmemeval/evaluation/aggregation.py",
        "src/agentmemeval/evaluation/campaign_progress.py",
        "src/agentmemeval/evaluation/campaign_reporting.py",
        "src/agentmemeval/evaluation/formal_freeze.py",
        "src/agentmemeval/evaluation/pilot.py",
        "src/agentmemeval/evaluation/relevance_review.py",
        "src/agentmemeval/evaluation/runtime_lock.py",
        "src/agentmemeval/evaluation/study_reporting.py",
        "src/agentmemeval/experiments/admission.py",
        "src/agentmemeval/experiments/campaign.py",
        "src/agentmemeval/storage/archive.py",
        "src/agentmemeval/storage/campaign_seal.py",
        "src/agentmemeval/storage/run_map.py",
        "src/agentmemeval/storage/snapshot_archive.py",
        "tests/unit/test_archive_manifest.py",
        "tests/unit/test_campaign_p_gate.py",
        "tests/unit/test_campaign_progress.py",
        "tests/unit/test_campaign_reporting.py",
        "tests/unit/test_campaign_seal.py",
        "tests/unit/test_config_validation.py",
        "tests/unit/test_formal_freeze.py",
        "tests/unit/test_pilot_power_plan.py",
        "tests/unit/test_protocol_admission.py",
        "tests/unit/test_relevance_review.py",
        "tests/unit/test_run_map.py",
        "tests/unit/test_runtime_lock.py",
        "tests/unit/test_snapshot_archive.py",
        "tests/unit/test_statistics_and_degeneracy.py",
        "tests/unit/test_study_reporting.py",
        "tests/integration/test_campaign.py",
        "tools/task4/audit_campaign_archive_handoff.py",
        "tools/task4/audit_pilot_prelaunch_code_paths.py",
        "tools/task4/audit_pilot_runtime_equivalence.py",
        "tools/task4/audit_campaign_seal.py",
        "tools/task4/build_formal_runtime_lock.py",
        "tools/task4/build_study_report.py",
        "tools/task4/campaign_progress.py",
        "tools/task4/gate_campaign_p_before_e.py",
        "tools/task4/retrieval_relevance_review.py",
        "tools/task4/snapshot_archive.py",
        "tools/task4/start_campaign_e_v7_pilot.sh",
    ]
    audit = build_pilot_runtime_equivalence_audit(
        _p(),
        campaign_e,
        changed_paths,
        dict(PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256),
    )
    assert audit["status"] == (
        "verified_execution_runtime_equivalent_for_pilot_power_only"
    )
    assert audit["disallowed_changed_paths"] == []
    assert audit["changed_paths"] == sorted(changed_paths)
    assert audit["required_diff_sha256_match"] is True


def test_pilot_prelaunch_code_audit_is_narrow_and_does_not_grant_runtime_equivalence(
) -> None:
    safe = build_pilot_prelaunch_code_audit(
        "a" * 40,
        "b" * 40,
        [
            "src/agentmemeval/storage/snapshot_archive.py",
            "tools/task4/start_campaign_e_v7_pilot.sh",
        ],
    )
    assert safe["status"] == (
        "verified_code_paths_safe_to_launch_campaign_e_pilot"
    )
    assert safe["runtime_equivalence_not_yet_granted"] is True
    assert safe["formal_homogeneity_not_granted"] is True
    assert safe["disallowed_changed_paths"] == []

    scheduler_safe = build_pilot_prelaunch_code_audit(
        "a" * 40,
        "b" * 40,
        ["src/agentmemeval/experiments/campaign.py"],
        dict(PILOT_RUNTIME_EQUIVALENCE_REQUIRED_DIFF_SHA256),
    )
    assert scheduler_safe["status"] == (
        "verified_code_paths_safe_to_launch_campaign_e_pilot"
    )
    scheduler_blocked = build_pilot_prelaunch_code_audit(
        "a" * 40,
        "b" * 40,
        ["src/agentmemeval/experiments/campaign.py"],
        {"src/agentmemeval/experiments/campaign.py": "wrong"},
    )
    assert scheduler_blocked["status"] == "no_go_code_paths_changed"
    assert any("diff hash mismatch" in item for item in scheduler_blocked["blockers"])

    blocked = build_pilot_prelaunch_code_audit(
        "short",
        "b" * 40,
        [
            "../src/agentmemeval/storage/snapshot_archive.py",
            "src/agentmemeval/experiments/fixed_table.py",
        ],
    )
    assert blocked["status"] == "no_go_code_paths_changed"
    assert blocked["invalid_changed_paths"] == [
        "../src/agentmemeval/storage/snapshot_archive.py"
    ]
    assert blocked["disallowed_changed_paths"] == [
        "src/agentmemeval/experiments/fixed_table.py"
    ]
    assert any("full lowercase SHA-1" in item for item in blocked["blockers"])


def test_campaign_e_start_requires_verified_campaign_p_archive_handoff() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = (
        repo / "tools" / "task4" / "start_campaign_e_v7_pilot.sh"
    ).read_text(encoding="utf-8")

    assert 'if [ "$#" -ne 6 ]; then' in script
    assert 'test -f "$p_seal_readiness"' in script
    assert 'test -f "$p_snapshot_receipt"' in script
    assert 'test ! -e "$archive_handoff_audit"' in script
    assert 'task4_campaign_p_before_e_gate_v7' in script
    audit_index = script.index("tools/task4/audit_campaign_archive_handoff.py")
    campaign_index = script.index("campaign \\\n")
    assert audit_index < campaign_index
    assert (
        'handoff["campaign_manifest_sha256"] == audit["campaign_manifest_sha256"]'
        in script
    )
    assert 'handoff["state_tsv_sha256"] == audit["state_tsv_sha256"]' in script


def test_pilot_runtime_equivalence_rejects_execution_relevant_change() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "later"], ["dirty", False]]
    }
    audit = build_pilot_runtime_equivalence_audit(
        _p(),
        campaign_e,
        ["src/agentmemeval/memory/mechanisms.py"],
    )
    assert audit["status"] == "no_go_runtime_equivalence_unverified"
    assert audit["disallowed_changed_paths"] == [
        "src/agentmemeval/memory/mechanisms.py"
    ]
    plan = build_pilot_power_plan(_p(), campaign_e, audit)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert "pilot runtime-equivalence audit is not verified" in plan["blockers"]


def test_pilot_power_plan_rejects_stale_runtime_equivalence_audit() -> None:
    campaign_e = _e()
    campaign_e["runtime_homogeneity"]["identity"] = {  # type: ignore[index]
        "code": [["commit", "later"], ["dirty", False]]
    }
    audit = build_pilot_runtime_equivalence_audit(
        _p(),
        campaign_e,
        ["tools/task4/gate_campaign_p_before_e.py"],
    )
    audit["campaign_e_code_sha"] = "stale"
    plan = build_pilot_power_plan(_p(), campaign_e, audit)
    assert plan["status"] == "blocked_invalid_or_incomplete_pilot"
    assert (
        "pilot runtime-equivalence audit mismatch for campaign_e_code_sha"
        in plan["blockers"]
    )


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
        "schema_version": "task4_retrieval_relevance_audit_v2",
        "review_status": "human_labels_verified",
        "retrieval_threshold_status": "frozen",
        "minimum_retrieval_score": 0.42,
        "sampled_row_count": 200,
        "labeled_row_count": 200,
        "review_pack_content_sha256": "a" * 64,
        "review_policy_sha256": hashlib.sha256(
            json.dumps(
                REVIEW_POLICY,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_campaign_count": 2,
        "source_designs": [
            "mixed_table",
            "target_vs_seven_no_memory",
        ],
        "source_evidence": [
            {
                "campaign_dir": "/evidence/p",
                "design": "mixed_table",
                "matrix_complete": True,
                "expected_state_rows": 1,
                "completed_state_rows": 1,
                "campaign_manifest_sha256": "b" * 64,
                "state_tsv_sha256": "c" * 64,
                "event_sources": [
                    {
                        "run_id": "p__s1__a01",
                        "events_sha256": "d" * 64,
                    }
                ],
            },
            {
                "campaign_dir": "/evidence/e",
                "design": "target_vs_seven_no_memory",
                "matrix_complete": True,
                "expected_state_rows": 1,
                "completed_state_rows": 1,
                "campaign_manifest_sha256": "e" * 64,
                "state_tsv_sha256": "f" * 64,
                "event_sources": [
                    {
                        "run_id": "e__s1__a01",
                        "events_sha256": "a" * 64,
                    }
                ],
            },
        ],
        "source_rebuild_verified": True,
        "source_rebuild_content_sha256": "a" * 64,
        "input_evidence": {
            "review_key_sha256": "b" * 64,
            "labels_sha256": "c" * 64,
            "label_row_count": 200,
            "human_reviewer_count": 1,
            "human_reviewer_ids_sha256": ["d" * 64],
        },
        "blockers": [],
    }


def _execution_health(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "valid": True,
        "status": "passed",
        "fallback_count": 0,
        "memory_revision_fallback_count": 0,
        "reward_conservation_violation_count": 0,
        "stack_conservation_violation_count": 0,
    }
    result.update(overrides)
    return result


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
            "execution_health": _execution_health(),
        }
        for _ in metrics
    ]
    e_audits = [
        {
            "evaluation_target_ids": ["fact_00"],
            "execution_health": _execution_health(),
        }
        for _ in range(15)
    ]
    proposal = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 5, e_audits, _review()
    )
    assert proposal["status"] == "ready_to_generate_immutable_formal_configs"
    assert proposal["retrieval_freeze"] == {
        "retrieval_threshold_status": "frozen",
        "minimum_retrieval_score": 0.42,
        "reason": "independent outcome-blind human relevance review",
        "review_audit_schema_version": "task4_retrieval_relevance_audit_v2",
        "source_designs": [
            "mixed_table",
            "target_vs_seven_no_memory",
        ],
        "review_key_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
    }
    p_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": _execution_health(valid=False),
    }
    blocked = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 5, e_audits, _review()
    )
    assert blocked["status"] == "no_go_pilot_freeze_blocked"
    assert blocked["execution_blockers"]
    p_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": _execution_health(),
    }
    e_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": _execution_health(valid=False),
    }
    blocked_e = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 5, e_audits, _review()
    )
    assert "campaign_e run 0 execution health is not valid" in blocked_e[
        "execution_blockers"
    ]
    e_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": _execution_health(fallback_count=1),
    }
    explicit_fallback_blocked = build_pilot_freeze_proposal(
        _p(), _e(), metrics, p_audits, metrics * 5, e_audits, _review()
    )
    assert "campaign_e run 0 execution fallback_count: 1" in (
        explicit_fallback_blocked["execution_blockers"]
    )
    e_audits[0] = {
        "evaluation_target_ids": ["fact_00"],
        "execution_health": _execution_health(),
    }
    fallback_metrics = [_metrics() for _ in range(15)]
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


def test_freeze_proposal_blocks_campaign_e_target_behavior_degeneracy() -> None:
    p_metrics = [_metrics(), _metrics(), _metrics()]
    e_metrics = [_metrics(fold_rate=1.0) for _ in range(15)]
    p_audits = [
        {
            "evaluation_target_ids": ["fact_00"],
            "execution_health": _execution_health(),
        }
        for _ in p_metrics
    ]
    e_audits = [
        {
            "evaluation_target_ids": ["fact_00"],
            "execution_health": _execution_health(),
        }
        for _ in e_metrics
    ]
    proposal = build_pilot_freeze_proposal(
        _p(),
        _e(),
        p_metrics,
        p_audits,
        e_metrics,
        e_audits,
        _review(),
    )
    assert proposal["status"] == "no_go_pilot_freeze_blocked"
    assert proposal["behavior_freeze"]["status"] == (
        "blocked_pilot_behavior_degenerate"
    )
    assert any(
        "campaign_e run" in blocker
        for blocker in proposal["behavior_freeze"]["blockers"]
    )


def test_freeze_path_loader_accepts_superseded_failed_attempt(tmp_path: Path) -> None:
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
    state_lines.append(
        f"t\tmixed\tmixed\t0\t1\tfailed\tpartial\t{tmp_path / 'partial'}"
        "\tinfrastructure\tpartial"
    )
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
                    "execution_health": _execution_health(),
                }
            ),
            encoding="utf-8",
        )
        attempt = 2 if index == 0 else 1
        state_lines.append(
            f"t\tmixed\tmixed\t{index}\t{attempt}\tcomplete\tr{index}\t"
            f"{run_dir}\t\t"
        )
    (p_dir / "state.tsv").write_text(
        "\n".join(state_lines) + "\n", encoding="utf-8"
    )
    e_state_lines = [state_lines[0]]
    for index in range(15):
        run_dir = tmp_path / f"e-run-{index}"
        run_dir.mkdir()
        (run_dir / "protocol_audit.json").write_text(
            json.dumps(
                {
                    "evaluation_target_ids": ["fact_00"],
                    "execution_health": _execution_health(),
                }
            ),
            encoding="utf-8",
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
    assert proposal["campaign_p_evidence"]["failed_state_rows"] == 1
    assert proposal["campaign_p_evidence"]["superseded_failed_state_rows"] == 1
    assert proposal["campaign_p_evidence"]["latest_failed_matrix_units"] == 0
    assert proposal["campaign_e_evidence"]["completed_state_rows"] == 15
    assert len(proposal["campaign_p_leaf_evidence"]) == 3
    assert len(proposal["campaign_e_leaf_evidence"]) == 15
    assert proposal["campaign_p_aggregate_evidence"]["sha256"]
    assert proposal["campaign_e_aggregate_evidence"]["sha256"]
    assert proposal["retrieval_review_evidence"]["sha256"]


def test_freeze_state_loader_rejects_stale_complete_after_failed_retry(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        "t\tmixed\tmixed\t7\t1\tcomplete\told\t/old\t\t\n"
        "t\tmixed\tmixed\t7\t2\tfailed\tnew\t/new\tinfrastructure\tfailed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latest attempt is not complete"):
        _completed_state_rows(campaign_dir, "campaign_p")


def test_freeze_state_loader_rejects_failed_then_complete_same_attempt(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        "t\tmixed\tmixed\t7\t1\tfailed\trun\t/run\tinfrastructure\tfailed\n"
        "t\tmixed\tmixed\t7\t1\tcomplete\trun\t/run\t\t\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed state preceding completion"):
        _completed_state_rows(campaign_dir, "campaign_p")


def test_freeze_state_loader_rejects_multiple_completed_attempts(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "state.tsv").write_text(
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
        "t\tmixed\tmixed\t7\t1\tcomplete\told\t/old\t\t\n"
        "t\tmixed\tmixed\t7\t2\tcomplete\tnew\t/new\t\t\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple completed attempts"):
        _completed_state_rows(campaign_dir, "campaign_p")


def test_freeze_path_loader_prefers_relocated_campaign_leaf(tmp_path: Path) -> None:
    p_dir = tmp_path / "campaign-p"
    e_dir = tmp_path / "campaign-e"
    p_dir.mkdir()
    e_dir.mkdir()
    header = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage\n"
    )
    for campaign_dir, prefix, count in ((p_dir, "p", 3), (e_dir, "e", 15)):
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
                        "execution_health": _execution_health(),
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
