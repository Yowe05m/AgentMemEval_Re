"""
模块说明：本模块测试固定桌、换桌和报告重建的离线集成路径。
核心职责：确保 mock Provider 下能生成标准工件并重建报告。
输入与输出：输入临时目录配置，输出 pytest 断言结果。
依赖边界：调用公开 runner 和 reporting 接口。
不负责：不调用真实在线 API。
"""

import json
from pathlib import Path
from uuid import uuid4

from agentmemeval.config.loader import load_config
from agentmemeval.evaluation.reporting import rebuild_report
from agentmemeval.experiments.runner import run_resolved_config


def base_config(output_root: Path) -> dict[str, object]:
    """
    功能：生成测试基础配置。
    参数：
        output_root：临时输出目录。
    返回：配置字典。
    副作用：无。
    异常：无。
    设计说明：集成测试使用很小手数，避免拖慢默认 pytest。
    """

    return {
        "provider": {
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "max_retries": 0,
        },
        "table": {
            "starting_stack": 200,
            "small_blind": 1,
            "big_blind": 2,
            "max_raises_per_street": 3,
        },
        "agent": {
            "mechanism": "fact_expr_sync",
            "memory_scope": "per_agent",
            "top_k": 4,
            "window_size": 3,
        },
        "opponent_agent": {"mechanism": "no_memory", "memory_scope": "per_agent"},
        "heldout_agent": {"mechanism": "no_memory", "memory_scope": "per_agent"},
        "experiment": {
            "scenario": "fixed_evolving_table",
            "seed": 123,
            "output_root": str(output_root),
            "run_id": "fixed_test",
            "train_hands": 2,
            "test_hands": 1,
            "table_size": 3,
            "target_agent_id": "agent_00",
            "update_memory_train": True,
            "update_memory_test": False,
        },
    }


def test_fixed_table_run_and_report() -> None:
    """
    功能：验证固定桌训练、快照、泛化和报告重建。
    参数：
        无。
    返回：无。
    副作用：写临时工件。
    异常：断言失败时由 pytest 报告。
    设计说明：覆盖本任务要求的训练到 snapshot 到泛化测试闭环。
    """

    output_root = Path("tmp") / "test_outputs" / f"fixed_{uuid4().hex}"
    config = base_config(output_root)
    result = run_resolved_config(config)
    run_dir = Path(result.artifacts["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "protocol_audit.json").exists()
    assert (run_dir / "async_evidence_review_queue.json").exists()
    assert (run_dir / "memory_snapshots" / "agent_00_after_train.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "gpu" in manifest["metadata"]
    assert "dirty" in manifest["metadata"]["code"]
    assert "decision_system_sha256" in manifest["metadata"]["prompts"]
    assert manifest["metadata"]["protocol"]["run_mode"] == "smoke"
    rebuilt = rebuild_report(run_dir, big_blind=2)
    assert Path(rebuilt["report_path"]).exists()
    hands = [
        json.loads(line)
        for line in (run_dir / "hand_summaries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_hands = [hand for hand in hands if hand["stage"] == "train"]
    assert [hand["dealer_index"] for hand in train_hands] == [0, 1]
    assert [hand["hand_number"] for hand in train_hands] == [1, 2]
    assert train_hands[0]["small_blind_agent_id"] == "agent_01"
    assert train_hands[1]["small_blind_agent_id"] == "agent_02"
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    action_event = next(event for event in events if event.get("event") == "action")
    assert action_event["prompt"]["template_version"] == (
        "2026-07-19-v6-counterfactual-calibrated-memory"
    )
    assert len(action_event["prompt"]["user_sha256"]) == 64
    assert "call_cost" in action_event["call_risk"]
    assert "made_hand_class" in action_event["call_risk"]
    assert "stage_per_agent" in rebuilt["metrics"]["primary_metrics"]
    assert rebuilt["metrics"]["execution_health"]["valid"] is True
    assert rebuilt["metrics"]["run_validity"]["paper_eligible"] is False
    assert rebuilt["metrics"]["run_validity"]["status"] == "not_for_main_table"
    protocol = json.loads((run_dir / "protocol_audit.json").read_text(encoding="utf-8"))
    assert protocol["paper_evolving_roster_match"] is False
    assert protocol["dealer_rotation"] == "hand_index modulo table_size"


def test_mixed_exp1_checkpoints_every_agent_independently() -> None:
    output_root = Path("tmp") / "test_outputs" / f"mixed_{uuid4().hex}"
    config = load_config("configs/experiments/paper_exp1_mixed_mock.yaml")
    config["experiment"].update(
        {
            "output_root": str(output_root),
            "run_id": "mixed_test",
            "train_hands": 2,
            "checkpoint_interval": 1,
            "checkpoint_test_hands": 1,
            "statistical_plan_status": "pending_pilot_power_calibration",
            "primary_endpoint": "final_test_bb_per_100",
            "primary_estimand": "same_seed_table_run_mechanism_effect_vs_baseline",
            "primary_baseline_mechanism": "fact",
            "within_table_mechanism_aggregation": "arithmetic_mean",
            "multiple_comparison_method": "holm",
        }
    )
    result = run_resolved_config(config)
    run_dir = Path(result.artifacts["run_dir"])
    audit = json.loads((run_dir / "protocol_audit.json").read_text(encoding="utf-8"))
    checkpoints = json.loads(
        (run_dir / "checkpoint_generalization.json").read_text(encoding="utf-8")
    )

    assert audit["paper_evolving_roster_match"] is True
    assert audit["generalization_schedule"] == "every_1_train_hands_and_final"
    assert audit["strategy_risk_gate"] == "disabled"
    assert audit["strategy_risk_gate_applied"] is False
    assert audit["action_guard_scope"] == "legality_and_format_only"
    assert audit["checkpoint_cost_budget"] == {
        "checkpoint_count_per_seed": 2,
        "evaluation_target_count": 8,
        "checkpoint_evaluations_per_seed": 16,
        "checkpoint_generalization_hands_per_seed": 16,
        "seed_count": 5,
        "checkpoint_generalization_hands_all_seeds": 80,
    }
    assert set(audit["evaluation_target_ids"]) == set(audit["train_agent_mechanisms"])
    assert len(checkpoints["results"]) == 16
    assert all("generalization_gap_chip_delta" in item for item in checkpoints["results"])
    assert set(checkpoints["summary"]["1"]["by_mechanism"]) == {
        "fact",
        "expr",
        "fact_expr_sync",
        "fact_expr_async",
    }
    assert checkpoints["summary"]["1"]["paired_mechanism_effects"]
    assert "test_bb_per_100_mean" in checkpoints["summary"]["1"]["by_mechanism"]["fact"]
    estimand = result.metrics["primary_metrics"].get("table_run_estimand")
    assert estimand["independent_unit"] == "one complete table/run within seed"
    assert estimand["baseline_mechanism"] == "fact"
    assert result.aggregate_metrics["paired_estimand_descriptive"]["status"] == (
        "descriptive_only"
    )
    assert result.aggregate_metrics["main_table"]["status"] == (
        "no_paper_eligible_runs"
    )
    assert len(result.artifacts["checkpoint_snapshots"]) == 2
    assert all(len(paths) == 8 for paths in result.artifacts["checkpoint_snapshots"].values())


def test_rotating_20_agents_run() -> None:
    """
    功能：验证 20 Agent 换桌 smoke run。
    参数：
        无。
    返回：无。
    副作用：写临时工件。
    异常：断言失败时由 pytest 报告。
    设计说明：覆盖 20+ Agent、暴露统计和离线 mock 集成要求。
    """

    output_root = Path("tmp") / "test_outputs" / f"rotating_{uuid4().hex}"
    config = base_config(output_root)
    config["agent"] = {
        "mechanism": "fact_expr_async",
        "memory_scope": "per_agent",
        "top_k": 4,
        "window_size": 3,
        "sweep_every": 2,
        "evidence_k": 2,
    }
    config["personas"] = ["INTJ", "ENFP", "ISTP", "ESFJ"]
    config["experiment"] = {
        "scenario": "rotating_table",
        "seed": 123,
        "output_root": str(output_root),
        "run_id": "rotating_test",
        "agent_count": 20,
        "table_size": 4,
        "rounds": 1,
        "hands_per_table_round": 1,
        "rotation_mode": "balanced",
        "update_memory_train": True,
        "rebuy_busted": True,
    }
    result = run_resolved_config(config)
    run_dir = Path(result.artifacts["run_dir"])
    assert (run_dir / "exposure_stats.json").exists()
    assert len(list((run_dir / "memory_snapshots").glob("*_final.json"))) == 20
    assert result.metrics["run_counters"]["agents"] == 20
