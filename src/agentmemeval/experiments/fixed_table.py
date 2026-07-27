"""
模块说明：本模块实现固定桌训练与泛化测试场景。
核心职责：跑通论文 evolving table 与 generalization table 的最小闭环。
输入与输出：输入 ExperimentContext，输出 ExperimentResult 和标准工件。
依赖边界：依赖 Agent 构建、单手运行器、评估和存储层。
不负责：不实现换桌排程，不执行真实在线 Provider smoke。
"""

from __future__ import annotations

import copy
import statistics
from itertools import combinations
from typing import Any

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.agents.llm_agent import build_agent
from agentmemeval.analysis.plots import generate_audit_plots, plot_stack_curves
from agentmemeval.config.loader import resolve_checkpoint_set, resolve_heldout_table_set
from agentmemeval.core.domain import ExperimentResult, MemorySnapshot
from agentmemeval.core.errors import ConfigError
from agentmemeval.core.seeds import derive_seed
from agentmemeval.evaluation.aggregation import (
    aggregate_metrics,
    build_table_run_estimand,
)
from agentmemeval.evaluation.degeneracy import (
    build_run_validity,
    evaluate_behavior_health,
    evaluate_execution_health,
)
from agentmemeval.evaluation.metrics import compute_metrics
from agentmemeval.evaluation.reporting import build_report_text
from agentmemeval.experiments.context import ExperimentContext
from agentmemeval.experiments.formal_protocol import (
    build_clone_audit,
    build_heldout_schedule_manifest,
    clone_memory_branches,
    derive_formal_hand_seed,
    sha256_json,
    verify_schedule_manifest,
)
from agentmemeval.experiments.table_play import run_single_hand
from agentmemeval.storage.snapshots import load_snapshot


class FixedTableScenario:
    """
    功能：运行固定桌训练和可选泛化测试。
    参数：无。
    返回：场景实例。
    副作用：run 时写入工件。
    异常：配置错误由下游模块抛出。
    设计说明：训练和测试阶段共享同一单手运行器，记忆冻结由 update_memory 控制。
    """

    name = "fixed_evolving_table"

    def run(self, context: ExperimentContext) -> ExperimentResult:
        """
        功能：执行固定桌场景。
        参数：
            context：实验上下文。
        返回：ExperimentResult。
        副作用：写入 manifest、事件、摘要、快照、指标、报告。
        异常：配置、Provider 或环境错误向上抛出。
        设计说明：先训练目标 Agent，再将快照带入未见对手泛化桌。
        """

        config = context.config
        artifacts = context.artifacts
        artifacts.write_manifest()
        exp = config["experiment"]
        table = config.get("table", {})
        small_blind = int(table.get("small_blind", 1))
        big_blind = int(table.get("big_blind", 2))
        max_raises = int(table.get("max_raises_per_street", 4))
        starting_stack = int(table.get("starting_stack", 1000))
        table_size = int(exp.get("table_size", 4))
        seed = int(exp["seed"])
        train_hands = int(exp.get("train_hands", 8))
        test_hands = int(exp.get("test_hands", 4))
        update_train = bool(exp.get("update_memory_train", True))
        update_test = bool(exp.get("update_memory_test", False))
        target_id = str(exp.get("target_agent_id", "agent_00"))
        train_ids, agent_configs = _resolve_train_roster(config, table_size, target_id)
        if target_id not in train_ids:
            target_id = train_ids[0]
        table_size = len(train_ids)
        evaluation_targets = _resolve_evaluation_targets(exp, train_ids, target_id)
        checkpoint_interval = max(0, int(exp.get("checkpoint_interval", 0)))
        checkpoint_set = resolve_checkpoint_set(exp)
        heldout_table_set = resolve_heldout_table_set(exp)
        checkpoint_test_hands = int(exp.get("checkpoint_test_hands", test_hands))
        table_lifecycle = str(table.get("lifecycle", "tournament_elimination"))
        if table_lifecycle not in {"tournament_elimination", "continuous_rebuy"}:
            raise ValueError(f"未知 table.lifecycle：{table_lifecycle}")
        agents = _build_agents(
            train_ids,
            config,
            context,
            target_id,
            agent_configs=agent_configs,
        )
        clone_transform_audit = _restore_initial_memory_snapshots(agents, exp)
        if clone_transform_audit:
            artifacts.write_json("clone_transform_audit.json", clone_transform_audit)
        stacks = {agent_id: starting_stack for agent_id in train_ids}
        checkpoint_snapshot_paths: dict[str, dict[str, str]] = {}
        checkpoint_results: list[dict[str, Any]] = []
        planned_checkpoints = checkpoint_set or _legacy_checkpoint_set(
            train_hands,
            checkpoint_interval,
            initial_checkpoint_hand=int(exp.get("initial_checkpoint_hand", 0)),
        )
        hands_by_checkpoint = {
            point: _checkpoint_test_hands(exp, point, checkpoint_test_hands)
            for point in planned_checkpoints
        }
        heldout_rosters = exp.get("heldout_table_rosters")
        default_heldout_roster = config.get(
            "heldout_agent", config.get("opponent_agent", {})
        )
        roster_identities = {
            table_id: sha256_json(
                heldout_rosters[table_id]
                if isinstance(heldout_rosters, dict) and table_id in heldout_rosters
                else default_heldout_roster
            )
            for table_id in heldout_table_set
        }
        schedule_manifest = build_heldout_schedule_manifest(
            root_seed=seed,
            checkpoint_set=planned_checkpoints,
            table_set=heldout_table_set,
            hands_by_checkpoint=hands_by_checkpoint,
            table_size=table_size,
            roster_identity=(
                {table_id: str(exp["heldout_roster_identity"]) for table_id in heldout_table_set}
                if exp.get("heldout_roster_identity")
                else roster_identities
            ),
        )
        verify_schedule_manifest(schedule_manifest)
        artifacts.write_json("schedule_manifest.json", schedule_manifest)
        rebuy_counts = {agent_id: 0 for agent_id in train_ids}
        executed_train_hands = 0
        for hand_index in range(train_hands):
            if (
                table_lifecycle == "tournament_elimination"
                and sum(stack > 0 for stack in stacks.values()) < 2
            ):
                artifacts.log_event(
                    {
                        "event": "training_stopped",
                        "reason": "fewer_than_two_active_players",
                        "requested_train_hands": train_hands,
                        "executed_train_hands": executed_train_hands,
                    }
                )
                break
            run_single_hand(
                agents=agents,
                table_id="train_table",
                agent_ids=train_ids,
                stacks=stacks,
                seed=derive_seed(seed, "fixed", "train", hand_index),
                stage="train",
                small_blind=small_blind,
                big_blind=big_blind,
                max_raises_per_street=max_raises,
                update_memory=update_train,
                artifacts=artifacts,
                dealer_index=hand_index % table_size,
                hand_number=hand_index + 1,
            )
            executed_train_hands += 1
            if table_lifecycle == "continuous_rebuy":
                _rebuy_busted_agents(
                    stacks,
                    starting_stack,
                    rebuy_counts,
                    artifacts,
                    hand_index + 1,
                    stage="train",
                )
            checkpoint_hand = hand_index + 1
            checkpoint_due = (
                checkpoint_hand in checkpoint_set
                if checkpoint_set is not None
                else checkpoint_interval > 0
                and (checkpoint_hand % checkpoint_interval == 0 or checkpoint_hand == train_hands)
            )
            if checkpoint_due:
                snapshots, results = _checkpoint_and_generalize(
                    checkpoint_hand=checkpoint_hand,
                    evaluation_targets=evaluation_targets,
                    agents=agents,
                    config=config,
                    context=context,
                    seed=seed,
                    table_size=table_size,
                    starting_stack=starting_stack,
                    small_blind=small_blind,
                    big_blind=big_blind,
                    max_raises=max_raises,
                    test_hands=_checkpoint_test_hands(
                        exp, checkpoint_hand, checkpoint_test_hands
                    ),
                    update_test=update_test,
                    table_lifecycle=table_lifecycle,
                    heldout_table_set=heldout_table_set,
                )
                checkpoint_snapshot_paths[str(checkpoint_hand)] = snapshots
                checkpoint_results.extend(results)
        if checkpoint_set is not None and executed_train_hands != train_hands:
            raise ConfigError(
                "显式 checkpoint_set 训练提前终止，拒绝产生缺失 checkpoint 的结果"
            )
        final_checkpoint_missing = str(executed_train_hands) not in checkpoint_snapshot_paths
        if checkpoint_set is None and (checkpoint_interval == 0 or final_checkpoint_missing):
            final_checkpoint_hand = int(
                exp.get("initial_checkpoint_hand", executed_train_hands)
            )
            snapshots, results = _checkpoint_and_generalize(
                checkpoint_hand=final_checkpoint_hand,
                evaluation_targets=evaluation_targets,
                agents=agents,
                config=config,
                context=context,
                seed=seed,
                table_size=table_size,
                starting_stack=starting_stack,
                small_blind=small_blind,
                big_blind=big_blind,
                max_raises=max_raises,
                test_hands=test_hands,
                update_test=update_test,
                table_lifecycle=table_lifecycle,
                heldout_table_set=heldout_table_set,
            )
            checkpoint_snapshot_paths[str(final_checkpoint_hand)] = snapshots
            checkpoint_results.extend(results)
        train_snapshot_paths = {
            agent_id: artifacts.save_snapshot(
                agent_id,
                agent.snapshot_memory(),
                suffix="after_train",
            )
            for agent_id, agent in agents.items()
        }
        final_snapshot_paths = {
            agent_id: artifacts.save_snapshot(agent_id, agent.snapshot_memory(), suffix="final")
            for agent_id, agent in agents.items()
        }
        hands = artifacts.hand_summaries.read_all()
        events = artifacts.events.read_all()
        memory_metrics = {agent_id: agent.memory_metrics() for agent_id, agent in agents.items()}
        _merge_generalization_experience_injection_audits(
            memory_metrics,
            checkpoint_results,
        )
        mechanism_counts: dict[str, int] = {}
        for agent_id in train_ids:
            mechanism = str(memory_metrics[agent_id].get("mechanism", "unknown"))
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        protocol_audit = {
            "scenario": self.name,
            "requested_train_hands": train_hands,
            "train_hands": executed_train_hands,
            "test_hands": test_hands,
            "table_size": table_size,
            "target_agent_id": target_id,
            "evaluation_target_ids": evaluation_targets,
            "train_agent_mechanisms": {
                agent_id: memory_metrics[agent_id].get("mechanism", "unknown")
                for agent_id in train_ids
            },
            "train_agent_raise_sizing_policies": {
                agent_id: agents[agent_id].raise_sizing_policy for agent_id in train_ids
            },
            "train_mechanism_counts": mechanism_counts,
            "memory_update_train": update_train,
            "memory_update_test": update_test,
            "dealer_rotation": "hand_index modulo table_size",
            "generalization_schedule": (
                "explicit_checkpoint_set"
                if checkpoint_set is not None
                else (
                    f"every_{checkpoint_interval}_train_hands_and_final"
                    if checkpoint_interval > 0
                    else "final_snapshot_only"
                )
            ),
            "checkpoint_set": planned_checkpoints,
            "heldout_table_set": heldout_table_set,
            "schedule_sha256": schedule_manifest["schedule_sha256"],
            "checkpoint_test_hands": checkpoint_test_hands,
            "checkpoint_cost_budget": _checkpoint_cost_budget(
                train_hands=train_hands,
                checkpoint_interval=checkpoint_interval,
                checkpoint_set=planned_checkpoints,
                evaluation_target_count=len(evaluation_targets),
                heldout_table_count=len(heldout_table_set),
                checkpoint_test_hands=checkpoint_test_hands,
                hands_by_checkpoint=hands_by_checkpoint,
                seed_count=(
                    len(exp["seeds"])
                    if isinstance(exp.get("seeds"), list)
                    else 1
                ),
            ),
            "heldout_seed_policy": (
                "derive_seed(root, phase, checkpoint, table, hand, rng_stream); "
                "target/mechanism/worker excluded"
            ),
            "table_lifecycle": table_lifecycle,
            "strategy_risk_gate": str(
                config.get("agent", {}).get("strategy_risk_gate", "disabled")
            ),
            "strategy_risk_gate_applied": False,
            "action_guard_scope": "legality_and_format_only",
            "embedding_protocol": {
                key: config.get("agent", {}).get(key)
                for key in (
                    "embedding_backend",
                    "embedding_model",
                    "embedding_revision",
                    "embedding_weights_hash",
                    "embedding_tokenizer_revision",
                    "embedding_base_url_env",
                    "embedding_query_instruction",
                    "embedding_query_policy",
                    "embedding_hybrid_weights",
                    "embedding_candidate_depth",
                    "embedding_colbert_rerank_depth",
                    "embedding_final_top_k_policy",
                    "embedding_cache_schema_version",
                    "embedding_cache_path",
                )
            },
            "rebuy_counts": rebuy_counts,
            "preregistered_primary_metrics": list(
                exp.get("preregistered_primary_metrics", [])
            ),
            "primary_endpoint": exp.get("primary_endpoint"),
            "primary_estimand": exp.get("primary_estimand"),
            "primary_baseline_mechanism": exp.get("primary_baseline_mechanism"),
            "within_table_mechanism_aggregation": exp.get(
                "within_table_mechanism_aggregation"
            ),
            "multiple_comparison_method": exp.get("multiple_comparison_method"),
            "required_seed_pairs": exp.get("required_seed_pairs"),
            "statistical_plan_status": exp.get("statistical_plan_status"),
            "auxiliary_metrics": list(exp.get("auxiliary_metrics", ["bb_per_100"])),
            "paper_evolving_roster_match": mechanism_counts
            == {
                "fact": 2,
                "expr": 2,
                "fact_expr_sync": 2,
                "fact_expr_async": 2,
            },
            "known_protocol_gaps": _known_protocol_gaps(
                evaluation_targets=evaluation_targets,
                train_ids=train_ids,
                checkpoint_interval=checkpoint_interval,
                checkpoint_set=planned_checkpoints,
                agent_configs=agent_configs,
            ),
            "generalization_experience_injection_audit": [
                {
                    "checkpoint_hand": item["checkpoint_hand"],
                    "heldout_table_id": item["heldout_table_id"],
                    "target_agent_id": item["target_agent_id"],
                    **dict(item["experience_injection_audit"]),
                }
                for item in checkpoint_results
                if isinstance(item.get("experience_injection_audit"), dict)
            ],
        }
        protocol_audit_path = artifacts.write_json("protocol_audit.json", protocol_audit)
        checkpoint_summary = _summarize_checkpoint_results(checkpoint_results)
        checkpoint_path = artifacts.write_json(
            "checkpoint_generalization.json",
            {
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_set": planned_checkpoints,
                "heldout_table_set": heldout_table_set,
                "schedule_sha256": schedule_manifest["schedule_sha256"],
                "checkpoint_test_hands": checkpoint_test_hands,
                "results": checkpoint_results,
                "summary": checkpoint_summary,
            },
        )
        metrics = compute_metrics(hands, events, big_blind=big_blind, memory_metrics=memory_metrics)
        metrics["primary_metrics"]["preregistered_metrics"] = list(
            exp.get("preregistered_primary_metrics", [])
        )
        metrics["primary_metrics"]["auxiliary_metrics"] = list(
            exp.get("auxiliary_metrics", ["bb_per_100"])
        )
        metrics["primary_metrics"]["checkpoint_generalization"] = checkpoint_summary
        primary_estimand = str(exp.get("primary_estimand", ""))
        if primary_estimand == "same_seed_table_run_mechanism_effect_vs_baseline":
            metrics["primary_metrics"]["table_run_estimand"] = build_table_run_estimand(
                checkpoint_results,
                seed=seed,
                run_id=artifacts.run_id,
                endpoint=str(exp.get("primary_endpoint", "")),
                baseline_mechanism=str(exp.get("primary_baseline_mechanism", "")),
                statistical_plan_status=str(exp.get("statistical_plan_status", "")),
                multiple_comparison_method=str(
                    exp.get("multiple_comparison_method", "holm")
                ),
                required_seed_pairs=(
                    int(exp["required_seed_pairs"])
                    if exp.get("required_seed_pairs") is not None
                    else None
                ),
            )
        behavior_health = evaluate_behavior_health(metrics, exp, evaluation_targets)
        execution_health = evaluate_execution_health(hands, metrics)
        admission = dict(exp.get("admission_audit", {}))
        run_validity = build_run_validity(
            admission,
            behavior_health,
            execution_health,
            str(exp.get("run_mode", "smoke")),
        )
        metrics["behavior_health"] = behavior_health
        metrics["execution_health"] = execution_health
        metrics["run_validity"] = run_validity
        protocol_audit["behavior_health"] = behavior_health
        protocol_audit["execution_health"] = execution_health
        protocol_audit["run_validity"] = run_validity
        artifacts.write_json("protocol_audit.json", protocol_audit)
        review_rows = [
            {"agent_id": agent_id, **row}
            for agent_id, memory in memory_metrics.items()
            for row in memory.get("evidence_review_queue", [])
            if isinstance(row, dict)
        ]
        async_review_path = artifacts.write_json(
            "async_evidence_review_queue.json",
            _build_async_evidence_artifact(review_rows),
        )
        aggregate = aggregate_metrics([metrics])
        plot_path = plot_stack_curves(hands, artifacts.run_dir / "plots")
        audit_plot_paths = generate_audit_plots(hands, events, artifacts.run_dir / "plots")
        artifacts.write_json("metrics.json", metrics)
        artifacts.write_json("aggregate_metrics.json", aggregate)
        report = build_report_text(
            run_id=artifacts.run_id,
            scenario=self.name,
            metrics=metrics,
            aggregate=aggregate,
            plot_paths=[plot_path, *audit_plot_paths],
            notes=[
                "已离线验证 mock Provider 路径；真实 Provider 需用户提供密钥后单独 smoke test。",
                "本地环境实现覆盖核心 betting flow，复杂边池规则在文档中标为待增强。",
                f"主表准入状态：{run_validity['status']}。",
            ],
        )
        report_path = artifacts.write_text("report.md", report)
        result = ExperimentResult(
            run_id=artifacts.run_id,
            scenario=self.name,
            metrics=metrics,
            aggregate_metrics=aggregate,
            artifacts={
                "run_dir": str(artifacts.run_dir),
                "report": report_path,
                "train_snapshots": train_snapshot_paths,
                "checkpoint_snapshots": checkpoint_snapshot_paths,
                "final_snapshots": final_snapshot_paths,
                "plot": plot_path,
                "audit_plots": audit_plot_paths,
                "protocol_audit": protocol_audit_path,
                "checkpoint_generalization": checkpoint_path,
                "async_evidence_review_queue": async_review_path,
            },
            notes=[
                "固定桌训练和泛化测试均可离线运行。",
                "泛化阶段是否继续更新记忆由 update_memory_test 控制。",
                f"主表准入状态：{run_validity['status']}。",
            ],
        )
        artifacts.finish(result)
        return result


def _build_async_evidence_artifact(
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize Async evidence without imposing a human gate on paper semantics."""

    status_counts: dict[str, int] = {}
    human_review_required = False
    for row in review_rows:
        status = str(row.get("classification_status", "pending_human_review"))
        status_counts[status] = status_counts.get(status, 0) + 1
        explicit_human_review = row.get("human_review_required")
        human_review_required = human_review_required or (
            explicit_human_review
            if isinstance(explicit_human_review, bool)
            else status == "pending_human_review"
        )
    if not review_rows:
        classification_status = "not_requested"
    elif human_review_required:
        classification_status = "pending_human_review"
    elif len(status_counts) == 1:
        classification_status = next(iter(status_counts))
    elif set(status_counts) <= {
        "paper_faithful_provisional_applied",
        "no_valid_stability_proposal",
    }:
        classification_status = "paper_faithful_mixed_outcomes"
    else:
        classification_status = "mixed_non_pending"
    return {
        "classification_status": classification_status,
        "human_review_required": human_review_required,
        "classification_status_counts": status_counts,
        "pending_human_review_count": status_counts.get("pending_human_review", 0),
        "paper_faithful_provisional_applied_count": status_counts.get(
            "paper_faithful_provisional_applied", 0
        ),
        "record_count": len(review_rows),
        "records": review_rows,
    }


def _build_agents(
    agent_ids: list[str],
    config: dict[str, Any],
    context: ExperimentContext,
    target_id: str,
    agent_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, LLMDecisionAgent]:
    """
    功能：创建训练桌 Agent。
    参数：
        agent_ids：Agent 列表。
        config：resolved 配置。
        context：实验上下文。
        target_id：目标 Agent。
    返回：Agent 字典。
    副作用：无。
    异常：配置错误由 build_agent 抛出。
    设计说明：目标和对手可以使用不同机制，也可通过 all_agents_same_mechanism 对齐。
    """

    provider = config["provider"]
    model = str(provider.get("model", "mock-deterministic-v1"))
    reviewer_provider = config.get("experience_reviewer_provider", {})
    reviewer_model = (
        str(reviewer_provider.get("model", ""))
        if isinstance(reviewer_provider, dict)
        else ""
    )
    agent_cfg = dict(config.get("agent", {}))
    opponent_cfg = dict(config.get("opponent_agent", {"mechanism": "no_memory"}))
    if config.get("experiment", {}).get("all_agents_same_mechanism", False):
        opponent_cfg = dict(agent_cfg)
    agents: dict[str, LLMDecisionAgent] = {}
    for agent_id in agent_ids:
        cfg = (agent_configs or {}).get(
            agent_id,
            agent_cfg if agent_id == target_id else opponent_cfg,
        )
        agents[agent_id] = build_agent(
            agent_id,
            cfg,
            context.llm_client,
            model,
            experience_reviewer_client=context.experience_reviewer_client,
            experience_reviewer_model=reviewer_model,
        )
        agents[agent_id].bind_memory_persistence(
            context.artifacts.agent_memory_directory(agent_id),
            reference_root=context.artifacts.run_dir,
        )
    return agents


def _resolve_train_roster(
    config: dict[str, Any], table_size: int, target_id: str
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Resolve either the legacy target/opponent layout or an explicit mixed roster."""

    roster = config.get("experiment", {}).get("agent_roster")
    if not roster:
        agent_ids = [f"agent_{index:02d}" for index in range(table_size)]
        if target_id not in agent_ids:
            agent_ids[0] = target_id
        return agent_ids, {}
    if not isinstance(roster, list):
        raise ValueError("experiment.agent_roster 必须是列表")
    base_agent = dict(config.get("agent", {}))
    agent_ids: list[str] = []
    configs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(roster):
        if not isinstance(raw, dict):
            raise ValueError("agent_roster 每项必须是对象")
        agent_id = str(raw.get("agent_id", f"agent_{index:02d}"))
        if agent_id in configs:
            raise ValueError(f"agent_roster 存在重复 agent_id：{agent_id}")
        override = dict(raw.get("config", {}))
        for key, value in raw.items():
            if key not in {"agent_id", "config"}:
                override[key] = value
        agent_ids.append(agent_id)
        configs[agent_id] = {**base_agent, **override}
    if len(agent_ids) < 2:
        raise ValueError("混合训练桌至少需要 2 个 Agent")
    return agent_ids, configs


def _resolve_evaluation_targets(
    exp: dict[str, Any], train_ids: list[str], target_id: str
) -> list[str]:
    configured = exp.get("evaluation_target_ids")
    targets = list(train_ids) if exp.get("evaluate_all_train_agents") else [target_id]
    if configured is not None:
        targets = [str(value) for value in configured]
    unknown = sorted(set(targets) - set(train_ids))
    if unknown:
        raise ValueError(f"evaluation_target_ids 不在训练桌：{unknown}")
    return targets


def _checkpoint_and_generalize(
    *,
    checkpoint_hand: int,
    evaluation_targets: list[str],
    agents: dict[str, LLMDecisionAgent],
    config: dict[str, Any],
    context: ExperimentContext,
    seed: int,
    table_size: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    max_raises: int,
    test_hands: int,
    update_test: bool,
    table_lifecycle: str,
    heldout_table_set: list[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Persist every target snapshot and evaluate each against independently seeded opponents."""

    artifacts = context.artifacts
    snapshot_paths = {
        agent_id: artifacts.save_snapshot(
            agent_id,
            agents[agent_id].snapshot_memory(),
            suffix=f"checkpoint_{checkpoint_hand:04d}",
        )
        for agent_id in evaluation_targets
    }
    results: list[dict[str, Any]] = []
    train_hands_so_far = [
        hand
        for hand in artifacts.hand_summaries.read_all()
        if hand.get("stage") == "train"
    ]
    train_chip_by_target = {
        target_id: sum(
            int((hand.get("rewards", {}) or {}).get(target_id, 0))
            for hand in train_hands_so_far
        )
        for target_id in evaluation_targets
    }
    train_hand_count_by_target = {
        target_id: sum(
            target_id in (hand.get("rewards", {}) or {}) for hand in train_hands_so_far
        )
        for target_id in evaluation_targets
    }
    if test_hands <= 0:
        return snapshot_paths, results
    for target_id in evaluation_targets:
        for table_offset, heldout_table_id in enumerate(heldout_table_set):
            heldout_ids = [
                target_id,
                *[
                    f"heldout_{heldout_table_id}_seat{index:02d}"
                    for index in range(1, table_size)
                ],
            ]
            heldout_agents = _build_heldout_agents(
                heldout_ids,
                config,
                context,
                target_id,
                agents[target_id],
                heldout_table_id,
            )
            for heldout_agent_id, heldout_agent in heldout_agents.items():
                heldout_agent.bind_memory_persistence(
                    artifacts.agent_memory_directory(
                        heldout_agent_id,
                        namespace=(
                            "g",
                            f"c{checkpoint_hand}",
                            f"t{sha256_json(target_id)[:8]}",
                            heldout_table_id,
                        ),
                    ),
                    reference_root=artifacts.run_dir,
                )
                heldout_agent.set_memory_frozen(not update_test)
            target_memory_before_test = heldout_agents[target_id].memory_metrics()
            target_snapshot_before_test = heldout_agents[target_id].snapshot_memory()
            test_stacks = {agent_id: starting_stack for agent_id in heldout_ids}
            test_rebuy_counts = {agent_id: 0 for agent_id in heldout_ids}
            reward = 0
            for hand_index in range(test_hands):
                result = run_single_hand(
                    agents=heldout_agents,
                    table_id=f"heldout_{heldout_table_id}_cp{checkpoint_hand}",
                    agent_ids=heldout_ids,
                    stacks=test_stacks,
                    seed=derive_formal_hand_seed(
                        seed,
                        "heldout",
                        checkpoint_hand,
                        heldout_table_id,
                        hand_index,
                        "deal_and_opponent",
                    ),
                    stage="test",
                    small_blind=small_blind,
                    big_blind=big_blind,
                    max_raises_per_street=max_raises,
                    update_memory=update_test,
                    artifacts=artifacts,
                    dealer_index=(hand_index + table_offset) % table_size,
                    hand_number=hand_index + 1,
                    hand_metadata={
                        "checkpoint_hand": checkpoint_hand,
                        "evaluation_target_id": target_id,
                        "heldout_table_id": heldout_table_id,
                    },
                )
                reward += int(result.rewards.get(target_id, 0))
                if table_lifecycle == "continuous_rebuy":
                    _rebuy_busted_agents(
                        test_stacks,
                        starting_stack,
                        test_rebuy_counts,
                        artifacts,
                        hand_index + 1,
                        stage="test",
                        metadata={
                            "checkpoint_hand": checkpoint_hand,
                            "evaluation_target_id": target_id,
                            "heldout_table_id": heldout_table_id,
                        },
                    )
            target_memory_after_test = heldout_agents[target_id].memory_metrics()
            target_snapshot_after_test = heldout_agents[target_id].snapshot_memory()
            frozen_memory_audit = _build_frozen_memory_audit(
                target_snapshot_before_test,
                target_snapshot_after_test,
                update_memory_test=update_test,
            )
            experience_injection_audit = _build_generalization_experience_injection_audit(
                target_memory_before_test,
                target_memory_after_test,
                update_memory_test=update_test,
                frozen_memory_audit=frozen_memory_audit,
            )
            train_hand_count = train_hand_count_by_target[target_id]
            train_chip_per_hand = train_chip_by_target[target_id] / max(1, train_hand_count)
            test_chip_per_hand = reward / max(1, test_hands)
            train_bb_per_100 = train_chip_per_hand / max(1, big_blind) * 100
            test_bb_per_100 = test_chip_per_hand / max(1, big_blind) * 100
            results.append(
                {
                    "checkpoint_hand": checkpoint_hand,
                    "heldout_table_id": heldout_table_id,
                    "target_agent_id": target_id,
                    "mechanism": agents[target_id].memory_metrics().get(
                        "mechanism", "unknown"
                    ),
                    "test_hands": test_hands,
                    "chip_delta": reward,
                    "train_chip_delta_at_checkpoint": train_chip_by_target[target_id],
                    "train_hands": train_hand_count,
                    "train_chip_per_hand": train_chip_per_hand,
                    "test_chip_per_hand": test_chip_per_hand,
                    "generalization_gap_chip_delta": train_chip_per_hand - test_chip_per_hand,
                    "generalization_gap_total_chip_delta_unscaled": (
                        train_chip_by_target[target_id] - reward
                    ),
                    "train_bb_per_100": train_bb_per_100,
                    "bb_per_100": test_bb_per_100,
                    "generalization_gap_bb_per_100": train_bb_per_100 - test_bb_per_100,
                    "snapshot_path": snapshot_paths[target_id],
                    "rebuy_counts": test_rebuy_counts,
                    "experience_injection_audit": experience_injection_audit,
                    "frozen_memory_audit": frozen_memory_audit,
                }
            )
    return snapshot_paths, results


_EXPERIENCE_MECHANISMS = {"expr", "fact_expr_sync", "fact_expr_async"}


def _build_generalization_experience_injection_audit(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    update_memory_test: bool,
    frozen_memory_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit whether a heldout clone consumed the frozen trained experience."""

    mechanism = str(after.get("mechanism", before.get("mechanism", "unknown")))
    if mechanism not in _EXPERIENCE_MECHANISMS:
        return {
            "schema_version": "generalization-experience-injection-v1",
            "mechanism": mechanism,
            "applicable": False,
            "passed": True,
            "status": "not_applicable",
        }

    before_version = int(before.get("experience_version", 0))
    after_version = int(after.get("experience_version", 0))
    before_revision_count = int(before.get("revision_count", 0))
    after_revision_count = int(after.get("revision_count", 0))
    before_context_count = int(before.get("experience_context_build_count", 0))
    after_context_count = int(after.get("experience_context_build_count", 0))
    injected_version = after.get("last_injected_experience_version")
    context_build_delta = after_context_count - before_context_count
    frozen_revision_state_unchanged = (
        after_version == before_version
        and after_revision_count == before_revision_count
    )
    injected_trained_version = (
        context_build_delta > 0
        and injected_version is not None
        and int(injected_version) == before_version
    )
    document_valid = bool(after.get("experience_document_valid", False))
    frozen_persistent_state_unchanged = bool(
        (frozen_memory_audit or {}).get(
            "persistent_state_unchanged",
            True,
        )
    )
    passed = (
        not update_memory_test
        and frozen_revision_state_unchanged
        and frozen_persistent_state_unchanged
        and injected_trained_version
        and document_valid
    )
    if update_memory_test:
        status = "not_frozen_update_memory_test_true"
    elif not frozen_revision_state_unchanged:
        status = "failed_frozen_memory_changed"
    elif not frozen_persistent_state_unchanged:
        status = "failed_frozen_persistent_state_changed"
    elif not injected_trained_version:
        status = "failed_trained_version_not_injected"
    elif not document_valid:
        status = "failed_invalid_experience_document"
    else:
        status = "passed"
    return {
        "schema_version": "generalization-experience-injection-v1",
        "mechanism": mechanism,
        "applicable": True,
        "update_memory_test": update_memory_test,
        "expected_experience_version": before_version,
        "post_test_experience_version": after_version,
        "last_injected_experience_version": injected_version,
        "context_build_count_before": before_context_count,
        "context_build_count_after": after_context_count,
        "context_build_count_delta": context_build_delta,
        "revision_count_before": before_revision_count,
        "revision_count_after": after_revision_count,
        "frozen_revision_state_unchanged": frozen_revision_state_unchanged,
        "frozen_persistent_state_unchanged": frozen_persistent_state_unchanged,
        "experience_document_valid": document_valid,
        "injected_trained_version": injected_trained_version,
        "passed": passed,
        "status": status,
    }


def _build_frozen_memory_audit(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    update_memory_test: bool,
) -> dict[str, Any]:
    """Prove that held-out reads did not mutate learned memory state."""

    before_payload = _persistent_memory_state(before)
    after_payload = _persistent_memory_state(after)
    before_sha256 = sha256_json(before_payload)
    after_sha256 = sha256_json(after_payload)
    unchanged = before_sha256 == after_sha256
    return {
        "schema_version": "generalization-frozen-memory-audit-v1",
        "mechanism": after.mechanism,
        "update_memory_test": update_memory_test,
        "persistent_state_sha256_before": before_sha256,
        "persistent_state_sha256_after": after_sha256,
        "frozen_memory_snapshot_sha256_before": before_sha256,
        "frozen_memory_snapshot_sha256_after": after_sha256,
        "persistent_state_unchanged": unchanged,
        "passed": (not update_memory_test and unchanged) or update_memory_test,
        "status": (
            "not_frozen_update_memory_test_true"
            if update_memory_test
            else "passed" if unchanged else "failed_frozen_persistent_state_changed"
        ),
    }


def _persistent_memory_state(snapshot: MemorySnapshot) -> dict[str, Any]:
    """Strip read-only telemetry while retaining every learned memory field."""

    payload = copy.deepcopy(snapshot.payload)
    mechanism = snapshot.mechanism
    if mechanism == "fact":
        for key in (
            "retrieval_request_count",
            "empty_retrieval_count",
            "retrieval_below_threshold_count",
            "retrieval_duplicate_excluded_count",
            "retrieval_audit_log",
            "embedding",
        ):
            payload.pop(key, None)
    elif mechanism == "expr":
        payload.pop("context_build_count", None)
        payload.pop("last_injected_experience_version", None)
    elif mechanism in {"fact_expr_sync", "fact_expr_async"}:
        payload["fact"] = _persistent_memory_state(
            MemorySnapshot("fact", snapshot.agent_id, snapshot.scope, payload.get("fact", {}))
        )["payload"]
        payload["expr"] = _persistent_memory_state(
            MemorySnapshot("expr", snapshot.agent_id, snapshot.scope, payload.get("expr", {}))
        )["payload"]
    elif mechanism == "personality_driven":
        wrapped_mechanism = str(payload.get("wrapped_mechanism", "no_memory"))
        payload["wrapped_payload"] = _persistent_memory_state(
            MemorySnapshot(
                wrapped_mechanism,
                snapshot.agent_id,
                snapshot.scope,
                payload.get("wrapped_payload", {}),
            )
        )["payload"]
    return {"mechanism": mechanism, "payload": payload}


def _merge_generalization_experience_injection_audits(
    memory_metrics: dict[str, dict[str, Any]],
    checkpoint_results: list[dict[str, Any]],
) -> None:
    """Merge real heldout-clone injections into the run-level Expr health signal."""

    for agent_id, memory in memory_metrics.items():
        if str(memory.get("mechanism", "")) not in _EXPERIENCE_MECHANISMS:
            continue
        source_current = bool(memory.get("experience_injection_current", False))
        current_version = int(memory.get("experience_version", 0))
        audits = [
            dict(item["experience_injection_audit"])
            for item in checkpoint_results
            if str(item.get("target_agent_id")) == agent_id
            and isinstance(item.get("experience_injection_audit"), dict)
            and bool(item["experience_injection_audit"].get("applicable", False))
            and int(
                item["experience_injection_audit"].get(
                    "expected_experience_version", -1
                )
            )
            == current_version
        ]
        generalization_current = bool(audits) and all(
            bool(item.get("passed", False)) for item in audits
        )
        memory["experience_injection_current_training_source"] = source_current
        memory["experience_injection_current_generalization"] = generalization_current
        memory["generalization_experience_injection_audits"] = audits
        memory["experience_injection_current"] = source_current or generalization_current


def _summarize_checkpoint_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the two agents per mechanism and emit within-checkpoint paired effects."""

    summary: dict[str, Any] = {}
    checkpoints = sorted({int(item["checkpoint_hand"]) for item in results})
    for checkpoint in checkpoints:
        checkpoint_rows = [
            item for item in results if int(item["checkpoint_hand"]) == checkpoint
        ]
        by_mechanism: dict[str, list[dict[str, Any]]] = {}
        for item in checkpoint_rows:
            by_mechanism.setdefault(str(item["mechanism"]), []).append(item)
        mechanism_summary = {
            mechanism: {
                "agent_ids": [str(item["target_agent_id"]) for item in items],
                "train_chip_delta_mean": statistics.mean(
                    float(item["train_chip_delta_at_checkpoint"]) for item in items
                ),
                "test_chip_delta_mean": statistics.mean(
                    float(item["chip_delta"]) for item in items
                ),
                "test_chip_per_hand_mean": statistics.mean(
                    float(item["test_chip_per_hand"]) for item in items
                ),
                "test_bb_per_100_mean": statistics.mean(
                    float(item["bb_per_100"]) for item in items
                ),
                "generalization_gap_mean": statistics.mean(
                    float(item["generalization_gap_chip_delta"]) for item in items
                ),
            }
            for mechanism, items in sorted(by_mechanism.items())
        }
        paired = {}
        for left, right in combinations(sorted(mechanism_summary), 2):
            paired[f"{left}_minus_{right}"] = {
                "train_chip_delta": (
                    mechanism_summary[left]["train_chip_delta_mean"]
                    - mechanism_summary[right]["train_chip_delta_mean"]
                ),
                "test_chip_delta": (
                    mechanism_summary[left]["test_chip_delta_mean"]
                    - mechanism_summary[right]["test_chip_delta_mean"]
                ),
                "generalization_gap": (
                    mechanism_summary[left]["generalization_gap_mean"]
                    - mechanism_summary[right]["generalization_gap_mean"]
                ),
            }
        summary[str(checkpoint)] = {
            "by_mechanism": mechanism_summary,
            "paired_mechanism_effects": paired,
        }
    return summary


def _rebuy_busted_agents(
    stacks: dict[str, int],
    starting_stack: int,
    rebuy_counts: dict[str, int],
    artifacts: Any,
    after_hand: int,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    for agent_id, stack in list(stacks.items()):
        if stack > 0:
            continue
        stacks[agent_id] = starting_stack
        rebuy_counts[agent_id] += 1
        artifacts.log_event(
            {
                "event": "rebuy",
                "stage": stage,
                "agent_id": agent_id,
                "after_hand": after_hand,
                "stack_added": starting_stack,
                **(metadata or {}),
            }
        )


def _known_protocol_gaps(
    *,
    evaluation_targets: list[str],
    train_ids: list[str],
    checkpoint_interval: int,
    checkpoint_set: list[int],
    agent_configs: dict[str, dict[str, Any]],
) -> list[str]:
    gaps = []
    if set(evaluation_targets) != set(train_ids):
        gaps.append("Not every training agent is evaluated against heldout opponents.")
    if checkpoint_interval <= 0 and len(checkpoint_set) <= 1:
        gaps.append("Generalization runs only after the final training hand.")
    strategies = {
        str(config.get("experience_revision_strategy", "deterministic"))
        for config in agent_configs.values()
        if str(config.get("mechanism", "")) in {"expr", "fact_expr_sync", "fact_expr_async"}
    }
    if "deterministic" in strategies:
        gaps.append("At least one experience mechanism uses deterministic revision.")
    return gaps


def _checkpoint_cost_budget(
    *,
    train_hands: int,
    checkpoint_interval: int,
    checkpoint_set: list[int],
    evaluation_target_count: int,
    heldout_table_count: int,
    checkpoint_test_hands: int,
    hands_by_checkpoint: dict[int, int],
    seed_count: int,
) -> dict[str, int]:
    """Precompute the heldout-hand budget implied by the checkpoint protocol."""

    checkpoint_count = len(checkpoint_set)
    evaluations_per_seed = checkpoint_count * evaluation_target_count * heldout_table_count
    hands_per_seed = (
        sum(int(hands_by_checkpoint[point]) for point in checkpoint_set)
        * evaluation_target_count
        * heldout_table_count
    )
    return {
        "checkpoint_count_per_seed": checkpoint_count,
        "evaluation_target_count": evaluation_target_count,
        "heldout_table_count": heldout_table_count,
        "checkpoint_evaluations_per_seed": evaluations_per_seed,
        "checkpoint_generalization_hands_per_seed": hands_per_seed,
        "seed_count": seed_count,
        "checkpoint_generalization_hands_all_seeds": hands_per_seed * seed_count,
    }


def _legacy_checkpoint_set(
    train_hands: int, checkpoint_interval: int, *, initial_checkpoint_hand: int = 0
) -> list[int]:
    """Resolve legacy interval semantics into the exact points used for audit and scheduling."""

    if train_hands <= 0:
        return [max(0, initial_checkpoint_hand)]
    if checkpoint_interval <= 0:
        return [train_hands]
    points = list(range(checkpoint_interval, train_hands + 1, checkpoint_interval))
    if not points or points[-1] != train_hands:
        points.append(train_hands)
    return points


def _checkpoint_test_hands(
    experiment: dict[str, Any], checkpoint_hand: int, default_hands: int
) -> int:
    raw = experiment.get("checkpoint_test_hands_by_checkpoint")
    if raw is None:
        return default_hands
    if not isinstance(raw, dict):
        raise ValueError("checkpoint_test_hands_by_checkpoint 必须是映射")
    value = raw.get(str(checkpoint_hand), raw.get(checkpoint_hand, default_hands))
    hands = int(value)
    if hands < 0:
        raise ValueError("checkpoint test hands 不能为负数")
    return hands


def _restore_initial_memory_snapshots(
    agents: dict[str, LLMDecisionAgent], experiment: dict[str, Any]
) -> dict[str, Any] | None:
    raw = experiment.get("initial_memory_snapshots")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError("initial_memory_snapshots 必须是非空 agent_id 到路径映射")
    mode = str(experiment.get("memory_mode", "Frozen"))
    if mode not in {"Frozen", "Online", "Without"}:
        raise ValueError(f"未知 memory_mode：{mode}")
    expected_update = mode == "Online"
    if bool(experiment.get("update_memory_test", False)) != expected_update:
        raise ValueError(
            f"memory_mode={mode} 要求 update_memory_test={str(expected_update).lower()}"
        )
    audits: dict[str, Any] = {}
    for agent_id, snapshot_path in sorted(raw.items()):
        if agent_id not in agents:
            raise ValueError(f"initial memory target 不在 roster：{agent_id}")
        parent = load_snapshot(str(snapshot_path))
        branches = clone_memory_branches(parent)
        agents[agent_id].restore_memory(branches[mode])
        audits[agent_id] = build_clone_audit(parent, branches)
    return {
        "schema_version": "task8-memory-clone-audit-set-v1",
        "memory_mode": mode,
        "targets": audits,
    }


def _build_heldout_agents(
    agent_ids: list[str],
    config: dict[str, Any],
    context: ExperimentContext,
    target_id: str,
    trained_target: LLMDecisionAgent,
    heldout_table_id: str,
) -> dict[str, LLMDecisionAgent]:
    """
    功能：创建泛化测试桌 Agent。
    参数：
        agent_ids：测试桌 Agent 列表。
        config：resolved 配置。
        context：实验上下文。
        target_id：目标 Agent。
        trained_target：训练后的目标 Agent。
    返回：Agent 字典。
    副作用：无。
    异常：配置错误由 build_agent 抛出。
    设计说明：目标 Agent 携带训练记忆，对手是未见的新 Agent。
    """

    provider = config["provider"]
    model = str(provider.get("model", "mock-deterministic-v1"))
    reviewer_provider = config.get("experience_reviewer_provider", {})
    reviewer_model = (
        str(reviewer_provider.get("model", ""))
        if isinstance(reviewer_provider, dict)
        else ""
    )
    experiment = dict(config.get("experiment", {}))
    table_rosters = experiment.get("heldout_table_rosters")
    opponent_cfg = dict(
        table_rosters[heldout_table_id]
        if isinstance(table_rosters, dict) and heldout_table_id in table_rosters
        else config.get(
            "heldout_agent",
            config.get("opponent_agent", {"mechanism": "no_memory"}),
        )
    )
    # 泛化阶段使用训练后 Agent 的隔离副本。即使 update_memory_test=true，
    # 也不能让某个 checkpoint 的测试轨迹污染后续训练或其他 checkpoint。
    agents: dict[str, LLMDecisionAgent] = {target_id: copy.deepcopy(trained_target)}
    for agent_id in agent_ids:
        if agent_id == target_id:
            continue
        agents[agent_id] = build_agent(
            agent_id,
            opponent_cfg,
            context.llm_client,
            model,
            experience_reviewer_client=context.experience_reviewer_client,
            experience_reviewer_model=reviewer_model,
        )
    return agents
