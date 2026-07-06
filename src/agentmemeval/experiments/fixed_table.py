"""
模块说明：本模块实现固定桌训练与泛化测试场景。
核心职责：跑通论文 evolving table 与 generalization table 的最小闭环。
输入与输出：输入 ExperimentContext，输出 ExperimentResult 和标准工件。
依赖边界：依赖 Agent 构建、单手运行器、评估和存储层。
不负责：不实现换桌排程，不执行真实在线 Provider smoke。
"""

from __future__ import annotations

from typing import Any

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.agents.llm_agent import build_agent
from agentmemeval.analysis.plots import plot_stack_curves
from agentmemeval.core.domain import ExperimentResult
from agentmemeval.core.seeds import derive_seed
from agentmemeval.evaluation.aggregation import aggregate_metrics
from agentmemeval.evaluation.metrics import compute_metrics
from agentmemeval.evaluation.reporting import build_report_text
from agentmemeval.experiments.context import ExperimentContext
from agentmemeval.experiments.table_play import run_single_hand


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
        train_ids = [f"agent_{index:02d}" for index in range(table_size)]
        if target_id not in train_ids:
            train_ids[0] = target_id
        agents = _build_agents(train_ids, config, context, target_id)
        stacks = {agent_id: starting_stack for agent_id in train_ids}
        for hand_index in range(train_hands):
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
            )
        train_snapshot_paths = {
            agent_id: artifacts.save_snapshot(
                agent_id,
                agent.snapshot_memory(),
                suffix="after_train",
            )
            for agent_id, agent in agents.items()
        }
        if test_hands > 0:
            heldout_ids = [target_id, *[f"heldout_{index:02d}" for index in range(table_size - 1)]]
            heldout_agents = _build_heldout_agents(
                heldout_ids,
                config,
                context,
                target_id,
                agents[target_id],
            )
            test_stacks = {agent_id: starting_stack for agent_id in heldout_ids}
            for hand_index in range(test_hands):
                run_single_hand(
                    agents=heldout_agents,
                    table_id="generalization_table",
                    agent_ids=heldout_ids,
                    stacks=test_stacks,
                    seed=derive_seed(seed, "fixed", "test", hand_index),
                    stage="test",
                    small_blind=small_blind,
                    big_blind=big_blind,
                    max_raises_per_street=max_raises,
                    update_memory=update_test,
                    artifacts=artifacts,
                )
            agents.update({target_id: heldout_agents[target_id]})
        final_snapshot_paths = {
            agent_id: artifacts.save_snapshot(agent_id, agent.snapshot_memory(), suffix="final")
            for agent_id, agent in agents.items()
        }
        hands = artifacts.hand_summaries.read_all()
        events = artifacts.events.read_all()
        memory_metrics = {agent_id: agent.memory_metrics() for agent_id, agent in agents.items()}
        metrics = compute_metrics(hands, events, big_blind=big_blind, memory_metrics=memory_metrics)
        aggregate = aggregate_metrics([metrics])
        plot_path = plot_stack_curves(hands, artifacts.run_dir / "plots")
        artifacts.write_json("metrics.json", metrics)
        artifacts.write_json("aggregate_metrics.json", aggregate)
        report = build_report_text(
            run_id=artifacts.run_id,
            scenario=self.name,
            metrics=metrics,
            aggregate=aggregate,
            plot_paths=[plot_path],
            notes=[
                "已离线验证 mock Provider 路径；真实 Provider 需用户提供密钥后单独 smoke test。",
                "本地环境实现覆盖核心 betting flow，复杂边池规则在文档中标为待增强。",
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
                "final_snapshots": final_snapshot_paths,
                "plot": plot_path,
            },
            notes=[
                "固定桌训练和泛化测试均可离线运行。",
                "泛化阶段是否继续更新记忆由 update_memory_test 控制。",
            ],
        )
        artifacts.finish(result)
        return result


def _build_agents(
    agent_ids: list[str],
    config: dict[str, Any],
    context: ExperimentContext,
    target_id: str,
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
    agent_cfg = dict(config.get("agent", {}))
    opponent_cfg = dict(config.get("opponent_agent", {"mechanism": "no_memory"}))
    if config.get("experiment", {}).get("all_agents_same_mechanism", False):
        opponent_cfg = dict(agent_cfg)
    agents: dict[str, LLMDecisionAgent] = {}
    for agent_id in agent_ids:
        cfg = agent_cfg if agent_id == target_id else opponent_cfg
        agents[agent_id] = build_agent(agent_id, cfg, context.llm_client, model)
    return agents


def _build_heldout_agents(
    agent_ids: list[str],
    config: dict[str, Any],
    context: ExperimentContext,
    target_id: str,
    trained_target: LLMDecisionAgent,
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
    opponent_cfg = dict(
        config.get(
            "heldout_agent",
            config.get("opponent_agent", {"mechanism": "no_memory"}),
        )
    )
    agents: dict[str, LLMDecisionAgent] = {target_id: trained_target}
    for agent_id in agent_ids:
        if agent_id == target_id:
            continue
        agents[agent_id] = build_agent(agent_id, opponent_cfg, context.llm_client, model)
    return agents
