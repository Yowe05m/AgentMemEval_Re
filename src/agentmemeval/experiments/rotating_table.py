"""
模块说明：本模块实现 20+ Agent 换桌 / 对手池轮换实验。
核心职责：运行 seed 驱动的换桌排程，记录座位、轮空、暴露统计和标准指标。
输入与输出：输入 ExperimentContext，输出 ExperimentResult 和工件。
依赖边界：依赖调度器、单手运行器、Agent 构建、评估和存储层。
不负责：不声称严格 pairwise 均衡，不执行真实在线模型调用。
"""

from __future__ import annotations

from agentmemeval.agents.base import LLMDecisionAgent
from agentmemeval.agents.llm_agent import build_agent
from agentmemeval.analysis.plots import plot_stack_curves
from agentmemeval.core.domain import ExperimentResult
from agentmemeval.core.seeds import derive_seed
from agentmemeval.evaluation.aggregation import aggregate_metrics
from agentmemeval.evaluation.metrics import compute_metrics
from agentmemeval.evaluation.reporting import build_report_text
from agentmemeval.experiments.context import ExperimentContext
from agentmemeval.experiments.schedulers import RotationRound, TableRotationScheduler
from agentmemeval.experiments.table_play import run_single_hand


class RotatingTableScenario:
    """
    功能：运行换桌实验。
    参数：无。
    返回：场景实例。
    副作用：run 时写入事件、摘要、快照、指标和报告。
    异常：配置或环境错误向上抛出。
    设计说明：默认 balanced 调度量化实际暴露不均衡，不做过度承诺。
    """

    name = "rotating_table"

    def run(self, context: ExperimentContext) -> ExperimentResult:
        """
        功能：执行换桌场景。
        参数：
            context：实验上下文。
        返回：ExperimentResult。
        副作用：写入所有标准工件。
        异常：配置或运行错误向上抛出。
        设计说明：支持 N>=20 的 smoke run，同时也可配置 fixed mode 作对照。
        """

        config = context.config
        artifacts = context.artifacts
        artifacts.write_manifest()
        exp = config["experiment"]
        table = config.get("table", {})
        seed = int(exp["seed"])
        agent_count = int(exp.get("agent_count", 20))
        table_size = int(exp.get("table_size", 4))
        rounds = int(exp.get("rounds", 5))
        hands_per_round = int(exp.get("hands_per_table_round", 1))
        starting_stack = int(table.get("starting_stack", 1000))
        small_blind = int(table.get("small_blind", 1))
        big_blind = int(table.get("big_blind", 2))
        max_raises = int(table.get("max_raises_per_street", 4))
        update_memory = bool(exp.get("update_memory_train", True))
        rebuy_busted = bool(exp.get("rebuy_busted", True))
        mode = str(exp.get("rotation_mode", "balanced"))
        agent_ids = [f"agent_{index:02d}" for index in range(agent_count)]
        agents = _build_pool(agent_ids, config, context)
        stacks = {agent_id: starting_stack for agent_id in agent_ids}
        scheduler = TableRotationScheduler(agent_ids, table_size=table_size, seed=seed, mode=mode)
        executed_rounds: list[RotationRound] = []
        hand_counter = 0
        for round_index in range(rounds):
            rotation = scheduler.schedule_round(round_index)
            executed_rounds.append(rotation)
            artifacts.log_event({"event": "rotation", **rotation.to_dict()})
            for table_assignment in rotation.tables:
                for local_hand in range(hands_per_round):
                    result = run_single_hand(
                        agents=agents,
                        table_id=table_assignment.table_id,
                        agent_ids=table_assignment.seats,
                        stacks=stacks,
                        seed=derive_seed(
                            seed,
                            "rotating",
                            round_index,
                            table_assignment.table_id,
                            local_hand,
                        ),
                        stage="train",
                        small_blind=small_blind,
                        big_blind=big_blind,
                        max_raises_per_street=max_raises,
                        update_memory=update_memory,
                        artifacts=artifacts,
                    )
                    hand_counter += 1
                    if rebuy_busted:
                        for agent_id, stack in list(result.final_stacks.items()):
                            if stack < big_blind:
                                stacks[agent_id] = starting_stack
                                artifacts.log_event(
                                    {
                                        "event": "rebuy",
                                        "agent_id": agent_id,
                                        "after_hand_id": result.hand_id,
                                        "stack_before": stack,
                                        "stack_after": starting_stack,
                                    }
                                )
        exposure_stats = scheduler.exposure_stats(executed_rounds)
        artifacts.write_json("exposure_stats.json", exposure_stats)
        final_snapshot_paths = {
            agent_id: artifacts.save_snapshot(agent_id, agent.snapshot_memory(), suffix="final")
            for agent_id, agent in agents.items()
        }
        hands = artifacts.hand_summaries.read_all()
        events = artifacts.events.read_all()
        memory_metrics = {agent_id: agent.memory_metrics() for agent_id, agent in agents.items()}
        metrics = compute_metrics(
            hands,
            events,
            big_blind=big_blind,
            memory_metrics=memory_metrics,
            exposure_stats=exposure_stats,
        )
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
                f"已执行 {agent_count} 个 Agent、{rounds} 轮、{hand_counter} 手牌。",
                "暴露统计为实际排程结果；strict_pairwise_balanced 字段说明是否严格均衡。",
                "rebuy_busted=True 时使用现金桌式补码以保证 20 Agent 小规模赛程可完成。",
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
                "plot": plot_path,
                "exposure_stats": str(artifacts.run_dir / "exposure_stats.json"),
                "final_snapshots": final_snapshot_paths,
            },
            notes=[
                "换桌场景已离线验证，不依赖真实在线模型。",
                "固定桌对照可通过 experiment.rotation_mode=fixed 和相同预算配置运行。",
            ],
        )
        artifacts.finish(result)
        return result


def _build_pool(
    agent_ids: list[str],
    config: dict[str, object],
    context: ExperimentContext,
) -> dict[str, LLMDecisionAgent]:
    """
    功能：创建换桌 Agent 池。
    参数：
        agent_ids：Agent ID 列表。
        config：resolved 配置。
        context：实验上下文。
    返回：Agent 字典。
    副作用：无。
    异常：配置错误由 build_agent 抛出。
    设计说明：所有 Agent 默认同机制，persona 可按轮询方式分配。
    """

    provider = config["provider"]  # type: ignore[index]
    model = str(provider.get("model", "mock-deterministic-v1"))  # type: ignore[union-attr]
    agent_cfg = dict(config.get("agent", {}))  # type: ignore[union-attr]
    personas = list(config.get("personas", []) or [])  # type: ignore[union-attr]
    agents: dict[str, LLMDecisionAgent] = {}
    for index, agent_id in enumerate(agent_ids):
        cfg = dict(agent_cfg)
        if personas:
            cfg["persona"] = personas[index % len(personas)]
        agents[agent_id] = build_agent(agent_id, cfg, context.llm_client, model)
    return agents
