"""
模块说明：本模块生成实验图表。
核心职责：从手牌摘要生成筹码曲线 PNG，若 matplotlib 不可用则写文本占位。
输入与输出：输入手牌摘要和输出目录，输出图表路径。
依赖边界：matplotlib 为项目依赖，但函数内部保留降级路径。
不负责：不计算指标，不解释结果。
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def plot_stack_curves(hand_summaries: list[dict[str, Any]], output_dir: str | Path) -> str:
    """
    功能：生成每个 Agent 的累计筹码变化曲线。
    参数：
        hand_summaries：手牌摘要。
        output_dir：输出目录。
    返回：图表路径。
    副作用：写 PNG 或文本文件。
    异常：内部绘图失败时降级为文本。
    设计说明：smoke run 也应产出 plots 工件，便于验收闭环。
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cumulative_by_stage = _build_cumulative_by_stage(hand_summaries)
    try:
        mpl_config = Path(os.environ.get("MPLCONFIGDIR", output / ".matplotlib"))
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

        import matplotlib.pyplot as plt

        stages = _ordered_stages(cumulative_by_stage)
        fig, axes_grid = plt.subplots(
            1,
            max(1, len(stages)),
            figsize=(7.2 * max(1, len(stages)), 6.0),
            squeeze=False,
        )
        axes = list(axes_grid[0])
        for ax, stage in zip(axes, stages, strict=True):
            cumulative = cumulative_by_stage[stage]
            for agent_id, values in sorted(cumulative.items()):
                is_target = agent_id == "agent_00"
                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    label=_agent_label(agent_id),
                    linewidth=2.8 if is_target else 1.2,
                    alpha=1.0 if is_target else 0.72,
                    color="black" if is_target else None,
                    zorder=5 if is_target else 2,
                )
            title = stage.capitalize() if stage != "unknown" else "All hands"
            ax.set_title(f"{title}: cumulative chip delta (reset at 0)")
            ax.set_xlabel(f"{title} hand index")
            ax.set_ylabel("Chip delta")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                fontsize=7,
                frameon=True,
                title="Agent",
                ncol=2,
            )
            ax.grid(True, alpha=0.3)
        path = output / "stack_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        path = output / "stack_curve.txt"
        path.write_text(f"绘图失败，已降级为文本占位：{exc}\n", encoding="utf-8")
        return str(path)


def generate_audit_plots(
    hand_summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    """Generate action, table-composition, and risk-outcome audit charts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        paths: list[str] = []
        actions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event in events:
            if event.get("event") == "action":
                actions[str(event.get("agent_id"))][str(event.get("action_type"))] += 1
        action_types = ["fold", "check", "call", "raise"]
        agent_ids = sorted(actions)
        fig, ax = plt.subplots(figsize=(max(8, len(agent_ids) * 0.8), 5.5))
        bottoms = [0] * len(agent_ids)
        for action_type in action_types:
            values = [actions[agent_id].get(action_type, 0) for agent_id in agent_ids]
            ax.bar(agent_ids, values, bottom=bottoms, label=action_type)
            bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
        ax.set_title("Action counts by agent")
        ax.set_ylabel("Decision count")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        action_path = output / "action_counts_by_agent.png"
        fig.savefig(action_path, dpi=140)
        plt.close(fig)
        paths.append(str(action_path))

        train_hands = [hand for hand in hand_summaries if hand.get("stage") == "train"]
        effective_counts = [
            sum(int(stack) > 0 for stack in (hand.get("starting_stacks", {}) or {}).values())
            for hand in train_hands
        ]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.step(range(1, len(effective_counts) + 1), effective_counts, where="post")
        ax.set_title("Effective players at each training hand start")
        ax.set_xlabel("Train hand")
        ax.set_ylabel("Effective players")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        players_path = output / "effective_players_curve.png"
        fig.savefig(players_path, dpi=140)
        plt.close(fig)
        paths.append(str(players_path))

        risk_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event in events:
            if event.get("event") != "action":
                continue
            facts = event.get("decision_facts") or {}
            label = str((facts.get("call") or {}).get("risk_label", "unknown"))
            risk_counts[str(event.get("agent_id"))][label] += 1
        labels = ["low", "medium", "high", "all_in"]
        fig, ax = plt.subplots(figsize=(max(8, len(agent_ids) * 0.8), 5.5))
        bottoms = [0] * len(agent_ids)
        for label in labels:
            values = [risk_counts[agent_id].get(label, 0) for agent_id in agent_ids]
            ax.bar(agent_ids, values, bottom=bottoms, label=label)
            bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
        ax.set_title("Rule-engine call-risk labels by agent")
        ax.set_ylabel("Decision count")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        risk_path = output / "call_risk_labels_by_agent.png"
        fig.savefig(risk_path, dpi=140)
        plt.close(fig)
        paths.append(str(risk_path))
        return paths
    except Exception as exc:  # noqa: BLE001
        path = output / "audit_plots.txt"
        path.write_text(f"审计图绘制失败：{exc}\n", encoding="utf-8")
        return [str(path)]
def _build_cumulative_by_stage(
    hand_summaries: list[dict[str, Any]],
) -> dict[str, dict[str, list[int]]]:
    """Build independent cumulative series so stage resets cannot be connected."""

    hands_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hand in hand_summaries:
        hands_by_stage[str(hand.get("stage", "unknown"))].append(hand)
    result: dict[str, dict[str, list[int]]] = {}
    for stage, hands in hands_by_stage.items():
        cumulative: dict[str, list[int]] = defaultdict(list)
        totals: dict[str, int] = defaultdict(int)
        known_agents: set[str] = set()
        for hand_index, hand in enumerate(hands, start=1):
            rewards = hand.get("rewards", {}) or {}
            for agent_id in rewards:
                if agent_id not in known_agents:
                    cumulative[agent_id] = [0] * (hand_index - 1)
                    known_agents.add(agent_id)
            for agent_id, reward in rewards.items():
                totals[agent_id] += int(reward)
            for agent_id in sorted(known_agents):
                cumulative[agent_id].append(totals[agent_id])
        result[stage] = dict(cumulative)
    return result


def _ordered_stages(cumulative_by_stage: dict[str, object]) -> list[str]:
    """Place train/test first while preserving deterministic order for other stages."""

    priority = {"train": 0, "test": 1}
    return sorted(cumulative_by_stage, key=lambda stage: (priority.get(stage, 2), stage))


def _agent_label(agent_id: str) -> str:
    """
    功能：把内部 Agent ID 转成图例可读标签。
    参数：
        agent_id：内部 ID。
    返回：图例标签。
    副作用：无。
    异常：无。
    设计说明：报告图应能区分目标、训练对手和泛化对手。
    """

    if agent_id == "agent_00":
        return "agent_00 (target)"
    if agent_id.startswith("agent_"):
        return f"{agent_id} (train)"
    if agent_id.startswith("heldout_"):
        return f"{agent_id} (heldout)"
    return agent_id
