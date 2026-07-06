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
    cumulative: dict[str, list[int]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for hand in hand_summaries:
        for agent_id, reward in (hand.get("rewards", {}) or {}).items():
            totals[agent_id] += int(reward)
        for agent_id in sorted(totals):
            cumulative[agent_id].append(totals[agent_id])
    try:
        mpl_config = Path(os.environ.get("MPLCONFIGDIR", output / ".matplotlib"))
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for agent_id, values in sorted(cumulative.items()):
            ax.plot(range(1, len(values) + 1), values, label=agent_id)
        ax.set_title("Cumulative chip delta")
        ax.set_xlabel("Hand index")
        ax.set_ylabel("Chip delta")
        if len(cumulative) <= 12:
            ax.legend(fontsize=7)
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
