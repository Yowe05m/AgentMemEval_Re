"""
模块说明：本模块实现换桌调度器和对手暴露统计。
核心职责：按 seed 生成可复现的桌面分组，并量化 pairwise exposure。
输入与输出：输入 Agent 池、桌容量和轮次，输出桌面分配与统计。
依赖边界：只依赖核心 seed 工具和检索器中的熵函数。
不负责：不运行牌局，不创建 Agent。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from agentmemeval.core.domain import AgentId
from agentmemeval.core.seeds import make_rng
from agentmemeval.memory.retrievers import exposure_entropy


@dataclass(slots=True)
class TableAssignment:
    """
    功能：描述一轮换桌的一张桌。
    参数：
        table_id：桌号。
        seats：按座位顺序排列的 Agent。
    返回：桌面分配对象。
    副作用：无。
    异常：无。
    设计说明：座位顺序写入事件日志，便于复现实验。
    """

    table_id: str
    seats: list[AgentId]

    def to_dict(self) -> dict[str, object]:
        """
        功能：转换为 JSON 字典。
        参数：无。
        返回：字典。
        副作用：无。
        异常：无。
        设计说明：换桌事件日志使用该结构。
        """

        return {"table_id": self.table_id, "seats": list(self.seats)}


@dataclass(slots=True)
class RotationRound:
    """
    功能：描述一轮换桌排程。
    参数：
        round_index：轮次。
        tables：桌面分配列表。
        byes：轮空 Agent。
    返回：轮次对象。
    副作用：无。
    异常：无。
    设计说明：轮空显式记录，避免隐藏预算差异。
    """

    round_index: int
    tables: list[TableAssignment]
    byes: list[AgentId]

    def to_dict(self) -> dict[str, object]:
        """
        功能：转换为 JSON 字典。
        参数：无。
        返回：字典。
        副作用：无。
        异常：无。
        设计说明：写入 events.jsonl 的 rotation 事件。
        """

        return {
            "round_index": self.round_index,
            "tables": [table.to_dict() for table in self.tables],
            "byes": list(self.byes),
        }


class TableRotationScheduler:
    """
    功能：生成固定或随机均衡换桌排程。
    参数：
        agent_ids：Agent 池。
        table_size：桌容量。
        seed：调度 seed。
        mode：balanced 或 fixed。
    返回：调度器。
    副作用：无。
    异常：Agent 数量不足或桌容量非法时抛出 ValueError。
    设计说明：balanced 为 seed 驱动洗牌分组，不声称严格 pairwise 均衡。
    """

    def __init__(
        self,
        agent_ids: list[AgentId],
        table_size: int,
        seed: int,
        mode: str = "balanced",
    ) -> None:
        """
        功能：初始化调度器。
        参数：
            agent_ids：Agent 池。
            table_size：桌容量。
            seed：根 seed。
            mode：调度模式。
        返回：无。
        副作用：保存配置。
        异常：配置非法时抛出 ValueError。
        设计说明：N>=20 的要求由场景配置测试覆盖，调度器本身支持任意 N>=2。
        """

        if len(agent_ids) < 2:
            raise ValueError("换桌调度至少需要两个 Agent")
        if table_size < 2:
            raise ValueError("桌容量至少为 2")
        self.agent_ids = list(agent_ids)
        self.table_size = table_size
        self.seed = seed
        self.mode = mode

    def schedule_round(self, round_index: int) -> RotationRound:
        """
        功能：生成指定轮次的桌面分配。
        参数：
            round_index：轮次。
        返回：RotationRound。
        副作用：无。
        异常：无。
        设计说明：fixed 模式保持原始顺序，balanced 模式每轮 seed 洗牌。
        """

        if self.mode == "fixed":
            ordered = list(self.agent_ids)
        else:
            rng = make_rng(self.seed, "rotation", round_index)
            ordered = list(self.agent_ids)
            rng.shuffle(ordered)
        tables: list[TableAssignment] = []
        byes: list[AgentId] = []
        table_index = 0
        for start in range(0, len(ordered), self.table_size):
            seats = ordered[start : start + self.table_size]
            if len(seats) < 2:
                byes.extend(seats)
                continue
            table_id = f"round{round_index:03d}_table{table_index:02d}"
            tables.append(TableAssignment(table_id=table_id, seats=seats))
            table_index += 1
        return RotationRound(round_index=round_index, tables=tables, byes=byes)

    def exposure_stats(self, rounds: list[RotationRound]) -> dict[str, object]:
        """
        功能：计算对手暴露统计。
        参数：
            rounds：已执行轮次。
        返回：暴露统计字典。
        副作用：无。
        异常：无。
        设计说明：报告实际不均衡程度，而不是宣称严格均衡。
        """

        pair_counts: dict[tuple[str, str], int] = {}
        per_agent: dict[str, dict[str, int]] = {agent_id: {} for agent_id in self.agent_ids}
        for round_item in rounds:
            for table in round_item.tables:
                for left, right in combinations(table.seats, 2):
                    key = tuple(sorted((left, right)))
                    pair_counts[key] = pair_counts.get(key, 0) + 1
                    per_agent[left][right] = per_agent[left].get(right, 0) + 1
                    per_agent[right][left] = per_agent[right].get(left, 0) + 1
        pair_values = list(pair_counts.values())
        per_agent_summary = {}
        for agent_id, opponents in per_agent.items():
            counts = list(opponents.values())
            per_agent_summary[agent_id] = {
                "total_exposures": sum(counts),
                "unique_opponents": len(opponents),
                "exposure_entropy": exposure_entropy(counts),
                "opponents": dict(sorted(opponents.items())),
            }
        histogram: dict[str, int] = {}
        for value in pair_values:
            histogram[str(value)] = histogram.get(str(value), 0) + 1
        imbalance = (max(pair_values) - min(pair_values)) if pair_values else 0
        return {
            "mode": self.mode,
            "table_size": self.table_size,
            "rounds": len(rounds),
            "per_agent": per_agent_summary,
            "pairwise_exposure_histogram": histogram,
            "pairwise_imbalance": imbalance,
            "pair_count": len(pair_counts),
            "strict_pairwise_balanced": imbalance == 0 and bool(pair_values),
        }
