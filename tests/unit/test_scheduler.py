"""
模块说明：本模块测试换桌调度器。
核心职责：覆盖 seed 可复现性、轮空记录和暴露统计。
输入与输出：输入 Agent 池和桌容量，输出 pytest 断言结果。
依赖边界：只依赖 experiments.schedulers。
不负责：不运行牌局。
"""

from agentmemeval.experiments.schedulers import TableRotationScheduler


def test_scheduler_is_reproducible_and_reports_exposure() -> None:
    """
    功能：验证相同 seed 生成相同排程并输出暴露统计。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：换桌实验的研究问题依赖可复现对手暴露。
    """

    agents = [f"agent_{index:02d}" for index in range(20)]
    left = TableRotationScheduler(agents, table_size=4, seed=42)
    right = TableRotationScheduler(agents, table_size=4, seed=42)
    rounds_left = [left.schedule_round(index) for index in range(3)]
    rounds_right = [right.schedule_round(index) for index in range(3)]
    assert [item.to_dict() for item in rounds_left] == [item.to_dict() for item in rounds_right]
    stats = left.exposure_stats(rounds_left)
    assert stats["pairwise_exposure_histogram"]
    assert stats["per_agent"]["agent_00"]["unique_opponents"] > 0


def test_scheduler_records_byes() -> None:
    """
    功能：验证不能整除桌容量时记录轮空。
    参数：无。
    返回：无。
    副作用：无。
    异常：断言失败时由 pytest 报告。
    设计说明：预算公平性需要显式记录轮空。
    """

    agents = [f"agent_{index:02d}" for index in range(5)]
    scheduler = TableRotationScheduler(agents, table_size=4, seed=1, mode="fixed")
    rotation = scheduler.schedule_round(0)
    assert rotation.byes == ["agent_04"]
