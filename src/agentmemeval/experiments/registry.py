"""
模块说明：本模块注册可运行实验场景。
核心职责：把配置中的 scenario 名称映射到场景类。
输入与输出：输入场景名，输出 ExperimentScenario。
依赖边界：只导入场景类，不读取配置文件。
不负责：不创建运行上下文。
"""

from __future__ import annotations

from agentmemeval.core.errors import ConfigError
from agentmemeval.core.protocols import ExperimentScenario
from agentmemeval.experiments.fixed_table import FixedTableScenario
from agentmemeval.experiments.generalization import GeneralizationScenario
from agentmemeval.experiments.rotating_table import RotatingTableScenario


def get_scenario(name: str) -> ExperimentScenario:
    """
    功能：按名称返回场景实例。
    参数：
        name：场景名。
    返回：ExperimentScenario。
    副作用：无。
    异常：未知场景时抛出 ConfigError。
    设计说明：新增场景只需在注册表加入映射，不改 runner 主流程。
    """

    table = {
        "fixed_evolving_table": FixedTableScenario,
        "paper_baseline": FixedTableScenario,
        "generalization_table": GeneralizationScenario,
        "rotating_table": RotatingTableScenario,
        "rotating_20_agents": RotatingTableScenario,
    }
    cls = table.get(name)
    if cls is None:
        raise ConfigError(f"未知实验场景：{name}")
    return cls()
