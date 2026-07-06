"""
模块说明：本模块提供泛化场景入口。
核心职责：复用 FixedTableScenario 的训练到快照再测试流程。
输入与输出：输入 ExperimentContext，输出 ExperimentResult。
依赖边界：依赖 fixed_table 模块，不复制场景逻辑。
不负责：不实现独立调度器。
"""

from agentmemeval.experiments.fixed_table import FixedTableScenario


class GeneralizationScenario(FixedTableScenario):
    """
    功能：命名化的泛化场景。
    参数：无。
    返回：场景实例。
    副作用：run 时写入工件。
    异常：由 FixedTableScenario 向上抛出。
    设计说明：保留独立场景名，便于配置和后续扩展更复杂泛化协议。
    """

    name = "generalization_table"
