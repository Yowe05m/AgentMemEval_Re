"""
模块说明：本模块实现配置到实验结果的运行入口。
核心职责：加载配置、创建 Provider 和 ArtifactManager，并调用注册场景。
输入与输出：输入配置路径或字典，输出 ExperimentResult。
依赖边界：依赖配置、Provider 路由、场景注册和存储层。
不负责：不实现具体实验逻辑。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import load_config, validate_config
from agentmemeval.core.domain import ExperimentResult
from agentmemeval.core.seeds import seed_global
from agentmemeval.experiments.admission import assess_run_admission
from agentmemeval.experiments.context import ExperimentContext
from agentmemeval.experiments.registry import get_scenario
from agentmemeval.llm.router import build_llm_client
from agentmemeval.storage.artifacts import ArtifactManager, make_run_id


def run_config(path: str | Path) -> ExperimentResult:
    """
    功能：从配置文件运行实验。
    参数：
        path：YAML 配置路径。
    返回：ExperimentResult。
    副作用：创建输出目录并写入工件。
    异常：配置、Provider 或场景错误向上抛出。
    设计说明：CLI run 命令只调用该函数，保持入口简单。
    """

    config = load_config(path)
    return run_resolved_config(config)


def run_resolved_config(config: dict[str, Any]) -> ExperimentResult:
    """
    功能：运行已解析配置。
    参数：
        config：resolved 配置。
    返回：ExperimentResult。
    副作用：创建输出目录并写入工件。
    异常：配置、Provider 或场景错误向上抛出。
    设计说明：测试可直接传入字典，避免临时文件。
    """

    config = copy.deepcopy(config)
    validate_config(config)
    admission = assess_run_admission(config, Path.cwd())
    config["experiment"]["admission_audit"] = admission
    experiment = config["experiment"]
    scenario_name = str(experiment["scenario"])
    seed = int(experiment["seed"])
    seed_global(seed)
    run_id = str(experiment.get("run_id") or make_run_id(scenario_name, seed))
    output_root = Path(str(experiment.get("output_root", "outputs")))
    artifacts = ArtifactManager(output_root, run_id, config)
    llm_client = build_llm_client(config["provider"])
    context = ExperimentContext(config=config, artifacts=artifacts, llm_client=llm_client)
    scenario = get_scenario(scenario_name)
    return scenario.run(context)
