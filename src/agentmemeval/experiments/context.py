"""
模块说明：本模块定义实验运行上下文。
核心职责：集中保存 resolved config、Provider 和工件管理器。
输入与输出：输入运行器创建的依赖，输出上下文对象。
依赖边界：依赖核心协议和存储层，不依赖具体场景。
不负责：不运行实验，不计算指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmemeval.core.protocols import LLMClient
from agentmemeval.storage.artifacts import ArtifactManager


@dataclass(slots=True)
class ExperimentContext:
    """
    功能：保存一次实验运行的共享依赖。
    参数：
        config：resolved 配置。
        artifacts：工件管理器。
        llm_client：Provider 实例。
    返回：上下文对象。
    副作用：无。
    异常：无。
    设计说明：场景只通过上下文访问输出和模型，便于 runner 统一创建。
    """

    config: dict[str, Any]
    artifacts: ArtifactManager
    llm_client: LLMClient
