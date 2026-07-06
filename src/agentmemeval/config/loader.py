"""
模块说明：本模块负责 YAML 配置加载、继承合并和基础校验。
核心职责：把实验配置解析为单一 resolved config，供运行器快照保存。
输入与输出：输入配置路径，输出合并后的字典。
依赖边界：依赖 PyYAML 与标准库 pathlib，不依赖实验模块。
不负责：不创建 Provider，不运行实验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentmemeval.core.errors import ConfigError


def load_config(path: str | Path) -> dict[str, Any]:
    """
    功能：加载 YAML 配置并处理 extends。
    参数：
        path：配置文件路径。
    返回：合并后的配置字典。
    副作用：读取文件。
    异常：文件不存在或 YAML 结构非法时抛出 ConfigError。
    设计说明：配置集中在 YAML，避免实验参数散落在 Python 常量中。
    """

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在：{config_path}")
    config = _read_yaml(config_path)
    parent_name = config.pop("extends", None)
    if parent_name:
        parent_path = (config_path.parent / str(parent_name)).resolve()
        parent = load_config(parent_path)
        config = deep_merge(parent, config)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    功能：校验运行所需的关键配置字段。
    参数：
        config：配置字典。
    返回：无。
    副作用：无。
    异常：缺字段时抛出 ConfigError。
    设计说明：尽早失败，避免实验跑到一半才发现缺少 seed 或 scenario。
    """

    if "experiment" not in config:
        raise ConfigError("配置缺少 experiment 段")
    if "provider" not in config:
        raise ConfigError("配置缺少 provider 段")
    experiment = config["experiment"]
    if "scenario" not in experiment:
        raise ConfigError("experiment.scenario 不能为空")
    if "seed" not in experiment:
        raise ConfigError("experiment.seed 不能为空")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    功能：递归合并两个配置字典。
    参数：
        base：基础配置。
        override：覆盖配置。
    返回：合并结果。
    副作用：无。
    异常：无。
    设计说明：实验配置只写差异，resolved 快照保存完整结果。
    """

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def dump_yaml(data: dict[str, Any]) -> str:
    """
    功能：把配置字典转为稳定 YAML 文本。
    参数：
        data：配置字典。
    返回：YAML 字符串。
    副作用：无。
    异常：无。
    设计说明：resolved_config.yaml 由同一函数生成，便于比较。
    """

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    """
    功能：读取 YAML 文件。
    参数：
        path：文件路径。
    返回：字典。
    副作用：读取文件。
    异常：YAML 顶层不是字典时抛出 ConfigError。
    设计说明：私有函数集中处理文件格式错误。
    """

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败：{path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML 顶层必须是对象：{path}")
    return raw
