"""
模块说明：本模块负责实验随机性的集中管理。
核心职责：提供稳定 seed 派生、标准库随机数初始化和可追踪 seed 记录。
输入与输出：输入基础 seed 与标签，输出整数 seed 或已初始化的随机数对象。
依赖边界：只依赖 hashlib 与 random，不依赖具体环境或模型。
不负责：不保证第三方 SDK 的随机性控制，不改变全局配置文件。
"""

from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, *parts: object) -> int:
    """
    功能：从基础 seed 和若干标签派生稳定的 32 位整数 seed。
    参数：
        base_seed：实验配置中的根 seed。
        parts：场景、桌号、手数等可追踪标签。
    返回：稳定整数 seed。
    副作用：无。
    异常：无。
    设计说明：跨机制比较时可以复用相同标签，减少排程随机性带来的噪声。
    """

    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_rng(base_seed: int, *parts: object) -> random.Random:
    """
    功能：创建局部随机数生成器。
    参数：
        base_seed：实验根 seed。
        parts：派生标签。
    返回：`random.Random` 实例。
    副作用：无。
    异常：无。
    设计说明：避免使用全局随机状态，让换桌调度和发牌可以独立复现。
    """

    return random.Random(derive_seed(base_seed, *parts))


def seed_global(base_seed: int) -> None:
    """
    功能：初始化 Python 标准库全局随机状态。
    参数：
        base_seed：实验根 seed。
    返回：无。
    副作用：调用 `random.seed`。
    异常：无。
    设计说明：大部分实现使用局部 RNG；该函数仅作为兼容性兜底。
    """

    random.seed(base_seed)
