"""
模块说明：本模块提供轻量级 Provider 重试工具。
核心职责：为真实 Provider 调用提供统一重试边界。
输入与输出：输入可调用对象，输出调用结果。
依赖边界：只依赖标准库 time，不依赖具体 Provider。
不负责：不判断动作合法性，不记录完整日志。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from agentmemeval.core.errors import ProviderError

T = TypeVar("T")


def retry_call(fn: Callable[[], T], max_retries: int, delay_seconds: float = 0.2) -> tuple[T, int]:
    """
    功能：执行带固定延迟的重试调用。
    参数：
        fn：待执行函数。
        max_retries：最大重试次数。
        delay_seconds：每次失败后的等待时间。
    返回：结果与实际重试次数。
    副作用：失败时会 sleep。
    异常：全部失败后抛出 ProviderError。
    设计说明：真实 Provider 的失败处理集中管理，mock 不需要额外分支。
    """

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(delay_seconds)
    raise ProviderError(f"Provider 调用失败，已重试 {max_retries} 次：{last_error}") from last_error
