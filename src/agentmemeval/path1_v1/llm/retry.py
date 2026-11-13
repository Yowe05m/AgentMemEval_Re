from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from agentmemeval.path1_v1.core.errors import ProviderError

T = TypeVar("T")


def retry_call(fn: Callable[[], T], max_retries: int, delay_seconds: float = 0.2) -> tuple[T, int]:

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if getattr(exc, "retryable", True) is False:
                raise
            if attempt >= max_retries:
                break
            time.sleep(delay_seconds)
    raise ProviderError(
        f"provider call failed after {max_retries} retries: {last_error}"
    ) from last_error
