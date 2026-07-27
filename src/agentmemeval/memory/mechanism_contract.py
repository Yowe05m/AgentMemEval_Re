"""Backward-compatible re-export of the cycle-free core mechanism contract."""

from agentmemeval.core.memory_mechanism import (  # noqa: F401
    EXPERIENCE_MECHANISMS,
    FACTUAL_MECHANISMS,
    MECHANISM_ALIASES,
    canonical_memory_mechanism,
    resolve_memory_mechanism,
)
