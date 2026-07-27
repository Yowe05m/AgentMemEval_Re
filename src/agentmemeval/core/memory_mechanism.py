"""Canonical memory-mechanism names without importing the eager memory package."""

from __future__ import annotations

from collections.abc import Mapping

MECHANISM_ALIASES: dict[str, str] = {
    "no_memory": "no_memory",
    "none": "no_memory",
    "naive": "no_memory",
    "fact": "fact",
    "FactAgent": "fact",
    "expr": "expr",
    "ExprAgent": "expr",
    "fact_expr_sync": "fact_expr_sync",
    "fxsync": "fact_expr_sync",
    "FactExprSync": "fact_expr_sync",
    "fact_expr_async": "fact_expr_async",
    "fxasync": "fact_expr_async",
    "FactExprAsync": "fact_expr_async",
}

FACTUAL_MECHANISMS = frozenset({"fact", "fact_expr_sync", "fact_expr_async"})
EXPERIENCE_MECHANISMS = frozenset({"expr", "fact_expr_sync", "fact_expr_async"})


def canonical_memory_mechanism(value: object) -> str:
    """Return one canonical mechanism name and reject unknown/coerced values."""

    if not isinstance(value, str) or value not in MECHANISM_ALIASES:
        raise ValueError(f"未知记忆机制：{value}")
    return MECHANISM_ALIASES[value]


def resolve_memory_mechanism(config: Mapping[str, object]) -> str:
    """Resolve ``mechanism``/legacy ``type`` and reject conflicts."""

    has_mechanism = "mechanism" in config
    has_type = "type" in config
    if not has_mechanism and not has_type:
        return "no_memory"
    mechanism = canonical_memory_mechanism(config["mechanism"]) if has_mechanism else None
    legacy_type = canonical_memory_mechanism(config["type"]) if has_type else None
    if mechanism is not None and legacy_type is not None and mechanism != legacy_type:
        raise ValueError(
            "memory mechanism/type declarations conflict: "
            f"{config['mechanism']} != {config['type']}"
        )
    return mechanism or legacy_type or "no_memory"
