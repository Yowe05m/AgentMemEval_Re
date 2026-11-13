"""Canonical model-family identities used by independent-review contracts."""

from __future__ import annotations

import re


def canonical_model_family(value: str) -> str:
    """Collapse common aliases so one model family cannot self-certify."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("model family must be a non-empty string")
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    family_markers = (
        "deepseek",
        "qwen",
        "claude",
        "anthropic",
        "gemini",
        "openai",
        "chatgpt",
        "llama",
        "mistral",
        "minimax",
        "moonshot",
        "kimi",
    )
    for marker in family_markers:
        if re.search(rf"(?:^|-){re.escape(marker)}(?:[0-9]|-|$)", normalized):
            if marker == "anthropic":
                return "claude"
            if marker == "chatgpt":
                return "openai"
            if marker in {"moonshot", "kimi"}:
                return "kimi"
            return marker
    return normalized


__all__ = ["canonical_model_family"]
