from __future__ import annotations

from typing import Any

_ACTIVE_SECTION_LABELS = (
    "Starting Hands",
    "Postflop Play",
    "Turn and River",
    "Opponent Adaptation",
    "Bet Sizing and Position",
)

def active_experience_section_labels() -> tuple[str, ...]:
    """Return the maintained English section labels used by validators."""

    return _ACTIVE_SECTION_LABELS

def active_prompt_localization() -> dict[str, Any] | None:
    """Return no localization because the maintained English runtime stays active."""

    return None
