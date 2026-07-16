from __future__ import annotations

import pytest

from agentmemeval.config.loader import validate_config
from agentmemeval.core.errors import ConfigError


def _valid_config() -> dict[str, object]:
    return {
        "experiment": {"scenario": "fixed_evolving_table", "seed": 1, "table_size": 4},
        "provider": {"name": "mock"},
        "table": {"small_blind": 1, "big_blind": 2, "starting_stack": 100},
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("experiment", "table_size", 1),
        ("experiment", "train_hands", -1),
        ("table", "starting_stack", 1),
        ("table", "lifecycle", "unknown"),
    ],
)
def test_validate_config_rejects_invalid_ranges(
    section: str,
    field: str,
    value: object,
) -> None:
    config = _valid_config()
    config[section][field] = value  # type: ignore[index]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_accepts_minimal_valid_config() -> None:
    validate_config(_valid_config())
