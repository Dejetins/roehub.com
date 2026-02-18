from __future__ import annotations

import pytest

from apps.worker.strategy_live_runner.wiring.modules.strategy_live_runner import (
    _require_non_empty_env_value,
)


def test_require_non_empty_env_value_returns_secret() -> None:
    """
    Ensure helper returns configured non-empty environment variable value.
    """
    value = _require_non_empty_env_value(
        environ={"TELEGRAM_BOT_TOKEN": "token-value"},
        key="TELEGRAM_BOT_TOKEN",
        setting_name="strategy_live_runner.telegram.bot_token_env",
    )

    assert value == "token-value"


def test_require_non_empty_env_value_rejects_missing_or_blank_value() -> None:
    """
    Ensure helper fails fast when required environment variable is missing or blank.
    """
    with pytest.raises(ValueError):
        _require_non_empty_env_value(
            environ={},
            key="TELEGRAM_BOT_TOKEN",
            setting_name="strategy_live_runner.telegram.bot_token_env",
        )

    with pytest.raises(ValueError):
        _require_non_empty_env_value(
            environ={"TELEGRAM_BOT_TOKEN": "   "},
            key="TELEGRAM_BOT_TOKEN",
            setting_name="strategy_live_runner.telegram.bot_token_env",
        )
