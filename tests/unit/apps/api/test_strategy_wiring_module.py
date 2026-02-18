from __future__ import annotations

from pathlib import Path

import pytest

from apps.api.wiring.modules.strategy import is_strategy_api_enabled


def _write_strategy_config(tmp_path: Path, *, api_enabled: bool) -> Path:
    """
    Write minimal valid Strategy runtime config for API toggle tests.

    Args:
        tmp_path: pytest temporary path fixture.
        api_enabled: Desired `strategy.api.enabled` toggle value.
    Returns:
        Path: Written config path.
    Assumptions:
        Minimal payload still contains mandatory live-worker sections.
    Raises:
        OSError: If write fails.
    Side Effects:
        Creates one temporary YAML file.
    """
    path = tmp_path / "strategy.yaml"
    path.write_text(
        (
            "version: 1\n"
            "strategy:\n"
            f"  api:\n    enabled: {'true' if api_enabled else 'false'}\n"
            "  live_worker:\n"
            "    enabled: true\n"
            "    poll_interval_seconds: 5\n"
            "    redis_streams:\n"
            "      enabled: true\n"
            "      host: redis\n"
            "      port: 6379\n"
            "      db: 0\n"
            "      socket_timeout_s: 2.0\n"
            "      connect_timeout_s: 2.0\n"
            "      stream_prefix: md.candles.1m\n"
            "      consumer_group: strategy.live_runner.v1\n"
            "      read_count: 100\n"
            "      block_ms: 100\n"
            "  realtime_output:\n"
            "    redis_streams:\n"
            "      enabled: false\n"
            "      host: redis\n"
            "      port: 6379\n"
            "      db: 0\n"
            "      socket_timeout_s: 2.0\n"
            "      connect_timeout_s: 2.0\n"
            "      metrics_stream_prefix: strategy.metrics.v1.user\n"
            "      events_stream_prefix: strategy.events.v1.user\n"
            "  telegram:\n"
            "    enabled: false\n"
            "    mode: log_only\n"
            "    bot_token_env: TELEGRAM_BOT_TOKEN\n"
            "    api_base_url: https://api.telegram.org\n"
            "    send_timeout_s: 2.0\n"
            "    debounce_failed_seconds: 600\n"
            "  metrics:\n"
            "    port: 9203\n"
        ),
        encoding="utf-8",
    )
    return path


def test_is_strategy_api_enabled_reads_yaml_toggle(tmp_path: Path) -> None:
    """
    Verify Strategy API toggle is read from source-of-truth `strategy.yaml`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `ROEHUB_STRATEGY_CONFIG` points to temporary runtime config file.
    Raises:
        AssertionError: If toggle value does not match YAML.
    Side Effects:
        None.
    """
    disabled_path = _write_strategy_config(tmp_path, api_enabled=False)

    assert (
        is_strategy_api_enabled(
            environ={"ROEHUB_STRATEGY_CONFIG": str(disabled_path)},
        )
        is False
    )

    enabled_path = _write_strategy_config(tmp_path, api_enabled=True)
    assert (
        is_strategy_api_enabled(
            environ={"ROEHUB_STRATEGY_CONFIG": str(enabled_path)},
        )
        is True
    )


def test_is_strategy_api_enabled_env_override_has_priority(tmp_path: Path) -> None:
    """
    Verify `ROEHUB_STRATEGY_API_ENABLED` override has higher priority than YAML toggle.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Scalar env overrides follow STR-EPIC-06 precedence contract.
    Raises:
        AssertionError: If env override does not affect result.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(tmp_path, api_enabled=True)

    assert (
        is_strategy_api_enabled(
            environ={
                "ROEHUB_STRATEGY_CONFIG": str(config_path),
                "ROEHUB_STRATEGY_API_ENABLED": "0",
            },
        )
        is False
    )


def test_is_strategy_api_enabled_rejects_invalid_override_literal(tmp_path: Path) -> None:
    """
    Verify invalid boolean env override literal fails fast with deterministic error.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Boolean env parser accepts strict literals only.
    Raises:
        AssertionError: If invalid literal does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(tmp_path, api_enabled=True)

    with pytest.raises(ValueError, match="ROEHUB_STRATEGY_API_ENABLED"):
        is_strategy_api_enabled(
            environ={
                "ROEHUB_STRATEGY_CONFIG": str(config_path),
                "ROEHUB_STRATEGY_API_ENABLED": "enabled",
            },
        )
