from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.strategy.adapters.outbound.config import (
    load_strategy_runtime_config,
    resolve_strategy_config_path,
)


def _write_strategy_config(tmp_path: Path, *, body: str) -> Path:
    """
    Write temporary Strategy runtime YAML used by config-loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
        body: Full YAML content.
    Returns:
        Path: Written config path.
    Assumptions:
        Input text is valid UTF-8.
    Raises:
        OSError: If write fails.
    Side Effects:
        Creates one temp file.
    """
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_strategy_runtime_config_reads_yaml_sections(tmp_path: Path) -> None:
    """
    Verify loader parses source-of-truth `strategy.yaml` structure and defaults.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        YAML payload follows STR-EPIC-06 schema.
    Raises:
        AssertionError: If parsed values do not match input payload.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(
        tmp_path,
        body="""
version: 1
strategy:
  api:
    enabled: true
  live_worker:
    enabled: true
    poll_interval_seconds: 7
    redis_streams:
      enabled: true
      host: redis
      port: 6379
      db: 1
      password_env: ROEHUB_REDIS_PASSWORD
      socket_timeout_s: 2.5
      connect_timeout_s: 2.5
      stream_prefix: md.candles.1m
      consumer_group: strategy.live_runner.v1
      read_count: 10
      block_ms: 0
    repair:
      retry_attempts: 2
      retry_backoff_seconds: 1.5
  realtime_output:
    redis_streams:
      enabled: true
      host: redis
      port: 6380
      db: 2
      password_env: ROEHUB_REDIS_PASSWORD
      socket_timeout_s: 1.5
      connect_timeout_s: 1.5
      metrics_stream_prefix: strategy.metrics.v1.user
      events_stream_prefix: strategy.events.v1.user
  telegram:
    enabled: true
    mode: log_only
    bot_token_env: TELEGRAM_BOT_TOKEN
    api_base_url: https://api.telegram.org
    send_timeout_s: 1.0
    debounce_failed_seconds: 120
  metrics:
    port: 9300
""".strip(),
    )

    config = load_strategy_runtime_config(config_path, environ={})

    assert config.version == 1
    assert config.api.enabled is True
    assert config.live_worker.enabled is True
    assert config.live_worker.poll_interval_seconds == 7
    assert config.live_worker.redis_streams.host == "redis"
    assert config.live_worker.redis_streams.port == 6379
    assert config.live_worker.redis_streams.db == 1
    assert config.live_worker.repair.retry_attempts == 2
    assert config.live_worker.repair.retry_backoff_seconds == 1.5
    assert config.realtime_output.redis_streams.enabled is True
    assert config.realtime_output.redis_streams.port == 6380
    assert config.telegram.enabled is True
    assert config.telegram.mode == "log_only"
    assert config.metrics.port == 9300


def test_load_strategy_runtime_config_scalar_env_overrides_have_priority(
    tmp_path: Path,
) -> None:
    """
    Verify scalar env overrides have priority over YAML values for whitelisted keys.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Env overrides are limited to STR-EPIC-06 whitelist.
    Raises:
        AssertionError: If env values do not override YAML values.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(
        tmp_path,
        body="""
version: 1
strategy:
  api:
    enabled: true
  live_worker:
    enabled: true
    poll_interval_seconds: 5
    redis_streams:
      enabled: true
      host: redis
      port: 6379
      db: 0
      socket_timeout_s: 2.0
      connect_timeout_s: 2.0
      stream_prefix: md.candles.1m
      consumer_group: strategy.live_runner.v1
      read_count: 100
      block_ms: 100
  realtime_output:
    redis_streams:
      enabled: true
      host: redis
      port: 6379
      db: 0
      socket_timeout_s: 2.0
      connect_timeout_s: 2.0
      metrics_stream_prefix: strategy.metrics.v1.user
      events_stream_prefix: strategy.events.v1.user
  telegram:
    enabled: true
    mode: log_only
    bot_token_env: TELEGRAM_BOT_TOKEN
    api_base_url: https://api.telegram.org
    send_timeout_s: 2.0
    debounce_failed_seconds: 600
  metrics:
    port: 9203
""".strip(),
    )
    environ = {
        "ROEHUB_STRATEGY_API_ENABLED": "0",
        "ROEHUB_STRATEGY_LIVE_WORKER_ENABLED": "false",
        "ROEHUB_STRATEGY_REALTIME_OUTPUT_REDIS_STREAMS_ENABLED": "no",
        "ROEHUB_STRATEGY_TELEGRAM_ENABLED": "off",
        "ROEHUB_STRATEGY_METRICS_PORT": "9100",
    }

    config = load_strategy_runtime_config(config_path, environ=environ)

    assert config.api.enabled is False
    assert config.live_worker.enabled is False
    assert config.realtime_output.redis_streams.enabled is False
    assert config.telegram.enabled is False
    assert config.metrics.port == 9100


def test_load_strategy_runtime_config_rejects_invalid_boolean_override(
    tmp_path: Path,
) -> None:
    """
    Verify strict boolean parser rejects unsupported scalar override literals.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Boolean overrides accept only 1/0/true/false/yes/no/on/off.
    Raises:
        AssertionError: If invalid boolean literal does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(
        tmp_path,
        body="""
version: 1
strategy:
  live_worker:
    redis_streams:
      enabled: true
      host: redis
      port: 6379
      db: 0
      socket_timeout_s: 2.0
      connect_timeout_s: 2.0
      stream_prefix: md.candles.1m
      consumer_group: strategy.live_runner.v1
      read_count: 100
      block_ms: 100
""".strip(),
    )

    with pytest.raises(ValueError, match="ROEHUB_STRATEGY_API_ENABLED"):
        load_strategy_runtime_config(
            config_path,
            environ={"ROEHUB_STRATEGY_API_ENABLED": "enabled"},
        )


def test_resolve_strategy_config_path_precedence() -> None:
    """
    Verify Strategy config path resolution priority is CLI > env > ROEHUB_ENV fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fallback path format is `configs/<env>/strategy.yaml`.
    Raises:
        AssertionError: If precedence order differs from STR-EPIC-06 contract.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_STRATEGY_CONFIG": "configs/test/custom-strategy.yaml",
    }

    assert resolve_strategy_config_path(
        environ=environ,
        cli_config_path="configs/dev/from-cli.yaml",
    ) == Path("configs/dev/from-cli.yaml")
    assert resolve_strategy_config_path(environ=environ) == Path(
        "configs/test/custom-strategy.yaml"
    )

    assert resolve_strategy_config_path(environ={"ROEHUB_ENV": "test"}) == Path(
        "configs/test/strategy.yaml"
    )


def test_resolve_strategy_config_path_rejects_invalid_env_name() -> None:
    """
    Verify fallback path resolver rejects unsupported `ROEHUB_ENV` values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed env names are `dev`, `prod`, `test`.
    Raises:
        AssertionError: If invalid env value does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="ROEHUB_ENV"):
        resolve_strategy_config_path(environ={"ROEHUB_ENV": "staging"})
