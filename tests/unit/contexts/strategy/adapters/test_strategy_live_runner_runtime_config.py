from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.strategy.adapters.outbound.config import (
    load_strategy_live_runner_runtime_config,
)


def test_load_strategy_live_runner_runtime_config_parses_shim_file() -> None:
    """
    Ensure shim `strategy_live_runner.yaml` resolves to source-of-truth `strategy.yaml`.
    """
    path = Path("configs/test/strategy_live_runner.yaml")

    cfg = load_strategy_live_runner_runtime_config(path)

    assert cfg.version == 1
    assert cfg.live_worker_enabled is True
    assert cfg.poll_interval_seconds == 1
    assert cfg.metrics_port == 9207
    assert cfg.producer.enabled is False
    assert cfg.producer.allow_all is False
    assert cfg.producer.allowed_modes == ("paper", "testnet")
    assert cfg.producer.allowed_user_ids == ()
    assert cfg.producer.allowed_strategy_ids == ()

    assert cfg.redis_streams.enabled is True
    assert cfg.redis_streams.host == "localhost"
    assert cfg.redis_streams.stream_prefix == "md.candles.1m"
    assert cfg.redis_streams.consumer_group == "strategy.live_runner.v1"
    assert cfg.redis_streams.pending_claim_min_idle_ms == 0

    assert cfg.realtime_output.enabled is True
    assert cfg.realtime_output.metrics_stream_prefix == "strategy.metrics.v1.user"
    assert cfg.realtime_output.events_stream_prefix == "strategy.events.v1.user"

    assert cfg.telegram.enabled is True
    assert cfg.telegram.mode == "notifications"
    assert cfg.telegram.bot_token_env is None
    assert cfg.telegram.api_base_url == "https://api.telegram.org"
    assert cfg.telegram.send_timeout_s == 1.0
    assert cfg.telegram.debounce_failed_seconds == 600

    assert cfg.repair.retry_attempts == 0
    assert cfg.repair.retry_backoff_seconds == 0.0


def test_load_strategy_live_runner_runtime_config_parses_direct_strategy_yaml() -> None:
    """
    Ensure loader accepts direct `strategy.yaml` path for compatibility with new defaults.
    """
    cfg = load_strategy_live_runner_runtime_config(Path("configs/dev/strategy.yaml"))

    assert cfg.version == 1
    assert cfg.live_worker_enabled is True
    assert cfg.poll_interval_seconds == 5
    assert cfg.metrics_port == 9207
    assert cfg.producer.enabled is False
    assert cfg.producer.allowed_modes == ("paper", "testnet")
    assert cfg.redis_streams.pending_claim_min_idle_ms == 0


def test_load_strategy_live_runner_runtime_config_supports_legacy_payload(
    tmp_path: Path,
) -> None:
    """
    Ensure legacy full payload keeps working after shim migration.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
  poll_interval_seconds: 3
  redis_streams:
    enabled: true
    host: redis
    port: 6379
    db: 0
    socket_timeout_s: 2.0
    connect_timeout_s: 2.0
    stream_prefix: md.candles.1m
    consumer_group: strategy.live_runner.v1
    read_count: 200
    block_ms: 100
    pending_claim_min_idle_ms: 150
  realtime_output:
    enabled: false
  telegram:
    enabled: false
    mode: log_only
    bot_token_env: TELEGRAM_BOT_TOKEN
    api_base_url: https://api.telegram.org
    send_timeout_s: 2.0
    debounce_failed_seconds: 600
  repair:
    retry_attempts: 3
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    cfg = load_strategy_live_runner_runtime_config(config_path)

    assert cfg.version == 1
    assert cfg.live_worker_enabled is True
    assert cfg.poll_interval_seconds == 3
    assert cfg.metrics_port == 9207
    assert cfg.redis_streams.enabled is True
    assert cfg.redis_streams.pending_claim_min_idle_ms == 150
    assert cfg.realtime_output.enabled is False
    assert cfg.telegram.enabled is False
    assert cfg.producer.enabled is False
    assert cfg.producer.allowed_modes == ("paper", "testnet")


def test_load_strategy_live_runner_runtime_config_supports_producer_env_overrides() -> None:
    """
    Ensure producer admin switch and allowlists can be controlled without secrets in YAML.
    """
    cfg = load_strategy_live_runner_runtime_config(
        Path("configs/test/strategy_live_runner.yaml"),
        environ={
            "ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED": "true",
            "ROEHUB_STRATEGY_PRODUCER_ALLOW_ALL": "false",
            "ROEHUB_STRATEGY_PRODUCER_ALLOWED_MODES": "paper,testnet",
            "ROEHUB_STRATEGY_PRODUCER_ALLOWED_USER_IDS": (
                "00000000-0000-0000-0000-000000000901"
            ),
            "ROEHUB_STRATEGY_PRODUCER_ALLOWED_STRATEGY_IDS": (
                "00000000-0000-0000-0000-000000000902"
            ),
        },
    )

    assert cfg.producer.enabled is True
    assert cfg.producer.allow_all is False
    assert cfg.producer.allowed_modes == ("paper", "testnet")
    assert cfg.producer.allowed_user_ids == ("00000000-0000-0000-0000-000000000901",)
    assert cfg.producer.allowed_strategy_ids == (
        "00000000-0000-0000-0000-000000000902",
    )


def test_load_strategy_live_runner_runtime_config_rejects_live_producer_mode(
    tmp_path: Path,
) -> None:
    """
    Ensure supervised strategy producer config cannot enable live/mainnet mode.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
  poll_interval_seconds: 3
  redis_streams:
    enabled: true
    host: redis
    port: 6379
    db: 0
    socket_timeout_s: 2.0
    connect_timeout_s: 2.0
    stream_prefix: md.candles.1m
    consumer_group: strategy.live_runner.v1
    read_count: 200
    block_ms: 100
  producer:
    enabled: true
    allow_all: true
    allowed_modes:
      - paper
      - live
  repair:
    retry_attempts: 3
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper/testnet"):
        load_strategy_live_runner_runtime_config(config_path)


def test_load_strategy_live_runner_runtime_config_rejects_invalid_poll_interval(
    tmp_path: Path,
) -> None:
    """
    Ensure parser rejects non-positive polling interval.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
  poll_interval_seconds: 0
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
  repair:
    retry_attempts: 1
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_strategy_live_runner_runtime_config(config_path)


def test_load_strategy_live_runner_runtime_config_rejects_invalid_realtime_prefix(
    tmp_path: Path,
) -> None:
    """
    Ensure parser rejects invalid realtime output stream prefix when feature is enabled.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
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
    enabled: true
    host: redis
    port: 6379
    db: 0
    socket_timeout_s: 2.0
    connect_timeout_s: 2.0
    metrics_stream_prefix: "   "
    events_stream_prefix: strategy.events.v1.user
  repair:
    retry_attempts: 1
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_strategy_live_runner_runtime_config(config_path)


def test_load_strategy_live_runner_runtime_config_rejects_invalid_telegram_mode(
    tmp_path: Path,
) -> None:
    """
    Ensure parser rejects unsupported Telegram notifier mode value.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
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
  telegram:
    enabled: true
    mode: invalid_mode
    bot_token_env: TELEGRAM_BOT_TOKEN
    api_base_url: https://api.telegram.org
    send_timeout_s: 2.0
    debounce_failed_seconds: 600
  repair:
    retry_attempts: 1
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_strategy_live_runner_runtime_config(config_path)


def test_load_strategy_live_runner_runtime_config_rejects_direct_telegram_mode(
    tmp_path: Path,
) -> None:
    """
    Ensure direct raw-token `telegram` mode is rejected in favor of providers.
    """
    config_path = tmp_path / "strategy_live_runner.yaml"
    config_path.write_text(
        """
version: 1
strategy_live_runner:
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
  telegram:
    enabled: true
    mode: telegram
    bot_token_env: "   "
    api_base_url: https://api.telegram.org
    send_timeout_s: 2.0
    debounce_failed_seconds: 600
  repair:
    retry_attempts: 1
    retry_backoff_seconds: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_strategy_live_runner_runtime_config(config_path)
