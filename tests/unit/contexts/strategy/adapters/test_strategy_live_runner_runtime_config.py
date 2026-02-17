from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.strategy.adapters.outbound.config import (
    load_strategy_live_runner_runtime_config,
)


def test_load_strategy_live_runner_runtime_config_parses_defaults() -> None:
    """
    Ensure runtime config parser reads valid YAML and applies deterministic defaults.
    """
    path = Path("configs/test/strategy_live_runner.yaml")

    cfg = load_strategy_live_runner_runtime_config(path)

    assert cfg.version == 1
    assert cfg.poll_interval_seconds == 1
    assert cfg.redis_streams.enabled is True
    assert cfg.redis_streams.stream_prefix == "md.candles.1m"
    assert cfg.redis_streams.consumer_group == "strategy.live_runner.v1"
    assert cfg.repair.retry_attempts == 0
    assert cfg.repair.retry_backoff_seconds == 0.0


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
