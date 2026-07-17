from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)


def test_load_market_data_runtime_config_ok(tmp_path: Path) -> None:
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets:
    - market_id: 1
      exchange: binance
      market_type: spot
      market_code: binance:spot
      rest:
        base_url: https://example
        earliest_available_ts_utc: "2017-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 3
        backoff: { base_s: 0.5, max_s: 10.0, jitter_s: 0.2 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 4 }
      ws:
        url: wss://example
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 2000
    rest_concurrency_instruments: 7
    tail_lookback_minutes: 240
    rest_inter_instrument_delay_s: 2.0
  scheduler:
    jobs:
      refresh_catalog: { interval_seconds: 900 }
      enrich: { interval_seconds: 7200 }
      rest_insurance_catchup: { interval_seconds: 3600 }
      funding_rate_catchup:
        interval_seconds: 1800
        settlement_lag_minutes: 15
        binance_standard_interval_hours: 8
        allow_binance_funding_info_failure_fallback: false
        startup_bootstrap: true
        universe_refresh_interval_seconds: 21600
        tail_lookback_intervals: 3
        gap_audit_lookback_intervals: 21
        batch_size: 1000
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
""".strip(),
        encoding="utf-8",
    )

    cfg = load_market_data_runtime_config(p)
    assert cfg.version == 1
    assert cfg.market_ids() == (1,)
    assert cfg.raw_write.flush_interval_ms == 250
    assert cfg.ingestion.rest_concurrency_instruments == 7
    assert cfg.ingestion.tail_lookback_minutes == 240
    assert cfg.ingestion.rest_inter_instrument_delay_s == 2.0
    assert cfg.scheduler.jobs.refresh_catalog.interval_seconds == 900
    assert cfg.scheduler.jobs.funding_rate_catchup.interval_seconds == 1800
    assert cfg.scheduler.jobs.funding_rate_catchup.settlement_lag_minutes == 15
    assert cfg.scheduler.jobs.funding_rate_catchup.due_mode == "due_only"
    assert cfg.backfill.max_days_per_insert == 7
    assert str(cfg.markets[0].rest.earliest_available_ts_utc) == "2017-01-01T00:00:00.000Z"


def test_backfill_max_days_must_be_le_7(tmp_path: Path) -> None:
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  backfill: { max_days_per_insert: 8, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)


def test_flush_interval_must_be_le_500(tmp_path: Path) -> None:
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion:
    flush_interval_ms: 501
    max_buffer_rows: 1000
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)


def test_scheduler_defaults_when_section_is_missing(tmp_path: Path) -> None:
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    cfg = load_market_data_runtime_config(p)
    assert cfg.scheduler.jobs.refresh_catalog.interval_seconds == 3600
    assert cfg.scheduler.jobs.enrich.interval_seconds == 21600
    assert cfg.scheduler.jobs.rest_insurance_catchup.interval_seconds == 3600
    assert cfg.scheduler.jobs.funding_rate_catchup.interval_seconds == 1800
    assert cfg.scheduler.jobs.funding_rate_catchup.enabled is True
    assert (
        cfg.scheduler.jobs.funding_rate_catchup.allow_binance_funding_info_failure_fallback
        is False
    )
    assert cfg.ingestion.rest_inter_instrument_delay_s == 0.0


def test_funding_rate_catchup_due_mode_is_strict(tmp_path: Path) -> None:
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  scheduler:
    jobs:
      funding_rate_catchup: { due_mode: poll_all }
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)


def test_inter_instrument_delay_must_be_non_negative(tmp_path: Path) -> None:
    """
    Ensure ingestion inter-instrument delay rejects negative values.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - None.
    """
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 1000
    rest_inter_instrument_delay_s: -0.1
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)


def test_live_feed_defaults_to_disabled_when_section_is_missing(tmp_path: Path) -> None:
    """
    Ensure Redis streams live feed is disabled by default for backward compatibility.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - None.
    """
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    cfg = load_market_data_runtime_config(p)
    redis_streams = cfg.live_feed.redis_streams
    assert redis_streams.enabled is False
    assert redis_streams.host == "redis"
    assert redis_streams.port == 6379
    assert redis_streams.db == 0
    assert redis_streams.stream_mode == "per_instrument"
    assert redis_streams.stream_prefix == "md.candles.1m"
    assert redis_streams.maxlen_approx == 7 * 1440
    redis_hot_cache = cfg.live_feed.redis_hot_cache
    assert redis_hot_cache.enabled is False
    assert redis_hot_cache.key_prefix == "md:hot:1m"
    assert redis_hot_cache.retention_hours == 24
    assert redis_hot_cache.retention_ms == 24 * 60 * 60 * 1000


def test_live_feed_redis_sections_are_parsed_with_computed_defaults(tmp_path: Path) -> None:
    """
    Ensure parser reads live feed redis sections and computes defaults when omitted.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - None.
    """
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets: []
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  live_feed:
    redis_streams:
      enabled: true
      host: redis
      port: 6380
      db: 2
      password_env: "CUSTOM_REDIS_PASSWORD_ENV"
      socket_timeout_s: 1.5
      connect_timeout_s: 1.25
      stream_mode: per_instrument
      stream_prefix: md.candles.1m
      retention_days: 3
    redis_hot_cache:
      enabled: true
      key_prefix: md:hot:1m
      retention_hours: 12
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )

    cfg = load_market_data_runtime_config(p)
    redis_streams = cfg.live_feed.redis_streams
    assert redis_streams.enabled is True
    assert redis_streams.host == "redis"
    assert redis_streams.port == 6380
    assert redis_streams.db == 2
    assert redis_streams.password_env == "CUSTOM_REDIS_PASSWORD_ENV"
    assert redis_streams.socket_timeout_s == 1.5
    assert redis_streams.connect_timeout_s == 1.25
    assert redis_streams.stream_mode == "per_instrument"
    assert redis_streams.stream_prefix == "md.candles.1m"
    assert redis_streams.retention_days == 3
    assert redis_streams.maxlen_approx == 3 * 1440
    redis_hot_cache = cfg.live_feed.redis_hot_cache
    assert redis_hot_cache.enabled is True
    assert redis_hot_cache.key_prefix == "md:hot:1m"
    assert redis_hot_cache.retention_hours == 12
    assert redis_hot_cache.retention_ms == 12 * 60 * 60 * 1000


def test_market_earliest_available_timestamp_must_not_be_in_future(tmp_path: Path) -> None:
    """
    Ensure parser rejects market earliest boundary timestamps that are in the future.

    Parameters:
    - tmp_path: pytest temp path fixture.

    Returns:
    - None.
    """
    future_ts = (datetime.now(tz=timezone.utc) + timedelta(days=2)).isoformat()
    p = tmp_path / "market_data.yaml"
    p.write_text(
        f"""
version: 1
market_data:
  markets:
    - market_id: 1
      exchange: binance
      market_type: spot
      market_code: binance:spot
      rest:
        base_url: https://example
        earliest_available_ts_utc: "{future_ts}"
        timeout_s: 10.0
        retries: 1
        backoff: {{ base_s: 0.5, max_s: 10.0, jitter_s: 0.2 }}
        limiter: {{ mode: autodetect, safety_factor: 0.8, max_concurrency: 4 }}
      ws:
        url: wss://example
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: {{ min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }}
        max_symbols_per_connection: 200
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 2000
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)


def test_market_earliest_available_timestamp_is_required(tmp_path: Path) -> None:
    """
    Ensure parser requires explicit `rest.earliest_available_ts_utc` key per market.

    Parameters:
    - tmp_path: pytest temp path fixture.

    Returns:
    - None.
    """
    p = tmp_path / "market_data.yaml"
    p.write_text(
        """
version: 1
market_data:
  markets:
    - market_id: 1
      exchange: binance
      market_type: spot
      market_code: binance:spot
      rest:
        base_url: https://example
        timeout_s: 10.0
        retries: 1
        backoff: { base_s: 0.5, max_s: 10.0, jitter_s: 0.2 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 4 }
      ws:
        url: wss://example
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 2000
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_market_data_runtime_config(p)
