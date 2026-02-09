from __future__ import annotations

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
  scheduler:
    jobs:
      sync_whitelist: { interval_seconds: 900 }
      enrich: { interval_seconds: 7200 }
      rest_insurance_catchup: { interval_seconds: 3600 }
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
    assert cfg.scheduler.jobs.sync_whitelist.interval_seconds == 900
    assert cfg.backfill.max_days_per_insert == 7


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
    assert cfg.scheduler.jobs.sync_whitelist.interval_seconds == 3600
    assert cfg.scheduler.jobs.enrich.interval_seconds == 21600
    assert cfg.scheduler.jobs.rest_insurance_catchup.interval_seconds == 3600
