from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.clients.files.parquet_candle_ingest_source import (  # noqa: E501
    ParquetCandleIngestSource,
    ParquetScanner,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketDataRuntimeConfig,
    load_market_data_runtime_config,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class FixedClock:
    def __init__(self, ts: UtcTimestamp) -> None:
        self._ts = ts

    def now(self) -> UtcTimestamp:
        return self._ts


class InMemoryScanner(ParquetScanner):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = list(rows)

    def scan_filtered(
        self,
        *,
        market_id: int,
        symbol: str,
        start_ts_open,
        end_ts_open,
        columns: Sequence[str],
        batch_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        # Симулируем фильтр parquet: market_id/symbol/ts_open in [start,end)
        for r in self._rows:
            if int(r["market_id"]) != market_id:
                continue
            if str(r["symbol"]).upper() != symbol.upper():
                continue
            ts = r["ts_open"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if not (start_ts_open <= ts < end_ts_open):
                continue
            yield {k: r.get(k) for k in columns}


def _ts(dt: datetime) -> UtcTimestamp:
    return UtcTimestamp(dt)


def _runtime_config(tmp_path: Path) -> MarketDataRuntimeConfig:
    """
    Create minimal runtime config with market_id=1 mapped to binance spot.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - Parsed `MarketDataRuntimeConfig`.

    Assumptions/Invariants:
    - YAML contains all required runtime-config fields.

    Errors/Exceptions:
    - Propagates config parser validation errors.

    Side effects:
    - Writes temporary YAML file in pytest temp directory.
    """
    yaml_text = """
version: 1
market_data:
  markets:
    - market_id: 1
      exchange: binance
      market_type: spot
      market_code: binance:spot
      rest:
        base_url: "https://api.binance.com"
        earliest_available_ts_utc: "2017-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.01, max_s: 0.01, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: "wss://example"
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
  ingestion:
    raw_write: { flush_interval_ms: 250, max_buffer_rows: 2000 }
  backfill:
    max_days_per_insert: 7
    chunk_align: "utc_day"
"""
    p = tmp_path / "market_data.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return load_market_data_runtime_config(p)


def test_parquet_source_requires_market_id_and_symbol_and_generates_instrument_key(
    tmp_path: Path,
) -> None:
    """
    Ensure parquet ingestion emits canonical instrument_key resolved from runtime config.

    Parameters:
    - tmp_path: pytest temp path fixture.

    Returns:
    - None.

    Assumptions/Invariants:
    - parquet rows include required market_id/symbol columns.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    rows = [
        {
            "market_id": 1,
            "symbol": "btcusdt",
            "ts_open": datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
            "ts_close": None,  # проверим вычисление
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume_base": 1.0,
            "volume_quote": 2.0,
            "trades_count": None,
            "taker_buy_volume_base": None,
            "taker_buy_volume_quote": None,
        }
    ]

    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    tr = TimeRange(_ts(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)), _ts(datetime(2026, 2, 1, 0, 2, tzinfo=timezone.utc))) # noqa: E501

    clock = FixedClock(_ts(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)))
    src = ParquetCandleIngestSource(
        scanner=InMemoryScanner(rows),
        cfg=_runtime_config(tmp_path),
        clock=clock,
    )

    out = list(src.stream_1m(instrument, tr))
    assert len(out) == 1

    row = out[0]
    assert int(row.candle.instrument_id.market_id.value) == 1
    assert str(row.candle.instrument_id.symbol) == "BTCUSDT"
    # ts_close computed as +1m
    assert row.candle.ts_close.value == row.candle.ts_open.value + timedelta(minutes=1)
    assert row.meta.source == "file"
    assert row.meta.ingested_at.value == datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)
    # instrument_key generated
    assert row.meta.instrument_key == "binance:spot:BTCUSDT"


def test_parquet_source_filters_by_time_range_semantics_half_interval(tmp_path: Path) -> None:
    """
    Verify parquet source respects half-open time interval semantics `[start, end)`.

    Parameters:
    - tmp_path: pytest temp path fixture.

    Returns:
    - None.

    Assumptions/Invariants:
    - scanner applies same half-open boundaries as source contract.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    rows = [
        {
            "market_id": 1,
            "symbol": "BTCUSDT",
            "ts_open": datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
            "ts_close": None,
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume_base": 1.0,
            "volume_quote": None,
            "trades_count": None,
            "taker_buy_volume_base": None,
            "taker_buy_volume_quote": None,
        },
        {
            "market_id": 1,
            "symbol": "BTCUSDT",
            "ts_open": datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc),
            "ts_close": None,
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume_base": 1.0,
            "volume_quote": None,
            "trades_count": None,
            "taker_buy_volume_base": None,
            "taker_buy_volume_quote": None,
        },
    ]

    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    # [00:00, 00:01) -> только первая
    tr = TimeRange(_ts(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)), _ts(datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc))) # noqa: E501

    clock = FixedClock(_ts(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)))
    src = ParquetCandleIngestSource(
        scanner=InMemoryScanner(rows),
        cfg=_runtime_config(tmp_path),
        clock=clock,
    )

    out = list(src.stream_1m(instrument, tr))
    assert len(out) == 1
    assert out[0].candle.ts_open.value == datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
