from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from trading.contexts.market_data.adapters.outbound.clients.common_http import HttpResponse
from trading.contexts.market_data.adapters.outbound.clients.funding_rate_history_source import (
    BINANCE_STANDARD_8H_NO_ADJUSTMENT_ROW,
    BYBIT_FUTURES_CATEGORY,
    RestFundingRateHistorySource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class _Clock:
    def now(self):
        return UtcTimestamp(datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc))


class _Http:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get_json(self, *, url, params, timeout_s, retries, backoff_base_s, backoff_max_s, backoff_jitter_s):  # noqa: E501
        self.calls.append((url, dict(params)))
        key = (url.rsplit("/", 1)[-1], params.get("cursor", ""))
        value = self.routes.get(key, self.routes.get(url.rsplit("/", 1)[-1]))
        if isinstance(value, Exception):
            raise value
        return HttpResponse(status_code=200, headers={}, body=value)


def _cfg(tmp_path):
    path = tmp_path / "market_data.yaml"
    path.write_text(
        """
version: 1
market_data:
  markets:
    - market_id: 2
      exchange: binance
      market_type: futures
      market_code: binance:futures
      rest:
        base_url: "https://fapi.binance.com"
        earliest_available_ts_utc: "2019-09-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.01, max_s: 0.01, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: "wss://x"
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
    - market_id: 4
      exchange: bybit
      market_type: futures
      market_code: bybit:futures
      rest:
        base_url: "https://api.bybit.com"
        earliest_available_ts_utc: "2018-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.01, max_s: 0.01, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: "wss://x"
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )
    return load_market_data_runtime_config(path)


def test_binance_universe_uses_adjusted_rows_and_standard_absent_rows(tmp_path) -> None:
    http = _Http(
        {
            "exchangeInfo": {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                    },
                    {
                        "symbol": "ETHUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "baseAsset": "ETH",
                        "quoteAsset": "USDT",
                    },
                ]
            },
            "fundingInfo": [
                {
                    "symbol": "BTCUSDT",
                    "fundingIntervalHours": 4,
                    "adjustedFundingRateCap": "0.01",
                    "adjustedFundingRateFloor": "-0.01",
                }
            ],
        }
    )
    src = RestFundingRateHistorySource(
        cfg=_cfg(tmp_path),
        http=http,
        clock=_Clock(),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
    )

    rows = src.list_funding_instruments(MarketId(2))
    by_symbol = {str(row.instrument_id.symbol): row for row in rows}

    assert by_symbol["BTCUSDT"].funding_interval_minutes == 240
    assert by_symbol["BTCUSDT"].funding_interval_source == "binance_fundingInfo"
    assert by_symbol["ETHUSDT"].funding_interval_minutes == 480
    assert by_symbol["ETHUSDT"].funding_interval_source == BINANCE_STANDARD_8H_NO_ADJUSTMENT_ROW


def test_binance_funding_info_global_failure_blocks_without_fallback(tmp_path) -> None:
    http = _Http(
        {
            "exchangeInfo": {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"}
                ]
            },
            "fundingInfo": RuntimeError("fundingInfo unavailable"),
        }
    )
    src = RestFundingRateHistorySource(
        cfg=_cfg(tmp_path),
        http=http,
        clock=_Clock(),
        ingest_id=UUID("00000000-0000-0000-0000-000000000002"),
    )

    with pytest.raises(RuntimeError):
        src.list_funding_instruments(MarketId(2))


def test_bybit_universe_uses_linear_and_degrades_missing_interval(tmp_path) -> None:
    http = _Http(
        {
            ("instruments-info", ""): {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "status": "Trading",
                            "baseCoin": "BTC",
                            "quoteCoin": "USDT",
                            "fundingInterval": "480",
                        },
                        {"symbol": "ETHUSDT", "status": "Trading"},
                        {"symbol": "SOLUSDT", "status": "Trading", "fundingInterval": "0"},
                    ],
                    "nextPageCursor": "",
                },
            }
        }
    )
    src = RestFundingRateHistorySource(
        cfg=_cfg(tmp_path),
        http=http,
        clock=_Clock(),
        ingest_id=UUID("00000000-0000-0000-0000-000000000003"),
    )

    rows = src.list_funding_instruments(MarketId(4))
    assert http.calls[0][1]["category"] == BYBIT_FUTURES_CATEGORY
    by_symbol = {str(row.instrument_id.symbol): row for row in rows}
    assert by_symbol["BTCUSDT"].funding_interval_minutes == 480
    assert by_symbol["ETHUSDT"].funding_interval_minutes is None
    assert by_symbol["SOLUSDT"].funding_interval_minutes is None
    assert by_symbol["SOLUSDT"].funding_interval_source is None


def test_bybit_history_uses_linear_category_and_filters_half_open_window(tmp_path) -> None:
    start = datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 22, 8, 0, tzinfo=timezone.utc)
    http = _Http(
        {
            ("history", ""): {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "fundingRate": "0.0001",
                            "fundingRateTimestamp": str(int(start.timestamp() * 1000)),
                        },
                        {
                            "symbol": "BTCUSDT",
                            "fundingRate": "0.0002",
                            "fundingRateTimestamp": str(int(end.timestamp() * 1000)),
                        },
                    ]
                },
            }
        }
    )
    src = RestFundingRateHistorySource(
        cfg=_cfg(tmp_path),
        http=http,
        clock=_Clock(),
        ingest_id=UUID("00000000-0000-0000-0000-000000000004"),
    )

    rows = src.list_funding_rates(
        instrument_id=InstrumentId(MarketId(4), Symbol("BTCUSDT")),
        time_range=TimeRange(UtcTimestamp(start), UtcTimestamp(end)),
        funding_interval_minutes=480,
        funding_interval_source="bybit_instruments_info",
    )

    assert http.calls[0][1]["category"] == BYBIT_FUTURES_CATEGORY
    assert len(rows) == 1
    assert rows[0].funding_rate == 0.0001
