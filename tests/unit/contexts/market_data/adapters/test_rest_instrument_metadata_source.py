from __future__ import annotations

from pathlib import Path

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    HttpResponse,
)
from trading.contexts.market_data.adapters.outbound.clients.rest_instrument_metadata_source import (
    RestInstrumentMetadataSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.shared_kernel.primitives import MarketId


class _FakeHttp:
    """HTTP fake returning preconfigured responses in call order."""

    def __init__(self, responses):  # noqa: ANN001
        """Store response payload queue used by `get_json` calls."""
        self._responses = list(responses)

    def get_json(self, *, url, params, timeout_s, retries, backoff_base_s, backoff_max_s, backoff_jitter_s):  # noqa: E501, ANN001
        """Return next queued payload and ignore transport settings."""
        _ = url
        _ = params
        _ = timeout_s
        _ = retries
        _ = backoff_base_s
        _ = backoff_max_s
        _ = backoff_jitter_s
        return HttpResponse(status_code=200, headers={}, body=self._responses.pop(0))


def _config(tmp_path: Path) -> Path:
    """
    Create runtime config file with Binance and Bybit markets for metadata tests.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - Path to generated config file.
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
        base_url: https://api.binance.com
        earliest_available_ts_utc: "2017-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.1, max_s: 0.1, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: wss://stream.binance.com:9443/stream
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
    - market_id: 3
      exchange: bybit
      market_type: spot
      market_code: bybit:spot
      rest:
        base_url: https://api.bybit.com
        earliest_available_ts_utc: "2018-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.1, max_s: 0.1, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: wss://stream.bybit.com/v5/public/spot
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 10
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 1000
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
"""
    path = tmp_path / "market_data.yaml"
    path.write_text(yaml_text.strip(), encoding="utf-8")
    return path


def test_metadata_source_parses_binance_exchange_info(tmp_path: Path) -> None:
    """Ensure Binance exchangeInfo payload is mapped to enrichment metadata fields."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    source = RestInstrumentMetadataSource(
        cfg=cfg,
        http=_FakeHttp(
            [
                {
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "baseAsset": "BTC",
                            "quoteAsset": "USDT",
                            "filters": [
                                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                {"filterType": "LOT_SIZE", "stepSize": "0.0001"},
                                {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                            ],
                        }
                    ]
                }
            ]
        ),
    )

    rows = source.list_for_market(MarketId(1))

    assert len(rows) == 1
    assert str(rows[0].instrument_id.symbol) == "BTCUSDT"
    assert rows[0].price_step == 0.01
    assert rows[0].qty_step == 0.0001
    assert rows[0].min_notional == 10.0


def test_metadata_source_parses_bybit_paginated_payload(tmp_path: Path) -> None:
    """Ensure Bybit pagination is handled and all pages are flattened into metadata rows."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    source = RestInstrumentMetadataSource(
        cfg=cfg,
        http=_FakeHttp(
            [
                {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "symbol": "ETHUSDT",
                                "baseCoin": "ETH",
                                "quoteCoin": "USDT",
                                "priceFilter": {"tickSize": "0.01"},
                                "lotSizeFilter": {
                                    "qtyStep": "0.001",
                                    "minNotionalValue": "5",
                                },
                            }
                        ],
                        "nextPageCursor": "cursor-1",
                    },
                },
                {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "symbol": "ADAUSDT",
                                "baseCoin": "ADA",
                                "quoteCoin": "USDT",
                                "priceFilter": {"tickSize": "0.0001"},
                                "lotSizeFilter": {
                                    "qtyStep": "0.1",
                                    "minNotionalValue": "5",
                                },
                            }
                        ],
                        "nextPageCursor": "",
                    },
                },
            ]
        ),
    )

    rows = source.list_for_market(MarketId(3))

    assert {str(row.instrument_id.symbol) for row in rows} == {"ETHUSDT", "ADAUSDT"}
