from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from trading.contexts.market_data.adapters.outbound.clients import (
    RestInstrumentHistoryStartSource,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    HttpResponse,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


@dataclass(frozen=True, slots=True)
class _FixedClock:
    now_value: UtcTimestamp

    def now(self) -> UtcTimestamp:
        """Return deterministic current UTC time for binary-search tests."""
        return self.now_value


class _FakeHttp:
    def __init__(self, handler: Callable[[str, Mapping[str, Any]], Any]) -> None:
        """Store request handler and capture requests for later assertions."""
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(  # noqa: ANN001
        self,
        *,
        url,
        params,
        timeout_s,
        retries,
        backoff_base_s,
        backoff_max_s,
        backoff_jitter_s,
    ):
        _ = timeout_s
        _ = retries
        _ = backoff_base_s
        _ = backoff_max_s
        _ = backoff_jitter_s
        call_params = dict(params)
        self.calls.append((url, call_params))
        return HttpResponse(status_code=200, headers={}, body=self._handler(url, call_params))


def _config(tmp_path: Path) -> Path:
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
    - market_id: 2
      exchange: binance
      market_type: futures
      market_code: binance:futures
      rest:
        base_url: https://fapi.binance.com
        earliest_available_ts_utc: "2019-09-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.1, max_s: 0.1, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: wss://fstream.binance.com/stream
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
        max_symbols_per_connection: 50
    - market_id: 4
      exchange: bybit
      market_type: futures
      market_code: bybit:futures
      rest:
        base_url: https://api.bybit.com
        earliest_available_ts_utc: "2018-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.1, max_s: 0.1, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: wss://stream.bybit.com/v5/public/linear
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 50
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 1000
    rest_concurrency_instruments: 2
    tail_lookback_minutes: 180
    rest_inter_instrument_delay_s: 0.0
  scheduler:
    jobs:
      refresh_catalog: { interval_seconds: 3600 }
      enrich: { interval_seconds: 3600 }
      rest_insurance_catchup: { interval_seconds: 3600 }
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
"""
    path = tmp_path / "market_data.yaml"
    path.write_text(yaml_text.strip(), encoding="utf-8")
    return path


def test_history_start_source_confirms_binance_futures_first_kline(tmp_path: Path) -> None:
    """Ensure Binance futures does not treat pre-history onboardDate as first candle."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    http = _FakeHttp(
        lambda url, params: (
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "onboardDate": 1577836800000,
                    }
                ]
            }
            if url.endswith("/fapi/v1/exchangeInfo") and params == {}
            else [[1577923200000]]
            if url.endswith("/fapi/v1/klines")
            and params
            == {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": 1577836800000,
                "limit": 1,
            }
            else (_raise_unexpected(url, params))
        )
    )
    source = RestInstrumentHistoryStartSource(
        cfg=cfg,
        http=http,
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    first = source.get_history_start(InstrumentId(MarketId(2), Symbol("BTCUSDT")))
    second = source.get_history_start(InstrumentId(MarketId(2), Symbol("BTCUSDT")))

    assert str(first) == "2020-01-02T00:00:00.000Z"
    assert str(second) == "2020-01-02T00:00:00.000Z"
    assert len(http.calls) == 2


def test_history_start_source_binary_searches_binance_spot_first_available_minute(
    tmp_path: Path,
) -> None:
    """Ensure Binance spot infers first available minute via non-empty kline probe."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    first_minute = datetime(2017, 1, 1, 0, 4, tzinfo=timezone.utc)
    threshold_end_ms = int(first_minute.timestamp() * 1000) + 60_000 - 1

    http = _FakeHttp(
        lambda url, params: (
            [["dummy"]]
            if url.endswith("/api/v3/klines") and int(params["endTime"]) >= threshold_end_ms
            else []
        )
        if url.endswith("/api/v3/klines")
        else (_raise_unexpected(url, params))
    )
    source = RestInstrumentHistoryStartSource(
        cfg=cfg,
        http=http,
        clock=_FixedClock(UtcTimestamp(datetime(2017, 1, 1, 0, 10, tzinfo=timezone.utc))),
    )

    resolved = source.get_history_start(InstrumentId(MarketId(1), Symbol("BTCUSDT")))

    assert str(resolved) == "2017-01-01T00:04:00.000Z"
    assert len(http.calls) >= 2


def test_history_start_source_reads_bybit_futures_launch_time(tmp_path: Path) -> None:
    """Ensure Bybit futures uses `launchTime` from instruments-info."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    http = _FakeHttp(
        lambda url, params: {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "launchTime": "1585526400000",
                    }
                ]
            },
        }
        if url.endswith("/v5/market/instruments-info")
        and params == {"category": "linear", "symbol": "BTCUSDT"}
        else (_raise_unexpected(url, params))
    )
    source = RestInstrumentHistoryStartSource(
        cfg=cfg,
        http=http,
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    resolved = source.get_history_start(InstrumentId(MarketId(4), Symbol("BTCUSDT")))

    assert str(resolved) == "2020-03-30T00:00:00.000Z"


def test_history_start_source_binary_searches_bybit_spot_first_available_minute(
    tmp_path: Path,
) -> None:
    """Ensure Bybit spot infers first available minute from non-empty kline windows."""
    cfg = load_market_data_runtime_config(_config(tmp_path))
    first_minute = datetime(2018, 1, 1, 0, 3, tzinfo=timezone.utc)
    threshold_end_ms = int(first_minute.timestamp() * 1000) + 60_000 - 1

    http = _FakeHttp(
        lambda url, params: {
            "retCode": 0,
            "result": {"list": [["dummy"]]},
        }
        if url.endswith("/v5/market/kline") and int(params["end"]) >= threshold_end_ms
        else {"retCode": 0, "result": {"list": []}}
        if url.endswith("/v5/market/kline")
        else (_raise_unexpected(url, params))
    )
    source = RestInstrumentHistoryStartSource(
        cfg=cfg,
        http=http,
        clock=_FixedClock(UtcTimestamp(datetime(2018, 1, 1, 0, 10, tzinfo=timezone.utc))),
    )

    resolved = source.get_history_start(InstrumentId(MarketId(3), Symbol("BTCUSDT")))

    assert str(resolved) == "2018-01-01T00:03:00.000Z"
    assert len(http.calls) >= 2


def _raise_unexpected(url: str, params: Mapping[str, Any]) -> None:
    raise AssertionError(f"Unexpected request url={url!r} params={dict(params)!r}")
