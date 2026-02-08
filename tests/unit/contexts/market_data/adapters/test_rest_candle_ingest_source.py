from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    HttpResponse,
)
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class FakeClock:
    def __init__(self, dt):
        self._dt = dt

    def now(self):
        return UtcTimestamp(self._dt)


class FakeHttp:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def get_json(self, *, url, params, timeout_s, retries, backoff_base_s, backoff_max_s, backoff_jitter_s):  # noqa: E501
        self.calls.append((url, dict(params)))
        return HttpResponse(status_code=200, headers={}, body=self.body)


def test_binance_maps_to_candle_with_meta(tmp_path):
    # используем configs/dev/market_data.yaml из репо (здесь — через temp имитируем минимально)
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
  ingestion:
    raw_write: { flush_interval_ms: 250, max_buffer_rows: 2000 }
  backfill:
    max_days_per_insert: 7
    chunk_align: "utc_day"
"""
    p = tmp_path / "market_data.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_market_data_runtime_config(p)

    # одна свеча Binance kline item
    open_ms = int(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    body = [
        [open_ms, "1", "2", "0.5", "1.5", "10", open_ms + 59999, "15", 7, "3", "4", "0"]
    ]

    fake_http = FakeHttp(body)
    fake_clock = FakeClock(datetime(2026, 2, 1, 1, 0, tzinfo=timezone.utc))
    ingest_id = UUID("00000000-0000-0000-0000-000000000001")

    src = RestCandleIngestSource(cfg=cfg, clock=fake_clock, http=fake_http, ingest_id=ingest_id)

    inst = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    tr = TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc)),
    )
    rows = list(src.stream_1m(inst, tr))

    assert len(rows) == 1
    assert rows[0].meta.source == "rest"
    assert rows[0].meta.ingest_id == ingest_id
    assert rows[0].candle.volume_quote == 15.0
