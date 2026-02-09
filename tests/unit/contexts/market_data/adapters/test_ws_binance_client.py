from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.binance.ws_client import (
    parse_binance_closed_1m_message,
)
from trading.shared_kernel.primitives import MarketId, UtcTimestamp


class _FixedClock:
    """Clock fake returning one constant UTC timestamp."""

    def __init__(self, now_value: UtcTimestamp) -> None:
        """Store fixed timestamp for later `now()` calls."""
        self._now_value = now_value

    def now(self) -> UtcTimestamp:
        """Return preconfigured fixed timestamp."""
        return self._now_value


def test_binance_non_closed_update_is_ignored() -> None:
    """Ensure Binance parser ignores non-closed 1m updates and marks metric flag."""
    payload = {
        "stream": "btcusdt@kline_1m",
        "data": {
            "s": "BTCUSDT",
            "k": {
                "s": "BTCUSDT",
                "i": "1m",
                "x": False,
            },
        },
    }
    row, ignored = parse_binance_closed_1m_message(
        payload=json.dumps(payload),
        market_id=MarketId(1),
        market_type="spot",
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    assert row is None
    assert ignored is True


def test_binance_closed_update_yields_normalized_1m_candle() -> None:
    """Ensure Binance parser emits closed candle and normalizes `ts_open` to minute bucket."""
    payload = {
        "stream": "btcusdt@kline_1m",
        "data": {
            "s": "BTCUSDT",
            "k": {
                "s": "BTCUSDT",
                "i": "1m",
                "x": True,
                "t": 1765022416789,
                "o": "101.0",
                "h": "103.0",
                "l": "100.0",
                "c": "102.5",
                "v": "12.3",
                "q": "1260.0",
                "n": 10,
                "V": "6.1",
                "Q": "620.0",
            },
        },
    }
    row, ignored = parse_binance_closed_1m_message(
        payload=json.dumps(payload),
        market_id=MarketId(1),
        market_type="spot",
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    assert ignored is False
    assert row is not None
    assert row.candle.ts_open.value.second == 0
    assert row.candle.ts_open.value.microsecond == 0
    assert row.candle.instrument_id.market_id.value == 1
    assert str(row.candle.instrument_id.symbol) == "BTCUSDT"

