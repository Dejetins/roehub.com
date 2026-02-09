from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.bybit.ws_client import (
    parse_bybit_closed_1m_message,
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


def test_bybit_confirm_false_is_ignored() -> None:
    """Ensure Bybit parser ignores non-closed (`confirm=false`) updates."""
    payload = {
        "topic": "kline.1.BTCUSDT",
        "type": "snapshot",
        "data": [
            {
                "start": 1765022400000,
                "interval": "1",
                "open": "100",
                "high": "101",
                "low": "99",
                "close": "100.5",
                "volume": "3.1",
                "turnover": "310.0",
                "confirm": False,
            }
        ],
    }
    rows, ignored = parse_bybit_closed_1m_message(
        payload=json.dumps(payload),
        market_id=MarketId(3),
        market_type="spot",
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    assert rows == []
    assert ignored is True


def test_bybit_confirm_true_yields_closed_candle() -> None:
    """Ensure Bybit parser emits closed candle for `confirm=true` updates."""
    payload = {
        "topic": "kline.1.BTCUSDT",
        "type": "snapshot",
        "data": [
            {
                "start": 1765022455555,
                "interval": "1",
                "open": "100",
                "high": "102",
                "low": "99",
                "close": "101",
                "volume": "4.2",
                "turnover": "424.0",
                "confirm": True,
            }
        ],
    }
    rows, ignored = parse_bybit_closed_1m_message(
        payload=json.dumps(payload),
        market_id=MarketId(3),
        market_type="spot",
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    assert ignored is False
    assert len(rows) == 1
    row = rows[0]
    assert row.candle.ts_open.value.second == 0
    assert row.candle.ts_open.value.microsecond == 0
    assert row.candle.instrument_id.market_id.value == 3
    assert str(row.candle.instrument_id.symbol) == "BTCUSDT"

