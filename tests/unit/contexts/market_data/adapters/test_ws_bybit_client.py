from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.bybit.ws_client import (
    _chunk_symbols,
    _parse_subscribe_ack,
    _subscribe_topics,
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


class _FakeBybitSocket:
    """Minimal websocket stub for Bybit subscribe tests."""

    def __init__(self, *, recv_payloads: list[str]) -> None:
        """
        Store canned receive payloads and initialize send registry.

        Parameters:
        - recv_payloads: payloads returned in order by `recv()`.

        Returns:
        - None.
        """
        self._recv_payloads = list(recv_payloads)
        self.sent_payloads: list[str] = []

    async def send(self, payload: str) -> None:
        """
        Record outbound payload.

        Parameters:
        - payload: websocket frame sent by caller.

        Returns:
        - None.
        """
        self.sent_payloads.append(payload)

    async def recv(self) -> str:
        """
        Return next canned payload.

        Parameters:
        - None.

        Returns:
        - Next payload string.

        Errors/Exceptions:
        - Raises `RuntimeError` when no canned payload remains.
        """
        if not self._recv_payloads:
            raise RuntimeError("recv called without canned payloads")
        return self._recv_payloads.pop(0)


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


def test_chunk_symbols_splits_input_into_fixed_size_batches() -> None:
    """
    Ensure symbol chunk helper preserves order and enforces max chunk size.
    """
    symbols = tuple(f"S{i}" for i in range(14))
    chunks = _chunk_symbols(symbols=symbols, chunk_size=10)
    assert chunks == [
        tuple(f"S{i}" for i in range(10)),
        tuple(f"S{i}" for i in range(10, 14)),
    ]


def test_parse_subscribe_ack_extracts_failure_reason() -> None:
    """
    Ensure subscribe ack parser returns status and ret_msg for subscribe frames.
    """
    payload = json.dumps({"op": "subscribe", "success": False, "ret_msg": "args size >10"})
    ack = _parse_subscribe_ack(payload)
    assert ack == (False, "args size >10")


def test_subscribe_topics_sends_bybit_subscriptions_in_batches_of_ten() -> None:
    """
    Ensure Bybit subscribe helper sends multiple subscribe frames and validates ACKs.
    """
    socket = _FakeBybitSocket(
        recv_payloads=[
            json.dumps({"op": "subscribe", "success": True, "ret_msg": "OK"}),
            json.dumps({"op": "subscribe", "success": True, "ret_msg": "OK"}),
        ]
    )
    symbols = tuple(
        [
            "BTCUSDT",
            "ADAUSDT",
            "LTCUSDT",
            "TONUSDT",
            "DOTUSDT",
            "ETCUSDT",
            "SOLUSDT",
            "DOGEUSDT",
            "HYPEUSDT",
            "SUIUSDT",
            "SHIBUSDT",
            "UNIUSDT",
            "MNTUSDT",
            "AAVEUSDT",
        ]
    )

    asyncio.run(_subscribe_topics(socket, symbols))

    assert len(socket.sent_payloads) == 2
    first_payload = json.loads(socket.sent_payloads[0])
    second_payload = json.loads(socket.sent_payloads[1])
    assert first_payload["op"] == "subscribe"
    assert second_payload["op"] == "subscribe"
    assert len(first_payload["args"]) == 10
    assert len(second_payload["args"]) == 4


def test_subscribe_topics_raises_on_failed_ack() -> None:
    """
    Ensure failed Bybit subscribe ACK becomes explicit runtime error.
    """
    socket = _FakeBybitSocket(
        recv_payloads=[
            json.dumps({"op": "subscribe", "success": False, "ret_msg": "args size >10"}),
        ]
    )

    try:
        asyncio.run(_subscribe_topics(socket, ("BTCUSDT",)))
    except RuntimeError as exc:
        assert "args size >10" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on failed subscribe ACK")
