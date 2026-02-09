from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from uuid import UUID

import websockets

from trading.contexts.market_data.adapters.outbound.config.runtime_config import WsReconnectConfig
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BybitWsHooks:
    """
    Optional metric/logging hooks for Bybit WS stream runtime.

    Parameters:
    - on_connected: callback invoked with `1` on connect and `0` on disconnect.
    - on_reconnect: callback invoked on reconnect attempts.
    - on_message: callback invoked for every received WS frame.
    - on_error: callback invoked on connection/processing errors.
    - on_ignored_non_closed: callback invoked for non-closed 1m updates.
    """

    on_connected: Callable[[int], None] | None = None
    on_reconnect: Callable[[], None] | None = None
    on_message: Callable[[], None] | None = None
    on_error: Callable[[], None] | None = None
    on_ignored_non_closed: Callable[[], None] | None = None


class BybitWsClosedCandleStream:
    """
    Long-running Bybit V5 WS consumer yielding normalized closed 1m candles.

    Parameters:
    - market_id: market id (`3` spot, `4` futures).
    - market_type: market type string (`spot` or `futures`).
    - ws_url: websocket endpoint from runtime config.
    - symbols: symbol batch bound to one websocket connection.
    - ping_interval_s: websocket ping interval.
    - pong_timeout_s: websocket pong timeout.
    - reconnect: reconnect policy.
    - clock: UTC clock used for ingestion metadata.
    - ingest_id: session UUID attached to `CandleMeta`.
    - on_closed_candle: async callback for every closed normalized candle.
    - on_connected: optional async callback invoked on every successful connect.
    - hooks: optional metrics/logging hooks.

    Assumptions/Invariants:
    - One connection subscribes to `kline.1.<symbol>` topics for given symbols.
    """

    def __init__(
        self,
        *,
        market_id: MarketId,
        market_type: str,
        ws_url: str,
        symbols: tuple[str, ...],
        ping_interval_s: float,
        pong_timeout_s: float,
        reconnect: WsReconnectConfig,
        clock: Clock,
        ingest_id: UUID,
        on_closed_candle: Callable[[CandleWithMeta], Awaitable[None]],
        on_connected: Callable[[], Awaitable[None]] | None = None,
        hooks: BybitWsHooks | None = None,
    ) -> None:
        """
        Validate constructor arguments and initialize runtime state.

        Parameters:
        - See class-level documentation.

        Returns:
        - None.

        Assumptions/Invariants:
        - `symbols` tuple is non-empty.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if not symbols:
            raise ValueError("BybitWsClosedCandleStream requires non-empty symbols")
        if not ws_url.strip():
            raise ValueError("BybitWsClosedCandleStream requires ws_url")

        self._market_id = market_id
        self._market_type = market_type
        self._ws_url = ws_url
        self._symbols = symbols
        self._ping_interval_s = ping_interval_s
        self._pong_timeout_s = pong_timeout_s
        self._reconnect = reconnect
        self._clock = clock
        self._ingest_id = ingest_id
        self._on_closed_candle = on_closed_candle
        self._on_connected = on_connected
        self._hooks = hooks if hooks is not None else BybitWsHooks()

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Run websocket loop with reconnect backoff until stop event is set.

        Parameters:
        - stop_event: cooperative shutdown signal.

        Returns:
        - None.

        Assumptions/Invariants:
        - Method is executed inside active asyncio event loop.

        Errors/Exceptions:
        - Does not propagate connection/runtime exceptions; errors are logged and retried.

        Side effects:
        - Maintains outbound websocket connection.
        - Emits callbacks for metrics and candle events.
        """
        delay = self._reconnect.min_delay_s
        first_connect = True

        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=self._ping_interval_s,
                    ping_timeout=self._pong_timeout_s,
                ) as socket:
                    _emit_connected(self._hooks.on_connected, 1)
                    if not first_connect:
                        _emit_simple(self._hooks.on_reconnect)
                    first_connect = False
                    delay = self._reconnect.min_delay_s

                    await _subscribe_topics(socket, self._symbols)

                    if self._on_connected is not None:
                        await self._on_connected()

                    while not stop_event.is_set():
                        payload = await _recv_or_stop(socket, stop_event)
                        if payload is None:
                            break

                        _emit_simple(self._hooks.on_message)
                        rows, ignored_non_closed = parse_bybit_closed_1m_message(
                            payload=payload,
                            market_id=self._market_id,
                            market_type=self._market_type,
                            clock=self._clock,
                            ingest_id=self._ingest_id,
                        )
                        if ignored_non_closed:
                            _emit_simple(self._hooks.on_ignored_non_closed)
                        for row in rows:
                            await self._on_closed_candle(row)
            except Exception:  # noqa: BLE001
                _emit_simple(self._hooks.on_error)
                log.exception("bybit ws stream failed for market_id=%s", self._market_id.value)
            finally:
                _emit_connected(self._hooks.on_connected, 0)

            if stop_event.is_set():
                break

            await asyncio.sleep(
                _next_delay(
                    delay,
                    self._reconnect.max_delay_s,
                    self._reconnect.jitter_s,
                )
            )
            delay = min(delay * self._reconnect.factor, self._reconnect.max_delay_s)


def parse_bybit_closed_1m_message(
    *,
    payload: str,
    market_id: MarketId,
    market_type: str,
    clock: Clock,
    ingest_id: UUID,
) -> tuple[list[CandleWithMeta], bool]:
    """
    Parse Bybit WS frame and return normalized candles for closed 1m updates.

    Parameters:
    - payload: websocket JSON payload text.
    - market_id: market id used for `InstrumentId`.
    - market_type: market type used for `instrument_key`.
    - clock: UTC clock used for `meta.ingested_at`.
    - ingest_id: runtime ingest session UUID.

    Returns:
    - Tuple `(rows, ignored_non_closed)` where:
      - `rows` contains normalized `CandleWithMeta` records for closed 1m updates.
      - `ignored_non_closed` is true when at least one non-closed 1m update was filtered out.

    Assumptions/Invariants:
    - Bybit payload uses `topic=kline.1.<symbol>` and `data=[...]`.

    Errors/Exceptions:
    - Malformed payloads are ignored and returned as `([], False)`.

    Side effects:
    - None.
    """
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError:
        return ([], False)
    if not isinstance(envelope, dict):
        return ([], False)

    topic = str(envelope.get("topic", ""))
    if not topic.startswith("kline.1."):
        return ([], False)

    symbol_from_topic = topic.split(".")[-1]
    data = envelope.get("data")
    if not isinstance(data, list):
        return ([], False)

    rows: list[CandleWithMeta] = []
    ignored_non_closed = False
    for item in data:
        if not isinstance(item, dict):
            continue

        interval = str(item.get("interval", ""))
        if interval not in ("1", "1m"):
            continue

        confirm = bool(item.get("confirm", False))
        if not confirm:
            ignored_non_closed = True
            continue

        start_ms = _to_int(item.get("start"))
        if start_ms is None:
            continue

        symbol = str(item.get("symbol") or symbol_from_topic).strip()
        if not symbol:
            continue

        ts_open_dt = floor_to_minute_utc(datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc))
        ts_open = UtcTimestamp(ts_open_dt)
        ts_close = UtcTimestamp(ts_open_dt + timedelta(minutes=1))
        instrument_id = InstrumentId(market_id=market_id, symbol=Symbol(symbol))

        candle = Candle(
            instrument_id=instrument_id,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume_base=float(item["volume"]),
            volume_quote=_to_float_or_none(item.get("turnover")),
        )
        meta = CandleMeta(
            source="ws",
            ingested_at=clock.now(),
            ingest_id=ingest_id,
            instrument_key=f"bybit:{market_type}:{symbol}",
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )
        rows.append(CandleWithMeta(candle=candle, meta=meta))

    return (rows, ignored_non_closed)


async def _subscribe_topics(socket, symbols: tuple[str, ...]) -> None:
    """
    Send Bybit V5 subscription payload for `kline.1.<symbol>` topics.

    Parameters:
    - socket: active websocket connection object.
    - symbols: symbols bound to this connection.

    Returns:
    - None.

    Assumptions/Invariants:
    - Socket is connected and writable.

    Errors/Exceptions:
    - Propagates websocket send errors.

    Side effects:
    - Sends one subscribe frame through websocket.
    """
    args = [f"kline.1.{symbol}" for symbol in symbols]
    payload = {"op": "subscribe", "args": args}
    await socket.send(json.dumps(payload))


def _to_int(value) -> int | None:
    """
    Convert incoming scalar to int when possible.

    Parameters:
    - value: scalar value from WS payload.

    Returns:
    - Integer value or `None` when conversion fails.

    Assumptions/Invariants:
    - Boolean values are rejected.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value) -> float | None:
    """
    Convert incoming scalar to float when possible.

    Parameters:
    - value: scalar payload value.

    Returns:
    - Float value or `None` when input is absent/invalid.

    Assumptions/Invariants:
    - Boolean values are rejected.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_connected(callback: Callable[[int], None] | None, value: int) -> None:
    """
    Trigger connection-state callback if provided.

    Parameters:
    - callback: callback accepting connection state value.
    - value: `1` for connected, `0` for disconnected.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback side effects are controlled by caller.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is None:
        return
    callback(value)


def _emit_simple(callback: Callable[[], None] | None) -> None:
    """
    Trigger optional no-argument callback.

    Parameters:
    - callback: callback or `None`.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback side effects are controlled by caller.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is None:
        return
    callback()


def _next_delay(current: float, max_delay: float, jitter_s: float) -> float:
    """
    Compute reconnect delay with jitter for next retry iteration.

    Parameters:
    - current: current backoff delay.
    - max_delay: maximum allowed delay.
    - jitter_s: random additive jitter bound.

    Returns:
    - Delay in seconds.

    Assumptions/Invariants:
    - Delay values are positive.

    Errors/Exceptions:
    - None.

    Side effects:
    - Uses pseudo-random generator for jitter.
    """
    capped = min(current, max_delay)
    return capped + random.random() * jitter_s


async def _recv_or_stop(socket, stop_event: asyncio.Event) -> str | None:
    """
    Receive one websocket frame or return early on shutdown signal.

    Parameters:
    - socket: active websocket connection object.
    - stop_event: cooperative shutdown signal.

    Returns:
    - Payload text when received, otherwise `None` when stop is requested.

    Assumptions/Invariants:
    - `socket.recv()` returns text payload for kline streams.

    Errors/Exceptions:
    - Propagates websocket receive errors to outer reconnect loop.

    Side effects:
    - Allocates temporary asyncio tasks per receive attempt.
    """
    recv_task = asyncio.create_task(socket.recv())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {recv_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if stop_task in done and stop_event.is_set():
        recv_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)
        return None
    return str(recv_task.result())
