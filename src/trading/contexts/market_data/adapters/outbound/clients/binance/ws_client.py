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
class BinanceWsHooks:
    """
    Optional metric/logging hooks for Binance WS stream runtime.

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


class BinanceWsClosedCandleStream:
    """
    Long-running Binance WS consumer yielding normalized closed 1m candles.

    Parameters:
    - market_id: market id (`1` spot, `2` futures).
    - market_type: market type string (`spot` or `futures`).
    - ws_url: base combined-stream URL from runtime config.
    - symbols: symbol batch bound to one websocket connection.
    - ping_interval_s: websocket ping interval.
    - pong_timeout_s: websocket pong timeout.
    - reconnect: reconnect policy from runtime config.
    - clock: UTC clock used for ingestion metadata.
    - ingest_id: session UUID attached to `CandleMeta`.
    - on_closed_candle: async callback for every closed normalized candle.
    - on_connected: optional async callback invoked on every successful connect.
    - hooks: optional metrics/logging hooks.

    Assumptions/Invariants:
    - Stream URL uses Binance combined stream semantics.
    - Callback execution is quick; heavy work should be delegated to background queues.
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
        hooks: BinanceWsHooks | None = None,
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
            raise ValueError("BinanceWsClosedCandleStream requires non-empty symbols")
        if not ws_url.strip():
            raise ValueError("BinanceWsClosedCandleStream requires ws_url")

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
        self._hooks = hooks if hooks is not None else BinanceWsHooks()

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
        stream_url = build_binance_combined_stream_url(self._ws_url, self._symbols)

        while not stop_event.is_set():
            connected = False
            try:
                async with websockets.connect(
                    stream_url,
                    ping_interval=self._ping_interval_s,
                    ping_timeout=self._pong_timeout_s,
                ) as socket:
                    _emit_connected(self._hooks.on_connected, 1)
                    connected = True
                    if not first_connect:
                        _emit_simple(self._hooks.on_reconnect)
                    first_connect = False
                    delay = self._reconnect.min_delay_s

                    if self._on_connected is not None:
                        await self._on_connected()

                    while not stop_event.is_set():
                        payload = await _recv_or_stop(socket, stop_event)
                        if payload is None:
                            break

                        _emit_simple(self._hooks.on_message)
                        row, ignored_non_closed = parse_binance_closed_1m_message(
                            payload=payload,
                            market_id=self._market_id,
                            market_type=self._market_type,
                            clock=self._clock,
                            ingest_id=self._ingest_id,
                        )
                        if ignored_non_closed:
                            _emit_simple(self._hooks.on_ignored_non_closed)
                        if row is not None:
                            await self._on_closed_candle(row)
            except Exception:  # noqa: BLE001
                _emit_simple(self._hooks.on_error)
                log.exception("binance ws stream failed for market_id=%s", self._market_id.value)
            finally:
                if connected:
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


def parse_binance_closed_1m_message(
    *,
    payload: str,
    market_id: MarketId,
    market_type: str,
    clock: Clock,
    ingest_id: UUID,
) -> tuple[CandleWithMeta | None, bool]:
    """
    Parse Binance WS frame and return normalized candle only for closed 1m updates.

    Parameters:
    - payload: websocket JSON payload text.
    - market_id: market id used for `InstrumentId`.
    - market_type: market type used for `instrument_key`.
    - clock: UTC clock used for `meta.ingested_at`.
    - ingest_id: runtime ingest session UUID.

    Returns:
    - Tuple `(row, ignored_non_closed)` where:
      - `row` is normalized `CandleWithMeta` for closed 1m update.
      - `ignored_non_closed` is true when update is valid 1m but not yet closed.

    Assumptions/Invariants:
    - Binance combined stream payload contains `data.k` structure for kline events.

    Errors/Exceptions:
    - Malformed payloads are ignored and returned as `(None, False)`.

    Side effects:
    - None.
    """
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError:
        return (None, False)
    if not isinstance(envelope, dict):
        return (None, False)

    data = envelope.get("data")
    if not isinstance(data, dict):
        return (None, False)

    kline = data.get("k")
    if not isinstance(kline, dict):
        return (None, False)

    interval = str(kline.get("i", ""))
    if interval != "1m":
        return (None, False)

    is_closed = bool(kline.get("x", False))
    if not is_closed:
        return (None, True)

    symbol = str(kline.get("s") or data.get("s") or "").strip()
    if not symbol:
        return (None, False)

    open_ms = _to_int(kline.get("t"))
    if open_ms is None:
        return (None, False)
    ts_open_dt = floor_to_minute_utc(datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc))
    ts_open = UtcTimestamp(ts_open_dt)
    ts_close = UtcTimestamp(ts_open_dt + timedelta(minutes=1))

    instrument_id = InstrumentId(market_id=market_id, symbol=Symbol(symbol))

    candle = Candle(
        instrument_id=instrument_id,
        ts_open=ts_open,
        ts_close=ts_close,
        open=float(kline["o"]),
        high=float(kline["h"]),
        low=float(kline["l"]),
        close=float(kline["c"]),
        volume_base=float(kline["v"]),
        volume_quote=_to_float_or_none(kline.get("q")),
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=clock.now(),
        ingest_id=ingest_id,
        instrument_key=f"binance:{market_type}:{symbol}",
        trades_count=_to_int(kline.get("n")),
        taker_buy_volume_base=_to_float_or_none(kline.get("V")),
        taker_buy_volume_quote=_to_float_or_none(kline.get("Q")),
    )
    return (CandleWithMeta(candle=candle, meta=meta), False)


def build_binance_combined_stream_url(base_url: str, symbols: tuple[str, ...]) -> str:
    """
    Build Binance combined-stream URL for 1m kline subscriptions.

    Parameters:
    - base_url: base websocket URL from runtime config (typically ending with `/stream`).
    - symbols: symbol batch for one connection.

    Returns:
    - Combined stream URL with `streams=` query parameter.

    Assumptions/Invariants:
    - Symbols are valid Binance symbol strings.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    streams = "/".join(f"{s.lower()}@kline_1m" for s in symbols)
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}streams={streams}"


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
