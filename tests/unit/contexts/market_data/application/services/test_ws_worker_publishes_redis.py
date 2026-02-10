from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

from apps.worker.market_data_ws.wiring.modules.market_data_ws import MarketDataWsApp
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


class _InsertBufferStub:
    """Insert buffer stub recording submitted rows."""

    def __init__(self) -> None:
        """
        Initialize in-memory submit registry.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.rows: list[CandleWithMeta] = []

    def submit(self, row: CandleWithMeta) -> None:
        """
        Record one submitted candle row.

        Parameters:
        - row: submitted candle event.

        Returns:
        - None.
        """
        self.rows.append(row)


class _RestFillQueueStub:
    """REST fill queue stub recording enqueued tasks."""

    def __init__(self) -> None:
        """
        Initialize empty task list.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.tasks: list[object] = []

    async def enqueue(self, task: object) -> None:
        """
        Record one enqueued task.

        Parameters:
        - task: task object enqueued by worker.

        Returns:
        - None.
        """
        self.tasks.append(task)


class _PublisherStub:
    """Live publisher stub capturing calls and optionally raising exceptions."""

    def __init__(self, *, raise_on_call: bool) -> None:
        """
        Initialize publisher behavior and call registry.

        Parameters:
        - raise_on_call: whether `publish_1m_closed` should raise.

        Returns:
        - None.
        """
        self._raise_on_call = raise_on_call
        self.calls: list[CandleWithMeta] = []

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """
        Record publish call and optionally raise runtime error.

        Parameters:
        - candle: candle forwarded by worker.

        Returns:
        - None.
        """
        self.calls.append(candle)
        if self._raise_on_call:
            raise RuntimeError("simulated publish failure")


class _GapTrackerStub:
    """Gap tracker stub returning a predefined task."""

    def __init__(self, task: object | None) -> None:
        """
        Initialize gap tracker return value.

        Parameters:
        - task: task returned by `observe`.

        Returns:
        - None.
        """
        self._task = task
        self.rows: list[CandleWithMeta] = []

    def observe(self, row: CandleWithMeta) -> object | None:
        """
        Record observed row and return configured task.

        Parameters:
        - row: observed candle event.

        Returns:
        - Preconfigured task or `None`.
        """
        self.rows.append(row)
        return self._task


class _NoopInstrumentReader:
    """Instrument reader stub returning no instruments."""

    def list_enabled_tradable(self):
        """
        Return empty enabled instrument list.

        Parameters:
        - None.

        Returns:
        - Empty tuple.
        """
        return ()


class _NoopReconnectPlanner:
    """Reconnect planner stub returning no tasks."""

    def plan(self, instruments):
        """
        Return empty reconnect task list.

        Parameters:
        - instruments: instrument collection.

        Returns:
        - Empty list.
        """
        _ = instruments
        return []


def _row() -> CandleWithMeta:
    """
    Build deterministic WS closed candle for worker tests.

    Parameters:
    - None.

    Returns:
    - CandleWithMeta row.
    """
    ts_open = datetime(2026, 2, 10, 12, 34, tzinfo=timezone.utc)
    instrument_id = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume_base=2.0,
        volume_quote=200.0,
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=UtcTimestamp(datetime(2026, 2, 10, 12, 35, tzinfo=timezone.utc)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000123"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=10,
        taker_buy_volume_base=0.5,
        taker_buy_volume_quote=50.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def _app(
    *,
    insert_buffer: _InsertBufferStub,
    rest_queue: _RestFillQueueStub,
    publisher: _PublisherStub,
    gap_tracker: _GapTrackerStub,
) -> MarketDataWsApp:
    """
    Build minimally wired worker app for `_on_closed_candle` unit tests.

    Parameters:
    - insert_buffer: insert buffer stub.
    - rest_queue: rest queue stub.
    - publisher: publisher stub.
    - gap_tracker: gap tracker stub.

    Returns:
    - `MarketDataWsApp` instance.
    """
    return MarketDataWsApp(
        config=object(),  # type: ignore[arg-type]
        instrument_reader=_NoopInstrumentReader(),  # type: ignore[arg-type]
        index_reader=object(),  # type: ignore[arg-type]
        insert_buffer=insert_buffer,  # type: ignore[arg-type]
        rest_fill_queue=rest_queue,  # type: ignore[arg-type]
        live_candle_publisher=publisher,  # type: ignore[arg-type]
        gap_tracker=gap_tracker,  # type: ignore[arg-type]
        reconnect_planner=_NoopReconnectPlanner(),  # type: ignore[arg-type]
        ingest_id=UUID("00000000-0000-0000-0000-000000000999"),
        metrics=object(),  # type: ignore[arg-type]
        metrics_port=9201,
    )


def test_ws_worker_publishes_closed_candle_to_live_feed() -> None:
    """
    Ensure worker calls live candle publisher for each WS closed candle.
    """
    insert_buffer = _InsertBufferStub()
    rest_queue = _RestFillQueueStub()
    publisher = _PublisherStub(raise_on_call=False)
    gap_tracker = _GapTrackerStub(task=None)
    app = _app(
        insert_buffer=insert_buffer,
        rest_queue=rest_queue,
        publisher=publisher,
        gap_tracker=gap_tracker,
    )

    asyncio.run(app._on_closed_candle(_row()))

    assert len(insert_buffer.rows) == 1
    assert len(publisher.calls) == 1
    assert len(rest_queue.tasks) == 0


def test_ws_worker_keeps_processing_when_live_publish_fails() -> None:
    """
    Ensure publish exceptions do not stop gap handling and rest-fill enqueue flow.
    """
    insert_buffer = _InsertBufferStub()
    rest_queue = _RestFillQueueStub()
    publisher = _PublisherStub(raise_on_call=True)
    gap_task = object()
    gap_tracker = _GapTrackerStub(task=gap_task)
    app = _app(
        insert_buffer=insert_buffer,
        rest_queue=rest_queue,
        publisher=publisher,
        gap_tracker=gap_tracker,
    )

    asyncio.run(app._on_closed_candle(_row()))

    assert len(insert_buffer.rows) == 1
    assert len(publisher.calls) == 1
    assert len(rest_queue.tasks) == 1
    assert rest_queue.tasks[0] is gap_task
