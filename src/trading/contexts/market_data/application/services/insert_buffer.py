from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InsertBufferHooks:
    """
    Optional callbacks invoked by async insert buffer.

    Parameters:
    - on_ws_closed_to_insert_start: observe latency from WS receive to insert start.
    - on_ws_closed_to_insert_done: observe latency from WS receive to insert done.
    - on_insert_batch: callback `(rows, duration_seconds)` for successful batch writes.
    - on_insert_error: callback invoked when batch insert fails.

    Assumptions/Invariants:
    - Callbacks are lightweight and non-blocking.
    """

    on_ws_closed_to_insert_start: Callable[[float], None] | None = None
    on_ws_closed_to_insert_done: Callable[[float], None] | None = None
    on_insert_batch: Callable[[int, float], None] | None = None
    on_insert_error: Callable[[], None] | None = None


class AsyncRawInsertBuffer:
    """
    Async buffer for WS candle inserts with flush-by-size and flush-by-timer.

    Parameters:
    - writer: raw writer port used to persist batches.
    - clock: UTC clock used for latency measurements.
    - flush_interval_ms: periodic flush interval in milliseconds.
    - max_buffer_rows: hard flush threshold by batch size.
    - hooks: optional callback hooks for metrics integration.

    Assumptions/Invariants:
    - Incoming rows are closed 1m candles from WS path.
    - `row.meta.ingested_at` marks receive time used by SLO histograms.
    """

    def __init__(
        self,
        *,
        writer: RawKlineWriter,
        clock: Clock,
        flush_interval_ms: int,
        max_buffer_rows: int,
        hooks: InsertBufferHooks | None = None,
    ) -> None:
        """
        Initialize insert buffer state.

        Parameters:
        - writer: raw writer port.
        - clock: UTC clock.
        - flush_interval_ms: periodic flush interval in milliseconds.
        - max_buffer_rows: flush threshold by row count.
        - hooks: optional metrics callbacks.

        Returns:
        - None.

        Assumptions/Invariants:
        - `flush_interval_ms` and `max_buffer_rows` are positive.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("AsyncRawInsertBuffer requires writer")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("AsyncRawInsertBuffer requires clock")
        if flush_interval_ms <= 0:
            raise ValueError("flush_interval_ms must be > 0")
        if max_buffer_rows <= 0:
            raise ValueError("max_buffer_rows must be > 0")

        self._writer = writer
        self._clock = clock
        self._flush_interval_seconds = flush_interval_ms / 1000.0
        self._max_buffer_rows = max_buffer_rows
        self._hooks = hooks if hooks is not None else InsertBufferHooks()

        self._queue: asyncio.Queue[CandleWithMeta] = asyncio.Queue()
        self._buffer: list[CandleWithMeta] = []
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._consumer_task: asyncio.Task[None] | None = None
        self._timer_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """
        Start background consumer and timer tasks.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Called once for one runtime instance.

        Errors/Exceptions:
        - None.

        Side effects:
        - Spawns asyncio background tasks.
        """
        if self._consumer_task is not None:
            return
        self._consumer_task = asyncio.create_task(self._consume_loop(), name="raw-insert-consumer")
        self._timer_task = asyncio.create_task(self._timer_loop(), name="raw-insert-timer")

    def submit(self, row: CandleWithMeta) -> None:
        """
        Enqueue one WS candle row for asynchronous insertion.

        Parameters:
        - row: candle-with-meta DTO to buffer.

        Returns:
        - None.

        Assumptions/Invariants:
        - Buffer instance is started before active submission.

        Errors/Exceptions:
        - Raises `RuntimeError` when submit is attempted after shutdown signal.

        Side effects:
        - Pushes row into in-memory asyncio queue.
        """
        if self._stop_event.is_set():
            raise RuntimeError("insert buffer is stopping and does not accept new rows")
        self._queue.put_nowait(row)

    async def flush(self) -> None:
        """
        Flush current buffered rows immediately.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Method may run concurrently with queue consumer; lock serializes writes.

        Errors/Exceptions:
        - None. Insert errors are captured via hooks and logs.

        Side effects:
        - Writes batch into raw storage when buffer is non-empty.
        """
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        """
        Stop background tasks and flush pending rows.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Safe to call multiple times.

        Errors/Exceptions:
        - None.

        Side effects:
        - Stops background tasks and performs final raw write flush.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()

        if self._consumer_task is not None:
            await self._consumer_task
        if self._timer_task is not None:
            self._timer_task.cancel()
            await asyncio.gather(self._timer_task, return_exceptions=True)

        await self._drain_queue_to_buffer()
        await self.flush()

    async def _consume_loop(self) -> None:
        """
        Move queued rows into buffer and trigger size-based flush.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Loop exits when stop event is set and queue drain is complete.

        Errors/Exceptions:
        - None. Unexpected errors are logged and loop continues.

        Side effects:
        - Updates internal in-memory buffer.
        - Performs raw writes via `_flush_locked`.
        """
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return

            row = await self._try_get_row_with_timeout(timeout_seconds=0.1)
            if row is None:
                continue

            async with self._lock:
                self._buffer.append(row)
                if len(self._buffer) >= self._max_buffer_rows:
                    await self._flush_locked()

    async def _timer_loop(self) -> None:
        """
        Trigger periodic flush while buffer is running.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Timer stops on cancellation or stop event.

        Errors/Exceptions:
        - None.

        Side effects:
        - Invokes `flush()` periodically.
        """
        while not self._stop_event.is_set():
            await asyncio.sleep(self._flush_interval_seconds)
            await self.flush()

    async def _flush_locked(self) -> None:
        """
        Flush buffered rows with lock held by caller.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Caller holds `_lock`.

        Errors/Exceptions:
        - None. Insert failures are logged and batch is kept for retry.

        Side effects:
        - Writes one batch into raw storage on success.
        - Updates latency hooks for every written row.
        """
        if not self._buffer:
            return

        batch = list(self._buffer)
        self._buffer.clear()

        insert_start = self._clock.now().value
        for row in batch:
            _observe(
                self._hooks.on_ws_closed_to_insert_start,
                insert_start,
                row.meta.ingested_at.value,
            )

        try:
            await asyncio.to_thread(self._writer.write_1m, batch)
        except Exception:  # noqa: BLE001
            self._buffer = batch + self._buffer
            _invoke(self._hooks.on_insert_error)
            log.exception("raw insert failed; batch returned to buffer")
            return

        insert_done = self._clock.now().value
        for row in batch:
            _observe(
                self._hooks.on_ws_closed_to_insert_done,
                insert_done,
                row.meta.ingested_at.value,
            )

        duration_seconds = max((insert_done - insert_start).total_seconds(), 0.0)
        _invoke_batch(
            self._hooks.on_insert_batch,
            rows=len(batch),
            duration_seconds=duration_seconds,
        )

    async def _drain_queue_to_buffer(self) -> None:
        """
        Drain queued rows into buffer during shutdown.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Queue contains rows submitted before shutdown.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends all queued rows to in-memory buffer.
        """
        while True:
            try:
                row = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            async with self._lock:
                self._buffer.append(row)

    async def _try_get_row_with_timeout(self, *, timeout_seconds: float) -> CandleWithMeta | None:
        """
        Attempt to read one row from queue with timeout.

        Parameters:
        - timeout_seconds: timeout in seconds.

        Returns:
        - One row when queue item arrives within timeout, otherwise `None`.

        Assumptions/Invariants:
        - Timeout is positive.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except TimeoutError:
            return None


def _observe(
    callback: Callable[[float], None] | None,
    end_value,
    start_value,
) -> None:
    """
    Compute non-negative seconds delta and pass it into observer callback.

    Parameters:
    - callback: metric observer callback or `None`.
    - end_value: datetime boundary (later moment).
    - start_value: datetime boundary (earlier moment).

    Returns:
    - None.

    Assumptions/Invariants:
    - Both boundaries are datetime-like values supporting subtraction.

    Errors/Exceptions:
    - None.

    Side effects:
    - Calls metric observer when callback is provided.
    """
    if callback is None:
        return
    seconds = max((end_value - start_value).total_seconds(), 0.0)
    callback(seconds)


def _invoke(callback: Callable[[], None] | None) -> None:
    """
    Call optional no-argument callback.

    Parameters:
    - callback: optional callable.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback is side-effect-only.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is None:
        return
    callback()


def _invoke_batch(
    callback: Callable[[int, float], None] | None,
    *,
    rows: int,
    duration_seconds: float,
) -> None:
    """
    Call optional batch callback with row count and duration.

    Parameters:
    - callback: optional callback.
    - rows: written rows in batch.
    - duration_seconds: insert duration in seconds.

    Returns:
    - None.

    Assumptions/Invariants:
    - `rows >= 0`.
    - `duration_seconds >= 0`.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is None:
        return
    callback(rows, duration_seconds)
