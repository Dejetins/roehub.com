from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Sequence

from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    RecoveryEntry,
    StreamRecoveryStore,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    ErrorClassifier,
    retry_delay,
    transient_ingestion_error,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

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


class UnflushedBufferError(RuntimeError):
    """Shutdown did not confirm every candle; durable receipts must be replayed."""


class AsyncRawInsertBuffer:
    """Hard-bounded accepted rows, classified retry, durable receipt before acceptance.

    WS wiring supplies an async writer whose cancellation terminates its process.
    `submit_wait` applies backpressure; the received row's receipt survives cancellation.
    """

    def __init__(
        self,
        *,
        writer: RawKlineWriter,
        clock: Clock,
        flush_interval_ms: int,
        max_buffer_rows: int,
        hooks: InsertBufferHooks | None = None,
        async_writer: Callable[[Sequence[CandleWithMeta]], Awaitable[None]] | None = None,
        store: StreamRecoveryStore | None = None,
        classify_error: ErrorClassifier = transient_ingestion_error,
        retry_base_s: float = 15,
        shutdown_timeout_s: float = 3,
        on_deferred: Callable[[RecoveryEntry], Awaitable[None]] | None = None,
    ) -> None:
        if min(flush_interval_ms, max_buffer_rows, retry_base_s, shutdown_timeout_s) <= 0:
            raise ValueError("buffer bounds must be positive")
        self._writer = writer
        self._async_writer = async_writer
        self._clock = clock
        self._flush_interval_seconds = flush_interval_ms / 1000
        self._max_buffer_rows = max_buffer_rows
        self._hooks = hooks or InsertBufferHooks()
        self._store = store
        self._classify_error = classify_error
        self._retry_base_s = retry_base_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._on_deferred = on_deferred
        self._buffer: list[CandleWithMeta] = []
        self._receipts: list[RecoveryEntry | None] = []
        self._wake = asyncio.Event()
        self._space = asyncio.Event()
        self._space.set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._next_retry = 0.0
        self._flush_due = 0.0
        self._attempt = 0
        self.failure: BaseException | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="raw-insert-buffer")

    def _remember(self, row: CandleWithMeta) -> RecoveryEntry | None:
        if self._store is None:
            return None
        return self._store.remember(
            RestFillTask(
                row.candle.instrument_id,
                TimeRange(
                    row.candle.ts_open,
                    UtcTimestamp(row.candle.ts_open.value + timedelta(minutes=1)),
                ),
                "ws_recovery",
            )
        )

    def _accept(self, row: CandleWithMeta, receipt: RecoveryEntry | None) -> None:
        if self._stopping or self.failure:
            raise RuntimeError("insert buffer is stopping or failed")
        if len(self._buffer) >= self._max_buffer_rows:
            raise asyncio.QueueFull
        if not self._buffer:
            self._flush_due = asyncio.get_running_loop().time() + self._flush_interval_seconds
        self._buffer.append(row)
        self._receipts.append(receipt)
        self._wake.set()
        if len(self._buffer) >= self._max_buffer_rows:
            self._space.clear()
            self._wake.set()

    def submit(self, row: CandleWithMeta) -> None:
        if self._stopping or self.failure:
            raise RuntimeError("insert buffer is stopping or failed")
        if len(self._buffer) >= self._max_buffer_rows:
            raise asyncio.QueueFull
        self._accept(row, self._remember(row))

    async def submit_wait(self, row: CandleWithMeta) -> None:
        if self._stopping or self.failure:
            raise RuntimeError("insert buffer is stopping or failed")
        receipt = self._remember(row)
        try:
            while len(self._buffer) >= self._max_buffer_rows:
                await self._space.wait()
                if self._stopping or self.failure:
                    raise RuntimeError("insert buffer is stopping or failed")
            self._accept(row, receipt)
        except BaseException:
            # Subscription replacement may cancel a row waiting for capacity.
            # It must be repaired in this live worker, not only after a restart.
            if receipt is not None and self._on_deferred is not None:
                await self._on_deferred(receipt)
            raise

    async def _run(self) -> None:
        while not self._stopping and self.failure is None:
            if not self._buffer:
                await self._wake.wait()
                self._wake.clear()
                continue
            # Coalesce from the first received row, not an unrelated periodic tick.
            # Rows arriving during a write drain immediately after it completes.
            due = 0 if len(self._buffer) >= self._max_buffer_rows else self._flush_due
            delay = max(due, self._next_retry) - asyncio.get_running_loop().time()
            if delay <= 0:
                await self.flush()
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), delay)
            except TimeoutError:
                pass
            self._wake.clear()

    async def flush(self, *, force: bool = False) -> None:
        async with self._lock:
            if not self._buffer or self.failure:
                return
            loop = asyncio.get_running_loop()
            if not force and loop.time() < self._next_retry:
                return
            batch = self._buffer[: min(self._max_buffer_rows, 1000)]
            started = self._clock.now().value
            for row in batch:
                _observe(
                    self._hooks.on_ws_closed_to_insert_start, started, row.meta.ingested_at.value
                )
            try:
                if self._async_writer:
                    await self._async_writer(batch)
                else:
                    await asyncio.to_thread(self._writer.write_1m, batch)
                if self._store:
                    self._store.complete_many(
                        [e for e in self._receipts[: len(batch)] if e is not None]
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _invoke(self._hooks.on_insert_error)
                code = self._classify_error(exc)
                if code is None:
                    self.failure = exc
                    self._space.set()
                    if self._store:
                        self._store.fail_many(
                            [e for e in self._receipts[: len(batch)] if e is not None],
                            code="non_retryable_ingestion_error",
                        )
                    log.error(
                        "raw insert terminal error_type=%s; receipts retained", type(exc).__name__
                    )
                else:
                    self._attempt += 1
                    delay = retry_delay(exc, self._attempt, base=self._retry_base_s)
                    self._next_retry = loop.time() + delay
                    log.warning(
                        "raw insert waiting code=%s retry_in_s=%s rows=%s", code, delay, len(batch)
                    )
                return
            del self._receipts[: len(batch)]
            del self._buffer[: len(batch)]
            self._space.set()
            self._attempt = 0
            self._next_retry = 0
            self._flush_due = 0
            finished = self._clock.now().value
            for row in batch:
                _observe(
                    self._hooks.on_ws_closed_to_insert_done, finished, row.meta.ingested_at.value
                )
            _invoke_batch(
                self._hooks.on_insert_batch,
                rows=len(batch),
                duration_seconds=max(0, (finished - started).total_seconds()),
            )

    async def close(self) -> None:
        self._stopping = True
        self._space.set()
        self._wake.set()

        async def drain() -> None:
            if self._task:
                await self._task
            while self._buffer and self.failure is None:
                before = len(self._buffer)
                await self.flush(force=True)
                if len(self._buffer) == before:
                    break

        try:
            await asyncio.wait_for(drain(), self._shutdown_timeout_s)
        except TimeoutError:
            pass
        if self._buffer or self.failure:
            raise UnflushedBufferError(
                f"unconfirmed_rows={len(self._buffer)}; durable recovery required"
            ) from self.failure


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
