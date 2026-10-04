"""Failure/restart contracts, not implementation-shaped success assertions."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from test_insert_buffer import _MonotonicClock, _RecordingWriter, _row

from trading.contexts.market_data.application.dto import RestFillResult, RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    RecoveryEntry,
    recovery_key,
)
from trading.contexts.market_data.application.services.insert_buffer import (
    AsyncRawInsertBuffer,
    UnflushedBufferError,
)
from trading.contexts.market_data.application.services.rest_fill_queue import AsyncRestFillQueue
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

NOW = datetime(2026, 10, 4, tzinfo=UTC)


class MemoryReceipts:
    def __init__(self):
        self.rows = {}
        self.failed = set()

    def remember(self, task):
        key = recovery_key(task)
        if key not in self.rows:
            self.rows[key] = RecoveryEntry(task, uuid4())
        return self.rows[key]

    def pending(self):
        return [entry for key, entry in self.rows.items() if key not in self.failed]

    def advance(self, entry, *, start_at):
        next_task = replace(
            entry.task, time_range=TimeRange(UtcTimestamp(start_at), entry.task.time_range.end)
        )
        self.rows[entry.task_key] = replace(
            entry, task=next_task, key=entry.task_key, attempt=0, next_retry_at=None
        )

    def complete(self, entry):
        key = entry.task_key
        if self.rows.get(key) == entry:
            self.rows.pop(key)
        elif key in self.rows and self.rows[key].token == entry.token:
            self.rows.pop(key)

    def complete_many(self, entries):
        for entry in entries:
            self.complete(entry)

    def retry(self, entry, *, next_retry_at, code):
        self.rows[entry.task_key] = replace(entry, next_retry_at=next_retry_at, error_code=code)

    def fail(self, entry, *, code):
        self.failed.add(recovery_key(entry.task))

    def fail_many(self, entries, *, code):
        for entry in entries:
            self.fail(entry, code=code)


def task(symbol="AAVEUSDT"):
    return RestFillTask(
        InstrumentId(MarketId(1), Symbol(symbol)),
        TimeRange(UtcTimestamp(NOW), UtcTimestamp(NOW + timedelta(minutes=1))),
        "gap",
    )


def result(item):
    return RestFillResult(item, 1, 1, 1, UtcTimestamp(NOW), UtcTimestamp(NOW))


def test_retry_after_releases_slot_and_survives_restart():
    async def scenario():
        store = MemoryReceipts()
        seen = []
        failed = asyncio.Event()
        other_finished = asyncio.Event()

        async def execute(item):
            seen.append(str(item.instrument_id.symbol))
            if item.instrument_id.symbol == Symbol("AAVEUSDT"):
                failed.set()
                raise TemporarySourceError(retry_after_s=0.25)
            other_finished.set()
            return result(item)

        queue = AsyncRestFillQueue(
            async_executor=execute, worker_count=1, store=store, retry_base_s=0.01
        )
        await queue.start()
        await queue.enqueue(task())
        assert not await queue.enqueue(task())
        await queue.enqueue(task("ADAUSDT"))
        await asyncio.wait_for(other_finished.wait(), 1)
        await asyncio.sleep(0.03)
        assert seen == ["AAVEUSDT", "ADAUSDT"]
        await queue.close()
        assert len(store.pending()) == 1
        assert store.pending()[0].attempt == 1
        recovered = asyncio.Event()

        async def healthy(item):
            recovered.set()
            return result(item)

        restarted = AsyncRestFillQueue(async_executor=healthy, worker_count=1, store=store)
        await restarted.start()
        await asyncio.sleep(0.02)
        assert not recovered.is_set()  # persisted Retry-After still applies
        await asyncio.wait_for(recovered.wait(), 1)
        await asyncio.sleep(0.01)
        await restarted.close()
        assert not store.rows

    asyncio.run(scenario())


def test_permanent_rest_error_is_not_retried_and_remains_diagnosable():
    async def scenario():
        store = MemoryReceipts()
        attempts = 0

        async def execute(item):
            nonlocal attempts
            attempts += 1
            raise ValueError("schema mismatch")

        queue = AsyncRestFillQueue(
            async_executor=execute, worker_count=1, store=store, retry_base_s=0.01
        )
        await queue.start()
        await queue.enqueue(task())
        await asyncio.sleep(0.1)
        await queue.close()
        assert attempts == 1 and len(store.failed) == 1 and not store.pending()

    asyncio.run(scenario())


def test_full_buffer_backpressures_and_retains_cancelled_intake_for_restart():
    async def scenario():
        store = MemoryReceipts()
        calls = 0

        async def unavailable(rows):
            nonlocal calls
            calls += 1
            raise ConnectionError()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=2,
            store=store,
            async_writer=unavailable,
            retry_base_s=10,
        )
        await buffer.start()
        for i in range(2):
            await buffer.submit_wait(_row(NOW + timedelta(minutes=i), NOW))
        blocked = asyncio.create_task(buffer.submit_wait(_row(NOW + timedelta(minutes=2), NOW)))
        await asyncio.sleep(0.04)
        assert not blocked.done() and len(buffer._buffer) == 2
        assert len(store.pending()) == 3 and calls == 1  # no tight write retry
        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)
        with pytest.raises(UnflushedBufferError):
            await buffer.close()
        assert len(store.pending()) == 3
        repaired = []

        async def fill(item):
            repaired.append(item)
            return result(item)

        queue = AsyncRestFillQueue(async_executor=fill, worker_count=1, store=store)
        await queue.start()
        await asyncio.sleep(0.04)
        await queue.close()
        assert len(repaired) == 3 and not store.rows

    asyncio.run(scenario())


def test_transient_insert_then_success_confirms_receipts_once():
    async def scenario():
        store = MemoryReceipts()
        saved = []
        attempts = 0

        async def write(rows):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError()
            saved.extend(rows)

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=2,
            store=store,
            async_writer=write,
            retry_base_s=0.02,
        )
        await buffer.start()
        await buffer.submit_wait(_row(NOW, NOW))
        await asyncio.sleep(0.08)
        await buffer.close()
        assert attempts == 2 and len(saved) == 1 and not store.rows

    asyncio.run(scenario())


def test_permanent_insert_fails_visibly_without_repeating_or_erasing_receipt():
    async def scenario():
        store = MemoryReceipts()
        calls = 0

        async def invalid(rows):
            nonlocal calls
            calls += 1
            raise ValueError()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=2,
            store=store,
            async_writer=invalid,
        )
        await buffer.start()
        buffer.submit(_row(NOW, NOW))
        await asyncio.sleep(0.04)
        with pytest.raises(UnflushedBufferError):
            await buffer.close()
        assert calls == 1 and len(store.failed) == 1 and len(store.rows) == 1

    asyncio.run(scenario())


def test_shutdown_cancels_active_io_but_preserves_receipt():
    async def scenario():
        store = MemoryReceipts()
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def stuck(rows):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=2,
            store=store,
            async_writer=stuck,
            shutdown_timeout_s=0.05,
        )
        await buffer.start()
        buffer.submit(_row(NOW, NOW))
        await started.wait()
        with pytest.raises(UnflushedBufferError):
            await asyncio.wait_for(buffer.close(), 0.3)
        assert cancelled.is_set() and len(store.pending()) == 1

    asyncio.run(scenario())


def test_confirmed_windows_resume_after_restart_without_rereading_prefix():
    async def scenario():
        store = MemoryReceipts()
        original = replace(
            task(),
            time_range=TimeRange(UtcTimestamp(NOW), UtcTimestamp(NOW + timedelta(minutes=3))),
        )
        second_started = asyncio.Event()
        ranges = []

        async def fill(window):
            ranges.append(window.time_range)
            if len(ranges) == 2:
                second_started.set()
                await asyncio.Event().wait()
            return result(window)

        queue = AsyncRestFillQueue(
            async_executor=fill, worker_count=1, store=store, window_minutes=1
        )
        await queue.start()
        await queue.enqueue(original)
        await asyncio.wait_for(second_started.wait(), 1)
        await queue.close()
        assert store.pending()[0].task.time_range.start.value == NOW + timedelta(minutes=1)
        resumed = []

        async def healthy(window):
            resumed.append(window.time_range)
            return result(window)

        queue = AsyncRestFillQueue(
            async_executor=healthy, worker_count=1, store=store, window_minutes=1
        )
        await queue.start()
        await asyncio.sleep(0.04)
        await queue.close()
        assert [r.start.value for r in resumed] == [
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=2),
        ]
        assert not store.rows

    asyncio.run(scenario())


def test_cancelled_intake_repairs_without_restarting_live_queue():
    async def scenario():
        store = MemoryReceipts()
        repaired = asyncio.Event()

        async def fill(item):
            repaired.set()
            return result(item)

        queue = AsyncRestFillQueue(async_executor=fill, worker_count=1, store=store)
        await queue.start()

        async def defer(entry):
            await queue.enqueue(entry.task)

        async def unavailable(rows):
            raise ConnectionError()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=1,
            store=store,
            async_writer=unavailable,
            on_deferred=defer,
        )
        await buffer.start()
        buffer.submit(_row(NOW, NOW))
        blocked = asyncio.create_task(buffer.submit_wait(_row(NOW + timedelta(minutes=1), NOW)))
        await asyncio.sleep(0.02)
        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)
        await asyncio.wait_for(repaired.wait(), 1)
        await asyncio.sleep(0.02)
        assert len(store.pending()) == 1  # only the original full-buffer candle remains
        with pytest.raises(UnflushedBufferError):
            await buffer.close()
        await queue.close()

    asyncio.run(scenario())


def test_empty_rest_result_cannot_acknowledge_received_ws_minute():
    from apps.worker.market_data_ws.wiring.io_process import verify_received_range
    from trading.contexts.market_data.application.services.ingestion_retry import (
        TemporaryIngestionError,
    )
    class Index:
        def __init__(self, rows):
            self.rows = rows
        def distinct_ts_opens(self, **kwargs):
            return self.rows
    received = replace(task(), reason='ws_recovery')
    with pytest.raises(TemporaryIngestionError, match='source_incomplete'):
        verify_received_range(received, Index([]))
    verify_received_range(received, Index([received.time_range.start]))


def test_insert_success_with_ack_failure_keeps_receipt_for_idempotent_restart():
    async def scenario():
        class FailingAck(MemoryReceipts):
            def complete_many(self, entries):
                raise ConnectionError()
        store = FailingAck()
        saved = []
        async def write(rows):
            saved.extend(rows)
        buffer = AsyncRawInsertBuffer(writer=_RecordingWriter(), clock=_MonotonicClock(NOW),
                                      flush_interval_ms=5, max_buffer_rows=2, store=store,
                                      async_writer=write, retry_base_s=10)
        await buffer.start()
        buffer.submit(_row(NOW, NOW))
        await asyncio.sleep(.04)
        with pytest.raises(UnflushedBufferError):
            await buffer.close()
        assert saved and len(store.pending()) == 1
    asyncio.run(scenario())


def test_buffer_coalesces_first_arrival_and_drains_inflight_arrivals():
    async def scenario():
        batches = []
        first_started = asyncio.Event()
        release = asyncio.Event()
        drained = asyncio.Event()

        async def write(rows):
            batches.append(list(rows))
            if len(batches) == 1:
                first_started.set()
                await release.wait()
            else:
                drained.set()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(), clock=_MonotonicClock(NOW),
            flush_interval_ms=300, max_buffer_rows=10, async_writer=write,
        )
        await buffer.start()
        # No global timer phase should split two arrivals within one batch interval.
        await asyncio.sleep(0.25)
        buffer.submit(_row(NOW, NOW))
        await asyncio.sleep(0.1)
        assert not batches
        buffer.submit(_row(NOW + timedelta(minutes=1), NOW))
        await asyncio.wait_for(first_started.wait(), 1)
        assert len(batches[0]) == 2
        buffer.submit(_row(NOW + timedelta(minutes=2), NOW))
        release.set()
        # An arrival during INSERT must not pay another coalescing interval.
        await asyncio.wait_for(drained.wait(), 0.15)
        await buffer.close()
        assert [len(batch) for batch in batches] == [2, 1]

    asyncio.run(scenario())
