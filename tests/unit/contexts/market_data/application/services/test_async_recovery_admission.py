"""Ownership and bounded-memory tests for asynchronous recovery admission."""

import asyncio

import pytest
from test_insert_buffer import _MonotonicClock, _RecordingWriter, _row
from test_stream_recovery import NOW, MemoryReceipts, task

from trading.contexts.market_data.application.services.batched_recovery import (
    BatchedRecoveryAdmission,
)
from trading.contexts.market_data.application.services.insert_buffer import (
    AsyncRawInsertBuffer,
    UnflushedBufferError,
)


class AsyncMemory:
    def __init__(self):
        self.sync = MemoryReceipts()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.batches = []

    def __getattr__(self, name):
        return getattr(self.sync, name)

    async def pending(self):
        return self.sync.pending()

    async def failed_history_repairs(self):
        return []

    async def advance(self, entry, *, start_at):
        self.sync.advance(entry, start_at=start_at)

    async def complete(self, entry):
        self.sync.complete(entry)

    async def retry(self, entry, *, next_retry_at, code):
        self.sync.retry(entry, next_retry_at=next_retry_at, code=code)

    async def fail(self, entry, *, code):
        self.sync.fail(entry, code=code)

    async def remember_many(self, tasks):
        self.batches.append(tasks)
        self.entered.set()
        await self.release.wait()
        return [self.remember(t) for t in tasks]

    async def complete_many(self, entries):
        for entry in entries:
            self.sync.complete(entry)

    async def fail_many(self, entries, *, code):
        for entry in entries:
            self.sync.fail(entry, code=code)


def test_double_cancellation_does_not_cancel_shared_commit():
    async def scenario():
        store = AsyncMemory()
        admission = BatchedRecoveryAdmission(store)
        first = asyncio.create_task(admission.remember(task()))
        second = asyncio.create_task(admission.remember(task("ADAUSDT")))
        await store.entered.wait()
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not second.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await second).task == task("ADAUSDT")
        await admission.close()
        assert len(store.rows) == 2 and len(store.batches) == 1

    asyncio.run(scenario())


def test_capacity_wait_cancels_without_receipt_or_shutdown_deadlock():
    async def scenario():
        store = AsyncMemory()
        store.release.set()
        admission = BatchedRecoveryAdmission(store)

        async def unavailable(rows):
            raise ConnectionError()

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=1,
            async_store=store,
            admission=admission,
            async_writer=unavailable,
            shutdown_timeout_s=0.05,
        )
        await buffer.start()
        await buffer.submit_wait(_row(NOW, NOW))
        waiting = asyncio.create_task(buffer.submit_wait(_row(NOW, NOW)))
        await asyncio.sleep(0.02)
        assert buffer._retained_rows == 2
        with pytest.raises(asyncio.QueueFull):
            await buffer.submit_wait(_row(NOW, NOW))
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, 0.2)
        assert buffer._retained_rows == 1 and len(store.rows) == 1
        with pytest.raises(UnflushedBufferError):
            await asyncio.wait_for(buffer.close(), 0.3)
        await admission.close()

    asyncio.run(scenario())


def test_cancel_after_commit_keeps_row_owned_until_canonical_write():
    async def scenario():
        store = AsyncMemory()
        admission = BatchedRecoveryAdmission(store)
        written = []

        async def write(rows):
            written.extend(rows)

        buffer = AsyncRawInsertBuffer(
            writer=_RecordingWriter(),
            clock=_MonotonicClock(NOW),
            flush_interval_ms=5,
            max_buffer_rows=2,
            async_store=store,
            admission=admission,
            async_writer=write,
        )
        await buffer.start()
        producer = asyncio.create_task(buffer.submit_wait(_row(NOW, NOW)))
        await store.entered.wait()
        producer.cancel()
        await asyncio.sleep(0)
        producer.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await buffer.close()
        await admission.close()
        assert len(written) == 1 and not store.rows
        assert buffer._retained_rows == buffer._retained_bytes == 0

    asyncio.run(scenario())
