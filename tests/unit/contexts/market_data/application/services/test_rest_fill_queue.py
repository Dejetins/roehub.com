from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

from trading.contexts.market_data.application.dto import RestFillResult, RestFillTask
from trading.contexts.market_data.application.services.rest_fill_queue import AsyncRestFillQueue
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


def _task() -> RestFillTask:
    """
    Build deterministic rest fill task for queue tests.

    Parameters:
    - None.

    Returns:
    - One rest fill task instance.
    """
    start = UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 5, 12, 5, tzinfo=timezone.utc))
    return RestFillTask(
        instrument_id=InstrumentId(MarketId(1), Symbol("BTCUSDT")),
        time_range=TimeRange(start=start, end=end),
        reason="gap",
    )


def test_rest_fill_queue_enqueue_is_non_blocking_for_slow_executor() -> None:
    """Ensure enqueue path returns quickly and does not await slow task execution."""
    async def _scenario() -> None:
        started = threading.Event()
        finished = threading.Event()

        def _slow_executor(task: RestFillTask) -> RestFillResult:
            started.set()
            time.sleep(0.3)
            finished.set()
            return RestFillResult(
                task=task,
                rows_read=0,
                rows_written=0,
                batches_written=0,
                started_at=UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)),
                finished_at=UtcTimestamp(datetime(2026, 2, 5, 12, 0, 1, tzinfo=timezone.utc)),
            )

        queue = AsyncRestFillQueue(executor=_slow_executor, worker_count=1)
        await queue.start()

        enqueue_started = time.monotonic()
        accepted = await queue.enqueue(_task())
        enqueue_elapsed = time.monotonic() - enqueue_started

        assert accepted is True
        assert enqueue_elapsed < 0.05
        assert await asyncio.to_thread(started.wait, 0.2)
        assert not finished.is_set()

        await asyncio.sleep(0.35)
        assert finished.is_set()
        await queue.close()

    asyncio.run(_scenario())


def test_rest_fill_queue_deduplicates_same_task_key() -> None:
    """Ensure queue rejects duplicate pending tasks with the same key."""
    async def _scenario() -> None:
        done = threading.Event()

        def _executor(task: RestFillTask) -> RestFillResult:
            time.sleep(0.15)
            done.set()
            return RestFillResult(
                task=task,
                rows_read=1,
                rows_written=1,
                batches_written=1,
                started_at=UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)),
                finished_at=UtcTimestamp(datetime(2026, 2, 5, 12, 0, 1, tzinfo=timezone.utc)),
            )

        queue = AsyncRestFillQueue(executor=_executor, worker_count=1)
        await queue.start()

        task = _task()
        accepted_first = await queue.enqueue(task)
        accepted_second = await queue.enqueue(task)

        assert accepted_first is True
        assert accepted_second is False

        assert await asyncio.to_thread(done.wait, 1.0)
        await queue.close()

    asyncio.run(_scenario())
