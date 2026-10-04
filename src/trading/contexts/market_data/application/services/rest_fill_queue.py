from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable
from uuid import uuid4

from trading.contexts.market_data.application.dto import RestFillResult, RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    RecoveryEntry,
    StreamRecoveryStore,
    recovery_key,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    ErrorClassifier,
    retry_delay,
    transient_ingestion_error,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RestFillQueueHooks:
    """
    Optional lifecycle callbacks for asynchronous REST fill queue.

    Parameters:
    - on_task_enqueued: callback invoked when task is accepted into queue.
    - on_task_started: callback invoked when worker starts processing task.
    - on_task_succeeded: callback with `(task, result, duration_seconds)` after success.
    - on_task_failed: callback with `(task, exc, duration_seconds)` after failure.

    Assumptions/Invariants:
    - Callbacks are lightweight and non-blocking.
    """

    on_task_enqueued: Callable[[RestFillTask], None] | None = None
    on_task_started: Callable[[RestFillTask], None] | None = None
    on_task_succeeded: Callable[[RestFillTask, RestFillResult, float], None] | None = None
    on_task_failed: Callable[[RestFillTask, Exception, float], None] | None = None
    on_task_cancelled: Callable[[RestFillTask], None] | None = None


class AsyncRestFillQueue:
    """Bounded repair queue. Cooldowns release worker slots; durable receipts survive stop.

    The optional async executor must terminate and join its I/O when cancelled.
    The legacy synchronous executor remains supported by the scheduler; it must
    bound its own calls. WS wiring uses a managed process instead of a thread.
    """

    def __init__(
        self,
        *,
        executor: Callable[[RestFillTask], RestFillResult] | None = None,
        worker_count: int,
        hooks: RestFillQueueHooks | None = None,
        async_executor: Callable[[RestFillTask], Awaitable[RestFillResult]] | None = None,
        store: StreamRecoveryStore | None = None,
        classify_error: ErrorClassifier = transient_ingestion_error,
        retry_base_s: float = 15,
        max_pending: int = 20000,
        window_minutes: int | None = None,
    ) -> None:
        if worker_count <= 0 or max_pending <= 0 or retry_base_s <= 0:
            raise ValueError("queue bounds must be positive")
        if window_minutes is not None and window_minutes <= 0:
            raise ValueError("window_minutes must be positive")
        if (executor is None) == (async_executor is None):
            raise ValueError("exactly one executor is required")
        self._executor = executor
        self._async_executor = async_executor
        self._worker_count = worker_count
        self._hooks = hooks or RestFillQueueHooks()
        self._store = store
        self._classify_error = classify_error
        self._retry_base_s = retry_base_s
        self._max_pending = max_pending
        self._window_minutes = window_minutes
        self._pending: dict[str, RecoveryEntry] = {}
        self._active: set[str] = set()
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = False
        self.failure: BaseException | None = None

    async def start(self) -> None:
        if self._workers:
            return
        if self._store is not None:
            for entry in self._store.pending():
                self._pending[entry.task_key] = entry
        self._workers = [
            asyncio.create_task(self._worker_loop(), name=f"rest-fill-{i}")
            for i in range(self._worker_count)
        ]

    async def enqueue(self, task: RestFillTask) -> bool:
        async with self._condition:
            if self._stopping:
                raise RuntimeError("rest fill queue is stopping")
            key = recovery_key(task)
            if key in self._pending:
                return False
            if len(self._pending) >= self._max_pending:
                raise RuntimeError("rest fill queue capacity exceeded")
            entry = self._store.remember(task) if self._store else RecoveryEntry(task, uuid4())
            self._pending[key] = entry
            self._condition.notify_all()
        _emit_enqueued(self._hooks.on_task_enqueued, task)
        return True

    async def close(self) -> None:
        async with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._async_executor is not None:
            # Receipts have already been saved; cancel also joins each managed child.
            for worker in self._workers:
                worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self.failure is not None:
            raise RuntimeError("rest fill queue failed; repair receipts retained") from self.failure

    async def _next(self) -> tuple[str, RecoveryEntry] | None:
        async with self._condition:
            while True:
                if self.failure is not None:
                    return None
                now = datetime.now(UTC)
                active_instruments = {
                    self._pending[key].task.instrument_id
                    for key in self._active
                    if key in self._pending
                }
                available = [
                    (key, entry)
                    for key, entry in self._pending.items()
                    if key not in self._active
                    and entry.task.instrument_id not in active_instruments
                ]
                ready = [
                    (key, entry)
                    for key, entry in available
                    if entry.next_retry_at is None or entry.next_retry_at <= now
                ]
                if ready:
                    key, entry = ready[0]
                    self._active.add(key)
                    return key, entry
                if self._stopping:
                    return None
                delay = min(
                    (
                        (entry.next_retry_at - now).total_seconds()
                        for _, entry in available
                        if entry.next_retry_at
                    ),
                    default=60,
                )
                try:
                    await asyncio.wait_for(self._condition.wait(), max(0.001, delay))
                except TimeoutError:
                    pass

    async def _worker_loop(self) -> None:
        try:
            while (item := await self._next()) is not None:
                key, entry = item
                await self._run_task(key, entry)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
            # Safe metadata only; driver exception messages may contain connection details.
            log.error("rest fill queue stopped error_type=%s", type(exc).__name__)

    async def _run_task(self, key: str, entry: RecoveryEntry) -> None:
        task = entry.task
        work = task
        if self._window_minutes is not None:
            end = min(
                task.time_range.end.value,
                task.time_range.start.value + timedelta(minutes=self._window_minutes),
            )
            work = RestFillTask(
                task.instrument_id, TimeRange(task.time_range.start, UtcTimestamp(end)), task.reason
            )
        started = asyncio.get_running_loop().time()
        _emit_started(self._hooks.on_task_started, task)
        try:
            if self._async_executor is not None:
                result = await self._async_executor(work)
            else:
                assert self._executor is not None
                result = await asyncio.to_thread(self._executor, work)
        except asyncio.CancelledError:
            if self._hooks.on_task_cancelled:
                self._hooks.on_task_cancelled(task)
            raise
        except Exception as exc:
            _emit_failed(
                self._hooks.on_task_failed, task, exc, asyncio.get_running_loop().time() - started
            )
            code = self._classify_error(exc)
            if code is not None:
                entry = replace(entry, attempt=entry.attempt + 1)
                next_at = datetime.now(UTC) + timedelta(
                    seconds=retry_delay(exc, entry.attempt, base=self._retry_base_s)
                )
                if self._store:
                    self._store.retry(entry, next_retry_at=next_at, code=code)
                self._pending[key] = replace(entry, next_retry_at=next_at, error_code=code)
                log.warning(
                    "rest fill waiting code=%s attempt=%s next_retry_at=%s",
                    code,
                    entry.attempt,
                    next_at.isoformat(),
                )
            else:
                if self._store:
                    self._store.fail(entry, code="non_retryable_ingestion_error")
                self._pending.pop(key, None)
                log.error("rest fill terminal error_type=%s", type(exc).__name__)
        else:
            if work.time_range.end.value < task.time_range.end.value:
                if self._store:
                    self._store.advance(entry, start_at=work.time_range.end.value)
                continued = RestFillTask(
                    task.instrument_id,
                    TimeRange(work.time_range.end, task.time_range.end),
                    task.reason,
                )
                self._pending.pop(key, None)
                self._pending[key] = replace(
                    entry,
                    task=continued,
                    key=entry.task_key,
                    attempt=0,
                    next_retry_at=None,
                    error_code=None,
                )
            else:
                if self._store:
                    self._store.complete(entry)
                self._pending.pop(key, None)
            _emit_succeeded(
                self._hooks.on_task_succeeded,
                task,
                result,
                asyncio.get_running_loop().time() - started,
            )
        finally:
            async with self._condition:
                self._active.discard(key)
                self._condition.notify_all()


def _emit_enqueued(
    callback: Callable[[RestFillTask], None] | None,
    task: RestFillTask,
) -> None:
    """
    Trigger enqueue hook when callback is provided.

    Parameters:
    - callback: enqueue callback.
    - task: accepted task.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback side effects are owned by caller.

    Errors/Exceptions:
    - None.

    Side effects:
    - Invokes callback when present.
    """
    if callback is None:
        return
    callback(task)


def _emit_started(
    callback: Callable[[RestFillTask], None] | None,
    task: RestFillTask,
) -> None:
    """
    Trigger task-start hook when callback exists.

    Parameters:
    - callback: start callback.
    - task: started task.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback side effects are owned by caller.

    Errors/Exceptions:
    - None.

    Side effects:
    - Invokes callback when present.
    """
    if callback is None:
        return
    callback(task)


def _emit_succeeded(
    callback: Callable[[RestFillTask, RestFillResult, float], None] | None,
    task: RestFillTask,
    result: RestFillResult,
    duration_seconds: float,
) -> None:
    """
    Trigger success hook when callback exists.

    Parameters:
    - callback: success callback.
    - task: successful task.
    - result: execution result.
    - duration_seconds: wall-clock duration.

    Returns:
    - None.

    Assumptions/Invariants:
    - Duration is non-negative.

    Errors/Exceptions:
    - None.

    Side effects:
    - Invokes callback when present.
    """
    if callback is None:
        return
    callback(task, result, duration_seconds)


def _emit_failed(
    callback: Callable[[RestFillTask, Exception, float], None] | None,
    task: RestFillTask,
    exc: Exception,
    duration_seconds: float,
) -> None:
    """
    Trigger failure hook when callback exists.

    Parameters:
    - callback: failure callback.
    - task: failed task.
    - exc: captured exception.
    - duration_seconds: wall-clock duration.

    Returns:
    - None.

    Assumptions/Invariants:
    - Duration is non-negative.

    Errors/Exceptions:
    - None.

    Side effects:
    - Invokes callback when present.
    """
    if callback is None:
        return
    callback(task, exc, duration_seconds)
