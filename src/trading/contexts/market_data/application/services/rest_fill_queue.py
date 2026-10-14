from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable, Sequence
from uuid import uuid4

from trading.contexts.market_data.application.dto import RestFillResult, RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    AsyncStreamRecoveryStore,
    RecoveryEntry,
    StreamRecoveryStore,
    recovery_key,
)
from trading.contexts.market_data.application.services.batched_recovery import (
    BatchedRecoveryAdmission,
    await_owned,
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
        async_store: AsyncStreamRecoveryStore | None = None,
        admission: BatchedRecoveryAdmission | None = None,
        classify_error: ErrorClassifier = transient_ingestion_error,
        retry_base_s: float = 15,
        max_pending: int = 20000,
        window_minutes: int | None = None,
        failed_history_repairs: Sequence[RecoveryEntry] = (),
        should_run: Callable[[RestFillTask], Awaitable[bool]] | None = None,
    ) -> None:
        if worker_count <= 0 or max_pending <= 0 or retry_base_s <= 0:
            raise ValueError("queue bounds must be positive")
        if window_minutes is not None and window_minutes <= 0:
            raise ValueError("window_minutes must be positive")
        if (executor is None) == (async_executor is None):
            raise ValueError("exactly one executor is required")
        self._should_run = should_run
        self._executor = executor
        self._async_executor = async_executor
        self._worker_count = worker_count
        self._hooks = hooks or RestFillQueueHooks()
        self._store = store
        self._async_store = async_store
        self._admission = admission
        self._enqueuing: dict[str, RestFillTask] = {}
        self._admissions: set[asyncio.Task[bool]] = set()
        if (async_store is None) != (admission is None) or (store and async_store):
            raise ValueError("async store requires admission coordinator")
        self._classify_error = classify_error
        self._retry_base_s = retry_base_s
        self._max_pending = max_pending
        self._window_minutes = window_minutes
        self._blocked_history = list(failed_history_repairs)
        self._pending: dict[str, RecoveryEntry] = {}
        self._active: set[str] = set()
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = False
        self.failure: BaseException | None = None

    async def start(self) -> None:
        if self._workers:
            return
        if self._async_store is not None:
            self._blocked_history = await self._async_store.failed_history_repairs()
            for entry in await self._async_store.pending():
                self._pending[entry.task_key] = entry
        if self._store is not None:
            for entry in self._store.pending():
                self._pending[entry.task_key] = entry
        self._workers = [
            asyncio.create_task(self._worker_loop(), name=f"rest-fill-{i}")
            for i in range(self._worker_count)
        ]

    async def enqueue(self, task: RestFillTask) -> bool:
        if self._async_store is not None:
            return await self._enqueue_async(task)
        async with self._condition:
            if self._stopping:
                raise RuntimeError("rest fill queue is stopping")
            if task.reason == "stream_history_repair":
                # A moving now must not bypass an instrument's pending cooldown.
                if any(e.task.instrument_id == task.instrument_id for e in self._pending.values()):
                    return False
                if any(
                    e.task.instrument_id == task.instrument_id
                    and e.task.time_range.start.value < task.time_range.end.value
                    and task.time_range.start.value < e.task.time_range.end.value
                    for e in self._blocked_history
                ):
                    return False
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

    async def _enqueue_async(self, task: RestFillTask) -> bool:
        key = recovery_key(task)
        # No await between checking and reserving; never hold Condition over PG.
        if self._stopping:
            raise RuntimeError("rest fill queue is stopping")
        if key in self._pending or key in self._enqueuing:
            return False
        if task.reason == "stream_history_repair":
            if any(e.task.instrument_id == task.instrument_id for e in self._pending.values()):
                return False
            if any(t.instrument_id == task.instrument_id for t in self._enqueuing.values()):
                return False
            if any(e.task.instrument_id == task.instrument_id
                   and e.task.time_range.start.value < task.time_range.end.value
                   and task.time_range.start.value < e.task.time_range.end.value
                   for e in self._blocked_history):
                return False
        if len(self._pending) + len(self._enqueuing) >= self._max_pending:
            raise RuntimeError("rest fill queue capacity exceeded")
        self._enqueuing[key] = task

        async def persist() -> bool:
            try:
                assert self._admission is not None
                entry = await self._admission.remember(task)
                async with self._condition:
                    if not self._stopping:
                        self._pending[key] = entry
                        self._condition.notify_all()
                _emit_enqueued(self._hooks.on_task_enqueued, task)
                return True
            finally:
                self._enqueuing.pop(key, None)

        owned = asyncio.create_task(persist(), name="durable-repair-admission")
        self._admissions.add(owned)
        try:
            return await await_owned(owned)
        finally:
            self._admissions.discard(owned)

    async def close(self) -> None:
        async with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._admissions:
            await await_owned(asyncio.gather(*self._admissions, return_exceptions=True))
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
            if self._should_run is not None and not await self._should_run(work):
                if self._async_store:
                    await self._async_store.complete(entry)
                if self._store:
                    self._store.complete(entry)
                self._pending.pop(key, None)
                if self._hooks.on_task_cancelled:
                    self._hooks.on_task_cancelled(task)
                log.info("stream_history_cancelled market_id=%s symbol=%s",
                         task.instrument_id.market_id.value, str(task.instrument_id.symbol))
                return
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
            exhausted_history = (
                task.reason == "stream_history_repair"
                and code == "source_history_incomplete"
                and entry.error_code == code
                and entry.attempt >= 3
            )
            if code is not None and not exhausted_history:
                entry = replace(
                    entry,
                    attempt=(
                        1
                        if task.reason == "stream_history_repair" and entry.error_code != code
                        else entry.attempt + 1
                    ),
                )
                next_at = datetime.now(UTC) + timedelta(
                    seconds=retry_delay(exc, entry.attempt, base=self._retry_base_s)
                )
                if self._async_store:
                    await self._async_store.retry(entry, next_retry_at=next_at, code=code)
                if self._store:
                    self._store.retry(entry, next_retry_at=next_at, code=code)
                self._pending[key] = replace(entry, next_retry_at=next_at, error_code=code)
                if entry.error_code != code or entry.attempt & (entry.attempt - 1) == 0:
                    log.warning(
                        "stream_recovery_waiting market_id=%s symbol=%s reason=%s "
                        "start=%s end=%s code=%s attempt=%s next_retry_at=%s",
                        task.instrument_id.market_id.value, str(task.instrument_id.symbol),
                        task.reason, task.time_range.start.value.isoformat(),
                        task.time_range.end.value.isoformat(), code, entry.attempt,
                        next_at.isoformat(),
                    )
            else:
                terminal_code = (
                    "source_history_incomplete"
                    if exhausted_history
                    else "non_retryable_ingestion_error"
                )
                if self._async_store:
                    await self._async_store.fail(entry, code=terminal_code)
                if self._store:
                    self._store.fail(entry, code=terminal_code)
                if task.reason == "stream_history_repair":
                    self._blocked_history.append(replace(entry, error_code=terminal_code))
                self._pending.pop(key, None)
                log.error("rest fill terminal error_type=%s", type(exc).__name__)
        else:
            if work.time_range.end.value < task.time_range.end.value:
                if self._async_store:
                    await self._async_store.advance(entry, start_at=work.time_range.end.value)
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
                if self._async_store:
                    await self._async_store.complete(entry)
                if self._store:
                    self._store.complete(entry)
                self._pending.pop(key, None)
            log.info(
                "stream_recovery_window_completed market_id=%s symbol=%s reason=%s "
                "start=%s end=%s rows_read=%s rows_written=%s",
                work.instrument_id.market_id.value, str(work.instrument_id.symbol),
                work.reason, work.time_range.start.value.isoformat(),
                work.time_range.end.value.isoformat(), result.rows_read, result.rows_written,
            )
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
