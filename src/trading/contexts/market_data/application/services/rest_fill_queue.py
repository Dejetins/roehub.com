from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from trading.contexts.market_data.application.dto import RestFillResult, RestFillTask

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


class AsyncRestFillQueue:
    """
    Background queue executing REST fill tasks with bounded worker concurrency.

    Parameters:
    - executor: sync callable that executes one task and returns report.
    - worker_count: number of queue workers.
    - hooks: optional metrics/log hooks.

    Assumptions/Invariants:
    - Executor is thread-safe for concurrent calls.
    - Tasks are idempotent enough to tolerate retries/reordering at orchestration level.
    """

    def __init__(
        self,
        *,
        executor: Callable[[RestFillTask], RestFillResult],
        worker_count: int,
        hooks: RestFillQueueHooks | None = None,
    ) -> None:
        """
        Initialize queue internals and validate constructor arguments.

        Parameters:
        - executor: synchronous task execution callable.
        - worker_count: worker pool size.
        - hooks: optional callbacks.

        Returns:
        - None.

        Assumptions/Invariants:
        - `worker_count` must be positive.

        Errors/Exceptions:
        - Raises `ValueError` on invalid arguments.

        Side effects:
        - None.
        """
        if executor is None:  # type: ignore[truthy-bool]
            raise ValueError("AsyncRestFillQueue requires executor")
        if worker_count <= 0:
            raise ValueError("worker_count must be > 0")

        self._executor = executor
        self._worker_count = worker_count
        self._hooks = hooks if hooks is not None else RestFillQueueHooks()

        self._queue: asyncio.Queue[RestFillTask] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._pending_keys: set[str] = set()
        self._pending_lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Start background workers.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Method may be called repeatedly; workers are spawned only once.

        Errors/Exceptions:
        - None.

        Side effects:
        - Spawns worker tasks.
        """
        if self._workers:
            return
        for index in range(self._worker_count):
            task = asyncio.create_task(self._worker_loop(), name=f"rest-fill-worker-{index}")
            self._workers.append(task)

    async def enqueue(self, task: RestFillTask) -> bool:
        """
        Enqueue task if equivalent key is not already pending or running.

        Parameters:
        - task: fill task to enqueue.

        Returns:
        - `True` when task was accepted, `False` when deduplicated.

        Assumptions/Invariants:
        - Equality key is based on instrument, range, and reason.

        Errors/Exceptions:
        - None.

        Side effects:
        - Adds task into in-memory queue and invokes enqueue hook on success.
        """
        key = _task_key(task)
        async with self._pending_lock:
            if key in self._pending_keys:
                return False
            self._pending_keys.add(key)

        self._queue.put_nowait(task)
        _emit_enqueued(self._hooks.on_task_enqueued, task)
        return True

    async def close(self) -> None:
        """
        Signal shutdown and wait for workers to stop after queue drain.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Safe to call multiple times.

        Errors/Exceptions:
        - None.

        Side effects:
        - Stops worker loops after queued tasks finish.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker_loop(self) -> None:
        """
        Process queued tasks until shutdown and queue drain.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Loop exits only when stop signal is set and queue becomes empty.

        Errors/Exceptions:
        - None. Task-level failures are logged and reported through hooks.

        Side effects:
        - Executes REST fill tasks via executor callable.
        """
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return

            item = await self._try_get_task()
            if item is None:
                continue
            await self._run_task(item)

    async def _run_task(self, task: RestFillTask) -> None:
        """
        Execute one queued task and trigger corresponding hooks.

        Parameters:
        - task: one queued fill task.

        Returns:
        - None.

        Assumptions/Invariants:
        - Executor is synchronous and therefore executed in worker thread.

        Errors/Exceptions:
        - None. Exceptions are captured and converted into failed-task hook calls.

        Side effects:
        - Runs REST fill execution function.
        - Updates pending-key registry.
        """
        started = asyncio.get_running_loop().time()
        _emit_started(self._hooks.on_task_started, task)
        key = _task_key(task)

        try:
            result = await asyncio.to_thread(self._executor, task)
        except Exception as exc:  # noqa: BLE001
            duration_seconds = max(asyncio.get_running_loop().time() - started, 0.0)
            _emit_failed(self._hooks.on_task_failed, task, exc, duration_seconds)
            log.exception("rest fill task failed: %s", task)
        else:
            duration_seconds = max(asyncio.get_running_loop().time() - started, 0.0)
            _emit_succeeded(self._hooks.on_task_succeeded, task, result, duration_seconds)
        finally:
            async with self._pending_lock:
                self._pending_keys.discard(key)

    async def _try_get_task(self) -> RestFillTask | None:
        """
        Poll queue with timeout to keep stop checks responsive.

        Parameters:
        - None.

        Returns:
        - Next task or `None` on timeout.

        Assumptions/Invariants:
        - Queue is unbounded and local to this class.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.2)
        except TimeoutError:
            return None


def _task_key(task: RestFillTask) -> str:
    """
    Build deterministic deduplication key for task registry.

    Parameters:
    - task: fill task.

    Returns:
    - Text key containing instrument/range/reason fields.

    Assumptions/Invariants:
    - `instrument_id` and time-range primitives are stable string representations.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return (
        f"{task.instrument_id.market_id.value}:{task.instrument_id.symbol}:"
        f"{task.time_range.start}:{task.time_range.end}:{task.reason}"
    )


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

