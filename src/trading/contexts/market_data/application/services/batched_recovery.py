"""Bounded admission coalescer. One task owns each shared commit, never a caller."""

import asyncio
from collections import deque
from typing import TypeVar

from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    AsyncStreamRecoveryStore,
    RecoveryEntry,
)

T = TypeVar("T")


async def await_owned(future: asyncio.Future[T]) -> T:
    """Repeated caller cancellation cannot propagate into owned durable work."""
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    try:
        result = future.result()
    except BaseException:
        if cancelled:
            raise asyncio.CancelledError from None
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class BatchedRecoveryAdmission:
    def __init__(
        self,
        store: AsyncStreamRecoveryStore,
        *,
        capacity: int = 20000,
        batch_size: int = 256,
        delay_s: float = 0.002,
    ) -> None:
        if not 0 < batch_size <= 256 or capacity <= 0 or delay_s < 0:
            raise ValueError("invalid recovery admission bounds")
        self.store = store
        self.capacity = capacity
        self.batch_size = batch_size
        self.delay_s = delay_s
        self._pending: deque[tuple[RestFillTask, asyncio.Future[RecoveryEntry]]] = deque()
        self._count = 0
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    async def remember(self, task: RestFillTask) -> RecoveryEntry:
        if self._closed or self._count >= self.capacity:
            raise RuntimeError("recovery admission closed or capacity exceeded")
        # Bounded strings prevent an unbounded metadata payload inside a row budget.
        if len(str(task.instrument_id.symbol)) > 128 or len(task.reason) > 128:
            raise ValueError("recovery metadata exceeds byte bound")
        future = asyncio.get_running_loop().create_future()
        self._pending.append((task, future))
        self._count += 1
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name="recovery-batch-admission")
        return await await_owned(future)

    async def _run(self) -> None:
        try:
            while self._pending:
                await asyncio.sleep(self.delay_s)
                batch = [
                    self._pending.popleft() for _ in range(min(self.batch_size, len(self._pending)))
                ]
                try:
                    entries = await self.store.remember_many([task for task, _ in batch])
                    if len(entries) != len(batch):
                        raise RuntimeError("recovery batch response length mismatch")
                except Exception as exc:
                    for _, future in batch:
                        future.set_exception(exc)
                else:
                    for (_, future), entry in zip(batch, entries):
                        future.set_result(entry)
                finally:
                    self._count -= len(batch)
        finally:
            self._runner = None

    async def close(self) -> None:
        self._closed = True
        if self._runner is not None:
            await await_owned(self._runner)
