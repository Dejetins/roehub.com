from __future__ import annotations

from uuid import UUID

from trading.contexts.backtest.application.ports import BacktestJobExecutionTrigger
from trading.shared_kernel.primitives import UserId


class DatabaseBacktestJobExecutionTrigger(BacktestJobExecutionTrigger):
    """
    Durable-queue trigger backed by the already-persisted `backtest_jobs` row.

    The current runtime uses the jobs table plus `claim_next` as the queue boundary.
    No extra broker write is required here; this adapter keeps the enqueue boundary explicit
    for API wiring and future worker notification without putting compute in the request path.
    """

    def enqueue(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        request_hash: str,
    ) -> None:
        _ = job_id, user_id, request_hash


__all__ = ["DatabaseBacktestJobExecutionTrigger"]
