from __future__ import annotations

from trading.contexts.backtest.domain.entities import BacktestJob


class DatabaseBacktestJobExecutionTrigger:
    """
    No-op queue trigger for the database-backed Backtest job queue.

    Persisting a row with state=`queued` is the durable enqueue operation. This adapter
    exists so API wiring depends on a bounded trigger port instead of on runtime compute.
    """

    def enqueue(self, *, job: BacktestJob) -> None:
        """
        Accept a queued job notification without doing any scoring work.

        Args:
            job: Persisted queued job snapshot.
        Returns:
            None.
        Assumptions:
            A background worker claims jobs from `backtest_jobs`; there is no external
            broker to notify in the default single-host v1 runtime.
        Raises:
            ValueError: If the adapter is accidentally called for a non-queued job.
        Side Effects:
            None.
        """
        if job.state != "queued":
            raise ValueError("DatabaseBacktestJobExecutionTrigger requires queued job")


__all__ = ["DatabaseBacktestJobExecutionTrigger"]
