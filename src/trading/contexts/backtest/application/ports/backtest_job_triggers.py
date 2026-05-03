from __future__ import annotations

from typing import Protocol

from trading.contexts.backtest.domain.entities import BacktestJob


class BacktestJobExecutionTrigger(Protocol):
    """
    Bounded signal port used after a Backtest job is durably persisted as `queued`.

    The default v1 queue is the `backtest_jobs` row itself. Implementations may notify a
    background worker, but they must not perform scoring or other long-running compute.
    """

    def enqueue(self, *, job: BacktestJob) -> None:
        """
        Signal that a persisted queued job is available for background execution.

        Args:
            job: Persisted queued job snapshot.
        Returns:
            None.
        Assumptions:
            The durable enqueue already happened when `job_repository.create` committed
            state=`queued`; this method is a bounded notification hook only.
        Raises:
            Exception: Adapter-specific failures when notification cannot be completed.
        Side Effects:
            Adapter-specific bounded notification only.
        """
        ...


__all__ = ["BacktestJobExecutionTrigger"]
