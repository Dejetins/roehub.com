from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId


class BacktestJobExecutionTrigger(Protocol):
    """
    Signal that a queued Backtest job is available for background execution.
    """

    def enqueue(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        request_hash: str,
    ) -> None:
        """
        Notify the runtime that a persisted queued job should be claimed by a worker.
        """
        ...


__all__ = ["BacktestJobExecutionTrigger"]
