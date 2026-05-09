from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestPreflightResult
from trading.contexts.backtest.application.ports import (
    BacktestJobLeaseRepository,
    BacktestJobRepository,
)
from trading.contexts.backtest.application.services.v2 import BacktestPreflightService
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
)


class BacktestJobExecutor(Protocol):
    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class BacktestJobWorkerResult:
    job: BacktestJob | None
    claimed: bool


@dataclass(frozen=True, slots=True)
class BacktestJobWorkerUseCase:
    lease_repository: BacktestJobLeaseRepository
    job_repository: BacktestJobRepository
    preflight_service: BacktestPreflightService
    executor: BacktestJobExecutor
    lease_seconds: int
    locked_by: str | None = None

    def run_next(self) -> BacktestJobWorkerResult:
        now = datetime.now(UTC)
        owner = self._locked_by()
        job = self.lease_repository.claim_next(
            now=now,
            locked_by=owner,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return BacktestJobWorkerResult(job=None, claimed=False)

        self.lease_repository.update_progress(
            job_id=job.job_id,
            now=datetime.now(UTC),
            locked_by=owner,
            stage="stage_a",
            processed_units=0,
            total_units=1,
        )
        try:
            preflight = self.preflight_service.execute(dict(job.request_json))
            execution_result = self.executor.execute(
                job_id=job.job_id,
                preflight=preflight,
                updated_at=datetime.now(UTC),
            )
            finished = self.job_repository.finish_with_top_variants(
                job_id=job.job_id,
                user_id=job.user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="succeeded",
                top_variants=tuple(execution_result.top_variants),
            )
            return BacktestJobWorkerResult(job=finished, claimed=True)
        except Exception as error:  # noqa: BLE001
            failed = self.job_repository.finish_with_top_variants(
                job_id=job.job_id,
                user_id=job.user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="failed",
                top_variants=(),
                last_error=str(error),
                last_error_json=BacktestJobErrorPayload(
                    code="unexpected_error",
                    message="Backtest job execution failed",
                    details={"reason": str(error)},
                ),
            )
            return BacktestJobWorkerResult(job=failed, claimed=True)

    def _locked_by(self) -> str:
        if self.locked_by is not None and self.locked_by.strip():
            return self.locked_by.strip()
        return f"backtest-worker:{socket.gethostname()}"


__all__ = [
    "BacktestJobExecutor",
    "BacktestJobWorkerResult",
    "BacktestJobWorkerUseCase",
]
