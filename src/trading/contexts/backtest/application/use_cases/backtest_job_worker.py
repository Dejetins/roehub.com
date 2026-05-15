from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestPreflightResult,
    BacktestRuntimeGuardrails,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobLeaseRepository,
    BacktestJobRepository,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobHeavyPromotion,
)
from trading.contexts.backtest.application.services.v2.preflight import (
    BacktestPreflightService,
)
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
        cancel_event: threading.Event | None = None,
    ) -> Any:
        ...


class BacktestJobCancellationRequested(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BacktestJobWorkerResult:
    job: BacktestJob | None
    claimed: bool
    lease_lost: bool = False
    status: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestJobWorkerUseCase:
    lease_repository: BacktestJobLeaseRepository
    job_repository: BacktestJobRepository
    preflight_service: BacktestPreflightService
    executor: BacktestJobExecutor
    lease_seconds: int
    heartbeat_interval_seconds: float = 30.0
    locked_by: str | None = None
    scheduling_classes: tuple[str, ...] | None = None
    validation_guardrails: BacktestRuntimeGuardrails | None = None

    def run_next(self) -> BacktestJobWorkerResult:
        now = datetime.now(UTC)
        owner = self._locked_by()
        job = self._claim_next(now=now, locked_by=owner)
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
            cancel_event = threading.Event()
            if self.validation_guardrails is None:
                preflight = self.preflight_service.execute(dict(job.request_json))
            else:
                preflight = self.preflight_service.execute(
                    dict(job.request_json),
                    validation_guardrails=self.validation_guardrails,
                )
            with _LeaseHeartbeat(
                lease_repository=self.lease_repository,
                job_id=job.job_id,
                locked_by=owner,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_interval_seconds,
                cancel_event=cancel_event,
            ) as heartbeat:
                execution_result = self.executor.execute(
                    job_id=job.job_id,
                    preflight=preflight,
                    updated_at=datetime.now(UTC),
                    cancel_event=cancel_event,
                )
            if cancel_event.is_set() or heartbeat.cancel_requested:
                cancelled = self.job_repository.finish_with_top_variants(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    now=datetime.now(UTC),
                    locked_by=owner,
                    next_state="cancelled",
                    top_variants=(),
                )
                return BacktestJobWorkerResult(
                    job=cancelled,
                    claimed=True,
                    lease_lost=heartbeat.lease_lost or cancelled is None,
                    status="cancelled",
                )
            if isinstance(execution_result, BacktestJobHeavyPromotion):
                requeued = self.lease_repository.promote_to_heavy_and_requeue(
                    job_id=job.job_id,
                    now=datetime.now(UTC),
                    locked_by=owner,
                    estimated_combinations_upper_bound=(
                        execution_result.estimated_combinations_upper_bound
                    ),
                    actual_combinations=execution_result.actual_combinations,
                    reason=execution_result.reason,
                )
                return BacktestJobWorkerResult(
                    job=requeued,
                    claimed=True,
                    lease_lost=heartbeat.lease_lost or requeued is None,
                    status="requeued_heavy",
                )
            finished = self.job_repository.finish_with_top_variants(
                job_id=job.job_id,
                user_id=job.user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="succeeded",
                top_variants=tuple(execution_result.top_variants),
            )
            return BacktestJobWorkerResult(
                job=finished,
                claimed=True,
                lease_lost=heartbeat.lease_lost or finished is None,
            )
        except BacktestJobCancellationRequested:
            cancelled = self.job_repository.finish_with_top_variants(
                job_id=job.job_id,
                user_id=job.user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="cancelled",
                top_variants=(),
            )
            return BacktestJobWorkerResult(
                job=cancelled,
                claimed=True,
                lease_lost=cancelled is None,
                status="cancelled",
            )
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
            return BacktestJobWorkerResult(job=failed, claimed=True, lease_lost=failed is None)

    def _claim_next(self, *, now: datetime, locked_by: str) -> BacktestJob | None:
        if self.scheduling_classes is None:
            return self.lease_repository.claim_next(
                now=now,
                locked_by=locked_by,
                lease_seconds=self.lease_seconds,
            )
        return self.lease_repository.claim_next(
            now=now,
            locked_by=locked_by,
            lease_seconds=self.lease_seconds,
            scheduling_classes=self.scheduling_classes,
        )

    def _locked_by(self) -> str:
        if self.locked_by is not None and self.locked_by.strip():
            return self.locked_by.strip()
        return f"backtest-worker:{socket.gethostname()}"


class _LeaseHeartbeat:
    def __init__(
        self,
        *,
        lease_repository: BacktestJobLeaseRepository,
        job_id: UUID,
        locked_by: str,
        lease_seconds: int,
        interval_seconds: float,
        cancel_event: threading.Event,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        self._lease_repository = lease_repository
        self._job_id = job_id
        self._locked_by = locked_by
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._cancel_event = cancel_event
        self._stop = threading.Event()
        self._lease_lost = False
        self._cancel_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"backtest-job-heartbeat-{job_id}",
            daemon=True,
        )

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def __enter__(self) -> "_LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self._interval_seconds, 1.0))

    def _run(self) -> None:
        next_heartbeat = monotonic() + self._interval_seconds
        while not self._stop.wait(max(next_heartbeat - monotonic(), 0.0)):
            updated = self._lease_repository.heartbeat(
                job_id=self._job_id,
                now=datetime.now(UTC),
                locked_by=self._locked_by,
                lease_seconds=self._lease_seconds,
            )
            if updated is None:
                self._lease_lost = True
                return
            if updated.cancel_requested_at is not None:
                self._cancel_requested = True
                self._cancel_event.set()
            next_heartbeat = monotonic() + self._interval_seconds


__all__ = [
    "BacktestJobExecutor",
    "BacktestJobCancellationRequested",
    "BacktestJobWorkerResult",
    "BacktestJobWorkerUseCase",
]
