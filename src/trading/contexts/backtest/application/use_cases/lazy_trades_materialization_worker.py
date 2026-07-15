from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestLazyTradesDetailReadModel
from trading.contexts.backtest.application.ports import (
    BacktestJobRepository,
    BacktestLazyTradesMaterializationRepository,
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.services.v2 import BacktestLazyTradesDetailService
from trading.platform.errors import RoehubError


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationWorkerResult:
    task: BacktestLazyTradesMaterializationTask | None
    claimed: bool
    lease_lost: bool = False
    cache_status: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationExecutionResult:
    cache_status: str
    cache_path: str | None = None


class BacktestLazyTradesMaterializationExecutor(Protocol):
    def execute(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
    ) -> BacktestLazyTradesMaterializationExecutionResult: ...


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationWorkerUseCase:
    materialization_repository: BacktestLazyTradesMaterializationRepository
    job_repository: BacktestJobRepository
    lazy_trades_service: BacktestLazyTradesDetailService
    lease_seconds: int
    heartbeat_interval_seconds: float = 30.0
    locked_by: str | None = None
    executor: BacktestLazyTradesMaterializationExecutor | None = None

    def run_next(self) -> BacktestLazyTradesMaterializationWorkerResult:
        now = datetime.now(UTC)
        owner = self._locked_by()
        task = self.materialization_repository.claim_next(
            now=now,
            locked_by=owner,
            lease_seconds=self.lease_seconds,
        )
        if task is None:
            return BacktestLazyTradesMaterializationWorkerResult(task=None, claimed=False)

        try:
            with _LazyTradesMaterializationHeartbeat(
                materialization_repository=self.materialization_repository,
                task_id=task.task_id,
                locked_by=owner,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_interval_seconds,
            ) as heartbeat:
                execution = self._execute(task=task)
            finished = self.materialization_repository.finish_completed(
                task_id=task.task_id,
                owner_user_id=task.owner_user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                cache_status=execution.cache_status,
                cache_path=execution.cache_path,
            )
            return BacktestLazyTradesMaterializationWorkerResult(
                task=finished,
                claimed=True,
                lease_lost=heartbeat.lease_lost or finished is None,
                cache_status=None if finished is None else finished.cache_status,
            )
        except Exception as error:  # noqa: BLE001
            error_payload = _error_payload(error=error, task=task)
            failed = self.materialization_repository.finish_failed(
                task_id=task.task_id,
                owner_user_id=task.owner_user_id,
                now=datetime.now(UTC),
                locked_by=owner,
                last_error=str(error),
                last_error_json=error_payload,
            )
            return BacktestLazyTradesMaterializationWorkerResult(
                task=failed,
                claimed=True,
                lease_lost=failed is None,
                cache_status=None if failed is None else failed.cache_status,
            )

    def _execute(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
    ) -> BacktestLazyTradesMaterializationExecutionResult:
        if self.executor is not None:
            return self.executor.execute(task=task)
        detail = self._execute_in_process_for_tests(task=task)
        return BacktestLazyTradesMaterializationExecutionResult(
            cache_status=str(detail.cache.get("status", "unknown")),
            cache_path=_optional_str(detail.cache.get("cache_path")),
        )

    def _execute_in_process_for_tests(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
    ) -> BacktestLazyTradesDetailReadModel:
        job = self.job_repository.get(
            job_id=task.job_id,
            organization_id=task.organization_id,
            user_id=task.owner_user_id,
        )
        if job is None:
            raise RoehubError(
                code="backtest.job_not_found",
                message="Backtest job was not found for lazy trades materialization",
                details={
                    "job_id": str(task.job_id),
                    "task_id": str(task.task_id),
                    "retryable": False,
                },
            )
        row = self.job_repository.get_top_variant_by_public_key(
            job_id=task.job_id,
            organization_id=task.organization_id,
            public_variant_key=task.public_variant_key,
        )
        if row is None:
            raise RoehubError(
                code="backtest.variant_not_found",
                message="Backtest variant was not found for lazy trades materialization",
                details={
                    "job_id": str(task.job_id),
                    "variant_key": task.public_variant_key,
                    "task_id": str(task.task_id),
                    "retryable": False,
                },
            )
        return self.lazy_trades_service.execute(
            job=job,
            row=row,
            public_variant_key=task.public_variant_key,
        )

    def _locked_by(self) -> str:
        if self.locked_by is not None and self.locked_by.strip():
            return self.locked_by.strip()
        return f"backtest-lazy-detail-worker:{socket.gethostname()}"


class _LazyTradesMaterializationHeartbeat:
    def __init__(
        self,
        *,
        materialization_repository: BacktestLazyTradesMaterializationRepository,
        task_id: UUID,
        locked_by: str,
        lease_seconds: int,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        self._materialization_repository = materialization_repository
        self._task_id = task_id
        self._locked_by = locked_by
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lease_lost = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"backtest-lazy-detail-heartbeat-{task_id}",
            daemon=True,
        )

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    def __enter__(self) -> "_LazyTradesMaterializationHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self._interval_seconds, 1.0))

    def _run(self) -> None:
        next_heartbeat = monotonic() + self._interval_seconds
        while not self._stop.wait(max(next_heartbeat - monotonic(), 0.0)):
            updated = self._materialization_repository.heartbeat(
                task_id=self._task_id,
                now=datetime.now(UTC),
                locked_by=self._locked_by,
                lease_seconds=self._lease_seconds,
            )
            if updated is None:
                self._lease_lost = True
                return
            next_heartbeat = monotonic() + self._interval_seconds


def _error_payload(
    *,
    error: Exception,
    task: BacktestLazyTradesMaterializationTask,
) -> Mapping[str, Any]:
    if isinstance(error, RoehubError):
        return {
            "code": error.code,
            "message": error.message,
            "details": dict(error.details or {}),
        }
    return {
        "code": "unexpected_error",
        "message": "Backtest lazy trades materialization failed",
        "details": {
            "task_id": str(task.task_id),
            "job_id": str(task.job_id),
            "retryable": True,
            "reason": str(error),
        },
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "BacktestLazyTradesMaterializationExecutionResult",
    "BacktestLazyTradesMaterializationExecutor",
    "BacktestLazyTradesMaterializationWorkerResult",
    "BacktestLazyTradesMaterializationWorkerUseCase",
]
