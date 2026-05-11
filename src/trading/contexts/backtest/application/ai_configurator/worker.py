from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Callable, TypeVar, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.ports.backtest_ai_configurator import (
    BacktestAiConfigJobRepository,
    BacktestAiConfigLeaseRepository,
)

from .dto import (
    BacktestAiConfigEvent,
    BacktestAiConfigEventName,
    BacktestAiConfigJob,
    BacktestAiConfigLlmAttempt,
)
from .jobs import BACKTEST_AI_CONFIG_SOURCE_PAGE
from .services import BacktestAiConfigPipeline

_T = TypeVar("_T")

_PRE_INPUT_GATE_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("preparing_catalog", "Preparing the current /backtests catalog.", 20),
)
_VALIDATION_STAGE_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("assembling_prompt", "Assembling deterministic catalog-bound request.", 35),
    ("generating", "Generating configuration draft.", 55),
    ("validating_json", "Checking JSON shape and output safety.", 70),
    ("validating_business", "Checking /backtests business rules.", 85),
)


@dataclass(frozen=True, slots=True)
class BacktestAiConfigWorkerResult:
    job: BacktestAiConfigJob | None
    claimed: bool
    lease_lost: bool = False
    skipped_source_page: bool = False
    llm_attempts: tuple[BacktestAiConfigLlmAttempt, ...] = ()


class BacktestAiConfigGenerationLimiter:
    def __init__(
        self,
        *,
        active_generations: int = 1,
        active_callback: Callable[[bool], None] | None = None,
    ) -> None:
        if active_generations <= 0:
            raise ValueError("active_generations must be > 0")
        self.active_generations = active_generations
        self._active_callback = active_callback
        self._semaphore = threading.BoundedSemaphore(value=active_generations)

    def run_locked(self, callback: Callable[[], _T]) -> _T:
        with self._semaphore:
            if self._active_callback is not None:
                self._active_callback(True)
            try:
                return callback()
            finally:
                if self._active_callback is not None:
                    self._active_callback(False)


@dataclass(frozen=True, slots=True)
class BacktestAiConfigWorkerUseCase:
    job_repository: BacktestAiConfigJobRepository
    lease_repository: BacktestAiConfigLeaseRepository
    pipeline: BacktestAiConfigPipeline
    lease_seconds: int
    max_attempts: int
    heartbeat_interval_seconds: float = 30.0
    locked_by: str | None = None
    generation_limiter: BacktestAiConfigGenerationLimiter | None = None

    def run_next(self) -> BacktestAiConfigWorkerResult:
        now = datetime.now(UTC)
        owner = self._locked_by()
        claimed = self.lease_repository.claim_next(
            now=now,
            locked_by=owner,
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_attempts,
        )
        if claimed is None:
            return BacktestAiConfigWorkerResult(job=None, claimed=False)
        if claimed.source_page != BACKTEST_AI_CONFIG_SOURCE_PAGE:
            failed = self.lease_repository.finish(
                job_id=claimed.job_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="failed",
                assistant_message="AI configurator source page is unsupported.",
                last_error="unsupported_source_page",
                last_error_json={
                    "code": "unsupported_source_page",
                    "source_page": claimed.source_page,
                },
            )
            return BacktestAiConfigWorkerResult(
                job=failed,
                claimed=True,
                lease_lost=failed is None,
                skipped_source_page=True,
            )

        for event_name, message, progress in _PRE_INPUT_GATE_EVENTS:
            self.job_repository.append_event(
                event=_event(
                    job=claimed,
                    event_name=event_name,
                    message=message,
                    progress=progress,
                    created_at=now,
                )
            )

        limiter = self.generation_limiter or BacktestAiConfigGenerationLimiter()
        try:
            with _AiConfigLeaseHeartbeat(
                lease_repository=self.lease_repository,
                job_id=claimed.job_id,
                locked_by=owner,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_interval_seconds,
            ) as heartbeat:
                pipeline_result = limiter.run_locked(lambda: self.pipeline.run(job=claimed))

            if pipeline_result.stage == "validation":
                for event_name, message, progress in _VALIDATION_STAGE_EVENTS:
                    self.job_repository.append_event(
                        event=_event(
                            job=claimed,
                            event_name=event_name,
                            message=message,
                            progress=progress,
                            created_at=datetime.now(UTC),
                        )
                    )
                if any(
                    attempt.attempt_kind == "repair"
                    for attempt in pipeline_result.llm_attempts
                ):
                    self.job_repository.append_event(
                        event=_event(
                            job=claimed,
                            event_name="repairing",
                            message=(
                                "Repairing configuration draft from validation errors."
                            ),
                            progress=90,
                            created_at=datetime.now(UTC),
                        )
                    )

            for attempt in pipeline_result.llm_attempts:
                self.job_repository.record_llm_attempt(attempt=attempt)

            finished = self.lease_repository.finish(
                job_id=claimed.job_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state=pipeline_result.status,
                assistant_message=pipeline_result.assistant_message,
                validated_config_json=pipeline_result.validated_config,
                suggestions_json=(
                    pipeline_result.warnings + pipeline_result.suggestions
                ),
                validation_errors_json=pipeline_result.validation_errors,
                model_id=pipeline_result.model_id,
                model_path_hash=pipeline_result.model_path_hash,
                last_error=pipeline_result.last_error,
                last_error_json=pipeline_result.last_error_json,
            )
            if finished is not None:
                self.job_repository.append_event(
                    event=_event(
                        job=finished,
                        event_name=pipeline_result.status,
                        message=pipeline_result.assistant_message,
                        progress=100 if pipeline_result.status == "ready" else 95,
                        created_at=datetime.now(UTC),
                    )
                )
            return BacktestAiConfigWorkerResult(
                job=finished,
                claimed=True,
                lease_lost=heartbeat.lease_lost or finished is None,
                llm_attempts=pipeline_result.llm_attempts,
            )
        except Exception as error:  # noqa: BLE001
            failed = self.lease_repository.finish(
                job_id=claimed.job_id,
                now=datetime.now(UTC),
                locked_by=owner,
                next_state="failed",
                assistant_message="AI configurator request failed during generation.",
                last_error="worker_runtime_error",
                last_error_json={
                    "code": "worker_runtime_error",
                    "message": str(error),
                },
            )
            if failed is not None:
                self.job_repository.append_event(
                    event=_event(
                        job=failed,
                        event_name="failed",
                        message="AI configurator request failed during generation.",
                        progress=95,
                        created_at=datetime.now(UTC),
                    )
                )
            return BacktestAiConfigWorkerResult(
                job=failed,
                claimed=True,
                lease_lost=failed is None,
            )

    def _locked_by(self) -> str:
        if self.locked_by is not None and self.locked_by.strip():
            return self.locked_by.strip()
        hostname = socket.gethostname().strip() or "unknown-host"
        return f"backtest-ai-configurator-worker:{hostname}"


class _AiConfigLeaseHeartbeat:
    def __init__(
        self,
        *,
        lease_repository: BacktestAiConfigLeaseRepository,
        job_id: UUID,
        locked_by: str,
        lease_seconds: int,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        self._lease_repository = lease_repository
        self._job_id = job_id
        self._locked_by = locked_by
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lease_lost = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"backtest-ai-config-heartbeat-{job_id}",
            daemon=True,
        )

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    def __enter__(self) -> "_AiConfigLeaseHeartbeat":
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
            next_heartbeat = monotonic() + self._interval_seconds


def _event(
    *,
    job: BacktestAiConfigJob,
    event_name: str,
    message: str,
    progress: int,
    created_at: datetime,
) -> BacktestAiConfigEvent:
    return BacktestAiConfigEvent(
        event_id=uuid4(),
        job_id=job.job_id,
        owner_user_id=job.owner_user_id,
        event_name=cast(BacktestAiConfigEventName, event_name),
        message=message,
        payload_json={
            "job_id": str(job.job_id),
            "status": event_name,
            "message": message,
            "progress": progress,
        },
        created_at=created_at,
    )


__all__ = [
    "BacktestAiConfigGenerationLimiter",
    "BacktestAiConfigWorkerResult",
    "BacktestAiConfigWorkerUseCase",
]
