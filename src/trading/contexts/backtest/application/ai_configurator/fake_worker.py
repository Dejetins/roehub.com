from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from trading.contexts.backtest.application.ports.backtest_ai_configurator import (
    BacktestAiConfigJobRepository,
    BacktestAiConfigLeaseRepository,
)

from .dto import BacktestAiConfigEvent, BacktestAiConfigEventName, BacktestAiConfigJob
from .services import BacktestAiConfigPipeline

_FAKE_WORKER_ID = "backtest-ai-config-fake-worker"
_PRE_INPUT_GATE_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("preparing_catalog", "Preparing the current /backtests catalog.", 20),
)
_VALIDATION_STAGE_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("collecting_context", "Collecting deterministic tool-agent context.", 35),
    ("generating", "Generating deterministic configuration draft.", 55),
    ("validating_json", "Checking JSON shape and output safety.", 70),
    ("validating_business", "Checking /backtests business rules.", 85),
)


@dataclass(frozen=True, slots=True)
class BacktestAiConfigFakeWorkerUseCase:
    """
    Deterministic Iteration 02 worker path that proves queue and event semantics.
    """

    job_repository: BacktestAiConfigJobRepository
    lease_repository: BacktestAiConfigLeaseRepository
    pipeline: BacktestAiConfigPipeline
    lease_seconds: int = 60
    max_attempts: int = 1
    locked_by: str = _FAKE_WORKER_ID

    def process_next(self, *, now: datetime | None = None) -> BacktestAiConfigJob | None:
        effective_now = datetime.now(UTC) if now is None else now
        claimed = self.lease_repository.claim_next(
            now=effective_now,
            locked_by=self.locked_by,
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_attempts,
        )
        if claimed is None:
            return None

        for event_name, message, progress in _PRE_INPUT_GATE_EVENTS:
            self.job_repository.append_event(
                event=_event(
                    job=claimed,
                    event_name=event_name,
                    message=message,
                    progress=progress,
                    created_at=effective_now,
                )
            )

        pipeline_result = self.pipeline.run(job=claimed)
        if pipeline_result.stage == "validation":
            for event_name, message, progress in _VALIDATION_STAGE_EVENTS:
                self.job_repository.append_event(
                    event=_event(
                        job=claimed,
                        event_name=event_name,
                        message=message,
                        progress=progress,
                        created_at=effective_now,
                    )
                )
            if any(attempt.attempt_kind == "repair" for attempt in pipeline_result.llm_attempts):
                self.job_repository.append_event(
                    event=_event(
                        job=claimed,
                        event_name="repairing",
                        message="Repairing configuration draft from validation errors.",
                        progress=90,
                        created_at=effective_now,
                    )
                )

        for attempt in pipeline_result.llm_attempts:
            self.job_repository.record_llm_attempt(attempt=attempt)

        finished = self.lease_repository.finish(
            job_id=claimed.job_id,
            now=effective_now,
            locked_by=self.locked_by,
            next_state=pipeline_result.status,
            assistant_message=pipeline_result.assistant_message,
            validated_config_json=pipeline_result.validated_config,
            suggestions_json=pipeline_result.warnings + pipeline_result.suggestions,
            validation_errors_json=pipeline_result.validation_errors,
            model_id=pipeline_result.model_id,
            model_path_hash=pipeline_result.model_path_hash,
            last_error=pipeline_result.last_error,
            last_error_json=pipeline_result.last_error_json,
        )
        if finished is None:
            return None

        self.job_repository.append_event(
            event=_event(
                job=finished,
                event_name=pipeline_result.status,
                message=pipeline_result.assistant_message,
                progress=100 if pipeline_result.status == "ready" else 95,
                created_at=effective_now,
            )
        )
        return finished


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

__all__ = ["BacktestAiConfigFakeWorkerUseCase"]
