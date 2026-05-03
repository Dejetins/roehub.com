from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobLeaseRepository,
    BacktestJobRepository,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobState,
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
class BacktestJobWorkerRunResult:
    claimed_job_id: UUID | None
    state: BacktestJobState | None
    executed: bool
    lease_lost: bool = False


@dataclass(frozen=True, slots=True)
class BacktestJobWorkerUseCase:
    """
    Claim one queued Backtest job and run it outside the API request path.
    """

    job_repository: BacktestJobRepository
    lease_repository: BacktestJobLeaseRepository
    executor: BacktestJobExecutor
    lease_seconds: int = 900

    def run_next(
        self,
        *,
        locked_by: str,
        now: datetime | None = None,
    ) -> BacktestJobWorkerRunResult:
        claim_time = now or datetime.now(UTC)
        claimed = self.lease_repository.claim_next(
            now=claim_time,
            locked_by=locked_by,
            lease_seconds=self.lease_seconds,
        )
        if claimed is None:
            return BacktestJobWorkerRunResult(
                claimed_job_id=None,
                state=None,
                executed=False,
            )

        if claimed.cancel_requested_at is not None:
            return self._finish_cancelled(job=claimed, locked_by=locked_by, executed=False)

        progressed = self.lease_repository.update_progress(
            job_id=claimed.job_id,
            now=datetime.now(UTC),
            locked_by=locked_by,
            stage="stage_a",
            processed_units=0,
            total_units=1,
        )
        if progressed is None:
            return BacktestJobWorkerRunResult(
                claimed_job_id=claimed.job_id,
                state=None,
                executed=False,
                lease_lost=True,
            )

        preflight = _preflight_from_job(job=progressed)
        try:
            execution_result = self.executor.execute(
                job_id=progressed.job_id,
                preflight=preflight,
                updated_at=datetime.now(UTC),
            )
        except Exception as error:  # noqa: BLE001
            finished = self.job_repository.finish_with_top_variants(
                job_id=progressed.job_id,
                user_id=progressed.user_id,
                now=datetime.now(UTC),
                locked_by=locked_by,
                next_state="failed",
                top_variants=(),
                last_error=str(error),
                last_error_json=BacktestJobErrorPayload(
                    code="unexpected_error",
                    message="Backtest job execution failed",
                    details={"reason": str(error)},
                ),
            )
            return BacktestJobWorkerRunResult(
                claimed_job_id=progressed.job_id,
                state=finished.state if finished is not None else None,
                executed=True,
                lease_lost=finished is None,
            )

        current = self.job_repository.get(
            job_id=progressed.job_id,
            user_id=progressed.user_id,
        )
        if current is not None and current.cancel_requested_at is not None:
            return self._finish_cancelled(job=current, locked_by=locked_by, executed=True)

        finished = self.job_repository.finish_with_top_variants(
            job_id=progressed.job_id,
            user_id=progressed.user_id,
            now=datetime.now(UTC),
            locked_by=locked_by,
            next_state="succeeded",
            top_variants=tuple(execution_result.top_variants),
        )
        return BacktestJobWorkerRunResult(
            claimed_job_id=progressed.job_id,
            state=finished.state if finished is not None else None,
            executed=True,
            lease_lost=finished is None,
        )

    def _finish_cancelled(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        executed: bool,
    ) -> BacktestJobWorkerRunResult:
        finished = self.job_repository.finish_with_top_variants(
            job_id=job.job_id,
            user_id=job.user_id,
            now=datetime.now(UTC),
            locked_by=locked_by,
            next_state="cancelled",
            top_variants=(),
        )
        return BacktestJobWorkerRunResult(
            claimed_job_id=job.job_id,
            state=finished.state if finished is not None else None,
            executed=executed,
            lease_lost=finished is None,
        )


def _preflight_from_job(*, job: BacktestJob) -> BacktestPreflightResult:
    request_payload = dict(job.request_json)
    artifact_metadata_payload = request_payload.pop("artifact_metadata", None)
    request_payload.pop("idempotency", None)
    if not isinstance(artifact_metadata_payload, Mapping):
        raise ValueError("Backtest queued job is missing artifact_metadata")
    return BacktestPreflightResult(
        normalized_request=request_payload,
        request_hash=job.request_hash,
        result_config_hash=job.engine_params_hash,
        artifact_metadata=BacktestArtifactMetadata(
            artifact_slot=str(artifact_metadata_payload["artifact_slot"]),
            artifact_slot_generation=int(artifact_metadata_payload["artifact_slot_generation"]),
            artifact_manifest_hash=str(artifact_metadata_payload["artifact_manifest_hash"]),
            artifact_asof_date=str(artifact_metadata_payload["artifact_asof_date"]),
            hit_times_manifest_hash=_optional_str(
                value=artifact_metadata_payload.get("hit_times_manifest_hash")
            ),
            published_at_utc=str(artifact_metadata_payload["published_at_utc"]),
        ),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=0,
            candidate_combinations=0,
            tp_sl_cells=0,
            cost_class="unknown",
        ),
    )


def _optional_str(*, value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "BacktestJobExecutor",
    "BacktestJobWorkerRunResult",
    "BacktestJobWorkerUseCase",
]
