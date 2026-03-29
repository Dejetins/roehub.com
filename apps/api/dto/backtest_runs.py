"""
Pydantic models and deterministic cursor/mapper helpers for public Backtest runs API.

Docs:
  - docs/architecture/backtest/backtest-runs-history-v2.md
  - docs/architecture/roadmap/base_refactor_plan.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from trading.contexts.backtest.application.use_cases import BacktestRunTopReadResult
from trading.contexts.backtest.domain.entities import BacktestJob
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor

from .backtest_jobs import (
    BacktestJobArtifactSlotLiteral,
    BacktestJobExecutionModeLiteral,
    BacktestJobsStateLiteral,
    decode_backtest_jobs_cursor,
    decode_backtest_jobs_state,
    encode_backtest_jobs_cursor,
)
from .backtests import BacktestVariantPayloadRequest

BacktestRunsStateLiteral = BacktestJobsStateLiteral
BacktestRunsStageLiteral = Literal["stage_a", "stage_b", "finalizing"]


class BacktestRunErrorResponse(BaseModel):
    """
    API response model for persisted Roehub-like failed-run payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any]


class BacktestRunStatusResponse(BaseModel):
    """
    API response model for one public Backtest run status/metadata snapshot.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    """

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    mode: Literal["saved", "template"]
    state: BacktestRunsStateLiteral
    stage: BacktestRunsStageLiteral
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    progress_updated_at: datetime | None = None
    processed_units: int
    total_units: int
    execution_mode: BacktestJobExecutionModeLiteral | None = None
    market_id: int | None = None
    symbol: str | None = None
    timeframe: str | None = None
    requested_top_n: int | None = None
    ranking_primary_metric: str | None = None
    ranking_secondary_metric: str | None = None
    artifact_slot: BacktestJobArtifactSlotLiteral | None = None
    artifact_slot_generation: int | None = None
    artifact_manifest_hash: str | None = None
    artifact_asof_date: str | None = None
    last_error: str | None = None
    last_error_json: BacktestRunErrorResponse | None = None


class BacktestRunsListItemResponse(BaseModel):
    """
    API response model for one public Backtest history item.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    mode: Literal["saved", "template"]
    state: BacktestRunsStateLiteral
    stage: BacktestRunsStageLiteral
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    processed_units: int
    total_units: int
    execution_mode: BacktestJobExecutionModeLiteral | None = None
    market_id: int | None = None
    symbol: str | None = None
    timeframe: str | None = None
    requested_top_n: int | None = None
    ranking_primary_metric: str | None = None
    ranking_secondary_metric: str | None = None
    artifact_slot: BacktestJobArtifactSlotLiteral | None = None
    artifact_slot_generation: int | None = None
    artifact_manifest_hash: str | None = None
    artifact_asof_date: str | None = None


class BacktestRunsListResponse(BaseModel):
    """
    API response model for deterministic keyset-paginated public Backtest runs history.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
    """

    model_config = ConfigDict(extra="forbid")

    items: list[BacktestRunsListItemResponse]
    next_cursor: str | None


class BacktestRunTopItemResponse(BaseModel):
    """
    API response model for one public summary-only top row.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    """

    model_config = ConfigDict(extra="forbid")

    rank: int
    variant_key: str
    indicator_variant_key: str
    variant_index: int
    total_return_pct: float
    payload: dict[str, Any]
    summary_metrics_json: dict[str, Any]
    best_tp_pct: float | None = None
    best_sl_pct: float | None = None


class BacktestRunTopResponse(BaseModel):
    """
    API response model for public Backtest run summary table payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    state: BacktestRunsStateLiteral
    execution_mode: BacktestJobExecutionModeLiteral | None = None
    items: list[BacktestRunTopItemResponse]


class BacktestRunVariantReportPostRequest(BaseModel):
    """
    API request envelope for run-scoped lazy `POST /backtests/runs/{run_id}/variant-report`.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    variant: BacktestVariantPayloadRequest
    include_trades: bool = False


def build_backtest_run_status_response(*, run: BacktestJob) -> BacktestRunStatusResponse:
    """
    Convert persisted-run aggregate into strict public status response payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        run: Persisted run aggregate backed by unified jobs storage.
    Returns:
        BacktestRunStatusResponse: Strict public status payload using `run_id` vocabulary.
    Assumptions:
        Public runs payload hides internal reproducibility hashes exposed by legacy jobs routes.
    Raises:
        None.
    Side Effects:
        None.
    """
    error_payload: BacktestRunErrorResponse | None = None
    if run.last_error_json is not None:
        error_payload = BacktestRunErrorResponse(
            code=run.last_error_json.code,
            message=run.last_error_json.message,
            details=dict(run.last_error_json.details),
        )

    return BacktestRunStatusResponse(
        run_id=run.job_id,
        mode=run.mode,
        state=run.state,
        stage=run.stage,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        cancel_requested_at=run.cancel_requested_at,
        progress_updated_at=run.progress_updated_at,
        processed_units=run.processed_units,
        total_units=run.total_units,
        execution_mode=run.execution_mode,
        market_id=run.market_id,
        symbol=run.symbol,
        timeframe=run.timeframe,
        requested_top_n=run.requested_top_n,
        ranking_primary_metric=run.ranking_primary_metric,
        ranking_secondary_metric=run.ranking_secondary_metric,
        artifact_slot=run.artifact_pin.artifact_slot if run.artifact_pin is not None else None,
        artifact_slot_generation=(
            run.artifact_pin.artifact_slot_generation if run.artifact_pin is not None else None
        ),
        artifact_manifest_hash=(
            run.artifact_pin.artifact_manifest_hash if run.artifact_pin is not None else None
        ),
        artifact_asof_date=(
            run.artifact_pin.artifact_asof_date if run.artifact_pin is not None else None
        ),
        last_error=run.last_error,
        last_error_json=error_payload,
    )


def build_backtest_runs_list_response(
    *,
    items: tuple[BacktestJob, ...],
    next_cursor: BacktestJobListCursor | None,
) -> BacktestRunsListResponse:
    """
    Build strict public history response from deterministic repository page payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    Args:
        items: Deterministically ordered persisted runs page items.
        next_cursor: Optional keyset cursor for the next page.
    Returns:
        BacktestRunsListResponse: Strict public history payload.
    Assumptions:
        Items are ordered by `created_at DESC, job_id DESC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestRunsListResponse(
        items=[
            BacktestRunsListItemResponse(
                run_id=item.job_id,
                mode=item.mode,
                state=item.state,
                stage=item.stage,
                created_at=item.created_at,
                updated_at=item.updated_at,
                started_at=item.started_at,
                finished_at=item.finished_at,
                cancel_requested_at=item.cancel_requested_at,
                processed_units=item.processed_units,
                total_units=item.total_units,
                execution_mode=item.execution_mode,
                market_id=item.market_id,
                symbol=item.symbol,
                timeframe=item.timeframe,
                requested_top_n=item.requested_top_n,
                ranking_primary_metric=item.ranking_primary_metric,
                ranking_secondary_metric=item.ranking_secondary_metric,
                artifact_slot=(
                    item.artifact_pin.artifact_slot if item.artifact_pin is not None else None
                ),
                artifact_slot_generation=(
                    item.artifact_pin.artifact_slot_generation
                    if item.artifact_pin is not None
                    else None
                ),
                artifact_manifest_hash=(
                    item.artifact_pin.artifact_manifest_hash
                    if item.artifact_pin is not None
                    else None
                ),
                artifact_asof_date=(
                    item.artifact_pin.artifact_asof_date if item.artifact_pin is not None else None
                ),
            )
            for item in items
        ],
        next_cursor=encode_backtest_runs_cursor(cursor=next_cursor),
    )


def build_backtest_run_top_response(*, result: BacktestRunTopReadResult) -> BacktestRunTopResponse:
    """
    Build strict public summary-only `/runs/{run_id}/top` response payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    Args:
        result: Public runs top-read use-case payload.
    Returns:
        BacktestRunTopResponse: Strict public summary table payload.
    Assumptions:
        Persisted rows exclude `report/trades` bodies and expose only summary fields.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestRunTopResponse(
        run_id=result.job.job_id,
        state=result.job.state,
        execution_mode=result.job.execution_mode,
        items=[
            BacktestRunTopItemResponse(
                rank=row.rank,
                variant_key=row.variant_key,
                indicator_variant_key=row.indicator_variant_key,
                variant_index=row.variant_index,
                total_return_pct=row.total_return_pct,
                payload=dict(row.payload_json),
                summary_metrics_json=dict(row.summary_metrics_json),
                best_tp_pct=row.best_tp_pct,
                best_sl_pct=row.best_sl_pct,
            )
            for row in result.rows
        ],
    )


def encode_backtest_runs_cursor(*, cursor: BacktestJobListCursor | None) -> str | None:
    """
    Encode deterministic public runs cursor into opaque `base64url(json)` transport payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/dto/backtest_jobs.py
      - apps/api/routes/backtest_runs.py
    Args:
        cursor: Optional keyset cursor value object.
    Returns:
        str | None: Opaque cursor string without base64 padding.
    Assumptions:
        Public runs reuse the same keyset cursor transport as legacy jobs endpoints.
    Raises:
        None.
    Side Effects:
        None.
    """
    return encode_backtest_jobs_cursor(cursor=cursor)


def decode_backtest_runs_state(*, state: str | None) -> BacktestRunsStateLiteral | None:
    """
    Decode optional public runs `state` query value with blank-to-none compatibility.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/dto/backtest_jobs.py
      - apps/api/routes/backtest_runs.py
    Args:
        state: Optional raw `state` query value.
    Returns:
        BacktestRunsStateLiteral | None: Normalized state literal or `None`.
    Assumptions:
        Public runs preserve legacy blank-state compatibility for migration clients.
    Raises:
        BacktestValidationError: If non-empty state is not one of allowed literals.
    Side Effects:
        None.
    """
    try:
        decoded = decode_backtest_jobs_state(state=state)
    except BacktestValidationError as error:
        raise BacktestValidationError(
            "Invalid runs state filter",
            errors=error.errors,
        ) from error
    return cast(BacktestRunsStateLiteral | None, decoded)


def decode_backtest_runs_cursor(*, cursor: str | None) -> BacktestJobListCursor | None:
    """
    Decode optional public runs cursor from opaque `base64url(json)` transport payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/dto/backtest_jobs.py
      - apps/api/routes/backtest_runs.py
    Args:
        cursor: Optional raw cursor query value.
    Returns:
        BacktestJobListCursor | None: Decoded keyset cursor or `None`.
    Assumptions:
        Public runs preserve the legacy cursor transport format for deterministic pagination.
    Raises:
        BacktestValidationError: Deterministic cursor validation exception.
    Side Effects:
        None.
    """
    try:
        return decode_backtest_jobs_cursor(cursor=cursor)
    except BacktestValidationError as error:
        raise BacktestValidationError(
            "Invalid runs cursor",
            errors=error.errors,
        ) from error


__all__ = [
    "BacktestRunErrorResponse",
    "BacktestRunStatusResponse",
    "BacktestRunsListItemResponse",
    "BacktestRunsListResponse",
    "BacktestRunsStageLiteral",
    "BacktestRunsStateLiteral",
    "BacktestRunVariantReportPostRequest",
    "BacktestRunTopItemResponse",
    "BacktestRunTopResponse",
    "build_backtest_run_status_response",
    "build_backtest_run_top_response",
    "build_backtest_runs_list_response",
    "decode_backtest_runs_cursor",
    "decode_backtest_runs_state",
    "encode_backtest_runs_cursor",
]
