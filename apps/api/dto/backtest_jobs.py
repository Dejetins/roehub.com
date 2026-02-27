"""
Pydantic models and deterministic cursor/mapper helpers for Backtest Jobs API v1.

Docs:
  - docs/architecture/backtest/backtest-jobs-api-v1.md
  - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any, Literal, Mapping, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from trading.contexts.backtest.application.use_cases import BacktestJobTopReadResult
from trading.contexts.backtest.domain.entities import BacktestJob
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor

BacktestJobsStateLiteral = Literal["queued", "running", "succeeded", "failed", "cancelled"]
BacktestJobsStageLiteral = Literal["stage_a", "stage_b", "finalizing"]
_BACKTEST_JOBS_STATE_VALUES: tuple[BacktestJobsStateLiteral, ...] = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)


class BacktestJobErrorResponse(BaseModel):
    """
    API response model for persisted Roehub-like failed-job payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any]


class BacktestJobStatusResponse(BaseModel):
    """
    API response model for one Backtest job status/progress snapshot.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    mode: Literal["saved", "template"]
    state: BacktestJobsStateLiteral
    stage: BacktestJobsStageLiteral
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    progress_updated_at: datetime | None = None
    processed_units: int
    total_units: int
    request_hash: str
    engine_params_hash: str
    backtest_runtime_config_hash: str
    spec_hash: str | None = None
    last_error: str | None = None
    last_error_json: BacktestJobErrorResponse | None = None


class BacktestJobsListItemResponse(BaseModel):
    """
    API response model for one Backtest jobs list item.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    mode: Literal["saved", "template"]
    state: BacktestJobsStateLiteral
    stage: BacktestJobsStageLiteral
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    processed_units: int
    total_units: int


class BacktestJobsListResponse(BaseModel):
    """
    API response model for deterministic keyset-paginated Backtest jobs list.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    items: list[BacktestJobsListItemResponse]
    next_cursor: str | None


class BacktestJobTopItemResponse(BaseModel):
    """
    API response model for one persisted top-variant row in jobs `/top` endpoint.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    rank: int
    variant_key: str
    indicator_variant_key: str
    variant_index: int
    total_return_pct: float
    payload: dict[str, Any]
    report_table_md: str | None = None
    trades: list[dict[str, Any]] | None = None


class BacktestJobTopResponse(BaseModel):
    """
    API response model for Backtest jobs `/top` payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    state: BacktestJobsStateLiteral
    items: list[BacktestJobTopItemResponse]



def build_backtest_job_status_response(*, job: BacktestJob) -> BacktestJobStatusResponse:
    """
    Convert immutable domain job snapshot into strict status API response model.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - apps/api/routes/backtest_jobs.py

    Args:
        job: Domain job snapshot.
    Returns:
        BacktestJobStatusResponse: Strict status response payload.
    Assumptions:
        Failed jobs persist both `last_error` and `last_error_json`.
    Raises:
        ValueError: If state-specific failure payload invariant is violated.
    Side Effects:
        None.
    """
    error_payload: BacktestJobErrorResponse | None = None
    if job.last_error_json is not None:
        error_payload = BacktestJobErrorResponse(
            code=job.last_error_json.code,
            message=job.last_error_json.message,
            details=dict(job.last_error_json.details),
        )

    return BacktestJobStatusResponse(
        job_id=job.job_id,
        mode=job.mode,
        state=job.state,
        stage=job.stage,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        cancel_requested_at=job.cancel_requested_at,
        progress_updated_at=job.progress_updated_at,
        processed_units=job.processed_units,
        total_units=job.total_units,
        request_hash=job.request_hash,
        engine_params_hash=job.engine_params_hash,
        backtest_runtime_config_hash=job.backtest_runtime_config_hash,
        spec_hash=job.spec_hash,
        last_error=job.last_error,
        last_error_json=error_payload,
    )



def build_backtest_jobs_list_response(
    *,
    items: tuple[BacktestJob, ...],
    next_cursor: BacktestJobListCursor | None,
) -> BacktestJobsListResponse:
    """
    Build strict list API response from deterministic repository page payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py

    Args:
        items: Deterministically ordered jobs page items.
        next_cursor: Optional keyset cursor for next page.
    Returns:
        BacktestJobsListResponse: Strict list response payload.
    Assumptions:
        Items are ordered by `created_at DESC, job_id DESC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestJobsListResponse(
        items=[
            BacktestJobsListItemResponse(
                job_id=item.job_id,
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
            )
            for item in items
        ],
        next_cursor=encode_backtest_jobs_cursor(cursor=next_cursor),
    )



def build_backtest_job_top_response(*, result: BacktestJobTopReadResult) -> BacktestJobTopResponse:
    """
    Build `/top` response payload with EPIC-11 state-dependent details visibility policy.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py

    Args:
        result: Top-read use-case result payload.
    Returns:
        BacktestJobTopResponse: Strict `/top` response.
    Assumptions:
        `report_table_md` and `trades` are returned only for `succeeded` jobs.
    Raises:
        None.
    Side Effects:
        None.
    """
    include_details = result.job.state == "succeeded"
    return BacktestJobTopResponse(
        job_id=result.job.job_id,
        state=result.job.state,
        items=[
            BacktestJobTopItemResponse(
                rank=row.rank,
                variant_key=row.variant_key,
                indicator_variant_key=row.indicator_variant_key,
                variant_index=row.variant_index,
                total_return_pct=row.total_return_pct,
                payload=dict(row.payload_json),
                report_table_md=row.report_table_md if include_details else None,
                trades=[dict(item) for item in row.trades_json]
                if include_details and row.trades_json is not None
                else None,
            )
            for row in result.rows
        ],
    )



def encode_backtest_jobs_cursor(*, cursor: BacktestJobListCursor | None) -> str | None:
    """
    Encode deterministic keyset cursor into opaque `base64url(json)` transport payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
      - apps/api/routes/backtest_jobs.py

    Args:
        cursor: Optional keyset cursor value object.
    Returns:
        str | None: Opaque cursor string without base64 padding.
    Assumptions:
        Canonical JSON uses sorted keys and compact separators.
    Raises:
        None.
    Side Effects:
        None.
    """
    if cursor is None:
        return None

    payload = cursor.to_payload()
    canonical_json = json.dumps(
        _normalize_json_value(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return base64.urlsafe_b64encode(canonical_json.encode("utf-8")).decode("ascii").rstrip("=")


def decode_backtest_jobs_state(*, state: str | None) -> BacktestJobsStateLiteral | None:
    """
    Decode optional jobs list `state` query value with blank-to-none compatibility behavior.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/routes/backtest_jobs.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

    Args:
        state: Optional raw `state` query value.
    Returns:
        BacktestJobsStateLiteral | None: Normalized state literal or `None`.
    Assumptions:
        Empty `state` query value from legacy clients should be treated as missing filter.
    Raises:
        BacktestValidationError: If non-empty state value is not one of allowed literals.
    Side Effects:
        None.
    """
    if state is None:
        return None

    normalized_state = state.strip().lower()
    if not normalized_state:
        return None

    if normalized_state not in _BACKTEST_JOBS_STATE_VALUES:
        allowed_values = ", ".join(_BACKTEST_JOBS_STATE_VALUES)
        raise BacktestValidationError(
            "Invalid jobs state filter",
            errors=(
                {
                    "path": "query.state",
                    "code": "invalid_value",
                    "message": f"state must be one of: {allowed_values}",
                },
            ),
        )

    return cast(BacktestJobsStateLiteral, normalized_state)


def decode_backtest_jobs_cursor(*, cursor: str | None) -> BacktestJobListCursor | None:
    """
    Decode opaque `base64url(json)` cursor transport payload into cursor value object.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
      - apps/api/routes/backtest_jobs.py

    Args:
        cursor: Optional opaque cursor query value.
    Returns:
        BacktestJobListCursor | None: Decoded cursor or `None` when not provided.
    Assumptions:
        Payload is produced by `encode_backtest_jobs_cursor`; blank cursor is treated as missing.
    Raises:
        BacktestValidationError: If cursor payload is malformed or cannot be parsed.
    Side Effects:
        None.
    """
    if cursor is None:
        return None

    normalized_cursor = cursor.strip()
    if not normalized_cursor:
        return None

    try:
        padding = "=" * (-len(normalized_cursor) % 4)
        raw_payload = base64.urlsafe_b64decode(normalized_cursor + padding)
        parsed_payload = json.loads(raw_payload.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise _invalid_cursor_error() from None

    if not isinstance(parsed_payload, Mapping):
        raise _invalid_cursor_error()

    try:
        return BacktestJobListCursor.from_payload(payload=dict(parsed_payload))
    except BacktestValidationError as error:
        raise _invalid_cursor_error(reason=str(error)) from error



def _invalid_cursor_error(*, reason: str | None = None) -> BacktestValidationError:
    """
    Build canonical validation error for malformed Backtest jobs list cursor payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - apps/api/routes/backtest_jobs.py
      - src/trading/contexts/backtest/domain/errors/backtest_errors.py

    Args:
        reason: Optional low-level parse reason.
    Returns:
        BacktestValidationError: Deterministic cursor validation exception.
    Assumptions:
        Cursor parsing errors must map to `validation_error` with stable details order.
    Raises:
        None.
    Side Effects:
        None.
    """
    message = "Invalid jobs cursor"
    if reason is not None and reason.strip():
        message = f"Invalid jobs cursor: {reason.strip()}"
    return BacktestValidationError(
        message,
        errors=(
            {
                "path": "query.cursor",
                "code": "invalid_cursor",
                "message": "cursor must be base64url(json)",
            },
        ),
    )



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary payload node into deterministic JSON-compatible structure.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py

    Args:
        value: Raw JSON-like node.
    Returns:
        Any: Deterministic mapping/list/scalar value.
    Assumptions:
        Mapping keys are stringified and sorted recursively.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda key: str(key)):
            normalized[str(raw_key)] = _normalize_json_value(value=value[raw_key])
        return normalized

    if isinstance(value, list | tuple):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


__all__ = [
    "BacktestJobErrorResponse",
    "BacktestJobStatusResponse",
    "BacktestJobsListItemResponse",
    "BacktestJobsListResponse",
    "BacktestJobsStateLiteral",
    "BacktestJobTopItemResponse",
    "BacktestJobTopResponse",
    "build_backtest_job_status_response",
    "build_backtest_job_top_response",
    "build_backtest_jobs_list_response",
    "decode_backtest_jobs_state",
    "decode_backtest_jobs_cursor",
    "encode_backtest_jobs_cursor",
]
