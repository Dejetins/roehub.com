from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import BacktestJobLeaseRepository
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStage,
    BacktestJobState,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError

from .backtest_job_repository import (
    _BACKTEST_JOB_SELECT_COLUMNS,
    _json_dumps,
    _map_job_row,
)


class PostgresBacktestJobLeaseRepository(BacktestJobLeaseRepository):
    """
    Explicit SQL adapter for Backtest jobs claim/lease/progress/finish worker operations.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def __init__(
        self,
        *,
        gateway: BacktestPostgresGateway,
        jobs_table: str = "backtest_jobs",
    ) -> None:
        """
        Initialize lease repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            jobs_table: Backtest jobs table name.
        Returns:
            None.
        Assumptions:
            Table schema follows Backtest jobs v1 migration contract.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresBacktestJobLeaseRepository requires gateway")
        normalized_table = jobs_table.strip()
        if not normalized_table:
            raise ValueError("PostgresBacktestJobLeaseRepository requires non-empty jobs_table")
        self._gateway = gateway
        self._jobs_table = normalized_table

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Claim one queued/reclaim candidate job using FIFO order and SKIP LOCKED semantics.

        Args:
            now: Claim timestamp in UTC.
            locked_by: Worker owner identity.
            lease_seconds: Lease duration in seconds.
        Returns:
            BacktestJob | None: Claimed running job snapshot or `None` when no claimable rows.
        Assumptions:
            FIFO order is `created_at ASC, job_id ASC` for queued jobs.
        Raises:
            BacktestStorageError: If storage update or row mapping fails.
        Side Effects:
            Executes one SQL CTE statement with `FOR UPDATE SKIP LOCKED`.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        validated_lease_seconds = _validate_lease_seconds(lease_seconds=lease_seconds)
        lease_expires_at = now + timedelta(seconds=validated_lease_seconds)

        query = f"""
        WITH queued_candidate AS (
            SELECT
                job_id
            FROM {self._jobs_table}
            WHERE state = 'queued'
            ORDER BY created_at ASC, job_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ),
        reclaim_candidate AS (
            SELECT
                job_id
            FROM {self._jobs_table}
            WHERE state = 'running'
              AND lease_expires_at <= %(now)s
            ORDER BY lease_expires_at ASC, created_at ASC, job_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ),
        candidate AS (
            SELECT job_id, 1 AS priority FROM queued_candidate
            UNION ALL
            SELECT job_id, 2 AS priority FROM reclaim_candidate
            ORDER BY priority ASC
            LIMIT 1
        ),
        claimed AS (
            UPDATE {self._jobs_table} AS jobs
            SET
                state = 'running',
                stage = CASE
                    WHEN jobs.state = 'queued' THEN 'stage_a'
                    ELSE jobs.stage
                END,
                started_at = CASE
                    WHEN jobs.started_at IS NULL THEN %(now)s
                    ELSE jobs.started_at
                END,
                updated_at = %(now)s,
                locked_by = %(locked_by)s,
                locked_at = %(now)s,
                lease_expires_at = %(lease_expires_at)s,
                heartbeat_at = %(now)s,
                attempt = jobs.attempt + 1
            FROM candidate
            WHERE jobs.job_id = candidate.job_id
            RETURNING
                {_BACKTEST_JOB_SELECT_COLUMNS}
        )
        SELECT
            {_BACKTEST_JOB_SELECT_COLUMNS}
        FROM claimed
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "now": now,
                "locked_by": normalized_owner,
                "lease_expires_at": lease_expires_at,
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Extend running lease by owner with conditional active-lease predicate.

        Args:
            job_id: Job identifier.
            now: Heartbeat timestamp in UTC.
            locked_by: Worker owner identity.
            lease_seconds: Lease extension duration in seconds.
        Returns:
            BacktestJob | None: Updated job snapshot or `None` when lease is lost.
        Assumptions:
            Conditional write uses `(job_id, locked_by, lease_expires_at > now)` predicate.
        Raises:
            BacktestStorageError: If storage update or row mapping fails.
        Side Effects:
            Executes one SQL update statement.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        validated_lease_seconds = _validate_lease_seconds(lease_seconds=lease_seconds)
        lease_expires_at = now + timedelta(seconds=validated_lease_seconds)

        query = f"""
        UPDATE {self._jobs_table}
        SET
            updated_at = %(now)s,
            heartbeat_at = %(now)s,
            lease_expires_at = %(lease_expires_at)s
        WHERE job_id = %(job_id)s
          AND state = 'running'
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_BACKTEST_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": normalized_owner,
                "lease_expires_at": lease_expires_at,
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: BacktestJobStage,
        processed_units: int,
        total_units: int,
    ) -> BacktestJob | None:
        """
        Persist stage/progress counters guarded by active lease-owner predicate.

        Args:
            job_id: Job identifier.
            now: Progress timestamp in UTC.
            locked_by: Worker owner identity.
            stage: Stage literal (`stage_a|stage_b|finalizing`).
            processed_units: Processed units counter.
            total_units: Total units counter.
        Returns:
            BacktestJob | None: Updated job snapshot or `None` when lease is lost.
        Assumptions:
            Progress writes are accepted only while job is running under active lease.
        Raises:
            BacktestStorageError: If storage update or row mapping fails.
        Side Effects:
            Executes one SQL update statement.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        _validate_progress_counts(processed_units=processed_units, total_units=total_units)

        query = f"""
        UPDATE {self._jobs_table}
        SET
            updated_at = %(now)s,
            stage = %(stage)s,
            processed_units = %(processed_units)s,
            total_units = %(total_units)s,
            progress_updated_at = %(now)s
        WHERE job_id = %(job_id)s
          AND state = 'running'
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_BACKTEST_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": normalized_owner,
                "stage": stage,
                "processed_units": processed_units,
                "total_units": total_units,
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        """
        Finish running job under active lease owner conditional write guard.

        Args:
            job_id: Job identifier.
            now: Finish timestamp in UTC.
            locked_by: Worker owner identity.
            next_state: Terminal state literal (`succeeded|failed|cancelled`).
            last_error: Failure summary text for `failed` state.
            last_error_json: RoehubError-like payload for `failed` state.
        Returns:
            BacktestJob | None: Updated terminal snapshot or `None` when lease is lost.
        Assumptions:
            Method is used only for running jobs; `queued -> failed` is not possible here.
        Raises:
            BacktestStorageError: If terminal payload invariants or SQL mapping fail.
        Side Effects:
            Executes one SQL update statement.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        normalized_state = next_state.strip().lower()
        if normalized_state not in {"succeeded", "failed", "cancelled"}:
            raise BacktestStorageError(
                "PostgresBacktestJobLeaseRepository.finish requires terminal next_state"
            )

        normalized_last_error: str | None = None
        normalized_error_json: Mapping[str, Any] | None = None
        if normalized_state == "failed":
            if last_error is None or not last_error.strip():
                raise BacktestStorageError(
                    "PostgresBacktestJobLeaseRepository.finish failed state requires last_error"
                )
            if last_error_json is None:
                raise BacktestStorageError(
                    "PostgresBacktestJobLeaseRepository.finish failed state requires "
                    "last_error_json"
                )
            normalized_last_error = last_error.strip()
            normalized_error_json = last_error_json.to_mapping()

        query = f"""
        UPDATE {self._jobs_table}
        SET
            state = %(next_state)s,
            stage = CASE
                WHEN %(next_state)s = 'succeeded' THEN 'finalizing'
                ELSE stage
            END,
            updated_at = %(now)s,
            finished_at = %(now)s,
            locked_by = NULL,
            locked_at = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            last_error = CASE
                WHEN %(next_state)s = 'failed' THEN %(last_error)s
                ELSE NULL
            END,
            last_error_json = CASE
                WHEN %(next_state)s = 'failed' THEN %(last_error_json)s::jsonb
                ELSE NULL
            END
        WHERE job_id = %(job_id)s
          AND state = 'running'
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_BACKTEST_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": normalized_owner,
                "next_state": normalized_state,
                "last_error": normalized_last_error,
                "last_error_json": _json_dumps(payload=normalized_error_json)
                if normalized_error_json is not None
                else None,
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)



def _normalize_locked_by(*, value: str) -> str:
    """
    Validate and normalize worker lease owner literal.

    Args:
        value: Raw worker owner literal.
    Returns:
        str: Trimmed non-empty owner literal.
    Assumptions:
        Owner literal format follows `<hostname>-<pid>` style contract.
    Raises:
        BacktestStorageError: If owner literal is blank.
    Side Effects:
        None.
    """
    normalized = value.strip()
    if not normalized:
        raise BacktestStorageError("locked_by must be non-empty")
    return normalized



def _validate_lease_seconds(*, lease_seconds: int) -> int:
    """
    Validate positive lease TTL value in seconds.

    Args:
        lease_seconds: Lease duration in seconds.
    Returns:
        int: Lease duration value.
    Assumptions:
        Lease TTL must be strictly positive.
    Raises:
        BacktestStorageError: If value is invalid.
    Side Effects:
        None.
    """
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int):
        raise BacktestStorageError("lease_seconds must be integer")
    if lease_seconds <= 0:
        raise BacktestStorageError("lease_seconds must be > 0")
    return lease_seconds



def _validate_progress_counts(*, processed_units: int, total_units: int) -> None:
    """
    Validate progress counters before persisting lease-guarded update.

    Args:
        processed_units: Processed units counter.
        total_units: Total units counter.
    Returns:
        None.
    Assumptions:
        Both counters are non-negative and processed cannot exceed total when total > 0.
    Raises:
        BacktestStorageError: If counters are invalid.
    Side Effects:
        None.
    """
    if isinstance(processed_units, bool) or not isinstance(processed_units, int):
        raise BacktestStorageError("processed_units must be integer")
    if isinstance(total_units, bool) or not isinstance(total_units, int):
        raise BacktestStorageError("total_units must be integer")
    if processed_units < 0:
        raise BacktestStorageError("processed_units must be >= 0")
    if total_units < 0:
        raise BacktestStorageError("total_units must be >= 0")
    if total_units > 0 and processed_units > total_units:
        raise BacktestStorageError("processed_units cannot exceed total_units")


__all__ = ["PostgresBacktestJobLeaseRepository"]
