from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestJobRepository,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobMode,
    BacktestJobStage,
    BacktestJobState,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.shared_kernel.primitives import UserId

_BACKTEST_JOB_SELECT_COLUMNS = """
    job_id,
    user_id,
    mode,
    state,
    created_at,
    updated_at,
    started_at,
    finished_at,
    cancel_requested_at,
    request_json,
    request_hash,
    spec_hash,
    spec_payload_json,
    engine_params_hash,
    backtest_runtime_config_hash,
    stage,
    processed_units,
    total_units,
    progress_updated_at,
    locked_by,
    locked_at,
    lease_expires_at,
    heartbeat_at,
    attempt,
    last_error,
    last_error_json
"""


class PostgresBacktestJobRepository(BacktestJobRepository):
    """
    Explicit SQL adapter implementing Backtest job core storage repository port.

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
        Initialize repository with SQL gateway and target table name.

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
            raise ValueError("PostgresBacktestJobRepository requires gateway")
        normalized_table = jobs_table.strip()
        if not normalized_table:
            raise ValueError("PostgresBacktestJobRepository requires non-empty jobs_table")
        self._gateway = gateway
        self._jobs_table = normalized_table

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Persist new job row and return mapped immutable aggregate snapshot.

        Args:
            job: Prepared queued Backtest job aggregate.
        Returns:
            BacktestJob: Persisted immutable snapshot.
        Assumptions:
            Saved/template invariants are pre-validated by domain aggregate.
        Raises:
            BacktestStorageError: If insert fails or row cannot be mapped.
        Side Effects:
            Executes one SQL insert statement.
        """
        query = f"""
        INSERT INTO {self._jobs_table}
        (
            job_id,
            user_id,
            mode,
            state,
            created_at,
            updated_at,
            started_at,
            finished_at,
            cancel_requested_at,
            request_json,
            request_hash,
            spec_hash,
            spec_payload_json,
            engine_params_hash,
            backtest_runtime_config_hash,
            stage,
            processed_units,
            total_units,
            progress_updated_at,
            locked_by,
            locked_at,
            lease_expires_at,
            heartbeat_at,
            attempt,
            last_error,
            last_error_json
        )
        VALUES
        (
            %(job_id)s,
            %(user_id)s,
            %(mode)s,
            %(state)s,
            %(created_at)s,
            %(updated_at)s,
            %(started_at)s,
            %(finished_at)s,
            %(cancel_requested_at)s,
            %(request_json)s::jsonb,
            %(request_hash)s,
            %(spec_hash)s,
            %(spec_payload_json)s::jsonb,
            %(engine_params_hash)s,
            %(backtest_runtime_config_hash)s,
            %(stage)s,
            %(processed_units)s,
            %(total_units)s,
            %(progress_updated_at)s,
            %(locked_by)s,
            %(locked_at)s,
            %(lease_expires_at)s,
            %(heartbeat_at)s,
            %(attempt)s,
            %(last_error)s,
            %(last_error_json)s::jsonb
        )
        RETURNING
            {_BACKTEST_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job.job_id),
                "user_id": str(job.user_id),
                "mode": job.mode,
                "state": job.state,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "cancel_requested_at": job.cancel_requested_at,
                "request_json": _json_dumps(payload=job.request_json),
                "request_hash": job.request_hash,
                "spec_hash": job.spec_hash,
                "spec_payload_json": _json_dumps(payload=job.spec_payload_json)
                if job.spec_payload_json is not None
                else None,
                "engine_params_hash": job.engine_params_hash,
                "backtest_runtime_config_hash": job.backtest_runtime_config_hash,
                "stage": job.stage,
                "processed_units": job.processed_units,
                "total_units": job.total_units,
                "progress_updated_at": job.progress_updated_at,
                "locked_by": job.locked_by,
                "locked_at": job.locked_at,
                "lease_expires_at": job.lease_expires_at,
                "heartbeat_at": job.heartbeat_at,
                "attempt": job.attempt,
                "last_error": job.last_error,
                "last_error_json": _json_dumps(payload=job.last_error_json.to_mapping())
                if job.last_error_json is not None
                else None,
            },
        )
        if row is None:
            raise BacktestStorageError("PostgresBacktestJobRepository.create returned no row")
        return _map_job_row(row=row)

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Load one job snapshot by id with optional owner filter.

        Args:
            job_id: Job identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Job snapshot or `None`.
        Assumptions:
            Owner checks are explicit and deterministic in higher layers.
        Raises:
            BacktestStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        owner_filter = ""
        parameters: dict[str, Any] = {"job_id": str(job_id)}
        if user_id is not None:
            owner_filter = "AND user_id = %(user_id)s"
            parameters["user_id"] = str(user_id)

        query = f"""
        SELECT
            {_BACKTEST_JOB_SELECT_COLUMNS}
        FROM {self._jobs_table}
        WHERE job_id = %(job_id)s
          {owner_filter}
        """
        row = self._gateway.fetch_one(query=query, parameters=parameters)
        if row is None:
            return None
        return _map_job_row(row=row)

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        """
        List owner jobs by deterministic keyset ordering and optional state filter.

        Args:
            query: User list query payload.
        Returns:
            BacktestJobListPage: Deterministic keyset page payload.
        Assumptions:
            SQL order is fixed to `created_at DESC, job_id DESC`.
        Raises:
            BacktestStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        limit_with_probe = query.limit + 1
        cursor_created_at = query.cursor.created_at if query.cursor is not None else None
        cursor_job_id = str(query.cursor.job_id) if query.cursor is not None else None

        sql = f"""
        SELECT
            {_BACKTEST_JOB_SELECT_COLUMNS}
        FROM {self._jobs_table}
        WHERE user_id = %(user_id)s
          AND (%(state)s::text IS NULL OR state = %(state)s::text)
          AND (
            %(cursor_created_at)s::timestamptz IS NULL
            OR (created_at, job_id) < (
              %(cursor_created_at)s::timestamptz,
              %(cursor_job_id)s::uuid
            )
          )
        ORDER BY created_at DESC, job_id DESC
        LIMIT %(limit)s
        """
        rows = self._gateway.fetch_all(
            query=sql,
            parameters={
                "user_id": str(query.user_id),
                "state": query.state,
                "cursor_created_at": cursor_created_at,
                "cursor_job_id": cursor_job_id,
                "limit": limit_with_probe,
            },
        )
        mapped_jobs = tuple(_map_job_row(row=row) for row in rows)
        if len(mapped_jobs) <= query.limit:
            return BacktestJobListPage(items=mapped_jobs, next_cursor=None)

        page_items = mapped_jobs[: query.limit]
        last_item = page_items[-1]
        next_cursor = BacktestJobListCursor(
            created_at=last_item.created_at,
            job_id=last_item.job_id,
        )
        return BacktestJobListPage(items=page_items, next_cursor=next_cursor)

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        """
        Request cancel for owner job (`queued` immediate cancel, `running` mark request).

        Args:
            job_id: Job identifier.
            user_id: Job owner identifier.
            cancel_requested_at: Cancel request timestamp in UTC.
        Returns:
            BacktestJob | None: Updated snapshot or `None` when job is missing.
        Assumptions:
            Cancel operation is idempotent for already-terminal jobs.
        Raises:
            BacktestStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL update and optional fallback select.
        """
        update_sql = f"""
        UPDATE {self._jobs_table}
        SET
            state = CASE
                WHEN state = 'queued' THEN 'cancelled'
                ELSE state
            END,
            finished_at = CASE
                WHEN state = 'queued' THEN %(cancel_requested_at)s
                ELSE finished_at
            END,
            cancel_requested_at = %(cancel_requested_at)s,
            updated_at = %(cancel_requested_at)s
        WHERE job_id = %(job_id)s
          AND user_id = %(user_id)s
          AND state IN ('queued', 'running')
        RETURNING
            {_BACKTEST_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=update_sql,
            parameters={
                "job_id": str(job_id),
                "user_id": str(user_id),
                "cancel_requested_at": cancel_requested_at,
            },
        )
        if row is not None:
            return _map_job_row(row=row)

        return self.get(job_id=job_id, user_id=user_id)

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Count active owner jobs (`queued + running`) for deterministic quota checks.

        Args:
            user_id: Owner identifier.
        Returns:
            int: Active jobs count.
        Assumptions:
            Active state set is fixed by Backtest Jobs v1 contract.
        Raises:
            BacktestStorageError: If count row is missing or invalid.
        Side Effects:
            Executes one SQL aggregate select.
        """
        sql = f"""
        SELECT
            COUNT(*) AS active_total
        FROM {self._jobs_table}
        WHERE user_id = %(user_id)s
          AND state IN ('queued', 'running')
        """
        row = self._gateway.fetch_one(query=sql, parameters={"user_id": str(user_id)})
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.count_active_for_user returned no row"
            )
        try:
            return int(row["active_total"])
        except Exception as error:  # noqa: BLE001
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.count_active_for_user invalid count row"
            ) from error


def _map_job_row(*, row: Mapping[str, Any]) -> BacktestJob:
    """
    Map SQL row payload into immutable `BacktestJob` aggregate.

    Args:
        row: SQL row mapping.
    Returns:
        BacktestJob: Mapped immutable job aggregate.
    Assumptions:
        Row schema follows Backtest jobs v1 storage contract.
    Raises:
        BacktestStorageError: If one field cannot be mapped.
    Side Effects:
        None.
    """
    try:
        last_error_json_payload = _parse_json_object(
            value=row.get("last_error_json"),
            field_name="last_error_json",
            required=False,
        )
        last_error_payload = None
        if last_error_json_payload is not None:
            last_error_payload = BacktestJobErrorPayload(
                code=str(last_error_json_payload.get("code", "")),
                message=str(last_error_json_payload.get("message", "")),
                details=cast(
                    Mapping[str, Any],
                    last_error_json_payload.get("details")
                    if isinstance(last_error_json_payload.get("details"), Mapping)
                    else {},
                ),
            )

        request_payload = _parse_json_object(
            value=row.get("request_json"),
            field_name="request_json",
            required=True,
        )
        if request_payload is None:
            raise BacktestStorageError("backtest_jobs.request_json must be JSON object")

        spec_payload = _parse_json_object(
            value=row.get("spec_payload_json"),
            field_name="spec_payload_json",
            required=False,
        )
        return BacktestJob(
            job_id=UUID(str(row["job_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            mode=_parse_job_mode(value=row["mode"]),
            state=_parse_job_state(value=row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            cancel_requested_at=row.get("cancel_requested_at"),
            request_json=request_payload,
            request_hash=str(row["request_hash"]),
            spec_hash=str(row["spec_hash"]).strip() if row.get("spec_hash") is not None else None,
            spec_payload_json=spec_payload,
            engine_params_hash=str(row["engine_params_hash"]),
            backtest_runtime_config_hash=str(row["backtest_runtime_config_hash"]),
            stage=_parse_job_stage(value=row["stage"]),
            processed_units=int(row["processed_units"]),
            total_units=int(row["total_units"]),
            progress_updated_at=row.get("progress_updated_at"),
            locked_by=str(row["locked_by"]) if row.get("locked_by") is not None else None,
            locked_at=row.get("locked_at"),
            lease_expires_at=row.get("lease_expires_at"),
            heartbeat_at=row.get("heartbeat_at"),
            attempt=int(row["attempt"]),
            last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
            last_error_json=last_error_payload,
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError("PostgresBacktestJobRepository cannot map job row") from error


def _parse_json_object(
    *,
    value: Any,
    field_name: str,
    required: bool,
) -> Mapping[str, Any] | None:
    """
    Parse JSON object column value from gateway row into mapping payload.

    Args:
        value: Raw gateway value.
        field_name: Column name for deterministic error messages.
        required: Whether object value is mandatory.
    Returns:
        Mapping[str, Any] | None: Parsed mapping or `None` when optional and absent.
    Assumptions:
        Gateway may return dict, bytes, memoryview, or JSON text.
    Raises:
        BacktestStorageError: If payload is missing or not JSON object.
    Side Effects:
        None.
    """
    if value is None:
        if required:
            raise BacktestStorageError(f"backtest_jobs.{field_name} must be JSON object")
        return None

    if isinstance(value, Mapping):
        return dict(value)

    raw_value = value
    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes().decode("utf-8")
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = bytes(raw_value).decode("utf-8")

    if isinstance(raw_value, str):
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError as error:
            raise BacktestStorageError(f"backtest_jobs.{field_name} has invalid JSON") from error
        if not isinstance(decoded, Mapping):
            raise BacktestStorageError(f"backtest_jobs.{field_name} must be JSON object")
        return dict(decoded)

    raise BacktestStorageError(
        f"backtest_jobs.{field_name} has unsupported type {type(value).__name__}"
    )


def _parse_job_mode(*, value: Any) -> BacktestJobMode:
    """
    Parse and validate storage mode literal into `BacktestJobMode` type.

    Args:
        value: Raw storage mode value.
    Returns:
        BacktestJobMode: Typed mode literal.
    Assumptions:
        Storage mode values are constrained by migration check literal set.
    Raises:
        BacktestStorageError: If value is unknown.
    Side Effects:
        None.
    """
    normalized = str(value).strip().lower()
    if normalized not in {"saved", "template"}:
        raise BacktestStorageError(f"Unexpected backtest job mode value: {normalized!r}")
    return cast(BacktestJobMode, normalized)


def _parse_job_state(*, value: Any) -> BacktestJobState:
    """
    Parse and validate storage state literal into `BacktestJobState` type.

    Args:
        value: Raw storage state value.
    Returns:
        BacktestJobState: Typed state literal.
    Assumptions:
        Storage state values are constrained by migration check literal set.
    Raises:
        BacktestStorageError: If value is unknown.
    Side Effects:
        None.
    """
    normalized = str(value).strip().lower()
    if normalized not in {"queued", "running", "succeeded", "failed", "cancelled"}:
        raise BacktestStorageError(f"Unexpected backtest job state value: {normalized!r}")
    return cast(BacktestJobState, normalized)


def _parse_job_stage(*, value: Any) -> BacktestJobStage:
    """
    Parse and validate storage stage literal into `BacktestJobStage` type.

    Args:
        value: Raw storage stage value.
    Returns:
        BacktestJobStage: Typed stage literal.
    Assumptions:
        Storage stage values are constrained by migration check literal set.
    Raises:
        BacktestStorageError: If value is unknown.
    Side Effects:
        None.
    """
    normalized = str(value).strip().lower()
    if normalized not in {"stage_a", "stage_b", "finalizing"}:
        raise BacktestStorageError(f"Unexpected backtest job stage value: {normalized!r}")
    return cast(BacktestJobStage, normalized)


def _json_dumps(*, payload: Mapping[str, Any] | None) -> str | None:
    """
    Serialize optional mapping payload into canonical JSON string.

    Args:
        payload: Optional mapping payload.
    Returns:
        str | None: Canonical JSON text or `None`.
    Assumptions:
        JSON canonicalization uses sorted keys and compact separators.
    Raises:
        TypeError: If payload is not JSON-serializable.
    Side Effects:
        None.
    """
    if payload is None:
        return None
    # json.dumps does not support MappingProxyType directly.
    # Domain aggregates intentionally store JSON payloads as immutable mappings.
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = ["PostgresBacktestJobRepository"]
