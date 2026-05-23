from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesMaterializationRepository,
    BacktestLazyTradesMaterializationRequest,
    BacktestLazyTradesMaterializationStatus,
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.shared_kernel.primitives import UserId

_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS = """
    task_id,
    owner_user_id,
    job_id,
    public_variant_key,
    variant_hash,
    request_hash,
    engine_params_hash,
    artifact_manifest_hash,
    cache_key,
    status,
    priority_class,
    created_at,
    updated_at,
    started_at,
    finished_at,
    locked_by,
    locked_at,
    lease_expires_at,
    heartbeat_at,
    attempt,
    last_error,
    last_error_json,
    cache_status,
    cache_path,
    ttl_seconds
"""


class PostgresBacktestLazyTradesMaterializationRepository(
    BacktestLazyTradesMaterializationRepository
):
    def __init__(
        self,
        *,
        gateway: BacktestPostgresGateway,
        table: str = "backtest_lazy_trades_materializations",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "PostgresBacktestLazyTradesMaterializationRepository requires gateway"
            )
        normalized_table = table.strip()
        if not normalized_table:
            raise ValueError(
                "PostgresBacktestLazyTradesMaterializationRepository requires non-empty table"
            )
        self._gateway = gateway
        self._table = normalized_table

    def find_by_identity(
        self,
        *,
        owner_user_id: UserId,
        job_id: UUID,
        public_variant_key: str,
        cache_key: str,
    ) -> BacktestLazyTradesMaterializationTask | None:
        query = f"""
        SELECT
            {_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS}
        FROM {self._table}
        WHERE owner_user_id = %(owner_user_id)s
          AND job_id = %(job_id)s
          AND public_variant_key = %(public_variant_key)s
          AND cache_key = %(cache_key)s
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "job_id": str(job_id),
                "public_variant_key": public_variant_key,
                "cache_key": cache_key,
            },
        )
        if row is None:
            return None
        return _map_materialization_row(row=row)

    def request_materialization(
        self,
        *,
        request: BacktestLazyTradesMaterializationRequest,
    ) -> BacktestLazyTradesMaterializationTask:
        task_id = uuid4()
        query = f"""
        INSERT INTO {self._table}
        (
            task_id,
            owner_user_id,
            job_id,
            public_variant_key,
            variant_hash,
            request_hash,
            engine_params_hash,
            artifact_manifest_hash,
            cache_key,
            status,
            priority_class,
            created_at,
            updated_at,
            started_at,
            finished_at,
            locked_by,
            locked_at,
            lease_expires_at,
            heartbeat_at,
            attempt,
            last_error,
            last_error_json,
            cache_status,
            cache_path,
            ttl_seconds
        )
        VALUES
        (
            %(task_id)s,
            %(owner_user_id)s,
            %(job_id)s,
            %(public_variant_key)s,
            %(variant_hash)s,
            %(request_hash)s,
            %(engine_params_hash)s,
            %(artifact_manifest_hash)s,
            %(cache_key)s,
            'queued',
            %(priority_class)s,
            %(requested_at)s,
            %(requested_at)s,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            0,
            NULL,
            NULL::jsonb,
            %(cache_status)s,
            NULL,
            %(ttl_seconds)s
        )
        ON CONFLICT (owner_user_id, job_id, public_variant_key, cache_key)
        DO UPDATE SET
            status = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN 'queued'
                ELSE {self._table}.status
            END,
            priority_class = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN EXCLUDED.priority_class
                ELSE {self._table}.priority_class
            END,
            updated_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN EXCLUDED.updated_at
                ELSE {self._table}.updated_at
            END,
            started_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.started_at
            END,
            finished_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.finished_at
            END,
            locked_by = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.locked_by
            END,
            locked_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.locked_at
            END,
            lease_expires_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.lease_expires_at
            END,
            heartbeat_at = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.heartbeat_at
            END,
            last_error = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.last_error
            END,
            last_error_json = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.last_error_json
            END,
            cache_status = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN EXCLUDED.cache_status
                ELSE {self._table}.cache_status
            END,
            cache_path = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN NULL
                ELSE {self._table}.cache_path
            END,
            ttl_seconds = CASE
                WHEN {self._table}.status IN ('completed', 'failed', 'cancelled')
                    THEN EXCLUDED.ttl_seconds
                ELSE {self._table}.ttl_seconds
            END
        RETURNING
            {_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "task_id": str(task_id),
                "owner_user_id": str(request.owner_user_id),
                "job_id": str(request.job_id),
                "public_variant_key": request.public_variant_key,
                "variant_hash": request.variant_hash,
                "request_hash": request.request_hash,
                "engine_params_hash": request.engine_params_hash,
                "artifact_manifest_hash": request.artifact_manifest_hash,
                "cache_key": request.cache_key,
                "priority_class": request.priority_class,
                "requested_at": request.requested_at,
                "cache_status": request.cache_status,
                "ttl_seconds": request.ttl_seconds,
            },
        )
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestLazyTradesMaterializationRepository returned no row"
            )
        return _map_materialization_row(row=row)

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        query = f"""
        SELECT
            COUNT(*) AS active_total
        FROM {self._table}
        WHERE owner_user_id = %(owner_user_id)s
          AND status IN ('queued', 'running')
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        return _count_from_row(row=row, field_name="active_total")

    def count_created_for_user_since(
        self,
        *,
        owner_user_id: UserId,
        created_after: datetime,
    ) -> int:
        query = f"""
        SELECT
            COUNT(*) AS created_total
        FROM {self._table}
        WHERE owner_user_id = %(owner_user_id)s
          AND created_at >= %(created_after)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "created_after": created_after,
            },
        )
        return _count_from_row(row=row, field_name="created_total")

    def count_active_global(self) -> int:
        query = f"""
        SELECT
            COUNT(*) AS active_total
        FROM {self._table}
        WHERE status IN ('queued', 'running')
        """
        row = self._gateway.fetch_one(query=query, parameters={})
        return _count_from_row(row=row, field_name="active_total")

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestLazyTradesMaterializationTask | None:
        normalized_owner = _normalize_locked_by(value=locked_by)
        validated_lease_seconds = _validate_lease_seconds(lease_seconds=lease_seconds)
        lease_expires_at = now + timedelta(seconds=validated_lease_seconds)
        claimed_columns = _qualify_select_columns(relation_alias="tasks")
        final_columns = _qualify_select_columns(relation_alias="claimed")
        query = f"""
        WITH queued_candidate AS (
            SELECT
                task_id
            FROM {self._table}
            WHERE status = 'queued'
            ORDER BY created_at ASC, task_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ),
        reclaim_candidate AS (
            SELECT
                task_id
            FROM {self._table}
            WHERE status = 'running'
              AND lease_expires_at <= %(now)s
            ORDER BY lease_expires_at ASC, created_at ASC, task_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ),
        candidate AS (
            SELECT task_id, 1 AS priority FROM queued_candidate
            UNION ALL
            SELECT task_id, 2 AS priority FROM reclaim_candidate
            ORDER BY priority ASC
            LIMIT 1
        ),
        claimed AS (
            UPDATE {self._table} AS tasks
            SET
                status = 'running',
                started_at = CASE
                    WHEN tasks.started_at IS NULL THEN %(now)s
                    ELSE tasks.started_at
                END,
                updated_at = %(now)s,
                locked_by = %(locked_by)s,
                locked_at = %(now)s,
                lease_expires_at = %(lease_expires_at)s,
                heartbeat_at = %(now)s,
                attempt = tasks.attempt + 1
            FROM candidate
            WHERE tasks.task_id = candidate.task_id
            RETURNING
                {claimed_columns}
        )
        SELECT
            {final_columns}
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
        return _map_materialization_row(row=row)

    def heartbeat(
        self,
        *,
        task_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestLazyTradesMaterializationTask | None:
        normalized_owner = _normalize_locked_by(value=locked_by)
        validated_lease_seconds = _validate_lease_seconds(lease_seconds=lease_seconds)
        lease_expires_at = now + timedelta(seconds=validated_lease_seconds)
        query = f"""
        UPDATE {self._table}
        SET
            updated_at = %(now)s,
            heartbeat_at = %(now)s,
            lease_expires_at = %(lease_expires_at)s
        WHERE task_id = %(task_id)s
          AND status = 'running'
          AND locked_by = %(locked_by)s
        RETURNING
            {_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "task_id": str(task_id),
                "now": now,
                "locked_by": normalized_owner,
                "lease_expires_at": lease_expires_at,
            },
        )
        if row is None:
            return None
        return _map_materialization_row(row=row)

    def finish_completed(
        self,
        *,
        task_id: UUID,
        owner_user_id: UserId,
        now: datetime,
        locked_by: str,
        cache_status: str,
        cache_path: str | None,
    ) -> BacktestLazyTradesMaterializationTask | None:
        query = f"""
        UPDATE {self._table}
        SET
            status = 'completed',
            updated_at = %(now)s,
            finished_at = %(now)s,
            locked_by = NULL,
            locked_at = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            last_error = NULL,
            last_error_json = NULL,
            cache_status = %(cache_status)s,
            cache_path = %(cache_path)s
        WHERE task_id = %(task_id)s
          AND owner_user_id = %(owner_user_id)s
          AND status = 'running'
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "task_id": str(task_id),
                "owner_user_id": str(owner_user_id),
                "now": now,
                "locked_by": _normalize_locked_by(value=locked_by),
                "cache_status": _normalize_cache_status(value=cache_status),
                "cache_path": cache_path,
            },
        )
        if row is None:
            return None
        return _map_materialization_row(row=row)

    def finish_failed(
        self,
        *,
        task_id: UUID,
        owner_user_id: UserId,
        now: datetime,
        locked_by: str,
        last_error: str,
        last_error_json: Mapping[str, Any],
    ) -> BacktestLazyTradesMaterializationTask | None:
        query = f"""
        UPDATE {self._table}
        SET
            status = 'failed',
            updated_at = %(now)s,
            finished_at = %(now)s,
            locked_by = NULL,
            locked_at = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            last_error = %(last_error)s,
            last_error_json = %(last_error_json)s::jsonb
        WHERE task_id = %(task_id)s
          AND owner_user_id = %(owner_user_id)s
          AND status = 'running'
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "task_id": str(task_id),
                "owner_user_id": str(owner_user_id),
                "now": now,
                "locked_by": _normalize_locked_by(value=locked_by),
                "last_error": last_error[:2000],
                "last_error_json": json.dumps(
                    dict(last_error_json),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            },
        )
        if row is None:
            return None
        return _map_materialization_row(row=row)


def _map_materialization_row(
    *,
    row: Mapping[str, Any],
) -> BacktestLazyTradesMaterializationTask:
    try:
        return BacktestLazyTradesMaterializationTask(
            task_id=UUID(str(row["task_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            job_id=UUID(str(row["job_id"])),
            public_variant_key=str(row["public_variant_key"]),
            variant_hash=str(row["variant_hash"]),
            request_hash=str(row["request_hash"]),
            engine_params_hash=str(row["engine_params_hash"]),
            artifact_manifest_hash=str(row["artifact_manifest_hash"]),
            cache_key=str(row["cache_key"]),
            status=_materialization_status(row["status"]),
            priority_class=str(row["priority_class"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
            started_at=_optional_datetime(row["started_at"]),
            finished_at=_optional_datetime(row["finished_at"]),
            locked_by=None if row["locked_by"] is None else str(row["locked_by"]),
            locked_at=_optional_datetime(row["locked_at"]),
            lease_expires_at=_optional_datetime(row["lease_expires_at"]),
            heartbeat_at=_optional_datetime(row["heartbeat_at"]),
            attempt=int(row["attempt"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            last_error_json=_json_mapping(row["last_error_json"]),
            cache_status=str(row["cache_status"]),
            cache_path=None if row["cache_path"] is None else str(row["cache_path"]),
            ttl_seconds=int(row["ttl_seconds"]),
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError(
            "Cannot map backtest lazy trades materialization row"
        ) from error


def _materialization_status(value: Any) -> BacktestLazyTradesMaterializationStatus:
    normalized = str(value).strip().lower()
    if normalized not in {"queued", "running", "completed", "failed", "cancelled"}:
        raise BacktestStorageError(
            f"Unexpected backtest lazy trades materialization status: {normalized!r}"
        )
    return cast(BacktestLazyTradesMaterializationStatus, normalized)


def _datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise BacktestStorageError("Expected datetime column")
    return value


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise BacktestStorageError("Expected JSON object column")


def _count_from_row(*, row: Mapping[str, Any] | None, field_name: str) -> int:
    if row is None:
        raise BacktestStorageError(
            "PostgresBacktestLazyTradesMaterializationRepository count returned no row"
        )
    try:
        return int(row[field_name])
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError(
            "PostgresBacktestLazyTradesMaterializationRepository count invalid row"
        ) from error


def _qualify_select_columns(*, relation_alias: str) -> str:
    columns = [
        column.strip()
        for column in _BACKTEST_LAZY_TRADES_MATERIALIZATION_SELECT_COLUMNS.split(",")
        if column.strip()
    ]
    return ",\n    ".join(f"{relation_alias}.{column}" for column in columns)


def _normalize_locked_by(*, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("locked_by must be non-empty")
    if len(normalized) > 256:
        raise ValueError("locked_by must be <= 256 chars")
    return normalized


def _validate_lease_seconds(*, lease_seconds: int) -> int:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be > 0")
    return lease_seconds


def _normalize_cache_status(*, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("cache_status must be non-empty")
    if len(normalized) > 64:
        raise ValueError("cache_status must be <= 64 chars")
    return normalized


__all__ = ["PostgresBacktestLazyTradesMaterializationRepository"]
