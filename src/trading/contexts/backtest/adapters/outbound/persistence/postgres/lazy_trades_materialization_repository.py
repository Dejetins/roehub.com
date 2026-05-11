from __future__ import annotations

import json
from datetime import datetime
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
            updated_at = {self._table}.updated_at
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


__all__ = ["PostgresBacktestLazyTradesMaterializationRepository"]
