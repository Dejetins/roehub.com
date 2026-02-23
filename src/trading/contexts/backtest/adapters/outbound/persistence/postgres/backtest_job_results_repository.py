from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import BacktestJobResultsRepository
from trading.contexts.backtest.domain.entities import (
    BacktestJobStageAShortlist,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError


class PostgresBacktestJobResultsRepository(BacktestJobResultsRepository):
    """
    Explicit SQL adapter for Backtest jobs top-k snapshots and Stage-A shortlist storage.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def __init__(
        self,
        *,
        gateway: BacktestPostgresGateway,
        jobs_table: str = "backtest_jobs",
        top_variants_table: str = "backtest_job_top_variants",
        stage_a_shortlist_table: str = "backtest_job_stage_a_shortlist",
    ) -> None:
        """
        Initialize repository with SQL gateway and target table names.

        Args:
            gateway: SQL gateway abstraction.
            jobs_table: Jobs table name.
            top_variants_table: Top variants table name.
            stage_a_shortlist_table: Stage-A shortlist table name.
        Returns:
            None.
        Assumptions:
            Table schemas follow Backtest jobs v1 migration contract.
        Raises:
            ValueError: If one dependency/table name is invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresBacktestJobResultsRepository requires gateway")

        normalized_jobs_table = jobs_table.strip()
        normalized_top_table = top_variants_table.strip()
        normalized_shortlist_table = stage_a_shortlist_table.strip()
        if not normalized_jobs_table:
            raise ValueError("PostgresBacktestJobResultsRepository requires non-empty jobs_table")
        if not normalized_top_table:
            raise ValueError(
                "PostgresBacktestJobResultsRepository requires non-empty top_variants_table"
            )
        if not normalized_shortlist_table:
            raise ValueError(
                "PostgresBacktestJobResultsRepository requires non-empty stage_a_shortlist_table"
            )

        self._gateway = gateway
        self._jobs_table = normalized_jobs_table
        self._top_variants_table = normalized_top_table
        self._stage_a_shortlist_table = normalized_shortlist_table

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[BacktestJobTopVariant, ...],
    ) -> bool:
        """
        Replace full top-k snapshot using one SQL statement (delete old + insert new rows).

        Args:
            job_id: Job identifier.
            now: Snapshot timestamp in UTC.
            locked_by: Expected worker owner identity.
            rows: Full ranked rows payload.
        Returns:
            bool: `True` when lease-guarded replace is applied; `False` when lease is lost.
        Assumptions:
            Snapshot persistence keeps `report_table_md` null for best-so-far running writes.
        Raises:
            BacktestStorageError: If SQL execution fails or row payload is invalid.
        Side Effects:
            Replaces rows in `backtest_job_top_variants` table.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        serialized_rows = _serialize_top_rows(job_id=job_id, rows=rows)

        query = f"""
        WITH lease_owner AS (
            SELECT
                job_id
            FROM {self._jobs_table}
            WHERE job_id = %(job_id)s
              AND state = 'running'
              AND locked_by = %(locked_by)s
              AND lease_expires_at > %(now)s
        ),
        deleted AS (
            DELETE FROM {self._top_variants_table}
            WHERE job_id IN (SELECT job_id FROM lease_owner)
        ),
        source_rows AS (
            SELECT item
            FROM jsonb_array_elements(%(rows_json)s::jsonb) AS item
        ),
        inserted AS (
            INSERT INTO {self._top_variants_table}
            (
                job_id,
                rank,
                variant_key,
                indicator_variant_key,
                variant_index,
                total_return_pct,
                payload_json,
                report_table_md,
                trades_json,
                updated_at
            )
            SELECT
                %(job_id)s::uuid AS job_id,
                (item ->> 'rank')::INTEGER AS rank,
                item ->> 'variant_key' AS variant_key,
                item ->> 'indicator_variant_key' AS indicator_variant_key,
                (item ->> 'variant_index')::INTEGER AS variant_index,
                (item ->> 'total_return_pct')::DOUBLE PRECISION AS total_return_pct,
                item -> 'payload_json' AS payload_json,
                item ->> 'report_table_md' AS report_table_md,
                item -> 'trades_json' AS trades_json,
                %(now)s AS updated_at
            FROM source_rows
            WHERE EXISTS (SELECT 1 FROM lease_owner)
            ORDER BY
                (item ->> 'rank')::INTEGER ASC,
                (item ->> 'variant_key') ASC
        )
        SELECT EXISTS(SELECT 1 FROM lease_owner) AS applied
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": normalized_owner,
                "rows_json": json.dumps(
                    serialized_rows,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            },
        )
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestJobResultsRepository.replace_top_variants_snapshot"
                " returned no status row"
            )
        try:
            return bool(row["applied"])
        except Exception as error:  # noqa: BLE001
            raise BacktestStorageError(
                "PostgresBacktestJobResultsRepository.replace_top_variants_snapshot"
                " invalid status row"
            ) from error

    def list_top_variants(self, *, job_id: UUID, limit: int) -> tuple[BacktestJobTopVariant, ...]:
        """
        Load persisted top variants ordered deterministically by rank and key.

        Args:
            job_id: Job identifier.
            limit: Max row count.
        Returns:
            tuple[BacktestJobTopVariant, ...]: Ordered top-variant snapshots.
        Assumptions:
            SQL order is explicitly fixed to `rank ASC, variant_key ASC`.
        Raises:
            BacktestStorageError: If SQL read or row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        if limit <= 0:
            raise BacktestStorageError(
                "PostgresBacktestJobResultsRepository.list_top_variants limit must be > 0"
            )

        query = f"""
        SELECT
            job_id,
            rank,
            variant_key,
            indicator_variant_key,
            variant_index,
            total_return_pct,
            payload_json,
            report_table_md,
            trades_json,
            updated_at
        FROM {self._top_variants_table}
        WHERE job_id = %(job_id)s
        ORDER BY rank ASC, variant_key ASC
        LIMIT %(limit)s
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={
                "job_id": str(job_id),
                "limit": limit,
            },
        )
        return tuple(_map_top_variant_row(row=row) for row in rows)

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: BacktestJobStageAShortlist,
    ) -> bool:
        """
        Upsert Stage-A shortlist payload with active lease-owner conditional guard.

        Args:
            job_id: Job identifier.
            now: Upsert timestamp in UTC.
            locked_by: Expected worker owner identity.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: `True` when upsert is applied; `False` when lease is lost.
        Assumptions:
            Stage-A shortlist uses deterministic ordered integer index array.
        Raises:
            BacktestStorageError: If SQL write fails.
        Side Effects:
            Upserts one row in `backtest_job_stage_a_shortlist` table.
        """
        normalized_owner = _normalize_locked_by(value=locked_by)
        if shortlist.job_id != job_id:
            raise BacktestStorageError(
                "PostgresBacktestJobResultsRepository.save_stage_a_shortlist"
                " shortlist.job_id must match method job_id"
            )

        query = f"""
        WITH lease_owner AS (
            SELECT
                job_id
            FROM {self._jobs_table}
            WHERE job_id = %(job_id)s
              AND state = 'running'
              AND locked_by = %(locked_by)s
              AND lease_expires_at > %(now)s
        )
        INSERT INTO {self._stage_a_shortlist_table}
        (
            job_id,
            stage_a_indexes_json,
            stage_a_variants_total,
            risk_total,
            preselect_used,
            updated_at
        )
        SELECT
            %(job_id)s::uuid,
            %(stage_a_indexes_json)s::jsonb,
            %(stage_a_variants_total)s,
            %(risk_total)s,
            %(preselect_used)s,
            %(now)s
        FROM lease_owner
        ON CONFLICT (job_id)
        DO UPDATE SET
            stage_a_indexes_json = EXCLUDED.stage_a_indexes_json,
            stage_a_variants_total = EXCLUDED.stage_a_variants_total,
            risk_total = EXCLUDED.risk_total,
            preselect_used = EXCLUDED.preselect_used,
            updated_at = EXCLUDED.updated_at
        RETURNING job_id
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "locked_by": normalized_owner,
                "now": now,
                "stage_a_indexes_json": json.dumps(
                    shortlist.to_json_array(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                "stage_a_variants_total": shortlist.stage_a_variants_total,
                "risk_total": shortlist.risk_total,
                "preselect_used": shortlist.preselect_used,
            },
        )
        return row is not None

    def get_stage_a_shortlist(self, *, job_id: UUID) -> BacktestJobStageAShortlist | None:
        """
        Load Stage-A shortlist payload for one job id.

        Args:
            job_id: Job identifier.
        Returns:
            BacktestJobStageAShortlist | None: Snapshot payload or `None`.
        Assumptions:
            One shortlist row exists per job id at most.
        Raises:
            BacktestStorageError: If SQL read or row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            job_id,
            stage_a_indexes_json,
            stage_a_variants_total,
            risk_total,
            preselect_used,
            updated_at
        FROM {self._stage_a_shortlist_table}
        WHERE job_id = %(job_id)s
        """
        row = self._gateway.fetch_one(query=query, parameters={"job_id": str(job_id)})
        if row is None:
            return None
        return _map_stage_a_shortlist_row(row=row)



def _serialize_top_rows(
    *,
    job_id: UUID,
    rows: tuple[BacktestJobTopVariant, ...],
) -> list[dict[str, Any]]:
    """
    Serialize top-variant rows into deterministic JSON payload for SQL statement input.

    Args:
        job_id: Method-level job identifier.
        rows: Ranked rows payload.
    Returns:
        list[dict[str, Any]]: Deterministic JSON-serializable list payload.
    Assumptions:
        Rows represent full snapshot and are replaced atomically.
    Raises:
        BacktestStorageError: If row job id mismatches method job id.
    Side Effects:
        None.
    """
    serialized: list[dict[str, Any]] = []
    for row in rows:
        if row.job_id != job_id:
            raise BacktestStorageError(
                "PostgresBacktestJobResultsRepository.replace_top_variants_snapshot"
                " row.job_id must match method job_id"
            )
        serialized.append(
            {
                "rank": row.rank,
                "variant_key": row.variant_key,
                "indicator_variant_key": row.indicator_variant_key,
                "variant_index": row.variant_index,
                "total_return_pct": row.total_return_pct,
                "payload_json": dict(row.payload_json),
                "report_table_md": row.report_table_md,
                "trades_json": [dict(item) for item in row.trades_json]
                if row.trades_json is not None
                else None,
            }
        )
    return serialized



def _map_top_variant_row(*, row: Mapping[str, Any]) -> BacktestJobTopVariant:
    """
    Map SQL row payload into immutable `BacktestJobTopVariant` entity.

    Args:
        row: SQL row mapping.
    Returns:
        BacktestJobTopVariant: Mapped top-variant snapshot.
    Assumptions:
        Row schema follows Backtest jobs v1 storage contract.
    Raises:
        BacktestStorageError: If row cannot be mapped.
    Side Effects:
        None.
    """
    try:
        trades_value = _parse_json_array(value=row.get("trades_json"), field_name="trades_json")
        trades_json: tuple[Mapping[str, Any], ...] | None = None
        if trades_value is not None:
            parsed_items: list[Mapping[str, Any]] = []
            for item in trades_value:
                if not isinstance(item, Mapping):
                    raise BacktestStorageError(
                        "backtest_job_top_variants.trades_json must contain JSON objects"
                    )
                parsed_items.append(dict(item))
            trades_json = tuple(parsed_items)

        payload_json = _parse_json_object(
            value=row.get("payload_json"),
            field_name="payload_json",
            required=True,
        )
        if payload_json is None:
            raise BacktestStorageError("backtest_job_top_variants.payload_json is required")

        return BacktestJobTopVariant(
            job_id=UUID(str(row["job_id"])),
            rank=int(row["rank"]),
            variant_key=str(row["variant_key"]),
            indicator_variant_key=str(row["indicator_variant_key"]),
            variant_index=int(row["variant_index"]),
            total_return_pct=float(row["total_return_pct"]),
            payload_json=payload_json,
            report_table_md=str(row["report_table_md"])
            if row.get("report_table_md") is not None
            else None,
            trades_json=trades_json,
            updated_at=row["updated_at"],
        )
    except Exception as error:  # noqa: BLE001
        if isinstance(error, BacktestStorageError):
            raise
        raise BacktestStorageError(
            "PostgresBacktestJobResultsRepository cannot map top variant row"
        ) from error



def _map_stage_a_shortlist_row(*, row: Mapping[str, Any]) -> BacktestJobStageAShortlist:
    """
    Map SQL row payload into immutable `BacktestJobStageAShortlist` entity.

    Args:
        row: SQL row mapping.
    Returns:
        BacktestJobStageAShortlist: Mapped shortlist snapshot.
    Assumptions:
        Row schema follows Backtest jobs v1 storage contract.
    Raises:
        BacktestStorageError: If row cannot be mapped.
    Side Effects:
        None.
    """
    try:
        raw_indexes = _parse_json_array(
            value=row.get("stage_a_indexes_json"),
            field_name="stage_a_indexes_json",
        )
        if raw_indexes is None:
            raise BacktestStorageError(
                "backtest_job_stage_a_shortlist.stage_a_indexes_json is required"
            )

        indexes: list[int] = []
        for raw_item in raw_indexes:
            if isinstance(raw_item, bool) or not isinstance(raw_item, int):
                raise BacktestStorageError(
                    "backtest_job_stage_a_shortlist.stage_a_indexes_json must contain integers"
                )
            indexes.append(raw_item)

        return BacktestJobStageAShortlist(
            job_id=UUID(str(row["job_id"])),
            stage_a_indexes=tuple(indexes),
            stage_a_variants_total=int(row["stage_a_variants_total"]),
            risk_total=int(row["risk_total"]),
            preselect_used=int(row["preselect_used"]),
            updated_at=row["updated_at"],
        )
    except Exception as error:  # noqa: BLE001
        if isinstance(error, BacktestStorageError):
            raise
        raise BacktestStorageError(
            "PostgresBacktestJobResultsRepository cannot map stage_a shortlist row"
        ) from error



def _parse_json_object(
    *,
    value: Any,
    field_name: str,
    required: bool,
) -> Mapping[str, Any] | None:
    """
    Parse JSON object column payload into mapping.

    Args:
        value: Raw gateway value.
        field_name: Column name for deterministic error message.
        required: Whether value is mandatory.
    Returns:
        Mapping[str, Any] | None: Parsed JSON object mapping.
    Assumptions:
        Gateway may return dict/bytes/memoryview/JSON text representations.
    Raises:
        BacktestStorageError: If value is invalid or missing.
    Side Effects:
        None.
    """
    decoded = _decode_json_value(value=value, field_name=field_name, required=required)
    if decoded is None:
        return None
    if not isinstance(decoded, Mapping):
        raise BacktestStorageError(f"{field_name} must be JSON object")
    return dict(decoded)



def _parse_json_array(*, value: Any, field_name: str) -> list[Any] | None:
    """
    Parse optional JSON array column payload into Python list.

    Args:
        value: Raw gateway value.
        field_name: Column name for deterministic error message.
    Returns:
        list[Any] | None: Parsed JSON array payload.
    Assumptions:
        Missing value is represented as `None`.
    Raises:
        BacktestStorageError: If value is not JSON array payload.
    Side Effects:
        None.
    """
    decoded = _decode_json_value(value=value, field_name=field_name, required=False)
    if decoded is None:
        return None
    if not isinstance(decoded, list):
        raise BacktestStorageError(f"{field_name} must be JSON array")
    return decoded



def _decode_json_value(*, value: Any, field_name: str, required: bool) -> Any:
    """
    Decode JSON payload from gateway row preserving object/array scalar forms.

    Args:
        value: Raw gateway value.
        field_name: Column name for deterministic error messages.
        required: Whether value is mandatory.
    Returns:
        Any: Decoded JSON-compatible value.
    Assumptions:
        Gateway may return dict/list directly or encoded bytes/text payload.
    Raises:
        BacktestStorageError: If value is missing or cannot be decoded.
    Side Effects:
        None.
    """
    if value is None:
        if required:
            raise BacktestStorageError(f"{field_name} is required")
        return None

    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
        return value

    raw_value = value
    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes().decode("utf-8")
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = bytes(raw_value).decode("utf-8")

    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as error:
            raise BacktestStorageError(f"{field_name} has invalid JSON text") from error

    raise BacktestStorageError(
        f"{field_name} has unsupported type {type(value).__name__}"
    )



def _normalize_locked_by(*, value: str) -> str:
    """
    Validate and normalize worker lease owner literal.

    Args:
        value: Raw owner literal.
    Returns:
        str: Trimmed owner literal.
    Assumptions:
        Lease owner string must be non-empty.
    Raises:
        BacktestStorageError: If owner literal is blank.
    Side Effects:
        None.
    """
    normalized = value.strip()
    if not normalized:
        raise BacktestStorageError("locked_by must be non-empty")
    return normalized


__all__ = ["PostgresBacktestJobResultsRepository"]
