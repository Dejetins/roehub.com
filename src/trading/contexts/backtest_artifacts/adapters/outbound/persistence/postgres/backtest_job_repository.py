from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestJobRepository,
)
from trading.contexts.backtest.domain.entities import (
    BacktestArtifactSlotLiteral,
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobExecutionMode,
    BacktestJobMode,
    BacktestJobStage,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.application.services.v2.metrics_kernel import (
    normalize_persisted_summary_metrics_v2,
)
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
    artifact_slot,
    artifact_slot_generation,
    artifact_manifest_hash,
    artifact_asof_date,
    execution_mode,
    execution_profile_mode_hint,
    effective_execution_profile_mode,
    market_id,
    symbol,
    timeframe,
    requested_top_n,
    ranking_primary_metric,
    ranking_secondary_metric,
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
      - docs/architecture/backtest/README.md
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
        top_variants_table: str = "backtest_job_top_variants",
        stage_a_shortlist_table: str = "backtest_job_stage_a_shortlist",
    ) -> None:
        """
        Initialize repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            jobs_table: Backtest jobs table name.
            top_variants_table: Backtest job top-variants table name.
            stage_a_shortlist_table: Backtest Stage A shortlist table name.
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
        normalized_top_variants_table = top_variants_table.strip()
        normalized_stage_a_shortlist_table = stage_a_shortlist_table.strip()
        if not normalized_table:
            raise ValueError("PostgresBacktestJobRepository requires non-empty jobs_table")
        if not normalized_top_variants_table:
            raise ValueError(
                "PostgresBacktestJobRepository requires non-empty top_variants_table"
            )
        if not normalized_stage_a_shortlist_table:
            raise ValueError(
                "PostgresBacktestJobRepository requires non-empty stage_a_shortlist_table"
            )
        self._gateway = gateway
        self._jobs_table = normalized_table
        self._top_variants_table = normalized_top_variants_table
        self._stage_a_shortlist_table = normalized_stage_a_shortlist_table

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Persist new job row and return mapped immutable aggregate snapshot.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
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
        insert_parameters = _build_job_insert_parameters(job=job)
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
            artifact_slot,
            artifact_slot_generation,
            artifact_manifest_hash,
            artifact_asof_date,
            execution_mode,
            execution_profile_mode_hint,
            effective_execution_profile_mode,
            market_id,
            symbol,
            timeframe,
            requested_top_n,
            ranking_primary_metric,
            ranking_secondary_metric,
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
            %(artifact_slot)s,
            %(artifact_slot_generation)s,
            %(artifact_manifest_hash)s,
            %(artifact_asof_date)s,
            %(execution_mode)s,
            %(execution_profile_mode_hint)s,
            %(effective_execution_profile_mode)s,
            %(market_id)s,
            %(symbol)s,
            %(timeframe)s,
            %(requested_top_n)s,
            %(ranking_primary_metric)s,
            %(ranking_secondary_metric)s,
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
            parameters=insert_parameters,
        )
        if row is None:
            raise BacktestStorageError("PostgresBacktestJobRepository.create returned no row")
        return _map_job_row(row=row)

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        """
        Persist one terminal run row, summary-only top rows, and optional shortlist atomically.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
        Args:
            job: Prepared terminal persisted-run aggregate.
            top_variants: Summary-only top rows ordered by `rank ASC, variant_key ASC`.
            stage_a_shortlist:
                Optional internal shortlist snapshot for `exact_no_risk_parity` sync runs.
        Returns:
            BacktestJob: Persisted immutable job snapshot.
        Assumptions:
            Sync-inline cutover persists only final succeeded rows and does not store detail
            payloads in `report_table_md/trades_json`; internal shortlist persistence remains
            additive and summary-only transport stays unchanged.
        Raises:
            BacktestStorageError: If SQL execution fails or row mapping breaks.
        Side Effects:
            Writes one row in `backtest_jobs` and zero or more rows in
            `backtest_job_top_variants`, plus at most one row in
            `backtest_job_stage_a_shortlist`.
        """
        insert_parameters = _build_job_insert_parameters(job=job)
        insert_parameters["rows_json"] = json.dumps(
            _serialize_top_rows(job_id=job.job_id, rows=top_variants),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        insert_parameters.update(
            _build_stage_a_shortlist_insert_parameters(shortlist=stage_a_shortlist)
        )
        query = f"""
        WITH inserted_job AS (
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
                artifact_slot,
                artifact_slot_generation,
                artifact_manifest_hash,
                artifact_asof_date,
                execution_mode,
                execution_profile_mode_hint,
                effective_execution_profile_mode,
                market_id,
                symbol,
                timeframe,
                requested_top_n,
                ranking_primary_metric,
                ranking_secondary_metric,
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
                %(artifact_slot)s,
                %(artifact_slot_generation)s,
                %(artifact_manifest_hash)s,
                %(artifact_asof_date)s,
                %(execution_mode)s,
                %(execution_profile_mode_hint)s,
                %(effective_execution_profile_mode)s,
                %(market_id)s,
                %(symbol)s,
                %(timeframe)s,
                %(requested_top_n)s,
                %(ranking_primary_metric)s,
                %(ranking_secondary_metric)s,
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
        ),
        source_rows AS (
            SELECT item
            FROM jsonb_array_elements(%(rows_json)s::jsonb) AS item
        ),
        inserted_shortlist AS (
            INSERT INTO {self._stage_a_shortlist_table}
            (
                job_id,
                stage_a_indexes_json,
                stage_a_variants_total,
                risk_total,
                preselect_used,
                no_risk_exact_rows_json,
                parity_runtime_state_json,
                updated_at
            )
            SELECT
                %(job_id)s::uuid AS job_id,
                %(stage_a_indexes_json)s::jsonb AS stage_a_indexes_json,
                %(stage_a_variants_total)s AS stage_a_variants_total,
                %(risk_total)s AS risk_total,
                %(preselect_used)s AS preselect_used,
                %(no_risk_exact_rows_json)s::jsonb AS no_risk_exact_rows_json,
                %(parity_runtime_state_json)s::jsonb AS parity_runtime_state_json,
                %(updated_at)s AS updated_at
            FROM inserted_job
            WHERE %(stage_a_indexes_json)s::jsonb IS NOT NULL
        ),
        inserted_rows AS (
            INSERT INTO {self._top_variants_table}
            (
                job_id,
                rank,
                variant_key,
                indicator_variant_key,
                variant_index,
                total_return_pct,
                payload_json,
                summary_metrics_json,
                best_tp_pct,
                best_sl_pct,
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
                item -> 'summary_metrics_json' AS summary_metrics_json,
                (item ->> 'best_tp_pct')::DOUBLE PRECISION AS best_tp_pct,
                (item ->> 'best_sl_pct')::DOUBLE PRECISION AS best_sl_pct,
                NULL::TEXT AS report_table_md,
                NULL::JSONB AS trades_json,
                %(updated_at)s AS updated_at
            FROM source_rows
            ORDER BY
                (item ->> 'rank')::INTEGER ASC,
                (item ->> 'variant_key') ASC
        )
        SELECT
            {_BACKTEST_JOB_SELECT_COLUMNS}
        FROM inserted_job
        """
        row = self._gateway.fetch_one(query=query, parameters=insert_parameters)
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.create_with_top_variants returned no row"
            )
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
            Cancel operation is idempotent for terminal jobs and preserves the first running
            `cancel_requested_at` marker for deterministic R8-03 lifecycle visibility.
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
            cancel_requested_at = CASE
                WHEN state = 'running' AND cancel_requested_at IS NOT NULL
                    THEN cancel_requested_at
                ELSE %(cancel_requested_at)s
            END,
            updated_at = CASE
                WHEN state = 'running' AND cancel_requested_at IS NOT NULL
                    THEN updated_at
                ELSE %(cancel_requested_at)s
            END
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

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Count active jobs pinning one previously published inactive-slot manifest identity.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
        Args:
            market_id: Canonical market id for the symbol-root being published.
            symbol: Instrument symbol pinned by the active jobs.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 of the inactive slot `manifest.yaml`.
        Returns:
            int: Number of active jobs blocking rebuild/publish of this slot content.
        Assumptions:
            R8-03 blocking set is explicit: only `queued|running` rows with
            `execution_mode in ('background_auto', 'background_manual_legacy')` participate in
            inactive-slot publish guard. R7-01 rows prefer denormalized `(market_id, symbol)`
            columns, while legacy rows still fall back to canonical payload snapshots inside
            `request_json/spec_payload_json`.
        Raises:
            BacktestStorageError: If count row is missing or invalid.
        Side Effects:
            Executes one SQL aggregate select.
        """
        sql = f"""
        SELECT
            COUNT(*) AS active_total
        FROM {self._jobs_table}
        WHERE state IN ('queued', 'running')
          AND execution_mode IN ('background_auto', 'background_manual_legacy')
          AND artifact_slot = %(artifact_slot)s
          AND artifact_manifest_hash = %(artifact_manifest_hash)s
          AND (
                (
                    market_id = %(market_id)s
                    AND symbol = %(symbol)s
                )
                OR (
                    market_id IS NULL
                    AND symbol IS NULL
                    AND (
                        (
                            request_json -> 'template' -> 'instrument_id' ->> 'market_id'
                        )::integer = %(market_id)s
                        AND request_json -> 'template' -> 'instrument_id' ->> 'symbol'
                            = %(symbol)s
                        OR (
                            spec_payload_json -> 'instrument_id' ->> 'market_id'
                        )::integer = %(market_id)s
                        AND spec_payload_json -> 'instrument_id' ->> 'symbol' = %(symbol)s
                    )
                )
          )
        """
        row = self._gateway.fetch_one(
            query=sql,
            parameters={
                "market_id": market_id,
                "symbol": symbol,
                "artifact_slot": artifact_slot,
                "artifact_manifest_hash": artifact_manifest_hash,
            },
        )
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.count_active_for_artifact_manifest returned no row"
            )
        try:
            return int(row["active_total"])
        except Exception as error:  # noqa: BLE001
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.count_active_for_artifact_manifest "
                "invalid count row"
            ) from error


def _map_job_row(*, row: Mapping[str, Any]) -> BacktestJob:
    """
    Map SQL row payload into immutable `BacktestJob` aggregate.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    Args:
        row: SQL row mapping.
    Returns:
        BacktestJob: Mapped immutable job aggregate.
    Assumptions:
        Row schema follows additive Backtest jobs persisted-run storage contract, and legacy rows
        may still require explicit `request_json.execution_profile_mode` fallback when the new
        metadata columns are null.
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
        execution_profile_mode_hint = _normalize_optional_execution_profile_mode_metadata(
            value=row.get("execution_profile_mode_hint"),
            field_name="execution_profile_mode_hint",
        )
        effective_execution_profile_mode = _normalize_optional_execution_profile_mode_metadata(
            value=row.get("effective_execution_profile_mode"),
            field_name="effective_execution_profile_mode",
        )
        if effective_execution_profile_mode is None:
            effective_execution_profile_mode = _legacy_execution_profile_mode_from_request_json(
                request_json=request_payload
            )

        spec_payload = _parse_json_object(
            value=row.get("spec_payload_json"),
            field_name="spec_payload_json",
            required=False,
        )
        artifact_pin = _parse_artifact_pin(row=row)
        created_at = _normalize_storage_datetime_utc(
            value=row["created_at"],
            field_name="created_at",
        )
        updated_at = _normalize_storage_datetime_utc(
            value=row["updated_at"],
            field_name="updated_at",
        )
        if created_at is None or updated_at is None:
            raise BacktestStorageError(
                "backtest_jobs.created_at/updated_at must be non-null UTC datetimes"
            )
        return BacktestJob(
            job_id=UUID(str(row["job_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            mode=_parse_job_mode(value=row["mode"]),
            state=_parse_job_state(value=row["state"]),
            created_at=created_at,
            updated_at=updated_at,
            started_at=_normalize_storage_datetime_utc(
                value=row.get("started_at"),
                field_name="started_at",
                required=False,
            ),
            finished_at=_normalize_storage_datetime_utc(
                value=row.get("finished_at"),
                field_name="finished_at",
                required=False,
            ),
            cancel_requested_at=_normalize_storage_datetime_utc(
                value=row.get("cancel_requested_at"),
                field_name="cancel_requested_at",
                required=False,
            ),
            request_json=request_payload,
            request_hash=str(row["request_hash"]),
            spec_hash=str(row["spec_hash"]).strip() if row.get("spec_hash") is not None else None,
            spec_payload_json=spec_payload,
            engine_params_hash=str(row["engine_params_hash"]),
            backtest_runtime_config_hash=str(row["backtest_runtime_config_hash"]),
            artifact_pin=artifact_pin,
            execution_mode=_parse_execution_mode(value=row.get("execution_mode")),
            execution_profile_mode_hint=execution_profile_mode_hint,
            effective_execution_profile_mode=effective_execution_profile_mode,
            market_id=int(row["market_id"]) if row.get("market_id") is not None else None,
            symbol=str(row["symbol"]) if row.get("symbol") is not None else None,
            timeframe=str(row["timeframe"]) if row.get("timeframe") is not None else None,
            requested_top_n=int(row["requested_top_n"])
            if row.get("requested_top_n") is not None
            else None,
            ranking_primary_metric=str(row["ranking_primary_metric"])
            if row.get("ranking_primary_metric") is not None
            else None,
            ranking_secondary_metric=str(row["ranking_secondary_metric"])
            if row.get("ranking_secondary_metric") is not None
            else None,
            stage=_parse_job_stage(value=row["stage"]),
            processed_units=int(row["processed_units"]),
            total_units=int(row["total_units"]),
            progress_updated_at=_normalize_storage_datetime_utc(
                value=row.get("progress_updated_at"),
                field_name="progress_updated_at",
                required=False,
            ),
            locked_by=str(row["locked_by"]) if row.get("locked_by") is not None else None,
            locked_at=_normalize_storage_datetime_utc(
                value=row.get("locked_at"),
                field_name="locked_at",
                required=False,
            ),
            lease_expires_at=_normalize_storage_datetime_utc(
                value=row.get("lease_expires_at"),
                field_name="lease_expires_at",
                required=False,
            ),
            heartbeat_at=_normalize_storage_datetime_utc(
                value=row.get("heartbeat_at"),
                field_name="heartbeat_at",
                required=False,
            ),
            attempt=int(row["attempt"]),
            last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
            last_error_json=last_error_payload,
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError("PostgresBacktestJobRepository cannot map job row") from error


def _build_job_insert_parameters(*, job: BacktestJob) -> dict[str, Any]:
    """
    Build canonical SQL parameters mapping for one `backtest_jobs` insert statement.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
    Args:
        job: Prepared immutable job aggregate.
    Returns:
        dict[str, Any]: SQL parameters mapping for one insert statement.
    Assumptions:
        JSON payloads are canonicalized via stable `json.dumps(... sort_keys=True)`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
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
        "artifact_slot": job.artifact_pin.artifact_slot if job.artifact_pin is not None else None,
        "artifact_slot_generation": (
            job.artifact_pin.artifact_slot_generation if job.artifact_pin is not None else None
        ),
        "artifact_manifest_hash": (
            job.artifact_pin.artifact_manifest_hash if job.artifact_pin is not None else None
        ),
        "artifact_asof_date": (
            job.artifact_pin.artifact_asof_date if job.artifact_pin is not None else None
        ),
        "execution_mode": job.execution_mode,
        "execution_profile_mode_hint": job.execution_profile_mode_hint,
        "effective_execution_profile_mode": job.effective_execution_profile_mode,
        "market_id": job.market_id,
        "symbol": job.symbol,
        "timeframe": job.timeframe,
        "requested_top_n": job.requested_top_n,
        "ranking_primary_metric": job.ranking_primary_metric,
        "ranking_secondary_metric": job.ranking_secondary_metric,
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
    }


def _build_stage_a_shortlist_insert_parameters(
    *,
    shortlist: BacktestJobStageAShortlist | None,
) -> dict[str, Any]:
    """
    Build canonical SQL parameters for the optional sync-inline shortlist insert branch.

    Args:
        shortlist: Optional internal Stage A shortlist snapshot carried from live sync execution.
    Returns:
        dict[str, Any]: SQL parameters mapping consumed by the optional shortlist CTE.
    Assumptions:
        When shortlist is absent the terminal sync write must remain backward-compatible and skip
        the `backtest_job_stage_a_shortlist` insert branch, while no-risk shortlist payloads must
        carry persisted parity runtime-state literals.
    Raises:
        BacktestStorageError: If no-risk shortlist rows are provided without
            `parity_runtime_state` evidence.
    Side Effects:
        None.
    """
    if shortlist is None:
        return {
            "stage_a_indexes_json": None,
            "stage_a_variants_total": None,
            "risk_total": None,
            "preselect_used": None,
            "no_risk_exact_rows_json": None,
            "parity_runtime_state_json": None,
        }
    if shortlist.no_risk_exact_rows is not None and shortlist.parity_runtime_state is None:
        raise BacktestStorageError(
            "backtest_job_stage_a_shortlist requires DB-backed runtime-shape literals when "
            "no_risk_exact_rows_json is populated"
        )
    return {
        "stage_a_indexes_json": _json_dumps(payload=shortlist.to_json_array()),
        "stage_a_variants_total": shortlist.stage_a_variants_total,
        "risk_total": shortlist.risk_total,
        "preselect_used": shortlist.preselect_used,
        "no_risk_exact_rows_json": _json_dumps(
            payload=shortlist.to_no_risk_exact_rows_json_array()
        )
        if shortlist.no_risk_exact_rows is not None
        else None,
        "parity_runtime_state_json": _json_dumps(
            payload=shortlist.to_parity_runtime_state_json_object()
        )
        if shortlist.parity_runtime_state is not None
        else None,
    }


def _normalize_optional_execution_profile_mode_metadata(
    *,
    value: Any,
    field_name: str,
) -> str | None:
    """
    Normalize one optional persisted execution-profile metadata column from a SQL row.

    Args:
        value: Raw SQL row value.
        field_name: Column name used in deterministic error messages.
    Returns:
        str | None: Lowercase stripped metadata literal, or `None`.
    Assumptions:
        Dedicated execution-profile metadata columns are additive and nullable for historical
        rows.
    Raises:
        BacktestStorageError: If the column is present but normalizes to an empty string.
    Side Effects:
        None.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        raise BacktestStorageError(f"backtest_jobs.{field_name} must be non-empty when set")
    return normalized


def _legacy_execution_profile_mode_from_request_json(
    *,
    request_json: Mapping[str, Any],
) -> str | None:
    """
    Read legacy persisted execution-profile metadata from `request_json` as an explicit fallback.

    Args:
        request_json: Canonical persisted request payload.
    Returns:
        str | None: Lowercase execution-profile metadata literal, or `None`.
    Assumptions:
        Only historical rows use `request_json.execution_profile_mode`; new rows should hydrate
        from dedicated metadata columns instead.
    Raises:
        None.
    Side Effects:
        None.
    """
    raw_mode = request_json.get("execution_profile_mode")
    if not isinstance(raw_mode, str):
        return None
    normalized = raw_mode.strip().lower()
    if not normalized:
        return None
    return normalized


def _serialize_top_rows(
    *,
    job_id: UUID,
    rows: tuple[BacktestJobTopVariant, ...],
) -> list[dict[str, Any]]:
    """
    Serialize summary-only top rows into canonical JSON array for one atomic SQL insert.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
    Args:
        job_id: Parent job identifier expected on every row.
        rows: Summary-only top rows ordered by rank.
    Returns:
        list[dict[str, Any]]: Canonical JSON-serializable rows payload.
    Assumptions:
        Persisted sync-inline rows keep `report_table_md/trades_json` null-only.
    Raises:
        BacktestStorageError: If one row belongs to another job id.
    Side Effects:
        None.
    """
    serialized_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.rank, item.variant_key)):
        if row.job_id != job_id:
            raise BacktestStorageError(
                "PostgresBacktestJobRepository.create_with_top_variants "
                "received mismatched top row job_id"
            )
        summary_metrics_payload = dict(row.summary_metrics_json)
        summary_metrics_payload["total_return_pct"] = row.total_return_pct
        serialized_rows.append(
            {
                "rank": row.rank,
                "variant_key": row.variant_key,
                "indicator_variant_key": row.indicator_variant_key,
                "variant_index": row.variant_index,
                "total_return_pct": row.total_return_pct,
                "payload_json": dict(row.payload_json),
                "summary_metrics_json": dict(
                    normalize_persisted_summary_metrics_v2(metrics=summary_metrics_payload)
                ),
                "best_tp_pct": row.best_tp_pct,
                "best_sl_pct": row.best_sl_pct,
            }
        )
    return serialized_rows


def _parse_artifact_pin(*, row: Mapping[str, Any]) -> BacktestJobArtifactPin | None:
    """
    Parse nullable artifact pin columns into immutable domain pin metadata.

    Args:
        row: SQL row mapping from `backtest_jobs`.
    Returns:
        BacktestJobArtifactPin | None: Parsed pin metadata or `None` when all fields are null.
    Assumptions:
        Artifact pin columns are additive and follow all-or-none nullability contract.
    Raises:
        BacktestStorageError: If partial nullable fields or invalid scalar values are present.
    Side Effects:
        None.
    """
    raw_slot = row.get("artifact_slot")
    raw_generation = row.get("artifact_slot_generation")
    raw_manifest_hash = row.get("artifact_manifest_hash")
    raw_asof_date = row.get("artifact_asof_date")
    if all(item is None for item in (raw_slot, raw_generation, raw_manifest_hash, raw_asof_date)):
        return None
    if any(item is None for item in (raw_slot, raw_generation, raw_manifest_hash, raw_asof_date)):
        raise BacktestStorageError("backtest_jobs artifact pin columns must be all null or all set")
    assert raw_generation is not None
    assert raw_slot is not None
    assert raw_manifest_hash is not None
    assert raw_asof_date is not None
    return BacktestJobArtifactPin(
        artifact_slot=cast(BacktestArtifactSlotLiteral, str(raw_slot)),
        artifact_slot_generation=int(raw_generation),
        artifact_manifest_hash=str(raw_manifest_hash),
        artifact_asof_date=str(raw_asof_date),
    )


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


def _normalize_storage_datetime_utc(
    *,
    value: Any,
    field_name: str,
    required: bool = True,
) -> datetime | None:
    """
    Normalize one storage `timestamptz` value into timezone-aware UTC datetime.

    Args:
        value: Raw storage value from psycopg row mapping.
        field_name: Storage column name for diagnostic errors.
        required: Whether the column must be present and non-null.
    Returns:
        datetime | None: UTC-normalized datetime or `None` for nullable fields.
    Assumptions:
        Storage layer may deserialize `timestamptz` in session timezone rather than UTC.
    Raises:
        BacktestStorageError: If value is missing, null for required columns, or not datetime-like.
    Side Effects:
        None.
    """
    if value is None:
        if required:
            raise BacktestStorageError(f"backtest_jobs.{field_name} must be non-null datetime")
        return None
    if not isinstance(value, datetime):
        raise BacktestStorageError(
            f"backtest_jobs.{field_name} must be datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None:
        raise BacktestStorageError(
            f"backtest_jobs.{field_name} must be timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


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


def _parse_execution_mode(*, value: Any) -> BacktestJobExecutionMode | None:
    """
    Parse and validate nullable persisted-run execution mode literal.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    Args:
        value: Raw storage execution mode value.
    Returns:
        BacktestJobExecutionMode | None: Typed execution mode literal or `None`.
    Assumptions:
        Legacy rows may keep persisted-run metadata columns null during additive rollout.
    Raises:
        BacktestStorageError: If value is unknown.
    Side Effects:
        None.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"sync_inline", "background_auto", "background_manual_legacy"}:
        raise BacktestStorageError(
            f"Unexpected backtest job execution_mode value: {normalized!r}"
        )
    return cast(BacktestJobExecutionMode, normalized)


def _json_dumps(*, payload: Any) -> str | None:
    """
    Serialize optional JSON-compatible payload into canonical JSON string.

    Args:
        payload: Optional JSON-compatible payload.
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
    return json.dumps(
        _normalize_json_payload_for_dumps(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_json_payload_for_dumps(*, value: Any) -> Any:
    """
    Normalize immutable mapping/sequence wrappers into `json.dumps`-compatible builtins.

    Args:
        value: Raw JSON-compatible payload possibly containing mapping proxies or tuples.
    Returns:
        Any: Builtin dict/list/scalar tree accepted by `json.dumps`.
    Assumptions:
        Non-finite numeric scalars are normalized to `None` so persisted payloads stay finite and
        JSON-safe without silently rewriting values into misleading finite sentinels.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_payload_for_dumps(value=item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_json_payload_for_dumps(value=item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Real):
        normalized_numeric = float(value)
        if not math.isfinite(normalized_numeric):
            return None
        return normalized_numeric
    return value


__all__ = ["PostgresBacktestJobRepository"]
