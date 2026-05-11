from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigEvent,
    BacktestAiConfigJob,
    BacktestAiConfigJobState,
    BacktestAiConfigLlmAttempt,
    BacktestAiConfigTerminalState,
    BacktestAiQuotaEvent,
    BacktestAiTrainingExportRecord,
)
from trading.contexts.backtest.application.ports import (
    BacktestAiConfigJobRepository,
    BacktestAiConfigLeaseRepository,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.shared_kernel.primitives import UserId

_AI_CONFIG_JOB_SELECT_COLUMNS = """
    job_id,
    idempotency_key,
    owner_user_id,
    mode,
    locale,
    state,
    source_page,
    user_prompt_text,
    user_prompt_hash,
    current_config_hash,
    current_config_json,
    validated_config_json,
    assistant_message,
    suggestions_json,
    validation_errors_json,
    model_id,
    model_path_hash,
    system_prompt_version,
    system_prompt_hash,
    catalog_snapshot_hash,
    runtime_defaults_hash,
    queued_at,
    started_at,
    finished_at,
    updated_at,
    locked_by,
    locked_at,
    lease_expires_at,
    heartbeat_at,
    attempt,
    quota_charged,
    applied_at,
    user_feedback_json,
    last_error,
    last_error_json
"""
_AI_CONFIG_JOB_RETURNING_COLUMNS_FOR_JOBS_ALIAS = "\n".join(
    f"    jobs.{column.strip()}"
    for column in _AI_CONFIG_JOB_SELECT_COLUMNS.strip().splitlines()
)

_ACTIVE_STATES = ("queued", "running", "repairing")
_OWNER_ACTIVE_STATES = ("running", "repairing")
_TERMINAL_STATES = {
    "ready",
    "needs_clarification",
    "blocked_by_policy",
    "input_too_large",
    "security_review",
    "failed",
    "cancelled",
}


class PostgresBacktestAiConfigRepository(
    BacktestAiConfigJobRepository,
    BacktestAiConfigLeaseRepository,
):
    """
    SQL adapter for Backtest AI configurator queue, audit, quota and lease storage.
    """

    def __init__(
        self,
        *,
        gateway: BacktestPostgresGateway,
        jobs_table: str = "backtest_ai_config_jobs",
        events_table: str = "backtest_ai_config_events",
        llm_attempts_table: str = "backtest_ai_config_llm_attempts",
        quota_events_table: str = "backtest_ai_quota_events",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresBacktestAiConfigRepository requires gateway")
        self._gateway = gateway
        self._jobs_table = _table_name(jobs_table, field_name="jobs_table")
        self._events_table = _table_name(events_table, field_name="events_table")
        self._llm_attempts_table = _table_name(
            llm_attempts_table,
            field_name="llm_attempts_table",
        )
        self._quota_events_table = _table_name(
            quota_events_table,
            field_name="quota_events_table",
        )

    def create_with_quota_event(
        self,
        *,
        job: BacktestAiConfigJob,
        event: BacktestAiConfigEvent,
        quota_event: BacktestAiQuotaEvent,
    ) -> BacktestAiConfigJob:
        query = f"""
        WITH inserted_job AS (
            INSERT INTO {self._jobs_table}
            (
                job_id,
                idempotency_key,
                owner_user_id,
                mode,
                locale,
                state,
                source_page,
                user_prompt_text,
                user_prompt_hash,
                current_config_hash,
                current_config_json,
                validated_config_json,
                assistant_message,
                suggestions_json,
                validation_errors_json,
                model_id,
                model_path_hash,
                system_prompt_version,
                system_prompt_hash,
                catalog_snapshot_hash,
                runtime_defaults_hash,
                queued_at,
                started_at,
                finished_at,
                updated_at,
                locked_by,
                locked_at,
                lease_expires_at,
                heartbeat_at,
                attempt,
                quota_charged,
                applied_at,
                user_feedback_json,
                last_error,
                last_error_json
            )
            VALUES
            (
                %(job_id)s,
                %(idempotency_key)s,
                %(owner_user_id)s,
                %(mode)s,
                %(locale)s,
                %(state)s,
                %(source_page)s,
                %(user_prompt_text)s,
                %(user_prompt_hash)s,
                %(current_config_hash)s,
                %(current_config_json)s::jsonb,
                %(validated_config_json)s::jsonb,
                %(assistant_message)s,
                %(suggestions_json)s::jsonb,
                %(validation_errors_json)s::jsonb,
                %(model_id)s,
                %(model_path_hash)s,
                %(system_prompt_version)s,
                %(system_prompt_hash)s,
                %(catalog_snapshot_hash)s,
                %(runtime_defaults_hash)s,
                %(queued_at)s,
                %(started_at)s,
                %(finished_at)s,
                %(updated_at)s,
                %(locked_by)s,
                %(locked_at)s,
                %(lease_expires_at)s,
                %(heartbeat_at)s,
                %(attempt)s,
                %(quota_charged)s,
                %(applied_at)s,
                %(user_feedback_json)s::jsonb,
                %(last_error)s,
                %(last_error_json)s::jsonb
            )
            RETURNING
                {_AI_CONFIG_JOB_SELECT_COLUMNS}
        ),
        inserted_event AS (
            INSERT INTO {self._events_table}
            (
                event_id,
                job_id,
                owner_user_id,
                event_name,
                message,
                payload_json,
                created_at
            )
            VALUES
            (
                %(event_id)s,
                %(job_id)s,
                %(owner_user_id)s,
                %(event_name)s,
                %(event_message)s,
                %(event_payload_json)s::jsonb,
                %(event_created_at)s
            )
        ),
        inserted_quota_event AS (
            INSERT INTO {self._quota_events_table}
            (
                quota_event_id,
                job_id,
                owner_user_id,
                paid_level,
                quota_action,
                units,
                idempotency_key,
                reason,
                metadata_json,
                occurred_at
            )
            VALUES
            (
                %(quota_event_id)s,
                %(job_id)s,
                %(owner_user_id)s,
                %(paid_level)s,
                %(quota_action)s,
                %(quota_units)s,
                %(quota_idempotency_key)s,
                %(quota_reason)s,
                %(quota_metadata_json)s::jsonb,
                %(quota_occurred_at)s
            )
        )
        SELECT
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        FROM inserted_job
        """
        parameters = _job_parameters(job=job)
        parameters.update(_event_parameters(event=event))
        parameters.update(_quota_event_parameters(event=quota_event))
        row = self._gateway.fetch_one(query=query, parameters=parameters)
        if row is None:
            raise BacktestStorageError(
                "PostgresBacktestAiConfigRepository.create_with_quota_event returned no row"
            )
        return _map_job_row(row=row)

    def get(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId | None = None,
    ) -> BacktestAiConfigJob | None:
        owner_filter = ""
        parameters: dict[str, Any] = {"job_id": str(job_id)}
        if owner_user_id is not None:
            owner_filter = "AND owner_user_id = %(owner_user_id)s"
            parameters["owner_user_id"] = str(owner_user_id)
        query = f"""
        SELECT
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        FROM {self._jobs_table}
        WHERE job_id = %(job_id)s
          {owner_filter}
        LIMIT 1
        """
        row = self._gateway.fetch_one(query=query, parameters=parameters)
        if row is None:
            return None
        return _map_job_row(row=row)

    def find_by_idempotency_key(
        self,
        *,
        owner_user_id: UserId,
        idempotency_key: str,
    ) -> BacktestAiConfigJob | None:
        query = f"""
        SELECT
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        FROM {self._jobs_table}
        WHERE owner_user_id = %(owner_user_id)s
          AND idempotency_key = %(idempotency_key)s
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "idempotency_key": idempotency_key,
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def record_quota_event(self, *, event: BacktestAiQuotaEvent) -> None:
        query = f"""
        INSERT INTO {self._quota_events_table}
        (
            quota_event_id,
            job_id,
            owner_user_id,
            paid_level,
            quota_action,
            units,
            idempotency_key,
            reason,
            metadata_json,
            occurred_at
        )
        VALUES
        (
            %(quota_event_id)s,
            %(quota_job_id)s,
            %(quota_owner_user_id)s,
            %(paid_level)s,
            %(quota_action)s,
            %(quota_units)s,
            %(quota_idempotency_key)s,
            %(quota_reason)s,
            %(quota_metadata_json)s::jsonb,
            %(quota_occurred_at)s
        )
        """
        self._gateway.execute(query=query, parameters=_quota_event_parameters(event=event))

    def append_event(self, *, event: BacktestAiConfigEvent) -> None:
        query = f"""
        INSERT INTO {self._events_table}
        (
            event_id,
            job_id,
            owner_user_id,
            event_name,
            message,
            payload_json,
            created_at
        )
        VALUES
        (
            %(event_id)s,
            %(event_job_id)s,
            %(event_owner_user_id)s,
            %(event_name)s,
            %(event_message)s,
            %(event_payload_json)s::jsonb,
            %(event_created_at)s
        )
        """
        self._gateway.execute(query=query, parameters=_event_parameters(event=event))

    def record_llm_attempt(self, *, attempt: BacktestAiConfigLlmAttempt) -> None:
        query = f"""
        INSERT INTO {self._llm_attempts_table}
        (
            attempt_id,
            job_id,
            owner_user_id,
            attempt_no,
            attempt_kind,
            prompt_profile,
            system_prompt_version,
            system_prompt_hash,
            user_prompt_text,
            catalog_subset_json,
            raw_model_response,
            parsed_json_draft,
            validation_errors_json,
            input_tokens_estimate,
            output_tokens_estimate,
            latency_ms,
            finish_reason,
            success,
            failure_reason,
            created_at
        )
        VALUES
        (
            %(attempt_id)s,
            %(attempt_job_id)s,
            %(attempt_owner_user_id)s,
            %(attempt_no)s,
            %(attempt_kind)s,
            %(prompt_profile)s,
            %(attempt_system_prompt_version)s,
            %(attempt_system_prompt_hash)s,
            %(attempt_user_prompt_text)s,
            %(catalog_subset_json)s::jsonb,
            %(raw_model_response)s,
            %(parsed_json_draft)s::jsonb,
            %(attempt_validation_errors_json)s::jsonb,
            %(input_tokens_estimate)s,
            %(output_tokens_estimate)s,
            %(latency_ms)s,
            %(finish_reason)s,
            %(success)s,
            %(failure_reason)s,
            %(attempt_created_at)s
        )
        """
        self._gateway.execute(query=query, parameters=_llm_attempt_parameters(attempt=attempt))

    def list_events(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConfigEvent, ...]:
        query = f"""
        SELECT
            event_id,
            job_id,
            owner_user_id,
            event_name,
            message,
            payload_json,
            created_at
        FROM {self._events_table}
        WHERE job_id = %(job_id)s
          AND owner_user_id = %(owner_user_id)s
        ORDER BY event_seq ASC, created_at ASC, event_id ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={
                "job_id": str(job_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        return tuple(_map_event_row(row=row) for row in rows)

    def record_feedback(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
        applied: bool,
        feedback_json: Mapping[str, object],
        now: datetime,
    ) -> BacktestAiConfigJob | None:
        query = f"""
        UPDATE {self._jobs_table}
        SET
            applied_at = CASE WHEN %(applied)s THEN %(now)s ELSE applied_at END,
            user_feedback_json = %(user_feedback_json)s::jsonb,
            updated_at = %(now)s
        WHERE job_id = %(job_id)s
          AND owner_user_id = %(owner_user_id)s
        RETURNING
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "owner_user_id": str(owner_user_id),
                "applied": applied,
                "now": now,
                "user_feedback_json": _json_dumps(feedback_json),
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def count_quota_events(
        self,
        *,
        owner_user_id: UserId,
        occurred_after: datetime,
    ) -> int:
        query = f"""
        SELECT count(*) AS count
        FROM {self._quota_events_table}
        WHERE owner_user_id = %(owner_user_id)s
          AND quota_action = 'request_charged'
          AND units > 0
          AND occurred_at >= %(occurred_after)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "occurred_after": occurred_after,
            },
        )
        return 0 if row is None else int(row["count"])

    def count_queued_for_user(self, *, owner_user_id: UserId) -> int:
        return self._count_jobs(
            where_clause="owner_user_id = %(owner_user_id)s AND state = 'queued'",
            parameters={"owner_user_id": str(owner_user_id)},
        )

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        return self._count_jobs(
            where_clause=(
                "owner_user_id = %(owner_user_id)s "
                "AND state = ANY(%(states)s)"
            ),
            parameters={
                "owner_user_id": str(owner_user_id),
                "states": list(_OWNER_ACTIVE_STATES),
            },
        )

    def count_active_global(self) -> int:
        return self._count_jobs(
            where_clause="state = ANY(%(states)s)",
            parameters={"states": list(_ACTIVE_STATES)},
        )

    def count_jobs_by_state(self, *, state: BacktestAiConfigJobState) -> int:
        return self._count_jobs(
            where_clause="state = %(state)s",
            parameters={"state": state},
        )

    def list_training_export_records(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[BacktestAiTrainingExportRecord, ...]:
        if limit is not None and limit <= 0:
            raise BacktestStorageError("training export limit must be > 0")
        limit_clause = "" if limit is None else "LIMIT %(limit)s"
        parameters: dict[str, Any] = {
            "states": [
                "ready",
                "needs_clarification",
                "blocked_by_policy",
                "security_review",
                "failed",
            ]
        }
        if limit is not None:
            parameters["limit"] = limit
        jobs_query = f"""
        SELECT
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        FROM {self._jobs_table}
        WHERE state = ANY(%(states)s)
        ORDER BY finished_at ASC NULLS LAST, queued_at ASC, job_id ASC
        {limit_clause}
        """
        job_rows = self._gateway.fetch_all(query=jobs_query, parameters=parameters)
        jobs = tuple(_map_job_row(row=row) for row in job_rows)
        if not jobs:
            return ()
        attempt_rows = self._gateway.fetch_all(
            query=f"""
            SELECT
                attempt_id,
                job_id,
                owner_user_id,
                attempt_no,
                attempt_kind,
                prompt_profile,
                system_prompt_version,
                system_prompt_hash,
                user_prompt_text,
                catalog_subset_json,
                raw_model_response,
                parsed_json_draft,
                validation_errors_json,
                input_tokens_estimate,
                output_tokens_estimate,
                latency_ms,
                finish_reason,
                success,
                failure_reason,
                created_at
            FROM {self._llm_attempts_table}
            WHERE job_id = ANY(%(job_ids)s)
            ORDER BY job_id ASC, attempt_no ASC, created_at ASC
            """,
            parameters={"job_ids": [str(job.job_id) for job in jobs]},
        )
        attempts_by_job: dict[UUID, list[BacktestAiConfigLlmAttempt]] = {
            job.job_id: [] for job in jobs
        }
        for row in attempt_rows:
            attempt = _map_llm_attempt_row(row=row)
            attempts_by_job.setdefault(attempt.job_id, []).append(attempt)
        return tuple(
            BacktestAiTrainingExportRecord(
                job=job,
                attempts=tuple(attempts_by_job.get(job.job_id, ())),
            )
            for job in jobs
        )

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> BacktestAiConfigJob | None:
        owner = _normalize_locked_by(value=locked_by)
        if lease_seconds <= 0:
            raise BacktestStorageError("lease_seconds must be > 0")
        if max_attempts <= 0:
            raise BacktestStorageError("max_attempts must be > 0")
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        query = f"""
        WITH expired_attempts AS (
            UPDATE {self._jobs_table}
            SET
                state = 'failed',
                finished_at = %(now)s,
                updated_at = %(now)s,
                locked_by = NULL,
                locked_at = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                assistant_message = COALESCE(
                    assistant_message,
                    'AI configurator request could not be completed after retries.'
                ),
                last_error = COALESCE(last_error, 'lease_attempt_limit_exceeded'),
                last_error_json = COALESCE(
                    last_error_json,
                    %(lease_attempt_limit_exceeded_json)s::jsonb
                )
            WHERE state IN ('running', 'repairing')
              AND lease_expires_at <= %(now)s
              AND attempt >= %(max_attempts)s
            RETURNING job_id
        ),
        queued_candidate AS (
            SELECT job_id
            FROM {self._jobs_table}
            WHERE state = 'queued'
              AND source_page = 'backtests'
              AND attempt < %(max_attempts)s
            ORDER BY queued_at ASC, job_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ),
        reclaim_candidate AS (
            SELECT job_id
            FROM {self._jobs_table}
            WHERE state IN ('running', 'repairing')
              AND source_page = 'backtests'
              AND lease_expires_at <= %(now)s
              AND attempt < %(max_attempts)s
            ORDER BY lease_expires_at ASC, queued_at ASC, job_id ASC
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
                state = CASE
                    WHEN jobs.state = 'repairing' THEN 'repairing'
                    ELSE 'running'
                END,
                started_at = COALESCE(jobs.started_at, %(now)s),
                updated_at = %(now)s,
                locked_by = %(locked_by)s,
                locked_at = %(now)s,
                lease_expires_at = %(lease_expires_at)s,
                heartbeat_at = %(now)s,
                attempt = jobs.attempt + 1
            FROM candidate
            WHERE jobs.job_id = candidate.job_id
            RETURNING
                {_AI_CONFIG_JOB_RETURNING_COLUMNS_FOR_JOBS_ALIAS}
        )
        SELECT
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        FROM claimed
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "now": now,
                "locked_by": owner,
                "lease_expires_at": lease_expires_at,
                "max_attempts": max_attempts,
                "lease_attempt_limit_exceeded_json": json.dumps(
                    {"code": "lease_attempt_limit_exceeded"}
                ),
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
    ) -> BacktestAiConfigJob | None:
        owner = _normalize_locked_by(value=locked_by)
        if lease_seconds <= 0:
            raise BacktestStorageError("lease_seconds must be > 0")
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        query = f"""
        UPDATE {self._jobs_table}
        SET
            updated_at = %(now)s,
            heartbeat_at = %(now)s,
            lease_expires_at = %(lease_expires_at)s
        WHERE job_id = %(job_id)s
          AND state IN ('running', 'repairing')
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": owner,
                "lease_expires_at": lease_expires_at,
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
        next_state: BacktestAiConfigTerminalState,
        assistant_message: str | None = None,
        validated_config_json: dict[str, object] | None = None,
        suggestions_json: tuple[dict[str, object], ...] = (),
        validation_errors_json: tuple[dict[str, object], ...] = (),
        model_id: str | None = None,
        model_path_hash: str | None = None,
        last_error: str | None = None,
        last_error_json: dict[str, object] | None = None,
    ) -> BacktestAiConfigJob | None:
        owner = _normalize_locked_by(value=locked_by)
        normalized_state = str(next_state).strip().lower()
        if normalized_state not in _TERMINAL_STATES:
            raise BacktestStorageError(
                "PostgresBacktestAiConfigRepository.finish requires terminal state"
            )
        query = f"""
        UPDATE {self._jobs_table}
        SET
            state = %(next_state)s,
            assistant_message = %(assistant_message)s,
            validated_config_json = %(validated_config_json)s::jsonb,
            suggestions_json = %(suggestions_json)s::jsonb,
            validation_errors_json = %(validation_errors_json)s::jsonb,
            model_id = %(model_id)s,
            model_path_hash = %(model_path_hash)s,
            finished_at = %(now)s,
            updated_at = %(now)s,
            locked_by = NULL,
            locked_at = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            last_error = %(last_error)s,
            last_error_json = %(last_error_json)s::jsonb
        WHERE job_id = %(job_id)s
          AND state IN ('running', 'repairing')
          AND locked_by = %(locked_by)s
          AND lease_expires_at > %(now)s
        RETURNING
            {_AI_CONFIG_JOB_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "job_id": str(job_id),
                "now": now,
                "locked_by": owner,
                "next_state": normalized_state,
                "assistant_message": assistant_message,
                "validated_config_json": _json_dumps_optional(validated_config_json),
                "suggestions_json": _json_dumps(list(suggestions_json)),
                "validation_errors_json": _json_dumps(list(validation_errors_json)),
                "model_id": model_id,
                "model_path_hash": model_path_hash,
                "last_error": last_error,
                "last_error_json": _json_dumps_optional(last_error_json),
            },
        )
        if row is None:
            return None
        return _map_job_row(row=row)

    def _count_jobs(
        self,
        *,
        where_clause: str,
        parameters: Mapping[str, Any],
    ) -> int:
        query = f"""
        SELECT count(*) AS count
        FROM {self._jobs_table}
        WHERE {where_clause}
        """
        row = self._gateway.fetch_one(query=query, parameters=parameters)
        return 0 if row is None else int(row["count"])


def _table_name(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if not normalized.replace("_", "").isalnum():
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _job_parameters(*, job: BacktestAiConfigJob) -> dict[str, Any]:
    return {
        "job_id": str(job.job_id),
        "idempotency_key": job.idempotency_key,
        "owner_user_id": str(job.owner_user_id),
        "mode": job.mode,
        "locale": job.locale,
        "state": job.state,
        "source_page": job.source_page,
        "user_prompt_text": job.user_prompt_text,
        "user_prompt_hash": job.user_prompt_hash,
        "current_config_hash": job.current_config_hash,
        "current_config_json": _json_dumps_optional(job.current_config_json),
        "validated_config_json": _json_dumps_optional(job.validated_config_json),
        "assistant_message": job.assistant_message,
        "suggestions_json": _json_dumps([dict(item) for item in job.suggestions_json]),
        "validation_errors_json": _json_dumps(
            [dict(item) for item in job.validation_errors_json]
        ),
        "model_id": job.model_id,
        "model_path_hash": job.model_path_hash,
        "system_prompt_version": job.system_prompt_version,
        "system_prompt_hash": job.system_prompt_hash,
        "catalog_snapshot_hash": job.catalog_snapshot_hash,
        "runtime_defaults_hash": job.runtime_defaults_hash,
        "queued_at": job.queued_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "updated_at": job.updated_at,
        "locked_by": job.locked_by,
        "locked_at": job.locked_at,
        "lease_expires_at": job.lease_expires_at,
        "heartbeat_at": job.heartbeat_at,
        "attempt": job.attempt,
        "quota_charged": job.quota_charged,
        "applied_at": job.applied_at,
        "user_feedback_json": _json_dumps_optional(job.user_feedback_json),
        "last_error": job.last_error,
        "last_error_json": _json_dumps_optional(job.last_error_json),
    }


def _event_parameters(*, event: BacktestAiConfigEvent) -> dict[str, Any]:
    return {
        "event_id": str(event.event_id),
        "event_job_id": str(event.job_id),
        "event_owner_user_id": str(event.owner_user_id),
        "event_name": event.event_name,
        "event_message": event.message,
        "event_payload_json": _json_dumps(event.payload_json),
        "event_created_at": event.created_at,
    }


def _quota_event_parameters(*, event: BacktestAiQuotaEvent) -> dict[str, Any]:
    return {
        "quota_event_id": str(event.quota_event_id),
        "quota_job_id": None if event.job_id is None else str(event.job_id),
        "quota_owner_user_id": str(event.owner_user_id),
        "paid_level": str(event.paid_level),
        "quota_action": event.quota_action,
        "quota_units": event.units,
        "quota_idempotency_key": event.idempotency_key,
        "quota_reason": event.reason,
        "quota_metadata_json": _json_dumps(event.metadata_json),
        "quota_occurred_at": event.occurred_at,
    }


def _llm_attempt_parameters(*, attempt: BacktestAiConfigLlmAttempt) -> dict[str, Any]:
    return {
        "attempt_id": str(attempt.attempt_id),
        "attempt_job_id": str(attempt.job_id),
        "attempt_owner_user_id": str(attempt.owner_user_id),
        "attempt_no": attempt.attempt_no,
        "attempt_kind": attempt.attempt_kind,
        "prompt_profile": attempt.prompt_profile,
        "attempt_system_prompt_version": attempt.system_prompt_version,
        "attempt_system_prompt_hash": attempt.system_prompt_hash,
        "attempt_user_prompt_text": attempt.user_prompt_text,
        "catalog_subset_json": _json_dumps(attempt.catalog_subset_json),
        "raw_model_response": attempt.raw_model_response,
        "parsed_json_draft": _json_dumps_optional(attempt.parsed_json_draft),
        "attempt_validation_errors_json": _json_dumps(
            [dict(item) for item in attempt.validation_errors_json]
        ),
        "input_tokens_estimate": attempt.input_tokens_estimate,
        "output_tokens_estimate": attempt.output_tokens_estimate,
        "latency_ms": attempt.latency_ms,
        "finish_reason": attempt.finish_reason,
        "success": attempt.success,
        "failure_reason": attempt.failure_reason,
        "attempt_created_at": attempt.created_at,
    }


def _map_job_row(*, row: Mapping[str, Any]) -> BacktestAiConfigJob:
    try:
        return BacktestAiConfigJob(
            job_id=UUID(str(row["job_id"])),
            idempotency_key=_optional_str(row["idempotency_key"]),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            mode=cast(Any, str(row["mode"])),
            locale=cast(Any, str(row["locale"])),
            state=_job_state(row["state"]),
            source_page=str(row["source_page"]),
            user_prompt_text=str(row["user_prompt_text"]),
            user_prompt_hash=str(row["user_prompt_hash"]),
            current_config_hash=_optional_str(row["current_config_hash"]),
            current_config_json=_json_mapping(row["current_config_json"]),
            validated_config_json=_json_mapping(row["validated_config_json"]),
            assistant_message=_optional_str(row["assistant_message"]),
            suggestions_json=_json_tuple(row["suggestions_json"]),
            validation_errors_json=_json_tuple(row["validation_errors_json"]),
            model_id=_optional_str(row["model_id"]),
            model_path_hash=_optional_str(row["model_path_hash"]),
            system_prompt_version=str(row["system_prompt_version"]),
            system_prompt_hash=str(row["system_prompt_hash"]),
            catalog_snapshot_hash=str(row["catalog_snapshot_hash"]),
            runtime_defaults_hash=str(row["runtime_defaults_hash"]),
            queued_at=_datetime(row["queued_at"]),
            started_at=_optional_datetime(row["started_at"]),
            finished_at=_optional_datetime(row["finished_at"]),
            updated_at=_datetime(row["updated_at"]),
            locked_by=_optional_str(row["locked_by"]),
            locked_at=_optional_datetime(row["locked_at"]),
            lease_expires_at=_optional_datetime(row["lease_expires_at"]),
            heartbeat_at=_optional_datetime(row["heartbeat_at"]),
            attempt=int(row["attempt"]),
            quota_charged=bool(row["quota_charged"]),
            applied_at=_optional_datetime(row["applied_at"]),
            user_feedback_json=_json_mapping(row["user_feedback_json"]),
            last_error=_optional_str(row["last_error"]),
            last_error_json=_json_mapping(row["last_error_json"]),
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError("Cannot map backtest AI config job row") from error


def _map_event_row(*, row: Mapping[str, Any]) -> BacktestAiConfigEvent:
    try:
        return BacktestAiConfigEvent(
            event_id=UUID(str(row["event_id"])),
            job_id=UUID(str(row["job_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            event_name=cast(Any, str(row["event_name"])),
            message=str(row["message"]),
            payload_json=_json_mapping_required(row["payload_json"]),
            created_at=_datetime(row["created_at"]),
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError("Cannot map backtest AI config event row") from error


def _map_llm_attempt_row(*, row: Mapping[str, Any]) -> BacktestAiConfigLlmAttempt:
    try:
        return BacktestAiConfigLlmAttempt(
            attempt_id=UUID(str(row["attempt_id"])),
            job_id=UUID(str(row["job_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            attempt_no=int(row["attempt_no"]),
            attempt_kind=cast(Any, str(row["attempt_kind"])),
            prompt_profile=str(row["prompt_profile"]),
            system_prompt_version=str(row["system_prompt_version"]),
            system_prompt_hash=str(row["system_prompt_hash"]),
            user_prompt_text=str(row["user_prompt_text"]),
            catalog_subset_json=_json_mapping_required(row["catalog_subset_json"]),
            raw_model_response=_optional_str(row["raw_model_response"]),
            parsed_json_draft=_json_mapping(row["parsed_json_draft"]),
            validation_errors_json=_json_tuple(row["validation_errors_json"]),
            input_tokens_estimate=_optional_int(row["input_tokens_estimate"]),
            output_tokens_estimate=_optional_int(row["output_tokens_estimate"]),
            latency_ms=_optional_int(row["latency_ms"]),
            finish_reason=_optional_str(row["finish_reason"]),
            success=bool(row["success"]),
            failure_reason=_optional_str(row["failure_reason"]),
            created_at=_datetime(row["created_at"]),
        )
    except Exception as error:  # noqa: BLE001
        raise BacktestStorageError("Cannot map backtest AI config LLM attempt row") from error


def _job_state(value: Any) -> BacktestAiConfigJobState:
    normalized = str(value).strip().lower()
    if normalized not in _ACTIVE_STATES and normalized not in _TERMINAL_STATES:
        raise BacktestStorageError(f"Unexpected AI config job state: {normalized!r}")
    return cast(BacktestAiConfigJobState, normalized)


def _normalize_locked_by(*, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise BacktestStorageError("locked_by must be non-empty")
    return normalized


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        _json_serializable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _json_dumps_optional(payload: Any | None) -> str | None:
    if payload is None:
        return None
    return _json_dumps(payload)


def _json_serializable(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {str(key): _json_serializable(value) for key, value in payload.items()}
    if isinstance(payload, tuple):
        return [_json_serializable(value) for value in payload]
    if isinstance(payload, list):
        return [_json_serializable(value) for value in payload]
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        return [_json_serializable(value) for value in payload]
    return payload


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise BacktestStorageError("Expected JSON object column")
    return dict(value)


def _json_mapping_required(value: Any) -> Mapping[str, Any]:
    result = _json_mapping(value)
    if result is None:
        raise BacktestStorageError("Expected required JSON object column")
    return result


def _json_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise BacktestStorageError("Expected JSON array column")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BacktestStorageError("Expected JSON array of objects")
        result.append(dict(item))
    return tuple(result)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise BacktestStorageError("Expected datetime column")
    return value


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


__all__ = ["PostgresBacktestAiConfigRepository"]
