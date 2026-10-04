from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Callable, Iterator, Mapping
from uuid import UUID, uuid4, uuid5

import psycopg
from psycopg.errors import UniqueViolation

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.market_data.application.services.work_requests import WorkRequest


class ActiveWorkRequestError(ValueError):
    """Admission rejected duplicate work or a full bounded queue."""

    def __init__(self, code: str = "duplicate_work") -> None:
        self.code = code
        super().__init__(code)


class PostgresMarketDataWorkRequestRepository:
    """Durable request inbox; atomic claim, fenced updates, checkpointed pause and recovery."""

    def __init__(self, *, gateway: BacktestPostgresGateway,
                 execution_dsn: str | None = None,
                 execution_lock_key: int = 744948345610) -> None:
        self._gateway = gateway
        self._execution_dsn = execution_dsn
        self._execution_lock_key = execution_lock_key

    @contextmanager
    def execution(self) -> Iterator[Callable[[], None] | None]:
        """One live coordinator and its bounded worker pool, including stalled I/O.

        The yielded check must precede each persistence side effect. A lost lock
        connection fails closed. Already in-flight ClickHouse inserts can finish;
        the canonical reader and the existing REST fill deduplicate minute keys.
        """
        if self._execution_dsn is None:
            raise RuntimeError("execution coordination DSN is required")
        with psycopg.connect(self._execution_dsn, autocommit=True) as connection:
            row = connection.execute("SELECT pg_try_advisory_lock(%s)",
                                     (self._execution_lock_key,)).fetchone()
            if row is None or not row[0]:
                yield None
                return
            check_lock = Lock()

            def check() -> None:
                with check_lock:
                    connection.execute("SELECT 1").fetchone()
            try:
                yield check
            finally:
                # Closing this exact session releases the lock even after an error.
                connection.close()

    def backlog(self, *, organization_id: UUID) -> Mapping[str, Any]:
        row = self._gateway.fetch_one(
            query="""
            SELECT count(*) FILTER (WHERE state = 'queued') AS queued,
                   count(*) FILTER (WHERE state = 'running') AS running,
                   count(*) FILTER (WHERE state = 'cancel_requested') AS cancel_requested,
                   count(*) FILTER (WHERE state IN ('paused', 'pause_requested')) AS paused,
                   count(*) FILTER (WHERE state = 'retry_wait') AS retry_wait,
                   min(created_at) FILTER (WHERE state = 'queued') AS oldest_queued_at
            FROM market_data_work_requests
            WHERE organization_id = %(org)s
              AND state IN ('queued', 'running', 'pause_requested', 'paused',
                            'retry_wait', 'cancel_requested')
            """,
            parameters={"org": organization_id},
        )
        if row is None:
            raise RuntimeError("work request backlog read returned no result")
        return row

    def instrument_jobs(
        self, *, organization_id: UUID, market_ids: list[int],
        start_at: datetime, end_at: datetime,
    ) -> tuple[Mapping[str, Any], ...]:
        """Latest request per instrument overlapping the displayed historical range."""
        return self._gateway.fetch_all(
            query="""
            SELECT DISTINCT ON (market_id, symbol) market_id, symbol,
                   job_id, state, completed_units, total_units, position, start_at, end_at
            FROM market_data_work_requests,
                 LATERAL jsonb_array_elements_text(symbols)
                 WITH ORDINALITY AS entry(symbol, position)
            WHERE organization_id = %(org)s AND market_id = ANY(%(markets)s)
              AND kind = 'candle_ingestion'
              AND (state IN ('queued', 'running', 'pause_requested', 'paused',
                             'retry_wait', 'cancel_requested')
                   OR (start_at < %(end)s AND end_at > %(start)s))
            ORDER BY market_id, symbol,
                (state IN ('queued', 'running', 'pause_requested', 'paused',
                            'retry_wait', 'cancel_requested')) DESC,
                updated_at DESC, created_at DESC, job_id DESC
            """,
            parameters={"org": organization_id, "markets": market_ids,
                        "start": start_at, "end": end_at},
        )

    def submit(
        self,
        *,
        organization_id: UUID,
        actor_user_id: UUID,
        key: UUID,
        request: WorkRequest,
        now: datetime,
    ) -> Mapping[str, Any]:
        total = request.validate(now=now)
        params = dict(
            org=organization_id,
            actor=actor_user_id,
            key=key,
            id=uuid4(),
            kind=request.kind,
            market=request.market_id,
            symbols=json.dumps(request.symbols),
            start=request.start_at,
            end=request.end_at,
            total=total,
            now=now,
        )
        existing = self._gateway.fetch_one(
            query="""
            SELECT * FROM market_data_work_requests
            WHERE organization_id = %(org)s AND idempotency_key = %(key)s
            """,
            parameters=params,
        )
        if existing is not None:
            self._assert_same(existing, request)
            return existing
        try:
            row = self._gateway.fetch_one(
                query="""
                WITH authorized AS (
                    SELECT m.user_id FROM identity_memberships m
                    JOIN identity_organizations o ON o.organization_id = m.organization_id
                    WHERE m.organization_id = %(org)s AND m.user_id = %(actor)s
                      AND m.status = 'active' AND o.status = 'active'
                    FOR SHARE OF m, o
                )
                INSERT INTO market_data_work_requests
                    (job_id, organization_id, actor_user_id, idempotency_key, kind,
                     market_id, symbols, start_at, end_at, state, total_units,
                     created_at, updated_at)
                SELECT %(id)s, %(org)s, %(actor)s, %(key)s, %(kind)s, %(market)s,
                        %(symbols)s::jsonb, %(start)s, %(end)s, 'queued', %(total)s,
                        %(now)s, %(now)s FROM authorized
                RETURNING *
                """,
                parameters=params,
            )
        except UniqueViolation as error:
            existing = self._gateway.fetch_one(
                query="""
                SELECT * FROM market_data_work_requests
                WHERE organization_id = %(org)s AND idempotency_key = %(key)s
                """,
                parameters=params,
            )
            if existing is None:
                raise ActiveWorkRequestError(
                    "queue_full" if error.diag.constraint_name == "market_data_work_queue_full"
                    else "duplicate_work"
                ) from error
            self._assert_same(existing, request)
            return existing
        if row is None:
            raise PermissionError("active organization membership is required")
        return row

    @staticmethod
    def _assert_same(row: Mapping[str, Any], request: WorkRequest) -> None:
        if (
            row["kind"],
            row["market_id"],
            tuple(row["symbols"]),
            row["start_at"],
            row["end_at"],
        ) != (request.kind, request.market_id, request.symbols, request.start_at, request.end_at):
            raise ValueError("idempotency key belongs to a different request")

    def get(self, *, organization_id: UUID, job_id: UUID) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            SELECT j.*,
                EXISTS (SELECT 1 FROM market_data_work_requests q
                        WHERE q.job_id <> j.job_id
                          AND (q.state = 'running' OR (q.organization_id=j.organization_id
                              AND (q.state='queued' OR
                                   (q.state='retry_wait' AND q.next_retry_at <= now())))))
                    AS queue_waiting,
                (SELECT next_retry_at FROM market_data_storage_cooldown
                 WHERE next_retry_at > now()) AS storage_retry_at
            FROM market_data_work_requests j
            WHERE j.organization_id = %(org)s AND j.job_id = %(id)s
            """,
            parameters={"org": organization_id, "id": job_id},
        )

    def lookup(self, *, organization_id: UUID, key: UUID) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            SELECT * FROM market_data_work_requests
            WHERE organization_id = %(org)s AND (idempotency_key = %(key)s OR batch_key = %(key)s)
            ORDER BY symbols::text, job_id LIMIT 1
            """,
            parameters={"org": organization_id, "key": key},
        )

    def submit_batch(self, *, organization_id: UUID, actor_user_id: UUID, key: UUID,
                     requests: tuple[WorkRequest, ...], now: datetime
                     ) -> tuple[Mapping[str, Any], ...]:
        if not 1 <= len(requests) <= 50:
            raise ValueError("select 1 to 50 instruments")
        commands = []
        for request in sorted(requests, key=lambda r: (r.market_id, r.symbols)):
            total = request.validate(now=now)
            if request.kind != "candle_ingestion" or len(request.symbols) != 1:
                raise ValueError("each queued download must have one instrument")
            assert request.start_at is not None and request.end_at is not None
            commands.append(dict(
                id=str(uuid5(key, f"{request.market_id}:{request.symbols[0]}")),
                market_id=request.market_id, symbol=request.symbols[0], total_units=total,
                start_at=request.start_at.astimezone(UTC).isoformat(),
                end_at=request.end_at.astimezone(UTC).isoformat(),
            ))
        if len({c["id"] for c in commands}) != len(commands):
            raise ValueError("select distinct instruments")
        try:
            return self._gateway.fetch_all(
                query="SELECT * FROM market_data_submit_batch(%(org)s,%(actor)s,%(key)s,"
                      "%(commands)s::jsonb,%(now)s)",
                parameters=dict(org=organization_id, actor=actor_user_id, key=key,
                                commands=json.dumps(commands), now=now),
            )
        except UniqueViolation as error:
            raise ActiveWorkRequestError(
                "queue_full" if error.diag.constraint_name == "market_data_work_queue_full"
                else "duplicate_work"
            ) from error
        except psycopg.errors.InsufficientPrivilege as error:
            raise PermissionError("active membership required") from error
        except psycopg.errors.InvalidParameterValue as error:
            raise ValueError("batch key belongs to different work") from error

    def page(
        self,
        *,
        organization_id: UUID,
        limit: int,
        before: datetime | None,
        before_id: UUID | None,
        kind: str | None,
        state: str | None,
    ) -> tuple[Mapping[str, Any], ...]:
        return self._gateway.fetch_all(
            query="""
            SELECT * FROM market_data_work_requests WHERE organization_id = %(org)s
              AND (%(kind)s::text IS NULL OR kind = %(kind)s)
              AND (%(state)s::text IS NULL OR state = %(state)s)
              AND (%(before)s::timestamptz IS NULL OR
                   (created_at, job_id) < (%(before)s, %(before_id)s::uuid))
            ORDER BY created_at DESC, job_id DESC LIMIT %(limit)s
            """,
            parameters=dict(
                org=organization_id,
                limit=limit,
                before=before,
                before_id=before_id,
                kind=kind,
                state=state,
            ),
        )

    def pause(
        self, *, organization_id: UUID, job_id: UUID, version: int, now: datetime
    ) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            UPDATE market_data_work_requests
            SET state = CASE WHEN state = 'running' THEN 'pause_requested' ELSE 'paused' END,
                control_version = control_version + 1, updated_at = %(now)s
            WHERE organization_id = %(org)s AND job_id = %(id)s
              AND control_version = %(version)s AND state IN ('queued', 'running', 'retry_wait')
            RETURNING *
            """,
            parameters=dict(org=organization_id, id=job_id, version=version, now=now),
        )

    def resume(
        self, *, organization_id: UUID, job_id: UUID, version: int, now: datetime
    ) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            UPDATE market_data_work_requests
            SET state = CASE WHEN next_retry_at > %(now)s THEN 'retry_wait' ELSE 'queued' END,
                control_version = control_version + 1, progress_epoch = progress_epoch + 1,
                updated_at = %(now)s
            WHERE organization_id = %(org)s AND job_id = %(id)s
              AND control_version = %(version)s AND state = 'paused' RETURNING *
            """,
            parameters=dict(org=organization_id, id=job_id, version=version, now=now),
        )

    def cancel(
        self, *, organization_id: UUID, job_id: UUID, now: datetime
    ) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            UPDATE market_data_work_requests
            SET state = CASE WHEN worker_token IS NULL THEN 'cancelled'
                             ELSE 'cancel_requested' END,
                finished_at = CASE WHEN worker_token IS NULL THEN %(now)s ELSE NULL END,
                next_retry_at = NULL, error_code = NULL,
                control_version = control_version + 1, updated_at = %(now)s
            WHERE organization_id = %(org)s AND job_id = %(id)s
              AND state IN ('queued', 'running', 'pause_requested', 'paused', 'retry_wait')
            RETURNING *
            """,
            parameters=dict(org=organization_id, id=job_id, now=now),
        )

    def retry(
        self, *, organization_id: UUID, job_id: UUID, attempt: int, now: datetime
    ) -> Mapping[str, Any] | None:
        try:
            return self._gateway.fetch_one(
                query="""
                UPDATE market_data_work_requests
                SET state = 'queued', attempt = attempt + 1,
                    error_code = NULL, error_phase = NULL, worker_token = NULL,
                    finished_at = NULL, updated_at = %(now)s,
                    control_version = control_version + 1, progress_epoch = progress_epoch + 1,
                    retry_count = 0, next_retry_at = NULL
                WHERE organization_id = %(org)s AND job_id = %(id)s AND state = 'failed'
                  AND attempt = %(attempt)s AND attempt < 5 RETURNING *
                """,
                parameters=dict(org=organization_id, id=job_id, attempt=attempt, now=now),
            )
        except UniqueViolation as error:
            raise ActiveWorkRequestError(
                "queue_full" if error.diag.constraint_name == "market_data_work_queue_full"
                else "duplicate_work"
            ) from error

    def recover(self, *, now: datetime) -> None:
        """Maintenance once under execution(), before any workers are dispatched."""
        self._gateway.execute(
            query="""DELETE FROM market_data_work_events WHERE event_id IN (
                SELECT event_id FROM market_data_work_events
                WHERE occurred_at < %(now)s::timestamptz - INTERVAL '30 days'
                ORDER BY occurred_at LIMIT 1000)""",
            parameters={"now": now},
        )
        self._gateway.execute(
            query="""
            UPDATE market_data_work_requests
            SET state = CASE WHEN state = 'cancel_requested' THEN 'cancelled'
                             WHEN state = 'pause_requested' THEN 'paused' ELSE 'retry_wait' END,
                worker_token = NULL, updated_at = %(now)s,
                error_code = CASE WHEN state = 'running' THEN 'worker_lost' ELSE NULL END,
                finished_at = CASE WHEN state = 'cancel_requested' THEN %(now)s ELSE NULL END,
                next_retry_at = CASE WHEN state = 'running'
                    THEN %(now)s::timestamptz + INTERVAL '15 seconds' ELSE NULL END,
                progress_epoch = progress_epoch + 1
            WHERE state IN ('running', 'pause_requested', 'cancel_requested')
              AND updated_at < %(now)s::timestamptz - INTERVAL '5 minutes'
            """,
            parameters={"now": now},
        )

    def claim(self, *, now: datetime) -> Mapping[str, Any] | None:
        # The coordinator claims sequentially under execution(). Do not recover
        # here: an old heartbeat can belong to a live slow sibling in this pool.
        return self._gateway.fetch_one(
            query="""
            WITH candidate AS (
                SELECT q.job_id FROM market_data_work_requests q
                WHERE (q.state = 'queued' OR
                       (q.state = 'retry_wait' AND q.next_retry_at <= %(now)s))
                  AND NOT EXISTS (SELECT 1 FROM market_data_storage_cooldown
                                  WHERE next_retry_at > %(now)s)
                  AND NOT EXISTS (
                      SELECT 1 FROM market_data_work_requests active
                      WHERE active.state IN ('running', 'pause_requested', 'cancel_requested')
                        AND active.market_id = q.market_id AND active.kind = q.kind
                        AND (q.kind = 'catalog_refresh' OR (
                            active.symbols ?| ARRAY(SELECT jsonb_array_elements_text(q.symbols))
                            AND active.start_at < q.end_at AND q.start_at < active.end_at)))
                ORDER BY q.updated_at, q.job_id FOR UPDATE OF q SKIP LOCKED LIMIT 1
            )
            UPDATE market_data_work_requests j
            SET state = 'running', worker_token = %(token)s,
                started_at = COALESCE(started_at, %(now)s), next_retry_at = NULL,
                error_code = NULL, error_phase = NULL, updated_at = %(now)s
            FROM candidate c WHERE j.job_id = c.job_id RETURNING j.*
            """,
            parameters={"now": now, "token": uuid4()},
        )

    def checkpoint(
        self,
        *,
        job: Mapping[str, Any],
        units: int,
        rows_read: int,
        rows_written: int,
        now: datetime,
    ) -> str | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE market_data_work_requests
            SET completed_units = %(units)s, rows_read = %(read)s,
                rows_written = %(written)s, updated_at = %(now)s,
                retry_count = CASE WHEN %(units)s > completed_units THEN 0 ELSE retry_count END
            WHERE job_id = %(id)s AND worker_token = %(token)s
              AND state IN ('running', 'pause_requested', 'cancel_requested') RETURNING state
            """,
            parameters=dict(
                id=job["job_id"],
                token=job["worker_token"],
                units=units,
                read=rows_read,
                written=rows_written,
                now=now,
            ),
        )
        return str(row["state"]) if row else None

    def release(self, *, job: Mapping[str, Any], now: datetime) -> None:
        """Yield a checkpointed request so other organizations and metadata can run."""
        self._gateway.execute(
            query="""
            UPDATE market_data_work_requests SET
                state = CASE WHEN state = 'cancel_requested' THEN 'cancelled'
                             WHEN state = 'pause_requested' THEN 'paused' ELSE 'queued' END,
                worker_token = NULL, updated_at = %(now)s,
                finished_at = CASE WHEN state = 'cancel_requested' THEN %(now)s ELSE NULL END,
                attempts = CASE WHEN state = 'cancel_requested' THEN
                    attempts || jsonb_build_array(jsonb_build_object(
                        'attempt', attempt, 'state', 'cancelled', 'error_code', NULL,
                        'finished_at', %(now)s::timestamptz,
                        'rows_read', rows_read, 'rows_written', rows_written)) ELSE attempts END
            WHERE job_id = %(id)s AND worker_token = %(token)s
              AND state IN ('running', 'pause_requested', 'cancel_requested')
            """,
            parameters={"id": job["job_id"], "token": job["worker_token"], "now": now},
        )

    def defer(
        self, *, job: Mapping[str, Any], error_code: str, retry_after_s: float, now: datetime
    ) -> None:
        """Durable backoff; manual pause/cancel wins even during failing provider I/O."""
        self._gateway.execute(
            query="""
            WITH deferred AS (UPDATE market_data_work_requests SET
                state = CASE WHEN state = 'cancel_requested' THEN 'cancelled'
                             WHEN state = 'pause_requested' THEN 'paused' ELSE 'retry_wait' END,
                worker_token = NULL, updated_at = %(now)s,
                finished_at = CASE WHEN state = 'cancel_requested' THEN %(now)s ELSE NULL END,
                error_code = CASE WHEN state = 'cancel_requested' THEN NULL ELSE %(error)s END,
                error_phase = CASE WHEN state = 'cancel_requested' THEN NULL ELSE %(phase)s END,
                next_retry_at = CASE WHEN state = 'cancel_requested' THEN NULL ELSE
                    %(now)s::timestamptz + make_interval(secs => GREATEST(
                        %(after)s, LEAST(300, 15 * power(2, LEAST(retry_count, 5))))) END,
                retry_count = LEAST(retry_count + 1, 1000000),
                progress_epoch = progress_epoch + 1
            WHERE job_id = %(id)s AND worker_token = %(token)s
              AND state IN ('running', 'pause_requested', 'cancel_requested')
            RETURNING next_retry_at)
            INSERT INTO market_data_storage_cooldown (singleton, next_retry_at, error_code)
            SELECT TRUE, COALESCE(next_retry_at, %(now)s::timestamptz + INTERVAL '15 seconds'),
                   %(error)s FROM deferred
            WHERE %(error)s IN ('storage_unavailable', 'storage_memory_pressure')
            ON CONFLICT (singleton) DO UPDATE SET
                next_retry_at = GREATEST(market_data_storage_cooldown.next_retry_at,
                                        EXCLUDED.next_retry_at),
                error_code = EXCLUDED.error_code
            """,
            parameters=dict(id=job["job_id"], token=job["worker_token"],
                            error=error_code, phase=job.get("error_phase"),
                            after=retry_after_s, now=now),
        )

    def finish(
        self, *, job: Mapping[str, Any], state: str, error_code: str | None, now: datetime
    ) -> None:
        if state not in {"succeeded", "cancelled", "failed"}:
            raise ValueError("invalid terminal outcome")
        self._gateway.execute(
            query="""
            UPDATE market_data_work_requests SET
                state = CASE WHEN state = 'cancel_requested' THEN 'cancelled'
                             WHEN state = 'pause_requested' AND %(state)s = 'failed'
                             THEN 'paused' ELSE %(state)s END,
                error_code = CASE WHEN state = 'cancel_requested' THEN NULL ELSE %(error)s END,
                error_phase = CASE WHEN state = 'cancel_requested' THEN NULL ELSE %(phase)s END,
                finished_at = CASE WHEN state = 'pause_requested' AND %(state)s = 'failed'
                                   THEN NULL ELSE %(now)s END, updated_at = %(now)s,
                worker_token = NULL, next_retry_at = NULL,
                attempts = CASE WHEN state = 'pause_requested' AND %(state)s = 'failed'
                    THEN attempts ELSE attempts || jsonb_build_array(jsonb_build_object(
                    'attempt', attempt, 'state',
                    CASE WHEN state = 'cancel_requested' THEN 'cancelled' ELSE %(state)s END,
                    'error_code', CASE WHEN state = 'cancel_requested' THEN NULL
                                      ELSE %(error)s::text END,
                    'finished_at', %(now)s::timestamptz,
                    'rows_read', rows_read, 'rows_written', rows_written)) END
            WHERE job_id = %(id)s AND worker_token = %(token)s
              AND state IN ('running', 'pause_requested', 'cancel_requested')
            """,
            parameters=dict(
                id=job["job_id"], token=job["worker_token"], state=state, error=error_code,
                phase=job.get("error_phase"), now=now
            ),
        )

    def events(self, *, organization_id: UUID, job_id: UUID, before: int | None,
               limit: int) -> tuple[Mapping[str, Any], ...]:
        return self._gateway.fetch_all(
            query="""SELECT event_id,event_type,occurred_at,state,attempt,completed_units,
                            total_units,rows_written,error_code,error_phase,next_retry_at
                     FROM market_data_work_events
                     WHERE organization_id = %(org)s AND job_id = %(id)s
                       AND (%(before)s::bigint IS NULL OR event_id < %(before)s)
                     ORDER BY event_id DESC LIMIT %(limit)s""",
            parameters=dict(org=organization_id, id=job_id, before=before, limit=limit),
        )
