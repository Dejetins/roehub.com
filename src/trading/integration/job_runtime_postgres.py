"""Durable PostgreSQL catalog for isolated jobs and attempts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from trading.integration.job_runtime import JobEnvelope, JobResultManifest


class JobRuntimeCatalogError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ClaimedJobAttempt:
    envelope: JobEnvelope
    worker_id: str
    claimed_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryClaim:
    envelope: JobEnvelope
    outcome: str
    error_code: str
    recovery_owner_id: str


@dataclass(frozen=True, slots=True)
class JobState:
    organization_id: UUID
    job_id: UUID
    status: str
    attempt_count: int
    cancel_requested: bool
    result_artifact_manifest_digest: str | None


class PostgresJobRuntimeCatalog:
    """PostgreSQL is the source of truth; queue transports are optional hints."""

    def __init__(self, *, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PostgresJobRuntimeCatalog requires dsn")
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    def submit(self, *, envelope: JobEnvelope, created_at: datetime) -> UUID:
        payload = envelope.model_dump(mode="json", by_alias=True)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 15))",
                    (f"{envelope.organization_id}:{envelope.semantic_job_key}",),
                )
                cursor.execute(
                    """SELECT job_id, semantic_spec_digest
                       FROM job_runtime_jobs
                       WHERE organization_id = %s AND semantic_job_key = %s
                       FOR UPDATE""",
                    (envelope.organization_id, envelope.semantic_job_key),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["semantic_spec_digest"] != envelope.semantic_spec_digest:
                        raise JobRuntimeCatalogError(code="job.semantic_key_conflict")
                    return cast(UUID, existing["job_id"])
                if envelope.attempt_number != 1:
                    raise JobRuntimeCatalogError(code="job.first_attempt_invalid")
                cursor.execute(
                    """INSERT INTO job_runtime_jobs
                           (organization_id, job_id, semantic_job_key,
                            semantic_spec_digest, capability, status,
                            attempt_count, created_at, updated_at)
                       VALUES (%s,%s,%s,%s,%s,'queued',1,%s,%s)""",
                    (
                        envelope.organization_id,
                        envelope.job_id,
                        envelope.semantic_job_key,
                        envelope.semantic_spec_digest,
                        envelope.capability,
                        created_at,
                        created_at,
                    ),
                )
                self._insert_attempt(cursor=cursor, envelope=envelope, payload=payload)
        except JobRuntimeCatalogError:
            raise
        except psycopg.errors.ForeignKeyViolation as error:
            raise JobRuntimeCatalogError(code="job.organization_not_found") from error
        except psycopg.Error as error:
            raise JobRuntimeCatalogError(code="job.catalog_unavailable") from error
        return envelope.job_id

    def claim_next(self, *, worker_id: str, now: datetime) -> ClaimedJobAttempt | None:
        if not worker_id or len(worker_id) > 128:
            raise JobRuntimeCatalogError(code="job.worker_id_invalid")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                self._expire_queued(cursor=cursor, now=now)
                cursor.execute(
                    """SELECT job.organization_id, job.job_id
                       FROM job_runtime_jobs AS job
                       WHERE job.status = 'queued'
                         AND job.cancel_requested_at IS NULL
                         AND EXISTS (
                             SELECT 1 FROM job_runtime_attempts AS attempt
                             WHERE attempt.organization_id = job.organization_id
                               AND attempt.job_id = job.job_id
                               AND attempt.status = 'queued'
                               AND attempt.deadline > %s
                         )
                       ORDER BY job.organization_id, job.job_id
                       FOR UPDATE OF job SKIP LOCKED
                       LIMIT 1""",
                    (now,),
                )
                job = cursor.fetchone()
                if job is None:
                    return None
                cursor.execute(
                    """SELECT attempt_id, envelope
                       FROM job_runtime_attempts
                       WHERE organization_id = %s AND job_id = %s
                         AND status = 'queued' AND deadline > %s
                       ORDER BY attempt_number
                       FOR UPDATE""",
                    (job["organization_id"], job["job_id"], now),
                )
                row = cursor.fetchone()
                if row is None:
                    raise JobRuntimeCatalogError(code="job.queued_attempt_missing")
                cursor.execute(
                    """UPDATE job_runtime_attempts
                       SET status = 'running', worker_id = %s,
                           claimed_at = %s, heartbeat_at = %s
                       WHERE organization_id = %s AND attempt_id = %s""",
                    (
                        worker_id,
                        now,
                        now,
                        job["organization_id"],
                        row["attempt_id"],
                    ),
                )
                cursor.execute(
                    """UPDATE job_runtime_jobs SET status = 'running', updated_at = %s
                       WHERE organization_id = %s AND job_id = %s""",
                    (now, job["organization_id"], job["job_id"]),
                )
                envelope = JobEnvelope.model_validate(row["envelope"])
        except JobRuntimeCatalogError:
            raise
        except psycopg.Error as error:
            raise JobRuntimeCatalogError(code="job.catalog_unavailable") from error
        return ClaimedJobAttempt(envelope=envelope, worker_id=worker_id, claimed_at=now)

    def heartbeat(
        self,
        *,
        organization_id: UUID,
        attempt_id: UUID,
        worker_id: str,
        now: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE job_runtime_attempts SET heartbeat_at = %s
                   WHERE organization_id = %s AND attempt_id = %s
                     AND status = 'running' AND worker_id = %s""",
                (now, organization_id, attempt_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise JobRuntimeCatalogError(code="job.attempt_not_running")

    def request_cancel(
        self, *, organization_id: UUID, job_id: UUID, requested_at: datetime
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT status FROM job_runtime_jobs
                   WHERE organization_id = %s AND job_id = %s FOR UPDATE""",
                (organization_id, job_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise JobRuntimeCatalogError(code="job.not_found")
            if row["status"] not in {"queued", "running", "recovering"}:
                raise JobRuntimeCatalogError(code="job.already_finished")
            cursor.execute(
                """UPDATE job_runtime_jobs
                   SET cancel_requested_at = COALESCE(cancel_requested_at, %s),
                       status = CASE WHEN status = 'queued' THEN 'canceled' ELSE status END,
                       updated_at = %s
                   WHERE organization_id = %s AND job_id = %s""",
                (requested_at, requested_at, organization_id, job_id),
            )
            cursor.execute(
                """UPDATE job_runtime_attempts
                   SET status = 'canceled', finished_at = %s,
                       error_code = 'job.canceled'
                   WHERE organization_id = %s AND job_id = %s AND status = 'queued'""",
                (requested_at, organization_id, job_id),
            )

    def is_cancel_requested(self, *, organization_id: UUID, job_id: UUID) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT cancel_requested_at IS NOT NULL AS requested
                   FROM job_runtime_jobs
                   WHERE organization_id = %s AND job_id = %s""",
                (organization_id, job_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise JobRuntimeCatalogError(code="job.not_found")
        return bool(row["requested"])

    def finish_attempt(
        self,
        *,
        envelope: JobEnvelope,
        worker_id: str,
        result: JobResultManifest,
    ) -> JobResultManifest:
        if (
            result.job_id != envelope.job_id
            or result.attempt_id != envelope.attempt_id
            or result.organization_id != envelope.organization_id
            or result.envelope_digest != envelope.envelope_digest
        ):
            raise JobRuntimeCatalogError(code="job.result_identity_mismatch")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """SELECT status, cancel_requested_at FROM job_runtime_jobs
                       WHERE organization_id = %s AND job_id = %s FOR UPDATE""",
                    (envelope.organization_id, envelope.job_id),
                )
                job = cursor.fetchone()
                if job is None:
                    raise JobRuntimeCatalogError(code="job.not_found")
                cursor.execute(
                    """SELECT status, worker_id FROM job_runtime_attempts
                       WHERE organization_id = %s AND attempt_id = %s FOR UPDATE""",
                    (envelope.organization_id, envelope.attempt_id),
                )
                attempt = cursor.fetchone()
                if attempt is None:
                    raise JobRuntimeCatalogError(code="job.attempt_not_found")
                if (
                    attempt["status"] != "running"
                    or attempt["worker_id"] != worker_id
                    or job["status"] != "running"
                ):
                    raise JobRuntimeCatalogError(code="job.attempt_not_running")
                actual_result = result
                if job["cancel_requested_at"] is not None:
                    actual_result = result.model_copy(
                        update={
                            "outcome": "canceled",
                            "output_artifact_manifest_digest": None,
                            "outputs": (),
                            "strategy_decisions": (),
                            "error_code": "job.canceled",
                        }
                    )
                cursor.execute(
                    """UPDATE job_runtime_attempts
                       SET status = %s, finished_at = %s, result = %s,
                           exit_code = %s, error_code = %s
                       WHERE organization_id = %s AND attempt_id = %s""",
                    (
                        actual_result.outcome,
                        actual_result.completed_at,
                        Jsonb(actual_result.model_dump(mode="json", by_alias=True)),
                        actual_result.exit_code,
                        actual_result.error_code,
                        envelope.organization_id,
                        envelope.attempt_id,
                    ),
                )
                cursor.execute(
                    """UPDATE job_runtime_jobs
                       SET status = %s, result_artifact_manifest_digest = %s,
                           updated_at = %s
                       WHERE organization_id = %s AND job_id = %s""",
                    (
                        actual_result.outcome,
                        actual_result.output_artifact_manifest_digest,
                        actual_result.completed_at,
                        envelope.organization_id,
                        envelope.job_id,
                    ),
                )
        except JobRuntimeCatalogError:
            raise
        except psycopg.errors.ForeignKeyViolation as error:
            raise JobRuntimeCatalogError(code="job.result_artifact_missing") from error
        except psycopg.Error as error:
            raise JobRuntimeCatalogError(code="job.catalog_unavailable") from error
        return actual_result

    def retry(self, *, envelope: JobEnvelope, queued_at: datetime) -> None:
        payload = envelope.model_dump(mode="json", by_alias=True)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT semantic_spec_digest, status, attempt_count,
                          cancel_requested_at
                   FROM job_runtime_jobs
                   WHERE organization_id = %s AND job_id = %s FOR UPDATE""",
                (envelope.organization_id, envelope.job_id),
            )
            job = cursor.fetchone()
            if job is None:
                raise JobRuntimeCatalogError(code="job.not_found")
            if job["cancel_requested_at"] is not None:
                raise JobRuntimeCatalogError(code="job.cancel_requested")
            if job["status"] not in {
                "failed",
                "crashed",
                "timed_out",
                "resource_exhausted",
            }:
                raise JobRuntimeCatalogError(code="job.retry_not_allowed")
            if job["semantic_spec_digest"] != envelope.semantic_spec_digest:
                raise JobRuntimeCatalogError(code="job.retry_spec_changed")
            next_attempt = int(job["attempt_count"]) + 1
            if envelope.attempt_number != next_attempt:
                raise JobRuntimeCatalogError(code="job.attempt_number_invalid")
            cursor.execute(
                """UPDATE job_runtime_jobs
                   SET status = 'queued', attempt_count = %s, updated_at = %s
                   WHERE organization_id = %s AND job_id = %s""",
                (next_attempt, queued_at, envelope.organization_id, envelope.job_id),
            )
            self._insert_attempt(cursor=cursor, envelope=envelope, payload=payload)

    def claim_stale_for_recovery(
        self,
        *,
        now: datetime,
        worker_heartbeat_before: datetime,
        recovery_claimed_before: datetime,
        recovery_owner_id: str,
    ) -> tuple[RecoveryClaim, ...]:
        if not recovery_owner_id or len(recovery_owner_id) > 128:
            raise JobRuntimeCatalogError(code="job.recovery_owner_invalid")
        claims: list[RecoveryClaim] = []
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT job.organization_id, job.job_id, job.cancel_requested_at
                   FROM job_runtime_jobs AS job
                   WHERE job.status IN ('running', 'recovering')
                     AND EXISTS (
                         SELECT 1 FROM job_runtime_attempts AS attempt
                         WHERE attempt.organization_id = job.organization_id
                           AND attempt.job_id = job.job_id
                           AND (
                               (attempt.status = 'recovering'
                                AND attempt.recovery_claimed_at < %s)
                               OR (
                                   attempt.status = 'running'
                                   AND (attempt.deadline <= %s
                                        OR attempt.heartbeat_at < %s)
                               )
                           )
                     )
                   ORDER BY job.organization_id, job.job_id
                   FOR UPDATE OF job SKIP LOCKED""",
                (recovery_claimed_before, now, worker_heartbeat_before),
            )
            jobs = cursor.fetchall()
            for job in jobs:
                cursor.execute(
                    """SELECT attempt_id, deadline, envelope, status
                       FROM job_runtime_attempts
                       WHERE organization_id = %s AND job_id = %s
                         AND (
                             (status = 'recovering' AND recovery_claimed_at < %s)
                             OR (status = 'running'
                                 AND (deadline <= %s OR heartbeat_at < %s))
                         )
                       ORDER BY attempt_number DESC
                       FOR UPDATE""",
                    (
                        job["organization_id"],
                        job["job_id"],
                        recovery_claimed_before,
                        now,
                        worker_heartbeat_before,
                    ),
                )
                attempt = cursor.fetchone()
                if attempt is None:
                    continue
                if job["cancel_requested_at"] is not None:
                    outcome = "canceled"
                    error_code = "job.canceled"
                else:
                    outcome = "timed_out" if attempt["deadline"] <= now else "crashed"
                    error_code = (
                        "job.deadline_exceeded"
                        if outcome == "timed_out"
                        else "job.worker_lost"
                    )
                cursor.execute(
                    """UPDATE job_runtime_attempts
                       SET status = 'recovering', recovery_owner_id = %s,
                           recovery_claimed_at = %s
                       WHERE organization_id = %s AND attempt_id = %s""",
                    (
                        recovery_owner_id,
                        now,
                        job["organization_id"],
                        attempt["attempt_id"],
                    ),
                )
                if attempt["status"] == "running":
                    cursor.execute(
                        """UPDATE job_runtime_jobs
                           SET status = 'recovering', updated_at = %s
                           WHERE organization_id = %s AND job_id = %s""",
                        (now, job["organization_id"], job["job_id"]),
                    )
                claims.append(
                    RecoveryClaim(
                        envelope=JobEnvelope.model_validate(attempt["envelope"]),
                        outcome=outcome,
                        error_code=error_code,
                        recovery_owner_id=recovery_owner_id,
                    )
                )
        return tuple(claims)

    def complete_recovery(
        self, *, claim: RecoveryClaim, completed_at: datetime
    ) -> JobResultManifest:
        envelope = claim.envelope
        result = JobResultManifest(
            schema="JobResultManifest/v1",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            organization_id=envelope.organization_id,
            outcome=cast(Any, claim.outcome),
            envelope_digest=envelope.envelope_digest,
            output_artifact_manifest_digest=None,
            outputs=(),
            strategy_decisions=(),
            completed_at=completed_at,
            exit_code=None,
            error_code=claim.error_code,
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT status, cancel_requested_at FROM job_runtime_jobs
                   WHERE organization_id = %s AND job_id = %s FOR UPDATE""",
                (envelope.organization_id, envelope.job_id),
            )
            job = cursor.fetchone()
            if job is None:
                raise JobRuntimeCatalogError(code="job.not_found")
            cursor.execute(
                """SELECT status, recovery_owner_id FROM job_runtime_attempts
                   WHERE organization_id = %s AND attempt_id = %s FOR UPDATE""",
                (envelope.organization_id, envelope.attempt_id),
            )
            attempt = cursor.fetchone()
            if (
                attempt is None
                or attempt["status"] != "recovering"
                or attempt["recovery_owner_id"] != claim.recovery_owner_id
                or job["status"] != "recovering"
            ):
                raise JobRuntimeCatalogError(code="job.recovery_not_owned")
            if job["cancel_requested_at"] is not None:
                result = result.model_copy(
                    update={"outcome": "canceled", "error_code": "job.canceled"}
                )
            cursor.execute(
                """UPDATE job_runtime_attempts
                   SET status = %s, finished_at = %s, result = %s,
                       exit_code = NULL, error_code = %s
                   WHERE organization_id = %s AND attempt_id = %s""",
                (
                    result.outcome,
                    completed_at,
                    Jsonb(result.model_dump(mode="json", by_alias=True)),
                    result.error_code,
                    envelope.organization_id,
                    envelope.attempt_id,
                ),
            )
            cursor.execute(
                """UPDATE job_runtime_jobs SET status = %s, updated_at = %s
                   WHERE organization_id = %s AND job_id = %s""",
                (
                    result.outcome,
                    completed_at,
                    envelope.organization_id,
                    envelope.job_id,
                ),
            )
        return result

    def get_state(self, *, organization_id: UUID, job_id: UUID) -> JobState:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT status, attempt_count,
                          cancel_requested_at IS NOT NULL AS cancel_requested,
                          result_artifact_manifest_digest
                   FROM job_runtime_jobs
                   WHERE organization_id = %s AND job_id = %s""",
                (organization_id, job_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise JobRuntimeCatalogError(code="job.not_found")
        return JobState(
            organization_id=organization_id,
            job_id=job_id,
            status=cast(str, row["status"]),
            attempt_count=int(row["attempt_count"]),
            cancel_requested=bool(row["cancel_requested"]),
            result_artifact_manifest_digest=cast(
                str | None, row["result_artifact_manifest_digest"]
            ),
        )

    @staticmethod
    def _insert_attempt(
        *,
        cursor: psycopg.Cursor[Any],
        envelope: JobEnvelope,
        payload: dict[str, Any],
    ) -> None:
        cursor.execute(
            """INSERT INTO job_runtime_attempts
                   (organization_id, job_id, attempt_id, attempt_number,
                    envelope_digest, image_digest, envelope, status, deadline)
               VALUES (%s,%s,%s,%s,%s,%s,%s,'queued',%s)""",
            (
                envelope.organization_id,
                envelope.job_id,
                envelope.attempt_id,
                envelope.attempt_number,
                envelope.envelope_digest,
                envelope.image_digest,
                Jsonb(payload),
                envelope.deadline,
            ),
        )

    @staticmethod
    def _expire_queued(*, cursor: psycopg.Cursor[Any], now: datetime) -> None:
        cursor.execute(
            """SELECT job.organization_id, job.job_id
               FROM job_runtime_jobs AS job
               WHERE job.status = 'queued'
                 AND EXISTS (
                     SELECT 1 FROM job_runtime_attempts AS attempt
                     WHERE attempt.organization_id = job.organization_id
                       AND attempt.job_id = job.job_id
                       AND attempt.status = 'queued' AND attempt.deadline <= %s
                 )
               ORDER BY job.organization_id, job.job_id
               FOR UPDATE OF job SKIP LOCKED""",
            (now,),
        )
        for job in cursor.fetchall():
            cursor.execute(
                """UPDATE job_runtime_attempts
                   SET status = 'timed_out', finished_at = %s,
                       error_code = 'job.deadline_exceeded'
                   WHERE organization_id = %s AND job_id = %s
                     AND status = 'queued' AND deadline <= %s""",
                (now, job["organization_id"], job["job_id"], now),
            )
            cursor.execute(
                """UPDATE job_runtime_jobs
                   SET status = 'timed_out', updated_at = %s
                   WHERE organization_id = %s AND job_id = %s""",
                (now, job["organization_id"], job["job_id"]),
            )


__all__ = [
    "ClaimedJobAttempt",
    "RecoveryClaim",
    "JobRuntimeCatalogError",
    "JobState",
    "PostgresJobRuntimeCatalog",
]
