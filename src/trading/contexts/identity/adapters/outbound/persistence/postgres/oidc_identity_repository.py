from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from trading.contexts.identity.application.ports.authentication_provider import (
    OidcAttemptPurpose,
    OidcIdentityCompletion,
    OidcIdentityRepository,
    OidcIdentityRepositoryError,
    OidcLoginAttempt,
)
from trading.shared_kernel.primitives import UserId


class PostgresOidcIdentityRepository(OidcIdentityRepository):
    """Transactional OIDC identity adapter with invitation-first provisioning."""

    def __init__(self, *, dsn: str) -> None:
        self._dsn = dsn.strip()
        if not self._dsn:
            raise ValueError("PostgresOidcIdentityRepository requires non-empty dsn")

    def create_attempt(
        self,
        *,
        provider_id: str,
        issuer: str,
        purpose: OidcAttemptPurpose,
        state_sha256: str,
        nonce_sha256: str,
        code_verifier: str,
        linking_user_id: UserId | None,
        next_path: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> OidcLoginAttempt:
        attempt_id = uuid4()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO identity_oidc_login_attempts (
                    attempt_id, provider_id, issuer, purpose, state_sha256,
                    nonce_sha256, code_verifier, linking_user_id, next_path,
                    created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(attempt_id), provider_id, issuer, purpose, state_sha256,
                    nonce_sha256, code_verifier,
                    None if linking_user_id is None else str(linking_user_id),
                    next_path, created_at, expires_at,
                ),
            )
        return OidcLoginAttempt(
            attempt_id=attempt_id,
            provider_id=provider_id,
            issuer=issuer,
            purpose=purpose,
            state_sha256=state_sha256,
            nonce_sha256=nonce_sha256,
            code_verifier=code_verifier,
            linking_user_id=linking_user_id,
            next_path=next_path,
            created_at=created_at,
            expires_at=expires_at,
        )

    def find_attempt(self, *, attempt_id: UUID, now: datetime) -> OidcLoginAttempt | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT attempt_id, provider_id, issuer, purpose, state_sha256,
                       nonce_sha256, code_verifier, linking_user_id, next_path,
                       created_at, expires_at, exchange_started_at, consumed_at
                FROM identity_oidc_login_attempts
                WHERE attempt_id = %s
                  AND exchange_started_at IS NULL
                  AND consumed_at IS NULL
                  AND expires_at > %s
                """,
                (str(attempt_id), now),
            )
            row = cursor.fetchone()
        return None if row is None else _attempt(row)

    def claim_attempt(
        self, *, attempt_id: UUID, claimed_at: datetime
    ) -> OidcLoginAttempt | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE identity_oidc_login_attempts
                SET exchange_started_at = %s
                WHERE attempt_id = %s
                  AND exchange_started_at IS NULL
                  AND consumed_at IS NULL
                  AND expires_at > %s
                RETURNING attempt_id, provider_id, issuer, purpose, state_sha256,
                          nonce_sha256, code_verifier, linking_user_id, next_path,
                          created_at, expires_at, exchange_started_at, consumed_at
                """,
                (claimed_at, str(attempt_id), claimed_at),
            )
            row = cursor.fetchone()
        return None if row is None else _attempt(row)

    def reject_attempt(
        self,
        *,
        attempt_id: UUID,
        reason_code: str,
        rejected_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE identity_oidc_login_attempts
                SET consumed_at = %s, rejection_reason = %s
                WHERE attempt_id = %s AND consumed_at IS NULL
                RETURNING provider_id
                """,
                (rejected_at, reason_code, str(attempt_id)),
            )
            row = cursor.fetchone()
            if row is not None:
                self._event(
                    cursor=cursor,
                    attempt_id=attempt_id,
                    provider_id=str(row["provider_id"]),
                    user_id=None,
                    subject_sha256=None,
                    action="oidc.callback",
                    outcome="rejected",
                    reason_code=reason_code,
                    created_at=rejected_at,
                )

    def complete_attempt(
        self,
        *,
        attempt_id: UUID,
        provider_id: str,
        issuer: str,
        subject_sha256: str,
        email_sha256: str | None,
        email_verified: bool,
        callback_user_id: UserId | None,
        completed_at: datetime,
    ) -> OidcIdentityCompletion:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT attempt_id, provider_id, issuer, purpose, state_sha256,
                           nonce_sha256, code_verifier, linking_user_id, next_path,
                           created_at, expires_at, exchange_started_at, consumed_at
                    FROM identity_oidc_login_attempts
                    WHERE attempt_id = %s
                    FOR UPDATE
                    """,
                    (str(attempt_id),),
                )
                row = cursor.fetchone()
                if row is None:
                    raise OidcIdentityRepositoryError(code="oidc_attempt_invalid")
                attempt = _attempt(row)
                if (
                    attempt.exchange_started_at is None
                    or attempt.consumed_at is not None
                    or attempt.expires_at <= completed_at
                ):
                    raise OidcIdentityRepositoryError(code="oidc_attempt_invalid")
                if attempt.provider_id != provider_id or attempt.issuer != issuer:
                    raise OidcIdentityRepositoryError(code="oidc_attempt_invalid")

                cursor.execute(
                    """
                    SELECT user_id
                    FROM identity_external_identities
                    WHERE provider_id = %s AND issuer = %s AND subject_sha256 = %s
                    FOR UPDATE
                    """,
                    (provider_id, issuer, subject_sha256),
                )
                linked = cursor.fetchone()
                if linked is not None:
                    user_id = UserId.from_string(str(linked["user_id"]))
                    if attempt.purpose == "link" and user_id != attempt.linking_user_id:
                        raise OidcIdentityRepositoryError(code="oidc_identity_conflict")
                    self._record_login(
                        cursor=cursor,
                        attempt_id=attempt_id,
                        provider_id=provider_id,
                        user_id=user_id,
                        subject_sha256=subject_sha256,
                        completed_at=completed_at,
                    )
                    return OidcIdentityCompletion(
                        user_id=user_id,
                        provisioned=False,
                        linked=False,
                        accepted_invitation_count=0,
                    )

                if attempt.purpose == "link":
                    user_id = attempt.linking_user_id
                    if user_id is None or callback_user_id != user_id:
                        raise OidcIdentityRepositoryError(code="oidc_link_session_required")
                    cursor.execute(
                        "SELECT 1 FROM identity_users WHERE user_id = %s FOR UPDATE",
                        (str(user_id),),
                    )
                    if cursor.fetchone() is None:
                        raise OidcIdentityRepositoryError(code="oidc_link_session_required")
                    self._insert_identity(
                        cursor=cursor,
                        provider_id=provider_id,
                        issuer=issuer,
                        subject_sha256=subject_sha256,
                        user_id=user_id,
                        email_sha256=email_sha256 if email_verified else None,
                        completed_at=completed_at,
                    )
                    self._consume_attempt(cursor=cursor, attempt_id=attempt_id, at=completed_at)
                    self._event(
                        cursor=cursor,
                        attempt_id=attempt_id,
                        provider_id=provider_id,
                        user_id=user_id,
                        subject_sha256=subject_sha256,
                        action="oidc.identity_linked",
                        outcome="succeeded",
                        reason_code="linked",
                        created_at=completed_at,
                    )
                    return OidcIdentityCompletion(
                        user_id=user_id,
                        provisioned=False,
                        linked=True,
                        accepted_invitation_count=0,
                    )

                if not email_verified or email_sha256 is None:
                    raise OidcIdentityRepositoryError(code="oidc_verified_email_required")
                cursor.execute(
                    """
                    SELECT invitation_id, organization_id, role
                    FROM identity_invitations
                    WHERE recipient_email_sha256 = %s
                      AND status = 'pending'
                      AND expires_at > %s
                    ORDER BY organization_id, invitation_id
                    FOR UPDATE
                    """,
                    (email_sha256, completed_at),
                )
                invitations = cursor.fetchall()
                if not invitations:
                    raise OidcIdentityRepositoryError(code="oidc_invitation_required")
                user_id = UserId(uuid4())
                cursor.execute(
                    """
                    INSERT INTO identity_users (
                        user_id, paid_level, created_at, last_login_at, is_deleted
                    ) VALUES (%s, 'free', %s, %s, FALSE)
                    """,
                    (str(user_id), completed_at, completed_at),
                )
                self._insert_identity(
                    cursor=cursor,
                    provider_id=provider_id,
                    issuer=issuer,
                    subject_sha256=subject_sha256,
                    user_id=user_id,
                    email_sha256=email_sha256,
                    completed_at=completed_at,
                )
                for invitation in invitations:
                    cursor.execute(
                        """
                        INSERT INTO identity_memberships (
                            organization_id, user_id, role, status, created_at, updated_at
                        ) VALUES (%s, %s, %s, 'active', %s, %s)
                        """,
                        (
                            str(invitation["organization_id"]), str(user_id),
                            str(invitation["role"]), completed_at, completed_at,
                        ),
                    )
                    cursor.execute(
                        """
                        UPDATE identity_invitations
                        SET status = 'accepted', accepted_by_user_id = %s, accepted_at = %s
                        WHERE invitation_id = %s AND status = 'pending'
                        """,
                        (str(user_id), completed_at, str(invitation["invitation_id"])),
                    )
                self._consume_attempt(cursor=cursor, attempt_id=attempt_id, at=completed_at)
                self._event(
                    cursor=cursor,
                    attempt_id=attempt_id,
                    provider_id=provider_id,
                    user_id=user_id,
                    subject_sha256=subject_sha256,
                    action="oidc.user_provisioned",
                    outcome="succeeded",
                    reason_code="invitation_accepted",
                    created_at=completed_at,
                )
                return OidcIdentityCompletion(
                    user_id=user_id,
                    provisioned=True,
                    linked=True,
                    accepted_invitation_count=len(invitations),
                )
        except psycopg.errors.UniqueViolation as error:
            raise OidcIdentityRepositoryError(code="oidc_identity_conflict") from error
        except psycopg.errors.ForeignKeyViolation as error:
            raise OidcIdentityRepositoryError(code="oidc_identity_conflict") from error

    def _record_login(
        self,
        *,
        cursor: Any,
        attempt_id: UUID,
        provider_id: str,
        user_id: UserId,
        subject_sha256: str,
        completed_at: datetime,
    ) -> None:
        cursor.execute(
            """
            UPDATE identity_users
            SET last_login_at = %s, is_deleted = FALSE
            WHERE user_id = %s
            """,
            (completed_at, str(user_id)),
        )
        cursor.execute(
            """
            UPDATE identity_external_identities
            SET last_login_at = %s
            WHERE provider_id = %s AND user_id = %s
            """,
            (completed_at, provider_id, str(user_id)),
        )
        self._consume_attempt(cursor=cursor, attempt_id=attempt_id, at=completed_at)
        self._event(
            cursor=cursor,
            attempt_id=attempt_id,
            provider_id=provider_id,
            user_id=user_id,
            subject_sha256=subject_sha256,
            action="oidc.login",
            outcome="succeeded",
            reason_code="authenticated",
            created_at=completed_at,
        )

    @staticmethod
    def _insert_identity(
        *,
        cursor: Any,
        provider_id: str,
        issuer: str,
        subject_sha256: str,
        user_id: UserId,
        email_sha256: str | None,
        completed_at: datetime,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO identity_external_identities (
                external_identity_id, provider_id, issuer, subject_sha256,
                user_id, verified_email_sha256, created_at, last_login_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid4()), provider_id, issuer, subject_sha256, str(user_id),
                email_sha256, completed_at, completed_at,
            ),
        )

    @staticmethod
    def _consume_attempt(*, cursor: Any, attempt_id: UUID, at: datetime) -> None:
        cursor.execute(
            """
            UPDATE identity_oidc_login_attempts
            SET consumed_at = %s
            WHERE attempt_id = %s AND consumed_at IS NULL
            """,
            (at, str(attempt_id)),
        )

    @staticmethod
    def _event(
        *,
        cursor: Any,
        attempt_id: UUID,
        provider_id: str,
        user_id: UserId | None,
        subject_sha256: str | None,
        action: str,
        outcome: str,
        reason_code: str,
        created_at: datetime,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO identity_oidc_auth_events (
                event_id, attempt_id, provider_id, user_id, subject_sha256,
                action, outcome, reason_code, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid4()), str(attempt_id), provider_id,
                None if user_id is None else str(user_id), subject_sha256,
                action, outcome, reason_code, created_at,
            ),
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))


def _attempt(row: Mapping[str, Any]) -> OidcLoginAttempt:
    return OidcLoginAttempt(
        attempt_id=UUID(str(row["attempt_id"])),
        provider_id=str(row["provider_id"]),
        issuer=str(row["issuer"]),
        purpose=str(row["purpose"]),  # type: ignore[arg-type]
        state_sha256=str(row["state_sha256"]),
        nonce_sha256=str(row["nonce_sha256"]),
        code_verifier=str(row["code_verifier"]),
        linking_user_id=(
            None
            if row["linking_user_id"] is None
            else UserId.from_string(str(row["linking_user_id"]))
        ),
        next_path=str(row["next_path"]),
        created_at=_utc(row["created_at"]),
        expires_at=_utc(row["expires_at"]),
        exchange_started_at=(
            None
            if row["exchange_started_at"] is None
            else _utc(row["exchange_started_at"])
        ),
        consumed_at=None if row["consumed_at"] is None else _utc(row["consumed_at"]),
    )


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("OIDC timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)
