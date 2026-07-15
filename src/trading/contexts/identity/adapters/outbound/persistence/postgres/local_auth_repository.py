from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from trading.contexts.identity.application.ports.local_auth_repository import (
    LocalAccount,
    LocalAuthChallenge,
    LocalAuthPurpose,
    LocalAuthRepository,
    LocalAuthRepositoryError,
    LocalPasskey,
    RecoveryCodeHash,
)
from trading.shared_kernel.primitives import UserId

_RATE_WINDOW = timedelta(minutes=15)
_LOCKOUT = timedelta(minutes=15)
_MAX_FAILURES = 5


class PostgresLocalAuthRepository(LocalAuthRepository):
    def __init__(self, *, dsn: str) -> None:
        normalized = dsn.strip()
        if not normalized:
            raise ValueError("PostgresLocalAuthRepository requires non-empty dsn")
        self._dsn = normalized

    def bootstrap_required(self) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT NOT EXISTS (SELECT 1 FROM identity_installations) AS value")
            row = cursor.fetchone()
        return bool(row and row["value"])

    def issue_bootstrap_ticket(
        self, *, token_sha256: str, created_at: datetime, expires_at: datetime
    ) -> UUID:
        ticket_id = uuid4()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('local-auth-bootstrap'))")
            cursor.execute("SELECT EXISTS (SELECT 1 FROM identity_installations) AS exists")
            row = cursor.fetchone()
            if row is None or row["exists"]:
                raise LocalAuthRepositoryError(code="bootstrap_unavailable")
            cursor.execute(
                """
                UPDATE identity_local_bootstrap_tickets
                SET consumed_at = %s
                WHERE consumed_at IS NULL
                """,
                (created_at,),
            )
            cursor.execute(
                """
                INSERT INTO identity_local_bootstrap_tickets (
                    ticket_id, token_sha256, created_at, expires_at
                ) VALUES (%s, %s, %s, %s)
                """,
                (str(ticket_id), token_sha256, created_at, expires_at),
            )
        return ticket_id

    def find_bootstrap_ticket(self, *, token_sha256: str, now: datetime) -> UUID | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT ticket_id
                FROM identity_local_bootstrap_tickets
                WHERE token_sha256 = %s AND consumed_at IS NULL AND expires_at > %s
                """,
                (token_sha256, now),
            )
            row = cursor.fetchone()
        return None if row is None else UUID(str(row["ticket_id"]))

    def create_challenge(
        self,
        *,
        purpose: LocalAuthPurpose,
        challenge_sha256: str,
        user_id: UserId | None,
        context: Mapping[str, str],
        created_at: datetime,
        expires_at: datetime,
    ) -> LocalAuthChallenge:
        challenge = LocalAuthChallenge(
            challenge_id=uuid4(),
            purpose=purpose,
            challenge_sha256=challenge_sha256,
            user_id=user_id,
            context=dict(context),
            created_at=created_at,
            expires_at=expires_at,
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO identity_local_auth_challenges (
                    challenge_id, purpose, challenge_sha256, user_id,
                    context_json, created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    str(challenge.challenge_id),
                    purpose,
                    challenge_sha256,
                    None if user_id is None else str(user_id),
                    json.dumps(dict(context), sort_keys=True),
                    created_at,
                    expires_at,
                ),
            )
        return challenge

    def find_challenge(
        self, *, challenge_id: UUID, purpose: LocalAuthPurpose, now: datetime
    ) -> LocalAuthChallenge | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT challenge_id, purpose, challenge_sha256, user_id,
                       context_json, created_at, expires_at, consumed_at
                FROM identity_local_auth_challenges
                WHERE challenge_id = %s AND purpose = %s
                  AND consumed_at IS NULL AND expires_at > %s
                """,
                (str(challenge_id), purpose, now),
            )
            row = cursor.fetchone()
        return None if row is None else _challenge(row)

    def complete_bootstrap(
        self,
        *,
        challenge_id: UUID,
        ticket_id: UUID,
        user_id: UserId,
        username: str,
        display_name: str,
        password_hash: str | None,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        passkey: LocalPasskey,
        recovery_code_hashes: tuple[str, ...],
        completed_at: datetime,
    ) -> None:
        installation_id = uuid4()
        organization_id = uuid4()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('local-auth-bootstrap'))")
            cursor.execute("SELECT EXISTS (SELECT 1 FROM identity_installations) AS exists")
            row = cursor.fetchone()
            if row is None or row["exists"]:
                raise LocalAuthRepositoryError(code="bootstrap_unavailable")
            cursor.execute(
                """
                UPDATE identity_local_bootstrap_tickets SET consumed_at = %s
                WHERE ticket_id = %s AND consumed_at IS NULL AND expires_at > %s
                RETURNING ticket_id
                """,
                (completed_at, str(ticket_id), completed_at),
            )
            if cursor.fetchone() is None:
                raise LocalAuthRepositoryError(code="bootstrap_invalid")
            self._consume_challenge(
                cursor=cursor,
                challenge_id=challenge_id,
                purpose="bootstrap",
                completed_at=completed_at,
            )
            cursor.execute(
                """
                INSERT INTO identity_users (
                    user_id, telegram_user_id, paid_level, created_at,
                    last_login_at, is_deleted, keycloak_subject
                ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
                """,
                (str(user_id), completed_at, completed_at),
            )
            cursor.execute(
                """
                INSERT INTO identity_local_accounts (
                    user_id, username, display_name, password_hash, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (str(user_id), username, display_name, password_hash, completed_at, completed_at),
            )
            cursor.execute(
                """
                INSERT INTO identity_webauthn_credentials (
                    credential_id, user_id, public_key, sign_count, transports,
                    created_at, last_used_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    passkey.credential_id,
                    str(user_id),
                    passkey.public_key,
                    passkey.sign_count,
                    list(passkey.transports),
                    completed_at,
                    completed_at,
                ),
            )
            cursor.execute(
                """
                INSERT INTO identity_installations (
                    installation_id, singleton_key, display_name, created_at
                ) VALUES (%s, TRUE, %s, %s)
                """,
                (str(installation_id), installation_name, completed_at),
            )
            cursor.execute(
                """
                INSERT INTO identity_installation_owners (
                    installation_id, user_id, granted_by_user_id, granted_at
                ) VALUES (%s, %s, %s, %s)
                """,
                (str(installation_id), str(user_id), str(user_id), completed_at),
            )
            cursor.execute(
                """
                INSERT INTO identity_organizations (
                    organization_id, installation_id, slug, display_name, status, created_at
                ) VALUES (%s, %s, %s, %s, 'active', %s)
                """,
                (
                    str(organization_id),
                    str(installation_id),
                    organization_slug,
                    organization_name,
                    completed_at,
                ),
            )
            cursor.execute(
                """
                INSERT INTO identity_memberships (
                    organization_id, user_id, role, status, created_at, updated_at
                ) VALUES (%s, %s, 'owner', 'active', %s, %s)
                """,
                (str(organization_id), str(user_id), completed_at, completed_at),
            )
            cursor.executemany(
                """
                INSERT INTO identity_local_recovery_codes (
                    recovery_code_id, user_id, code_hash, created_at
                ) VALUES (%s, %s, %s, %s)
                """,
                [
                    (str(uuid4()), str(user_id), code_hash, completed_at)
                    for code_hash in recovery_code_hashes
                ],
            )
            cursor.execute(
                """
                INSERT INTO identity_administrative_audit_events (
                    event_id, installation_id, organization_id, actor_user_id,
                    action, target_type, target_id, outcome, metadata_json, created_at
                ) VALUES (%s, %s, %s, %s, 'installation.bootstrap', 'installation', %s,
                          'succeeded', %s::jsonb, %s)
                """,
                (
                    str(uuid4()),
                    str(installation_id),
                    str(organization_id),
                    str(user_id),
                    str(installation_id),
                    json.dumps({"organization_id": str(organization_id)}, sort_keys=True),
                    completed_at,
                ),
            )
            self._insert_auth_event(
                cursor=cursor,
                user_id=user_id,
                subject_sha256=hashlib.sha256(b"bootstrap").hexdigest(),
                action="local_auth.bootstrap",
                outcome="succeeded",
                reason_code="completed",
                created_at=completed_at,
            )

    def find_account_by_username(self, *, username: str) -> LocalAccount | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, username, display_name, password_hash, created_at
                FROM identity_local_accounts WHERE username = %s
                """,
                (username,),
            )
            row = cursor.fetchone()
        return None if row is None else _account(row)

    def find_account_by_user_id(self, *, user_id: UserId) -> LocalAccount | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, username, display_name, password_hash, created_at
                FROM identity_local_accounts WHERE user_id = %s
                """,
                (str(user_id),),
            )
            row = cursor.fetchone()
        return None if row is None else _account(row)

    def find_passkey(self, *, credential_id: str) -> LocalPasskey | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT credential_id, user_id, public_key, sign_count, transports, created_at
                FROM identity_webauthn_credentials WHERE credential_id = %s
                """,
                (credential_id,),
            )
            row = cursor.fetchone()
        return None if row is None else _passkey(row)

    def list_passkeys(self, *, user_id: UserId) -> tuple[LocalPasskey, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT credential_id, user_id, public_key, sign_count, transports, created_at
                FROM identity_webauthn_credentials WHERE user_id = %s
                ORDER BY created_at, credential_id
                """,
                (str(user_id),),
            )
            rows = cursor.fetchall()
        return tuple(_passkey(row) for row in rows)

    def add_passkey_and_consume_challenge(
        self,
        *,
        challenge_id: UUID,
        passkey: LocalPasskey,
        completed_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            self._consume_challenge(
                cursor=cursor,
                challenge_id=challenge_id,
                purpose="register",
                completed_at=completed_at,
                user_id=passkey.user_id,
            )
            cursor.execute(
                """
                INSERT INTO identity_webauthn_credentials (
                    credential_id, user_id, public_key, sign_count, transports,
                    created_at, last_used_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    passkey.credential_id,
                    str(passkey.user_id),
                    passkey.public_key,
                    passkey.sign_count,
                    list(passkey.transports),
                    completed_at,
                    completed_at,
                ),
            )

    def finish_passkey_authentication(
        self,
        *,
        challenge_id: UUID,
        credential_id: str,
        new_sign_count: int,
        completed_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT user_id FROM identity_webauthn_credentials WHERE credential_id = %s",
                (credential_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise LocalAuthRepositoryError(code="credential_not_found")
            self._consume_challenge(
                cursor=cursor,
                challenge_id=challenge_id,
                purpose=None,
                completed_at=completed_at,
                user_id=None,
            )
            cursor.execute(
                """
                UPDATE identity_webauthn_credentials
                SET sign_count = %s, last_used_at = %s
                WHERE credential_id = %s
                """,
                (new_sign_count, completed_at, credential_id),
            )

    def list_recovery_code_hashes(
        self, *, user_id: UserId
    ) -> tuple[RecoveryCodeHash, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT recovery_code_id, code_hash
                FROM identity_local_recovery_codes
                WHERE user_id = %s AND consumed_at IS NULL
                ORDER BY recovery_code_id
                """,
                (str(user_id),),
            )
            rows = cursor.fetchall()
        return tuple(
            RecoveryCodeHash(
                recovery_code_id=UUID(str(row["recovery_code_id"])),
                code_hash=str(row["code_hash"]),
            )
            for row in rows
        )

    def consume_recovery_code(
        self,
        *,
        recovery_code_id: UUID,
        user_id: UserId,
        consumed_at: datetime,
    ) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE identity_local_recovery_codes SET consumed_at = %s
                WHERE recovery_code_id = %s AND user_id = %s AND consumed_at IS NULL
                RETURNING recovery_code_id
                """,
                (consumed_at, str(recovery_code_id), str(user_id)),
            )
            return cursor.fetchone() is not None

    def is_rate_limited(self, *, subject_sha256: str, now: datetime) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT locked_until > %s AS locked
                FROM identity_local_auth_rate_limits WHERE subject_sha256 = %s
                """,
                (now, subject_sha256),
            )
            row = cursor.fetchone()
        return bool(row and row["locked"])

    def record_auth_failure(self, *, subject_sha256: str, now: datetime) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO identity_local_auth_rate_limits (
                    subject_sha256, failed_count, window_started_at, locked_until, updated_at
                ) VALUES (%s, 1, %s, NULL, %s)
                ON CONFLICT (subject_sha256) DO UPDATE SET
                    failed_count = CASE
                        WHEN identity_local_auth_rate_limits.window_started_at <= %s
                        THEN 1 ELSE identity_local_auth_rate_limits.failed_count + 1 END,
                    window_started_at = CASE
                        WHEN identity_local_auth_rate_limits.window_started_at <= %s
                        THEN %s ELSE identity_local_auth_rate_limits.window_started_at END,
                    locked_until = CASE
                        WHEN (CASE
                            WHEN identity_local_auth_rate_limits.window_started_at <= %s
                            THEN 1 ELSE identity_local_auth_rate_limits.failed_count + 1 END) >= %s
                        THEN %s ELSE identity_local_auth_rate_limits.locked_until END,
                    updated_at = %s
                """,
                (
                    subject_sha256,
                    now,
                    now,
                    now - _RATE_WINDOW,
                    now - _RATE_WINDOW,
                    now,
                    now - _RATE_WINDOW,
                    _MAX_FAILURES,
                    now + _LOCKOUT,
                    now,
                ),
            )

    def clear_auth_failures(self, *, subject_sha256: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM identity_local_auth_rate_limits WHERE subject_sha256 = %s",
                (subject_sha256,),
            )

    def record_auth_event(
        self,
        *,
        user_id: UserId | None,
        subject_sha256: str,
        action: str,
        outcome: Literal["succeeded", "rejected"],
        reason_code: str,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            self._insert_auth_event(
                cursor=cursor,
                user_id=user_id,
                subject_sha256=subject_sha256,
                action=action,
                outcome=outcome,
                reason_code=reason_code,
                created_at=created_at,
            )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    @staticmethod
    def _consume_challenge(
        *,
        cursor: Any,
        challenge_id: UUID,
        purpose: LocalAuthPurpose | None,
        completed_at: datetime,
        user_id: UserId | None = None,
    ) -> None:
        conditions = ["challenge_id = %s", "consumed_at IS NULL", "expires_at > %s"]
        parameters: list[object] = [str(challenge_id), completed_at]
        if purpose is not None:
            conditions.append("purpose = %s")
            parameters.append(purpose)
        if user_id is not None:
            conditions.append("user_id = %s")
            parameters.append(str(user_id))
        cursor.execute(
            "UPDATE identity_local_auth_challenges SET consumed_at = %s WHERE "
            + " AND ".join(conditions)
            + " RETURNING challenge_id",
            (completed_at, *parameters),
        )
        if cursor.fetchone() is None:
            raise LocalAuthRepositoryError(code="challenge_invalid")

    @staticmethod
    def _insert_auth_event(
        *,
        cursor: Any,
        user_id: UserId | None,
        subject_sha256: str,
        action: str,
        outcome: str,
        reason_code: str,
        created_at: datetime,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO identity_local_auth_events (
                event_id, user_id, subject_sha256, action, outcome, reason_code, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid4()),
                None if user_id is None else str(user_id),
                subject_sha256,
                action,
                outcome,
                reason_code,
                created_at,
            ),
        )


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("local auth timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _account(row: Mapping[str, Any]) -> LocalAccount:
    return LocalAccount(
        user_id=UserId(UUID(str(row["user_id"]))),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        password_hash=None if row["password_hash"] is None else str(row["password_hash"]),
        created_at=_utc(row["created_at"]),
    )


def _challenge(row: Mapping[str, Any]) -> LocalAuthChallenge:
    context = row["context_json"]
    if not isinstance(context, dict):
        raise ValueError("local auth challenge context must be object")
    return LocalAuthChallenge(
        challenge_id=UUID(str(row["challenge_id"])),
        purpose=cast(LocalAuthPurpose, str(row["purpose"])),
        challenge_sha256=str(row["challenge_sha256"]),
        user_id=None if row["user_id"] is None else UserId(UUID(str(row["user_id"]))),
        context={str(key): str(value) for key, value in context.items()},
        created_at=_utc(row["created_at"]),
        expires_at=_utc(row["expires_at"]),
        consumed_at=None if row["consumed_at"] is None else _utc(row["consumed_at"]),
    )


def _passkey(row: Mapping[str, Any]) -> LocalPasskey:
    transports = row["transports"]
    return LocalPasskey(
        credential_id=str(row["credential_id"]),
        user_id=UserId(UUID(str(row["user_id"]))),
        public_key=bytes(row["public_key"]),
        sign_count=int(row["sign_count"]),
        transports=tuple(str(value) for value in transports or ()),
        created_at=_utc(row["created_at"]),
    )
