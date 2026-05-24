from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, cast
from uuid import UUID

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from trading.contexts.exchange_control.application.connections import (
    ExchangeConnectionRecord,
    ExchangeConnectionRepository,
    ExchangeCredentialVersionRecord,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialValidationResult,
)
from trading.shared_kernel.primitives import UserId


class PostgresExchangeConnectionRepository(ExchangeConnectionRepository):
    def __init__(self, *, dsn: str) -> None:
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("PostgresExchangeConnectionRepository requires non-empty dsn")
        self._dsn = normalized_dsn

    def create(
        self,
        *,
        connection: ExchangeConnectionRecord,
        credential_version: ExchangeCredentialVersionRecord,
    ) -> ExchangeConnectionRecord | None:
        try:
            with psycopg.connect(
                self._dsn,
                row_factory=cast(Any, dict_row),
            ) as postgres_connection:
                with postgres_connection.cursor() as cursor:
                    cursor.execute(
                        cast(
                            Any,
                            """
                            SELECT 1
                            FROM exchange_connections AS connection
                            JOIN exchange_credential_versions AS credential
                              ON credential.credential_version_id =
                                 connection.active_credential_version_id
                            WHERE connection.owner_user_id = %(owner_user_id)s
                              AND connection.exchange_name = %(exchange_name)s
                              AND connection.market_type = %(market_type)s
                              AND connection.environment = %(environment)s
                              AND connection.status = 'active'
                              AND credential.api_key_fingerprint_hmac =
                                  %(api_key_fingerprint_hmac)s
                            LIMIT 1
                            """,
                        ),
                        {
                            "owner_user_id": str(connection.owner_user_id),
                            "exchange_name": connection.exchange_name,
                            "market_type": connection.market_type,
                            "environment": connection.environment,
                            "api_key_fingerprint_hmac": _fingerprint_bytes(
                                credential_version.api_key_fingerprint_hmac
                            ),
                        },
                    )
                    if cursor.fetchone() is not None:
                        return None

                    cursor.execute(
                        cast(
                            Any,
                            """
                            INSERT INTO exchange_connections (
                                connection_id,
                                owner_user_id,
                                exchange_name,
                                market_type,
                                environment,
                                label,
                                active_credential_version_id,
                                status,
                                status_reason,
                                permission_summary_json,
                                ip_restriction_status,
                                created_at,
                                updated_at,
                                disabled_at
                            )
                            VALUES (
                                %(connection_id)s,
                                %(owner_user_id)s,
                                %(exchange_name)s,
                                %(market_type)s,
                                %(environment)s,
                                %(label)s,
                                %(active_credential_version_id)s,
                                %(status)s,
                                %(status_reason)s,
                                jsonb_build_object(
                                    'permissions', %(permissions)s::text,
                                    'validation_status', %(validation_status)s::text,
                                    'validation_reason', %(validation_reason)s::text
                                ),
                                'unknown',
                                %(created_at)s,
                                %(updated_at)s,
                                %(disabled_at)s
                            )
                            RETURNING
                                connection_id,
                                owner_user_id,
                                exchange_name,
                                market_type,
                                environment,
                                label,
                                active_credential_version_id,
                                status,
                                status_reason,
                                permission_summary_json ->> 'permissions' AS permissions,
                                permission_summary_json ->> 'validation_status'
                                    AS validation_status,
                                permission_summary_json ->> 'validation_reason'
                                    AS validation_reason,
                                ip_restriction_status,
                                last_validated_at,
                                created_at,
                                updated_at,
                                disabled_at
                            """,
                        ),
                        _connection_parameters(connection=connection),
                    )
                    created_row = cursor.fetchone()
                    cursor.execute(
                        cast(
                            Any,
                            """
                            INSERT INTO exchange_credential_versions (
                                credential_version_id,
                                connection_id,
                                api_key_ciphertext,
                                api_secret_ciphertext,
                                passphrase_ciphertext,
                                api_key_last4,
                                api_key_fingerprint_hmac,
                                secret_cipher,
                                transit_key_id,
                                credential_scheme,
                                status,
                                created_by_user_id,
                                created_by_session_id,
                                created_at,
                                rotated_at,
                                disabled_at
                            )
                            VALUES (
                                %(credential_version_id)s,
                                %(connection_id)s,
                                %(api_key_ciphertext)s,
                                %(api_secret_ciphertext)s,
                                %(passphrase_ciphertext)s,
                                %(api_key_last4)s,
                                %(api_key_fingerprint_hmac)s,
                                %(secret_cipher)s,
                                %(transit_key_id)s,
                                %(credential_scheme)s,
                                %(status)s,
                                %(created_by_user_id)s,
                                NULL,
                                %(created_at)s,
                                %(rotated_at)s,
                                %(disabled_at)s
                            )
                            """,
                        ),
                        _credential_parameters(credential_version=credential_version),
                    )
        except errors.UniqueViolation:
            return None
        if created_row is None:
            return None
        return _map_connection(row=dict(created_row))

    def get(self, *, connection_id: UUID) -> ExchangeConnectionRecord | None:
        row = self._fetch_connection(
            where="connection.connection_id = %(connection_id)s",
            parameters={"connection_id": str(connection_id)},
        )
        return _map_connection(row=row) if row is not None else None

    def list_for_user(self, *, owner_user_id: UserId) -> tuple[ExchangeConnectionRecord, ...]:
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT
                            connection.connection_id,
                            connection.owner_user_id,
                            connection.exchange_name,
                            connection.market_type,
                            connection.environment,
                            connection.label,
                            connection.active_credential_version_id,
                            connection.status,
                            connection.status_reason,
                            connection.permission_summary_json ->> 'permissions'
                                AS permissions,
                            connection.permission_summary_json ->> 'validation_status'
                                AS validation_status,
                            connection.permission_summary_json ->> 'validation_reason'
                                AS validation_reason,
                            connection.ip_restriction_status,
                            connection.last_validated_at,
                            connection.created_at,
                            connection.updated_at,
                            connection.disabled_at
                        FROM exchange_connections AS connection
                        WHERE connection.owner_user_id = %(owner_user_id)s
                        ORDER BY connection.created_at ASC, connection.connection_id ASC
                        """,
                    ),
                    {"owner_user_id": str(owner_user_id)},
                )
                rows = cursor.fetchall()
        return tuple(_map_connection(row=dict(row)) for row in rows)

    def get_active_credential(
        self, *, connection_id: UUID
    ) -> ExchangeCredentialVersionRecord | None:
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT credential.*
                        FROM exchange_connections AS connection
                        JOIN exchange_credential_versions AS credential
                          ON credential.credential_version_id =
                             connection.active_credential_version_id
                        WHERE connection.connection_id = %(connection_id)s
                        """,
                    ),
                    {"connection_id": str(connection_id)},
                )
                row = cursor.fetchone()
        return _map_credential(row=dict(row)) if row is not None else None

    def replace_active_credential(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        credential_version: ExchangeCredentialVersionRecord,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT active_credential_version_id
                        FROM exchange_connections
                        WHERE connection_id = %(connection_id)s
                          AND owner_user_id = %(owner_user_id)s
                          AND status = 'active'
                        FOR UPDATE
                        """,
                    ),
                    {
                        "connection_id": str(connection_id),
                        "owner_user_id": str(owner_user_id),
                    },
                )
                current = cursor.fetchone()
                if current is None:
                    return None
                current_row = dict(current)
                cursor.execute(
                    cast(
                        Any,
                        """
                        UPDATE exchange_credential_versions
                        SET status = 'rotated',
                            rotated_at = %(rotated_at)s
                        WHERE credential_version_id = %(credential_version_id)s
                        """,
                    ),
                    {
                        "rotated_at": updated_at,
                        "credential_version_id": str(
                            current_row["active_credential_version_id"]
                        ),
                    },
                )
                cursor.execute(
                    cast(
                        Any,
                        """
                        INSERT INTO exchange_credential_versions (
                            credential_version_id,
                            connection_id,
                            api_key_ciphertext,
                            api_secret_ciphertext,
                            passphrase_ciphertext,
                            api_key_last4,
                            api_key_fingerprint_hmac,
                            secret_cipher,
                            transit_key_id,
                            credential_scheme,
                            status,
                            created_by_user_id,
                            created_by_session_id,
                            created_at,
                            rotated_at,
                            disabled_at
                        )
                        VALUES (
                            %(credential_version_id)s,
                            %(connection_id)s,
                            %(api_key_ciphertext)s,
                            %(api_secret_ciphertext)s,
                            %(passphrase_ciphertext)s,
                            %(api_key_last4)s,
                            %(api_key_fingerprint_hmac)s,
                            %(secret_cipher)s,
                            %(transit_key_id)s,
                            %(credential_scheme)s,
                            %(status)s,
                            %(created_by_user_id)s,
                            NULL,
                            %(created_at)s,
                            %(rotated_at)s,
                            %(disabled_at)s
                        )
                        """,
                    ),
                    _credential_parameters(credential_version=credential_version),
                )
                cursor.execute(
                    cast(
                        Any,
                        """
                        UPDATE exchange_connections AS connection
                        SET active_credential_version_id = %(credential_version_id)s,
                            permission_summary_json =
                                connection.permission_summary_json
                                || jsonb_build_object(
                                    'validation_status', 'skipped_external_validation',
                                    'validation_reason', 'credential_rotated'
                                ),
                            ip_restriction_status = 'unknown',
                            last_validated_at = NULL,
                            updated_at = %(updated_at)s
                        WHERE connection.connection_id = %(connection_id)s
                          AND connection.owner_user_id = %(owner_user_id)s
                          AND connection.status = 'active'
                        RETURNING
                            connection.connection_id,
                            connection.owner_user_id,
                            connection.exchange_name,
                            connection.market_type,
                            connection.environment,
                            connection.label,
                            connection.active_credential_version_id,
                            connection.status,
                            connection.status_reason,
                            connection.permission_summary_json ->> 'permissions'
                                AS permissions,
                            connection.permission_summary_json ->> 'validation_status'
                                AS validation_status,
                            connection.permission_summary_json ->> 'validation_reason'
                                AS validation_reason,
                            connection.ip_restriction_status,
                            connection.last_validated_at,
                            connection.created_at,
                            connection.updated_at,
                            connection.disabled_at
                        """,
                    ),
                    {
                        "credential_version_id": str(
                            credential_version.credential_version_id
                        ),
                        "updated_at": updated_at,
                        "connection_id": str(connection_id),
                        "owner_user_id": str(owner_user_id),
                    },
                )
                row = cursor.fetchone()
        return _map_connection(row=dict(row)) if row is not None else None

    def disable(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        disabled_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT active_credential_version_id, status
                        FROM exchange_connections
                        WHERE connection_id = %(connection_id)s
                          AND owner_user_id = %(owner_user_id)s
                        FOR UPDATE
                        """,
                    ),
                    {
                        "connection_id": str(connection_id),
                        "owner_user_id": str(owner_user_id),
                    },
                )
                current = cursor.fetchone()
                if current is None:
                    return None
                current_row = dict(current)
                if str(current_row["status"]) == "disabled":
                    return None
                cursor.execute(
                    cast(
                        Any,
                        """
                        UPDATE exchange_credential_versions
                        SET status = 'disabled',
                            disabled_at = %(disabled_at)s
                        WHERE credential_version_id = %(credential_version_id)s
                        """,
                    ),
                    {
                        "disabled_at": disabled_at,
                        "credential_version_id": str(
                            current_row["active_credential_version_id"]
                        ),
                    },
                )
                cursor.execute(
                    cast(
                        Any,
                        """
                        UPDATE exchange_connections AS connection
                        SET status = 'disabled',
                            status_reason = 'user_disabled',
                            updated_at = %(updated_at)s,
                            disabled_at = %(disabled_at)s
                        WHERE connection.connection_id = %(connection_id)s
                          AND connection.owner_user_id = %(owner_user_id)s
                        RETURNING
                            connection.connection_id,
                            connection.owner_user_id,
                            connection.exchange_name,
                            connection.market_type,
                            connection.environment,
                            connection.label,
                            connection.active_credential_version_id,
                            connection.status,
                            connection.status_reason,
                            connection.permission_summary_json ->> 'permissions'
                                AS permissions,
                            connection.permission_summary_json ->> 'validation_status'
                                AS validation_status,
                            connection.permission_summary_json ->> 'validation_reason'
                                AS validation_reason,
                            connection.ip_restriction_status,
                            connection.last_validated_at,
                            connection.created_at,
                            connection.updated_at,
                            connection.disabled_at
                        """,
                    ),
                    {
                        "updated_at": disabled_at,
                        "disabled_at": disabled_at,
                        "connection_id": str(connection_id),
                        "owner_user_id": str(owner_user_id),
                    },
                )
                row = cursor.fetchone()
        return _map_connection(row=dict(row)) if row is not None else None

    def record_validation(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        result: ExchangeCredentialValidationResult,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        observed_at = result.observed_at or updated_at
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        UPDATE exchange_connections AS connection
                        SET permission_summary_json =
                                connection.permission_summary_json
                                || %(permission_summary_json)s::jsonb
                                || jsonb_build_object(
                                    'validation_status', %(validation_status)s::text,
                                    'validation_reason', %(validation_reason)s::text
                                ),
                            ip_restriction_status = %(ip_restriction_status)s,
                            last_validated_at = %(last_validated_at)s,
                            updated_at = %(updated_at)s
                        WHERE connection.connection_id = %(connection_id)s
                          AND connection.owner_user_id = %(owner_user_id)s
                          AND connection.status = 'active'
                        RETURNING
                            connection.connection_id,
                            connection.owner_user_id,
                            connection.exchange_name,
                            connection.market_type,
                            connection.environment,
                            connection.label,
                            connection.active_credential_version_id,
                            connection.status,
                            connection.status_reason,
                            connection.permission_summary_json ->> 'permissions'
                                AS permissions,
                            connection.permission_summary_json ->> 'validation_status'
                                AS validation_status,
                            connection.permission_summary_json ->> 'validation_reason'
                                AS validation_reason,
                            connection.ip_restriction_status,
                            connection.last_validated_at,
                            connection.created_at,
                            connection.updated_at,
                            connection.disabled_at
                        """,
                    ),
                    {
                        "permission_summary_json": json.dumps(
                            result.permission_summary or {}
                        ),
                        "validation_status": result.status,
                        "validation_reason": result.reason,
                        "ip_restriction_status": result.ip_restriction_status,
                        "last_validated_at": observed_at,
                        "updated_at": updated_at,
                        "connection_id": str(connection_id),
                        "owner_user_id": str(owner_user_id),
                    },
                )
                row = cursor.fetchone()
        return _map_connection(row=dict(row)) if row is not None else None

    def _fetch_connection(
        self,
        *,
        where: str,
        parameters: Mapping[str, object],
    ) -> Mapping[str, Any] | None:
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as postgres_connection:
            with postgres_connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        f"""
                        SELECT
                            connection.connection_id,
                            connection.owner_user_id,
                            connection.exchange_name,
                            connection.market_type,
                            connection.environment,
                            connection.label,
                            connection.active_credential_version_id,
                            connection.status,
                            connection.status_reason,
                            connection.permission_summary_json ->> 'permissions'
                                AS permissions,
                            connection.permission_summary_json ->> 'validation_status'
                                AS validation_status,
                            connection.permission_summary_json ->> 'validation_reason'
                                AS validation_reason,
                            connection.ip_restriction_status,
                            connection.last_validated_at,
                            connection.created_at,
                            connection.updated_at,
                            connection.disabled_at
                        FROM exchange_connections AS connection
                        WHERE {where}
                        """,
                    ),
                    parameters,
                )
                row = cursor.fetchone()
        return dict(row) if row is not None else None


def _connection_parameters(
    *,
    connection: ExchangeConnectionRecord,
) -> dict[str, object]:
    return {
        "connection_id": str(connection.connection_id),
        "owner_user_id": str(connection.owner_user_id),
        "exchange_name": connection.exchange_name,
        "market_type": connection.market_type,
        "environment": connection.environment,
        "label": connection.label,
        "active_credential_version_id": str(connection.active_credential_version_id),
        "status": connection.status,
        "status_reason": connection.status_reason,
        "permissions": connection.permissions,
        "validation_status": connection.validation_status,
        "validation_reason": connection.validation_reason,
        "created_at": connection.created_at,
        "updated_at": connection.updated_at,
        "disabled_at": connection.disabled_at,
    }


def _credential_parameters(
    *,
    credential_version: ExchangeCredentialVersionRecord,
) -> dict[str, object]:
    return {
        "credential_version_id": str(credential_version.credential_version_id),
        "connection_id": str(credential_version.connection_id),
        "api_key_ciphertext": credential_version.api_key_ciphertext,
        "api_secret_ciphertext": credential_version.api_secret_ciphertext,
        "passphrase_ciphertext": credential_version.passphrase_ciphertext,
        "api_key_last4": credential_version.api_key_last4,
        "api_key_fingerprint_hmac": _fingerprint_bytes(
            credential_version.api_key_fingerprint_hmac
        ),
        "secret_cipher": credential_version.secret_cipher,
        "transit_key_id": credential_version.transit_key_id,
        "credential_scheme": credential_version.credential_scheme,
        "status": credential_version.status,
        "created_by_user_id": str(credential_version.created_by_user_id),
        "created_at": credential_version.created_at,
        "rotated_at": credential_version.rotated_at,
        "disabled_at": credential_version.disabled_at,
    }


def _map_connection(*, row: Mapping[str, Any]) -> ExchangeConnectionRecord:
    permissions = row.get("permissions")
    return ExchangeConnectionRecord(
        connection_id=UUID(str(row["connection_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_name=str(row["exchange_name"]),
        market_type=str(row["market_type"]),
        environment=str(row["environment"]),
        label=str(row["label"]) if row["label"] is not None else None,
        permissions=str(permissions) if permissions is not None else "read",
        active_credential_version_id=UUID(str(row["active_credential_version_id"])),
        status=str(row["status"]),
        status_reason=(
            str(row["status_reason"]) if row["status_reason"] is not None else None
        ),
        validation_status=str(
            row.get("validation_status") or "skipped_external_validation"
        ),
        validation_reason=(
            str(row.get("validation_reason"))
            if row.get("validation_reason") is not None
            else "not_validated"
        ),
        ip_restriction_status=str(row.get("ip_restriction_status") or "unknown"),
        last_validated_at=_normalize_optional_utc_datetime(
            value=row.get("last_validated_at")
        ),
        created_at=_normalize_utc_datetime(value=row["created_at"]),
        updated_at=_normalize_utc_datetime(value=row["updated_at"]),
        disabled_at=_normalize_optional_utc_datetime(value=row["disabled_at"]),
    )


def _map_credential(*, row: Mapping[str, Any]) -> ExchangeCredentialVersionRecord:
    return ExchangeCredentialVersionRecord(
        credential_version_id=UUID(str(row["credential_version_id"])),
        connection_id=UUID(str(row["connection_id"])),
        api_key_ciphertext=str(row["api_key_ciphertext"]),
        api_secret_ciphertext=str(row["api_secret_ciphertext"]),
        passphrase_ciphertext=(
            str(row["passphrase_ciphertext"])
            if row["passphrase_ciphertext"] is not None
            else None
        ),
        api_key_last4=str(row["api_key_last4"]),
        api_key_fingerprint_hmac=_fingerprint_text(row["api_key_fingerprint_hmac"]),
        secret_cipher=str(row["secret_cipher"]),
        transit_key_id=str(row["transit_key_id"]),
        credential_scheme=str(row["credential_scheme"]),
        status=str(row["status"]),
        created_by_user_id=UserId.from_string(str(row["created_by_user_id"])),
        created_at=_normalize_utc_datetime(value=row["created_at"]),
        rotated_at=_normalize_optional_utc_datetime(value=row["rotated_at"]),
        disabled_at=_normalize_optional_utc_datetime(value=row["disabled_at"]),
    )


def _fingerprint_bytes(value: str) -> bytes:
    return value.encode("utf-8")


def _fingerprint_text(value: object) -> str:
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return bytes(value).decode("utf-8")  # type: ignore[arg-type]


def _normalize_utc_datetime(*, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Postgres datetime value is invalid")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_optional_utc_datetime(*, value: object) -> datetime | None:
    if value is None:
        return None
    return _normalize_utc_datetime(value=value)


__all__ = ["PostgresExchangeConnectionRepository"]
