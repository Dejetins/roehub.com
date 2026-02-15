from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.two_factor_repository import TwoFactorRepository
from trading.contexts.identity.domain.entities import TwoFactorAuth
from trading.shared_kernel.primitives import UserId


class PostgresIdentityTwoFactorRepository(TwoFactorRepository):
    """
    PostgresIdentityTwoFactorRepository — Postgres adapter for identity 2FA storage port.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_repository.py
      - migrations/postgres/0002_identity_2fa_totp_v1.sql
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
    """

    def __init__(
        self,
        *,
        gateway: IdentityPostgresGateway,
        two_factor_table: str = "identity_2fa",
    ) -> None:
        """
        Initialize repository with SQL gateway and target 2FA table name.

        Args:
            gateway: SQL gateway abstraction.
            two_factor_table: Target 2FA table name.
        Returns:
            None.
        Assumptions:
            Table schema follows migration `0002_identity_2fa_totp_v1.sql`.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresIdentityTwoFactorRepository requires gateway")
        normalized_table = two_factor_table.strip()
        if not normalized_table:
            raise ValueError("PostgresIdentityTwoFactorRepository requires non-empty table name")

        self._gateway = gateway
        self._table = normalized_table

    def find_by_user_id(self, *, user_id: UserId) -> TwoFactorAuth | None:
        """
        Find 2FA state row by stable user id.

        Args:
            user_id: Identity user identifier.
        Returns:
            TwoFactorAuth | None: Persisted 2FA state or `None`.
        Assumptions:
            `user_id` is primary key in identity_2fa table.
        Raises:
            ValueError: If row mapping is malformed.
        Side Effects:
            Executes one SQL SELECT statement.
        """
        query = f"""
        SELECT
            user_id,
            totp_secret_enc,
            enabled,
            enabled_at,
            updated_at
        FROM {self._table}
        WHERE user_id = %(user_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"user_id": str(user_id)},
        )
        if row is None:
            return None
        return _map_two_factor_row(row=row)

    def upsert_pending_secret(
        self,
        *,
        user_id: UserId,
        totp_secret_enc: bytes,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Store pending encrypted secret, skipping overwrite when state is already enabled.

        Args:
            user_id: Identity user identifier.
            totp_secret_enc: Encrypted opaque TOTP secret blob.
            updated_at: UTC update timestamp.
        Returns:
            TwoFactorAuth: Persisted or concurrent existing row snapshot.
        Assumptions:
            Caller should already enforce Option 1 policy before invoking this method.
        Raises:
            ValueError: If repository cannot return resulting row mapping.
        Side Effects:
            Executes one SQL upsert statement and optional fallback select.
        """
        query = f"""
        INSERT INTO {self._table}
        (
            user_id,
            totp_secret_enc,
            enabled,
            enabled_at,
            updated_at
        )
        VALUES
        (
            %(user_id)s,
            %(totp_secret_enc)s,
            FALSE,
            NULL,
            %(updated_at)s
        )
        ON CONFLICT (user_id)
        DO UPDATE
        SET
            totp_secret_enc = EXCLUDED.totp_secret_enc,
            updated_at = EXCLUDED.updated_at
        WHERE NOT {self._table}.enabled
        RETURNING
            user_id,
            totp_secret_enc,
            enabled,
            enabled_at,
            updated_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "totp_secret_enc": bytes(totp_secret_enc),
                "updated_at": updated_at,
            },
        )
        if row is None:
            existing = self.find_by_user_id(user_id=user_id)
            if existing is None:
                raise ValueError("PostgresIdentityTwoFactorRepository upsert returned no row")
            return existing
        return _map_two_factor_row(row=row)

    def enable(
        self,
        *,
        user_id: UserId,
        enabled_at: datetime,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Mark existing row as enabled and persist `enabled_at` plus `updated_at`.

        Args:
            user_id: Identity user identifier.
            enabled_at: UTC timestamp when 2FA becomes enabled.
            updated_at: UTC update timestamp.
        Returns:
            TwoFactorAuth: Persisted enabled state snapshot.
        Assumptions:
            Pending setup row exists in storage.
        Raises:
            ValueError: If row does not exist and cannot be enabled.
        Side Effects:
            Executes one SQL update statement and optional fallback select.
        """
        query = f"""
        UPDATE {self._table}
        SET
            enabled = TRUE,
            enabled_at = %(enabled_at)s,
            updated_at = %(updated_at)s
        WHERE user_id = %(user_id)s
          AND enabled = FALSE
        RETURNING
            user_id,
            totp_secret_enc,
            enabled,
            enabled_at,
            updated_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "enabled_at": enabled_at,
                "updated_at": updated_at,
            },
        )
        if row is None:
            existing = self.find_by_user_id(user_id=user_id)
            if existing is None:
                raise ValueError("PostgresIdentityTwoFactorRepository cannot enable missing row")
            return existing
        return _map_two_factor_row(row=row)


def _map_two_factor_row(*, row: Mapping[str, Any]) -> TwoFactorAuth:
    """
    Map SQL row mapping into immutable domain `TwoFactorAuth` entity.

    Args:
        row: SQL result mapping.
    Returns:
        TwoFactorAuth: Domain 2FA state entity.
    Assumptions:
        Row follows schema from `identity_2fa` table.
    Raises:
        ValueError: If required fields are missing or malformed.
    Side Effects:
        None.
    """
    try:
        secret_raw = row["totp_secret_enc"]
        if isinstance(secret_raw, memoryview):
            secret_enc = secret_raw.tobytes()
        else:
            secret_enc = bytes(secret_raw)
        return TwoFactorAuth(
            user_id=UserId.from_string(str(row["user_id"])),
            totp_secret_enc=secret_enc,
            enabled=bool(row["enabled"]),
            enabled_at=row["enabled_at"],
            updated_at=row["updated_at"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PostgresIdentityTwoFactorRepository cannot map 2FA row") from error
