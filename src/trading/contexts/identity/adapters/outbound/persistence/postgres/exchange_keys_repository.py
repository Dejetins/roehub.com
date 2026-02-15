from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.exchange_keys_repository import (
    ExchangeKeysRepository,
)
from trading.contexts.identity.domain.entities import ExchangeKey
from trading.shared_kernel.primitives import UserId


class PostgresIdentityExchangeKeysRepository(ExchangeKeysRepository):
    """
    PostgresIdentityExchangeKeysRepository — Postgres adapter for exchange keys storage.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
    """

    def __init__(
        self,
        *,
        gateway: IdentityPostgresGateway,
        table_name: str = "identity_exchange_keys",
    ) -> None:
        """
        Initialize repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            table_name: Target exchange keys table name.
        Returns:
            None.
        Assumptions:
            Table schema follows migration `0004_identity_exchange_keys_v2.sql`.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresIdentityExchangeKeysRepository requires gateway")
        normalized_table_name = table_name.strip()
        if not normalized_table_name:
            raise ValueError("PostgresIdentityExchangeKeysRepository requires non-empty table_name")
        self._gateway = gateway
        self._table_name = normalized_table_name

    def create(
        self,
        *,
        key_id: UUID,
        user_id: UserId,
        exchange_name: str,
        market_type: str,
        label: str | None,
        permissions: str,
        api_key_enc: bytes,
        api_key_hash: bytes,
        api_key_last4: str,
        api_secret_enc: bytes,
        passphrase_enc: bytes | None,
        created_at: datetime,
        updated_at: datetime,
    ) -> ExchangeKey | None:
        """
        Insert active exchange key row and return persisted snapshot.

        Args:
            key_id: Generated exchange key identifier.
            user_id: Owner identity user id.
            exchange_name: Exchange literal.
            market_type: Market type literal.
            label: Optional label.
            permissions: Permission literal.
            api_key_enc: Encrypted API key bytes.
            api_key_hash: Deterministic API key hash bytes.
            api_key_last4: Deterministic API key suffix for masked responses.
            api_secret_enc: Encrypted API secret bytes.
            passphrase_enc: Optional encrypted passphrase bytes.
            created_at: UTC creation timestamp.
            updated_at: UTC update timestamp.
        Returns:
            ExchangeKey | None: Created row or `None` when insert conflicts with active duplicate.
        Assumptions:
            Active uniqueness is enforced by partial unique index.
        Raises:
            ValueError: If adapter cannot map returned row.
        Side Effects:
            Executes one SQL insert statement.
        """
        query = f"""
        INSERT INTO {self._table_name}
        (
            key_id,
            user_id,
            exchange_name,
            market_type,
            label,
            permissions,
            api_key_enc,
            api_key_hash,
            api_key_last4,
            api_secret_enc,
            passphrase_enc,
            created_at,
            updated_at,
            is_deleted,
            deleted_at
        )
        VALUES
        (
            %(key_id)s,
            %(user_id)s,
            %(exchange_name)s,
            %(market_type)s,
            %(label)s,
            %(permissions)s,
            %(api_key_enc)s,
            %(api_key_hash)s,
            %(api_key_last4)s,
            %(api_secret_enc)s,
            %(passphrase_enc)s,
            %(created_at)s,
            %(updated_at)s,
            FALSE,
            NULL
        )
        ON CONFLICT DO NOTHING
        RETURNING
            key_id,
            user_id,
            exchange_name,
            market_type,
            label,
            permissions,
            api_key_enc,
            api_key_hash,
            api_key_last4,
            api_secret_enc,
            passphrase_enc,
            created_at,
            updated_at,
            is_deleted,
            deleted_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "key_id": str(key_id),
                "user_id": str(user_id),
                "exchange_name": exchange_name,
                "market_type": market_type,
                "label": label,
                "permissions": permissions,
                "api_key_enc": bytes(api_key_enc),
                "api_key_hash": bytes(api_key_hash),
                "api_key_last4": api_key_last4,
                "api_secret_enc": bytes(api_secret_enc),
                "passphrase_enc": bytes(passphrase_enc) if passphrase_enc is not None else None,
                "created_at": created_at,
                "updated_at": updated_at,
            },
        )
        if row is None:
            return None
        return _map_exchange_key_row(row=row)

    def list_active_for_user(self, *, user_id: UserId) -> tuple[ExchangeKey, ...]:
        """
        Return active rows for user sorted by `created_at ASC, key_id ASC`.

        Args:
            user_id: Owner identity user id.
        Returns:
            tuple[ExchangeKey, ...]: Sorted active exchange key rows.
        Assumptions:
            Soft-deleted rows are excluded by SQL filter.
        Raises:
            ValueError: If one of rows cannot be mapped.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            key_id,
            user_id,
            exchange_name,
            market_type,
            label,
            permissions,
            api_key_enc,
            api_key_hash,
            api_key_last4,
            api_secret_enc,
            passphrase_enc,
            created_at,
            updated_at,
            is_deleted,
            deleted_at
        FROM {self._table_name}
        WHERE user_id = %(user_id)s
          AND is_deleted = FALSE
        ORDER BY created_at ASC, key_id ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={"user_id": str(user_id)},
        )
        return tuple(_map_exchange_key_row(row=row) for row in rows)

    def soft_delete(
        self,
        *,
        user_id: UserId,
        key_id: UUID,
        deleted_at: datetime,
        updated_at: datetime,
    ) -> bool:
        """
        Soft-delete owned active row and return operation status.

        Args:
            user_id: Owner identity user id.
            key_id: Exchange key identifier.
            deleted_at: UTC delete timestamp.
            updated_at: UTC update timestamp.
        Returns:
            bool: `True` when row was updated, otherwise `False`.
        Assumptions:
            Missing, foreign, and already-deleted rows are treated as no-op.
        Raises:
            ValueError: If gateway update execution fails.
        Side Effects:
            Executes one SQL update statement.
        """
        query = f"""
        UPDATE {self._table_name}
        SET
            is_deleted = TRUE,
            deleted_at = %(deleted_at)s,
            updated_at = %(updated_at)s
        WHERE key_id = %(key_id)s
          AND user_id = %(user_id)s
          AND is_deleted = FALSE
        RETURNING key_id
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "key_id": str(key_id),
                "user_id": str(user_id),
                "deleted_at": deleted_at,
                "updated_at": updated_at,
            },
        )
        return row is not None



def _map_exchange_key_row(*, row: Mapping[str, Any]) -> ExchangeKey:
    """
    Map SQL row mapping into immutable domain `ExchangeKey` entity.

    Args:
        row: SQL result mapping.
    Returns:
        ExchangeKey: Domain exchange key entity.
    Assumptions:
        Row follows schema from `identity_exchange_keys` table.
    Raises:
        ValueError: If row fields are missing or malformed.
    Side Effects:
        None.
    """
    try:
        api_key_enc_raw = row["api_key_enc"]
        api_key_hash_raw = row["api_key_hash"]
        api_key_last4_raw = row["api_key_last4"]
        api_secret_raw = row["api_secret_enc"]
        passphrase_raw = row["passphrase_enc"]

        if isinstance(api_key_enc_raw, memoryview):
            api_key_enc = api_key_enc_raw.tobytes()
        else:
            api_key_enc = bytes(api_key_enc_raw)

        if isinstance(api_key_hash_raw, memoryview):
            api_key_hash = api_key_hash_raw.tobytes()
        else:
            api_key_hash = bytes(api_key_hash_raw)
        if api_key_last4_raw is None:
            raise ValueError("api_key_last4 must be present in exchange key row")
        api_key_last4 = str(api_key_last4_raw)

        if isinstance(api_secret_raw, memoryview):
            api_secret_enc = api_secret_raw.tobytes()
        else:
            api_secret_enc = bytes(api_secret_raw)

        if passphrase_raw is None:
            passphrase_enc = None
        elif isinstance(passphrase_raw, memoryview):
            passphrase_enc = passphrase_raw.tobytes()
        else:
            passphrase_enc = bytes(passphrase_raw)

        return ExchangeKey(
            key_id=UUID(str(row["key_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            exchange_name=str(row["exchange_name"]),
            market_type=str(row["market_type"]),
            label=str(row["label"]) if row["label"] is not None else None,
            permissions=str(row["permissions"]),
            api_key_enc=api_key_enc,
            api_key_hash=api_key_hash,
            api_key_last4=api_key_last4,
            api_secret_enc=api_secret_enc,
            passphrase_enc=passphrase_enc,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
            deleted_at=row["deleted_at"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "PostgresIdentityExchangeKeysRepository cannot map exchange key row"
        ) from error
