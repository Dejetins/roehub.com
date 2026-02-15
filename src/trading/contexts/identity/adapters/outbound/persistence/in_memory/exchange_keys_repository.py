from __future__ import annotations

from datetime import datetime
from uuid import UUID

from trading.contexts.identity.application.ports.exchange_keys_repository import (
    ExchangeKeysRepository,
)
from trading.contexts.identity.domain.entities import ExchangeKey
from trading.shared_kernel.primitives import UserId


class InMemoryIdentityExchangeKeysRepository(ExchangeKeysRepository):
    """
    InMemoryIdentityExchangeKeysRepository — deterministic in-memory exchange keys storage.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/
        exchange_keys_repository.py
      - tests/unit/contexts/identity/application/test_exchange_keys_use_cases.py
    """

    def __init__(self) -> None:
        """
        Initialize isolated in-memory rows map keyed by `key_id` string.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Repository instance is process-local and deterministic for tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._rows: dict[str, ExchangeKey] = {}

    def create(
        self,
        *,
        key_id: UUID,
        user_id: UserId,
        exchange_name: str,
        market_type: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret_enc: bytes,
        passphrase_enc: bytes | None,
        created_at: datetime,
        updated_at: datetime,
    ) -> ExchangeKey | None:
        """
        Persist new active key or return `None` for active duplicate.

        Args:
            key_id: Generated exchange key identifier.
            user_id: Owner identity user id.
            exchange_name: Exchange literal.
            market_type: Market type literal.
            label: Optional user label.
            permissions: Permission literal.
            api_key: API key value.
            api_secret_enc: Encrypted API secret bytes.
            passphrase_enc: Optional encrypted passphrase bytes.
            created_at: UTC creation timestamp.
            updated_at: UTC update timestamp.
        Returns:
            ExchangeKey | None: Created row or `None` when active duplicate exists.
        Assumptions:
            Active duplicate uniqueness is `(user_id, exchange_name, market_type, api_key)`.
        Raises:
            ValueError: If domain invariants reject constructed row.
        Side Effects:
            Mutates in-memory rows map.
        """
        for existing in self._rows.values():
            if existing.is_deleted:
                continue
            if existing.user_id != user_id:
                continue
            if existing.exchange_name != exchange_name:
                continue
            if existing.market_type != market_type:
                continue
            if existing.api_key != api_key:
                continue
            return None

        row = ExchangeKey(
            key_id=key_id,
            user_id=user_id,
            exchange_name=exchange_name,
            market_type=market_type,
            label=label,
            permissions=permissions,
            api_key=api_key,
            api_secret_enc=bytes(api_secret_enc),
            passphrase_enc=bytes(passphrase_enc) if passphrase_enc is not None else None,
            created_at=created_at,
            updated_at=updated_at,
            is_deleted=False,
            deleted_at=None,
        )
        self._rows[str(key_id)] = row
        return row

    def list_active_for_user(self, *, user_id: UserId) -> tuple[ExchangeKey, ...]:
        """
        Return active rows for user sorted by `created_at ASC, key_id ASC`.

        Args:
            user_id: Owner identity user id.
        Returns:
            tuple[ExchangeKey, ...]: Sorted active rows.
        Assumptions:
            Soft-deleted rows must not appear in result.
        Raises:
            None.
        Side Effects:
            None.
        """
        rows = [
            row
            for row in self._rows.values()
            if row.user_id == user_id and not row.is_deleted
        ]
        rows.sort(key=lambda item: (item.created_at, str(item.key_id)))
        return tuple(rows)

    def soft_delete(
        self,
        *,
        user_id: UserId,
        key_id: UUID,
        deleted_at: datetime,
        updated_at: datetime,
    ) -> bool:
        """
        Soft-delete row if it belongs to user and is currently active.

        Args:
            user_id: Owner identity user id.
            key_id: Exchange key identifier.
            deleted_at: UTC delete timestamp.
            updated_at: UTC update timestamp.
        Returns:
            bool: `True` when row was soft-deleted, otherwise `False`.
        Assumptions:
            Already-deleted rows are treated as not found.
        Raises:
            ValueError: If domain invariants reject updated row.
        Side Effects:
            Mutates one in-memory row when key is found.
        """
        existing = self._rows.get(str(key_id))
        if existing is None:
            return False
        if existing.user_id != user_id:
            return False
        if existing.is_deleted:
            return False

        deleted_row = ExchangeKey(
            key_id=existing.key_id,
            user_id=existing.user_id,
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            label=existing.label,
            permissions=existing.permissions,
            api_key=existing.api_key,
            api_secret_enc=existing.api_secret_enc,
            passphrase_enc=existing.passphrase_enc,
            created_at=existing.created_at,
            updated_at=updated_at,
            is_deleted=True,
            deleted_at=deleted_at,
        )
        self._rows[str(key_id)] = deleted_row
        return True
