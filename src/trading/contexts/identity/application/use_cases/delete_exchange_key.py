from __future__ import annotations

from datetime import datetime
from uuid import UUID

from trading.contexts.identity.application.ports import ExchangeKeysRepository, IdentityClock
from trading.contexts.identity.application.use_cases.exchange_keys_errors import (
    ExchangeKeyNotFoundError,
)
from trading.shared_kernel.primitives import UserId


class DeleteExchangeKeyUseCase:
    """
    DeleteExchangeKeyUseCase — soft-delete one owned active exchange key.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    def __init__(
        self,
        *,
        repository: ExchangeKeysRepository,
        clock: IdentityClock,
    ) -> None:
        """
        Initialize use-case dependencies for repository and deterministic time.

        Args:
            repository: Exchange keys storage port.
            clock: UTC clock port.
        Returns:
            None.
        Assumptions:
            Repository implements soft-delete semantics.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("DeleteExchangeKeyUseCase requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("DeleteExchangeKeyUseCase requires clock")
        self._repository = repository
        self._clock = clock

    def delete(self, *, user_id: UserId, key_id: UUID) -> None:
        """
        Soft-delete key row if it belongs to user and is still active.

        Args:
            user_id: Owner identity user id.
            key_id: Exchange key identifier.
        Returns:
            None.
        Assumptions:
            Missing, not-owned, and already-deleted states are all treated as not found.
        Raises:
            ExchangeKeyNotFoundError: If key cannot be soft-deleted.
        Side Effects:
            Writes one storage record when row is found.
        """
        now = _ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        deleted = self._repository.soft_delete(
            user_id=user_id,
            key_id=key_id,
            deleted_at=now,
            updated_at=now,
        )
        if not deleted:
            raise ExchangeKeyNotFoundError()



def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate datetime is timezone-aware UTC and return same value.

    Args:
        value: Datetime value to validate.
        field_name: Field label for deterministic error messages.
    Returns:
        datetime: Same validated datetime.
    Assumptions:
        UTC datetimes have zero offset.
    Raises:
        ValueError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC datetime")
    return value
