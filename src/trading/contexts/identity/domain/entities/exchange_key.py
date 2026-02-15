from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from trading.shared_kernel.primitives import UserId

_ALLOWED_EXCHANGE_NAMES = {"binance", "bybit"}
_ALLOWED_MARKET_TYPES = {"spot", "futures"}
_ALLOWED_PERMISSIONS = {"read", "trade"}


@dataclass(frozen=True, slots=True)
class ExchangeKey:
    """
    ExchangeKey — immutable identity exchange key storage snapshot.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
    """

    key_id: UUID
    user_id: UserId
    exchange_name: str
    market_type: str
    label: str | None
    permissions: str
    api_key_enc: bytes
    api_key_hash: bytes
    api_key_last4: str
    api_secret_enc: bytes
    passphrase_enc: bytes | None
    created_at: datetime
    updated_at: datetime
    is_deleted: bool
    deleted_at: datetime | None

    def __post_init__(self) -> None:
        """
        Validate exchange key invariants for enums, timestamps, and soft-delete state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `created_at`, `updated_at`, and `deleted_at` (when set) are timezone-aware UTC.
        Raises:
            ValueError: If enum values, payload fields, or soft-delete invariants are invalid.
        Side Effects:
            None.
        """
        if self.exchange_name not in _ALLOWED_EXCHANGE_NAMES:
            raise ValueError("ExchangeKey.exchange_name must be one of {'binance', 'bybit'}")
        if self.market_type not in _ALLOWED_MARKET_TYPES:
            raise ValueError("ExchangeKey.market_type must be one of {'spot', 'futures'}")
        if self.permissions not in _ALLOWED_PERMISSIONS:
            raise ValueError("ExchangeKey.permissions must be one of {'read', 'trade'}")
        if not self.api_key_enc:
            raise ValueError("ExchangeKey.api_key_enc must be non-empty")
        if len(self.api_key_hash) != 32:
            raise ValueError("ExchangeKey.api_key_hash must be exactly 32 bytes (SHA-256)")
        normalized_api_key_last4 = self.api_key_last4.strip()
        if not normalized_api_key_last4:
            raise ValueError("ExchangeKey.api_key_last4 must be non-empty")
        if len(normalized_api_key_last4) > 4:
            raise ValueError("ExchangeKey.api_key_last4 length must be <= 4")
        if not self.api_secret_enc:
            raise ValueError("ExchangeKey.api_secret_enc must be non-empty")
        if self.passphrase_enc is not None and not self.passphrase_enc:
            raise ValueError("ExchangeKey.passphrase_enc must be non-empty when provided")

        _ensure_utc_datetime(name="created_at", value=self.created_at)
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("ExchangeKey.updated_at cannot be before created_at")

        if self.is_deleted:
            if self.deleted_at is None:
                raise ValueError("ExchangeKey.deleted_at must be set when is_deleted is true")
            _ensure_utc_datetime(name="deleted_at", value=self.deleted_at)
            if self.deleted_at < self.created_at:
                raise ValueError("ExchangeKey.deleted_at cannot be before created_at")
            if self.updated_at < self.deleted_at:
                raise ValueError("ExchangeKey.updated_at cannot be before deleted_at")
            return

        if self.deleted_at is not None:
            raise ValueError("ExchangeKey.deleted_at must be None when is_deleted is false")



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone awareness and UTC offset for datetime fields.

    Args:
        name: Field name for deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        UTC datetimes are represented with timezone info and zero offset.
    Raises:
        ValueError: If datetime is naive or not in UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
