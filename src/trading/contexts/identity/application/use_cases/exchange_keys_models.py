from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from trading.contexts.identity.domain.entities import ExchangeKey


@dataclass(frozen=True, slots=True)
class ExchangeKeyView:
    """
    ExchangeKeyView — non-secret API-safe exchange key projection.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/application/use_cases/list_exchange_keys.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    key_id: UUID
    exchange_name: str
    market_type: str
    label: str | None
    permissions: str
    api_key: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """
        Validate non-secret projection invariants and UTC timestamps.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `api_key` field stores masked representation for API responses.
        Raises:
            ValueError: If API key is empty or timestamps are invalid.
        Side Effects:
            None.
        """
        if not self.api_key:
            raise ValueError("ExchangeKeyView.api_key must be non-empty")
        _ensure_utc_datetime(name="created_at", value=self.created_at)
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)



def to_exchange_key_view(*, entity: ExchangeKey) -> ExchangeKeyView:
    """
    Convert full storage entity into API-safe non-secret projection.

    Args:
        entity: Full exchange key entity from repository.
    Returns:
        ExchangeKeyView: API-safe projection with masked API key.
    Assumptions:
        `entity.api_key` contains plaintext value persisted in storage.
    Raises:
        ValueError: If entity fields violate projection invariants.
    Side Effects:
        None.
    """
    return ExchangeKeyView(
        key_id=entity.key_id,
        exchange_name=entity.exchange_name,
        market_type=entity.market_type,
        label=entity.label,
        permissions=entity.permissions,
        api_key=_mask_api_key(api_key=entity.api_key),
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )



def _mask_api_key(*, api_key: str) -> str:
    """
    Mask API key value for safe response representation.

    Args:
        api_key: Plaintext API key from storage.
    Returns:
        str: Masked API key showing only last four characters.
    Assumptions:
        Masking strategy is deterministic and stable for UI rendering.
    Raises:
        ValueError: If API key is empty.
    Side Effects:
        None.
    """
    normalized = api_key.strip()
    if not normalized:
        raise ValueError("_mask_api_key requires non-empty api_key")
    visible_suffix = normalized[-4:] if len(normalized) >= 4 else normalized
    return f"****{visible_suffix}"



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
