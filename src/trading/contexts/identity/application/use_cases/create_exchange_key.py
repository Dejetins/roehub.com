from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from trading.contexts.identity.application.ports import ExchangeKeysRepository, IdentityClock
from trading.contexts.identity.application.ports.exchange_keys_secret_cipher import (
    ExchangeKeysSecretCipher,
)
from trading.contexts.identity.application.use_cases.exchange_keys_errors import (
    ExchangeKeyAlreadyExistsError,
    ExchangeKeyValidationError,
)
from trading.contexts.identity.application.use_cases.exchange_keys_models import (
    ExchangeKeyView,
    to_exchange_key_view,
)
from trading.shared_kernel.primitives import UserId

_ALLOWED_EXCHANGE_NAMES = {"binance", "bybit"}
_ALLOWED_MARKET_TYPES = {"spot", "futures"}
_ALLOWED_PERMISSIONS = {"read", "trade"}


class CreateExchangeKeyUseCase:
    """
    CreateExchangeKeyUseCase — create encrypted exchange key for authenticated user.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - src/trading/contexts/identity/application/ports/exchange_keys_secret_cipher.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    def __init__(
        self,
        *,
        repository: ExchangeKeysRepository,
        secret_cipher: ExchangeKeysSecretCipher,
        clock: IdentityClock,
    ) -> None:
        """
        Initialize use-case dependencies for storage, encryption, and deterministic time.

        Args:
            repository: Exchange keys storage port.
            secret_cipher: Envelope cipher port for secret fields.
            clock: UTC clock port for timestamps.
        Returns:
            None.
        Assumptions:
            Dependencies are initialized and non-null.
        Raises:
            ValueError: If any dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateExchangeKeyUseCase requires repository")
        if secret_cipher is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateExchangeKeyUseCase requires secret_cipher")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateExchangeKeyUseCase requires clock")

        self._repository = repository
        self._secret_cipher = secret_cipher
        self._clock = clock

    def create(
        self,
        *,
        user_id: UserId,
        exchange_name: str,
        market_type: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
    ) -> ExchangeKeyView:
        """
        Validate payload, encrypt secrets, and persist new active exchange key.

        Args:
            user_id: Owner identity user id.
            exchange_name: Exchange literal.
            market_type: Market type literal.
            label: Optional user label.
            permissions: Permission literal.
            api_key: Plain API key value.
            api_secret: Plain API secret value.
            passphrase: Optional plain passphrase value.
        Returns:
            ExchangeKeyView: API-safe projection of created key.
        Assumptions:
            Use-case never exposes plaintext secrets in return value.
        Raises:
            ExchangeKeyValidationError: If payload fields are invalid.
            ExchangeKeyAlreadyExistsError: If active duplicate key already exists.
        Side Effects:
            Encrypts secrets and writes one record in repository.
        """
        normalized_exchange_name = _normalize_exchange_name(exchange_name=exchange_name)
        normalized_market_type = _normalize_market_type(market_type=market_type)
        normalized_permissions = _normalize_permissions(permissions=permissions)
        normalized_label = _normalize_label(label=label)
        normalized_api_key = _normalize_required_secret(
            value=api_key,
            field_name="api_key",
            error_message="Exchange API key must be non-empty.",
        )
        normalized_api_secret = _normalize_required_secret(
            value=api_secret,
            field_name="api_secret",
            error_message="Exchange API secret must be non-empty.",
        )
        normalized_passphrase = _normalize_optional_secret(passphrase=passphrase)

        now = _ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        api_secret_enc = self._secret_cipher.encrypt_secret(secret=normalized_api_secret)
        passphrase_enc = (
            self._secret_cipher.encrypt_secret(secret=normalized_passphrase)
            if normalized_passphrase is not None
            else None
        )

        created = self._repository.create(
            key_id=uuid4(),
            user_id=user_id,
            exchange_name=normalized_exchange_name,
            market_type=normalized_market_type,
            label=normalized_label,
            permissions=normalized_permissions,
            api_key=normalized_api_key,
            api_secret_enc=api_secret_enc,
            passphrase_enc=passphrase_enc,
            created_at=now,
            updated_at=now,
        )
        if created is None:
            raise ExchangeKeyAlreadyExistsError()
        return to_exchange_key_view(entity=created)



def _normalize_exchange_name(*, exchange_name: str) -> str:
    """
    Normalize and validate exchange name enum value.

    Args:
        exchange_name: Raw exchange name input.
    Returns:
        str: Normalized exchange name literal.
    Assumptions:
        Only `binance` and `bybit` are supported in v1.
    Raises:
        ExchangeKeyValidationError: If value is unsupported.
    Side Effects:
        None.
    """
    normalized = exchange_name.strip().lower()
    if normalized not in _ALLOWED_EXCHANGE_NAMES:
        raise ExchangeKeyValidationError(
            message="exchange_name must be one of: binance, bybit.",
        )
    return normalized



def _normalize_market_type(*, market_type: str) -> str:
    """
    Normalize and validate market type enum value.

    Args:
        market_type: Raw market type input.
    Returns:
        str: Normalized market type literal.
    Assumptions:
        Variant A stores market_type without market_id.
    Raises:
        ExchangeKeyValidationError: If value is unsupported.
    Side Effects:
        None.
    """
    normalized = market_type.strip().lower()
    if normalized not in _ALLOWED_MARKET_TYPES:
        raise ExchangeKeyValidationError(
            message="market_type must be one of: spot, futures.",
        )
    return normalized



def _normalize_permissions(*, permissions: str) -> str:
    """
    Normalize and validate permissions enum value.

    Args:
        permissions: Raw permissions input.
    Returns:
        str: Normalized permissions literal.
    Assumptions:
        v1 supports only `read` and `trade` permissions.
    Raises:
        ExchangeKeyValidationError: If value is unsupported.
    Side Effects:
        None.
    """
    normalized = permissions.strip().lower()
    if normalized not in _ALLOWED_PERMISSIONS:
        raise ExchangeKeyValidationError(
            message="permissions must be one of: read, trade.",
        )
    return normalized



def _normalize_label(*, label: str | None) -> str | None:
    """
    Normalize optional label value for deterministic storage.

    Args:
        label: Optional raw label input.
    Returns:
        str | None: Stripped label or `None`.
    Assumptions:
        Empty labels are treated as absent.
    Raises:
        None.
    Side Effects:
        None.
    """
    if label is None:
        return None
    normalized = label.strip()
    if not normalized:
        return None
    return normalized



def _normalize_required_secret(*, value: str, field_name: str, error_message: str) -> str:
    """
    Normalize required secret-like value without leaking input in errors.

    Args:
        value: Raw string value.
        field_name: Field name used only for internal guard checks.
        error_message: Deterministic public-safe error message.
    Returns:
        str: Normalized non-empty value.
    Assumptions:
        Returned value should be persisted or encrypted immediately.
    Raises:
        ExchangeKeyValidationError: If value is empty after normalization.
    Side Effects:
        None.
    """
    normalized = value.strip()
    if not normalized:
        _ = field_name
        raise ExchangeKeyValidationError(message=error_message)
    return normalized



def _normalize_optional_secret(*, passphrase: str | None) -> str | None:
    """
    Normalize optional passphrase value.

    Args:
        passphrase: Optional passphrase input.
    Returns:
        str | None: Normalized passphrase or `None`.
    Assumptions:
        Empty passphrase should be treated as absent.
    Raises:
        None.
    Side Effects:
        None.
    """
    if passphrase is None:
        return None
    normalized = passphrase.strip()
    if not normalized:
        return None
    return normalized



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
