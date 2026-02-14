from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.jwt_codec import IdentityJwtClaims, JwtCodec
from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthPayloadValidator,
)
from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.shared_kernel.primitives import PaidLevel, UserId


@dataclass(frozen=True, slots=True)
class TelegramLoginResult:
    """
    TelegramLoginResult — результат use-case Telegram Login Widget (Variant A).

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
      - src/trading/contexts/identity/application/ports/jwt_codec.py
    """

    user_id: UserId
    paid_level: PaidLevel
    jwt_token: str
    jwt_expires_at: datetime


class TelegramLoginUseCase:
    """
    TelegramLoginUseCase — логин через Telegram Widget с upsert пользователя и JWT cookie.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/user_repository.py
      - src/trading/contexts/identity/application/ports/jwt_codec.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
    """

    def __init__(
        self,
        *,
        validator: TelegramAuthPayloadValidator,
        user_repository: UserRepository,
        jwt_codec: JwtCodec,
        clock: IdentityClock,
        jwt_ttl_days: int,
    ) -> None:
        """
        Initialize use-case dependencies and immutable runtime policy.

        Args:
            validator: Telegram payload validator (signature + freshness).
            user_repository: Identity storage port.
            jwt_codec: JWT signer/verifier port.
            clock: Current time source.
            jwt_ttl_days: JWT TTL in days; v1 expects 7.
        Returns:
            None.
        Assumptions:
            All dependencies are initialized and non-null.
        Raises:
            ValueError: If dependencies are missing or TTL is non-positive.
        Side Effects:
            None.
        """
        if validator is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramLoginUseCase requires validator")
        if user_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramLoginUseCase requires user_repository")
        if jwt_codec is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramLoginUseCase requires jwt_codec")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramLoginUseCase requires clock")
        if jwt_ttl_days <= 0:
            raise ValueError("TelegramLoginUseCase requires jwt_ttl_days > 0")

        self._validator = validator
        self._user_repository = user_repository
        self._jwt_codec = jwt_codec
        self._clock = clock
        self._jwt_ttl_days = jwt_ttl_days

    def login(self, *, payload: Mapping[str, str]) -> TelegramLoginResult:
        """
        Validate Telegram payload, upsert user, and issue signed JWT token.

        Args:
            payload: Raw Telegram widget payload fields.
        Returns:
            TelegramLoginResult: User identity and issued JWT metadata.
        Assumptions:
            Payload is deterministic mapping with string keys/values.
        Raises:
            TelegramAuthValidationError: If payload signature/freshness check fails.
            ValueError: If repository or JWT conversion fails.
        Side Effects:
            Persists user login timestamp and may create a new user record.
        """
        now = _ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        normalized_payload = _normalize_payload(payload=payload)

        telegram_user_id = self._validator.validate(payload=normalized_payload, now=now)
        user = self._user_repository.upsert_telegram_login(
            telegram_user_id=telegram_user_id,
            login_at=now,
        )

        expires_at = now + timedelta(days=self._jwt_ttl_days)
        claims = IdentityJwtClaims(
            user_id=user.user_id,
            paid_level=user.paid_level,
            issued_at=now,
            expires_at=expires_at,
        )
        jwt_token = self._jwt_codec.encode(claims=claims)

        return TelegramLoginResult(
            user_id=user.user_id,
            paid_level=user.paid_level,
            jwt_token=jwt_token,
            jwt_expires_at=expires_at,
        )


def _normalize_payload(*, payload: Mapping[str, str]) -> dict[str, str]:
    """
    Build deterministic normalized payload mapping for Telegram verification.

    Args:
        payload: Raw mapping from HTTP request body.
    Returns:
        dict[str, str]: Deterministically ordered payload with stripped string values.
    Assumptions:
        Keys and values are representable as strings.
    Raises:
        ValueError: If normalized payload is empty.
    Side Effects:
        None.
    """
    normalized_items = [
        (str(key).strip(), str(value).strip())
        for key, value in payload.items()
        if str(key).strip() and value is not None
    ]
    normalized_items.sort(key=lambda item: item[0])
    normalized = dict(normalized_items)
    if not normalized:
        raise ValueError("TelegramLoginUseCase payload cannot be empty")
    return normalized


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate that datetime is timezone-aware UTC.

    Args:
        value: Datetime value to validate.
        field_name: Field label used in error messages.
    Returns:
        datetime: Same validated datetime value.
    Assumptions:
        UTC datetimes use zero offset.
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
