from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from trading.shared_kernel.primitives import PaidLevel, UserId


@dataclass(frozen=True, slots=True)
class IdentityJwtClaims:
    """
    IdentityJwtClaims — типизированные JWT claims для identity cookie.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    user_id: UserId
    paid_level: PaidLevel
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        """
        Validate claims datetime invariants for JWT serialization.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `issued_at` and `expires_at` are timezone-aware UTC datetimes.
        Raises:
            ValueError: If datetimes are naive, non-UTC, or expiration is not after issue time.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="issued_at", value=self.issued_at)
        _ensure_utc_datetime(name="expires_at", value=self.expires_at)
        if self.expires_at <= self.issued_at:
            raise ValueError("IdentityJwtClaims.expires_at must be after issued_at")


class JwtDecodeError(ValueError):
    """
    JwtDecodeError — детерминированная ошибка проверки JWT.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/jwt_codec.py
      - src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize decode error with stable error code.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable error description.
        Returns:
            None.
        Assumptions:
            API dependency maps this error into structured 401 payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message


class JwtCodec(Protocol):
    """
    JwtCodec — порт подписи и проверки identity JWT токенов.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    def encode(self, *, claims: IdentityJwtClaims) -> str:
        """
        Sign claims into deterministic compact JWT string.

        Args:
            claims: Typed identity claims.
        Returns:
            str: Signed compact JWT.
        Assumptions:
            HS256 secret key is configured and non-empty.
        Raises:
            ValueError: If claims cannot be serialized.
        Side Effects:
            None.
        """
        ...

    def decode(self, *, token: str) -> IdentityJwtClaims:
        """
        Verify token signature and claims, then return typed identity claims.

        Args:
            token: Compact JWT token string.
        Returns:
            IdentityJwtClaims: Verified and normalized claims.
        Assumptions:
            Token was issued by same signer and not expired.
        Raises:
            JwtDecodeError: If signature or claims are invalid.
        Side Effects:
            None.
        """
        ...


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Ensure datetime field is timezone-aware UTC.

    Args:
        name: Field name for deterministic error message.
        value: Datetime to validate.
    Returns:
        None.
    Assumptions:
        UTC datetimes have zero offset.
    Raises:
        ValueError: If datetime is naive or not UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
