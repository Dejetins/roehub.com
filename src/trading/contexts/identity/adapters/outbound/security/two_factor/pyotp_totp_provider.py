from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import importlib
import os
from datetime import datetime
from types import ModuleType
from typing import Any
from urllib.parse import quote, urlencode

from trading.contexts.identity.application.ports.two_factor_totp_provider import (
    TwoFactorTotpProvider,
)
from trading.shared_kernel.primitives import UserId

try:
    _PYOTP_MODULE: ModuleType | None = importlib.import_module("pyotp")
except ModuleNotFoundError:  # pragma: no cover - covered by fallback-path tests
    _PYOTP_MODULE = None

_DEFAULT_TOTP_DIGITS = 6
_DEFAULT_TOTP_PERIOD_SECONDS = 30
_DEFAULT_VALID_WINDOW = 1


class PyOtpTwoFactorTotpProvider(TwoFactorTotpProvider):
    """
    PyOtpTwoFactorTotpProvider — TOTP provider for setup/verify with pyotp + deterministic fallback.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_totp_provider.py
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
    """

    def __init__(
        self,
        *,
        digits: int = _DEFAULT_TOTP_DIGITS,
        period_seconds: int = _DEFAULT_TOTP_PERIOD_SECONDS,
        valid_window: int = _DEFAULT_VALID_WINDOW,
    ) -> None:
        """
        Initialize TOTP provider parameters for setup URI and verification.

        Args:
            digits: Number of code digits (identity v1 expects 6).
            period_seconds: TOTP period in seconds.
            valid_window: Number of time-steps accepted before/after current step.
        Returns:
            None.
        Assumptions:
            Defaults align with common authenticator apps and identity policy expectations.
        Raises:
            ValueError: If arguments are outside supported ranges.
        Side Effects:
            None.
        """
        if digits <= 0:
            raise ValueError("PyOtpTwoFactorTotpProvider digits must be > 0")
        if period_seconds <= 0:
            raise ValueError("PyOtpTwoFactorTotpProvider period_seconds must be > 0")
        if valid_window < 0:
            raise ValueError("PyOtpTwoFactorTotpProvider valid_window must be >= 0")

        self._digits = digits
        self._period_seconds = period_seconds
        self._valid_window = valid_window

    def create_secret(self) -> str:
        """
        Generate new base32 TOTP secret string.

        Args:
            None.
        Returns:
            str: Base32 secret string suitable for TOTP providers.
        Assumptions:
            Secret generation must use cryptographically secure randomness.
        Raises:
            ValueError: If generated secret is empty.
        Side Effects:
            Uses OS random source.
        """
        if _PYOTP_MODULE is not None:
            random_base32: Any = getattr(_PYOTP_MODULE, "random_base32")
            secret = str(random_base32())
        else:
            secret = base64.b32encode(os.urandom(20)).decode("ascii").rstrip("=")
        normalized = secret.strip().upper()
        if not normalized:
            raise ValueError("PyOtpTwoFactorTotpProvider generated empty secret")
        return normalized

    def build_otpauth_uri(self, *, secret: str, user_id: UserId, issuer: str) -> str:
        """
        Build standard otpauth URI for UI QR rendering from setup result.

        Args:
            secret: Base32 TOTP secret.
            user_id: Stable user identifier used as account label.
            issuer: Issuer label shown in authenticator apps.
        Returns:
            str: URI string starting with `otpauth://totp`.
        Assumptions:
            Account label in v1 is based on `user_id` and contains no PII.
        Raises:
            ValueError: If secret or issuer is empty.
        Side Effects:
            None.
        """
        normalized_secret = secret.strip().upper()
        normalized_issuer = issuer.strip()
        account_label = str(user_id).strip()
        if not normalized_secret:
            raise ValueError("PyOtpTwoFactorTotpProvider requires non-empty secret")
        if not normalized_issuer:
            raise ValueError("PyOtpTwoFactorTotpProvider requires non-empty issuer")
        if not account_label:
            raise ValueError("PyOtpTwoFactorTotpProvider requires non-empty user_id label")

        if _PYOTP_MODULE is not None:
            totp_factory: Any = getattr(_PYOTP_MODULE, "TOTP")
            totp = totp_factory(
                normalized_secret,
                digits=self._digits,
                interval=self._period_seconds,
            )
            uri = totp.provisioning_uri(
                name=account_label,
                issuer_name=normalized_issuer,
            )
        else:
            label = quote(f"{normalized_issuer}:{account_label}", safe="")
            query = urlencode(
                {
                    "digits": str(self._digits),
                    "issuer": normalized_issuer,
                    "period": str(self._period_seconds),
                    "secret": normalized_secret,
                }
            )
            uri = f"otpauth://totp/{label}?{query}"
        if not uri.startswith("otpauth://totp"):
            raise ValueError("PyOtpTwoFactorTotpProvider produced invalid otpauth URI")
        return uri

    def verify_code(self, *, secret: str, code: str, at_time: datetime) -> bool:
        """
        Verify TOTP code for provided UTC timestamp.

        Args:
            secret: Base32 TOTP secret.
            code: User submitted code string.
            at_time: Current timezone-aware UTC datetime.
        Returns:
            bool: `True` when verification succeeds.
        Assumptions:
            Caller already normalized code format for deterministic behavior.
        Raises:
            ValueError: If timestamp is not UTC or secret/code are empty.
        Side Effects:
            None.
        """
        normalized_secret = secret.strip().upper()
        normalized_code = code.strip()
        if not normalized_secret:
            raise ValueError("PyOtpTwoFactorTotpProvider verify requires non-empty secret")
        if not normalized_code:
            raise ValueError("PyOtpTwoFactorTotpProvider verify requires non-empty code")
        now = _ensure_utc_datetime(value=at_time, field_name="at_time")
        now_seconds = int(now.timestamp())
        if _PYOTP_MODULE is not None:
            totp_factory: Any = getattr(_PYOTP_MODULE, "TOTP")
            totp = totp_factory(
                normalized_secret,
                digits=self._digits,
                interval=self._period_seconds,
            )
            return bool(
                totp.verify(
                    normalized_code,
                    for_time=now_seconds,
                    valid_window=self._valid_window,
                )
            )
        return _verify_with_fallback(
            secret=normalized_secret,
            code=normalized_code,
            now_seconds=now_seconds,
            digits=self._digits,
            period_seconds=self._period_seconds,
            valid_window=self._valid_window,
        )


def _verify_with_fallback(
    *,
    secret: str,
    code: str,
    now_seconds: int,
    digits: int,
    period_seconds: int,
    valid_window: int,
) -> bool:
    """
    Verify TOTP code using RFC 6238 fallback implementation when pyotp is unavailable.

    Args:
        secret: Base32 TOTP secret.
        code: User submitted code.
        now_seconds: Current UNIX timestamp in seconds.
        digits: Number of TOTP digits.
        period_seconds: TOTP period in seconds.
        valid_window: Accepted offset window in periods.
    Returns:
        bool: `True` when any accepted time-slot matches submitted code.
    Assumptions:
        `code` is normalized to digits-only by caller.
    Raises:
        ValueError: If secret cannot be decoded as base32.
    Side Effects:
        None.
    """
    key = _decode_base32_secret(secret=secret)
    for offset in range(-valid_window, valid_window + 1):
        candidate_seconds = now_seconds + (offset * period_seconds)
        generated_code = _build_totp_code(
            key=key,
            timestamp_seconds=candidate_seconds,
            digits=digits,
            period_seconds=period_seconds,
        )
        if hmac.compare_digest(generated_code, code):
            return True
    return False


def _decode_base32_secret(*, secret: str) -> bytes:
    """
    Decode base32 secret into raw bytes with deterministic error handling.

    Args:
        secret: Base32 TOTP secret.
    Returns:
        bytes: Decoded secret bytes.
    Assumptions:
        Padding can be omitted by providers and is restored locally.
    Raises:
        ValueError: If secret is not valid base32.
    Side Effects:
        None.
    """
    normalized = secret.strip().upper()
    padding = "=" * ((8 - (len(normalized) % 8)) % 8)
    candidate = f"{normalized}{padding}"
    try:
        return base64.b32decode(candidate, casefold=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("PyOtpTwoFactorTotpProvider secret is not valid base32") from error


def _build_totp_code(
    *,
    key: bytes,
    timestamp_seconds: int,
    digits: int,
    period_seconds: int,
) -> str:
    """
    Build TOTP code for given key and timestamp using HMAC-SHA1 dynamic truncation.

    Args:
        key: Raw decoded secret key bytes.
        timestamp_seconds: UNIX timestamp in seconds.
        digits: Number of output code digits.
        period_seconds: TOTP period in seconds.
    Returns:
        str: Zero-padded code string with requested digits.
    Assumptions:
        RFC 6238 default HMAC-SHA1 is used for compatibility.
    Raises:
        ValueError: If timestamp is negative.
    Side Effects:
        None.
    """
    if timestamp_seconds < 0:
        raise ValueError("PyOtpTwoFactorTotpProvider timestamp must be non-negative")
    counter = int(timestamp_seconds // period_seconds)
    counter_bytes = counter.to_bytes(8, byteorder="big", signed=False)
    digest = hmac.new(key, counter_bytes, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary_code = (
        ((digest[offset] & 0x7F) << 24)
        | (digest[offset + 1] << 16)
        | (digest[offset + 2] << 8)
        | digest[offset + 3]
    )
    numeric_code = binary_code % (10**digits)
    return f"{numeric_code:0{digits}d}"


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate datetime is timezone-aware UTC and return same value.

    Args:
        value: Datetime value to validate.
        field_name: Field label for deterministic error message.
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
