from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from uuid import UUID

import pytest

from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityTwoFactorRepository,
)
from trading.contexts.identity.adapters.outbound.security.two_factor import (
    AesGcmEnvelopeTwoFactorSecretCipher,
    PyOtpTwoFactorTotpProvider,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases import (
    SetupTwoFactorTotpUseCase,
    TwoFactorAlreadyEnabledError,
    TwoFactorInvalidCodeError,
    VerifyTwoFactorTotpUseCase,
)
from trading.shared_kernel.primitives import UserId


class _MutableClock(IdentityClock):
    """
    Mutable deterministic UTC clock for testing setup/verify flows.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize mutable clock with initial UTC value.

        Args:
            now_value: Initial timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Test controls all clock changes through `set_now`.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            None.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def set_now(self, *, now_value: datetime) -> None:
        """
        Replace current deterministic timestamp.

        Args:
            now_value: New timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Timestamp changes are deterministic and explicit in tests.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            Mutates internal clock state.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def now(self) -> datetime:
        """
        Return current deterministic UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current test-controlled timestamp.
        Assumptions:
            Time does not auto-progress between calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_setup_returns_otpauth_uri_and_stores_encrypted_secret() -> None:
    """
    Verify setup returns otpauth URI and persists only encrypted TOTP secret bytes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        In-memory repository reflects writes from setup use-case deterministically.
    Raises:
        AssertionError: If setup output or persisted encrypted state violates policy.
    Side Effects:
        None.
    """
    clock = _MutableClock(now_value=datetime(2026, 2, 14, 16, 0, 0, tzinfo=timezone.utc))
    repository = InMemoryIdentityTwoFactorRepository()
    secret_cipher = AesGcmEnvelopeTwoFactorSecretCipher(
        kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    )
    totp_provider = PyOtpTwoFactorTotpProvider()
    setup_use_case = SetupTwoFactorTotpUseCase(
        repository=repository,
        secret_cipher=secret_cipher,
        totp_provider=totp_provider,
        clock=clock,
        issuer="Roehub",
    )
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000101")

    setup_result = setup_use_case.setup(user_id=user_id)

    assert setup_result.otpauth_uri.startswith("otpauth://totp")
    secret = _extract_secret_from_uri(otpauth_uri=setup_result.otpauth_uri)
    stored = repository.find_by_user_id(user_id=user_id)
    assert stored is not None
    assert stored.enabled is False
    assert stored.enabled_at is None
    assert stored.updated_at == clock.now()
    assert stored.totp_secret_enc
    assert secret.encode("utf-8") not in stored.totp_secret_enc


def test_verify_enables_two_factor_on_correct_code_and_rejects_wrong_code() -> None:
    """
    Verify wrong code is rejected and correct code enables 2FA with timestamps.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Test-generated code follows same RFC 6238 defaults as provider.
    Raises:
        AssertionError: If verify does not enforce expected success/error behavior.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 14, 16, 30, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    repository = InMemoryIdentityTwoFactorRepository()
    secret_cipher = AesGcmEnvelopeTwoFactorSecretCipher(
        kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    )
    totp_provider = PyOtpTwoFactorTotpProvider()
    setup_use_case = SetupTwoFactorTotpUseCase(
        repository=repository,
        secret_cipher=secret_cipher,
        totp_provider=totp_provider,
        clock=clock,
        issuer="Roehub",
    )
    verify_use_case = VerifyTwoFactorTotpUseCase(
        repository=repository,
        secret_cipher=secret_cipher,
        totp_provider=totp_provider,
        clock=clock,
    )
    user_id = UserId(UUID("00000000-0000-0000-0000-000000000102"))

    setup_result = setup_use_case.setup(user_id=user_id)
    secret = _extract_secret_from_uri(otpauth_uri=setup_result.otpauth_uri)
    valid_code = _build_totp_code(
        secret=secret,
        timestamp_seconds=int(now.timestamp()),
        digits=6,
        period_seconds=30,
    )
    wrong_code = "000000" if valid_code != "000000" else "999999"

    with pytest.raises(TwoFactorInvalidCodeError):
        verify_use_case.verify(user_id=user_id, code=wrong_code)

    verify_result = verify_use_case.verify(user_id=user_id, code=valid_code)

    assert verify_result.enabled is True
    stored = repository.find_by_user_id(user_id=user_id)
    assert stored is not None
    assert stored.enabled is True
    assert stored.enabled_at == now
    assert stored.updated_at == now


def test_option_one_rejects_setup_and_verify_after_two_factor_is_enabled() -> None:
    """
    Verify Option 1 policy rejects re-setup and re-verify when 2FA is already enabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        First setup+verify transition succeeds and stores enabled state.
    Raises:
        AssertionError: If already-enabled policy is not enforced.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 14, 17, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    repository = InMemoryIdentityTwoFactorRepository()
    secret_cipher = AesGcmEnvelopeTwoFactorSecretCipher(
        kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    )
    totp_provider = PyOtpTwoFactorTotpProvider()
    setup_use_case = SetupTwoFactorTotpUseCase(
        repository=repository,
        secret_cipher=secret_cipher,
        totp_provider=totp_provider,
        clock=clock,
        issuer="Roehub",
    )
    verify_use_case = VerifyTwoFactorTotpUseCase(
        repository=repository,
        secret_cipher=secret_cipher,
        totp_provider=totp_provider,
        clock=clock,
    )
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000103")

    setup_result = setup_use_case.setup(user_id=user_id)
    secret = _extract_secret_from_uri(otpauth_uri=setup_result.otpauth_uri)
    valid_code = _build_totp_code(
        secret=secret,
        timestamp_seconds=int(now.timestamp()),
        digits=6,
        period_seconds=30,
    )
    verify_use_case.verify(user_id=user_id, code=valid_code)

    with pytest.raises(TwoFactorAlreadyEnabledError):
        setup_use_case.setup(user_id=user_id)
    with pytest.raises(TwoFactorAlreadyEnabledError):
        verify_use_case.verify(user_id=user_id, code=valid_code)


def _extract_secret_from_uri(*, otpauth_uri: str) -> str:
    """
    Extract base32 secret query parameter from otpauth URI.

    Args:
        otpauth_uri: URI returned by setup use-case.
    Returns:
        str: Base32 secret string from `secret` query parameter.
    Assumptions:
        URI follows standard otpauth query structure.
    Raises:
        AssertionError: If URI does not contain exactly one secret query value.
    Side Effects:
        None.
    """
    parsed = urlparse(otpauth_uri)
    query = parse_qs(parsed.query)
    secret_values = query.get("secret", [])
    assert len(secret_values) == 1
    return secret_values[0]


def _build_totp_code(
    *,
    secret: str,
    timestamp_seconds: int,
    digits: int,
    period_seconds: int,
) -> str:
    """
    Build deterministic RFC 6238 TOTP code for test verification.

    Args:
        secret: Base32 TOTP secret.
        timestamp_seconds: UNIX timestamp in seconds.
        digits: Number of code digits.
        period_seconds: TOTP period in seconds.
    Returns:
        str: Zero-padded numeric code string.
    Assumptions:
        HMAC-SHA1 profile matches provider defaults for identity 2FA v1.
    Raises:
        ValueError: If secret cannot be decoded as base32.
    Side Effects:
        None.
    """
    key = _decode_base32(secret=secret)
    counter = int(timestamp_seconds // period_seconds)
    digest = hmac.new(
        key,
        counter.to_bytes(8, byteorder="big", signed=False),
        hashlib.sha1,
    ).digest()
    offset = digest[-1] & 0x0F
    binary_code = (
        ((digest[offset] & 0x7F) << 24)
        | (digest[offset + 1] << 16)
        | (digest[offset + 2] << 8)
        | digest[offset + 3]
    )
    return f"{binary_code % (10**digits):0{digits}d}"


def _decode_base32(*, secret: str) -> bytes:
    """
    Decode base32 secret with optional missing padding.

    Args:
        secret: Base32 secret string.
    Returns:
        bytes: Decoded secret bytes.
    Assumptions:
        Secret may omit trailing `=` padding.
    Raises:
        ValueError: If string is not valid base32.
    Side Effects:
        None.
    """
    normalized = secret.strip().upper()
    padding = "=" * ((8 - (len(normalized) % 8)) % 8)
    return base64.b32decode(f"{normalized}{padding}", casefold=True)


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate datetime is timezone-aware UTC and return same value.

    Args:
        value: Datetime value to validate.
        field_name: Label for deterministic error messages.
    Returns:
        datetime: Same validated datetime.
    Assumptions:
        UTC datetimes have zero UTC offset.
    Raises:
        ValueError: If datetime is naive or not UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC datetime")
    return value
