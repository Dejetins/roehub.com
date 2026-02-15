from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityTwoFactorRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    JwtCookieCurrentUser,
)
from trading.contexts.identity.adapters.outbound.security.jwt import Hs256JwtCodec
from trading.contexts.identity.adapters.outbound.security.telegram import (
    TelegramLoginWidgetPayloadValidator,
)
from trading.contexts.identity.adapters.outbound.security.two_factor import (
    AesGcmEnvelopeTwoFactorSecretCipher,
    PyOtpTwoFactorTotpProvider,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases import (
    SetupTwoFactorTotpUseCase,
    TelegramLoginUseCase,
    VerifyTwoFactorTotpUseCase,
)
from trading.shared_kernel.primitives import UserId


class _FixedClock(IdentityClock):
    """
    Deterministic UTC clock for identity 2FA route tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize fixed clock value.

        Args:
            now_value: Fixed timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Datetime remains constant across whole request flow in each test.
        Raises:
            ValueError: If datetime is naive/non-UTC.
        Side Effects:
            None.
        """
        offset = now_value.utcoffset()
        if now_value.tzinfo is None or offset is None:
            raise ValueError("_FixedClock requires timezone-aware UTC datetime")
        if offset.total_seconds() != 0:
            raise ValueError("_FixedClock requires timezone-aware UTC datetime")
        self._now_value = now_value

    def now(self) -> datetime:
        """
        Return fixed UTC datetime.

        Args:
            None.
        Returns:
            datetime: Fixed timestamp.
        Assumptions:
            Deterministic value is sufficient for route tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_two_factor_setup_route_returns_otpauth_uri_and_persists_encrypted_secret() -> None:
    """
    Verify `/2fa/setup` returns only `otpauth_uri` and stores encrypted secret in repository.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authenticated user is established through Telegram login route.
    Raises:
        AssertionError: If setup route payload or repository state is incorrect.
    Side Effects:
        None.
    """
    client, two_factor_repository, now, bot_token = _build_identity_test_client()

    login_response = client.post(
        "/auth/telegram/login",
        json=_build_signed_telegram_payload(
            bot_token=bot_token,
            user_id=1002001,
            auth_date=int(now.timestamp()),
        ),
    )
    assert login_response.status_code == 200

    setup_response = client.post("/2fa/setup")

    assert setup_response.status_code == 200
    payload = setup_response.json()
    assert list(payload.keys()) == ["otpauth_uri"]
    assert payload["otpauth_uri"].startswith("otpauth://totp")

    user_id = login_response.json()["user_id"]
    state = two_factor_repository.find_by_user_id(user_id=_user_id_from_string(value=user_id))
    assert state is not None
    assert state.enabled is False
    assert state.enabled_at is None
    assert state.totp_secret_enc


def test_two_factor_verify_route_enables_state_and_rejects_wrong_code() -> None:
    """
    Verify `/2fa/verify` rejects wrong code and enables 2FA on correct code.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Setup route has created pending encrypted secret for authenticated user.
    Raises:
        AssertionError: If verify route does not enforce expected behavior.
    Side Effects:
        None.
    """
    client, two_factor_repository, now, bot_token = _build_identity_test_client()
    login_response = client.post(
        "/auth/telegram/login",
        json=_build_signed_telegram_payload(
            bot_token=bot_token,
            user_id=1002002,
            auth_date=int(now.timestamp()),
        ),
    )
    assert login_response.status_code == 200

    setup_response = client.post("/2fa/setup")
    assert setup_response.status_code == 200
    otpauth_uri = setup_response.json()["otpauth_uri"]
    secret = _extract_secret_from_uri(otpauth_uri=otpauth_uri)
    valid_code = _build_totp_code(secret=secret, timestamp_seconds=int(now.timestamp()))
    wrong_code = "000000" if valid_code != "000000" else "999999"

    wrong_verify_response = client.post("/2fa/verify", json={"code": wrong_code})
    assert wrong_verify_response.status_code == 422
    assert wrong_verify_response.json() == {
        "detail": {
            "error": "invalid_two_factor_code",
            "message": "Invalid two-factor authentication code.",
        }
    }

    verify_response = client.post("/2fa/verify", json={"code": valid_code})
    assert verify_response.status_code == 200
    assert verify_response.json() == {"enabled": True}

    user_id = login_response.json()["user_id"]
    state = two_factor_repository.find_by_user_id(user_id=_user_id_from_string(value=user_id))
    assert state is not None
    assert state.enabled is True
    assert state.enabled_at == now


def test_two_factor_option_one_rejects_repeated_setup_and_verify_when_enabled() -> None:
    """
    Verify Option 1 route behavior rejects setup/verify once 2FA is already enabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        2FA is enabled after one successful setup+verify sequence.
    Raises:
        AssertionError: If endpoints allow re-setup/re-verify while enabled.
    Side Effects:
        None.
    """
    client, _, now, bot_token = _build_identity_test_client()
    login_response = client.post(
        "/auth/telegram/login",
        json=_build_signed_telegram_payload(
            bot_token=bot_token,
            user_id=1002003,
            auth_date=int(now.timestamp()),
        ),
    )
    assert login_response.status_code == 200

    setup_response = client.post("/2fa/setup")
    secret = _extract_secret_from_uri(otpauth_uri=setup_response.json()["otpauth_uri"])
    valid_code = _build_totp_code(secret=secret, timestamp_seconds=int(now.timestamp()))
    verify_response = client.post("/2fa/verify", json={"code": valid_code})
    assert verify_response.status_code == 200

    repeated_setup = client.post("/2fa/setup")
    repeated_verify = client.post("/2fa/verify", json={"code": valid_code})

    expected_payload = {
        "detail": {
            "error": "two_factor_already_enabled",
            "message": "Two-factor authentication is already enabled.",
        }
    }
    assert repeated_setup.status_code == 409
    assert repeated_setup.json() == expected_payload
    assert repeated_verify.status_code == 409
    assert repeated_verify.json() == expected_payload


def _build_identity_test_client() -> tuple[
    TestClient,
    InMemoryIdentityTwoFactorRepository,
    datetime,
    str,
]:
    """
    Build FastAPI TestClient with wired identity router and in-memory 2FA repository.

    Args:
        None.
    Returns:
        tuple[TestClient, InMemoryIdentityTwoFactorRepository, datetime, str]:
            `(client, two_factor_repository, now, bot_token)`.
    Assumptions:
        Clock and signing secrets are deterministic for reproducible route tests.
    Raises:
        ValueError: If dependency wiring is invalid.
    Side Effects:
        Creates in-memory FastAPI application.
    """
    now = datetime(2026, 2, 14, 18, 0, 0, tzinfo=timezone.utc)
    bot_token = "two-factor-route-bot-token"
    clock = _FixedClock(now_value=now)
    user_repository = InMemoryIdentityUserRepository()
    two_factor_repository = InMemoryIdentityTwoFactorRepository()
    validator = TelegramLoginWidgetPayloadValidator(bot_token=bot_token)
    jwt_codec = Hs256JwtCodec(secret_key="two-factor-route-secret", clock=clock)
    current_user_port = JwtCookieCurrentUser(
        jwt_codec=jwt_codec,
        user_repository=user_repository,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name="roehub_identity_jwt",
    )
    telegram_login = TelegramLoginUseCase(
        validator=validator,
        user_repository=user_repository,
        jwt_codec=jwt_codec,
        clock=clock,
        jwt_ttl_days=7,
    )
    two_factor_setup = SetupTwoFactorTotpUseCase(
        repository=two_factor_repository,
        secret_cipher=AesGcmEnvelopeTwoFactorSecretCipher(
            kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
        ),
        totp_provider=PyOtpTwoFactorTotpProvider(),
        clock=clock,
        issuer="Roehub",
    )
    two_factor_verify = VerifyTwoFactorTotpUseCase(
        repository=two_factor_repository,
        secret_cipher=AesGcmEnvelopeTwoFactorSecretCipher(
            kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
        ),
        totp_provider=PyOtpTwoFactorTotpProvider(),
        clock=clock,
    )

    app = FastAPI()
    app.include_router(
        build_identity_router(
            telegram_login=telegram_login,
            two_factor_setup=two_factor_setup,
            two_factor_verify=two_factor_verify,
            current_user_dependency=current_user_dependency,
            cookie_name="roehub_identity_jwt",
            cookie_secure=False,
            cookie_samesite="lax",
            cookie_path="/",
        )
    )
    return TestClient(app), two_factor_repository, now, bot_token


def _build_signed_telegram_payload(
    *,
    bot_token: str,
    user_id: int,
    auth_date: int,
) -> dict[str, object]:
    """
    Build signed Telegram login payload for route authentication setup.

    Args:
        bot_token: Telegram bot token used for payload hash signing.
        user_id: Telegram user identifier.
        auth_date: UNIX auth timestamp.
    Returns:
        dict[str, object]: JSON payload accepted by `/auth/telegram/login`.
    Assumptions:
        Auth date and clock value are aligned for freshness check.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload = {
        "auth_date": str(auth_date),
        "first_name": "Roe",
        "id": str(user_id),
        "username": "identity_2fa_user",
    }
    data_check_string = "\n".join(
        f"{key}={value}" for key, value in sorted(payload.items(), key=lambda item: item[0])
    )
    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    payload_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "id": user_id,
        "auth_date": auth_date,
        "hash": payload_hash,
        "first_name": "Roe",
        "username": "identity_2fa_user",
    }


def _extract_secret_from_uri(*, otpauth_uri: str) -> str:
    """
    Extract base32 secret from `otpauth_uri` query string.

    Args:
        otpauth_uri: URI returned by `/2fa/setup`.
    Returns:
        str: Base32 secret value.
    Assumptions:
        URI contains exactly one `secret` query parameter.
    Raises:
        AssertionError: If secret query parameter is missing or duplicated.
    Side Effects:
        None.
    """
    parsed = urlparse(otpauth_uri)
    query = parse_qs(parsed.query)
    secret_values = query.get("secret", [])
    assert len(secret_values) == 1
    return secret_values[0]


def _build_totp_code(*, secret: str, timestamp_seconds: int) -> str:
    """
    Build deterministic six-digit RFC 6238 code for provided secret and timestamp.

    Args:
        secret: Base32 TOTP secret.
        timestamp_seconds: UNIX timestamp in seconds.
    Returns:
        str: Six-digit zero-padded TOTP code.
    Assumptions:
        Test code generation matches provider defaults: SHA1, period 30s, digits 6.
    Raises:
        ValueError: If secret is not valid base32.
    Side Effects:
        None.
    """
    normalized_secret = secret.strip().upper()
    padding = "=" * ((8 - (len(normalized_secret) % 8)) % 8)
    key = base64.b32decode(f"{normalized_secret}{padding}", casefold=True)
    counter = int(timestamp_seconds // 30)
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
    return f"{binary_code % 1_000_000:06d}"


def _user_id_from_string(*, value: str) -> UserId:
    """
    Convert user id string to shared-kernel `UserId` for repository assertions.

    Args:
        value: Canonical UUID string representation of user id.
    Returns:
        UserId: Parsed user identifier object.
    Assumptions:
        Login endpoint returns canonical UUID string.
    Raises:
        ValueError: If value is not valid UUID.
    Side Effects:
        None.
    """
    return UserId.from_string(value)
