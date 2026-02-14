from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Mapping

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    JwtCookieCurrentUser,
)
from trading.contexts.identity.adapters.outbound.security.jwt import Hs256JwtCodec
from trading.contexts.identity.adapters.outbound.security.telegram import (
    TelegramLoginWidgetPayloadValidator,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.jwt_codec import IdentityJwtClaims
from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthPayloadValidator,
)
from trading.contexts.identity.application.use_cases import TelegramLoginUseCase
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel


class _FixedClock(IdentityClock):
    """
    Deterministic UTC clock used by identity JWT and login tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize fixed clock value.

        Args:
            now_value: Fixed timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Datetime remains immutable through test lifecycle.
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
            No monotonic progression is required for these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


class _TelegramValidatorStub(TelegramAuthPayloadValidator):
    """
    Validator stub returning fixed Telegram user id for API route tests.
    """

    def __init__(self, *, telegram_user_id: TelegramUserId) -> None:
        """
        Store fixed Telegram user id.

        Args:
            telegram_user_id: Deterministic Telegram user id.
        Returns:
            None.
        Assumptions:
            Payload validation is out of scope for these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._telegram_user_id = telegram_user_id

    def validate(self, *, payload: Mapping[str, str], now: datetime) -> TelegramUserId:
        """
        Return preconfigured Telegram id and ignore payload details.

        Args:
            payload: Raw request payload.
            now: Current UTC datetime.
        Returns:
            TelegramUserId: Preconfigured id.
        Assumptions:
            Use-case normalization already ensures non-empty payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = payload
        _ = now
        return self._telegram_user_id



def _build_cookie_test_client() -> tuple[TestClient, str]:
    """
    Build test client with identity router and return one valid JWT cookie token.

    Args:
        None.
    Returns:
        tuple[TestClient, str]: `(client, valid_token)` pair.
    Assumptions:
        In-memory repository stores one active user for token subject.
    Raises:
        ValueError: If test dependency construction is invalid.
    Side Effects:
        Creates in-memory FastAPI application.
    """
    now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
    clock = _FixedClock(now_value=now)
    repository = InMemoryIdentityUserRepository()
    created_user = repository.upsert_telegram_login(
        telegram_user_id=TelegramUserId(411001),
        login_at=now,
    )

    jwt_codec = Hs256JwtCodec(secret_key="identity-routes-secret", clock=clock)
    claims = IdentityJwtClaims(
        user_id=created_user.user_id,
        paid_level=PaidLevel.free(),
        issued_at=now,
        expires_at=now + timedelta(days=7),
    )
    valid_token = jwt_codec.encode(claims=claims)

    current_user_port = JwtCookieCurrentUser(jwt_codec=jwt_codec, user_repository=repository)
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name="roehub_identity_jwt",
    )

    login_use_case = TelegramLoginUseCase(
        validator=_TelegramValidatorStub(telegram_user_id=TelegramUserId(411001)),
        user_repository=repository,
        jwt_codec=jwt_codec,
        clock=clock,
        jwt_ttl_days=7,
    )

    app = FastAPI()
    app.include_router(
        build_identity_router(
            telegram_login=login_use_case,
            current_user_dependency=current_user_dependency,
            cookie_name="roehub_identity_jwt",
            cookie_secure=False,
            cookie_samesite="lax",
            cookie_path="/",
        )
    )
    return TestClient(app), valid_token



def _build_signed_telegram_payload(
    *,
    bot_token: str,
    user_id: int,
    auth_date: int,
) -> dict[str, object]:
    """
    Build signed Telegram login payload for integration-style login route test.

    Args:
        bot_token: Telegram bot token used for hash signing.
        user_id: Telegram user identifier.
    Returns:
        dict[str, object]: JSON payload accepted by `/auth/telegram/login`.
    Assumptions:
        `auth_date` and use-case clock are aligned for freshness check.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload = {
        "auth_date": str(auth_date),
        "first_name": "Roe",
        "id": str(user_id),
        "username": "identity_user",
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
        "username": "identity_user",
    }



def test_current_user_dependency_rejects_missing_cookie() -> None:
    """
    Verify protected endpoint returns 401 when JWT cookie is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        CurrentUser dependency maps missing cookie to deterministic error code.
    Raises:
        AssertionError: If endpoint does not return expected 401 payload.
    Side Effects:
        None.
    """
    client, _ = _build_cookie_test_client()

    response = client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "missing_jwt_cookie",
            "message": "JWT cookie is required",
        }
    }



def test_current_user_dependency_rejects_invalid_cookie() -> None:
    """
    Verify protected endpoint returns 401 for malformed JWT cookie value.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        JWT codec reports invalid token format for malformed string.
    Raises:
        AssertionError: If endpoint accepts malformed token.
    Side Effects:
        None.
    """
    client, _ = _build_cookie_test_client()
    client.cookies.set("roehub_identity_jwt", "invalid-token")

    response = client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "invalid_token_format"



def test_current_user_dependency_accepts_valid_cookie() -> None:
    """
    Verify protected endpoint returns stable `user_id` for valid JWT cookie.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        In-memory user exists and token is signed with same secret.
    Raises:
        AssertionError: If valid cookie does not authorize request.
    Side Effects:
        None.
    """
    client, valid_token = _build_cookie_test_client()
    client.cookies.set("roehub_identity_jwt", valid_token)

    response = client.get("/auth/current-user")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"]
    assert payload["paid_level"] == "free"



def test_post_auth_telegram_login_sets_http_only_cookie() -> None:
    """
    Verify wired login route sets HttpOnly JWT cookie and authorizes current-user endpoint.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Router wiring uses in-memory repository when PG DSN is not configured.
    Raises:
        AssertionError: If login response misses auth cookie or protected endpoint fails.
    Side Effects:
        None.
    """
    bot_token = "integration-bot-token"
    now = datetime(2026, 2, 14, 14, 0, 0, tzinfo=timezone.utc)
    clock = _FixedClock(now_value=now)
    repository = InMemoryIdentityUserRepository()
    validator = TelegramLoginWidgetPayloadValidator(bot_token=bot_token)
    jwt_codec = Hs256JwtCodec(secret_key="integration-jwt-secret", clock=clock)
    current_user_port = JwtCookieCurrentUser(jwt_codec=jwt_codec, user_repository=repository)
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name="roehub_identity_jwt",
    )
    login_use_case = TelegramLoginUseCase(
        validator=validator,
        user_repository=repository,
        jwt_codec=jwt_codec,
        clock=clock,
        jwt_ttl_days=7,
    )

    app = FastAPI()
    app.include_router(
        build_identity_router(
            telegram_login=login_use_case,
            current_user_dependency=current_user_dependency,
            cookie_name="roehub_identity_jwt",
            cookie_secure=False,
            cookie_samesite="lax",
            cookie_path="/",
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/auth/telegram/login",
        json=_build_signed_telegram_payload(
            bot_token=bot_token,
            user_id=5123001,
            auth_date=int(now.timestamp()),
        ),
    )

    assert response.status_code == 200
    assert response.json()["paid_level"] == "free"
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "roehub_identity_jwt=" in set_cookie_header
    assert "HttpOnly" in set_cookie_header

    current_user_response = client.get("/auth/current-user")
    assert current_user_response.status_code == 200
    assert current_user_response.json()["user_id"]
