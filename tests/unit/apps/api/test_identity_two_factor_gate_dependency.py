from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
    RequireTwoFactorEnabledDependency,
    register_two_factor_required_exception_handler,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityTwoFactorRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.policy import RepositoryTwoFactorPolicyGate
from trading.contexts.identity.adapters.outbound.security.current_user import (
    JwtCookieCurrentUser,
)
from trading.contexts.identity.adapters.outbound.security.jwt import Hs256JwtCodec
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.ports.jwt_codec import IdentityJwtClaims
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel, UserId


class _FixedClock(IdentityClock):
    """
    Fixed UTC clock used for deterministic token and timestamp operations.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize fixed clock with timezone-aware UTC datetime value.

        Args:
            now_value: Fixed UTC datetime.
        Returns:
            None.
        Assumptions:
            Same timestamp is used for JWT issuance and test assertions.
        Raises:
            ValueError: If datetime is naive or non-UTC.
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
            Time progression is unnecessary for these gate tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_gate_dependency_returns_exact_403_payload_when_two_factor_disabled() -> None:
    """
    Verify reusable 2FA gate dependency returns exact required 403 payload when disabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authenticated user has valid JWT but no enabled 2FA state.
    Raises:
        AssertionError: If status code or payload deviates from required contract.
    Side Effects:
        None.
    """
    client, token, _two_factor_repository, _user_id, _now = _build_gate_test_client()
    client.cookies.set("roehub_identity_jwt", token)

    response = client.get("/exchange-keys/ping")

    assert response.status_code == 403
    assert response.json() == {
        "error": "two_factor_required",
        "message": "Two-factor authentication must be enabled.",
    }


def test_gate_dependency_allows_request_when_two_factor_is_enabled() -> None:
    """
    Verify reusable 2FA gate dependency allows request when 2FA state is enabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        In-memory 2FA repository can transition from pending to enabled state.
    Raises:
        AssertionError: If enabled state is not accepted by gate dependency.
    Side Effects:
        None.
    """
    client, token, two_factor_repository, user_id, now = _build_gate_test_client()
    two_factor_repository.upsert_pending_secret(
        user_id=user_id,
        totp_secret_enc=b"\x01\x02\x03",
        updated_at=now,
    )
    two_factor_repository.enable(
        user_id=user_id,
        enabled_at=now,
        updated_at=now,
    )
    client.cookies.set("roehub_identity_jwt", token)

    response = client.get("/exchange-keys/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _build_gate_test_client() -> tuple[
    TestClient,
    str,
    InMemoryIdentityTwoFactorRepository,
    UserId,
    datetime,
]:
    """
    Build test client and dependencies for reusable 2FA gate behavior checks.

    Args:
        None.
    Returns:
        tuple[TestClient, str, InMemoryIdentityTwoFactorRepository, UserId, datetime]:
            `(client, jwt_token, two_factor_repository, user_id, now)` tuple.
    Assumptions:
        User repository contains one active user matching JWT claims subject.
    Raises:
        ValueError: If dependency wiring is invalid.
    Side Effects:
        Creates FastAPI app with registered 2FA required exception handler.
    """
    now = datetime(2026, 2, 14, 19, 0, 0, tzinfo=timezone.utc)
    clock = _FixedClock(now_value=now)
    user_repository = InMemoryIdentityUserRepository()
    two_factor_repository = InMemoryIdentityTwoFactorRepository()
    user = user_repository.upsert_telegram_login(
        telegram_user_id=TelegramUserId(990001),
        login_at=now,
    )
    jwt_codec = Hs256JwtCodec(secret_key="gate-dependency-secret", clock=clock)
    claims = IdentityJwtClaims(
        user_id=user.user_id,
        paid_level=PaidLevel.free(),
        issued_at=now,
        expires_at=now + timedelta(days=7),
    )
    token = jwt_codec.encode(claims=claims)
    current_user_port = JwtCookieCurrentUser(
        jwt_codec=jwt_codec,
        user_repository=user_repository,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name="roehub_identity_jwt",
    )
    policy_gate = RepositoryTwoFactorPolicyGate(repository=two_factor_repository)
    two_factor_dependency = RequireTwoFactorEnabledDependency(
        current_user_dependency=current_user_dependency,
        policy_gate=policy_gate,
    )

    app = FastAPI()
    register_two_factor_required_exception_handler(app=app)

    @app.get("/exchange-keys/ping")
    def get_exchange_keys_ping(
        _principal: CurrentUserPrincipal = Depends(two_factor_dependency),
    ) -> dict[str, str]:
        """
        Dummy protected endpoint for verifying reusable 2FA gate dependency behavior.

        Args:
            _principal: Principal resolved by current user + 2FA gate dependency.
        Returns:
            dict[str, str]: Stable success payload.
        Assumptions:
            Any request reaching endpoint has passed authentication and 2FA gate checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"status": "ok"}

    return TestClient(app), token, two_factor_repository, user.user_id, now
