from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
    RequireTwoFactorEnabledDependency,
    register_two_factor_required_exception_handler,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityExchangeKeysRepository,
    InMemoryIdentityTwoFactorRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.policy import RepositoryTwoFactorPolicyGate
from trading.contexts.identity.adapters.outbound.security.current_user import (
    JwtCookieCurrentUser,
)
from trading.contexts.identity.adapters.outbound.security.exchange_keys import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
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
from trading.contexts.identity.application.ports.jwt_codec import IdentityJwtClaims
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
    SetupTwoFactorTotpUseCase,
    TelegramLoginUseCase,
    VerifyTwoFactorTotpUseCase,
)
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel, UserId


class _MutableClock(IdentityClock):
    """
    Mutable deterministic UTC clock for exchange keys route tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize deterministic clock with initial UTC value.

        Args:
            now_value: Initial timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Test updates time via explicit `set_now` calls.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            None.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def set_now(self, *, now_value: datetime) -> None:
        """
        Replace deterministic clock value.

        Args:
            now_value: New timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Test controls timeline deterministically.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            Mutates internal clock value.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def now(self) -> datetime:
        """
        Return current deterministic UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current timestamp.
        Assumptions:
            Time does not auto-progress in tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_exchange_keys_routes_require_two_factor_enabled_on_all_operations() -> None:
    """
    Verify create/list/delete exchange keys endpoints return exact 403 payload when 2FA is disabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        User is authenticated but repository has no enabled 2FA row.
    Raises:
        AssertionError: If any endpoint bypasses 2FA gate or payload changes.
    Side Effects:
        None.
    """
    client, _clock, _user_id, _two_factor_repository, _exchange_repository = _build_test_client()

    responses = [
        client.get("/exchange-keys"),
        client.post(
            "/exchange-keys",
            json={
                "exchange_name": "binance",
                "market_type": "spot",
                "label": "main",
                "permissions": "read",
                "api_key": "ROUTE-KEY-0001",
                "api_secret": "route-secret-1",
                "passphrase": "route-passphrase-1",
            },
        ),
        client.delete("/exchange-keys/00000000-0000-0000-0000-00000000aaaa"),
    ]

    for response in responses:
        assert response.status_code == 403
        assert response.json() == {
            "error": "two_factor_required",
            "message": "Two-factor authentication must be enabled.",
        }


def test_exchange_keys_crud_routes_hide_secrets_and_apply_soft_delete() -> None:
    """
    Verify create/list/delete route flow excludes secrets from responses and performs soft-delete.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authenticated user has enabled 2FA before calling exchange keys endpoints.
    Raises:
        AssertionError: If response shape leaks secret fields or delete semantics are broken.
    Side Effects:
        None.
    """
    client, clock, user_id, two_factor_repository, exchange_repository = _build_test_client()
    _enable_two_factor(
        two_factor_repository=two_factor_repository,
        user_id=user_id,
        now=clock.now(),
    )

    create_response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "main",
            "permissions": "trade",
            "api_key": "ROUTE-KEY-1234",
            "api_secret": "route-secret-2",
            "passphrase": "route-passphrase-2",
        },
    )

    assert create_response.status_code == 201
    created_payload = create_response.json()
    assert list(created_payload.keys()) == [
        "key_id",
        "exchange_name",
        "market_type",
        "label",
        "permissions",
        "api_key",
        "created_at",
        "updated_at",
    ]
    assert created_payload["api_key"] == "****1234"
    for forbidden_field in (
        "api_secret",
        "passphrase",
        "api_key_enc",
        "api_secret_enc",
        "passphrase_enc",
        "api_key_hash",
    ):
        assert forbidden_field not in created_payload

    list_response = client.get("/exchange-keys")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert len(list_payload) == 1
    assert list_payload[0]["key_id"] == created_payload["key_id"]
    for forbidden_field in (
        "api_secret",
        "passphrase",
        "api_key_enc",
        "api_secret_enc",
        "passphrase_enc",
        "api_key_hash",
    ):
        assert forbidden_field not in list_payload[0]

    clock.set_now(now_value=clock.now() + timedelta(minutes=1))
    delete_response = client.delete(f"/exchange-keys/{created_payload['key_id']}")
    assert delete_response.status_code == 204

    empty_list_response = client.get("/exchange-keys")
    assert empty_list_response.status_code == 200
    assert empty_list_response.json() == []

    stored_row = exchange_repository._rows[created_payload["key_id"]]
    assert not hasattr(stored_row, "api_key")
    assert stored_row.api_key_enc != b"ROUTE-KEY-1234"
    assert stored_row.api_key_hash == hashlib.sha256(b"ROUTE-KEY-1234").digest()
    assert stored_row.api_key_last4 == "1234"
    assert stored_row.is_deleted is True
    assert stored_row.deleted_at is not None


def test_exchange_keys_create_route_returns_deterministic_409_for_active_duplicate() -> None:
    """
    Verify create route returns deterministic 409 payload for active duplicate keys.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Duplicate semantics rely on normalized API key hash and active-row uniqueness.
    Raises:
        AssertionError: If duplicate request does not produce deterministic 409 payload.
    Side Effects:
        None.
    """
    client, clock, user_id, two_factor_repository, _exchange_repository = _build_test_client()
    _enable_two_factor(
        two_factor_repository=two_factor_repository,
        user_id=user_id,
        now=clock.now(),
    )

    first_response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "main",
            "permissions": "trade",
            "api_key": "DUPLICATE-ROUTE-0001",
            "api_secret": "duplicate-secret-1",
            "passphrase": None,
        },
    )
    assert first_response.status_code == 201

    duplicate_response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "duplicate",
            "permissions": "trade",
            "api_key": "  DUPLICATE-ROUTE-0001  ",
            "api_secret": "duplicate-secret-2",
            "passphrase": None,
        },
    )

    assert duplicate_response.status_code == 409
    duplicate_payload = duplicate_response.json()
    assert list(duplicate_payload.keys()) == ["detail"]
    assert list(duplicate_payload["detail"].keys()) == ["error", "message"]
    assert duplicate_payload == {
        "detail": {
            "error": "exchange_key_already_exists",
            "message": "Exchange API key already exists.",
        }
    }


def test_exchange_keys_delete_route_returns_404_for_missing_key_id() -> None:
    """
    Verify delete route returns deterministic 404 payload for missing key identifiers.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Missing/foreign/already-deleted keys share the same not-found contract.
    Raises:
        AssertionError: If missing-key delete does not return deterministic 404 payload.
    Side Effects:
        None.
    """
    client, clock, user_id, two_factor_repository, _exchange_repository = _build_test_client()
    _enable_two_factor(
        two_factor_repository=two_factor_repository,
        user_id=user_id,
        now=clock.now(),
    )

    response = client.delete("/exchange-keys/00000000-0000-0000-0000-00000000beef")
    assert response.status_code == 404
    payload = response.json()
    assert list(payload.keys()) == ["detail"]
    assert list(payload["detail"].keys()) == ["error", "message"]
    assert payload == {
        "detail": {
            "error": "exchange_key_not_found",
            "message": "Exchange API key was not found.",
        }
    }


def test_exchange_keys_list_route_is_deterministically_sorted() -> None:
    """
    Verify list route returns keys in deterministic order by creation timestamp.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Create route timestamps come from deterministic mutable clock.
    Raises:
        AssertionError: If list ordering deviates from deterministic contract.
    Side Effects:
        None.
    """
    client, clock, user_id, two_factor_repository, _exchange_repository = _build_test_client()
    _enable_two_factor(
        two_factor_repository=two_factor_repository,
        user_id=user_id,
        now=clock.now(),
    )

    first_response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "bybit",
            "market_type": "futures",
            "label": "first",
            "permissions": "read",
            "api_key": "ORDER-ROUTE-0001",
            "api_secret": "order-secret-1",
            "passphrase": None,
        },
    )
    assert first_response.status_code == 201

    clock.set_now(now_value=clock.now() + timedelta(minutes=1))
    second_response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "second",
            "permissions": "trade",
            "api_key": "ORDER-ROUTE-0002",
            "api_secret": "order-secret-2",
            "passphrase": None,
        },
    )
    assert second_response.status_code == 201

    list_response = client.get("/exchange-keys")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert [item["label"] for item in payload] == ["first", "second"]


def _build_test_client() -> tuple[
    TestClient,
    _MutableClock,
    UserId,
    InMemoryIdentityTwoFactorRepository,
    InMemoryIdentityExchangeKeysRepository,
]:
    """
    Build TestClient with identity router, authenticated cookie, and exchange keys dependencies.

    Args:
        None.
    Returns:
        tuple[TestClient, _MutableClock, UserId, InMemoryIdentityTwoFactorRepository,
            InMemoryIdentityExchangeKeysRepository]:
            `(client, clock, user_id, two_factor_repository, exchange_repository)` tuple.
    Assumptions:
        JWT cookie is valid for one pre-created in-memory user.
    Raises:
        ValueError: If dependency wiring is invalid.
    Side Effects:
        Creates in-memory FastAPI app and sets auth cookie on test client.
    """
    now = datetime(2026, 2, 15, 13, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)

    user_repository = InMemoryIdentityUserRepository()
    two_factor_repository = InMemoryIdentityTwoFactorRepository()
    exchange_repository = InMemoryIdentityExchangeKeysRepository()

    user = user_repository.upsert_telegram_login(
        telegram_user_id=TelegramUserId(922001),
        login_at=now,
    )

    jwt_codec = Hs256JwtCodec(secret_key="exchange-keys-routes-secret", clock=clock)
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

    two_factor_policy_gate = RepositoryTwoFactorPolicyGate(repository=two_factor_repository)
    two_factor_enabled_dependency = RequireTwoFactorEnabledDependency(
        current_user_dependency=current_user_dependency,
        policy_gate=two_factor_policy_gate,
    )

    telegram_login_use_case = TelegramLoginUseCase(
        validator=TelegramLoginWidgetPayloadValidator(bot_token="exchange-keys-bot-token"),
        user_repository=user_repository,
        jwt_codec=jwt_codec,
        clock=clock,
        jwt_ttl_days=7,
    )

    two_factor_secret_cipher = AesGcmEnvelopeTwoFactorSecretCipher(
        kek_b64="cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    )
    two_factor_setup = SetupTwoFactorTotpUseCase(
        repository=two_factor_repository,
        secret_cipher=two_factor_secret_cipher,
        totp_provider=PyOtpTwoFactorTotpProvider(),
        clock=clock,
        issuer="Roehub",
    )
    two_factor_verify = VerifyTwoFactorTotpUseCase(
        repository=two_factor_repository,
        secret_cipher=two_factor_secret_cipher,
        totp_provider=PyOtpTwoFactorTotpProvider(),
        clock=clock,
    )

    exchange_secret_cipher = AesGcmEnvelopeExchangeKeysSecretCipher(
        kek_b64="cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    )
    create_exchange_key_use_case = CreateExchangeKeyUseCase(
        repository=exchange_repository,
        secret_cipher=exchange_secret_cipher,
        clock=clock,
    )
    list_exchange_keys_use_case = ListExchangeKeysUseCase(repository=exchange_repository)
    delete_exchange_key_use_case = DeleteExchangeKeyUseCase(
        repository=exchange_repository,
        clock=clock,
    )

    app = FastAPI()
    register_two_factor_required_exception_handler(app=app)
    app.include_router(
        build_identity_router(
            telegram_login=telegram_login_use_case,
            two_factor_setup=two_factor_setup,
            two_factor_verify=two_factor_verify,
            current_user_dependency=current_user_dependency,
            cookie_name="roehub_identity_jwt",
            cookie_secure=False,
            cookie_samesite="lax",
            cookie_path="/",
            create_exchange_key_use_case=create_exchange_key_use_case,
            list_exchange_keys_use_case=list_exchange_keys_use_case,
            delete_exchange_key_use_case=delete_exchange_key_use_case,
            two_factor_enabled_dependency=two_factor_enabled_dependency,
        )
    )

    client = TestClient(app)
    client.cookies.set("roehub_identity_jwt", token)
    return client, clock, user.user_id, two_factor_repository, exchange_repository



def _enable_two_factor(
    *,
    two_factor_repository: InMemoryIdentityTwoFactorRepository,
    user_id: UserId,
    now: datetime,
) -> None:
    """
    Enable 2FA state in repository for target user to satisfy policy gate in tests.

    Args:
        two_factor_repository: In-memory 2FA repository instance.
        user_id: Target identity user id.
        now: UTC timestamp for setup/enable rows.
    Returns:
        None.
    Assumptions:
        Any non-empty encrypted secret placeholder is sufficient for gate policy tests.
    Raises:
        ValueError: If repository invariants reject provided timestamp.
    Side Effects:
        Writes and updates one 2FA repository row.
    """
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
