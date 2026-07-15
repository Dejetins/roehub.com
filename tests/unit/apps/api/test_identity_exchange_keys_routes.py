from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryAccountSettingsRepository,
    InMemoryIdentityExchangeKeysRepository,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    PostgresIdentityExchangeKeysRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.adapters.outbound.security.exchange_keys import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
)
from trading.shared_kernel.primitives import UserId

_KEYCLOAK_AUTH_URL = "https://auth.roehub.local/realms/roehub/protocol/openid-connect/auth"
_KEYCLOAK_TOKEN_URL = "https://auth.roehub.local/realms/roehub/protocol/openid-connect/token"
_KEYCLOAK_CLIENT_ID = "roehub-api"
_KEYCLOAK_CLIENT_SECRET = "test-client-secret"
_KEYCLOAK_REDIRECT_URI = "http://127.0.0.1:8010/auth/callback"
_KEYCLOAK_LOGOUT_REDIRECT_URI = "http://127.0.0.1:8010/login"
_KEYCLOAK_INTROSPECTION_URL = (
    "https://auth.roehub.local/realms/roehub/protocol/openid-connect/token/introspect"
)
_SESSION_COOKIE_NAME = "roehub_session_id"
_KEYCLOAK_SUBJECT = "keycloak-exchange-keys-user-1"
_MUTATION_HEADERS = {
    "origin": "http://testserver",
    "x-csrf-token": "test-csrf-token",
}


class _ProjectionGateway:
    def fetch_one(self, *, query: str, parameters: dict[str, object]) -> None:
        _ = query, parameters
        return None

    def execute(self, *, query: str, parameters: dict[str, object]) -> None:
        _ = query, parameters

    def fetch_all(
        self,
        *,
        query: str,
        parameters: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        _ = parameters
        if "exchange_connections" not in query:
            return ()
        return (
            {
                "key_id": "00000000-0000-0000-0000-000000000444",
                "user_id": "00000000-0000-0000-0000-000000000111",
                "exchange_name": "binance",
                "market_type": "spot",
                "label": "backfilled",
                "permissions": "read",
                "api_key_last4": "1234",
                "api_key_hash": b"1" * 32,
                "created_at": datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 5, 24, 12, 1, tzinfo=timezone.utc),
            },
        )


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


def test_exchange_keys_routes_require_authenticated_user_on_all_operations() -> None:
    """
    Verify create/list/delete exchange keys endpoints return deterministic 401
    when session cookie is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Current-user dependency resolves authenticated principal from Roehub session cookie only.
    Raises:
        AssertionError: If endpoint bypasses auth guard or payload changes.
    Side Effects:
        None.
    """
    client, _clock, _exchange_repository, _account_repository = _build_test_client()
    client.cookies.clear()

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
        assert response.status_code == 401
        assert response.json() == {
            "detail": {
                "error": "missing_session_id",
                "message": "Session id is required",
            }
        }


def test_exchange_keys_routes_reject_non_uuid_session_cookie_value() -> None:
    """
    Verify protected routes reject malformed non-UUID values in session cookie.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Browser auth cookie must contain Roehub UUID session id.
    Raises:
        AssertionError: If malformed cookie value is accepted as authenticated session.
    Side Effects:
        None.
    """
    client, _clock, _exchange_repository, _account_repository = _build_test_client()
    client.cookies.set(_SESSION_COOKIE_NAME, "malformed-cookie-value")

    response = client.get("/exchange-keys")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "invalid_session_id",
            "message": "Session id must be UUID",
        }
    }


def test_exchange_keys_crud_routes_hide_secrets_and_apply_soft_delete() -> None:
    """
    Verify create/list/delete route flow excludes secrets from responses and performs soft-delete.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authenticated user can operate exchange keys without local 2FA gate.
    Raises:
        AssertionError: If response shape leaks secret fields or delete semantics are broken.
    Side Effects:
        None.
    """
    client, clock, exchange_repository, account_repository = _build_test_client()

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
        headers=_MUTATION_HEADERS,
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
    delete_response = client.delete(
        f"/exchange-keys/{created_payload['key_id']}",
        headers=_MUTATION_HEADERS,
    )
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
    audit_events = account_repository._audit_events
    assert [event.event_type for event in audit_events] == [
        "exchange_key_created",
        "exchange_key_deleted",
    ]
    assert all("route-secret" not in str(event.metadata) for event in audit_events)
    assert audit_events[0].metadata == {
        "surface": "api",
        "key_id": created_payload["key_id"],
        "exchange_name": "binance",
        "market_type": "spot",
        "permissions": "trade",
    }


def test_legacy_exchange_keys_projection_reads_exchange_connections_first() -> None:
    repository = PostgresIdentityExchangeKeysRepository(
        gateway=_ProjectionGateway(),  # type: ignore[arg-type]
    )
    rows = repository.list_active_for_user(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
    )

    assert len(rows) == 1
    assert str(rows[0].key_id) == "00000000-0000-0000-0000-000000000444"
    assert rows[0].exchange_name == "binance"
    assert rows[0].market_type == "spot"
    assert rows[0].api_key_last4 == "1234"
    assert rows[0].api_key_enc == b"compatibility-projection"


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
    client, _clock, _exchange_repository, _account_repository = _build_test_client()

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
        headers=_MUTATION_HEADERS,
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
        headers=_MUTATION_HEADERS,
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
    client, _clock, _exchange_repository, _account_repository = _build_test_client()

    response = client.delete(
        "/exchange-keys/00000000-0000-0000-0000-00000000beef",
        headers=_MUTATION_HEADERS,
    )
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


def test_exchange_keys_mutations_fail_closed_without_origin_or_referer() -> None:
    client, _clock, _exchange_repository, _account_repository = _build_test_client()

    response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "blocked",
            "permissions": "read",
            "api_key": "CSRF-BLOCKED-0001",
            "api_secret": "csrf-blocked-secret",
            "passphrase": None,
        },
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "error": "csrf_required",
            "message": "CSRF protection is required for exchange credential mutations.",
            "reason": "csrf_required",
        }
    }


def test_exchange_keys_mutations_reject_cross_origin_requests() -> None:
    client, _clock, _exchange_repository, _account_repository = _build_test_client()

    response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "blocked",
            "permissions": "read",
            "api_key": "CSRF-CROSS-0001",
            "api_secret": "csrf-cross-origin-secret",
            "passphrase": None,
        },
        headers={"origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "error": "csrf_required",
            "message": "CSRF protection is required for exchange credential mutations.",
            "reason": "csrf_origin_mismatch",
        }
    }


def test_exchange_keys_mutations_require_recent_auth_after_same_origin_check() -> None:
    client, clock, _exchange_repository, _account_repository = _build_test_client()
    clock.set_now(now_value=clock.now() + timedelta(minutes=11))

    response = client.post(
        "/exchange-keys",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "needs-recent-auth",
            "permissions": "read",
            "api_key": "RECENT-AUTH-0001",
            "api_secret": "recent-auth-secret",
            "passphrase": None,
        },
        headers=_MUTATION_HEADERS,
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "error": "recent_auth_required",
            "message": "Recent Keycloak authentication is required.",
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
    client, clock, _exchange_repository, _account_repository = _build_test_client()

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
        headers=_MUTATION_HEADERS,
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
        headers=_MUTATION_HEADERS,
    )
    assert second_response.status_code == 201

    list_response = client.get("/exchange-keys")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert [item["label"] for item in payload] == ["first", "second"]


def _build_test_client() -> tuple[
    TestClient,
    _MutableClock,
    InMemoryIdentityExchangeKeysRepository,
    InMemoryAccountSettingsRepository,
]:
    """
    Build TestClient with identity router, Roehub session auth, and exchange key dependencies.

    Args:
        None.
    Returns:
        tuple[TestClient, _MutableClock, InMemoryIdentityExchangeKeysRepository]:
            `(client, clock, exchange_repository)` tuple.
    Assumptions:
        User and session are persisted in local in-memory repositories before route calls.
    Raises:
        ValueError: If dependency wiring is invalid.
    Side Effects:
        Creates in-memory FastAPI app and sets opaque Roehub session cookie on test client.
    """
    now = datetime(2026, 2, 15, 13, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)

    exchange_repository = InMemoryIdentityExchangeKeysRepository()
    account_repository = InMemoryAccountSettingsRepository()
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()

    user = user_repository.upsert_keycloak_login(
        keycloak_subject=_KEYCLOAK_SUBJECT,
        login_at=now,
    )
    session = session_repository.create_session(
        user_id=user.user_id,
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    current_user_port = RoehubSessionCurrentUser(
        session_repository=session_repository,
        user_repository=user_repository,
        clock=clock,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name=_SESSION_COOKIE_NAME,
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
    app.include_router(
        build_identity_router(
            current_user_dependency=current_user_dependency,
            audit_events_repository=account_repository,
            user_repository=user_repository,
            session_repository=session_repository,
            clock=clock,
            cookie_name=_SESSION_COOKIE_NAME,
            cookie_secure=False,
            session_idle_ttl_seconds=1800,
            session_absolute_ttl_seconds=43200,
            cookie_samesite="lax",
            cookie_path="/",
            create_exchange_key_use_case=create_exchange_key_use_case,
            list_exchange_keys_use_case=list_exchange_keys_use_case,
            delete_exchange_key_use_case=delete_exchange_key_use_case,
        )
    )

    client = TestClient(app)
    client.cookies.set(_SESSION_COOKIE_NAME, str(session.session_id))
    return client, clock, exchange_repository, account_repository



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
