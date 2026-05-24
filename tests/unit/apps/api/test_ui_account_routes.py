from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.exchange_control_client import (
    ExchangeControlClientConfig,
    ExchangeControlClientError,
    HttpExchangeControlClient,
    InMemoryExchangeControlClient,
)
from apps.api.routes.ui_account import build_ui_account_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryAccountSettingsRepository,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountSessionView,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
)
from trading.shared_kernel.primitives import UserId

_SESSION_COOKIE_NAME = "roehub_session_id"


class _MutableClock(IdentityClock):
    def __init__(self, *, now_value: datetime) -> None:
        self._now = now_value

    def now(self) -> datetime:
        return self._now


def test_ui_account_routes_require_authenticated_user() -> None:
    client, _account_repository, _session_ids = _build_test_client()
    client.cookies.clear()

    responses = [
        client.get("/ui/account/profile"),
        client.get("/ui/account/preferences"),
        client.get("/ui/account/sessions"),
        client.get("/ui/account/audit-events"),
        client.put(
            "/ui/account/preferences",
            json={
                "theme": "terminal-orange",
                "locale": "en",
                "density": "compact",
                "autorefresh_preset": "15s",
            },
        ),
    ]

    for response in responses:
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "auth.required"


def test_ui_account_profile_limits_integrations_and_notifications_contracts() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    profile = client.get("/ui/account/profile")
    limits = client.get("/ui/account/limits")
    integrations = client.get("/ui/account/integrations")
    notifications = client.get("/ui/account/notifications")

    assert profile.status_code == 200
    assert profile.json()["username"] == "quant_trader"
    assert profile.json()["locale"] == "en"
    assert profile.json()["subscription_status"] == "free"
    assert limits.status_code == 200
    assert limits.json()["plan"] == "free"
    assert limits.json()["exchange_connections_used"] == 0
    assert limits.json()["api_keys_used"] == 0
    assert limits.json()["exchange_connections_limit"] == 10
    assert integrations.status_code == 200
    assert [item["integration_key"] for item in integrations.json()["items"]] == [
        "telegram",
        "discord",
        "slack",
    ]
    assert notifications.status_code == 200
    assert len(notifications.json()["items"]) == 7


def test_ui_account_exchange_connections_create_list_rotate_disable_are_secret_safe() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    created = client.post(
        "/ui/account/exchange-connections",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "environment": "testnet",
            "label": "readonly",
            "permissions": "read",
            "api_key": "ACCOUNTKEY1234",
            "api_secret": "TEST_SECRET_STAGE4",
        },
        headers={"origin": "http://testserver"},
    )

    assert created.status_code == 201
    created_payload = created.json()
    connection_id = created_payload["connection_id"]
    first_version_id = created_payload["credential_version_id"]
    assert created_payload["api_key"] == "****1234"
    assert created_payload["permissions"] == "read"
    assert created_payload["environment"] == "testnet"
    assert created_payload["validation_status"] == "skipped_external_validation"
    for forbidden in ("TEST_SECRET_STAGE4", "ciphertext", "hmac"):
        assert forbidden not in created.text

    listed = client.get("/ui/account/exchange-connections")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload["items"][0]["connection_id"] == connection_id
    assert listed_payload["items"][0]["api_key"] == "****1234"

    limits_after_create = client.get("/ui/account/limits")
    assert limits_after_create.status_code == 200
    assert limits_after_create.json()["exchange_connections_used"] == 1
    assert limits_after_create.json()["api_keys_used"] == 1

    rotated = client.post(
        f"/ui/account/exchange-connections/{connection_id}/rotate",
        json={
            "api_key": "ACCOUNTKEY9876",
            "api_secret": "TEST_SECRET_ROTATED",
        },
        headers={"origin": "http://testserver"},
    )
    assert rotated.status_code == 200
    rotated_payload = rotated.json()
    assert rotated_payload["connection_id"] == connection_id
    assert rotated_payload["credential_version_id"] != first_version_id
    assert rotated_payload["api_key"] == "****9876"
    assert "TEST_SECRET_ROTATED" not in rotated.text

    disabled = client.post(
        f"/ui/account/exchange-connections/{connection_id}/disable",
        headers={"origin": "http://testserver"},
    )
    assert disabled.status_code == 200
    assert disabled.json()["connection_id"] == connection_id
    assert disabled.json()["status"] == "disabled"

    limits_after_disable = client.get("/ui/account/limits")
    assert limits_after_disable.status_code == 200
    assert limits_after_disable.json()["exchange_connections_used"] == 0
    assert limits_after_disable.json()["api_keys_used"] == 0


def test_ui_account_exchange_connections_allow_forwarded_public_same_origin() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    for index, forwarded_proto in enumerate(("https", "http")):
        response = client.post(
            "/ui/account/exchange-connections",
            json={
                "exchange_name": "bybit",
                "market_type": "spot",
                "environment": "mainnet",
                "permissions": "read",
                "api_key": f"ACCOUNTKEY111{index}",
                "api_secret": f"TEST_SECRET_PROXY_ORIGIN_{index}",
            },
            headers={
                "host": "macstudio-daniil.tail0ebbbc.ts.net",
                "origin": "https://roehub.com",
                "x-forwarded-host": "roehub.com",
                "x-forwarded-proto": forwarded_proto,
            },
        )

        assert response.status_code == 201
        assert response.json()["exchange_name"] == "bybit"
        assert "TEST_SECRET_PROXY_ORIGIN" not in response.text


def test_ui_account_exchange_connection_permissions_default_to_read() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    created = client.post(
        "/ui/account/exchange-connections",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "environment": "mainnet",
            "api_key": "ACCOUNTKEY7777",
            "api_secret": "TEST_SECRET_DEFAULT_READ",
        },
        headers={"origin": "http://testserver"},
    )

    assert created.status_code == 201
    assert created.json()["permissions"] == "read"
    assert "TEST_SECRET_DEFAULT_READ" not in created.text


def test_ui_account_exchange_connections_reject_passphrase_for_supported_exchanges() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    created = client.post(
        "/ui/account/exchange-connections",
        json={
            "exchange_name": "binance",
            "market_type": "spot",
            "environment": "testnet",
            "permissions": "read",
            "api_key": "ACCOUNTKEY1234",
            "api_secret": "TEST_SECRET_STAGE4",
            "passphrase": "SHOULD_NOT_BE_ACCEPTED",
        },
        headers={"origin": "http://testserver"},
    )
    rotated = client.post(
        "/ui/account/exchange-connections/00000000-0000-0000-0000-000000000001/rotate",
        json={
            "api_key": "ACCOUNTKEY9876",
            "api_secret": "TEST_SECRET_ROTATED",
            "passphrase": "SHOULD_NOT_BE_ACCEPTED",
        },
        headers={"origin": "http://testserver"},
    )

    assert created.status_code == 422
    assert rotated.status_code == 422
    for response in (created, rotated):
        assert "SHOULD_NOT_BE_ACCEPTED" not in response.text
        assert "TEST_SECRET" not in response.text


def test_ui_account_exchange_connection_validate_is_secret_safe_and_audited() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    created = client.post(
        "/ui/account/exchange-connections",
        json={
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "testnet",
            "label": "readonly",
            "permissions": "read",
            "api_key": "ACCOUNTKEY5555",
            "api_secret": "TEST_SECRET_STAGE5",
        },
        headers={"origin": "http://testserver"},
    )
    connection_id = created.json()["connection_id"]

    validated = client.post(
        f"/ui/account/exchange-connections/{connection_id}/validate",
        headers={"origin": "http://testserver"},
    )

    assert validated.status_code == 200
    payload = validated.json()
    assert payload["connection_id"] == connection_id
    assert payload["validation_status"] == "valid_readonly"
    assert payload["validation_reason"] == "fake_client_readonly"
    assert payload["last_validated_at"] is not None
    assert "TEST_SECRET_STAGE5" not in validated.text

    audit = client.get("/ui/account/audit-events")
    assert audit.status_code == 200
    assert audit.json()["items"][0]["event_type"] == "exchange_connection_validated"
    assert audit.json()["items"][0]["metadata"] == {
        "exchange": "bybit",
        "validation_status": "valid_readonly",
    }


def test_ui_account_exchange_connections_reject_linear_and_inverse_market_types() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    for market_type in ("linear", "inverse"):
        response = client.post(
            "/ui/account/exchange-connections",
            json={
                "exchange_name": "bybit",
                "market_type": market_type,
                "environment": "testnet",
                "permissions": "read",
                "api_key": "ACCOUNTKEY1234",
                "api_secret": "TEST_SECRET_STAGE4",
            },
            headers={"origin": "http://testserver"},
        )

        assert response.status_code == 422
        assert "TEST_SECRET_STAGE4" not in response.text


def test_ui_account_preferences_persist_locale_theme_autorefresh_and_write_audit() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    update = client.put(
        "/ui/account/preferences",
        json={
            "theme": "matrix-green",
            "locale": "ru",
            "density": "comfortable",
            "autorefresh_preset": "custom",
            "refresh_interval_seconds": 45,
        },
        headers={"origin": "http://testserver"},
    )
    assert update.status_code == 200
    payload = update.json()
    assert payload["theme"] == "matrix-green"
    assert payload["locale"] == "ru"
    assert payload["autorefresh"]["preset_key"] == "custom"
    assert payload["autorefresh"]["refresh_interval_seconds"] == 45

    restored = client.get("/ui/account/preferences")
    assert restored.status_code == 200
    assert restored.json()["theme"] == "matrix-green"
    assert restored.json()["locale"] == "ru"
    assert restored.json()["autorefresh"]["refresh_interval_seconds"] == 45

    audit = client.get("/ui/account/audit-events")
    assert audit.status_code == 200
    assert audit.json()["items"][0]["event_type"] == "preferences_updated"
    assert audit.json()["items"][0]["metadata"]["locale"] == "ru"


def test_ui_account_preferences_reject_unsupported_locale_and_too_low_interval() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    unsupported_locale = client.put(
        "/ui/account/preferences",
        json={
            "theme": "terminal-orange",
            "locale": "de",
            "density": "compact",
            "autorefresh_preset": "15s",
        },
    )
    assert unsupported_locale.status_code == 422
    assert unsupported_locale.json()["error"]["details"]["errors"][0] == {
        "code": "unsupported_locale",
        "message": "Unsupported locale preference.",
        "path": "locale",
    }

    too_low_interval = client.put(
        "/ui/account/preferences",
        json={
            "theme": "terminal-orange",
            "locale": "en",
            "density": "compact",
            "autorefresh_preset": "custom",
            "refresh_interval_seconds": 5,
        },
    )
    assert too_low_interval.status_code == 422
    assert too_low_interval.json()["error"]["details"]["errors"][0]["code"] == (
        "autorefresh_interval_too_low"
    )


def test_ui_account_integrations_and_notifications_mutations_mask_webhook_and_audit() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    integration = client.put(
        "/ui/account/integrations",
        json={
            "integration_key": "slack",
            "mode": "alerts",
            "webhook_url": "https://hooks.slack.test/services/T000/B000/SECRET",
        },
    )
    assert integration.status_code == 200
    assert integration.json()["status"] == "connected"
    assert integration.json()["webhook_url_masked"] == "https://...CRET"
    assert "SECRET" not in integration.text

    notification = client.put(
        "/ui/account/notifications",
        json={"channel_key": "risk_alerts", "mode": "critical"},
    )
    assert notification.status_code == 200
    assert notification.json()["mode"] == "critical"

    audit = client.get("/ui/account/audit-events?limit=10")
    event_types = [item["event_type"] for item in audit.json()["items"]]
    assert event_types == ["notifications_updated", "integration_updated"]


def test_ui_account_sessions_and_audit_are_owner_scoped_and_cursor_paginated() -> None:
    client, account_repository, session_ids = _build_test_client()
    account_repository.append_audit_event(
        owner_user_id=session_ids["first_user_id"],
        event_type="profile_updated",
        summary="first user event 1",
        metadata={},
        created_at=datetime(2026, 2, 15, 14, 0, tzinfo=timezone.utc),
    )
    account_repository.append_audit_event(
        owner_user_id=session_ids["first_user_id"],
        event_type="preferences_updated",
        summary="first user event 2",
        metadata={},
        created_at=datetime(2026, 2, 15, 15, 0, tzinfo=timezone.utc),
    )
    account_repository.append_audit_event(
        owner_user_id=session_ids["second_user_id"],
        event_type="profile_updated",
        summary="foreign user event",
        metadata={},
        created_at=datetime(2026, 2, 15, 16, 0, tzinfo=timezone.utc),
    )

    first_page = client.get("/ui/account/audit-events?limit=1")
    assert first_page.status_code == 200
    assert first_page.json()["items"][0]["summary"] == "first user event 2"
    assert first_page.json()["next_cursor"] == "1"

    second_page = client.get(
        f"/ui/account/audit-events?limit=1&cursor={first_page.json()['next_cursor']}"
    )
    assert second_page.status_code == 200
    assert second_page.json()["items"][0]["summary"] == "first user event 1"
    assert "foreign user event" not in second_page.text

    sessions = client.get("/ui/account/sessions?limit=1")
    assert sessions.status_code == 200
    assert len(sessions.json()["items"]) == 1
    assert sessions.json()["next_cursor"] == "1"


def test_ui_account_mutations_reject_cross_origin_requests() -> None:
    client, _account_repository, _session_ids = _build_test_client()

    response = client.put(
        "/ui/account/profile",
        json={"username": "quant_trader"},
        headers={"origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["details"] == {"reason": "csrf_origin_mismatch"}


def test_exchange_control_client_config_fails_closed_when_public_routes_enabled() -> None:
    with pytest.raises(ValueError, match="INTERNAL_BASE_URL"):
        ExchangeControlClientConfig.from_environ(
            {"ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED": "true"}
        )
    with pytest.raises(ValueError, match="INTERNAL_API_TOKEN"):
        ExchangeControlClientConfig.from_environ(
            {
                "ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED": "true",
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL": "http://127.0.0.1:9205",
            }
        )

    config = ExchangeControlClientConfig.from_environ(
        {
            "ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED": "true",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL": "http://127.0.0.1:9205",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        }
    )

    assert config.public_routes_enabled
    assert config.base_url == "http://127.0.0.1:9205"
    assert config.build_client() is not None


def test_exchange_control_http_client_sends_internal_auth_headers() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            status_code=200,
            json={
                "service": "exchange-control",
                "service_identity": "exchange-control",
                "contract_version": "internal-v1",
                "capabilities": ["capabilities.read"],
            },
        )

    client = HttpExchangeControlClient(
        base_url="http://127.0.0.1:9205",
        internal_api_token="internal-token",
        transport=httpx.MockTransport(handler),
    )

    capabilities = client.get_capabilities(request_id="stage-3c-test")

    assert capabilities.service == "exchange-control"
    assert capabilities.service_identity == "exchange-control"
    assert capabilities.contract_version == "internal-v1"
    assert capabilities.capabilities == ("capabilities.read",)
    assert captured_headers["authorization"] == "Bearer internal-token"
    assert captured_headers["x-roehub-internal-service"] == "apps/api"
    assert captured_headers["x-request-id"] == "stage-3c-test"


def test_exchange_control_http_client_sanitizes_failures() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=403,
            json={"error": {"message": "internal-token leaked"}},
        )

    client = HttpExchangeControlClient(
        base_url="http://127.0.0.1:9205",
        internal_api_token="internal-token",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ExchangeControlClientError) as exc_info:
        client.get_capabilities(request_id="stage-3c-test")

    message = str(exc_info.value)
    assert "403" in message
    assert "internal-token" not in message


def test_exchange_control_fake_client_is_deterministic() -> None:
    client = InMemoryExchangeControlClient()

    first = client.get_capabilities(request_id="first")
    second = client.get_capabilities(request_id="second")

    assert first == second
    assert first.service == "exchange-control"


def _build_test_client() -> tuple[
    TestClient,
    InMemoryAccountSettingsRepository,
    dict[str, UserId],
]:
    now = datetime(2026, 2, 15, 13, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()
    account_repository = InMemoryAccountSettingsRepository()

    first_user = user_repository.upsert_keycloak_login(
        keycloak_subject="ui-account-user-1",
        login_at=now,
    )
    second_user = user_repository.upsert_keycloak_login(
        keycloak_subject="ui-account-user-2",
        login_at=now,
    )
    first_session = session_repository.create_session(
        user_id=first_user.user_id,
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    second_session = session_repository.create_session(
        user_id=second_user.user_id,
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    account_repository.record_session(
        session=AccountSessionView(
            session_id=str(first_session.session_id),
            owner_user_id=first_user.user_id,
            created_at=now,
            last_seen_at=now + timedelta(minutes=5),
            idle_expires_at=first_session.idle_expires_at,
            absolute_expires_at=first_session.absolute_expires_at,
            revoked_at=None,
            device="Chrome / macOS",
            ip_address="185.67.23.118",
            location="Moscow, RU",
            is_current=True,
        )
    )
    account_repository.record_session(
        session=AccountSessionView(
            session_id="older-session",
            owner_user_id=first_user.user_id,
            created_at=now,
            last_seen_at=now,
            idle_expires_at=first_session.idle_expires_at,
            absolute_expires_at=first_session.absolute_expires_at,
            revoked_at=None,
            device="Safari / iOS",
            ip_address="185.130.155.156",
            location="Kiev, UA",
            is_current=False,
        )
    )
    account_repository.record_session(
        session=AccountSessionView(
            session_id=str(second_session.session_id),
            owner_user_id=second_user.user_id,
            created_at=now,
            last_seen_at=now + timedelta(minutes=15),
            idle_expires_at=second_session.idle_expires_at,
            absolute_expires_at=second_session.absolute_expires_at,
            revoked_at=None,
            device="Foreign browser",
            ip_address="203.0.113.10",
            location="foreign",
            is_current=True,
        )
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
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_account_router(
            account_settings=AccountSettingsUseCase(
                repository=account_repository,
                clock=clock,
            ),
            current_user_dependency=current_user_dependency,
            clock=clock,
            exchange_control_client=InMemoryExchangeControlClient(),
        )
    )
    client = TestClient(app)
    client.cookies.set(_SESSION_COOKIE_NAME, str(first_session.session_id))
    return client, account_repository, {
        "first_user_id": first_user.user_id,
        "second_user_id": second_user.user_id,
    }
