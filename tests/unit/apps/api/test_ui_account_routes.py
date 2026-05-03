from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.ui_account import build_ui_account_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityAccountSettingsRepository,
    InMemoryIdentityExchangeKeysRepository,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases import (
    AccountSettingsUseCase,
    ListExchangeKeysUseCase,
)
from trading.shared_kernel.primitives import UserId

"""
Stage 5 local endpoint contract:
- method/path: browser `/api/ui/account/...`, backend `/ui/account/...`.
- owner scope: all routes use identity current-user principal and owner_user_id storage.
- request DTO: profile/preferences/integrations/notifications are strict JSON bodies.
- response DTO: profile, limits, preferences, notifications, integrations, sessions, audit.
- status/error: 200 for reads/writes, 401 `auth.required`, 403 same-origin mutation guard,
  422 deterministic validation, 409 not introduced by account routes.
- pagination: sessions and audit use opaque cursor, stable newest-first ordering, max limit 100.
- cache identity: none; user-scoped live settings read-models.
- compatibility: additive compatible-change API/DTO/schema.
"""

_SESSION_COOKIE_NAME = "roehub_session_id"
_ORIGIN = "http://testserver"


class _MutableClock(IdentityClock):
    def __init__(self, *, now_value: datetime) -> None:
        self._now_value = now_value

    def tick(self, *, seconds: int = 60) -> datetime:
        self._now_value = self._now_value + timedelta(seconds=seconds)
        return self._now_value

    def now(self) -> datetime:
        return self._now_value


def test_ui_account_routes_require_authenticated_owner() -> None:
    context = _build_test_context()
    context.client.cookies.clear()

    response = context.client.get("/ui/account/preferences")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_ui_account_preferences_are_owner_scoped_and_write_audit_events() -> None:
    context = _build_test_context()

    response = context.client.put(
        "/ui/account/preferences",
        headers={"origin": _ORIGIN},
        json={"theme": "graphite", "locale": "ru", "density": "comfortable"},
    )

    assert response.status_code == 200
    assert response.json()["theme"] == "graphite"
    assert response.json()["locale"] == "ru"
    assert response.json()["density"] == "comfortable"

    audit_response = context.client.get("/ui/account/audit-events")
    assert audit_response.status_code == 200
    assert audit_response.json()["items"][0]["event_type"] == "account.preferences.updated"

    context.client.cookies.set(_SESSION_COOKIE_NAME, str(context.second_session_id))
    second_owner_response = context.client.get("/ui/account/preferences")
    assert second_owner_response.status_code == 200
    assert second_owner_response.json()["theme"] == "terminal-orange"
    assert second_owner_response.json()["locale"] == "en"
    assert context.client.get("/ui/account/audit-events").json()["items"] == []


def test_ui_account_mutations_require_same_origin_gate() -> None:
    context = _build_test_context()

    response = context.client.put("/ui/account/preferences", json={"locale": "ru"})

    assert response.status_code == 403
    assert response.json()["error"]["details"] == {"reason": "csrf_origin"}


def test_ui_account_rejects_unsupported_locale_with_deterministic_validation() -> None:
    context = _build_test_context()

    response = context.client.put(
        "/ui/account/preferences",
        headers={"origin": _ORIGIN},
        json={"locale": "de"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"]["errors"][0]["path"] == "body.locale"


def test_ui_account_sessions_and_audit_are_cursor_paginated() -> None:
    context = _build_test_context()
    for _index in range(3):
        context.clock.tick()
        context.session_repository.create_session(
            user_id=context.first_user_id,
            now=context.clock.now(),
            idle_ttl_seconds=1800,
            absolute_ttl_seconds=43200,
        )

    first_page = context.client.get("/ui/account/sessions?limit=2")
    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert len(first_payload["items"]) == 2
    assert first_payload["next_cursor"]

    second_page = context.client.get(
        f"/ui/account/sessions?limit=2&cursor={first_payload['next_cursor']}"
    )
    assert second_page.status_code == 200
    assert len(second_page.json()["items"]) == 2

    for locale in ("ru", "en", "ru"):
        context.clock.tick()
        response = context.client.put(
            "/ui/account/preferences",
            headers={"origin": _ORIGIN},
            json={"locale": locale},
        )
        assert response.status_code == 200

    audit_page = context.client.get("/ui/account/audit-events?limit=2")
    assert audit_page.status_code == 200
    audit_payload = audit_page.json()
    assert len(audit_payload["items"]) == 2
    assert audit_payload["next_cursor"]
    next_audit_page = context.client.get(
        f"/ui/account/audit-events?limit=2&cursor={audit_payload['next_cursor']}"
    )
    assert next_audit_page.status_code == 200
    assert len(next_audit_page.json()["items"]) == 1


def test_ui_account_integrations_and_notifications_are_validated_and_persisted() -> None:
    context = _build_test_context()

    integrations_response = context.client.put(
        "/ui/account/integrations",
        headers={"origin": _ORIGIN},
        json={
            "integrations": [
                {"provider": "telegram", "enabled": True},
                {"provider": "webhook_alerts", "enabled": False},
            ]
        },
    )
    assert integrations_response.status_code == 200
    integrations = {
        item["provider"]: item["enabled"]
        for item in integrations_response.json()["integrations"]
    }
    assert integrations["telegram"] is True
    assert integrations["webhook_alerts"] is False

    context.clock.tick()
    notifications_response = context.client.put(
        "/ui/account/notifications",
        headers={"origin": _ORIGIN},
        json={
            "email_notifications_enabled": False,
            "trade_alerts_enabled": True,
            "product_updates_enabled": True,
        },
    )
    assert notifications_response.status_code == 200
    assert notifications_response.json()["email_notifications_enabled"] is False
    assert context.client.get("/ui/account/audit-events").json()["items"][0][
        "event_type"
    ] == "account.notifications.updated"


@dataclass(frozen=True, slots=True)
class _TestContext:
    client: TestClient
    clock: _MutableClock
    session_repository: InMemoryIdentitySessionRepository
    first_user_id: UserId
    second_session_id: UUID


def _build_test_context() -> _TestContext:
    now = datetime(2026, 5, 3, 9, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()
    exchange_repository = InMemoryIdentityExchangeKeysRepository()
    account_repository = InMemoryIdentityAccountSettingsRepository(
        session_repository=session_repository
    )

    first_user = user_repository.upsert_keycloak_login(
        keycloak_subject="account-user-1",
        login_at=now,
    )
    first_session = session_repository.create_session(
        user_id=first_user.user_id,
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    second_user = user_repository.upsert_keycloak_login(
        keycloak_subject="account-user-2",
        login_at=now,
    )
    second_session = session_repository.create_session(
        user_id=second_user.user_id,
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=RoehubSessionCurrentUser(
            session_repository=session_repository,
            user_repository=user_repository,
            clock=clock,
        ),
        cookie_name=_SESSION_COOKIE_NAME,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_account_router(
            account_settings_use_case=AccountSettingsUseCase(
                repository=account_repository,
                clock=clock,
            ),
            list_exchange_keys_use_case=ListExchangeKeysUseCase(
                repository=exchange_repository
            ),
            current_user_dependency=current_user_dependency,
            allowed_ui_origins=(_ORIGIN,),
        )
    )
    client = TestClient(app)
    client.cookies.set(_SESSION_COOKIE_NAME, str(first_session.session_id))
    return _TestContext(
        client=client,
        clock=clock,
        session_repository=session_repository,
        first_user_id=first_user.user_id,
        second_session_id=second_session.session_id,
    )
