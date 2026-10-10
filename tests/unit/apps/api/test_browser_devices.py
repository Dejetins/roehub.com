from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from trading.contexts.identity.adapters.inbound.api.browser_device import browser_label
from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.inbound.api.routes.browser_devices import (
    build_browser_devices_router,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory.session_repository import (
    InMemoryIdentitySessionRepository,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


class Clock:
    def now(self):
        return NOW


def stored(repo, session_id):
    session = repo.find_by_session_id(session_id=session_id)
    assert session is not None
    return session


def fixture():
    repo = InMemoryIdentitySessionRepository()
    owner = UserId(uuid4())
    current = repo.create_session(
        user_id=owner, now=NOW, idle_ttl_seconds=3600, absolute_ttl_seconds=7200
    )

    def require(request: Request):
        try:
            session = repo.find_by_session_id(session_id=UUID(request.cookies["session"]))
        except (ValueError, KeyError):
            raise HTTPException(401) from None
        if session is None or not session.is_active_at(at=NOW):
            raise HTTPException(401)
        return CurrentUserPrincipal(
            user_id=session.user_id, paid_level=PaidLevel("free"), session_id=session.session_id
        )

    app = FastAPI()
    app.include_router(
        build_browser_devices_router(
            repository=repo,
            current_user_dependency=cast(RequireCurrentUserDependency, require),
            clock=Clock(),
            cookie_secure=False,
            cookie_name="session",
            cookie_path="/",
        )
    )
    client = TestClient(app)
    client.cookies.set("session", str(current.session_id))
    client.cookies.set("roehub_csrf", "disposable-csrf")
    return client, repo, owner, current


def test_adopts_current_and_never_exposes_credentials_or_raw_user_agent():
    client, repo, owner, current = fixture()
    repo.create_session(
        user_id=owner,
        now=NOW - timedelta(days=1),
        idle_ttl_seconds=7 * 86400,
        absolute_ttl_seconds=7 * 86400,
    )
    response = client.get(
        "/ui/account/devices", headers={"User-Agent": "Chrome/123 Macintosh secret"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["legacy_active_count"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["is_current"] is True
    assert data["items"][0]["label"] == "Chrome · macOS"
    assert str(current.session_id) not in response.text
    assert "secret" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert "HttpOnly" in response.headers["set-cookie"]
    assert client.get("/ui/account/devices").json()["items"] == data["items"]


def test_revocation_requires_csrf_and_is_owner_scoped():
    client, repo, owner, current = fixture()
    did = client.get("/ui/account/devices").json()["items"][0]["device_id"]
    other = repo.create_session(
        user_id=UserId(uuid4()), now=NOW, idle_ttl_seconds=3600, absolute_ttl_seconds=3600
    )
    other_did = repo.bind_device(
        session_id=other.session_id, device_key="b" * 64, label="Firefox", now=NOW
    )
    route = f"/ui/account/devices/{did}/revoke"
    assert client.post(route).status_code == 403
    assert (
        client.post(
            route, headers={"Origin": "https://evil.example", "X-CSRF-Token": "disposable-csrf"}
        ).status_code
        == 403
    )
    headers = {"Origin": "http://testserver", "X-CSRF-Token": "disposable-csrf"}
    assert (
        client.post(f"/ui/account/devices/{other_did}/revoke", headers=headers).status_code == 200
    )
    assert stored(repo, other.session_id).is_active_at(at=NOW)
    assert len(client.get("/ui/account/devices").json()["items"]) == 1
    result = client.post(route, headers=headers)
    assert result.json() == {"revoked": True, "current": True}
    assert stored(repo, current.session_id).revoked_at == NOW
    assert client.get("/ui/account/devices").status_code == 401


def test_end_legacy_preserves_current_and_other_accounts():
    client, repo, owner, current = fixture()
    legacy = repo.create_session(
        user_id=owner, now=NOW, idle_ttl_seconds=3600, absolute_ttl_seconds=3600
    )
    other = repo.create_session(
        user_id=UserId(uuid4()), now=NOW, idle_ttl_seconds=3600, absolute_ttl_seconds=3600
    )
    headers = {"Origin": "http://testserver", "X-CSRF-Token": "disposable-csrf"}
    assert client.post("/ui/account/devices/legacy/revoke", headers=headers).status_code == 200
    assert not stored(repo, legacy.session_id).is_active_at(at=NOW)
    assert stored(repo, current.session_id).is_active_at(at=NOW)
    assert stored(repo, other.session_id).is_active_at(at=NOW)
    assert client.get("/ui/account/devices").json()["legacy_active_count"] == 0


def test_browser_labels_are_bounded_and_conservative():
    assert browser_label("Mozilla Android Chrome/123 Edg/123") == "Edge · Android"
    assert browser_label("<script>" * 1000) == "Browser"


def test_in_memory_activity_cannot_restore_a_concurrently_revoked_session(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
        session_repository as module,
    )

    _, repo, _, current = fixture()
    entered, release, revoke_started = Event(), Event(), Event()
    original = module.replace

    def paused_replace(value, **fields):
        if "last_seen_at" in fields:
            entered.set()
            assert release.wait(2)
        return original(value, **fields)

    monkeypatch.setattr(module, "replace", paused_replace)
    later = NOW + timedelta(minutes=2)

    def revoke():
        revoke_started.set()
        repo.revoke_session(session_id=current.session_id, revoked_at=later)

    with ThreadPoolExecutor(max_workers=2) as executor:
        touch = executor.submit(repo.touch_session, session_id=current.session_id, now=later)
        assert entered.wait(2)
        revoked = executor.submit(revoke)
        assert revoke_started.wait(2)
        release.set()
        touch.result(timeout=2)
        revoked.result(timeout=2)
    assert not stored(repo, current.session_id).is_active_at(at=later)
