from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, Lock
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    InMemoryOidcIdentityRepository,
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application import (
    AuthenticationProviderError,
    IdentityClock,
    OidcAuthenticationError,
    OidcAuthenticationService,
    VerifiedExternalIdentity,
)
from trading.shared_kernel.primitives import UserId

_SESSION_COOKIE_NAME = "roehub_session_id"
_BASE_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_ISSUER = "https://identity.example.test"
_INVITED_EMAIL = "invited@example.test"


class _MutableClock(IdentityClock):
    def __init__(self) -> None:
        self.value = _BASE_NOW

    def now(self) -> datetime:
        return self.value


@dataclass
class _FixtureProvider:
    external: VerifiedExternalIdentity
    unavailable: bool = False
    last_verifier: str | None = None
    last_nonce_hash: str | None = None
    block_exchange: bool = False
    exchange_calls: int = 0
    exchange_started: Event = field(init=False)
    exchange_release: Event = field(init=False)
    exchange_lock: Lock = field(init=False)
    exchange_hook: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.exchange_started = Event()
        self.exchange_release = Event()
        self.exchange_lock = Lock()

    @property
    def provider_id(self) -> str:
        return "fixture"

    @property
    def issuer(self) -> str:
        return _ISSUER

    @property
    def display_name(self) -> str:
        return "Fixture Identity"

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str:
        if self.unavailable:
            raise AuthenticationProviderError(code="provider_unavailable", retryable=True)
        return "https://identity.example.test/authorize?" + urlencode(
            {
                "state": state,
                "nonce": nonce,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        expected_nonce_sha256: str,
    ) -> VerifiedExternalIdentity:
        with self.exchange_lock:
            self.exchange_calls += 1
        self.exchange_started.set()
        if self.block_exchange:
            assert self.exchange_release.wait(timeout=2)
        if self.exchange_hook is not None:
            self.exchange_hook()
        if self.unavailable:
            raise AuthenticationProviderError(
                code="token_result_unknown", token_result_unknown=True
            )
        assert code == "disposable-code"
        self.last_verifier = code_verifier
        self.last_nonce_hash = expected_nonce_sha256
        return self.external


def test_current_user_dependency_rejects_missing_cookie() -> None:
    fixture = _build_fixture()

    response = fixture.client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "missing_session_id"


def test_current_user_dependency_rejects_unknown_session_cookie() -> None:
    fixture = _build_fixture()
    fixture.client.cookies.set(
        _SESSION_COOKIE_NAME, "00000000-0000-0000-0000-000000000001"
    )

    response = fixture.client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "session_not_found"


def test_oidc_login_uses_state_nonce_and_pkce_without_provider_payload_cookies() -> None:
    fixture = _build_fixture()

    response = fixture.client.get(
        "/auth/oidc/login?next=/strategies", follow_redirects=False
    )

    assert response.status_code == 303
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert len(query["state"][0]) >= 32
    assert len(query["nonce"][0]) >= 32
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["code_challenge"][0]) == 43
    assert fixture.client.cookies.get("roehub_oidc_attempt") is not None
    assert fixture.client.cookies.get("roehub_oidc_state") is None


def test_oidc_invitation_provisions_user_and_issues_opaque_session() -> None:
    fixture = _build_fixture(invite=True)
    state = _begin_login(fixture.client)

    response = fixture.client.get(
        f"/auth/oidc/callback?code=disposable-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/strategies"
    session_value = fixture.client.cookies.get(_SESSION_COOKIE_NAME)
    assert session_value is not None and session_value != "disposable-code"
    current_user = fixture.client.get("/auth/current-user")
    assert current_user.status_code == 200
    assert current_user.json()["user_id"] != str(fixture.owner_user_id)
    assert fixture.provider.last_verifier is not None
    assert fixture.provider.last_nonce_hash is not None


def test_oidc_uninvited_identity_is_rejected_without_session() -> None:
    fixture = _build_fixture(invite=False)
    state = _begin_login(fixture.client)

    response = fixture.client.get(
        f"/auth/oidc/callback?code=disposable-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "oidc_invitation_required"
    assert fixture.client.cookies.get(_SESSION_COOKIE_NAME) is None
    replay = fixture.client.get(
        f"/auth/oidc/callback?code=disposable-code&state={state}",
        follow_redirects=False,
    )
    assert replay.status_code == 400


def test_oidc_state_mismatch_is_rejected_before_code_exchange() -> None:
    fixture = _build_fixture(invite=True)
    _begin_login(fixture.client)

    response = fixture.client.get(
        "/auth/oidc/callback?code=disposable-code&state=wrong",
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert fixture.provider.last_verifier is None
    assert fixture.client.cookies.get(_SESSION_COOKIE_NAME) is None


def test_concurrent_oidc_callbacks_exchange_code_exactly_once() -> None:
    fixture = _build_fixture(invite=True)
    fixture.provider.block_exchange = True
    start = fixture.service.begin_login(next_path="/strategies")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            fixture.service.complete,
            attempt_id=start.attempt_id,
            state=start.state,
            code="disposable-code",
            callback_user_id=None,
        )
        assert fixture.provider.exchange_started.wait(timeout=1)
        second = executor.submit(
            fixture.service.complete,
            attempt_id=start.attempt_id,
            state=start.state,
            code="disposable-code",
            callback_user_id=None,
        )
        with pytest.raises(OidcAuthenticationError, match="oidc_attempt_invalid"):
            second.result(timeout=1)
        fixture.provider.exchange_release.set()
        result = first.result(timeout=1)

    assert result.provisioned is True
    assert fixture.provider.exchange_calls == 1


def test_attempt_expiring_during_provider_exchange_cannot_create_session() -> None:
    fixture = _build_fixture(invite=True)
    start = fixture.service.begin_login(next_path="/strategies")
    fixture.provider.exchange_hook = lambda: setattr(
        fixture.clock, "value", _BASE_NOW + timedelta(minutes=11)
    )

    with pytest.raises(OidcAuthenticationError, match="oidc_attempt_invalid"):
        fixture.service.complete(
            attempt_id=start.attempt_id,
            state=start.state,
            code="disposable-code",
            callback_user_id=None,
        )

    assert fixture.provider.exchange_calls == 1


def test_authenticated_linking_binds_provider_without_replacing_local_session() -> None:
    fixture = _build_fixture()
    owner_session = fixture.sessions.create_session(
        user_id=fixture.owner_user_id,
        now=_BASE_NOW,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    fixture.client.cookies.set(_SESSION_COOKIE_NAME, str(owner_session.session_id))

    start = fixture.client.get("/auth/oidc/link?next=/account", follow_redirects=False)
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    callback = fixture.client.get(
        f"/auth/oidc/callback?code=disposable-code&state={state}",
        follow_redirects=False,
    )

    assert callback.status_code == 303
    assert callback.headers["location"] == "/account"
    assert fixture.client.cookies.get(_SESSION_COOKIE_NAME) == str(owner_session.session_id)

    fixture.client.cookies.clear()
    login_state = _begin_login(fixture.client)
    login = fixture.client.get(
        f"/auth/oidc/callback?code=disposable-code&state={login_state}",
        follow_redirects=False,
    )
    assert login.status_code == 303
    assert fixture.client.get("/auth/current-user").json()["user_id"] == str(
        fixture.owner_user_id
    )


def test_provider_outage_does_not_break_local_existing_session() -> None:
    fixture = _build_fixture(provider_unavailable=True)
    owner_session = fixture.sessions.create_session(
        user_id=fixture.owner_user_id,
        now=_BASE_NOW,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )
    fixture.client.cookies.set(_SESSION_COOKIE_NAME, str(owner_session.session_id))

    provider_response = fixture.client.get("/auth/oidc/login", follow_redirects=False)
    local_response = fixture.client.get("/auth/current-user")

    assert provider_response.status_code == 503
    assert local_response.status_code == 200
    assert local_response.json()["user_id"] == str(fixture.owner_user_id)


@dataclass
class _Fixture:
    client: TestClient
    provider: _FixtureProvider
    sessions: InMemoryIdentitySessionRepository
    owner_user_id: UserId
    service: OidcAuthenticationService
    clock: _MutableClock


def _build_fixture(
    *, invite: bool = False, provider_unavailable: bool = False
) -> _Fixture:
    clock = _MutableClock()
    users = InMemoryIdentityUserRepository()
    sessions = InMemoryIdentitySessionRepository()
    organizations = InMemoryOrganizationRepository()
    owner_user_id = UserId(uuid4())
    users.create_local_user(user_id=owner_user_id, created_at=_BASE_NOW)
    _, organization = organizations.bootstrap_installation(
        owner_user_id=owner_user_id,
        installation_name="Fixture",
        organization_slug="fixture",
        organization_name="Fixture",
        created_at=_BASE_NOW,
    )
    if invite:
        organizations.create_invitation(
            organization_id=organization.organization_id,
            recipient_email_sha256=hashlib.sha256(_INVITED_EMAIL.encode()).hexdigest(),
            role="viewer",
            actor_user_id=owner_user_id,
            expires_at=_BASE_NOW.replace(day=14),
            created_at=_BASE_NOW,
        )
    oidc_repository = InMemoryOidcIdentityRepository(
        user_repository=users,
        organization_repository=organizations,
    )
    provider = _FixtureProvider(
        external=VerifiedExternalIdentity(
            issuer=_ISSUER,
            subject="disposable-subject",
            email=_INVITED_EMAIL,
            email_verified=True,
        ),
        unavailable=provider_unavailable,
    )
    service = OidcAuthenticationService(
        provider=provider,
        repository=oidc_repository,
        session_repository=sessions,
        clock=clock,
        session_idle_ttl_seconds=1800,
        session_absolute_ttl_seconds=43200,
    )
    current_user = RequireCurrentUserDependency(
        current_user=RoehubSessionCurrentUser(
            session_repository=sessions,
            user_repository=users,
            clock=clock,
        ),
        cookie_name=_SESSION_COOKIE_NAME,
    )
    app = FastAPI()
    app.include_router(
        build_identity_router(
            current_user_dependency=current_user,
            user_repository=users,
            session_repository=sessions,
            clock=clock,
            cookie_name=_SESSION_COOKIE_NAME,
            cookie_secure=False,
            session_idle_ttl_seconds=1800,
            session_absolute_ttl_seconds=43200,
            oidc_authentication_service=service,
        )
    )
    return _Fixture(
        client=TestClient(app),
        provider=provider,
        sessions=sessions,
        owner_user_id=owner_user_id,
        service=service,
        clock=clock,
    )


def _begin_login(client: TestClient) -> str:
    response = client.get(
        "/auth/oidc/login?next=/strategies", follow_redirects=False
    )
    assert response.status_code == 303
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]
