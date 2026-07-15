from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.inbound.api.routes import build_auth_local_router
from trading.contexts.identity.adapters.outbound import (
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    InMemoryLocalAuthRepository,
    InMemoryOrganizationRepository,
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application.use_cases import LocalAuthError, LocalAuthService
from trading.contexts.identity.application.use_cases import local_auth as local_auth_module


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _credential(*, challenge: str, credential_id: str, ceremony: str) -> dict[str, object]:
    client_data = json.dumps(
        {
            "type": ceremony,
            "challenge": challenge,
            "origin": "http://testserver",
        }
    ).encode()
    return {
        "id": credential_id,
        "rawId": credential_id,
        "type": "public-key",
        "response": {
            "clientDataJSON": _base64url(client_data),
            "attestationObject": _base64url(b"attestation"),
            "authenticatorData": _base64url(b"authenticator"),
            "signature": _base64url(b"signature"),
            "transports": ["internal"],
        },
        "clientExtensionResults": {},
    }


def _build_client(monkeypatch):
    clock = MutableClock()
    users = InMemoryIdentityUserRepository()
    sessions = InMemoryIdentitySessionRepository()
    organizations = InMemoryOrganizationRepository()
    repository = InMemoryLocalAuthRepository(
        user_repository=users,
        organization_repository=organizations,
    )
    service = LocalAuthService(
        repository=repository,
        user_repository=users,
        session_repository=sessions,
        clock=clock,
        rp_id="testserver",
        rp_name="Roehub Test",
        expected_origin="http://testserver",
        session_idle_ttl_seconds=1800,
        session_absolute_ttl_seconds=43200,
    )
    current_user = RequireCurrentUserDependency(
        current_user=RoehubSessionCurrentUser(
            session_repository=sessions,
            user_repository=users,
            clock=clock,
        ),
        cookie_name="roehub_session_id",
    )
    app = FastAPI()
    app.include_router(
        build_auth_local_router(
            service=service,
            current_user_dependency=current_user,
            session_repository=sessions,
            clock=clock,
            cookie_name="roehub_session_id",
            cookie_secure=False,
            session_absolute_ttl_seconds=43200,
        )
    )
    credential_bytes = b"local-passkey-credential-01"
    credential_id = _base64url(credential_bytes)
    monkeypatch.setattr(
        local_auth_module,
        "verify_registration_response",
        lambda **_kwargs: SimpleNamespace(
            credential_id=credential_bytes,
            credential_public_key=b"public-key-material-for-tests",
            sign_count=0,
        ),
    )
    monkeypatch.setattr(
        local_auth_module,
        "verify_authentication_response",
        lambda **_kwargs: SimpleNamespace(new_sign_count=1),
    )
    return TestClient(app), service, sessions, clock, credential_id


def _bootstrap(client: TestClient, service: LocalAuthService, credential_id: str):
    bootstrap_value = service.issue_bootstrap_ticket()
    fallback_value = "Valid-Local-Fallback-2026"
    options_response = client.post(
        "/auth/local/bootstrap/options",
        headers={"Origin": "http://testserver"},
        json={
            "ticket": bootstrap_value,
            "username": "owner",
            "display_name": "Local Owner",
            "installation_name": "Roehub Test",
            "organization_slug": "default",
            "organization_name": "Default Organization",
            "password": fallback_value,
        },
    )
    assert options_response.status_code == 200
    options = options_response.json()
    completion_response = client.post(
        "/auth/local/bootstrap/complete",
        headers={"Origin": "http://testserver"},
        json={
            "challenge_id": options["challenge_id"],
            "credential": _credential(
                challenge=options["publicKey"]["challenge"],
                credential_id=credential_id,
                ceremony="webauthn.create",
            ),
        },
    )
    assert completion_response.status_code == 200
    return completion_response, fallback_value


def test_bootstrap_passkey_login_recent_auth_and_logout(monkeypatch) -> None:
    client, service, sessions, _clock, credential_id = _build_client(monkeypatch)
    bootstrap_response, _fallback_value = _bootstrap(client, service, credential_id)
    first_session = client.cookies.get("roehub_session_id")
    csrf_value = client.cookies.get("roehub_csrf")

    assert len(bootstrap_response.json()["recovery_codes"]) == 8
    assert first_session is not None
    assert csrf_value is not None

    recent_options = client.post(
        "/auth/local/recent-auth/options",
        headers={"Origin": "http://testserver", "x-csrf-token": csrf_value},
    )
    assert recent_options.status_code == 200
    recent_payload = recent_options.json()
    recent_complete = client.post(
        "/auth/local/recent-auth/complete",
        headers={"Origin": "http://testserver", "x-csrf-token": csrf_value},
        json={
            "challenge_id": recent_payload["challenge_id"],
            "credential": _credential(
                challenge=recent_payload["publicKey"]["challenge"],
                credential_id=credential_id,
                ceremony="webauthn.get",
            ),
        },
    )
    assert recent_complete.status_code == 200
    rotated_session = client.cookies.get("roehub_session_id")
    assert rotated_session != first_session
    first_session_record = sessions.find_by_session_id(session_id=UUID(first_session))
    assert first_session_record is not None
    assert first_session_record.revoked_at is not None

    new_csrf_value = client.cookies.get("roehub_csrf")
    assert new_csrf_value is not None
    missing_csrf = client.post("/auth/local/logout", headers={"Origin": "http://testserver"})
    assert missing_csrf.status_code == 403
    logout = client.post(
        "/auth/local/logout",
        headers={"Origin": "http://testserver", "x-csrf-token": new_csrf_value},
    )
    assert logout.status_code == 204
    assert client.cookies.get("roehub_session_id") is None


def test_recent_auth_fails_closed_when_previous_session_cannot_be_revoked(
    monkeypatch,
) -> None:
    client, service, sessions, _clock, credential_id = _build_client(monkeypatch)
    _bootstrap(client, service, credential_id)
    current_session = client.cookies.get("roehub_session_id")
    assert current_session is not None
    session = sessions.find_by_session_id(session_id=UUID(current_session))
    assert session is not None

    options = service.begin_recent_auth(user_id=session.user_id)
    with pytest.raises(LocalAuthError):
        service.complete_recent_auth(
            challenge_id=options.challenge_id,
            user_id=session.user_id,
            credential=_credential(
                challenge=str(options.public_key["challenge"]),
                credential_id=credential_id,
                ceremony="webauthn.get",
            ),
            session_id_to_rotate=uuid4(),
        )


def test_password_errors_are_generic_and_rate_limit_locks_subject(monkeypatch) -> None:
    client, service, _sessions, clock, credential_id = _build_client(monkeypatch)
    _bootstrap_response, fallback_value = _bootstrap(client, service, credential_id)
    client.cookies.clear()

    error_payloads = []
    for username in ("unknown", "owner"):
        response = client.post(
            "/auth/local/password",
            headers={"Origin": "http://testserver"},
            json={"username": username, "password": "not-the-value"},
        )
        assert response.status_code == 401
        error_payloads.append(response.json())
    assert error_payloads[0] == error_payloads[1]

    for _attempt in range(4):
        client.post(
            "/auth/local/password",
            headers={"Origin": "http://testserver"},
            json={"username": "owner", "password": "not-the-value"},
        )
    locked = client.post(
        "/auth/local/password",
        headers={"Origin": "http://testserver"},
        json={"username": "owner", "password": fallback_value},
    )
    assert locked.status_code == 401

    clock.value += timedelta(minutes=16)
    unlocked = client.post(
        "/auth/local/password",
        headers={"Origin": "http://testserver"},
        json={"username": "owner", "password": fallback_value},
    )
    assert unlocked.status_code == 200


def test_recovery_code_is_one_time_and_revokes_prior_sessions(monkeypatch) -> None:
    client, service, sessions, _clock, credential_id = _build_client(monkeypatch)
    bootstrap_response, _fallback_value = _bootstrap(client, service, credential_id)
    recovery_value = bootstrap_response.json()["recovery_codes"][0]
    prior_session = client.cookies.get("roehub_session_id")
    assert prior_session is not None
    client.cookies.clear()

    recovered = client.post(
        "/auth/local/recovery",
        headers={"Origin": "http://testserver"},
        json={"username": "owner", "recovery_code": recovery_value},
    )
    assert recovered.status_code == 200
    prior_session_record = sessions.find_by_session_id(session_id=UUID(prior_session))
    assert prior_session_record is not None
    assert prior_session_record.revoked_at is not None

    client.cookies.clear()
    replay = client.post(
        "/auth/local/recovery",
        headers={"Origin": "http://testserver"},
        json={"username": "owner", "recovery_code": recovery_value},
    )
    assert replay.status_code == 401
