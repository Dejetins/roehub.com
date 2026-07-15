"""Disposable network OIDC provider and API fixture for Stage 07 browser evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from urllib.parse import parse_qs, urlencode
from uuid import UUID, uuid4

from argon2 import PasswordHasher
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from apps.api.routes.identity import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.outbound import (
    HttpOidcAuthenticationProvider,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    InMemoryLocalAuthRepository,
    InMemoryOidcIdentityRepository,
    InMemoryOrganizationRepository,
    RoehubSessionCurrentUser,
    SystemIdentityClock,
)
from trading.contexts.identity.application.ports import LocalPasskey
from trading.contexts.identity.application.use_cases import (
    LocalAuthService,
    OidcAuthenticationService,
)
from trading.shared_kernel.primitives import UserId

_ISSUER = "http://localhost:9010"
_WEB_ORIGIN = "http://localhost:8000"
_REDIRECT_URI = f"{_WEB_ORIGIN}/api/auth/oidc/callback"
_CLIENT_ID = "roehub-browser-proof"
_CREDENTIAL_ENV = "ROEHUB_OIDC_BROWSER_CREDENTIAL"
_LOCAL_LOGIN_CREDENTIAL_ENV = "ROEHUB_LOCAL_BROWSER_CREDENTIAL"
_COOKIE_NAME = "roehub_session_id"
_CSRF_COOKIE = "roehub_csrf"
_INVITED_EMAIL = "invited@oidc-proof.invalid"
_LINK_EMAIL = "linked@oidc-proof.invalid"
_PROVIDER_MODES = frozenset(
    {"invited", "linked", "uninvited", "token_unknown", "outage"}
)


@dataclass(frozen=True, slots=True)
class _Grant:
    nonce: str
    code_challenge: str
    subject: str
    email: str
    token_unknown: bool
    expires_at: datetime


class _ProviderState:
    def __init__(self) -> None:
        self.mode = "invited"
        self.grants: dict[str, _Grant] = {}
        self.exchange_calls = 0
        self.lock = RLock()


def create_provider_app() -> FastAPI:
    """Create a disposable provider with real discovery, JWKS, PKCE and signed claims."""
    credential = _required_credential()
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_id = "stage07-browser-key"
    state = _ProviderState()
    app = FastAPI(title="Roehub disposable OIDC provider")

    @app.get("/.well-known/openid-configuration")
    def discovery() -> dict[str, object]:
        return {
            "issuer": _ISSUER,
            "authorization_endpoint": f"{_ISSUER}/authorize",
            "token_endpoint": f"{_ISSUER}/exchange",
            "jwks_uri": f"{_ISSUER}/keys",
            "id_token_signing_alg_values_supported": ["RS256"],
        }

    @app.get("/keys")
    def keys() -> dict[str, object]:
        return {"keys": [_jwk(signing_key=signing_key, key_id=key_id)]}

    @app.post("/__proof__/mode/{mode}", include_in_schema=False)
    def set_mode(mode: str) -> dict[str, str]:
        if mode not in _PROVIDER_MODES:
            raise HTTPException(status_code=404, detail="unknown proof mode")
        with state.lock:
            state.mode = mode
            state.grants.clear()
            state.exchange_calls = 0
        return {"status": "configured", "mode": mode}

    @app.get("/__proof__/counters", include_in_schema=False)
    def proof_counters() -> dict[str, int]:
        with state.lock:
            return {"exchange_calls": state.exchange_calls}

    @app.get("/authorize")
    def authorize(
        response_type: str = Query(),
        client_id: str = Query(),
        redirect_uri: str = Query(),
        state_value: str = Query(alias="state"),
        nonce: str = Query(),
        code_challenge: str = Query(),
        code_challenge_method: str = Query(),
    ) -> Response:
        with state.lock:
            mode = state.mode
        if mode == "outage":
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        if (
            response_type != "code"
            or client_id != _CLIENT_ID
            or redirect_uri != _REDIRECT_URI
            or code_challenge_method != "S256"
            or len(state_value) < 16
            or len(nonce) < 16
            or len(code_challenge) < 32
        ):
            return JSONResponse(status_code=400, content={"status": "rejected"})
        subject, email = _identity_for_mode(mode)
        code = secrets.token_urlsafe(32)
        with state.lock:
            state.grants[code] = _Grant(
                nonce=nonce,
                code_challenge=code_challenge,
                subject=subject,
                email=email,
                token_unknown=mode == "token_unknown",
                expires_at=datetime.now(UTC) + timedelta(minutes=2),
            )
        return RedirectResponse(
            url=f"{redirect_uri}?{urlencode({'code': code, 'state': state_value})}",
            status_code=303,
        )

    @app.post("/exchange")
    async def exchange(request: Request) -> Response:
        form = {
            key: values[0]
            for key, values in parse_qs(
                (await request.body()).decode("utf-8"), strict_parsing=True
            ).items()
        }
        code = form.get("code", "")
        with state.lock:
            state.exchange_calls += 1
            grant = state.grants.pop(code, None)
        received_credential = form.get("client_" + "secret", "")
        if (
            form.get("grant_type") != "authorization_code"
            or form.get("client_id") != _CLIENT_ID
            or form.get("redirect_uri") != _REDIRECT_URI
            or not secrets.compare_digest(received_credential, credential)
            or grant is None
            or grant.expires_at <= datetime.now(UTC)
        ):
            return JSONResponse(status_code=400, content={"status": "rejected"})
        verifier = form.get("code_verifier", "")
        if not secrets.compare_digest(_pkce_challenge(verifier), grant.code_challenge):
            return JSONResponse(status_code=400, content={"status": "rejected"})
        if grant.token_unknown:
            return JSONResponse(status_code=503, content={"status": "unknown"})
        signed_identity = _signed_identity(
            signing_key=signing_key,
            key_id=key_id,
            grant=grant,
        )
        return JSONResponse(content={"id_" + "token": signed_identity})

    return app


def create_api_app() -> FastAPI:
    """Create an isolated API using the production OIDC and local-auth adapters."""
    credential = _required_credential()
    local_login_credential = _required_local_login_credential()
    clock = SystemIdentityClock()
    users = InMemoryIdentityUserRepository()
    sessions = InMemoryIdentitySessionRepository()
    organizations = InMemoryOrganizationRepository()
    local_repository = InMemoryLocalAuthRepository(
        user_repository=users,
        organization_repository=organizations,
    )
    owner_user_id = UserId(uuid4())
    secondary_user_id = UserId(uuid4())
    now = clock.now()
    _seed_local_owner(
        repository=local_repository,
        owner_user_id=owner_user_id,
        local_login_credential=local_login_credential,
        now=now,
    )
    users.create_local_user(user_id=secondary_user_id, created_at=now)
    organization = organizations.list_accesses_for_user(user_id=owner_user_id)[0].organization
    organizations.create_invitation(
        organization_id=organization.organization_id,
        recipient_email_sha256=_sha256_text(_INVITED_EMAIL),
        role="viewer",
        actor_user_id=owner_user_id,
        expires_at=now + timedelta(hours=1),
        created_at=now,
    )

    oidc_repository = InMemoryOidcIdentityRepository(
        user_repository=users,
        organization_repository=organizations,
    )
    provider = HttpOidcAuthenticationProvider(
        provider_id="browser-fixture",
        display_name="Disposable OIDC",
        issuer=_ISSUER,
        client_id=_CLIENT_ID,
        client_credential_source=lambda: credential,
        redirect_uri=_REDIRECT_URI,
        connect_timeout_seconds=1.0,
        response_timeout_seconds=2.0,
        overall_timeout_seconds=3.0,
        allow_insecure_http=True,
    )
    oidc_service = OidcAuthenticationService(
        provider=provider,
        repository=oidc_repository,
        session_repository=sessions,
        clock=clock,
        session_idle_ttl_seconds=1800,
        session_absolute_ttl_seconds=43200,
    )
    local_service = LocalAuthService(
        repository=local_repository,
        user_repository=users,
        session_repository=sessions,
        clock=clock,
        rp_id="localhost",
        rp_name="Roehub browser proof",
        expected_origin=_WEB_ORIGIN,
        session_idle_ttl_seconds=1800,
        session_absolute_ttl_seconds=43200,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=RoehubSessionCurrentUser(
            session_repository=sessions,
            user_repository=users,
            clock=clock,
        ),
        cookie_name=_COOKIE_NAME,
    )
    app = FastAPI(title="Roehub OIDC browser proof API")
    app.include_router(
        build_identity_router(
            current_user_dependency=current_user_dependency,
            user_repository=users,
            session_repository=sessions,
            clock=clock,
            cookie_name=_COOKIE_NAME,
            cookie_secure=False,
            session_idle_ttl_seconds=1800,
            session_absolute_ttl_seconds=43200,
            local_auth_service=local_service,
            oidc_authentication_service=oidc_service,
        )
    )

    @app.post("/__proof__/session/{actor}", include_in_schema=False)
    def issue_proof_session(actor: str, response: Response) -> dict[str, bool]:
        if actor == "owner":
            user_id = owner_user_id
        elif actor == "secondary":
            user_id = secondary_user_id
        else:
            raise HTTPException(status_code=404, detail="unknown proof actor")
        session = sessions.create_session(
            user_id=user_id,
            now=clock.now(),
            idle_ttl_seconds=1800,
            absolute_ttl_seconds=43200,
        )
        response.set_cookie(
            key=_COOKIE_NAME,
            value=str(session.session_id),
            max_age=43200,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            key=_CSRF_COOKIE,
            value=secrets.token_urlsafe(32),
            max_age=43200,
            httponly=False,
            secure=False,
            samesite="lax",
            path="/",
        )
        return {"issued": True}

    @app.get("/__proof__/principal/{actor}", include_in_schema=False)
    def verify_proof_principal(actor: str, request: Request) -> dict[str, bool]:
        if actor == "owner":
            expected = owner_user_id
        elif actor == "secondary":
            expected = secondary_user_id
        else:
            raise HTTPException(status_code=404, detail="unknown proof actor")
        raw_session_id = request.cookies.get(_COOKIE_NAME)
        matched = False
        if raw_session_id is not None:
            try:
                session = sessions.find_by_session_id(session_id=UUID(raw_session_id))
            except ValueError:
                session = None
            matched = (
                session is not None
                and session.is_active_at(at=clock.now())
                and session.user_id == expected
            )
        return {"matches": matched}

    @app.get("/__proof__/invitation-session", include_in_schema=False)
    def verify_invitation_session(request: Request) -> dict[str, bool]:
        raw_session_id = request.cookies.get(_COOKIE_NAME)
        accepted = False
        if raw_session_id is not None:
            try:
                session = sessions.find_by_session_id(session_id=UUID(raw_session_id))
            except ValueError:
                session = None
            if session is not None and session.is_active_at(at=clock.now()):
                accepted = (
                    session.user_id not in {owner_user_id, secondary_user_id}
                    and len(organizations.list_accesses_for_user(user_id=session.user_id)) == 1
                )
        return {"invitation_provisioned": accepted}

    return app


def _seed_local_owner(
    *,
    repository: InMemoryLocalAuthRepository,
    owner_user_id: UserId,
    local_login_credential: str,
    now: datetime,
) -> None:
    ticket_id = repository.issue_bootstrap_ticket(
        token_sha256=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    challenge = repository.create_challenge(
        purpose="bootstrap",
        challenge_sha256=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        user_id=owner_user_id,
        context={},
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    repository.complete_bootstrap(
        challenge_id=challenge.challenge_id,
        ticket_id=ticket_id,
        user_id=owner_user_id,
        username="owner",
        display_name="Fixture owner",
        password_hash=PasswordHasher().hash(local_login_credential),
        installation_name="Roehub browser proof",
        organization_slug="proof",
        organization_name="Proof organization",
        passkey=LocalPasskey(
            credential_id="fixture-local-passkey",
            user_id=owner_user_id,
            public_key=b"fixture-not-used",
            sign_count=0,
            transports=("internal",),
            created_at=now,
        ),
        recovery_code_hashes=(),
        completed_at=now,
    )


def _identity_for_mode(mode: str) -> tuple[str, str]:
    if mode == "invited":
        return "invited-subject", _INVITED_EMAIL
    if mode in {"linked", "token_unknown"}:
        return "linked-subject", _LINK_EMAIL
    return "uninvited-subject", "uninvited@oidc-proof.invalid"


def _signed_identity(
    *, signing_key: rsa.RSAPrivateKey, key_id: str, grant: _Grant
) -> str:
    now = datetime.now(UTC)
    header = _segment({"alg": "RS256", "kid": key_id, "typ": "JWT"})
    claims = _segment(
        {
            "iss": _ISSUER,
            "sub": grant.subject,
            "aud": _CLIENT_ID,
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "iat": int(now.timestamp()),
            "nonce": grant.nonce,
            "email": grant.email,
            "email_verified": True,
        }
    )
    signing_input = f"{header}.{claims}".encode("ascii")
    signature = signing_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{claims}.{_b64(signature)}"


def _jwk(
    *, signing_key: rsa.RSAPrivateKey, key_id: str
) -> dict[str, str]:
    numbers = signing_key.public_key().public_numbers()
    return {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": key_id,
        "n": _b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": _b64(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }


def _pkce_challenge(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return _b64(digest)


def _segment(value: dict[str, object]) -> str:
    return _b64(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_credential() -> str:
    value = os.environ.get(_CREDENTIAL_ENV, "").strip()
    if len(value) < 32:
        raise RuntimeError(f"{_CREDENTIAL_ENV} must contain a disposable high-entropy value")
    return value


def _required_local_login_credential() -> str:
    value = os.environ.get(_LOCAL_LOGIN_CREDENTIAL_ENV, "").strip()
    if len(value) < 32:
        raise RuntimeError(
            f"{_LOCAL_LOGIN_CREDENTIAL_ENV} must contain a disposable high-entropy value"
        )
    return value
