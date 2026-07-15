"""Isolated real-browser fixture for the local-auth acceptance boundary."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

from fastapi import FastAPI

from apps.api.routes.identity import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.outbound import (
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    InMemoryLocalAuthRepository,
    InMemoryOrganizationRepository,
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application.use_cases import LocalAuthService

_BOOTSTRAP_FILE_ENV = "ROEHUB_LOCAL_AUTH_BOOTSTRAP_FILE"
_COOKIE_NAME = "roehub_session_id"
_ORIGIN = "http://localhost:8000"


class _ProofClock:
    """Thread-safe mutable UTC clock used only to prove deterministic expiry."""

    def __init__(self) -> None:
        self._value = datetime.now(UTC)
        self._lock = RLock()

    def now(self) -> datetime:
        with self._lock:
            return self._value

    def expire_all_sessions(self) -> None:
        with self._lock:
            self._value += timedelta(days=1)


def create_app() -> FastAPI:
    """Create an in-memory API that exercises production local-auth adapters."""
    bootstrap_path = _resolve_bootstrap_path()
    clock = _ProofClock()
    users = InMemoryIdentityUserRepository()
    sessions = InMemoryIdentitySessionRepository()
    organizations = InMemoryOrganizationRepository()
    local_repository = InMemoryLocalAuthRepository(
        user_repository=users,
        organization_repository=organizations,
    )
    local_service = LocalAuthService(
        repository=local_repository,
        user_repository=users,
        session_repository=sessions,
        clock=clock,
        rp_id="localhost",
        rp_name="Roehub browser proof",
        expected_origin=_ORIGIN,
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

    _write_exclusive_bootstrap_file(
        path=bootstrap_path,
        value=local_service.issue_bootstrap_ticket(),
    )

    app = FastAPI(title="Roehub local-auth browser proof")
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
        )
    )

    @app.post("/__proof__/expire-sessions", include_in_schema=False)
    def expire_sessions() -> dict[str, str]:
        clock.expire_all_sessions()
        return {"status": "expired"}

    return app


def _resolve_bootstrap_path() -> Path:
    raw_path = os.environ.get(_BOOTSTRAP_FILE_ENV, "").strip()
    if not raw_path:
        raise RuntimeError(f"{_BOOTSTRAP_FILE_ENV} is required")
    return Path(raw_path).expanduser().resolve()


def _write_exclusive_bootstrap_file(*, path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, value.encode("utf-8"))
    finally:
        os.close(descriptor)
