from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from apps.api.wiring.modules.identity import _resolve_identity_runtime_settings
from tools.qa.backtests_client_fixture import preview_identity_environment
from trading.contexts.identity.adapters.outbound.persistence.in_memory.session_repository import (
    InMemoryIdentitySessionRepository,
)
from trading.shared_kernel.primitives import UserId


def test_preview_session_survives_short_timeouts_and_remains_revocable() -> None:
    settings = _resolve_identity_runtime_settings(
        environ=preview_identity_environment({"ROEHUB_ENV": "test"})
    )
    repository = InMemoryIdentitySessionRepository()
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)
    session = repository.create_session(
        user_id=UserId(uuid4()),
        now=now,
        idle_ttl_seconds=settings.identity_session_idle_ttl_seconds,
        absolute_ttl_seconds=settings.identity_session_absolute_ttl_seconds,
    )
    for age in (timedelta(minutes=31), timedelta(hours=13), timedelta(days=30)):
        assert session.is_active_at(at=now + age)
    assert session.absolute_expires_at == now + timedelta(days=365)
    revoked = repository.revoke_session(session_id=session.session_id, revoked_at=now)
    assert revoked is not None
    assert not revoked.is_active_at(at=now + timedelta(seconds=1))


def test_preview_policy_preserves_explicit_test_timeouts_and_input() -> None:
    environ = {
        "ROEHUB_ENV": "test",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "60",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "120",
    }
    assert preview_identity_environment(environ) == environ
    assert preview_identity_environment(environ) is not environ


def test_preview_policy_cannot_be_used_for_production() -> None:
    with pytest.raises(ValueError, match="local-only"):
        preview_identity_environment({"ROEHUB_ENV": "prod"})
