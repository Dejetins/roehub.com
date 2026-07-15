from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from trading.contexts.identity.adapters.inbound.api.routes.organizations import (
    build_organizations_router,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import OrganizationAccessService
from trading.shared_kernel.primitives import PaidLevel, UserId

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def now(self) -> datetime:
        return NOW


def test_versioned_organization_api_uses_authenticated_principal_scope() -> None:
    owner = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel.free(),
        session_created_at=NOW,
    )

    current = {"principal": owner}

    def current_user() -> CurrentUserPrincipal:
        return current["principal"]

    app = FastAPI()
    register_api_error_handlers(app=app)
    service = OrganizationAccessService(repository=InMemoryOrganizationRepository())
    app.include_router(
        build_organizations_router(
            service=service,
            current_user_dependency=current_user,  # type: ignore[arg-type]
            clock=_Clock(),
        )
    )
    client = TestClient(app)
    mutation_headers = {"Origin": "http://testserver"}

    bootstrap = client.post(
        "/api/v1/installations/bootstrap",
        json={
            "installation_name": "Roehub Site",
            "organization_slug": "primary-org",
            "organization_name": "Primary Organization",
        },
        headers=mutation_headers,
    )
    assert bootstrap.status_code == 201
    organization_id = bootstrap.json()["organization"]["organization_id"]

    listed = client.get("/api/v1/organizations")
    assert listed.status_code == 200
    assert listed.json()[0]["organization"]["organization_id"] == organization_id
    assert "mainnet.approve" in listed.json()[0]["permissions"]

    invited = client.post(
        f"/api/v1/organizations/{organization_id}/invitations",
        json={
            "recipient_email": "member@example.invalid",
            "role": "viewer",
            "expires_at": "2026-07-14T12:00:00Z",
        },
        headers=mutation_headers,
    )
    assert invited.status_code == 201
    assert "recipient_email" not in invited.json()

    admin = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel.free(),
        session_created_at=NOW,
    )
    added = client.post(
        f"/api/v1/organizations/{organization_id}/members",
        json={"user_id": str(admin.user_id), "role": "admin"},
        headers=mutation_headers,
    )
    assert added.status_code == 201
    current["principal"] = admin
    rejected = client.patch(
        f"/api/v1/organizations/{organization_id}/members/{admin.user_id}",
        json={"role": "owner"},
        headers=mutation_headers,
    )
    assert rejected.status_code == 403
    assert rejected.json()["error"]["code"] == "owner_role_required"
    current["principal"] = owner
    audit = client.get(f"/api/v1/organizations/{organization_id}/audit")
    assert audit.status_code == 200
    rejected_events = [event for event in audit.json() if event["outcome"] == "rejected"]
    assert rejected_events[0]["metadata"] == {"reason_code": "owner_role_required"}

    duplicate = client.post(
        "/api/v1/installations/bootstrap",
        json={
            "installation_name": "Other Site",
            "organization_slug": "other-org",
            "organization_name": "Other Organization",
        },
        headers=mutation_headers,
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "installation_already_initialized"


def test_organization_router_requires_all_dependencies() -> None:
    service = OrganizationAccessService(repository=InMemoryOrganizationRepository())
    try:
        build_organizations_router(
            service=service,
            current_user_dependency=None,  # type: ignore[arg-type]
            clock=_Clock(),
        )
    except ValueError as error:
        assert "current_user_dependency" in str(error)
    else:
        raise AssertionError("missing current-user dependency must fail closed")
