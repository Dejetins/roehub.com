"""Actual identity use-case and HTTP guards; transport absence is explicit."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.artifact_storage import build_artifact_storage_router
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import OrganizationAccessService
from trading.contexts.operations import ControlOperationError
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_storage_requires_owner_origin_and_closed_dto_without_recent_login():
    now = datetime.now(UTC)
    owner = CurrentUserPrincipal(
        user_id=UserId(uuid4()), paid_level=PaidLevel.free(), session_created_at=now
    )
    repo = InMemoryOrganizationRepository()
    service = OrganizationAccessService(repository=repo)
    _, org = service.bootstrap_installation(
        principal=owner,
        installation_name="Test",
        organization_slug="test-org",
        organization_name="Test org",
        now=now,
    )
    principals = {"owner": owner}
    for role in ("owner", "admin", "operator", "viewer"):
        p = replace(owner, user_id=UserId(uuid4()))
        repo.add_membership(
            organization_id=org.organization_id,
            user_id=p.user_id,
            role=role,
            actor_user_id=owner.user_id,
            created_at=now,
        )
        principals["org_" + role] = p
    current = {"user": owner}

    class Unavailable:
        calls = 0

        def artifact_storage(self, envelope):
            self.calls += 1
            raise ControlOperationError(code="control_agent.unavailable")

    transport = Unavailable()
    app = FastAPI()
    app.include_router(
        build_artifact_storage_router(
            current_user_dependency=lambda: current["user"],
            organization_service=service,
            client=transport,
        )
    )
    client = TestClient(app)
    url = f"/api/v1/admin/organizations/{org.organization_id.value}/backtest-artifacts/storage"
    body = {"action": "validate", "expected_version": 0}
    headers = {"Origin": "http://testserver", "Idempotency-Key": "storage-key-001"}
    for role in ("org_owner", "org_admin", "org_operator", "org_viewer"):
        current["user"] = principals[role]
        assert client.get(url).status_code == 403
        assert client.post(url + "/validate", json=body, headers=headers).status_code == 403
    assert transport.calls == 0
    current["user"] = replace(owner, session_created_at=now - timedelta(hours=1))
    # An older authenticated owner session reaches the privileged transport.
    assert client.post(url + "/validate", json=body, headers=headers).status_code == 503
    assert transport.calls == 1
    current["user"] = replace(owner, session_created_at=None)
    assert client.post(url + "/validate", json=body, headers=headers).status_code == 503
    assert transport.calls == 2
    current["user"] = owner
    assert (
        client.post(
            url + "/validate", json=body, headers={"Idempotency-Key": "storage-key-001"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            url + "/validate", json=dict(body, path="/tmp/evil"), headers=headers
        ).status_code
        == 422
    )
    response = client.post(url + "/validate", json=body, headers=headers)
    assert response.status_code == 503 and response.headers["Retry-After"] == "30"
    assert response.json() == {"detail": {"code": "control_unavailable"}}
    assert client.get(url).status_code == 503
    assert client.post(url + "/validate", content="x" * 1048577, headers=headers).status_code == 413
