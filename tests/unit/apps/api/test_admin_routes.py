from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4, uuid5

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.admin import build_admin_router
from apps.monitoring.operational_health import OperationalSnapshot, OperationalStatus
from trading.contexts.extensions.domain import PluginEvent
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import OrganizationAccessService
from trading.contexts.operations import (
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
)
from trading.shared_kernel.primitives import (
    InstallationId,
    OrganizationId,
    PaidLevel,
    UserId,
)

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW


class _PluginService:
    def list_inventory(self, **kwargs: object) -> tuple[tuple[object, ...], ...]:
        organization_id = kwargs["organization_id"]
        assert isinstance(organization_id, OrganizationId)
        event = PluginEvent(
            event_id=uuid4(),
            installation_id=InstallationId(uuid4()),
            organization_id=organization_id,
            actor_user_id=UserId(uuid4()),
            event_type="plugin.update.requested",
            target_type="plugin",
            target_id="fixture.plugin",
            outcome="succeeded",
            metadata={},
            created_at=NOW,
        )
        return (), (), (), (event,)


class _ControlClient:
    def __init__(self) -> None:
        self.submitted: list[UUID] = []

    def submit(self, request: OperationRequest) -> OperationResult:
        operation_id = request.operation_id
        self.submitted.append(operation_id)
        return OperationResult(
            operation_id=operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.ACCEPTED,
            detail_code="operation.accepted",
        )

    def get(self, operation_id: UUID) -> OperationResult:
        return OperationResult(
            operation_id=operation_id,
            action=OperationAction.RESTART,
            profile="base",
            state=OperationState.UNKNOWN,
            detail_code="operation.state_unknown",
        )

    def reconcile(self, operation_id: UUID) -> OperationResult:
        return OperationResult(
            operation_id=operation_id,
            action=OperationAction.RESTART,
            profile="base",
            state=OperationState.SUCCEEDED,
            detail_code="operation.reconciled",
        )


class _OperationalHealthClient:
    def __init__(
        self,
        *,
        state: str = "stopped",
        action_ref: str = "restart_service",
        service_id: str = "api",
    ) -> None:
        self.state = state
        self.action_ref = action_ref
        self.service_id = service_id

    def snapshot(self) -> OperationalSnapshot:
        status = OperationalStatus(
            service_id=self.service_id,
            capability="product.web_api",
            state=self.state,  # type: ignore[arg-type]
            detail_code=f"probe.{self.state}",
            runbook_id="web.api-health-degraded",
            action_ref=self.action_ref,
            required=True,
            observed_at=NOW,
        )
        return OperationalSnapshot(
            profile="base",
            generated_at=NOW,
            overall_state=self.state,  # type: ignore[arg-type]
            services=(status,),
        )


def _build_client() -> tuple[
    TestClient,
    dict[str, CurrentUserPrincipal],
    dict[str, CurrentUserPrincipal],
    OrganizationId,
    _ControlClient,
    _OperationalHealthClient,
]:
    owner = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel.free(),
        session_created_at=NOW,
    )
    repository = InMemoryOrganizationRepository()
    service = OrganizationAccessService(repository=repository)
    _, organization = service.bootstrap_installation(
        principal=owner,
        installation_name="Roehub Site",
        organization_slug="primary-org",
        organization_name="Primary Organization",
        now=NOW,
    )
    principals = {"owner": owner}
    for role in ("admin", "operator", "trader", "viewer"):
        principal = CurrentUserPrincipal(
            user_id=UserId(uuid4()),
            paid_level=PaidLevel.free(),
            session_created_at=NOW,
        )
        repository.add_membership(
            organization_id=organization.organization_id,
            user_id=principal.user_id,
            role=role,  # type: ignore[arg-type]
            actor_user_id=owner.user_id,
            created_at=NOW,
        )
        principals[role] = principal
    current = {"principal": owner}

    def current_user() -> CurrentUserPrincipal:
        return current["principal"]

    control = _ControlClient()
    operational_health = _OperationalHealthClient()
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_admin_router(
            organization_service=service,
            plugin_service=_PluginService(),  # type: ignore[arg-type]
            current_user_dependency=current_user,  # type: ignore[arg-type]
            clock=_Clock(),
            control_agent_client=control,  # type: ignore[arg-type]
            operational_health_client=operational_health,
        )
    )
    client = TestClient(app)
    return (
        client,
        principals,
        current,
        organization.organization_id,
        control,
        operational_health,
    )


def test_admin_snapshot_is_redacted_and_readable_by_every_role() -> None:
    client, principals, current, organization_id, _, _ = _build_client()

    for role, principal in principals.items():
        current["principal"] = principal
        response = client.get(
            f"/api/v1/admin/organizations/{organization_id}/snapshot"
        )
        assert response.status_code == 200, role
        payload = response.json()
        assert payload["schema"] == "io.roehub.admin-snapshot/v1alpha1"
        assert payload["role"] == role
        assert payload["installation_owner"] is (role == "owner")
        assert bool(payload["events"]) is (role in {"owner", "admin", "operator"})
        assert payload["operational_health"]["overall_state"] == "stopped"
        assert payload["operational_health"]["services"][0]["runbook_path"] == (
            "/runbooks/web.api-health-degraded"
        )
        serialized = response.text.lower()
        for forbidden in ("password", "secret", "credential", "authorization"):
            assert forbidden not in serialized


def test_admin_operations_enforce_role_recent_auth_origin_and_idempotency() -> None:
    client, principals, current, organization_id, control, health = _build_client()
    url = f"/api/v1/admin/organizations/{organization_id}/operations"
    headers = {
        "Origin": "http://testserver",
        "Idempotency-Key": "admin-restart-0001",
    }

    for role in ("owner", "admin", "operator"):
        current["principal"] = principals[role]
        response = client.post(
            url,
            json={"action": "restart", "profile": "base", "services": ["api"]},
            headers=headers,
        )
        assert response.status_code == 202, role

    assert control.submitted[0] == control.submitted[1] == control.submitted[2]

    for role in ("trader", "viewer"):
        current["principal"] = principals[role]
        response = client.post(
            url,
            json={"action": "restart", "profile": "base", "services": ["api"]},
            headers=headers,
        )
        assert response.status_code == 403, role

    current["principal"] = CurrentUserPrincipal(
        user_id=principals["admin"].user_id,
        paid_level=PaidLevel.free(),
        session_created_at=NOW - timedelta(minutes=11),
    )
    stale = client.post(
        url,
        json={"action": "restart", "profile": "base", "services": ["api"]},
        headers=headers,
    )
    assert stale.status_code == 403
    assert stale.json()["error"]["code"] == "recent_auth_required"

    current["principal"] = principals["owner"]
    missing_origin = client.post(
        url,
        json={"action": "restart", "profile": "base", "services": ["api"]},
        headers={"Idempotency-Key": "admin-restart-0002"},
    )
    assert missing_origin.status_code == 403
    assert missing_origin.json()["error"]["code"] == "admin.csrf_required"

    current["principal"] = principals["admin"]
    release_forbidden = client.post(
        url,
        json={
            "action": "update",
            "profile": "base",
            "release_version": "0.1.1",
        },
        headers={**headers, "Idempotency-Key": "admin-update-0001"},
    )
    assert release_forbidden.status_code == 403
    assert release_forbidden.json()["error"]["code"] == "installation_owner_required"

    current["principal"] = principals["owner"]
    release_allowed = client.post(
        url,
        json={
            "action": "update",
            "profile": "base",
            "release_version": "0.1.1",
        },
        headers={**headers, "Idempotency-Key": "admin-update-0001"},
    )
    assert release_allowed.status_code == 202

    for state in ("ready", "degraded", "unknown"):
        health.state = state
        rejected = client.post(
            url,
            json={"action": "restart", "profile": "base", "services": ["api"]},
            headers={**headers, "Idempotency-Key": f"admin-restart-{state}-0001"},
        )
        assert rejected.status_code == 409
        assert rejected.json()["error"]["code"] == "admin.operation_not_allowlisted"

    health.state = "stopped"
    health.service_id = "postgresql"
    health.action_ref = "diagnostics"
    stateful_rejected = client.post(
        url,
        json={"action": "restart", "profile": "base", "services": ["postgresql"]},
        headers={**headers, "Idempotency-Key": "admin-restart-postgresql-0001"},
    )
    assert stateful_rejected.status_code == 409
    assert stateful_rejected.json()["error"]["code"] == (
        "admin.operation_not_allowlisted"
    )


def test_admin_unknown_operation_can_be_reconciled_without_runtime_details() -> None:
    client, principals, current, organization_id, _, _ = _build_client()
    current["principal"] = principals["operator"]
    idempotency_key = "admin-restart-reconcile-0001"
    operation_id = uuid5(
        organization_id.value,
        f"admin-operation:{idempotency_key}",
    )
    operation_url = (
        f"/api/v1/admin/organizations/{organization_id}/operations/{operation_id}"
    )

    unknown = client.get(
        operation_url,
        headers={"Idempotency-Key": idempotency_key},
    )
    assert unknown.status_code == 200
    assert unknown.json()["state"] == "unknown"
    assert set(unknown.json()) == {
        "schema",
        "operation_id",
        "action",
        "profile",
        "state",
        "detail_code",
        "active_services",
        "journal_sequence",
    }

    reconciled = client.post(
        f"{operation_url}:reconcile",
        headers={
            "Origin": "http://testserver",
            "Idempotency-Key": idempotency_key,
        },
    )
    assert reconciled.status_code == 200
    assert reconciled.json()["state"] == "succeeded"


def test_admin_operation_read_is_bound_to_organization_namespace() -> None:
    client, principals, current, organization_id, _, _ = _build_client()
    current["principal"] = principals["operator"]
    idempotency_key = "admin-restart-foreign-0001"
    foreign_operation_id = uuid5(
        uuid4(),
        f"admin-operation:{idempotency_key}",
    )

    response = client.get(
        f"/api/v1/admin/organizations/{organization_id}/operations/"
        f"{foreign_operation_id}",
        headers={"Idempotency-Key": idempotency_key},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "admin.operation_not_found"
