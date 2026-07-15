"""Secret-free API fixture linked to the real Stage 20 operational-health HTTP service."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import Depends, FastAPI, Request

from apps.api.common import register_api_error_handlers
from apps.api.operational_health_client import HttpOperationalHealthClient
from apps.api.routes.admin import build_admin_router
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import OrganizationAccessService
from trading.contexts.operations import (
    OperationRequest,
    OperationResult,
    OperationState,
)
from trading.shared_kernel.primitives import PaidLevel, UserId

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW


class _PluginService:
    def list_inventory(self, **_kwargs: object) -> tuple[tuple[object, ...], ...]:
        return (), (), (), ()


class _HarmlessControlClient:
    def __init__(self) -> None:
        self.submitted: list[OperationRequest] = []

    def submit(self, request: OperationRequest) -> OperationResult:
        self.submitted.append(request)
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.ACCEPTED,
            detail_code="operation.accepted",
        )

    def get(self, operation_id: UUID) -> OperationResult:
        return OperationResult(
            operation_id=operation_id,
            action=self.submitted[-1].action,
            profile=self.submitted[-1].profile,
            state=OperationState.SUCCEEDED,
            detail_code="operation.completed",
        )

    def reconcile(self, operation_id: UUID) -> OperationResult:
        return self.get(operation_id)


repository = InMemoryOrganizationRepository()
organization_service = OrganizationAccessService(repository=repository)
owner = CurrentUserPrincipal(
    user_id=UserId(UUID("20000000-0000-0000-0000-000000000020")),
    paid_level=PaidLevel.free(),
    session_created_at=NOW,
)
installation, organization = organization_service.bootstrap_installation(
    principal=owner,
    installation_name="Stage 20 Runtime",
    organization_slug="stage20-runtime",
    organization_name="Stage 20 Runtime",
    now=NOW,
)
principals: dict[str, CurrentUserPrincipal] = {"owner": owner}
for index, role in enumerate(("admin", "operator", "trader", "viewer"), start=1):
    principal = CurrentUserPrincipal(
        user_id=UserId(UUID(f"20000000-0000-0000-0000-{index:012d}")),
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


def current_user(request: Request) -> CurrentUserPrincipal:
    role = request.cookies.get("qa_role", "owner")
    principal = principals.get(role, principals["viewer"])
    if request.cookies.get("qa_recent", "1") == "1":
        return principal
    return CurrentUserPrincipal(
        user_id=principal.user_id,
        paid_level=principal.paid_level,
        session_created_at=NOW - timedelta(minutes=11),
    )


operational_url = os.environ["ROEHUB_OPERATIONAL_HEALTH_URL"]
control_client = _HarmlessControlClient()
app = FastAPI(title="Stage 20 real-chain API fixture")
register_api_error_handlers(app=app)
app.include_router(
    build_admin_router(
        organization_service=organization_service,
        plugin_service=_PluginService(),  # type: ignore[arg-type]
        current_user_dependency=current_user,  # type: ignore[arg-type]
        clock=_Clock(),
        control_agent_client=control_client,  # type: ignore[arg-type]
        operational_health_client=HttpOperationalHealthClient(base_url=operational_url),
    )
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/api/v1/organizations")
def list_organizations(
    principal: CurrentUserPrincipal = Depends(current_user),
) -> list[dict[str, object]]:
    access = organization_service.list_organizations(principal=principal)[0]
    return [
        {
            "organization": {
                "organization_id": str(access.organization.organization_id.value),
                "installation_id": str(installation.installation_id.value),
                "slug": access.organization.slug,
                "display_name": access.organization.display_name,
                "status": access.organization.status,
                "created_at": access.organization.created_at.isoformat(),
            },
            "role": access.role,
            "permissions": sorted(access.permissions),
        }
    ]


@app.get("/__qa/operations")
def submitted_operations() -> list[dict[str, object]]:
    return [
        {
            "operation_id": str(item.operation_id),
            "action": item.action.value,
            "profile": item.profile,
            "services": list(item.services),
        }
        for item in control_client.submitted
    ]
