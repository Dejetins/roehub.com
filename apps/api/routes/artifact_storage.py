"""Browser owner checks precede the independently authenticated privileged capability."""

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from apps.api.routes.backtest_artifacts import (
    BoundedArtifactRoute,
    artifact_error,
    require_mutation,
)
from trading.contexts.backtest_artifacts.application.services.management import ManagementError
from trading.contexts.identity.application.use_cases import OrganizationAccessError
from trading.contexts.operations import ControlOperationError
from trading.contexts.operations.artifact_storage_contracts import (
    StorageCommand,
    StorageEnvelope,
    StorageResult,
)
from trading.shared_kernel.primitives import OrganizationId


def build_artifact_storage_router(*, current_user_dependency, organization_service, client):
    router = APIRouter(
        prefix="/api/v1/admin/organizations/{organization_id}/backtest-artifacts/storage",
        tags=["artifact-storage"],
        route_class=BoundedArtifactRoute,
    )

    def authorize(principal, org):
        try:
            access = organization_service.get_access(
                principal=principal, organization_id=OrganizationId(org)
            )
            if "operations.execute" not in access.permissions:
                raise HTTPException(403, {"code": "forbidden"})
            organization_service.require_installation_control(principal=principal)
        except OrganizationAccessError as error:
            raise HTTPException(403, {"code": "installation_forbidden"}) from error
        if client is None:
            raise HTTPException(503, {"code": "control_unavailable"}, headers={"Retry-After": "30"})

    def execute(org, principal, body, key):
        try:
            return client.artifact_storage(
                StorageEnvelope(
                    organization_id=org,
                    actor_id=principal.user_id.value,
                    idempotency_key=key,
                    command=body,
                )
            )
        except ControlOperationError as error:
            reason = error.code.removeprefix("storage.")
            if reason == "unavailable" or error.code.startswith("control_agent."):
                raise HTTPException(
                    503, {"code": "control_unavailable"}, headers={"Retry-After": "30"}
                ) from error
            if reason in {"preview_stale", "cleanup_busy", "cleanup_current", "cleanup_pinned"}:
                raise HTTPException(409, {"code": reason}) from error
            if reason in {
                "untrusted_root",
                "storage_access_unavailable",
                "disk_full",
                "storage_inventory_unavailable",
            }:
                raise HTTPException(503, {"code": reason}, headers={"Retry-After": "30"}) from error
            raise artifact_error(ManagementError(reason)) from error

    @router.get("", response_model=StorageResult)
    def inspect(
        organization_id: UUID,
        expected_version: int = Query(default=0, ge=0),
        root_id: str = Query(default="backtest-primary", max_length=64),
        principal=Depends(current_user_dependency),
    ):
        authorize(principal, organization_id)
        return execute(
            organization_id,
            principal,
            StorageCommand(action="inspect", expected_version=expected_version, root_id=root_id),
            "inspect-" + uuid4().hex,
        )

    @router.post("/validate", response_model=StorageResult)
    @router.put("", response_model=StorageResult)
    @router.post("/cleanup:preview", response_model=StorageResult)
    @router.post("/cleanup:confirm", response_model=StorageResult)
    def mutate(
        organization_id: UUID,
        body: StorageCommand,
        request: Request,
        key: str = Header(alias="Idempotency-Key"),
        principal=Depends(current_user_dependency),
    ):
        require_mutation(request, key)
        authorize(principal, organization_id)
        expected = {
            "validate": "validate",
            "cleanup:preview": "cleanup_preview",
            "cleanup:confirm": "cleanup_confirm",
            "storage": "configure",
        }
        if body.action != expected[request.url.path.rsplit("/", 1)[-1]]:
            raise HTTPException(422, {"code": "action_mismatch"})
        return execute(organization_id, principal, body, key)

    return router
