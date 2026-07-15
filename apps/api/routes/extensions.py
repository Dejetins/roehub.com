from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from trading.contexts.extensions.application import (
    DataSourceQueryError,
    DataSourceQueryService,
    PluginBundleValidationError,
    PluginBundleValidator,
    PluginLifecycleError,
    PluginLifecycleService,
)
from trading.contexts.extensions.domain import PluginOperation, ValidatedPluginBundle
from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import DataSourceQueryRequest, RoehubDataFrame
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId


class ValidatePluginBundleRequest(BaseModel):
    bundle_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,127}$")


class ValidatedPluginBundleResponse(BaseModel):
    contract: str = "ValidatedPluginBundle/v1alpha1"
    plugin_id: str
    version: str
    package_digest: str
    image_digest: str
    publisher_key_id: str | None
    permissions: list[str]


class InstallPluginRequest(ValidatePluginBundleRequest):
    instance_name: str = Field(min_length=1, max_length=120)
    permissions: list[str] = Field(max_length=32)
    config: dict[str, Any] = Field(default_factory=dict)


class PluginOperationResponse(BaseModel):
    contract: str = "PluginOperation/v1alpha1"
    operation_id: UUID
    organization_id: UUID
    kind: str
    target_id: str
    status: str
    result: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def build_extensions_router(
    *,
    service: PluginLifecycleService,
    validator: PluginBundleValidator,
    bundle_spool_root: Path,
    current_user_dependency: RequireCurrentUserDependency,
    data_source_service: DataSourceQueryService | None = None,
) -> APIRouter:
    """Build organization-scoped typed plugin management operations."""

    router = APIRouter(prefix="/api/v1", tags=["plugins"])

    if data_source_service is not None:

        @router.post(
            "/plugins/data-sources/{instance_id}:query",
            response_model=RoehubDataFrame,
        )
        async def query_data_source(
            instance_id: UUID,
            request: DataSourceQueryRequest,
            principal: CurrentUserPrincipal = Depends(current_user_dependency),
        ) -> RoehubDataFrame:
            try:
                return await data_source_service.query(
                    principal=principal,
                    instance_id=instance_id,
                    request=request,
                )
            except DataSourceQueryError as error:
                raise RoehubError(code=error.code, message=error.message) from error

    def validate_bundle(bundle_id: str) -> ValidatedPluginBundle:
        try:
            return validator.validate(bundle_spool_root / bundle_id)
        except PluginBundleValidationError as error:
            raise RoehubError(code=error.code, message=error.message) from error

    @router.post(
        "/organizations/{organization_id}/plugins/bundles:validate",
        response_model=ValidatedPluginBundleResponse,
    )
    def validate_plugin_bundle(
        organization_id: UUID,
        request: ValidatePluginBundleRequest,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ValidatedPluginBundleResponse:
        try:
            service.require_manage(
                principal=principal,
                organization_id=OrganizationId(organization_id),
            )
        except PermissionError as error:
            raise RoehubError(
                code="plugin.forbidden", message="Plugin access is forbidden"
            ) from error
        bundle = validate_bundle(request.bundle_id)
        return ValidatedPluginBundleResponse(
            plugin_id=bundle.manifest.plugin_id,
            version=bundle.manifest.version,
            package_digest=bundle.manifest.package_digest,
            image_digest=bundle.manifest.image_digest,
            publisher_key_id=bundle.manifest.publisher_key_id,
            permissions=list(bundle.manifest.permissions),
        )

    @router.post(
        "/organizations/{organization_id}/plugins/installations",
        response_model=PluginOperationResponse,
        status_code=202,
    )
    def install_or_update_plugin(
        organization_id: UUID,
        request: InstallPluginRequest,
        http_request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> PluginOperationResponse:
        _enforce_same_origin_mutation(request=http_request)
        bundle = validate_bundle(request.bundle_id)
        try:
            operation = service.submit_install_or_update(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                bundle=bundle,
                requested_permissions=tuple(request.permissions),
                instance_name=request.instance_name,
                config=request.config,
                idempotency_key=idempotency_key,
                now=datetime.now(UTC),
            )
        except PermissionError as error:
            raise RoehubError(
                code="plugin.forbidden", message="Plugin access is forbidden"
            ) from error
        except PluginLifecycleError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        return _operation_response(operation)

    @router.post(
        "/organizations/{organization_id}/plugins/installations/{plugin_id}:rollback",
        response_model=PluginOperationResponse,
        status_code=202,
    )
    def rollback_plugin(
        organization_id: UUID,
        plugin_id: str,
        http_request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> PluginOperationResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            operation = service.submit_rollback(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                plugin_id=plugin_id,
                idempotency_key=idempotency_key,
                now=datetime.now(UTC),
            )
        except PermissionError as error:
            raise RoehubError(
                code="plugin.forbidden", message="Plugin access is forbidden"
            ) from error
        except PluginLifecycleError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        return _operation_response(operation)

    @router.get(
        "/organizations/{organization_id}/plugins/operations/{operation_id}",
        response_model=PluginOperationResponse,
    )
    def get_plugin_operation(
        organization_id: UUID,
        operation_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> PluginOperationResponse:
        try:
            operation = service.get_operation(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                operation_id=operation_id,
            )
        except PermissionError as error:
            raise RoehubError(
                code="plugin.forbidden", message="Plugin access is forbidden"
            ) from error
        except PluginLifecycleError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        return _operation_response(operation)

    return router


def _enforce_same_origin_mutation(*, request: Request) -> None:
    rejection_reason = same_origin_rejection_reason(
        request=request,
        fail_closed_without_origin=True,
    )
    if rejection_reason is not None:
        raise RoehubError(
            code="plugin.csrf_required",
            message="Plugin mutation origin is not allowed",
            details={"reason": rejection_reason},
        )


def _operation_response(operation: PluginOperation) -> PluginOperationResponse:
    return PluginOperationResponse(
        operation_id=operation.operation_id,
        organization_id=operation.organization_id.value,
        kind=operation.kind,
        target_id=operation.target_id,
        status=operation.status,
        result=dict(operation.result),
        created_at=operation.created_at,
        updated_at=operation.updated_at,
    )
