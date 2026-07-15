from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid5

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from apps.api.control_agent_client import ApiControlAgentClient
from apps.api.operational_health_client import (
    OperationalHealthClient,
    OperationalHealthClientError,
)
from trading.contexts.extensions.application import PluginLifecycleError, PluginLifecycleService
from trading.contexts.extensions.domain import (
    PluginEvent,
    PluginInstallation,
    PluginInstance,
    PluginOperation,
)
from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports import (
    CurrentUserPrincipal,
    IdentityClock,
)
from trading.contexts.identity.application.use_cases import (
    OrganizationAccessError,
    OrganizationAccessService,
)
from trading.contexts.identity.domain.entities import (
    AdministrativeAuditEvent,
    OrganizationMembership,
)
from trading.contexts.operations import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
    OperationResult,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId

_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_RECENT_AUTH_WINDOW = timedelta(minutes=10)
_INSTALLATION_ACTIONS = frozenset(
    {
        OperationAction.RECOVER,
        OperationAction.INSTALL,
        OperationAction.UPDATE,
        OperationAction.ROLLBACK,
        OperationAction.BACKUP,
        OperationAction.RESTORE,
    }
)


class AdminMemberResponse(BaseModel):
    user_id: UUID
    role: Literal["owner", "admin", "operator", "trader", "viewer"]
    status: Literal["active", "suspended"]
    created_at: datetime


class AdminPluginInstallationResponse(BaseModel):
    plugin_installation_id: UUID
    plugin_id: str
    package_id: UUID
    rollback_available: bool
    granted_permissions: list[str]
    status: Literal["enabled", "disabled", "degraded"]
    updated_at: datetime


class AdminPluginInstanceResponse(BaseModel):
    instance_id: UUID
    plugin_installation_id: UUID
    name: str
    config_revision: int
    status: Literal["enabled", "disabled", "degraded"]
    updated_at: datetime


class AdminPluginOperationResponse(BaseModel):
    operation_id: UUID
    kind: str
    target_id: str
    status: Literal["pending", "running", "succeeded", "failed", "unknown"]
    created_at: datetime
    updated_at: datetime


class AdminEventResponse(BaseModel):
    event_id: UUID
    category: Literal["identity", "plugin"]
    action: str
    target_type: str
    target_id: str
    outcome: Literal["succeeded", "rejected"]
    created_at: datetime


class AdminCapabilityResponse(BaseModel):
    providers: Literal["available"] = "available"
    backups: Literal["deferred"] = "deferred"
    updates: Literal["ready", "degraded"]
    services: Literal["ready", "degraded"]
    observability: Literal["ready", "degraded"]


class AdminOperationalServiceResponse(BaseModel):
    service_id: str
    capability: str
    state: Literal["ready", "degraded", "stopped", "unknown"]
    detail_code: str
    runbook_id: str
    runbook_path: str
    action_ref: str
    observed_at: datetime


class AdminOperationalHealthResponse(BaseModel):
    schema_id: Literal["io.roehub.admin-operational-health/v1alpha1"] = Field(
        default="io.roehub.admin-operational-health/v1alpha1",
        alias="schema",
    )
    profile: Literal["base", "trading", "ml", "unknown"]
    overall_state: Literal["ready", "degraded", "stopped", "unknown"]
    generated_at: datetime
    grafana_path: str | None = None
    services: list[AdminOperationalServiceResponse]


class AdminSnapshotResponse(BaseModel):
    schema_id: Literal["io.roehub.admin-snapshot/v1alpha1"] = Field(
        default="io.roehub.admin-snapshot/v1alpha1",
        alias="schema",
    )
    organization_id: UUID
    organization_name: str
    role: Literal["owner", "admin", "operator", "trader", "viewer"]
    permissions: list[str]
    recent_auth: bool
    installation_owner: bool
    members: list[AdminMemberResponse]
    plugin_installations: list[AdminPluginInstallationResponse]
    plugin_instances: list[AdminPluginInstanceResponse]
    plugin_operations: list[AdminPluginOperationResponse]
    events: list[AdminEventResponse]
    capabilities: AdminCapabilityResponse
    operational_health: AdminOperationalHealthResponse


class AdminOperationRequest(BaseModel):
    action: Literal[
        "diagnostics",
        "start",
        "stop",
        "restart",
        "recover",
        "install",
        "update",
        "rollback",
        "backup",
        "restore",
    ]
    profile: Literal["base", "trading", "ml"] = "base"
    services: list[str] = Field(default_factory=list, max_length=32)
    release_version: str | None = Field(
        default=None,
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$",
    )
    subject_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$",
    )


def build_admin_router(
    *,
    organization_service: OrganizationAccessService,
    plugin_service: PluginLifecycleService,
    current_user_dependency: RequireCurrentUserDependency,
    clock: IdentityClock,
    control_agent_client: ApiControlAgentClient | None,
    operational_health_client: OperationalHealthClient | None = None,
) -> APIRouter:
    """Build the browser-safe administrative API composition boundary."""

    router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

    @router.get(
        "/organizations/{organization_id}/snapshot",
        response_model=AdminSnapshotResponse,
    )
    def get_snapshot(
        organization_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> AdminSnapshotResponse:
        scoped_id = OrganizationId(organization_id)
        try:
            access = organization_service.get_access(
                principal=principal,
                organization_id=scoped_id,
            )
            members = organization_service.list_members(
                principal=principal,
                organization_id=scoped_id,
            )
            installations, instances, operations, plugin_events = (
                plugin_service.list_inventory(
                    principal=principal,
                    organization_id=scoped_id,
                )
            )
            identity_events = (
                organization_service.list_audit_events(
                    principal=principal,
                    organization_id=scoped_id,
                    limit=100,
                )
                if "audit.read" in access.permissions
                else ()
            )
        except (OrganizationAccessError, PermissionError) as error:
            code = error.code if isinstance(error, OrganizationAccessError) else "admin.forbidden"
            raise RoehubError(
                code=code,
                message="Administrative snapshot access is forbidden",
            ) from error
        except PluginLifecycleError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        visible_plugin_events = (
            plugin_events if "audit.read" in access.permissions else ()
        )
        events = [
            *_identity_events(identity_events),
            *_plugin_events(visible_plugin_events),
        ]
        events.sort(key=lambda event: (event.created_at, str(event.event_id)), reverse=True)
        control_ready = (
            control_agent_client is not None
            and "operations.execute" in access.permissions
        )
        operational_health = _operational_health_response(
            client=operational_health_client,
            now=clock.now(),
        )
        return AdminSnapshotResponse(
            organization_id=organization_id,
            organization_name=access.organization.display_name,
            role=access.role,
            permissions=sorted(access.permissions),
            recent_auth=_is_recent(principal=principal, now=clock.now()),
            installation_owner=organization_service.is_installation_owner(
                principal=principal
            ),
            members=[_member_response(member) for member in members],
            plugin_installations=[
                _plugin_installation_response(installation)
                for installation in installations
            ],
            plugin_instances=[_plugin_instance_response(instance) for instance in instances],
            plugin_operations=[
                _plugin_operation_response(operation) for operation in operations
            ],
            events=events[:100],
            capabilities=AdminCapabilityResponse(
                updates="ready" if control_ready else "degraded",
                services="ready" if control_ready else "degraded",
                observability=(
                    "ready"
                    if operational_health.overall_state == "ready"
                    else "degraded"
                ),
            ),
            operational_health=operational_health,
        )

    @router.post(
        "/organizations/{organization_id}/operations",
        response_model=OperationResult,
        status_code=202,
    )
    def submit_operation(
        organization_id: UUID,
        payload: AdminOperationRequest,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> OperationResult:
        _enforce_same_origin_mutation(request=request)
        scoped_id = OrganizationId(organization_id)
        action = OperationAction(payload.action)
        try:
            organization_service.require_operation_execute(
                principal=principal,
                organization_id=scoped_id,
                now=clock.now(),
            )
            if action in _INSTALLATION_ACTIONS:
                organization_service.require_installation_control(
                    principal=principal,
                )
        except OrganizationAccessError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        _enforce_operational_action(
            action=action,
            services=tuple(payload.services),
            client=operational_health_client,
        )
        client = _require_control_client(control_agent_client)
        operation_id = _operation_id(
            organization_id=organization_id,
            idempotency_key=idempotency_key,
        )
        try:
            operation = OperationRequest(
                operation_id=operation_id,
                action=action,
                profile=payload.profile,
                services=tuple(payload.services),
                release_version=payload.release_version,
                subject_id=payload.subject_id,
            )
            return client.submit(operation)
        except (ControlOperationError, ValueError) as error:
            if isinstance(error, ControlOperationError):
                raise _admin_control_error(
                    error=error,
                    message="Administrative operation failed",
                ) from error
            raise RoehubError(
                code="admin.operation_invalid",
                message="Administrative operation failed",
            ) from error

    @router.get(
        "/organizations/{organization_id}/operations/{operation_id}",
        response_model=OperationResult,
    )
    def get_operation(
        organization_id: UUID,
        operation_id: UUID,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> OperationResult:
        _require_operation_read(
            organization_service=organization_service,
            principal=principal,
            organization_id=organization_id,
        )
        _require_operation_scope(
            organization_id=organization_id,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
        )
        client = _require_control_client(control_agent_client)
        try:
            return client.get(operation_id)
        except ControlOperationError as error:
            raise _admin_control_error(
                error=error,
                message="Operation state is unavailable",
            ) from error

    @router.post(
        "/organizations/{organization_id}/operations/{operation_id}:reconcile",
        response_model=OperationResult,
    )
    def reconcile_operation(
        organization_id: UUID,
        operation_id: UUID,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> OperationResult:
        _enforce_same_origin_mutation(request=request)
        _require_operation_read(
            organization_service=organization_service,
            principal=principal,
            organization_id=organization_id,
        )
        _require_operation_scope(
            organization_id=organization_id,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
        )
        client = _require_control_client(control_agent_client)
        try:
            return client.reconcile(operation_id)
        except ControlOperationError as error:
            raise _admin_control_error(
                error=error,
                message="Operation reconciliation failed",
            ) from error

    return router


def _enforce_operational_action(
    *,
    action: OperationAction,
    services: tuple[str, ...],
    client: OperationalHealthClient | None,
) -> None:
    if action == OperationAction.DIAGNOSTICS:
        return
    if action != OperationAction.RESTART or not services or client is None:
        if services or action == OperationAction.RESTART:
            raise RoehubError(
                code="admin.operation_not_allowlisted",
                message="Administrative operation is not allowed for this service state",
            )
        return
    try:
        snapshot = client.snapshot()
    except OperationalHealthClientError as error:
        raise RoehubError(
            code="admin.operation_not_allowlisted",
            message="Administrative operation is not allowed for this service state",
        ) from error
    by_service = {status.service_id: status for status in snapshot.services}
    if any(
        (status := by_service.get(service)) is None
        or status.state != "stopped"
        or status.action_ref != "restart_service"
        for service in services
    ):
        raise RoehubError(
            code="admin.operation_not_allowlisted",
            message="Administrative operation is not allowed for this service state",
        )


def _operational_health_response(
    *,
    client: OperationalHealthClient | None,
    now: datetime,
) -> AdminOperationalHealthResponse:
    if client is None:
        return AdminOperationalHealthResponse(
            profile="unknown",
            overall_state="unknown",
            generated_at=now,
            services=[],
        )
    try:
        snapshot = client.snapshot()
    except OperationalHealthClientError:
        return AdminOperationalHealthResponse(
            profile="unknown",
            overall_state="unknown",
            generated_at=now,
            services=[],
        )
    return AdminOperationalHealthResponse(
        profile=snapshot.profile,
        overall_state=snapshot.overall_state,
        generated_at=snapshot.generated_at,
        services=[
            AdminOperationalServiceResponse(
                service_id=item.service_id,
                capability=item.capability,
                state=item.state,
                detail_code=item.detail_code,
                runbook_id=item.runbook_id,
                runbook_path=f"/runbooks/{item.runbook_id}",
                action_ref=item.action_ref,
                observed_at=item.observed_at,
            )
            for item in snapshot.services
        ],
    )


def _require_operation_read(
    *,
    organization_service: OrganizationAccessService,
    principal: CurrentUserPrincipal,
    organization_id: UUID,
) -> None:
    try:
        organization_service.require_operation_read(
            principal=principal,
            organization_id=OrganizationId(organization_id),
        )
    except OrganizationAccessError as error:
        raise RoehubError(code=error.code, message=error.message) from error


def _require_control_client(
    client: ApiControlAgentClient | None,
) -> ApiControlAgentClient:
    if client is None:
        raise RoehubError(
            code="admin.control_agent_unavailable",
            message="Control agent is unavailable",
        )
    return client


def _operation_id(*, organization_id: UUID, idempotency_key: str) -> UUID:
    if _IDEMPOTENCY_KEY.fullmatch(idempotency_key) is None:
        raise RoehubError(
            code="admin.idempotency_key_invalid",
            message="Idempotency-Key is invalid",
        )
    return uuid5(organization_id, f"admin-operation:{idempotency_key}")


def _require_operation_scope(
    *,
    organization_id: UUID,
    operation_id: UUID,
    idempotency_key: str,
) -> None:
    expected = _operation_id(
        organization_id=organization_id,
        idempotency_key=idempotency_key,
    )
    if operation_id != expected:
        raise RoehubError(
            code="admin.operation_not_found",
            message="Administrative operation is not found",
        )


def _admin_control_error(
    *,
    error: ControlOperationError,
    message: str,
) -> RoehubError:
    if error.code == "operation.not_found":
        code = "admin.operation_not_found"
    elif error.code == "operation.idempotency_conflict":
        code = "admin.idempotency_conflict"
    else:
        code = "admin.control_agent_unavailable"
    return RoehubError(code=code, message=message)


def _enforce_same_origin_mutation(*, request: Request) -> None:
    reason = same_origin_rejection_reason(
        request=request,
        fail_closed_without_origin=True,
    )
    if reason is not None:
        raise RoehubError(
            code="admin.csrf_required",
            message="Administrative mutation origin is not allowed",
            details={"reason": reason},
        )


def _is_recent(*, principal: CurrentUserPrincipal, now: datetime) -> bool:
    authenticated_at = principal.session_created_at
    if authenticated_at is None:
        return False
    normalized_now = now.astimezone(UTC)
    normalized_auth = authenticated_at.astimezone(UTC)
    return (
        normalized_auth <= normalized_now
        and normalized_now - normalized_auth <= _RECENT_AUTH_WINDOW
    )


def _member_response(member: OrganizationMembership) -> AdminMemberResponse:
    return AdminMemberResponse(
        user_id=member.user_id.value,
        role=member.role,
        status=member.status,
        created_at=member.created_at,
    )


def _plugin_installation_response(
    installation: PluginInstallation,
) -> AdminPluginInstallationResponse:
    return AdminPluginInstallationResponse(
        plugin_installation_id=installation.plugin_installation_id,
        plugin_id=installation.plugin_id,
        package_id=installation.package_id,
        rollback_available=installation.previous_package_id is not None,
        granted_permissions=list(installation.granted_permissions),
        status=installation.status,
        updated_at=installation.updated_at,
    )


def _plugin_instance_response(instance: PluginInstance) -> AdminPluginInstanceResponse:
    return AdminPluginInstanceResponse(
        instance_id=instance.instance_id,
        plugin_installation_id=instance.plugin_installation_id,
        name=instance.name,
        config_revision=instance.config_revision,
        status=instance.status,
        updated_at=instance.updated_at,
    )


def _plugin_operation_response(
    operation: PluginOperation,
) -> AdminPluginOperationResponse:
    return AdminPluginOperationResponse(
        operation_id=operation.operation_id,
        kind=operation.kind,
        target_id=operation.target_id,
        status=operation.status,
        created_at=operation.created_at,
        updated_at=operation.updated_at,
    )


def _identity_events(
    events: tuple[AdministrativeAuditEvent, ...],
) -> list[AdminEventResponse]:
    return [
        AdminEventResponse(
            event_id=event.event_id,
            category="identity",
            action=event.action,
            target_type=event.target_type,
            target_id=event.target_id,
            outcome=event.outcome,
            created_at=event.created_at,
        )
        for event in events
    ]


def _plugin_events(events: tuple[PluginEvent, ...]) -> list[AdminEventResponse]:
    return [
        AdminEventResponse(
            event_id=event.event_id,
            category="plugin",
            action=event.event_type,
            target_type=event.target_type,
            target_id=event.target_id,
            outcome=event.outcome,
            created_at=event.created_at,
        )
        for event in events
    ]


__all__ = ["AdminSnapshotResponse", "build_admin_router"]
