from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports import CurrentUserPrincipal, IdentityClock
from trading.contexts.identity.application.use_cases.organizations import (
    OrganizationAccessError,
    OrganizationAccessService,
)
from trading.contexts.identity.domain.entities import (
    AdministrativeAuditEvent,
    Organization,
    OrganizationAccess,
    OrganizationInvitation,
    OrganizationMembership,
    PluginPermissionGrant,
    SupportAccessGrant,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, UserId

Role = Literal["owner", "admin", "operator", "trader", "viewer"]
PluginPermission = Literal["read", "configure", "operate"]


class BootstrapInstallationRequest(BaseModel):
    installation_name: str = Field(min_length=2, max_length=120)
    organization_slug: str = Field(min_length=3, max_length=64)
    organization_name: str = Field(min_length=2, max_length=120)


class CreateOrganizationRequest(BaseModel):
    slug: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=2, max_length=120)


class MembershipRequest(BaseModel):
    user_id: UUID
    role: Role


class ChangeMembershipRoleRequest(BaseModel):
    role: Role


class InvitationRequest(BaseModel):
    recipient_email: str = Field(min_length=3, max_length=320)
    role: Role
    expires_at: datetime


class PluginPermissionRequest(BaseModel):
    permission: PluginPermission


class SupportAccessRequest(BaseModel):
    support_user_id: UUID
    reason: str = Field(min_length=8, max_length=240)
    expires_at: datetime


class OrganizationResponse(BaseModel):
    organization_id: UUID
    installation_id: UUID
    slug: str
    display_name: str
    status: Literal["active", "archived"]
    created_at: datetime


class BootstrapInstallationResponse(BaseModel):
    installation_id: UUID
    organization: OrganizationResponse


class OrganizationAccessResponse(BaseModel):
    organization: OrganizationResponse
    role: Role
    permissions: list[str]


class MembershipResponse(BaseModel):
    organization_id: UUID
    user_id: UUID
    role: Role
    status: Literal["active", "suspended"]
    created_at: datetime


class InvitationResponse(BaseModel):
    invitation_id: UUID
    organization_id: UUID
    role: Role
    expires_at: datetime
    created_at: datetime


class PluginPermissionResponse(BaseModel):
    organization_id: UUID
    plugin_id: str
    user_id: UUID
    permission: PluginPermission
    updated_at: datetime


class SupportAccessResponse(BaseModel):
    grant_id: UUID
    installation_id: UUID
    support_user_id: UUID
    expires_at: datetime
    created_at: datetime


class AuditEventResponse(BaseModel):
    event_id: UUID
    organization_id: UUID | None
    actor_user_id: UUID
    action: str
    target_type: str
    target_id: str
    outcome: Literal["succeeded", "rejected"]
    metadata: dict[str, str]
    created_at: datetime


def build_organizations_router(
    *,
    service: OrganizationAccessService,
    current_user_dependency: RequireCurrentUserDependency,
    clock: IdentityClock,
) -> APIRouter:
    """Build versioned installation and organization administration routes."""

    if service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_organizations_router requires service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_organizations_router requires current_user_dependency")
    if clock is None:  # type: ignore[truthy-bool]
        raise ValueError("build_organizations_router requires clock")
    router = APIRouter(prefix="/api/v1", tags=["organizations"])

    @router.post(
        "/installations/bootstrap",
        response_model=BootstrapInstallationResponse,
        status_code=201,
    )
    def bootstrap_installation(
        request: BootstrapInstallationRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BootstrapInstallationResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            installation, organization = service.bootstrap_installation(
                principal=principal,
                installation_name=request.installation_name,
                organization_slug=request.organization_slug,
                organization_name=request.organization_name,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=None,
                action="installation.bootstrap",
                target_type="installation",
                target_id="singleton",
                error=error,
                now=clock.now(),
            ) from error
        return BootstrapInstallationResponse(
            installation_id=installation.installation_id.value,
            organization=_organization_response(organization),
        )

    @router.get("/organizations", response_model=list[OrganizationAccessResponse])
    def list_organizations(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> list[OrganizationAccessResponse]:
        return [
            _organization_access_response(access)
            for access in service.list_organizations(principal=principal)
        ]

    @router.post("/organizations", response_model=OrganizationResponse, status_code=201)
    def create_organization(
        request: CreateOrganizationRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> OrganizationResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            organization = service.create_organization(
                principal=principal,
                slug=request.slug,
                display_name=request.display_name,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=None,
                action="organization.create",
                target_type="organization",
                target_id=request.slug,
                error=error,
                now=clock.now(),
            ) from error
        return _organization_response(organization)

    @router.post(
        "/organizations/{organization_id}/members",
        response_model=MembershipResponse,
        status_code=201,
    )
    def add_member(
        organization_id: UUID,
        request: MembershipRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MembershipResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            membership = service.add_member(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                user_id=UserId(request.user_id),
                role=request.role,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="membership.create",
                target_type="membership",
                target_id=str(request.user_id),
                error=error,
                now=clock.now(),
            ) from error
        return _membership_response(membership)

    @router.patch(
        "/organizations/{organization_id}/members/{user_id}",
        response_model=MembershipResponse,
    )
    def change_member_role(
        organization_id: UUID,
        user_id: UUID,
        request: ChangeMembershipRoleRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MembershipResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            membership = service.change_member_role(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                user_id=UserId(user_id),
                role=request.role,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="membership.role_change",
                target_type="membership",
                target_id=str(user_id),
                error=error,
                now=clock.now(),
            ) from error
        return _membership_response(membership)

    @router.delete(
        "/organizations/{organization_id}/members/{user_id}",
        status_code=204,
    )
    def remove_member(
        organization_id: UUID,
        user_id: UUID,
        request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> Response:
        _enforce_same_origin_mutation(request=request)
        try:
            service.remove_member(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                user_id=UserId(user_id),
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="membership.remove",
                target_type="membership",
                target_id=str(user_id),
                error=error,
                now=clock.now(),
            ) from error
        return Response(status_code=204)

    @router.post(
        "/organizations/{organization_id}/invitations",
        response_model=InvitationResponse,
        status_code=201,
    )
    def create_invitation(
        organization_id: UUID,
        request: InvitationRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> InvitationResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            invitation = service.create_invitation(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                recipient_email=request.recipient_email,
                role=request.role,
                expires_at=request.expires_at,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="invitation.create",
                target_type="invitation",
                target_id="new",
                error=error,
                now=clock.now(),
            ) from error
        return _invitation_response(invitation)

    @router.put(
        "/organizations/{organization_id}/plugins/{plugin_id}/permissions/{user_id}",
        response_model=PluginPermissionResponse,
    )
    def set_plugin_permission(
        organization_id: UUID,
        plugin_id: str,
        user_id: UUID,
        request: PluginPermissionRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> PluginPermissionResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            grant = service.set_plugin_permission(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                plugin_id=plugin_id,
                user_id=UserId(user_id),
                permission=request.permission,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="plugin.permission_set",
                target_type="plugin_permission",
                target_id=f"{plugin_id}:{user_id}",
                error=error,
                now=clock.now(),
            ) from error
        return _plugin_permission_response(grant)

    @router.post(
        "/installations/support-access",
        response_model=SupportAccessResponse,
        status_code=201,
    )
    def grant_support_access(
        request: SupportAccessRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> SupportAccessResponse:
        _enforce_same_origin_mutation(request=http_request)
        try:
            grant = service.grant_support_access(
                principal=principal,
                support_user_id=UserId(request.support_user_id),
                reason=request.reason,
                expires_at=request.expires_at,
                now=clock.now(),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=None,
                action="support_access.grant",
                target_type="support_access",
                target_id=str(request.support_user_id),
                error=error,
                now=clock.now(),
            ) from error
        return _support_access_response(grant)

    @router.get(
        "/organizations/{organization_id}/members",
        response_model=list[MembershipResponse],
    )
    def list_members(
        organization_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> list[MembershipResponse]:
        try:
            memberships = service.list_members(
                principal=principal,
                organization_id=OrganizationId(organization_id),
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="membership.read",
                target_type="membership",
                target_id=str(organization_id),
                error=error,
                now=clock.now(),
            ) from error
        return [_membership_response(membership) for membership in memberships]

    @router.get(
        "/organizations/{organization_id}/audit",
        response_model=list[AuditEventResponse],
    )
    def list_audit_events(
        organization_id: UUID,
        limit: int = 100,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> list[AuditEventResponse]:
        try:
            events = service.list_audit_events(
                principal=principal,
                organization_id=OrganizationId(organization_id),
                limit=limit,
            )
        except OrganizationAccessError as error:
            raise _audited_api_error(
                service=service,
                principal=principal,
                organization_id=OrganizationId(organization_id),
                action="audit.read",
                target_type="audit",
                target_id=str(organization_id),
                error=error,
                now=clock.now(),
            ) from error
        return [_audit_response(event) for event in events]

    return router


def _enforce_same_origin_mutation(*, request: Request) -> None:
    rejection_reason = same_origin_rejection_reason(
        request=request,
        fail_closed_without_origin=True,
    )
    if rejection_reason is not None:
        raise RoehubError(
            code="identity.csrf_required",
            message="Administrative mutation origin is not allowed",
            details={"reason": rejection_reason},
        )


def _api_error(error: OrganizationAccessError) -> RoehubError:
    return RoehubError(code=error.code, message=error.message)


def _audited_api_error(
    *,
    service: OrganizationAccessService,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId | None,
    action: str,
    target_type: str,
    target_id: str,
    error: OrganizationAccessError,
    now: datetime,
) -> RoehubError:
    service.record_rejected_operation(
        principal=principal,
        organization_id=organization_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        reason_code=error.code,
        now=now,
    )
    return _api_error(error)


def _organization_response(value: Organization) -> OrganizationResponse:
    return OrganizationResponse(
        organization_id=value.organization_id.value,
        installation_id=value.installation_id.value,
        slug=value.slug,
        display_name=value.display_name,
        status=value.status,
        created_at=value.created_at,
    )


def _organization_access_response(value: OrganizationAccess) -> OrganizationAccessResponse:
    return OrganizationAccessResponse(
        organization=_organization_response(value.organization),
        role=value.role,
        permissions=sorted(value.permissions),
    )


def _membership_response(value: OrganizationMembership) -> MembershipResponse:
    return MembershipResponse(
        organization_id=value.organization_id.value,
        user_id=value.user_id.value,
        role=value.role,
        status=value.status,
        created_at=value.created_at,
    )


def _invitation_response(value: OrganizationInvitation) -> InvitationResponse:
    return InvitationResponse(
        invitation_id=value.invitation_id,
        organization_id=value.organization_id.value,
        role=value.role,
        expires_at=value.expires_at,
        created_at=value.created_at,
    )


def _plugin_permission_response(value: PluginPermissionGrant) -> PluginPermissionResponse:
    return PluginPermissionResponse(
        organization_id=value.organization_id.value,
        plugin_id=value.plugin_id,
        user_id=value.user_id.value,
        permission=value.permission,
        updated_at=value.updated_at,
    )


def _support_access_response(value: SupportAccessGrant) -> SupportAccessResponse:
    return SupportAccessResponse(
        grant_id=value.grant_id,
        installation_id=value.installation_id.value,
        support_user_id=value.support_user_id.value,
        expires_at=value.expires_at,
        created_at=value.created_at,
    )


def _audit_response(value: AdministrativeAuditEvent) -> AuditEventResponse:
    return AuditEventResponse(
        event_id=value.event_id,
        organization_id=(None if value.organization_id is None else value.organization_id.value),
        actor_user_id=value.actor_user_id.value,
        action=value.action,
        target_type=value.target_type,
        target_id=value.target_id,
        outcome=value.outcome,
        metadata=dict(value.metadata),
        created_at=value.created_at,
    )
