from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import (
    OrganizationAccessError,
    OrganizationAccessService,
)
from trading.contexts.identity.domain.entities import permissions_for_role
from trading.shared_kernel.primitives import PaidLevel, UserId

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _principal(*, minutes_ago: int = 0) -> CurrentUserPrincipal:
    return CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel.free(),
        session_created_at=NOW - timedelta(minutes=minutes_ago),
    )


def _bootstrap() -> tuple[
    OrganizationAccessService,
    InMemoryOrganizationRepository,
    CurrentUserPrincipal,
    object,
]:
    repository = InMemoryOrganizationRepository()
    service = OrganizationAccessService(repository=repository)
    owner = _principal()
    _installation, organization = service.bootstrap_installation(
        principal=owner,
        installation_name="Roehub Site",
        organization_slug="primary-org",
        organization_name="Primary Organization",
        now=NOW,
    )
    return service, repository, owner, organization


def test_role_matrix_keeps_owner_only_mainnet_and_admin_management() -> None:
    assert "mainnet.approve" in permissions_for_role("owner")
    assert "mainnet.approve" not in permissions_for_role("admin")
    assert "plugins.manage" in permissions_for_role("admin")
    assert "roles.manage" in permissions_for_role("admin")
    assert "members.manage" not in permissions_for_role("operator")
    assert permissions_for_role("viewer") == frozenset(
        {"organization.read", "members.read", "plugins.read"}
    )


def test_bootstrap_requires_recent_auth_and_is_singleton() -> None:
    repository = InMemoryOrganizationRepository()
    service = OrganizationAccessService(repository=repository)
    stale = _principal(minutes_ago=11)

    with pytest.raises(OrganizationAccessError, match="Recent authentication") as raised:
        service.bootstrap_installation(
            principal=stale,
            installation_name="Roehub Site",
            organization_slug="primary-org",
            organization_name="Primary Organization",
            now=NOW,
        )
    assert raised.value.code == "recent_auth_required"

    owner = _principal()
    service.bootstrap_installation(
        principal=owner,
        installation_name="Roehub Site",
        organization_slug="primary-org",
        organization_name="Primary Organization",
        now=NOW,
    )
    with pytest.raises(OrganizationAccessError) as duplicate:
        service.bootstrap_installation(
            principal=owner,
            installation_name="Other Site",
            organization_slug="other-org",
            organization_name="Other Organization",
            now=NOW,
        )
    assert duplicate.value.code == "installation_already_initialized"


def test_two_organization_scope_is_server_derived_and_cross_org_is_denied() -> None:
    service, _repository, owner, primary = _bootstrap()
    secondary = service.create_organization(
        principal=owner,
        slug="secondary-org",
        display_name="Secondary Organization",
        now=NOW,
    )
    primary_id = primary.organization_id  # type: ignore[attr-defined]
    secondary_id = secondary.organization_id
    primary_admin = _principal()
    service.add_member(
        principal=owner,
        organization_id=primary_id,
        user_id=primary_admin.user_id,
        role="admin",
        now=NOW,
    )

    with pytest.raises(OrganizationAccessError) as cross_org:
        service.add_member(
            principal=primary_admin,
            organization_id=secondary_id,
            user_id=_principal().user_id,
            role="viewer",
            now=NOW,
        )
    assert cross_org.value.code == "organization_forbidden"

    visible = service.list_organizations(principal=primary_admin)
    assert [access.organization.organization_id for access in visible] == [primary_id]

    secondary_owner = _principal()
    service.add_member(
        principal=owner,
        organization_id=secondary_id,
        user_id=secondary_owner.user_id,
        role="owner",
        now=NOW,
    )
    service.remove_member(
        principal=owner,
        organization_id=secondary_id,
        user_id=owner.user_id,
        now=NOW,
    )
    with pytest.raises(OrganizationAccessError) as installation_owner_escalation:
        service.add_member(
            principal=owner,
            organization_id=secondary_id,
            user_id=_principal().user_id,
            role="viewer",
            now=NOW,
        )
    assert installation_owner_escalation.value.code == "organization_forbidden"
    assert secondary_id not in {
        access.organization.organization_id
        for access in service.list_organizations(principal=owner)
    }


def test_admin_member_role_plugin_and_last_owner_invariants() -> None:
    service, _repository, owner, organization = _bootstrap()
    organization_id = organization.organization_id  # type: ignore[attr-defined]
    admin = _principal()
    member = _principal()
    service.add_member(
        principal=owner,
        organization_id=organization_id,
        user_id=admin.user_id,
        role="admin",
        now=NOW,
    )
    service.add_member(
        principal=admin,
        organization_id=organization_id,
        user_id=member.user_id,
        role="viewer",
        now=NOW,
    )
    changed = service.change_member_role(
        principal=admin,
        organization_id=organization_id,
        user_id=member.user_id,
        role="trader",
        now=NOW,
    )
    assert changed.role == "trader"
    grant = service.set_plugin_permission(
        principal=admin,
        organization_id=organization_id,
        plugin_id="roehub.sample-plugin",
        user_id=member.user_id,
        permission="operate",
        now=NOW,
    )
    assert grant.permission == "operate"

    with pytest.raises(OrganizationAccessError) as escalation:
        service.change_member_role(
            principal=admin,
            organization_id=organization_id,
            user_id=member.user_id,
            role="owner",
            now=NOW,
        )
    assert escalation.value.code == "owner_role_required"

    with pytest.raises(OrganizationAccessError) as admin_removes_owner:
        service.remove_member(
            principal=admin,
            organization_id=organization_id,
            user_id=owner.user_id,
            now=NOW,
        )
    assert admin_removes_owner.value.code == "owner_role_required"

    with pytest.raises(OrganizationAccessError) as last_owner:
        service.change_member_role(
            principal=owner,
            organization_id=organization_id,
            user_id=owner.user_id,
            role="admin",
            now=NOW,
        )
    assert last_owner.value.code == "last_owner"


def test_invitations_support_access_and_audit_never_expose_sensitive_payload() -> None:
    service, _repository, owner, organization = _bootstrap()
    organization_id = organization.organization_id  # type: ignore[attr-defined]
    support = _principal()
    invitation = service.create_invitation(
        principal=owner,
        organization_id=organization_id,
        recipient_email="operator@example.invalid",
        role="operator",
        expires_at=NOW + timedelta(days=1),
        now=NOW,
    )
    assert invitation.role == "operator"
    grant = service.grant_support_access(
        principal=owner,
        support_user_id=support.user_id,
        reason="Investigate installation health",
        expires_at=NOW + timedelta(hours=2),
        now=NOW,
    )
    assert grant.expires_at == NOW + timedelta(hours=2)

    events = service.list_audit_events(
        principal=owner,
        organization_id=organization_id,
        limit=100,
    )
    serialized = repr(events).lower()
    assert "operator@example.invalid" not in serialized
    assert "password" not in serialized
    assert "token" not in serialized
    assert {event.action for event in events} >= {
        "installation.bootstrap",
        "invitation.created",
    }

    with pytest.raises(OrganizationAccessError) as too_long:
        service.grant_support_access(
            principal=owner,
            support_user_id=support.user_id,
            reason="Investigate installation health",
            expires_at=NOW + timedelta(hours=25),
            now=NOW,
        )
    assert too_long.value.code == "invalid_support_access_expiry"
