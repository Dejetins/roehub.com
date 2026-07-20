from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast
from uuid import uuid4

import pytest

from trading.contexts.identity.application.authorization import (
    AuthorizationDenyReason,
    AuthorizationRequest,
    AuthorizationResource,
    AuthorizationScope,
    CapabilityAuthorizationService,
    CapabilityId,
)
from trading.contexts.identity.application.authorization.policy import CAPABILITY_POLICIES
from trading.contexts.identity.application.ports import CurrentUserPrincipal, OrganizationRepository
from trading.contexts.identity.domain.entities import OrganizationMembership, OrganizationRole
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


@dataclass
class _AuthorizationRepository:
    memberships: dict[tuple[OrganizationId, UserId], OrganizationMembership]
    installation_owners: set[UserId]

    def get_membership(
        self, *, organization_id: OrganizationId, user_id: UserId
    ) -> OrganizationMembership | None:
        return self.memberships.get((organization_id, user_id))

    def is_installation_owner(self, *, user_id: UserId) -> bool:
        return user_id in self.installation_owners


def _principal() -> CurrentUserPrincipal:
    return CurrentUserPrincipal(user_id=UserId(uuid4()), paid_level=PaidLevel.free())


def _service(
    *,
    actor: CurrentUserPrincipal,
    organization_id: OrganizationId,
    role: OrganizationRole = "owner",
    membership_status: Literal["active", "suspended"] = "active",
    installation_owner: bool = False,
) -> CapabilityAuthorizationService:
    repository = _AuthorizationRepository(
        memberships={
            (organization_id, actor.user_id): OrganizationMembership(
                organization_id=organization_id,
                user_id=actor.user_id,
                role=role,
                status=membership_status,
                created_at=NOW,
            )
        },
        installation_owners={actor.user_id} if installation_owner else set(),
    )
    return CapabilityAuthorizationService(
        organization_repository=cast(OrganizationRepository, repository)
    )


def test_static_policy_covers_every_stable_capability_identifier() -> None:
    assert set(CAPABILITY_POLICIES) == set(CapabilityId)


@pytest.mark.parametrize(
    ("role", "capability", "resource_owner", "expected_scope"),
    [
        ("owner", CapabilityId.STRATEGIES_MANAGE, None, AuthorizationScope.ORGANIZATION),
        ("admin", CapabilityId.CONNECTIONS_MANAGE, None, AuthorizationScope.ORGANIZATION),
        (
            "operator",
            CapabilityId.STRATEGIES_SAFE_STOP,
            None,
            AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
        ),
        ("trader", CapabilityId.STRATEGIES_MANAGE, "actor", AuthorizationScope.OWN),
        ("viewer", CapabilityId.DASHBOARD_READ, None, AuthorizationScope.SERVER_FILTERED_READ),
    ],
)
def test_persisted_role_matrix_decides_each_role_family(
    role: OrganizationRole,
    capability: CapabilityId,
    resource_owner: str | None,
    expected_scope: AuthorizationScope,
) -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(actor=actor, organization_id=organization_id, role=role)
    resource = (
        None
        if resource_owner is None
        else AuthorizationResource(organization_id=organization_id, owner_user_id=actor.user_id)
    )

    decision = service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=capability,
            selected_organization_id=organization_id,
            resource=resource,
        )
    )

    assert decision.allowed is True
    assert decision.scope is expected_scope
    assert decision.deny_reason is None


def test_client_role_and_missing_organization_context_deny() -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(actor=actor, organization_id=organization_id)

    client_role = service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=CapabilityId.STRATEGIES_MANAGE,
            selected_organization_id=organization_id,
            client_supplied_role="owner",
        )
    )
    missing_context = service.decide(
        request=AuthorizationRequest(actor=actor, capability=CapabilityId.DASHBOARD_READ)
    )

    assert client_role.deny_reason is AuthorizationDenyReason.CLIENT_ROLE_SUPPLIED
    assert missing_context.deny_reason is AuthorizationDenyReason.ORGANIZATION_CONTEXT_REQUIRED


def test_unknown_capability_and_inactive_membership_default_to_deny() -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    unknown_service = _service(actor=actor, organization_id=organization_id)
    inactive_service = _service(
        actor=actor,
        organization_id=organization_id,
        membership_status="suspended",
    )

    unknown = unknown_service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability="unrecognized.capability",
            selected_organization_id=organization_id,
        )
    )
    inactive = inactive_service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=CapabilityId.DASHBOARD_READ,
            selected_organization_id=organization_id,
        )
    )

    assert unknown.allowed is False
    assert unknown.deny_reason is AuthorizationDenyReason.UNKNOWN_CAPABILITY
    assert inactive.deny_reason is AuthorizationDenyReason.INACTIVE_OR_MISSING_MEMBERSHIP


def test_cross_organization_and_other_owned_resource_deny() -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(actor=actor, organization_id=organization_id, role="trader")

    cross_organization = service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=CapabilityId.STRATEGIES_MANAGE,
            selected_organization_id=organization_id,
            resource=AuthorizationResource(organization_id=OrganizationId(uuid4())),
        )
    )
    another_owner = service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=CapabilityId.STRATEGIES_MANAGE,
            selected_organization_id=organization_id,
            resource=AuthorizationResource(
                organization_id=organization_id,
                owner_user_id=UserId(uuid4()),
            ),
        )
    )

    assert cross_organization.deny_reason is AuthorizationDenyReason.RESOURCE_ORGANIZATION_MISMATCH
    assert another_owner.deny_reason is AuthorizationDenyReason.OWNERSHIP_REQUIRED


def test_installation_owner_overlay_is_independent_from_organization_membership() -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    installation_owner = _service(
        actor=actor,
        organization_id=organization_id,
        installation_owner=True,
    )
    organization_owner = _service(actor=actor, organization_id=organization_id)

    permitted = installation_owner.decide(
        request=AuthorizationRequest(actor=actor, capability=CapabilityId.INSTALLATION_TRUST_MANAGE)
    )
    rejected = organization_owner.decide(
        request=AuthorizationRequest(actor=actor, capability=CapabilityId.INSTALLATION_TRUST_MANAGE)
    )

    assert permitted.allowed is True
    assert permitted.scope is AuthorizationScope.INSTALLATION
    assert rejected.deny_reason is AuthorizationDenyReason.INSTALLATION_OWNER_REQUIRED


def test_stored_secret_reveal_is_denied_for_every_actor() -> None:
    actor = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(
        actor=actor,
        organization_id=organization_id,
        installation_owner=True,
    )

    decision = service.decide(
        request=AuthorizationRequest(
            actor=actor,
            capability=CapabilityId.CONNECTIONS_SECRET_REVEAL,
        )
    )

    assert decision.allowed is False
    assert decision.deny_reason is AuthorizationDenyReason.STORED_SECRET_REVEAL_FORBIDDEN
