from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from trading.contexts.identity.adapters.outbound.persistence.in_memory.delegation_repository import (  # noqa: E501
    InMemoryDelegationRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory.organization_repository import (  # noqa: E501
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.authorization import (
    AuthorizationResource,
    CapabilityAuthorizationService,
    CapabilityId,
)
from trading.contexts.identity.application.delegation.models import (
    DELEGABLE_CAPABILITIES,
    DELEGABLE_CAPABILITY_GRANTEE_ROLES,
    DelegationAuthoritySource,
    DelegationDenyReason,
    DelegationEvaluationRequest,
    DelegationResourceScope,
    GrantDelegatedCapabilityCommand,
    RevokeDelegatedCapabilityCommand,
)
from trading.contexts.identity.application.delegation.service import (
    DelegatedCapabilityService,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.domain.entities import OrganizationRole
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
SCOPE = DelegationResourceScope.ORGANIZATION


@dataclass(frozen=True, slots=True)
class _DelegationFixture:
    organization_repository: InMemoryOrganizationRepository
    delegation_repository: InMemoryDelegationRepository
    service: DelegatedCapabilityService
    owner: CurrentUserPrincipal
    admin: CurrentUserPrincipal
    trader: CurrentUserPrincipal
    operator: CurrentUserPrincipal
    organization_id: OrganizationId


def _principal() -> CurrentUserPrincipal:
    return CurrentUserPrincipal(user_id=UserId(uuid4()), paid_level=PaidLevel.free())


def _fixture() -> _DelegationFixture:
    organization_repository = InMemoryOrganizationRepository()
    owner = _principal()
    admin = _principal()
    trader = _principal()
    operator = _principal()
    _, organization = organization_repository.bootstrap_installation(
        owner_user_id=owner.user_id,
        installation_name="Delegation test installation",
        organization_slug="delegation-test",
        organization_name="Delegation test organization",
        created_at=NOW,
    )
    members: tuple[tuple[CurrentUserPrincipal, OrganizationRole], ...] = (
        (admin, "admin"),
        (trader, "trader"),
        (operator, "operator"),
    )
    for principal, role in members:
        organization_repository.add_membership(
            organization_id=organization.organization_id,
            user_id=principal.user_id,
            role=role,
            actor_user_id=owner.user_id,
            created_at=NOW,
        )
    delegation_repository = InMemoryDelegationRepository()
    service = DelegatedCapabilityService(
        capability_authorization_service=CapabilityAuthorizationService(
            organization_repository=organization_repository
        ),
        organization_repository=organization_repository,
        delegation_repository=delegation_repository,
    )
    return _DelegationFixture(
        organization_repository=organization_repository,
        delegation_repository=delegation_repository,
        service=service,
        owner=owner,
        admin=admin,
        trader=trader,
        operator=operator,
        organization_id=organization.organization_id,
    )


def _grant_command(
    fixture: _DelegationFixture,
    *,
    grantee: CurrentUserPrincipal | None = None,
    capability: CapabilityId = CapabilityId.MODELS_MANAGE,
    resource_scope: DelegationResourceScope | str = SCOPE,
    requested_at: datetime = NOW,
    expires_at: datetime | None = None,
    recent_authentication_verified: bool = True,
) -> GrantDelegatedCapabilityCommand:
    return GrantDelegatedCapabilityCommand(
        actor=fixture.owner,
        organization_id=fixture.organization_id,
        grantee_user_id=(grantee or fixture.admin).user_id,
        capability=capability,
        resource_scope=resource_scope,
        expires_at=expires_at or requested_at + timedelta(hours=1),
        requested_at=requested_at,
        recent_authentication_verified=recent_authentication_verified,
    )


def _evaluation_request(
    fixture: _DelegationFixture,
    *,
    actor: CurrentUserPrincipal | None = None,
    organization_id: OrganizationId | None = None,
    resource: AuthorizationResource | None = None,
    resource_scope: DelegationResourceScope | str = SCOPE,
    evaluated_at: datetime = NOW,
) -> DelegationEvaluationRequest:
    return DelegationEvaluationRequest(
        actor=actor or fixture.admin,
        capability=CapabilityId.MODELS_MANAGE,
        selected_organization_id=organization_id or fixture.organization_id,
        resource=resource,
        resource_scope=resource_scope,
        evaluated_at=evaluated_at,
    )


def test_grant_expiry_revoke_and_authority_provenance_are_exact_and_redacted() -> None:
    fixture = _fixture()

    granted = fixture.service.grant(command=_grant_command(fixture))
    delegated = fixture.service.decide(request=_evaluation_request(fixture))
    direct = fixture.service.decide(request=_evaluation_request(fixture, actor=fixture.owner))
    expired = fixture.service.decide(
        request=_evaluation_request(fixture, evaluated_at=NOW + timedelta(hours=1))
    )

    assert granted.allowed is True
    assert granted.idempotent is False
    assert granted.grant is not None
    assert delegated.allowed is True
    assert delegated.authority_source is DelegationAuthoritySource.DELEGATION
    assert delegated.delegation == granted.grant
    assert direct.allowed is True
    assert direct.authority_source is DelegationAuthoritySource.CAPABILITY_KERNEL
    assert direct.delegation is None
    assert expired.allowed is False
    assert expired.delegation_deny_reason is DelegationDenyReason.DELEGATION_NOT_ACTIVE

    revoked = fixture.service.revoke(
        command=RevokeDelegatedCapabilityCommand(
            actor=fixture.owner,
            organization_id=fixture.organization_id,
            delegation_id=granted.grant.delegation_id,
            requested_at=NOW + timedelta(minutes=1),
            recent_authentication_verified=True,
        )
    )
    revoked_again = fixture.service.revoke(
        command=RevokeDelegatedCapabilityCommand(
            actor=fixture.owner,
            organization_id=fixture.organization_id,
            delegation_id=granted.grant.delegation_id,
            requested_at=NOW + timedelta(minutes=2),
            recent_authentication_verified=True,
        )
    )
    denied_after_revoke = fixture.service.decide(request=_evaluation_request(fixture))
    audit_events = fixture.delegation_repository.list_audit_events(
        organization_id=fixture.organization_id
    )

    assert revoked.allowed is True
    assert revoked.idempotent is False
    assert revoked_again.allowed is True
    assert revoked_again.idempotent is True
    assert denied_after_revoke.allowed is False
    assert denied_after_revoke.delegation_deny_reason is DelegationDenyReason.DELEGATION_NOT_ACTIVE
    assert [event.action for event in audit_events] == ["delegation.revoked", "delegation.granted"]
    expected_audit_keys = {"capability_id", "grantee_user_id", "resource_scope"}
    assert all(set(event.metadata) == expected_audit_keys for event in audit_events)
    assert all("secret" not in repr(event.metadata).lower() for event in audit_events)
    assert all(event.metadata["resource_scope"] == "organization" for event in audit_events)


def test_expired_grant_requires_explicit_revoke_before_a_new_grant() -> None:
    fixture = _fixture()
    first = fixture.service.grant(
        command=_grant_command(fixture, expires_at=NOW + timedelta(hours=1))
    )
    assert first.grant is not None

    renewal_without_revoke = fixture.service.grant(
        command=_grant_command(
            fixture,
            requested_at=NOW + timedelta(hours=1),
            expires_at=NOW + timedelta(hours=2),
        )
    )
    revocation = fixture.service.revoke(
        command=RevokeDelegatedCapabilityCommand(
            actor=fixture.owner,
            organization_id=fixture.organization_id,
            delegation_id=first.grant.delegation_id,
            requested_at=NOW + timedelta(hours=1),
            recent_authentication_verified=True,
        )
    )
    renewal_after_revoke = fixture.service.grant(
        command=_grant_command(
            fixture,
            requested_at=NOW + timedelta(hours=1),
            expires_at=NOW + timedelta(hours=2),
        )
    )

    assert renewal_without_revoke.deny_reason is DelegationDenyReason.GRANT_CONFLICT
    assert revocation.allowed is True
    assert renewal_after_revoke.allowed is True


def test_grant_rejects_self_grant_redelegation_nondelegable_and_ineligible_grantee() -> None:
    fixture = _fixture()

    self_grant = fixture.service.grant(command=_grant_command(fixture, grantee=fixture.owner))
    no_recent_auth = fixture.service.grant(
        command=_grant_command(fixture, recent_authentication_verified=False)
    )
    secret_capability = fixture.service.grant(
        command=_grant_command(
            fixture,
            capability=CapabilityId.CONNECTIONS_SECRET_REVEAL,
        )
    )
    operator_grantee = fixture.service.grant(
        command=_grant_command(fixture, grantee=fixture.operator)
    )
    unsafe_scope = fixture.service.grant(
        command=_grant_command(fixture, resource_scope="invalid-scope")
    )
    admin_redelegation = fixture.service.grant(
        command=GrantDelegatedCapabilityCommand(
            actor=fixture.admin,
            organization_id=fixture.organization_id,
            grantee_user_id=fixture.trader.user_id,
            capability=CapabilityId.MODELS_MANAGE,
            resource_scope=SCOPE,
            expires_at=NOW + timedelta(hours=1),
            requested_at=NOW,
            recent_authentication_verified=True,
        )
    )

    assert self_grant.deny_reason is DelegationDenyReason.SELF_GRANT_FORBIDDEN
    assert no_recent_auth.deny_reason is DelegationDenyReason.RECENT_AUTHENTICATION_REQUIRED
    assert secret_capability.deny_reason is DelegationDenyReason.NON_DELEGABLE_CAPABILITY
    assert operator_grantee.deny_reason is DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE
    assert unsafe_scope.deny_reason is DelegationDenyReason.INVALID_RESOURCE_SCOPE
    assert admin_redelegation.deny_reason is DelegationDenyReason.REDELEGATION_FORBIDDEN
    assert (
        fixture.delegation_repository.list_audit_events(organization_id=fixture.organization_id)
        == ()
    )


def test_exact_accepted_capability_to_grantee_role_matrix_is_enforced() -> None:
    fixture = _fixture()

    trader_data_selection = fixture.service.grant(
        command=_grant_command(
            fixture,
            grantee=fixture.trader,
            capability=CapabilityId.DATA_SELECTION_MANAGE,
        )
    )
    admin_data_selection = fixture.service.grant(
        command=_grant_command(
            fixture,
            capability=CapabilityId.DATA_SELECTION_MANAGE,
        )
    )
    trader_models_manage = fixture.service.grant(
        command=_grant_command(
            fixture,
            grantee=fixture.trader,
            capability=CapabilityId.MODELS_MANAGE,
        )
    )

    assert set(DELEGABLE_CAPABILITY_GRANTEE_ROLES) == set(DELEGABLE_CAPABILITIES)
    assert DELEGABLE_CAPABILITY_GRANTEE_ROLES[CapabilityId.DATA_SELECTION_MANAGE] == {"trader"}
    assert all(
        eligible_roles == {"admin"}
        for capability, eligible_roles in DELEGABLE_CAPABILITY_GRANTEE_ROLES.items()
        if capability is not CapabilityId.DATA_SELECTION_MANAGE
    )
    assert trader_data_selection.allowed is True
    assert admin_data_selection.deny_reason is DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE
    assert trader_models_manage.deny_reason is DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE


def test_evaluation_denies_a_persisted_grant_with_an_ineligible_grantee_role() -> None:
    fixture = _fixture()
    fixture.delegation_repository.create_or_get_active_grant(
        organization_id=fixture.organization_id,
        grantee_user_id=fixture.trader.user_id,
        capability=CapabilityId.MODELS_MANAGE,
        resource_scope=SCOPE,
        granted_by_owner_user_id=fixture.owner.user_id,
        granted_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )

    decision = fixture.service.decide(request=_evaluation_request(fixture, actor=fixture.trader))

    assert decision.allowed is False
    assert decision.delegation is None
    assert decision.delegation_deny_reason is DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE


def test_other_organization_and_resource_context_never_match_a_delegation() -> None:
    fixture = _fixture()
    assert fixture.service.grant(command=_grant_command(fixture)).allowed is True
    other_organization = fixture.organization_repository.create_organization(
        actor_user_id=fixture.owner.user_id,
        slug="delegation-other",
        display_name="Other organization",
        created_at=NOW,
    )
    fixture.organization_repository.add_membership(
        organization_id=other_organization.organization_id,
        user_id=fixture.admin.user_id,
        role="admin",
        actor_user_id=fixture.owner.user_id,
        created_at=NOW,
    )

    other_organization_decision = fixture.service.decide(
        request=_evaluation_request(
            fixture,
            organization_id=other_organization.organization_id,
        )
    )
    mismatched_resource_decision = fixture.service.decide(
        request=_evaluation_request(
            fixture,
            resource=AuthorizationResource(organization_id=other_organization.organization_id),
        )
    )
    mismatched_scope_decision = fixture.service.decide(
        request=_evaluation_request(fixture, resource_scope="organization:limited")
    )

    assert other_organization_decision.allowed is False
    assert (
        other_organization_decision.delegation_deny_reason
        is DelegationDenyReason.DELEGATION_NOT_ACTIVE
    )
    assert mismatched_resource_decision.allowed is False
    assert mismatched_resource_decision.delegation is None
    assert (
        mismatched_resource_decision.delegation_deny_reason
        is DelegationDenyReason.DELEGATION_NOT_ACTIVE
    )
    assert mismatched_scope_decision.allowed is False
    assert (
        mismatched_scope_decision.delegation_deny_reason
        is DelegationDenyReason.INVALID_RESOURCE_SCOPE
    )


def test_in_memory_grant_is_concurrent_idempotent_and_rejects_a_material_conflict() -> None:
    fixture = _fixture()
    command = _grant_command(fixture)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: fixture.service.grant(command=command), range(16)))

    grant_ids = {result.grant.delegation_id for result in results if result.grant is not None}
    conflict = fixture.service.grant(
        command=_grant_command(fixture, expires_at=NOW + timedelta(hours=2))
    )
    audit_events = fixture.delegation_repository.list_audit_events(
        organization_id=fixture.organization_id
    )

    assert all(result.allowed for result in results)
    assert len(grant_ids) == 1
    assert sum(not result.idempotent for result in results) == 1
    assert conflict.deny_reason is DelegationDenyReason.GRANT_CONFLICT
    assert [event.action for event in audit_events] == ["delegation.granted"]
