from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Mapping, cast
from uuid import uuid4

import pytest

from trading.contexts.identity.adapters.outbound.persistence.in_memory.delegation_repository import (  # noqa: E501
    InMemoryDelegationRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory.organization_repository import (  # noqa: E501
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.authorization import (
    AuthorizationDenyReason,
    AuthorizationResource,
    CapabilityAuthorizationService,
    CapabilityId,
)
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityGrant,
    DelegationDenyReason,
    DelegationResourceScope,
    GrantDelegatedCapabilityCommand,
    RevokeDelegatedCapabilityCommand,
)
from trading.contexts.identity.application.delegation.service import (
    DelegatedCapabilityService,
)
from trading.contexts.identity.application.mutation_security import (
    CapabilityAuthorizationAdapter,
    DelegatedCapabilityAuthorizationAdapter,
    EffectiveAuthoritySource,
    EffectiveAuthorizationDecision,
    EffectiveAuthorizationRequest,
    EffectiveAuthorizer,
    IdempotencyDisposition,
    IdempotencyRecordState,
    InMemoryMutationIdempotencyStore,
    JsonValue,
    MutationActionPolicy,
    MutationAuditEvent,
    MutationSecurityDecision,
    MutationSecurityDenyReason,
    MutationSecurityRequest,
    MutationSecurityService,
)
from trading.contexts.identity.application.ports import (
    CurrentUserPrincipal,
    OrganizationRepository,
)
from trading.contexts.identity.domain.entities import OrganizationMembership
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


@dataclass
class _OrganizationRepository:
    actor: CurrentUserPrincipal
    organization_id: OrganizationId
    role: Literal["owner", "admin", "operator", "trader", "viewer"] = "owner"

    def get_membership(
        self, *, organization_id: OrganizationId, user_id: UserId
    ) -> OrganizationMembership | None:
        if organization_id != self.organization_id or user_id != self.actor.user_id:
            return None
        return OrganizationMembership(
            organization_id=organization_id,
            user_id=user_id,
            role=self.role,
            status="active",
            created_at=NOW,
        )

    def is_installation_owner(self, *, user_id: UserId) -> bool:
        return user_id == self.actor.user_id


class _Validator:
    def validate(self, *, payload: Mapping[str, object]) -> Mapping[str, JsonValue]:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name is required")
        note = payload.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError("note must be a string")
        return {"name": name.strip(), "note": note}


class _AuditSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[MutationAuditEvent] = []
        self.fail = fail

    def record(self, *, event: MutationAuditEvent) -> None:
        if self.fail:
            raise RuntimeError("audit unavailable")
        self.events.append(event)


@dataclass(frozen=True)
class _TransportProof:
    cookie_authenticated: bool = True
    accepted: bool = True


def _principal(*, authenticated_at: datetime = NOW) -> CurrentUserPrincipal:
    return CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel.free(),
        session_created_at=authenticated_at,
    )


def _service(
    *,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId,
    store: InMemoryMutationIdempotencyStore | None = None,
    audit: _AuditSink | None = None,
    role: Literal["owner", "admin", "operator", "trader", "viewer"] = "owner",
    capability: CapabilityId | str = CapabilityId.STRATEGIES_MANAGE,
    resource_required: bool = True,
    recent_auth_required: bool = False,
    authorizer: EffectiveAuthorizer | None = None,
) -> MutationSecurityService:
    repository = _OrganizationRepository(
        actor=principal,
        organization_id=organization_id,
        role=role,
    )
    return MutationSecurityService(
        authorizer=authorizer
        or CapabilityAuthorizationAdapter(
            service=CapabilityAuthorizationService(
                organization_repository=cast(OrganizationRepository, repository)
            )
        ),
        idempotency_store=store,
        audit_sink=audit,
        action_policies={
            "strategy.update": MutationActionPolicy(
                capability=capability,
                resource_required=resource_required,
                recent_auth_required=recent_auth_required,
            )
        },
    )


@dataclass(frozen=True, slots=True)
class _DelegationFixture:
    owner: CurrentUserPrincipal
    grantee: CurrentUserPrincipal
    organization_id: OrganizationId
    service: DelegatedCapabilityService


def _delegation_fixture() -> _DelegationFixture:
    organization_repository = InMemoryOrganizationRepository()
    owner = _principal()
    grantee = _principal()
    _, organization = organization_repository.bootstrap_installation(
        owner_user_id=owner.user_id,
        installation_name="Mutation delegation fixture",
        organization_slug="mutation-delegation",
        organization_name="Mutation delegation",
        created_at=NOW,
    )
    organization_repository.add_membership(
        organization_id=organization.organization_id,
        user_id=grantee.user_id,
        role="admin",
        actor_user_id=owner.user_id,
        created_at=NOW,
    )
    service = DelegatedCapabilityService(
        capability_authorization_service=CapabilityAuthorizationService(
            organization_repository=organization_repository
        ),
        organization_repository=organization_repository,
        delegation_repository=InMemoryDelegationRepository(),
    )
    return _DelegationFixture(
        owner=owner,
        grantee=grantee,
        organization_id=organization.organization_id,
        service=service,
    )


def _grant(
    fixture: _DelegationFixture,
    *,
    requested_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(hours=1),
) -> DelegatedCapabilityGrant:
    result = fixture.service.grant(
        command=GrantDelegatedCapabilityCommand(
            actor=fixture.owner,
            organization_id=fixture.organization_id,
            grantee_user_id=fixture.grantee.user_id,
            capability=CapabilityId.MODELS_MANAGE,
            resource_scope=DelegationResourceScope.ORGANIZATION,
            expires_at=expires_at,
            requested_at=requested_at,
            recent_authentication_verified=True,
        )
    )
    assert result.allowed is True
    assert result.grant is not None
    return result.grant


class _UnavailableAuthorizer:
    def decide(self, *, request: EffectiveAuthorizationRequest) -> EffectiveAuthorizationDecision:
        _ = request
        raise RuntimeError("effective authorizer unavailable")


def _decide(
    service: MutationSecurityService,
    request: MutationSecurityRequest,
    *,
    cookie_authenticated: bool = True,
    transport_accepted: bool = True,
) -> MutationSecurityDecision:
    return service.decide(
        request=request,
        transport_proof=_TransportProof(
            cookie_authenticated=cookie_authenticated,
            accepted=transport_accepted,
        ),
    )


def _request(
    *,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId,
    payload: Mapping[str, object] | None = None,
    resource: AuthorizationResource | None = None,
    idempotency_key: str = "mutation-request-0001",
    now: datetime = NOW,
) -> MutationSecurityRequest:
    return MutationSecurityRequest(
        actor=principal,
        selected_organization_id=organization_id,
        resource=resource
        or AuthorizationResource(
            organization_id=organization_id,
            owner_user_id=principal.user_id,
        ),
        resource_reference="strategy:fixture",
        action="strategy.update",
        raw_payload=payload or {"name": "alpha", "note": "<redacted>"},
        validator=_Validator(),
        now=now,
        idempotency_key=idempotency_key,
    )


def test_direct_capability_authority_is_normalized_into_decision_and_audit() -> None:
    fixture = _delegation_fixture()
    audit = _AuditSink()
    service = _service(
        principal=fixture.owner,
        organization_id=fixture.organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=audit,
        capability=CapabilityId.MODELS_MANAGE,
        authorizer=DelegatedCapabilityAuthorizationAdapter(service=fixture.service),
    )

    decision = _decide(
        service,
        _request(principal=fixture.owner, organization_id=fixture.organization_id),
    )

    assert decision.allowed is True
    assert decision.authorization is not None
    assert decision.authorization.allowed is True
    assert decision.authorization.capability is CapabilityId.MODELS_MANAGE
    assert decision.authorization.scope == "organization"
    assert decision.authorization.authority_source is EffectiveAuthoritySource.CAPABILITY_KERNEL
    assert decision.authorization.delegated_organization_id is None
    assert audit.events[0].authority_source is EffectiveAuthoritySource.CAPABILITY_KERNEL
    assert audit.events[0].delegated_organization_id is None


def test_active_delegation_authority_is_normalized_without_exposing_grant() -> None:
    fixture = _delegation_fixture()
    grant = _grant(fixture)
    audit = _AuditSink()
    service = _service(
        principal=fixture.grantee,
        organization_id=fixture.organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=audit,
        capability=CapabilityId.MODELS_MANAGE,
        authorizer=DelegatedCapabilityAuthorizationAdapter(service=fixture.service),
    )

    decision = _decide(
        service,
        _request(
            principal=fixture.grantee,
            organization_id=fixture.organization_id,
        ),
    )

    assert decision.allowed is True
    assert decision.authorization is not None
    assert decision.authorization.capability is CapabilityId.MODELS_MANAGE
    assert decision.authorization.scope == "delegated_organization"
    assert decision.authorization.authority_source is EffectiveAuthoritySource.DELEGATION
    assert decision.authorization.delegated_organization_id == fixture.organization_id
    assert not hasattr(decision.authorization, "delegation")
    assert str(grant.delegation_id) not in repr(decision)
    assert "granted_by_owner_user_id" not in repr(decision)
    assert audit.events[0].authority_source is EffectiveAuthoritySource.DELEGATION
    assert audit.events[0].delegated_organization_id == fixture.organization_id
    assert str(grant.delegation_id) not in repr(audit.events[0])


def test_revoked_delegation_fails_closed() -> None:
    fixture = _delegation_fixture()
    grant = _grant(fixture)
    revoked = fixture.service.revoke(
        command=RevokeDelegatedCapabilityCommand(
            actor=fixture.owner,
            organization_id=fixture.organization_id,
            delegation_id=grant.delegation_id,
            requested_at=NOW + timedelta(minutes=1),
            recent_authentication_verified=True,
        )
    )
    assert revoked.allowed is True
    audit = _AuditSink()
    service = _service(
        principal=fixture.grantee,
        organization_id=fixture.organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=audit,
        capability=CapabilityId.MODELS_MANAGE,
        authorizer=DelegatedCapabilityAuthorizationAdapter(service=fixture.service),
    )

    decision = _decide(
        service,
        _request(
            principal=fixture.grantee,
            organization_id=fixture.organization_id,
            now=NOW + timedelta(minutes=2),
        ),
    )

    assert decision.allowed is False
    assert decision.reason is MutationSecurityDenyReason.AUTHORIZATION_DENIED
    assert decision.authorization is not None
    assert decision.authorization.deny_reason == DelegationDenyReason.DELEGATION_NOT_ACTIVE.value
    assert decision.authorization.authority_source is None
    assert decision.authorization.delegated_organization_id is None
    assert audit.events[0].authority_source is None
    assert audit.events[0].delegated_organization_id is None


def test_expired_delegation_fails_closed() -> None:
    fixture = _delegation_fixture()
    _grant(fixture, expires_at=NOW + timedelta(hours=1))
    service = _service(
        principal=fixture.grantee,
        organization_id=fixture.organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=_AuditSink(),
        capability=CapabilityId.MODELS_MANAGE,
        authorizer=DelegatedCapabilityAuthorizationAdapter(service=fixture.service),
    )

    decision = _decide(
        service,
        _request(
            principal=fixture.grantee,
            organization_id=fixture.organization_id,
            now=NOW + timedelta(hours=1),
        ),
    )

    assert decision.allowed is False
    assert decision.reason is MutationSecurityDenyReason.AUTHORIZATION_DENIED
    assert decision.authorization is not None
    assert decision.authorization.deny_reason == DelegationDenyReason.DELEGATION_NOT_ACTIVE.value
    assert decision.authorization.authority_source is None
    assert decision.authorization.delegated_organization_id is None


def test_authorizer_unavailability_fails_closed_without_authority_metadata() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    audit = _AuditSink()
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=audit,
        authorizer=_UnavailableAuthorizer(),
    )

    decision = _decide(
        service,
        _request(principal=principal, organization_id=organization_id),
    )

    assert decision.allowed is False
    assert decision.reason is MutationSecurityDenyReason.AUTHORIZATION_UNAVAILABLE
    assert decision.authorization is None
    assert audit.events[0].authority_source is None
    assert audit.events[0].delegated_organization_id is None


def test_terminal_repeat_replays_and_changed_content_conflicts() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    store = InMemoryMutationIdempotencyStore()
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=store,
        audit=_AuditSink(),
    )
    request = _request(principal=principal, organization_id=organization_id)

    first = _decide(service, request)
    assert first.allowed is True
    assert first.idempotency is IdempotencyDisposition.NEW
    assert first.payload_hash is not None
    assert first.idempotency_key_hash is not None
    service.finish_idempotency(
        decision=first,
        state=IdempotencyRecordState.SUCCEEDED,
        terminal_reference="operation:result-0001",
    )

    replay = _decide(service, request)
    conflict = _decide(
        service,
        _request(
            principal=principal,
            organization_id=organization_id,
            payload={"name": "changed", "note": "<redacted-changed>"},
        ),
    )

    assert replay.allowed is True
    assert replay.idempotency is IdempotencyDisposition.REPLAY_TERMINAL
    assert replay.terminal_reference == "operation:result-0001"
    assert conflict.allowed is False
    assert conflict.reason is MutationSecurityDenyReason.IDEMPOTENCY_CONFLICT


def test_same_idempotency_key_is_isolated_by_actor_and_organization_scope() -> None:
    first_actor = _principal()
    second_actor = _principal()
    first_organization = OrganizationId(uuid4())
    second_organization = OrganizationId(uuid4())
    store = InMemoryMutationIdempotencyStore()

    decisions = (
        _decide(
            _service(
                principal=first_actor,
                organization_id=first_organization,
                store=store,
                audit=_AuditSink(),
            ),
            _request(
                principal=first_actor,
                organization_id=first_organization,
            ),
        ),
        _decide(
            _service(
                principal=first_actor,
                organization_id=second_organization,
                store=store,
                audit=_AuditSink(),
            ),
            _request(
                principal=first_actor,
                organization_id=second_organization,
            ),
        ),
        _decide(
            _service(
                principal=second_actor,
                organization_id=first_organization,
                store=store,
                audit=_AuditSink(),
            ),
            _request(
                principal=second_actor,
                organization_id=first_organization,
            ),
        ),
    )

    assert all(decision.allowed for decision in decisions)
    assert all(decision.idempotency is IdempotencyDisposition.NEW for decision in decisions)


def test_cross_organization_resource_is_denied_by_capability_kernel() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=_AuditSink(),
        role="admin",
        capability=CapabilityId.CONNECTIONS_MANAGE,
    )

    decision = _decide(
        service,
        _request(
            principal=principal,
            organization_id=organization_id,
            resource=AuthorizationResource(
                organization_id=OrganizationId(uuid4()),
                owner_user_id=principal.user_id,
            ),
        ),
    )

    assert decision.allowed is False
    assert decision.reason is MutationSecurityDenyReason.AUTHORIZATION_DENIED
    assert decision.authorization is not None
    assert (
        decision.authorization.deny_reason
        == AuthorizationDenyReason.RESOURCE_ORGANIZATION_MISMATCH.value
    )


def test_own_scope_rejects_resource_owned_by_another_user() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=_AuditSink(),
        role="trader",
    )

    decision = _decide(
        service,
        _request(
            principal=principal,
            organization_id=organization_id,
            resource=AuthorizationResource(
                organization_id=organization_id,
                owner_user_id=UserId(uuid4()),
            ),
        ),
    )

    assert decision.allowed is False
    assert decision.authorization is not None
    assert decision.authorization.deny_reason == AuthorizationDenyReason.OWNERSHIP_REQUIRED.value


def test_stale_recent_auth_and_unknown_capability_fail_closed() -> None:
    stale = _principal(authenticated_at=NOW - timedelta(minutes=11))
    organization_id = OrganizationId(uuid4())
    service = _service(
        principal=stale,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=_AuditSink(),
        role="admin",
        capability=CapabilityId.CONNECTIONS_MANAGE,
    )

    stale_decision = _decide(
        service,
        _request(
            principal=stale,
            organization_id=organization_id,
        ),
    )
    unknown_decision = _decide(
        _service(
            principal=stale,
            organization_id=organization_id,
            store=InMemoryMutationIdempotencyStore(),
            audit=_AuditSink(),
            capability="unknown.capability",
        ),
        _request(
            principal=stale,
            organization_id=organization_id,
        ),
    )

    assert stale_decision.reason is MutationSecurityDenyReason.RECENT_AUTH_REQUIRED
    assert unknown_decision.reason is MutationSecurityDenyReason.AUTHORIZATION_DENIED
    assert unknown_decision.authorization is not None
    assert (
        unknown_decision.authorization.deny_reason
        == AuthorizationDenyReason.UNKNOWN_CAPABILITY.value
    )


def test_audit_contains_hashes_not_raw_payload_or_idempotency_key() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    audit = _AuditSink()
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=audit,
    )
    request = _request(
        principal=principal,
        organization_id=organization_id,
        payload={
            "name": " alpha ",
            "note": "<redacted-private-material>",
            "ignored": "<redacted-ignored>",
        },
        idempotency_key="raw-idempotency-key-0001",
    )

    decision = _decide(service, request)

    assert decision.allowed is True
    assert decision.audit_recorded is True
    assert len(audit.events) == 1
    serialized = repr(audit.events[0])
    assert "<redacted-private-material>" not in serialized
    assert "raw-idempotency-key-0001" not in serialized
    assert audit.events[0].request_payload_hash == decision.payload_hash
    assert audit.events[0].idempotency_key_hash == decision.idempotency_key_hash
    assert decision.validated_payload == {
        "name": "alpha",
        "note": "<redacted-private-material>",
    }
    assert decision.validated_payload is not None
    assert "ignored" not in decision.validated_payload
    with pytest.raises(TypeError):
        decision.validated_payload["name"] = "changed"  # type: ignore[index]


def test_missing_or_rejected_transport_proof_cannot_authorize_browser_mutation() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=_AuditSink(),
    )
    request = _request(principal=principal, organization_id=organization_id)

    api_client = _decide(service, request, cookie_authenticated=False)
    rejected_browser = _decide(service, request, transport_accepted=False)

    assert api_client.applicable is False
    assert api_client.allowed is False
    assert rejected_browser.applicable is True
    assert rejected_browser.reason is MutationSecurityDenyReason.TRANSPORT_DENIED


def test_invalid_input_and_missing_validation_resource_idempotency_audit_deny() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=InMemoryMutationIdempotencyStore(),
        audit=None,
    )
    base = _request(principal=principal, organization_id=organization_id)

    missing_validation = _decide(
        service,
        MutationSecurityRequest(
            actor=base.actor,
            selected_organization_id=base.selected_organization_id,
            resource=base.resource,
            resource_reference=base.resource_reference,
            action=base.action,
            raw_payload=base.raw_payload,
            validator=None,
            now=base.now,
        ),
    )
    missing_resource = _decide(
        service,
        MutationSecurityRequest(
            actor=base.actor,
            selected_organization_id=base.selected_organization_id,
            resource=None,
            resource_reference=None,
            action=base.action,
            raw_payload=base.raw_payload,
            validator=base.validator,
            now=base.now,
        ),
    )
    missing_audit = _decide(service, base)
    missing_idempotency = _decide(
        _service(
            principal=principal,
            organization_id=organization_id,
            store=None,
            audit=_AuditSink(),
        ),
        base,
    )
    invalid_input = _decide(
        _service(
            principal=principal,
            organization_id=organization_id,
            store=InMemoryMutationIdempotencyStore(),
            audit=_AuditSink(),
        ),
        _request(
            principal=principal,
            organization_id=organization_id,
            payload={"name": "", "note": "<redacted>"},
        ),
    )

    assert missing_validation.reason is MutationSecurityDenyReason.VALIDATION_REQUIRED
    assert missing_resource.reason is MutationSecurityDenyReason.RESOURCE_CONTEXT_REQUIRED
    assert missing_audit.reason is MutationSecurityDenyReason.AUDIT_REQUIRED
    assert missing_idempotency.reason is MutationSecurityDenyReason.IDEMPOTENCY_REQUIRED
    assert invalid_input.reason is MutationSecurityDenyReason.REQUEST_INVALID


def test_audit_failure_marks_new_idempotency_result_unknown() -> None:
    principal = _principal()
    organization_id = OrganizationId(uuid4())
    store = InMemoryMutationIdempotencyStore()
    service = _service(
        principal=principal,
        organization_id=organization_id,
        store=store,
        audit=_AuditSink(fail=True),
    )
    request = _request(principal=principal, organization_id=organization_id)

    first = _decide(service, request)
    retry = _decide(service, request)

    assert first.reason is MutationSecurityDenyReason.AUDIT_UNAVAILABLE
    assert retry.reason is (MutationSecurityDenyReason.IDEMPOTENCY_RECONCILIATION_REQUIRED)
