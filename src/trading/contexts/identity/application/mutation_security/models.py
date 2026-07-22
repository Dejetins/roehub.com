from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping, Protocol, TypeAlias

from trading.contexts.identity.application.authorization import (
    AuthorizationResource,
    CapabilityId,
)
from trading.contexts.identity.application.mutation_security.effective_authorizer import (
    EffectiveAuthoritySource,
    EffectiveAuthorizationDecision,
)
from trading.contexts.identity.application.mutation_security.idempotency import (
    IdempotencyIdentity,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import OrganizationId, UserId

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | Mapping[str, "JsonValue"]
ValidatedJsonValue: TypeAlias = (
    JsonScalar | tuple["ValidatedJsonValue", ...] | Mapping[str, "ValidatedJsonValue"]
)


class MutationSecurityDenyReason(StrEnum):
    VALIDATION_REQUIRED = "validation_required"
    REQUEST_INVALID = "request_invalid"
    UNKNOWN_ACTION = "unknown_action"
    TRANSPORT_DENIED = "transport_denied"
    RESOURCE_CONTEXT_REQUIRED = "resource_context_required"
    AUTHORIZATION_DENIED = "authorization_denied"
    RECENT_AUTH_REQUIRED = "recent_auth_required"
    IDEMPOTENCY_REQUIRED = "idempotency_required"
    IDEMPOTENCY_UNAVAILABLE = "idempotency_unavailable"
    IDEMPOTENCY_KEY_INVALID = "idempotency_key_invalid"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    IDEMPOTENCY_IN_PROGRESS = "idempotency_in_progress"
    IDEMPOTENCY_RECONCILIATION_REQUIRED = "idempotency_reconciliation_required"
    AUDIT_REQUIRED = "audit_required"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    AUTHORIZATION_UNAVAILABLE = "authorization_unavailable"
    INVALID_TIME_CONTEXT = "invalid_time_context"


class IdempotencyDisposition(StrEnum):
    NOT_REQUIRED = "not_required"
    NEW = "new"
    REPLAY_TERMINAL = "replay_terminal"


class MutationAuditOutcome(StrEnum):
    AUTHORIZED = "authorized"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class MutationSecurityRequest:
    actor: CurrentUserPrincipal
    selected_organization_id: OrganizationId | None
    resource: AuthorizationResource | None
    resource_reference: str | None
    action: str
    raw_payload: Mapping[str, object]
    validator: MutationPayloadValidator | None
    now: datetime
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class MutationActionPolicy:
    """Server-owned requirements for one registered mutation action."""

    capability: CapabilityId | str
    resource_required: bool
    recent_auth_required: bool = False


@dataclass(frozen=True, slots=True)
class MutationSecurityDecision:
    applicable: bool
    allowed: bool
    reason: MutationSecurityDenyReason | None
    authorization: EffectiveAuthorizationDecision | None
    idempotency: IdempotencyDisposition
    payload_hash: str | None = None
    idempotency_key_hash: str | None = None
    terminal_reference: str | None = None
    audit_recorded: bool = False
    idempotency_identity: IdempotencyIdentity | None = None
    validated_payload: Mapping[str, ValidatedJsonValue] | None = None


@dataclass(frozen=True, slots=True)
class MutationAuditEvent:
    occurred_at: datetime
    actor_user_id: UserId
    organization_id: OrganizationId | None
    capability: str
    action: str
    outcome: MutationAuditOutcome
    reason_code: str | None
    authority_source: EffectiveAuthoritySource | None
    delegated_organization_id: OrganizationId | None
    idempotency_key_hash: str | None
    request_payload_hash: str | None


class MutationPayloadValidator(Protocol):
    def validate(self, *, payload: Mapping[str, object]) -> Mapping[str, JsonValue]: ...


class MutationAuditSink(Protocol):
    def record(self, *, event: MutationAuditEvent) -> None: ...


class BrowserMutationTransportProof(Protocol):
    @property
    def cookie_authenticated(self) -> bool: ...

    @property
    def accepted(self) -> bool: ...
