"""Shared fail-closed security envelope for cookie-authenticated mutations."""

from .effective_authorizer import (
    CapabilityAuthorizationAdapter,
    DelegatedCapabilityAuthorizationAdapter,
    EffectiveAuthoritySource,
    EffectiveAuthorizationDecision,
    EffectiveAuthorizationRequest,
    EffectiveAuthorizer,
)
from .idempotency import (
    IdempotencyBeginResult,
    IdempotencyBeginStatus,
    IdempotencyIdentity,
    IdempotencyRecord,
    IdempotencyRecordState,
    InMemoryMutationIdempotencyStore,
    MutationIdempotencyStore,
)
from .models import (
    BrowserMutationTransportProof,
    IdempotencyDisposition,
    JsonValue,
    MutationActionPolicy,
    MutationAuditEvent,
    MutationAuditOutcome,
    MutationAuditSink,
    MutationPayloadValidator,
    MutationSecurityDecision,
    MutationSecurityDenyReason,
    MutationSecurityRequest,
    ValidatedJsonValue,
)
from .service import MutationSecurityService

__all__ = [
    "CapabilityAuthorizationAdapter",
    "DelegatedCapabilityAuthorizationAdapter",
    "EffectiveAuthorizationDecision",
    "EffectiveAuthorizationRequest",
    "EffectiveAuthorizer",
    "EffectiveAuthoritySource",
    "IdempotencyBeginResult",
    "IdempotencyBeginStatus",
    "IdempotencyDisposition",
    "IdempotencyIdentity",
    "IdempotencyRecord",
    "IdempotencyRecordState",
    "InMemoryMutationIdempotencyStore",
    "JsonValue",
    "BrowserMutationTransportProof",
    "MutationActionPolicy",
    "MutationAuditEvent",
    "MutationAuditOutcome",
    "MutationAuditSink",
    "MutationIdempotencyStore",
    "MutationPayloadValidator",
    "MutationSecurityDecision",
    "MutationSecurityDenyReason",
    "MutationSecurityRequest",
    "MutationSecurityService",
    "ValidatedJsonValue",
]
