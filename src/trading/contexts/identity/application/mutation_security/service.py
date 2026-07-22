from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Mapping

from trading.contexts.identity.application.authorization import (
    CapabilityId,
)
from trading.contexts.identity.application.mutation_security.effective_authorizer import (
    EffectiveAuthoritySource,
    EffectiveAuthorizationDecision,
    EffectiveAuthorizationRequest,
    EffectiveAuthorizer,
)
from trading.contexts.identity.application.mutation_security.idempotency import (
    IdempotencyBeginStatus,
    IdempotencyIdentity,
    IdempotencyRecordState,
    MutationIdempotencyStore,
)
from trading.contexts.identity.application.mutation_security.models import (
    BrowserMutationTransportProof,
    IdempotencyDisposition,
    JsonValue,
    MutationActionPolicy,
    MutationAuditEvent,
    MutationAuditOutcome,
    MutationAuditSink,
    MutationSecurityDecision,
    MutationSecurityDenyReason,
    MutationSecurityRequest,
    ValidatedJsonValue,
)
from trading.shared_kernel.primitives import OrganizationId

_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_ACTION_ID = re.compile(r"^[a-z][a-z0-9._:-]{2,127}$")
_TERMINAL_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_RECENT_AUTH_CAPABILITIES = frozenset(
    {
        CapabilityId.CONNECTIONS_MANAGE,
        CapabilityId.MODELS_PROMOTE_OR_ROLLBACK,
        CapabilityId.ADMIN_MEMBERS_MANAGE,
        CapabilityId.INSTALLATION_TRUST_MANAGE,
        CapabilityId.INSTALLATION_RESOURCES_MANAGE,
        CapabilityId.INSTALLATION_RECOVERY_EXECUTE,
    }
)


class MutationSecurityService:
    """Fail-closed application envelope for one validated browser mutation."""

    def __init__(
        self,
        *,
        authorizer: EffectiveAuthorizer,
        idempotency_store: MutationIdempotencyStore | None,
        audit_sink: MutationAuditSink | None,
        action_policies: Mapping[str, MutationActionPolicy],
        recent_auth_window: timedelta = timedelta(minutes=10),
    ) -> None:
        if authorizer is None:  # type: ignore[truthy-bool]
            raise ValueError("MutationSecurityService requires authorizer")
        if recent_auth_window <= timedelta(0):
            raise ValueError("recent_auth_window must be positive")
        normalized_policies: dict[str, MutationActionPolicy] = {}
        for action, policy in action_policies.items():
            if _ACTION_ID.fullmatch(action) is None:
                raise ValueError("mutation action policy has an invalid action id")
            normalized_policies[action] = policy
        self._authorizer = authorizer
        self._idempotency_store = idempotency_store
        self._audit_sink = audit_sink
        self._recent_auth_window = recent_auth_window
        self._action_policies = MappingProxyType(normalized_policies)

    def decide(
        self,
        *,
        request: MutationSecurityRequest,
        transport_proof: BrowserMutationTransportProof,
    ) -> MutationSecurityDecision:
        if not transport_proof.cookie_authenticated:
            return MutationSecurityDecision(
                applicable=False,
                allowed=False,
                reason=None,
                authorization=None,
                idempotency=IdempotencyDisposition.NOT_REQUIRED,
            )
        now = request.now
        if now.tzinfo is None or now.utcoffset() is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.INVALID_TIME_CONTEXT,
            )
        now = now.astimezone(UTC)
        policy = self._action_policies.get(request.action)
        if policy is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.UNKNOWN_ACTION,
                now=now,
            )
        if not transport_proof.accepted:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.TRANSPORT_DENIED,
                now=now,
                capability=policy.capability,
            )
        if request.validator is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.VALIDATION_REQUIRED,
                now=now,
                capability=policy.capability,
            )
        try:
            validated = request.validator.validate(payload=request.raw_payload)
            payload_hash, frozen_payload = _canonical_payload(validated)
        except Exception:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.REQUEST_INVALID,
                now=now,
                capability=policy.capability,
            )

        resource_reference = (
            None if request.resource_reference is None else request.resource_reference.strip()
        )
        if policy.resource_required and (request.resource is None or not resource_reference):
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.RESOURCE_CONTEXT_REQUIRED,
                now=now,
                payload_hash=payload_hash,
                capability=policy.capability,
            )
        if resource_reference is not None and not 1 <= len(resource_reference) <= 512:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.REQUEST_INVALID,
                now=now,
                payload_hash=payload_hash,
                capability=policy.capability,
            )
        resource_reference_hash = (
            None if resource_reference is None else _sha256(resource_reference)
        )

        try:
            authorization = self._authorizer.decide(
                request=EffectiveAuthorizationRequest(
                    actor=request.actor,
                    capability=policy.capability,
                    selected_organization_id=request.selected_organization_id,
                    resource=request.resource,
                    evaluated_at=now,
                )
            )
        except Exception:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.AUTHORIZATION_UNAVAILABLE,
                now=now,
                payload_hash=payload_hash,
                capability=policy.capability,
            )
        if not _authorization_matches_request(
            authorization=authorization,
            capability=policy.capability,
            organization_id=request.selected_organization_id,
        ):
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.AUTHORIZATION_DENIED,
                now=now,
                payload_hash=payload_hash,
                capability=policy.capability,
            )
        if not authorization.allowed or authorization.capability is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.AUTHORIZATION_DENIED,
                now=now,
                payload_hash=payload_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        recent_auth_required = (
            policy.recent_auth_required or policy.capability in _RECENT_AUTH_CAPABILITIES
        )
        if recent_auth_required and not self._is_recent(request=request, now=now):
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.RECENT_AUTH_REQUIRED,
                now=now,
                payload_hash=payload_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        if self._audit_sink is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.AUDIT_REQUIRED,
                now=now,
                payload_hash=payload_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        if self._idempotency_store is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.IDEMPOTENCY_REQUIRED,
                now=now,
                payload_hash=payload_hash,
                authorization=authorization,
                capability=policy.capability,
            )

        key = "" if request.idempotency_key is None else request.idempotency_key.strip()
        if _IDEMPOTENCY_KEY.fullmatch(key) is None:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.IDEMPOTENCY_KEY_INVALID,
                now=now,
                payload_hash=payload_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        key_hash = _sha256(key)
        identity = IdempotencyIdentity(
            actor_user_id=request.actor.user_id,
            organization_id=request.selected_organization_id,
            capability=authorization.capability,
            action=request.action,
            resource_reference_hash=resource_reference_hash,
            key_hash=key_hash,
        )
        try:
            begin = self._idempotency_store.begin(
                identity=identity,
                payload_hash=payload_hash,
            )
        except Exception:
            return self._deny(
                request=request,
                reason=MutationSecurityDenyReason.IDEMPOTENCY_UNAVAILABLE,
                now=now,
                payload_hash=payload_hash,
                idempotency_key_hash=key_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        deny_by_status = {
            IdempotencyBeginStatus.CONFLICT: MutationSecurityDenyReason.IDEMPOTENCY_CONFLICT,
            IdempotencyBeginStatus.IN_PROGRESS: MutationSecurityDenyReason.IDEMPOTENCY_IN_PROGRESS,
            IdempotencyBeginStatus.RECONCILIATION_REQUIRED: (
                MutationSecurityDenyReason.IDEMPOTENCY_RECONCILIATION_REQUIRED
            ),
        }
        if begin.status in deny_by_status:
            return self._deny(
                request=request,
                reason=deny_by_status[begin.status],
                now=now,
                payload_hash=payload_hash,
                idempotency_key_hash=key_hash,
                authorization=authorization,
                capability=policy.capability,
            )
        if begin.status is IdempotencyBeginStatus.REPLAY_TERMINAL:
            disposition = IdempotencyDisposition.REPLAY_TERMINAL
            terminal_reference = begin.record.terminal_reference
        else:
            disposition = IdempotencyDisposition.NEW
            terminal_reference = None

        if not self._record_audit(
            request=request,
            now=now,
            outcome=MutationAuditOutcome.AUTHORIZED,
            reason=None,
            payload_hash=payload_hash,
            idempotency_key_hash=key_hash,
            capability=policy.capability,
            authorization=authorization,
        ):
            if disposition is IdempotencyDisposition.NEW:
                try:
                    self._idempotency_store.finish(
                        identity=identity,
                        payload_hash=payload_hash,
                        state=IdempotencyRecordState.UNKNOWN,
                        terminal_reference=None,
                    )
                except Exception:
                    pass
            return MutationSecurityDecision(
                applicable=True,
                allowed=False,
                reason=MutationSecurityDenyReason.AUDIT_UNAVAILABLE,
                authorization=authorization,
                idempotency=disposition,
                payload_hash=payload_hash,
                idempotency_key_hash=key_hash,
            )
        return MutationSecurityDecision(
            applicable=True,
            allowed=True,
            reason=None,
            authorization=authorization,
            idempotency=disposition,
            payload_hash=payload_hash,
            idempotency_key_hash=key_hash,
            terminal_reference=terminal_reference,
            audit_recorded=True,
            idempotency_identity=identity,
            validated_payload=frozen_payload,
        )

    def finish_idempotency(
        self,
        *,
        decision: MutationSecurityDecision,
        state: IdempotencyRecordState,
        terminal_reference: str | None,
    ) -> None:
        """Finalize a new reservation after execution, or mark an unknown outcome."""
        if (
            not decision.applicable
            or not decision.allowed
            or decision.idempotency is not IdempotencyDisposition.NEW
            or decision.idempotency_identity is None
            or decision.payload_hash is None
            or self._idempotency_store is None
        ):
            raise ValueError("decision has no new idempotency reservation")
        if state is IdempotencyRecordState.PROCESSING:
            raise ValueError("idempotency reservation cannot finish as processing")
        if state is IdempotencyRecordState.UNKNOWN:
            if terminal_reference is not None:
                raise ValueError("unknown result cannot have a terminal reference")
        elif (
            terminal_reference is None or _TERMINAL_REFERENCE.fullmatch(terminal_reference) is None
        ):
            raise ValueError("terminal result requires a safe reference")
        self._idempotency_store.finish(
            identity=decision.idempotency_identity,
            payload_hash=decision.payload_hash,
            state=state,
            terminal_reference=terminal_reference,
        )

    def _deny(
        self,
        *,
        request: MutationSecurityRequest,
        reason: MutationSecurityDenyReason,
        now: datetime | None = None,
        payload_hash: str | None = None,
        idempotency_key_hash: str | None = None,
        authorization: EffectiveAuthorizationDecision | None = None,
        capability: CapabilityId | str | None = None,
    ) -> MutationSecurityDecision:
        audit_recorded = False
        if now is not None and self._audit_sink is not None:
            audit_recorded = self._record_audit(
                request=request,
                now=now,
                outcome=MutationAuditOutcome.REJECTED,
                reason=reason,
                payload_hash=payload_hash,
                idempotency_key_hash=idempotency_key_hash,
                capability=capability,
                authorization=authorization,
            )
        return MutationSecurityDecision(
            applicable=True,
            allowed=False,
            reason=reason,
            authorization=authorization,
            idempotency=IdempotencyDisposition.NOT_REQUIRED,
            payload_hash=payload_hash,
            idempotency_key_hash=idempotency_key_hash,
            audit_recorded=audit_recorded,
        )

    def _record_audit(
        self,
        *,
        request: MutationSecurityRequest,
        now: datetime,
        outcome: MutationAuditOutcome,
        reason: MutationSecurityDenyReason | None,
        payload_hash: str | None,
        idempotency_key_hash: str | None,
        capability: CapabilityId | str | None,
        authorization: EffectiveAuthorizationDecision | None = None,
    ) -> bool:
        if self._audit_sink is None:
            return False
        try:
            audit_capability = CapabilityId(capability).value
        except (TypeError, ValueError):
            audit_capability = "unknown"
        event = MutationAuditEvent(
            occurred_at=now,
            actor_user_id=request.actor.user_id,
            organization_id=request.selected_organization_id,
            capability=audit_capability,
            action=request.action if _ACTION_ID.fullmatch(request.action) else "invalid",
            outcome=outcome,
            reason_code=None if reason is None else reason.value,
            authority_source=(None if authorization is None else authorization.authority_source),
            delegated_organization_id=(
                None if authorization is None else authorization.delegated_organization_id
            ),
            idempotency_key_hash=idempotency_key_hash,
            request_payload_hash=payload_hash,
        )
        try:
            self._audit_sink.record(event=event)
        except Exception:
            return False
        return True

    def _is_recent(self, *, request: MutationSecurityRequest, now: datetime) -> bool:
        authenticated_at = request.actor.session_created_at
        if authenticated_at is None:
            return False
        if authenticated_at.tzinfo is None or authenticated_at.utcoffset() is None:
            return False
        normalized = authenticated_at.astimezone(UTC)
        return normalized <= now <= normalized + self._recent_auth_window


def _canonical_payload(
    payload: Mapping[str, JsonValue],
) -> tuple[str, Mapping[str, ValidatedJsonValue]]:
    normalized = _normalize_json(payload)
    if not isinstance(normalized, Mapping):
        raise TypeError("validated mutation payload must be an object")
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    frozen = _freeze_json(normalized)
    if not isinstance(frozen, Mapping):
        raise TypeError("validated mutation payload must be an object")
    return _sha256(serialized), frozen


def _authorization_matches_request(
    *,
    authorization: EffectiveAuthorizationDecision,
    capability: CapabilityId | str,
    organization_id: OrganizationId | None,
) -> bool:
    try:
        requested_capability = CapabilityId(capability)
    except ValueError:
        requested_capability = None
    if not authorization.allowed:
        return (
            authorization.capability in {None, requested_capability}
            and authorization.scope is None
            and authorization.authority_source is None
            and authorization.delegated_organization_id is None
        )
    if (
        requested_capability is None
        or authorization.capability is not requested_capability
        or not authorization.scope
        or authorization.authority_source is None
    ):
        return False
    if authorization.authority_source is EffectiveAuthoritySource.CAPABILITY_KERNEL:
        return authorization.delegated_organization_id is None
    if authorization.authority_source is EffectiveAuthoritySource.DELEGATION:
        return (
            organization_id is not None
            and authorization.delegated_organization_id == organization_id
        )
    return False


def _normalize_json(value: JsonValue) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("payload keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    raise TypeError("payload is not JSON-compatible")


def _freeze_json(value: JsonValue) -> ValidatedJsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
