from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

from .execution_source import ExecutionIntent
from .order_execution import ExchangeExecutionConnection

ExecutionProviderKind = Literal["core", "verified", "plugin"]
ExecutionSafetyMode = Literal["research", "paper", "testnet", "mainnet"]
ExecutionSubmitGuardStatus = Literal["accepted", "rejected"]
ExecutionSubmitGuardPhase = Literal["preflight", "pre_submit"]

MAINNET_RECENT_AUTH_WINDOW = timedelta(minutes=10)
MAINNET_APPROVAL_MAX_TTL = timedelta(minutes=15)


class ExecutionGatewayPolicyError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionAdapterIdentity:
    provider_id: str
    provider_version: str
    provider_kind: ExecutionProviderKind
    exchange_name: str
    revision_hash: str


@dataclass(frozen=True, slots=True)
class ExecutionPolicyPrincipal:
    """Authenticated identity evidence; PostgreSQL remains the authority source."""

    user_id: UserId
    session_id: UUID
    authenticated_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionProviderRegistration:
    provider_id: str
    provider_version: str
    provider_kind: ExecutionProviderKind
    exchange_name: str
    revision_hash: str
    order_submit_capability: bool
    enabled: bool
    approved_by_user_id: UserId
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionAccountSafetyState:
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_connection_id: UUID
    mode: ExecutionSafetyMode
    risk_revision_hash: str
    account_revision_hash: str
    secret_reference_hash: str
    risk_allows_submit: bool
    max_order_notional: Decimal
    daily_notional_limit: Decimal
    max_account_exposure_notional: Decimal
    risk_valid_until: datetime
    updated_by_user_id: UserId
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionCurrentRiskSnapshot:
    account_snapshot_fresh: bool
    config_guard_verified: bool
    daily_notional_used: Decimal
    account_exposure_notional: Decimal
    account_exposure_complete: bool
    observed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ExecutionMainnetApproval:
    approval_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_connection_id: UUID
    exchange_name: str
    market_type: str
    provider_id: str
    risk_revision_hash: str
    account_revision_hash: str
    provider_revision_hash: str
    recent_auth_session_id: UUID
    recent_auth_at: datetime
    approved_at: datetime
    expires_at: datetime
    audit_event_id: UUID
    revoked_at: datetime | None = None
    revocation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionKillSwitchState:
    scope_type: Literal["installation", "organization", "account"]
    active: bool
    reason: str
    updated_by_user_id: UserId
    updated_at: datetime
    organization_id: OrganizationId | None = None
    exchange_connection_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSubmitGuardQuery:
    intent: ExecutionIntent
    connection: ExchangeExecutionConnection
    adapter: ExecutionAdapterIdentity
    phase: ExecutionSubmitGuardPhase
    submission_attempt_id: UUID
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionSubmitGuardDecision:
    status: ExecutionSubmitGuardStatus
    reason: str
    check_name: str
    phase: ExecutionSubmitGuardPhase
    evaluated_at: datetime
    approval_id: UUID | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    audit_event_id: UUID | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


@dataclass(frozen=True, slots=True)
class ExecutionGatewayAuditEvent:
    event_id: UUID
    organization_id: OrganizationId | None
    owner_user_id: UserId | None
    exchange_connection_id: UUID | None
    intent_id: UUID | None
    approval_id: UUID | None
    event_type: str
    decision: str
    reason: str
    actor_user_id: UserId | None
    created_at: datetime
    metadata: Mapping[str, str] = field(default_factory=dict)


def evaluate_execution_submit_guard(
    *,
    query: ExecutionSubmitGuardQuery,
    safety: ExecutionAccountSafetyState | None,
    provider: ExecutionProviderRegistration | None,
    kill_switch_active: bool,
    approval: ExecutionMainnetApproval | None,
    risk_snapshot: ExecutionCurrentRiskSnapshot,
) -> ExecutionSubmitGuardDecision:
    intent = query.intent
    connection = query.connection
    adapter = query.adapter
    now = query.evaluated_at

    checks: tuple[tuple[str, bool, str], ...] = (
        (
            "organization_ownership",
            connection.organization_id == intent.organization_id,
            "organization_ownership_mismatch",
        ),
        (
            "account_ownership",
            connection.connection_id == intent.exchange_connection_id
            and connection.owner_user_id == intent.owner_user_id,
            "account_ownership_mismatch",
        ),
        (
            "connection_readiness",
            connection.connection_readiness == "ready_for_trading"
            and connection.effective_capability == "trading",
            "exchange_connection_not_ready_for_trading",
        ),
        (
            "safety_state",
            safety is not None,
            "execution_safety_state_unavailable",
        ),
    )
    rejection = _first_rejection(query=query, checks=checks)
    if rejection is not None:
        return rejection
    assert safety is not None

    order_notional = _order_notional(intent=intent)
    safety_checks: tuple[tuple[str, bool, str], ...] = (
        (
            "safety_ownership",
            safety.organization_id == intent.organization_id
            and safety.owner_user_id == intent.owner_user_id
            and safety.exchange_connection_id == intent.exchange_connection_id,
            "execution_safety_scope_mismatch",
        ),
        (
            "mode",
            safety.mode == connection.environment and safety.mode in {"testnet", "mainnet"},
            "execution_mode_not_approved",
        ),
        (
            "risk",
            safety.risk_allows_submit
            and safety.updated_at <= now < safety.risk_valid_until,
            "execution_risk_denied_or_stale",
        ),
        (
            "account_snapshot",
            risk_snapshot.account_snapshot_fresh,
            "execution_account_snapshot_not_fresh",
        ),
        (
            "config_guard",
            risk_snapshot.config_guard_verified,
            "execution_account_config_not_verified",
        ),
        (
            "order_notional",
            order_notional is not None
            and order_notional > 0
            and order_notional <= safety.max_order_notional,
            "execution_order_notional_limit_exceeded",
        ),
        (
            "daily_notional",
            order_notional is not None
            and risk_snapshot.daily_notional_used + order_notional
            <= safety.daily_notional_limit,
            "execution_daily_notional_limit_exceeded",
        ),
        (
            "account_exposure",
            order_notional is not None
            and risk_snapshot.account_exposure_complete
            and risk_snapshot.account_exposure_notional + order_notional
            <= safety.max_account_exposure_notional,
            "execution_account_exposure_limit_exceeded",
        ),
        ("kill_switch", not kill_switch_active, "execution_kill_switch_active"),
        ("provider", provider is not None, "execution_provider_not_allowlisted"),
    )
    rejection = _first_rejection(query=query, checks=safety_checks)
    if rejection is not None:
        return rejection
    assert provider is not None

    provider_checks: tuple[tuple[str, bool, str], ...] = (
        (
            "provider_trust",
            provider.enabled
            and provider.provider_kind in {"core", "verified"}
            and provider.order_submit_capability,
            "execution_provider_untrusted",
        ),
        (
            "provider_identity",
            provider.provider_id == adapter.provider_id
            and provider.provider_version == adapter.provider_version
            and provider.provider_kind == adapter.provider_kind
            and provider.exchange_name == adapter.exchange_name
            and provider.exchange_name == connection.exchange_name,
            "execution_provider_identity_mismatch",
        ),
        (
            "provider_revision",
            len(adapter.revision_hash) == 64
            and provider.revision_hash == adapter.revision_hash,
            "execution_provider_identity_mismatch",
        ),
        (
            "secret_reference",
            len(connection.secret_reference_hash) == 64
            and safety.secret_reference_hash == connection.secret_reference_hash,
            "execution_secret_reference_unbound",
        ),
        (
            "account_revision",
            len(connection.account_revision_hash) == 64
            and safety.account_revision_hash == connection.account_revision_hash,
            "execution_account_revision_changed",
        ),
        (
            "intent_identity",
            len(intent.canonical_intent_hash) == 64,
            "canonical_intent_identity_missing",
        ),
        (
            "intent_expiry",
            not _intent_expired(intent=intent, now=now),
            "execution_intent_expired",
        ),
    )
    rejection = _first_rejection(query=query, checks=provider_checks)
    if rejection is not None:
        return rejection

    if safety.mode == "mainnet":
        approval_rejection = _evaluate_mainnet_approval(
            query=query,
            safety=safety,
            provider=provider,
            approval=approval,
        )
        if approval_rejection is not None:
            return approval_rejection

    return ExecutionSubmitGuardDecision(
        status="accepted",
        reason="execution_submit_guard_accepted",
        check_name="all",
        phase=query.phase,
        evaluated_at=now,
        approval_id=approval.approval_id if approval is not None else None,
        metadata={
            "mode": safety.mode,
            "provider_id": provider.provider_id,
            "provider_kind": provider.provider_kind,
            "provider_revision_hash": provider.revision_hash,
            "risk_revision_hash": safety.risk_revision_hash,
            "account_revision_hash": safety.account_revision_hash,
        },
    )


def validate_mainnet_approval(
    *, approval: ExecutionMainnetApproval, principal: ExecutionPolicyPrincipal, now: datetime
) -> None:
    if approval.owner_user_id != principal.user_id:
        raise ExecutionGatewayPolicyError(reason="mainnet_owner_approval_required")
    if approval.recent_auth_session_id != principal.session_id:
        raise ExecutionGatewayPolicyError(reason="mainnet_recent_auth_session_mismatch")
    if approval.recent_auth_at != principal.authenticated_at:
        raise ExecutionGatewayPolicyError(reason="mainnet_recent_auth_timestamp_mismatch")
    if approval.approved_at != now:
        raise ExecutionGatewayPolicyError(reason="mainnet_approval_timestamp_mismatch")
    if approval.recent_auth_at > now or now - approval.recent_auth_at > MAINNET_RECENT_AUTH_WINDOW:
        raise ExecutionGatewayPolicyError(reason="mainnet_recent_auth_required")
    if approval.expires_at <= now or approval.expires_at - now > MAINNET_APPROVAL_MAX_TTL:
        raise ExecutionGatewayPolicyError(reason="mainnet_approval_ttl_invalid")
    if approval.revoked_at is not None:
        raise ExecutionGatewayPolicyError(reason="mainnet_approval_already_revoked")


def _evaluate_mainnet_approval(
    *,
    query: ExecutionSubmitGuardQuery,
    safety: ExecutionAccountSafetyState,
    provider: ExecutionProviderRegistration,
    approval: ExecutionMainnetApproval | None,
) -> ExecutionSubmitGuardDecision | None:
    checks: tuple[tuple[str, bool, str], ...] = (
        ("mainnet_approval", approval is not None, "mainnet_owner_approval_required"),
    )
    rejection = _first_rejection(query=query, checks=checks)
    if rejection is not None:
        return rejection
    assert approval is not None
    matches = (
        approval.organization_id == query.intent.organization_id
        and approval.owner_user_id == query.intent.owner_user_id
        and approval.exchange_connection_id == query.intent.exchange_connection_id
        and approval.exchange_name == query.connection.exchange_name
        and approval.market_type == query.intent.market_type
        and approval.provider_id == provider.provider_id
    )
    checks = (
        ("mainnet_approval_scope", matches, "mainnet_approval_scope_mismatch"),
        ("mainnet_approval_revocation", approval.revoked_at is None, "mainnet_approval_revoked"),
        (
            "mainnet_approval_expiry",
            approval.approved_at <= query.evaluated_at < approval.expires_at,
            "mainnet_approval_expired",
        ),
        (
            "mainnet_recent_auth",
            approval.recent_auth_at <= query.evaluated_at
            and query.evaluated_at - approval.recent_auth_at <= MAINNET_RECENT_AUTH_WINDOW,
            "mainnet_recent_auth_required",
        ),
        (
            "mainnet_risk_revision",
            approval.risk_revision_hash == safety.risk_revision_hash,
            "mainnet_approval_invalidated_by_risk_change",
        ),
        (
            "mainnet_account_revision",
            approval.account_revision_hash == safety.account_revision_hash,
            "mainnet_approval_invalidated_by_account_change",
        ),
        (
            "mainnet_provider_revision",
            approval.provider_revision_hash == provider.revision_hash,
            "mainnet_approval_invalidated_by_provider_change",
        ),
    )
    return _first_rejection(query=query, checks=checks)


def _intent_expired(*, intent: ExecutionIntent, now: datetime) -> bool:
    raw = intent.constraints.get("expires_at")
    if raw is None:
        return False
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return now >= parsed


def _order_notional(*, intent: ExecutionIntent) -> Decimal | None:
    if intent.quote_notional is not None:
        return intent.quote_notional
    if intent.quantity is not None and intent.limit_price is not None:
        return intent.quantity * intent.limit_price
    return None


def _first_rejection(
    *,
    query: ExecutionSubmitGuardQuery,
    checks: tuple[tuple[str, bool, str], ...],
) -> ExecutionSubmitGuardDecision | None:
    for check_name, allowed, reason in checks:
        if not allowed:
            return ExecutionSubmitGuardDecision(
                status="rejected",
                reason=reason,
                check_name=check_name,
                phase=query.phase,
                evaluated_at=query.evaluated_at,
            )
    return None
