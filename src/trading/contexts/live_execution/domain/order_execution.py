from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal, Mapping
from uuid import UUID

from trading.contexts.live_execution.domain.execution_source import ExecutionIntent
from trading.shared_kernel.primitives import OrganizationId, UserId

ExchangeExecutionOrderStatus = Literal[
    "guard_rejected",
    "submit_pending",
    "submitted",
    "status_checked",
    "cancelled",
    "adapter_error",
    "unknown",
    "reconciled",
]
ExchangePrivateStreamStatus = Literal["ready", "degraded", "not_ready"]
ExecutionOrderEventType = Literal[
    "guard_rejected",
    "submit_pending",
    "submitted",
    "status_checked",
    "cancelled",
    "adapter_error",
    "private_stream_backfill",
    "reconciled",
]
ExecutionReconciliationStatus = Literal["matched", "mismatch", "pending", "failed"]
ExecutionPitrDrillStatus = Literal["verified", "blocked", "failed"]


@dataclass(frozen=True, repr=False, slots=True)
class ExchangeExecutionCredential:
    api_key: str
    api_secret: str
    passphrase: str | None = None

    def __repr__(self) -> str:
        return "ExchangeExecutionCredential(<redacted>)"


@dataclass(frozen=True, slots=True)
class ExchangeExecutionConnection:
    connection_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    connection_readiness: str
    effective_capability: str
    secret_reference_hash: str
    account_revision_hash: str
    credential: ExchangeExecutionCredential


def execution_secret_reference_hash(
    *, connection_id: UUID, credential_version_id: UUID
) -> str:
    """Bind submit authorization to a non-sensitive credential-version reference."""

    parts = (
        "io.roehub.execution-credential-reference/v1",
        str(connection_id),
        str(credential_version_id),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def execution_account_revision_hash(
    *,
    connection_id: UUID,
    credential_version_id: UUID,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    exchange_name: str,
    market_type: str,
    environment: str,
    connection_readiness: str,
    effective_capability: str,
    updated_at: datetime,
) -> str:
    """Fingerprint material, non-sensitive account state used at submit time."""

    parts = (
        "io.roehub.execution-account-revision/v1",
        str(connection_id),
        str(credential_version_id),
        str(organization_id),
        str(owner_user_id),
        exchange_name,
        market_type,
        environment,
        connection_readiness,
        effective_capability,
        updated_at.isoformat(),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExchangeOrderCommand:
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_connection_id: UUID
    exchange_name: str
    environment: str
    market_type: str
    instrument_key: str
    side: str
    order_type: str
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None
    client_order_id: str
    constraints: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_intent(
        cls,
        *,
        intent: ExecutionIntent,
        exchange_name: str,
        environment: str,
        client_order_id: str,
    ) -> "ExchangeOrderCommand":
        return cls(
            intent_id=intent.intent_id,
            organization_id=intent.organization_id,
            owner_user_id=intent.owner_user_id,
            exchange_connection_id=intent.exchange_connection_id,
            exchange_name=exchange_name,
            environment=environment,
            market_type=intent.market_type,
            instrument_key=intent.instrument_key,
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            quote_notional=intent.quote_notional,
            limit_price=intent.limit_price,
            client_order_id=client_order_id,
            constraints=intent.constraints,
        )


@dataclass(frozen=True, slots=True)
class ExecutionSubmitClaim:
    order: "ExchangeExecutionOrderRecord"
    claim_id: UUID
    acquired: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ExchangeOrderSubmitResult:
    exchange_order_id: str
    exchange_status: str
    submitted_at: datetime
    latency_ms: float
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExchangeOrderStatusResult:
    exchange_order_id: str
    exchange_status: str
    checked_at: datetime
    latency_ms: float
    metadata: Mapping[str, int | float | str]
    lookup_outcome: Literal["found", "confirmed_absent", "unknown"] = "found"
    fills: tuple["ExecutionFillFact", ...] = ()
    funding_events: tuple["ExecutionFundingFact", ...] = ()


@dataclass(frozen=True, slots=True)
class ExchangeOrderCancelResult:
    exchange_order_id: str
    exchange_status: str
    cancelled_at: datetime
    latency_ms: float
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExchangePrivateStreamSession:
    session_id: UUID
    organization_id: OrganizationId
    exchange_name: str
    environment: str
    market_type: str
    status: ExchangePrivateStreamStatus
    status_reason: str
    opened_at: datetime
    keepalive_at: datetime | None
    expires_at: datetime | None
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExecutionFillFact:
    provider_trade_id: str
    price: Decimal
    quantity: Decimal
    fee_amount: Decimal
    fee_asset: str
    filled_at: datetime
    liquidity: str | None = None
    metadata: Mapping[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_trade_id.strip():
            raise ValueError("ExecutionFillFact.provider_trade_id must be non-empty")
        if self.quantity <= 0:
            raise ValueError("ExecutionFillFact.quantity must be > 0")
        if self.price <= 0:
            raise ValueError("ExecutionFillFact.price must be > 0")
        if self.fee_amount < 0:
            raise ValueError("ExecutionFillFact.fee_amount must be >= 0")
        if not self.fee_asset.strip():
            raise ValueError("ExecutionFillFact.fee_asset must be non-empty")


@dataclass(frozen=True, slots=True)
class ExecutionFundingFact:
    provider_event_id: str
    amount: Decimal
    asset: str
    funding_at: datetime
    reason: str
    metadata: Mapping[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_event_id.strip():
            raise ValueError("ExecutionFundingFact.provider_event_id must be non-empty")
        if not self.asset.strip():
            raise ValueError("ExecutionFundingFact.asset must be non-empty")
        if not self.reason.strip():
            raise ValueError("ExecutionFundingFact.reason must be non-empty")


@dataclass(frozen=True, slots=True)
class ExecutionOrderEvent:
    event_id: UUID
    order_id: UUID
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    event_type: ExecutionOrderEventType
    status: str
    reason: str
    provider_order_id: str | None
    provider_event_id: str | None
    observed_at: datetime
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    fill_id: UUID
    order_id: UUID
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    provider_trade_id: str
    price: Decimal
    quantity: Decimal
    fee_amount: Decimal
    fee_asset: str
    filled_at: datetime
    liquidity: str | None
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExecutionFundingEvent:
    funding_event_id: UUID
    order_id: UUID
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    provider_event_id: str
    amount: Decimal
    asset: str
    funding_at: datetime
    reason: str
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExecutionReconciliationRun:
    reconciliation_run_id: UUID
    order_id: UUID
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_name: str
    environment: str
    status: ExecutionReconciliationStatus
    reason: str
    local_status: str
    provider_status: str | None
    fill_count: int
    funding_event_count: int
    started_at: datetime
    completed_at: datetime
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExecutionLedgerRetentionPolicy:
    policy_name: str
    table_name: str
    partition_key: str
    retention_days: int
    archive_before_purge: bool
    pitr_required: bool
    checked_at: datetime
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ExecutionLedgerPitrDrill:
    drill_id: UUID
    target_time: datetime
    status: ExecutionPitrDrillStatus
    reason: str
    verified_at: datetime
    row_counts: Mapping[str, int]
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionOrderRecord:
    order_id: UUID
    intent_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_connection_id: UUID
    exchange_name: str
    environment: str
    market_type: str
    instrument_key: str
    side: str
    order_type: str
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None
    client_order_id: str
    exchange_order_id: str | None
    status: ExchangeExecutionOrderStatus
    status_reason: str
    submitted_at: datetime | None
    cancel_requested_at: datetime | None
    cancelled_at: datetime | None
    last_checked_at: datetime | None
    adapter_attempt_count: int
    latency_ms: float | None
    metadata: Mapping[str, int | float | str]
    created_at: datetime
    updated_at: datetime
    submit_claim_id: UUID | None = None
    submit_claimed_at: datetime | None = None
    submit_claim_expires_at: datetime | None = None
    submit_guard_audit_event_id: UUID | None = None
    mainnet_approval_id: UUID | None = None
