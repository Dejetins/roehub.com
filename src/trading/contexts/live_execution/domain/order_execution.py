from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Mapping
from uuid import UUID

from trading.contexts.live_execution.domain.execution_source import ExecutionIntent
from trading.shared_kernel.primitives import UserId

ExchangeExecutionOrderStatus = Literal[
    "guard_rejected",
    "submit_pending",
    "submitted",
    "status_checked",
    "cancelled",
    "adapter_error",
    "unknown",
]
ExchangePrivateStreamStatus = Literal["ready", "degraded", "not_ready"]


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
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    connection_readiness: str
    effective_capability: str
    credential: ExchangeExecutionCredential


@dataclass(frozen=True, slots=True)
class ExchangeOrderCommand:
    intent_id: UUID
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
        )


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
class ExchangeExecutionOrderRecord:
    order_id: UUID
    intent_id: UUID
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
