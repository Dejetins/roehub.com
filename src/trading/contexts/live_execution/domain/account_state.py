from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from trading.shared_kernel.primitives import UserId

AccountReadinessStatus = Literal["fresh", "stale", "degraded", "config_mismatch"]
ConfigGuardStatus = Literal["verified", "mismatch", "degraded"]


@dataclass(frozen=True, slots=True)
class ExchangeBalanceSnapshot:
    asset: str
    free: Decimal
    locked: Decimal = Decimal("0")
    total: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExchangePositionSnapshot:
    instrument_key: str
    side: Literal["long", "short", "net"]
    quantity: Decimal
    entry_price: Decimal | None = None
    leverage: Decimal | None = None
    margin_mode: str | None = None
    position_mode: str | None = None


@dataclass(frozen=True, slots=True)
class ExchangeOpenOrderSnapshot:
    instrument_key: str
    exchange_order_ref: str
    side: Literal["buy", "sell"]
    order_type: str
    quantity: Decimal
    price: Decimal | None
    status: str


@dataclass(frozen=True, slots=True)
class ExchangeInstrumentFilterSnapshot:
    instrument_key: str
    tick_size: Decimal | None = None
    step_size: Decimal | None = None
    min_qty: Decimal | None = None
    min_notional: Decimal | None = None
    max_leverage: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExchangeAccountProjection:
    account_snapshot_id: UUID
    owner_user_id: UserId
    exchange_connection_id: UUID
    exchange_name: str
    market_type: str
    environment: str
    account_mode: str
    balances: tuple[ExchangeBalanceSnapshot, ...]
    positions: tuple[ExchangePositionSnapshot, ...]
    open_orders: tuple[ExchangeOpenOrderSnapshot, ...]
    instrument_filters: tuple[ExchangeInstrumentFilterSnapshot, ...]
    source_hash: str
    observed_at: datetime
    synced_at: datetime
    sync_status: Literal["fresh", "degraded"] = "fresh"
    sync_reason: str = "read_only_sync_ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_seconds(self, *, now: datetime) -> int:
        observed = _ensure_aware(self.observed_at)
        current = _ensure_aware(now)
        return max(0, int((current - observed).total_seconds()))


@dataclass(frozen=True, slots=True)
class ExpectedInstrumentConfig:
    instrument_key: str
    market_type: str
    side: Literal["long", "short"] | None = None
    expected_margin_mode: str | None = None
    expected_position_mode: str | None = None
    required_leverage: Decimal | None = None
    order_notional: Decimal | None = None
    required_balance_asset: str | None = None
    min_notional: Decimal | None = None
    tick_size: Decimal | None = None
    step_size: Decimal | None = None


@dataclass(frozen=True, slots=True)
class AccountConfigGuardResult:
    config_guard_result_id: UUID
    account_snapshot_id: UUID | None
    owner_user_id: UserId
    exchange_connection_id: UUID
    instrument_key: str
    market_type: str
    status: ConfigGuardStatus
    reason_codes: tuple[str, ...]
    checked_at: datetime
    requirement: ExpectedInstrumentConfig

    @property
    def mismatch(self) -> bool:
        return self.status == "mismatch"


@dataclass(frozen=True, slots=True)
class AccountProjectionReadiness:
    status: AccountReadinessStatus
    reason_codes: tuple[str, ...]
    exchange_connection_id: UUID | None
    instrument_key: str | None
    market_type: str | None
    account_snapshot_id: UUID | None
    config_guard_result_id: UUID | None
    age_seconds: int | None
    source_hash: str | None
    checked_at: datetime

    @property
    def ready_for_risk(self) -> bool:
        return self.status == "fresh"


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
