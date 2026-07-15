from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID = UUID("00000000-0000-0000-0000-00000000a007")

CapitalReservationState = Literal["reserved", "released", "rejected", "stale_requires_repair"]
PaperOrderStatus = Literal["filled", "rejected"]
PaperSide = Literal["buy", "sell"]


class CapitalReservationBlockedError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CapitalReservation:
    reservation_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_connection_id: UUID
    strategy_id: UUID
    live_profile_id: UUID | None
    strategy_run_id: UUID
    asset: str
    requested_amount: Decimal
    reserved_amount: Decimal
    state: CapitalReservationState
    source_account_snapshot_id: UUID | None
    acquired_at: datetime
    released_at: datetime | None
    reason: str
    fee_model: str
    funding_model: str
    pnl_complete: bool


@dataclass(frozen=True, slots=True)
class PaperOrder:
    paper_order_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    strategy_id: UUID
    strategy_run_id: UUID
    reservation_id: UUID
    source_signal_id: UUID
    instrument_key: str
    market_type: str
    side: PaperSide
    order_type: Literal["market"]
    quantity: Decimal
    quote_notional: Decimal
    reference_price: Decimal
    status: PaperOrderStatus
    reason: str
    created_at: datetime
    source_event_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class PaperFill:
    paper_fill_id: UUID
    paper_order_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    strategy_id: UUID
    strategy_run_id: UUID
    instrument_key: str
    side: PaperSide
    quantity: Decimal
    fill_price: Decimal
    quote_notional: Decimal
    fee_amount: Decimal
    fee_asset: str
    funding_amount: Decimal
    funding_asset: str
    filled_at: datetime


@dataclass(frozen=True, slots=True)
class StrategyPaperAccountingSnapshot:
    accounting_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    strategy_id: UUID
    strategy_run_id: UUID
    reservation_id: UUID
    paper_fill_id: UUID
    instrument_key: str
    market_type: str
    position_quantity: Decimal
    average_entry_price: Decimal | None
    reserved_budget: Decimal
    cash_balance: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fee_total: Decimal
    funding_total: Decimal
    fee_model: str
    funding_model: str
    pnl_complete: bool
    completeness_reason: str
    created_at: datetime
