from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialPlaintext,
)


@dataclass(frozen=True, slots=True)
class ExchangeBalanceState:
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal | None


@dataclass(frozen=True, slots=True)
class ExchangePositionState:
    instrument_key: str
    side: str
    quantity: Decimal
    entry_price: Decimal | None = None
    leverage: Decimal | None = None
    margin_mode: str | None = None
    position_mode: str | None = None


@dataclass(frozen=True, slots=True)
class ExchangeOpenOrderState:
    instrument_key: str
    exchange_order_ref: str
    side: str
    order_type: str
    quantity: Decimal
    price: Decimal | None
    status: str


@dataclass(frozen=True, slots=True)
class ExchangeInstrumentFilterState:
    instrument_key: str
    tick_size: Decimal | None = None
    step_size: Decimal | None = None
    min_qty: Decimal | None = None
    min_notional: Decimal | None = None
    max_leverage: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExchangeAccountStateReadRequest:
    exchange_name: str
    market_type: str
    environment: str
    credential: ExchangeCredentialPlaintext
    instrument_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExchangeAccountStateReadResult:
    exchange_name: str
    market_type: str
    environment: str
    account_mode: str
    balances: tuple[ExchangeBalanceState, ...]
    positions: tuple[ExchangePositionState, ...]
    open_orders: tuple[ExchangeOpenOrderState, ...]
    instrument_filters: tuple[ExchangeInstrumentFilterState, ...]
    observed_at: datetime
    source_hash: str
    sync_status: str = "fresh"
    sync_reason: str = "account_state_read_ok"


class ExchangeAccountStateReader(Protocol):
    requires_plaintext: bool

    def read_account_state(
        self,
        *,
        request: ExchangeAccountStateReadRequest,
        now: datetime,
    ) -> ExchangeAccountStateReadResult: ...
