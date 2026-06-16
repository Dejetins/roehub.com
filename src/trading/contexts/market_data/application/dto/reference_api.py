from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class EnabledMarketReference:
    """
    Read-model row for one enabled market in reference API use-cases.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
      - apps/api/dto/market_data_reference.py
    """

    market_id: MarketId
    exchange_name: str
    market_type: str
    market_code: str

    def __post_init__(self) -> None:
        """
        Validate enabled market read-model invariants.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Text fields are non-empty after trimming.

        Errors/Exceptions:
        - Raises `ValueError` when one of fields is blank.

        Side effects:
        - None.
        """
        if not self.exchange_name.strip():
            raise ValueError("EnabledMarketReference.exchange_name must be non-empty")
        if not self.market_type.strip():
            raise ValueError("EnabledMarketReference.market_type must be non-empty")
        if not self.market_code.strip():
            raise ValueError("EnabledMarketReference.market_code must be non-empty")


BTCUSDT_MARKET_READINESS_SYMBOL = "BTCUSDT"
BTCUSDT_MARKET_READINESS_MARKETS: tuple[tuple[str, str, str], ...] = (
    ("binance", "spot", "binance:spot"),
    ("binance", "futures", "binance:futures"),
    ("bybit", "spot", "bybit:spot"),
    ("bybit", "futures", "bybit:futures"),
)

BTCUSDTReferenceState = Literal["ready", "missing", "disabled", "incomplete"]
BTCUSDTMarketReadinessState = Literal["ready", "missing", "stale", "pending", "blocked"]


@dataclass(frozen=True, slots=True)
class BTCUSDTMarketReferenceSnapshot:
    market_id: MarketId | None
    exchange_name: str
    market_type: str
    market_code: str
    market_enabled: bool
    symbol: str
    status: str | None
    is_tradable: int | None
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None

    @property
    def instrument_key(self) -> str:
        return f"{self.exchange_name}:{self.market_type}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class BTCUSDTStreamReadinessSnapshot:
    state: Literal["ready", "missing", "stale", "pending"]
    reason_code: str
    stream_name: str
    stream_length: int | None
    last_message_id: str | None
    last_observed_at: datetime | None
    age_seconds: int | None


@dataclass(frozen=True, slots=True)
class BTCUSDTMarketReadinessRow:
    market_id: MarketId | None
    exchange_name: str
    market_type: str
    market_code: str
    symbol: str
    instrument_key: str
    readiness_state: BTCUSDTMarketReadinessState
    reason_codes: tuple[str, ...]
    reference_state: BTCUSDTReferenceState
    reference_reason_codes: tuple[str, ...]
    market_enabled: bool
    status: str | None
    is_tradable: int | None
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None
    stream_state: Literal["ready", "missing", "stale", "pending"]
    stream_reason_code: str
    stream_name: str
    stream_length: int | None
    stream_last_message_id: str | None
    stream_last_observed_at: datetime | None
    stream_age_seconds: int | None
    checked_at: datetime


@dataclass(frozen=True, slots=True)
class BTCUSDTMarketReadinessReport:
    symbol: str
    freshness_threshold_seconds: int
    rows: tuple[BTCUSDTMarketReadinessRow, ...]
    checked_at: datetime
