from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives.instrument_id import InstrumentId
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol
from trading.shared_kernel.primitives.utc_timestamp import UtcTimestamp


@dataclass(frozen=True, slots=True)
class WhitelistInstrumentRow:
    instrument_id: InstrumentId
    is_enabled: bool


@dataclass(frozen=True, slots=True)
class InstrumentRefUpsert:
    market_id: MarketId
    symbol: Symbol
    status: str  # 'ENABLED' | 'DISABLED'
    is_tradable: int  # 0 | 1
    updated_at: UtcTimestamp

    def __post_init__(self) -> None:
        if self.status not in ("ENABLED", "DISABLED"):
            raise ValueError(f"InstrumentRefUpsert.status must be ENABLED|DISABLED, got {self.status!r}")  # noqa: E501
        if self.is_tradable not in (0, 1):
            raise ValueError(f"InstrumentRefUpsert.is_tradable must be 0|1, got {self.is_tradable!r}")  # noqa: E501


@dataclass(frozen=True, slots=True)
class RefMarketRow:
    market_id: MarketId
    exchange_name: str
    market_type: str
    market_code: str
    is_enabled: int  # 0|1
    count_symbols: int
    updated_at: UtcTimestamp

    def __post_init__(self) -> None:
        if self.is_enabled not in (0, 1):
            raise ValueError(f"RefMarketRow.is_enabled must be 0|1, got {self.is_enabled!r}")
        if not self.exchange_name.strip():
            raise ValueError("RefMarketRow.exchange_name must be non-empty")
        if not self.market_type.strip():
            raise ValueError("RefMarketRow.market_type must be non-empty")
        if not self.market_code.strip():
            raise ValueError("RefMarketRow.market_code must be non-empty")
        if self.count_symbols < 0:
            raise ValueError("RefMarketRow.count_symbols must be >= 0")
