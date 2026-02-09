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
class InstrumentRefEnrichmentUpsert:
    market_id: MarketId
    symbol: Symbol
    status: str
    is_tradable: int
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None
    updated_at: UtcTimestamp

    def __post_init__(self) -> None:
        """
        Validate enrichment upsert invariants for storage safety.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Status/is_tradable keep the same constraints as whitelist sync rows.
        - Numeric enrichment fields, when present, are strictly positive.

        Errors/Exceptions:
        - Raises `ValueError` on invalid field values.

        Side effects:
        - None.
        """
        if self.status not in ("ENABLED", "DISABLED"):
            raise ValueError(
                "InstrumentRefEnrichmentUpsert.status must be ENABLED|DISABLED, "
                f"got {self.status!r}"
            )
        if self.is_tradable not in (0, 1):
            raise ValueError(
                "InstrumentRefEnrichmentUpsert.is_tradable must be 0|1, "
                f"got {self.is_tradable!r}"
            )
        if self.base_asset is not None and not self.base_asset.strip():
            raise ValueError("InstrumentRefEnrichmentUpsert.base_asset must not be blank")
        if self.quote_asset is not None and not self.quote_asset.strip():
            raise ValueError("InstrumentRefEnrichmentUpsert.quote_asset must not be blank")
        if self.price_step is not None and self.price_step <= 0:
            raise ValueError("InstrumentRefEnrichmentUpsert.price_step must be > 0")
        if self.qty_step is not None and self.qty_step <= 0:
            raise ValueError("InstrumentRefEnrichmentUpsert.qty_step must be > 0")
        if self.min_notional is not None and self.min_notional <= 0:
            raise ValueError("InstrumentRefEnrichmentUpsert.min_notional must be > 0")


@dataclass(frozen=True, slots=True)
class InstrumentRefEnrichmentSnapshot:
    """
    Latest persisted enrichment state for one instrument in `ref_instruments`.

    Parameters:
    - status: current status field (`ENABLED`/`DISABLED`).
    - is_tradable: current tradable flag (`0`/`1`).
    - base_asset: current base asset value.
    - quote_asset: current quote asset value.
    - price_step: current price step value.
    - qty_step: current quantity step value.
    - min_notional: current min notional value.
    """

    status: str
    is_tradable: int
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None


@dataclass(frozen=True, slots=True)
class ExchangeInstrumentMetadata:
    instrument_id: InstrumentId
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None

    def __post_init__(self) -> None:
        """
        Validate normalized exchange metadata values.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Blank asset strings are not allowed.
        - Numeric values, when present, are strictly positive.

        Errors/Exceptions:
        - Raises `ValueError` on invalid values.

        Side effects:
        - None.
        """
        if self.base_asset is not None and not self.base_asset.strip():
            raise ValueError("ExchangeInstrumentMetadata.base_asset must not be blank")
        if self.quote_asset is not None and not self.quote_asset.strip():
            raise ValueError("ExchangeInstrumentMetadata.quote_asset must not be blank")
        if self.price_step is not None and self.price_step <= 0:
            raise ValueError("ExchangeInstrumentMetadata.price_step must be > 0")
        if self.qty_step is not None and self.qty_step <= 0:
            raise ValueError("ExchangeInstrumentMetadata.qty_step must be > 0")
        if self.min_notional is not None and self.min_notional <= 0:
            raise ValueError("ExchangeInstrumentMetadata.min_notional must be > 0")


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
