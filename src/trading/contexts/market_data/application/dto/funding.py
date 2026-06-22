from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp


@dataclass(frozen=True, slots=True)
class FundingInstrument:
    instrument_id: InstrumentId
    instrument_key: str
    exchange: str
    market_type: str
    status: str
    is_tradable: int
    base_asset: str | None
    quote_asset: str | None
    funding_interval_minutes: int | None
    funding_interval_source: str | None
    funding_cap: float | None
    funding_floor: float | None
    updated_at: UtcTimestamp

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("FundingInstrument requires instrument_id")
        _require_non_empty("FundingInstrument.instrument_key", self.instrument_key)
        _require_non_empty("FundingInstrument.exchange", self.exchange)
        _require_non_empty("FundingInstrument.market_type", self.market_type)
        _require_non_empty("FundingInstrument.status", self.status)
        if self.is_tradable not in (0, 1):
            raise ValueError("FundingInstrument.is_tradable must be 0 or 1")
        _require_optional_non_blank("FundingInstrument.base_asset", self.base_asset)
        _require_optional_non_blank("FundingInstrument.quote_asset", self.quote_asset)
        _require_optional_non_blank(
            "FundingInstrument.funding_interval_source",
            self.funding_interval_source,
        )
        _require_optional_positive_int(
            "FundingInstrument.funding_interval_minutes",
            self.funding_interval_minutes,
        )


@dataclass(frozen=True, slots=True)
class FundingRateRecord:
    instrument_id: InstrumentId
    instrument_key: str
    funding_time: UtcTimestamp
    funding_rate: float
    funding_interval_minutes: int
    funding_interval_source: str
    source: str
    ingested_at: UtcTimestamp
    ingest_id: str | None = None
    mark_price: float | None = None
    bybit_category: str | None = None

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("FundingRateRecord requires instrument_id")
        _require_non_empty("FundingRateRecord.instrument_key", self.instrument_key)
        _require_positive_int(
            "FundingRateRecord.funding_interval_minutes",
            self.funding_interval_minutes,
        )
        _require_non_empty(
            "FundingRateRecord.funding_interval_source",
            self.funding_interval_source,
        )
        _require_non_empty("FundingRateRecord.source", self.source)
        _require_optional_non_blank("FundingRateRecord.ingest_id", self.ingest_id)
        _require_optional_non_blank("FundingRateRecord.bybit_category", self.bybit_category)


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_optional_non_blank(name: str, value: str | None) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{name} must be non-empty when provided")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    _require_positive_int(name, value)
