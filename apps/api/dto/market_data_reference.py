"""
Pydantic API models and converters for Market Data reference API v1 endpoints.

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReadinessReport,
    BTCUSDTMarketReadinessRow,
    EnabledMarketReference,
)
from trading.shared_kernel.primitives import InstrumentId


class MarketDataMarketItemResponse(BaseModel):
    """
    API item payload for one enabled market.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/routes/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
      - migrations/clickhouse/market_data_ddl.sql
    """

    market_id: int
    exchange_name: str
    market_type: str
    market_code: str


class MarketDataMarketsResponse(BaseModel):
    """
    API response wrapper for `GET /market-data/markets`.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/routes/market_data_reference.py
      - apps/api/dto/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
    """

    items: list[MarketDataMarketItemResponse]


class MarketDataInstrumentItemResponse(BaseModel):
    """
    API item payload for one market instrument tuple.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/routes/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
      - src/trading/shared_kernel/primitives/instrument_id.py
    """

    market_id: int
    symbol: str


class MarketDataInstrumentsResponse(BaseModel):
    """
    API response wrapper for `GET /market-data/instruments`.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/routes/market_data_reference.py
      - apps/api/dto/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
    """

    items: list[MarketDataInstrumentItemResponse]


class BTCUSDTMarketReadinessItemResponse(BaseModel):
    market_id: int | None
    exchange_name: str
    market_type: str
    market_code: str
    symbol: str
    instrument_key: str
    readiness_state: str
    reason_codes: list[str]
    reference_state: str
    reference_reason_codes: list[str]
    market_enabled: bool
    status: str | None
    is_tradable: int | None
    base_asset: str | None
    quote_asset: str | None
    price_step: float | None
    qty_step: float | None
    min_notional: float | None
    stream_state: str
    stream_reason_code: str
    stream_name: str
    stream_length: int | None
    stream_last_message_id: str | None
    stream_last_observed_at: str | None
    stream_age_seconds: int | None
    checked_at: str


class BTCUSDTMarketReadinessResponse(BaseModel):
    symbol: str
    freshness_threshold_seconds: int
    items: list[BTCUSDTMarketReadinessItemResponse]
    checked_at: str


def build_market_data_markets_response(
    *,
    markets: Sequence[EnabledMarketReference],
) -> MarketDataMarketsResponse:
    """
    Convert market_data application read-models into API response payload.

    Parameters:
    - markets: enabled markets returned by application use-case.

    Returns:
    - `MarketDataMarketsResponse` with deterministic item mapping.

    Assumptions/Invariants:
    - Input order is already deterministic in use-case layer.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return MarketDataMarketsResponse(
        items=[
            MarketDataMarketItemResponse(
                market_id=market.market_id.value,
                exchange_name=market.exchange_name,
                market_type=market.market_type,
                market_code=market.market_code,
            )
            for market in markets
        ]
    )


def build_market_data_instruments_response(
    *,
    instruments: Sequence[InstrumentId],
) -> MarketDataInstrumentsResponse:
    """
    Convert instrument id rows into API response payload.

    Parameters:
    - instruments: instrument ids returned by application use-case.

    Returns:
    - `MarketDataInstrumentsResponse` with deterministic item mapping.

    Assumptions/Invariants:
    - Symbols are normalized by shared-kernel `Symbol` primitive.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return MarketDataInstrumentsResponse(
        items=[
            MarketDataInstrumentItemResponse(
                market_id=instrument.market_id.value,
                symbol=str(instrument.symbol),
            )
            for instrument in instruments
        ]
    )


def build_btcusdt_market_readiness_response(
    *,
    report: BTCUSDTMarketReadinessReport,
) -> BTCUSDTMarketReadinessResponse:
    return BTCUSDTMarketReadinessResponse(
        symbol=report.symbol,
        freshness_threshold_seconds=report.freshness_threshold_seconds,
        items=[_build_btcusdt_market_readiness_item(row=row) for row in report.rows],
        checked_at=report.checked_at.isoformat(),
    )


def _build_btcusdt_market_readiness_item(
    *,
    row: BTCUSDTMarketReadinessRow,
) -> BTCUSDTMarketReadinessItemResponse:
    return BTCUSDTMarketReadinessItemResponse(
        market_id=row.market_id.value if row.market_id is not None else None,
        exchange_name=row.exchange_name,
        market_type=row.market_type,
        market_code=row.market_code,
        symbol=row.symbol,
        instrument_key=row.instrument_key,
        readiness_state=row.readiness_state,
        reason_codes=list(row.reason_codes),
        reference_state=row.reference_state,
        reference_reason_codes=list(row.reference_reason_codes),
        market_enabled=row.market_enabled,
        status=row.status,
        is_tradable=row.is_tradable,
        base_asset=row.base_asset,
        quote_asset=row.quote_asset,
        price_step=row.price_step,
        qty_step=row.qty_step,
        min_notional=row.min_notional,
        stream_state=row.stream_state,
        stream_reason_code=row.stream_reason_code,
        stream_name=row.stream_name,
        stream_length=row.stream_length,
        stream_last_message_id=row.stream_last_message_id,
        stream_last_observed_at=(
            row.stream_last_observed_at.isoformat()
            if row.stream_last_observed_at is not None
            else None
        ),
        stream_age_seconds=row.stream_age_seconds,
        checked_at=row.checked_at.isoformat(),
    )
