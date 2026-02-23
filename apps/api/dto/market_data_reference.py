"""
Pydantic API models and converters for Market Data reference API v1 endpoints.

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
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
