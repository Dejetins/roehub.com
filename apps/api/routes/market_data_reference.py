"""
Market Data reference API routes (auth-only).

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Query, Request

from apps.api.dto import (
    MarketDataInstrumentsResponse,
    MarketDataMarketsResponse,
    build_market_data_instruments_response,
    build_market_data_markets_response,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.application.use_cases import (
    DEFAULT_INSTRUMENT_SEARCH_LIMIT,
    MAX_INSTRUMENT_SEARCH_LIMIT,
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.shared_kernel.primitives import MarketId

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_market_data_reference_router(
    *,
    list_enabled_markets_use_case: ListEnabledMarketsUseCase,
    search_enabled_tradable_instruments_use_case: SearchEnabledTradableInstrumentsUseCase,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    """
    Build auth-only Market Data reference API router.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/dto/market_data_reference.py
      - apps/api/wiring/modules/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py

    Args:
        list_enabled_markets_use_case: Use-case returning enabled markets.
        search_enabled_tradable_instruments_use_case: Use-case searching market instruments.
        current_user_dependency: Identity dependency resolving authenticated principal.
    Returns:
        APIRouter: Router with `/market-data/markets` and `/market-data/instruments`.
    Assumptions:
        Business rules are implemented in use-cases and adapters; route layer maps transport only.
    Raises:
        ValueError: If one of required dependencies is missing.
    Side Effects:
        None.
    """
    if list_enabled_markets_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError(
            "build_market_data_reference_router requires list_enabled_markets_use_case"
        )
    if search_enabled_tradable_instruments_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError(
            "build_market_data_reference_router requires "
            "search_enabled_tradable_instruments_use_case"
        )
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_market_data_reference_router requires current_user_dependency")

    router = APIRouter(tags=["market-data"])

    @router.get("/market-data/markets", response_model=MarketDataMarketsResponse)
    def get_market_data_markets(
        _principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataMarketsResponse:
        """
        Return enabled markets ordered deterministically by `market_id ASC`.

        Args:
            _principal: Authenticated identity principal from cookie dependency.
        Returns:
            MarketDataMarketsResponse: Enabled markets response wrapper.
        Assumptions:
            Endpoint is auth-only and available only for authenticated users.
        Raises:
            HTTPException: 401 when authentication dependency rejects request.
        Side Effects:
            Executes one use-case read over ClickHouse reference table.
        """
        markets = list_enabled_markets_use_case.execute()
        return build_market_data_markets_response(markets=markets)

    @router.get("/market-data/instruments", response_model=MarketDataInstrumentsResponse)
    def get_market_data_instruments(
        *,
        market_id: int = Query(..., ge=1),
        q: str | None = Query(default=None),
        limit: int = Query(
            default=DEFAULT_INSTRUMENT_SEARCH_LIMIT,
            ge=1,
            le=MAX_INSTRUMENT_SEARCH_LIMIT,
        ),
        _principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataInstrumentsResponse:
        """
        Search enabled tradable instruments for one market with optional prefix filter.

        Args:
            market_id: Required market id query parameter.
            q: Optional symbol prefix filter, blank means no filter.
            limit: Optional max result size (`default=50`, `max=200`).
            _principal: Authenticated identity principal from cookie dependency.
        Returns:
            MarketDataInstrumentsResponse: Market instrument tuples ordered by `symbol ASC`.
        Assumptions:
            Unknown or disabled market id is represented as `items=[]`.
        Raises:
            HTTPException: 401 for unauthenticated requests, 422 for invalid query values.
        Side Effects:
            Executes one use-case read over ClickHouse reference tables.
        """
        instruments = search_enabled_tradable_instruments_use_case.execute(
            market_id=MarketId(market_id),
            q=q,
            limit=limit,
        )
        return build_market_data_instruments_response(instruments=instruments)

    return router


__all__ = ["build_market_data_reference_router"]
