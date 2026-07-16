"""
Market Data reference API routes (auth-only).

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable

from fastapi import APIRouter, Depends, Query, Request

from apps.api.dto import (
    BTCUSDTMarketReadinessResponse,
    MarketDataCatalogItemResponse,
    MarketDataCatalogResponse,
    MarketDataInstrumentsResponse,
    MarketDataMarketsResponse,
    MarketDataSelectionItemResponse,
    MarketDataSelectionsResponse,
    build_btcusdt_market_readiness_response,
    build_market_data_instruments_response,
    build_market_data_markets_response,
)
from trading.contexts.backtest.application.ports import (
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.adapters.outbound.persistence.artifact_inventory_reader import (
    FileSystemActiveArtifactInventoryReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseInstrumentCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.application.use_cases import (
    DEFAULT_INSTRUMENT_SEARCH_LIMIT,
    MAX_INSTRUMENT_SEARCH_LIMIT,
    BTCUSDTMarketReadinessUseCase,
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import MarketId

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_market_data_reference_router(
    *,
    list_enabled_markets_use_case: ListEnabledMarketsUseCase,
    search_enabled_tradable_instruments_use_case: SearchEnabledTradableInstrumentsUseCase,
    btcusdt_market_readiness_use_case: BTCUSDTMarketReadinessUseCase | None = None,
    current_user_dependency: CurrentUserDependency,
    organization_scope_resolver: ResearchOrganizationScopeResolver | None,
    instrument_selection_repository: PostgresInstrumentSelectionRepository | None = None,
    coverage_reader: ClickHouseInstrumentCoverageReader | None = None,
    artifact_inventory_reader: FileSystemActiveArtifactInventoryReader | None = None,
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
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
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
        _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
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
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
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
        _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        instruments = search_enabled_tradable_instruments_use_case.execute(
            market_id=MarketId(market_id),
            q=q,
            limit=limit,
        )
        return build_market_data_instruments_response(instruments=instruments)

    @router.get("/market-data/catalog", response_model=MarketDataCatalogResponse)
    def get_market_data_catalog(
        *,
        market_id: int = Query(..., ge=1),
        q: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=MAX_INSTRUMENT_SEARCH_LIMIT),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataCatalogResponse:
        """Return the visible exchange catalog with current-organization state only."""
        scope = _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        repository = _require_selection_repository(instrument_selection_repository)
        selected = {
            (record.instrument_id.market_id.value, str(record.instrument_id.symbol))
            for record in repository.list_for_organization(organization_id=scope.organization_id)
        }
        instruments = search_enabled_tradable_instruments_use_case.execute(
            market_id=MarketId(market_id), q=q, limit=limit
        )
        history_bounds = repository.list_history_bounds(instrument_ids=instruments)
        coverage_end_at = datetime.now(UTC).replace(second=0, microsecond=0)
        items: list[MarketDataCatalogItemResponse] = []
        for instrument in instruments:
            selection_key = (instrument.market_id.value, str(instrument.symbol))
            strategy_pinned = repository.is_strategy_pinned(
                organization_id=scope.organization_id,
                instrument_id=instrument,
            )
            coverage_state = "unknown"
            coverage_percent: float | None = None
            history_bound = history_bounds.get(selection_key)
            if history_bound is not None and coverage_reader is not None:
                try:
                    coverage = coverage_reader.read(
                        instrument_id=instrument,
                        expected_start_at=history_bound.expected_start_at,
                        expected_end_at=coverage_end_at,
                    )
                except (OSError, ValueError):
                    coverage = None
                if coverage is not None:
                    coverage_state = coverage.state
                    coverage_percent = coverage.percent

            artifact_state = "unavailable"
            artifact_bytes = 0
            if artifact_inventory_reader is not None:
                try:
                    artifact_bytes = artifact_inventory_reader.active_slot_bytes(
                        instrument_id=instrument
                    )
                except (OSError, ValueError):
                    artifact_state = "unavailable"
                else:
                    artifact_state = "ready"

            items.append(
                MarketDataCatalogItemResponse(
                    market_id=instrument.market_id.value,
                    symbol=str(instrument.symbol),
                    selected=selection_key in selected,
                    strategy_pinned=strategy_pinned,
                    effective=selection_key in selected or strategy_pinned,
                    coverage_state=coverage_state,
                    coverage_percent=coverage_percent,
                    artifact_state=artifact_state,
                    artifact_bytes=artifact_bytes,
                )
            )
        return MarketDataCatalogResponse(
            catalog_state=repository.catalog_state(
                market_id=MarketId(market_id), now=datetime.now(UTC)
            ),
            items=items,
        )

    @router.get("/market-data/selections", response_model=MarketDataSelectionsResponse)
    def get_market_data_selections(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataSelectionsResponse:
        scope = _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        repository = _require_selection_repository(instrument_selection_repository)
        items: list[MarketDataSelectionItemResponse] = []
        for record in repository.list_for_organization(organization_id=scope.organization_id):
            strategy_pinned = repository.is_strategy_pinned(
                organization_id=scope.organization_id,
                instrument_id=record.instrument_id,
            )
            items.append(
                MarketDataSelectionItemResponse(
                    market_id=record.instrument_id.market_id.value,
                    symbol=str(record.instrument_id.symbol),
                    strategy_pinned=strategy_pinned,
                    effective=True,
                )
            )
        return MarketDataSelectionsResponse(items=items)

    @router.put(
        "/market-data/selections/{market_id}/{symbol}",
        response_model=MarketDataSelectionItemResponse,
    )
    def select_market_data_instrument(
        *,
        market_id: int,
        symbol: str,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataSelectionItemResponse:
        scope = _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        repository = _require_selection_repository(instrument_selection_repository)
        instrument = _catalog_instrument_or_error(
            search_use_case=search_enabled_tradable_instruments_use_case,
            market_id=market_id,
            symbol=symbol,
        )
        repository.select(
            organization_id=scope.organization_id,
            actor_user_id=principal.user_id,
            instrument_id=instrument,
            now=datetime.now(UTC),
        )
        return MarketDataSelectionItemResponse(
            market_id=instrument.market_id.value,
            symbol=str(instrument.symbol),
            strategy_pinned=repository.is_strategy_pinned(
                organization_id=scope.organization_id, instrument_id=instrument
            ),
            effective=True,
        )

    @router.delete(
        "/market-data/selections/{market_id}/{symbol}",
        response_model=MarketDataSelectionItemResponse,
    )
    def unselect_market_data_instrument(
        *,
        market_id: int,
        symbol: str,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> MarketDataSelectionItemResponse:
        scope = _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        repository = _require_selection_repository(instrument_selection_repository)
        instrument = _catalog_instrument_or_error(
            search_use_case=search_enabled_tradable_instruments_use_case,
            market_id=market_id,
            symbol=symbol,
        )
        repository.unselect(
            organization_id=scope.organization_id,
            actor_user_id=principal.user_id,
            instrument_id=instrument,
            now=datetime.now(UTC),
        )
        strategy_pinned = repository.is_strategy_pinned(
            organization_id=scope.organization_id, instrument_id=instrument
        )
        return MarketDataSelectionItemResponse(
            market_id=instrument.market_id.value,
            symbol=str(instrument.symbol),
            strategy_pinned=strategy_pinned,
            effective=strategy_pinned,
        )

    @router.get(
        "/market-data/btcusdt-readiness",
        response_model=BTCUSDTMarketReadinessResponse,
    )
    def get_btcusdt_market_readiness(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BTCUSDTMarketReadinessResponse:
        """
        Return BTCUSDT market-data readiness matrix for strategy producer launch gates.
        """
        if btcusdt_market_readiness_use_case is None:
            raise ValueError("BTCUSDT market readiness use-case is not configured")
        _resolve_research_scope(
            resolver=organization_scope_resolver,
            principal=principal,
        )
        report = btcusdt_market_readiness_use_case.execute()
        return build_btcusdt_market_readiness_response(report=report)

    return router


def _resolve_research_scope(
    *,
    resolver: ResearchOrganizationScopeResolver | None,
    principal: CurrentUserPrincipal,
) -> ResearchOrganizationScope:
    if resolver is None:
        raise RoehubError(
            code="research.organization_scope_unavailable",
            message="Research organization scope is unavailable",
            details={"reason": "scope_resolver_unavailable"},
        )
    return resolver.resolve(user_id=principal.user_id)


def _require_selection_repository(
    repository: PostgresInstrumentSelectionRepository | None,
) -> PostgresInstrumentSelectionRepository:
    if repository is None:
        raise RoehubError(
            code="market_data.selection_store_unavailable",
            message="Instrument selection storage is unavailable",
            details={"reason": "selection_store_unavailable"},
        )
    return repository


def _catalog_instrument_or_error(
    *,
    search_use_case: SearchEnabledTradableInstrumentsUseCase,
    market_id: int,
    symbol: str,
):
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise RoehubError(
            code="market_data.instrument_not_available",
            message="Instrument is not available in the current catalog",
            details={"reason": "empty_symbol"},
        )
    candidates = search_use_case.execute(
        market_id=MarketId(market_id), q=normalized_symbol, limit=MAX_INSTRUMENT_SEARCH_LIMIT
    )
    for candidate in candidates:
        if str(candidate.symbol) == normalized_symbol:
            return candidate
    raise RoehubError(
        code="market_data.instrument_not_available",
        message="Instrument is not available in the current catalog",
        details={"reason": "instrument_not_in_catalog"},
    )


__all__ = ["build_market_data_reference_router"]
