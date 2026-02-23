"""
Composition helpers for Market Data reference API module.

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import (
    build_market_data_reference_router as build_market_data_reference_api_router,
)
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseEnabledMarketReader,
    ClickHouseEnabledTradableInstrumentSearchReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.application.use_cases import (
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)


def build_market_data_reference_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """
    Build fully wired auth-only router for Market Data reference endpoints.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - apps/api/routes/market_data_reference.py
      - apps/api/main/app.py
      - apps/cli/wiring/db/clickhouse.py

    Args:
        environ: Runtime environment mapping.
        current_user_dependency: Shared identity auth dependency.
    Returns:
        APIRouter: Router exposing reference markets and instruments endpoints.
    Assumptions:
        ClickHouse settings loader applies repository-wide fail-fast validation policy.
    Raises:
        ValueError: If auth dependency or ClickHouse settings are invalid.
    Side Effects:
        Configures thread-local ClickHouse gateway factory for request-time reads.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_market_data_reference_router requires current_user_dependency")

    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )

    list_enabled_markets_use_case = ListEnabledMarketsUseCase(
        reader=ClickHouseEnabledMarketReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        )
    )
    search_enabled_tradable_instruments_use_case = SearchEnabledTradableInstrumentsUseCase(
        reader=ClickHouseEnabledTradableInstrumentSearchReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        )
    )

    return build_market_data_reference_api_router(
        list_enabled_markets_use_case=list_enabled_markets_use_case,
        search_enabled_tradable_instruments_use_case=search_enabled_tradable_instruments_use_case,
        current_user_dependency=current_user_dependency,
    )


__all__ = ["build_market_data_reference_router"]
