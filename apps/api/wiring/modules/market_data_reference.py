"""
Composition helpers for Market Data reference API module.

Docs:
  - docs/architecture/market_data/market-data-reference-api-v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import (
    build_market_data_reference_router as build_market_data_reference_api_router,
)
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.backtest.adapters.outbound import (
    PsycopgBacktestPostgresGateway,
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.market_data.adapters.outbound.persistence.artifact_inventory_reader import (
    FileSystemActiveArtifactInventoryReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseBTCUSDTMarketReadinessReader,
    ClickHouseEnabledMarketReader,
    ClickHouseEnabledTradableInstrumentSearchReader,
    ClickHouseInstrumentCoverageReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTStreamReadinessSnapshot,
)
from trading.contexts.market_data.application.use_cases import (
    BTCUSDTMarketReadinessUseCase,
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.contexts.strategy.adapters.outbound import (
    RedisMarketDataReadinessReader,
    RedisStrategyLiveCandleStreamConfig,
    load_strategy_runtime_config,
    resolve_strategy_config_path,
)

from .research_tenancy import build_research_organization_scope_resolver


@dataclass(frozen=True, slots=True)
class MarketDataReferenceUseCases:
    list_enabled_markets: ListEnabledMarketsUseCase
    search_enabled_tradable_instruments: SearchEnabledTradableInstrumentsUseCase
    btcusdt_market_readiness: BTCUSDTMarketReadinessUseCase
    coverage_reader: ClickHouseInstrumentCoverageReader


def build_market_data_reference_use_cases(
    *, environ: Mapping[str, str]
) -> MarketDataReferenceUseCases:
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )

    return MarketDataReferenceUseCases(
        list_enabled_markets=ListEnabledMarketsUseCase(
            reader=ClickHouseEnabledMarketReader(
                gateway=clickhouse_gateway,
                database=clickhouse_settings.database,
            )
        ),
        search_enabled_tradable_instruments=SearchEnabledTradableInstrumentsUseCase(
            reader=ClickHouseEnabledTradableInstrumentSearchReader(
                gateway=clickhouse_gateway,
                database=clickhouse_settings.database,
            )
        ),
        btcusdt_market_readiness=BTCUSDTMarketReadinessUseCase(
            reference_reader=ClickHouseBTCUSDTMarketReadinessReader(
                gateway=clickhouse_gateway,
                database=clickhouse_settings.database,
            ),
            stream_reader=_build_btcusdt_stream_readiness_reader(environ=environ),
        ),
        coverage_reader=ClickHouseInstrumentCoverageReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        ),
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

    use_cases = build_market_data_reference_use_cases(environ=environ)

    return build_market_data_reference_api_router(
        list_enabled_markets_use_case=use_cases.list_enabled_markets,
        search_enabled_tradable_instruments_use_case=use_cases.search_enabled_tradable_instruments,
        btcusdt_market_readiness_use_case=use_cases.btcusdt_market_readiness,
        current_user_dependency=current_user_dependency,
        organization_scope_resolver=build_research_organization_scope_resolver(
            environ=environ
        ),
        instrument_selection_repository=_build_instrument_selection_repository(environ=environ),
        coverage_reader=use_cases.coverage_reader,
        artifact_inventory_reader=_build_artifact_inventory_reader(environ=environ),
    )


def _build_instrument_selection_repository(
    *, environ: Mapping[str, str]
) -> PostgresInstrumentSelectionRepository | None:
    dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not dsn:
        return None
    return PostgresInstrumentSelectionRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=dsn)
    )


def _build_artifact_inventory_reader(
    *, environ: Mapping[str, str]
) -> FileSystemActiveArtifactInventoryReader | None:
    configured = environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "").strip()
    path = Path(configured) if configured else Path("/etc/roehub/backtest-artifacts.yaml")
    if not path.is_file():
        return None
    config = load_backtest_artifacts_runtime_config(path)
    return FileSystemActiveArtifactInventoryReader(
        artifact_root=config.artifact_root_path()
    )


class _UnavailableBTCUSDTStreamReadinessReader:
    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at,
    ) -> BTCUSDTStreamReadinessSnapshot:
        _ = timeframe
        _ = observed_at
        return BTCUSDTStreamReadinessSnapshot(
            state="pending",
            reason_code="market_data_readiness_reader_unavailable",
            stream_name=f"md.candles.1m.{instrument_key}",
            stream_length=None,
            last_message_id=None,
            last_observed_at=None,
            age_seconds=None,
        )


class _StrategyRedisBTCUSDTStreamReadinessReader:
    def __init__(self, *, reader: RedisMarketDataReadinessReader) -> None:
        self._reader = reader

    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at,
    ) -> BTCUSDTStreamReadinessSnapshot:
        snapshot = self._reader.check(
            instrument_key=instrument_key,
            timeframe=timeframe,
            observed_at=observed_at,
        )
        return BTCUSDTStreamReadinessSnapshot(
            state=snapshot.state,
            reason_code=snapshot.reason_code,
            stream_name=snapshot.stream_name,
            stream_length=snapshot.stream_length,
            last_message_id=snapshot.last_message_id,
            last_observed_at=snapshot.last_observed_at,
            age_seconds=snapshot.age_seconds,
        )


def _build_btcusdt_stream_readiness_reader(*, environ: Mapping[str, str]):
    try:
        runtime_config = load_strategy_runtime_config(
            resolve_strategy_config_path(environ=environ),
            environ=environ,
        )
        redis_config = runtime_config.live_worker.redis_streams
        if not redis_config.enabled:
            return _UnavailableBTCUSDTStreamReadinessReader()
        return _StrategyRedisBTCUSDTStreamReadinessReader(
            reader=RedisMarketDataReadinessReader(
                config=RedisStrategyLiveCandleStreamConfig(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password_env=redis_config.password_env,
                    socket_timeout_s=redis_config.socket_timeout_s,
                    connect_timeout_s=redis_config.connect_timeout_s,
                    stream_prefix=redis_config.stream_prefix,
                    consumer_group=redis_config.consumer_group,
                    consumer_name="api-btcusdt-readiness",
                    read_count=1,
                    block_ms=0,
                ),
                environ=environ,
            )
        )
    except Exception:
        return _UnavailableBTCUSDTStreamReadinessReader()


__all__ = [
    "MarketDataReferenceUseCases",
    "build_market_data_reference_router",
    "build_market_data_reference_use_cases",
]
