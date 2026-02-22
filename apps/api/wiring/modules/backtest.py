"""
Composition helpers for backtests API module.

Docs:
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import build_backtests_router
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.backtest.adapters.outbound import (
    StrategyRepositoryBacktestStrategyReader,
    YamlBacktestGridDefaultsProvider,
    load_backtest_runtime_config,
    resolve_backtest_config_path,
)
from trading.contexts.backtest.application.use_cases import RunBacktestUseCase
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.indicators.adapters.outbound import MarketDataCandleFeed
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.strategy.adapters.outbound import (
    InMemoryStrategyRepository,
    PostgresStrategyRepository,
    PsycopgStrategyPostgresGateway,
)
from trading.contexts.strategy.application import StrategyRepository
from trading.platform.time.system_clock import SystemClock

_ENV_NAME_KEY = "ROEHUB_ENV"
_BACKTEST_FAIL_FAST_KEY = "BACKTEST_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class BacktestRuntimeSettings:
    """
    Runtime settings for backtests module repository fail-fast composition policy.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - apps/api/main/app.py
      - apps/api/routes/backtests.py
    """

    env_name: str
    fail_fast: bool
    strategy_postgres_dsn: str

    def __post_init__(self) -> None:
        """
        Validate runtime settings invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `env_name` is normalized by resolver before dataclass construction.
        Raises:
            ValueError: If one invariant is violated.
        Side Effects:
            None.
        """
        if self.env_name not in _ALLOWED_ENVS:
            raise ValueError(
                f"BacktestRuntimeSettings.env_name must be one of {_ALLOWED_ENVS}, "
                f"got {self.env_name!r}"
            )


def build_backtest_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
    indicator_compute: IndicatorCompute,
) -> APIRouter:
    """
    Build fully wired `POST /backtests` router with fail-fast runtime configuration.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

    Args:
        environ: Runtime environment mapping.
        current_user_dependency: Shared identity dependency resolving authenticated principal.
        indicator_compute: Pre-warmed indicators compute adapter.
    Returns:
        APIRouter: Backtests router exposing `POST /backtests` endpoint.
    Assumptions:
        Defaults/provider/config are validated on startup (fail-fast).
    Raises:
        ValueError: If required runtime dependencies are invalid or missing.
        FileNotFoundError: If `backtest.yaml` or `indicators.yaml` cannot be resolved.
    Side Effects:
        Reads runtime YAML files and configures storage/candle-feed adapters.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_router requires current_user_dependency")
    if indicator_compute is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_router requires indicator_compute")

    runtime_settings = _resolve_backtest_runtime_settings(environ=environ)
    runtime_config_path = resolve_backtest_config_path(environ=environ)
    runtime_config = load_backtest_runtime_config(runtime_config_path)

    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(environ=environ)
    strategy_repository = _build_strategy_repository(settings=runtime_settings)
    strategy_reader = StrategyRepositoryBacktestStrategyReader(repository=strategy_repository)
    candle_feed = _build_backtest_candle_feed(environ=environ)

    run_use_case = RunBacktestUseCase(
        candle_feed=candle_feed,
        indicator_compute=indicator_compute,
        strategy_reader=strategy_reader,
        defaults_provider=defaults_provider,
        warmup_bars_default=runtime_config.warmup_bars_default,
        top_k_default=runtime_config.top_k_default,
        preselect_default=runtime_config.preselect_default,
        top_trades_n_default=runtime_config.reporting.top_trades_n_default,
        init_cash_quote_default=runtime_config.execution.init_cash_quote_default,
        fixed_quote_default=runtime_config.execution.fixed_quote_default,
        safe_profit_percent_default=runtime_config.execution.safe_profit_percent_default,
        slippage_pct_default=runtime_config.execution.slippage_pct_default,
        fee_pct_default_by_market_id=runtime_config.execution.fee_pct_default_by_market_id,
    )
    return build_backtests_router(
        run_use_case=run_use_case,
        strategy_reader=strategy_reader,
        current_user_dependency=current_user_dependency,
    )


def _resolve_backtest_runtime_settings(*, environ: Mapping[str, str]) -> BacktestRuntimeSettings:
    """
    Resolve backtests module runtime settings with environment-aware fail-fast policy.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - apps/api/main/app.py
      - apps/api/wiring/modules/strategy.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        BacktestRuntimeSettings: Normalized runtime settings.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If env values are invalid.
    Side Effects:
        None.
    """
    env_name = _resolve_env_name(environ=environ)
    fail_fast = _resolve_fail_fast(environ=environ, env_name=env_name)
    strategy_postgres_dsn = environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    return BacktestRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        strategy_postgres_dsn=strategy_postgres_dsn,
    )


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized runtime environment name for backtests wiring.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - apps/api/main/app.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        str: Environment literal (`dev`, `prod`, `test`).
    Assumptions:
        Missing env variable defaults to `dev`.
    Raises:
        ValueError: If env literal is unsupported.
    Side Effects:
        None.
    """
    raw_env = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env!r}")
    return raw_env


def _resolve_fail_fast(*, environ: Mapping[str, str], env_name: str) -> bool:
    """
    Resolve backtests fail-fast mode from explicit override or environment default.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - apps/api/wiring/modules/strategy.py
      - apps/api/main/app.py

    Args:
        environ: Runtime environment mapping.
        env_name: Normalized environment name.
    Returns:
        bool: True when fail-fast is enabled.
    Assumptions:
        Default policy enables fail-fast only in `prod`.
    Raises:
        ValueError: If explicit override is not boolean-like.
    Side Effects:
        None.
    """
    raw_value = environ.get(_BACKTEST_FAIL_FAST_KEY)
    if raw_value is None:
        return env_name == "prod"

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_BACKTEST_FAIL_FAST_KEY} must be boolean-like value, got {raw_value!r}"
    )


def _build_strategy_repository(*, settings: BacktestRuntimeSettings) -> StrategyRepository:
    """
    Build strategy repository dependency for saved-mode strategy loading.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - src/trading/contexts/backtest/adapters/outbound/acl/strategy_repository_reader.py

    Args:
        settings: Resolved runtime settings.
    Returns:
        StrategyRepository: Postgres or in-memory repository adapter.
    Assumptions:
        In-memory fallback is allowed only when fail-fast mode is disabled.
    Raises:
        ValueError: If fail-fast is enabled but Postgres DSN is missing.
    Side Effects:
        None.
    """
    if settings.strategy_postgres_dsn:
        gateway = PsycopgStrategyPostgresGateway(dsn=settings.strategy_postgres_dsn)
        return PostgresStrategyRepository(gateway=gateway)

    if settings.fail_fast:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required when backtest fail-fast mode is enabled"
        )

    return InMemoryStrategyRepository()


def _build_backtest_candle_feed(*, environ: Mapping[str, str]) -> MarketDataCandleFeed:
    """
    Build backtests candle-feed adapter backed by market_data canonical candle reader.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/
        market_data_candle_feed.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        canonical_candle_reader.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        MarketDataCandleFeed: Candle-feed adapter for backtest use-case.
    Assumptions:
        ClickHouse client is created lazily by thread-local gateway factory.
    Raises:
        ValueError: If ClickHouse settings are invalid.
    Side Effects:
        Configures clickhouse gateway factory callable for runtime reads.
    """
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )
    canonical_reader = ClickHouseCanonicalCandleReader(
        gateway=clickhouse_gateway,
        clock=SystemClock(),
        database=clickhouse_settings.database,
    )
    return MarketDataCandleFeed(canonical_candle_reader=canonical_reader)


__all__ = [
    "BacktestRuntimeSettings",
    "build_backtest_router",
]
