"""
Composition helpers for backtests API module.

Docs:
  - docs/architecture/backtest/README.md
  - docs/architecture/backtest/backtest-core-refactor-prompt-pack-v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter

from apps.api.dto import (
    build_backtest_runtime_defaults_response,
)
from apps.api.routes import (
    build_backtest_jobs_router,
    build_backtest_runs_router,
    build_backtests_router,
)
from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobRepository,
    PostgresBacktestJobResultsRepository,
    PsycopgBacktestPostgresGateway,
    StrategyRepositoryBacktestStrategyReader,
    YamlBacktestGridDefaultsProvider,
    build_backtest_runtime_config_hash,
    load_backtest_runtime_config,
    resolve_backtest_config_path,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestRunProgressSnapshotBuilder,
    CancelBacktestJobUseCase,
    CancelBacktestRunUseCase,
    CreateBacktestJobUseCase,
    GetBacktestJobStatusUseCase,
    GetBacktestJobTopUseCase,
    GetBacktestRunStatusUseCase,
    GetBacktestRunTopUseCase,
    LaunchBacktestGatewayUseCase,
    ListBacktestJobsUseCase,
    ListBacktestRunsUseCase,
)
from trading.contexts.backtest_artifacts.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    BacktestArtifactsRuntimeConfig,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest_artifacts.application.services import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.strategy.adapters.outbound import (
    InMemoryStrategyRepository,
    PostgresStrategyRepository,
    PsycopgStrategyPostgresGateway,
)
from trading.contexts.strategy.application import StrategyRepository

_ENV_NAME_KEY = "ROEHUB_ENV"
_BACKTEST_FAIL_FAST_KEY = "BACKTEST_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_ALLOWED_ENVS = ("dev", "prod", "test")
_BACKTEST_API_EXECUTION_BOUNDARY = "gateway_background_only"


@dataclass(frozen=True, slots=True)
class BacktestRuntimeSettings:
    """
    Runtime settings for backtests module repository fail-fast composition policy.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-core-refactor-prompt-pack-v1.md
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
    Build fully wired Backtest API router (`POST /backtests` + optional jobs endpoints).

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-core-refactor-prompt-pack-v1.md
    Related:
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

    Args:
        environ: Runtime environment mapping.
        current_user_dependency: Shared identity dependency resolving authenticated principal.
        indicator_compute:
            Retained compatibility dependency no longer used by the gateway-only launch path.
    Returns:
        APIRouter: Backtests router with optional EPIC-11 jobs endpoints.
    Assumptions:
        API runtime is gateway-only in production and enqueues background compute jobs only.
        Defaults/provider/config are validated on startup (fail-fast).
    Raises:
        ValueError: If required runtime dependencies are invalid or missing.
        FileNotFoundError: If `backtest.yaml`, `backtest_artifacts.yaml`, or `indicators.yaml`
            cannot be resolved.
    Side Effects:
        Reads runtime YAML files and configures storage/artifact/runtime adapters.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_router requires current_user_dependency")
    if indicator_compute is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_router requires indicator_compute")

    runtime_settings = _resolve_backtest_runtime_settings(environ=environ)
    runtime_config_path = resolve_backtest_config_path(environ=environ)
    runtime_config = load_backtest_runtime_config(runtime_config_path)
    artifact_runtime_config = _load_backtest_artifacts_runtime_config(environ=environ)
    backtest_runtime_config_hash = build_backtest_runtime_config_hash(config=runtime_config)

    if not runtime_settings.strategy_postgres_dsn:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required for persisted POST /backtests storage"
        )

    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(environ=environ)
    strategy_repository = _build_strategy_repository(settings=runtime_settings)
    strategy_reader = StrategyRepositoryBacktestStrategyReader(repository=strategy_repository)
    _ = indicator_compute
    artifact_loader = YamlBacktestArtifactLoaderV2(
        path_resolver=BacktestArtifactPathBuilderV2(
            root=artifact_runtime_config.artifact_root_path()
        )
    )
    jobs_gateway = _build_jobs_gateway(settings=runtime_settings)
    job_repository = PostgresBacktestJobRepository(gateway=jobs_gateway)
    create_use_case = CreateBacktestJobUseCase(
        job_repository=job_repository,
        strategy_reader=strategy_reader,
        top_k_persisted_default=runtime_config.jobs.top_k_persisted_default,
        max_active_jobs_per_user=runtime_config.jobs.max_active_jobs_per_user,
        warmup_bars_default=runtime_config.warmup_bars_default,
        top_k_default=runtime_config.top_k_default,
        preselect_default=runtime_config.preselect_default,
        init_cash_quote_default=runtime_config.execution.init_cash_quote_default,
        fixed_quote_default=runtime_config.execution.fixed_quote_default,
        safe_profit_percent_default=runtime_config.execution.safe_profit_percent_default,
        slippage_pct_default=runtime_config.execution.slippage_pct_default,
        fee_pct_default_by_market_id=runtime_config.execution.fee_pct_default_by_market_id,
        backtest_runtime_config_hash=backtest_runtime_config_hash,
        artifact_loader=artifact_loader,
        defaults_provider=defaults_provider,
        allowed_request_timeframes=runtime_config.contracts.allowed_request_timeframes,
        forbidden_request_timeframes=runtime_config.contracts.forbidden_request_timeframes,
    )
    backtests_launch_use_case = LaunchBacktestGatewayUseCase(
        background_create_use_case=create_use_case,
        engine_version=runtime_config.contracts.risk_model,
    )
    _ = _BACKTEST_API_EXECUTION_BOUNDARY
    runtime_defaults_response = build_backtest_runtime_defaults_response(
        config=runtime_config,
        defaults_provider=defaults_provider,
    )
    backtests_router = build_backtests_router(
        run_use_case=backtests_launch_use_case,
        strategy_reader=strategy_reader,
        runtime_defaults_response=runtime_defaults_response,
        current_user_dependency=current_user_dependency,
        sync_deadline_seconds=runtime_config.sync.sync_deadline_seconds,
        eager_top_reports_enabled=runtime_config.reporting.eager_top_reports_enabled,
    )

    results_repository = PostgresBacktestJobResultsRepository(gateway=jobs_gateway)
    runs_router = build_backtest_runs_router(
        get_status_use_case=GetBacktestRunStatusUseCase(job_repository=job_repository),
        get_top_use_case=GetBacktestRunTopUseCase(
            job_repository=job_repository,
            results_repository=results_repository,
            top_k_persisted_default=runtime_config.jobs.top_k_persisted_default,
        ),
        list_use_case=ListBacktestRunsUseCase(job_repository=job_repository),
        cancel_use_case=CancelBacktestRunUseCase(job_repository=job_repository),
        current_user_dependency=current_user_dependency,
        sync_deadline_seconds=runtime_config.sync.sync_deadline_seconds,
        run_progress_builder=BacktestRunProgressSnapshotBuilder(
            execution_profiles=runtime_config.execution_profiles,
            benchmark_corpus=runtime_config.runtime_acceleration_benchmark_corpus,
        ),
    )
    backtests_router.include_router(runs_router)
    if not runtime_config.jobs.enabled:
        return backtests_router

    jobs_router = build_backtest_jobs_router(
        create_use_case=create_use_case,
        get_status_use_case=GetBacktestJobStatusUseCase(job_repository=job_repository),
        get_top_use_case=GetBacktestJobTopUseCase(
            job_repository=job_repository,
            results_repository=results_repository,
            top_k_persisted_default=runtime_config.jobs.top_k_persisted_default,
        ),
        list_use_case=ListBacktestJobsUseCase(job_repository=job_repository),
        cancel_use_case=CancelBacktestJobUseCase(job_repository=job_repository),
        current_user_dependency=current_user_dependency,
    )
    backtests_router.include_router(jobs_router)
    return backtests_router


def _load_backtest_artifacts_runtime_config(
    *,
    environ: Mapping[str, str],
) -> BacktestArtifactsRuntimeConfig:
    """
    Resolve and fail-fast load strict artifact pipeline config for backtest wiring.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        BacktestArtifactsRuntimeConfig: Parsed strict artifact pipeline config.
    Assumptions:
        Startup must validate artifact root and validation-plan contracts before jobs wiring.
    Raises:
        FileNotFoundError: If `backtest_artifacts.yaml` cannot be resolved.
        ValueError: If the artifact config payload is invalid.
    Side Effects:
        Reads one UTF-8 YAML file from filesystem.
    """
    artifact_environ = dict(environ)
    artifact_environ.setdefault(_ENV_NAME_KEY, _resolve_env_name(environ=environ))
    config_path = resolve_backtest_artifacts_config_path(environ=artifact_environ)
    return load_backtest_artifacts_runtime_config(config_path)


def _resolve_backtest_runtime_settings(*, environ: Mapping[str, str]) -> BacktestRuntimeSettings:
    """
    Resolve backtests module runtime settings with environment-aware fail-fast policy.

    Docs:
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
    raise ValueError(f"{_BACKTEST_FAIL_FAST_KEY} must be boolean-like value, got {raw_value!r}")


def _build_strategy_repository(*, settings: BacktestRuntimeSettings) -> StrategyRepository:
    """
    Build strategy repository dependency for saved-mode strategy loading.

    Docs:
      - docs/architecture/backtest/README.md
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


def _build_jobs_gateway(*, settings: BacktestRuntimeSettings) -> PsycopgBacktestPostgresGateway:
    """
    Build fail-fast Postgres gateway for Backtest Jobs API repositories.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/gateway.py
      - apps/api/routes/backtest_jobs.py

    Args:
        settings: Resolved runtime settings with strategy Postgres DSN.
    Returns:
        PsycopgBacktestPostgresGateway: Configured gateway for jobs repositories.
    Assumptions:
        Jobs endpoints are mounted only when runtime toggle is enabled.
    Raises:
        ValueError: If Postgres DSN is missing.
    Side Effects:
        None.
    """
    if not settings.strategy_postgres_dsn:
        raise ValueError(f"{_STRATEGY_PG_DSN_KEY} is required when backtest.jobs.enabled is true")
    return PsycopgBacktestPostgresGateway(dsn=settings.strategy_postgres_dsn)

__all__ = [
    "BacktestRuntimeSettings",
    "build_backtest_router",
]
