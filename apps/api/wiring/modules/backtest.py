from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import build_backtests_router as build_backtests_api_router
from trading.contexts.backtest.adapters.outbound import (
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    LocalFileBacktestLazyTradesCache,
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestComboPlanningService,
    BacktestLazyTradesDetailService,
    BacktestNoRiskExactScoringService,
    BacktestPreflightService,
    BacktestPreparePoolsService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
    BacktestRuntimeJobOrchestrationService,
    BacktestTpSlExactScoringService,
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency


def build_backtests_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """
    Build fully wired Iteration 1 backtests API router.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires current_user_dependency")

    effective_environ = _with_local_dev_default(environ=environ)
    artifact_config_path = resolve_backtest_artifacts_config_path(environ=effective_environ)
    artifact_config = load_backtest_artifacts_runtime_config(artifact_config_path)
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=effective_environ,
        artifact_config_path=artifact_config_path,
    )
    artifact_path_builder = BacktestArtifactPathBuilderV2(
        root=artifact_config.artifact_root_path()
    )
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=artifact_path_builder)
    artifact_context_resolver = FilesystemBacktestArtifactContextResolver(
        artifact_loader=artifact_loader
    )
    artifact_array_loader = FilesystemBacktestArtifactArrayLoader(
        artifact_loader=artifact_loader
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=artifact_config.hit_times_grid.tp_levels_pct,
        hit_times_sl_levels_pct=artifact_config.hit_times_grid.sl_levels_pct,
        artifact_config_hash=build_backtest_artifacts_runtime_config_hash(
            config=artifact_config
        ),
    )
    runtime_defaults_service = BacktestRuntimeDefaultsService(
        defaults_provider=defaults_provider,
        runtime_config=runtime_config,
    )
    preflight_service = BacktestPreflightService(
        defaults_provider=defaults_provider,
        artifact_context_resolver=artifact_context_resolver,
        runtime_config=runtime_config,
    )
    jobs_use_case = _build_jobs_use_case(
        environ=effective_environ,
        defaults_provider=defaults_provider,
        artifact_array_loader=artifact_array_loader,
        preflight_service=preflight_service,
        runtime_config=runtime_config,
    )
    return build_backtests_api_router(
        runtime_defaults_service=runtime_defaults_service,
        preflight_service=preflight_service,
        current_user_dependency=current_user_dependency,
        jobs_use_case=jobs_use_case,
    )


def _with_local_dev_default(*, environ: Mapping[str, str]) -> Mapping[str, str]:
    if environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "").strip():
        return environ
    if environ.get("ROEHUB_ENV", "").strip():
        return environ
    return {**environ, "ROEHUB_ENV": "dev"}


def _build_jobs_use_case(
    *,
    environ: Mapping[str, str],
    defaults_provider: YamlBacktestGridDefaultsProvider,
    artifact_array_loader: FilesystemBacktestArtifactArrayLoader,
    preflight_service: BacktestPreflightService,
    runtime_config: BacktestRuntimeConfig,
) -> BacktestJobsUseCase | None:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return None
    job_repository = PostgresBacktestJobRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    )
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
    )
    tp_sl_hit_times = BacktestTpSlHitTimesService(
        artifact_array_loader=artifact_array_loader
    )
    executor = BacktestRuntimeJobOrchestrationService(
        prepare_pools=prepare_pools,
        combo_planning=BacktestComboPlanningService(),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=tp_sl_hit_times,
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=artifact_array_loader,
    )
    return BacktestJobsUseCase(
        job_repository=job_repository,
        preflight_service=preflight_service,
        runtime_config=runtime_config,
        executor=executor,
        lazy_trades_service=BacktestLazyTradesDetailService(
            prepare_pools=prepare_pools,
            tp_sl_hit_times=tp_sl_hit_times,
            cache=LocalFileBacktestLazyTradesCache(
                root=Path(
                    environ.get(
                        "ROEHUB_BACKTEST_TRADES_CACHE_ROOT",
                        str(DEFAULT_LAZY_TRADES_CACHE_ROOT),
                    )
                )
            ),
        ),
    )


__all__ = ["build_backtests_router"]
