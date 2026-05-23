from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import build_backtests_router as build_backtests_api_router
from trading.contexts.backtest.adapters.outbound import (
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    BacktestArtifactPathBuilderV2,
    DatabaseBacktestJobExecutionTrigger,
    FilesystemBacktestArtifactContextResolver,
    LocalFileBacktestLazyTradesCache,
    PostgresBacktestJobRepository,
    PostgresBacktestLazyTradesMaterializationRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_admission_config,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_admission_config_path,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2 import (
    DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
    BacktestAdmissionService,
    BacktestLazyTradesDetailConfig,
    BacktestLazyTradesDetailService,
    BacktestPreflightService,
    BacktestPreparePoolsService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
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
    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    job_repository = PostgresBacktestJobRepository(gateway=postgres_gateway)
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
    )
    tp_sl_hit_times = BacktestTpSlHitTimesService(
        artifact_array_loader=artifact_array_loader
    )
    return BacktestJobsUseCase(
        job_repository=job_repository,
        preflight_service=preflight_service,
        runtime_config=runtime_config,
        admission_service=BacktestAdmissionService(
            config=load_backtest_admission_config(
                resolve_backtest_admission_config_path(environ=environ)
            )
        ),
        execution_trigger=DatabaseBacktestJobExecutionTrigger(),
        lazy_trades_materialization_repository=(
            PostgresBacktestLazyTradesMaterializationRepository(gateway=postgres_gateway)
        ),
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
            config=BacktestLazyTradesDetailConfig(
                cache_ttl_seconds=_lazy_trades_cache_ttl_seconds(
                    environ=environ,
                )
            ),
        ),
    )


def _lazy_trades_cache_ttl_seconds(*, environ: Mapping[str, str]) -> int:
    raw = environ.get("ROEHUB_BACKTEST_DETAIL_CACHE_TTL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS
    return int(raw)


__all__ = [
    "build_backtests_router",
]
