from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter

from apps.api.routes import build_backtests_router as build_backtests_api_router
from trading.contexts.backtest.adapters.outbound import (
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    BacktestAiConfiguratorRuntimeConfig,
    BacktestArtifactPathBuilderV2,
    DatabaseBacktestJobExecutionTrigger,
    DeterministicBacktestConfigLLMGateway,
    FilesystemBacktestArtifactContextResolver,
    LocalFileBacktestLazyTradesCache,
    PostgresBacktestAiConfigRepository,
    PostgresBacktestJobRepository,
    PostgresBacktestLazyTradesMaterializationRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_admission_config,
    load_backtest_ai_configurator_runtime_config,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_admission_config_path,
    resolve_backtest_ai_configurator_config_path,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiCatalogResolver,
    BacktestAiConfigJobsUseCase,
    BacktestAiConfigPipeline,
    BacktestAiConfigValidator,
    BacktestAiInputGate,
    BacktestAiOutputGate,
    BacktestAiQuotaService,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigLLMGateway,
)
from trading.contexts.backtest.application.ports import BacktestAiConfigLeaseRepository
from trading.contexts.backtest.application.services.v2 import (
    BacktestAdmissionService,
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


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorUseCases:
    jobs: BacktestAiConfigJobsUseCase
    lease_repository: BacktestAiConfigLeaseRepository
    runtime_config: BacktestAiConfiguratorRuntimeConfig
    pipeline: BacktestAiConfigPipeline


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
        ),
    )


def build_backtest_ai_configurator_use_cases(
    *,
    environ: Mapping[str, str],
    llm_gateway: BacktestConfigLLMGateway | None = None,
) -> BacktestAiConfiguratorUseCases | None:
    """
    Build Stage 1 Backtest AI configurator storage/use-case boundary without API routes.
    """
    effective_environ = _with_local_dev_default(environ=environ)
    config_path = resolve_backtest_ai_configurator_config_path(environ=effective_environ)
    ai_runtime_config = load_backtest_ai_configurator_runtime_config(config_path)
    postgres_dsn = effective_environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return None
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
    backtest_runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=artifact_config.hit_times_grid.tp_levels_pct,
        hit_times_sl_levels_pct=artifact_config.hit_times_grid.sl_levels_pct,
        artifact_config_hash=build_backtest_artifacts_runtime_config_hash(
            config=artifact_config
        ),
    )
    runtime_defaults_service = BacktestRuntimeDefaultsService(
        defaults_provider=defaults_provider,
        runtime_config=backtest_runtime_config,
    )
    preflight_service = BacktestPreflightService(
        defaults_provider=defaults_provider,
        artifact_context_resolver=artifact_context_resolver,
        runtime_config=backtest_runtime_config,
    )
    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    repository = PostgresBacktestAiConfigRepository(gateway=postgres_gateway)
    return BacktestAiConfiguratorUseCases(
        jobs=BacktestAiConfigJobsUseCase(
            repository=repository,
            quota_service=BacktestAiQuotaService(
                config=ai_runtime_config.to_quota_config(),
            ),
        ),
        lease_repository=repository,
        runtime_config=ai_runtime_config,
        pipeline=BacktestAiConfigPipeline(
            catalog_resolver=BacktestAiCatalogResolver(
                runtime_defaults_service=runtime_defaults_service,
                supported_symbols=_discover_ai_artifact_symbols(artifact_config=artifact_config),
            ),
            validator=BacktestAiConfigValidator(
                preflight_service=preflight_service,
                output_gate=BacktestAiOutputGate(),
            ),
            input_gate=BacktestAiInputGate(),
            llm_gateway=llm_gateway or DeterministicBacktestConfigLLMGateway(),
        ),
    )


def _discover_ai_artifact_symbols(*, artifact_config: Any) -> tuple[str, ...]:
    root = artifact_config.artifact_root_path()
    discovered: set[str] = set()
    if not root.exists() or not root.is_dir():
        return ("BTCUSDT",)
    for exchange_root in root.iterdir():
        if not exchange_root.is_dir():
            continue
        for market_type_root in exchange_root.iterdir():
            if not market_type_root.is_dir():
                continue
            for child in market_type_root.iterdir():
                if child.is_dir() and (child / "current.yaml").exists():
                    discovered.add(child.name.upper())
    return tuple(sorted(discovered)) or ("BTCUSDT",)


__all__ = [
    "BacktestAiConfiguratorUseCases",
    "build_backtest_ai_configurator_use_cases",
    "build_backtests_router",
]
