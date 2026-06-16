from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

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
from trading.contexts.backtest.application.ports import BacktestJobRepository
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
from trading.contexts.strategy.adapters.outbound import (
    PostgresStrategyBacktestVariantProvenanceRepository,
    PostgresStrategyCompatibilityReadinessRepository,
    PostgresStrategyEventRepository,
    PostgresStrategyRepository,
    PostgresStrategyVariantScenarioMatrixRepository,
    PsycopgStrategyPostgresGateway,
    RedisMarketDataReadinessReader,
    RedisStrategyLiveCandleStreamConfig,
    SystemStrategyClock,
    load_strategy_runtime_config,
    resolve_strategy_config_path,
)
from trading.contexts.strategy.application import (
    BacktestVariantLaunchReader,
    BacktestVariantLaunchSnapshot,
    CreateStrategyFromBacktestVariantUseCase,
    StrategyCompatibilityReadinessService,
    StrategyVariantScenarioMatrixService,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


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
    job_repository = _build_job_repository(environ=effective_environ)
    jobs_use_case = _build_jobs_use_case(
        environ=effective_environ,
        job_repository=job_repository,
        defaults_provider=defaults_provider,
        artifact_array_loader=artifact_array_loader,
        preflight_service=preflight_service,
        runtime_config=runtime_config,
    )
    create_strategy_from_variant_use_case = _build_create_strategy_from_variant_use_case(
        environ=effective_environ,
        job_repository=job_repository,
    )
    variant_launch_reader = (
        _BacktestJobRepositoryVariantLaunchReader(repository=job_repository)
        if job_repository is not None
        else None
    )
    compatibility_readiness_service = _build_compatibility_readiness_service(
        environ=effective_environ,
    )
    scenario_matrix_service = _build_scenario_matrix_service(
        environ=effective_environ,
        compatibility_readiness_service=compatibility_readiness_service,
    )
    return build_backtests_api_router(
        runtime_defaults_service=runtime_defaults_service,
        preflight_service=preflight_service,
        current_user_dependency=current_user_dependency,
        jobs_use_case=jobs_use_case,
        create_strategy_from_variant_use_case=create_strategy_from_variant_use_case,
        compatibility_readiness_service=compatibility_readiness_service,
        scenario_matrix_service=scenario_matrix_service,
        backtest_variant_launch_reader=variant_launch_reader,
    )


def _with_local_dev_default(*, environ: Mapping[str, str]) -> Mapping[str, str]:
    if environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "").strip():
        return environ
    if environ.get("ROEHUB_ENV", "").strip():
        return environ
    return {**environ, "ROEHUB_ENV": "dev"}


def _build_job_repository(*, environ: Mapping[str, str]) -> BacktestJobRepository | None:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return None
    return PostgresBacktestJobRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    )


def _build_jobs_use_case(
    *,
    environ: Mapping[str, str],
    defaults_provider: YamlBacktestGridDefaultsProvider,
    artifact_array_loader: FilesystemBacktestArtifactArrayLoader,
    preflight_service: BacktestPreflightService,
    runtime_config: BacktestRuntimeConfig,
    job_repository: BacktestJobRepository | None = None,
) -> BacktestJobsUseCase | None:
    if job_repository is None:
        job_repository = _build_job_repository(environ=environ)
    if job_repository is None:
        return None
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
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


def _build_create_strategy_from_variant_use_case(
    *,
    environ: Mapping[str, str],
    job_repository: BacktestJobRepository | None,
) -> CreateStrategyFromBacktestVariantUseCase | None:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn or job_repository is None:
        return None
    strategy_gateway = PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
    strategy_repository = PostgresStrategyRepository(gateway=strategy_gateway)
    return CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_BacktestJobRepositoryVariantLaunchReader(repository=job_repository),
        strategy_repository=strategy_repository,
        provenance_repository=PostgresStrategyBacktestVariantProvenanceRepository(
            gateway=strategy_gateway,
        ),
        event_repository=PostgresStrategyEventRepository(gateway=strategy_gateway),
        clock=SystemStrategyClock(),
    )


def _build_compatibility_readiness_service(
    *, environ: Mapping[str, str]
) -> StrategyCompatibilityReadinessService | None:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return None
    strategy_gateway = PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
    redis_reader = None
    try:
        runtime_config = load_strategy_runtime_config(
            resolve_strategy_config_path(environ=environ),
            environ=environ,
        )
        redis_config = runtime_config.live_worker.redis_streams
        if redis_config.enabled:
            redis_reader = RedisMarketDataReadinessReader(
                config=RedisStrategyLiveCandleStreamConfig(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password_env=redis_config.password_env,
                    socket_timeout_s=redis_config.socket_timeout_s,
                    connect_timeout_s=redis_config.connect_timeout_s,
                    stream_prefix=redis_config.stream_prefix,
                    consumer_group=redis_config.consumer_group,
                    consumer_name="api-readiness",
                    read_count=1,
                    block_ms=0,
                ),
                environ=environ,
            )
    except Exception:
        redis_reader = None
    return StrategyCompatibilityReadinessService(
        strategy_repository=PostgresStrategyRepository(gateway=strategy_gateway),
        compatibility_repository=PostgresStrategyCompatibilityReadinessRepository(
            gateway=strategy_gateway,
        ),
        market_data_reader=redis_reader,
        clock=SystemStrategyClock(),
    )


def _build_scenario_matrix_service(
    *,
    environ: Mapping[str, str],
    compatibility_readiness_service: StrategyCompatibilityReadinessService | None,
) -> StrategyVariantScenarioMatrixService | None:
    if compatibility_readiness_service is None:
        return None
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    repository = None
    if postgres_dsn:
        repository = PostgresStrategyVariantScenarioMatrixRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
        )
    return StrategyVariantScenarioMatrixService(
        compatibility_readiness_service=compatibility_readiness_service,
        repository=repository,
        clock=SystemStrategyClock(),
    )


class _BacktestJobRepositoryVariantLaunchReader(BacktestVariantLaunchReader):
    def __init__(self, *, repository: BacktestJobRepository) -> None:
        self._repository = repository

    def get(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestVariantLaunchSnapshot:
        job = self._repository.get(job_id=job_id)
        if job is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest job was not found",
                details={"reason": "not_found", "job_id": str(job_id)},
            )
        if job.user_id != user_id:
            raise RoehubError(
                code="strategy_variant_launch.forbidden",
                message="Backtest job does not belong to current user",
                details={"reason": "forbidden", "job_id": str(job_id)},
            )
        row = self._repository.get_top_variant_by_public_key(
            job_id=job_id,
            public_variant_key=variant_key,
        )
        if row is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest variant was not found",
                details={"reason": "not_found", "job_id": str(job_id), "variant_key": variant_key},
            )
        request = dict(job.request_json)
        coordinates = _mapping(request.get("coordinates"))
        payload = dict(row.payload_json)
        market_id = job.market_id
        if market_id is None:
            raise RoehubError(
                code="strategy_variant_launch.not_launchable",
                message="Backtest job has no launchable market id",
                details={"reason": "not_launchable", "job_id": str(job_id)},
            )
        return BacktestVariantLaunchSnapshot(
            job_id=job.job_id,
            owner_user_id=job.user_id,
            job_state=job.state,
            request_hash=job.request_hash,
            result_config_hash=job.engine_params_hash,
            market_id=int(market_id),
            exchange=str(coordinates.get("exchange", "binance")),
            market_type=str(coordinates.get("market_type", "spot")),
            symbol=str(coordinates.get("symbol", job.symbol)),
            timeframe=str(job.timeframe),
            variant_key=str(payload.get("public_variant_key") or variant_key),
            variant_hash=str(payload.get("variant_hash") or row.variant_key),
            indicator_variant_hash=(
                str(payload.get("indicator_variant_hash") or row.indicator_variant_key)
                if (payload.get("indicator_variant_hash") or row.indicator_variant_key)
                else None
            ),
            rank=row.rank,
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=_mapping(payload.get("canonical_variant_params")),
            readable_params=_mapping(payload.get("readable_params")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _lazy_trades_cache_ttl_seconds(*, environ: Mapping[str, str]) -> int:
    raw = environ.get("ROEHUB_BACKTEST_DETAIL_CACHE_TTL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS
    return int(raw)


__all__ = [
    "build_backtests_router",
]
