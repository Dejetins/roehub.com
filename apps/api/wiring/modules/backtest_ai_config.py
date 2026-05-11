from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter

from apps.api.routes import (
    build_backtest_ai_config_router as build_backtest_ai_config_api_router,
)
from apps.api.wiring.modules.backtest import build_backtest_ai_configurator_use_cases
from trading.contexts.backtest.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    DeterministicBacktestConfigLLMGateway,
    FilesystemBacktestArtifactContextResolver,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiCatalogResolver,
    BacktestAiConfigFakeWorkerUseCase,
    BacktestAiConfigPipeline,
    BacktestAiConfigValidator,
    BacktestAiInputGate,
    BacktestAiOutputGate,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency


def build_backtest_ai_config_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_ai_config_router requires current_user_dependency")
    use_cases = build_backtest_ai_configurator_use_cases(environ=environ)
    return build_backtest_ai_config_api_router(
        current_user_dependency=current_user_dependency,
        jobs_use_case=None if use_cases is None else use_cases.jobs,
    )


def build_backtest_ai_config_fake_worker(
    *,
    environ: Mapping[str, str],
) -> BacktestAiConfigFakeWorkerUseCase | None:
    use_cases = build_backtest_ai_configurator_use_cases(environ=environ)
    if use_cases is None:
        return None
    return BacktestAiConfigFakeWorkerUseCase(
        job_repository=use_cases.jobs.repository,
        lease_repository=use_cases.lease_repository,
        pipeline=_build_pipeline(environ=environ),
        lease_seconds=use_cases.runtime_config.queue.lease_seconds,
        max_attempts=use_cases.runtime_config.queue.repair_attempts + 1,
    )


def _build_pipeline(*, environ: Mapping[str, str]) -> BacktestAiConfigPipeline:
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
    return BacktestAiConfigPipeline(
        catalog_resolver=BacktestAiCatalogResolver(
            runtime_defaults_service=runtime_defaults_service,
            supported_symbols=_discover_ai_artifact_symbols(artifact_config=artifact_config),
        ),
        validator=BacktestAiConfigValidator(
            preflight_service=BacktestPreflightService(
                defaults_provider=defaults_provider,
                artifact_context_resolver=artifact_context_resolver,
                runtime_config=runtime_config,
            ),
            output_gate=BacktestAiOutputGate(),
        ),
        input_gate=BacktestAiInputGate(),
        llm_gateway=DeterministicBacktestConfigLLMGateway(),
    )


def _with_local_dev_default(*, environ: Mapping[str, str]) -> Mapping[str, str]:
    if environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "").strip():
        return environ
    if environ.get("ROEHUB_ENV", "").strip():
        return environ
    return {**environ, "ROEHUB_ENV": "dev"}


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
    "build_backtest_ai_config_fake_worker",
    "build_backtest_ai_config_router",
]
