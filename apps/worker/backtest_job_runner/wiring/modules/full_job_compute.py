from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from apps.api.wiring.modules.indicators import (
    build_artifact_precompute_indicators_compute,
    build_indicators_registry,
)
from trading.contexts.backtest.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.dto.artifact_inputs import BacktestAttemptInputs
from trading.contexts.backtest.application.services.v2.combo_planning import (
    BacktestComboPlanningConfig,
    BacktestComboPlanningService,
)
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy
from trading.contexts.backtest.application.services.v2.job_orchestration import (
    BacktestRuntimeJobOrchestrationService,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    resolve_backtest_numba_thread_decision,
)
from trading.contexts.backtest.application.services.v2.no_risk_exact import (
    BacktestNoRiskExactScoringService,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_exact import (
    BacktestTpSlExactScoringService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
    BacktestArtifactManifestValidatorV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (
    BacktestArtifactPrecomputeRunnerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
)
from trading.contexts.indicators.application.services import GridBuilder
from trading.platform.config import load_indicators_compute_numba_config


def build_full_job_compute_executor(
    *,
    environ: Mapping[str, str],
    compute_policy: BacktestComputePolicy | None = None,
    attempt_inputs: BacktestAttemptInputs | None = None,
) -> BacktestRuntimeJobOrchestrationService:
    policy = compute_policy or BacktestComputePolicy(
        threads=resolve_backtest_numba_thread_decision(
            environ=environ, scheduling_class="heavy", inherited=True
        )
    )
    artifact_config_path = resolve_backtest_artifacts_config_path(environ=environ)
    artifact_config = load_backtest_artifacts_runtime_config(Path(artifact_config_path))
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=environ,
        artifact_config_path=Path(artifact_config_path),
    )
    artifact_path_builder = BacktestArtifactPathBuilderV2(
        root=artifact_config.artifact_root_path()
    )
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=artifact_path_builder)
    artifact_array_loader = FilesystemBacktestArtifactArrayLoader(
        artifact_loader=artifact_loader
    )
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
        config=BacktestPreparePoolsConfig(
            row_prefilter_top_fraction=1.0,
            row_prefilter_min_nonzero=1,
        ),
    )
    return BacktestRuntimeJobOrchestrationService(
        prepare_pools=prepare_pools,
        attempt_inputs=attempt_inputs,
        input_validator=BacktestArtifactManifestValidatorV2(artifact_loader=artifact_loader),
        derivative_builder=BacktestArtifactPrecomputeRunnerV2(
            runtime_settings=artifact_config.to_precompute_runtime_settings(
                config_sha256=build_backtest_artifacts_runtime_config_hash(config=artifact_config)
            ),
            artifact_loader=artifact_loader,
            defaults_provider=defaults_provider,
            signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
            indicator_compute=build_artifact_precompute_indicators_compute(
                environ=environ,
                config=replace(
                    load_indicators_compute_numba_config(
                        environ=environ, artifact_config_path=Path(artifact_config_path)
                    ),
                    # Builder warmup shares this child with scoring; the admitted
                    # job budget owns its thread mask, not the indicators API setting.
                    numba_num_threads=policy.threads.num_threads,
                ),
                artifact_config_path=Path(artifact_config_path),
            ),
            indicator_grid_builder=GridBuilder(
                registry=build_indicators_registry(
                    environ=environ,
                    artifact_config_path=Path(artifact_config_path),
                )
            ),
        ),
        compute_policy=policy,
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(
                combo_top_frac=1.0,
                combo_min_confirm=1,
            ),
        ),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=artifact_array_loader),
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=artifact_array_loader,
    )
