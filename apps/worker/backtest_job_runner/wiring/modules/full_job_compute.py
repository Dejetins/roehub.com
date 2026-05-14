from __future__ import annotations

from pathlib import Path
from typing import Mapping

from trading.contexts.backtest.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    YamlBacktestGridDefaultsProvider,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2.combo_planning import (
    BacktestComboPlanningConfig,
    BacktestComboPlanningService,
)
from trading.contexts.backtest.application.services.v2.job_orchestration import (
    BacktestRuntimeJobOrchestrationService,
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


def build_full_job_compute_executor(
    *,
    environ: Mapping[str, str],
) -> BacktestRuntimeJobOrchestrationService:
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
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(
                combo_top_frac=1.0,
                combo_min_confirm=1,
            ),
        ),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=BacktestTpSlHitTimesService(
            artifact_array_loader=artifact_array_loader
        ),
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=artifact_array_loader,
    )
