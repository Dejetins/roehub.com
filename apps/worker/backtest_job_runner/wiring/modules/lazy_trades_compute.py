from __future__ import annotations

from pathlib import Path
from typing import Mapping

from trading.contexts.backtest.adapters.outbound import (
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    BacktestArtifactPathBuilderV2,
    LocalFileBacktestLazyTradesCache,
    YamlBacktestGridDefaultsProvider,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2.lazy_trades_detail import (
    DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
    BacktestLazyTradesDetailConfig,
    BacktestLazyTradesDetailService,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)


def build_lazy_trades_compute_service(
    *,
    environ: Mapping[str, str],
) -> BacktestLazyTradesDetailService:
    artifact_config_path = resolve_backtest_artifacts_config_path(environ=environ)
    artifact_config = load_backtest_artifacts_runtime_config(Path(artifact_config_path))
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=environ,
        artifact_config_path=Path(artifact_config_path),
    )
    artifact_path_builder = BacktestArtifactPathBuilderV2(root=artifact_config.artifact_root_path())
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=artifact_path_builder)
    artifact_array_loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=artifact_loader)
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
    )
    return BacktestLazyTradesDetailService(
        prepare_pools=prepare_pools,
        tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=artifact_array_loader),
        cache=LocalFileBacktestLazyTradesCache(
            root=Path(
                environ.get(
                    "ROEHUB_BACKTEST_TRADES_CACHE_ROOT",
                    str(DEFAULT_LAZY_TRADES_CACHE_ROOT),
                )
            )
        ),
        config=BacktestLazyTradesDetailConfig(
            cache_ttl_seconds=_lazy_trades_cache_ttl_seconds(environ=environ)
        ),
    )


def _lazy_trades_cache_ttl_seconds(*, environ: Mapping[str, str]) -> int:
    raw = environ.get("ROEHUB_BACKTEST_DETAIL_CACHE_TTL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS
    return int(raw)
