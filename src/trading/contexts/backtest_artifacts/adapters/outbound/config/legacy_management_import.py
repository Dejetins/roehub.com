"""Explicit legacy import projection; callers page only observed/configured coordinates."""

from dataclasses import asdict
from typing import Any

from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    BacktestArtifactsRuntimeConfig,
    build_backtest_artifacts_runtime_config_hash,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest_artifacts.application.services.management import Coordinate, digest


def legacy_demand(
    config: BacktestArtifactsRuntimeConfig,
    coordinate: Coordinate,
    *,
    defaults: YamlBacktestGridDefaultsProvider,
) -> tuple[Coordinate, str, dict[str, Any]]:
    policy = next(
        (
            p
            for p in config.retention_policy
            if (p.exchange, p.market_type, p.symbol) == coordinate.key
        ),
        None,
    )
    indicator_defaults = {}
    for name in sorted({s.indicator_id for s in config.validation_plan.signal_artifacts}):
        grid = defaults.compute_defaults(indicator_id=name)
        indicator_defaults[name] = {
            "sources": list(defaults.allowed_source_values(indicator_id=name)),
            "params": {
                key: list(spec.materialize()) for key, spec in (grid.params.items() if grid else ())
            },
            "signal_params": {
                key: list(spec.materialize())
                for key, spec in defaults.signal_param_defaults(indicator_id=name).items()
            },
        }
    snapshot = {
        "signals": policy.signals if policy else "precompute",
        "hit_times": policy.hit_times if policy else "precompute",
        "signal_artifacts": [asdict(s) for s in config.validation_plan.signal_artifacts],
        "tp_levels_pct": list(config.hit_times_grid.tp_levels_pct),
        "sl_levels_pct": list(config.hit_times_grid.sl_levels_pct),
        "origin": "explicit_override" if policy else "implicit_legacy_precompute",
        "indicator_defaults": indicator_defaults,
    }
    return (
        coordinate,
        digest(
            {
                "runtime_config": build_backtest_artifacts_runtime_config_hash(config=config),
                "indicator_defaults": indicator_defaults,
            }
        ),
        snapshot,
    )
