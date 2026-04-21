"""
Composition helpers for indicators API module.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md,
  docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md,
  docs/architecture/indicators/indicators-ma-compute-numba-v1.md
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from trading.contexts.indicators.adapters.outbound import (
    MarketDataCandleFeed,
    NumbaIndicatorCompute,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.platform.config import (
    IndicatorsComputeNumbaConfig,
    load_indicators_compute_numba_config,
    resolve_indicators_config_path,
)

_ARTIFACT_PRECOMPUTE_MAX_COMPUTE_BYTES_TOTAL = sys.maxsize


def build_indicators_registry(
    *,
    environ: Mapping[str, str],
    artifact_config_path: str | Path | None = None,
) -> YamlIndicatorRegistry:
    """
    Build fail-fast indicators registry from environment-aware YAML config.

    Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

    Args:
        environ: Process environment mapping.
        artifact_config_path: Optional explicit artifact-config path used to derive the matching
            sibling `indicators.yaml` for artifact-aware wiring.
    Returns:
        YamlIndicatorRegistry: Ready-to-use merged registry adapter.
    Assumptions:
        Config path resolves to `configs/<env>/indicators.yaml` unless overridden.
    Raises:
        FileNotFoundError: If YAML file does not exist.
        ValueError: If environment/config is invalid or YAML validation fails.
    Side Effects:
        Reads defaults YAML from filesystem.
    """
    config_path = resolve_indicators_config_path(
        environ=environ,
        artifact_config_path=artifact_config_path,
    )
    return YamlIndicatorRegistry.from_yaml(
        defs=all_defs(),
        config_path=config_path,
    )


def build_indicators_compute(
    *,
    environ: Mapping[str, str],
    config: IndicatorsComputeNumbaConfig | None = None,
) -> NumbaIndicatorCompute:
    """
    Build indicators CPU/Numba compute adapter and run startup warmup.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md

    Args:
        environ: Process environment mapping.
        config: Optional preloaded runtime config to avoid duplicate disk/env reads.
    Returns:
        NumbaIndicatorCompute: Warmed-up compute adapter instance.
    Assumptions:
        Numba runtime settings are loaded from env + indicators YAML config.
    Raises:
        FileNotFoundError: If indicators config path cannot be resolved/read.
        ValueError: If runtime config is invalid or cache dir is not writable.
    Side Effects:
        Applies Numba runtime config and performs JIT warmup at startup.
    """
    compute_config = config or load_indicators_compute_numba_config(environ=environ)
    compute = NumbaIndicatorCompute(defs=all_defs(), config=compute_config)
    compute.warmup()
    return compute


def build_artifact_precompute_indicators_compute(
    *,
    environ: Mapping[str, str],
    config: IndicatorsComputeNumbaConfig | None = None,
    artifact_config_path: str | Path | None = None,
) -> NumbaIndicatorCompute:
    """
    Build a dedicated indicators compute adapter for offline artifact precompute.

    Docs: docs/architecture/backtest/README.md

    Args:
        environ: Process environment mapping.
        config: Optional preloaded runtime config to avoid duplicate disk/env reads.
        artifact_config_path: Optional explicit artifact-config path used to derive the matching
            sibling `indicators.yaml` when no explicit override is provided.
    Returns:
        NumbaIndicatorCompute: Warmed-up compute adapter with an effectively unbounded total
            compute-budget guard for offline artifact materialization.
    Assumptions:
        Artifact publish is an offline batch flow guarded by artifact-specific slot validation and
        should not inherit the public API/runtime `max_compute_bytes_total` ceiling from
        `configs/<env>/indicators.yaml`.
    Raises:
        FileNotFoundError: If indicators config path cannot be resolved/read.
        ValueError: If runtime config is invalid or cache dir is not writable.
    Side Effects:
        Applies Numba runtime config and performs JIT warmup at startup.
    """
    base_config = config or load_indicators_compute_numba_config(
        environ=environ,
        artifact_config_path=artifact_config_path,
    )
    precompute_config = replace(
        base_config,
        max_compute_bytes_total=_ARTIFACT_PRECOMPUTE_MAX_COMPUTE_BYTES_TOTAL,
    )
    return build_indicators_compute(environ=environ, config=precompute_config)


def build_indicators_candle_feed(
    *,
    canonical_candle_reader: CanonicalCandleReader,
) -> CandleFeed:
    """
    Bind indicators `CandleFeed` port to `market_data_acl` adapter implementation.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/application/ports/feeds/candle_feed.py,
      src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py,
      src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py

    Args:
        canonical_candle_reader: market_data canonical 1m reader port implementation.
    Returns:
        CandleFeed: Ready adapter for dense timeline loading.
    Assumptions:
        `canonical_candle_reader` enforces source-specific dedup rules for canonical candles.
    Raises:
        ValueError: If reader dependency is missing.
    Side Effects:
        None.
    """
    return MarketDataCandleFeed(canonical_candle_reader=canonical_candle_reader)


def bind_indicators_runtime_dependencies(
    *,
    app_state: object,
    compute: IndicatorCompute,
    candle_feed: CandleFeed | None = None,
) -> None:
    """
    Bind indicators compute/feed runtime dependencies into FastAPI app state.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.main.app,
      apps.api.routes.indicators,
      src/trading/contexts/indicators/application/ports/compute/indicator_compute.py

    Args:
        app_state: Mutable app state object (typically `FastAPI.state`).
        compute: Configured indicators compute adapter.
        candle_feed: Optional CandleFeed adapter for `/indicators/compute`.
    Returns:
        None.
    Assumptions:
        `app_state` supports dynamic attribute assignment.
    Raises:
        ValueError: If compute dependency is missing.
    Side Effects:
        Mutates app state by setting `indicators_compute` and `indicators_candle_feed`.
    """
    if compute is None:  # type: ignore[truthy-bool]
        raise ValueError("bind_indicators_runtime_dependencies requires compute")
    setattr(app_state, "indicators_compute", compute)
    setattr(app_state, "indicators_candle_feed", candle_feed)
