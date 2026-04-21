from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from trading.contexts.backtest.adapters.outbound.artifacts_fs import BacktestArtifactPathBuilderV2
from trading.contexts.backtest.adapters.outbound.config import (
    BacktestArtifactsRuntimeConfig,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactCoordinatesV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotLiteralV2,
    ArtifactSlotValidationSpecV2,
    inactive_artifact_slot_v2,
)


@dataclass(frozen=True, slots=True)
class SyntheticArtifactStoreV2:
    """
    Synthetic artifact store fixture with deterministic strict manifests and `.npy` payloads.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py
    """

    builder: BacktestArtifactPathBuilderV2
    loader: YamlBacktestArtifactLoaderV2
    coordinates: ArtifactCoordinatesV2
    validation_spec: ArtifactSlotValidationSpecV2
    active_slot: ArtifactSlotLiteralV2
    inactive_slot: ArtifactSlotLiteralV2


@dataclass(frozen=True, slots=True)
class ArtifactPrecomputeFixtureV2:
    """
    Minimal R3-02 fixture with strict config, pointer file, and explicit artifact paths.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    config_path: Path
    runtime_config: BacktestArtifactsRuntimeConfig
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2
    builder: BacktestArtifactPathBuilderV2
    loader: YamlBacktestArtifactLoaderV2
    coordinates: ArtifactCoordinatesV2
    active_slot: ArtifactSlotLiteralV2
    inactive_slot: ArtifactSlotLiteralV2


def build_synthetic_artifact_store_v2(
    *,
    tmp_path: Path,
    active_slot: ArtifactSlotLiteralV2 = "slot_a",
    current_slot_generation: int = 4,
    inactive_slot_generation: int = 5,
    inactive_signal_values: np.ndarray | None = None,
    inactive_mapping_open_idx: np.ndarray | None = None,
    inactive_mapping_close_idx: np.ndarray | None = None,
    inactive_long_tp: np.ndarray | None = None,
    inactive_long_sl: np.ndarray | None = None,
    inactive_short_tp: np.ndarray | None = None,
    inactive_short_sl: np.ndarray | None = None,
    active_include_signal_features: bool = True,
    inactive_include_signal_features: bool = True,
    omit_inactive_files: tuple[str, ...] = (),
) -> SyntheticArtifactStoreV2:
    """
    Build a deterministic two-slot artifact tree with strict R2-03 manifests under `tmp_path`.

    Args:
        tmp_path: pytest temporary path fixture.
        active_slot: Slot literal referenced by `current.yaml`.
        current_slot_generation: Current published slot generation.
        inactive_slot_generation: Slot generation written into the inactive slot manifests.
        inactive_signal_values: Optional override for the inactive signal matrix.
        inactive_mapping_open_idx: Optional override for inactive `bar_open_1m_idx`.
        inactive_mapping_close_idx: Optional override for inactive `bar_close_1m_idx`.
        inactive_long_tp: Optional override for inactive `long_tp`.
        inactive_long_sl: Optional override for inactive `long_sl`.
        inactive_short_tp: Optional override for inactive `short_tp`.
        inactive_short_sl: Optional override for inactive `short_sl`.
        active_include_signal_features: Whether the active slot publishes additive
            `signal_features` metadata and files.
        inactive_include_signal_features: Whether the inactive slot publishes additive
            `signal_features` metadata and files.
        omit_inactive_files: Optional tuple of slot-relative paths to skip for the inactive slot.
    Returns:
        SyntheticArtifactStoreV2: Builder, loader, coordinates, validation spec, and slot ids.
    Assumptions:
        Tests need a valid strict artifact store by default and mutate only the inactive slot.
    Raises:
        OSError: If one synthetic artifact file cannot be written.
    Side Effects:
        Creates a deterministic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    inactive_slot: ArtifactSlotLiteralV2 = "slot_b" if active_slot == "slot_a" else "slot_a"
    validation_spec = ArtifactSlotValidationSpecV2(
        price_timeframes=("1m", "15m"),
        mapping_timeframes=("15m",),
        signal_artifacts=(ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.ema"),),
        require_hit_times_manifest=True,
    )

    _write_slot_payloads(
        builder=builder,
        coordinates=coordinates,
        slot=active_slot,
        slot_generation=current_slot_generation,
        asof_date="2026-03-25",
        signal_values=_default_signal_values(),
        mapping_open_idx=_default_mapping_open_idx(),
        mapping_close_idx=_default_mapping_close_idx(),
        long_tp=_default_long_tp(),
        long_sl=_default_long_sl(),
        short_tp=_default_short_tp(),
        short_sl=_default_short_sl(),
        include_signal_features=active_include_signal_features,
        omit_files=(),
    )
    _write_slot_payloads(
        builder=builder,
        coordinates=coordinates,
        slot=inactive_slot,
        slot_generation=inactive_slot_generation,
        asof_date="2026-03-26",
        signal_values=(
            inactive_signal_values.copy()
            if inactive_signal_values is not None
            else _default_signal_values()
        ),
        mapping_open_idx=(
            inactive_mapping_open_idx.copy()
            if inactive_mapping_open_idx is not None
            else _default_mapping_open_idx()
        ),
        mapping_close_idx=(
            inactive_mapping_close_idx.copy()
            if inactive_mapping_close_idx is not None
            else _default_mapping_close_idx()
        ),
        long_tp=inactive_long_tp.copy() if inactive_long_tp is not None else _default_long_tp(),
        long_sl=inactive_long_sl.copy() if inactive_long_sl is not None else _default_long_sl(),
        short_tp=(
            inactive_short_tp.copy() if inactive_short_tp is not None else _default_short_tp()
        ),
        short_sl=(
            inactive_short_sl.copy() if inactive_short_sl is not None else _default_short_sl()
        ),
        include_signal_features=inactive_include_signal_features,
        omit_files=omit_inactive_files,
    )

    current_pointer_payload = {
        "schema_version": 1,
        "active_slot": active_slot,
        "slot_generation": current_slot_generation,
        "asof_date": "2026-03-25",
        "manifest_sha256": _file_sha256_hex_v2(
            builder.slot_manifest_path(coordinates, active_slot)
        ),
        "published_at_utc": "2026-03-25T02:00:00Z",
    }
    _write_yaml(
        builder.current_pointer_path(coordinates),
        current_pointer_payload,
    )

    return SyntheticArtifactStoreV2(
        builder=builder,
        loader=loader,
        coordinates=coordinates,
        validation_spec=validation_spec,
        active_slot=active_slot,
        inactive_slot=inactive_slot,
    )


def build_artifact_precompute_fixture_v2(
    *,
    tmp_path: Path,
    active_slot: ArtifactSlotLiteralV2 = "slot_a",
    current_slot_generation: int = 4,
    price_tail_bars_1m: int = 2,
    mapping_tail_bars_1m: int = 10,
    signal_tail_bars_1m: int = 10,
    hit_times_tail_bars_1m: int = 10,
    hit_times_tp_levels_pct: tuple[float, ...] = (1.0,),
    hit_times_sl_levels_pct: tuple[float, ...] = (1.0,),
    validation_price_timeframes: tuple[str, ...] = ARTIFACT_PRICE_TIMEFRAMES_V2,
    validation_mapping_timeframes: tuple[str, ...] = ARTIFACT_MAPPING_TIMEFRAMES_V2,
    validation_signal_artifacts: tuple[tuple[str, str], ...] | str = (),
    precompute_signal_artifacts: tuple[tuple[str, str], ...] | str = (),
    require_hit_times_manifest: bool = False,
    max_hit_times_cells: int = 1_000_000,
    max_hit_times_cells_full_rebuild: int | None = None,
    max_open_timeframe_sessions: int = 1,
    signal_worker_processes: int = 4,
    signal_worker_memory_budget_bytes: int = 2_147_483_648,
    signal_chunk_rows_min: int = 32,
    signal_chunk_rows_max: int = 256,
) -> ArtifactPrecomputeFixtureV2:
    """
    Build a minimal strict R3-02 fixture with config and `current.yaml` only.

    Args:
        tmp_path: pytest temporary path fixture.
        active_slot: Active slot literal referenced by `current.yaml`.
        current_slot_generation: Current published slot generation.
        price_tail_bars_1m: Strict positive `prices/1m` tail reread budget.
        mapping_tail_bars_1m: Strict positive `mappings/<tf>` tail rebuild budget in `1m`
            bars.
        signal_tail_bars_1m: Strict positive `signals/<tf>/<indicator_id>` tail rebuild budget
            expressed in `1m` bars.
        hit_times_tail_bars_1m: Strict positive `hit_times/1m` tail rebuild budget in canonical
            `1m` bars.
        hit_times_tp_levels_pct: Explicit TP levels in human-percent units written into
            `backtest_artifacts.hit_times_grid.tp_levels_pct`.
        hit_times_sl_levels_pct: Explicit SL levels in human-percent units written into
            `backtest_artifacts.hit_times_grid.sl_levels_pct`.
        validation_signal_artifacts: Explicit `(timeframe, indicator_id)` targets or the
            machine-readable literal `all_supported_v1` written into
            `backtest_artifacts.validation_plan.signal_artifacts`.
        validation_price_timeframes: Explicit `prices/<tf>` validation-plan timeframes.
        validation_mapping_timeframes: Explicit `mappings/<tf>` validation-plan timeframes.
        precompute_signal_artifacts: Explicit `(timeframe, indicator_id)` targets or the same
            machine-readable literal `all_supported_v1` enabled for real R12 signal
            materialization by the runner.
        require_hit_times_manifest: Whether the generated runtime config should require real
            `hit_times/1m/manifest.yaml` during whole-slot validation.
        max_hit_times_cells: Strict positive upper bound for materialized hit-times table cells.
        max_hit_times_cells_full_rebuild: Optional strict positive upper bound for full-rebuild
            or bootstrap hit-times table cells. When omitted, uses `max_hit_times_cells`.
        max_open_timeframe_sessions: Strict upper bound for simultaneously open timeframe
            sessions inside the R12 coordinator.
        signal_worker_processes: Strict worker-process upper bound reserved for later chunked
            signal execution.
        signal_worker_memory_budget_bytes: Strict per-worker memory ceiling reserved for later
            chunked signal execution.
        signal_chunk_rows_min: Smallest acceptable signal chunk size reserved for later chunked
            signal execution.
        signal_chunk_rows_max: Largest acceptable signal chunk size reserved for later chunked
            signal execution.
    Returns:
        ArtifactPrecomputeFixtureV2: Strict config/loader/path fixture for R3-02 runner tests.
    Assumptions:
        Runner tests own inactive-slot contents and start without prebuilt `prices/<tf>` files,
        while R3-04 tests may still request a full later-stage validation plan from the same
        source-of-truth config.
    Raises:
        OSError: If config or pointer files cannot be written.
    Side Effects:
        Creates strict config YAML and `current.yaml` under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    inactive_slot = inactive_artifact_slot_v2(active_slot)
    current_pointer_payload = {
        "schema_version": 1,
        "active_slot": active_slot,
        "slot_generation": current_slot_generation,
        "asof_date": "2026-03-25",
        "manifest_sha256": "0" * 64,
        "published_at_utc": "2026-03-25T02:00:00Z",
    }
    _write_yaml(builder.current_pointer_path(coordinates), current_pointer_payload)
    effective_full_rebuild_budget = (
        max_hit_times_cells
        if max_hit_times_cells_full_rebuild is None
        else max_hit_times_cells_full_rebuild
    )
    config_path = tmp_path / "backtest_artifacts.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backtest_artifacts": {
                    "artifact_root": str(builder.root),
                    "validation_plan": {
                        "price_timeframes": list(validation_price_timeframes),
                        "mapping_timeframes": list(validation_mapping_timeframes),
                        "signal_artifacts": _serialize_signal_artifacts_config_v2(
                            signal_artifacts=validation_signal_artifacts
                        ),
                        "require_hit_times_manifest": require_hit_times_manifest,
                    },
                    "hit_times_grid": {
                        "tp_levels_pct": list(hit_times_tp_levels_pct),
                        "sl_levels_pct": list(hit_times_sl_levels_pct),
                    },
                    "slot_policy": {"slots": ["slot_a", "slot_b"]},
                    "publish_schedule": {
                        "full_rebuild_hour_utc": 2,
                        "full_rebuild_minute_utc": 0,
                    },
                    "lookback_policy": {
                        "price_tail_bars_1m": price_tail_bars_1m,
                        "mapping_tail_bars_1m": mapping_tail_bars_1m,
                        "signal_tail_bars_1m": signal_tail_bars_1m,
                        "hit_times_tail_bars_1m": hit_times_tail_bars_1m,
                    },
                    "validation_budgets": {
                            "max_price_bars_per_timeframe": 1000000,
                            "max_mapping_rows_per_timeframe": 1000000,
                            "max_signal_rows_per_artifact": 1000000,
                            "max_hit_times_cells": max_hit_times_cells,
                            "max_hit_times_cells_full_rebuild": (
                                effective_full_rebuild_budget
                            ),
                        },
                    "execution_policy": {
                        "max_open_timeframe_sessions": max_open_timeframe_sessions,
                        "signal_worker_processes": signal_worker_processes,
                        "signal_worker_memory_budget_bytes": (
                            signal_worker_memory_budget_bytes
                        ),
                        "signal_chunk_rows_min": signal_chunk_rows_min,
                        "signal_chunk_rows_max": signal_chunk_rows_max,
                    },
                    },
                },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runtime_config = load_backtest_artifacts_runtime_config(config_path)
    return ArtifactPrecomputeFixtureV2(
        config_path=config_path,
        runtime_config=runtime_config,
        runtime_settings=ArtifactPrecomputeRuntimeSettingsV2(
            price_tail_bars_1m=runtime_config.lookback_policy.price_tail_bars_1m,
            mapping_tail_bars_1m=runtime_config.lookback_policy.mapping_tail_bars_1m,
            signal_tail_bars_1m=runtime_config.lookback_policy.signal_tail_bars_1m,
            hit_times_tail_bars_1m=runtime_config.lookback_policy.hit_times_tail_bars_1m,
            hit_times_tp_levels_pct=runtime_config.hit_times_grid.tp_levels_pct,
            hit_times_sl_levels_pct=runtime_config.hit_times_grid.sl_levels_pct,
            price_timeframes=runtime_config.validation_plan.price_timeframes,
            mapping_timeframes=runtime_config.validation_plan.mapping_timeframes,
            config_sha256=build_backtest_artifacts_runtime_config_hash(config=runtime_config),
            execution_policy=runtime_config.execution_policy.to_execution_policy(),
            signal_artifacts=_runtime_settings_signal_artifacts_v2(
                runtime_config=runtime_config,
                precompute_signal_artifacts=precompute_signal_artifacts,
            ),
            max_signal_rows_per_artifact=(
                runtime_config.validation_budgets.max_signal_rows_per_artifact
            ),
            max_hit_times_cells=runtime_config.validation_budgets.max_hit_times_cells,
            max_hit_times_cells_full_rebuild=(
                runtime_config.validation_budgets.max_hit_times_cells_full_rebuild
            ),
        ),
        builder=builder,
        loader=loader,
        coordinates=coordinates,
        active_slot=active_slot,
        inactive_slot=inactive_slot,
    )


def _serialize_signal_artifacts_config_v2(
    *,
    signal_artifacts: tuple[tuple[str, str], ...] | str,
) -> str | list[Mapping[str, str]]:
    """
    Serialize fixture signal-target inputs into the strict YAML config shape.

    Args:
        signal_artifacts: Explicit `(timeframe, indicator_id)` targets or the literal
            `all_supported_v1`.
    Returns:
        str | list[Mapping[str, str]]: YAML-ready literal or explicit target list.
    Assumptions:
        Fixture helpers need to cover both explicit small test target sets and full-registry
        `all_supported_v1` expansion through the real config loader.
    Raises:
        ValueError: If the literal value is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    if isinstance(signal_artifacts, str):
        normalized_literal = signal_artifacts.strip().lower()
        if normalized_literal != "all_supported_v1":
            raise ValueError(
                "signal_artifacts literal must be 'all_supported_v1'; got "
                f"{signal_artifacts!r}"
            )
        return normalized_literal
    return [
        {
            "timeframe": timeframe,
            "indicator_id": indicator_id,
        }
        for timeframe, indicator_id in signal_artifacts
    ]


def _runtime_settings_signal_artifacts_v2(
    *,
    runtime_config: BacktestArtifactsRuntimeConfig,
    precompute_signal_artifacts: tuple[tuple[str, str], ...] | str,
) -> tuple[ArtifactSignalValidationSpecV2, ...]:
    """
    Resolve fixture precompute targets into runner runtime settings.

    Args:
        runtime_config: Loaded strict runtime config used as source-of-truth for literal
            expansion.
        precompute_signal_artifacts: Explicit `(timeframe, indicator_id)` targets or the literal
            `all_supported_v1`.
    Returns:
        tuple[ArtifactSignalValidationSpecV2, ...]: Runner-ready deterministic target tuple.
    Assumptions:
        Explicit precompute targets may intentionally differ from the full validation plan in
        tests, while the `all_supported_v1` literal should reuse the config loader's canonical
        expansion order.
    Raises:
        ValueError: If the literal value is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    if isinstance(precompute_signal_artifacts, str):
        normalized_literal = precompute_signal_artifacts.strip().lower()
        if normalized_literal != "all_supported_v1":
            raise ValueError(
                "precompute_signal_artifacts literal must be 'all_supported_v1'; got "
                f"{precompute_signal_artifacts!r}"
            )
        return tuple(
            item.to_validation_spec() for item in runtime_config.validation_plan.signal_artifacts
        )
    return tuple(
        ArtifactSignalValidationSpecV2(
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        for timeframe, indicator_id in precompute_signal_artifacts
    )


def _write_slot_payloads(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    slot_generation: int,
    asof_date: str,
    signal_values: np.ndarray,
    mapping_open_idx: np.ndarray,
    mapping_close_idx: np.ndarray,
    long_tp: np.ndarray,
    long_sl: np.ndarray,
    short_tp: np.ndarray,
    short_sl: np.ndarray,
    include_signal_features: bool,
    omit_files: tuple[str, ...],
) -> None:
    """
    Write one complete slot payload with strict manifests and artifact files.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal to populate.
        slot_generation: Slot generation written into manifests.
        asof_date: As-of date literal written into manifests.
        signal_values: Signal matrix for `signals/15m/ma.ema/signals.i8.npy`.
        mapping_open_idx: Mapping open indexes for `mappings/15m/bar_open_1m_idx.u32.npy`.
        mapping_close_idx: Mapping close indexes for `mappings/15m/bar_close_1m_idx.u32.npy`.
        long_tp: Hit-times `long_tp`.
        long_sl: Hit-times `long_sl`.
        short_tp: Hit-times `short_tp`.
        short_sl: Hit-times `short_sl`.
        include_signal_features: Whether to publish additive `signal_features` for this slot.
        omit_files: Optional tuple of slot-relative file paths to skip.
    Returns:
        None.
    Assumptions:
        Helper writes files first, then signal/hit manifests, then root manifest for hash linking.
    Raises:
        OSError: If one file cannot be written.
    Side Effects:
        Creates files and manifests under one slot root.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    one_minute_open_time = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    one_minute_close_time = np.array([1599, 2599, 3599, 4599], dtype=np.int64)
    one_minute_ohlcv = np.array(
        [
            [1.0, 1.1, 0.9, 1.05, 10.0],
            [1.05, 1.15, 1.0, 1.1, 12.0],
            [1.1, 1.2, 1.05, 1.15, 9.0],
            [1.15, 1.25, 1.1, 1.2, 11.0],
        ],
        dtype=np.float32,
    )
    fifteen_minute_open_time = np.array([1000, 3000], dtype=np.int64)
    fifteen_minute_close_time = np.array([2599, 4599], dtype=np.int64)
    fifteen_minute_ohlcv = np.array(
        [
            [1.0, 1.15, 0.9, 1.1, 22.0],
            [1.1, 1.25, 1.05, 1.2, 20.0],
        ],
        dtype=np.float32,
    )
    tp_values = np.array([0.01, 0.02], dtype=np.float32)
    sl_values = np.array([0.01, 0.02], dtype=np.float32)

    price_1m_paths = builder.price_paths(coordinates, slot, "1m")
    price_15m_paths = builder.price_paths(coordinates, slot, "15m")
    mapping_paths = builder.mapping_paths(coordinates, slot, "15m")
    signal_paths = builder.signal_paths(coordinates, slot, "15m", "ma.ema")
    signal_features_paths = builder.signal_features_paths(coordinates, slot, "15m", "ma.ema")
    hit_times_paths = builder.hit_times_paths(coordinates, slot)

    _write_npy_if_needed(
        price_1m_paths.open_time,
        one_minute_open_time,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            price_1m_paths.open_time,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        price_1m_paths.close_time,
        one_minute_close_time,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            price_1m_paths.close_time,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        price_1m_paths.ohlcv,
        one_minute_ohlcv,
        slot_relative_path=_slot_relative_path(builder, coordinates, slot, price_1m_paths.ohlcv),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        price_15m_paths.open_time,
        fifteen_minute_open_time,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            price_15m_paths.open_time,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        price_15m_paths.close_time,
        fifteen_minute_close_time,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            price_15m_paths.close_time,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        price_15m_paths.ohlcv,
        fifteen_minute_ohlcv,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            price_15m_paths.ohlcv,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        mapping_paths.bar_open_1m_idx,
        mapping_open_idx,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            mapping_paths.bar_open_1m_idx,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        mapping_paths.bar_close_1m_idx,
        mapping_close_idx,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            mapping_paths.bar_close_1m_idx,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        signal_paths.signals,
        signal_values,
        slot_relative_path=_slot_relative_path(builder, coordinates, slot, signal_paths.signals),
        omit_files=omit_files,
    )
    signal_features_reference_payload: dict[str, str] | None = None
    if include_signal_features:
        signal_feature_matrix = _build_signal_feature_matrix_v2(signal_values=signal_values)
        _write_npy_if_needed(
            signal_features_paths.features,
            signal_feature_matrix,
            slot_relative_path=_slot_relative_path(
                builder,
                coordinates,
                slot,
                signal_features_paths.features,
            ),
            omit_files=omit_files,
        )
        signal_features_manifest_payload = {
            "schema_version": 1,
            "manifest_kind": "signal_features",
            "slot": slot,
            "slot_generation": slot_generation,
            "asof_date": asof_date,
            "indicator_id": "ma.ema",
            "timeframe": "15m",
            "features": _array_metadata_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                absolute_path=signal_features_paths.features,
            ),
            "rows_count": int(signal_values.shape[0]),
            "feature_names": list(SIGNAL_FEATURE_NAMES_V2),
            "provenance": _provenance_payload(),
        }
        _write_yaml(signal_features_paths.manifest, signal_features_manifest_payload)
        signal_features_reference_payload = {
            "manifest_path": _slot_relative_path(
                builder,
                coordinates,
                slot,
                signal_features_paths.manifest,
            ),
            "manifest_sha256": _file_sha256_hex_v2(signal_features_paths.manifest),
        }
    _write_npy_if_needed(
        hit_times_paths.tp_values,
        tp_values,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            hit_times_paths.tp_values,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        hit_times_paths.sl_values,
        sl_values,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            hit_times_paths.sl_values,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        hit_times_paths.long_tp,
        long_tp,
        slot_relative_path=_slot_relative_path(builder, coordinates, slot, hit_times_paths.long_tp),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        hit_times_paths.long_sl,
        long_sl,
        slot_relative_path=_slot_relative_path(builder, coordinates, slot, hit_times_paths.long_sl),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        hit_times_paths.short_tp,
        short_tp,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            hit_times_paths.short_tp,
        ),
        omit_files=omit_files,
    )
    _write_npy_if_needed(
        hit_times_paths.short_sl,
        short_sl,
        slot_relative_path=_slot_relative_path(
            builder,
            coordinates,
            slot,
            hit_times_paths.short_sl,
        ),
        omit_files=omit_files,
    )

    signal_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "signal",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "indicator_id": "ma.ema",
        "timeframe": "15m",
        "signals": _array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=signal_paths.signals,
        ),
        "rows_count": int(signal_values.shape[0]),
        "timeline": _timeline_payload(
            open_time=fifteen_minute_open_time,
            close_time=fifteen_minute_close_time,
        ),
        "signal_value_set": [-1, 0, 1],
        "grid": {
            "variant_key_version": 1,
            "variant_keys_sha256": "d" * 64,
            "signals_v1_params_defaults": {},
        },
        "provenance": _provenance_payload(),
    }
    if signal_features_reference_payload is not None:
        signal_manifest_payload["signal_features"] = signal_features_reference_payload
    if _slot_relative_path(builder, coordinates, slot, signal_paths.manifest) not in omit_files:
        _write_yaml(signal_paths.manifest, signal_manifest_payload)

    hit_times_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "hit_times_1m",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "timeframe": "1m",
        "timeline_bar_count": int(one_minute_open_time.shape[0]),
        "sentinel_index": int(one_minute_open_time.shape[0]),
        "tp_values": _level_array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=hit_times_paths.tp_values,
        ),
        "sl_values": _level_array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=hit_times_paths.sl_values,
        ),
        "tables": {
            "long_tp": _hit_times_table_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                absolute_path=hit_times_paths.long_tp,
            ),
            "long_sl": _hit_times_table_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                absolute_path=hit_times_paths.long_sl,
            ),
            "short_tp": _hit_times_table_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                absolute_path=hit_times_paths.short_tp,
            ),
            "short_sl": _hit_times_table_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                absolute_path=hit_times_paths.short_sl,
            ),
        },
        "provenance": _provenance_payload(),
    }
    if _slot_relative_path(builder, coordinates, slot, hit_times_paths.manifest) not in omit_files:
        _write_yaml(hit_times_paths.manifest, hit_times_manifest_payload)

    root_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "slot_root",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "identity": {
            "exchange": coordinates.exchange,
            "market_type": coordinates.market_type,
            "symbol": coordinates.symbol,
        },
        "prices": [
            _price_manifest_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                timeframe="1m",
                open_time_path=price_1m_paths.open_time,
                close_time_path=price_1m_paths.close_time,
                ohlcv_path=price_1m_paths.ohlcv,
                open_time=one_minute_open_time,
                close_time=one_minute_close_time,
            ),
            _price_manifest_payload(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                timeframe="15m",
                open_time_path=price_15m_paths.open_time,
                close_time_path=price_15m_paths.close_time,
                ohlcv_path=price_15m_paths.ohlcv,
                open_time=fifteen_minute_open_time,
                close_time=fifteen_minute_close_time,
            ),
        ],
        "mappings": [
            {
                "timeframe": "15m",
                "bar_open_1m_idx": _array_metadata_payload(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    absolute_path=mapping_paths.bar_open_1m_idx,
                ),
                "bar_close_1m_idx": _array_metadata_payload(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    absolute_path=mapping_paths.bar_close_1m_idx,
                ),
            }
        ],
        "signals": {
            "supported_timeframes": ["15m"],
            "supported_indicator_ids": ["ma.ema"],
            "manifests": [
                {
                    "timeframe": "15m",
                    "indicator_id": "ma.ema",
                    "manifest_path": _slot_relative_path(
                        builder,
                        coordinates,
                        slot,
                        signal_paths.manifest,
                    ),
                    "manifest_sha256": _file_sha256_hex_v2(signal_paths.manifest),
                }
            ],
        },
        "hit_times": {
            "timeframe": "1m",
            "manifest_path": _slot_relative_path(
                builder,
                coordinates,
                slot,
                hit_times_paths.manifest,
            ),
            "manifest_sha256": _file_sha256_hex_v2(hit_times_paths.manifest),
        },
        "signal_encoding": {
            "dtype": "int8",
            "axis_order": ["variant", "time"],
            "value_set": [-1, 0, 1],
        },
        "provenance": _provenance_payload(),
    }
    root_manifest_relative_path = _slot_relative_path(
        builder,
        coordinates,
        slot,
        builder.slot_manifest_path(coordinates, slot),
    )
    if root_manifest_relative_path not in omit_files:
        _write_yaml(builder.slot_manifest_path(coordinates, slot), root_manifest_payload)

    for omitted_relative_path in omit_files:
        omitted_path = builder.slot_root(coordinates, slot) / omitted_relative_path
        if omitted_path.exists():
            omitted_path.unlink()


def _price_manifest_payload(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    timeframe: str,
    open_time_path: Path,
    close_time_path: Path,
    ohlcv_path: Path,
    open_time: np.ndarray,
    close_time: np.ndarray,
) -> dict[str, Any]:
    """
    Build one strict root-manifest price section payload.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal being serialized.
        timeframe: Price timeframe literal.
        open_time_path: Absolute `open_time` path.
        close_time_path: Absolute `close_time` path.
        ohlcv_path: Absolute `ohlcv` path.
        open_time: Materialized open-time array.
        close_time: Materialized close-time array.
    Returns:
        dict[str, Any]: Strict root-manifest price section payload.
    Assumptions:
        OHLCV arrays always use shape `(T, 5)` in this synthetic fixture.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return {
        "timeframe": timeframe,
        "open_time": _array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=open_time_path,
        ),
        "close_time": _array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=close_time_path,
        ),
        "ohlcv": _array_metadata_payload(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            absolute_path=ohlcv_path,
        ),
        "coverage": _timeline_payload(open_time=open_time, close_time=close_time),
    }


def _array_metadata_payload(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    absolute_path: Path,
) -> dict[str, Any]:
    """
    Build one strict array metadata payload from an existing `.npy` file.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal being serialized.
        absolute_path: Absolute `.npy` path.
    Returns:
        dict[str, Any]: Strict array metadata payload.
    Assumptions:
        Arrays were already written to disk before metadata generation.
    Raises:
        FileNotFoundError: If the file does not exist.
    Side Effects:
        Reads the `.npy` file from disk to derive dtype and shape.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    array = np.load(absolute_path, mmap_mode="r", allow_pickle=False)
    return {
        "path": _slot_relative_path(builder, coordinates, slot, absolute_path),
        "dtype": array.dtype.name,
        "shape": [int(value) for value in array.shape],
        "axis_order": _axis_order_for_shape(array.shape),
        "sha256": _file_sha256_hex_v2(absolute_path),
    }


def _hit_times_table_payload(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    absolute_path: Path,
) -> dict[str, Any]:
    """
    Build one strict hit-times table metadata payload from an existing `.npy` file.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal being serialized.
        absolute_path: Absolute hit-times table path.
    Returns:
        dict[str, Any]: Strict hit-times table payload.
    Assumptions:
        Table arrays were already written to disk before metadata generation.
    Raises:
        FileNotFoundError: If the file does not exist.
    Side Effects:
        Reads the `.npy` file from disk to derive dtype and shape.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    payload = _array_metadata_payload(
        builder=builder,
        coordinates=coordinates,
        slot=slot,
        absolute_path=absolute_path,
    )
    payload["monotonicity"] = "non_decreasing_by_level"
    return payload


def _level_array_metadata_payload(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    absolute_path: Path,
) -> dict[str, Any]:
    """
    Build strict metadata payload for one hit-times level grid array.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal being serialized.
        absolute_path: Absolute grid-array path.
    Returns:
        dict[str, Any]: Strict array metadata payload with `axis_order=['level']`.
    Assumptions:
        TP/SL level arrays are one-dimensional and use the `level` axis contract.
    Raises:
        FileNotFoundError: If the file does not exist.
    Side Effects:
        Reads the `.npy` file from disk to derive dtype and shape.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    payload = _array_metadata_payload(
        builder=builder,
        coordinates=coordinates,
        slot=slot,
        absolute_path=absolute_path,
    )
    payload["axis_order"] = ["level"]
    return payload


def _timeline_payload(*, open_time: np.ndarray, close_time: np.ndarray) -> dict[str, Any]:
    """
    Build strict timeline coverage payload from paired open/close arrays.

    Args:
        open_time: Open-time integer array.
        close_time: Close-time integer array.
    Returns:
        dict[str, Any]: Strict timeline coverage payload.
    Assumptions:
        Arrays are non-empty and aligned by row count.
    Raises:
        IndexError: If one array is empty.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return {
        "bar_count": int(open_time.shape[0]),
        "open_time_start": int(open_time[0]),
        "open_time_end": int(open_time[-1]),
        "close_time_start": int(close_time[0]),
        "close_time_end": int(close_time[-1]),
    }


def _provenance_payload() -> dict[str, Any]:
    """
    Build a deterministic provenance payload used by all synthetic manifests.

    Args:
        None.
    Returns:
        dict[str, Any]: Strict provenance payload.
    Assumptions:
        Synthetic tests need stable non-empty provenance and hash literals only.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return {
        "generator": "backtest-precompute-runner-v2",
        "generator_version": "r4-02",
        "generated_at_utc": "2026-03-26T03:00:00Z",
        "config_sha256": "a" * 64,
        "inputs_sha256": "b" * 64,
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write one YAML payload with deterministic field order.

    Args:
        path: Target file path.
        payload: YAML payload to serialize.
    Returns:
        None.
    Assumptions:
        Callers already prepared a deterministic key order in the input mapping.
    Raises:
        OSError: If the file cannot be written.
    Side Effects:
        Creates parent directories and writes UTF-8 YAML to disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_npy_if_needed(
    path: Path,
    array: np.ndarray,
    *,
    slot_relative_path: str,
    omit_files: tuple[str, ...],
) -> None:
    """
    Write one `.npy` file unless the caller requested to omit that exact relative path.

    Args:
        path: Target `.npy` path.
        array: Array payload to serialize.
        slot_relative_path: Slot-relative path literal used for omit matching.
        omit_files: Tuple of slot-relative paths to skip.
    Returns:
        None.
    Assumptions:
        Missing-file tests omit only inactive-slot files after metadata generation design time.
    Raises:
        OSError: If the file cannot be written.
    Side Effects:
        Creates parent directories and writes `.npy` bytes to disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        np.save(file_handle, array, allow_pickle=False)


def _slot_relative_path(
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    absolute_path: Path,
) -> str:
    """
    Convert one absolute slot-local path into the canonical slot-relative literal.

    Args:
        builder: Deterministic artifact path builder.
        coordinates: Artifact coordinates under test.
        slot: Slot literal being serialized.
        absolute_path: Absolute file path under the slot root.
    Returns:
        str: POSIX-style slot-relative path literal.
    Assumptions:
        All synthetic artifact paths live strictly under the slot root.
    Raises:
        ValueError: If the path is outside the slot root.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return absolute_path.relative_to(builder.slot_root(coordinates, slot)).as_posix()


def _axis_order_for_shape(shape: tuple[int, ...]) -> list[str]:
    """
    Infer the deterministic axis-order literal used by the synthetic strict manifests.

    Args:
        shape: Array shape tuple.
    Returns:
        list[str]: Canonical axis-order literal list for the array family.
    Assumptions:
        Synthetic fixture uses only the fixed artifact families needed by R2-03 tests.
    Raises:
        ValueError: If shape does not match one of the supported artifact families.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    if shape == (4,) or shape == (2,):
        return ["time"]
    if shape == (4, 5) or shape == (2, 5):
        return ["time", "field"]
    if shape == (2, 2):
        return ["variant", "time"]
    if len(shape) == 2 and shape[1] == len(SIGNAL_FEATURE_NAMES_V2):
        return ["variant", "feature"]
    if shape == (2, 4):
        return ["level", "time"]
    raise ValueError(f"unsupported synthetic shape for axis order inference: {shape!r}")


def _file_sha256_hex_v2(path: Path) -> str:
    """
    Compute a lowercase SHA-256 hex digest for one file.

    Args:
        path: Existing file path to hash.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        Synthetic tests use file hashes to mirror strict publish-time manifest contracts.
    Raises:
        OSError: If the file cannot be read.
    Side Effects:
        Reads the file from disk in binary mode.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_signal_values() -> np.ndarray:
    """
    Return the default valid synthetic signal matrix.

    Args:
        None.
    Returns:
        np.ndarray: Valid `int8` signal matrix with values from `{-1,0,1}`.
    Assumptions:
        Tests need a small deterministic `[variant, time]` signal fixture.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([[-1, 0], [1, 0]], dtype=np.int8)


def _build_signal_feature_matrix_v2(*, signal_values: np.ndarray) -> np.ndarray:
    """
    Derive the fixed additive row-local feature matrix from synthetic signal rows.

    Args:
        signal_values: Synthetic strict signal matrix with shape `[variant, time]`.
    Returns:
        np.ndarray: Contiguous `float32` feature matrix with shape `[variant, feature]`.
    Assumptions:
        Synthetic fixtures must mirror the production feature ordering exactly for loader and
        validator tests.
    Raises:
        ValueError: If the provided signal matrix is not two-dimensional or has empty axes.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    if signal_values.ndim != 2:
        raise ValueError(f"signal_values must be 2D; got ndim={signal_values.ndim!r}")
    row_count = int(signal_values.shape[0])
    timeline_length = int(signal_values.shape[1])
    if row_count <= 0 or timeline_length <= 0:
        raise ValueError("signal_values must have positive variant and timeline dimensions")
    nonzero_count = np.count_nonzero(signal_values != 0, axis=1).astype(np.float32, copy=False)
    long_count = np.count_nonzero(signal_values > 0, axis=1).astype(np.float32, copy=False)
    short_count = np.count_nonzero(signal_values < 0, axis=1).astype(np.float32, copy=False)
    activity_ratio = np.ascontiguousarray(
        nonzero_count / np.float32(timeline_length),
        dtype=np.float32,
    )
    direction_balance = np.zeros(row_count, dtype=np.float32)
    np.divide(
        long_count - short_count,
        nonzero_count,
        out=direction_balance,
        where=nonzero_count > 0.0,
    )
    if timeline_length < 2:
        transition_count = np.zeros(row_count, dtype=np.float32)
    else:
        transition_count = np.count_nonzero(
            signal_values[:, 1:] != signal_values[:, :-1],
            axis=1,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(
        np.column_stack(
            (
                nonzero_count,
                long_count,
                short_count,
                activity_ratio,
                direction_balance,
                transition_count,
            )
        ),
        dtype=np.float32,
    )


def _default_mapping_open_idx() -> np.ndarray:
    """
    Return the default valid synthetic mapping-open array.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` open-index mapping array.
    Assumptions:
        Default mappings align `15m` bars to the four-bar synthetic `1m` timeline.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([0, 2], dtype=np.uint32)


def _default_mapping_close_idx() -> np.ndarray:
    """
    Return the default valid synthetic mapping-close array.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` close-index mapping array.
    Assumptions:
        Default mappings align `15m` bars to the four-bar synthetic `1m` timeline.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([1, 3], dtype=np.uint32)


def _default_long_tp() -> np.ndarray:
    """
    Return the default valid synthetic `long_tp` hit-times table.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` hit-times table.
    Assumptions:
        Table values stay within the four-bar sentinel contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([[1, 2, 4, 4], [1, 3, 4, 4]], dtype=np.uint32)


def _default_long_sl() -> np.ndarray:
    """
    Return the default valid synthetic `long_sl` hit-times table.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` hit-times table.
    Assumptions:
        Table values stay within the four-bar sentinel contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([[1, 2, 4, 4], [2, 3, 4, 4]], dtype=np.uint32)


def _default_short_tp() -> np.ndarray:
    """
    Return the default valid synthetic `short_tp` hit-times table.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` hit-times table.
    Assumptions:
        Table values stay within the four-bar sentinel contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([[1, 2, 4, 4], [2, 3, 4, 4]], dtype=np.uint32)


def _default_short_sl() -> np.ndarray:
    """
    Return the default valid synthetic `short_sl` hit-times table.

    Args:
        None.
    Returns:
        np.ndarray: Valid monotone `uint32` hit-times table.
    Assumptions:
        Table values stay within the four-bar sentinel contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
    """
    return np.array([[1, 2, 4, 4], [1, 3, 4, 4]], dtype=np.uint32)
