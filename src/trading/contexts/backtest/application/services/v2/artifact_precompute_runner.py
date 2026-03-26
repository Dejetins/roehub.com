"""Deterministic R3-03/R4-02 artifact materialization into the inactive slot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    indicator_primary_output_series_from_tensor_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, ComputeRequest
from trading.contexts.indicators.application.dto.variant_key import (
    IndicatorVariantSelection,
    build_variant_key_v1,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.services import GridBuilder
from trading.contexts.indicators.domain.entities import IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

from .contracts import (
    ARTIFACT_MANIFEST_FILENAME_V2,
    ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PLACEHOLDER_SHA256_V2,
    ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
    ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    ARTIFACT_SIGNAL_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_VALUE_SET_V2,
    ARTIFACT_TIME_AXIS_ORDER_V2,
    HIT_TIMES_DIRECTORY_LITERAL_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    SIGNAL_ARTIFACT_MANIFEST_KIND_V2,
    SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
    ArtifactArrayMetadataV2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactCoordinatesV2,
    ArtifactHitTimesReferenceV2,
    ArtifactManifestDocumentV2,
    ArtifactManifestProvenanceV2,
    ArtifactMappingPathsV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactPricePathsV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalCatalogV2,
    ArtifactSignalEncodingContractV2,
    ArtifactSignalGridContractV2,
    ArtifactSignalManifestDocumentV2,
    ArtifactSignalPathsV2,
    ArtifactSignalValidationSpecV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    SignalRuleEvaluationRequestV2,
    artifact_market_id_from_coordinates_v2,
    inactive_artifact_slot_v2,
    validate_artifact_slot_v2,
)
from .signal_rules_engine_v2 import BacktestSignalRulesEngineV2

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2 = "1m"
_CANONICAL_CANDLE_SOURCE_LITERAL_V2 = "market_data.canonical_candles_1m"
_PRECOMPUTE_GENERATOR_LITERAL_V2 = "backtest-artifact-precompute-runner-v2"
_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2 = "r4-02"
_ONE_MINUTE_MILLIS_V2 = 60 * 1000
_ROLLED_PRICE_TIMEFRAME_LITERALS_V2 = tuple(
    timeframe
    for timeframe in ARTIFACT_PRICE_TIMEFRAMES_V2
    if timeframe != _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2
)


@dataclass(frozen=True, slots=True)
class _CanonicalPriceArraysV2:
    """
    Internal immutable container for `open_time/close_time/ohlcv` price arrays.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    open_time: np.ndarray
    close_time: np.ndarray
    ohlcv: np.ndarray


@dataclass(frozen=True, slots=True)
class _TimeframeMappingArraysV2:
    """
    Internal immutable container for one `tf -> 1m` mapping artifact family.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    bar_open_1m_idx: np.ndarray
    bar_close_1m_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class _CanonicalPriceTailPlanV2:
    """
    Internal deterministic plan describing prefix reuse and source reread bounds.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    prefix: _CanonicalPriceArraysV2 | None
    source_time_range: TimeRange


@dataclass(frozen=True, slots=True)
class _RootManifestScaffoldV2:
    """
    Internal scaffold for root-manifest sections not owned by R3-03 price/mapping materialization.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    preserved_prices: tuple[ArtifactPriceTimeframeManifestV2, ...]
    mappings: tuple[ArtifactMappingTimeframeManifestV2, ...]
    signals: ArtifactSignalCatalogV2
    hit_times: ArtifactHitTimesReferenceV2
    signal_encoding: ArtifactSignalEncodingContractV2


@dataclass(frozen=True, slots=True)
class _SignalVariantRowV2:
    """
    Internal immutable row descriptor for one exported signal-variant selection.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
    """

    inputs_source: str | None
    variant_key: str


@dataclass(frozen=True, slots=True)
class _SignalArtifactBuildResultV2:
    """
    Internal immutable output of one full R4-02 signal materialization pass.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    catalog: ArtifactSignalCatalogV2
    manifests: tuple[ArtifactSignalManifestDocumentV2, ...]


@dataclass(frozen=True, slots=True)
class BacktestArtifactPrecomputeRunnerV2:
    """
    Materialize canonical prices, mappings, and optional R4-02 signal artifacts.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """

    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2
    artifact_loader: BacktestArtifactLoaderV2
    canonical_candle_reader: CanonicalCandleReader
    defaults_provider: BacktestGridDefaultsProvider | None = None
    signal_rules_engine: BacktestSignalRulesEngineV2 | None = None
    indicator_compute: IndicatorCompute | None = None
    indicator_grid_builder: GridBuilder | None = None

    def __post_init__(self) -> None:
        """
        Validate mandatory dependencies for deterministic inactive-slot precompute.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Loader paths are already wired to `backtest_artifacts.artifact_root`.
        Raises:
            ValueError: If runtime config, artifact loader, or candle reader is missing.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/config/
            backtest_artifacts_runtime_config.py
        """
        if self.runtime_settings is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPrecomputeRunnerV2.runtime_settings is required")
        if self.artifact_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPrecomputeRunnerV2.artifact_loader is required")
        if self.canonical_candle_reader is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestArtifactPrecomputeRunnerV2.canonical_candle_reader is required"
            )
        signal_dependencies = (
            self.defaults_provider,
            self.signal_rules_engine,
            self.indicator_compute,
            self.indicator_grid_builder,
        )
        if any(dependency is not None for dependency in signal_dependencies) and any(
            dependency is None for dependency in signal_dependencies
        ):
            raise ValueError(
                "BacktestArtifactPrecomputeRunnerV2 signal materialization requires "
                "defaults_provider, signal_rules_engine, indicator_compute, and "
                "indicator_grid_builder together"
            )
        if len(self.runtime_settings.signal_artifacts) > 0 and any(
            dependency is None for dependency in signal_dependencies
        ):
            raise ValueError(
                "BacktestArtifactPrecomputeRunnerV2 signal_artifacts require "
                "defaults_provider, signal_rules_engine, indicator_compute, and "
                "indicator_grid_builder"
            )

    def export_canonical_price_1m(
        self,
        request: ArtifactCanonicalPriceExportRequestV2,
    ) -> ArtifactCanonicalPriceExportResultV2:
        """
        Export canonical `1m`, rolled `prices/<tf>`, and `tf -> 1m` mappings into the inactive
        slot.

        Args:
            request: Explicit export identity with symbol coordinates and `TimeRange [start, end)`.
        Returns:
            ArtifactCanonicalPriceExportResultV2: Structured write result for the inactive slot.
        Assumptions:
            Public API stays rooted in canonical `1m`, while R3-03 also materializes every
            allowed request timeframe plus deterministic `tf -> 1m` mapping arrays under the same
            root manifest update.
        Raises:
            FileNotFoundError: If strict `current.yaml` is missing for the symbol root.
            ValueError: If existing inactive-slot metadata, source candles, or derived
                price-to-mapping correspondence violate strict ordering/dtype/path contracts.
            OSError: If one atomic file write fails.
        Side Effects:
            Reads canonical candles through the port and atomically replaces inactive-slot price,
            mapping, and root-manifest files.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        current_pointer = self.artifact_loader.load_current_pointer(request.coordinates)
        inactive_slot = inactive_artifact_slot_v2(current_pointer.active_slot)
        target_slot_generation = current_pointer.slot_generation + 1
        price_paths = self.artifact_loader.resolve_price_paths(
            request.coordinates,
            inactive_slot,
            _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
        )
        manifest_path = self.artifact_loader.resolve_slot_manifest_path(
            request.coordinates,
            inactive_slot,
        )
        slot_root = manifest_path.parent
        existing_manifest = _load_existing_inactive_manifest_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            manifest_path=manifest_path,
        )
        existing_arrays = _load_existing_canonical_price_arrays_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            existing_manifest=existing_manifest,
        )
        tail_plan = _build_tail_plan_v2(
            request=request,
            existing_arrays=existing_arrays,
            lookback_bars=self.runtime_settings.price_tail_bars_1m,
        )
        source_rows = tuple(
            self.canonical_candle_reader.read_1m(
                instrument_id=_instrument_id_from_coordinates_v2(request.coordinates),
                time_range=tail_plan.source_time_range,
            )
        )
        tail_arrays = _canonical_price_arrays_from_rows_v2(
            rows=source_rows,
            source_time_range=tail_plan.source_time_range,
        )
        materialized_arrays = _merge_canonical_price_arrays_v2(
            prefix=tail_plan.prefix,
            tail=tail_arrays,
        )
        _validate_canonical_price_arrays_v2(
            arrays=materialized_arrays,
            label="materialized canonical prices/1m",
        )
        _validate_rollup_source_one_minute_arrays_v2(
            arrays=materialized_arrays,
            label="materialized canonical prices/1m",
        )
        _write_price_arrays_atomically_v2(price_paths=price_paths, arrays=materialized_arrays)
        one_minute_manifest = _build_one_minute_price_manifest_v2(
            slot_root=slot_root,
            price_paths=price_paths,
            arrays=materialized_arrays,
        )
        rollup_source_arrays = _load_materialized_price_arrays_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
            manifest_section=one_minute_manifest,
            location_prefix="materialized prices[1m] rollup source",
        )
        rolled_price_manifests = _materialize_rolled_price_timeframes_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            slot_root=slot_root,
            existing_manifest=existing_manifest,
            source_arrays=rollup_source_arrays,
            source_tail_time_range=tail_plan.source_time_range,
        )
        mapping_manifests = _materialize_mapping_timeframes_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            slot_root=slot_root,
            existing_manifest=existing_manifest,
            one_minute_arrays=rollup_source_arrays,
            price_manifests=rolled_price_manifests,
            mapping_tail_bars_1m=self.runtime_settings.mapping_tail_bars_1m,
        )
        scaffold = _build_root_manifest_scaffold_v2(existing_manifest=existing_manifest)
        price_manifest_by_timeframe = {
            section.timeframe: section
            for section in (one_minute_manifest, *rolled_price_manifests)
        }
        signal_build_result = _materialize_signal_artifacts_v2(
            artifact_loader=self.artifact_loader,
            coordinates=request.coordinates,
            slot=inactive_slot,
            slot_root=slot_root,
            request=request,
            slot_generation=target_slot_generation,
            runtime_settings=self.runtime_settings,
            signal_targets=self.runtime_settings.signal_artifacts,
            price_manifest_by_timeframe=price_manifest_by_timeframe,
            defaults_provider=self.defaults_provider,
            signal_rules_engine=self.signal_rules_engine,
            indicator_compute=self.indicator_compute,
            indicator_grid_builder=self.indicator_grid_builder,
        )
        root_signals = (
            scaffold.signals if signal_build_result is None else signal_build_result.catalog
        )
        effective_scaffold = _RootManifestScaffoldV2(
            preserved_prices=scaffold.preserved_prices,
            mappings=scaffold.mappings,
            signals=root_signals,
            hit_times=scaffold.hit_times,
            signal_encoding=scaffold.signal_encoding,
        )
        provenance = _build_root_manifest_provenance_v2(
            runtime_settings=self.runtime_settings,
            request=request,
            arrays=materialized_arrays,
            rolled_sections=rolled_price_manifests,
            mapping_sections=mapping_manifests,
            signal_entries=root_signals.manifests,
        )
        root_manifest_payload = _build_root_manifest_payload_v2(
            request=request,
            slot=inactive_slot,
            slot_generation=target_slot_generation,
            root_scaffold=effective_scaffold,
            price_manifests=(one_minute_manifest, *rolled_price_manifests),
            mapping_manifests=mapping_manifests,
            provenance=provenance,
        )
        _write_yaml_atomically_v2(path=manifest_path, payload=root_manifest_payload)
        return ArtifactCanonicalPriceExportResultV2(
            coordinates=request.coordinates,
            slot=inactive_slot,
            slot_generation=target_slot_generation,
            asof_date=request.asof_date,
            manifest_path=manifest_path,
            manifest_sha256=_file_sha256_hex_v2(manifest_path),
            price_paths=price_paths,
            coverage=one_minute_manifest.coverage,
            source_time_range=tail_plan.source_time_range,
            source_candle_count=len(source_rows),
            reused_prefix_bars=(
                0 if tail_plan.prefix is None else int(tail_plan.prefix.open_time.shape[0])
            ),
            rewritten_tail_bars=int(tail_arrays.open_time.shape[0]),
        )


def _materialize_signal_artifacts_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_targets: tuple[ArtifactSignalValidationSpecV2, ...],
    price_manifest_by_timeframe: Mapping[str, ArtifactPriceTimeframeManifestV2],
    defaults_provider: BacktestGridDefaultsProvider | None,
    signal_rules_engine: BacktestSignalRulesEngineV2 | None,
    indicator_compute: IndicatorCompute | None,
    indicator_grid_builder: GridBuilder | None,
) -> _SignalArtifactBuildResultV2 | None:
    """
    Materialize all configured R4-02 signal artifacts for the current inactive slot.

    Args:
        artifact_loader: Explicit-path artifact loader used to resolve signal and price paths.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the signal artifacts.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying slot identity and timestamps.
        runtime_settings: Strict runner runtime settings with signal target and guard budgets.
        signal_targets: Canonically ordered `(timeframe, indicator_id)` signal targets.
        price_manifest_by_timeframe: Fresh materialized price sections keyed by timeframe.
        defaults_provider: Runtime defaults provider for compute grids and signal defaults.
        signal_rules_engine: Startup-validated signal rules engine.
        indicator_compute: Indicator compute port used for tensor materialization.
        indicator_grid_builder: Grid builder used for deterministic variant ordering.
    Returns:
        _SignalArtifactBuildResultV2 | None: Real signal catalog/manifests when targets are
            configured, otherwise `None`.
    Assumptions:
        R4-02 owns only explicitly configured signal targets and must not discover directories.
    Raises:
        ValueError: If one signal dependency is missing, one target lacks a price manifest, or one
            materialized signal artifact violates the strict contract.
        OSError: If writing signal arrays or manifests fails.
    Side Effects:
        Writes `signals/<tf>/<indicator_id>/signals.i8.npy` and `manifest.yaml` files.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if len(signal_targets) == 0:
        return None
    if (
        defaults_provider is None
        or signal_rules_engine is None
        or indicator_compute is None
        or indicator_grid_builder is None
    ):
        raise ValueError(
            "R4-02 signal materialization requires defaults_provider, "
            "signal_rules_engine, indicator_compute, and indicator_grid_builder"
        )

    manifests: list[ArtifactSignalManifestDocumentV2] = []
    catalog_entries: list[ArtifactSignalCatalogEntryV2] = []
    for signal_target in signal_targets:
        price_manifest = price_manifest_by_timeframe.get(signal_target.timeframe)
        if price_manifest is None:
            raise ValueError(
                "signal target requires materialized prices for timeframe "
                f"{signal_target.timeframe!r}"
            )
        signal_manifest = _materialize_signal_artifact_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            slot_root=slot_root,
            request=request,
            slot_generation=slot_generation,
            runtime_settings=runtime_settings,
            signal_target=signal_target,
            price_manifest=price_manifest,
            defaults_provider=defaults_provider,
            signal_rules_engine=signal_rules_engine,
            indicator_compute=indicator_compute,
            indicator_grid_builder=indicator_grid_builder,
        )
        manifests.append(signal_manifest)
        catalog_entries.append(
            ArtifactSignalCatalogEntryV2(
                timeframe=signal_manifest.timeframe,
                indicator_id=signal_manifest.indicator_id,
                manifest_path=_slot_relative_path_v2(
                    slot_root=slot_root,
                    absolute_path=signal_manifest.path,
                ),
                manifest_sha256=_file_sha256_hex_v2(signal_manifest.path),
            )
        )
    return _SignalArtifactBuildResultV2(
        catalog=ArtifactSignalCatalogV2(
            supported_timeframes=tuple(entry.timeframe for entry in catalog_entries),
            supported_indicator_ids=tuple(entry.indicator_id for entry in catalog_entries),
            manifests=tuple(catalog_entries),
        ),
        manifests=tuple(manifests),
    )


def _materialize_signal_artifact_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    defaults_provider: BacktestGridDefaultsProvider,
    signal_rules_engine: BacktestSignalRulesEngineV2,
    indicator_compute: IndicatorCompute,
    indicator_grid_builder: GridBuilder,
) -> ArtifactSignalManifestDocumentV2:
    """
    Materialize one strict per-timeframe/per-indicator signal artifact family.

    Args:
        artifact_loader: Explicit-path artifact loader used to resolve fixed output paths.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the signal artifact.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying root identity and timestamps.
        runtime_settings: Strict runtime settings with signal guard budgets.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        price_manifest: Fresh materialized price section for the same timeframe.
        defaults_provider: Runtime defaults provider for compute grids and signal defaults.
        signal_rules_engine: Startup-validated signal rules engine.
        indicator_compute: Indicator compute port used for primary/dependency tensors.
        indicator_grid_builder: Grid builder used for deterministic variant ordering.
    Returns:
        ArtifactSignalManifestDocumentV2: Typed strict signal manifest describing the written
            artifact family.
    Assumptions:
        Signal rows reuse v1 variant-key semantics and the fixed `[variant, time]` matrix layout.
    Raises:
        ValueError: If defaults, tensor shapes, value sets, or dependency alignment drift.
        OSError: If writing the signal matrix or manifest fails.
    Side Effects:
        Loads price arrays for the target timeframe and writes signal files under the slot root.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    price_arrays = _load_materialized_price_arrays_v2(
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        timeframe=signal_target.timeframe,
        manifest_section=price_manifest,
        location_prefix=(
            "materialized prices for signal export "
            f"{signal_target.timeframe}:{signal_target.indicator_id}"
        ),
    )
    candles = _candle_arrays_from_price_arrays_v2(
        coordinates=coordinates,
        timeframe=signal_target.timeframe,
        arrays=price_arrays,
    )
    defaults_grid = defaults_provider.compute_defaults(indicator_id=signal_target.indicator_id)
    if defaults_grid is None:
        raise ValueError(
            "signal target requires compute defaults for indicator_id "
            f"{signal_target.indicator_id!r}"
        )
    compute_grid = _grid_with_layout_v2(
        grid=defaults_grid,
        indicator_id=signal_target.indicator_id,
        layout=Layout.VARIANT_MAJOR,
    )
    materialized_grid = indicator_grid_builder.materialize_indicator(grid=compute_grid)
    primary_tensor = indicator_compute.compute(
        ComputeRequest(
            candles=candles,
            grid=compute_grid,
            max_variants_guard=runtime_settings.max_signal_rows_per_artifact,
        )
    )
    signal_rows = _build_signal_variant_rows_v2(
        coordinates=coordinates,
        timeframe=signal_target.timeframe,
        materialized_grid=materialized_grid,
    )
    if len(signal_rows) != primary_tensor.meta.variants:
        raise ValueError(
            "signal variant ordering drift detected for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; "
            f"grid rows={len(signal_rows)!r}, tensor variants={primary_tensor.meta.variants!r}"
        )
    dependency_tensors = _compute_signal_dependency_tensors_v2(
        candles=candles,
        compute_grid=compute_grid,
        indicator_id=signal_target.indicator_id,
        indicator_compute=indicator_compute,
        max_signal_rows_per_artifact=runtime_settings.max_signal_rows_per_artifact,
        signal_rules_engine=signal_rules_engine,
        expected_variants=primary_tensor.meta.variants,
        expected_t=primary_tensor.meta.t,
    )
    signal_matrix, signal_params_defaults = _evaluate_signal_matrix_v2(
        candles=candles,
        indicator_id=signal_target.indicator_id,
        primary_tensor=primary_tensor,
        dependency_tensors=dependency_tensors,
        signal_rows=signal_rows,
        signal_rules_engine=signal_rules_engine,
    )
    signal_paths = artifact_loader.resolve_signal_paths(
        coordinates,
        slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    _write_npy_atomically_v2(path=signal_paths.signals, array=signal_matrix)
    signal_manifest = _build_signal_manifest_v2(
        coordinates=coordinates,
        slot=slot,
        slot_root=slot_root,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        signal_paths=signal_paths,
        signal_matrix=signal_matrix,
        timeline=_timeline_coverage_from_arrays_v2(arrays=price_arrays),
        price_manifest=price_manifest,
        signal_rows=signal_rows,
        signal_params_defaults=signal_params_defaults,
        signal_rules_engine=signal_rules_engine,
    )
    _write_yaml_atomically_v2(
        path=signal_paths.manifest,
        payload=_serialize_signal_manifest_v2(signal_manifest),
    )
    return signal_manifest


def _candle_arrays_from_price_arrays_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    timeframe: str,
    arrays: _CanonicalPriceArraysV2,
) -> CandleArrays:
    """
    Convert one materialized `prices/<tf>` family into `CandleArrays` for indicator compute.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        timeframe: Price timeframe literal represented by the arrays.
        arrays: Strict price arrays already validated and loaded from disk.
    Returns:
        CandleArrays: Dense OHLCV arrays aligned to the target timeframe timeline.
    Assumptions:
        Price arrays already satisfy strict dtype/shape/timeline invariants before conversion.
    Raises:
        ValueError: If `CandleArrays` construction detects one alignment drift.
    Side Effects:
        Allocates contiguous float32 column vectors for indicator compute.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return CandleArrays(
        market_id=MarketId(artifact_market_id_from_coordinates_v2(coordinates)),
        symbol=Symbol(coordinates.symbol),
        time_range=TimeRange(
            start=_epoch_millis_to_utc_timestamp_v2(int(arrays.open_time[0])),
            end=_epoch_millis_to_utc_timestamp_v2(int(arrays.close_time[-1])),
        ),
        timeframe=Timeframe(timeframe),
        ts_open=np.ascontiguousarray(arrays.open_time, dtype=np.int64),
        open=np.ascontiguousarray(arrays.ohlcv[:, 0], dtype=np.float32),
        high=np.ascontiguousarray(arrays.ohlcv[:, 1], dtype=np.float32),
        low=np.ascontiguousarray(arrays.ohlcv[:, 2], dtype=np.float32),
        close=np.ascontiguousarray(arrays.ohlcv[:, 3], dtype=np.float32),
        volume=np.ascontiguousarray(arrays.ohlcv[:, 4], dtype=np.float32),
    )


def _grid_with_layout_v2(
    *,
    grid: GridSpec,
    indicator_id: str,
    layout: Layout,
) -> GridSpec:
    """
    Clone one defaults grid with explicit indicator id and layout preference.

    Args:
        grid: Source compute defaults grid.
        indicator_id: Indicator identifier to assign to the cloned grid.
        layout: Explicit tensor layout preference for compute.
    Returns:
        GridSpec: Cloned grid preserving params/source while replacing id/layout.
    Assumptions:
        R4-02 signal export always requests `variant_major` tensors for direct `[V, T]` writes.
    Raises:
        ValueError: If the cloned indicator id or grid payload is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/indicators/domain/entities/layout.py
    """
    return GridSpec(
        indicator_id=IndicatorId(indicator_id),
        params=grid.params,
        source=grid.source,
        layout_preference=layout,
    )


def _build_signal_variant_rows_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    timeframe: str,
    materialized_grid: Any,
) -> tuple[_SignalVariantRowV2, ...]:
    """
    Build deterministic signal row descriptors matching the compute tensor variant order.

    Args:
        coordinates: Artifact coordinates used for v1 variant-key identity.
        timeframe: Signal timeframe literal.
        materialized_grid: Materialized indicator grid returned by `GridBuilder`.
    Returns:
        tuple[_SignalVariantRowV2, ...]: Ordered signal-row descriptors for `[V, T_tf]` export.
    Assumptions:
        Variant flattening order matches axis cartesian-product order with the last axis varying
        fastest, consistent with compute tensor materialization.
    Raises:
        ValueError: If variant-key construction fails for one explicit row.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
    """
    instrument_id = str(_instrument_id_from_coordinates_v2(coordinates))
    axis_values = tuple(axis.values for axis in materialized_grid.axes)
    ordered_value_rows = product(*axis_values) if len(axis_values) > 0 else ((),)
    rows: list[_SignalVariantRowV2] = []
    for value_row in ordered_value_rows:
        inputs: dict[str, int | float | str] = {}
        params: dict[str, int | float | str] = {}
        resolved_source: str | None = None
        for axis, value in zip(materialized_grid.axes, value_row):
            if axis.name == "source":
                resolved_source = str(value)
                inputs["source"] = resolved_source
                continue
            params[axis.name] = value
        rows.append(
            _SignalVariantRowV2(
                inputs_source=resolved_source,
                variant_key=build_variant_key_v1(
                    instrument_id=instrument_id,
                    timeframe=timeframe,
                    indicators=(
                        IndicatorVariantSelection(
                            indicator_id=materialized_grid.indicator_id,
                            inputs=inputs,
                            params=params,
                        ),
                    ),
                ),
            )
        )
    return tuple(rows)


def _compute_signal_dependency_tensors_v2(
    *,
    candles: CandleArrays,
    compute_grid: GridSpec,
    indicator_id: str,
    indicator_compute: IndicatorCompute,
    max_signal_rows_per_artifact: int,
    signal_rules_engine: BacktestSignalRulesEngineV2,
    expected_variants: int,
    expected_t: int,
) -> Mapping[str, Any]:
    """
    Compute dependency tensors required by one signal rule family.

    Args:
        candles: Dense candle arrays aligned to the target signal timeframe.
        compute_grid: Primary indicator grid used for the target indicator.
        indicator_id: Primary indicator identifier.
        indicator_compute: Indicator compute port used for dependency tensors.
        max_signal_rows_per_artifact: Strict compute guard for dependency grids.
        signal_rules_engine: Explicit signal rules engine used to resolve dependency ids.
        expected_variants: Expected variant count shared with the primary tensor.
        expected_t: Expected timeline length shared with the primary tensor.
    Returns:
        Mapping[str, Any]: Dependency tensors keyed by dependency indicator id.
    Assumptions:
        Wrapper dependencies reuse the same parameterization and row ordering as the primary grid.
    Raises:
        ValueError: If one dependency tensor drifts in variant count or timeline length.
    Side Effects:
        Computes additional indicator tensors for composite signal rule families.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
    """
    dependency_tensors: dict[str, Any] = {}
    dependency_ids = signal_rules_engine.rule_spec(
        indicator_id=indicator_id
    ).required_dependency_ids
    for dependency_id in dependency_ids:
        dependency_grid = _grid_with_layout_v2(
            grid=compute_grid,
            indicator_id=dependency_id,
            layout=Layout.VARIANT_MAJOR,
        )
        tensor = indicator_compute.compute(
            ComputeRequest(
                candles=candles,
                grid=dependency_grid,
                max_variants_guard=max_signal_rows_per_artifact,
            )
        )
        if tensor.meta.variants != expected_variants:
            raise ValueError(
                "signal dependency variants must match primary grid for "
                f"{indicator_id!r}; dependency {dependency_id!r} produced "
                f"{tensor.meta.variants!r}, expected {expected_variants!r}"
            )
        if tensor.meta.t != expected_t:
            raise ValueError(
                "signal dependency timeline length must match primary grid for "
                f"{indicator_id!r}; dependency {dependency_id!r} produced "
                f"{tensor.meta.t!r}, expected {expected_t!r}"
            )
        dependency_tensors[dependency_id] = tensor
    return dependency_tensors


def _evaluate_signal_matrix_v2(
    *,
    candles: CandleArrays,
    indicator_id: str,
    primary_tensor: Any,
    dependency_tensors: Mapping[str, Any],
    signal_rows: tuple[_SignalVariantRowV2, ...],
    signal_rules_engine: BacktestSignalRulesEngineV2,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """
    Evaluate compact `int8` signals for every row in the exported signal matrix.

    Args:
        candles: Dense candle arrays aligned to the target signal timeframe.
        indicator_id: Primary indicator identifier.
        primary_tensor: Computed primary indicator tensor.
        dependency_tensors: Dependency tensors keyed by indicator id.
        signal_rows: Ordered row descriptors matching tensor variant order.
        signal_rules_engine: Explicit signal rules engine preserving R4-01 semantics.
    Returns:
        tuple[np.ndarray, Mapping[str, Any]]: Export-ready signal matrix and the resolved
            `signals.v1.params` default mapping serialized into the manifest grid section.
    Assumptions:
        Every row uses the same `signals.v1.params` defaults under the default-only contract.
    Raises:
        ValueError: If one evaluated series shape or value set violates the strict matrix
            contract.
    Side Effects:
        Allocates one contiguous `int8` matrix in memory.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    row_count = int(primary_tensor.meta.variants)
    time_count = int(primary_tensor.meta.t)
    signal_matrix = np.empty((row_count, time_count), dtype=np.int8)
    resolved_signal_params_defaults: Mapping[str, Any] | None = None
    for row_index, signal_row in enumerate(signal_rows):
        dependency_outputs = {
            dependency_id: indicator_primary_output_series_from_tensor_v1(
                tensor=dependency_tensor,
                variant_index=row_index,
            )
            for dependency_id, dependency_tensor in dependency_tensors.items()
        }
        evaluation_result = signal_rules_engine.evaluate(
            request=SignalRuleEvaluationRequestV2(
                indicator_id=indicator_id,
                candles=candles,
                primary_output=indicator_primary_output_series_from_tensor_v1(
                    tensor=primary_tensor,
                    variant_index=row_index,
                ),
                inputs_source=signal_row.inputs_source,
                signal_params={},
                dependency_outputs=dependency_outputs,
            )
        )
        signal_matrix[row_index, :] = evaluation_result.signal_codes
        if resolved_signal_params_defaults is None:
            resolved_signal_params_defaults = evaluation_result.signal_params
    _validate_signal_matrix_v2(
        signal_matrix=signal_matrix,
        expected_shape=(row_count, time_count),
        label=f"signals[{indicator_id}]",
    )
    return signal_matrix, dict(resolved_signal_params_defaults or {})


def _validate_signal_matrix_v2(
    *,
    signal_matrix: np.ndarray,
    expected_shape: tuple[int, int],
    label: str,
) -> None:
    """
    Validate strict R4-02 signal matrix dtype, shape, and encoded value-set invariants.

    Args:
        signal_matrix: Candidate compact signal matrix.
        expected_shape: Expected `[V, T_tf]` shape.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Signal matrices are stored only as `int8` with the fixed `{-1,0,1}` encoding.
    Raises:
        ValueError: If dtype, shape, dimensionality, or encoded values drift from the contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if signal_matrix.dtype.name != ARTIFACT_SIGNAL_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label} dtype must be {ARTIFACT_SIGNAL_DTYPE_LITERAL_V2}; "
            f"got {signal_matrix.dtype.name!r}"
        )
    if signal_matrix.ndim != 2:
        raise ValueError(f"{label} shape must be [V, T_tf]; got {signal_matrix.shape!r}")
    if signal_matrix.shape != expected_shape:
        raise ValueError(f"{label} shape must be {expected_shape!r}; got {signal_matrix.shape!r}")
    invalid_mask = (
        (signal_matrix != ARTIFACT_SIGNAL_VALUE_SET_V2[0])
        & (signal_matrix != ARTIFACT_SIGNAL_VALUE_SET_V2[1])
        & (signal_matrix != ARTIFACT_SIGNAL_VALUE_SET_V2[2])
    )
    if np.any(invalid_mask):
        raise ValueError(f"{label} values must be exactly {ARTIFACT_SIGNAL_VALUE_SET_V2!r}")


def _build_signal_manifest_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_paths: ArtifactSignalPathsV2,
    signal_matrix: np.ndarray,
    timeline: ArtifactTimelineCoverageV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    signal_rows: tuple[_SignalVariantRowV2, ...],
    signal_params_defaults: Mapping[str, Any],
    signal_rules_engine: BacktestSignalRulesEngineV2,
) -> ArtifactSignalManifestDocumentV2:
    """
    Build the strict typed per-indicator signal manifest for one freshly written matrix.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the manifest.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying root identity and timestamps.
        runtime_settings: Strict runtime settings contributing config hash and guards.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        signal_paths: Fixed signal file paths under the inactive slot.
        signal_matrix: Freshly written compact signal matrix.
        timeline: Timeline coverage matching the target `prices/<tf>` manifest.
        price_manifest: Fresh target timeframe price section used for provenance hashing.
        signal_rows: Ordered signal row descriptors used for `variant_keys_sha256`.
        signal_params_defaults: Resolved `signals.v1.params` default mapping.
        signal_rules_engine: Explicit signal rules engine used to capture dependency metadata.
    Returns:
        ArtifactSignalManifestDocumentV2: Typed strict signal manifest.
    Assumptions:
        The signal file already exists on disk and is ready for `sha256` hashing.
    Raises:
        ValueError: If one manifest field violates the strict typed contract.
        OSError: If one written signal file cannot be hashed.
    Side Effects:
        Reads the freshly written signal matrix file to compute its manifest hash.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    validated_slot = validate_artifact_slot_v2(slot)
    signal_metadata = ArtifactArrayMetadataV2(
        path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=signal_paths.signals),
        dtype=signal_matrix.dtype.name,
        shape=tuple(int(value) for value in signal_matrix.shape),
        axis_order=ARTIFACT_SIGNAL_AXIS_ORDER_V2,
        sha256=_file_sha256_hex_v2(signal_paths.signals),
    )
    grid_contract = ArtifactSignalGridContractV2(
        variant_key_version=1,
        variant_keys_sha256=_variant_keys_sha256_v2(signal_rows=signal_rows),
        signals_v1_params_defaults=signal_params_defaults,
    )
    provenance = _build_signal_manifest_provenance_v2(
        coordinates=coordinates,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        price_manifest=price_manifest,
        grid_contract=grid_contract,
        timeline=timeline,
        signal_rules_engine=signal_rules_engine,
        signal_params_defaults=signal_params_defaults,
    )
    payload = {
        "schema_version": SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        "manifest_kind": SIGNAL_ARTIFACT_MANIFEST_KIND_V2,
        "slot": validated_slot,
        "slot_generation": slot_generation,
        "asof_date": request.asof_date,
        "indicator_id": signal_target.indicator_id,
        "timeframe": signal_target.timeframe,
        "signals": _serialize_array_metadata_v2(signal_metadata),
        "rows_count": int(signal_matrix.shape[0]),
        "timeline": _serialize_timeline_coverage_v2(timeline),
        "signal_value_set": [int(value) for value in ARTIFACT_SIGNAL_VALUE_SET_V2],
        "grid": _serialize_signal_grid_contract_v2(grid_contract),
        "provenance": _serialize_provenance_v2(provenance),
    }
    return ArtifactSignalManifestDocumentV2(
        path=signal_paths.manifest,
        raw_payload=payload,
        slot=validated_slot,
        schema_version=SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        manifest_kind=SIGNAL_ARTIFACT_MANIFEST_KIND_V2,
        slot_generation=slot_generation,
        asof_date=request.asof_date,
        indicator_id=signal_target.indicator_id,
        timeframe=signal_target.timeframe,
        signals=signal_metadata,
        rows_count=int(signal_matrix.shape[0]),
        timeline=timeline,
        signal_value_set=ARTIFACT_SIGNAL_VALUE_SET_V2,
        grid=grid_contract,
        provenance=provenance,
    )


def _variant_keys_sha256_v2(
    *,
    signal_rows: tuple[_SignalVariantRowV2, ...],
) -> str:
    """
    Hash the ordered variant-key catalog used by one signal matrix.

    Args:
        signal_rows: Ordered signal row descriptors for the matrix.
    Returns:
        str: Lowercase SHA-256 digest of the ordered variant-key list.
    Assumptions:
        Runtime row addressing depends only on variant-key order, not on manifest formatting.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    canonical_json = json.dumps(
        [row.variant_key for row in signal_rows],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _load_existing_inactive_manifest_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    manifest_path: Path,
) -> ArtifactManifestDocumentV2 | None:
    """
    Load the inactive-slot root manifest when it already exists on disk.

    Args:
        artifact_loader: Explicit-path manifest loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        manifest_path: Resolved inactive-slot `manifest.yaml` path.
    Returns:
        ArtifactManifestDocumentV2 | None: Parsed manifest when present, otherwise `None`.
    Assumptions:
        Missing inactive-slot manifest means R3-01 performs a full initial build.
    Raises:
        ValueError: If an existing inactive-slot manifest violates strict root schema contracts.
    Side Effects:
        Reads one YAML manifest from disk when the file exists.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    if not manifest_path.is_file():
        return None
    return artifact_loader.load_slot_manifest(coordinates, slot)


def _load_existing_canonical_price_arrays_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    existing_manifest: ArtifactManifestDocumentV2 | None,
) -> _CanonicalPriceArraysV2 | None:
    """
    Load and validate existing inactive-slot `prices/1m` arrays for bounded tail reuse.

    Args:
        artifact_loader: Explicit-path manifest loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        existing_manifest: Already-loaded inactive-slot root manifest, if any.
    Returns:
        _CanonicalPriceArraysV2 | None: Existing arrays when `prices/1m` is already materialized.
    Assumptions:
        A valid manifest without `prices/1m` indicates first R3-01 build for that slot.
    Raises:
        ValueError: If manifest metadata or referenced files violate strict price contracts.
    Side Effects:
        Reads existing `.npy` files from the inactive slot when `prices/1m` is present.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    return _load_existing_price_timeframe_arrays_v2(
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        existing_manifest=existing_manifest,
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
    )


def _build_tail_plan_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    existing_arrays: _CanonicalPriceArraysV2 | None,
    lookback_bars: int,
) -> _CanonicalPriceTailPlanV2:
    """
    Build deterministic tail-reread bounds using `lookback_policy.price_tail_bars_1m`.

    Args:
        request: Explicit export request with the full target `TimeRange [start, end)`.
        existing_arrays: Existing inactive-slot `prices/1m` arrays when available.
        lookback_bars: Strict positive tail reread budget in `1m` bars.
    Returns:
        _CanonicalPriceTailPlanV2: Prefix reuse slice and source reread bounds.
    Assumptions:
        Tail update reuses only inactive-slot prefix inside the requested range and rereads the
        last `lookback_bars` source-of-truth rows to keep deterministic overlap/replace semantics.
    Raises:
        ValueError: If existing arrays are malformed or the derived tail time range is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if existing_arrays is None:
        return _CanonicalPriceTailPlanV2(prefix=None, source_time_range=request.time_range)

    requested_start_ms = _utc_timestamp_to_epoch_millis_v2(request.time_range.start)
    requested_end_ms = _utc_timestamp_to_epoch_millis_v2(request.time_range.end)
    existing_start_ms = int(existing_arrays.open_time[0])
    if requested_start_ms < existing_start_ms:
        return _CanonicalPriceTailPlanV2(prefix=None, source_time_range=request.time_range)

    selected_start_idx = int(
        np.searchsorted(existing_arrays.open_time, requested_start_ms, side="left")
    )
    selected_end_idx = int(
        np.searchsorted(existing_arrays.open_time, requested_end_ms, side="left")
    )
    if selected_start_idx >= selected_end_idx:
        return _CanonicalPriceTailPlanV2(prefix=None, source_time_range=request.time_range)

    selected_arrays = _slice_canonical_price_arrays_v2(
        arrays=existing_arrays,
        start_idx=selected_start_idx,
        end_idx=selected_end_idx,
    )
    if int(selected_arrays.open_time.shape[0]) <= lookback_bars:
        return _CanonicalPriceTailPlanV2(prefix=None, source_time_range=request.time_range)

    prefix_bar_count = int(selected_arrays.open_time.shape[0]) - lookback_bars
    prefix = _slice_canonical_price_arrays_v2(
        arrays=selected_arrays,
        start_idx=0,
        end_idx=prefix_bar_count,
    )
    source_start = _epoch_millis_to_utc_timestamp_v2(
        int(selected_arrays.open_time[prefix_bar_count])
    )
    return _CanonicalPriceTailPlanV2(
        prefix=prefix,
        source_time_range=TimeRange(start=source_start, end=request.time_range.end),
    )


def _instrument_id_from_coordinates_v2(coordinates: ArtifactCoordinatesV2) -> InstrumentId:
    """
    Translate artifact coordinates into the canonical market-data instrument identity.

    Args:
        coordinates: Artifact coordinates selecting one backtest symbol root.
    Returns:
        InstrumentId: Shared-kernel instrument identity for `CanonicalCandleReader.read_1m(...)`.
    Assumptions:
        R2/R3 still bridge artifact market scope to `market_id` through fixed coordinates.
    Raises:
        ValueError: If the coordinate scope has no supported market-id bridge.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """
    return InstrumentId(
        market_id=MarketId(artifact_market_id_from_coordinates_v2(coordinates)),
        symbol=Symbol(coordinates.symbol),
    )


def _canonical_price_arrays_from_rows_v2(
    *,
    rows: tuple[CandleWithMeta, ...],
    source_time_range: TimeRange,
) -> _CanonicalPriceArraysV2:
    """
    Convert canonical candle rows into contiguous strict `open_time/close_time/ohlcv` arrays.

    Args:
        rows: Canonical candle rows returned by `CanonicalCandleReader.read_1m(...)`.
        source_time_range: Exact source reread window used for stable error messages.
    Returns:
        _CanonicalPriceArraysV2: Strict contiguous arrays ready for slot materialization.
    Assumptions:
        Export keeps sparse canonical `1m` rows as-is and never backfills missing minutes with
        dense `NaN` placeholders.
    Raises:
        ValueError: If the source produced no rows or violates strict timeline monotonicity.
    Side Effects:
        Allocates contiguous numpy arrays in memory.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """
    if len(rows) == 0:
        raise ValueError(
            "canonical 1m source returned no candles for "
            f"TimeRange [start, end)={_time_range_literal_v2(source_time_range)}"
        )
    ordered_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                _utc_timestamp_to_epoch_millis_v2(row.candle.ts_open),
                _utc_timestamp_to_epoch_millis_v2(row.candle.ts_close),
            ),
        )
    )
    open_time = np.asarray(
        [_utc_timestamp_to_epoch_millis_v2(row.candle.ts_open) for row in ordered_rows],
        dtype=np.int64,
    )
    close_time = np.asarray(
        [_utc_timestamp_to_epoch_millis_v2(row.candle.ts_close) for row in ordered_rows],
        dtype=np.int64,
    )
    ohlcv = np.asarray(
        [
            (
                float(row.candle.open),
                float(row.candle.high),
                float(row.candle.low),
                float(row.candle.close),
                float(row.candle.volume_base),
            )
            for row in ordered_rows
        ],
        dtype=np.float32,
    )
    arrays = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(open_time, dtype=np.int64),
        close_time=np.ascontiguousarray(close_time, dtype=np.int64),
        ohlcv=np.ascontiguousarray(ohlcv, dtype=np.float32),
    )
    _validate_canonical_price_arrays_v2(
        arrays=arrays,
        label="canonical 1m source candles",
    )
    return arrays


def _validate_canonical_price_arrays_v2(
    *,
    arrays: _CanonicalPriceArraysV2,
    label: str,
) -> None:
    """
    Validate deterministic dtype/shape/timeline invariants for materialized price arrays.

    Args:
        arrays: Candidate `open_time/close_time/ohlcv` arrays.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Both canonical `1m` and rolled request timeframes store timestamps separately from OHLCV
        and use `volume_base` as the fifth field.
    Raises:
        ValueError: If dtypes, shapes, or monotonicity invariants are violated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if arrays.open_time.dtype.name != ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label} open_time dtype must be {ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2}; "
            f"got {arrays.open_time.dtype.name!r}"
        )
    if arrays.close_time.dtype.name != ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label} close_time dtype must be {ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2}; "
            f"got {arrays.close_time.dtype.name!r}"
        )
    if arrays.ohlcv.dtype.name != ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label} ohlcv dtype must be {ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2}; "
            f"got {arrays.ohlcv.dtype.name!r}"
        )
    if len(arrays.open_time.shape) != 1:
        raise ValueError(f"{label} open_time shape must be [T]; got {arrays.open_time.shape!r}")
    if len(arrays.close_time.shape) != 1:
        raise ValueError(f"{label} close_time shape must be [T]; got {arrays.close_time.shape!r}")
    if arrays.ohlcv.ndim != 2 or arrays.ohlcv.shape[1] != 5:
        raise ValueError(f"{label} ohlcv shape must be [T, 5]; got {arrays.ohlcv.shape!r}")
    if arrays.open_time.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one bar")
    if arrays.close_time.shape[0] != arrays.open_time.shape[0]:
        raise ValueError(
            f"{label} close_time length must equal open_time length; got "
            f"{arrays.close_time.shape[0]!r} and {arrays.open_time.shape[0]!r}"
        )
    if arrays.ohlcv.shape[0] != arrays.open_time.shape[0]:
        raise ValueError(
            f"{label} ohlcv rows must equal open_time length; got "
            f"{arrays.ohlcv.shape[0]!r} and {arrays.open_time.shape[0]!r}"
        )
    if arrays.open_time.shape[0] > 1 and not np.all(np.diff(arrays.open_time) > 0):
        raise ValueError(f"{label} must be strictly increasing by open_time")
    if arrays.close_time.shape[0] > 1 and not np.all(np.diff(arrays.close_time) > 0):
        raise ValueError(f"{label} must be strictly increasing by close_time")
    if not np.all(arrays.close_time > arrays.open_time):
        raise ValueError(f"{label} must satisfy close_time[i] > open_time[i] for every bar")


def _validate_rollup_source_one_minute_arrays_v2(
    *,
    arrays: _CanonicalPriceArraysV2,
    label: str,
) -> None:
    """
    Validate strict rollup-source invariants for canonical `prices/1m` arrays.

    Args:
        arrays: Candidate canonical `1m` arrays intended to drive derived rollups.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        R3-02 rollup reads only from materialized `prices/1m` and expects exact `1m` bucket
        boundaries with no overlapping rows.
    Raises:
        ValueError: If one timestamp is not `1m`-aligned, if `close_time != open_time + 1m`, or
            if adjacent rows overlap in time.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    _validate_canonical_price_arrays_v2(arrays=arrays, label=label)
    source_timeframe = Timeframe(_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2)
    if not np.all(
        arrays.open_time
        == np.asarray(
            [
                _bucket_open_epoch_millis_v2(timeframe=source_timeframe, value=int(open_time))
                for open_time in arrays.open_time
            ],
            dtype=np.int64,
        )
    ):
        raise ValueError(f"{label} open_time must be epoch-aligned to 1m bucket boundaries")
    expected_close = arrays.open_time + np.int64(_ONE_MINUTE_MILLIS_V2)
    if not np.array_equal(arrays.close_time, expected_close):
        raise ValueError(f"{label} close_time must equal open_time + 60000 for every 1m bar")
    if arrays.open_time.shape[0] > 1 and not np.all(arrays.open_time[1:] >= arrays.close_time[:-1]):
        raise ValueError(f"{label} must not contain overlapping 1m bars")


def _validate_rolled_price_arrays_v2(
    *,
    arrays: _CanonicalPriceArraysV2,
    timeframe: str,
) -> None:
    """
    Validate strict dtype, shape, and bucket-boundary invariants for rolled prices.

    Args:
        arrays: Candidate rolled `open_time/close_time/ohlcv` arrays.
        timeframe: Target rolled timeframe literal.
    Returns:
        None.
    Assumptions:
        R3-02 stores timestamps outside `ohlcv` and writes only epoch-aligned full buckets.
    Raises:
        ValueError: If arrays violate dtype, shape, monotonicity, or boundary alignment rules.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    label = f"rolled prices[{timeframe}]"
    _validate_canonical_price_arrays_v2(arrays=arrays, label=label)
    target_timeframe = Timeframe(timeframe)
    if not np.all(
        arrays.open_time
        == np.asarray(
            [
                _bucket_open_epoch_millis_v2(timeframe=target_timeframe, value=int(open_time))
                for open_time in arrays.open_time
            ],
            dtype=np.int64,
        )
    ):
        raise ValueError(
            f"{label} open_time must be epoch-aligned to {timeframe} bucket boundaries"
        )
    expected_close = arrays.open_time + np.int64(_timeframe_duration_millis_v2(target_timeframe))
    if not np.array_equal(arrays.close_time, expected_close):
        raise ValueError(f"{label} close_time must equal open_time + {timeframe} duration")
    if arrays.open_time.shape[0] > 1 and not np.all(arrays.open_time[1:] >= arrays.close_time[:-1]):
        raise ValueError(f"{label} must not contain overlapping rolled buckets")


def _slice_canonical_price_arrays_v2(
    *,
    arrays: _CanonicalPriceArraysV2,
    start_idx: int,
    end_idx: int,
) -> _CanonicalPriceArraysV2:
    """
    Slice canonical price arrays by row index while preserving contiguous dtypes.

    Args:
        arrays: Source canonical arrays.
        start_idx: Inclusive slice start.
        end_idx: Exclusive slice end.
    Returns:
        _CanonicalPriceArraysV2: Contiguous sliced arrays.
    Assumptions:
        Index bounds were already derived from monotone `open_time` search.
    Raises:
        ValueError: If slice indexes are negative or inconsistent.
    Side Effects:
        Allocates new contiguous array views/copies.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if start_idx < 0:
        raise ValueError(f"canonical price slice start_idx must be >= 0; got {start_idx!r}")
    if end_idx < start_idx:
        raise ValueError(
            "canonical price slice end_idx must be >= start_idx; got "
            f"{end_idx!r} and {start_idx!r}"
        )
    return _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(arrays.open_time[start_idx:end_idx], dtype=np.int64),
        close_time=np.ascontiguousarray(arrays.close_time[start_idx:end_idx], dtype=np.int64),
        ohlcv=np.ascontiguousarray(arrays.ohlcv[start_idx:end_idx], dtype=np.float32),
    )


def _merge_canonical_price_arrays_v2(
    *,
    prefix: _CanonicalPriceArraysV2 | None,
    tail: _CanonicalPriceArraysV2,
) -> _CanonicalPriceArraysV2:
    """
    Merge reused prefix bars with freshly reread tail bars in deterministic order.

    Args:
        prefix: Existing inactive-slot prefix kept unchanged before the tail overlap.
        tail: Fresh canonical source rows read from the overlap boundary onward.
    Returns:
        _CanonicalPriceArraysV2: Contiguous merged arrays.
    Assumptions:
        Prefix bars always end strictly before the first tail bar when prefix is present.
    Raises:
        ValueError: If the merged arrays violate strict canonical timeline invariants.
    Side Effects:
        Allocates merged contiguous arrays.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if prefix is None or int(prefix.open_time.shape[0]) == 0:
        return tail
    merged = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(
            np.concatenate((prefix.open_time, tail.open_time)),
            dtype=np.int64,
        ),
        close_time=np.ascontiguousarray(
            np.concatenate((prefix.close_time, tail.close_time)),
            dtype=np.int64,
        ),
        ohlcv=np.ascontiguousarray(np.concatenate((prefix.ohlcv, tail.ohlcv)), dtype=np.float32),
    )
    _validate_canonical_price_arrays_v2(
        arrays=merged,
        label="merged canonical prices/1m",
    )
    return merged


def _write_price_arrays_atomically_v2(
    *,
    price_paths: ArtifactPricePathsV2,
    arrays: _CanonicalPriceArraysV2,
) -> None:
    """
    Atomically replace inactive-slot `prices/1m/*.npy` files with deterministic bytes.

    Args:
        price_paths: Explicit inactive-slot target paths for `open_time`, `close_time`, and
            `ohlcv`.
        arrays: Strict canonical arrays to serialize.
    Returns:
        None.
    Assumptions:
        Temp files are written in the same directory so `os.replace` remains atomic.
    Raises:
        OSError: If temp-file write or atomic replace fails.
    Side Effects:
        Creates parent directories and replaces three `.npy` files under the inactive slot.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    _write_npy_atomically_v2(path=price_paths.open_time, array=arrays.open_time)
    _write_npy_atomically_v2(path=price_paths.close_time, array=arrays.close_time)
    _write_npy_atomically_v2(path=price_paths.ohlcv, array=arrays.ohlcv)


def _write_npy_atomically_v2(*, path: Path, array: np.ndarray) -> None:
    """
    Serialize one `.npy` payload through temp-file write plus atomic replace.

    Args:
        path: Canonical target `.npy` path under the inactive slot.
        array: Contiguous array payload to serialize.
    Returns:
        None.
    Assumptions:
        Callers already validated dtype/shape contracts before serialization.
    Raises:
        OSError: If temp-file write or atomic replace fails.
    Side Effects:
        Creates parent directories and replaces one `.npy` file on disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _build_one_minute_price_manifest_v2(
    *,
    slot_root: Path,
    price_paths: ArtifactPricePathsV2,
    arrays: _CanonicalPriceArraysV2,
) -> ArtifactPriceTimeframeManifestV2:
    """
    Build strict root-manifest metadata for the freshly written `prices/1m` family.

    Args:
        slot_root: Absolute inactive-slot root directory.
        price_paths: Explicit inactive-slot `prices/1m` file paths.
        arrays: Freshly written strict canonical arrays.
    Returns:
        ArtifactPriceTimeframeManifestV2: Strict `prices/1m` manifest section.
    Assumptions:
        Files were already atomically written and are ready for `sha256` calculation.
    Raises:
        OSError: If one written file cannot be hashed.
    Side Effects:
        Reads written `.npy` files to compute `sha256`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return _build_price_manifest_v2(
        slot_root=slot_root,
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
        price_paths=price_paths,
        arrays=arrays,
    )


def _build_price_manifest_v2(
    *,
    slot_root: Path,
    timeframe: str,
    price_paths: ArtifactPricePathsV2,
    arrays: _CanonicalPriceArraysV2,
) -> ArtifactPriceTimeframeManifestV2:
    """
    Build strict root-manifest metadata for one freshly written `prices/<tf>` family.

    Args:
        slot_root: Absolute inactive-slot root directory.
        timeframe: Price timeframe literal addressed by `price_paths`.
        price_paths: Explicit inactive-slot `prices/<tf>` file paths.
        arrays: Freshly written strict price arrays for the timeframe.
    Returns:
        ArtifactPriceTimeframeManifestV2: Strict `prices/<tf>` manifest section.
    Assumptions:
        Files were already atomically written and are ready for `sha256` calculation.
    Raises:
        OSError: If one written file cannot be hashed.
    Side Effects:
        Reads written `.npy` files to compute `sha256`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactPriceTimeframeManifestV2(
        timeframe=timeframe,
        open_time=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=price_paths.open_time),
            dtype=arrays.open_time.dtype.name,
            shape=tuple(int(value) for value in arrays.open_time.shape),
            axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(price_paths.open_time),
        ),
        close_time=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=price_paths.close_time),
            dtype=arrays.close_time.dtype.name,
            shape=tuple(int(value) for value in arrays.close_time.shape),
            axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(price_paths.close_time),
        ),
        ohlcv=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=price_paths.ohlcv),
            dtype=arrays.ohlcv.dtype.name,
            shape=tuple(int(value) for value in arrays.ohlcv.shape),
            axis_order=ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(price_paths.ohlcv),
        ),
        coverage=_timeline_coverage_from_arrays_v2(arrays=arrays),
    )


def _load_materialized_price_arrays_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    timeframe: str,
    manifest_section: ArtifactPriceTimeframeManifestV2,
    location_prefix: str,
) -> _CanonicalPriceArraysV2:
    """
    Load one already-materialized `prices/<tf>` family using strict manifest metadata.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Slot literal containing the price files.
        timeframe: Price timeframe literal for the section.
        manifest_section: Typed root-manifest price section referencing the files.
        location_prefix: Stable diagnostic prefix used in validation errors.
    Returns:
        _CanonicalPriceArraysV2: Contiguous validated price arrays loaded from disk.
    Assumptions:
        R3-02 rollup must read from materialized `prices/1m` artifacts rather than from source
        rows directly.
    Raises:
        FileNotFoundError: If one expected `.npy` file is missing.
        ValueError: If strict metadata or actual file contents drift from the manifest section.
    Side Effects:
        Reads three `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    price_paths = artifact_loader.resolve_price_paths(coordinates, slot, timeframe)
    open_time = _load_validated_array_v2(
        metadata=manifest_section.open_time,
        expected_path=price_paths.open_time,
        expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
        expected_shape=None,
        location=f"{location_prefix}.open_time",
    )
    close_time = _load_validated_array_v2(
        metadata=manifest_section.close_time,
        expected_path=price_paths.close_time,
        expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
        expected_shape=None,
        location=f"{location_prefix}.close_time",
    )
    ohlcv = _load_validated_array_v2(
        metadata=manifest_section.ohlcv,
        expected_path=price_paths.ohlcv,
        expected_dtype=ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
        expected_shape=None,
        location=f"{location_prefix}.ohlcv",
    )
    arrays = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(open_time, dtype=np.int64),
        close_time=np.ascontiguousarray(close_time, dtype=np.int64),
        ohlcv=np.ascontiguousarray(ohlcv, dtype=np.float32),
    )
    if timeframe == _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2:
        _validate_rollup_source_one_minute_arrays_v2(arrays=arrays, label=location_prefix)
    else:
        _validate_rolled_price_arrays_v2(arrays=arrays, timeframe=timeframe)
    expected_coverage = _timeline_coverage_from_arrays_v2(arrays=arrays)
    if manifest_section.coverage != expected_coverage:
        raise ValueError(
            f"{location_prefix}.coverage must match materialized arrays; got "
            f"{manifest_section.coverage!r}, expected {expected_coverage!r}"
        )
    return arrays


def _load_existing_price_timeframe_arrays_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    timeframe: str,
) -> _CanonicalPriceArraysV2 | None:
    """
    Load existing inactive-slot `prices/<tf>` arrays when that timeframe is already present.

    Args:
        artifact_loader: Explicit-path manifest loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        existing_manifest: Already-loaded inactive-slot root manifest, if any.
        timeframe: Price timeframe literal to load.
    Returns:
        _CanonicalPriceArraysV2 | None: Existing arrays when the timeframe is already materialized.
    Assumptions:
        Tail update may reuse a strict prefix only from the current inactive-slot artifact files.
    Raises:
        ValueError: If manifest metadata or referenced files violate strict price contracts.
    Side Effects:
        Reads existing `.npy` files from the inactive slot when the timeframe is present.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if existing_manifest is None:
        return None
    existing_section = _select_price_manifest_v2(
        price_sections=existing_manifest.prices,
        timeframe=timeframe,
    )
    if existing_section is None:
        return None
    return _load_materialized_price_arrays_v2(
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        timeframe=timeframe,
        manifest_section=existing_section,
        location_prefix=f"existing prices[{timeframe}]",
    )


def _materialize_rolled_price_timeframes_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    source_arrays: _CanonicalPriceArraysV2,
    source_tail_time_range: TimeRange,
) -> tuple[ArtifactPriceTimeframeManifestV2, ...]:
    """
    Materialize deterministic rolled price arrays for all allowed request timeframes.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving the rolled arrays.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest when present.
        source_arrays: Canonical `prices/1m` arrays loaded back from the materialized artifact.
        source_tail_time_range: Effective canonical source reread window used for the `1m` build.
    Returns:
        tuple[ArtifactPriceTimeframeManifestV2, ...]: Canonically ordered rolled price sections.
    Assumptions:
        R3-02 owns every request-level timeframe listed in `ARTIFACT_PRICE_TIMEFRAMES_V2`
        except canonical `1m`.
    Raises:
        ValueError: If source arrays, reused prefixes, or rolled outputs violate strict contracts.
        OSError: If one `.npy` write or hash read fails.
    Side Effects:
        Atomically writes `prices/<tf>/*.npy` files for every rolled timeframe.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    rolled_sections: list[ArtifactPriceTimeframeManifestV2] = []
    for timeframe in _ROLLED_PRICE_TIMEFRAME_LITERALS_V2:
        existing_arrays = _load_existing_price_timeframe_arrays_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            existing_manifest=existing_manifest,
            timeframe=timeframe,
        )
        rolled_arrays = _build_rolled_price_arrays_with_tail_update_v2(
            source_arrays=source_arrays,
            existing_arrays=existing_arrays,
            timeframe=timeframe,
            source_tail_time_range=source_tail_time_range,
        )
        price_paths = artifact_loader.resolve_price_paths(coordinates, slot, timeframe)
        _write_price_arrays_atomically_v2(price_paths=price_paths, arrays=rolled_arrays)
        rolled_sections.append(
            _build_price_manifest_v2(
                slot_root=slot_root,
                timeframe=timeframe,
                price_paths=price_paths,
                arrays=rolled_arrays,
            )
        )
    return tuple(rolled_sections)


def _select_mapping_manifest_v2(
    *,
    mapping_sections: tuple[ArtifactMappingTimeframeManifestV2, ...],
    timeframe: str,
) -> ArtifactMappingTimeframeManifestV2 | None:
    """
    Select one mapping timeframe section from typed root-manifest mapping sections.

    Args:
        mapping_sections: Typed root-manifest mapping sections.
        timeframe: Target mapping timeframe literal.
    Returns:
        ArtifactMappingTimeframeManifestV2 | None: Matching section when present.
    Assumptions:
        Typed root manifests already enforce one mapping section per timeframe.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    for section in mapping_sections:
        if section.timeframe == timeframe:
            return section
    return None


def _load_existing_mapping_arrays_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    timeframe: str,
) -> _TimeframeMappingArraysV2 | None:
    """
    Load existing inactive-slot `mappings/<tf>` arrays when that timeframe already exists.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        existing_manifest: Already-loaded inactive-slot root manifest, if any.
        timeframe: Mapping timeframe literal to load.
    Returns:
        _TimeframeMappingArraysV2 | None: Existing mapping arrays when present in the slot.
    Assumptions:
        Mapping tail update may reuse only already-materialized inactive-slot prefixes.
    Raises:
        FileNotFoundError: If manifest metadata references a missing mapping file.
        ValueError: If existing mapping metadata or files violate strict `uint32/time` contracts.
    Side Effects:
        Reads existing mapping `.npy` files from the inactive slot when present.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if existing_manifest is None:
        return None
    existing_section = _select_mapping_manifest_v2(
        mapping_sections=existing_manifest.mappings,
        timeframe=timeframe,
    )
    if existing_section is None:
        return None
    mapping_paths = artifact_loader.resolve_mapping_paths(coordinates, slot, timeframe)
    arrays = _TimeframeMappingArraysV2(
        bar_open_1m_idx=np.ascontiguousarray(
            _load_validated_array_v2(
                metadata=existing_section.bar_open_1m_idx,
                expected_path=mapping_paths.bar_open_1m_idx,
                expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
                expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
                expected_shape=None,
                location=f"existing mappings[{timeframe}].bar_open_1m_idx",
            ),
            dtype=np.uint32,
        ),
        bar_close_1m_idx=np.ascontiguousarray(
            _load_validated_array_v2(
                metadata=existing_section.bar_close_1m_idx,
                expected_path=mapping_paths.bar_close_1m_idx,
                expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
                expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
                expected_shape=None,
                location=f"existing mappings[{timeframe}].bar_close_1m_idx",
            ),
            dtype=np.uint32,
        ),
    )
    _validate_mapping_index_arrays_v2(
        arrays=arrays,
        timeframe=timeframe,
        target_bar_count=None,
        one_minute_bar_count=None,
        label=f"existing mappings[{timeframe}]",
    )
    return arrays


def _materialize_mapping_timeframes_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    one_minute_arrays: _CanonicalPriceArraysV2,
    price_manifests: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_tail_bars_1m: int,
) -> tuple[ArtifactMappingTimeframeManifestV2, ...]:
    """
    Materialize deterministic `tf -> 1m` mapping arrays for all allowed request timeframes.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving the mapping arrays.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest when present.
        one_minute_arrays: Materialized artifact-backed canonical `prices/1m` arrays.
        price_manifests: Fresh strict `prices/<tf>` manifest sections for allowed request TFs.
        mapping_tail_bars_1m: Effective `lookback_policy.mapping_tail_bars_1m`.
    Returns:
        tuple[ArtifactMappingTimeframeManifestV2, ...]: Canonically ordered mapping sections.
    Assumptions:
        Mapping build uses only materialized artifact-backed `prices/1m` plus `prices/<tf>`
        arrays and never rereads external sources.
    Raises:
        ValueError: If price arrays or reused mapping prefixes violate strict mapping contracts.
        OSError: If one `.npy` write or hash read fails.
    Side Effects:
        Atomically writes `mappings/<tf>/*.npy` files for every allowed request timeframe.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    price_by_timeframe = {section.timeframe: section for section in price_manifests}
    mapping_sections: list[ArtifactMappingTimeframeManifestV2] = []
    for timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2:
        price_manifest = price_by_timeframe.get(timeframe)
        if price_manifest is None:
            raise ValueError(f"materialized prices[{timeframe}] manifest section is required")
        timeframe_arrays = _load_materialized_price_arrays_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            timeframe=timeframe,
            manifest_section=price_manifest,
            location_prefix=f"materialized prices[{timeframe}] mapping target",
        )
        existing_arrays = _load_existing_mapping_arrays_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            existing_manifest=existing_manifest,
            timeframe=timeframe,
        )
        mapping_arrays = _build_mapping_arrays_with_tail_update_v2(
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            existing_arrays=existing_arrays,
            timeframe=timeframe,
            mapping_tail_bars_1m=mapping_tail_bars_1m,
        )
        mapping_paths = artifact_loader.resolve_mapping_paths(coordinates, slot, timeframe)
        _write_mapping_arrays_atomically_v2(mapping_paths=mapping_paths, arrays=mapping_arrays)
        mapping_sections.append(
            _build_mapping_manifest_v2(
                slot_root=slot_root,
                timeframe=timeframe,
                mapping_paths=mapping_paths,
                arrays=mapping_arrays,
            )
        )
    return tuple(mapping_sections)


def _build_mapping_arrays_with_tail_update_v2(
    *,
    one_minute_arrays: _CanonicalPriceArraysV2,
    timeframe_arrays: _CanonicalPriceArraysV2,
    existing_arrays: _TimeframeMappingArraysV2 | None,
    timeframe: str,
    mapping_tail_bars_1m: int,
) -> _TimeframeMappingArraysV2:
    """
    Build one `tf -> 1m` mapping family using bounded tail rebuild plus deterministic prefix reuse.

    Args:
        one_minute_arrays: Materialized artifact-backed `prices/1m` arrays.
        timeframe_arrays: Materialized artifact-backed `prices/<tf>` arrays.
        existing_arrays: Existing inactive-slot mapping arrays for the timeframe, when present.
        timeframe: Target request timeframe literal.
        mapping_tail_bars_1m: Effective `lookback_policy.mapping_tail_bars_1m`.
    Returns:
        _TimeframeMappingArraysV2: Final strict mapping arrays for the timeframe.
    Assumptions:
        A request-TF bar is unaffected when its `close_time` stays strictly before the first
        `1m` bar open included in the rebuilt tail window.
    Raises:
        ValueError: If prefix reuse, rebuilt tail, or final correspondence violates strict
            mapping contracts.
    Side Effects:
        Allocates contiguous numpy arrays in memory.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if existing_arrays is None:
        return _build_mapping_arrays_from_price_timelines_v2(
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            timeframe=timeframe,
        )

    one_minute_bar_count = int(one_minute_arrays.open_time.shape[0])
    if one_minute_bar_count <= mapping_tail_bars_1m:
        return _build_mapping_arrays_from_price_timelines_v2(
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            timeframe=timeframe,
        )

    affected_one_minute_start_idx = one_minute_bar_count - mapping_tail_bars_1m
    affected_one_minute_open_ms = int(one_minute_arrays.open_time[affected_one_minute_start_idx])
    prefix_end_idx = int(
        np.searchsorted(
            timeframe_arrays.close_time,
            np.int64(affected_one_minute_open_ms),
            side="right",
        )
    )
    prefix_bar_count = min(prefix_end_idx, int(existing_arrays.bar_open_1m_idx.shape[0]))
    prefix = (
        None
        if prefix_bar_count <= 0
        else _slice_mapping_arrays_v2(
            arrays=existing_arrays,
            start_idx=0,
            end_idx=prefix_bar_count,
        )
    )
    if prefix_bar_count >= int(timeframe_arrays.open_time.shape[0]):
        if prefix is None:
            raise ValueError(f"mappings[{timeframe}] prefix reuse produced no rows")
        _validate_mapping_arrays_v2(
            arrays=prefix,
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            timeframe=timeframe,
            label=f"mappings[{timeframe}]",
        )
        return prefix

    tail_price_arrays = _slice_canonical_price_arrays_v2(
        arrays=timeframe_arrays,
        start_idx=prefix_bar_count,
        end_idx=int(timeframe_arrays.open_time.shape[0]),
    )
    tail = _build_mapping_arrays_from_price_timelines_v2(
        one_minute_arrays=one_minute_arrays,
        timeframe_arrays=tail_price_arrays,
        timeframe=timeframe,
    )
    merged = _merge_mapping_arrays_v2(prefix=prefix, tail=tail)
    _validate_mapping_arrays_v2(
        arrays=merged,
        one_minute_arrays=one_minute_arrays,
        timeframe_arrays=timeframe_arrays,
        timeframe=timeframe,
        label=f"mappings[{timeframe}]",
    )
    return merged


def _build_mapping_arrays_from_price_timelines_v2(
    *,
    one_minute_arrays: _CanonicalPriceArraysV2,
    timeframe_arrays: _CanonicalPriceArraysV2,
    timeframe: str,
) -> _TimeframeMappingArraysV2:
    """
    Build strict `tf -> 1m` mappings directly from materialized `prices/1m` and `prices/<tf>`.

    Args:
        one_minute_arrays: Materialized artifact-backed canonical `prices/1m` arrays.
        timeframe_arrays: Materialized artifact-backed target `prices/<tf>` arrays.
        timeframe: Target request timeframe literal.
    Returns:
        _TimeframeMappingArraysV2: Strict `uint32` mapping arrays with shape `[T_tf]`.
    Assumptions:
        `prices/1m.open_time` and `prices/1m.close_time` are strict monotone arrays and
        `prices/<tf>` was already materialized from the same artifact-backed timeline.
    Raises:
        ValueError: If bounds, monotonicity, or timeline correspondence contracts fail.
    Side Effects:
        Allocates contiguous numpy arrays in memory.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    arrays = _TimeframeMappingArraysV2(
        bar_open_1m_idx=np.ascontiguousarray(
            np.searchsorted(
                one_minute_arrays.open_time,
                timeframe_arrays.open_time,
                side="left",
            ),
            dtype=np.uint32,
        ),
        bar_close_1m_idx=np.ascontiguousarray(
            np.searchsorted(
                one_minute_arrays.close_time,
                timeframe_arrays.close_time,
                side="left",
            ),
            dtype=np.uint32,
        ),
    )
    _validate_mapping_arrays_v2(
        arrays=arrays,
        one_minute_arrays=one_minute_arrays,
        timeframe_arrays=timeframe_arrays,
        timeframe=timeframe,
        label=f"mappings[{timeframe}]",
    )
    return arrays


def _slice_mapping_arrays_v2(
    *,
    arrays: _TimeframeMappingArraysV2,
    start_idx: int,
    end_idx: int,
) -> _TimeframeMappingArraysV2:
    """
    Slice one mapping family into a contiguous `[start_idx:end_idx]` view copy.

    Args:
        arrays: Source mapping arrays.
        start_idx: Inclusive slice start.
        end_idx: Exclusive slice end.
    Returns:
        _TimeframeMappingArraysV2: Contiguous mapping sub-slice.
    Assumptions:
        Callers already computed valid slice bounds against the source arrays.
    Raises:
        IndexError: If slice bounds are outside the source arrays.
    Side Effects:
        Allocates contiguous numpy arrays.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return _TimeframeMappingArraysV2(
        bar_open_1m_idx=np.ascontiguousarray(
            arrays.bar_open_1m_idx[start_idx:end_idx],
            dtype=np.uint32,
        ),
        bar_close_1m_idx=np.ascontiguousarray(
            arrays.bar_close_1m_idx[start_idx:end_idx],
            dtype=np.uint32,
        ),
    )


def _merge_mapping_arrays_v2(
    *,
    prefix: _TimeframeMappingArraysV2 | None,
    tail: _TimeframeMappingArraysV2,
) -> _TimeframeMappingArraysV2:
    """
    Merge reused mapping prefix rows with a freshly rebuilt tail.

    Args:
        prefix: Existing mapping rows strictly before the rebuilt tail window.
        tail: Freshly rebuilt mapping rows from the tail window onward.
    Returns:
        _TimeframeMappingArraysV2: Contiguous merged mapping arrays.
    Assumptions:
        Prefix rows always end strictly before the rebuilt target-row window when present.
    Raises:
        None.
    Side Effects:
        Allocates contiguous numpy arrays.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if prefix is None or int(prefix.bar_open_1m_idx.shape[0]) == 0:
        return tail
    return _TimeframeMappingArraysV2(
        bar_open_1m_idx=np.ascontiguousarray(
            np.concatenate((prefix.bar_open_1m_idx, tail.bar_open_1m_idx)),
            dtype=np.uint32,
        ),
        bar_close_1m_idx=np.ascontiguousarray(
            np.concatenate((prefix.bar_close_1m_idx, tail.bar_close_1m_idx)),
            dtype=np.uint32,
        ),
    )


def _write_mapping_arrays_atomically_v2(
    *,
    mapping_paths: ArtifactMappingPathsV2,
    arrays: _TimeframeMappingArraysV2,
) -> None:
    """
    Atomically replace inactive-slot `mappings/<tf>/*.npy` files with deterministic bytes.

    Args:
        mapping_paths: Explicit inactive-slot target paths for `bar_open_1m_idx` and
            `bar_close_1m_idx`.
        arrays: Strict mapping arrays to serialize.
    Returns:
        None.
    Assumptions:
        Temp files are written in the same directory so `os.replace` remains atomic.
    Raises:
        OSError: If temp-file write or atomic replace fails.
    Side Effects:
        Creates parent directories and replaces two `.npy` files under the inactive slot.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    _write_npy_atomically_v2(path=mapping_paths.bar_open_1m_idx, array=arrays.bar_open_1m_idx)
    _write_npy_atomically_v2(path=mapping_paths.bar_close_1m_idx, array=arrays.bar_close_1m_idx)


def _build_mapping_manifest_v2(
    *,
    slot_root: Path,
    timeframe: str,
    mapping_paths: ArtifactMappingPathsV2,
    arrays: _TimeframeMappingArraysV2,
) -> ArtifactMappingTimeframeManifestV2:
    """
    Build strict root-manifest metadata for one freshly written `mappings/<tf>` family.

    Args:
        slot_root: Absolute inactive-slot root directory.
        timeframe: Mapping timeframe literal addressed by `mapping_paths`.
        mapping_paths: Explicit inactive-slot mapping file paths.
        arrays: Freshly written strict mapping arrays for the timeframe.
    Returns:
        ArtifactMappingTimeframeManifestV2: Strict `mappings/<tf>` manifest section.
    Assumptions:
        Files were already atomically written and are ready for `sha256` calculation.
    Raises:
        OSError: If one written file cannot be hashed.
    Side Effects:
        Reads written `.npy` files to compute `sha256`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactMappingTimeframeManifestV2(
        timeframe=timeframe,
        bar_open_1m_idx=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=mapping_paths.bar_open_1m_idx,
            ),
            dtype=arrays.bar_open_1m_idx.dtype.name,
            shape=tuple(int(value) for value in arrays.bar_open_1m_idx.shape),
            axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(mapping_paths.bar_open_1m_idx),
        ),
        bar_close_1m_idx=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=mapping_paths.bar_close_1m_idx,
            ),
            dtype=arrays.bar_close_1m_idx.dtype.name,
            shape=tuple(int(value) for value in arrays.bar_close_1m_idx.shape),
            axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(mapping_paths.bar_close_1m_idx),
        ),
    )


def _validate_mapping_index_arrays_v2(
    *,
    arrays: _TimeframeMappingArraysV2,
    timeframe: str,
    target_bar_count: int | None,
    one_minute_bar_count: int | None,
    label: str,
) -> None:
    """
    Validate intrinsic mapping index invariants independent from price correspondence checks.

    Args:
        arrays: Candidate mapping arrays.
        timeframe: Target request timeframe literal.
        target_bar_count: Optional expected number of request-timeframe rows.
        one_minute_bar_count: Optional `T_1m` upper bound for index validation.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Every mapping family stores `uint32` non-decreasing indexes with shape `[T_tf]`.
    Raises:
        ValueError: If dtype, shape, monotonicity, ordering, or bounds are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    del timeframe
    if arrays.bar_open_1m_idx.dtype.name != ARTIFACT_MAPPING_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label}.bar_open_1m_idx dtype must be {ARTIFACT_MAPPING_DTYPE_LITERAL_V2!r}; got "
            f"{arrays.bar_open_1m_idx.dtype.name!r}"
        )
    if arrays.bar_close_1m_idx.dtype.name != ARTIFACT_MAPPING_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{label}.bar_close_1m_idx dtype must be {ARTIFACT_MAPPING_DTYPE_LITERAL_V2!r}; got "
            f"{arrays.bar_close_1m_idx.dtype.name!r}"
        )
    if arrays.bar_open_1m_idx.shape != arrays.bar_close_1m_idx.shape:
        raise ValueError(
            f"{label} open/close mapping shapes must match; got "
            f"{arrays.bar_open_1m_idx.shape!r} and {arrays.bar_close_1m_idx.shape!r}"
        )
    if target_bar_count is not None and arrays.bar_open_1m_idx.shape != (target_bar_count,):
        raise ValueError(
            f"{label} must have shape ({target_bar_count},); got "
            f"{arrays.bar_open_1m_idx.shape!r}"
        )
    if (
        arrays.bar_open_1m_idx.shape[0] > 1
        and not np.all(arrays.bar_open_1m_idx[1:] >= arrays.bar_open_1m_idx[:-1])
    ):
        raise ValueError(f"{label}.bar_open_1m_idx must be non-decreasing")
    if (
        arrays.bar_close_1m_idx.shape[0] > 1
        and not np.all(arrays.bar_close_1m_idx[1:] >= arrays.bar_close_1m_idx[:-1])
    ):
        raise ValueError(f"{label}.bar_close_1m_idx must be non-decreasing")
    if not np.all(arrays.bar_open_1m_idx <= arrays.bar_close_1m_idx):
        raise ValueError(f"{label} must satisfy bar_open_1m_idx <= bar_close_1m_idx")
    if one_minute_bar_count is not None:
        if not np.all(arrays.bar_open_1m_idx < one_minute_bar_count):
            raise ValueError(
                f"{label}.bar_open_1m_idx must stay within [0, {one_minute_bar_count})"
            )
        if not np.all(arrays.bar_close_1m_idx < one_minute_bar_count):
            raise ValueError(
                f"{label}.bar_close_1m_idx must stay within [0, {one_minute_bar_count})"
            )


def _validate_mapping_arrays_v2(
    *,
    arrays: _TimeframeMappingArraysV2,
    one_minute_arrays: _CanonicalPriceArraysV2,
    timeframe_arrays: _CanonicalPriceArraysV2,
    timeframe: str,
    label: str,
) -> None:
    """
    Validate full mapping contracts including bounds, monotonicity, and price correspondence.

    Args:
        arrays: Candidate mapping arrays.
        one_minute_arrays: Materialized artifact-backed canonical `prices/1m` arrays.
        timeframe_arrays: Materialized artifact-backed target `prices/<tf>` arrays.
        timeframe: Target request timeframe literal.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Strict runtime contract requires `prices/1m.open_time[bar_open_1m_idx]` and
        `prices/1m.close_time[bar_close_1m_idx]` to match `prices/<tf>` exactly.
    Raises:
        ValueError: If intrinsic invariants or exact timeline correspondence fail.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    _validate_mapping_index_arrays_v2(
        arrays=arrays,
        timeframe=timeframe,
        target_bar_count=int(timeframe_arrays.open_time.shape[0]),
        one_minute_bar_count=int(one_minute_arrays.open_time.shape[0]),
        label=label,
    )
    open_indexes = np.asarray(arrays.bar_open_1m_idx, dtype=np.intp)
    close_indexes = np.asarray(arrays.bar_close_1m_idx, dtype=np.intp)
    if not np.array_equal(one_minute_arrays.open_time[open_indexes], timeframe_arrays.open_time):
        raise ValueError(
            f"{label} open-time correspondence must satisfy "
            f"prices/1m.open_time[bar_open_1m_idx] == prices[{timeframe}].open_time"
        )
    if not np.array_equal(one_minute_arrays.close_time[close_indexes], timeframe_arrays.close_time):
        raise ValueError(
            f"{label} close-time correspondence must satisfy "
            f"prices/1m.close_time[bar_close_1m_idx] == prices[{timeframe}].close_time"
        )


def _build_rolled_price_arrays_with_tail_update_v2(
    *,
    source_arrays: _CanonicalPriceArraysV2,
    existing_arrays: _CanonicalPriceArraysV2 | None,
    timeframe: str,
    source_tail_time_range: TimeRange,
) -> _CanonicalPriceArraysV2:
    """
    Build one rolled timeframe using bounded tail recompute plus deterministic prefix reuse.

    Args:
        source_arrays: Final canonical `prices/1m` arrays loaded from the artifact slot.
        existing_arrays: Existing inactive-slot rolled arrays for the timeframe, when present.
        timeframe: Target rolled timeframe literal.
        source_tail_time_range: Effective `1m` reread window that may affect derived buckets.
    Returns:
        _CanonicalPriceArraysV2: Final rolled arrays for the target timeframe.
    Assumptions:
        Every derived bar is a pure function of canonical `1m` buckets aligned by
        `Timeframe.bucket_open/bucket_close`.
    Raises:
        ValueError: If tail slicing, prefix reuse, or rolled arrays violate strict contracts.
    Side Effects:
        Allocates contiguous numpy arrays in memory.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    """
    if existing_arrays is None:
        rolled_arrays = _rollup_price_arrays_from_one_minute_v2(
            source_arrays=source_arrays,
            timeframe=timeframe,
            allow_empty=False,
        )
        if rolled_arrays is None:
            raise ValueError(
                f"rolled prices[{timeframe}] produced no full buckets from prices[1m] source"
            )
        return rolled_arrays

    target_timeframe = Timeframe(timeframe)
    affected_bucket_open_ms = _bucket_open_epoch_millis_v2(
        timeframe=target_timeframe,
        value=_utc_timestamp_to_epoch_millis_v2(source_tail_time_range.start),
    )
    prefix_end_idx = int(
        np.searchsorted(existing_arrays.open_time, np.int64(affected_bucket_open_ms), side="left")
    )
    prefix = (
        None
        if prefix_end_idx <= 0
        else _slice_canonical_price_arrays_v2(
            arrays=existing_arrays,
            start_idx=0,
            end_idx=prefix_end_idx,
        )
    )
    source_start_idx = int(
        np.searchsorted(source_arrays.open_time, np.int64(affected_bucket_open_ms), side="left")
    )
    tail_source_arrays = _slice_canonical_price_arrays_v2(
        arrays=source_arrays,
        start_idx=source_start_idx,
        end_idx=int(source_arrays.open_time.shape[0]),
    )
    tail = _rollup_price_arrays_from_one_minute_v2(
        source_arrays=tail_source_arrays,
        timeframe=timeframe,
        allow_empty=True,
    )
    if tail is None:
        if prefix is None:
            raise ValueError(
                f"rolled prices[{timeframe}] produced no full buckets from prices[1m] source"
            )
        return prefix
    return _merge_rolled_price_arrays_v2(prefix=prefix, tail=tail, timeframe=timeframe)


def _rollup_price_arrays_from_one_minute_v2(
    *,
    source_arrays: _CanonicalPriceArraysV2,
    timeframe: str,
    allow_empty: bool,
) -> _CanonicalPriceArraysV2 | None:
    """
    Roll canonical `1m` arrays into strict `prices/<tf>` buckets with full-bucket semantics.

    Args:
        source_arrays: Canonical `prices/1m` arrays read from the artifact slot.
        timeframe: Target rolled timeframe literal.
        allow_empty: Whether returning `None` is allowed when no full buckets are present.
    Returns:
        _CanonicalPriceArraysV2 | None: Rolled arrays, or `None` when `allow_empty=True` and the
            source slice contains no full buckets.
    Assumptions:
        R3-02 stores only fully covered epoch-aligned buckets; partial leading or trailing buckets
        are skipped deterministically instead of being best-effort backfilled.
    Raises:
        ValueError: If the `1m` rollup source violates strict alignment or the result is empty
            while `allow_empty=False`.
    Side Effects:
        Allocates contiguous numpy arrays in memory.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    """
    if int(source_arrays.open_time.shape[0]) == 0:
        if allow_empty:
            return None
        raise ValueError(f"rolled prices[{timeframe}] source arrays must contain at least one bar")
    _validate_rollup_source_one_minute_arrays_v2(
        arrays=source_arrays,
        label=f"rollup source prices[1m] for {timeframe}",
    )
    target_timeframe = Timeframe(timeframe)
    bucket_bar_count = _timeframe_duration_millis_v2(target_timeframe) // _ONE_MINUTE_MILLIS_V2

    open_values: list[int] = []
    close_values: list[int] = []
    ohlcv_values: list[tuple[float, float, float, float, float]] = []
    bucket_open_ms: int | None = None
    bucket_start_idx = 0

    for index in range(int(source_arrays.open_time.shape[0])):
        row_open_ms = int(source_arrays.open_time[index])
        row_bucket_open_ms = _bucket_open_epoch_millis_v2(
            timeframe=target_timeframe,
            value=row_open_ms,
        )
        if bucket_open_ms is None:
            bucket_open_ms = row_bucket_open_ms
            bucket_start_idx = index
            continue
        if row_bucket_open_ms == bucket_open_ms:
            continue
        _append_complete_rollup_bucket_v2(
            source_arrays=source_arrays,
            timeframe=target_timeframe,
            bucket_open_ms=bucket_open_ms,
            bucket_start_idx=bucket_start_idx,
            bucket_end_idx=index,
            bucket_bar_count=bucket_bar_count,
            open_values=open_values,
            close_values=close_values,
            ohlcv_values=ohlcv_values,
        )
        bucket_open_ms = row_bucket_open_ms
        bucket_start_idx = index

    if bucket_open_ms is not None:
        _append_complete_rollup_bucket_v2(
            source_arrays=source_arrays,
            timeframe=target_timeframe,
            bucket_open_ms=bucket_open_ms,
            bucket_start_idx=bucket_start_idx,
            bucket_end_idx=int(source_arrays.open_time.shape[0]),
            bucket_bar_count=bucket_bar_count,
            open_values=open_values,
            close_values=close_values,
            ohlcv_values=ohlcv_values,
        )

    if len(open_values) == 0:
        if allow_empty:
            return None
        raise ValueError(f"rolled prices[{timeframe}] produced no full buckets from prices[1m]")

    rolled_arrays = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(np.asarray(open_values, dtype=np.int64), dtype=np.int64),
        close_time=np.ascontiguousarray(np.asarray(close_values, dtype=np.int64), dtype=np.int64),
        ohlcv=np.ascontiguousarray(np.asarray(ohlcv_values, dtype=np.float32), dtype=np.float32),
    )
    _validate_rolled_price_arrays_v2(arrays=rolled_arrays, timeframe=timeframe)
    return rolled_arrays


def _append_complete_rollup_bucket_v2(
    *,
    source_arrays: _CanonicalPriceArraysV2,
    timeframe: Timeframe,
    bucket_open_ms: int,
    bucket_start_idx: int,
    bucket_end_idx: int,
    bucket_bar_count: int,
    open_values: list[int],
    close_values: list[int],
    ohlcv_values: list[tuple[float, float, float, float, float]],
) -> None:
    """
    Append one fully covered epoch-aligned bucket into the rolled output buffers.

    Args:
        source_arrays: Canonical `1m` source arrays.
        timeframe: Target timeframe primitive used for bucket boundaries.
        bucket_open_ms: Epoch-millisecond open boundary of the candidate bucket.
        bucket_start_idx: Inclusive source row index for the bucket slice.
        bucket_end_idx: Exclusive source row index for the bucket slice.
        bucket_bar_count: Required number of `1m` bars for a full bucket.
        open_values: Mutable output buffer for rolled `open_time`.
        close_values: Mutable output buffer for rolled `close_time`.
        ohlcv_values: Mutable output buffer for rolled `ohlcv`.
    Returns:
        None.
    Assumptions:
        Partial buckets are skipped deterministically; only exact complete coverage is stored.
    Raises:
        None.
    Side Effects:
        Appends to the mutable output buffers when the bucket is complete.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    bucket_open_slice = source_arrays.open_time[bucket_start_idx:bucket_end_idx]
    if int(bucket_open_slice.shape[0]) != bucket_bar_count:
        return
    expected_open = (
        np.arange(bucket_bar_count, dtype=np.int64) * np.int64(_ONE_MINUTE_MILLIS_V2)
    ) + np.int64(bucket_open_ms)
    if not np.array_equal(bucket_open_slice, expected_open):
        return
    bucket_ohlcv = source_arrays.ohlcv[bucket_start_idx:bucket_end_idx]
    open_values.append(bucket_open_ms)
    close_values.append(bucket_open_ms + _timeframe_duration_millis_v2(timeframe))
    ohlcv_values.append(
        (
            float(bucket_ohlcv[0, 0]),
            float(np.max(bucket_ohlcv[:, 1])),
            float(np.min(bucket_ohlcv[:, 2])),
            float(bucket_ohlcv[-1, 3]),
            float(np.sum(bucket_ohlcv[:, 4], dtype=np.float64)),
        )
    )


def _merge_rolled_price_arrays_v2(
    *,
    prefix: _CanonicalPriceArraysV2 | None,
    tail: _CanonicalPriceArraysV2,
    timeframe: str,
) -> _CanonicalPriceArraysV2:
    """
    Merge reused rolled prefix bars with a freshly rebuilt tail for one timeframe.

    Args:
        prefix: Existing rolled bars strictly before the affected bucket boundary.
        tail: Freshly rebuilt rolled bars from the affected boundary onward.
        timeframe: Target rolled timeframe literal used in validation labels.
    Returns:
        _CanonicalPriceArraysV2: Contiguous merged rolled arrays.
    Assumptions:
        Prefix bars always end before the first rebuilt tail bucket when prefix is present.
    Raises:
        ValueError: If the merged arrays violate strict rolled-price invariants.
    Side Effects:
        Allocates merged contiguous arrays.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if prefix is None or int(prefix.open_time.shape[0]) == 0:
        return tail
    merged = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(
            np.concatenate((prefix.open_time, tail.open_time)),
            dtype=np.int64,
        ),
        close_time=np.ascontiguousarray(
            np.concatenate((prefix.close_time, tail.close_time)),
            dtype=np.int64,
        ),
        ohlcv=np.ascontiguousarray(np.concatenate((prefix.ohlcv, tail.ohlcv)), dtype=np.float32),
    )
    _validate_rolled_price_arrays_v2(arrays=merged, timeframe=timeframe)
    return merged


def _build_root_manifest_scaffold_v2(
    *,
    existing_manifest: ArtifactManifestDocumentV2 | None,
) -> _RootManifestScaffoldV2:
    """
    Build the non-owned root-manifest scaffold for R3-03 stage boundaries.

    Args:
        existing_manifest: Existing inactive-slot root manifest when one is already present.
    Returns:
        _RootManifestScaffoldV2: Preserved sections or explicit deterministic placeholders.
    Assumptions:
        R3-03 owns all supported `prices/<tf>` and `mappings/<tf>` sections, while
        `signals/hit_times` may still be placeholders or preserved later-stage sections.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    """
    if existing_manifest is None:
        return _RootManifestScaffoldV2(
            preserved_prices=(),
            mappings=(),
            signals=_empty_signal_catalog_v2(),
            hit_times=_placeholder_hit_times_reference_v2(),
            signal_encoding=_default_signal_encoding_contract_v2(),
        )
    return _RootManifestScaffoldV2(
        preserved_prices=tuple(
            section
            for section in existing_manifest.prices
            if section.timeframe not in ARTIFACT_PRICE_TIMEFRAMES_V2
        ),
        mappings=tuple(
            section
            for section in existing_manifest.mappings
            if section.timeframe not in ARTIFACT_MAPPING_TIMEFRAMES_V2
        ),
        signals=existing_manifest.signals,
        hit_times=existing_manifest.hit_times,
        signal_encoding=existing_manifest.signal_encoding,
    )


def _empty_signal_catalog_v2() -> ArtifactSignalCatalogV2:
    """
    Build the explicit empty signal catalog placeholder used before R4 materialization.

    Args:
        None.
    Returns:
        ArtifactSignalCatalogV2: Empty deterministic signal catalog placeholder.
    Assumptions:
        Root manifest keeps `signal_encoding` fixed even before any signal manifests exist.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactSignalCatalogV2(
        supported_timeframes=(),
        supported_indicator_ids=(),
        manifests=(),
    )


def _placeholder_hit_times_reference_v2() -> ArtifactHitTimesReferenceV2:
    """
    Build the explicit fixed-path hit-times placeholder used before R5 materialization.

    Args:
        None.
    Returns:
        ArtifactHitTimesReferenceV2: Deterministic fixed-path placeholder reference.
    Assumptions:
        R3-01 keeps root-manifest schema strict without pretending that `hit_times/1m` already
        exists; later epics must replace this placeholder with a real manifest hash.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    """
    return ArtifactHitTimesReferenceV2(
        timeframe=HIT_TIMES_TIMEFRAME_LITERAL_V2,
        manifest_path=f"{HIT_TIMES_DIRECTORY_LITERAL_V2}/{HIT_TIMES_TIMEFRAME_LITERAL_V2}/{ARTIFACT_MANIFEST_FILENAME_V2}",
        manifest_sha256=ARTIFACT_PLACEHOLDER_SHA256_V2,
    )


def _default_signal_encoding_contract_v2() -> ArtifactSignalEncodingContractV2:
    """
    Build the fixed signal encoding contract reused even before any signal artifacts exist.

    Args:
        None.
    Returns:
        ArtifactSignalEncodingContractV2: Fixed signal runtime encoding contract.
    Assumptions:
        Signal storage rules are global and independent from R3-01 `prices/1m` ownership.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactSignalEncodingContractV2(
        dtype=ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
        axis_order=ARTIFACT_SIGNAL_AXIS_ORDER_V2,
        value_set=ARTIFACT_SIGNAL_VALUE_SET_V2,
    )


def _build_signal_manifest_provenance_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    grid_contract: ArtifactSignalGridContractV2,
    timeline: ArtifactTimelineCoverageV2,
    signal_rules_engine: BacktestSignalRulesEngineV2,
    signal_params_defaults: Mapping[str, Any],
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic provenance for one strict per-indicator signal manifest.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        request: Explicit export request identity.
        slot_generation: Target inactive-slot generation assigned to the build.
        runtime_settings: Strict runtime settings contributing the config hash.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        price_manifest: Fresh target timeframe price section used as source-of-truth timeline.
        grid_contract: Strict grid metadata carrying `variant_keys_sha256`.
        timeline: Timeline coverage serialized into the signal manifest.
        signal_rules_engine: Explicit rules engine used to resolve dependency metadata.
        signal_params_defaults: Resolved default-only signal parameter mapping.
    Returns:
        ArtifactManifestProvenanceV2: Strict signal-manifest provenance payload.
    Assumptions:
        Per-manifest provenance hashes source identity and row-order metadata, not YAML bytes.
    Raises:
        TypeError: If provenance hashing encounters an unsupported JSON payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactManifestProvenanceV2(
        generator=_PRECOMPUTE_GENERATOR_LITERAL_V2,
        generator_version=_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2,
        generated_at_utc=request.generated_at_utc,
        config_sha256=runtime_settings.config_sha256,
        inputs_sha256=_build_signal_manifest_inputs_sha256_v2(
            coordinates=coordinates,
            request=request,
            slot_generation=slot_generation,
            signal_target=signal_target,
            price_manifest=price_manifest,
            grid_contract=grid_contract,
            timeline=timeline,
            required_dependency_ids=signal_rules_engine.rule_spec(
                indicator_id=signal_target.indicator_id
            ).required_dependency_ids,
            signal_params_defaults=signal_params_defaults,
        ),
    )


def _build_signal_manifest_inputs_sha256_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    signal_target: ArtifactSignalValidationSpecV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    grid_contract: ArtifactSignalGridContractV2,
    timeline: ArtifactTimelineCoverageV2,
    required_dependency_ids: tuple[str, ...],
    signal_params_defaults: Mapping[str, Any],
) -> str:
    """
    Hash normalized signal-manifest source identity into deterministic provenance.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        request: Explicit export request identity.
        slot_generation: Target inactive-slot generation assigned to the build.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        price_manifest: Fresh source price section for the target timeframe.
        grid_contract: Strict grid metadata carrying the ordered variant-key hash.
        timeline: Timeline coverage serialized into the signal manifest.
        required_dependency_ids: Dependency indicator ids required by the rule family.
        signal_params_defaults: Resolved default-only signal parameter mapping.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash tracks source identity and row ordering rather than manifest serialization bytes.
    Raises:
        TypeError: If canonical JSON serialization receives an unsupported payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    canonical_payload = json.dumps(
        {
            "coordinates": {
                "exchange": coordinates.exchange,
                "market_type": coordinates.market_type,
                "symbol": coordinates.symbol,
            },
            "time_range": {
                "start": _utc_timestamp_to_epoch_millis_v2(request.time_range.start),
                "end": _utc_timestamp_to_epoch_millis_v2(request.time_range.end),
            },
            "slot_generation": slot_generation,
            "asof_date": request.asof_date,
            "timeframe": signal_target.timeframe,
            "indicator_id": signal_target.indicator_id,
            "required_dependency_ids": required_dependency_ids,
            "price_manifest_sha256": {
                "open_time": price_manifest.open_time.sha256,
                "close_time": price_manifest.close_time.sha256,
                "ohlcv": price_manifest.ohlcv.sha256,
            },
            "timeline": _serialize_timeline_coverage_v2(timeline),
            "grid": {
                "variant_key_version": grid_contract.variant_key_version,
                "variant_keys_sha256": grid_contract.variant_keys_sha256,
            },
            "signals_v1_params_defaults": signal_params_defaults,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _build_root_manifest_provenance_v2(
    *,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    arrays: _CanonicalPriceArraysV2,
    rolled_sections: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_sections: tuple[ArtifactMappingTimeframeManifestV2, ...],
    signal_entries: tuple[ArtifactSignalCatalogEntryV2, ...] = (),
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic root-manifest provenance for canonical prices, rolled prices, and
    `tf -> 1m` mappings.

    Args:
        runtime_settings: Strict service runtime settings used by the precompute runner.
        request: Explicit export request identity.
        arrays: Final merged canonical `1m` arrays written into the inactive slot.
        rolled_sections: Rolled price-manifest sections emitted during the same R3-03 build.
        mapping_sections: Mapping-manifest sections emitted during the same R3-03 build.
        signal_entries: Signal catalog entries emitted during the same build when R4-02 is
            enabled.
    Returns:
        ArtifactManifestProvenanceV2: Strict provenance payload for the root manifest.
    Assumptions:
        At R3-03 `inputs_sha256` identifies the normalized export request plus emitted artifact
        metadata derived from `market_data.canonical_candles_1m`.
    Raises:
        TypeError: If config hashing encounters an unsupported JSON payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    return ArtifactManifestProvenanceV2(
        generator=_PRECOMPUTE_GENERATOR_LITERAL_V2,
        generator_version=_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2,
        generated_at_utc=request.generated_at_utc,
        config_sha256=runtime_settings.config_sha256,
        inputs_sha256=_build_inputs_sha256_v2(
            request=request,
            arrays=arrays,
            rolled_sections=rolled_sections,
            mapping_sections=mapping_sections,
            signal_entries=signal_entries,
            price_lookback_bars=runtime_settings.price_tail_bars_1m,
            mapping_lookback_bars=runtime_settings.mapping_tail_bars_1m,
        ),
    )


def _build_inputs_sha256_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    arrays: _CanonicalPriceArraysV2,
    rolled_sections: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_sections: tuple[ArtifactMappingTimeframeManifestV2, ...],
    signal_entries: tuple[ArtifactSignalCatalogEntryV2, ...],
    price_lookback_bars: int,
    mapping_lookback_bars: int,
) -> str:
    """
    Hash normalized export identity and emitted arrays into deterministic provenance.

    Args:
        request: Explicit export request identity.
        arrays: Final merged canonical `1m` arrays emitted by the runner.
        rolled_sections: Rolled price-manifest sections emitted by the same build.
        mapping_sections: Mapping-manifest sections emitted by the same build.
        signal_entries: Signal catalog entries emitted by the same build.
        price_lookback_bars: Effective `lookback_policy.price_tail_bars_1m` used for the build.
        mapping_lookback_bars: Effective `lookback_policy.mapping_tail_bars_1m` used for the
            build.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash is an R3-03 input-identity digest, not a runtime validation checksum.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    digest = hashlib.sha256()
    rolled_timeframes = tuple(section.timeframe for section in rolled_sections)
    normalized_identity = json.dumps(
        {
            "source_table": _CANONICAL_CANDLE_SOURCE_LITERAL_V2,
            "coordinates": {
                "exchange": request.coordinates.exchange,
                "market_type": request.coordinates.market_type,
                "symbol": request.coordinates.symbol,
            },
            "time_range": {
                "start": _utc_timestamp_to_epoch_millis_v2(request.time_range.start),
                "end": _utc_timestamp_to_epoch_millis_v2(request.time_range.end),
            },
            "asof_date": request.asof_date,
            "lookback_policy.price_tail_bars_1m": price_lookback_bars,
            "lookback_policy.mapping_tail_bars_1m": mapping_lookback_bars,
            "rolled_price_timeframes": rolled_timeframes,
            "mapping_timeframes": tuple(section.timeframe for section in mapping_sections),
            "signal_targets": tuple(
                (entry.timeframe, entry.indicator_id) for entry in signal_entries
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest.update(normalized_identity.encode("utf-8"))
    for array in (arrays.open_time, arrays.close_time, arrays.ohlcv):
        digest.update(array.dtype.name.encode("ascii"))
        digest.update(
            json.dumps(tuple(int(value) for value in array.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    for section in rolled_sections:
        digest.update(section.timeframe.encode("ascii"))
        for metadata in (section.open_time, section.close_time, section.ohlcv):
            digest.update(metadata.dtype.encode("ascii"))
            digest.update(
                json.dumps(
                    tuple(int(value) for value in metadata.shape),
                    separators=(",", ":"),
                ).encode("ascii")
            )
            digest.update(metadata.sha256.encode("ascii"))
    for section in mapping_sections:
        digest.update(section.timeframe.encode("ascii"))
        for metadata in (section.bar_open_1m_idx, section.bar_close_1m_idx):
            digest.update(metadata.dtype.encode("ascii"))
            digest.update(
                json.dumps(
                    tuple(int(value) for value in metadata.shape),
                    separators=(",", ":"),
                ).encode("ascii")
            )
            digest.update(metadata.sha256.encode("ascii"))
    for entry in signal_entries:
        digest.update(entry.timeframe.encode("ascii"))
        digest.update(entry.indicator_id.encode("ascii"))
        digest.update(entry.manifest_path.encode("utf-8"))
        digest.update(entry.manifest_sha256.encode("ascii"))
    return digest.hexdigest()


def _build_root_manifest_payload_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot: str,
    slot_generation: int,
    root_scaffold: _RootManifestScaffoldV2,
    price_manifests: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_manifests: tuple[ArtifactMappingTimeframeManifestV2, ...],
    provenance: ArtifactManifestProvenanceV2,
) -> dict[str, Any]:
    """
    Build the strict root `manifest.yaml` payload for R3-03 price and mapping materialization.

    Args:
        request: Explicit export request identity.
        slot: Inactive slot literal receiving the new root manifest.
        slot_generation: Target slot generation reserved for the next publish switch.
        root_scaffold: Preserved or placeholder non-price manifest sections.
        price_manifests: Fresh strict `prices/<tf>` sections owned by R3-03.
        mapping_manifests: Fresh strict `mappings/<tf>` sections owned by R3-03.
        provenance: Deterministic root-manifest provenance payload.
    Returns:
        dict[str, Any]: Deterministic YAML payload ready for atomic serialization.
    Assumptions:
        R3-03 updates all price and mapping sections while preserving strict schema for later
        stages.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    merged_prices = _merge_price_sections_v2(
        preserved_prices=root_scaffold.preserved_prices,
        price_manifests=price_manifests,
    )
    merged_mappings = _merge_mapping_sections_v2(
        preserved_mappings=root_scaffold.mappings,
        mapping_manifests=mapping_manifests,
    )
    return {
        "schema_version": 1,
        "manifest_kind": "slot_root",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": request.asof_date,
        "identity": {
            "exchange": request.coordinates.exchange,
            "market_type": request.coordinates.market_type,
            "symbol": request.coordinates.symbol,
        },
        "prices": [_serialize_price_manifest_v2(section) for section in merged_prices],
        "mappings": [_serialize_mapping_manifest_v2(section) for section in merged_mappings],
        "signals": _serialize_signal_catalog_v2(root_scaffold.signals),
        "hit_times": _serialize_hit_times_reference_v2(root_scaffold.hit_times),
        "signal_encoding": _serialize_signal_encoding_v2(root_scaffold.signal_encoding),
        "provenance": _serialize_provenance_v2(provenance),
    }


def _merge_price_sections_v2(
    *,
    preserved_prices: tuple[ArtifactPriceTimeframeManifestV2, ...],
    price_manifests: tuple[ArtifactPriceTimeframeManifestV2, ...],
) -> tuple[ArtifactPriceTimeframeManifestV2, ...]:
    """
    Merge preserved non-owned price sections with the freshly written R3-02 price sections.

    Args:
        preserved_prices: Existing root price sections outside the R3-02 ownership scope.
        price_manifests: Fresh strict price sections written during the current build.
    Returns:
        tuple[ArtifactPriceTimeframeManifestV2, ...]: Canonically ordered root price sections.
    Assumptions:
        Root manifest ordering must remain deterministic by the fixed artifact timeframe contract.
    Raises:
        ValueError: If duplicated timeframe sections are detected.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    merged_by_timeframe: dict[str, ArtifactPriceTimeframeManifestV2] = {}
    for section in price_manifests:
        if section.timeframe in merged_by_timeframe:
            raise ValueError(
                "root manifest price sections contain duplicate timeframe "
                f"{section.timeframe!r}"
            )
        merged_by_timeframe[section.timeframe] = section
    for section in preserved_prices:
        if section.timeframe in merged_by_timeframe:
            raise ValueError(
                "root manifest price sections contain duplicate timeframe "
                f"{section.timeframe!r}"
            )
        merged_by_timeframe[section.timeframe] = section
    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_PRICE_TIMEFRAMES_V2)
    }
    ordered_sections = sorted(
        merged_by_timeframe.values(),
        key=lambda section: timeframe_order[section.timeframe],
    )
    return tuple(ordered_sections)


def _merge_mapping_sections_v2(
    *,
    preserved_mappings: tuple[ArtifactMappingTimeframeManifestV2, ...],
    mapping_manifests: tuple[ArtifactMappingTimeframeManifestV2, ...],
) -> tuple[ArtifactMappingTimeframeManifestV2, ...]:
    """
    Merge preserved non-owned mapping sections with freshly written R3-03 mapping sections.

    Args:
        preserved_mappings: Existing root mapping sections outside the R3-03 ownership scope.
        mapping_manifests: Fresh strict mapping sections written during the current build.
    Returns:
        tuple[ArtifactMappingTimeframeManifestV2, ...]: Canonically ordered root mapping sections.
    Assumptions:
        Root manifest ordering must remain deterministic by the fixed mapping timeframe contract.
    Raises:
        ValueError: If duplicated timeframe sections are detected.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    merged_by_timeframe: dict[str, ArtifactMappingTimeframeManifestV2] = {}
    for section in mapping_manifests:
        if section.timeframe in merged_by_timeframe:
            raise ValueError(
                "root manifest mapping sections contain duplicate timeframe "
                f"{section.timeframe!r}"
            )
        merged_by_timeframe[section.timeframe] = section
    for section in preserved_mappings:
        if section.timeframe in merged_by_timeframe:
            raise ValueError(
                "root manifest mapping sections contain duplicate timeframe "
                f"{section.timeframe!r}"
            )
        merged_by_timeframe[section.timeframe] = section
    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_MAPPING_TIMEFRAMES_V2)
    }
    ordered_sections = sorted(
        merged_by_timeframe.values(),
        key=lambda section: timeframe_order[section.timeframe],
    )
    return tuple(ordered_sections)


def _serialize_price_manifest_v2(section: ArtifactPriceTimeframeManifestV2) -> dict[str, Any]:
    """
    Serialize one typed root price section into deterministic YAML-ready payload order.

    Args:
        section: Typed root-manifest price section.
    Returns:
        dict[str, Any]: YAML-ready price section payload.
    Assumptions:
        Typed section already satisfies strict root-manifest contracts.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "timeframe": section.timeframe,
        "open_time": _serialize_array_metadata_v2(section.open_time),
        "close_time": _serialize_array_metadata_v2(section.close_time),
        "ohlcv": _serialize_array_metadata_v2(section.ohlcv),
        "coverage": _serialize_timeline_coverage_v2(section.coverage),
    }


def _serialize_mapping_manifest_v2(
    section: ArtifactMappingTimeframeManifestV2,
) -> dict[str, Any]:
    """
    Serialize one typed root mapping section into deterministic YAML-ready payload order.

    Args:
        section: Typed root-manifest mapping section.
    Returns:
        dict[str, Any]: YAML-ready mapping section payload.
    Assumptions:
        Mapping sections are preserved verbatim by R3-01 when already present in the inactive slot.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "timeframe": section.timeframe,
        "bar_open_1m_idx": _serialize_array_metadata_v2(section.bar_open_1m_idx),
        "bar_close_1m_idx": _serialize_array_metadata_v2(section.bar_close_1m_idx),
    }


def _serialize_signal_catalog_v2(catalog: ArtifactSignalCatalogV2) -> dict[str, Any]:
    """
    Serialize the typed root signal catalog into deterministic YAML-ready payload order.

    Args:
        catalog: Typed root signal catalog or explicit R3-01 placeholder.
    Returns:
        dict[str, Any]: YAML-ready signal catalog payload.
    Assumptions:
        Empty lists remain explicit placeholders before R4 signal materialization.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "supported_timeframes": [item for item in catalog.supported_timeframes],
        "supported_indicator_ids": [item for item in catalog.supported_indicator_ids],
        "manifests": [
            _serialize_signal_catalog_entry_v2(entry) for entry in catalog.manifests
        ],
    }


def _serialize_signal_catalog_entry_v2(
    entry: ArtifactSignalCatalogEntryV2,
) -> dict[str, Any]:
    """
    Serialize one typed signal-catalog entry into deterministic YAML-ready payload order.

    Args:
        entry: Typed root signal-catalog entry.
    Returns:
        dict[str, Any]: YAML-ready signal-catalog entry payload.
    Assumptions:
        Entry paths remain slot-relative literals under the strict root-manifest contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "timeframe": entry.timeframe,
        "indicator_id": entry.indicator_id,
        "manifest_path": entry.manifest_path,
        "manifest_sha256": entry.manifest_sha256,
    }


def _serialize_signal_manifest_v2(
    manifest: ArtifactSignalManifestDocumentV2,
) -> dict[str, Any]:
    """
    Serialize one typed strict signal manifest into deterministic YAML-ready payload order.

    Args:
        manifest: Typed strict signal manifest.
    Returns:
        dict[str, Any]: YAML-ready signal manifest payload.
    Assumptions:
        Manifest fields are already validated and use canonical slot-relative path literals.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": manifest.manifest_kind,
        "slot": manifest.slot,
        "slot_generation": manifest.slot_generation,
        "asof_date": manifest.asof_date,
        "indicator_id": manifest.indicator_id,
        "timeframe": manifest.timeframe,
        "signals": _serialize_array_metadata_v2(manifest.signals),
        "rows_count": manifest.rows_count,
        "timeline": _serialize_timeline_coverage_v2(manifest.timeline),
        "signal_value_set": [int(value) for value in manifest.signal_value_set],
        "grid": _serialize_signal_grid_contract_v2(manifest.grid),
        "provenance": _serialize_provenance_v2(manifest.provenance),
    }


def _serialize_signal_grid_contract_v2(
    grid: ArtifactSignalGridContractV2,
) -> dict[str, Any]:
    """
    Serialize typed signal-grid metadata into deterministic YAML-ready payload order.

    Args:
        grid: Typed signal-grid metadata carried by a strict signal manifest.
    Returns:
        dict[str, Any]: YAML-ready signal-grid payload.
    Assumptions:
        `signals_v1_params_defaults` is already an immutable canonical mapping.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "variant_key_version": grid.variant_key_version,
        "variant_keys_sha256": grid.variant_keys_sha256,
        "signals_v1_params_defaults": dict(grid.signals_v1_params_defaults),
    }


def _serialize_hit_times_reference_v2(
    reference: ArtifactHitTimesReferenceV2,
) -> dict[str, Any]:
    """
    Serialize the typed root hit-times reference into deterministic YAML-ready payload order.

    Args:
        reference: Typed hit-times reference or explicit R3-01 placeholder.
    Returns:
        dict[str, Any]: YAML-ready hit-times reference payload.
    Assumptions:
        Placeholder reference keeps the strict schema visible until R5 owns the real files.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "timeframe": reference.timeframe,
        "manifest_path": reference.manifest_path,
        "manifest_sha256": reference.manifest_sha256,
    }


def _serialize_signal_encoding_v2(
    signal_encoding: ArtifactSignalEncodingContractV2,
) -> dict[str, Any]:
    """
    Serialize the typed root signal-encoding contract into YAML-ready payload order.

    Args:
        signal_encoding: Typed signal encoding contract.
    Returns:
        dict[str, Any]: YAML-ready signal encoding payload.
    Assumptions:
        Signal encoding stays fixed even when `signals.manifests` is empty at R3-01 stage.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "dtype": signal_encoding.dtype,
        "axis_order": [item for item in signal_encoding.axis_order],
        "value_set": [int(item) for item in signal_encoding.value_set],
    }


def _serialize_provenance_v2(
    provenance: ArtifactManifestProvenanceV2,
) -> dict[str, Any]:
    """
    Serialize typed strict provenance into deterministic YAML-ready payload order.

    Args:
        provenance: Typed strict provenance payload.
    Returns:
        dict[str, Any]: YAML-ready provenance payload.
    Assumptions:
        Root-manifest provenance is regenerated on every R3-01 export attempt.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "generator": provenance.generator,
        "generator_version": provenance.generator_version,
        "generated_at_utc": provenance.generated_at_utc,
        "config_sha256": provenance.config_sha256,
        "inputs_sha256": provenance.inputs_sha256,
    }


def _serialize_array_metadata_v2(
    metadata: ArtifactArrayMetadataV2,
) -> dict[str, Any]:
    """
    Serialize typed strict array metadata into deterministic YAML-ready payload order.

    Args:
        metadata: Typed strict array metadata.
    Returns:
        dict[str, Any]: YAML-ready array metadata payload.
    Assumptions:
        Array paths are already stored as canonical slot-relative literals.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "path": metadata.path,
        "dtype": metadata.dtype,
        "shape": [int(value) for value in metadata.shape],
        "axis_order": [axis for axis in metadata.axis_order],
        "sha256": metadata.sha256,
    }


def _serialize_timeline_coverage_v2(
    coverage: ArtifactTimelineCoverageV2,
) -> dict[str, Any]:
    """
    Serialize typed timeline coverage into deterministic YAML-ready payload order.

    Args:
        coverage: Typed root/signal timeline coverage payload.
    Returns:
        dict[str, Any]: YAML-ready timeline coverage payload.
    Assumptions:
        Coverage boundaries were already derived from strict `open_time/close_time` arrays.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return {
        "bar_count": coverage.bar_count,
        "open_time_start": coverage.open_time_start,
        "open_time_end": coverage.open_time_end,
        "close_time_start": coverage.close_time_start,
        "close_time_end": coverage.close_time_end,
    }


def _timeline_coverage_from_arrays_v2(
    *,
    arrays: _CanonicalPriceArraysV2,
) -> ArtifactTimelineCoverageV2:
    """
    Build strict timeline coverage metadata from canonical `open_time/close_time` arrays.

    Args:
        arrays: Strict canonical `1m` arrays.
    Returns:
        ArtifactTimelineCoverageV2: Strict timeline coverage payload.
    Assumptions:
        Arrays were already validated to be non-empty and strictly monotone.
    Raises:
        IndexError: If callers bypass validation and pass empty arrays.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactTimelineCoverageV2(
        bar_count=int(arrays.open_time.shape[0]),
        open_time_start=int(arrays.open_time[0]),
        open_time_end=int(arrays.open_time[-1]),
        close_time_start=int(arrays.close_time[0]),
        close_time_end=int(arrays.close_time[-1]),
    )


def _write_yaml_atomically_v2(*, path: Path, payload: Mapping[str, Any]) -> None:
    """
    Serialize one YAML payload through temp-file write plus atomic replace.

    Args:
        path: Canonical target YAML path under the inactive slot.
        payload: Deterministic YAML payload to serialize.
    Returns:
        None.
    Assumptions:
        Caller already prepared canonical field order with plain lists/dicts only.
    Raises:
        OSError: If temp-file write or atomic replace fails.
    Side Effects:
        Creates parent directories and replaces one YAML file on disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    serialized_payload = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=False,
    )
    if not serialized_payload.endswith("\n"):
        serialized_payload = serialized_payload + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _select_price_manifest_v2(
    *,
    price_sections: tuple[ArtifactPriceTimeframeManifestV2, ...],
    timeframe: str,
) -> ArtifactPriceTimeframeManifestV2 | None:
    """
    Select one price timeframe section from typed root-manifest price sections.

    Args:
        price_sections: Typed root-manifest price sections.
        timeframe: Target price timeframe literal.
    Returns:
        ArtifactPriceTimeframeManifestV2 | None: Matching section when present.
    Assumptions:
        Typed root manifests already enforce one section per timeframe.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    for section in price_sections:
        if section.timeframe == timeframe:
            return section
    return None


def _load_validated_array_v2(
    *,
    metadata: ArtifactArrayMetadataV2,
    expected_path: Path,
    expected_dtype: str,
    expected_axis_order: tuple[str, ...],
    expected_shape: tuple[int, ...] | None,
    location: str,
) -> np.ndarray:
    """
    Load one existing `.npy` file and fail fast on strict metadata drift.

    Args:
        metadata: Strict array metadata from the existing root manifest.
        expected_path: Explicit deterministic artifact path for the array.
        expected_dtype: Required dtype literal for the array family.
        expected_axis_order: Required axis-order literal for the array family.
        expected_shape: Optional required array shape when known ahead of time.
        location: Stable label used in deterministic error messages.
    Returns:
        np.ndarray: Loaded numpy array.
    Assumptions:
        Existing inactive-slot arrays must already satisfy the strict root-manifest contract if
        they are reused for bounded tail update.
    Raises:
        FileNotFoundError: If the expected array path is absent.
        ValueError: If metadata path/hash/dtype/shape/axis-order mismatches the actual file.
    Side Effects:
        Reads one `.npy` file from disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if metadata.path != expected_path.relative_to(expected_path.parents[2]).as_posix():
        expected_relative_path = expected_path.relative_to(expected_path.parents[2]).as_posix()
        raise ValueError(
            f"{location} manifest path must be {expected_relative_path!r}; "
            f"got {metadata.path!r}"
        )
    if metadata.dtype != expected_dtype:
        raise ValueError(
            f"{location} manifest dtype must be {expected_dtype!r}; got {metadata.dtype!r}"
        )
    if metadata.axis_order != expected_axis_order:
        raise ValueError(
            f"{location} manifest axis_order must be {expected_axis_order!r}; "
            f"got {metadata.axis_order!r}"
        )
    if not expected_path.is_file():
        raise FileNotFoundError(f"{location} artifact file is missing: {expected_path}")
    array = np.load(expected_path, allow_pickle=False)
    actual_shape = tuple(int(value) for value in array.shape)
    if metadata.shape != actual_shape:
        raise ValueError(
            f"{location} manifest shape must match actual file; got {metadata.shape!r}, "
            f"expected {actual_shape!r}"
        )
    if expected_shape is not None and actual_shape != expected_shape:
        raise ValueError(
            f"{location} file shape must be {expected_shape!r}; got {actual_shape!r}"
        )
    if array.dtype.name != expected_dtype:
        raise ValueError(
            f"{location} file dtype must be {expected_dtype!r}; got {array.dtype.name!r}"
        )
    actual_sha256 = _file_sha256_hex_v2(expected_path)
    if metadata.sha256 != actual_sha256:
        raise ValueError(
            f"{location} manifest sha256 must match actual file; got {metadata.sha256!r}, "
            f"expected {actual_sha256!r}"
        )
    return array


def _slot_relative_path_v2(*, slot_root: Path, absolute_path: Path) -> str:
    """
    Convert one absolute artifact path under a slot root into canonical POSIX-relative form.

    Args:
        slot_root: Absolute slot-root path.
        absolute_path: Absolute artifact path under that slot root.
    Returns:
        str: Canonical POSIX-style slot-relative path literal.
    Assumptions:
        All root-manifest artifact paths are serialized relative to the slot root.
    Raises:
        ValueError: If the absolute path is outside the slot root.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return absolute_path.relative_to(slot_root).as_posix()


def _file_sha256_hex_v2(path: Path) -> str:
    """
    Compute lowercase SHA-256 for one file using deterministic chunked I/O.

    Args:
        path: Existing filesystem path to hash.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        Hashes are publish-time metadata and must be stable for identical file bytes.
    Raises:
        OSError: If the file cannot be read.
    Side Effects:
        Reads the file from disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timeframe_duration_millis_v2(timeframe: Timeframe) -> int:
    """
    Convert one shared-kernel timeframe duration into integer epoch milliseconds.

    Args:
        timeframe: Shared-kernel timeframe primitive.
    Returns:
        int: Whole-millisecond duration of the timeframe.
    Assumptions:
        Supported artifact timeframes are fixed UTC durations expressible as whole milliseconds.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/shared-kernel-primitives.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return int(timeframe.duration() // timedelta(milliseconds=1))


def _bucket_open_epoch_millis_v2(*, timeframe: Timeframe, value: int) -> int:
    """
    Resolve one epoch-millisecond timestamp to its epoch-aligned bucket open boundary.

    Args:
        timeframe: Target shared-kernel timeframe primitive.
        value: Epoch milliseconds for the source timestamp.
    Returns:
        int: Epoch milliseconds for the bucket-open boundary.
    Assumptions:
        Bucket alignment must flow through `Timeframe.bucket_open(...)`, not ad-hoc day math.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/shared-kernel-primitives.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return _utc_timestamp_to_epoch_millis_v2(
        timeframe.bucket_open(_epoch_millis_to_utc_timestamp_v2(value))
    )


def _utc_timestamp_to_epoch_millis_v2(value: UtcTimestamp) -> int:
    """
    Convert strict UTC timestamp value object into epoch milliseconds without float rounding.

    Args:
        value: Shared-kernel UTC timestamp.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        `UtcTimestamp` already guarantees timezone-aware UTC with millisecond precision.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/shared-kernel-primitives.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
    """
    return int((value.value - _EPOCH_UTC) // timedelta(milliseconds=1))


def _epoch_millis_to_utc_timestamp_v2(value: int) -> UtcTimestamp:
    """
    Convert epoch milliseconds into strict shared-kernel UTC timestamp value object.

    Args:
        value: Epoch milliseconds.
    Returns:
        UtcTimestamp: Shared-kernel UTC timestamp.
    Assumptions:
        Millisecond timestamps already follow canonical `DateTime64(3, 'UTC')` precision.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/shared-kernel-primitives.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
    """
    return UtcTimestamp(_EPOCH_UTC + timedelta(milliseconds=value))


def _time_range_literal_v2(time_range: TimeRange) -> str:
    """
    Render one `TimeRange [start, end)` into a deterministic debug literal.

    Args:
        time_range: Shared-kernel half-open time range.
    Returns:
        str: Deterministic UTC debug literal.
    Assumptions:
        Stable error messages should not depend on locale-specific datetime formatting.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/time_range.py
    """
    return f"{time_range.start} .. {time_range.end}"
