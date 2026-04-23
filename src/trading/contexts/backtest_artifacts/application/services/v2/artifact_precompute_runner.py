"""Deterministic R3-03/R4-03/R5-01 artifact materialization into the inactive slot."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

import numpy as np
import yaml

from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    IndicatorSignalEvaluationInputV1,
    evaluate_indicator_signal_encoded_v1,
    indicator_primary_output_series_from_tensor_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, ComputeRequest
from trading.contexts.indicators.application.dto.variant_key import (
    IndicatorVariantSelection,
    build_variant_key_v1,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.services import GridBuilder
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.contexts.market_data.application.dto import CanonicalCandleBatch1m
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

from .artifact_precompute_coordinator import ArtifactPrecomputeCoordinatorV2
from .contracts import (
    ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
    ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
    ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2,
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
    ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    ARTIFACT_SIGNAL_VALUE_SET_V2,
    ARTIFACT_TIME_AXIS_ORDER_V2,
    HIT_TIMES_ARTIFACT_MANIFEST_KIND_V2,
    HIT_TIMES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
    HIT_TIMES_DIRECTORY_LITERAL_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    SIGNAL_ARTIFACT_MANIFEST_KIND_V2,
    SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
    SIGNAL_FEATURE_NAMES_V2,
    SIGNAL_FEATURES_ARTIFACT_MANIFEST_KIND_V2,
    SIGNAL_FEATURES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
    ArtifactArrayMetadataV2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactCoordinatesV2,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactHitTimesPathsV2,
    ArtifactHitTimesReferenceV2,
    ArtifactHitTimesTableManifestV2,
    ArtifactManifestDocumentV2,
    ArtifactManifestProvenanceV2,
    ArtifactMappingPathsV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactPrecomputeStageInputV2,
    ArtifactPrecomputeStageOutputV2,
    ArtifactPricePathsV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalCatalogV2,
    ArtifactSignalChunkJobV2,
    ArtifactSignalChunkPlanningRequestV2,
    ArtifactSignalEncodingContractV2,
    ArtifactSignalFeaturesManifestDocumentV2,
    ArtifactSignalFeaturesPathsV2,
    ArtifactSignalFeaturesReferenceV2,
    ArtifactSignalGridContractV2,
    ArtifactSignalManifestDocumentV2,
    ArtifactSignalPathsV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotLiteralV2,
    ArtifactStageRebuildStatsCollectionV2,
    ArtifactStageRebuildStatsV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    SignalRuleSpecV2,
    artifact_market_id_from_coordinates_v2,
    inactive_artifact_slot_v2,
    validate_artifact_slot_v2,
    validate_signal_input_source_v2,
)
from .hit_times_compute_v2 import (
    HitTimesArraysV2,
    hit_times_table_cell_count_v2,
    materialize_hit_times_from_ohlcv_v2,
    merge_hit_times_prefix_with_rebuilt_tail_v2,
)
from .signal_chunk_planner_v2 import DeterministicSignalChunkPlannerV2
from .signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
    _indicator_inputs_mapping_v2,
    _normalize_signal_codes_v2,
)

log = logging.getLogger(__name__)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2 = "1m"
_CANONICAL_CANDLE_SOURCE_LITERAL_V2 = "market_data.canonical_candles_1m"
_PRECOMPUTE_GENERATOR_LITERAL_V2 = "backtest-artifact-precompute-runner-v2"
_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2 = "r5-01"
_ONE_MINUTE_MILLIS_V2 = 60 * 1000
_ZERO_AXIS_SIGNAL_TARGET_IDS_V2 = (
    "structure.candle_stats",
    "volatility.tr",
    "volume.ad_line",
    "volume.obv",
)
_INDICATOR_AXES_BY_ID_V2 = {
    definition.indicator_id.value: tuple(definition.axes) for definition in all_defs()
}


def _log_precompute_stage_started_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_generation: int,
    force_full_rebuild: bool,
    stage: str,
    details: Mapping[str, Any],
) -> float:
    """
    Emit one structured stage-start log entry and return the monotonic start timestamp.

    Args:
        coordinates: Artifact coordinates identifying the symbol root under build.
        slot: Inactive slot literal receiving the current build.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        force_full_rebuild: Whether this run rebuilds from scratch instead of reusing prefixes.
        stage: Stable stage literal for log search and troubleshooting.
        details: Small JSON-serializable diagnostic payload for the stage.
    Returns:
        float: `time.perf_counter()` snapshot for later elapsed-time logging.
    Assumptions:
        Logging must stay deterministic and concise enough for operator grep/tail workflows.
    Raises:
        TypeError: If `details` cannot be JSON-serialized.
    Side Effects:
        Writes one INFO log record.
    """
    started_at = time.perf_counter()
    log.info(
        "event=artifact_precompute_stage_started component=backtest-artifact-precompute-runner "
        "stage=%s exchange=%s market_type=%s symbol=%s slot=%s slot_generation=%s "
        "force_full_rebuild=%s details=%s",
        stage,
        coordinates.exchange,
        coordinates.market_type,
        coordinates.symbol,
        slot,
        slot_generation,
        force_full_rebuild,
        json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )
    return started_at


def _log_precompute_stage_finished_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_generation: int,
    force_full_rebuild: bool,
    stage: str,
    started_at: float,
    details: Mapping[str, Any],
) -> None:
    """
    Emit one structured stage-finished log entry with elapsed wall-clock seconds.

    Args:
        coordinates: Artifact coordinates identifying the symbol root under build.
        slot: Inactive slot literal receiving the current build.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        force_full_rebuild: Whether this run rebuilds from scratch instead of reusing prefixes.
        stage: Stable stage literal for log search and troubleshooting.
        started_at: Earlier `time.perf_counter()` snapshot for this stage.
        details: Small JSON-serializable diagnostic payload for the stage result.
    Returns:
        None.
    Assumptions:
        Operators need elapsed timings even when per-stage metrics are not yet available.
    Raises:
        TypeError: If `details` cannot be JSON-serialized.
    Side Effects:
        Writes one INFO log record.
    """
    log.info(
        "event=artifact_precompute_stage_finished component=backtest-artifact-precompute-runner "
        "stage=%s exchange=%s market_type=%s symbol=%s slot=%s slot_generation=%s "
        "force_full_rebuild=%s elapsed_seconds=%.3f details=%s",
        stage,
        coordinates.exchange,
        coordinates.market_type,
        coordinates.symbol,
        slot,
        slot_generation,
        force_full_rebuild,
        time.perf_counter() - started_at,
        json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


class _SignalChunkWorkerSnapshotCapableV2(Protocol):
    """
    Internal protocol for compute adapters that can be rehydrated inside spawned chunk workers.

    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
    """

    def to_signal_chunk_worker_snapshot_v2(self) -> Mapping[str, Any]:
        """
        Serialize immutable worker snapshot state.

        Args:
            None.
        Returns:
            Mapping[str, Any]: Pickle-safe adapter snapshot.
        Assumptions:
            Snapshot contents are trusted immutable constructor inputs.
        Raises:
            None.
        Side Effects:
            None.
        """
        ...

    @classmethod
    def from_signal_chunk_worker_snapshot_v2(
        cls,
        *,
        snapshot: Mapping[str, Any],
    ) -> IndicatorCompute:
        """
        Rehydrate one compute adapter from a worker snapshot.

        Args:
            snapshot: Immutable snapshot mapping previously created by the live adapter.
        Returns:
            IndicatorCompute: Worker-local compute adapter.
        Assumptions:
            The worker uses the same code version as the parent runner process.
        Raises:
            ValueError: If snapshot contents violate adapter invariants.
        Side Effects:
            Reconstructs adapter-local runtime state.
        """
        ...


def _log_signal_chunk_progress_v2(
    *,
    event: str,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_generation: int,
    force_full_rebuild: bool,
    current_timeframe: str,
    current_indicator_id: str,
    chunk_job: ArtifactSignalChunkJobV2,
    completed_chunks_total: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """
    Emit one structured chunk-progress log entry for artifact signal execution.

    Args:
        event: Stable progress event literal for chunk start or finish.
        coordinates: Artifact coordinates identifying the symbol root under build.
        slot: Inactive slot literal receiving the current build.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        force_full_rebuild: Whether this run rebuilds from scratch instead of reusing prefixes.
        current_timeframe: Open timeframe session owning the chunk.
        current_indicator_id: Indicator id currently being materialized.
        chunk_job: Deterministic chunk job describing row ownership.
        completed_chunks_total: Optional completed-chunk counter emitted on finish.
        elapsed_seconds: Optional chunk wall-clock duration emitted on finish.
    Returns:
        None.
    Assumptions:
        Structured logs, not metrics, are the primary operator tool for per-indicator/per-chunk
        progress during long bootstrap runs.
    Raises:
        None.
    Side Effects:
        Writes one INFO log record.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    details = {
        "current_timeframe": current_timeframe,
        "current_indicator_id": current_indicator_id,
        "chunk_index": chunk_job.chunk_index,
        "chunk_count": chunk_job.chunk_count,
        "row_start_inclusive": chunk_job.row_start_inclusive,
        "row_end_exclusive": chunk_job.row_end_exclusive,
        "chunk_rows": chunk_job.chunk_rows,
    }
    if completed_chunks_total is not None:
        details["completed_chunks_total"] = completed_chunks_total
    if elapsed_seconds is None:
        log.info(
            "event=%s component=backtest-artifact-precompute-runner stage=%s exchange=%s "
            "market_type=%s symbol=%s slot=%s slot_generation=%s force_full_rebuild=%s "
            "current_timeframe=%s details=%s",
            event,
            "timeframe_session",
            coordinates.exchange,
            coordinates.market_type,
            coordinates.symbol,
            slot,
            slot_generation,
            force_full_rebuild,
            current_timeframe,
            json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
        return
    log.info(
        "event=%s component=backtest-artifact-precompute-runner stage=%s exchange=%s "
        "market_type=%s symbol=%s slot=%s slot_generation=%s force_full_rebuild=%s "
        "current_timeframe=%s elapsed_seconds=%.3f details=%s",
        event,
        "timeframe_session",
        coordinates.exchange,
        coordinates.market_type,
        coordinates.symbol,
        slot,
        slot_generation,
        force_full_rebuild,
        current_timeframe,
        elapsed_seconds,
        json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


@dataclass(frozen=True, slots=True)
class _CanonicalPriceArraysV2:
    """
    Internal immutable container for `open_time/close_time/ohlcv` price arrays.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    prefix: _CanonicalPriceArraysV2 | None
    source_time_range: TimeRange


@dataclass(frozen=True, slots=True)
class _CanonicalPriceStageBuildResultV2:
    """
    Internal immutable output of the canonical-prices stage before timeframe sessions start.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    tail_plan: _CanonicalPriceTailPlanV2
    tail_arrays: _CanonicalPriceArraysV2
    materialized_arrays: _CanonicalPriceArraysV2
    one_minute_manifest: ArtifactPriceTimeframeManifestV2
    rollup_source_arrays: _CanonicalPriceArraysV2


@dataclass(frozen=True, slots=True)
class _RootManifestScaffoldV2:
    """
    Internal scaffold for root-manifest sections not owned by R3-03 price/mapping materialization.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
    """

    inputs_source: str | None
    variant_key: str


@dataclass(frozen=True, slots=True)
class _SignalArtifactMaterializationResultV2:
    """
    Internal immutable output of one strict signal artifact materialization target.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    manifest: ArtifactSignalManifestDocumentV2
    reused_prefix_bars: int
    rewritten_tail_bars: int
    completed_chunks_total: int


@dataclass(frozen=True, slots=True)
class _SignalFeaturesArtifactBuildResultV2:
    """
    Internal immutable output of one strict additive signal-feature artifact materialization.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """

    manifest: ArtifactSignalFeaturesManifestDocumentV2
    reference: ArtifactSignalFeaturesReferenceV2


@dataclass(frozen=True, slots=True)
class _HitTimesArtifactBuildResultV2:
    """
    Internal immutable output of one strict R5-01 `hit_times/15m` materialization pass.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    manifest: ArtifactHitTimesManifestDocumentV2
    reference: ArtifactHitTimesReferenceV2
    reused_prefix_bars: int
    rewritten_tail_bars: int


@dataclass(frozen=True, slots=True)
class _TimeframeMappingBuildResultV2:
    """
    Internal immutable output of one `mappings/<tf>` tail rebuild decision.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    arrays: _TimeframeMappingArraysV2
    reused_prefix_bars: int
    rewritten_tail_bars: int


@dataclass(frozen=True, slots=True)
class _MappingArtifactMaterializationResultV2:
    """
    Internal immutable result of one full `mappings/<tf>` materialization target.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    manifest: ArtifactMappingTimeframeManifestV2
    reused_prefix_bars: int
    rewritten_tail_bars: int


@dataclass(frozen=True, slots=True)
class _ExistingSignalArtifactV2:
    """
    Internal immutable snapshot of one existing inactive-slot signal family eligible for reuse.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    catalog_entry: ArtifactSignalCatalogEntryV2
    manifest: ArtifactSignalManifestDocumentV2
    signals_path: Path


@dataclass(frozen=True, slots=True)
class _ExistingHitTimesArtifactV2:
    """
    Internal immutable snapshot of one existing inactive-slot `hit_times/15m` family.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    manifest: ArtifactHitTimesManifestDocumentV2
    arrays: HitTimesArraysV2


@dataclass(frozen=True, slots=True)
class _SignalArtifactTailPlanV2:
    """
    Internal deterministic plan for signal prefix reuse and bounded tail rebuild.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    reused_prefix_bars: int
    compute_start_idx: int
    trim_prefix_bars: int
    effective_tail_bars: int


@dataclass(frozen=True, slots=True)
class _SignalChunkGridBlockV2:
    """
    Internal immutable subgrid snapshot covering one contiguous row block inside a chunk.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
    """

    row_start_inclusive: int
    row_end_exclusive: int
    source_values: tuple[str, ...] | None
    param_values_by_name: tuple[tuple[str, tuple[int | float | str, ...]], ...]


@dataclass(frozen=True, slots=True)
class _SignalChunkWorkerBootstrapV2:
    """
    Immutable worker-bootstrap payload shared once per spawned chunk worker session.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    indicator_compute_worker_class: type[Any]
    indicator_compute_worker_snapshot: Mapping[str, Any]
    candles: CandleArrays


@dataclass(frozen=True, slots=True)
class _SignalChunkWorkerStateV2:
    """
    Worker-local runtime state reused across all chunk jobs inside one spawned process.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    indicator_compute: IndicatorCompute
    candles: CandleArrays


@dataclass(frozen=True, slots=True)
class _SignalChunkWorkerResultV2:
    """
    Internal immutable summary returned by one completed chunk worker.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    chunk_job: ArtifactSignalChunkJobV2
    signal_params_defaults: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _HitTimesArtifactTailPlanV2:
    """
    Internal deterministic plan for `hit_times/15m` prefix reuse and bounded tail rebuild.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    prefix: HitTimesArraysV2 | None
    prefix_bars: int
    effective_tail_bars: int


@dataclass(frozen=True, slots=True)
class _TimeframeSessionBuildResultV2:
    """
    Internal immutable output of one explicit R12 timeframe session.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
    """

    price_manifest: ArtifactPriceTimeframeManifestV2
    mapping_manifest: ArtifactMappingTimeframeManifestV2
    signal_manifests: tuple[ArtifactSignalManifestDocumentV2, ...]
    reused_mapping_prefix_bars: int
    rewritten_mapping_tail_bars: int
    reused_signal_prefix_bars: int
    rewritten_signal_tail_bars: int
    completed_chunks_total: int
    completed_indicators_total: int


@dataclass(frozen=True, slots=True)
class BacktestArtifactPrecomputeRunnerV2:
    """
    Materialize canonical prices, mappings, `hit_times/15m`, and optional signal artifacts.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
        Export canonical `1m`, `hit_times/15m`, and timeframe-local `rolled_prices` sessions
        into the inactive slot.

        Args:
            request: Explicit export identity with symbol coordinates and `TimeRange [start, end)`.
        Returns:
            ArtifactCanonicalPriceExportResultV2: Structured write result for the inactive slot.
        Assumptions:
            Public API stays rooted in canonical `1m`, while R12-03 keeps both
            `canonical_prices` and `hit_times` in the canonical `1m` scope before the explicit
            per-timeframe `rolled_prices -> mappings -> signals` loop begins.
        Raises:
            FileNotFoundError: If strict `current.yaml` is missing for the symbol root.
            ValueError: If existing inactive-slot metadata, source candles, or derived
                artifact contracts violate strict ordering, dtype, path, or hit-times budgets.
            OSError: If one atomic file write fails.
        Side Effects:
            Reads canonical candles through the port and atomically replaces inactive-slot price,
            mapping, hit-times, and root-manifest files.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        inactive_slot, target_slot_generation = _resolve_export_target_v2(
            artifact_loader=self.artifact_loader,
            request=request,
        )
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
        existing_manifest = None
        existing_arrays = None
        if not request.force_full_rebuild:
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
        export_started_at = time.perf_counter()
        coordinator = ArtifactPrecomputeCoordinatorV2(
            coordinates=request.coordinates,
            slot=inactive_slot,
            slot_generation=target_slot_generation,
            force_full_rebuild=request.force_full_rebuild,
            execution_policy=self.runtime_settings.execution_policy,
        )
        try:
            canonical_stage_result = coordinator.run_stage(
                stage_input=ArtifactPrecomputeStageInputV2(
                    stage="canonical_prices",
                    details={
                        "price_tail_bars_1m": self.runtime_settings.price_tail_bars_1m,
                        "source_mode": "columnar_arrays",
                    },
                ),
                execute=lambda: _materialize_canonical_prices_stage_v2(
                    artifact_loader=self.artifact_loader,
                    canonical_candle_reader=self.canonical_candle_reader,
                    coordinates=request.coordinates,
                    slot=inactive_slot,
                    slot_root=slot_root,
                    existing_manifest=existing_manifest,
                    existing_arrays=existing_arrays,
                    request=request,
                    runtime_settings=self.runtime_settings,
                ),
                build_output=lambda stage_result: ArtifactPrecomputeStageOutputV2(
                    stage="canonical_prices",
                    reused_prefix_bars=(
                        0
                        if stage_result.tail_plan.prefix is None
                        else int(stage_result.tail_plan.prefix.open_time.shape[0])
                    ),
                    rewritten_tail_bars=int(stage_result.tail_arrays.open_time.shape[0]),
                    details={
                        "source_time_range": _time_range_literal_v2(
                            stage_result.tail_plan.source_time_range
                        ),
                        "source_candle_count": int(stage_result.tail_arrays.open_time.shape[0]),
                        "timeline_bar_count": int(
                            stage_result.materialized_arrays.open_time.shape[0]
                        ),
                    },
                ),
            )

            hit_times_source_arrays = _resolve_hit_times_source_arrays_v2(
                one_minute_arrays=canonical_stage_result.rollup_source_arrays,
            )
            hit_times_budget = _resolve_hit_times_cell_budget_v2(
                runtime_settings=self.runtime_settings,
                force_full_rebuild=request.force_full_rebuild,
                has_existing_slot_manifest=existing_manifest is not None,
            )
            hit_times_timeline_bar_count = int(
                hit_times_source_arrays.open_time.shape[0]
            )
            hit_times_tp_level_count = len(self.runtime_settings.hit_times_tp_levels_pct)
            hit_times_sl_level_count = len(self.runtime_settings.hit_times_sl_levels_pct)
            hit_times_build_result = coordinator.run_stage(
                stage_input=ArtifactPrecomputeStageInputV2(
                    stage="hit_times",
                    details={
                        "hit_times_tail_bars_1m": self.runtime_settings.hit_times_tail_bars_1m,
                        "max_hit_times_cells": hit_times_budget,
                        "timeline_bar_count": hit_times_timeline_bar_count,
                        "tp_level_count": hit_times_tp_level_count,
                        "sl_level_count": hit_times_sl_level_count,
                        "table_cell_count": hit_times_table_cell_count_v2(
                            timeline_bar_count=hit_times_timeline_bar_count,
                            tp_level_count=hit_times_tp_level_count,
                            sl_level_count=hit_times_sl_level_count,
                        ),
                    },
                ),
                execute=lambda: _materialize_hit_times_artifacts_v2(
                    artifact_loader=self.artifact_loader,
                    coordinates=request.coordinates,
                    slot=inactive_slot,
                    slot_root=slot_root,
                    existing_manifest=existing_manifest,
                    request=request,
                    slot_generation=target_slot_generation,
                    runtime_settings=self.runtime_settings,
                    hit_times_source_arrays=hit_times_source_arrays,
                    one_minute_manifest=canonical_stage_result.one_minute_manifest,
                    max_hit_times_cells=hit_times_budget,
                ),
                build_output=lambda stage_result: ArtifactPrecomputeStageOutputV2(
                    stage="hit_times",
                    reused_prefix_bars=stage_result.reused_prefix_bars,
                    rewritten_tail_bars=stage_result.rewritten_tail_bars,
                    details={
                        "reused_prefix_bars": stage_result.reused_prefix_bars,
                        "rewritten_tail_bars": stage_result.rewritten_tail_bars,
                        "timeline_bar_count": stage_result.manifest.timeline_bar_count,
                        "tp_level_count": int(stage_result.manifest.tp_values.shape[0]),
                        "sl_level_count": int(stage_result.manifest.sl_values.shape[0]),
                        "table_cell_count": hit_times_table_cell_count_v2(
                            timeline_bar_count=stage_result.manifest.timeline_bar_count,
                            tp_level_count=int(stage_result.manifest.tp_values.shape[0]),
                            sl_level_count=int(stage_result.manifest.sl_values.shape[0]),
                        ),
                    },
                ),
            )

            rolled_price_manifests: list[ArtifactPriceTimeframeManifestV2] = []
            mapping_manifests: list[ArtifactMappingTimeframeManifestV2] = []
            signal_manifests: list[ArtifactSignalManifestDocumentV2] = []
            mapping_reused_prefix_bars = 0
            mapping_rewritten_tail_bars = 0
            signal_reused_prefix_bars = 0
            signal_rewritten_tail_bars = 0
            signal_targets_by_timeframe = _group_signal_targets_by_timeframe_v2(
                signal_targets=self.runtime_settings.signal_artifacts
            )
            for timeframe in self.runtime_settings.mapping_timeframes:
                timeframe_signal_targets = signal_targets_by_timeframe.get(timeframe, ())
                timeframe_stage_result: _TimeframeSessionBuildResultV2 | None = None
                with coordinator.open_timeframe_session(
                    timeframe=timeframe,
                    details={
                        "current_timeframe": timeframe,
                        "signal_target_count": len(timeframe_signal_targets),
                        "signal_worker_processes": (
                            self.runtime_settings.execution_policy.signal_worker_processes
                        ),
                        "signal_worker_memory_budget_bytes": (
                            self.runtime_settings.execution_policy.signal_worker_memory_budget_bytes
                        ),
                        "signal_chunk_rows_min": (
                            self.runtime_settings.execution_policy.signal_chunk_rows_min
                        ),
                        "signal_chunk_rows_max": (
                            self.runtime_settings.execution_policy.signal_chunk_rows_max
                        ),
                    },
                ) as session:
                    completed_session_result = coordinator.run_stage(
                        stage_input=ArtifactPrecomputeStageInputV2(
                            stage="timeframe_session",
                            current_timeframe=timeframe,
                            details={
                                "current_timeframe": timeframe,
                                "mapping_tail_bars_1m": (
                                    self.runtime_settings.mapping_tail_bars_1m
                                ),
                                "signal_tail_bars_1m": (
                                    self.runtime_settings.signal_tail_bars_1m
                                ),
                                "signal_target_count": len(timeframe_signal_targets),
                                "signal_worker_processes": (
                                    self.runtime_settings.execution_policy.signal_worker_processes
                                ),
                                "signal_worker_memory_budget_bytes": (
                                    self.runtime_settings.execution_policy.signal_worker_memory_budget_bytes
                                ),
                                "signal_chunk_rows_min": (
                                    self.runtime_settings.execution_policy.signal_chunk_rows_min
                                ),
                                "signal_chunk_rows_max": (
                                    self.runtime_settings.execution_policy.signal_chunk_rows_max
                                ),
                                "max_open_timeframe_sessions": (
                                    self.runtime_settings.execution_policy.max_open_timeframe_sessions
                                ),
                            },
                        ),
                        execute=lambda timeframe=timeframe, timeframe_signal_targets=(
                            timeframe_signal_targets
                        ): _materialize_timeframe_session_v2(
                            artifact_loader=self.artifact_loader,
                            coordinates=request.coordinates,
                            slot=inactive_slot,
                            slot_root=slot_root,
                            existing_manifest=existing_manifest,
                            request=request,
                            slot_generation=target_slot_generation,
                            runtime_settings=self.runtime_settings,
                            timeframe=timeframe,
                            one_minute_arrays=canonical_stage_result.rollup_source_arrays,
                            source_tail_time_range=(
                                canonical_stage_result.tail_plan.source_time_range
                            ),
                            signal_targets=timeframe_signal_targets,
                            defaults_provider=self.defaults_provider,
                            signal_rules_engine=self.signal_rules_engine,
                            indicator_compute=self.indicator_compute,
                            indicator_grid_builder=self.indicator_grid_builder,
                        ),
                        build_output=lambda stage_result, timeframe=timeframe: (
                            ArtifactPrecomputeStageOutputV2(
                                stage="timeframe_session",
                                current_timeframe=timeframe,
                                reused_prefix_bars=(
                                    stage_result.reused_mapping_prefix_bars
                                    + stage_result.reused_signal_prefix_bars
                                ),
                                rewritten_tail_bars=(
                                    stage_result.rewritten_mapping_tail_bars
                                    + stage_result.rewritten_signal_tail_bars
                                ),
                                details={
                                    "current_timeframe": timeframe,
                                    "timeframe_bar_count": (
                                        stage_result.price_manifest.coverage.bar_count
                                    ),
                                    "mapping_reused_prefix_bars": (
                                        stage_result.reused_mapping_prefix_bars
                                    ),
                                    "mapping_rewritten_tail_bars": (
                                        stage_result.rewritten_mapping_tail_bars
                                    ),
                                    "signal_reused_prefix_bars": (
                                        stage_result.reused_signal_prefix_bars
                                    ),
                                    "signal_rewritten_tail_bars": (
                                        stage_result.rewritten_signal_tail_bars
                                    ),
                                    "completed_chunks_total": (
                                        stage_result.completed_chunks_total
                                    ),
                                    "completed_indicators_total": (
                                        stage_result.completed_indicators_total
                                    ),
                                    "signal_manifest_count": len(
                                        stage_result.signal_manifests
                                    ),
                                },
                            )
                        ),
                    )
                    session.set_finish_details(
                        details={
                            "current_timeframe": timeframe,
                            "timeframe_bar_count": (
                                completed_session_result.price_manifest.coverage.bar_count
                            ),
                            "mapping_reused_prefix_bars": (
                                completed_session_result.reused_mapping_prefix_bars
                            ),
                            "mapping_rewritten_tail_bars": (
                                completed_session_result.rewritten_mapping_tail_bars
                            ),
                            "signal_reused_prefix_bars": (
                                completed_session_result.reused_signal_prefix_bars
                            ),
                            "signal_rewritten_tail_bars": (
                                completed_session_result.rewritten_signal_tail_bars
                            ),
                            "completed_chunks_total": (
                                completed_session_result.completed_chunks_total
                            ),
                            "completed_indicators_total": (
                                completed_session_result.completed_indicators_total
                            ),
                            "signal_manifest_count": len(
                                completed_session_result.signal_manifests
                            ),
                        }
                    )
                    timeframe_stage_result = completed_session_result
                if timeframe_stage_result is None:
                    raise ValueError(
                        f"timeframe session {timeframe!r} completed without a stage result"
                    )
                rolled_price_manifests.append(timeframe_stage_result.price_manifest)
                mapping_manifests.append(timeframe_stage_result.mapping_manifest)
                signal_manifests.extend(timeframe_stage_result.signal_manifests)
                mapping_reused_prefix_bars += timeframe_stage_result.reused_mapping_prefix_bars
                mapping_rewritten_tail_bars += timeframe_stage_result.rewritten_mapping_tail_bars
                signal_reused_prefix_bars += timeframe_stage_result.reused_signal_prefix_bars
                signal_rewritten_tail_bars += timeframe_stage_result.rewritten_signal_tail_bars

            scaffold = _build_root_manifest_scaffold_v2(existing_manifest=existing_manifest)
            root_signals = (
                scaffold.signals
                if len(signal_manifests) == 0
                else _build_signal_catalog_from_manifests_v2(
                    slot_root=slot_root,
                    signal_manifests=tuple(signal_manifests),
                )
            )
            effective_scaffold = _RootManifestScaffoldV2(
                preserved_prices=scaffold.preserved_prices,
                mappings=scaffold.mappings,
                signals=root_signals,
                hit_times=hit_times_build_result.reference,
                signal_encoding=scaffold.signal_encoding,
            )
            written_manifest_path = coordinator.run_stage(
                stage_input=ArtifactPrecomputeStageInputV2(
                    stage="root_manifest",
                    details={
                        "price_section_count": 1 + len(rolled_price_manifests),
                        "mapping_section_count": len(mapping_manifests),
                        "signal_manifest_count": len(root_signals.manifests),
                    },
                ),
                execute=lambda: _write_root_manifest_stage_v2(
                    manifest_path=manifest_path,
                    request=request,
                    slot=inactive_slot,
                    slot_generation=target_slot_generation,
                    runtime_settings=self.runtime_settings,
                    root_scaffold=effective_scaffold,
                    materialized_arrays=canonical_stage_result.materialized_arrays,
                    price_manifests=(
                        canonical_stage_result.one_minute_manifest,
                        *tuple(rolled_price_manifests),
                    ),
                    mapping_manifests=tuple(mapping_manifests),
                    signal_catalog=root_signals,
                    hit_times_reference=hit_times_build_result.reference,
                ),
                build_output=lambda written_path: ArtifactPrecomputeStageOutputV2(
                    stage="root_manifest",
                    details={"manifest_path": str(written_path)},
                ),
            )

            stage_rebuild_stats = ArtifactStageRebuildStatsCollectionV2(
                prices=ArtifactStageRebuildStatsV2(
                    reused_prefix_bars=(
                        0
                        if canonical_stage_result.tail_plan.prefix is None
                        else int(canonical_stage_result.tail_plan.prefix.open_time.shape[0])
                    ),
                    rewritten_tail_bars=int(
                        canonical_stage_result.tail_arrays.open_time.shape[0]
                    ),
                ),
                mappings=ArtifactStageRebuildStatsV2(
                    reused_prefix_bars=mapping_reused_prefix_bars,
                    rewritten_tail_bars=mapping_rewritten_tail_bars,
                ),
                signals=ArtifactStageRebuildStatsV2(
                    reused_prefix_bars=signal_reused_prefix_bars,
                    rewritten_tail_bars=signal_rewritten_tail_bars,
                ),
                hit_times=ArtifactStageRebuildStatsV2(
                    reused_prefix_bars=hit_times_build_result.reused_prefix_bars,
                    rewritten_tail_bars=hit_times_build_result.rewritten_tail_bars,
                ),
            )
            result = ArtifactCanonicalPriceExportResultV2(
                coordinates=request.coordinates,
                slot=inactive_slot,
                slot_generation=target_slot_generation,
                asof_date=request.asof_date,
                manifest_path=written_manifest_path,
                manifest_sha256=_file_sha256_hex_v2(written_manifest_path),
                price_paths=price_paths,
                coverage=canonical_stage_result.one_minute_manifest.coverage,
                source_time_range=canonical_stage_result.tail_plan.source_time_range,
                source_candle_count=int(canonical_stage_result.tail_arrays.open_time.shape[0]),
                reused_prefix_bars=stage_rebuild_stats.prices.reused_prefix_bars,
                rewritten_tail_bars=stage_rebuild_stats.prices.rewritten_tail_bars,
                stage_results=coordinator.stage_results(),
                stage_rebuild_stats=stage_rebuild_stats,
                tail_rebuild_bars=stage_rebuild_stats.tail_rebuild_bars(),
            )
            coordinator.emit_completed(
                elapsed_seconds=time.perf_counter() - export_started_at,
                tail_rebuild_bars=result.tail_rebuild_bars,
            )
            return result
        except Exception:
            log.exception(
                "event=artifact_precompute_failed component=backtest-artifact-precompute-runner "
                "exchange=%s market_type=%s symbol=%s slot=%s slot_generation=%s "
                "force_full_rebuild=%s elapsed_seconds=%.3f",
                request.coordinates.exchange,
                request.coordinates.market_type,
                request.coordinates.symbol,
                inactive_slot,
                target_slot_generation,
                request.force_full_rebuild,
                time.perf_counter() - export_started_at,
            )
            raise


def _materialize_canonical_prices_stage_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    canonical_candle_reader: CanonicalCandleReader,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    existing_arrays: _CanonicalPriceArraysV2 | None,
    request: ArtifactCanonicalPriceExportRequestV2,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
) -> _CanonicalPriceStageBuildResultV2:
    """
    Materialize the canonical `prices/1m` stage that seeds later R12 timeframe sessions.

    Args:
        artifact_loader: Explicit-path artifact loader.
        canonical_candle_reader: Canonical `1m` source reader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving the canonical arrays.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest, when present.
        existing_arrays: Existing inactive-slot canonical arrays, when present.
        request: Explicit export request with time-range and identity metadata.
        runtime_settings: Strict runtime settings including the canonical tail-rebuild budget.
    Returns:
        _CanonicalPriceStageBuildResultV2: Canonical arrays plus the one-minute manifest and
            rollup source arrays for later stages.
    Assumptions:
        The canonical stage owns source reread and `prices/1m` writes before any timeframe
        session opens.
    Raises:
        ValueError: If source candles, reused prefixes, or written arrays violate strict
            canonical-price contracts.
        OSError: If writing the canonical artifact family fails.
    Side Effects:
        Reads canonical candles and atomically writes the inactive-slot `prices/1m` family.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """
    tail_plan = _build_tail_plan_v2(
        request=request,
        existing_arrays=existing_arrays,
        lookback_bars=runtime_settings.price_tail_bars_1m,
    )
    price_paths = artifact_loader.resolve_price_paths(
        coordinates,
        slot,
        _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
    )
    tail_arrays = _read_canonical_price_arrays_v2(
        canonical_candle_reader=canonical_candle_reader,
        coordinates=coordinates,
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
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
        manifest_section=one_minute_manifest,
        location_prefix="materialized prices[1m] rollup source",
    )
    return _CanonicalPriceStageBuildResultV2(
        tail_plan=tail_plan,
        tail_arrays=tail_arrays,
        materialized_arrays=materialized_arrays,
        one_minute_manifest=one_minute_manifest,
        rollup_source_arrays=rollup_source_arrays,
    )


def _group_signal_targets_by_timeframe_v2(
    *,
    signal_targets: tuple[ArtifactSignalValidationSpecV2, ...],
) -> Mapping[str, tuple[ArtifactSignalValidationSpecV2, ...]]:
    """
    Group canonical signal targets by timeframe while preserving deterministic order.

    Args:
        signal_targets: Canonically ordered `(timeframe, indicator_id)` signal targets.
    Returns:
        Mapping[str, tuple[ArtifactSignalValidationSpecV2, ...]]: Timeframe-keyed immutable target
            groups preserving the original ordering within each timeframe.
    Assumptions:
        Runtime settings already canonicalize target ordering before the runner groups them.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    grouped: dict[str, list[ArtifactSignalValidationSpecV2]] = {}
    for target in signal_targets:
        grouped.setdefault(target.timeframe, []).append(target)
    return {
        timeframe: tuple(targets)
        for timeframe, targets in grouped.items()
    }


def _materialize_timeframe_session_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: ArtifactSlotLiteralV2,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    timeframe: str,
    one_minute_arrays: _CanonicalPriceArraysV2,
    source_tail_time_range: TimeRange,
    signal_targets: tuple[ArtifactSignalValidationSpecV2, ...],
    defaults_provider: BacktestGridDefaultsProvider | None,
    signal_rules_engine: BacktestSignalRulesEngineV2 | None,
    indicator_compute: IndicatorCompute | None,
    indicator_grid_builder: GridBuilder | None,
) -> _TimeframeSessionBuildResultV2:
    """
    Materialize one explicit R12 timeframe session from `rolled_prices` through `signals`.

    Args:
        artifact_loader: Explicit-path artifact loader used to resolve fixed output paths.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving timeframe-scoped artifacts.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest, when present.
        request: Explicit export request carrying shared timestamps and identity.
        slot_generation: Target inactive-slot generation assigned by the caller.
        runtime_settings: Strict runtime settings for mapping/signal lookbacks and budgets.
        timeframe: Target request timeframe opened by the current session.
        one_minute_arrays: Materialized canonical `prices/1m` arrays reused by this session.
        source_tail_time_range: Canonical tail-reread window used by the current build.
        signal_targets: Explicit signal targets for this timeframe only.
        defaults_provider: Runtime defaults provider for compute grids and signal defaults.
        signal_rules_engine: Startup-validated signal rules engine.
        indicator_compute: Indicator compute port used for signal tensors.
        indicator_grid_builder: Grid builder used for deterministic variant ordering.
    Returns:
        _TimeframeSessionBuildResultV2: Strict rolled-price, mapping, and signal outputs for the
            opened timeframe session.
    Assumptions:
        One session owns exactly one `current_timeframe` and fully releases it before the next
        session opens.
    Raises:
        ValueError: If rolled prices, mappings, or signals drift from strict contracts.
        OSError: If writing one timeframe-scoped artifact family fails.
    Side Effects:
        Atomically writes `prices/<tf>`, `mappings/<tf>`, and optional `signals/<tf>/*`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    price_manifest = _materialize_rolled_price_timeframe_v2(
        timeframe,
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        slot_root=slot_root,
        existing_manifest=existing_manifest,
        source_arrays=one_minute_arrays,
        source_tail_time_range=source_tail_time_range,
    )
    mapping_result = _materialize_mapping_timeframe_v2(
        timeframe,
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        slot_root=slot_root,
        existing_manifest=existing_manifest,
        one_minute_arrays=one_minute_arrays,
        price_by_timeframe={timeframe: price_manifest},
        mapping_tail_bars_1m=runtime_settings.mapping_tail_bars_1m,
    )
    session_price_arrays: _CanonicalPriceArraysV2 | None = None
    if signal_targets != ():
        session_price_arrays = _load_materialized_price_arrays_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            timeframe=timeframe,
            manifest_section=price_manifest,
            location_prefix=f"materialized prices[{timeframe}] timeframe session",
        )
    signal_manifests: list[ArtifactSignalManifestDocumentV2] = []
    reused_signal_prefix_bars = 0
    rewritten_signal_tail_bars = 0
    completed_chunks_total = 0
    completed_indicators_total = 0
    for signal_target in signal_targets:
        if (
            defaults_provider is None
            or signal_rules_engine is None
            or indicator_compute is None
            or indicator_grid_builder is None
        ):
            raise ValueError(
                "R12 timeframe-session signal materialization requires defaults_provider, "
                "signal_rules_engine, indicator_compute, and indicator_grid_builder"
            )
        signal_build_result = _materialize_signal_artifact_v2(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            slot_root=slot_root,
            existing_manifest=existing_manifest,
            request=request,
            slot_generation=slot_generation,
            runtime_settings=runtime_settings,
            signal_target=signal_target,
            price_manifest=price_manifest,
            session_price_arrays=session_price_arrays,
            defaults_provider=defaults_provider,
            signal_rules_engine=signal_rules_engine,
            indicator_compute=indicator_compute,
            indicator_grid_builder=indicator_grid_builder,
        )
        signal_manifests.append(signal_build_result.manifest)
        reused_signal_prefix_bars += signal_build_result.reused_prefix_bars
        rewritten_signal_tail_bars += signal_build_result.rewritten_tail_bars
        completed_chunks_total += signal_build_result.completed_chunks_total
        completed_indicators_total += 1
    return _TimeframeSessionBuildResultV2(
        price_manifest=price_manifest,
        mapping_manifest=mapping_result.manifest,
        signal_manifests=tuple(signal_manifests),
        reused_mapping_prefix_bars=mapping_result.reused_prefix_bars,
        rewritten_mapping_tail_bars=mapping_result.rewritten_tail_bars,
        reused_signal_prefix_bars=reused_signal_prefix_bars,
        rewritten_signal_tail_bars=rewritten_signal_tail_bars,
        completed_chunks_total=completed_chunks_total,
        completed_indicators_total=completed_indicators_total,
    )


def _build_signal_catalog_from_manifests_v2(
    *,
    slot_root: Path,
    signal_manifests: tuple[ArtifactSignalManifestDocumentV2, ...],
) -> ArtifactSignalCatalogV2:
    """
    Build the deterministic root signal catalog from already written signal manifests.

    Args:
        slot_root: Absolute inactive-slot root directory.
        signal_manifests: Finished strict signal manifests written during timeframe sessions.
    Returns:
        ArtifactSignalCatalogV2: Deterministic catalog preserving canonical timeframe/indicator
            ordering.
    Assumptions:
        Signal manifests already exist on disk and may be hashed directly without rescanning the
        filesystem for discovery.
    Raises:
        OSError: If one manifest hash cannot be read from disk.
    Side Effects:
        Reads manifest files to compute stable SHA-256 catalog entries.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_SIGNAL_TIMEFRAMES_V2)
    }
    catalog_entries = tuple(
        sorted(
            (
                ArtifactSignalCatalogEntryV2(
                    timeframe=signal_manifest.timeframe,
                    indicator_id=signal_manifest.indicator_id,
                    manifest_path=_slot_relative_path_v2(
                        slot_root=slot_root,
                        absolute_path=signal_manifest.path,
                    ),
                    manifest_sha256=_file_sha256_hex_v2(signal_manifest.path),
                )
                for signal_manifest in signal_manifests
            ),
            key=lambda entry: (timeframe_order[entry.timeframe], entry.indicator_id),
        )
    )
    return ArtifactSignalCatalogV2(
        supported_timeframes=tuple(dict.fromkeys(entry.timeframe for entry in catalog_entries)),
        supported_indicator_ids=tuple(sorted({entry.indicator_id for entry in catalog_entries})),
        manifests=catalog_entries,
    )


def _write_root_manifest_stage_v2(
    *,
    manifest_path: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot: ArtifactSlotLiteralV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    root_scaffold: _RootManifestScaffoldV2,
    materialized_arrays: _CanonicalPriceArraysV2,
    price_manifests: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_manifests: tuple[ArtifactMappingTimeframeManifestV2, ...],
    signal_catalog: ArtifactSignalCatalogV2,
    hit_times_reference: ArtifactHitTimesReferenceV2,
) -> Path:
    """
    Write the strict root manifest after all R12 build stages have finished.

    Args:
        manifest_path: Absolute inactive-slot root `manifest.yaml` path.
        request: Explicit export request carrying shared timestamps and identity.
        slot: Inactive slot literal receiving the current build.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        runtime_settings: Strict runtime settings used for provenance.
        root_scaffold: Existing-manifest scaffold for sections not recomputed in this helper.
        materialized_arrays: Fresh canonical `prices/1m` arrays used for provenance.
        price_manifests: Full strict price-manifest tuple including `1m` and rolled timeframes.
        mapping_manifests: Full strict mapping-manifest tuple for request timeframes.
        signal_catalog: Final root signal catalog for the build.
        hit_times_reference: Final hit-times reference for the build.
    Returns:
        Path: The written root manifest path.
    Assumptions:
        Whole-slot publish validation remains external; this helper only finalizes root metadata.
    Raises:
        OSError: If writing the root manifest fails.
    Side Effects:
        Atomically writes the inactive-slot root `manifest.yaml`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    provenance = _build_root_manifest_provenance_v2(
        runtime_settings=runtime_settings,
        request=request,
        arrays=materialized_arrays,
        rolled_sections=price_manifests[1:],
        mapping_sections=mapping_manifests,
        signal_entries=signal_catalog.manifests,
        hit_times_reference=hit_times_reference,
    )
    root_manifest_payload = _build_root_manifest_payload_v2(
        request=request,
        slot=slot,
        slot_generation=slot_generation,
        root_scaffold=_RootManifestScaffoldV2(
            preserved_prices=root_scaffold.preserved_prices,
            mappings=root_scaffold.mappings,
            signals=signal_catalog,
            hit_times=hit_times_reference,
            signal_encoding=root_scaffold.signal_encoding,
        ),
        price_manifests=price_manifests,
        mapping_manifests=mapping_manifests,
        provenance=provenance,
    )
    _write_yaml_atomically_v2(path=manifest_path, payload=root_manifest_payload)
    return manifest_path


def _resolve_export_target_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    request: ArtifactCanonicalPriceExportRequestV2,
) -> tuple[ArtifactSlotLiteralV2, int]:
    """
    Resolve the explicit inactive-slot target for one precompute export request.

    Args:
        artifact_loader: Artifact loader used when the request does not override target identity.
        request: Export request that may carry an explicit target slot/generation override.
    Returns:
        tuple[ArtifactSlotLiteralV2, int]: Deterministic target slot and slot generation.
    Assumptions:
        Shared orchestration may pre-resolve bootstrap target identity, while legacy callers still
        derive it from strict `current.yaml`.
    Raises:
        FileNotFoundError: If strict `current.yaml` is required but missing.
        ValueError: If `current.yaml` or explicit target fields violate strict contracts.
    Side Effects:
        Reads `current.yaml` only when request does not already specify target identity.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if request.target_slot is not None and request.target_slot_generation is not None:
        return request.target_slot, request.target_slot_generation
    current_pointer = artifact_loader.load_current_pointer(request.coordinates)
    return (
        inactive_artifact_slot_v2(current_pointer.active_slot),
        current_pointer.slot_generation + 1,
    )


def _materialize_signal_artifact_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    session_price_arrays: _CanonicalPriceArraysV2 | None,
    defaults_provider: BacktestGridDefaultsProvider,
    signal_rules_engine: BacktestSignalRulesEngineV2,
    indicator_compute: IndicatorCompute,
    indicator_grid_builder: GridBuilder,
) -> _SignalArtifactMaterializationResultV2:
    """
    Materialize one strict per-timeframe/per-indicator signal artifact family.

    Args:
        artifact_loader: Explicit-path artifact loader used to resolve fixed output paths.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the signal artifact.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Previously materialized inactive-slot root manifest, when present.
        request: Explicit export request carrying root identity and timestamps.
        runtime_settings: Strict runtime settings with signal guard budgets.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        price_manifest: Fresh materialized price section for the same timeframe.
        session_price_arrays: Already loaded session-owned `prices/<tf>` arrays reused across
            every signal target in the current timeframe session.
        defaults_provider: Runtime defaults provider for compute grids and signal defaults.
        signal_rules_engine: Startup-validated signal rules engine.
        indicator_compute: Indicator compute port used for primary/dependency tensors.
        indicator_grid_builder: Grid builder used for deterministic variant ordering.
    Returns:
        _SignalArtifactMaterializationResultV2: Typed strict signal-manifest result plus the
            deterministic count of rebuilt target-timeframe bars.
    Assumptions:
        Signal rows reuse v1 variant-key semantics and the fixed `[variant, time]` matrix layout.
    Raises:
        ValueError: If defaults, tensor shapes, value sets, or dependency alignment drift.
        OSError: If writing the signal matrix or manifest fails.
    Side Effects:
        Writes signal files under the slot root while reusing the already loaded timeframe-session
        inputs.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    if price_manifest.timeframe != signal_target.timeframe:
        raise ValueError(
            "signal export requires timeframe-local prices aligned to the current target; got "
            f"price_manifest.timeframe={price_manifest.timeframe!r} and "
            f"signal_target.timeframe={signal_target.timeframe!r}"
        )
    if session_price_arrays is None:
        raise ValueError(
            "signal export requires session-owned price arrays for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}"
        )
    price_arrays = session_price_arrays
    compute_grid = _resolve_signal_target_compute_grid_v2(
        defaults_provider=defaults_provider,
        indicator_id=signal_target.indicator_id,
    )
    materialized_grid = indicator_grid_builder.materialize_indicator(grid=compute_grid)
    signal_rows = _build_signal_variant_rows_v2(
        coordinates=coordinates,
        timeframe=signal_target.timeframe,
        materialized_grid=materialized_grid,
    )
    signal_variant_keys_sha256 = _variant_keys_sha256_v2(signal_rows=signal_rows)
    effective_tail_bars = _effective_signal_tail_bars_v2(
        timeframe=signal_target.timeframe,
        runtime_settings=runtime_settings,
    )
    rebuild_context_bars = _signal_rebuild_context_bars_v2(
        materialized_grid=materialized_grid,
        defaults_provider=defaults_provider,
        indicator_id=signal_target.indicator_id,
    )
    existing_signal_artifact = _load_existing_signal_artifact_v2(
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        existing_manifest=existing_manifest,
        signal_target=signal_target,
        expected_variant_keys_sha256=signal_variant_keys_sha256,
        expected_row_count=len(signal_rows),
    )
    default_inputs_source, signal_params_defaults = signal_rules_engine.resolved_defaults(
        indicator_id=signal_target.indicator_id
    )
    if existing_signal_artifact is not None:
        _validate_existing_signal_defaults_for_reuse_v2(
            existing_signal_artifact=existing_signal_artifact,
            signal_target=signal_target,
            signal_params_defaults=signal_params_defaults,
        )
    signal_tail_plan = _build_signal_tail_plan_v2(
        price_arrays=price_arrays,
        existing_signal_artifact=existing_signal_artifact,
        effective_tail_bars=effective_tail_bars,
        rebuild_context_bars=rebuild_context_bars,
    )
    signal_tail_price_arrays = _slice_canonical_price_arrays_v2(
        arrays=price_arrays,
        start_idx=signal_tail_plan.compute_start_idx,
        end_idx=int(price_arrays.open_time.shape[0]),
    )
    candles = _candle_arrays_from_price_arrays_v2(
        coordinates=coordinates,
        timeframe=signal_target.timeframe,
        arrays=signal_tail_price_arrays,
    )
    signal_paths = artifact_loader.resolve_signal_paths(
        coordinates,
        slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    signal_shape = (len(signal_rows), int(price_arrays.open_time.shape[0]))
    rule_spec = signal_rules_engine.rule_spec(indicator_id=signal_target.indicator_id)
    chunk_jobs = _plan_signal_chunk_jobs_v2(
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        timeline_bar_count=signal_shape[1],
        compute_bar_count=int(candles.close.shape[0]),
        variant_count=signal_shape[0],
        dependency_count=len(rule_spec.required_dependency_ids),
    )
    chunk_blocks = tuple(
        _build_signal_chunk_blocks_v2(
            materialized_grid=materialized_grid,
            chunk_job=chunk_job,
        )
        for chunk_job in chunk_jobs
    )
    _write_signal_matrix_in_chunks_v2(
        coordinates=coordinates,
        slot=slot,
        slot_generation=slot_generation,
        force_full_rebuild=request.force_full_rebuild,
        signal_target=signal_target,
        signal_paths=signal_paths,
        signal_shape=signal_shape,
        candles=candles,
        signal_worker_processes=runtime_settings.execution_policy.signal_worker_processes,
        chunk_jobs=chunk_jobs,
        chunk_blocks=chunk_blocks,
        signal_rows=signal_rows,
        signal_tail_plan=signal_tail_plan,
        existing_signal_artifact=existing_signal_artifact,
        indicator_compute=indicator_compute,
        rule_spec=rule_spec,
        default_inputs_source=default_inputs_source,
        signal_params_defaults=signal_params_defaults,
        max_signal_rows_per_artifact=runtime_settings.max_signal_rows_per_artifact,
    )
    signal_features_paths = artifact_loader.resolve_signal_features_paths(
        coordinates,
        slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    signal_features_build_result = _materialize_signal_features_artifact_v2(
        slot=slot,
        slot_root=slot_root,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        signal_shape=signal_shape,
        signal_paths=signal_paths,
        signal_features_paths=signal_features_paths,
    )
    signal_manifest = _build_signal_manifest_v2(
        coordinates=coordinates,
        slot=slot,
        slot_root=slot_root,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        signal_paths=signal_paths,
        signal_shape=signal_shape,
        timeline=_timeline_coverage_from_arrays_v2(arrays=price_arrays),
        price_manifest=price_manifest,
        signal_rows=signal_rows,
        signal_params_defaults=signal_params_defaults,
        signal_rules_engine=signal_rules_engine,
        effective_tail_bars=signal_tail_plan.effective_tail_bars,
        signal_features=signal_features_build_result.reference,
    )
    _write_yaml_atomically_v2(
        path=signal_paths.manifest,
        payload=_serialize_signal_manifest_v2(signal_manifest),
    )
    return _SignalArtifactMaterializationResultV2(
        manifest=signal_manifest,
        reused_prefix_bars=signal_tail_plan.reused_prefix_bars,
        rewritten_tail_bars=signal_shape[1] - signal_tail_plan.reused_prefix_bars,
        completed_chunks_total=len(chunk_jobs),
    )


def _materialize_signal_features_artifact_v2(
    *,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_shape: tuple[int, int],
    signal_paths: ArtifactSignalPathsV2,
    signal_features_paths: ArtifactSignalFeaturesPathsV2,
) -> _SignalFeaturesArtifactBuildResultV2:
    """
    Materialize one additive signal-feature family from the already written signal matrix.

    Args:
        slot: Inactive slot literal receiving the feature artifact.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying root identity and timestamps.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        runtime_settings: Strict runtime settings contributing config identity to provenance.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        signal_shape: Final signal matrix shape used as the strict row/timeline contract.
        signal_paths: Fixed signal family paths whose written matrix becomes the feature source.
        signal_features_paths: Fixed signal-feature family paths under the inactive slot.
    Returns:
        _SignalFeaturesArtifactBuildResultV2: Typed feature manifest plus the reference written
            back into the owning signal manifest.
    Assumptions:
        Feature derivation is intentionally row-local, deterministic, and depends only on the
        final signal rows plus their timeline length.
    Raises:
        ValueError: If the written signal matrix shape drifts from the expected contract.
        OSError: If writing the feature array or manifest fails.
    Side Effects:
        Memory-maps the final signal matrix, writes `features.f32.npy`, and writes the additive
        feature `manifest.yaml`.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    signal_matrix = np.load(signal_paths.signals, mmap_mode="r", allow_pickle=False)
    actual_signal_shape = tuple(int(value) for value in signal_matrix.shape)
    if actual_signal_shape != signal_shape:
        raise ValueError(
            "written signal matrix shape must match the expected strict contract; got "
            f"{actual_signal_shape!r}, expected {signal_shape!r}"
        )
    feature_matrix = _build_signal_features_matrix_v2(signal_matrix=signal_matrix)
    _write_npy_atomically_v2(path=signal_features_paths.features, array=feature_matrix)
    feature_manifest = _build_signal_features_manifest_v2(
        slot=slot,
        slot_root=slot_root,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        signal_shape=signal_shape,
        signal_paths=signal_paths,
        signal_features_paths=signal_features_paths,
    )
    _write_yaml_atomically_v2(
        path=signal_features_paths.manifest,
        payload=_serialize_signal_features_manifest_v2(feature_manifest),
    )
    reference = ArtifactSignalFeaturesReferenceV2(
        manifest_path=_slot_relative_path_v2(
            slot_root=slot_root,
            absolute_path=signal_features_paths.manifest,
        ),
        manifest_sha256=_file_sha256_hex_v2(signal_features_paths.manifest),
    )
    return _SignalFeaturesArtifactBuildResultV2(
        manifest=feature_manifest,
        reference=reference,
    )


def _build_signal_features_matrix_v2(*, signal_matrix: np.ndarray) -> np.ndarray:
    """
    Derive the fixed warm-cache signal features from one final `[variant, time]` signal matrix.

    Args:
        signal_matrix: Final strict signal matrix whose values must already satisfy `{-1, 0, 1}`.
    Returns:
        np.ndarray: Contiguous `float32` feature matrix with shape `[variant, feature]`.
    Assumptions:
        Feature derivation stays intentionally small, explicit, and row-local for Milestone C1.
    Raises:
        ValueError: If the matrix is not two-dimensional or has an empty feature axis.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if signal_matrix.ndim != 2:
        raise ValueError(
            f"signal feature source matrix must be 2D; got ndim={signal_matrix.ndim!r}"
        )
    variant_count = int(signal_matrix.shape[0])
    timeline_bar_count = int(signal_matrix.shape[1])
    if variant_count <= 0 or timeline_bar_count <= 0:
        raise ValueError(
            "signal feature source matrix must have positive variant and timeline dimensions"
        )
    nonzero_count = np.count_nonzero(signal_matrix != 0, axis=1).astype(np.float32, copy=False)
    long_count = np.count_nonzero(signal_matrix > 0, axis=1).astype(np.float32, copy=False)
    short_count = np.count_nonzero(signal_matrix < 0, axis=1).astype(np.float32, copy=False)
    activity_ratio = np.ascontiguousarray(
        nonzero_count / np.float32(timeline_bar_count),
        dtype=np.float32,
    )
    direction_balance = np.zeros(variant_count, dtype=np.float32)
    np.divide(
        long_count - short_count,
        nonzero_count,
        out=direction_balance,
        where=nonzero_count > 0.0,
    )
    if timeline_bar_count < 2:
        transition_count = np.zeros(variant_count, dtype=np.float32)
    else:
        transition_count = np.count_nonzero(
            signal_matrix[:, 1:] != signal_matrix[:, :-1],
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


def _plan_signal_chunk_jobs_v2(
    *,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    timeline_bar_count: int,
    compute_bar_count: int,
    variant_count: int,
    dependency_count: int,
) -> tuple[ArtifactSignalChunkJobV2, ...]:
    """
    Build deterministic chunk jobs for one signal artifact target.

    Args:
        runtime_settings: Strict runtime settings carrying execution-policy limits.
        signal_target: Explicit `(timeframe, indicator_id)` signal target.
        timeline_bar_count: Final target timeframe bar count for `signals/<tf>/<indicator_id>`.
        compute_bar_count: Actual candle count of the bounded compute window.
        variant_count: Deterministic total variant row count for the target.
        dependency_count: Number of dependency tensors required by the signal rule family.
    Returns:
        tuple[ArtifactSignalChunkJobV2, ...]: Ordered non-overlapping row-range jobs.
    Assumptions:
        Planner sizing is conservative and may underutilize memory, but must never exceed the
        configured worker budget.
    Raises:
        ValueError: If planner inputs or budgets are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_chunk_planner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    planner = DeterministicSignalChunkPlannerV2()
    return planner.plan(
        request=ArtifactSignalChunkPlanningRequestV2(
            indicator_id=signal_target.indicator_id,
            timeframe=signal_target.timeframe,
            timeline_bar_count=timeline_bar_count,
            variant_count=variant_count,
            estimated_bytes_per_row=_estimate_signal_chunk_bytes_per_row_v2(
                indicator_id=signal_target.indicator_id,
                timeline_bar_count=timeline_bar_count,
                compute_bar_count=compute_bar_count,
                dependency_count=dependency_count,
            ),
            worker_memory_budget_bytes=(
                runtime_settings.execution_policy.signal_worker_memory_budget_bytes
            ),
            signal_chunk_rows_min=runtime_settings.execution_policy.signal_chunk_rows_min,
            signal_chunk_rows_max=runtime_settings.execution_policy.signal_chunk_rows_max,
        )
    )


def _estimate_signal_chunk_bytes_per_row_v2(
    *,
    indicator_id: str,
    timeline_bar_count: int,
    compute_bar_count: int,
    dependency_count: int,
) -> int:
    """
    Estimate conservative per-row memory for ChunkPlanner sizing.

    Args:
        indicator_id: Indicator identifier currently being materialized.
        timeline_bar_count: Final target timeframe bar count written to the artifact.
        compute_bar_count: Candle count of the bounded compute window.
        dependency_count: Number of dependency tensors required by the rule family.
    Returns:
        int: Conservative bytes-per-row estimate for one chunk worker.
    Assumptions:
        Estimate includes primary/dependency float32 rows plus compact signal/output ownership and
        intentionally overestimates MA-family sources to stay below the worker budget.
    Raises:
        ValueError: If the time dimensions or dependency count are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_chunk_planner_v2.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
    """
    if timeline_bar_count <= 0:
        raise ValueError(
            f"signal timeline_bar_count must be > 0; got {timeline_bar_count!r}"
        )
    if compute_bar_count <= 0:
        raise ValueError(f"signal compute_bar_count must be > 0; got {compute_bar_count!r}")
    if dependency_count < 0:
        raise ValueError(f"signal dependency_count must be >= 0; got {dependency_count!r}")
    compute_bytes = compute_bar_count * (4 * (1 + dependency_count) + 1)
    output_bytes = timeline_bar_count
    ma_family_workspace_bytes = compute_bar_count * 4 if indicator_id.startswith("ma.") else 0
    return max(1, compute_bytes + output_bytes + ma_family_workspace_bytes)


def _build_signal_chunk_blocks_v2(
    *,
    materialized_grid: Any,
    chunk_job: ArtifactSignalChunkJobV2,
) -> tuple[_SignalChunkGridBlockV2, ...]:
    """
    Decompose one contiguous row slice into deterministic explicit subgrids.

    Args:
        materialized_grid: Materialized grid defining canonical variant ordering.
        chunk_job: Planned contiguous row range owned by the chunk.
    Returns:
        tuple[_SignalChunkGridBlockV2, ...]: Ordered subgrids covering the chunk rows exactly once.
    Assumptions:
        Contiguous row ranges may cross cartesian-product block boundaries and therefore need
        decomposition into smaller explicit grids.
    Raises:
        ValueError: If one emitted block drifts from the requested row range.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    axis_names = tuple(str(axis.name) for axis in materialized_grid.axes)
    axis_values = tuple(tuple(axis.values) for axis in materialized_grid.axes)
    if len(axis_names) == 0:
        return (
            _SignalChunkGridBlockV2(
                row_start_inclusive=chunk_job.row_start_inclusive,
                row_end_exclusive=chunk_job.row_end_exclusive,
                source_values=None,
                param_values_by_name=(),
            ),
        )
    suffix_products = _axis_suffix_products_v2(axis_values=axis_values)
    blocks: list[_SignalChunkGridBlockV2] = []
    _append_signal_chunk_blocks_recursive_v2(
        axis_names=axis_names,
        axis_values=axis_values,
        suffix_products=suffix_products,
        axis_index=0,
        base_row_offset=0,
        local_start=chunk_job.row_start_inclusive,
        local_end=chunk_job.row_end_exclusive,
        prefix_indices=(),
        blocks=blocks,
    )
    if not blocks:
        raise ValueError(
            "signal chunk block decomposition produced no blocks for "
            f"{chunk_job.indicator_id}:{chunk_job.timeframe} chunk_index={chunk_job.chunk_index}"
        )
    if blocks[0].row_start_inclusive != chunk_job.row_start_inclusive:
        raise ValueError("signal chunk blocks must start at the requested row boundary")
    if blocks[-1].row_end_exclusive != chunk_job.row_end_exclusive:
        raise ValueError("signal chunk blocks must end at the requested row boundary")
    return tuple(blocks)


def _axis_suffix_products_v2(
    *,
    axis_values: tuple[tuple[Any, ...], ...],
) -> tuple[int, ...]:
    """
    Compute mixed-radix suffix products for deterministic variant-index decomposition.

    Args:
        axis_values: Materialized axis value tuples in canonical order.
    Returns:
        tuple[int, ...]: Row span covered by each axis position.
    Assumptions:
        Variant flattening order is the same cartesian-product order used by `GridBuilder`.
    Raises:
        ValueError: If one axis is empty.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    running_product = 1
    suffix_products = [1] * len(axis_values)
    for axis_index in range(len(axis_values) - 1, -1, -1):
        values = axis_values[axis_index]
        if len(values) == 0:
            raise ValueError(f"signal chunk axis {axis_index} must be non-empty")
        suffix_products[axis_index] = running_product
        running_product *= len(values)
    return tuple(suffix_products)


def _append_signal_chunk_blocks_recursive_v2(
    *,
    axis_names: tuple[str, ...],
    axis_values: tuple[tuple[Any, ...], ...],
    suffix_products: tuple[int, ...],
    axis_index: int,
    base_row_offset: int,
    local_start: int,
    local_end: int,
    prefix_indices: tuple[int, ...],
    blocks: list[_SignalChunkGridBlockV2],
) -> None:
    """
    Recursively decompose one row interval into maximal explicit cartesian-product blocks.

    Args:
        axis_names: Canonical axis names from the materialized grid.
        axis_values: Canonical axis values from the materialized grid.
        suffix_products: Mixed-radix suffix products for variant flattening.
        axis_index: Current recursion axis.
        base_row_offset: Absolute global row offset of the current recursion subtree.
        local_start: Inclusive row offset within the current recursion subtree.
        local_end: Exclusive row offset within the current recursion subtree.
        prefix_indices: Fixed axis-value indexes chosen by parent recursion frames.
        blocks: Mutable result accumulator receiving ordered chunk blocks.
    Returns:
        None.
    Assumptions:
        Each full-covered subtree can be emitted as one explicit subgrid while partial overlaps
        recurse to the next axis.
    Raises:
        ValueError: If one overlap interval is inconsistent.
    Side Effects:
        Appends deterministic block snapshots into `blocks`.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if local_end <= local_start:
        raise ValueError(
            f"signal chunk recursion requires local_end > local_start; got {local_end!r} and "
            f"{local_start!r}"
        )
    child_span = suffix_products[axis_index]
    start_child = local_start // child_span
    end_child = (local_end - 1) // child_span
    for child_index in range(start_child, end_child + 1):
        child_range_start = child_index * child_span
        child_range_end = child_range_start + child_span
        overlap_start = max(local_start, child_range_start)
        overlap_end = min(local_end, child_range_end)
        if overlap_end <= overlap_start:
            raise ValueError("signal chunk recursion overlap must stay positive")
        child_prefix = (*prefix_indices, child_index)
        if overlap_start == child_range_start and overlap_end == child_range_end:
            blocks.append(
                _build_signal_chunk_grid_block_v2(
                    axis_names=axis_names,
                    axis_values=axis_values,
                    prefix_indices=child_prefix,
                    row_start_inclusive=base_row_offset + overlap_start,
                    row_end_exclusive=base_row_offset + overlap_end,
                )
            )
            continue
        if axis_index + 1 >= len(axis_names):
            blocks.append(
                _build_signal_chunk_grid_block_v2(
                    axis_names=axis_names,
                    axis_values=axis_values,
                    prefix_indices=child_prefix,
                    row_start_inclusive=base_row_offset + overlap_start,
                    row_end_exclusive=base_row_offset + overlap_end,
                )
            )
            continue
        _append_signal_chunk_blocks_recursive_v2(
            axis_names=axis_names,
            axis_values=axis_values,
            suffix_products=suffix_products,
            axis_index=axis_index + 1,
            base_row_offset=base_row_offset + child_range_start,
            local_start=overlap_start - child_range_start,
            local_end=overlap_end - child_range_start,
            prefix_indices=child_prefix,
            blocks=blocks,
        )


def _build_signal_chunk_grid_block_v2(
    *,
    axis_names: tuple[str, ...],
    axis_values: tuple[tuple[Any, ...], ...],
    prefix_indices: tuple[int, ...],
    row_start_inclusive: int,
    row_end_exclusive: int,
) -> _SignalChunkGridBlockV2:
    """
    Build one explicit chunk subgrid snapshot from fixed-prefix axis indexes.

    Args:
        axis_names: Canonical axis names from the materialized grid.
        axis_values: Canonical axis values from the materialized grid.
        prefix_indices: Fixed axis-value indexes for already-covered leading axes.
        row_start_inclusive: Inclusive global row index covered by the block.
        row_end_exclusive: Exclusive global row index covered by the block.
    Returns:
        _SignalChunkGridBlockV2: Snapshot describing one explicit subgrid.
    Assumptions:
        Leading axes addressed by `prefix_indices` are fixed to one value while the remaining
        suffix axes keep their full canonical value order.
    Raises:
        ValueError: If the row range is invalid or one prefix index is out of bounds.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if row_end_exclusive <= row_start_inclusive:
        raise ValueError(
            "signal chunk grid block requires positive row range; got "
            f"[{row_start_inclusive!r}, {row_end_exclusive!r})"
        )
    source_values: tuple[str, ...] | None = None
    param_values: list[tuple[str, tuple[int | float | str, ...]]] = []
    for axis_index, axis_name in enumerate(axis_names):
        values = axis_values[axis_index]
        if axis_index < len(prefix_indices):
            fixed_index = prefix_indices[axis_index]
            if fixed_index < 0 or fixed_index >= len(values):
                raise ValueError(
                    f"signal chunk fixed axis index out of bounds for {axis_name!r}: "
                    f"{fixed_index!r}"
                )
            selected_values = (values[fixed_index],)
        else:
            selected_values = tuple(values)
        if axis_name == "source":
            source_values = tuple(str(value) for value in selected_values)
            continue
        param_values.append(
            (
                axis_name,
                tuple(cast(tuple[int | float | str, ...], selected_values)),
            )
        )
    return _SignalChunkGridBlockV2(
        row_start_inclusive=row_start_inclusive,
        row_end_exclusive=row_end_exclusive,
        source_values=source_values,
        param_values_by_name=tuple(param_values),
    )


def _write_signal_matrix_in_chunks_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_generation: int,
    force_full_rebuild: bool,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_paths: ArtifactSignalPathsV2,
    signal_shape: tuple[int, int],
    candles: CandleArrays,
    signal_worker_processes: int,
    chunk_jobs: tuple[ArtifactSignalChunkJobV2, ...],
    chunk_blocks: tuple[tuple[_SignalChunkGridBlockV2, ...], ...],
    signal_rows: tuple[_SignalVariantRowV2, ...],
    signal_tail_plan: _SignalArtifactTailPlanV2,
    existing_signal_artifact: _ExistingSignalArtifactV2 | None,
    indicator_compute: IndicatorCompute,
    rule_spec: SignalRuleSpecV2,
    default_inputs_source: str | None,
    signal_params_defaults: Mapping[str, Any],
    max_signal_rows_per_artifact: int,
) -> None:
    """
    Materialize one signal matrix through deterministic chunk-local `np.memmap` row writes.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the signal artifact.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        force_full_rebuild: Whether this run rebuilds from scratch instead of reusing prefixes.
        signal_target: Explicit `(timeframe, indicator_id)` signal target.
        signal_paths: Deterministic inactive-slot output paths for this target.
        signal_shape: Final `[variant, time]` matrix shape.
        candles: Bounded compute-window candles for chunk-local indicator compute.
        signal_worker_processes: Configured upper bound of concurrent chunk workers.
        chunk_jobs: Deterministic ordered chunk jobs for the target.
        chunk_blocks: Explicit subgrids covering each chunk job in canonical order.
        signal_rows: Ordered row descriptors for the full target matrix.
        signal_tail_plan: Prefix-reuse/tail-rebuild plan for the current target.
        existing_signal_artifact: Existing reusable signal artifact, when present.
        indicator_compute: Compute adapter used for primary/dependency tensors.
        rule_spec: Signal rule specification for the target indicator.
        default_inputs_source: Default `inputs.source` literal resolved once per indicator.
        signal_params_defaults: Default-only `signals.v1.params` mapping.
        max_signal_rows_per_artifact: Strict compute guard forwarded into chunk-local requests.
    Returns:
        None.
    Assumptions:
        Chunk ownership is always a non-overlapping row slice, so direct `np.memmap` writes stay
        deterministic even when chunks complete out of order.
    Raises:
        ValueError: If one chunk writes an inconsistent shape or signal defaults drift.
        OSError: If temp-file creation, memmap writes, or atomic replace fail.
    Side Effects:
        Creates a temp `.npy`, writes row slices through `np.memmap`, and atomically replaces the
        final `signals/<tf>/<indicator_id>/signals.i8.npy` path.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/signal_chunk_planner_v2.py
    """
    signal_paths.signals.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{signal_paths.signals.name}.",
        suffix=".tmp",
        dir=signal_paths.signals.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    output_memmap: np.memmap = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.int8,
        shape=signal_shape,
    )
    output_memmap.flush()
    del output_memmap

    completed_chunks_total = 0
    worker_count = max(1, min(len(chunk_jobs), signal_worker_processes))
    compute_worker_factory = _resolve_indicator_compute_worker_factory_v2(
        indicator_compute=indicator_compute
    )
    existing_signals_path = (
        None if existing_signal_artifact is None else existing_signal_artifact.signals_path
    )
    try:
        if compute_worker_factory is None or worker_count == 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_meta = {}
                for chunk_job, chunk_job_blocks in zip(chunk_jobs, chunk_blocks, strict=True):
                    _log_signal_chunk_progress_v2(
                        event="artifact_precompute_chunk_started",
                        coordinates=coordinates,
                        slot=slot,
                        slot_generation=slot_generation,
                        force_full_rebuild=force_full_rebuild,
                        current_timeframe=signal_target.timeframe,
                        current_indicator_id=signal_target.indicator_id,
                        chunk_job=chunk_job,
                    )
                    future = executor.submit(
                        _execute_signal_chunk_job_v2,
                        indicator_compute=indicator_compute,
                        candles=candles,
                        chunk_job=chunk_job,
                        chunk_blocks=chunk_job_blocks,
                        signal_rows=signal_rows,
                        rule_spec=rule_spec,
                        default_inputs_source=default_inputs_source,
                        signal_params_defaults=signal_params_defaults,
                        output_path=temp_path,
                        output_shape=signal_shape,
                        existing_signals_path=existing_signals_path,
                        reused_prefix_bars=signal_tail_plan.reused_prefix_bars,
                        trim_prefix_bars=signal_tail_plan.trim_prefix_bars,
                        max_signal_rows_per_artifact=max_signal_rows_per_artifact,
                    )
                    future_to_meta[future] = (chunk_job, time.perf_counter())
                for future in as_completed(future_to_meta):
                    chunk_job, started_at = future_to_meta[future]
                    chunk_result = future.result()
                    if dict(chunk_result.signal_params_defaults) != dict(signal_params_defaults):
                        raise ValueError(
                            "signal chunk defaults drift detected for "
                            f"{signal_target.timeframe}:{signal_target.indicator_id}"
                        )
                    completed_chunks_total += 1
                    _log_signal_chunk_progress_v2(
                        event="artifact_precompute_chunk_finished",
                        coordinates=coordinates,
                        slot=slot,
                        slot_generation=slot_generation,
                        force_full_rebuild=force_full_rebuild,
                        current_timeframe=signal_target.timeframe,
                        current_indicator_id=signal_target.indicator_id,
                        chunk_job=chunk_job,
                        completed_chunks_total=completed_chunks_total,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
        else:
            worker_class, worker_snapshot = compute_worker_factory
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
                initializer=_initialize_signal_chunk_worker_v2,
                initargs=(
                    _SignalChunkWorkerBootstrapV2(
                        indicator_compute_worker_class=worker_class,
                        indicator_compute_worker_snapshot=worker_snapshot,
                        candles=candles,
                    ),
                ),
            ) as executor:
                future_to_meta = {}
                for chunk_job, chunk_job_blocks in zip(chunk_jobs, chunk_blocks, strict=True):
                    _log_signal_chunk_progress_v2(
                        event="artifact_precompute_chunk_started",
                        coordinates=coordinates,
                        slot=slot,
                        slot_generation=slot_generation,
                        force_full_rebuild=force_full_rebuild,
                        current_timeframe=signal_target.timeframe,
                        current_indicator_id=signal_target.indicator_id,
                        chunk_job=chunk_job,
                    )
                    future = executor.submit(
                        _execute_signal_chunk_job_v2,
                        indicator_compute=None,
                        candles=None,
                        chunk_job=chunk_job,
                        chunk_blocks=chunk_job_blocks,
                        signal_rows=signal_rows,
                        rule_spec=rule_spec,
                        default_inputs_source=default_inputs_source,
                        signal_params_defaults=signal_params_defaults,
                        output_path=temp_path,
                        output_shape=signal_shape,
                        existing_signals_path=existing_signals_path,
                        reused_prefix_bars=signal_tail_plan.reused_prefix_bars,
                        trim_prefix_bars=signal_tail_plan.trim_prefix_bars,
                        max_signal_rows_per_artifact=max_signal_rows_per_artifact,
                    )
                    future_to_meta[future] = (chunk_job, time.perf_counter())
                for future in as_completed(future_to_meta):
                    chunk_job, started_at = future_to_meta[future]
                    chunk_result = future.result()
                    if dict(chunk_result.signal_params_defaults) != dict(signal_params_defaults):
                        raise ValueError(
                            "signal chunk defaults drift detected for "
                            f"{signal_target.timeframe}:{signal_target.indicator_id}"
                        )
                    completed_chunks_total += 1
                    _log_signal_chunk_progress_v2(
                        event="artifact_precompute_chunk_finished",
                        coordinates=coordinates,
                        slot=slot,
                        slot_generation=slot_generation,
                        force_full_rebuild=force_full_rebuild,
                        current_timeframe=signal_target.timeframe,
                        current_indicator_id=signal_target.indicator_id,
                        chunk_job=chunk_job,
                        completed_chunks_total=completed_chunks_total,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
        if completed_chunks_total != len(chunk_jobs):
            raise ValueError(
                "signal chunk execution completed an unexpected number of chunks; got "
                f"{completed_chunks_total!r}, expected {len(chunk_jobs)!r}"
            )
        _fsync_path_v2(path=temp_path)
        os.replace(temp_path, signal_paths.signals)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _resolve_indicator_compute_worker_factory_v2(
    *,
    indicator_compute: IndicatorCompute,
) -> tuple[type[Any], Mapping[str, Any]] | None:
    """
    Resolve an optional spawned-worker factory for signal chunk compute.

    Args:
        indicator_compute: Live compute adapter owned by the runner.
    Returns:
        tuple[type[Any], Mapping[str, Any]] | None: Rehydration class plus snapshot when the
            adapter supports spawned-worker reconstruction, otherwise `None`.
    Assumptions:
        Test doubles may remain thread-only, while production Numba compute can opt into real
        process workers through an explicit snapshot contract.
    Raises:
        ValueError: If the adapter advertises an incomplete snapshot interface.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    snapshot_method = getattr(indicator_compute, "to_signal_chunk_worker_snapshot_v2", None)
    factory_method = getattr(
        indicator_compute.__class__,
        "from_signal_chunk_worker_snapshot_v2",
        None,
    )
    if snapshot_method is None and factory_method is None:
        return None
    if snapshot_method is None or factory_method is None:
        raise ValueError(
            "indicator_compute must expose both to_signal_chunk_worker_snapshot_v2 and "
            "from_signal_chunk_worker_snapshot_v2"
        )
    snapshot_capable = cast(_SignalChunkWorkerSnapshotCapableV2, indicator_compute)
    return indicator_compute.__class__, snapshot_capable.to_signal_chunk_worker_snapshot_v2()


_SIGNAL_CHUNK_WORKER_STATE_V2: _SignalChunkWorkerStateV2 | None = None


def _initialize_signal_chunk_worker_v2(
    worker_bootstrap: _SignalChunkWorkerBootstrapV2,
) -> None:
    """
    Bootstrap one spawned chunk worker with session-local compute and candle inputs.

    Args:
        worker_bootstrap: Immutable worker bootstrap payload created once per process.
    Returns:
        None.
    Assumptions:
        macOS `spawn` requires explicit process-local reconstruction instead of sharing the live
        parent process object graph.
    Raises:
        ValueError: If worker bootstrap cannot rebuild the compute adapter.
    Side Effects:
        Stores worker-local state in a private module global for later chunk jobs.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    global _SIGNAL_CHUNK_WORKER_STATE_V2
    _SIGNAL_CHUNK_WORKER_STATE_V2 = _SignalChunkWorkerStateV2(
        indicator_compute=cast(
            IndicatorCompute,
            worker_bootstrap.indicator_compute_worker_class.from_signal_chunk_worker_snapshot_v2(
                snapshot=worker_bootstrap.indicator_compute_worker_snapshot
            ),
        ),
        candles=worker_bootstrap.candles,
    )


def _require_signal_chunk_worker_state_v2() -> _SignalChunkWorkerStateV2:
    """
    Return the initialized spawned-worker state for one chunk execution.

    Args:
        None.
    Returns:
        _SignalChunkWorkerStateV2: Process-local chunk worker state.
    Assumptions:
        Spawned workers must run `_initialize_signal_chunk_worker_v2(...)` before accepting jobs.
    Raises:
        ValueError: If the worker was used before bootstrap finished.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if _SIGNAL_CHUNK_WORKER_STATE_V2 is None:
        raise ValueError("signal chunk worker state is not initialized")
    return _SIGNAL_CHUNK_WORKER_STATE_V2


def _execute_signal_chunk_job_v2(
    *,
    indicator_compute: IndicatorCompute | None,
    candles: CandleArrays | None,
    chunk_job: ArtifactSignalChunkJobV2,
    chunk_blocks: tuple[_SignalChunkGridBlockV2, ...],
    signal_rows: tuple[_SignalVariantRowV2, ...],
    rule_spec: SignalRuleSpecV2,
    default_inputs_source: str | None,
    signal_params_defaults: Mapping[str, Any],
    output_path: Path,
    output_shape: tuple[int, int],
    existing_signals_path: Path | None,
    reused_prefix_bars: int,
    trim_prefix_bars: int,
    max_signal_rows_per_artifact: int,
) -> _SignalChunkWorkerResultV2:
    """
    Compute one chunk and write its owned row slice directly into the final memmap file.

    Args:
        indicator_compute: Live in-process compute adapter for thread-based execution.
        candles: Bounded compute-window candles shared by every block of the chunk.
        chunk_job: Deterministic row-range owner for this chunk.
        chunk_blocks: Explicit subgrids covering `chunk_job` in canonical order.
        signal_rows: Full ordered signal row catalog for the target matrix.
        rule_spec: Signal rule specification for the target indicator.
        default_inputs_source: Default `inputs.source` literal resolved once per indicator.
        signal_params_defaults: Default-only `signals.v1.params` mapping.
        output_path: Temp `.npy` path already preallocated to the final matrix shape.
        output_shape: Final `[variant, time]` matrix shape.
        existing_signals_path: Existing reusable signal file used for prefix copy, when present.
        reused_prefix_bars: Number of leading time-axis bars copied unchanged from the old file.
        trim_prefix_bars: Number of computed warmup bars trimmed from chunk-local outputs.
        max_signal_rows_per_artifact: Strict compute guard forwarded to chunk-local requests.
    Returns:
        _SignalChunkWorkerResultV2: Completed chunk summary for progress aggregation.
    Assumptions:
        `chunk_blocks` cover only the rows owned by `chunk_job` and preserve their canonical order.
    Raises:
        ValueError: If a block compute drifts in row count, timeline length, or final slice width.
        OSError: If opening or flushing the writable memmap fails.
    Side Effects:
        Writes this chunk's non-overlapping row slice inside the temp signal `.npy` file.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
    """
    if indicator_compute is None or candles is None:
        worker_state = _require_signal_chunk_worker_state_v2()
        if indicator_compute is None:
            indicator_compute = worker_state.indicator_compute
        if candles is None:
            candles = worker_state.candles
    signal_memmap: np.memmap = cast(
        np.memmap,
        np.load(output_path, mmap_mode="r+", allow_pickle=False),
    )
    existing_memmap: np.memmap | None = None
    try:
        if existing_signals_path is not None and reused_prefix_bars > 0:
            existing_memmap = cast(
                np.memmap,
                np.load(existing_signals_path, mmap_mode="r", allow_pickle=False),
            )
            signal_memmap[
                chunk_job.row_start_inclusive : chunk_job.row_end_exclusive,
                :reused_prefix_bars,
            ] = existing_memmap[
                chunk_job.row_start_inclusive : chunk_job.row_end_exclusive,
                :reused_prefix_bars,
            ]
        expected_tail_bars = output_shape[1] - reused_prefix_bars
        for chunk_block in chunk_blocks:
            block_grid = _build_grid_from_signal_chunk_block_v2(
                indicator_id=rule_spec.indicator_id,
                chunk_block=chunk_block,
            )
            expected_variants = (
                chunk_block.row_end_exclusive - chunk_block.row_start_inclusive
            )
            primary_tensor = indicator_compute.compute(
                ComputeRequest(
                    candles=candles,
                    grid=block_grid,
                    max_variants_guard=max_signal_rows_per_artifact,
                )
            )
            if primary_tensor.meta.variants != expected_variants:
                raise ValueError(
                    "signal chunk primary tensor variants drift detected for "
                    f"{rule_spec.indicator_id!r}; got {primary_tensor.meta.variants!r}, "
                    f"expected {expected_variants!r}"
                )
            dependency_tensors = _compute_signal_dependency_tensors_v2(
                candles=candles,
                compute_grid=block_grid,
                required_dependency_ids=rule_spec.required_dependency_ids,
                indicator_id=rule_spec.indicator_id,
                indicator_compute=indicator_compute,
                max_signal_rows_per_artifact=max_signal_rows_per_artifact,
                expected_variants=primary_tensor.meta.variants,
                expected_t=primary_tensor.meta.t,
            )
            rebuilt_compute_window = _evaluate_signal_matrix_v2(
                candles=candles,
                indicator_id=rule_spec.indicator_id,
                primary_tensor=primary_tensor,
                dependency_tensors=dependency_tensors,
                signal_rows=signal_rows[
                    chunk_block.row_start_inclusive : chunk_block.row_end_exclusive
                ],
                rule_spec=rule_spec,
                default_inputs_source=default_inputs_source,
                signal_params_defaults=signal_params_defaults,
            )
            rebuilt_tail = rebuilt_compute_window[:, trim_prefix_bars:]
            if rebuilt_tail.shape[1] != expected_tail_bars:
                raise ValueError(
                    "signal chunk rebuilt tail width drift detected for "
                    f"{rule_spec.indicator_id!r}; got {rebuilt_tail.shape[1]!r}, "
                    f"expected {expected_tail_bars!r}"
                )
            signal_memmap[
                chunk_block.row_start_inclusive : chunk_block.row_end_exclusive,
                reused_prefix_bars:,
            ] = rebuilt_tail
        signal_memmap.flush()
    finally:
        del signal_memmap
        if existing_memmap is not None:
            del existing_memmap
    return _SignalChunkWorkerResultV2(
        chunk_job=chunk_job,
        signal_params_defaults=dict(signal_params_defaults),
    )


def _build_grid_from_signal_chunk_block_v2(
    *,
    indicator_id: str,
    chunk_block: _SignalChunkGridBlockV2,
) -> GridSpec:
    """
    Rebuild one explicit compute grid from a picklable chunk-block snapshot.

    Args:
        indicator_id: Indicator identifier to assign to the rebuilt grid.
        chunk_block: Picklable chunk-block snapshot describing explicit axis values.
    Returns:
        GridSpec: Variant-major explicit grid covering only the block rows.
    Assumptions:
        Chunk blocks already preserve canonical per-axis value order.
    Raises:
        ValueError: If one explicit axis specification is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return GridSpec(
        indicator_id=IndicatorId(indicator_id),
        params={
            axis_name: ExplicitValuesSpec(name=axis_name, values=values)
            for axis_name, values in chunk_block.param_values_by_name
        },
        source=(
            None
            if chunk_block.source_values is None
            else ExplicitValuesSpec(name="source", values=chunk_block.source_values)
        ),
        layout_preference=Layout.VARIANT_MAJOR,
    )


def _fsync_path_v2(*, path: Path) -> None:
    """
    Flush one fully written temp file to stable storage before atomic replace.

    Args:
        path: Temp file path to fsync.
    Returns:
        None.
    Assumptions:
        Temp files are created in the destination directory so `os.replace` remains atomic.
    Raises:
        OSError: If opening or fsyncing the file fails.
    Side Effects:
        Opens the file in binary mode and calls `os.fsync`.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _resolve_signal_target_compute_grid_v2(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
    indicator_id: str,
) -> GridSpec:
    """
    Resolve a variant-major compute grid for one signal target with explicit zero-axis fallback.

    Args:
        defaults_provider: Startup-validated defaults provider used by signal artifact precompute.
        indicator_id: Signal target indicator identifier.
    Returns:
        GridSpec: Deterministic variant-major compute grid for the target indicator.
    Assumptions:
        Only the four approved zero-axis signal targets may omit YAML compute defaults.
    Raises:
        ValueError: If compute defaults are missing for a non-approved or non-zero-axis signal
            target.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
    """
    defaults_grid = defaults_provider.compute_defaults(indicator_id=indicator_id)
    if defaults_grid is not None:
        return _grid_with_layout_v2(
            grid=defaults_grid,
            indicator_id=indicator_id,
            layout=Layout.VARIANT_MAJOR,
        )
    return _zero_axis_signal_target_grid_v2(indicator_id=indicator_id)


def _zero_axis_signal_target_grid_v2(*, indicator_id: str) -> GridSpec:
    """
    Build the deterministic single-variant `GridSpec` for approved zero-axis signal targets.

    Args:
        indicator_id: Signal target indicator identifier.
    Returns:
        GridSpec: Empty-axis `Layout.VARIANT_MAJOR` grid producing one deterministic variant row.
    Assumptions:
        Hard indicator definitions are the source of truth for zero-axis eligibility.
    Raises:
        ValueError: If the indicator is not one of the approved zero-axis signal targets or if its
            hard definition exposes axes.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/indicators/domain/definitions/structure.py
      - src/trading/contexts/indicators/domain/definitions/volatility.py
      - src/trading/contexts/indicators/domain/definitions/volume.py
    """
    normalized_indicator_id = indicator_id.strip().lower()
    if normalized_indicator_id not in _ZERO_AXIS_SIGNAL_TARGET_IDS_V2:
        raise ValueError(
            "signal target requires compute defaults for indicator_id "
            f"{normalized_indicator_id!r}"
        )
    hard_axes = _INDICATOR_AXES_BY_ID_V2.get(normalized_indicator_id)
    if hard_axes is None:
        raise ValueError(f"unknown hard definition for indicator_id {normalized_indicator_id!r}")
    if len(hard_axes) != 0:
        raise ValueError(
            "signal target requires compute defaults for indicator_id "
            f"{normalized_indicator_id!r}"
        )
    return GridSpec(
        indicator_id=IndicatorId(normalized_indicator_id),
        params={},
        source=None,
        layout_preference=Layout.VARIANT_MAJOR,
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    required_dependency_ids: tuple[str, ...],
    indicator_id: str,
    indicator_compute: IndicatorCompute,
    max_signal_rows_per_artifact: int,
    expected_variants: int,
    expected_t: int,
) -> Mapping[str, Any]:
    """
    Compute dependency tensors required by one signal rule family.

    Args:
        candles: Dense candle arrays aligned to the target signal timeframe.
        compute_grid: Primary indicator grid used for the target indicator.
        required_dependency_ids: Explicit dependency indicator ids required by the rule family.
        indicator_id: Primary indicator identifier.
        indicator_compute: Indicator compute port used for dependency tensors.
        max_signal_rows_per_artifact: Strict compute guard for dependency grids.
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
    """
    dependency_tensors: dict[str, Any] = {}
    for dependency_id in required_dependency_ids:
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
    rule_spec: SignalRuleSpecV2,
    default_inputs_source: str | None,
    signal_params_defaults: Mapping[str, Any],
) -> np.ndarray:
    """
    Evaluate compact `int8` signals for every row in the exported signal matrix.

    Args:
        candles: Dense candle arrays aligned to the target signal timeframe.
        indicator_id: Primary indicator identifier.
        primary_tensor: Computed primary indicator tensor.
        dependency_tensors: Dependency tensors keyed by indicator id.
        signal_rows: Ordered row descriptors matching tensor variant order.
        rule_spec: Explicit signal rule specification for the target indicator.
        default_inputs_source: Default `inputs.source` literal resolved once per indicator.
        signal_params_defaults: Default-only `signals.v1.params` mapping shared by every row.
    Returns:
        np.ndarray: Export-ready compact signal matrix for the chunk-local compute window.
    Assumptions:
        Every row uses the same default-only signal params already resolved by the caller.
    Raises:
        ValueError: If one evaluated series shape or value set violates the strict matrix
            contract.
    Side Effects:
        Allocates one contiguous `int8` matrix in memory.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    row_count = int(primary_tensor.meta.variants)
    time_count = int(primary_tensor.meta.t)
    signal_matrix = np.empty((row_count, time_count), dtype=np.int8)
    for row_index, signal_row in enumerate(signal_rows):
        dependency_outputs = {
            dependency_id: indicator_primary_output_series_from_tensor_v1(
                tensor=dependency_tensor,
                variant_index=row_index,
            )
            for dependency_id, dependency_tensor in dependency_tensors.items()
        }
        v1_indicator_input = IndicatorSignalEvaluationInputV1(
            indicator_id=indicator_id,
            primary_output=indicator_primary_output_series_from_tensor_v1(
                tensor=primary_tensor,
                variant_index=row_index,
            ),
            indicator_inputs=_indicator_inputs_mapping_v2(
                resolved_source=_resolve_signal_row_inputs_source_v2(
                    signal_row=signal_row,
                    rule_spec=rule_spec,
                    default_inputs_source=default_inputs_source,
                ),
                spec=rule_spec,
            ),
            signal_params=signal_params_defaults,
            dependency_outputs=dependency_outputs,
        )
        signal_matrix[row_index, :] = _normalize_signal_codes_v2(
            indicator_id=indicator_id,
            signal_codes=evaluate_indicator_signal_encoded_v1(
                candles=candles,
                indicator_input=v1_indicator_input,
            ),
        )
    _validate_signal_matrix_v2(
        signal_matrix=signal_matrix,
        expected_shape=(row_count, time_count),
        label=f"signals[{indicator_id}]",
    )
    return signal_matrix


def _resolve_signal_row_inputs_source_v2(
    *,
    signal_row: _SignalVariantRowV2,
    rule_spec: SignalRuleSpecV2,
    default_inputs_source: str | None,
) -> str | None:
    """
    Resolve the effective row-level `inputs.source` literal from explicit row data.

    Args:
        signal_row: One ordered signal row descriptor.
        rule_spec: Explicit signal rule specification for the target indicator.
        default_inputs_source: Default `inputs.source` literal resolved once per indicator.
    Returns:
        str | None: Effective row-level source literal or `None` when the rule family ignores it.
    Assumptions:
        `signal_row.inputs_source` already follows canonical grid ordering and is preferred over
        the indicator-level default when present.
    Raises:
        ValueError: If a row requiring `inputs.source` has neither an explicit nor default value.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if signal_row.inputs_source is not None:
        return validate_signal_input_source_v2(signal_row.inputs_source)
    if not rule_spec.uses_inputs_source:
        return None
    if default_inputs_source is None:
        raise ValueError(
            f"{rule_spec.indicator_id}: chunked signal evaluation requires a default inputs.source"
        )
    return validate_signal_input_source_v2(default_inputs_source)


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _build_signal_features_manifest_v2(
    *,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_shape: tuple[int, int],
    signal_paths: ArtifactSignalPathsV2,
    signal_features_paths: ArtifactSignalFeaturesPathsV2,
) -> ArtifactSignalFeaturesManifestDocumentV2:
    """
    Build the strict typed signal-feature manifest for one freshly written feature matrix.

    Args:
        slot: Inactive slot literal receiving the manifest.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying root identity and timestamps.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        runtime_settings: Strict runtime settings contributing config hash identity.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        signal_shape: Final source signal matrix shape `[V, T_tf]`.
        signal_paths: Fixed source signal family paths used for provenance hashing.
        signal_features_paths: Fixed signal-feature family paths under the inactive slot.
    Returns:
        ArtifactSignalFeaturesManifestDocumentV2: Typed strict signal-feature manifest.
    Assumptions:
        The feature file already exists on disk and is ready for `sha256` hashing.
    Raises:
        ValueError: If one manifest field violates the strict typed contract.
        OSError: If one written artifact file cannot be hashed.
    Side Effects:
        Reads the freshly written feature matrix file to compute its manifest hash.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """
    validated_slot = validate_artifact_slot_v2(slot)
    features_metadata = ArtifactArrayMetadataV2(
        path=_slot_relative_path_v2(
            slot_root=slot_root,
            absolute_path=signal_features_paths.features,
        ),
        dtype=ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
        shape=(int(signal_shape[0]), len(SIGNAL_FEATURE_NAMES_V2)),
        axis_order=ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
        sha256=_file_sha256_hex_v2(signal_features_paths.features),
    )
    provenance = _build_signal_features_manifest_provenance_v2(
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        signal_target=signal_target,
        signal_shape=signal_shape,
        signal_paths=signal_paths,
        signal_features_paths=signal_features_paths,
    )
    payload = {
        "schema_version": SIGNAL_FEATURES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        "manifest_kind": SIGNAL_FEATURES_ARTIFACT_MANIFEST_KIND_V2,
        "slot": validated_slot,
        "slot_generation": slot_generation,
        "asof_date": request.asof_date,
        "indicator_id": signal_target.indicator_id,
        "timeframe": signal_target.timeframe,
        "features": _serialize_array_metadata_v2(features_metadata),
        "rows_count": int(signal_shape[0]),
        "feature_names": [name for name in SIGNAL_FEATURE_NAMES_V2],
        "provenance": _serialize_provenance_v2(provenance),
    }
    return ArtifactSignalFeaturesManifestDocumentV2(
        path=signal_features_paths.manifest,
        raw_payload=payload,
        slot=validated_slot,
        schema_version=SIGNAL_FEATURES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        manifest_kind=SIGNAL_FEATURES_ARTIFACT_MANIFEST_KIND_V2,
        slot_generation=slot_generation,
        asof_date=request.asof_date,
        indicator_id=signal_target.indicator_id,
        timeframe=signal_target.timeframe,
        features=features_metadata,
        rows_count=int(signal_shape[0]),
        feature_names=SIGNAL_FEATURE_NAMES_V2,
        provenance=provenance,
    )


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
    signal_shape: tuple[int, int],
    timeline: ArtifactTimelineCoverageV2,
    price_manifest: ArtifactPriceTimeframeManifestV2,
    signal_rows: tuple[_SignalVariantRowV2, ...],
    signal_params_defaults: Mapping[str, Any],
    signal_rules_engine: BacktestSignalRulesEngineV2,
    effective_tail_bars: int,
    signal_features: ArtifactSignalFeaturesReferenceV2 | None,
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
        signal_shape: Freshly written compact signal matrix shape.
        timeline: Timeline coverage matching the target `prices/<tf>` manifest.
        price_manifest: Fresh target timeframe price section used for provenance hashing.
        signal_rows: Ordered signal row descriptors used for `variant_keys_sha256`.
        signal_params_defaults: Resolved `signals.v1.params` default mapping.
        signal_rules_engine: Explicit signal rules engine used to capture dependency metadata.
        effective_tail_bars: Effective target-timeframe tail window used for rebuild planning.
        signal_features: Optional additive feature-manifest reference for the same signal target.
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    validated_slot = validate_artifact_slot_v2(slot)
    signal_metadata = ArtifactArrayMetadataV2(
        path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=signal_paths.signals),
        dtype=ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
        shape=tuple(int(value) for value in signal_shape),
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
        effective_tail_bars=effective_tail_bars,
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
        "rows_count": int(signal_shape[0]),
        "timeline": _serialize_timeline_coverage_v2(timeline),
        "signal_value_set": [int(value) for value in ARTIFACT_SIGNAL_VALUE_SET_V2],
        "grid": _serialize_signal_grid_contract_v2(grid_contract),
        "provenance": _serialize_provenance_v2(provenance),
    }
    if signal_features is not None:
        payload["signal_features"] = _serialize_signal_features_reference_v2(signal_features)
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
        rows_count=int(signal_shape[0]),
        timeline=timeline,
        signal_value_set=ARTIFACT_SIGNAL_VALUE_SET_V2,
        grid=grid_contract,
        provenance=provenance,
        signal_features=signal_features,
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _effective_signal_tail_bars_v2(
    *,
    timeframe: str,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
) -> int:
    """
    Derive the bounded signal tail window for one `(timeframe, indicator_id)` target.

    Args:
        timeframe: Target signal timeframe literal.
        runtime_settings: Strict runtime settings carrying `signal_tail_bars_1m`.
    Returns:
        int: Effective target-timeframe tail length in bars.
    Assumptions:
        Tail planning rewrites only the bounded overlap derived from
        `lookback_policy.signal_tail_bars_1m`.
    Raises:
        ValueError: If the timeframe literal is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return max(
        1,
        _signal_target_tail_bars_from_1m_v2(
            timeframe=timeframe,
            signal_tail_bars_1m=runtime_settings.signal_tail_bars_1m,
        ),
    )


def _signal_rebuild_context_bars_v2(
    *,
    materialized_grid: Any,
    defaults_provider: BacktestGridDefaultsProvider,
    indicator_id: str,
) -> int:
    """
    Derive conservative leading-history bars required to compute a bounded signal tail.

    Args:
        materialized_grid: Materialized compute grid for the target indicator.
        defaults_provider: Runtime defaults provider exposing `signals.v1.params`.
        indicator_id: Target indicator identifier.
    Returns:
        int: Conservative leading-history budget in target-timeframe bars.
    Assumptions:
        Warmup/lag history is used only to seed deterministic tail recomputation and must not
        expand the final rewritten tail segment by itself.
    Raises:
        ValueError: If signal defaults cannot be materialized deterministically.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    return (
        _signal_compute_context_bars_v2(materialized_grid=materialized_grid)
        + _signal_param_context_bars_v2(
            signal_param_specs=defaults_provider.signal_param_defaults(
                indicator_id=indicator_id
            )
        )
    )


def _signal_target_tail_bars_from_1m_v2(
    *,
    timeframe: str,
    signal_tail_bars_1m: int,
) -> int:
    """
    Convert configured `signal_tail_bars_1m` into the target-timeframe bar budget.

    Args:
        timeframe: Target signal timeframe literal.
        signal_tail_bars_1m: Configured tail budget expressed in `1m` bars.
    Returns:
        int: Ceil-divided target-timeframe tail length with minimum `1`.
    Assumptions:
        Signal tail policy is configured in `1m` bars so all target timeframes share one source
        of truth.
    Raises:
        ValueError: If the timeframe literal is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - src/trading/shared_kernel/primitives/timeframe.py
    """
    target_duration_millis = _timeframe_duration_millis_v2(Timeframe(timeframe))
    requested_duration_millis = signal_tail_bars_1m * _ONE_MINUTE_MILLIS_V2
    return max(
        1,
        int((requested_duration_millis + target_duration_millis - 1) // target_duration_millis),
    )


def _signal_compute_context_bars_v2(*, materialized_grid: Any) -> int:
    """
    Estimate conservative compute-history bars from integer-valued materialized grid axes.

    Args:
        materialized_grid: Materialized indicator grid returned by `GridBuilder`.
    Returns:
        int: Conservative history budget in target-timeframe bars.
    Assumptions:
        Positive integer axes such as `window`, `left`, `right`, or `signal_window` dominate the
        finite warmup/lag semantics for bounded signal rebuild planning.
    Raises:
        ValueError: If one axis exposes an invalid empty value sequence.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    total_context_bars = 0
    for axis in getattr(materialized_grid, "axes"):
        axis_name = str(getattr(axis, "name")).strip().lower()
        if axis_name == "source":
            continue
        integer_values = tuple(
            abs(int(value))
            for value in getattr(axis, "values")
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if len(integer_values) == 0:
            continue
        total_context_bars += max(integer_values)
    return total_context_bars


def _signal_param_context_bars_v2(
    *,
    signal_param_specs: Mapping[str, Any],
) -> int:
    """
    Estimate conservative lag bars contributed by default-only `signals.v1.params`.

    Args:
        signal_param_specs: Default-only signal parameter specs keyed by param name.
    Returns:
        int: Conservative lag budget in target-timeframe bars.
    Assumptions:
        Only integer-valued params such as delta periods contribute directly to bar-history
        planning, while thresholds remain lag-free.
    Raises:
        ValueError: If one provided spec cannot materialize deterministically.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
      - src/trading/contexts/indicators/domain/specifications/grid_param_spec.py
    """
    total_context_bars = 0
    for spec in signal_param_specs.values():
        integer_values = tuple(
            abs(int(value))
            for value in spec.materialize()
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if len(integer_values) == 0:
            continue
        total_context_bars += max(integer_values)
    return total_context_bars


def _select_signal_catalog_entry_v2(
    *,
    catalog: ArtifactSignalCatalogV2,
    signal_target: ArtifactSignalValidationSpecV2,
) -> ArtifactSignalCatalogEntryV2 | None:
    """
    Select one existing root-catalog entry for the requested signal target.

    Args:
        catalog: Existing root-manifest signal catalog.
        signal_target: Explicit `(timeframe, indicator_id)` signal target.
    Returns:
        ArtifactSignalCatalogEntryV2 | None: Matching catalog entry when present.
    Assumptions:
        Root manifests keep at most one entry per `(timeframe, indicator_id)` identity.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    for entry in catalog.manifests:
        if (
            entry.timeframe == signal_target.timeframe
            and entry.indicator_id == signal_target.indicator_id
        ):
            return entry
    return None


def _load_existing_signal_artifact_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    signal_target: ArtifactSignalValidationSpecV2,
    expected_variant_keys_sha256: str,
    expected_row_count: int,
) -> _ExistingSignalArtifactV2 | None:
    """
    Load one existing inactive-slot signal family when it is safe to reuse for tail rebuild.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        existing_manifest: Previously materialized inactive-slot root manifest, when present.
        signal_target: Explicit `(timeframe, indicator_id)` signal target.
        expected_variant_keys_sha256: Current deterministic row-order hash for the target.
        expected_row_count: Current deterministic row count for the target.
    Returns:
        _ExistingSignalArtifactV2 | None: Existing signal payload when safely reusable, otherwise
            `None`.
    Assumptions:
        Missing existing target files may trigger a deterministic full build, while manifest/data
        drift during a reuse attempt must fail fast.
    Raises:
        ValueError: If existing root/signal metadata drifts from strict reuse contracts.
        FileNotFoundError: Propagated only after manifest existence prechecks pass and actual data
            drift is detected.
    Side Effects:
        Reads existing manifest and signal matrix from disk when the target is present.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if existing_manifest is None:
        return None
    existing_entry = _select_signal_catalog_entry_v2(
        catalog=existing_manifest.signals,
        signal_target=signal_target,
    )
    if existing_entry is None:
        return None
    signal_paths = artifact_loader.resolve_signal_paths(
        coordinates,
        slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    expected_manifest_path = _slot_relative_path_v2(
        slot_root=signal_paths.manifest.parents[3],
        absolute_path=signal_paths.manifest,
    )
    if existing_entry.manifest_path != expected_manifest_path:
        raise ValueError(
            "existing signal catalog manifest_path must match deterministic path for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{existing_entry.manifest_path!r}, expected {expected_manifest_path!r}"
        )
    if not signal_paths.manifest.is_file() or not signal_paths.signals.is_file():
        return None
    actual_manifest_sha256 = _file_sha256_hex_v2(signal_paths.manifest)
    if existing_entry.manifest_sha256 != actual_manifest_sha256:
        raise ValueError(
            "existing signal catalog manifest_sha256 must match actual file for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{existing_entry.manifest_sha256!r}, expected {actual_manifest_sha256!r}"
        )
    signal_manifest = artifact_loader.load_signal_manifest(
        coordinates,
        slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    expected_signal_path = _slot_relative_path_v2(
        slot_root=signal_paths.signals.parents[3],
        absolute_path=signal_paths.signals,
    )
    if signal_manifest.signals.path != expected_signal_path:
        raise ValueError(
            "existing signal manifest signals.path must match deterministic path for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.signals.path!r}, expected {expected_signal_path!r}"
        )
    if signal_manifest.signals.dtype != ARTIFACT_SIGNAL_DTYPE_LITERAL_V2:
        raise ValueError(
            "existing signal manifest signals.dtype must match the strict contract for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.signals.dtype!r}"
        )
    if signal_manifest.signals.axis_order != ARTIFACT_SIGNAL_AXIS_ORDER_V2:
        raise ValueError(
            "existing signal manifest signals.axis_order must match the strict contract for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.signals.axis_order!r}"
        )
    signal_matrix = cast(
        np.memmap,
        np.load(signal_paths.signals, mmap_mode="r", allow_pickle=False),
    )
    actual_shape = tuple(int(value) for value in signal_matrix.shape)
    if signal_manifest.signals.shape != actual_shape:
        raise ValueError(
            "existing signal manifest signals.shape must match the actual file for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.signals.shape!r}, expected {actual_shape!r}"
        )
    if actual_shape != (signal_manifest.rows_count, signal_manifest.timeline.bar_count):
        raise ValueError(
            "existing signal file shape must match manifest rows/timeline for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got {actual_shape!r}"
        )
    if signal_matrix.dtype.name != ARTIFACT_SIGNAL_DTYPE_LITERAL_V2:
        raise ValueError(
            "existing signal file dtype must match the strict contract for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_matrix.dtype.name!r}"
        )
    actual_signal_sha256 = _file_sha256_hex_v2(signal_paths.signals)
    if signal_manifest.signals.sha256 != actual_signal_sha256:
        raise ValueError(
            "existing signal manifest signals.sha256 must match the actual file for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.signals.sha256!r}, expected {actual_signal_sha256!r}"
        )
    del signal_matrix
    if signal_manifest.slot_generation != existing_manifest.slot_generation:
        raise ValueError(
            "existing signal manifest slot_generation must match root manifest for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.slot_generation!r}, expected "
            f"{existing_manifest.slot_generation!r}"
        )
    if signal_manifest.asof_date != existing_manifest.asof_date:
        raise ValueError(
            "existing signal manifest asof_date must match root manifest for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.asof_date!r}, expected {existing_manifest.asof_date!r}"
        )
    if signal_manifest.grid.variant_keys_sha256 != expected_variant_keys_sha256:
        raise ValueError(
            "existing signal manifest grid.variant_keys_sha256 must match current row order for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.grid.variant_keys_sha256!r}, expected "
            f"{expected_variant_keys_sha256!r}"
        )
    if signal_manifest.rows_count != expected_row_count:
        raise ValueError(
            "existing signal manifest rows_count must match current grid rows for "
            f"{signal_target.timeframe}:{signal_target.indicator_id}; got "
            f"{signal_manifest.rows_count!r}, expected {expected_row_count!r}"
        )
    return _ExistingSignalArtifactV2(
        catalog_entry=existing_entry,
        manifest=signal_manifest,
        signals_path=signal_paths.signals,
    )


def _build_signal_tail_plan_v2(
    *,
    price_arrays: _CanonicalPriceArraysV2,
    existing_signal_artifact: _ExistingSignalArtifactV2 | None,
    effective_tail_bars: int,
    rebuild_context_bars: int,
) -> _SignalArtifactTailPlanV2:
    """
    Build deterministic prefix reuse bounds for one signal artifact tail rebuild.

    Args:
        price_arrays: Fresh materialized target-timeframe price arrays.
        existing_signal_artifact: Existing inactive-slot signal family, when safely reusable.
        effective_tail_bars: Derived target-timeframe tail length in bars.
        rebuild_context_bars: Additional leading-history bars required for deterministic compute.
    Returns:
        _SignalArtifactTailPlanV2: Prefix slice and compute-start index for `prefix + rebuilt_tail`.
    Assumptions:
        Prefix reuse is allowed only when the existing signal timeline shares the same leading
        timeline start and there are enough overlapping bars left after the rebuilt tail window.
    Raises:
        ValueError: If derived slice indexes are inconsistent.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    current_timeline = _timeline_coverage_from_arrays_v2(arrays=price_arrays)
    current_bar_count = current_timeline.bar_count
    bounded_tail_bars = min(current_bar_count, effective_tail_bars)
    if existing_signal_artifact is None:
        return _SignalArtifactTailPlanV2(
            reused_prefix_bars=0,
            compute_start_idx=0,
            trim_prefix_bars=0,
            effective_tail_bars=bounded_tail_bars,
        )
    existing_timeline = existing_signal_artifact.manifest.timeline
    if (
        existing_timeline.open_time_start != current_timeline.open_time_start
        or existing_timeline.close_time_start != current_timeline.close_time_start
    ):
        return _SignalArtifactTailPlanV2(
            reused_prefix_bars=0,
            compute_start_idx=0,
            trim_prefix_bars=0,
            effective_tail_bars=bounded_tail_bars,
        )
    overlapping_bar_count = min(current_bar_count, existing_timeline.bar_count)
    if overlapping_bar_count <= bounded_tail_bars:
        return _SignalArtifactTailPlanV2(
            reused_prefix_bars=0,
            compute_start_idx=0,
            trim_prefix_bars=0,
            effective_tail_bars=bounded_tail_bars,
        )
    prefix_bar_count = overlapping_bar_count - bounded_tail_bars
    compute_start_idx = max(0, prefix_bar_count - rebuild_context_bars)
    return _SignalArtifactTailPlanV2(
        reused_prefix_bars=prefix_bar_count,
        compute_start_idx=compute_start_idx,
        trim_prefix_bars=prefix_bar_count - compute_start_idx,
        effective_tail_bars=bounded_tail_bars,
    )


def _slice_signal_matrix_v2(
    *,
    signal_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """
    Slice one `[V, T_tf]` signal matrix on the time axis into a contiguous sub-matrix.

    Args:
        signal_matrix: Source signal matrix.
        start_idx: Inclusive time-axis slice start.
        end_idx: Exclusive time-axis slice end.
    Returns:
        np.ndarray: Contiguous `int8` signal sub-matrix.
    Assumptions:
        Callers already validate that row order remains unchanged across reuse attempts.
    Raises:
        ValueError: If slice indexes are negative or inconsistent.
    Side Effects:
        Allocates one contiguous array.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if start_idx < 0:
        raise ValueError(f"signal slice start_idx must be >= 0; got {start_idx!r}")
    if end_idx < start_idx:
        raise ValueError(
            f"signal slice end_idx must be >= start_idx; got {end_idx!r} and {start_idx!r}"
        )
    return np.ascontiguousarray(signal_matrix[:, start_idx:end_idx], dtype=np.int8)


def _merge_signal_matrices_v2(
    *,
    prefix_matrix: np.ndarray | None,
    rebuilt_tail: np.ndarray,
) -> np.ndarray:
    """
    Merge one reused signal prefix with a freshly rebuilt tail matrix.

    Args:
        prefix_matrix: Existing unchanged prefix matrix, or `None` for full rebuild.
        rebuilt_tail: Freshly rebuilt signal tail matrix.
    Returns:
        np.ndarray: Contiguous merged `[V, T_tf]` matrix.
    Assumptions:
        Prefix and rebuilt tail share identical row ordering and dtype contracts.
    Raises:
        ValueError: If row counts differ across prefix and rebuilt tail.
    Side Effects:
        Allocates one contiguous merged matrix when prefix is present.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if prefix_matrix is None or int(prefix_matrix.shape[1]) == 0:
        return rebuilt_tail
    if prefix_matrix.shape[0] != rebuilt_tail.shape[0]:
        raise ValueError(
            "signal prefix row count must match rebuilt tail row count; got "
            f"{prefix_matrix.shape[0]!r} and {rebuilt_tail.shape[0]!r}"
        )
    return np.ascontiguousarray(
        np.concatenate((prefix_matrix, rebuilt_tail), axis=1),
        dtype=np.int8,
    )


def _validate_existing_signal_defaults_for_reuse_v2(
    *,
    existing_signal_artifact: _ExistingSignalArtifactV2,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_params_defaults: Mapping[str, Any],
) -> None:
    """
    Validate default-only signal params remain identical across signal prefix reuse attempts.

    Args:
        existing_signal_artifact: Existing inactive-slot signal family selected for reuse.
        signal_target: Explicit `(timeframe, indicator_id)` signal target.
        signal_params_defaults: Freshly resolved default-only signal params for the rebuild.
    Returns:
        None.
    Assumptions:
        Reusing an existing prefix is safe only when `signals.v1.params` remain `default-only`
        and byte-for-byte identical at the manifest contract level.
    Raises:
        ValueError: If existing manifest defaults drift from the current resolved defaults.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if dict(existing_signal_artifact.manifest.grid.signals_v1_params_defaults) != dict(
        signal_params_defaults
    ):
        raise ValueError(
            "existing signal manifest grid.signals_v1_params_defaults must match current "
            f"default-only params for {signal_target.timeframe}:{signal_target.indicator_id}"
        )


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """
    return InstrumentId(
        market_id=MarketId(artifact_market_id_from_coordinates_v2(coordinates)),
        symbol=Symbol(coordinates.symbol),
    )


def _read_canonical_price_arrays_v2(
    *,
    canonical_candle_reader: CanonicalCandleReader,
    coordinates: ArtifactCoordinatesV2,
    source_time_range: TimeRange,
) -> _CanonicalPriceArraysV2:
    """
    Read canonical `1m` source candles through the columnar reader contract and validate arrays.

    Args:
        canonical_candle_reader: Canonical candle reader port with array fast-path support.
        coordinates: Artifact coordinates selecting one backtest symbol root.
        source_time_range: Exact source reread window used for deterministic bootstrap/tail reads.
    Returns:
        _CanonicalPriceArraysV2: Strict contiguous arrays ready for slot materialization.
    Assumptions:
        Offline/precompute workloads may bypass `CandleWithMeta` allocation and request columnar
        arrays directly from storage adapters.
    Raises:
        ValueError: If the source produced no rows or violates strict timeline monotonicity.
    Side Effects:
        Reads canonical candle storage once for the requested range.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    batch = canonical_candle_reader.read_1m_arrays(
        instrument_id=_instrument_id_from_coordinates_v2(coordinates),
        time_range=source_time_range,
    )
    return _canonical_price_arrays_from_batch_v2(
        batch=batch,
        source_time_range=source_time_range,
    )


def _canonical_price_arrays_from_batch_v2(
    *,
    batch: CanonicalCandleBatch1m,
    source_time_range: TimeRange,
) -> _CanonicalPriceArraysV2:
    """
    Convert a columnar canonical candle batch into contiguous strict price arrays.

    Args:
        batch: Canonical candle batch returned by `CanonicalCandleReader.read_1m_arrays(...)`.
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
    """
    if batch.row_count() == 0:
        raise ValueError(
            "canonical 1m source returned no candles for "
            f"TimeRange [start, end)={_time_range_literal_v2(source_time_range)}"
        )
    arrays = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(batch.open_time_ms, dtype=np.int64),
        close_time=np.ascontiguousarray(batch.close_time_ms, dtype=np.int64),
        ohlcv=np.ascontiguousarray(batch.ohlcv_f32, dtype=np.float32),
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _materialize_rolled_price_timeframe_v2(
    timeframe: str,
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    source_arrays: _CanonicalPriceArraysV2,
    source_tail_time_range: TimeRange,
) -> ArtifactPriceTimeframeManifestV2:
    """
    Materialize one rolled `prices/<tf>` family for the inactive slot.

    Args:
        timeframe: Target rolled timeframe literal.
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving the rolled arrays.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest when present.
        source_arrays: Canonical `prices/1m` arrays loaded back from the materialized artifact.
        source_tail_time_range: Effective canonical source reread window used for the `1m` build.
    Returns:
        ArtifactPriceTimeframeManifestV2: Strict manifest section for the rolled timeframe.
    Assumptions:
        R12-03 intentionally calls this helper inside one active `timeframe_session` at a time so
        only the current target `rolled_prices` arrays stay live before mappings/signals consume
        them.
    Raises:
        ValueError: If source arrays or prefix reuse violate strict contracts.
        OSError: If writing one timeframe artifact family fails.
    Side Effects:
        Atomically writes `prices/<tf>/*.npy` files for one rolled timeframe.
    """
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
    return _build_price_manifest_v2(
        slot_root=slot_root,
        timeframe=timeframe,
        price_paths=price_paths,
        arrays=rolled_arrays,
    )


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _materialize_mapping_timeframe_v2(
    timeframe: str,
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    one_minute_arrays: _CanonicalPriceArraysV2,
    price_by_timeframe: Mapping[str, ArtifactPriceTimeframeManifestV2],
    mapping_tail_bars_1m: int,
) -> _MappingArtifactMaterializationResultV2:
    """
    Materialize one strict `mappings/<tf>` family for the inactive slot.

    Args:
        timeframe: Target mapping timeframe literal.
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot receiving the mapping arrays.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Existing inactive-slot root manifest when present.
        one_minute_arrays: Materialized artifact-backed `prices/1m` arrays.
        price_by_timeframe: Fresh strict `prices/<tf>` sections keyed by timeframe.
        mapping_tail_bars_1m: Effective `lookback_policy.mapping_tail_bars_1m`.
    Returns:
        _MappingArtifactMaterializationResultV2: Strict manifest plus per-timeframe rebuild stats.
    Assumptions:
        R12-03 keeps mapping work inside the current `timeframe_session` so one target timeframe
        fully completes before the next `rolled_prices` family is opened.
    Raises:
        ValueError: If target price arrays or prefix reuse violate strict mapping contracts.
        OSError: If writing one mapping family fails.
    Side Effects:
        Atomically writes `mappings/<tf>/*.npy` files for one timeframe.
    """
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
    mapping_build_result = _build_mapping_arrays_with_tail_update_v2(
        one_minute_arrays=one_minute_arrays,
        timeframe_arrays=timeframe_arrays,
        existing_arrays=existing_arrays,
        timeframe=timeframe,
        mapping_tail_bars_1m=mapping_tail_bars_1m,
    )
    mapping_paths = artifact_loader.resolve_mapping_paths(coordinates, slot, timeframe)
    _write_mapping_arrays_atomically_v2(
        mapping_paths=mapping_paths,
        arrays=mapping_build_result.arrays,
    )
    return _MappingArtifactMaterializationResultV2(
        manifest=_build_mapping_manifest_v2(
            slot_root=slot_root,
            timeframe=timeframe,
            mapping_paths=mapping_paths,
            arrays=mapping_build_result.arrays,
        ),
        reused_prefix_bars=mapping_build_result.reused_prefix_bars,
        rewritten_tail_bars=mapping_build_result.rewritten_tail_bars,
    )


def _build_mapping_arrays_with_tail_update_v2(
    *,
    one_minute_arrays: _CanonicalPriceArraysV2,
    timeframe_arrays: _CanonicalPriceArraysV2,
    existing_arrays: _TimeframeMappingArraysV2 | None,
    timeframe: str,
    mapping_tail_bars_1m: int,
) -> _TimeframeMappingBuildResultV2:
    """
    Build one `tf -> 1m` mapping family using bounded tail rebuild plus deterministic prefix reuse.

    Args:
        one_minute_arrays: Materialized artifact-backed `prices/1m` arrays.
        timeframe_arrays: Materialized artifact-backed `prices/<tf>` arrays.
        existing_arrays: Existing inactive-slot mapping arrays for the timeframe, when present.
        timeframe: Target request timeframe literal.
        mapping_tail_bars_1m: Effective `lookback_policy.mapping_tail_bars_1m`.
    Returns:
        _TimeframeMappingBuildResultV2: Final strict mapping arrays plus the number of rebuilt
            target-timeframe bars.
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if existing_arrays is None:
        arrays = _build_mapping_arrays_from_price_timelines_v2(
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            timeframe=timeframe,
        )
        return _TimeframeMappingBuildResultV2(
            arrays=arrays,
            reused_prefix_bars=0,
            rewritten_tail_bars=int(timeframe_arrays.open_time.shape[0]),
        )

    one_minute_bar_count = int(one_minute_arrays.open_time.shape[0])
    if one_minute_bar_count <= mapping_tail_bars_1m:
        arrays = _build_mapping_arrays_from_price_timelines_v2(
            one_minute_arrays=one_minute_arrays,
            timeframe_arrays=timeframe_arrays,
            timeframe=timeframe,
        )
        return _TimeframeMappingBuildResultV2(
            arrays=arrays,
            reused_prefix_bars=0,
            rewritten_tail_bars=int(timeframe_arrays.open_time.shape[0]),
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
        return _TimeframeMappingBuildResultV2(
            arrays=prefix,
            reused_prefix_bars=prefix_bar_count,
            rewritten_tail_bars=0,
        )

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
    return _TimeframeMappingBuildResultV2(
        arrays=merged,
        reused_prefix_bars=prefix_bar_count,
        rewritten_tail_bars=int(tail.bar_open_1m_idx.shape[0]),
    )


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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _load_existing_hit_times_artifact_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
) -> _ExistingHitTimesArtifactV2 | None:
    """
    Load an existing inactive-slot `hit_times/15m` family when it is safe to reuse.

    Args:
        artifact_loader: Explicit-path artifact loader.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Candidate inactive slot literal.
        existing_manifest: Previously materialized inactive-slot root manifest, when present.
        one_minute_manifest: Fresh current-build `prices/1m` manifest for start-alignment checks.
        runtime_settings: Strict runtime settings carrying current TP/SL grids.
    Returns:
        _ExistingHitTimesArtifactV2 | None: Existing artifact snapshot eligible for prefix reuse,
            otherwise `None` to trigger deterministic stage-local full rebuild.
    Assumptions:
        Missing files or reuse-precondition drift should not block the symbol root; the stage may
        fall back to full rebuild instead.
    Raises:
        None.
    Side Effects:
        Reads existing hit-times manifest and arrays from the inactive slot when present.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    if existing_manifest is None:
        return None
    expected_tp_values = np.ascontiguousarray(
        np.asarray(
            [value / 100.0 for value in runtime_settings.hit_times_tp_levels_pct],
            dtype=np.float32,
        )
    )
    expected_sl_values = np.ascontiguousarray(
        np.asarray(
            [value / 100.0 for value in runtime_settings.hit_times_sl_levels_pct],
            dtype=np.float32,
        )
    )
    existing_one_minute_manifest = _select_price_manifest_v2(
        price_sections=existing_manifest.prices,
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
    )
    if existing_one_minute_manifest is None:
        return None
    if (
        existing_one_minute_manifest.coverage.open_time_start
        != one_minute_manifest.coverage.open_time_start
        or existing_one_minute_manifest.coverage.close_time_start
        != one_minute_manifest.coverage.close_time_start
    ):
        return None

    hit_times_paths = artifact_loader.resolve_hit_times_paths(coordinates, slot)
    expected_manifest_path = _slot_relative_path_v2(
        slot_root=hit_times_paths.manifest.parents[2],
        absolute_path=hit_times_paths.manifest,
    )
    if existing_manifest.hit_times.manifest_path != expected_manifest_path:
        return None
    required_paths = (
        hit_times_paths.manifest,
        hit_times_paths.tp_values,
        hit_times_paths.sl_values,
        hit_times_paths.long_tp,
        hit_times_paths.long_sl,
        hit_times_paths.short_tp,
        hit_times_paths.short_sl,
    )
    if not all(path.is_file() for path in required_paths):
        return None
    if existing_manifest.hit_times.manifest_sha256 != _file_sha256_hex_v2(hit_times_paths.manifest):
        return None

    try:
        hit_times_manifest = artifact_loader.load_hit_times_manifest(coordinates, slot)
        if (
            hit_times_manifest.slot_generation != existing_manifest.slot_generation
            or hit_times_manifest.asof_date != existing_manifest.asof_date
        ):
            return None
        tp_values = np.ascontiguousarray(
            _load_validated_array_v2(
                metadata=hit_times_manifest.tp_values,
                expected_path=hit_times_paths.tp_values,
                expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
                expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
                expected_shape=(int(hit_times_manifest.tp_values.shape[0]),),
                location="existing hit_times.tp_values",
            ),
            dtype=np.float32,
        )
        sl_values = np.ascontiguousarray(
            _load_validated_array_v2(
                metadata=hit_times_manifest.sl_values,
                expected_path=hit_times_paths.sl_values,
                expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
                expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
                expected_shape=(int(hit_times_manifest.sl_values.shape[0]),),
                location="existing hit_times.sl_values",
            ),
            dtype=np.float32,
        )
        if not np.array_equal(tp_values, expected_tp_values):
            return None
        if not np.array_equal(sl_values, expected_sl_values):
            return None
        arrays = HitTimesArraysV2(
            tp_values=tp_values,
            sl_values=sl_values,
            long_tp=np.ascontiguousarray(
                _load_validated_array_v2(
                    metadata=hit_times_manifest.long_tp.array,
                    expected_path=hit_times_paths.long_tp,
                    expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
                    expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
                    expected_shape=(
                        int(hit_times_manifest.long_tp.array.shape[0]),
                        hit_times_manifest.timeline_bar_count,
                    ),
                    location="existing hit_times.long_tp",
                ),
                dtype=np.uint32,
            ),
            long_sl=np.ascontiguousarray(
                _load_validated_array_v2(
                    metadata=hit_times_manifest.long_sl.array,
                    expected_path=hit_times_paths.long_sl,
                    expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
                    expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
                    expected_shape=(
                        int(hit_times_manifest.long_sl.array.shape[0]),
                        hit_times_manifest.timeline_bar_count,
                    ),
                    location="existing hit_times.long_sl",
                ),
                dtype=np.uint32,
            ),
            short_tp=np.ascontiguousarray(
                _load_validated_array_v2(
                    metadata=hit_times_manifest.short_tp.array,
                    expected_path=hit_times_paths.short_tp,
                    expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
                    expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
                    expected_shape=(
                        int(hit_times_manifest.short_tp.array.shape[0]),
                        hit_times_manifest.timeline_bar_count,
                    ),
                    location="existing hit_times.short_tp",
                ),
                dtype=np.uint32,
            ),
            short_sl=np.ascontiguousarray(
                _load_validated_array_v2(
                    metadata=hit_times_manifest.short_sl.array,
                    expected_path=hit_times_paths.short_sl,
                    expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
                    expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
                    expected_shape=(
                        int(hit_times_manifest.short_sl.array.shape[0]),
                        hit_times_manifest.timeline_bar_count,
                    ),
                    location="existing hit_times.short_sl",
                ),
                dtype=np.uint32,
            ),
            sentinel_index=hit_times_manifest.sentinel_index,
        )
    except (FileNotFoundError, ValueError):
        return None
    return _ExistingHitTimesArtifactV2(manifest=hit_times_manifest, arrays=arrays)


def _build_hit_times_tail_plan_v2(
    *,
    hit_times_source_arrays: _CanonicalPriceArraysV2,
    existing_hit_times_artifact: _ExistingHitTimesArtifactV2 | None,
    hit_times_tail_bars_1m: int,
) -> _HitTimesArtifactTailPlanV2:
    """
    Build deterministic prefix reuse bounds for bounded `hit_times/15m` rebuilds.

    Args:
        hit_times_source_arrays: Fresh source arrays for the configured hit-times timeframe.
        existing_hit_times_artifact: Existing inactive-slot hit-times family, when reusable.
        hit_times_tail_bars_1m: Configured bounded tail window in canonical `1m` bars.
    Returns:
        _HitTimesArtifactTailPlanV2: Prefix slice plus effective tail window.
    Assumptions:
        Reused prefix bars are preserved verbatim, while the tail overlap and any appended suffix
        are rebuilt from current hit-times source arrays.
    Raises:
        ValueError: If the derived prefix slice indexes are inconsistent.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    current_bar_count = int(hit_times_source_arrays.open_time.shape[0])
    effective_tail_bars = min(
        current_bar_count,
        _convert_one_minute_tail_bars_to_hit_times_bars_v2(
            hit_times_tail_bars_1m=hit_times_tail_bars_1m
        ),
    )
    if existing_hit_times_artifact is None or current_bar_count <= effective_tail_bars:
        return _HitTimesArtifactTailPlanV2(
            prefix=None,
            prefix_bars=0,
            effective_tail_bars=effective_tail_bars,
        )
    overlapping_bar_count = min(
        current_bar_count,
        existing_hit_times_artifact.manifest.timeline_bar_count,
    )
    if overlapping_bar_count <= effective_tail_bars:
        return _HitTimesArtifactTailPlanV2(
            prefix=None,
            prefix_bars=0,
            effective_tail_bars=effective_tail_bars,
        )
    prefix_bars = overlapping_bar_count - effective_tail_bars
    if prefix_bars <= 0:
        raise ValueError(f"hit-times prefix_bars must be > 0; got {prefix_bars!r}")
    return _HitTimesArtifactTailPlanV2(
        prefix=_slice_hit_times_prefix_v2(
            hit_times=existing_hit_times_artifact.arrays,
            end_idx=prefix_bars,
        ),
        prefix_bars=prefix_bars,
        effective_tail_bars=effective_tail_bars,
    )


def _slice_hit_times_prefix_v2(
    *,
    hit_times: HitTimesArraysV2,
    end_idx: int,
) -> HitTimesArraysV2:
    """
    Slice one reusable hit-times prefix into a self-consistent standalone snapshot.

    Args:
        hit_times: Existing full hit-times arrays.
        end_idx: Exclusive prefix end index on the time axis.
    Returns:
        HitTimesArraysV2: Prefix slice with `sentinel_index == end_idx`.
    Assumptions:
        Prefix reuse keeps TP/SL grids unchanged and preserves only the leading table columns.
    Raises:
        ValueError: If `end_idx` is outside the existing timeline bounds.
    Side Effects:
        Allocates fresh contiguous arrays for the prefix slice.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    if end_idx <= 0:
        raise ValueError(f"hit-times prefix end_idx must be > 0; got {end_idx!r}")
    if end_idx > hit_times.sentinel_index:
        raise ValueError(
            "hit-times prefix end_idx must stay within the existing timeline; got "
            f"{end_idx!r}, sentinel_index={hit_times.sentinel_index!r}"
        )
    return HitTimesArraysV2(
        tp_values=np.ascontiguousarray(hit_times.tp_values, dtype=np.float32),
        sl_values=np.ascontiguousarray(hit_times.sl_values, dtype=np.float32),
        long_tp=np.ascontiguousarray(hit_times.long_tp[:, :end_idx], dtype=np.uint32),
        long_sl=np.ascontiguousarray(hit_times.long_sl[:, :end_idx], dtype=np.uint32),
        short_tp=np.ascontiguousarray(hit_times.short_tp[:, :end_idx], dtype=np.uint32),
        short_sl=np.ascontiguousarray(hit_times.short_sl[:, :end_idx], dtype=np.uint32),
        sentinel_index=end_idx,
    )


def _resolve_hit_times_cell_budget_v2(
    *,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    force_full_rebuild: bool,
    has_existing_slot_manifest: bool,
) -> int:
    """
    Resolve the effective hit-times cell budget for the current rebuild mode.

    Args:
        runtime_settings: Strict precompute runtime settings derived from artifact config.
        force_full_rebuild: Whether the current build intentionally rebuilds the full slot state.
        has_existing_slot_manifest: Whether the inactive slot already has a materialized root
            manifest that permits steady-state incremental rebuild logic.
    Returns:
        int: Effective hit-times table cell budget for this stage.
    Assumptions:
        Bootstrap and explicit full-rebuild runs need a larger budget than steady-state
        incremental tail refreshes, while incremental runs with an existing slot manifest should
        keep the tighter guard rail.
    Raises:
        None.
    Side Effects:
        None.
    """
    if force_full_rebuild or not has_existing_slot_manifest:
        return runtime_settings.max_hit_times_cells_full_rebuild
    return runtime_settings.max_hit_times_cells


def _resolve_hit_times_source_arrays_v2(
    *,
    one_minute_arrays: _CanonicalPriceArraysV2,
) -> _CanonicalPriceArraysV2:
    """
    Resolve source arrays for hit-times materialization from canonical `prices/1m`.

    Args:
        one_minute_arrays: Materialized canonical `prices/1m` arrays.
    Returns:
        _CanonicalPriceArraysV2: Source arrays aligned to `HIT_TIMES_TIMEFRAME_LITERAL_V2`.
    Assumptions:
        Hit-times may use either canonical `1m` directly or one rolled timeframe built from it.
    Raises:
        ValueError: If the configured hit-times timeframe cannot be materialized.
    Side Effects:
        Allocates rolled arrays when hit-times timeframe is not `1m`.
    """
    if HIT_TIMES_TIMEFRAME_LITERAL_V2 == _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2:
        return one_minute_arrays
    resolved = _rollup_price_arrays_from_one_minute_v2(
        source_arrays=one_minute_arrays,
        timeframe=HIT_TIMES_TIMEFRAME_LITERAL_V2,
        allow_empty=False,
    )
    if resolved is None:
        raise ValueError(
            "configured hit-times timeframe produced no full buckets from prices/1m; "
            f"timeframe={HIT_TIMES_TIMEFRAME_LITERAL_V2!r}"
        )
    return resolved


def _convert_one_minute_tail_bars_to_hit_times_bars_v2(*, hit_times_tail_bars_1m: int) -> int:
    """
    Convert `hit_times_tail_bars_1m` into bars of the configured hit-times timeframe.

    Args:
        hit_times_tail_bars_1m: Tail window configured in canonical `1m` bars.
    Returns:
        int: Equivalent tail budget measured in `HIT_TIMES_TIMEFRAME_LITERAL_V2` bars.
    Assumptions:
        Tail overlap should keep at least one hit-times bar when configured budget is positive.
    Raises:
        ValueError: If timeframe duration conversion is invalid.
    Side Effects:
        None.
    """
    if hit_times_tail_bars_1m <= 0:
        raise ValueError(
            "hit_times_tail_bars_1m must be > 0; "
            f"got {hit_times_tail_bars_1m!r}"
        )
    if HIT_TIMES_TIMEFRAME_LITERAL_V2 == _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2:
        return hit_times_tail_bars_1m
    duration_millis = _timeframe_duration_millis_v2(Timeframe(HIT_TIMES_TIMEFRAME_LITERAL_V2))
    if duration_millis <= 0 or duration_millis % _ONE_MINUTE_MILLIS_V2 != 0:
        raise ValueError(
            "hit-times timeframe duration must be a positive multiple of one minute; got "
            f"{duration_millis!r} ms for {HIT_TIMES_TIMEFRAME_LITERAL_V2!r}"
        )
    bars_per_hit_times_bar = duration_millis // _ONE_MINUTE_MILLIS_V2
    converted = (hit_times_tail_bars_1m + bars_per_hit_times_bar - 1) // bars_per_hit_times_bar
    return max(1, int(converted))


def _materialize_hit_times_artifacts_v2(
    *,
    artifact_loader: BacktestArtifactLoaderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    existing_manifest: ArtifactManifestDocumentV2 | None,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    hit_times_source_arrays: _CanonicalPriceArraysV2,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    max_hit_times_cells: int,
) -> _HitTimesArtifactBuildResultV2:
    """
    Materialize the strict R5-01 `hit_times/15m` artifact family for the inactive slot.

    Args:
        artifact_loader: Explicit-path artifact loader used to resolve fixed hit-times paths.
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the hit-times files.
        slot_root: Absolute inactive-slot root directory.
        existing_manifest: Previously materialized inactive-slot root manifest, when present.
        request: Explicit export request carrying slot identity and timestamps.
        slot_generation: Target inactive-slot generation assigned to the build.
        runtime_settings: Strict runtime settings carrying hit-times grids and guard budgets.
        hit_times_source_arrays: Materialized artifact-backed source arrays for the configured
            hit-times timeframe.
        one_minute_manifest: Fresh strict `prices/1m` manifest used for provenance hashing.
        max_hit_times_cells: Effective hit-times cell budget selected for this rebuild mode.
    Returns:
        _HitTimesArtifactBuildResultV2: Typed manifest plus root-manifest reference payload.
    Assumptions:
        Hit-times are derived only from already materialized `prices/1m` artifacts.
    Raises:
        ValueError: If computed tables violate the strict contract or exceed configured budgets.
        OSError: If writing arrays or manifest files fails.
    Side Effects:
        Writes `hit_times/15m/*.npy` and `hit_times/15m/manifest.yaml` under the inactive slot.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    hit_times_paths = artifact_loader.resolve_hit_times_paths(coordinates, slot)
    existing_hit_times_artifact = _load_existing_hit_times_artifact_v2(
        artifact_loader=artifact_loader,
        coordinates=coordinates,
        slot=slot,
        existing_manifest=existing_manifest,
        one_minute_manifest=one_minute_manifest,
        runtime_settings=runtime_settings,
    )
    tail_plan = _build_hit_times_tail_plan_v2(
        hit_times_source_arrays=hit_times_source_arrays,
        existing_hit_times_artifact=existing_hit_times_artifact,
        hit_times_tail_bars_1m=runtime_settings.hit_times_tail_bars_1m,
    )
    rebuilt_tail = materialize_hit_times_from_ohlcv_v2(
        ohlcv=hit_times_source_arrays.ohlcv[tail_plan.prefix_bars :, :],
        tp_levels_pct=runtime_settings.hit_times_tp_levels_pct,
        sl_levels_pct=runtime_settings.hit_times_sl_levels_pct,
        max_hit_times_cells=max_hit_times_cells,
    )
    hit_times_arrays = merge_hit_times_prefix_with_rebuilt_tail_v2(
        prefix=tail_plan.prefix,
        rebuilt_tail=rebuilt_tail,
        prefix_bars=tail_plan.prefix_bars,
        total_timeline_bars=int(hit_times_source_arrays.open_time.shape[0]),
    )
    _write_hit_times_arrays_atomically_v2(
        hit_times_paths=hit_times_paths,
        arrays=hit_times_arrays,
    )
    manifest = _build_hit_times_manifest_v2(
        coordinates=coordinates,
        slot=slot,
        slot_root=slot_root,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        one_minute_manifest=one_minute_manifest,
        expected_timeline_bar_count=int(hit_times_source_arrays.open_time.shape[0]),
        hit_times_paths=hit_times_paths,
        arrays=hit_times_arrays,
        effective_tail_bars=tail_plan.effective_tail_bars,
        max_hit_times_cells=max_hit_times_cells,
    )
    _write_yaml_atomically_v2(
        path=hit_times_paths.manifest,
        payload=_serialize_hit_times_manifest_v2(manifest),
    )
    return _HitTimesArtifactBuildResultV2(
        manifest=manifest,
        reference=ArtifactHitTimesReferenceV2(
            timeframe=manifest.timeframe,
            manifest_path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=hit_times_paths.manifest,
            ),
            manifest_sha256=_file_sha256_hex_v2(hit_times_paths.manifest),
        ),
        reused_prefix_bars=tail_plan.prefix_bars,
        rewritten_tail_bars=int(hit_times_arrays.sentinel_index - tail_plan.prefix_bars),
    )


def _write_hit_times_arrays_atomically_v2(
    *,
    hit_times_paths: ArtifactHitTimesPathsV2,
    arrays: HitTimesArraysV2,
) -> None:
    """
    Atomically replace inactive-slot `hit_times/15m/*.npy` files with deterministic bytes.

    Args:
        hit_times_paths: Explicit inactive-slot target paths for the hit-times family.
        arrays: Strict hit-times arrays to serialize.
    Returns:
        None.
    Assumptions:
        Temp files are written in the same directory so `os.replace` remains atomic.
    Raises:
        OSError: If temp-file write or atomic replace fails.
    Side Effects:
        Creates parent directories and replaces all six hit-times `.npy` files.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    _write_npy_atomically_v2(path=hit_times_paths.tp_values, array=arrays.tp_values)
    _write_npy_atomically_v2(path=hit_times_paths.sl_values, array=arrays.sl_values)
    _write_npy_atomically_v2(path=hit_times_paths.long_tp, array=arrays.long_tp)
    _write_npy_atomically_v2(path=hit_times_paths.long_sl, array=arrays.long_sl)
    _write_npy_atomically_v2(path=hit_times_paths.short_tp, array=arrays.short_tp)
    _write_npy_atomically_v2(path=hit_times_paths.short_sl, array=arrays.short_sl)


def _build_hit_times_manifest_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    slot_root: Path,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    expected_timeline_bar_count: int,
    hit_times_paths: ArtifactHitTimesPathsV2,
    arrays: HitTimesArraysV2,
    effective_tail_bars: int,
    max_hit_times_cells: int,
) -> ArtifactHitTimesManifestDocumentV2:
    """
    Build the strict typed `hit_times/15m/manifest.yaml` document for freshly written arrays.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        slot: Inactive slot literal receiving the manifest.
        slot_root: Absolute inactive-slot root directory.
        request: Explicit export request carrying root identity and timestamps.
        slot_generation: Target inactive-slot generation assigned to the build.
        runtime_settings: Strict runtime settings contributing config hash and hit-times grids.
        one_minute_manifest: Fresh strict `prices/1m` manifest used for provenance hashing.
        expected_timeline_bar_count: Expected timeline width for the configured hit-times
            timeframe.
        hit_times_paths: Fixed hit-times file paths under the inactive slot.
        arrays: Freshly written strict hit-times arrays.
        effective_tail_bars: Effective bounded `1m` tail overlap used for rebuild planning.
    Returns:
        ArtifactHitTimesManifestDocumentV2: Typed strict hit-times manifest.
    Assumptions:
        Hit-times files already exist on disk and are ready for `sha256` hashing.
    Raises:
        ValueError: If timeline counts drift from the configured hit-times timeframe or one
            metadata field is invalid.
        OSError: If one written hit-times file cannot be hashed.
    Side Effects:
        Reads the freshly written hit-times files to compute manifest hashes.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    validated_slot = validate_artifact_slot_v2(slot)
    timeline_bar_count = int(expected_timeline_bar_count)
    if arrays.sentinel_index != timeline_bar_count:
        raise ValueError(
            "hit-times sentinel_index must equal configured hit-times timeline bar_count; got "
            f"{arrays.sentinel_index!r}, expected {timeline_bar_count!r}"
        )
    table_time_count = int(arrays.long_tp.shape[1])
    if table_time_count != timeline_bar_count:
        raise ValueError(
            "hit-times timeline must match configured hit-times timeline bar_count; got "
            f"{table_time_count!r}, expected {timeline_bar_count!r}"
        )

    tp_values = ArtifactArrayMetadataV2(
        path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=hit_times_paths.tp_values),
        dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
        shape=tuple(int(value) for value in arrays.tp_values.shape),
        axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
        sha256=_file_sha256_hex_v2(hit_times_paths.tp_values),
    )
    sl_values = ArtifactArrayMetadataV2(
        path=_slot_relative_path_v2(slot_root=slot_root, absolute_path=hit_times_paths.sl_values),
        dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
        shape=tuple(int(value) for value in arrays.sl_values.shape),
        axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
        sha256=_file_sha256_hex_v2(hit_times_paths.sl_values),
    )
    long_tp = ArtifactHitTimesTableManifestV2(
        array=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=hit_times_paths.long_tp,
            ),
            dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            shape=tuple(int(value) for value in arrays.long_tp.shape),
            axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(hit_times_paths.long_tp),
        ),
        monotonicity=ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2,
    )
    long_sl = ArtifactHitTimesTableManifestV2(
        array=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=hit_times_paths.long_sl,
            ),
            dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            shape=tuple(int(value) for value in arrays.long_sl.shape),
            axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(hit_times_paths.long_sl),
        ),
        monotonicity=ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2,
    )
    short_tp = ArtifactHitTimesTableManifestV2(
        array=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=hit_times_paths.short_tp,
            ),
            dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            shape=tuple(int(value) for value in arrays.short_tp.shape),
            axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(hit_times_paths.short_tp),
        ),
        monotonicity=ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2,
    )
    short_sl = ArtifactHitTimesTableManifestV2(
        array=ArtifactArrayMetadataV2(
            path=_slot_relative_path_v2(
                slot_root=slot_root,
                absolute_path=hit_times_paths.short_sl,
            ),
            dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            shape=tuple(int(value) for value in arrays.short_sl.shape),
            axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            sha256=_file_sha256_hex_v2(hit_times_paths.short_sl),
        ),
        monotonicity=ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2,
    )
    provenance = _build_hit_times_manifest_provenance_v2(
        coordinates=coordinates,
        request=request,
        slot_generation=slot_generation,
        runtime_settings=runtime_settings,
        one_minute_manifest=one_minute_manifest,
        arrays=arrays,
        effective_tail_bars=effective_tail_bars,
        max_hit_times_cells=max_hit_times_cells,
    )
    payload = {
        "schema_version": HIT_TIMES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        "manifest_kind": HIT_TIMES_ARTIFACT_MANIFEST_KIND_V2,
        "slot": validated_slot,
        "slot_generation": slot_generation,
        "asof_date": request.asof_date,
        "timeframe": HIT_TIMES_TIMEFRAME_LITERAL_V2,
        "timeline_bar_count": timeline_bar_count,
        "sentinel_index": arrays.sentinel_index,
        "tp_values": _serialize_array_metadata_v2(tp_values),
        "sl_values": _serialize_array_metadata_v2(sl_values),
        "tables": {
            "long_tp": _serialize_hit_times_table_manifest_v2(long_tp),
            "long_sl": _serialize_hit_times_table_manifest_v2(long_sl),
            "short_tp": _serialize_hit_times_table_manifest_v2(short_tp),
            "short_sl": _serialize_hit_times_table_manifest_v2(short_sl),
        },
        "provenance": _serialize_provenance_v2(provenance),
    }
    return ArtifactHitTimesManifestDocumentV2(
        path=hit_times_paths.manifest,
        raw_payload=payload,
        slot=validated_slot,
        schema_version=HIT_TIMES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
        manifest_kind=HIT_TIMES_ARTIFACT_MANIFEST_KIND_V2,
        slot_generation=slot_generation,
        asof_date=request.asof_date,
        timeframe=HIT_TIMES_TIMEFRAME_LITERAL_V2,
        timeline_bar_count=timeline_bar_count,
        sentinel_index=arrays.sentinel_index,
        tp_values=tp_values,
        sl_values=sl_values,
        long_tp=long_tp,
        long_sl=long_sl,
        short_tp=short_tp,
        short_sl=short_sl,
        provenance=provenance,
    )


def _build_hit_times_manifest_provenance_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    arrays: HitTimesArraysV2,
    effective_tail_bars: int,
    max_hit_times_cells: int,
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic provenance for one strict `hit_times/15m` manifest.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        request: Explicit export request identity.
        slot_generation: Target inactive-slot generation assigned to the build.
        runtime_settings: Strict runtime settings contributing config hash and hit-times grids.
        one_minute_manifest: Fresh strict `prices/1m` manifest used as source-of-truth identity.
        arrays: Fresh hit-times arrays whose sentinel/timeline facts must be hashed.
        effective_tail_bars: Effective bounded `1m` tail overlap used for this rebuild.
        max_hit_times_cells: Effective hit-times cell budget selected for this rebuild mode.
    Returns:
        ArtifactManifestProvenanceV2: Strict hit-times-manifest provenance payload.
    Assumptions:
        Per-manifest provenance hashes source identity and configured grids, not YAML bytes.
    Raises:
        TypeError: If provenance hashing encounters an unsupported JSON payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    return ArtifactManifestProvenanceV2(
        generator=_PRECOMPUTE_GENERATOR_LITERAL_V2,
        generator_version=_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2,
        generated_at_utc=request.generated_at_utc,
        config_sha256=runtime_settings.config_sha256,
        inputs_sha256=_build_hit_times_manifest_inputs_sha256_v2(
            coordinates=coordinates,
            request=request,
            slot_generation=slot_generation,
            runtime_settings=runtime_settings,
            one_minute_manifest=one_minute_manifest,
            arrays=arrays,
            effective_tail_bars=effective_tail_bars,
            max_hit_times_cells=max_hit_times_cells,
        ),
    )


def _build_hit_times_manifest_inputs_sha256_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    arrays: HitTimesArraysV2,
    effective_tail_bars: int,
    max_hit_times_cells: int,
) -> str:
    """
    Hash normalized hit-times source identity into deterministic provenance.

    Args:
        coordinates: Artifact coordinates selecting one symbol root.
        request: Explicit export request identity.
        slot_generation: Target inactive-slot generation assigned to the build.
        runtime_settings: Strict runtime settings carrying hit-times grids and budgets.
        one_minute_manifest: Fresh strict `prices/1m` manifest used as source-of-truth identity.
        arrays: Fresh hit-times arrays whose sentinel/timeline facts must be hashed.
        effective_tail_bars: Effective bounded `1m` tail overlap used for this rebuild.
        max_hit_times_cells: Effective hit-times cell budget selected for this rebuild mode.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash tracks source identity and configured grids rather than manifest bytes.
    Raises:
        TypeError: If canonical JSON serialization receives an unsupported payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
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
            "timeframe": HIT_TIMES_TIMEFRAME_LITERAL_V2,
            "lookback_policy.hit_times_tail_bars_1m": runtime_settings.hit_times_tail_bars_1m,
            "effective_target_tail_bars": effective_tail_bars,
            "rebuild_strategy": "prefix + rebuilt_tail",
            "price_manifest_sha256": {
                "open_time": one_minute_manifest.open_time.sha256,
                "close_time": one_minute_manifest.close_time.sha256,
                "ohlcv": one_minute_manifest.ohlcv.sha256,
            },
            "hit_times_grid": {
                "tp_levels_pct": list(runtime_settings.hit_times_tp_levels_pct),
                "sl_levels_pct": list(runtime_settings.hit_times_sl_levels_pct),
            },
            "max_hit_times_cells": max_hit_times_cells,
            "timeline_bar_count": int(arrays.long_tp.shape[1]),
            "sentinel_index": arrays.sentinel_index,
            "tp_level_count": int(arrays.tp_values.shape[0]),
            "sl_level_count": int(arrays.sl_values.shape[0]),
            "table_cell_count": hit_times_table_cell_count_v2(
                timeline_bar_count=int(arrays.long_tp.shape[1]),
                tp_level_count=int(arrays.tp_values.shape[0]),
                sl_level_count=int(arrays.sl_values.shape[0]),
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
    Related:
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
        R3-01 keeps root-manifest schema strict without pretending that `hit_times/15m` already
        exists; later epics must replace this placeholder with a real manifest hash.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactSignalEncodingContractV2(
        dtype=ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
        axis_order=ARTIFACT_SIGNAL_AXIS_ORDER_V2,
        value_set=ARTIFACT_SIGNAL_VALUE_SET_V2,
    )


def _build_signal_features_manifest_provenance_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_shape: tuple[int, int],
    signal_paths: ArtifactSignalPathsV2,
    signal_features_paths: ArtifactSignalFeaturesPathsV2,
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic provenance for one strict additive signal-feature manifest.

    Args:
        request: Explicit export request carrying shared identity and timestamps.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        runtime_settings: Strict runtime settings contributing the config hash.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        signal_shape: Final source signal matrix shape `[V, T_tf]`.
        signal_paths: Fixed source signal family paths used for provenance hashing.
        signal_features_paths: Fixed signal-feature family paths under the inactive slot.
    Returns:
        ArtifactManifestProvenanceV2: Strict signal-feature manifest provenance payload.
    Assumptions:
        Feature provenance tracks only explicit source identity and the fixed feature schema, not
        YAML serialization bytes.
    Raises:
        OSError: If one source artifact hash cannot be read from disk.
    Side Effects:
        Reads source and feature artifact files to compute stable hashes.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactManifestProvenanceV2(
        generator=_PRECOMPUTE_GENERATOR_LITERAL_V2,
        generator_version=_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2,
        generated_at_utc=request.generated_at_utc,
        config_sha256=runtime_settings.config_sha256,
        inputs_sha256=_build_signal_features_manifest_inputs_sha256_v2(
            request=request,
            slot_generation=slot_generation,
            signal_target=signal_target,
            signal_shape=signal_shape,
            signal_paths=signal_paths,
            signal_features_paths=signal_features_paths,
        ),
    )


def _build_signal_features_manifest_inputs_sha256_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot_generation: int,
    signal_target: ArtifactSignalValidationSpecV2,
    signal_shape: tuple[int, int],
    signal_paths: ArtifactSignalPathsV2,
    signal_features_paths: ArtifactSignalFeaturesPathsV2,
) -> str:
    """
    Hash normalized signal-feature source identity into deterministic manifest provenance.

    Args:
        request: Explicit export request carrying root identity and timestamps.
        slot_generation: Deterministic generation assigned to the inactive slot build.
        signal_target: Explicit `(timeframe, indicator_id)` materialization target.
        signal_shape: Final source signal matrix shape `[V, T_tf]`.
        signal_paths: Fixed source signal family paths used for provenance hashing.
        signal_features_paths: Fixed signal-feature family paths under the inactive slot.
    Returns:
        str: Lowercase SHA-256 digest over normalized signal-feature manifest inputs.
    Assumptions:
        Feature provenance is driven by the immutable signal source plus the fixed feature schema.
    Raises:
        OSError: If one source artifact hash cannot be read from disk.
    Side Effects:
        Reads source and feature artifact files to compute stable hashes.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    normalized_payload = {
        "identity": {
            "exchange": request.coordinates.exchange,
            "market_type": request.coordinates.market_type,
            "symbol": request.coordinates.symbol,
        },
        "slot_generation": slot_generation,
        "asof_date": request.asof_date,
        "signal_target": {
            "timeframe": signal_target.timeframe,
            "indicator_id": signal_target.indicator_id,
        },
        "signal_shape": [int(signal_shape[0]), int(signal_shape[1])],
        "feature_names": [name for name in SIGNAL_FEATURE_NAMES_V2],
        "source_signal": {
            "path": signal_paths.signals.name,
            "sha256": _file_sha256_hex_v2(signal_paths.signals),
        },
        "feature_matrix": {
            "path": signal_features_paths.features.name,
            "sha256": _file_sha256_hex_v2(signal_features_paths.features),
        },
    }
    return hashlib.sha256(
        json.dumps(
            normalized_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    effective_tail_bars: int,
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
        effective_tail_bars: Effective target-timeframe tail window used for the rebuild.
    Returns:
        ArtifactManifestProvenanceV2: Strict signal-manifest provenance payload.
    Assumptions:
        Per-manifest provenance hashes source identity and row-order metadata, not YAML bytes.
    Raises:
        TypeError: If provenance hashing encounters an unsupported JSON payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
            signal_tail_bars_1m=runtime_settings.signal_tail_bars_1m,
            effective_tail_bars=effective_tail_bars,
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
    signal_tail_bars_1m: int,
    effective_tail_bars: int,
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
        signal_tail_bars_1m: Configured signal tail budget expressed in `1m` bars.
        effective_tail_bars: Effective target-timeframe tail window used for the rebuild.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash tracks source identity and row ordering rather than manifest serialization bytes.
    Raises:
        TypeError: If canonical JSON serialization receives an unsupported payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
            "lookback_policy.signal_tail_bars_1m": signal_tail_bars_1m,
            "effective_target_tail_bars": effective_tail_bars,
            "rebuild_strategy": "prefix + rebuilt_tail",
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
    hit_times_reference: ArtifactHitTimesReferenceV2 | None = None,
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic root-manifest provenance for emitted artifact families.

    Args:
        runtime_settings: Strict service runtime settings used by the precompute runner.
        request: Explicit export request identity.
        arrays: Final merged canonical `1m` arrays written into the inactive slot.
        rolled_sections: Rolled price-manifest sections emitted during the same R3-03 build.
        mapping_sections: Mapping-manifest sections emitted during the same R3-03 build.
        signal_entries: Signal catalog entries emitted during the same build when signals are
            enabled.
        hit_times_reference: Optional strict root hit-times reference emitted during the same
            build.
    Returns:
        ArtifactManifestProvenanceV2: Strict provenance payload for the root manifest.
    Assumptions:
        `inputs_sha256` identifies the normalized export request plus emitted artifact metadata
        derived from `market_data.canonical_candles_1m`.
    Raises:
        TypeError: If config hashing encounters an unsupported JSON payload.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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
            hit_times_reference=hit_times_reference,
            price_lookback_bars=runtime_settings.price_tail_bars_1m,
            mapping_lookback_bars=runtime_settings.mapping_tail_bars_1m,
            signal_lookback_bars=runtime_settings.signal_tail_bars_1m,
        ),
    )


def _build_inputs_sha256_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    arrays: _CanonicalPriceArraysV2,
    rolled_sections: tuple[ArtifactPriceTimeframeManifestV2, ...],
    mapping_sections: tuple[ArtifactMappingTimeframeManifestV2, ...],
    signal_entries: tuple[ArtifactSignalCatalogEntryV2, ...],
    hit_times_reference: ArtifactHitTimesReferenceV2 | None,
    price_lookback_bars: int,
    mapping_lookback_bars: int,
    signal_lookback_bars: int,
) -> str:
    """
    Hash normalized export identity and emitted arrays into deterministic provenance.

    Args:
        request: Explicit export request identity.
        arrays: Final merged canonical `1m` arrays emitted by the runner.
        rolled_sections: Rolled price-manifest sections emitted by the same build.
        mapping_sections: Mapping-manifest sections emitted by the same build.
        signal_entries: Signal catalog entries emitted by the same build.
        hit_times_reference: Optional strict hit-times reference emitted by the same build.
        price_lookback_bars: Effective `lookback_policy.price_tail_bars_1m` used for the build.
        mapping_lookback_bars: Effective `lookback_policy.mapping_tail_bars_1m` used for the
            build.
        signal_lookback_bars: Effective `lookback_policy.signal_tail_bars_1m` used for the
            build.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash is an emitted-artifact identity digest, not a runtime validation checksum.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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
            "lookback_policy.signal_tail_bars_1m": signal_lookback_bars,
            "rolled_price_timeframes": rolled_timeframes,
            "mapping_timeframes": tuple(section.timeframe for section in mapping_sections),
            "signal_targets": tuple(
                (entry.timeframe, entry.indicator_id) for entry in signal_entries
            ),
            "hit_times": (
                None
                if hit_times_reference is None
                else {
                    "timeframe": hit_times_reference.timeframe,
                    "manifest_path": hit_times_reference.manifest_path,
                    "manifest_sha256": hit_times_reference.manifest_sha256,
                }
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
    if hit_times_reference is not None:
        digest.update(hit_times_reference.timeframe.encode("ascii"))
        digest.update(hit_times_reference.manifest_path.encode("utf-8"))
        digest.update(hit_times_reference.manifest_sha256.encode("ascii"))
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    payload = {
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
    if manifest.signal_features is not None:
        payload["signal_features"] = _serialize_signal_features_reference_v2(
            manifest.signal_features
        )
    return payload


def _serialize_signal_features_manifest_v2(
    manifest: ArtifactSignalFeaturesManifestDocumentV2,
) -> dict[str, Any]:
    """
    Serialize one typed strict signal-feature manifest into deterministic YAML-ready payload order.

    Args:
        manifest: Typed strict signal-feature manifest.
    Returns:
        dict[str, Any]: YAML-ready signal-feature manifest payload.
    Assumptions:
        Manifest fields are already validated and use canonical slot-relative path literals.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
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
        "features": _serialize_array_metadata_v2(manifest.features),
        "rows_count": manifest.rows_count,
        "feature_names": [name for name in manifest.feature_names],
        "provenance": _serialize_provenance_v2(manifest.provenance),
    }


def _serialize_signal_features_reference_v2(
    reference: ArtifactSignalFeaturesReferenceV2,
) -> dict[str, Any]:
    """
    Serialize one typed signal-feature manifest reference into deterministic YAML-ready order.

    Args:
        reference: Typed signal-feature manifest reference.
    Returns:
        dict[str, Any]: YAML-ready reference payload.
    Assumptions:
        Signal manifests reference the additive feature family only through explicit manifest
        path/hash metadata.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "manifest_path": reference.manifest_path,
        "manifest_sha256": reference.manifest_sha256,
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return {
        "variant_key_version": grid.variant_key_version,
        "variant_keys_sha256": grid.variant_keys_sha256,
        "signals_v1_params_defaults": dict(grid.signals_v1_params_defaults),
    }


def _serialize_hit_times_table_manifest_v2(
    table_manifest: ArtifactHitTimesTableManifestV2,
) -> dict[str, Any]:
    """
    Serialize typed hit-times table metadata into deterministic YAML-ready payload order.

    Args:
        table_manifest: Typed hit-times table manifest payload.
    Returns:
        dict[str, Any]: YAML-ready hit-times table payload.
    Assumptions:
        Table metadata already uses canonical slot-relative paths and strict monotonicity literal.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    payload = _serialize_array_metadata_v2(table_manifest.array)
    payload["monotonicity"] = table_manifest.monotonicity
    return payload


def _serialize_hit_times_manifest_v2(
    manifest: ArtifactHitTimesManifestDocumentV2,
) -> dict[str, Any]:
    """
    Serialize one typed strict hit-times manifest into deterministic YAML-ready payload order.

    Args:
        manifest: Typed strict hit-times manifest.
    Returns:
        dict[str, Any]: YAML-ready hit-times manifest payload.
    Assumptions:
        Manifest fields are already validated and use canonical slot-relative path literals.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
        "timeframe": manifest.timeframe,
        "timeline_bar_count": manifest.timeline_bar_count,
        "sentinel_index": manifest.sentinel_index,
        "tp_values": _serialize_array_metadata_v2(manifest.tp_values),
        "sl_values": _serialize_array_metadata_v2(manifest.sl_values),
        "tables": {
            "long_tp": _serialize_hit_times_table_manifest_v2(manifest.long_tp),
            "long_sl": _serialize_hit_times_table_manifest_v2(manifest.long_sl),
            "short_tp": _serialize_hit_times_table_manifest_v2(manifest.short_tp),
            "short_sl": _serialize_hit_times_table_manifest_v2(manifest.short_sl),
        },
        "provenance": _serialize_provenance_v2(manifest.provenance),
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    slot_root: Path | None = None,
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
        slot_root: Explicit slot root used to validate manifest-relative paths when the artifact
            depth differs across families.
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    resolved_slot_root = expected_path.parents[2] if slot_root is None else slot_root
    if metadata.path != expected_path.relative_to(resolved_slot_root).as_posix():
        expected_relative_path = expected_path.relative_to(resolved_slot_root).as_posix()
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/shared_kernel/primitives/time_range.py
    """
    return f"{time_range.start} .. {time_range.end}"
