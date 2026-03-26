"""Deterministic R3-01 canonical `1m` price export into the inactive artifact slot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

from .contracts import (
    ARTIFACT_MANIFEST_FILENAME_V2,
    ARTIFACT_PLACEHOLDER_SHA256_V2,
    ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
    ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_VALUE_SET_V2,
    ARTIFACT_TIME_AXIS_ORDER_V2,
    HIT_TIMES_DIRECTORY_LITERAL_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    ArtifactArrayMetadataV2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactCoordinatesV2,
    ArtifactHitTimesReferenceV2,
    ArtifactManifestDocumentV2,
    ArtifactManifestProvenanceV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactPricePathsV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalCatalogV2,
    ArtifactSignalEncodingContractV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    artifact_market_id_from_coordinates_v2,
    inactive_artifact_slot_v2,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2 = "1m"
_CANONICAL_CANDLE_SOURCE_LITERAL_V2 = "market_data.canonical_candles_1m"
_PRECOMPUTE_GENERATOR_LITERAL_V2 = "backtest-artifact-precompute-runner-v2"
_PRECOMPUTE_GENERATOR_VERSION_LITERAL_V2 = "r3-01"


@dataclass(frozen=True, slots=True)
class _CanonicalPriceArraysV2:
    """
    Internal immutable container for canonical `1m` open/close/OHLCV arrays.

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
    Internal scaffold for root-manifest sections not owned by R3-01 `prices/1m`.

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
class BacktestArtifactPrecomputeRunnerV2:
    """
    Materialize canonical `1m` price arrays into the inactive slot for EPIC R3-01.

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

    def export_canonical_price_1m(
        self,
        request: ArtifactCanonicalPriceExportRequestV2,
    ) -> ArtifactCanonicalPriceExportResultV2:
        """
        Export canonical `1m` open/close/OHLCV arrays into the deterministic inactive slot.

        Args:
            request: Explicit export identity with symbol coordinates and `TimeRange [start, end)`.
        Returns:
            ArtifactCanonicalPriceExportResultV2: Structured write result for the inactive slot.
        Assumptions:
            R3-01 owns only `prices/1m/*` and root-manifest `1m` coverage; other sections are
            preserved or replaced with explicit placeholders.
        Raises:
            FileNotFoundError: If strict `current.yaml` is missing for the symbol root.
            ValueError: If existing inactive-slot metadata or source candles violate strict
                ordering/dtype/path contracts.
            OSError: If one atomic file write fails.
        Side Effects:
            Reads canonical candles through the port and atomically replaces inactive-slot files.
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
        _write_price_arrays_atomically_v2(price_paths=price_paths, arrays=materialized_arrays)
        one_minute_manifest = _build_one_minute_price_manifest_v2(
            slot_root=slot_root,
            price_paths=price_paths,
            arrays=materialized_arrays,
        )
        scaffold = _build_root_manifest_scaffold_v2(existing_manifest=existing_manifest)
        provenance = _build_root_manifest_provenance_v2(
            runtime_settings=self.runtime_settings,
            request=request,
            arrays=materialized_arrays,
        )
        root_manifest_payload = _build_root_manifest_payload_v2(
            request=request,
            slot=inactive_slot,
            slot_generation=target_slot_generation,
            root_scaffold=scaffold,
            one_minute_manifest=one_minute_manifest,
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
    if existing_manifest is None:
        return None
    existing_section = _select_price_manifest_v2(
        price_sections=existing_manifest.prices,
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
    )
    if existing_section is None:
        return None
    price_paths = artifact_loader.resolve_price_paths(
        coordinates,
        slot,
        _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
    )
    open_time = _load_validated_array_v2(
        metadata=existing_section.open_time,
        expected_path=price_paths.open_time,
        expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
        expected_shape=None,
        location="existing prices[1m].open_time",
    )
    close_time = _load_validated_array_v2(
        metadata=existing_section.close_time,
        expected_path=price_paths.close_time,
        expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
        expected_shape=None,
        location="existing prices[1m].close_time",
    )
    ohlcv = _load_validated_array_v2(
        metadata=existing_section.ohlcv,
        expected_path=price_paths.ohlcv,
        expected_dtype=ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
        expected_axis_order=ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
        expected_shape=None,
        location="existing prices[1m].ohlcv",
    )
    arrays = _CanonicalPriceArraysV2(
        open_time=np.ascontiguousarray(open_time, dtype=np.int64),
        close_time=np.ascontiguousarray(close_time, dtype=np.int64),
        ohlcv=np.ascontiguousarray(ohlcv, dtype=np.float32),
    )
    _validate_canonical_price_arrays_v2(
        arrays=arrays,
        label="existing canonical prices/1m",
    )
    expected_coverage = _timeline_coverage_from_arrays_v2(arrays=arrays)
    if existing_section.coverage != expected_coverage:
        raise ValueError(
            "existing prices[1m].coverage must match materialized arrays; "
            f"got {existing_section.coverage!r}, expected {expected_coverage!r}"
        )
    return arrays


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
    Validate deterministic dtype/shape/timeline invariants for canonical `1m` arrays.

    Args:
        arrays: Candidate `open_time/close_time/ohlcv` arrays.
        label: Stable human-readable label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        R3-01 stores timestamps separately from OHLCV and uses `volume_base` as the fifth field.
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
        raise ValueError(f"{label} open_time shape must be [T_1m]; got {arrays.open_time.shape!r}")
    if len(arrays.close_time.shape) != 1:
        raise ValueError(
            f"{label} close_time shape must be [T_1m]; got {arrays.close_time.shape!r}"
        )
    if arrays.ohlcv.ndim != 2 or arrays.ohlcv.shape[1] != 5:
        raise ValueError(
            f"{label} ohlcv shape must be [T_1m, 5]; got {arrays.ohlcv.shape!r}"
        )
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
    return ArtifactPriceTimeframeManifestV2(
        timeframe=_CANONICAL_PRICE_TIMEFRAME_LITERAL_V2,
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


def _build_root_manifest_scaffold_v2(
    *,
    existing_manifest: ArtifactManifestDocumentV2 | None,
) -> _RootManifestScaffoldV2:
    """
    Build the non-`prices/1m` root-manifest scaffold for R3-01 stage boundaries.

    Args:
        existing_manifest: Existing inactive-slot root manifest when one is already present.
    Returns:
        _RootManifestScaffoldV2: Preserved sections or explicit deterministic placeholders.
    Assumptions:
        R3-01 must keep root-manifest schema strict even when `mappings/signals/hit_times` are not
        materialized yet.
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
            if section.timeframe != _CANONICAL_PRICE_TIMEFRAME_LITERAL_V2
        ),
        mappings=existing_manifest.mappings,
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


def _build_root_manifest_provenance_v2(
    *,
    runtime_settings: ArtifactPrecomputeRuntimeSettingsV2,
    request: ArtifactCanonicalPriceExportRequestV2,
    arrays: _CanonicalPriceArraysV2,
) -> ArtifactManifestProvenanceV2:
    """
    Build deterministic root-manifest provenance for canonical `1m` export.

    Args:
        runtime_config: Strict artifact runtime config used by the precompute runner.
        request: Explicit export request identity.
        arrays: Final merged canonical arrays written into the inactive slot.
    Returns:
        ArtifactManifestProvenanceV2: Strict provenance payload for the root manifest.
    Assumptions:
        At R3-01 `inputs_sha256` identifies the normalized export request plus emitted canonical
        `1m` arrays derived from `market_data.canonical_candles_1m`.
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
            lookback_bars=runtime_settings.price_tail_bars_1m,
        ),
    )


def _build_inputs_sha256_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    arrays: _CanonicalPriceArraysV2,
    lookback_bars: int,
) -> str:
    """
    Hash normalized export identity and emitted arrays into deterministic provenance.

    Args:
        request: Explicit export request identity.
        arrays: Final merged canonical arrays emitted by the runner.
        lookback_bars: Effective `lookback_policy.price_tail_bars_1m` used for the build.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        The hash is an R3-01 input-identity digest, not a runtime validation checksum.
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
            "lookback_policy.price_tail_bars_1m": lookback_bars,
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
    return digest.hexdigest()


def _build_root_manifest_payload_v2(
    *,
    request: ArtifactCanonicalPriceExportRequestV2,
    slot: str,
    slot_generation: int,
    root_scaffold: _RootManifestScaffoldV2,
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
    provenance: ArtifactManifestProvenanceV2,
) -> dict[str, Any]:
    """
    Build the strict root `manifest.yaml` payload for R3-01 canonical `1m` export.

    Args:
        request: Explicit export request identity.
        slot: Inactive slot literal receiving the new root manifest.
        slot_generation: Target slot generation reserved for the next publish switch.
        root_scaffold: Preserved or placeholder non-price manifest sections.
        one_minute_manifest: Fresh strict `prices/1m` section.
        provenance: Deterministic root-manifest provenance payload.
    Returns:
        dict[str, Any]: Deterministic YAML payload ready for atomic serialization.
    Assumptions:
        R3-01 updates only the `1m` price base and root coverage while preserving strict schema.
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
        one_minute_manifest=one_minute_manifest,
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
        "mappings": [
            _serialize_mapping_manifest_v2(section) for section in root_scaffold.mappings
        ],
        "signals": _serialize_signal_catalog_v2(root_scaffold.signals),
        "hit_times": _serialize_hit_times_reference_v2(root_scaffold.hit_times),
        "signal_encoding": _serialize_signal_encoding_v2(root_scaffold.signal_encoding),
        "provenance": _serialize_provenance_v2(provenance),
    }


def _merge_price_sections_v2(
    *,
    preserved_prices: tuple[ArtifactPriceTimeframeManifestV2, ...],
    one_minute_manifest: ArtifactPriceTimeframeManifestV2,
) -> tuple[ArtifactPriceTimeframeManifestV2, ...]:
    """
    Merge preserved non-`1m` price sections with the freshly written strict `1m` section.

    Args:
        preserved_prices: Existing root price sections excluding `1m`.
        one_minute_manifest: Fresh `prices/1m` manifest section.
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
    merged_by_timeframe: dict[str, ArtifactPriceTimeframeManifestV2] = {
        one_minute_manifest.timeframe: one_minute_manifest
    }
    for section in preserved_prices:
        if section.timeframe in merged_by_timeframe:
            raise ValueError(
                "root manifest price sections contain duplicate timeframe "
                f"{section.timeframe!r}"
            )
        merged_by_timeframe[section.timeframe] = section
    timeframe_order = {
        literal: index
        for index, literal in enumerate(
            ("1m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "2d", "3d")
        )
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
