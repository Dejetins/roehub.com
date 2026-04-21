"""Explicit-path mmap loaders for prices, mappings, and `hit_times/1m` runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contracts import (
    ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
    ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
    ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
    ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
    ARTIFACT_TIME_AXIS_ORDER_V2,
    ArtifactArrayMetadataV2,
    ArtifactHitTimesArraysV2,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactMappingArraysV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPriceArraysV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSlotPinnedRuntimeContextV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    BacktestPriceArraysLoaderV2,
    validate_mapping_timeframe_v2,
    validate_price_timeframe_v2,
)


@dataclass(frozen=True, slots=True)
class MmapPriceArraysLoaderV2(BacktestPriceArraysLoaderV2):
    """
    Load runtime-side price families through explicit manifest-driven mmap paths only.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """

    artifact_loader: BacktestArtifactLoaderV2
    _price_cache: dict[
        tuple[Path, int, str, str],
        ArtifactPriceArraysV2,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _mapping_cache: dict[
        tuple[Path, int, str, str],
        ArtifactMappingArraysV2,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _hit_times_cache: dict[
        tuple[Path, int, str, str],
        ArtifactHitTimesArraysV2,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def run_scoped(self) -> MmapPriceArraysLoaderV2:
        """
        Build one `run-scoped` price loader that owns fresh mmap caches for a single caller.

        Args:
            None.
        Returns:
            MmapPriceArraysLoaderV2: Fresh loader sharing the same strict `artifact_loader`
                wiring and starting with empty price, mapping, and hit-times caches.
        Assumptions:
            Caller defines the returned loader lifetime and discards it after the owning runtime
            scope finishes.
        Raises:
            None.
        Side Effects:
            None.
        """
        return MmapPriceArraysLoaderV2(artifact_loader=self.artifact_loader)

    def load_price_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactPriceArraysV2:
        """
        Load one explicit `prices/<tf>` artifact family via `np.load(..., mmap_mode='r')`.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested artifact price timeframe.
        Returns:
            ArtifactPriceArraysV2: Cached-or-new memory-mapped price arrays and strict manifest
                metadata.
        Assumptions:
            Runtime must read price arrays only from root-manifest metadata and deterministic
            paths under the already pinned slot.
        Raises:
            FileNotFoundError: If one explicit `prices/<tf>` file is missing.
            ValueError: If manifest path/dtype/shape/axis/timeline metadata drifts from files.
        Side Effects:
            May memory-map three `.npy` files from the pinned slot on first access and reuse the
            validated payload afterwards.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        validated_timeframe = validate_price_timeframe_v2(timeframe)
        cache_key = _artifact_family_cache_key_v2(
            context=context,
            family_literal="prices",
            member_literal=validated_timeframe,
        )
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            return cached
        manifest = _price_manifest_for_timeframe(
            context=context,
            timeframe=validated_timeframe,
        )
        price_paths = self.artifact_loader.resolve_price_paths(
            context.coordinates,
            context.artifact_slot,
            validated_timeframe,
        )
        expected_bar_count = manifest.coverage.bar_count
        open_time = _load_mmap_array(
            context=context,
            metadata=manifest.open_time,
            expected_path=price_paths.open_time,
            expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(expected_bar_count,),
            location=f"prices/{validated_timeframe}/open_time",
        )
        close_time = _load_mmap_array(
            context=context,
            metadata=manifest.close_time,
            expected_path=price_paths.close_time,
            expected_dtype=ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(expected_bar_count,),
            location=f"prices/{validated_timeframe}/close_time",
        )
        ohlcv = _load_mmap_array(
            context=context,
            metadata=manifest.ohlcv,
            expected_path=price_paths.ohlcv,
            expected_dtype=ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
            expected_shape=(expected_bar_count, 5),
            location=f"prices/{validated_timeframe}/ohlcv",
        )
        _validate_price_timeline(
            timeframe=validated_timeframe,
            coverage=manifest.coverage,
            open_time=open_time,
            close_time=close_time,
            ohlcv=ohlcv,
        )
        loaded = ArtifactPriceArraysV2(
            timeframe=validated_timeframe,
            manifest=manifest,
            open_time=open_time,
            close_time=close_time,
            ohlcv=ohlcv,
        )
        self._price_cache[cache_key] = loaded
        return loaded

    def load_mapping_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactMappingArraysV2:
        """
        Load one explicit `mappings/<tf>` artifact family via `np.load(..., mmap_mode='r')`.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested artifact mapping timeframe.
        Returns:
            ArtifactMappingArraysV2: Cached-or-new memory-mapped mapping arrays and strict
                manifest metadata.
        Assumptions:
            Mapping arrays are rooted in already pinned slot metadata and must not be discovered
            by filesystem scanning.
        Raises:
            FileNotFoundError: If one explicit `mappings/<tf>` file is missing.
            ValueError: If manifest path/dtype/shape/axis/timeline metadata drifts from files.
        Side Effects:
            May memory-map two `.npy` files from the pinned slot on first access and reuse the
            validated payload afterwards.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        validated_timeframe = validate_mapping_timeframe_v2(timeframe)
        cache_key = _artifact_family_cache_key_v2(
            context=context,
            family_literal="mappings",
            member_literal=validated_timeframe,
        )
        cached = self._mapping_cache.get(cache_key)
        if cached is not None:
            return cached
        manifest = _mapping_manifest_for_timeframe(
            context=context,
            timeframe=validated_timeframe,
        )
        one_minute_manifest = _price_manifest_for_timeframe(context=context, timeframe="1m")
        target_price_manifest = _price_manifest_for_timeframe(
            context=context,
            timeframe=validated_timeframe,
        )
        mapping_paths = self.artifact_loader.resolve_mapping_paths(
            context.coordinates,
            context.artifact_slot,
            validated_timeframe,
        )
        expected_shape = (target_price_manifest.coverage.bar_count,)
        bar_open_1m_idx = _load_mmap_array(
            context=context,
            metadata=manifest.bar_open_1m_idx,
            expected_path=mapping_paths.bar_open_1m_idx,
            expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=expected_shape,
            location=f"mappings/{validated_timeframe}/bar_open_1m_idx",
        )
        bar_close_1m_idx = _load_mmap_array(
            context=context,
            metadata=manifest.bar_close_1m_idx,
            expected_path=mapping_paths.bar_close_1m_idx,
            expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=expected_shape,
            location=f"mappings/{validated_timeframe}/bar_close_1m_idx",
        )
        _validate_mapping_contract(
            timeframe=validated_timeframe,
            one_minute_bar_count=one_minute_manifest.coverage.bar_count,
            target_bar_count=target_price_manifest.coverage.bar_count,
            bar_open_1m_idx=bar_open_1m_idx,
            bar_close_1m_idx=bar_close_1m_idx,
        )
        loaded = ArtifactMappingArraysV2(
            timeframe=validated_timeframe,
            manifest=manifest,
            bar_open_1m_idx=bar_open_1m_idx,
            bar_close_1m_idx=bar_close_1m_idx,
        )
        self._mapping_cache[cache_key] = loaded
        return loaded

    def load_hit_times_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> ArtifactHitTimesArraysV2:
        """
        Load strict `hit_times/1m` arrays from explicit manifest-driven paths.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
        Returns:
            ArtifactHitTimesArraysV2: Cached-or-new memory-mapped hit-times arrays and strict
                manifest metadata.
        Assumptions:
            Runtime must consume shipped `hit_times/1m` artifacts only by explicit manifest path
            and fixed metadata; no recompute or directory discovery is allowed.
        Raises:
            FileNotFoundError: If hit-times manifest or one referenced `.npy` file is missing.
            ValueError: If manifest path/dtype/shape/axis/timeline metadata drifts from files.
        Side Effects:
            May memory-map six `.npy` files from the pinned slot on first access and reuse the
            validated payload afterwards.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        cache_key = _artifact_family_cache_key_v2(
            context=context,
            family_literal="hit_times",
            member_literal="1m",
        )
        cached = self._hit_times_cache.get(cache_key)
        if cached is not None:
            return cached
        hit_times_manifest_path = (
            context.slot_root_path / context.slot_manifest.hit_times.manifest_path
        )
        hit_times_manifest = self.artifact_loader.load_hit_times_manifest_from_path(
            hit_times_manifest_path,
            slot=context.artifact_slot,
        )
        hit_times_paths = self.artifact_loader.resolve_hit_times_paths(
            context.coordinates,
            context.artifact_slot,
        )
        tp_values = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.tp_values,
            expected_path=hit_times_paths.tp_values,
            expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
            expected_shape=(hit_times_manifest.tp_values.shape[0],),
            location="hit_times/1m/tp_values",
        )
        sl_values = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.sl_values,
            expected_path=hit_times_paths.sl_values,
            expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
            expected_shape=(hit_times_manifest.sl_values.shape[0],),
            location="hit_times/1m/sl_values",
        )
        timeline_shape = (hit_times_manifest.timeline_bar_count,)
        long_tp = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.long_tp.array,
            expected_path=hit_times_paths.long_tp,
            expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            expected_shape=(tp_values.shape[0], timeline_shape[0]),
            location="hit_times/1m/long_tp",
        )
        long_sl = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.long_sl.array,
            expected_path=hit_times_paths.long_sl,
            expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            expected_shape=(sl_values.shape[0], timeline_shape[0]),
            location="hit_times/1m/long_sl",
        )
        short_tp = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.short_tp.array,
            expected_path=hit_times_paths.short_tp,
            expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            expected_shape=(tp_values.shape[0], timeline_shape[0]),
            location="hit_times/1m/short_tp",
        )
        short_sl = _load_mmap_array(
            context=context,
            metadata=hit_times_manifest.short_sl.array,
            expected_path=hit_times_paths.short_sl,
            expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            expected_shape=(sl_values.shape[0], timeline_shape[0]),
            location="hit_times/1m/short_sl",
        )
        _validate_hit_times_contract(
            hit_times_manifest=hit_times_manifest,
            tp_values=tp_values,
            sl_values=sl_values,
            long_tp=long_tp,
            long_sl=long_sl,
            short_tp=short_tp,
            short_sl=short_sl,
        )
        loaded = ArtifactHitTimesArraysV2(
            manifest=hit_times_manifest,
            tp_values=tp_values,
            sl_values=sl_values,
            long_tp=long_tp,
            long_sl=long_sl,
            short_tp=short_tp,
            short_sl=short_sl,
        )
        self._hit_times_cache[cache_key] = loaded
        return loaded


def _artifact_family_cache_key_v2(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    family_literal: str,
    member_literal: str,
) -> tuple[Path, int, str, str]:
    """
    Build a run-local cache key for one validated artifact family inside a pinned slot.

    Args:
        context: Shared slot-pinned runtime context resolved at startup.
        family_literal: Stable artifact family literal such as `prices` or `signals`.
        member_literal: Family member discriminator such as timeframe or indicator id.
    Returns:
        tuple[Path, int, str, str]: Hashable key unique to one pinned family payload.
    Assumptions:
        Published slots are immutable once pinned, so reusing already validated mmap objects is
        safe within one loader instance.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """
    return (
        context.slot_root_path,
        context.slot_generation,
        context.artifact_manifest_hash,
        f"{family_literal}/{member_literal}",
    )


def _price_manifest_for_timeframe(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
) -> ArtifactPriceTimeframeManifestV2:
    """
    Look up one price-manifest section by timeframe from the pinned root manifest.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        timeframe: Canonical price timeframe literal.
    Returns:
        ArtifactPriceTimeframeManifestV2: Matching strict price-manifest section.
    Assumptions:
        Root manifest is the only source of truth for runtime artifact family membership.
    Raises:
        ValueError: If the requested timeframe is absent from the pinned root manifest.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    for manifest in context.slot_manifest.prices:
        if manifest.timeframe == timeframe:
            return manifest
    raise ValueError(f"slot-pinned context is missing prices/{timeframe} manifest metadata")


def _mapping_manifest_for_timeframe(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
) -> ArtifactMappingTimeframeManifestV2:
    """
    Look up one mapping-manifest section by timeframe from the pinned root manifest.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        timeframe: Canonical mapping timeframe literal.
    Returns:
        ArtifactMappingTimeframeManifestV2: Matching strict mapping-manifest section.
    Assumptions:
        Mapping families are enumerated explicitly in the pinned root manifest.
    Raises:
        ValueError: If the requested timeframe is absent from the pinned root manifest.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    for manifest in context.slot_manifest.mappings:
        if manifest.timeframe == timeframe:
            return manifest
    raise ValueError(f"slot-pinned context is missing mappings/{timeframe} manifest metadata")


def _load_mmap_array(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    metadata: ArtifactArrayMetadataV2,
    expected_path: Path,
    expected_dtype: str,
    expected_axis_order: tuple[str, ...],
    expected_shape: tuple[int, ...],
    location: str,
) -> np.ndarray:
    """
    Load one `.npy` file through strict metadata and explicit mmap path validation.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        metadata: Strict array metadata from the relevant manifest section.
        expected_path: Deterministic absolute path expected for this artifact family.
        expected_dtype: Required runtime dtype literal.
        expected_axis_order: Required runtime axis-order tuple.
        expected_shape: Required runtime shape tuple.
        location: Human-readable artifact location used in stable diagnostics.
    Returns:
        np.ndarray: Memory-mapped numpy array opened in read-only mode.
    Assumptions:
        Runtime loaders verify metadata path/dtype/shape/axis_order but intentionally avoid hot
        path SHA-256 recomputation loops.
    Raises:
        FileNotFoundError: If the explicit artifact file does not exist.
        ValueError: If manifest path/dtype/shape/axis metadata drifts from the file contract.
    Side Effects:
        Memory-maps one `.npy` file from disk with `allow_pickle=False`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    expected_relative_path = _relative_slot_path(context=context, absolute_path=expected_path)
    if metadata.path != expected_relative_path:
        raise ValueError(
            f"{location} manifest path must be {expected_relative_path!r}; got {metadata.path!r}"
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
    if metadata.shape != expected_shape:
        raise ValueError(
            f"{location} manifest shape must be {expected_shape!r}; got {metadata.shape!r}"
        )
    if not expected_path.is_file():
        raise FileNotFoundError(f"{location} artifact file is missing: {expected_path}")
    array = np.load(expected_path, mmap_mode='r', allow_pickle=False)
    actual_shape = tuple(int(value) for value in array.shape)
    if actual_shape != metadata.shape:
        raise ValueError(
            f"{location} file shape must match manifest metadata; got {actual_shape!r}, "
            f"expected {metadata.shape!r}"
        )
    if array.dtype.name != expected_dtype:
        raise ValueError(
            f"{location} file dtype must be {expected_dtype!r}; got {array.dtype.name!r}"
        )
    return array


def _relative_slot_path(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    absolute_path: Path,
) -> str:
    """
    Convert one explicit absolute slot path into the canonical slot-relative manifest literal.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        absolute_path: Absolute artifact file path under the pinned slot root.
    Returns:
        str: Canonical POSIX-style slot-relative artifact path.
    Assumptions:
        Every runtime artifact file must live strictly under the pinned slot root.
    Raises:
        ValueError: If the path does not belong to the pinned slot root.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    try:
        return absolute_path.relative_to(context.slot_root_path).as_posix()
    except ValueError as error:
        raise ValueError(
            f"{absolute_path} must stay under pinned slot root {context.slot_root_path}"
        ) from error


def _validate_price_timeline(
    *,
    timeframe: str,
    coverage: ArtifactTimelineCoverageV2,
    open_time: np.ndarray,
    close_time: np.ndarray,
    ohlcv: np.ndarray,
) -> None:
    """
    Reject timeline drift between price arrays and strict coverage metadata.

    Args:
        timeframe: Canonical price timeframe literal used in diagnostics.
        coverage: Strict timeline coverage from the root manifest.
        open_time: Memory-mapped `open_time` array.
        close_time: Memory-mapped `close_time` array.
        ohlcv: Memory-mapped `ohlcv` matrix.
    Returns:
        None.
    Assumptions:
        Runtime price loading must fail fast on timeline/count drift before downstream kernels use
        the arrays.
    Raises:
        ValueError: If bar counts, boundary timestamps, or monotone ordering are violated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    actual_bar_count = int(open_time.shape[0])
    if int(close_time.shape[0]) != actual_bar_count or int(ohlcv.shape[0]) != actual_bar_count:
        raise ValueError(
            f"prices/{timeframe} arrays must share the same bar count; got "
            f"{open_time.shape[0]!r}, {close_time.shape[0]!r}, {ohlcv.shape[0]!r}"
        )
    if actual_bar_count != coverage.bar_count:
        raise ValueError(
            f"prices/{timeframe} bar_count must match manifest coverage; got "
            f"{actual_bar_count!r}, expected {coverage.bar_count!r}"
        )
    if (
        int(open_time[0]) != coverage.open_time_start
        or int(open_time[-1]) != coverage.open_time_end
    ):
        raise ValueError(
            f"prices/{timeframe} open_time boundaries must match manifest coverage; got "
            f"{int(open_time[0])!r}/{int(open_time[-1])!r}, expected "
            f"{coverage.open_time_start!r}/{coverage.open_time_end!r}"
        )
    if (
        int(close_time[0]) != coverage.close_time_start
        or int(close_time[-1]) != coverage.close_time_end
    ):
        raise ValueError(
            f"prices/{timeframe} close_time boundaries must match manifest coverage; got "
            f"{int(close_time[0])!r}/{int(close_time[-1])!r}, expected "
            f"{coverage.close_time_start!r}/{coverage.close_time_end!r}"
        )
    if actual_bar_count > 1 and not np.all(open_time[1:] >= open_time[:-1]):
        raise ValueError(f"prices/{timeframe} open_time must be monotonically non-decreasing")
    if actual_bar_count > 1 and not np.all(close_time[1:] >= close_time[:-1]):
        raise ValueError(f"prices/{timeframe} close_time must be monotonically non-decreasing")


def _validate_mapping_contract(
    *,
    timeframe: str,
    one_minute_bar_count: int,
    target_bar_count: int,
    bar_open_1m_idx: np.ndarray,
    bar_close_1m_idx: np.ndarray,
) -> None:
    """
    Reject mapping drift relative to target timeframe length and `prices/1m` coverage.

    Args:
        timeframe: Canonical mapping timeframe literal used in diagnostics.
        one_minute_bar_count: Total `prices/1m` bar count from root-manifest coverage.
        target_bar_count: Total target timeframe bar count from root-manifest coverage.
        bar_open_1m_idx: Memory-mapped open-index mapping array.
        bar_close_1m_idx: Memory-mapped close-index mapping array.
    Returns:
        None.
    Assumptions:
        Runtime mapping arrays must remain bounded by `prices/1m` and aligned by target bars.
    Raises:
        ValueError: If bar counts, monotonicity, bounds, or open/close ordering drift.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if (
        int(bar_open_1m_idx.shape[0]) != target_bar_count
        or int(bar_close_1m_idx.shape[0]) != target_bar_count
    ):
        raise ValueError(
            f"mappings/{timeframe} row count must match prices/{timeframe}; got "
            f"{bar_open_1m_idx.shape[0]!r}/{bar_close_1m_idx.shape[0]!r}, expected "
            f"{target_bar_count!r}"
        )
    if target_bar_count > 1 and not np.all(bar_open_1m_idx[1:] >= bar_open_1m_idx[:-1]):
        raise ValueError(f"mappings/{timeframe} bar_open_1m_idx must be monotone")
    if target_bar_count > 1 and not np.all(bar_close_1m_idx[1:] >= bar_close_1m_idx[:-1]):
        raise ValueError(f"mappings/{timeframe} bar_close_1m_idx must be monotone")
    if np.any(bar_open_1m_idx > bar_close_1m_idx):
        raise ValueError(f"mappings/{timeframe} bar_open_1m_idx must be <= bar_close_1m_idx")
    if np.any(bar_close_1m_idx >= np.uint32(one_minute_bar_count)):
        raise ValueError(
            f"mappings/{timeframe} indexes must stay within prices/1m bounds {one_minute_bar_count}"
        )


def _validate_hit_times_contract(
    *,
    hit_times_manifest: ArtifactHitTimesManifestDocumentV2,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
    long_tp: np.ndarray,
    long_sl: np.ndarray,
    short_tp: np.ndarray,
    short_sl: np.ndarray,
) -> None:
    """
    Reject timeline and level drift inside the strict `hit_times/1m` artifact family.

    Args:
        hit_times_manifest: Strict loaded `hit_times/1m/manifest.yaml`.
        tp_values: Memory-mapped TP level grid.
        sl_values: Memory-mapped SL level grid.
        long_tp: Memory-mapped `long_tp` table.
        long_sl: Memory-mapped `long_sl` table.
        short_tp: Memory-mapped `short_tp` table.
        short_sl: Memory-mapped `short_sl` table.
    Returns:
        None.
    Assumptions:
        R5-01 fixed `sentinel_index == timeline_bar_count`, level axes are explicit, and runtime
        only needs fail-fast drift detection here.
    Raises:
        ValueError: If sentinel, level counts, or table timeline widths drift from the manifest.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if hit_times_manifest.sentinel_index != hit_times_manifest.timeline_bar_count:
        raise ValueError(
            "hit_times/1m sentinel_index must equal timeline_bar_count; got "
            f"{hit_times_manifest.sentinel_index!r} and "
            f"{hit_times_manifest.timeline_bar_count!r}"
        )
    if long_tp.shape[0] != tp_values.shape[0] or short_tp.shape[0] != tp_values.shape[0]:
        raise ValueError("hit_times/1m TP table level count must match tp_values length")
    if long_sl.shape[0] != sl_values.shape[0] or short_sl.shape[0] != sl_values.shape[0]:
        raise ValueError("hit_times/1m SL table level count must match sl_values length")
    expected_timeline_width = hit_times_manifest.timeline_bar_count
    for table_name, table in (
        ("long_tp", long_tp),
        ("long_sl", long_sl),
        ("short_tp", short_tp),
        ("short_sl", short_sl),
    ):
        if int(table.shape[1]) != expected_timeline_width:
            raise ValueError(
                f"hit_times/1m/{table_name} width must match timeline_bar_count; got "
                f"{table.shape[1]!r}, expected {expected_timeline_width!r}"
            )
