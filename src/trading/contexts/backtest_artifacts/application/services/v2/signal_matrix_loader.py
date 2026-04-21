"""Explicit-path mmap loaders for strict signal matrices and deterministic subset row reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contracts import (
    ARTIFACT_SIGNAL_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
    ArtifactArrayMetadataV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalMatrixV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestSignalMatrixLoaderV2,
    validate_indicator_id_v2,
    validate_signal_timeframe_v2,
)


@dataclass(frozen=True, slots=True)
class MmapSignalMatrixLoaderV2(BacktestSignalMatrixLoaderV2):
    """
    Load runtime-side signal matrices through explicit manifest-driven mmap paths only.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """

    artifact_loader: BacktestArtifactLoaderV2
    _signal_matrix_cache: dict[
        tuple[Path, int, str, str],
        ArtifactSignalMatrixV2,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def run_scoped(self) -> MmapSignalMatrixLoaderV2:
        """
        Build one `run-scoped` signal loader that owns fresh mmap caches for a single caller.

        Args:
            None.
        Returns:
            MmapSignalMatrixLoaderV2: Fresh loader sharing the same strict `artifact_loader`
                wiring and starting with an empty signal-matrix cache.
        Assumptions:
            Caller defines the returned loader lifetime and discards it after the owning runtime
            scope finishes.
        Raises:
            None.
        Side Effects:
            None.
        """
        return MmapSignalMatrixLoaderV2(artifact_loader=self.artifact_loader)

    def load_signal_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalMatrixV2:
        """
        Load one explicit signal matrix via `np.load(..., mmap_mode='r')`.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
        Returns:
            ArtifactSignalMatrixV2: Cached-or-new memory-mapped signal matrix and strict manifest
                metadata.
        Assumptions:
            Runtime must read signals only from root-manifest catalog entries and explicit paths
            under the already pinned slot.
        Raises:
            FileNotFoundError: If signal manifest or `signals.i8.npy` is missing.
            ValueError: If manifest path/dtype/shape/axis/timeline metadata drifts from files.
        Side Effects:
            Reads one strict signal manifest and may memory-map one `.npy` file from disk on the
            first access for this pinned matrix.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        validated_timeframe = validate_signal_timeframe_v2(timeframe)
        validated_indicator_id = validate_indicator_id_v2(indicator_id)
        cache_key = _signal_matrix_cache_key_v2(
            context=context,
            timeframe=validated_timeframe,
            indicator_id=validated_indicator_id,
        )
        cached = self._signal_matrix_cache.get(cache_key)
        if cached is not None:
            return cached
        catalog_entry = _signal_catalog_entry(
            context=context,
            timeframe=validated_timeframe,
            indicator_id=validated_indicator_id,
        )
        signal_paths = self.artifact_loader.resolve_signal_paths(
            context.coordinates,
            context.artifact_slot,
            validated_timeframe,
            validated_indicator_id,
        )
        expected_manifest_relative_path = _relative_slot_path(
            context=context,
            absolute_path=signal_paths.manifest,
        )
        if catalog_entry.manifest_path != expected_manifest_relative_path:
            raise ValueError(
                "signals catalog manifest_path must match the deterministic explicit path; got "
                f"{catalog_entry.manifest_path!r}, expected {expected_manifest_relative_path!r}"
            )
        signal_manifest = self.artifact_loader.load_signal_manifest_from_path(
            signal_paths.manifest,
            slot=context.artifact_slot,
        )
        price_manifest = _price_manifest_for_signal_timeframe(
            context=context,
            timeframe=validated_timeframe,
        )
        if signal_manifest.timeline != price_manifest.coverage:
            raise ValueError(
                f"signals/{validated_timeframe}/{validated_indicator_id} timeline must match "
                f"prices/{validated_timeframe} coverage"
            )
        if signal_manifest.signal_value_set != context.slot_manifest.signal_encoding.value_set:
            raise ValueError(
                "signal manifest value_set must match root signal_encoding value_set; got "
                f"{signal_manifest.signal_value_set!r}, expected "
                f"{context.slot_manifest.signal_encoding.value_set!r}"
            )
        matrix = _load_mmap_array(
            context=context,
            metadata=signal_manifest.signals,
            expected_path=signal_paths.signals,
            expected_dtype=ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_SIGNAL_AXIS_ORDER_V2,
            expected_shape=(signal_manifest.rows_count, price_manifest.coverage.bar_count),
            location=f"signals/{validated_timeframe}/{validated_indicator_id}/signals.i8.npy",
        )
        loaded = ArtifactSignalMatrixV2(
            timeframe=validated_timeframe,
            indicator_id=validated_indicator_id,
            manifest=signal_manifest,
            matrix=matrix,
        )
        self._signal_matrix_cache[cache_key] = loaded
        return loaded

    def load_signal_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> np.ndarray:
        """
        Load a deterministic subset of signal rows without runtime discovery.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
            row_selection: Either a positive-step `slice` or a strictly increasing row-index tuple.
        Returns:
            np.ndarray: Selected signal rows preserving deterministic caller ordering.
        Assumptions:
            Basic slices and contiguous explicit row tuples may return mmap-backed views, while
            non-contiguous explicit row tuples return stable advanced-index selections without
            loading unrelated artifact families.
        Raises:
            FileNotFoundError: If the underlying signal matrix file is missing.
            ValueError: If the row selection or manifest/file contract is invalid.
        Side Effects:
            May memory-map one `.npy` file from disk on first access and returns a slice or
            indexed selection from the cached strict matrix.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
        """
        signal_matrix = self.load_signal_matrix(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        normalized_selection = _normalize_row_selection(
            row_selection=row_selection,
            row_count=signal_matrix.manifest.rows_count,
        )
        return signal_matrix.matrix[normalized_selection, :]


def _signal_catalog_entry(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
    indicator_id: str,
) -> ArtifactSignalCatalogEntryV2:
    """
    Look up one signal catalog entry from the pinned root manifest.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        timeframe: Canonical signal timeframe literal.
        indicator_id: Canonical indicator identifier.
    Returns:
        ArtifactSignalCatalogEntryV2: Matching strict signal catalog entry.
    Assumptions:
        Root manifest signals catalog is the only source of truth for runtime signal discovery.
    Raises:
        ValueError: If the requested `(timeframe, indicator_id)` entry is absent.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    for entry in context.slot_manifest.signals.manifests:
        if entry.timeframe == timeframe and entry.indicator_id == indicator_id:
            return entry
    raise ValueError(
        "slot-pinned context is missing "
        f"signals/{timeframe}/{indicator_id}/manifest.yaml metadata"
    )


def _price_manifest_for_signal_timeframe(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
):
    """
    Look up the price coverage that a signal artifact timeline must match.

    Args:
        context: Shared slot-pinned context resolved at runtime start.
        timeframe: Canonical signal timeframe literal.
    Returns:
        ArtifactPriceTimeframeManifestV2: Matching strict price-manifest section.
    Assumptions:
        Signal timelines are anchored to the same `prices/<tf>` coverage published in the root
        manifest.
    Raises:
        ValueError: If the matching `prices/<tf>` section is absent from the root manifest.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """
    for manifest in context.slot_manifest.prices:
        if manifest.timeframe == timeframe:
            return manifest
    raise ValueError(f"slot-pinned context is missing prices/{timeframe} coverage metadata")


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
        metadata: Strict array metadata from the signal manifest.
        expected_path: Deterministic absolute path expected for this signal matrix.
        expected_dtype: Required runtime dtype literal.
        expected_axis_order: Required runtime axis-order tuple.
        expected_shape: Required runtime shape tuple.
        location: Human-readable artifact location used in stable diagnostics.
    Returns:
        np.ndarray: Memory-mapped numpy array opened in read-only mode.
    Assumptions:
        Runtime loaders verify manifest path/dtype/shape/axis_order but intentionally avoid
        hot-path SHA-256 recomputation loops.
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
      - src/trading/contexts/backtest/application/services/v2/contracts.py
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


def _normalize_row_selection(
    *,
    row_selection: slice | tuple[int, ...],
    row_count: int,
) -> slice | tuple[int, ...]:
    """
    Normalize and validate deterministic subset row selection for signal matrices.

    Args:
        row_selection: Either a positive-step `slice` or a tuple of explicit row indexes.
        row_count: Total available row count in the signal matrix.
    Returns:
        slice | tuple[int, ...]: Validated selection preserving deterministic caller order.
    Assumptions:
        Runtime subset reads must stay explicit and reject ambiguous ordering or out-of-range rows.
    Raises:
        ValueError: If slice step is non-positive or explicit row indexes are unsorted/invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """
    if isinstance(row_selection, slice):
        start, stop, step = row_selection.indices(row_count)
        if step <= 0:
            raise ValueError("signal row slice step must be > 0")
        return slice(start, stop, step)

    normalized: list[int] = []
    previous_index: int | None = None
    for index in row_selection:
        if index < 0 or index >= row_count:
            raise ValueError(
                f"signal row index must stay within [0, {row_count}); got {index!r}"
            )
        if previous_index is not None and index <= previous_index:
            raise ValueError(
                "signal row indexes must be strictly increasing for deterministic ordering"
            )
        normalized.append(index)
        previous_index = index
    if len(normalized) == 0:
        return tuple()
    if _row_selection_is_contiguous_v2(indexes=normalized):
        return slice(normalized[0], normalized[-1] + 1, 1)
    return tuple(normalized)


def _signal_matrix_cache_key_v2(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
    indicator_id: str,
) -> tuple[Path, int, str, str]:
    """
    Build a run-local cache key for one validated signal matrix inside a pinned slot.

    Args:
        context: Shared slot-pinned runtime context resolved at startup.
        timeframe: Canonical signal timeframe literal.
        indicator_id: Canonical indicator identifier.
    Returns:
        tuple[Path, int, str, str]: Hashable key unique to one pinned signal matrix.
    Assumptions:
        Published signal matrices stay immutable once the slot is pinned for one run.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """
    return (
        context.slot_root_path,
        context.slot_generation,
        context.artifact_manifest_hash,
        f"{timeframe}/{indicator_id}",
    )


def _row_selection_is_contiguous_v2(*, indexes: list[int]) -> bool:
    """
    Check whether explicit row indexes form one contiguous increasing range.

    Args:
        indexes: Validated strictly increasing row indexes.
    Returns:
        bool: `True` when the selection can be losslessly represented as one `slice`.
    Assumptions:
        Converting contiguous tuples into slices lets runtime keep memmap views for exact-path
        locality-sensitive reads.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    previous_index = indexes[0]
    for index in indexes[1:]:
        if index != previous_index + 1:
            return False
        previous_index = index
    return True
