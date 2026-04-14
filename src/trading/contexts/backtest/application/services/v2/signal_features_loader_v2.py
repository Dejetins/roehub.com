"""Explicit-path mmap loaders for additive signal feature matrices and subset row reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contracts import (
    ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactSignalFeaturesMatrixV2,
    ArtifactSignalFeaturesRowsV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestSignalFeaturesLoaderV2,
    validate_indicator_id_v2,
    validate_signal_timeframe_v2,
)
from .signal_matrix_loader import (
    _load_mmap_array,
    _normalize_row_selection,
    _price_manifest_for_signal_timeframe,
    _relative_slot_path,
    _signal_catalog_entry,
)


@dataclass(frozen=True, slots=True)
class MmapSignalFeaturesLoaderV2(BacktestSignalFeaturesLoaderV2):
    """
    Load additive signal-feature matrices through explicit manifest-driven mmap paths only.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """

    artifact_loader: BacktestArtifactLoaderV2
    _signal_features_matrix_cache: dict[
        tuple[Path, int, str, str],
        ArtifactSignalFeaturesMatrixV2,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def run_scoped(self) -> MmapSignalFeaturesLoaderV2:
        """
        Build one `run-scoped` signal-features loader that owns fresh mmap caches per caller.

        Args:
            None.
        Returns:
            MmapSignalFeaturesLoaderV2: Fresh loader sharing the same strict `artifact_loader`
                wiring and starting with an empty signal-features cache.
        Assumptions:
            Caller defines the returned loader lifetime and discards it after the owning runtime
            scope finishes.
        Raises:
            None.
        Side Effects:
            None.
        """
        return MmapSignalFeaturesLoaderV2(artifact_loader=self.artifact_loader)

    def load_signal_features_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalFeaturesMatrixV2:
        """
        Load one explicit signal-feature matrix via `np.load(..., mmap_mode='r')`.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
        Returns:
            ArtifactSignalFeaturesMatrixV2: Cached-or-new memory-mapped feature matrix and strict
                manifest metadata.
        Assumptions:
            Runtime must address signal-feature families only through the pinned root catalog and
            the owning signal manifest, never through filesystem discovery.
        Raises:
            FileNotFoundError: If the explicit feature manifest or `features.f32.npy` is missing.
            ValueError: If the owning signal manifest lacks the additive feature reference or any
                manifest/file contract drifts.
        Side Effects:
            Reads one owning signal manifest, one feature manifest, and may memory-map one feature
            matrix on first access for this pinned target.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        loaded = self._load_signal_features_matrix_impl(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
            allow_missing=False,
        )
        if loaded is None:
            raise AssertionError("strict signal_features loader unexpectedly returned None")
        return loaded

    def try_load_signal_features_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalFeaturesMatrixV2 | None:
        """
        Load one signal-feature matrix or return `None` for a legacy slot without the family.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
        Returns:
            ArtifactSignalFeaturesMatrixV2 | None: Loaded matrix when present, else `None` when
                the owning signal manifest does not declare `signal_features`.
        Assumptions:
            Legacy slots without additive feature artifacts must remain readable without forcing
            runtime feature availability.
        Raises:
            FileNotFoundError: If the feature family is declared but files are missing.
            ValueError: If the request or any declared manifest/file contract is invalid.
        Side Effects:
            May read manifests and memory-map one feature matrix on first access.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
          - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
        """
        return self._load_signal_features_matrix_impl(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
            allow_missing=True,
        )

    def load_signal_feature_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> ArtifactSignalFeaturesRowsV2:
        """
        Load a deterministic subset of signal-feature rows when the family is present.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
            row_selection: Either a positive-step `slice` or a strictly increasing row-index
                tuple.
        Returns:
            np.ndarray: Selected feature rows preserving deterministic caller ordering.
        Assumptions:
            Feature rows stay 1:1 aligned with signal rows, so row slicing semantics match the
            signal-matrix loader exactly.
        Raises:
            FileNotFoundError: If the underlying feature matrix file is missing.
            ValueError: If the feature family is absent or the row selection/contract is invalid.
        Side Effects:
            May memory-map one `.npy` file on first access and returns a slice or indexed
            selection from the cached feature matrix.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        signal_features_matrix = self.load_signal_features_matrix(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        return _materialize_signal_feature_rows_v2(
            signal_features_matrix=signal_features_matrix,
            row_selection=row_selection,
        )

    def try_load_signal_feature_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> ArtifactSignalFeaturesRowsV2 | None:
        """
        Load selected feature rows or return `None` for a legacy slot without the family.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
            row_selection: Either a positive-step `slice` or a strictly increasing row-index
                tuple.
        Returns:
            np.ndarray | None: Selected feature rows when present, else `None` for legacy slots.
        Assumptions:
            Only the absence of the additive `signal_features` reference is treated as a
            backward-compatible `None` case.
        Raises:
            FileNotFoundError: If the feature family is declared but the artifact file is missing.
            ValueError: If the row selection or any declared manifest/file contract is invalid.
        Side Effects:
            May read manifests and memory-map one feature matrix on first access.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        signal_features_matrix = self.try_load_signal_features_matrix(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        if signal_features_matrix is None:
            return None
        return _materialize_signal_feature_rows_v2(
            signal_features_matrix=signal_features_matrix,
            row_selection=row_selection,
        )

    def _load_signal_features_matrix_impl(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        allow_missing: bool,
    ) -> ArtifactSignalFeaturesMatrixV2 | None:
        """
        Load one strict signal-feature matrix with optional legacy-safe absence handling.

        Args:
            context: Shared slot-pinned context resolved at runtime start.
            timeframe: Requested signal timeframe.
            indicator_id: Requested indicator identifier.
            allow_missing: Whether a missing `signal_features` reference should yield `None`.
        Returns:
            ArtifactSignalFeaturesMatrixV2 | None: Loaded matrix or `None` for legacy slots when
                allowed.
        Assumptions:
            Only explicit contract absence is optional; all declared path/metadata drift remains a
            hard failure.
        Raises:
            FileNotFoundError: If declared feature artifacts are missing on disk.
            ValueError: If request literals or manifest/file contracts are invalid.
        Side Effects:
            Reads manifests and may memory-map one feature matrix from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        validated_timeframe = validate_signal_timeframe_v2(timeframe)
        validated_indicator_id = validate_indicator_id_v2(indicator_id)
        cache_key = _signal_features_cache_key_v2(
            context=context,
            timeframe=validated_timeframe,
            indicator_id=validated_indicator_id,
        )
        cached = self._signal_features_matrix_cache.get(cache_key)
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
        expected_signal_manifest_relative_path = _relative_slot_path(
            context=context,
            absolute_path=signal_paths.manifest,
        )
        if catalog_entry.manifest_path != expected_signal_manifest_relative_path:
            raise ValueError(
                "signals catalog manifest_path must match the deterministic explicit path; got "
                f"{catalog_entry.manifest_path!r}, expected "
                f"{expected_signal_manifest_relative_path!r}"
            )
        signal_manifest = self.artifact_loader.load_signal_manifest_from_path(
            signal_paths.manifest,
            slot=context.artifact_slot,
        )
        if signal_manifest.signal_features is None:
            if allow_missing:
                return None
            raise ValueError(
                "signal manifest does not declare optional signal_features metadata for "
                f"signals/{validated_timeframe}/{validated_indicator_id}"
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
        signal_features_paths = self.artifact_loader.resolve_signal_features_paths(
            context.coordinates,
            context.artifact_slot,
            validated_timeframe,
            validated_indicator_id,
        )
        expected_features_manifest_relative_path = _relative_slot_path(
            context=context,
            absolute_path=signal_features_paths.manifest,
        )
        if (
            signal_manifest.signal_features.manifest_path
            != expected_features_manifest_relative_path
        ):
            raise ValueError(
                "signal_features manifest_path must match the deterministic explicit path; got "
                f"{signal_manifest.signal_features.manifest_path!r}, expected "
                f"{expected_features_manifest_relative_path!r}"
            )
        signal_features_manifest = self.artifact_loader.load_signal_features_manifest_from_path(
            signal_features_paths.manifest,
            slot=context.artifact_slot,
        )
        if signal_features_manifest.timeframe != validated_timeframe:
            raise ValueError(
                "signal_features manifest timeframe must match request; got "
                f"{signal_features_manifest.timeframe!r}, expected {validated_timeframe!r}"
            )
        if signal_features_manifest.indicator_id != validated_indicator_id:
            raise ValueError(
                "signal_features manifest indicator_id must match request; got "
                f"{signal_features_manifest.indicator_id!r}, expected "
                f"{validated_indicator_id!r}"
            )
        if signal_features_manifest.slot_generation != context.slot_generation:
            raise ValueError(
                "signal_features manifest slot_generation must match pinned context; got "
                f"{signal_features_manifest.slot_generation!r}, expected "
                f"{context.slot_generation!r}"
            )
        if signal_features_manifest.asof_date != context.artifact_asof_date:
            raise ValueError(
                "signal_features manifest asof_date must match pinned context; got "
                f"{signal_features_manifest.asof_date!r}, expected "
                f"{context.artifact_asof_date!r}"
            )
        if signal_features_manifest.rows_count != signal_manifest.rows_count:
            raise ValueError(
                "signal_features rows_count must match owning signal rows_count; got "
                f"{signal_features_manifest.rows_count!r}, expected "
                f"{signal_manifest.rows_count!r}"
            )
        matrix = _load_mmap_array(
            context=context,
            metadata=signal_features_manifest.features,
            expected_path=signal_features_paths.features,
            expected_dtype=ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
            expected_shape=(signal_manifest.rows_count, len(SIGNAL_FEATURE_NAMES_V2)),
            location=(
                f"signal_features/{validated_timeframe}/{validated_indicator_id}/"
                "features.f32.npy"
            ),
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"signal_features/{validated_timeframe}/{validated_indicator_id} must contain "
                "only finite float32 values"
            )
        loaded = ArtifactSignalFeaturesMatrixV2(
            timeframe=validated_timeframe,
            indicator_id=validated_indicator_id,
            manifest=signal_features_manifest,
            matrix=matrix,
        )
        self._signal_features_matrix_cache[cache_key] = loaded
        return loaded


def _signal_features_cache_key_v2(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
    indicator_id: str,
) -> tuple[Path, int, str, str]:
    """
    Build a run-local cache key for one validated signal-feature matrix inside a pinned slot.

    Args:
        context: Shared slot-pinned runtime context resolved at startup.
        timeframe: Canonical signal timeframe literal.
        indicator_id: Canonical indicator identifier.
    Returns:
        tuple[Path, int, str, str]: Hashable key unique to one pinned feature matrix.
    Assumptions:
        Published feature matrices are immutable within one pinned slot and can safely reuse the
        same cache-key dimensions as signal matrices.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """
    return (
        context.slot_root_path,
        context.slot_generation,
        context.artifact_manifest_hash,
        f"{timeframe}/{indicator_id}",
    )


def _materialize_signal_feature_rows_v2(
    *,
    signal_features_matrix: ArtifactSignalFeaturesMatrixV2,
    row_selection: slice | tuple[int, ...],
) -> ArtifactSignalFeaturesRowsV2:
    """
    Materialize one deterministic selected feature-row payload from a validated feature matrix.

    Args:
        signal_features_matrix: Validated memory-mapped signal-feature matrix.
        row_selection: Deterministic row selection aligned with signal-row ordering.
    Returns:
        ArtifactSignalFeaturesRowsV2: Typed selected feature rows preserving caller order.
    Assumptions:
        Feature rows stay 1:1 aligned with the owning signal matrix rows.
    Raises:
        ValueError: If the row selection is invalid for the loaded matrix.
    Side Effects:
        Returns a memmap slice or indexed NumPy array view from the cached matrix.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """
    normalized_selection = _normalize_row_selection(
        row_selection=row_selection,
        row_count=signal_features_matrix.manifest.rows_count,
    )
    return ArtifactSignalFeaturesRowsV2(
        timeframe=signal_features_matrix.timeframe,
        indicator_id=signal_features_matrix.indicator_id,
        feature_names=signal_features_matrix.manifest.feature_names,
        rows=signal_features_matrix.matrix[normalized_selection, :],
    )
