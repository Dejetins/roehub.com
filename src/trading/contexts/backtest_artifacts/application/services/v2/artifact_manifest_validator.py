"""Strict manifest and array validators for deterministic artifact publish flow v2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
    ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
    ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
    ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
    ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
    ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
    ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
    ARTIFACT_SIGNAL_VALUE_SET_V2,
    ARTIFACT_TIME_AXIS_ORDER_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactArrayMetadataV2,
    ArtifactCoordinatesV2,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactHitTimesTableManifestV2,
    ArtifactManifestDocumentV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalFeaturesManifestDocumentV2,
    ArtifactSignalManifestDocumentV2,
    ArtifactSlotLiteralV2,
    ArtifactSlotValidationResultV2,
    ArtifactSlotValidationSpecV2,
    ArtifactTimelineCoverageV2,
    ArtifactValidationDiagnosticV2,
    BacktestArtifactLoaderV2,
)


@dataclass(frozen=True, slots=True)
class BacktestArtifactManifestValidatorV2:
    """
    Strict slot validator for root, signal, and hit-times manifest contracts.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    artifact_loader: BacktestArtifactLoaderV2

    def validate_slot(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        validation_spec: ArtifactSlotValidationSpecV2,
        expected_asof_date: str | None = None,
        expected_slot_generation: int | None = None,
    ) -> ArtifactSlotValidationResultV2:
        """
        Validate one prepared inactive slot with deterministic diagnostics ordering.

        Args:
            coordinates: Artifact symbol-root coordinates under validation.
            slot: Explicit inactive slot literal being validated.
            validation_spec: Explicit validation plan describing the whole slot surface.
            expected_asof_date: Optional strict `YYYY-MM-DD` literal expected from manifests.
            expected_slot_generation: Optional positive generation expected from manifests.
        Returns:
            ArtifactSlotValidationResultV2: Structured validation report with typed manifests and
                deterministic diagnostics.
        Assumptions:
            Validation runs in publish/precompute paths and may compute hashes over full files.
        Raises:
            ValueError: If explicit coordinates or slot literals are invalid before file access.
        Side Effects:
            Reads manifests and `.npy` arrays from the inactive slot on disk.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        root_manifest_path = self.artifact_loader.resolve_slot_manifest_path(coordinates, slot)
        try:
            slot_manifest = self.artifact_loader.load_slot_manifest(coordinates, slot)
        except (FileNotFoundError, ValueError) as error:
            return ArtifactSlotValidationResultV2(
                slot=slot,
                slot_manifest=None,
                signal_manifests=(),
                hit_times_manifest=None,
                manifest_sha256=None,
                validation_spec=validation_spec,
                diagnostics=(
                    ArtifactValidationDiagnosticV2(
                        code="root_manifest_invalid",
                        message=str(error),
                        location="manifest.yaml",
                        manifest_path=root_manifest_path,
                    ),
                ),
            )

        diagnostics: list[ArtifactValidationDiagnosticV2] = []
        slot_root = slot_manifest.path.parent
        manifest_sha256 = _file_sha256_hex_v2(slot_manifest.path)

        self._validate_root_manifest_contracts(
            slot_manifest=slot_manifest,
            coordinates=coordinates,
            validation_spec=validation_spec,
            expected_asof_date=expected_asof_date,
            expected_slot_generation=expected_slot_generation,
            diagnostics=diagnostics,
        )

        price_by_timeframe = {manifest.timeframe: manifest for manifest in slot_manifest.prices}
        price_time_arrays_by_timeframe: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        one_minute_bar_count: int | None = None
        hit_times_bar_count: int | None = None
        for price_manifest in slot_manifest.prices:
            loaded_time_arrays = self._validate_price_manifest(
                slot_root=slot_root,
                slot_manifest_path=slot_manifest.path,
                price_manifest=price_manifest,
                diagnostics=diagnostics,
            )
            if loaded_time_arrays is not None:
                price_time_arrays_by_timeframe[price_manifest.timeframe] = loaded_time_arrays
            if price_manifest.timeframe == "1m":
                one_minute_bar_count = price_manifest.coverage.bar_count
            if price_manifest.timeframe == HIT_TIMES_TIMEFRAME_LITERAL_V2:
                hit_times_bar_count = price_manifest.coverage.bar_count

        for mapping_manifest in slot_manifest.mappings:
            self._validate_mapping_manifest(
                coordinates=coordinates,
                slot=slot,
                slot_root=slot_root,
                slot_manifest_path=slot_manifest.path,
                mapping_manifest=mapping_manifest,
                price_by_timeframe=price_by_timeframe,
                price_time_arrays_by_timeframe=price_time_arrays_by_timeframe,
                one_minute_bar_count=one_minute_bar_count,
                diagnostics=diagnostics,
            )

        signal_manifests: list[ArtifactSignalManifestDocumentV2] = []
        for signal_reference in slot_manifest.signals.manifests:
            loaded_signal_manifest = self._load_signal_manifest_with_diagnostics(
                coordinates=coordinates,
                slot=slot,
                slot_manifest=slot_manifest,
                slot_root=slot_root,
                signal_reference=signal_reference,
                diagnostics=diagnostics,
            )
            if loaded_signal_manifest is None:
                continue
            signal_manifests.append(loaded_signal_manifest)
            self._validate_signal_manifest(
                coordinates=coordinates,
                slot=slot,
                slot_root=slot_root,
                slot_manifest=slot_manifest,
                signal_reference=signal_reference,
                signal_manifest=loaded_signal_manifest,
                price_by_timeframe=price_by_timeframe,
                diagnostics=diagnostics,
            )

        hit_times_manifest: ArtifactHitTimesManifestDocumentV2 | None = None
        if validation_spec.require_hit_times_manifest:
            hit_times_manifest = self._load_hit_times_manifest_with_diagnostics(
                coordinates=coordinates,
                slot=slot,
                slot_manifest=slot_manifest,
                slot_root=slot_root,
                diagnostics=diagnostics,
            )
            if hit_times_manifest is not None:
                self._validate_hit_times_manifest(
                    coordinates=coordinates,
                    slot=slot,
                    slot_root=slot_root,
                    slot_manifest=slot_manifest,
                    hit_times_manifest=hit_times_manifest,
                    hit_times_bar_count=hit_times_bar_count,
                    diagnostics=diagnostics,
                )

        return ArtifactSlotValidationResultV2(
            slot=slot,
            slot_manifest=slot_manifest,
            signal_manifests=tuple(signal_manifests),
            hit_times_manifest=hit_times_manifest,
            manifest_sha256=manifest_sha256,
            validation_spec=validation_spec,
            diagnostics=tuple(diagnostics),
        )

    def _validate_root_manifest_contracts(
        self,
        *,
        slot_manifest: ArtifactManifestDocumentV2,
        coordinates: ArtifactCoordinatesV2,
        validation_spec: ArtifactSlotValidationSpecV2,
        expected_asof_date: str | None,
        expected_slot_generation: int | None,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate root-manifest identity and whole-slot explicit coverage contracts.

        Args:
            slot_manifest: Parsed strict root manifest.
            coordinates: Expected artifact coordinates.
            validation_spec: Explicit whole-slot validation plan.
            expected_asof_date: Optional expected as-of date literal.
            expected_slot_generation: Optional expected slot generation.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Root manifest is the source of truth for fixed runtime metadata and slot contents.
        Raises:
            None.
        Side Effects:
            Appends deterministic diagnostics on contract mismatches.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if slot_manifest.identity != coordinates:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_identity_mismatch",
                    message=(
                        "root manifest identity must match publish coordinates; got "
                        f"{slot_manifest.identity!r}, expected {coordinates!r}"
                    ),
                    location="identity",
                    manifest_path=slot_manifest.path,
                )
            )
        if expected_asof_date is not None and slot_manifest.asof_date != expected_asof_date:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_asof_date_mismatch",
                    message=(
                        "root manifest asof_date must match publish request; got "
                        f"{slot_manifest.asof_date!r}, expected {expected_asof_date!r}"
                    ),
                    location="asof_date",
                    manifest_path=slot_manifest.path,
                )
            )
        if (
            expected_slot_generation is not None
            and slot_manifest.slot_generation != expected_slot_generation
        ):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_slot_generation_mismatch",
                    message=(
                        "root manifest slot_generation must match publish target; got "
                        f"{slot_manifest.slot_generation!r}, expected "
                        f"{expected_slot_generation!r}"
                    ),
                    location="slot_generation",
                    manifest_path=slot_manifest.path,
                )
            )

        root_price_timeframes = tuple(item.timeframe for item in slot_manifest.prices)
        if root_price_timeframes != validation_spec.price_timeframes:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_price_timeframes_mismatch",
                    message=(
                        "root manifest prices must match explicit validation spec; got "
                        f"{root_price_timeframes!r}, expected "
                        f"{validation_spec.price_timeframes!r}"
                    ),
                    location="prices",
                    manifest_path=slot_manifest.path,
                )
            )

        root_mapping_timeframes = tuple(item.timeframe for item in slot_manifest.mappings)
        if root_mapping_timeframes != validation_spec.mapping_timeframes:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_mapping_timeframes_mismatch",
                    message=(
                        "root manifest mappings must match explicit validation spec; got "
                        f"{root_mapping_timeframes!r}, expected "
                        f"{validation_spec.mapping_timeframes!r}"
                    ),
                    location="mappings",
                    manifest_path=slot_manifest.path,
                )
            )

        root_signal_targets = tuple(
            (item.timeframe, item.indicator_id) for item in slot_manifest.signals.manifests
        )
        spec_signal_targets = tuple(
            (item.timeframe, item.indicator_id) for item in validation_spec.signal_artifacts
        )
        if root_signal_targets != spec_signal_targets:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_signal_targets_mismatch",
                    message=(
                        "root manifest signals must match explicit validation spec; got "
                        f"{root_signal_targets!r}, expected {spec_signal_targets!r}"
                    ),
                    location="signals.manifests",
                    manifest_path=slot_manifest.path,
                )
            )

        supported_timeframes = tuple(
            dict.fromkeys(item.timeframe for item in slot_manifest.signals.manifests)
        )
        if slot_manifest.signals.supported_timeframes != supported_timeframes:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_supported_signal_timeframes_mismatch",
                    message=(
                        "root manifest signals.supported_timeframes must match manifests; got "
                        f"{slot_manifest.signals.supported_timeframes!r}, expected "
                        f"{supported_timeframes!r}"
                    ),
                    location="signals.supported_timeframes",
                    manifest_path=slot_manifest.path,
                )
            )

        supported_indicator_ids = tuple(
            sorted({item.indicator_id for item in slot_manifest.signals.manifests})
        )
        if slot_manifest.signals.supported_indicator_ids != supported_indicator_ids:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="root_manifest_supported_indicator_ids_mismatch",
                    message=(
                        "root manifest signals.supported_indicator_ids must match manifests; got "
                        f"{slot_manifest.signals.supported_indicator_ids!r}, expected "
                        f"{supported_indicator_ids!r}"
                    ),
                    location="signals.supported_indicator_ids",
                    manifest_path=slot_manifest.path,
                )
            )

    def _validate_price_manifest(
        self,
        *,
        slot_root: Path,
        slot_manifest_path: Path,
        price_manifest: ArtifactPriceTimeframeManifestV2,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Validate one strict `prices/<tf>/` manifest section against array contents.

        Args:
            slot_root: Absolute slot-root path.
            slot_manifest_path: Root manifest path used in diagnostics.
            price_manifest: Typed price-manifest section.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            tuple[np.ndarray, np.ndarray] | None: Loaded `open_time` and `close_time` arrays when
                both are available for downstream correspondence validation, otherwise `None`.
        Assumptions:
            Price arrays are stored as `.npy` files and are safe to inspect with mmap loading.
        Raises:
            None.
        Side Effects:
            Reads one price artifact family from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        bar_count = price_manifest.coverage.bar_count
        open_time_path = slot_root / price_manifest.open_time.path
        close_time_path = slot_root / price_manifest.close_time.path
        ohlcv_path = slot_root / price_manifest.ohlcv.path

        open_time_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=slot_manifest_path,
            location=f"prices[{price_manifest.timeframe}].open_time",
            metadata=price_manifest.open_time,
            expected_path=open_time_path,
            expected_dtype="int64",
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(bar_count,),
            diagnostics=diagnostics,
        )
        close_time_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=slot_manifest_path,
            location=f"prices[{price_manifest.timeframe}].close_time",
            metadata=price_manifest.close_time,
            expected_path=close_time_path,
            expected_dtype="int64",
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(bar_count,),
            diagnostics=diagnostics,
        )
        ohlcv_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=slot_manifest_path,
            location=f"prices[{price_manifest.timeframe}].ohlcv",
            metadata=price_manifest.ohlcv,
            expected_path=ohlcv_path,
            expected_dtype=ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2,
            expected_shape=(bar_count, 5),
            diagnostics=diagnostics,
        )

        if open_time_array is not None:
            self._validate_monotone_time_array(
                array=open_time_array,
                manifest_path=slot_manifest_path,
                location=f"prices[{price_manifest.timeframe}].open_time",
                code="price_open_time_not_strictly_increasing",
                diagnostics=diagnostics,
            )
            self._validate_timeline_coverage_edges(
                coverage=price_manifest.coverage,
                array=open_time_array,
                manifest_path=slot_manifest_path,
                location=f"prices[{price_manifest.timeframe}].open_time",
                expected_start=price_manifest.coverage.open_time_start,
                expected_end=price_manifest.coverage.open_time_end,
                diagnostics=diagnostics,
            )

        if close_time_array is not None:
            self._validate_monotone_time_array(
                array=close_time_array,
                manifest_path=slot_manifest_path,
                location=f"prices[{price_manifest.timeframe}].close_time",
                code="price_close_time_not_strictly_increasing",
                diagnostics=diagnostics,
            )
            self._validate_timeline_coverage_edges(
                coverage=price_manifest.coverage,
                array=close_time_array,
                manifest_path=slot_manifest_path,
                location=f"prices[{price_manifest.timeframe}].close_time",
                expected_start=price_manifest.coverage.close_time_start,
                expected_end=price_manifest.coverage.close_time_end,
                diagnostics=diagnostics,
            )

        if (
            open_time_array is not None
            and close_time_array is not None
            and open_time_array.shape == close_time_array.shape
            and not np.all(close_time_array > open_time_array)
        ):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="price_close_time_not_after_open_time",
                    message=(
                        f"prices[{price_manifest.timeframe}] close_time must be strictly after "
                        "open_time for every bar"
                    ),
                    location=f"prices[{price_manifest.timeframe}]",
                    manifest_path=slot_manifest_path,
                )
            )

        if (
            ohlcv_array is not None
            and int(ohlcv_array.shape[0]) != price_manifest.coverage.bar_count
        ):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="price_coverage_bar_count_mismatch",
                    message=(
                        f"prices[{price_manifest.timeframe}] coverage.bar_count must match "
                        f"ohlcv rows; got {price_manifest.coverage.bar_count!r}, expected "
                        f"{int(ohlcv_array.shape[0])!r}"
                    ),
                    location=f"prices[{price_manifest.timeframe}].coverage",
                    manifest_path=slot_manifest_path,
                )
            )
        if (
            open_time_array is None
            or close_time_array is None
            or open_time_array.shape != close_time_array.shape
        ):
            return None
        return (open_time_array, close_time_array)

    def _validate_mapping_manifest(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_root: Path,
        slot_manifest_path: Path,
        mapping_manifest: ArtifactMappingTimeframeManifestV2,
        price_by_timeframe: dict[str, ArtifactPriceTimeframeManifestV2],
        price_time_arrays_by_timeframe: dict[str, tuple[np.ndarray, np.ndarray]],
        one_minute_bar_count: int | None,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate one strict `mappings/<tf>/` manifest section against array contents.

        Args:
            coordinates: Artifact coordinates used for deterministic path resolution.
            slot: Explicit slot literal under validation.
            slot_root: Absolute slot-root path.
            slot_manifest_path: Root manifest path used in diagnostics.
            mapping_manifest: Typed mapping-manifest section.
            price_by_timeframe: Root price sections keyed by timeframe.
            price_time_arrays_by_timeframe: Loaded root price time arrays keyed by timeframe.
            one_minute_bar_count: Root `1m` timeline length when available.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Mapping arrays target the corresponding request timeframe and point into `1m`.
        Raises:
            None.
        Side Effects:
            Reads one mapping artifact family from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        price_manifest = price_by_timeframe.get(mapping_manifest.timeframe)
        target_bar_count = price_manifest.coverage.bar_count if price_manifest is not None else None
        resolved_paths = self.artifact_loader.resolve_mapping_paths(
            coordinates,
            slot,
            mapping_manifest.timeframe,
        )

        open_idx_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=slot_manifest_path,
            location=f"mappings[{mapping_manifest.timeframe}].bar_open_1m_idx",
            metadata=mapping_manifest.bar_open_1m_idx,
            expected_path=resolved_paths.bar_open_1m_idx,
            expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(target_bar_count,) if target_bar_count is not None else None,
            diagnostics=diagnostics,
        )
        close_idx_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=slot_manifest_path,
            location=f"mappings[{mapping_manifest.timeframe}].bar_close_1m_idx",
            metadata=mapping_manifest.bar_close_1m_idx,
            expected_path=resolved_paths.bar_close_1m_idx,
            expected_dtype=ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_TIME_AXIS_ORDER_V2,
            expected_shape=(target_bar_count,) if target_bar_count is not None else None,
            diagnostics=diagnostics,
        )

        if price_manifest is None:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="mapping_timeframe_without_price_manifest",
                    message=(
                        "mapping timeframe must have a corresponding root price manifest; got "
                        f"{mapping_manifest.timeframe!r}"
                    ),
                    location=f"mappings[{mapping_manifest.timeframe}]",
                    manifest_path=slot_manifest_path,
                )
            )

        if open_idx_array is not None:
            self._validate_non_decreasing_index_array(
                array=open_idx_array,
                manifest_path=slot_manifest_path,
                location=f"mappings[{mapping_manifest.timeframe}].bar_open_1m_idx",
                code="mapping_open_indexes_not_non_decreasing",
                diagnostics=diagnostics,
            )
        if close_idx_array is not None:
            self._validate_non_decreasing_index_array(
                array=close_idx_array,
                manifest_path=slot_manifest_path,
                location=f"mappings[{mapping_manifest.timeframe}].bar_close_1m_idx",
                code="mapping_close_indexes_not_non_decreasing",
                diagnostics=diagnostics,
            )

        if (
            open_idx_array is not None
            and close_idx_array is not None
            and open_idx_array.shape == close_idx_array.shape
            and not np.all(open_idx_array <= close_idx_array)
        ):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="mapping_open_index_after_close_index",
                    message=(
                        f"mappings[{mapping_manifest.timeframe}] must satisfy "
                        "bar_open_1m_idx <= bar_close_1m_idx"
                    ),
                    location=f"mappings[{mapping_manifest.timeframe}]",
                    manifest_path=slot_manifest_path,
                )
            )

        if one_minute_bar_count is not None and open_idx_array is not None:
            if not np.all(open_idx_array < one_minute_bar_count):
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="mapping_open_indexes_out_of_bounds",
                        message=(
                            f"mappings[{mapping_manifest.timeframe}] bar_open_1m_idx must stay "
                            f"within [0, {one_minute_bar_count}); mapping bounds contract failed"
                        ),
                        location=f"mappings[{mapping_manifest.timeframe}].bar_open_1m_idx",
                        manifest_path=slot_manifest_path,
                    )
                )
        if one_minute_bar_count is not None and close_idx_array is not None:
            if not np.all(close_idx_array < one_minute_bar_count):
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="mapping_close_indexes_out_of_bounds",
                        message=(
                            f"mappings[{mapping_manifest.timeframe}] bar_close_1m_idx must stay "
                            f"within [0, {one_minute_bar_count}); mapping bounds contract failed"
                        ),
                        location=f"mappings[{mapping_manifest.timeframe}].bar_close_1m_idx",
                        manifest_path=slot_manifest_path,
                    )
                )
        target_price_arrays = price_time_arrays_by_timeframe.get(mapping_manifest.timeframe)
        one_minute_price_arrays = price_time_arrays_by_timeframe.get("1m")
        if (
            open_idx_array is not None
            and close_idx_array is not None
            and target_price_arrays is not None
            and one_minute_price_arrays is not None
            and open_idx_array.shape == close_idx_array.shape
            and open_idx_array.shape == target_price_arrays[0].shape
            and (
                one_minute_bar_count is None
                or (
                    np.all(open_idx_array < one_minute_bar_count)
                    and np.all(close_idx_array < one_minute_bar_count)
                )
            )
        ):
            one_minute_open_time, one_minute_close_time = one_minute_price_arrays
            target_open_time, target_close_time = target_price_arrays
            open_index_positions = np.asarray(open_idx_array, dtype=np.intp)
            close_index_positions = np.asarray(close_idx_array, dtype=np.intp)
            if not np.array_equal(
                one_minute_open_time[open_index_positions],
                target_open_time,
            ):
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="mapping_open_time_correspondence_mismatch",
                        message=(
                            f"mappings[{mapping_manifest.timeframe}] must satisfy "
                            "prices/1m.open_time[bar_open_1m_idx] == "
                            f"prices[{mapping_manifest.timeframe}].open_time"
                        ),
                        location=f"mappings[{mapping_manifest.timeframe}].bar_open_1m_idx",
                        manifest_path=slot_manifest_path,
                    )
                )
            if not np.array_equal(
                one_minute_close_time[close_index_positions],
                target_close_time,
            ):
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="mapping_close_time_correspondence_mismatch",
                        message=(
                            f"mappings[{mapping_manifest.timeframe}] must satisfy "
                            "prices/1m.close_time[bar_close_1m_idx] == "
                            f"prices[{mapping_manifest.timeframe}].close_time"
                        ),
                        location=f"mappings[{mapping_manifest.timeframe}].bar_close_1m_idx",
                        manifest_path=slot_manifest_path,
                    )
                )

    def _load_signal_manifest_with_diagnostics(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_manifest: ArtifactManifestDocumentV2,
        slot_root: Path,
        signal_reference: ArtifactSignalCatalogEntryV2,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> ArtifactSignalManifestDocumentV2 | None:
        """
        Load one strict signal manifest while converting read failures into diagnostics.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_manifest: Root manifest used for path/hash cross-checks.
            slot_root: Absolute slot-root path.
            signal_reference: Root-manifest signal manifest reference.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            ArtifactSignalManifestDocumentV2 | None: Parsed signal manifest when loading succeeds.
        Assumptions:
            Root manifest already provides the canonical signal manifest reference set.
        Raises:
            None.
        Side Effects:
            Reads one signal manifest file from disk and appends diagnostics on failure.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        signal_paths = self.artifact_loader.resolve_signal_paths(
            coordinates,
            slot,
            signal_reference.timeframe,
            signal_reference.indicator_id,
        )
        expected_relative_path = _relative_slot_path_v2(slot_root, signal_paths.manifest)
        if signal_reference.manifest_path != expected_relative_path:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_reference_path_mismatch",
                    message=(
                        "root manifest signal reference must use deterministic manifest path; got "
                        f"{signal_reference.manifest_path!r}, expected "
                        f"{expected_relative_path!r}"
                    ),
                    location=(
                        "signals.manifests"
                        f"[{signal_reference.timeframe}:{signal_reference.indicator_id}]"
                    ),
                    manifest_path=slot_manifest.path,
                    artifact_path=signal_paths.manifest,
                )
            )
        if signal_paths.manifest.is_file():
            actual_manifest_hash = _file_sha256_hex_v2(signal_paths.manifest)
            if signal_reference.manifest_sha256 != actual_manifest_hash:
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="signal_manifest_reference_hash_mismatch",
                        message=(
                            "root manifest signal reference sha256 must match actual manifest; got "
                            f"{signal_reference.manifest_sha256!r}, expected "
                            f"{actual_manifest_hash!r}"
                        ),
                        location=(
                            "signals.manifests"
                            f"[{signal_reference.timeframe}:{signal_reference.indicator_id}]"
                        ),
                        manifest_path=slot_manifest.path,
                        artifact_path=signal_paths.manifest,
                    )
                )
        try:
            return self.artifact_loader.load_signal_manifest(
                coordinates,
                slot,
                signal_reference.timeframe,
                signal_reference.indicator_id,
            )
        except (FileNotFoundError, ValueError) as error:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_invalid",
                    message=str(error),
                    location=f"signals[{signal_reference.timeframe}:{signal_reference.indicator_id}]",
                    manifest_path=signal_paths.manifest,
                )
            )
            return None

    def _validate_signal_manifest(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_root: Path,
        slot_manifest: ArtifactManifestDocumentV2,
        signal_reference: ArtifactSignalCatalogEntryV2,
        signal_manifest: ArtifactSignalManifestDocumentV2,
        price_by_timeframe: dict[str, ArtifactPriceTimeframeManifestV2],
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate one strict signal manifest against root contract and signal matrix contents.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_root: Absolute slot-root path.
            slot_manifest: Root manifest used for cross-contract checks.
            signal_reference: Root-manifest signal manifest reference.
            signal_manifest: Parsed strict signal manifest.
            price_by_timeframe: Root price sections keyed by timeframe.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Signal matrix metadata must be sufficient for runtime without recomputation.
        Raises:
            None.
        Side Effects:
            Reads the signal matrix file from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if signal_manifest.timeframe != signal_reference.timeframe:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_timeframe_mismatch",
                    message=(
                        "signal manifest timeframe must match root catalog reference; got "
                        f"{signal_manifest.timeframe!r}, expected "
                        f"{signal_reference.timeframe!r}"
                    ),
                    location="timeframe",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.indicator_id != signal_reference.indicator_id:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_indicator_id_mismatch",
                    message=(
                        "signal manifest indicator_id must match root catalog reference; got "
                        f"{signal_manifest.indicator_id!r}, expected "
                        f"{signal_reference.indicator_id!r}"
                    ),
                    location="indicator_id",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.slot_generation != slot_manifest.slot_generation:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_slot_generation_mismatch",
                    message=(
                        "signal manifest slot_generation must match root manifest; got "
                        f"{signal_manifest.slot_generation!r}, expected "
                        f"{slot_manifest.slot_generation!r}"
                    ),
                    location="slot_generation",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.asof_date != slot_manifest.asof_date:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_asof_date_mismatch",
                    message=(
                        "signal manifest asof_date must match root manifest; got "
                        f"{signal_manifest.asof_date!r}, expected "
                        f"{slot_manifest.asof_date!r}"
                    ),
                    location="asof_date",
                    manifest_path=signal_manifest.path,
                )
            )

        price_manifest = price_by_timeframe.get(signal_manifest.timeframe)
        if price_manifest is None:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_timeframe_without_price_manifest",
                    message=(
                        "signal manifest timeframe must have a corresponding root price manifest; "
                        f"got {signal_manifest.timeframe!r}"
                    ),
                    location="timeframe",
                    manifest_path=signal_manifest.path,
                )
            )
            expected_timeline_bar_count = None
        else:
            expected_timeline_bar_count = price_manifest.coverage.bar_count
            if signal_manifest.timeline != price_manifest.coverage:
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="signal_manifest_timeline_mismatch",
                        message=(
                            "signal manifest timeline coverage must match root price coverage for "
                            f"{signal_manifest.timeframe!r}"
                        ),
                        location="timeline",
                        manifest_path=signal_manifest.path,
                    )
                )

        signal_paths = self.artifact_loader.resolve_signal_paths(
            coordinates,
            slot,
            signal_manifest.timeframe,
            signal_manifest.indicator_id,
        )
        signal_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=signal_manifest.path,
            location=f"signals[{signal_manifest.timeframe}:{signal_manifest.indicator_id}].signals",
            metadata=signal_manifest.signals,
            expected_path=signal_paths.signals,
            expected_dtype=slot_manifest.signal_encoding.dtype,
            expected_axis_order=slot_manifest.signal_encoding.axis_order,
            expected_shape=(
                signal_manifest.rows_count,
                expected_timeline_bar_count,
            )
            if expected_timeline_bar_count is not None
            else None,
            diagnostics=diagnostics,
        )

        if signal_manifest.rows_count != signal_manifest.signals.shape[0]:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_rows_count_mismatch",
                    message=(
                        "signal manifest rows_count must match signals.shape[0]; got "
                        f"{signal_manifest.rows_count!r}, expected "
                        f"{signal_manifest.signals.shape[0]!r}"
                    ),
                    location="rows_count",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.timeline.bar_count != signal_manifest.signals.shape[1]:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_timeline_bar_count_mismatch",
                    message=(
                        "signal manifest timeline.bar_count must match signals.shape[1]; got "
                        f"{signal_manifest.timeline.bar_count!r}, expected "
                        f"{signal_manifest.signals.shape[1]!r}"
                    ),
                    location="timeline.bar_count",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.signal_value_set != slot_manifest.signal_encoding.value_set:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_value_set_mismatch",
                    message=(
                        "signal manifest signal_value_set must match root signal encoding; got "
                        f"{signal_manifest.signal_value_set!r}, expected "
                        f"{slot_manifest.signal_encoding.value_set!r}"
                    ),
                    location="signal_value_set",
                    manifest_path=signal_manifest.path,
                )
            )
        if signal_manifest.grid.variant_key_version != 1:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_manifest_variant_key_version_unsupported",
                    message=(
                        "signal manifest grid.variant_key_version must preserve v1 semantics; got "
                        f"{signal_manifest.grid.variant_key_version!r}"
                    ),
                    location="grid.variant_key_version",
                    manifest_path=signal_manifest.path,
                )
            )

        if signal_array is not None:
            valid_mask = (
                (signal_array == ARTIFACT_SIGNAL_VALUE_SET_V2[0])
                | (signal_array == ARTIFACT_SIGNAL_VALUE_SET_V2[1])
                | (signal_array == ARTIFACT_SIGNAL_VALUE_SET_V2[2])
            )
            if not np.all(valid_mask):
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="signal_values_out_of_set",
                        message=(
                            "signal matrix must satisfy signal value set {-1,0,1}; "
                            "invalid encoded values detected"
                        ),
                        location=(
                            f"signals[{signal_manifest.timeframe}:{signal_manifest.indicator_id}]"
                        ),
                        manifest_path=signal_manifest.path,
                        artifact_path=signal_paths.signals,
                    )
                )
        if signal_manifest.signal_features is None:
            return
        loaded_signal_features_manifest = self._load_signal_features_manifest_with_diagnostics(
            coordinates=coordinates,
            slot=slot,
            slot_root=slot_root,
            signal_manifest=signal_manifest,
            diagnostics=diagnostics,
        )
        if loaded_signal_features_manifest is None:
            return
        self._validate_signal_features_manifest(
            coordinates=coordinates,
            slot=slot,
            slot_root=slot_root,
            slot_manifest=slot_manifest,
            signal_manifest=signal_manifest,
            signal_features_manifest=loaded_signal_features_manifest,
            diagnostics=diagnostics,
        )

    def _load_signal_features_manifest_with_diagnostics(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_root: Path,
        signal_manifest: ArtifactSignalManifestDocumentV2,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> ArtifactSignalFeaturesManifestDocumentV2 | None:
        """
        Load one optional signal-feature manifest while converting read failures into diagnostics.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_root: Absolute slot-root path.
            signal_manifest: Parsed signal manifest carrying the additive feature reference.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            ArtifactSignalFeaturesManifestDocumentV2 | None: Parsed feature manifest when loading
                succeeds, else `None`.
        Assumptions:
            Signal-feature discovery stays explicit and originates only from the owning signal
            manifest reference.
        Raises:
            None.
        Side Effects:
            Reads one signal-feature manifest file from disk and appends diagnostics on failure.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
        """
        signal_features_reference = signal_manifest.signal_features
        if signal_features_reference is None:
            return None
        signal_features_paths = self.artifact_loader.resolve_signal_features_paths(
            coordinates,
            slot,
            signal_manifest.timeframe,
            signal_manifest.indicator_id,
        )
        expected_relative_path = _relative_slot_path_v2(
            slot_root,
            signal_features_paths.manifest,
        )
        if signal_features_reference.manifest_path != expected_relative_path:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_reference_path_mismatch",
                    message=(
                        "signal manifest signal_features reference must use the deterministic "
                        f"manifest path; got {signal_features_reference.manifest_path!r}, "
                        f"expected {expected_relative_path!r}"
                    ),
                    location="signal_features.manifest_path",
                    manifest_path=signal_manifest.path,
                    artifact_path=signal_features_paths.manifest,
                )
            )
        if signal_features_paths.manifest.is_file():
            actual_manifest_hash = _file_sha256_hex_v2(signal_features_paths.manifest)
            if signal_features_reference.manifest_sha256 != actual_manifest_hash:
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="signal_features_manifest_reference_hash_mismatch",
                        message=(
                            "signal manifest signal_features sha256 must match actual manifest; "
                            f"got {signal_features_reference.manifest_sha256!r}, expected "
                            f"{actual_manifest_hash!r}"
                        ),
                        location="signal_features.manifest_sha256",
                        manifest_path=signal_manifest.path,
                        artifact_path=signal_features_paths.manifest,
                    )
                )
        try:
            return self.artifact_loader.load_signal_features_manifest(
                coordinates,
                slot,
                signal_manifest.timeframe,
                signal_manifest.indicator_id,
            )
        except (FileNotFoundError, ValueError) as error:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_invalid",
                    message=str(error),
                    location=(
                        "signal_features"
                        f"[{signal_manifest.timeframe}:{signal_manifest.indicator_id}]"
                    ),
                    manifest_path=signal_features_paths.manifest,
                )
            )
            return None

    def _validate_signal_features_manifest(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_root: Path,
        slot_manifest: ArtifactManifestDocumentV2,
        signal_manifest: ArtifactSignalManifestDocumentV2,
        signal_features_manifest: ArtifactSignalFeaturesManifestDocumentV2,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate one strict additive signal-feature manifest against owning signal contracts.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_root: Absolute slot-root path.
            slot_manifest: Root manifest used for cross-contract checks.
            signal_manifest: Owning signal manifest for the same `(timeframe, indicator_id)`.
            signal_features_manifest: Parsed strict additive feature manifest.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Signal features remain row-local warm-cache data for the exact same signal rows and
            therefore must align 1:1 with the owning signal manifest.
        Raises:
            None.
        Side Effects:
            Reads the feature matrix file from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
        """
        if signal_features_manifest.timeframe != signal_manifest.timeframe:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_timeframe_mismatch",
                    message=(
                        "signal_features manifest timeframe must match owning signal manifest; "
                        f"got {signal_features_manifest.timeframe!r}, expected "
                        f"{signal_manifest.timeframe!r}"
                    ),
                    location="timeframe",
                    manifest_path=signal_features_manifest.path,
                )
            )
        if signal_features_manifest.indicator_id != signal_manifest.indicator_id:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_indicator_id_mismatch",
                    message=(
                        "signal_features manifest indicator_id must match owning signal "
                        f"manifest; got {signal_features_manifest.indicator_id!r}, expected "
                        f"{signal_manifest.indicator_id!r}"
                    ),
                    location="indicator_id",
                    manifest_path=signal_features_manifest.path,
                )
            )
        if signal_features_manifest.slot_generation != slot_manifest.slot_generation:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_slot_generation_mismatch",
                    message=(
                        "signal_features manifest slot_generation must match root manifest; got "
                        f"{signal_features_manifest.slot_generation!r}, expected "
                        f"{slot_manifest.slot_generation!r}"
                    ),
                    location="slot_generation",
                    manifest_path=signal_features_manifest.path,
                )
            )
        if signal_features_manifest.asof_date != slot_manifest.asof_date:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_manifest_asof_date_mismatch",
                    message=(
                        "signal_features manifest asof_date must match root manifest; got "
                        f"{signal_features_manifest.asof_date!r}, expected "
                        f"{slot_manifest.asof_date!r}"
                    ),
                    location="asof_date",
                    manifest_path=signal_features_manifest.path,
                )
            )
        if signal_features_manifest.rows_count != signal_manifest.rows_count:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_rows_count_mismatch",
                    message=(
                        "signal_features rows_count must match owning signal rows_count; got "
                        f"{signal_features_manifest.rows_count!r}, expected "
                        f"{signal_manifest.rows_count!r}"
                    ),
                    location="rows_count",
                    manifest_path=signal_features_manifest.path,
                )
            )

        signal_features_paths = self.artifact_loader.resolve_signal_features_paths(
            coordinates,
            slot,
            signal_manifest.timeframe,
            signal_manifest.indicator_id,
        )
        feature_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=signal_features_manifest.path,
            location=(
                "signal_features"
                f"[{signal_manifest.timeframe}:{signal_manifest.indicator_id}].features"
            ),
            metadata=signal_features_manifest.features,
            expected_path=signal_features_paths.features,
            expected_dtype=ARTIFACT_SIGNAL_FEATURE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_SIGNAL_FEATURE_AXIS_ORDER_V2,
            expected_shape=(
                signal_manifest.rows_count,
                len(SIGNAL_FEATURE_NAMES_V2),
            ),
            diagnostics=diagnostics,
        )
        if feature_array is not None and not np.all(np.isfinite(feature_array)):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="signal_features_values_not_finite",
                    message="signal_features matrix must contain only finite float32 values",
                    location=(
                        "signal_features"
                        f"[{signal_manifest.timeframe}:{signal_manifest.indicator_id}]"
                    ),
                    manifest_path=signal_features_manifest.path,
                    artifact_path=signal_features_paths.features,
                )
            )

    def _load_hit_times_manifest_with_diagnostics(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_manifest: ArtifactManifestDocumentV2,
        slot_root: Path,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> ArtifactHitTimesManifestDocumentV2 | None:
        """
        Load the strict hit-times manifest while converting read failures into diagnostics.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_manifest: Root manifest used for path/hash cross-checks.
            slot_root: Absolute slot-root path.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            ArtifactHitTimesManifestDocumentV2 | None: Parsed hit-times manifest when loading
                succeeds.
        Assumptions:
            Root manifest already provides the canonical hit-times manifest reference.
        Raises:
            None.
        Side Effects:
            Reads one hit-times manifest file from disk and appends diagnostics on failure.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        hit_times_manifest_path = self.artifact_loader.resolve_hit_times_manifest_path(
            coordinates,
            slot,
        )
        expected_relative_path = _relative_slot_path_v2(slot_root, hit_times_manifest_path)
        if slot_manifest.hit_times.manifest_path != expected_relative_path:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_manifest_reference_path_mismatch",
                    message=(
                        "root manifest hit_times reference must use deterministic manifest path; "
                        f"got {slot_manifest.hit_times.manifest_path!r}, expected "
                        f"{expected_relative_path!r}"
                    ),
                    location="hit_times.manifest_path",
                    manifest_path=slot_manifest.path,
                    artifact_path=hit_times_manifest_path,
                )
            )
        if hit_times_manifest_path.is_file():
            actual_manifest_hash = _file_sha256_hex_v2(hit_times_manifest_path)
            if slot_manifest.hit_times.manifest_sha256 != actual_manifest_hash:
                diagnostics.append(
                    ArtifactValidationDiagnosticV2(
                        code="hit_times_manifest_reference_hash_mismatch",
                        message=(
                            "root manifest hit_times sha256 must match actual manifest; got "
                            f"{slot_manifest.hit_times.manifest_sha256!r}, expected "
                            f"{actual_manifest_hash!r}"
                        ),
                        location="hit_times.manifest_sha256",
                        manifest_path=slot_manifest.path,
                        artifact_path=hit_times_manifest_path,
                    )
                )
        try:
            return self.artifact_loader.load_hit_times_manifest(coordinates, slot)
        except (FileNotFoundError, ValueError) as error:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_manifest_invalid",
                    message=str(error),
                    location=f"hit_times/{HIT_TIMES_TIMEFRAME_LITERAL_V2}/manifest.yaml",
                    manifest_path=hit_times_manifest_path,
                )
            )
            return None

    def _validate_hit_times_manifest(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: ArtifactSlotLiteralV2,
        slot_root: Path,
        slot_manifest: ArtifactManifestDocumentV2,
        hit_times_manifest: ArtifactHitTimesManifestDocumentV2,
        hit_times_bar_count: int | None,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate the strict hit-times manifest against root contract and `.npy` contents.

        Args:
            coordinates: Artifact coordinates under validation.
            slot: Explicit slot literal under validation.
            slot_root: Absolute slot-root path.
            slot_manifest: Root manifest used for cross-contract checks.
            hit_times_manifest: Parsed strict hit-times manifest.
            hit_times_bar_count: Root timeline length for `HIT_TIMES_TIMEFRAME_LITERAL_V2` when
                available.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Hit-times tables operate only on the `1m` runtime timeline.
        Raises:
            None.
        Side Effects:
            Reads hit-times arrays from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if hit_times_manifest.slot_generation != slot_manifest.slot_generation:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_manifest_slot_generation_mismatch",
                    message=(
                        "hit_times manifest slot_generation must match root manifest; got "
                        f"{hit_times_manifest.slot_generation!r}, expected "
                        f"{slot_manifest.slot_generation!r}"
                    ),
                    location="slot_generation",
                    manifest_path=hit_times_manifest.path,
                )
            )
        if hit_times_manifest.asof_date != slot_manifest.asof_date:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_manifest_asof_date_mismatch",
                    message=(
                        "hit_times manifest asof_date must match root manifest; got "
                        f"{hit_times_manifest.asof_date!r}, expected "
                        f"{slot_manifest.asof_date!r}"
                    ),
                    location="asof_date",
                    manifest_path=hit_times_manifest.path,
                )
            )
        if (
            hit_times_bar_count is not None
            and hit_times_manifest.timeline_bar_count != hit_times_bar_count
        ):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_timeline_bar_count_mismatch",
                    message=(
                        "hit_times timeline_bar_count must match root hit-times timeframe "
                        "coverage; got "
                        f"{hit_times_manifest.timeline_bar_count!r}, expected "
                        f"{hit_times_bar_count!r}"
                    ),
                    location="timeline_bar_count",
                    manifest_path=hit_times_manifest.path,
                )
            )
        if hit_times_manifest.sentinel_index != hit_times_manifest.timeline_bar_count:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_sentinel_mismatch",
                    message=(
                        "hit_times sentinel_index must equal timeline_bar_count; got "
                        f"{hit_times_manifest.sentinel_index!r}, expected "
                        f"{hit_times_manifest.timeline_bar_count!r}"
                    ),
                    location="sentinel_index",
                    manifest_path=hit_times_manifest.path,
                )
            )

        hit_times_paths = self.artifact_loader.resolve_hit_times_paths(coordinates, slot)
        tp_values_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="hit_times.tp_values",
            metadata=hit_times_manifest.tp_values,
            expected_path=hit_times_paths.tp_values,
            expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
            expected_shape=None,
            diagnostics=diagnostics,
        )
        sl_values_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="hit_times.sl_values",
            metadata=hit_times_manifest.sl_values,
            expected_path=hit_times_paths.sl_values,
            expected_dtype=ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2,
            expected_shape=None,
            diagnostics=diagnostics,
        )

        if tp_values_array is not None and not np.all(np.diff(tp_values_array) > 0):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_tp_grid_not_strictly_increasing",
                    message="hit_times tp_values must be strictly increasing",
                    location="tp_values",
                    manifest_path=hit_times_manifest.path,
                    artifact_path=hit_times_paths.tp_values,
                )
            )
        if sl_values_array is not None and not np.all(np.diff(sl_values_array) > 0):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_sl_grid_not_strictly_increasing",
                    message="hit_times sl_values must be strictly increasing",
                    location="sl_values",
                    manifest_path=hit_times_manifest.path,
                    artifact_path=hit_times_paths.sl_values,
                )
            )

        self._validate_hit_times_table(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="tables.long_tp",
            table_manifest=hit_times_manifest.long_tp,
            expected_path=hit_times_paths.long_tp,
            expected_level_count=hit_times_manifest.tp_values.shape[0],
            expected_timeline_bar_count=hit_times_manifest.timeline_bar_count,
            sentinel_index=hit_times_manifest.sentinel_index,
            diagnostics=diagnostics,
        )
        self._validate_hit_times_table(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="tables.long_sl",
            table_manifest=hit_times_manifest.long_sl,
            expected_path=hit_times_paths.long_sl,
            expected_level_count=hit_times_manifest.sl_values.shape[0],
            expected_timeline_bar_count=hit_times_manifest.timeline_bar_count,
            sentinel_index=hit_times_manifest.sentinel_index,
            diagnostics=diagnostics,
        )
        self._validate_hit_times_table(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="tables.short_tp",
            table_manifest=hit_times_manifest.short_tp,
            expected_path=hit_times_paths.short_tp,
            expected_level_count=hit_times_manifest.tp_values.shape[0],
            expected_timeline_bar_count=hit_times_manifest.timeline_bar_count,
            sentinel_index=hit_times_manifest.sentinel_index,
            diagnostics=diagnostics,
        )
        self._validate_hit_times_table(
            slot_root=slot_root,
            manifest_path=hit_times_manifest.path,
            location="tables.short_sl",
            table_manifest=hit_times_manifest.short_sl,
            expected_path=hit_times_paths.short_sl,
            expected_level_count=hit_times_manifest.sl_values.shape[0],
            expected_timeline_bar_count=hit_times_manifest.timeline_bar_count,
            sentinel_index=hit_times_manifest.sentinel_index,
            diagnostics=diagnostics,
        )

    def _validate_hit_times_table(
        self,
        *,
        slot_root: Path,
        manifest_path: Path,
        location: str,
        table_manifest: ArtifactHitTimesTableManifestV2,
        expected_path: Path,
        expected_level_count: int,
        expected_timeline_bar_count: int,
        sentinel_index: int,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate one strict hit-times lookup table against declared invariants.

        Args:
            slot_root: Absolute slot-root path.
            manifest_path: Hit-times manifest path used in diagnostics.
            location: Stable diagnostic location label.
            table_manifest: Typed hit-times table metadata.
            expected_path: Explicit deterministic array path.
            expected_level_count: Expected TP/SL level count.
            expected_timeline_bar_count: Expected `1m` timeline length.
            sentinel_index: Expected sentinel upper bound.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Hit-times tables are monotone by level and bounded by sentinel index.
        Raises:
            None.
        Side Effects:
            Reads one hit-times table from disk and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        table_array = self._validate_array_metadata_and_load(
            slot_root=slot_root,
            manifest_path=manifest_path,
            location=location,
            metadata=table_manifest.array,
            expected_path=expected_path,
            expected_dtype=ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
            expected_axis_order=ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2,
            expected_shape=(expected_level_count, expected_timeline_bar_count),
            diagnostics=diagnostics,
        )
        if table_array is None:
            return
        if table_array.shape[0] > 1 and not np.all(table_array[1:, :] >= table_array[:-1, :]):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_table_not_monotone",
                    message=f"{location} must satisfy hit-time monotonicity by level",
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if not np.all(table_array <= sentinel_index):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="hit_times_table_out_of_bounds",
                    message=(
                        f"{location} values must stay within [0, {sentinel_index}] "
                        "under hit-time monotonicity contract"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )

    def _validate_array_metadata_and_load(
        self,
        *,
        slot_root: Path,
        manifest_path: Path,
        location: str,
        metadata: ArtifactArrayMetadataV2,
        expected_path: Path,
        expected_dtype: str,
        expected_axis_order: tuple[str, ...],
        expected_shape: tuple[int, ...] | None,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> Any | None:
        """
        Validate one array metadata contract and load the actual `.npy` file when present.

        Args:
            slot_root: Absolute slot-root path.
            manifest_path: Source manifest path used in diagnostics.
            location: Stable diagnostic location label.
            metadata: Strict manifest array metadata.
            expected_path: Explicit deterministic file path.
            expected_dtype: Required dtype literal for this artifact family.
            expected_axis_order: Required axis order for this artifact family.
            expected_shape: Optional required shape derived from other manifest metadata.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            Any | None: Loaded ndarray-like object when file loading succeeds.
        Assumptions:
            Publish validation may mmap `.npy` files to inspect dtype, shape, and values.
        Raises:
            None.
        Side Effects:
            Computes file hash, loads a `.npy` file from disk, and appends diagnostics.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        expected_relative_path = _relative_slot_path_v2(slot_root, expected_path)
        if metadata.path != expected_relative_path:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="array_path_mismatch",
                    message=(
                        f"{location} must use deterministic relative path "
                        f"{expected_relative_path!r}; got {metadata.path!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if metadata.dtype != expected_dtype:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="array_dtype_contract_mismatch",
                    message=(
                        f"{location} dtype contract must be {expected_dtype!r}; got "
                        f"{metadata.dtype!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if metadata.axis_order != expected_axis_order:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="array_axis_order_contract_mismatch",
                    message=(
                        f"{location} axis_order contract must be {expected_axis_order!r}; got "
                        f"{metadata.axis_order!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if expected_shape is not None and metadata.shape != expected_shape:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="array_shape_contract_mismatch",
                    message=(
                        f"{location} shape contract must be {expected_shape!r}; got "
                        f"{metadata.shape!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if not expected_path.is_file():
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="artifact_file_missing",
                    message=f"{location} file is missing at {expected_path}",
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
            return None

        actual_hash = _file_sha256_hex_v2(expected_path)
        if metadata.sha256 != actual_hash:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="artifact_file_hash_mismatch",
                    message=(
                        f"{location} sha256 must match file contents; got "
                        f"{metadata.sha256!r}, expected {actual_hash!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        try:
            array = np.load(expected_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="artifact_file_not_loadable",
                    message=f"{location} could not be loaded as .npy: {error}",
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
            return None

        actual_dtype = array.dtype.name
        actual_shape = tuple(int(value) for value in array.shape)
        if actual_dtype != metadata.dtype:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="artifact_array_dtype_mismatch",
                    message=(
                        f"{location} actual dtype must match manifest metadata; got "
                        f"{actual_dtype!r}, expected {metadata.dtype!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        if actual_shape != metadata.shape:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="artifact_array_shape_mismatch",
                    message=(
                        f"{location} actual shape must match manifest metadata; got "
                        f"{actual_shape!r}, expected {metadata.shape!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                    artifact_path=expected_path,
                )
            )
        return array

    def _validate_monotone_time_array(
        self,
        *,
        array: Any,
        manifest_path: Path,
        location: str,
        code: str,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate that one integer timeline array is strictly increasing.

        Args:
            array: Loaded integer timeline array.
            manifest_path: Source manifest path used in diagnostics.
            location: Stable diagnostic location label.
            code: Stable machine-readable diagnostic code.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Price open/close timelines are strictly increasing by bar index.
        Raises:
            None.
        Side Effects:
            Appends a diagnostic when monotonicity is violated.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if array.shape[0] > 1 and not np.all(array[1:] > array[:-1]):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code=code,
                    message=f"{location} must be strictly increasing",
                    location=location,
                    manifest_path=manifest_path,
                )
            )

    def _validate_timeline_coverage_edges(
        self,
        *,
        coverage: ArtifactTimelineCoverageV2,
        array: Any,
        manifest_path: Path,
        location: str,
        expected_start: int,
        expected_end: int,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate that timeline coverage metadata matches the first/last array elements.

        Args:
            coverage: Strict timeline coverage metadata.
            array: Loaded integer timeline array.
            manifest_path: Source manifest path used in diagnostics.
            location: Stable diagnostic location label.
            expected_start: Expected first timestamp literal from coverage metadata.
            expected_end: Expected last timestamp literal from coverage metadata.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Timeline coverage start/end fields must mirror the materialized arrays exactly.
        Raises:
            None.
        Side Effects:
            Appends diagnostics when metadata diverges from actual array edges.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if int(array.shape[0]) != coverage.bar_count:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="timeline_bar_count_mismatch",
                    message=(
                        f"{location} coverage.bar_count must match array length; got "
                        f"{coverage.bar_count!r}, expected {int(array.shape[0])!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                )
            )
        if int(array[0]) != expected_start:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="timeline_start_mismatch",
                    message=(
                        f"{location} start value must match manifest coverage; got "
                        f"{int(array[0])!r}, expected {expected_start!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                )
            )
        if int(array[-1]) != expected_end:
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code="timeline_end_mismatch",
                    message=(
                        f"{location} end value must match manifest coverage; got "
                        f"{int(array[-1])!r}, expected {expected_end!r}"
                    ),
                    location=location,
                    manifest_path=manifest_path,
                )
            )

    def _validate_non_decreasing_index_array(
        self,
        *,
        array: Any,
        manifest_path: Path,
        location: str,
        code: str,
        diagnostics: list[ArtifactValidationDiagnosticV2],
    ) -> None:
        """
        Validate that one unsigned integer mapping array is non-decreasing.

        Args:
            array: Loaded mapping index array.
            manifest_path: Source manifest path used in diagnostics.
            location: Stable diagnostic location label.
            code: Stable machine-readable diagnostic code.
            diagnostics: Mutable diagnostics buffer to extend.
        Returns:
            None.
        Assumptions:
            Request-TF to `1m` mapping indexes must be monotone by bar position.
        Raises:
            None.
        Side Effects:
            Appends a diagnostic when monotonicity is violated.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if array.shape[0] > 1 and not np.all(array[1:] >= array[:-1]):
            diagnostics.append(
                ArtifactValidationDiagnosticV2(
                    code=code,
                    message=f"{location} must be non-decreasing",
                    location=location,
                    manifest_path=manifest_path,
                )
            )


def _file_sha256_hex_v2(path: Path) -> str:
    """
    Compute a lowercase SHA-256 hex digest for one file.

    Args:
        path: Existing file path to hash.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        Publish validation may hash full artifact files because it runs outside runtime hot path.
    Raises:
        OSError: If the file cannot be read.
    Side Effects:
        Reads the file from disk in binary mode.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_slot_path_v2(slot_root: Path, artifact_path: Path) -> str:
    """
    Convert one absolute artifact path into the canonical slot-relative manifest literal.

    Args:
        slot_root: Absolute slot-root path.
        artifact_path: Absolute artifact file path located under the slot root.
    Returns:
        str: POSIX-style relative path literal from slot root to artifact file.
    Assumptions:
        Validators compare manifest paths to explicit deterministic resolver outputs.
    Raises:
        ValueError: If the artifact path is outside the slot root.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    return artifact_path.relative_to(slot_root).as_posix()
