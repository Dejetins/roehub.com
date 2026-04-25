"""Filesystem path builder for deterministic backtest artifact store v2 (R2-01)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_MANIFEST_FILENAME_V2,
    ARTIFACT_STORE_V2_ROOT_LITERAL,
    BAR_CLOSE_MAPPING_FILENAME_V2,
    BAR_OPEN_MAPPING_FILENAME_V2,
    CLOSE_TIME_FILENAME_V2,
    CURRENT_ARTIFACT_POINTER_FILENAME_V2,
    HIT_TIMES_DIRECTORY_LITERAL_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    LONG_SL_FILENAME_V2,
    LONG_TP_FILENAME_V2,
    MAPPINGS_DIRECTORY_LITERAL_V2,
    OHLCV_FILENAME_V2,
    OPEN_TIME_FILENAME_V2,
    PRICES_DIRECTORY_LITERAL_V2,
    SHORT_SL_FILENAME_V2,
    SHORT_TP_FILENAME_V2,
    SIGNAL_FEATURES_DIRECTORY_LITERAL_V2,
    SIGNAL_FEATURES_FILENAME_V2,
    SIGNALS_DIRECTORY_LITERAL_V2,
    SIGNALS_FILENAME_V2,
    SL_VALUES_FILENAME_V2,
    TP_VALUES_FILENAME_V2,
    ArtifactCoordinatesV2,
    ArtifactHitTimesPathsV2,
    ArtifactMappingPathsV2,
    ArtifactPricePathsV2,
    ArtifactSignalFeaturesPathsV2,
    ArtifactSignalPathsV2,
    ArtifactSlotLiteralV2,
    BacktestArtifactPathResolverV2,
    ordered_artifact_slots_v2,
    validate_artifact_slot_v2,
    validate_hit_times_timeframe_v2,
    validate_indicator_id_v2,
    validate_mapping_timeframe_v2,
    validate_price_timeframe_v2,
    validate_signal_timeframe_v2,
)


@dataclass(frozen=True, slots=True)
class BacktestArtifactPathBuilderV2(BacktestArtifactPathResolverV2):
    """
    Pure deterministic path builder for the R2-01 artifact store filesystem contract.

    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    root: Path = Path(ARTIFACT_STORE_V2_ROOT_LITERAL)

    def ordered_slots(self) -> tuple[ArtifactSlotLiteralV2, ...]:
        """
        Return the fixed slot order used by runtime-facing callers.

        Args:
            None.
        Returns:
            tuple[ArtifactSlotLiteralV2, ...]: Canonical ordered slot literals.
        Assumptions:
            Slot order must stay stable even if no directories exist on disk yet.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        return ordered_artifact_slots_v2()

    def symbol_root(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """
        Resolve the canonical symbol-root directory from validated artifact coordinates.

        Args:
            coordinates: Validated artifact coordinates.
        Returns:
            Path: `<artifact_root>/<exchange>/<market_type>/<symbol>/`.
        Assumptions:
            Coordinates are already validated and are appended as verbatim path components.
        Raises:
            ValueError: If coordinates are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        return self.root / coordinates.exchange / coordinates.market_type / coordinates.symbol

    def current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """
        Resolve the canonical `current.yaml` path for one symbol root.

        Args:
            coordinates: Validated artifact coordinates.
        Returns:
            Path: `<artifact_root>/<exchange>/<market_type>/<symbol>/current.yaml`.
        Assumptions:
            Pointer lookup never scans sibling directories to discover slots.
        Raises:
            ValueError: If coordinates are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        return self.symbol_root(coordinates) / CURRENT_ARTIFACT_POINTER_FILENAME_V2

    def slot_root(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """
        Resolve the canonical slot root for one symbol root and slot literal.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            Path: `<artifact_root>/<exchange>/<market_type>/<symbol>/<slot>/`.
        Assumptions:
            Only two fixed slots are valid during R2-01.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        validated_slot = validate_artifact_slot_v2(slot)
        return self.symbol_root(coordinates) / validated_slot

    def slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """
        Resolve the canonical slot `manifest.yaml` path.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            Path: `<artifact_root>/<exchange>/<market_type>/<symbol>/<slot>/manifest.yaml`.
        Assumptions:
            Slot manifests always live at the slot root.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        return self.slot_root(coordinates, slot) / ARTIFACT_MANIFEST_FILENAME_V2

    def price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """
        Resolve explicit paths for one `prices/<tf>/` directory.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate price timeframe literal.
        Returns:
            ArtifactPricePathsV2: Deterministic path set for price artifacts.
        Assumptions:
            Price artifacts are addressed directly without runtime filesystem scanning.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        prices_directory = self._timeframe_directory(
            coordinates=coordinates,
            slot=slot,
            top_level_directory=PRICES_DIRECTORY_LITERAL_V2,
            timeframe=validate_price_timeframe_v2(timeframe),
        )
        return ArtifactPricePathsV2(
            open_time=prices_directory / OPEN_TIME_FILENAME_V2,
            close_time=prices_directory / CLOSE_TIME_FILENAME_V2,
            ohlcv=prices_directory / OHLCV_FILENAME_V2,
        )

    def signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """
        Resolve explicit paths for one `signals/<tf>/<indicator_id>/` directory.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate signal timeframe literal.
            indicator_id: Candidate indicator id token.
        Returns:
            ArtifactSignalPathsV2: Deterministic path set for signal artifacts.
        Assumptions:
            Indicator ids remain single safe path tokens even when they contain dots.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        indicator_directory = self._timeframe_directory(
            coordinates=coordinates,
            slot=slot,
            top_level_directory=SIGNALS_DIRECTORY_LITERAL_V2,
            timeframe=validate_signal_timeframe_v2(timeframe),
        ) / validate_indicator_id_v2(indicator_id)
        return ArtifactSignalPathsV2(
            manifest=indicator_directory / ARTIFACT_MANIFEST_FILENAME_V2,
            signals=indicator_directory / SIGNALS_FILENAME_V2,
        )

    def signal_features_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalFeaturesPathsV2:
        """
        Resolve explicit paths for one `signal_features/<tf>/<indicator_id>/` directory.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate signal timeframe literal.
            indicator_id: Candidate indicator id token.
        Returns:
            ArtifactSignalFeaturesPathsV2: Deterministic path set for signal-feature artifacts.
        Assumptions:
            Signal-feature families mirror signal target coordinates and must stay directly
            addressable without runtime filesystem scanning.
        Raises:
            ValueError: If one input literal violates the explicit artifact contract.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
        """
        indicator_directory = self._timeframe_directory(
            coordinates=coordinates,
            slot=slot,
            top_level_directory=SIGNAL_FEATURES_DIRECTORY_LITERAL_V2,
            timeframe=validate_signal_timeframe_v2(timeframe),
        ) / validate_indicator_id_v2(indicator_id)
        return ArtifactSignalFeaturesPathsV2(
            manifest=indicator_directory / ARTIFACT_MANIFEST_FILENAME_V2,
            features=indicator_directory / SIGNAL_FEATURES_FILENAME_V2,
        )

    def mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """
        Resolve explicit paths for one `mappings/<tf>/` directory.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate mapping timeframe literal.
        Returns:
            ArtifactMappingPathsV2: Deterministic path set for mapping artifacts.
        Assumptions:
            Mapping artifacts are addressed directly without runtime filesystem scanning.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        mappings_directory = self._timeframe_directory(
            coordinates=coordinates,
            slot=slot,
            top_level_directory=MAPPINGS_DIRECTORY_LITERAL_V2,
            timeframe=validate_mapping_timeframe_v2(timeframe),
        )
        return ArtifactMappingPathsV2(
            bar_open_1m_idx=mappings_directory / BAR_OPEN_MAPPING_FILENAME_V2,
            bar_close_1m_idx=mappings_directory / BAR_CLOSE_MAPPING_FILENAME_V2,
        )

    def hit_times_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """
        Resolve the fixed `hit_times/15m/manifest.yaml` path for one slot.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            Path: Deterministic hit-times manifest path.
        Assumptions:
            R2-01 fixes hit-times layout under `hit_times/15m/`.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        return (
            self.slot_root(coordinates, slot)
            / HIT_TIMES_DIRECTORY_LITERAL_V2
            / validate_hit_times_timeframe_v2(HIT_TIMES_TIMEFRAME_LITERAL_V2)
            / ARTIFACT_MANIFEST_FILENAME_V2
        )

    def hit_times_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesPathsV2:
        """
        Resolve explicit paths for the fixed `hit_times/15m/` artifact family.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            ArtifactHitTimesPathsV2: Deterministic path set for hit-times artifacts.
        Assumptions:
            R2-03 keeps hit-times files under one fixed `hit_times/15m/` directory.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        hit_times_directory = (
            self.slot_root(coordinates, slot)
            / HIT_TIMES_DIRECTORY_LITERAL_V2
            / validate_hit_times_timeframe_v2(HIT_TIMES_TIMEFRAME_LITERAL_V2)
        )
        return ArtifactHitTimesPathsV2(
            manifest=hit_times_directory / ARTIFACT_MANIFEST_FILENAME_V2,
            tp_values=hit_times_directory / TP_VALUES_FILENAME_V2,
            sl_values=hit_times_directory / SL_VALUES_FILENAME_V2,
            long_tp=hit_times_directory / LONG_TP_FILENAME_V2,
            long_sl=hit_times_directory / LONG_SL_FILENAME_V2,
            short_tp=hit_times_directory / SHORT_TP_FILENAME_V2,
            short_sl=hit_times_directory / SHORT_SL_FILENAME_V2,
        )

    def _timeframe_directory(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        top_level_directory: str,
        timeframe: str,
    ) -> Path:
        """
        Resolve one `<slot>/<top_level_directory>/<timeframe>/` directory.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            top_level_directory: One canonical top-level artifact directory name.
            timeframe: Validated timeframe literal for the directory.
        Returns:
            Path: Deterministic timeframe directory path.
        Assumptions:
            Top-level directory literals are internal constants, not user input.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        return self.slot_root(coordinates, slot) / top_level_directory / timeframe
