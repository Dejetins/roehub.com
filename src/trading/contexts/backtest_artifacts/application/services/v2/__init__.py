"""Minimal v2 export surface for backtest_artifacts precompute/publish services."""

from .artifact_availability_summary import (
    BacktestArtifactAvailabilitySummaryGeneratorV2,
    BacktestArtifactAvailabilitySummaryResultV2,
)
from .artifact_manifest_loader import YamlBacktestArtifactLoaderV2
from .artifact_precompute_runner import (
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactCoordinatesV2,
    ArtifactStageRebuildStatsCollectionV2,
    BacktestArtifactPrecomputeRunnerV2,
)
from .artifact_slot_publisher import (
    ArtifactPublishPrecheckV2,
    ArtifactPublishResultV2,
    ArtifactSlotPublishErrorV2,
    BacktestArtifactSlotPublisherV2,
)
from .contracts import (
    ALLOWED_ARTIFACT_SLOTS_V2,
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    AVAILABILITY_SUMMARY_FILENAME_V2,
    ArtifactPrecomputeExecutionPolicyV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotValidationSpecV2,
    ArtifactTailRebuildBarsV2,
    artifact_market_id_from_coordinates_v2,
    ordered_artifact_slots_v2,
    validate_artifact_slot_v2,
    validate_current_pointer_published_at_utc_v2,
    validate_indicator_id_v2,
    validate_mapping_timeframe_v2,
    validate_price_timeframe_v2,
    validate_signal_timeframe_v2,
)
from .signal_rules_engine_v2 import supported_indicator_ids_for_signal_rules_v2

supported_indicator_ids_for_signals_v1 = supported_indicator_ids_for_signal_rules_v2

__all__ = [
    "ALLOWED_ARTIFACT_SLOTS_V2",
    "ARTIFACT_MAPPING_TIMEFRAMES_V2",
    "ARTIFACT_PRICE_TIMEFRAMES_V2",
    "ARTIFACT_SIGNAL_TIMEFRAMES_V2",
    "AVAILABILITY_SUMMARY_FILENAME_V2",
    "ArtifactCanonicalPriceExportRequestV2",
    "ArtifactCanonicalPriceExportResultV2",
    "ArtifactCoordinatesV2",
    "ArtifactPrecomputeExecutionPolicyV2",
    "ArtifactPrecomputeRuntimeSettingsV2",
    "ArtifactPublishPrecheckV2",
    "ArtifactPublishResultV2",
    "ArtifactSignalValidationSpecV2",
    "ArtifactSlotPublishErrorV2",
    "ArtifactSlotValidationSpecV2",
    "ArtifactStageRebuildStatsCollectionV2",
    "ArtifactTailRebuildBarsV2",
    "BacktestArtifactAvailabilitySummaryGeneratorV2",
    "BacktestArtifactAvailabilitySummaryResultV2",
    "BacktestArtifactPrecomputeRunnerV2",
    "BacktestArtifactSlotPublisherV2",
    "YamlBacktestArtifactLoaderV2",
    "artifact_market_id_from_coordinates_v2",
    "ordered_artifact_slots_v2",
    "supported_indicator_ids_for_signals_v1",
    "supported_indicator_ids_for_signal_rules_v2",
    "validate_artifact_slot_v2",
    "validate_current_pointer_published_at_utc_v2",
    "validate_indicator_id_v2",
    "validate_mapping_timeframe_v2",
    "validate_price_timeframe_v2",
    "validate_signal_timeframe_v2",
]
