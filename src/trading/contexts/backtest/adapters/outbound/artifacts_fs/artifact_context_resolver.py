from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)
from trading.contexts.backtest.application.ports import BacktestArtifactContextUnavailable
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    BacktestArtifactLoaderV2,
)


@dataclass(frozen=True, slots=True)
class FilesystemBacktestArtifactContextResolver:
    """
    Resolve active artifact context from trusted filesystem current pointer and manifests.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/artifact_context.py
      - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
    """

    artifact_loader: BacktestArtifactLoaderV2

    def resolve_context(
        self,
        *,
        coordinates: BacktestCoordinates,
    ) -> BacktestArtifactMetadata:
        """
        Resolve the currently published artifact metadata for normalized coordinates.

        Args:
            coordinates: Normalized public request coordinates.
        Returns:
            BacktestArtifactMetadata: Current slot and manifest hashes.
        Assumptions:
            The loader is already wired to a trusted root from runtime config. No user-supplied
            path is accepted by this adapter.
        Raises:
            BacktestArtifactContextUnavailable: If current pointer or manifests are unavailable.
        Side Effects:
            Reads current pointer, slot manifest, and hit-times manifest metadata from disk.
        """
        artifact_coordinates = ArtifactCoordinatesV2(
            exchange=coordinates.exchange,
            market_type=coordinates.market_type,
            symbol=coordinates.symbol,
        )
        try:
            current_pointer = self.artifact_loader.load_current_pointer(artifact_coordinates)
            root_manifest = self.artifact_loader.load_slot_manifest(
                artifact_coordinates,
                current_pointer.active_slot,
            )
            hit_times_manifest_hash = root_manifest.hit_times.manifest_sha256
            funding_manifest = root_manifest.funding
            self.artifact_loader.load_hit_times_manifest(
                artifact_coordinates,
                current_pointer.active_slot,
            )
        except (FileNotFoundError, ValueError) as error:
            raise BacktestArtifactContextUnavailable(str(error)) from error

        return BacktestArtifactMetadata(
            artifact_slot=current_pointer.active_slot,
            artifact_slot_generation=current_pointer.slot_generation,
            artifact_manifest_hash=current_pointer.manifest_sha256,
            artifact_asof_date=current_pointer.asof_date,
            hit_times_manifest_hash=hit_times_manifest_hash,
            published_at_utc=current_pointer.published_at_utc,
            funding_manifest_hash=(
                None if funding_manifest is None else funding_manifest.funding_manifest_hash
            ),
            funding_coverage_status=(
                None if funding_manifest is None else funding_manifest.coverage_status
            ),
            funding_coverage_policy=(
                None if funding_manifest is None else funding_manifest.coverage_policy
            ),
            funding_rows_count=None if funding_manifest is None else funding_manifest.rows_count,
            funding_expected_event_count=(
                None if funding_manifest is None else funding_manifest.expected_event_count
            ),
            funding_missing_event_count=(
                None if funding_manifest is None else funding_manifest.missing_event_count
            ),
            funding_reason_codes=(
                () if funding_manifest is None else funding_manifest.reason_codes
            ),
        )


__all__ = ["FilesystemBacktestArtifactContextResolver"]
