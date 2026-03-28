"""Shared slot-pinned runtime bootstrap over strict backtest artifact identities (R6-01)."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactManifestDocumentV2,
    ArtifactPinnedIdentityV2,
    ArtifactSlotLiteralV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactSlotResolverV2,
)


@dataclass(frozen=True, slots=True)
class ArtifactSlotResolverV2(BacktestArtifactSlotResolverV2):
    """
    Resolve one immutable slot-pinned context from `current.yaml` or persisted pin metadata.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    artifact_loader: BacktestArtifactLoaderV2

    def resolve_active_context(
        self,
        coordinates: ArtifactCoordinatesV2,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """
        Resolve the active slot-pinned context from strict `current.yaml`.

        Args:
            coordinates: Artifact coordinates for one `(exchange, market_type, symbol)` root.
        Returns:
            ArtifactSlotPinnedRuntimeContextV2: Shared immutable slot-pinned context for runtime.
        Assumptions:
            Active runtime startup pins the currently published slot identity once and must not
            scan directories or recompute manifest hashes afterward.
        Raises:
            FileNotFoundError: If `current.yaml` or the referenced slot manifest is missing.
            ValueError: If `current.yaml` and the resolved slot manifest disagree on identity.
        Side Effects:
            Reads strict `current.yaml` and one explicit slot `manifest.yaml` from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
        """
        current_pointer = self.artifact_loader.load_current_pointer(coordinates)
        slot_manifest = self._load_slot_manifest(
            coordinates=coordinates,
            artifact_slot=current_pointer.active_slot,
        )
        self._validate_current_pointer_alignment(
            current_pointer=current_pointer,
            slot_manifest=slot_manifest,
        )
        return ArtifactSlotPinnedRuntimeContextV2(
            coordinates=coordinates,
            artifact_slot=current_pointer.active_slot,
            slot_generation=current_pointer.slot_generation,
            artifact_asof_date=current_pointer.asof_date,
            artifact_manifest_hash=current_pointer.manifest_sha256,
            slot_root_path=slot_manifest.path.parent,
            slot_manifest_path=slot_manifest.path,
            slot_manifest=slot_manifest,
        )

    def resolve_pinned_context(
        self,
        coordinates: ArtifactCoordinatesV2,
        pinned_identity: ArtifactPinnedIdentityV2,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """
        Resolve a slot-pinned context from persisted run pin metadata only.

        Args:
            coordinates: Artifact coordinates for one `(exchange, market_type, symbol)` root.
            pinned_identity: Persisted immutable slot identity captured at job creation time.
        Returns:
            ArtifactSlotPinnedRuntimeContextV2: Shared immutable slot-pinned context for runtime.
        Assumptions:
            Published slots are immutable, so background runs reuse persisted
            `artifact_slot/slot_generation/artifact_asof_date/artifact_manifest_hash` without
            consulting `current.yaml`.
        Raises:
            FileNotFoundError: If the pinned slot manifest is missing.
            ValueError: If the pinned manifest drifts from persisted slot/date/generation fields.
        Side Effects:
            Reads one explicit slot `manifest.yaml` from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
        """
        slot_manifest = self._load_slot_manifest(
            coordinates=coordinates,
            artifact_slot=pinned_identity.artifact_slot,
        )
        self._validate_pinned_identity_alignment(
            pinned_identity=pinned_identity,
            slot_manifest=slot_manifest,
        )
        return ArtifactSlotPinnedRuntimeContextV2(
            coordinates=coordinates,
            artifact_slot=pinned_identity.artifact_slot,
            slot_generation=pinned_identity.slot_generation,
            artifact_asof_date=pinned_identity.artifact_asof_date,
            artifact_manifest_hash=pinned_identity.artifact_manifest_hash,
            slot_root_path=slot_manifest.path.parent,
            slot_manifest_path=slot_manifest.path,
            slot_manifest=slot_manifest,
        )

    def _load_slot_manifest(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        artifact_slot: ArtifactSlotLiteralV2,
    ) -> ArtifactManifestDocumentV2:
        """
        Load one slot manifest from its deterministic explicit path.

        Args:
            coordinates: Artifact coordinates for one symbol root.
            artifact_slot: Candidate slot literal resolved from pointer or pin metadata.
        Returns:
            ArtifactManifestDocumentV2: Strict root manifest for that slot.
        Assumptions:
            Runtime bootstrap must address slot manifests via deterministic paths only.
        Raises:
            FileNotFoundError: If the explicit slot manifest path does not exist.
            ValueError: If the slot literal or manifest payload is invalid.
        Side Effects:
            Reads one explicit slot `manifest.yaml` from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        manifest_path = self.artifact_loader.resolve_slot_manifest_path(
            coordinates,
            artifact_slot,
        )
        return self.artifact_loader.load_manifest_from_path(
            manifest_path,
            slot=artifact_slot,
        )

    def _validate_current_pointer_alignment(
        self,
        *,
        current_pointer: ArtifactCurrentPointerV2,
        slot_manifest: ArtifactManifestDocumentV2,
    ) -> None:
        """
        Fail fast when active `current.yaml` identity drifts from the loaded slot manifest.

        Args:
            current_pointer: Strict `current.yaml` payload.
            slot_manifest: Loaded slot root manifest for the active slot.
        Returns:
            None.
        Assumptions:
            Active runtime startup must reject slot/date/generation drift before hot-path loading.
        Raises:
            ValueError: If slot, generation, or as-of date differ between pointer and manifest.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if slot_manifest.slot != current_pointer.active_slot:
            raise ValueError(
                "slot manifest slot must match current.yaml active_slot; got "
                f"{slot_manifest.slot!r}, expected {current_pointer.active_slot!r}"
            )
        if slot_manifest.slot_generation != current_pointer.slot_generation:
            raise ValueError(
                "slot manifest slot_generation must match current.yaml slot_generation; got "
                f"{slot_manifest.slot_generation!r}, expected "
                f"{current_pointer.slot_generation!r}"
            )
        if slot_manifest.asof_date != current_pointer.asof_date:
            raise ValueError(
                "slot manifest asof_date must match current.yaml asof_date; got "
                f"{slot_manifest.asof_date!r}, expected {current_pointer.asof_date!r}"
            )

    def _validate_pinned_identity_alignment(
        self,
        *,
        pinned_identity: ArtifactPinnedIdentityV2,
        slot_manifest: ArtifactManifestDocumentV2,
    ) -> None:
        """
        Fail fast when a pinned background identity drifts from the loaded slot manifest.

        Args:
            pinned_identity: Persisted immutable slot identity from job metadata.
            slot_manifest: Loaded slot root manifest for the pinned slot.
        Returns:
            None.
        Assumptions:
            Background runtime startup uses persisted pin metadata rather than `current.yaml`.
        Raises:
            ValueError: If slot, generation, or as-of date differ between pin and manifest.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if slot_manifest.slot != pinned_identity.artifact_slot:
            raise ValueError(
                "slot manifest slot must match persisted artifact_slot; got "
                f"{slot_manifest.slot!r}, expected {pinned_identity.artifact_slot!r}"
            )
        if slot_manifest.slot_generation != pinned_identity.slot_generation:
            raise ValueError(
                "slot manifest slot_generation must match persisted slot_generation; got "
                f"{slot_manifest.slot_generation!r}, expected "
                f"{pinned_identity.slot_generation!r}"
            )
        if slot_manifest.asof_date != pinned_identity.artifact_asof_date:
            raise ValueError(
                "slot manifest asof_date must match persisted artifact_asof_date; got "
                f"{slot_manifest.asof_date!r}, expected "
                f"{pinned_identity.artifact_asof_date!r}"
            )
