"""Explicit-path YAML loader for deterministic backtest artifact store v2 (R2-01)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    ARTIFACT_MANIFEST_FILENAME_V2,
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    CURRENT_ARTIFACT_POINTER_FILENAME_V2,
    HIT_TIMES_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
    ROOT_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
    SIGNAL_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
    ArtifactArrayMetadataV2,
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactHitTimesPathsV2,
    ArtifactHitTimesReferenceV2,
    ArtifactHitTimesTableManifestV2,
    ArtifactManifestDocumentV2,
    ArtifactManifestProvenanceV2,
    ArtifactMappingPathsV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPricePathsV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalCatalogEntryV2,
    ArtifactSignalCatalogV2,
    ArtifactSignalEncodingContractV2,
    ArtifactSignalGridContractV2,
    ArtifactSignalManifestDocumentV2,
    ArtifactSignalPathsV2,
    ArtifactSlotLiteralV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactPathResolverV2,
    freeze_artifact_payload_mapping_v2,
    validate_artifact_slot_v2,
    validate_indicator_id_v2,
    validate_mapping_timeframe_v2,
    validate_price_timeframe_v2,
    validate_signal_timeframe_v2,
)


@dataclass(frozen=True, slots=True)
class YamlBacktestArtifactLoaderV2(BacktestArtifactLoaderV2):
    """
    Loader that reads `current.yaml` and slot manifests from explicit deterministic paths.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    path_resolver: BacktestArtifactPathResolverV2

    def load_current_pointer(self, coordinates: ArtifactCoordinatesV2) -> ArtifactCurrentPointerV2:
        """
        Read `current.yaml` for one coordinate triple via a known deterministic path.

        Args:
            coordinates: Validated artifact coordinates.
        Returns:
            ArtifactCurrentPointerV2: Parsed pointer document with typed slot identity.
        Assumptions:
            Pointer resolution must not depend on directory scanning or globbing.
        Raises:
            FileNotFoundError: If the deterministic pointer path does not exist.
            ValueError: If the YAML document is not a mapping or lacks valid `active_slot`.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.load_current_pointer_from_path(
            self.path_resolver.current_pointer_path(coordinates)
        )

    def load_current_pointer_from_path(self, path: Path) -> ArtifactCurrentPointerV2:
        """
        Read `current.yaml` from an explicit already-known filesystem path.

        Args:
            path: Full path to `current.yaml`.
        Returns:
            ArtifactCurrentPointerV2: Parsed strict pointer document with typed slot identity.
        Assumptions:
            R2-02 enforces the full `current.yaml` identity contract with exact required keys.
        Raises:
            FileNotFoundError: If the explicit pointer path does not exist.
            ValueError: If the YAML document is not a strict supported `current.yaml` payload.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = self._load_yaml_mapping(
            path=path,
            document_label=CURRENT_ARTIFACT_POINTER_FILENAME_V2,
        )
        return ArtifactCurrentPointerV2(
            path=path,
            active_slot=validate_artifact_slot_v2(
                str(self._required_yaml_field(payload=payload, key="active_slot", path=path))
            ),
            raw_payload=payload,
            schema_version=self._required_yaml_field(
                payload=payload,
                key="schema_version",
                path=path,
            ),
            slot_generation=self._required_yaml_field(
                payload=payload,
                key="slot_generation",
                path=path,
            ),
            asof_date=str(
                self._required_yaml_field(payload=payload, key="asof_date", path=path)
            ),
            manifest_sha256=str(
                self._required_yaml_field(
                    payload=payload,
                    key="manifest_sha256",
                    path=path,
                )
            ),
            published_at_utc=str(
                self._required_yaml_field(
                    payload=payload,
                    key="published_at_utc",
                    path=path,
                )
            ),
        )

    def load_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactManifestDocumentV2:
        """
        Read one slot `manifest.yaml` via deterministic coordinates and slot literal.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            ArtifactManifestDocumentV2: Parsed slot manifest document.
        Assumptions:
            Manifest resolution must not scan sibling directories to discover slots.
        Raises:
            FileNotFoundError: If the deterministic manifest path does not exist.
            ValueError: If the slot literal or YAML document violates the R2-01 contract.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        validated_slot = validate_artifact_slot_v2(slot)
        return self.load_manifest_from_path(
            self.path_resolver.slot_manifest_path(coordinates, validated_slot),
            slot=validated_slot,
        )

    def load_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactManifestDocumentV2:
        """
        Read one slot `manifest.yaml` from an explicit already-known filesystem path.

        Args:
            path: Full path to `manifest.yaml`.
            slot: Optional slot literal associated with the path.
        Returns:
            ArtifactManifestDocumentV2: Parsed slot manifest document.
        Assumptions:
            Detailed manifest schema checks are intentionally deferred beyond R2-01.
        Raises:
            FileNotFoundError: If the explicit manifest path does not exist.
            ValueError: If the YAML document is not a mapping or the slot is invalid.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = self._load_yaml_mapping(path=path, document_label=ARTIFACT_MANIFEST_FILENAME_V2)
        validated_slot = validate_artifact_slot_v2(slot)
        return self._parse_root_manifest_document(
            path=path,
            payload=payload,
            slot=validated_slot,
        )

    def load_signal_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalManifestDocumentV2:
        """
        Read one per-indicator signal manifest via deterministic coordinates and literals.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate signal timeframe literal.
            indicator_id: Candidate indicator identifier.
        Returns:
            ArtifactSignalManifestDocumentV2: Parsed strict signal manifest.
        Assumptions:
            Signal manifests are resolved only by explicit deterministic paths.
        Raises:
            FileNotFoundError: If the manifest path does not exist.
            ValueError: If inputs or YAML payload violate strict contracts.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        signal_paths = self.path_resolver.signal_paths(
            coordinates,
            validate_artifact_slot_v2(slot),
            validate_signal_timeframe_v2(timeframe),
            validate_indicator_id_v2(indicator_id),
        )
        return self.load_signal_manifest_from_path(
            signal_paths.manifest,
            slot=validate_artifact_slot_v2(slot),
        )

    def load_signal_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactSignalManifestDocumentV2:
        """
        Read one per-indicator signal manifest from an explicit already-known path.

        Args:
            path: Full path to signal `manifest.yaml`.
            slot: Explicit slot literal associated with the path.
        Returns:
            ArtifactSignalManifestDocumentV2: Parsed strict signal manifest.
        Assumptions:
            Signal manifests are schema-validated immediately on load.
        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the YAML payload violates strict contracts.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = self._load_yaml_mapping(path=path, document_label=ARTIFACT_MANIFEST_FILENAME_V2)
        return self._parse_signal_manifest_document(
            path=path,
            payload=payload,
            slot=validate_artifact_slot_v2(slot),
        )

    def load_hit_times_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesManifestDocumentV2:
        """
        Read the fixed `hit_times/1m/manifest.yaml` via deterministic coordinates.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            ArtifactHitTimesManifestDocumentV2: Parsed strict hit-times manifest.
        Assumptions:
            Hit-times manifest lookup is fixed to one explicit `1m` path.
        Raises:
            FileNotFoundError: If the manifest path does not exist.
            ValueError: If inputs or YAML payload violate strict contracts.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.load_hit_times_manifest_from_path(
            self.path_resolver.hit_times_manifest_path(
                coordinates,
                validate_artifact_slot_v2(slot),
            ),
            slot=validate_artifact_slot_v2(slot),
        )

    def load_hit_times_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactHitTimesManifestDocumentV2:
        """
        Read one `hit_times/1m/manifest.yaml` from an explicit already-known path.

        Args:
            path: Full path to `hit_times/1m/manifest.yaml`.
            slot: Explicit slot literal associated with the path.
        Returns:
            ArtifactHitTimesManifestDocumentV2: Parsed strict hit-times manifest.
        Assumptions:
            Hit-times manifests are schema-validated immediately on load.
        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the YAML payload violates strict contracts.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = self._load_yaml_mapping(path=path, document_label=ARTIFACT_MANIFEST_FILENAME_V2)
        return self._parse_hit_times_manifest_document(
            path=path,
            payload=payload,
            slot=validate_artifact_slot_v2(slot),
        )

    def load_active_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
    ) -> ArtifactManifestDocumentV2:
        """
        Read the active slot manifest by resolving the slot from `current.yaml`.

        Args:
            coordinates: Validated artifact coordinates.
        Returns:
            ArtifactManifestDocumentV2: Parsed manifest for the currently active slot.
        Assumptions:
            `current.yaml` is the single deterministic source of active slot identity.
        Raises:
            FileNotFoundError: If either `current.yaml` or the active manifest path is missing.
            ValueError: If either YAML document violates the R2-01 contract.
        Side Effects:
            Reads two UTF-8 YAML files from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        current_pointer = self.load_current_pointer(coordinates)
        return self.load_slot_manifest(coordinates, current_pointer.active_slot)

    def resolve_current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """
        Resolve `current.yaml` without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
        Returns:
            Path: Deterministic `current.yaml` path.
        Assumptions:
            Callers may use the returned path for explicit-path workflows outside this loader.
        Raises:
            ValueError: If coordinates are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.current_pointer_path(coordinates)

    def resolve_slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """
        Resolve one slot `manifest.yaml` path without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            Path: Deterministic slot manifest path.
        Assumptions:
            Slot choice must remain explicit and deterministic.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.slot_manifest_path(coordinates, slot)

    def resolve_price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """
        Resolve one `prices/<tf>/` path set without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate price timeframe literal.
        Returns:
            ArtifactPricePathsV2: Deterministic path set for price files.
        Assumptions:
            Runtime hot paths must reach price files directly without scanning directories.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.price_paths(coordinates, slot, timeframe)

    def resolve_signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """
        Resolve one `signals/<tf>/<indicator_id>/` path set without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate signal timeframe literal.
            indicator_id: Candidate indicator id token.
        Returns:
            ArtifactSignalPathsV2: Deterministic path set for signal files.
        Assumptions:
            Signal artifacts must be addressable directly from coordinates and literals.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.signal_paths(coordinates, slot, timeframe, indicator_id)

    def resolve_mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """
        Resolve one `mappings/<tf>/` path set without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
            timeframe: Candidate mapping timeframe literal.
        Returns:
            ArtifactMappingPathsV2: Deterministic path set for mapping files.
        Assumptions:
            Mapping artifacts must be addressable directly from coordinates and literals.
        Raises:
            ValueError: If one input literal violates the R2-01 contract.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.mapping_paths(coordinates, slot, timeframe)

    def resolve_hit_times_manifest_path(
        self, coordinates: ArtifactCoordinatesV2, slot: str
    ) -> Path:
        """
        Resolve `hit_times/1m/manifest.yaml` without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            Path: Deterministic hit-times manifest path.
        Assumptions:
            R2-01 fixes hit-times manifest lookup to a single known `1m` path.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.hit_times_manifest_path(coordinates, slot)

    def resolve_hit_times_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesPathsV2:
        """
        Resolve the fixed `hit_times/1m/` artifact paths without touching disk.

        Args:
            coordinates: Validated artifact coordinates.
            slot: Candidate slot literal.
        Returns:
            ArtifactHitTimesPathsV2: Deterministic hit-times path set.
        Assumptions:
            Hit-times artifacts must remain directly addressable without scanning.
        Raises:
            ValueError: If coordinates or slot are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        return self.path_resolver.hit_times_paths(coordinates, slot)

    def _parse_root_manifest_document(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactManifestDocumentV2:
        """
        Parse one strict root `manifest.yaml` payload into typed manifest DTOs.

        Args:
            path: Source manifest path.
            payload: Parsed YAML payload.
            slot: Explicit slot literal resolved from deterministic path.
        Returns:
            ArtifactManifestDocumentV2: Strict typed root manifest.
        Assumptions:
            R2-03 root manifests reject missing keys and unsupported nested drift.
        Raises:
            ValueError: If the payload shape or any nested strict field is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=ROOT_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=path,
        )
        manifest_slot = validate_artifact_slot_v2(
            str(self._required_yaml_field(payload=payload, key="slot", path=path))
        )
        if manifest_slot != slot:
            raise ValueError(f"{path} field 'slot' must be {slot!r}; got {manifest_slot!r}")
        return ArtifactManifestDocumentV2(
            path=path,
            raw_payload=payload,
            slot=manifest_slot,
            schema_version=self._required_yaml_field(
                payload=payload,
                key="schema_version",
                path=path,
            ),
            manifest_kind=str(
                self._required_yaml_field(payload=payload, key="manifest_kind", path=path)
            ),
            slot_generation=self._required_yaml_field(
                payload=payload,
                key="slot_generation",
                path=path,
            ),
            asof_date=str(self._required_yaml_field(payload=payload, key="asof_date", path=path)),
            identity=self._parse_identity_mapping(
                path=path,
                key="identity",
                payload=self._required_mapping_field(payload=payload, key="identity", path=path),
            ),
            prices=self._parse_price_manifests(
                path=path,
                values=self._required_sequence_field(payload=payload, key="prices", path=path),
            ),
            mappings=self._parse_mapping_manifests(
                path=path,
                values=self._required_sequence_field(payload=payload, key="mappings", path=path),
            ),
            signals=self._parse_signal_catalog(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="signals", path=path),
            ),
            hit_times=self._parse_hit_times_reference(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="hit_times", path=path),
            ),
            signal_encoding=self._parse_signal_encoding(
                path=path,
                payload=self._required_mapping_field(
                    payload=payload,
                    key="signal_encoding",
                    path=path,
                ),
            ),
            provenance=self._parse_provenance(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="provenance", path=path),
            ),
        )

    def _parse_signal_manifest_document(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactSignalManifestDocumentV2:
        """
        Parse one strict per-indicator signal manifest payload into typed DTOs.

        Args:
            path: Source manifest path.
            payload: Parsed YAML payload.
            slot: Explicit slot literal resolved from deterministic path.
        Returns:
            ArtifactSignalManifestDocumentV2: Strict typed signal manifest.
        Assumptions:
            Signal manifests are schema-validated eagerly during explicit-path reads.
        Raises:
            ValueError: If the payload shape or any nested strict field is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=SIGNAL_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=path,
        )
        manifest_slot = validate_artifact_slot_v2(
            str(self._required_yaml_field(payload=payload, key="slot", path=path))
        )
        if manifest_slot != slot:
            raise ValueError(f"{path} field 'slot' must be {slot!r}; got {manifest_slot!r}")
        return ArtifactSignalManifestDocumentV2(
            path=path,
            raw_payload=payload,
            slot=manifest_slot,
            schema_version=self._required_yaml_field(
                payload=payload,
                key="schema_version",
                path=path,
            ),
            manifest_kind=str(
                self._required_yaml_field(payload=payload, key="manifest_kind", path=path)
            ),
            slot_generation=self._required_yaml_field(
                payload=payload,
                key="slot_generation",
                path=path,
            ),
            asof_date=str(self._required_yaml_field(payload=payload, key="asof_date", path=path)),
            indicator_id=str(
                self._required_yaml_field(payload=payload, key="indicator_id", path=path)
            ),
            timeframe=str(self._required_yaml_field(payload=payload, key="timeframe", path=path)),
            signals=self._parse_array_metadata(
                path=path,
                key="signals",
                payload=self._required_mapping_field(payload=payload, key="signals", path=path),
            ),
            rows_count=self._required_yaml_field(payload=payload, key="rows_count", path=path),
            timeline=self._parse_timeline_coverage(
                path=path,
                key="timeline",
                payload=self._required_mapping_field(payload=payload, key="timeline", path=path),
            ),
            signal_value_set=self._parse_integer_tuple(
                path=path,
                key="signal_value_set",
                values=self._required_sequence_field(
                    payload=payload,
                    key="signal_value_set",
                    path=path,
                ),
            ),
            grid=self._parse_signal_grid(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="grid", path=path),
            ),
            provenance=self._parse_provenance(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="provenance", path=path),
            ),
        )

    def _parse_hit_times_manifest_document(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactHitTimesManifestDocumentV2:
        """
        Parse one strict `hit_times/1m/manifest.yaml` payload into typed DTOs.

        Args:
            path: Source manifest path.
            payload: Parsed YAML payload.
            slot: Explicit slot literal resolved from deterministic path.
        Returns:
            ArtifactHitTimesManifestDocumentV2: Strict typed hit-times manifest.
        Assumptions:
            Hit-times manifests are schema-validated eagerly during explicit-path reads.
        Raises:
            ValueError: If the payload shape or any nested strict field is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=HIT_TIMES_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=path,
        )
        manifest_slot = validate_artifact_slot_v2(
            str(self._required_yaml_field(payload=payload, key="slot", path=path))
        )
        if manifest_slot != slot:
            raise ValueError(f"{path} field 'slot' must be {slot!r}; got {manifest_slot!r}")
        tables_payload = self._required_mapping_field(payload=payload, key="tables", path=path)
        self._require_exact_yaml_keys(
            payload=tables_payload,
            required_keys=("long_tp", "long_sl", "short_tp", "short_sl"),
            path=path,
        )
        return ArtifactHitTimesManifestDocumentV2(
            path=path,
            raw_payload=payload,
            slot=manifest_slot,
            schema_version=self._required_yaml_field(
                payload=payload,
                key="schema_version",
                path=path,
            ),
            manifest_kind=str(
                self._required_yaml_field(payload=payload, key="manifest_kind", path=path)
            ),
            slot_generation=self._required_yaml_field(
                payload=payload,
                key="slot_generation",
                path=path,
            ),
            asof_date=str(self._required_yaml_field(payload=payload, key="asof_date", path=path)),
            timeframe=str(self._required_yaml_field(payload=payload, key="timeframe", path=path)),
            timeline_bar_count=self._required_yaml_field(
                payload=payload,
                key="timeline_bar_count",
                path=path,
            ),
            sentinel_index=self._required_yaml_field(
                payload=payload,
                key="sentinel_index",
                path=path,
            ),
            tp_values=self._parse_array_metadata(
                path=path,
                key="tp_values",
                payload=self._required_mapping_field(payload=payload, key="tp_values", path=path),
            ),
            sl_values=self._parse_array_metadata(
                path=path,
                key="sl_values",
                payload=self._required_mapping_field(payload=payload, key="sl_values", path=path),
            ),
            long_tp=self._parse_hit_times_table(
                path=path,
                key="tables.long_tp",
                payload=self._required_mapping_field(
                    payload=tables_payload,
                    key="long_tp",
                    path=path,
                ),
            ),
            long_sl=self._parse_hit_times_table(
                path=path,
                key="tables.long_sl",
                payload=self._required_mapping_field(
                    payload=tables_payload,
                    key="long_sl",
                    path=path,
                ),
            ),
            short_tp=self._parse_hit_times_table(
                path=path,
                key="tables.short_tp",
                payload=self._required_mapping_field(
                    payload=tables_payload,
                    key="short_tp",
                    path=path,
                ),
            ),
            short_sl=self._parse_hit_times_table(
                path=path,
                key="tables.short_sl",
                payload=self._required_mapping_field(
                    payload=tables_payload,
                    key="short_sl",
                    path=path,
                ),
            ),
            provenance=self._parse_provenance(
                path=path,
                payload=self._required_mapping_field(payload=payload, key="provenance", path=path),
            ),
        )

    def _parse_identity_mapping(
        self,
        *,
        path: Path,
        key: str,
        payload: Mapping[str, Any],
    ) -> ArtifactCoordinatesV2:
        """
        Parse one strict manifest identity mapping into artifact coordinates.

        Args:
            path: Source manifest path.
            key: Nested mapping field name.
            payload: Nested YAML mapping.
        Returns:
            ArtifactCoordinatesV2: Typed artifact coordinates.
        Assumptions:
            Identity mappings contain only `exchange`, `market_type`, and `symbol`.
        Raises:
            ValueError: If nested keys or coordinate literals are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("exchange", "market_type", "symbol"),
            path=path,
        )
        return ArtifactCoordinatesV2(
            exchange=str(self._required_yaml_field(payload=payload, key="exchange", path=path)),
            market_type=str(
                self._required_yaml_field(payload=payload, key="market_type", path=path)
            ),
            symbol=str(self._required_yaml_field(payload=payload, key="symbol", path=path)),
        )

    def _parse_price_manifests(
        self,
        *,
        path: Path,
        values: tuple[Any, ...],
    ) -> tuple[ArtifactPriceTimeframeManifestV2, ...]:
        """
        Parse and canonically order strict root-manifest price sections.

        Args:
            path: Source manifest path.
            values: Raw YAML sequence of price section payloads.
        Returns:
            tuple[ArtifactPriceTimeframeManifestV2, ...]: Canonically ordered price manifests.
        Assumptions:
            Each price timeframe appears at most once in the root manifest.
        Raises:
            ValueError: If one entry shape is invalid or a timeframe is duplicated.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        order = {literal: index for index, literal in enumerate(ARTIFACT_PRICE_TIMEFRAMES_V2)}
        seen: set[str] = set()
        parsed_entries: list[ArtifactPriceTimeframeManifestV2] = []
        for raw_value in values:
            entry_payload = self._coerce_mapping_value(
                value=raw_value,
                path=path,
                field_name="prices[]",
            )
            self._require_exact_yaml_keys(
                payload=entry_payload,
                required_keys=("timeframe", "open_time", "close_time", "ohlcv", "coverage"),
                path=path,
            )
            timeframe = validate_price_timeframe_v2(
                str(self._required_yaml_field(payload=entry_payload, key="timeframe", path=path))
            )
            if timeframe in seen:
                raise ValueError(f"{path} contains duplicate price timeframe {timeframe!r}")
            seen.add(timeframe)
            parsed_entries.append(
                ArtifactPriceTimeframeManifestV2(
                    timeframe=timeframe,
                    open_time=self._parse_array_metadata(
                        path=path,
                        key=f"prices[{timeframe}].open_time",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="open_time",
                            path=path,
                        ),
                    ),
                    close_time=self._parse_array_metadata(
                        path=path,
                        key=f"prices[{timeframe}].close_time",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="close_time",
                            path=path,
                        ),
                    ),
                    ohlcv=self._parse_array_metadata(
                        path=path,
                        key=f"prices[{timeframe}].ohlcv",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="ohlcv",
                            path=path,
                        ),
                    ),
                    coverage=self._parse_timeline_coverage(
                        path=path,
                        key=f"prices[{timeframe}].coverage",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="coverage",
                            path=path,
                        ),
                    ),
                )
            )
        return tuple(sorted(parsed_entries, key=lambda item: order[item.timeframe]))

    def _parse_mapping_manifests(
        self,
        *,
        path: Path,
        values: tuple[Any, ...],
    ) -> tuple[ArtifactMappingTimeframeManifestV2, ...]:
        """
        Parse and canonically order strict root-manifest mapping sections.

        Args:
            path: Source manifest path.
            values: Raw YAML sequence of mapping section payloads.
        Returns:
            tuple[ArtifactMappingTimeframeManifestV2, ...]: Canonically ordered mapping manifests.
        Assumptions:
            Each mapping timeframe appears at most once in the root manifest.
        Raises:
            ValueError: If one entry shape is invalid or a timeframe is duplicated.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        order = {literal: index for index, literal in enumerate(ARTIFACT_MAPPING_TIMEFRAMES_V2)}
        seen: set[str] = set()
        parsed_entries: list[ArtifactMappingTimeframeManifestV2] = []
        for raw_value in values:
            entry_payload = self._coerce_mapping_value(
                value=raw_value,
                path=path,
                field_name="mappings[]",
            )
            self._require_exact_yaml_keys(
                payload=entry_payload,
                required_keys=("timeframe", "bar_open_1m_idx", "bar_close_1m_idx"),
                path=path,
            )
            timeframe = validate_mapping_timeframe_v2(
                str(self._required_yaml_field(payload=entry_payload, key="timeframe", path=path))
            )
            if timeframe in seen:
                raise ValueError(f"{path} contains duplicate mapping timeframe {timeframe!r}")
            seen.add(timeframe)
            parsed_entries.append(
                ArtifactMappingTimeframeManifestV2(
                    timeframe=timeframe,
                    bar_open_1m_idx=self._parse_array_metadata(
                        path=path,
                        key=f"mappings[{timeframe}].bar_open_1m_idx",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="bar_open_1m_idx",
                            path=path,
                        ),
                    ),
                    bar_close_1m_idx=self._parse_array_metadata(
                        path=path,
                        key=f"mappings[{timeframe}].bar_close_1m_idx",
                        payload=self._required_mapping_field(
                            payload=entry_payload,
                            key="bar_close_1m_idx",
                            path=path,
                        ),
                    ),
                )
            )
        return tuple(sorted(parsed_entries, key=lambda item: order[item.timeframe]))

    def _parse_signal_catalog(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> ArtifactSignalCatalogV2:
        """
        Parse the strict root-manifest signal catalog section.

        Args:
            path: Source manifest path.
            payload: Nested YAML mapping.
        Returns:
            ArtifactSignalCatalogV2: Typed signal catalog.
        Assumptions:
            Signal catalog lists are explicit and contain no dynamic discovery fields.
        Raises:
            ValueError: If nested keys or manifest references are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("supported_timeframes", "supported_indicator_ids", "manifests"),
            path=path,
        )
        manifest_entries: list[ArtifactSignalCatalogEntryV2] = []
        for raw_value in self._required_sequence_field(payload=payload, key="manifests", path=path):
            entry_payload = self._coerce_mapping_value(
                value=raw_value,
                path=path,
                field_name="signals.manifests[]",
            )
            self._require_exact_yaml_keys(
                payload=entry_payload,
                required_keys=("timeframe", "indicator_id", "manifest_path", "manifest_sha256"),
                path=path,
            )
            manifest_entries.append(
                ArtifactSignalCatalogEntryV2(
                    timeframe=str(
                        self._required_yaml_field(payload=entry_payload, key="timeframe", path=path)
                    ),
                    indicator_id=str(
                        self._required_yaml_field(
                            payload=entry_payload,
                            key="indicator_id",
                            path=path,
                        )
                    ),
                    manifest_path=str(
                        self._required_yaml_field(
                            payload=entry_payload,
                            key="manifest_path",
                            path=path,
                        )
                    ),
                    manifest_sha256=str(
                        self._required_yaml_field(
                            payload=entry_payload,
                            key="manifest_sha256",
                            path=path,
                        )
                    ),
                )
            )
        return ArtifactSignalCatalogV2(
            supported_timeframes=self._parse_string_tuple(
                path=path,
                key="signals.supported_timeframes",
                values=self._required_sequence_field(
                    payload=payload,
                    key="supported_timeframes",
                    path=path,
                ),
            ),
            supported_indicator_ids=self._parse_string_tuple(
                path=path,
                key="signals.supported_indicator_ids",
                values=self._required_sequence_field(
                    payload=payload,
                    key="supported_indicator_ids",
                    path=path,
                ),
            ),
            manifests=tuple(manifest_entries),
        )

    def _parse_hit_times_reference(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> ArtifactHitTimesReferenceV2:
        """
        Parse the strict root-manifest hit-times reference section.

        Args:
            path: Source manifest path.
            payload: Nested YAML mapping.
        Returns:
            ArtifactHitTimesReferenceV2: Typed hit-times reference.
        Assumptions:
            Root manifests reference exactly one fixed `hit_times/1m/manifest.yaml`.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("timeframe", "manifest_path", "manifest_sha256"),
            path=path,
        )
        return ArtifactHitTimesReferenceV2(
            timeframe=str(self._required_yaml_field(payload=payload, key="timeframe", path=path)),
            manifest_path=str(
                self._required_yaml_field(payload=payload, key="manifest_path", path=path)
            ),
            manifest_sha256=str(
                self._required_yaml_field(payload=payload, key="manifest_sha256", path=path)
            ),
        )

    def _parse_signal_encoding(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> ArtifactSignalEncodingContractV2:
        """
        Parse the strict root-manifest signal-encoding section.

        Args:
            path: Source manifest path.
            payload: Nested YAML mapping.
        Returns:
            ArtifactSignalEncodingContractV2: Typed signal encoding contract.
        Assumptions:
            Runtime signal dtype, axis order, and value set are fixed metadata.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("dtype", "axis_order", "value_set"),
            path=path,
        )
        return ArtifactSignalEncodingContractV2(
            dtype=str(self._required_yaml_field(payload=payload, key="dtype", path=path)),
            axis_order=self._parse_string_tuple(
                path=path,
                key="signal_encoding.axis_order",
                values=self._required_sequence_field(payload=payload, key="axis_order", path=path),
            ),
            value_set=self._parse_integer_tuple(
                path=path,
                key="signal_encoding.value_set",
                values=self._required_sequence_field(payload=payload, key="value_set", path=path),
            ),
        )

    def _parse_provenance(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> ArtifactManifestProvenanceV2:
        """
        Parse one strict manifest provenance mapping.

        Args:
            path: Source manifest path.
            payload: Nested YAML mapping.
        Returns:
            ArtifactManifestProvenanceV2: Typed provenance contract.
        Assumptions:
            Provenance fields stay fixed across root/signal/hit-times manifests.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=(
                "generator",
                "generator_version",
                "generated_at_utc",
                "config_sha256",
                "inputs_sha256",
            ),
            path=path,
        )
        return ArtifactManifestProvenanceV2(
            generator=str(self._required_yaml_field(payload=payload, key="generator", path=path)),
            generator_version=str(
                self._required_yaml_field(payload=payload, key="generator_version", path=path)
            ),
            generated_at_utc=str(
                self._required_yaml_field(payload=payload, key="generated_at_utc", path=path)
            ),
            config_sha256=str(
                self._required_yaml_field(payload=payload, key="config_sha256", path=path)
            ),
            inputs_sha256=str(
                self._required_yaml_field(payload=payload, key="inputs_sha256", path=path)
            ),
        )

    def _parse_signal_grid(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> ArtifactSignalGridContractV2:
        """
        Parse one strict signal-grid metadata mapping.

        Args:
            path: Source manifest path.
            payload: Nested YAML mapping.
        Returns:
            ArtifactSignalGridContractV2: Typed signal-grid metadata.
        Assumptions:
            Signal-grid defaults remain explicit until a later config contract replaces them.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=(
                "variant_key_version",
                "variant_keys_sha256",
                "signals_v1_params_defaults",
            ),
            path=path,
        )
        return ArtifactSignalGridContractV2(
            variant_key_version=self._required_yaml_field(
                payload=payload,
                key="variant_key_version",
                path=path,
            ),
            variant_keys_sha256=str(
                self._required_yaml_field(payload=payload, key="variant_keys_sha256", path=path)
            ),
            signals_v1_params_defaults=freeze_artifact_payload_mapping_v2(
                self._required_mapping_field(
                    payload=payload,
                    key="signals_v1_params_defaults",
                    path=path,
                )
            ),
        )

    def _parse_array_metadata(
        self,
        *,
        path: Path,
        key: str,
        payload: Mapping[str, Any],
    ) -> ArtifactArrayMetadataV2:
        """
        Parse one strict array-metadata mapping shared across root/signal/hit-times manifests.

        Args:
            path: Source manifest path.
            key: Stable nested field label for diagnostics.
            payload: Nested YAML mapping.
        Returns:
            ArtifactArrayMetadataV2: Typed array metadata.
        Assumptions:
            Array metadata always declares `path`, `dtype`, `shape`, `axis_order`, and `sha256`.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("path", "dtype", "shape", "axis_order", "sha256"),
            path=path,
        )
        return ArtifactArrayMetadataV2(
            path=str(self._required_yaml_field(payload=payload, key="path", path=path)),
            dtype=str(self._required_yaml_field(payload=payload, key="dtype", path=path)),
            shape=self._parse_integer_tuple(
                path=path,
                key=f"{key}.shape",
                values=self._required_sequence_field(payload=payload, key="shape", path=path),
            ),
            axis_order=self._parse_string_tuple(
                path=path,
                key=f"{key}.axis_order",
                values=self._required_sequence_field(payload=payload, key="axis_order", path=path),
            ),
            sha256=str(self._required_yaml_field(payload=payload, key="sha256", path=path)),
        )

    def _parse_timeline_coverage(
        self,
        *,
        path: Path,
        key: str,
        payload: Mapping[str, Any],
    ) -> ArtifactTimelineCoverageV2:
        """
        Parse one strict timeline coverage mapping shared across price and signal manifests.

        Args:
            path: Source manifest path.
            key: Stable nested field label for diagnostics.
            payload: Nested YAML mapping.
        Returns:
            ArtifactTimelineCoverageV2: Typed timeline coverage metadata.
        Assumptions:
            Coverage metadata contains explicit counts and open/close boundary timestamps.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=(
                "bar_count",
                "open_time_start",
                "open_time_end",
                "close_time_start",
                "close_time_end",
            ),
            path=path,
        )
        return ArtifactTimelineCoverageV2(
            bar_count=self._required_yaml_field(payload=payload, key="bar_count", path=path),
            open_time_start=self._required_yaml_field(
                payload=payload,
                key="open_time_start",
                path=path,
            ),
            open_time_end=self._required_yaml_field(
                payload=payload,
                key="open_time_end",
                path=path,
            ),
            close_time_start=self._required_yaml_field(
                payload=payload,
                key="close_time_start",
                path=path,
            ),
            close_time_end=self._required_yaml_field(
                payload=payload,
                key="close_time_end",
                path=path,
            ),
        )

    def _parse_hit_times_table(
        self,
        *,
        path: Path,
        key: str,
        payload: Mapping[str, Any],
    ) -> ArtifactHitTimesTableManifestV2:
        """
        Parse one strict hit-times table metadata mapping.

        Args:
            path: Source manifest path.
            key: Stable nested field label for diagnostics.
            payload: Nested YAML mapping.
        Returns:
            ArtifactHitTimesTableManifestV2: Typed hit-times table metadata.
        Assumptions:
            Hit-times tables extend generic array metadata with one monotonicity literal.
        Raises:
            ValueError: If nested keys or values are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        self._require_exact_yaml_keys(
            payload=payload,
            required_keys=("path", "dtype", "shape", "axis_order", "sha256", "monotonicity"),
            path=path,
        )
        array_payload = {
            "path": self._required_yaml_field(payload=payload, key="path", path=path),
            "dtype": self._required_yaml_field(payload=payload, key="dtype", path=path),
            "shape": self._required_yaml_field(payload=payload, key="shape", path=path),
            "axis_order": self._required_yaml_field(
                payload=payload,
                key="axis_order",
                path=path,
            ),
            "sha256": self._required_yaml_field(payload=payload, key="sha256", path=path),
        }
        return ArtifactHitTimesTableManifestV2(
            array=self._parse_array_metadata(path=path, key=key, payload=array_payload),
            monotonicity=str(
                self._required_yaml_field(payload=payload, key="monotonicity", path=path)
            ),
        )

    def _required_mapping_field(
        self,
        *,
        payload: Mapping[str, Any],
        key: str,
        path: Path,
    ) -> Mapping[str, Any]:
        """
        Read one required nested YAML mapping field and validate its type.

        Args:
            payload: Parent YAML mapping.
            key: Required nested field name.
            path: Source manifest path used in deterministic errors.
        Returns:
            Mapping[str, Any]: Nested mapping payload.
        Assumptions:
            Strict manifest schemas never coerce scalars or sequences into mappings.
        Raises:
            ValueError: If the field is missing or not a mapping with string keys.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        return self._coerce_mapping_value(
            value=self._required_yaml_field(payload=payload, key=key, path=path),
            path=path,
            field_name=key,
        )

    def _required_sequence_field(
        self,
        *,
        payload: Mapping[str, Any],
        key: str,
        path: Path,
    ) -> tuple[Any, ...]:
        """
        Read one required YAML sequence field and validate its type.

        Args:
            payload: Parent YAML mapping.
            key: Required nested field name.
            path: Source manifest path used in deterministic errors.
        Returns:
            tuple[Any, ...]: Nested sequence payload converted to tuple.
        Assumptions:
            Strict manifest schemas distinguish sequences from scalar and mapping fields.
        Raises:
            ValueError: If the field is missing or not a YAML list.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        value = self._required_yaml_field(payload=payload, key=key, path=path)
        if not isinstance(value, list):
            raise ValueError(f"{path} field '{key}' must be a YAML list")
        return tuple(value)

    def _parse_string_tuple(
        self,
        *,
        path: Path,
        key: str,
        values: tuple[Any, ...],
    ) -> tuple[str, ...]:
        """
        Convert one YAML sequence into a tuple of strict string literals.

        Args:
            path: Source manifest path.
            key: Stable field label used in deterministic errors.
            values: Raw YAML sequence values.
        Returns:
            tuple[str, ...]: Tuple of string literals preserving input order.
        Assumptions:
            Loader does not normalize list ordering; canonical ordering is handled by contracts.
        Raises:
            ValueError: If one value is not a string.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        converted_values: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(f"{path} field '{key}' must contain only strings")
            converted_values.append(value)
        return tuple(converted_values)

    def _parse_integer_tuple(
        self,
        *,
        path: Path,
        key: str,
        values: tuple[Any, ...],
    ) -> tuple[int, ...]:
        """
        Convert one YAML sequence into a tuple of strict integer literals.

        Args:
            path: Source manifest path.
            key: Stable field label used in deterministic errors.
            values: Raw YAML sequence values.
        Returns:
            tuple[int, ...]: Tuple of integer literals preserving input order.
        Assumptions:
            Loader rejects implicit coercion of numeric strings or floats into integers.
        Raises:
            ValueError: If one value is not an integer literal.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        converted_values: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path} field '{key}' must contain only ints")
            converted_values.append(value)
        return tuple(converted_values)

    def _coerce_mapping_value(
        self,
        *,
        value: Any,
        path: Path,
        field_name: str,
    ) -> Mapping[str, Any]:
        """
        Validate that one nested YAML value is a mapping with string keys.

        Args:
            value: Raw YAML value.
            path: Source manifest path.
            field_name: Stable field label used in deterministic errors.
        Returns:
            Mapping[str, Any]: Nested mapping payload.
        Assumptions:
            Strict manifests reject scalars and sequences where mappings are expected.
        Raises:
            ValueError: If the value is not a mapping with string keys.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if not isinstance(value, dict):
            raise ValueError(f"{path} field '{field_name}' must be a YAML mapping")
        for nested_key in value.keys():
            if not isinstance(nested_key, str):
                raise ValueError(f"{path} field '{field_name}' must contain only string keys")
        return value

    def _require_exact_yaml_keys(
        self,
        *,
        payload: Mapping[str, Any],
        required_keys: tuple[str, ...],
        path: Path,
    ) -> None:
        """
        Enforce exact-key strictness for nested YAML mappings during manifest parsing.

        Args:
            payload: Nested YAML mapping payload.
            required_keys: Canonical required key tuple.
            path: Source manifest path used in deterministic errors.
        Returns:
            None.
        Assumptions:
            Strict manifest schemas reject both missing keys and extra unsupported keys.
        Raises:
            ValueError: If payload keys differ from the required key set.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload_keys = tuple(sorted(payload.keys()))
        expected_keys = tuple(sorted(required_keys))
        if payload_keys != expected_keys:
            missing_keys = tuple(key for key in required_keys if key not in payload)
            extra_keys = tuple(key for key in payload_keys if key not in required_keys)
            details: list[str] = []
            if len(missing_keys) > 0:
                details.append(f"missing keys {missing_keys}")
            if len(extra_keys) > 0:
                details.append(f"unexpected keys {extra_keys}")
            raise ValueError(
                f"{path} must contain exactly keys {required_keys}"
                + (f"; {'; '.join(details)}" if len(details) > 0 else "")
            )

    def _load_yaml_mapping(self, *, path: Path, document_label: str) -> Mapping[str, Any]:
        """
        Read one YAML document and ensure it is a mapping with string keys.

        Args:
            path: Explicit filesystem path to read.
            document_label: Stable document label used in error messages.
        Returns:
            Mapping[str, Any]: Parsed YAML mapping.
        Assumptions:
            YAML input is UTF-8 and should be parsed exactly once per explicit read call.
        Raises:
            FileNotFoundError: If the explicit path does not exist.
            ValueError: If the YAML document is not a mapping or contains non-string keys.
        Side Effects:
            Reads one UTF-8 YAML file from disk.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping for {document_label}")
        for key in payload.keys():
            if not isinstance(key, str):
                raise ValueError(f"{path} must contain only string YAML keys")
        return payload

    def _required_yaml_field(self, *, payload: Mapping[str, Any], key: str, path: Path) -> Any:
        """
        Read one required YAML field and fail fast when it is absent.

        Args:
            payload: Parsed YAML mapping payload.
            key: Required field name.
            path: Source path used in deterministic error messages.
        Returns:
            Any: Raw field value without implicit coercion.
        Assumptions:
            Detailed scalar validation is delegated to the strict typed contracts layer.
        Raises:
            ValueError: If the required field is absent from the YAML mapping.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if key not in payload:
            raise ValueError(f"{path} field '{key}' is required")
        return payload[key]
