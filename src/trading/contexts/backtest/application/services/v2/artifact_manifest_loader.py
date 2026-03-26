"""Explicit-path YAML loader for deterministic backtest artifact store v2 (R2-01)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    ARTIFACT_MANIFEST_FILENAME_V2,
    CURRENT_ARTIFACT_POINTER_FILENAME_V2,
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactManifestDocumentV2,
    ArtifactMappingPathsV2,
    ArtifactPricePathsV2,
    ArtifactSignalPathsV2,
    ArtifactSlotLiteralV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactPathResolverV2,
    validate_artifact_slot_v2,
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
        slot: ArtifactSlotLiteralV2 | None = None,
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
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        payload = self._load_yaml_mapping(path=path, document_label=ARTIFACT_MANIFEST_FILENAME_V2)
        validated_slot = validate_artifact_slot_v2(slot) if slot is not None else None
        return ArtifactManifestDocumentV2(path=path, raw_payload=payload, slot=validated_slot)

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
