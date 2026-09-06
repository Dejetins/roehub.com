"""Deterministic R2-02 publish orchestration for inactive artifact slots."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

from trading.contexts.backtest.application.ports import BacktestJobRepository

from .artifact_manifest_validator import BacktestArtifactManifestValidatorV2
from .artifact_precompute_runner import BacktestArtifactPrecomputeRunnerV2
from .contracts import (
    ARTIFACT_PUBLISH_FAILURE_CODE_INACTIVE_SLOT_PINNED_V2,
    ARTIFACT_SLOT_A_LITERAL_V2,
    CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactPricesMappingsPublishResultV2,
    ArtifactPublishPrecheckV2,
    ArtifactPublishResultV2,
    ArtifactSlotValidationResultV2,
    ArtifactSlotValidationSpecV2,
    ArtifactValidationDiagnosticV2,
    BacktestArtifactCurrentPointerWriterV2,
    BacktestArtifactLoaderV2,
    artifact_market_id_from_coordinates_v2,
    inactive_artifact_slot_v2,
    validate_current_pointer_asof_date_v2,
    validate_current_pointer_published_at_utc_v2,
)

NowProviderV2 = Callable[[], datetime]


def _default_now_provider_v2() -> datetime:
    """
    Return the default UTC wall-clock value used by artifact publish orchestration.

    Args:
        None.
    Returns:
        datetime: Timezone-aware UTC datetime.
    Assumptions:
        Default publisher timestamps are generated at second precision in UTC.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    return datetime.now(timezone.utc)


class ArtifactSlotPublishErrorV2(Exception):
    """
    Stable publish error with explicit code for R2-02 slot publish failures.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        diagnostics: tuple[ArtifactValidationDiagnosticV2, ...] = (),
    ) -> None:
        """
        Store deterministic publish failure code and message.

        Args:
            code: Stable machine-readable error code.
            message: Stable human-readable failure message.
            diagnostics: Optional structured validation diagnostics attached to the error.
        Returns:
            None.
        Assumptions:
            Callers may branch on `.code` while displaying `str(error)` to operators.
        Raises:
            None.
        Side Effects:
            Initializes exception state.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class BacktestArtifactSlotPublisherV2:
    """
    Publish orchestrator implementing `precheck -> validate -> atomic switch` for R2-02.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    artifact_loader: BacktestArtifactLoaderV2
    current_pointer_writer: BacktestArtifactCurrentPointerWriterV2
    job_repository: BacktestJobRepository
    now_provider: NowProviderV2 = _default_now_provider_v2

    def precheck_publish(self, coordinates: ArtifactCoordinatesV2) -> ArtifactPublishPrecheckV2:
        """
        Resolve current/inactive slots and fail fast when inactive slot is pinned.

        Args:
            coordinates: Symbol-root coordinates whose pointer is being prepared for publish.
        Returns:
            ArtifactPublishPrecheckV2: Deterministic readiness diagnostics for the inactive slot.
        Assumptions:
            Operators call this step before rebuilding the inactive slot in place.
        Raises:
            ValueError: If `current.yaml` violates the strict pointer contract or bootstrap
                preconditions are inconsistent.
        Side Effects:
            Reads `current.yaml` and, when present, the inactive slot `manifest.yaml`.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        current_pointer_path = self.artifact_loader.resolve_current_pointer_path(coordinates)
        try:
            current_pointer = self.artifact_loader.load_current_pointer(coordinates)
        except FileNotFoundError:
            bootstrap_manifest_path = self.artifact_loader.resolve_slot_manifest_path(
                coordinates,
                ARTIFACT_SLOT_A_LITERAL_V2,
            )
            alternate_manifest_path = self.artifact_loader.resolve_slot_manifest_path(
                coordinates,
                inactive_artifact_slot_v2(ARTIFACT_SLOT_A_LITERAL_V2),
            )
            conflicting_manifest_paths = tuple(
                path
                for path in (bootstrap_manifest_path, alternate_manifest_path)
                if path.is_file()
            )
            if len(conflicting_manifest_paths) > 0:
                raise ValueError(
                    "bootstrap requires missing current.yaml and no pre-existing slot manifests; "
                    f"found {conflicting_manifest_paths!r}"
                )
            return ArtifactPublishPrecheckV2(
                coordinates=coordinates,
                current_pointer_path=current_pointer_path,
                current_pointer=None,
                inactive_slot=ARTIFACT_SLOT_A_LITERAL_V2,
                target_slot_generation=1,
                inactive_manifest_path=bootstrap_manifest_path,
                inactive_manifest_hash=None,
                blocking_active_run_count=0,
                ready=True,
                bootstrap=True,
            )
        inactive_slot = inactive_artifact_slot_v2(current_pointer.active_slot)
        inactive_manifest_path = self.artifact_loader.resolve_slot_manifest_path(
            coordinates,
            inactive_slot,
        )
        inactive_manifest_hash: str | None = None
        blocking_active_run_count = 0
        if inactive_manifest_path.is_file():
            inactive_manifest_hash = _file_sha256_hex_v2(inactive_manifest_path)
            market_id = artifact_market_id_from_coordinates_v2(coordinates)
            blocking_active_run_count = self.job_repository.count_active_for_artifact_manifest(
                market_id=market_id,
                symbol=coordinates.symbol,
                artifact_slot=inactive_slot,
                artifact_manifest_hash=inactive_manifest_hash,
            )

        if blocking_active_run_count > 0:
            failure_message = (
                f"inactive artifact slot {inactive_slot} is pinned by "
                f"{blocking_active_run_count} active background job(s)"
            )
            return ArtifactPublishPrecheckV2(
                coordinates=coordinates,
                current_pointer_path=current_pointer.path,
                current_pointer=current_pointer,
                inactive_slot=inactive_slot,
                target_slot_generation=current_pointer.slot_generation + 1,
                inactive_manifest_path=inactive_manifest_path,
                inactive_manifest_hash=inactive_manifest_hash,
                blocking_active_run_count=blocking_active_run_count,
                ready=False,
                failure_code=ARTIFACT_PUBLISH_FAILURE_CODE_INACTIVE_SLOT_PINNED_V2,
                failure_message=failure_message,
            )

        return ArtifactPublishPrecheckV2(
            coordinates=coordinates,
            current_pointer_path=current_pointer.path,
            current_pointer=current_pointer,
            inactive_slot=inactive_slot,
            target_slot_generation=current_pointer.slot_generation + 1,
            inactive_manifest_path=inactive_manifest_path,
            inactive_manifest_hash=inactive_manifest_hash,
            blocking_active_run_count=0,
            ready=True,
        )

    def build_publish_prices_mappings_slot(
        self,
        *,
        request: ArtifactCanonicalPriceExportRequestV2,
        precompute_runner: BacktestArtifactPrecomputeRunnerV2,
        validation_spec: ArtifactSlotValidationSpecV2,
    ) -> ArtifactPricesMappingsPublishResultV2:
        """
        Execute `precheck -> build inactive slot -> validate whole slot -> atomically switch
        current.yaml` for the R3-04 `prices + mappings` stage.

        Args:
            request: Explicit `prices/1m` export request whose inactive slot becomes publish
                candidate.
            precompute_runner: Deterministic inactive-slot builder for `prices/<tf>` and
                `mappings/<tf>`.
            validation_spec: Explicit R3-04 prices+mappings validation scope derived from
                source-of-truth artifact config.
        Returns:
            ArtifactPricesMappingsPublishResultV2: Combined precheck/build/publish result for the
                published prices+mappings slot.
        Assumptions:
            This orchestration entrypoint is stage-specific and therefore requires
            `signal_artifacts=()` plus `require_hit_times_manifest=false`.
        Raises:
            ArtifactSlotPublishErrorV2: If precheck blocks the inactive slot or strict validation
                fails before the `current.yaml` switch.
            ValueError: If dependencies are missing or the supplied validation spec is not an
                explicit prices+mappings stage spec.
            FileNotFoundError: If `current.yaml` or the built root manifest is missing.
            OSError: If one artifact write or atomic pointer switch fails.
        Side Effects:
            Reads `current.yaml`, rebuilds the inactive slot, validates the root manifest, and
            replaces `current.yaml` atomically on success.
        Docs:
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
          - docs/runbooks/backtest-artifacts-rebuild.md
        """
        if precompute_runner is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestArtifactSlotPublisherV2.build_publish_prices_mappings_slot requires "
                "precompute_runner"
            )
        stage_validation_spec = _ensure_prices_mappings_publish_validation_spec_v2(validation_spec)
        precheck = self.precheck_publish(request.coordinates)
        self._ensure_precheck_ready(precheck)
        build_result = precompute_runner.export_canonical_price_1m(
            ArtifactCanonicalPriceExportRequestV2(
                coordinates=request.coordinates,
                time_range=request.time_range,
                asof_date=request.asof_date,
                generated_at_utc=request.generated_at_utc,
                target_slot=precheck.inactive_slot,
                target_slot_generation=precheck.target_slot_generation,
                reuse_source_slot=(
                    None
                    if precheck.current_pointer is None or request.force_full_rebuild
                    else precheck.current_pointer.active_slot
                ),
                force_full_rebuild=request.force_full_rebuild,
            )
        )
        publish_result = self.publish(
            precheck=precheck,
            validation_spec=stage_validation_spec,
            asof_date=request.asof_date,
        )
        return ArtifactPricesMappingsPublishResultV2(
            validation_spec=stage_validation_spec,
            precheck=precheck,
            build_result=build_result,
            publish_result=publish_result,
        )

    def validate_inactive_slot(
        self,
        *,
        precheck: ArtifactPublishPrecheckV2,
        validation_spec: ArtifactSlotValidationSpecV2,
        expected_asof_date: str | None = None,
    ) -> ArtifactSlotValidationResultV2:
        """
        Validate an already-built inactive slot through explicit deterministic paths only.

        Args:
            precheck: Publish readiness snapshot resolved before build/switch.
            validation_spec: Explicit path-validation plan for the built inactive slot,
                typically translated from `backtest_artifacts.validation_plan`.
            expected_asof_date: Optional strict `YYYY-MM-DD` literal expected from manifests.
        Returns:
            ArtifactSlotValidationResultV2: Validated slot manifest identity and plan snapshot.
        Assumptions:
            No directory scanning is allowed; callers must provide explicit validation targets.
        Raises:
            ArtifactSlotPublishErrorV2: If publish is blocked or one required explicit path is
                missing.
            FileNotFoundError: If slot `manifest.yaml` is absent.
            ValueError: If manifest or path coordinates violate deterministic contracts.
        Side Effects:
            Reads manifest files and checks required artifact files on disk.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        self._ensure_precheck_ready(precheck)
        validator = BacktestArtifactManifestValidatorV2(artifact_loader=self.artifact_loader)
        validation = validator.validate_slot(
            coordinates=precheck.coordinates,
            slot=precheck.inactive_slot,
            validation_spec=validation_spec,
            expected_asof_date=expected_asof_date,
            expected_slot_generation=precheck.target_slot_generation,
        )
        if len(validation.diagnostics) > 0:
            first_diagnostic = validation.diagnostics[0]
            raise ArtifactSlotPublishErrorV2(
                code="slot_validation_failed",
                message=first_diagnostic.message,
                diagnostics=validation.diagnostics,
            )
        return validation

    def publish(
        self,
        *,
        precheck: ArtifactPublishPrecheckV2,
        validation_spec: ArtifactSlotValidationSpecV2,
        asof_date: str,
    ) -> ArtifactPublishResultV2:
        """
        Validate the inactive slot and atomically switch `current.yaml` to the new identity.

        Args:
            precheck: Publish readiness snapshot resolved before build/switch.
            validation_spec: Explicit validation plan for the rebuilt inactive slot,
                typically translated from `backtest_artifacts.validation_plan`.
            asof_date: Strict `YYYY-MM-DD` literal for the newly published slot identity.
        Returns:
            ArtifactPublishResultV2: Structured previous/new pointer identity payload.
        Assumptions:
            Inactive slot contents were rebuilt after `precheck_publish` and before this call.
        Raises:
            ArtifactSlotPublishErrorV2: If precheck is blocked or validation fails.
            FileNotFoundError: If a required manifest file is missing.
            ValueError: If `asof_date` or the strict pointer payload is invalid.
            OSError: If atomic pointer replacement fails.
        Side Effects:
            Reads inactive slot files and atomically replaces `current.yaml`.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
        """
        validated_asof_date = validate_current_pointer_asof_date_v2(asof_date)
        validation = self.validate_inactive_slot(
            precheck=precheck,
            validation_spec=validation_spec,
            expected_asof_date=validated_asof_date,
        )
        if validation.manifest_sha256 is None:
            raise ArtifactSlotPublishErrorV2(
                code="slot_validation_failed",
                message="slot validation did not produce manifest_sha256",
                diagnostics=validation.diagnostics,
            )
        published_at_utc = _utc_now_literal_v2(self.now_provider())
        next_slot_generation = precheck.target_slot_generation
        raw_pointer_payload = {
            "schema_version": CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
            "active_slot": precheck.inactive_slot,
            "slot_generation": next_slot_generation,
            "asof_date": validated_asof_date,
            "manifest_sha256": validation.manifest_sha256,
            "published_at_utc": published_at_utc,
        }
        published_pointer = ArtifactCurrentPointerV2(
            path=precheck.current_pointer_path,
            active_slot=precheck.inactive_slot,
            raw_payload=raw_pointer_payload,
            schema_version=CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
            slot_generation=next_slot_generation,
            asof_date=validated_asof_date,
            manifest_sha256=validation.manifest_sha256,
            published_at_utc=published_at_utc,
        )
        self.current_pointer_writer.write_current_pointer_atomically(
            precheck.coordinates,
            published_pointer,
        )
        self._cleanup_previous_slot_after_publish(precheck=precheck)
        return ArtifactPublishResultV2(
            coordinates=precheck.coordinates,
            previous_pointer=precheck.current_pointer,
            published_pointer=published_pointer,
            precheck=precheck,
            validation=validation,
        )

    def _cleanup_previous_slot_after_publish(self, *, precheck: ArtifactPublishPrecheckV2) -> None:
        """
        Remove previous slot tree after publish when identity checks and pin guard allow cleanup.

        Args:
            precheck: Publish readiness snapshot holding the pre-switch pointer identity.
        Returns:
            None.
        Assumptions:
            Cleanup is best-effort and must never fail a publish after `current.yaml` switched.
        Raises:
            None.
        Side Effects:
            May delete one previous slot directory tree from disk.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        previous_pointer = precheck.current_pointer
        if previous_pointer is None:
            return
        previous_manifest_path = self.artifact_loader.resolve_slot_manifest_path(
            precheck.coordinates,
            previous_pointer.active_slot,
        )
        if not previous_manifest_path.is_file():
            return
        previous_manifest_hash = _file_sha256_hex_v2(previous_manifest_path)
        if previous_manifest_hash != previous_pointer.manifest_sha256:
            return
        blocking_active_run_count = self.job_repository.count_active_for_artifact_manifest(
            market_id=artifact_market_id_from_coordinates_v2(precheck.coordinates),
            symbol=precheck.coordinates.symbol,
            artifact_slot=previous_pointer.active_slot,
            artifact_manifest_hash=previous_manifest_hash,
        )
        if blocking_active_run_count > 0:
            return
        symbol_root = self.artifact_loader.resolve_current_pointer_path(precheck.coordinates).parent
        previous_slot_root = previous_manifest_path.parent
        if previous_slot_root.name != previous_pointer.active_slot:
            return
        if previous_slot_root.parent != symbol_root:
            return
        if symbol_root.is_symlink() or previous_slot_root.is_symlink():
            return
        if not previous_slot_root.is_dir():
            return
        try:
            shutil.rmtree(previous_slot_root)
        except OSError:
            return

    def _ensure_precheck_ready(self, precheck: ArtifactPublishPrecheckV2) -> None:
        """
        Raise a stable publish error when `precheck_publish` reported a blocking condition.

        Args:
            precheck: Publish readiness snapshot to enforce.
        Returns:
            None.
        Assumptions:
            Blocking diagnostics were already populated deterministically in precheck step.
        Raises:
            ArtifactSlotPublishErrorV2: If inactive slot is not publishable.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        if precheck.ready:
            return
        raise ArtifactSlotPublishErrorV2(
            code=precheck.failure_code or "publish_precheck_failed",
            message=precheck.failure_message or "artifact publish precheck failed",
        )


def _ensure_prices_mappings_publish_validation_spec_v2(
    validation_spec: ArtifactSlotValidationSpecV2,
) -> ArtifactSlotValidationSpecV2:
    """
    Enforce the explicit R3-04 validation boundary for the `prices + mappings` publish stage.

    Args:
        validation_spec: Candidate whole-slot validation spec supplied by the caller.
    Returns:
        ArtifactSlotValidationSpecV2: The original validated stage spec.
    Assumptions:
        R3-04 may validate full `prices/<tf>` and `mappings/<tf>` coverage, but must keep
        `signal_artifacts=()` and `require_hit_times_manifest=false` explicit instead of
        inferring stage scope from file presence.
    Raises:
        ValueError: If the spec still requires signal artifacts or a real hit-times manifest.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - docs/runbooks/backtest-artifacts-rebuild.md
    """
    if validation_spec.signal_artifacts != ():
        raise ValueError(
            "prices+mappings publish validation spec must set signal_artifacts=() explicitly"
        )
    if validation_spec.require_hit_times_manifest:
        raise ValueError(
            "prices+mappings publish validation spec must set "
            "require_hit_times_manifest=False explicitly"
        )
    return validation_spec


def _file_sha256_hex_v2(path: Path) -> str:
    """
    Compute deterministic SHA-256 hex digest for one filesystem artifact file.

    Args:
        path: Existing file path to hash.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        Slot manifest files are small enough to read eagerly during publish orchestration.
    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    Side Effects:
        Reads file bytes from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    return sha256(path.read_bytes()).hexdigest()


def _utc_now_literal_v2(value: datetime) -> str:
    """
    Serialize one timezone-aware UTC datetime into strict R2-02 pointer timestamp literal.

    Args:
        value: Datetime candidate returned by publisher clock dependency.
    Returns:
        str: Strict UTC timestamp literal `YYYY-MM-DDTHH:MM:SSZ`.
    Assumptions:
        Publisher clocks are expected to provide timezone-aware UTC datetimes.
    Raises:
        ValueError: If the datetime is naive or not UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError("publisher now_provider must return timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError("publisher now_provider must return UTC datetime")
    return validate_current_pointer_published_at_utc_v2(
        value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
