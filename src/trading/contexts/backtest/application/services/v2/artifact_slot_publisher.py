"""Deterministic R2-02 publish orchestration for inactive artifact slots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

from trading.contexts.backtest.application.ports import BacktestJobRepository

from .contracts import (
    CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactPublishPrecheckV2,
    ArtifactPublishResultV2,
    ArtifactSlotValidationResultV2,
    ArtifactSlotValidationSpecV2,
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
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    return datetime.now(timezone.utc)


class ArtifactSlotPublishErrorV2(Exception):
    """
    Stable publish error with explicit code for R2-02 slot publish failures.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Store deterministic publish failure code and message.

        Args:
            code: Stable machine-readable error code.
            message: Stable human-readable failure message.
        Returns:
            None.
        Assumptions:
            Callers may branch on `.code` while displaying `str(error)` to operators.
        Raises:
            None.
        Side Effects:
            Initializes exception state.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class BacktestArtifactSlotPublisherV2:
    """
    Publish orchestrator implementing `precheck -> validate -> atomic switch` for R2-02.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
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
            FileNotFoundError: If the active `current.yaml` pointer is missing.
            ValueError: If `current.yaml` violates the strict pointer contract.
        Side Effects:
            Reads `current.yaml` and, when present, the inactive slot `manifest.yaml`.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        current_pointer = self.artifact_loader.load_current_pointer(coordinates)
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
                current_pointer=current_pointer,
                inactive_slot=inactive_slot,
                inactive_manifest_path=inactive_manifest_path,
                inactive_manifest_hash=inactive_manifest_hash,
                blocking_active_run_count=blocking_active_run_count,
                ready=False,
                failure_code="inactive_slot_pinned",
                failure_message=failure_message,
            )

        return ArtifactPublishPrecheckV2(
            coordinates=coordinates,
            current_pointer=current_pointer,
            inactive_slot=inactive_slot,
            inactive_manifest_path=inactive_manifest_path,
            inactive_manifest_hash=inactive_manifest_hash,
            blocking_active_run_count=0,
            ready=True,
        )

    def validate_inactive_slot(
        self,
        *,
        precheck: ArtifactPublishPrecheckV2,
        validation_spec: ArtifactSlotValidationSpecV2,
    ) -> ArtifactSlotValidationResultV2:
        """
        Validate an already-built inactive slot through explicit deterministic paths only.

        Args:
            precheck: Publish readiness snapshot resolved before build/switch.
            validation_spec: Explicit path-validation plan for the built inactive slot.
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
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        self._ensure_precheck_ready(precheck)

        slot_manifest = self.artifact_loader.load_slot_manifest(
            precheck.coordinates,
            precheck.inactive_slot,
        )
        for timeframe in validation_spec.price_timeframes:
            price_paths = self.artifact_loader.resolve_price_paths(
                precheck.coordinates,
                precheck.inactive_slot,
                timeframe,
            )
            _require_existing_path_v2(price_paths.open_time, "price open_time")
            _require_existing_path_v2(price_paths.close_time, "price close_time")
            _require_existing_path_v2(price_paths.ohlcv, "price ohlcv")

        for timeframe in validation_spec.mapping_timeframes:
            mapping_paths = self.artifact_loader.resolve_mapping_paths(
                precheck.coordinates,
                precheck.inactive_slot,
                timeframe,
            )
            _require_existing_path_v2(mapping_paths.bar_open_1m_idx, "mapping bar_open_1m_idx")
            _require_existing_path_v2(
                mapping_paths.bar_close_1m_idx,
                "mapping bar_close_1m_idx",
            )

        for signal_artifact in validation_spec.signal_artifacts:
            signal_paths = self.artifact_loader.resolve_signal_paths(
                precheck.coordinates,
                precheck.inactive_slot,
                signal_artifact.timeframe,
                signal_artifact.indicator_id,
            )
            _require_existing_path_v2(signal_paths.manifest, "signal manifest")
            _require_existing_path_v2(signal_paths.signals, "signal payload")

        if validation_spec.require_hit_times_manifest:
            hit_times_manifest_path = self.artifact_loader.resolve_hit_times_manifest_path(
                precheck.coordinates,
                precheck.inactive_slot,
            )
            _require_existing_path_v2(hit_times_manifest_path, "hit_times manifest")

        return ArtifactSlotValidationResultV2(
            slot=precheck.inactive_slot,
            slot_manifest=slot_manifest,
            manifest_sha256=_file_sha256_hex_v2(slot_manifest.path),
            validation_spec=validation_spec,
        )

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
            validation_spec: Explicit validation plan for the rebuilt inactive slot.
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
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
        """
        validated_asof_date = validate_current_pointer_asof_date_v2(asof_date)
        validation = self.validate_inactive_slot(
            precheck=precheck,
            validation_spec=validation_spec,
        )
        published_at_utc = _utc_now_literal_v2(self.now_provider())
        next_slot_generation = precheck.current_pointer.slot_generation + 1
        raw_pointer_payload = {
            "schema_version": CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
            "active_slot": precheck.inactive_slot,
            "slot_generation": next_slot_generation,
            "asof_date": validated_asof_date,
            "manifest_sha256": validation.manifest_sha256,
            "published_at_utc": published_at_utc,
        }
        published_pointer = ArtifactCurrentPointerV2(
            path=precheck.current_pointer.path,
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
        return ArtifactPublishResultV2(
            coordinates=precheck.coordinates,
            previous_pointer=precheck.current_pointer,
            published_pointer=published_pointer,
            precheck=precheck,
            validation=validation,
        )

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
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        if precheck.ready:
            return
        raise ArtifactSlotPublishErrorV2(
            code=precheck.failure_code or "publish_precheck_failed",
            message=precheck.failure_message or "artifact publish precheck failed",
        )


def _require_existing_path_v2(path: Path, label: str) -> None:
    """
    Fail fast when one explicit artifact path required by publish validation is missing.

    Args:
        path: Explicit artifact path that must already exist on disk.
        label: Stable human-readable label used in failure messages.
    Returns:
        None.
    Assumptions:
        Validation works only with explicit deterministic paths and never scans directories.
    Raises:
        ArtifactSlotPublishErrorV2: If the required path is missing or is not a file.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not path.is_file():
        raise ArtifactSlotPublishErrorV2(
            code="artifact_slot_validation_failed",
            message=f"missing explicit artifact path for {label}: {path}",
        )


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
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
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
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
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
