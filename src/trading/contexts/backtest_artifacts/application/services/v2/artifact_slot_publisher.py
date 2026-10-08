"""Deterministic R2-02 publish orchestration for inactive artifact slots."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.ports.backtest_job_repositories import (
    ArtifactOwnershipConflict,
    ArtifactReaderReservation,
    ArtifactWriterReservation,
)

from .artifact_manifest_validator import BacktestArtifactManifestValidatorV2
from .artifact_precompute_runner import BacktestArtifactPrecomputeRunnerV2
from .contracts import (
    ARTIFACT_PUBLISH_FAILURE_CODE_INACTIVE_SLOT_PINNED_V2,
    ARTIFACT_SLOT_A_LITERAL_V2,
    CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
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
    validate_artifact_slot_v2,
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

    def recover_publications(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        validation_spec: ArtifactSlotValidationSpecV2,
    ) -> int:
        """Reconcile dead local owners; live flock holders and uncertain state remain protected."""
        symbol_root = self.artifact_loader.resolve_current_pointer_path(coordinates).parent
        recovered = 0
        if not symbol_root.exists():
            return recovered
        if symbol_root.is_symlink():
            raise ValueError("untrusted publication recovery root")
        for record in sorted(symbol_root.glob(".publication-*")):
            if record.is_symlink() or not record.is_dir():
                continue
            lifetime, marker = record / "lifetime.lock", record / "ownership.json"
            if lifetime.is_symlink() or marker.is_symlink() or not marker.is_file():
                continue
            fd = os.open(lifetime, os.O_RDWR | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    payload = json.loads(marker.read_text())
                except (ValueError, OSError):
                    # An unrecognized journal cannot prove ownership. Preserve it.
                    continue
                if (
                    not isinstance(payload, dict)
                    or payload.get("schema") != "artifact-publication-owner/v1"
                ):
                    continue
                value = payload["writer"]
                writer = ArtifactWriterReservation(
                    value["exchange"],
                    value["market_type"],
                    value["symbol"],
                    value["slot"],
                    UUID(value["owner_token"]),
                    int(value["attempt"]),
                    UUID(value["parent_incarnation"]),
                    int(value["epoch"]),
                )
                if (writer.exchange, writer.market_type, writer.symbol) != (
                    coordinates.exchange,
                    coordinates.market_type,
                    coordinates.symbol,
                ):
                    raise ValueError("publication recovery owner coordinates differ")
                source_reader = None
                value = payload["source_reader"]
                if value is not None:
                    source_reader = ArtifactReaderReservation(
                        value["exchange"],
                        value["market_type"],
                        value["symbol"],
                        value["slot"],
                        int(value["generation"]),
                        value["manifest_sha256"],
                        value["owner_kind"],
                        None,
                        UUID(value["owner_id"]),
                        UUID(value["owner_token"]),
                        int(value["attempt"]),
                        UUID(value["parent_incarnation"]),
                        int(value["epoch"]),
                    )
                    if (
                        source_reader.owner_token,
                        source_reader.owner_kind,
                        source_reader.parent_incarnation,
                    ) != (
                        writer.owner_token,
                        "publisher_source",
                        writer.parent_incarnation,
                    ):
                        raise ValueError("publication source owner mismatch")
                with self.job_repository.transaction():
                    if self.job_repository.owns_artifact_writer(writer=writer):
                        self.job_repository.quarantine_artifact_writer(writer=writer)
                        path = self.artifact_loader.resolve_slot_manifest_path(
                            coordinates, writer.slot
                        )
                        target = path.parent
                        try:
                            current = self.artifact_loader.load_current_pointer(coordinates)
                        except FileNotFoundError:
                            current = None
                        if target.is_symlink():
                            raise ValueError("untrusted recovered target")
                        if payload["operation"] == "cleanup":
                            if current is None or current.active_slot == writer.slot:
                                raise ArtifactOwnershipConflict("cannot_recover_active_cleanup")
                            if target.exists():
                                shutil.rmtree(target)
                                _fsync_directory(symbol_root)
                        elif payload["operation"] == "build":
                            previous = record / "previous-target"
                            if (
                                current is None
                                and payload.get("bootstrap") is True
                                and target.exists()
                            ):
                                # No pointer ever committed: return bootstrap to its
                                # empty state while exact dead writer ownership is held.
                                shutil.rmtree(target)
                                _fsync_directory(symbol_root)
                            if not target.exists() and previous.exists():
                                if current is not None and current.active_slot == writer.slot:
                                    raise ArtifactOwnershipConflict(
                                        "active_target_missing_during_recovery"
                                    )
                                os.rename(previous, target)
                                _fsync_directory(record)
                                _fsync_directory(symbol_root)
                        else:
                            raise ValueError("unknown publication recovery operation")
                        generation, digest = None, None
                        if path.exists():
                            manifest = self.artifact_loader.load_slot_manifest(
                                coordinates, writer.slot
                            )
                            generation, digest = manifest.slot_generation, _file_sha256_hex_v2(path)
                            validation = BacktestArtifactManifestValidatorV2(
                                artifact_loader=self.artifact_loader
                            ).validate_slot(
                                coordinates=coordinates,
                                slot=validate_artifact_slot_v2(writer.slot),
                                validation_spec=validation_spec,
                                expected_slot_generation=generation,
                            )
                            if validation.diagnostics:
                                raise ArtifactOwnershipConflict("recovered_manifest_invalid")
                        if (
                            current is not None
                            and current.active_slot == writer.slot
                            and (current.slot_generation, current.manifest_sha256)
                            != (generation, digest)
                        ):
                            raise ArtifactOwnershipConflict("recovered_pointer_manifest_mismatch")

                        def confirm_dead_owner() -> None:
                            if os.fstat(fd).st_ino != lifetime.stat(follow_symlinks=False).st_ino:
                                raise ValueError("recovery lifetime lock replaced")

                        if target.exists():
                            _fsync_candidate(target)
                        pointer_path = self.artifact_loader.resolve_current_pointer_path(
                            coordinates
                        )
                        if pointer_path.exists():
                            with pointer_path.open("rb") as pointer_file:
                                os.fsync(pointer_file.fileno())
                        _fsync_directory(record)
                        _fsync_directory(symbol_root)
                        self.job_repository.recover_artifact_writer(
                            writer=writer,
                            generation=generation,
                            manifest_sha256=digest,
                            reconcile_dead_owner=confirm_dead_owner,
                        )
                    if source_reader is not None:
                        self.job_repository.release_artifact_reader(reader=source_reader)
                shutil.rmtree(record)
                _fsync_directory(symbol_root)
                recovered += 1
            finally:
                os.close(fd)
        return recovered

    def build_and_publish(
        self,
        *,
        request: ArtifactCanonicalPriceExportRequestV2,
        precheck: ArtifactPublishPrecheckV2,
        precompute_runner: BacktestArtifactPrecomputeRunnerV2,
        validation_spec: ArtifactSlotValidationSpecV2,
    ) -> tuple[ArtifactCanonicalPriceExportResultV2, ArtifactPublishResultV2]:
        """Build privately under durable coordinate ownership and pin incremental source reads."""
        self._ensure_precheck_ready(precheck)
        coordinates = precheck.coordinates
        token, incarnation = uuid4(), uuid4()
        symbol_root = precheck.current_pointer_path.parent
        record = symbol_root / f".publication-{token}"
        source_reader = None
        writer = None
        lock_fd = -1
        target = precheck.inactive_manifest_path.parent
        candidate = record / "candidate"
        original_generation = None
        failed = False
        try:
            with self.job_repository.transaction():
                # Reservation itself locks BOTH rows. Expected physical metadata is
                # revalidated while the outer transaction retains those row locks.
                if precheck.inactive_manifest_hash is not None:
                    original_generation = self.artifact_loader.load_slot_manifest(
                        coordinates, precheck.inactive_slot
                    ).slot_generation
                writer = self.job_repository.reserve_artifact_writer(
                    exchange=coordinates.exchange,
                    market_type=coordinates.market_type,
                    symbol=coordinates.symbol,
                    slot=precheck.inactive_slot,
                    expected_generation=original_generation,
                    expected_manifest_sha256=precheck.inactive_manifest_hash,
                    owner_token=token,
                    attempt=1,
                    parent_incarnation=incarnation,
                )
                self._revalidate_precheck(precheck=precheck, check_target=True)
                if request.coordinates != coordinates or (
                    request.target_slot != precheck.inactive_slot
                    or request.target_slot_generation != precheck.target_slot_generation
                ):
                    raise ArtifactOwnershipConflict("builder_target_mismatch")
                incremental_source = (
                    None
                    if request.force_full_rebuild or precheck.current_pointer is None
                    else precheck.current_pointer.active_slot
                )
                if request.reuse_source_slot not in (None, incremental_source):
                    raise ArtifactOwnershipConflict("stale_incremental_source")
                request = replace(request, reuse_source_slot=incremental_source)
                if request.reuse_source_slot is not None:
                    source = precheck.current_pointer
                    if source is None or request.reuse_source_slot != source.active_slot:
                        raise ArtifactOwnershipConflict("unprotected_incremental_source")
                    source_reader = self.job_repository.reserve_artifact_reader(
                        reader=ArtifactReaderReservation(
                            coordinates.exchange,
                            coordinates.market_type,
                            coordinates.symbol,
                            source.active_slot,
                            source.slot_generation,
                            source.manifest_sha256,
                            "publisher_source",
                            None,
                            token,
                            token,
                            1,
                            incarnation,
                        )
                    )
                symbol_root.mkdir(parents=True, exist_ok=True)
                if symbol_root.is_symlink() or target.is_symlink():
                    raise ValueError("untrusted publication root")
                record.mkdir()
                lock_fd = os.open(
                    record / "lifetime.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _write_publication_journal(
                    record,
                    {
                        "schema": "artifact-publication-owner/v1",
                        "operation": "build",
                        "bootstrap": precheck.current_pointer is None,
                        "writer": {key: str(value) for key, value in writer.parameters().items()},
                        "source_reader": (
                            None
                            if source_reader is None
                            else {
                                key: None if value is None else str(value)
                                for key, value in source_reader.parameters().items()
                            }
                        ),
                    },
                )
            if precheck.bootstrap:
                # Preserve the existing two-root bootstrap layout, now only after
                # durable coordinate reservation instead of before admission.
                for slot in ("slot_a", "slot_b"):
                    self.artifact_loader.resolve_slot_manifest_path(coordinates, slot).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                _fsync_directory(symbol_root)
            # No detached writer child may outlive this lifetime lock. Native
            # computation/chunking remains shared; this publication uses one process.
            runner = replace(
                precompute_runner,
                runtime_settings=replace(
                    precompute_runner.runtime_settings,
                    execution_policy=replace(
                        precompute_runner.runtime_settings.execution_policy,
                        signal_worker_processes=1,
                    ),
                ),
            )
            build = runner.export_canonical_price_1m(request, output_directory=candidate)
            private_loader = self.artifact_loader.with_private_slot(
                coordinates=coordinates,
                slot=precheck.inactive_slot,
                root=candidate,
            )
            private_publisher = replace(self, artifact_loader=private_loader)
            private_publisher.validate_inactive_slot(
                precheck=precheck,
                validation_spec=validation_spec,
                expected_asof_date=request.asof_date,
            )
            _fsync_candidate(candidate)
            with self.job_repository.transaction():
                self.job_repository.verify_artifact_writer(writer=writer)
                self._revalidate_precheck(precheck=precheck, check_target=True)
                if target.exists():
                    os.rename(target, record / "previous-target")
                os.rename(candidate, target)
                _fsync_directory(record)
                _fsync_directory(symbol_root)
                published = self.publish(
                    precheck=precheck,
                    validation_spec=validation_spec,
                    asof_date=request.asof_date,
                    writer=writer,
                )
                self.job_repository.complete_artifact_writer(
                    writer=writer,
                    generation=published.published_pointer.slot_generation,
                    manifest_sha256=published.published_pointer.manifest_sha256,
                )
                if source_reader is not None:
                    self.job_repository.release_artifact_reader(reader=source_reader)
            shutil.rmtree(record)
            _fsync_directory(symbol_root)
            self._cleanup_previous_slot_after_publish(
                precheck=precheck, published_pointer=published.published_pointer
            )
            self.recover_publications(coordinates=coordinates, validation_spec=validation_spec)
            return replace(
                build,
                manifest_path=precheck.inactive_manifest_path,
                price_paths=self.artifact_loader.resolve_price_paths(
                    coordinates, precheck.inactive_slot, "1m"
                ),
            ), published
        except BaseException:
            failed = True
            if writer is not None:
                # Keep both source pin and record if filesystem/DB reconciliation
                # cannot prove a safe release. Expiration is never reclamation.
                self.job_repository.quarantine_artifact_writer(writer=writer)
            raise
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            if failed and record.exists():
                self.recover_publications(coordinates=coordinates, validation_spec=validation_spec)

    def _revalidate_precheck(
        self,
        *,
        precheck: ArtifactPublishPrecheckV2,
        check_target: bool,
    ) -> None:
        try:
            current = self.artifact_loader.load_current_pointer(precheck.coordinates)
        except FileNotFoundError:
            current = None
        if current != precheck.current_pointer:
            raise ArtifactOwnershipConflict("artifact_stale_current_pointer")
        if current is not None and current.active_slot == precheck.inactive_slot:
            raise ArtifactOwnershipConflict("artifact_target_is_active")
        if check_target:
            path = precheck.inactive_manifest_path
            digest = _file_sha256_hex_v2(path) if path.is_file() else None
            if digest != precheck.inactive_manifest_hash:
                raise ArtifactOwnershipConflict("artifact_stale_target_manifest")

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
        build_request = replace(
            request,
            target_slot=precheck.inactive_slot,
            target_slot_generation=precheck.target_slot_generation,
            reuse_source_slot=(
                None
                if precheck.current_pointer is None or request.force_full_rebuild
                else precheck.current_pointer.active_slot
            ),
        )
        build_result, publish_result = self.build_and_publish(
            request=build_request,
            precheck=precheck,
            precompute_runner=precompute_runner,
            validation_spec=stage_validation_spec,
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
        writer: ArtifactWriterReservation,
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
        if (writer.exchange, writer.market_type, writer.symbol, writer.slot) != (
            precheck.coordinates.exchange,
            precheck.coordinates.market_type,
            precheck.coordinates.symbol,
            precheck.inactive_slot,
        ):
            raise ArtifactOwnershipConflict("writer_target_mismatch")
        self.job_repository.verify_artifact_writer(writer=writer)
        self._revalidate_precheck(precheck=precheck, check_target=False)
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
        _fsync_directory(precheck.current_pointer_path.parent)
        return ArtifactPublishResultV2(
            coordinates=precheck.coordinates,
            previous_pointer=precheck.current_pointer,
            published_pointer=published_pointer,
            precheck=precheck,
            validation=validation,
        )

    def _cleanup_previous_slot_after_publish(
        self,
        *,
        precheck: ArtifactPublishPrecheckV2,
        published_pointer: ArtifactCurrentPointerV2,
    ) -> None:
        previous = precheck.current_pointer
        if previous is None:
            return
        path = self.artifact_loader.resolve_slot_manifest_path(
            precheck.coordinates, previous.active_slot
        )
        coordinates = precheck.coordinates
        writer = None
        lock_fd = -1
        record = published_pointer.path.parent / f".publication-{uuid4()}"
        try:
            with self.job_repository.transaction():
                writer = self.job_repository.reserve_artifact_writer(
                    exchange=coordinates.exchange,
                    market_type=coordinates.market_type,
                    symbol=coordinates.symbol,
                    slot=previous.active_slot,
                    expected_generation=previous.slot_generation,
                    expected_manifest_sha256=previous.manifest_sha256,
                    owner_token=uuid4(),
                    attempt=1,
                    parent_incarnation=uuid4(),
                )
                current = self.artifact_loader.load_current_pointer(coordinates)
                if current != published_pointer or current.active_slot == previous.active_slot:
                    raise ArtifactOwnershipConflict("artifact_stale_cleanup_pointer")
                if path.is_file() and _file_sha256_hex_v2(path) != previous.manifest_sha256:
                    raise ArtifactOwnershipConflict("artifact_stale_cleanup_manifest")
                root = path.parent
                if root.parent != published_pointer.path.parent or root.is_symlink():
                    raise ValueError("untrusted previous slot root")
                record.mkdir()
                lock_fd = os.open(
                    record / "lifetime.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _write_publication_journal(
                    record,
                    {
                        "schema": "artifact-publication-owner/v1",
                        "operation": "cleanup",
                        "writer": {key: str(value) for key, value in writer.parameters().items()},
                        "source_reader": None,
                    },
                )
            # Reservation is durable BEFORE destructive IO. Losing this transaction
            # cannot expose a live cleanup process to a successor writer.
            with self.job_repository.transaction():
                self.job_repository.verify_artifact_writer(writer=writer)
                current = self.artifact_loader.load_current_pointer(coordinates)
                if current != published_pointer or current.active_slot == previous.active_slot:
                    raise ArtifactOwnershipConflict("artifact_stale_cleanup_pointer")
                if root.exists():
                    shutil.rmtree(root)
                    _fsync_directory(root.parent)
                self.job_repository.complete_artifact_writer(
                    writer=writer,
                    generation=None,
                    manifest_sha256=None,
                )
            shutil.rmtree(record)
            _fsync_directory(record.parent)
        except ArtifactOwnershipConflict:
            if writer is not None:
                self.job_repository.quarantine_artifact_writer(writer=writer)
            return
        except BaseException:
            if writer is not None:
                self.job_repository.quarantine_artifact_writer(writer=writer)
            raise
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)

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


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_publication_journal(record: Path, payload: dict[str, object]) -> None:
    temporary = record / "ownership.pending"
    with temporary.open("x") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, record / "ownership.json")
    _fsync_directory(record)
    _fsync_directory(record.parent)


def _fsync_candidate(root: Path) -> None:
    # Files and directory entries must outlive DB completion and backup deletion.
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("candidate contains a symlink")
        if path.is_dir():
            directories.append(path)
        else:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in reversed(directories):
        _fsync_directory(path)
