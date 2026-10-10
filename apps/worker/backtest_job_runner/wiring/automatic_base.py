"""Inactive composition seam for the managed base lane (S05 owns dispatch/activation)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.automatic_base_repository import (  # noqa: E501
    PostgresAutomaticBaseRepository,
)
from trading.contexts.backtest_artifacts.application.services.automatic_base import (
    AutomaticBasePublisher,
    BasePrepublicationError,
    BasePublication,
    freeze_source,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    ManagementError,
    Progress,
    SourceTarget,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (
    ArtifactPublicationNotStarted,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_reader import (
    CanonicalCandleReader,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


@dataclass(frozen=True)
class AutomaticBaseWorker:
    repository: PostgresAutomaticBaseRepository
    reader: CanonicalCandleReader
    publisher: AutomaticBasePublisher
    # Trusted composition verifies root/version, current+pinned+staging bytes and free
    # space (including 1 GiB safety). An unknown storage estimate MUST reject admission.
    check_storage: Callable[[Mapping[str, Any], int], None]

    def run(self, *, installation_id: UUID, request: Mapping[str, Any]) -> BasePublication | None:
        request = self.repository.base_request(
            installation_id=installation_id, request_id=request["request_id"]
        )
        if request["kind"] != "base" or request["installation_id"] != installation_id:
            raise ManagementError("invalid_base_request")
        target = SourceTarget.model_validate(request["snapshot"]["source"])
        start, end = target.history.start_utc, target.history.end_utc
        assert start is not None and end is not None
        minutes = int((end - start).total_seconds() // 60)
        if minutes > 5_000_000:
            raise ManagementError("resource_conflict")
        reservation = max(4096, minutes * 128 + 1_048_576)
        self.check_storage(request["snapshot"], reservation)
        build = self.repository.claim(
            installation_id=installation_id,
            request_id=request["request_id"],
            owner_token=uuid4(),
            parent_incarnation=uuid4(),
            reservation_bytes=reservation,
        )
        if build is None:
            return None
        # Use only the database-returned immutable snapshot after the claim.
        target = SourceTarget.model_validate(build["snapshot"]["source"])
        start, end = target.history.start_utc, target.history.end_utc
        assert start is not None and end is not None
        coordinates = ArtifactCoordinatesV2(
            exchange=build["exchange"], market_type=build["market_type"], symbol=build["symbol"]
        )
        entered_publisher = False
        try:

            def checkpoint(done: int, total: int) -> None:
                self.repository.observe(
                    build=build,
                    progress=Progress(
                        phase="source_snapshot",
                        completed_units=done,
                        total_units=total,
                        unit="source_minutes",
                        observed_at=datetime.now(UTC),
                    ),
                )

            source = freeze_source(
                reader=self.reader,
                coordinates=coordinates,
                time_range=TimeRange(UtcTimestamp(start), UtcTimestamp(end)),
                checkpoint=checkpoint,
            )
            self.check_storage(build["snapshot"], reservation)
            entered_publisher = True
            result = self.publisher.publish(
                coordinates=coordinates, source=source, generated_at=datetime.now(UTC)
            )
            if result.publication is None:
                self.repository.finish(build=build, outcome="error", reason="source_unavailable")
                return result
            pointer = result.publication.published_pointer
            self.repository.finish_base(
                build=build,
                publication={
                    "generation": pointer.slot_generation,
                    "slot": pointer.active_slot,
                    "manifest_sha256": pointer.manifest_sha256,
                },
                coverage={
                    "state": result.state,
                    "source_sha256": result.source_sha256,
                    "timeframes": dict(result.timeframes),
                    "source": target.model_dump(mode="json"),
                },
            )
            return result
        except (BasePrepublicationError, ArtifactPublicationNotStarted) as error:
            self.repository.finish(
                build=build,
                outcome="error",
                reason=(
                    str(error) if isinstance(error, BasePrepublicationError) else "publisher_busy"
                ),
            )
            raise
        except Exception:
            # After publisher entry a pointer may already have committed. Never classify
            # an unknown cross-store result as safe to retry; explicit reconciliation owns it.
            self.repository.finish(
                build=build,
                outcome="quarantined" if entered_publisher else "error",
                reason="publication_reconciliation_required"
                if entered_publisher
                else "source_or_storage_unavailable",
            )
            raise
