"""Bounded canonical discovery. Notifications are hints; PostgreSQL owns obligations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, Sequence
from uuid import UUID

from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    History,
    SourceTarget,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    artifact_coordinates_from_market_id_v2,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
)
from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp


@dataclass(frozen=True)
class StoredSourceObservation:
    """One indexed coordinate, with actual minute bounds at a fixed exclusive cutoff.

    Empty bounds are legitimate for future-only/empty index states. Keep such rows
    in the page so keyset traversal always advances. Absence markers are excluded.
    """

    instrument: InstrumentId
    first: UtcTimestamp | None
    last: UtcTimestamp | None
    generation: UUID | None = None
    content_revision: int | None = None
    funding_revision: int | None = None
    actual_count: int | None = None
    index_state: str = "legacy"


class StoredCoordinates(Protocol):
    def page(self, *, after: tuple[int, str] | None, limit: int) -> Sequence[InstrumentId]: ...

    def observation_page(
        self,
        *,
        after: tuple[int, str] | None,
        limit: int,
        before: UtcTimestamp,
    ) -> Sequence[StoredSourceObservation]: ...

    def observation(
        self,
        *,
        instrument: InstrumentId,
        before: UtcTimestamp,
    ) -> StoredSourceObservation: ...


class BaseObligations(Protocol):
    def observe_unavailable(
        self,
        *,
        installation_id: UUID,
        observation: StoredSourceObservation,
        coordinate: Coordinate,
        observed_at: datetime,
    ) -> None: ...

    def observe_source(
        self,
        *,
        installation_id: UUID,
        coordinate: Coordinate,
        source: SourceTarget,
        observed_at: datetime,
    ) -> UUID | None: ...


@dataclass(frozen=True)
class BaseSourceReconciler:
    installation_id: UUID
    # Retained constructor dependency for existing composition callers; sweep/hints
    # use the consumer-owned batched source port, never this canonical reader.
    index: CanonicalCandleIndexReader
    stored: StoredCoordinates
    obligations: BaseObligations

    def observe(self, instrument: InstrumentId, *, now: datetime) -> UUID | None:
        before = self._closed_cutoff(now)
        row = self.stored.observation(instrument=instrument, before=before)
        return self._observe(row, now=now)

    @staticmethod
    def _closed_cutoff(now: datetime) -> UtcTimestamp:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("utc_required")
        return UtcTimestamp(now.astimezone(UTC).replace(second=0, microsecond=0))

    def _observe(self, row: StoredSourceObservation, *, now: datetime) -> UUID | None:
        instrument, first, last = row.instrument, row.first, row.last
        if row.content_revision is None or row.funding_revision is None:
            return None
        coordinates = artifact_coordinates_from_market_id_v2(
            market_id=instrument.market_id.value, symbol=str(instrument.symbol)
        )
        if row.index_state != "ready" or row.generation is None or first is None or last is None:
            if row.generation is not None:
                self.obligations.observe_unavailable(
                    installation_id=self.installation_id,
                    observation=row,
                    coordinate=Coordinate(
                        exchange=coordinates.exchange,
                        market_type=coordinates.market_type,
                        symbol=coordinates.symbol,
                    ),
                    observed_at=now.astimezone(UTC),
                )
            return None
        end = last.value + timedelta(minutes=1)
        return self.obligations.observe_source(
            installation_id=self.installation_id,
            coordinate=Coordinate(
                exchange=coordinates.exchange,
                market_type=coordinates.market_type,
                symbol=coordinates.symbol,
            ),
            source=SourceTarget(
                history=History(kind="range", start_utc=first.value, end_utc=end),
                watermark=end,
                revision=f"v2:{row.generation}:{row.content_revision}:{row.funding_revision}",
                source_generation=row.generation,
                content_revision=row.content_revision,
                funding_revision=row.funding_revision,
                compute_identity="native-base-v2",
                funding_identity=coordinates.market_type,
            ),
            observed_at=now.astimezone(UTC),
        )

    def reconcile_page(
        self, *, after: tuple[int, str] | None, now: datetime, limit: int = 50
    ) -> tuple[int, str] | None:
        if not 1 <= limit <= 50:
            raise ValueError("invalid_reconciliation_limit")
        before = self._closed_cutoff(now)
        rows = self.stored.observation_page(after=after, limit=limit, before=before)
        if len(rows) > limit:
            raise ValueError("unbounded_source_page")
        for row in rows:
            self._observe(row, now=now)
        if len(rows) < limit:
            return None
        return rows[-1].instrument.market_id.value, str(rows[-1].instrument.symbol)
