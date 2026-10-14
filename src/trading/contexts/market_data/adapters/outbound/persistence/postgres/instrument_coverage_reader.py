from dataclasses import dataclass

from .history_coverage_reader import PostgresHistoryCoverageReader
from .ingestion_gateway import IngestionGateway
from .query_support import CoverageUnavailable, key, minute


@dataclass(frozen=True)
class InstrumentCoverageSnapshot:
    state: str
    percent: float | None


@dataclass(frozen=True)
class PostgresInstrumentCoverageReader:
    gateway: IngestionGateway

    def read(self, *, instrument_id, expected_start_at, expected_end_at):
        expected = minute(expected_end_at) - minute(expected_start_at)
        if expected <= 0:
            return InstrumentCoverageSnapshot("unknown", None)
        try:
            counts = PostgresHistoryCoverageReader(self.gateway).counts(
                starts={key(instrument_id): expected_start_at}, end_at=expected_end_at
            )
        except CoverageUnavailable:
            return InstrumentCoverageSnapshot("unknown", None)
        actual = counts[key(instrument_id)]
        return InstrumentCoverageSnapshot(
            "complete" if actual == expected else "partial",
            min(100.0, round(actual / expected * 100, 2)),
        )
