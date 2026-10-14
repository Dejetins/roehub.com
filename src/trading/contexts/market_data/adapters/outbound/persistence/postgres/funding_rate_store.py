"""Funding ports; intervals remain per-event metadata, never a fixed eight hours."""

from trading.contexts.market_data.application.ports.stores.funding_rate_coverage_reader import (
    FundingRateArtifactRecord,
    FundingRateCoverageSnapshot,
)
from trading.shared_kernel.primitives import UtcTimestamp

from .funding_writer import PostgresFundingRateWriter
from .query_support import key, rows
from .reference_writers import PostgresFundingInstrumentStore

FUNDING_RANGE = """SELECT instrument_key,funding_time,funding_rate,funding_interval_minutes
FROM market_data.canonical_funding_rates WHERE market_id=%s AND symbol=%s
AND funding_time>=%s AND funding_time<%s ORDER BY funding_time"""
FUNDING_LATEST = """SELECT funding_time FROM market_data.canonical_funding_rates
WHERE market_id=%s AND symbol=%s ORDER BY funding_time DESC LIMIT 1"""


class PostgresFundingRateStore(PostgresFundingInstrumentStore):
    def write_funding_rates(self, rows):
        PostgresFundingRateWriter(self._gateway).write_funding_rates(rows)

    def latest_funding_time(self, instrument_id):
        result = self._gateway.run(
            lambda c: c.execute(FUNDING_LATEST, key(instrument_id)).fetchone()
        )
        return UtcTimestamp(result[0]) if result else None

    def read_funding_coverage(self, *, instrument_id, time_range):
        records = self._gateway.run(
            lambda c: tuple(
                FundingRateArtifactRecord(
                    instrument_id=instrument_id,
                    instrument_key=r["instrument_key"],
                    funding_time=UtcTimestamp(r["funding_time"]),
                    funding_rate=r["funding_rate"],
                    funding_interval_minutes=r["funding_interval_minutes"],
                    mark_price=None,
                    data_quality=1,
                )
                for r in rows(
                    c,
                    FUNDING_RANGE,
                    (*key(instrument_id), time_range.start.value, time_range.end.value),
                )
            )
        )
        if not records:
            return FundingRateCoverageSnapshot(
                status="unavailable",
                coverage_policy="unavailable",
                requested_range=time_range,
                available_start=None,
                available_end=None,
                expected_event_count=0,
                observed_event_count=0,
                missing_event_count=0,
                reason_codes=("no_funding_rows",),
                records=(),
            )
        missing = 0
        reasons = []
        if (records[0].funding_time.value - time_range.start.value).total_seconds() > records[
            0
        ].funding_interval_minutes * 60:
            missing += 1
            reasons.append("missing_leading_coverage")
        for previous, current in zip(records, records[1:]):
            # The current event declares the elapsed settlement interval.
            count = max(
                0,
                round(
                    (current.funding_time.value - previous.funding_time.value).total_seconds()
                    / (current.funding_interval_minutes * 60)
                )
                - 1,
            )
            missing += count
            if count and "funding_interval_gap" not in reasons:
                reasons.append("funding_interval_gap")
        if (time_range.end.value - records[-1].funding_time.value).total_seconds() > records[
            -1
        ].funding_interval_minutes * 60:
            missing += 1
            reasons.append("missing_trailing_coverage")
        return FundingRateCoverageSnapshot(
            status="degraded" if missing else "ready",
            coverage_policy="degraded_with_warning" if missing else "ready",
            requested_range=time_range,
            available_start=records[0].funding_time,
            available_end=records[-1].funding_time,
            expected_event_count=len(records) + missing,
            observed_event_count=len(records),
            missing_event_count=missing,
            reason_codes=tuple(reasons),
            records=records,
        )
