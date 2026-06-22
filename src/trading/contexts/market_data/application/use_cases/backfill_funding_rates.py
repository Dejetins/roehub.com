from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from trading.contexts.market_data.application.dto import FundingInstrument
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.funding_rate_history_source import (
    FundingRateHistorySource,
)
from trading.contexts.market_data.application.ports.stores.funding_instrument_universe_store import (  # noqa: E501
    FundingInstrumentUniverseStore,
)
from trading.contexts.market_data.application.ports.stores.funding_rate_writer import (
    FundingRateWriter,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class FundingCatchupInstrumentReport:
    instrument_id: InstrumentId
    exchange: str
    market_type: str
    status: str
    start: UtcTimestamp | None
    end: UtcTimestamp | None
    rows_read: int
    rows_written: int
    lag_seconds: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class BackfillFundingRatesReport:
    instruments_total: int
    instruments_due: int
    instruments_ok: int
    instruments_skipped: int
    instruments_failed: int
    rows_read: int
    rows_written: int
    dry_run: bool
    instrument_reports: tuple[FundingCatchupInstrumentReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instruments_total": self.instruments_total,
            "instruments_due": self.instruments_due,
            "instruments_ok": self.instruments_ok,
            "instruments_skipped": self.instruments_skipped,
            "instruments_failed": self.instruments_failed,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "dry_run": self.dry_run,
            "instrument_reports": [
                {
                    "instrument_id": str(r.instrument_id),
                    "exchange": r.exchange,
                    "market_type": r.market_type,
                    "status": r.status,
                    "start": _ts_to_iso(r.start),
                    "end": _ts_to_iso(r.end),
                    "rows_read": r.rows_read,
                    "rows_written": r.rows_written,
                    "lag_seconds": r.lag_seconds,
                    "reason": r.reason,
                }
                for r in self.instrument_reports
            ],
        }


class BackfillFundingRatesUseCase:
    def __init__(
        self,
        *,
        source: FundingRateHistorySource,
        writer: FundingRateWriter,
        universe_store: FundingInstrumentUniverseStore,
        clock: Clock,
        tail_lookback_intervals: int,
        settlement_lag_minutes: int,
    ) -> None:
        if source is None:  # type: ignore[truthy-bool]
            raise ValueError("BackfillFundingRatesUseCase requires source")
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("BackfillFundingRatesUseCase requires writer")
        if universe_store is None:  # type: ignore[truthy-bool]
            raise ValueError("BackfillFundingRatesUseCase requires universe_store")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("BackfillFundingRatesUseCase requires clock")
        if tail_lookback_intervals <= 0:
            raise ValueError("tail_lookback_intervals must be > 0")
        if settlement_lag_minutes < 0:
            raise ValueError("settlement_lag_minutes must be >= 0")
        self._source = source
        self._writer = writer
        self._universe_store = universe_store
        self._clock = clock
        self._tail_lookback_intervals = tail_lookback_intervals
        self._settlement_lag = timedelta(minutes=settlement_lag_minutes)

    def run_single(
        self,
        *,
        instrument: FundingInstrument,
        time_range: TimeRange | None = None,
        dry_run: bool = False,
    ) -> BackfillFundingRatesReport:
        report = self._run_one(instrument=instrument, time_range=time_range, dry_run=dry_run)
        return _build_report((report,), dry_run=dry_run)

    def run_due_universe(
        self,
        *,
        market_ids: Sequence[MarketId],
        dry_run: bool = False,
    ) -> BackfillFundingRatesReport:
        instruments = self._universe_store.list_tradable_funding_instruments(
            market_ids=market_ids,
        )
        reports = tuple(
            self._run_one(instrument=instrument, time_range=None, dry_run=dry_run)
            for instrument in instruments
        )
        return _build_report(reports, dry_run=dry_run)

    def _run_one(
        self,
        *,
        instrument: FundingInstrument,
        time_range: TimeRange | None,
        dry_run: bool,
    ) -> FundingCatchupInstrumentReport:
        if instrument.funding_interval_minutes is None:
            return self._skip(
                instrument=instrument,
                status="skipped_missing_interval",
                reason="funding interval metadata is missing",
            )
        if instrument.funding_interval_source is None:
            return self._skip(
                instrument=instrument,
                status="skipped_missing_interval",
                reason="funding interval source is missing",
            )

        effective_range = time_range or self._due_time_range(instrument)
        if effective_range is None:
            return self._skip(
                instrument=instrument,
                status="not_due",
                reason="next funding time is not settled yet",
            )

        if dry_run:
            return FundingCatchupInstrumentReport(
                instrument_id=instrument.instrument_id,
                exchange=instrument.exchange,
                market_type=instrument.market_type,
                status="dry_run",
                start=effective_range.start,
                end=effective_range.end,
                rows_read=0,
                rows_written=0,
                lag_seconds=self._lag_seconds(instrument),
                reason="dry run; provider history was not requested",
            )

        try:
            rows = list(
                self._source.list_funding_rates(
                    instrument_id=instrument.instrument_id,
                    time_range=effective_range,
                    funding_interval_minutes=instrument.funding_interval_minutes,
                    funding_interval_source=instrument.funding_interval_source,
                )
            )
            self._writer.write_funding_rates(rows)
        except Exception as exc:  # noqa: BLE001
            return FundingCatchupInstrumentReport(
                instrument_id=instrument.instrument_id,
                exchange=instrument.exchange,
                market_type=instrument.market_type,
                status="failed",
                start=effective_range.start,
                end=effective_range.end,
                rows_read=0,
                rows_written=0,
                lag_seconds=self._lag_seconds(instrument),
                reason=str(exc),
            )

        return FundingCatchupInstrumentReport(
            instrument_id=instrument.instrument_id,
            exchange=instrument.exchange,
            market_type=instrument.market_type,
            status="ok",
            start=effective_range.start,
            end=effective_range.end,
            rows_read=len(rows),
            rows_written=len(rows),
            lag_seconds=self._lag_seconds(instrument),
            reason="funding history fetched",
        )

    def _due_time_range(self, instrument: FundingInstrument) -> TimeRange | None:
        interval = timedelta(minutes=int(instrument.funding_interval_minutes or 0))
        now = self._clock.now()
        settled_end_dt = now.value - self._settlement_lag
        if interval.total_seconds() <= 0 or settled_end_dt <= now.value - timedelta(days=3650):
            return None

        last = self._writer.latest_funding_time(instrument.instrument_id)
        if last is None:
            start_dt = settled_end_dt - interval * self._tail_lookback_intervals
        else:
            next_dt = last.value + interval
            if now.value < next_dt + self._settlement_lag:
                return None
            start_dt = next_dt

        if start_dt >= settled_end_dt:
            return None
        return TimeRange(start=UtcTimestamp(start_dt), end=UtcTimestamp(settled_end_dt))

    def _lag_seconds(self, instrument: FundingInstrument) -> int | None:
        latest = self._writer.latest_funding_time(instrument.instrument_id)
        if latest is None:
            return None
        return max(int((self._clock.now().value - latest.value).total_seconds()), 0)

    def _skip(
        self,
        *,
        instrument: FundingInstrument,
        status: str,
        reason: str,
    ) -> FundingCatchupInstrumentReport:
        return FundingCatchupInstrumentReport(
            instrument_id=instrument.instrument_id,
            exchange=instrument.exchange,
            market_type=instrument.market_type,
            status=status,
            start=None,
            end=None,
            rows_read=0,
            rows_written=0,
            lag_seconds=self._lag_seconds(instrument),
            reason=reason,
        )


def _build_report(
    instrument_reports: Sequence[FundingCatchupInstrumentReport],
    *,
    dry_run: bool,
) -> BackfillFundingRatesReport:
    due_statuses = {"ok", "failed", "dry_run"}
    ok_statuses = {"ok", "dry_run"}
    skipped_statuses = {"not_due", "skipped_missing_interval"}
    return BackfillFundingRatesReport(
        instruments_total=len(instrument_reports),
        instruments_due=sum(1 for r in instrument_reports if r.status in due_statuses),
        instruments_ok=sum(1 for r in instrument_reports if r.status in ok_statuses),
        instruments_skipped=sum(1 for r in instrument_reports if r.status in skipped_statuses),
        instruments_failed=sum(1 for r in instrument_reports if r.status == "failed"),
        rows_read=sum(r.rows_read for r in instrument_reports),
        rows_written=sum(r.rows_written for r in instrument_reports),
        dry_run=dry_run,
        instrument_reports=tuple(instrument_reports),
    )


def _ts_to_iso(ts: UtcTimestamp | None) -> str | None:
    return str(ts) if ts is not None else None
