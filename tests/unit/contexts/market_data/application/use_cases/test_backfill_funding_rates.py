from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.market_data.application.dto import FundingInstrument, FundingRateRecord
from trading.contexts.market_data.application.use_cases.backfill_funding_rates import (
    BackfillFundingRatesUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


class _Clock:
    def now(self):
        return UtcTimestamp(datetime(2026, 6, 22, 12, 10, tzinfo=timezone.utc))


class _Source:
    def __init__(self):
        self.calls = []

    def list_funding_rates(self, *, instrument_id, time_range, funding_interval_minutes, funding_interval_source):  # noqa: E501
        self.calls.append((instrument_id, time_range))
        return [
            FundingRateRecord(
                instrument_id=instrument_id,
                instrument_key=f"binance:futures:{instrument_id.symbol}",
                funding_time=time_range.start,
                funding_rate=0.0001,
                funding_interval_minutes=funding_interval_minutes,
                funding_interval_source=funding_interval_source,
                source="test",
                ingested_at=UtcTimestamp(datetime(2026, 6, 22, 12, 11, tzinfo=timezone.utc)),
            )
        ]


class _Writer:
    def __init__(self, latest):
        self.latest = latest
        self.rows = []

    def latest_funding_time(self, instrument_id):
        return self.latest

    def write_funding_rates(self, rows):
        self.rows.extend(rows)


class _Universe:
    def __init__(self, rows):
        self.rows = rows

    def list_tradable_funding_instruments(self, *, market_ids):
        return self.rows

    def upsert_funding_instruments(self, rows):
        raise AssertionError("not used")

    def get_funding_instrument(self, instrument_id):
        return None


def _instrument(interval: int | None = 480):
    return FundingInstrument(
        instrument_id=InstrumentId(MarketId(2), Symbol("BTCUSDT")),
        instrument_key="binance:futures:BTCUSDT",
        exchange="binance",
        market_type="futures",
        status="TRADING",
        is_tradable=1,
        base_asset="BTC",
        quote_asset="USDT",
        funding_interval_minutes=interval,
        funding_interval_source="test_interval",
        funding_cap=None,
        funding_floor=None,
        updated_at=UtcTimestamp(datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc)),
    )


def test_non_due_symbol_does_not_call_provider() -> None:
    source = _Source()
    writer = _Writer(UtcTimestamp(datetime(2026, 6, 22, 8, 0, tzinfo=timezone.utc)))
    use_case = BackfillFundingRatesUseCase(
        source=source,
        writer=writer,
        universe_store=_Universe([_instrument()]),
        clock=_Clock(),
        tail_lookback_intervals=3,
        settlement_lag_minutes=30,
    )

    report = use_case.run_due_universe(market_ids=(MarketId(2),))

    assert report.instruments_skipped == 1
    assert report.instrument_reports[0].status == "not_due"
    assert source.calls == []
    assert writer.rows == []


def test_due_symbol_fetches_provider_and_writes_rows() -> None:
    source = _Source()
    writer = _Writer(UtcTimestamp(datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc)))
    use_case = BackfillFundingRatesUseCase(
        source=source,
        writer=writer,
        universe_store=_Universe([_instrument()]),
        clock=_Clock(),
        tail_lookback_intervals=3,
        settlement_lag_minutes=10,
    )

    report = use_case.run_due_universe(market_ids=(MarketId(2),))

    assert report.instruments_ok == 1
    assert len(source.calls) == 1
    assert len(writer.rows) == 1


def test_missing_interval_is_degraded_and_skipped() -> None:
    source = _Source()
    writer = _Writer(None)
    use_case = BackfillFundingRatesUseCase(
        source=source,
        writer=writer,
        universe_store=_Universe([_instrument(interval=None)]),
        clock=_Clock(),
        tail_lookback_intervals=3,
        settlement_lag_minutes=10,
    )

    report = use_case.run_due_universe(market_ids=(MarketId(2),))

    assert report.instrument_reports[0].status == "skipped_missing_interval"
    assert source.calls == []
