"""Retained storage-independent assertions from the removed adapter tests."""

from datetime import timedelta

import pytest

from tests.real_infra.market_data.pg_source_fixture import market_source
from tests.real_infra.market_data.test_postgres_readers_frontier import (
    BASE,
    DSN,
    candle,
    instrument,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.funding_rate_store import (
    PostgresFundingRateStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.instrument_coverage_reader import (  # noqa: E501
    PostgresInstrumentCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.application.dto import FundingRateRecord
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG16 required")


@pytest.fixture(scope="module")
def gateway():
    with market_source(DSN) as gateway:
        yield gateway


def test_empty_and_partial_funding_keep_quality_and_utc(gateway):
    store = PostgresFundingRateStore(gateway)
    target = InstrumentId(MarketId(2), Symbol("S07FUNDING"))
    window = TimeRange(UtcTimestamp(BASE), UtcTimestamp(BASE + timedelta(hours=24)))
    assert store.latest_funding_time(target) is None
    empty = store.read_funding_coverage(instrument_id=target, time_range=window)
    assert empty.status == "unavailable" and empty.reason_codes == ("no_funding_rows",)
    assert empty.records == ()
    store.write_funding_rates(
        [
            FundingRateRecord(
                instrument_id=target,
                instrument_key="binance:futures:S07FUNDING",
                funding_time=UtcTimestamp(BASE + timedelta(hours=hour)),
                funding_rate=rate,
                funding_interval_minutes=480,
                funding_interval_source="provider",
                source="rest",
                ingested_at=UtcTimestamp(BASE),
                ingest_id=None,
                mark_price=None,
                bybit_category=None,
            )
            for hour, rate in [(0, 0.0001), (16, -0.0002)]
        ]
    )
    partial = store.read_funding_coverage(instrument_id=target, time_range=window)
    assert partial.status == "degraded"
    assert partial.coverage_policy == "degraded_with_warning"
    assert partial.missing_event_count == 1 and partial.reason_codes == ("funding_interval_gap",)
    assert [r.funding_rate for r in partial.records] == [0.0001, -0.0002]
    assert store.latest_funding_time(target) == UtcTimestamp(BASE + timedelta(hours=16))


def test_duplicate_coverage_cap_and_inclusive_lower_bound(gateway):
    rows = [candle("S07INDEX", minute) for minute in range(65)]
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m(rows + rows)
    snapshot = PostgresInstrumentCoverageReader(gateway).read(
        instrument_id=instrument("S07INDEX"),
        expected_start_at=BASE,
        expected_end_at=BASE + timedelta(minutes=60),
    )
    assert snapshot.state == "complete" and snapshot.percent == 100.0
    index = PostgresCanonicalCandleIndexReader(gateway)
    assert index.max_ts_open_lt(
        instrument_id=instrument("S07INDEX"),
        before=UtcTimestamp(BASE + timedelta(minutes=61)),
        after=UtcTimestamp(BASE + timedelta(minutes=60)),
    ) == UtcTimestamp(BASE + timedelta(minutes=60))


def test_zero_history_gate_waits_for_last_actual_minute(gateway, monkeypatch):
    from datetime import datetime

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
        PostgresHistoryCoverageReader,
    )
    from trading.contexts.market_data.application.services import history_stream_gate
    from trading.contexts.market_data.application.services.history_readiness import HistoryReadiness

    end = BASE + timedelta(minutes=180)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return end

    monkeypatch.setattr(history_stream_gate, "datetime", Clock)
    target = instrument("S07GATE")
    desired = [target]
    class Enabled:
        def list_enabled_tradable(self):
            return desired

    class Origins:
        def ready_starts(self, **kwargs):
            return {(1, "S07GATE"): BASE}

        def read_or_request(self, **kwargs):
            raise AssertionError("ready_starts is the tested batched path")

    gate = history_stream_gate.HistoryReadyInstrumentReader(
        source=Enabled(),
        readiness=HistoryReadiness(
            origins=Origins(),
            coverage=PostgresHistoryCoverageReader(gateway),
        ),
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
        CoverageUnavailable,
    )

    with pytest.raises(CoverageUnavailable, match="canonical_coverage_unknown"):
        gate.list_enabled_tradable()

    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("S07GATE", minute) for minute in range(179)])
    assert not gate.list_enabled_tradable()
    writer.write_1m([candle("S07GATE", 179)])
    assert gate.list_enabled_tradable() == (target,)
    desired.clear()
    assert not gate.list_enabled_tradable()
