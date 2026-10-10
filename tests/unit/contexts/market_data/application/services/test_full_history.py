from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.ports.stores.history_coverage_reader import (
    HistoryCoverageReader,
)
from trading.contexts.market_data.application.services.history_readiness import (
    HistoryOrigins,
    HistoryReadiness,
    history_snapshot,
)
from trading.contexts.market_data.application.services.history_stream_gate import (
    HistoryReadyInstrumentReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


def test_full_history_ratio_never_rounds_a_gap_to_complete():
    end = datetime(2026, 1, 1, tzinfo=UTC)
    first = end - timedelta(minutes=1_000_000)
    result = history_snapshot(first=first, end=end, actual=999_999)
    assert result["state"] == "partial"
    assert result["coverage_percent"] == 99.99
    assert result["missing_candles"] == 1
    assert history_snapshot(first=None, end=end, actual=0)["coverage_percent"] is None


def test_exchange_halt_is_excluded_without_inventing_candles():
    end = datetime(2026, 1, 1, tzinfo=UTC)
    first = end - timedelta(minutes=120)
    result = history_snapshot(first=first, end=end, actual=60, source_absent=60)
    assert result["state"] == "complete"
    assert result["actual_candles"] == result["expected_candles"] == 60
    assert result["source_absent_minutes"] == 60
    assert result["missing_candles"] == 0
    assert result["coverage_percent"] == 100
    empty = history_snapshot(first=first, end=end, actual=0, source_absent=120)
    assert empty["state"] == "complete" and empty["coverage_percent"] == 100


def test_new_stream_and_restart_repair_history_before_ws_admission():
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    first = end - timedelta(days=3)
    key = (1, "ACTUSDT")
    missing = True
    origins = SimpleNamespace(ready_starts=lambda **kw: {key: first})
    coverage = SimpleNamespace(
        source_absent_counts=lambda **kw: {}, counts=lambda **kw: {
            key: int((kw["end_at"] - first).total_seconds() / 60) - (1 if missing else 0)
        },
        first_gap=lambda **kw: TimeRange(
            UtcTimestamp(first + timedelta(minutes=5)), UtcTimestamp(first + timedelta(minutes=6))
        ),
    )
    reader = HistoryReadyInstrumentReader(
        source=cast(
            EnabledInstrumentReader, SimpleNamespace(list_enabled_tradable=lambda: [instrument]),
        ),
        readiness=HistoryReadiness(
            origins=cast(HistoryOrigins, origins),
            coverage=cast(HistoryCoverageReader, coverage),
        ),
    )
    assert reader.list_enabled_tradable() == ()
    assert len(reader.repairs) == 1
    assert reader.repairs[0].reason == "stream_history_repair"
    missing = False
    assert reader.list_enabled_tradable() == (instrument,)
    missing = True  # An admitted live connection retains normal reconnect/tail repair.
    assert reader.list_enabled_tradable() == (instrument,)
    restarted = HistoryReadyInstrumentReader(source=reader.source, readiness=reader.readiness)
    assert restarted.list_enabled_tradable() == ()
    assert restarted.repairs


def test_unknown_origin_enqueues_discovery_not_ws():
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    requested = []
    reader = HistoryReadyInstrumentReader(
        source=cast(
            EnabledInstrumentReader, SimpleNamespace(list_enabled_tradable=lambda: [instrument]),
        ),
        readiness=HistoryReadiness(
            origins=cast(HistoryOrigins, SimpleNamespace(
                ready_starts=lambda **kw: {}, read_or_request=lambda **kw: requested.append(kw)
            )),
            coverage=cast(HistoryCoverageReader, SimpleNamespace(
                source_absent_counts=lambda **kw: {}, counts=lambda **kw: {})),
        ),
    )
    assert reader.list_enabled_tradable() == ()
    assert requested[0]["symbol"] == "ACTUSDT"
    assert reader.repairs == []


def test_empty_history_queues_whole_gap_and_disabling_prevents_admission():
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    first = end - timedelta(days=300)
    desired = [instrument]
    reader = HistoryReadyInstrumentReader(
        source=cast(
            EnabledInstrumentReader, SimpleNamespace(list_enabled_tradable=lambda: desired),
        ),
        readiness=HistoryReadiness(
            origins=cast(
                HistoryOrigins, SimpleNamespace(ready_starts=lambda **kw: {(1, "ACTUSDT"): first}),
            ),
            coverage=cast(HistoryCoverageReader, SimpleNamespace(
                source_absent_counts=lambda **kw: {}, counts=lambda **kw: {},
                first_gap=lambda **kw: kw["time_range"],
            )),
        ),
    )
    assert reader.list_enabled_tradable() == ()
    assert reader.repairs[0].time_range.start.value == first
    assert reader.repairs[0].time_range.end.value >= end
    desired.clear()
    assert reader.list_enabled_tradable() == ()
    assert not reader.repairs
