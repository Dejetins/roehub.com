"""Post-commit hints cannot change ingestion success or execute before the write."""

import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_artifact_precompute_runner_v2 as n,
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,
)


def test_signal_after_write_deduplicates_coordinates():
    events = []

    class Writer:
        def write_1m(self, rows):
            events.append(("commit", len(rows)))

    writer = StoredCandleSignalWriter(
        Writer(), lambda instrument: events.append(("hint", instrument))
    )
    writer.write_1m(iter(n._build_canonical_rows_v2(bar_indexes=(0, 1))))
    assert [e[0] for e in events] == ["commit", "hint"]
    assert events[0][1] == 2


def test_failed_write_has_no_hint():
    events = []

    class Writer:
        def write_1m(self, rows):
            raise OSError("fixture")

    writer = StoredCandleSignalWriter(Writer(), events.append)
    with pytest.raises(OSError):
        writer.write_1m(n._build_canonical_rows_v2(bar_indexes=(0,)))
    assert not events


def test_failed_hint_does_not_retry_committed_write():
    events = []

    class Writer:
        def write_1m(self, rows):
            events.append("commit")

    def unavailable(_):
        raise OSError("fixture")

    StoredCandleSignalWriter(Writer(), unavailable).write_1m(
        n._build_canonical_rows_v2(bar_indexes=(0,))
    )
    assert events == ["commit"]
