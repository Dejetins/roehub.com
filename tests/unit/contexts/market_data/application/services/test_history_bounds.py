from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from trading.contexts.market_data.application.services.history_bounds import HistoryBoundsRunner
from trading.shared_kernel.primitives import UtcTimestamp


def stub(**values: Any) -> Any:
    """Partial deterministic port doubles for the exercised runner paths."""
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "first",
    [
        None,
        datetime(2016, 1, 1, tzinfo=UTC),
        datetime(2030, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, tzinfo=UTC),
    ],
)
def test_discovery_persists_only_valid_confirmed_history(first):
    writes = []
    now = datetime(2026, 10, 3, tzinfo=UTC)
    store = stub(
        claim=lambda **_: dict(market_id=3, symbol="BTCUSDT"), finish=lambda **kw: writes.append(kw)
    )
    source = stub(get_history_start=lambda _: UtcTimestamp(first) if first else None)
    runner = HistoryBoundsRunner(
        store,
        stub(execution=lambda: nullcontext(lambda: None)),
        lambda: source,
        now=lambda: now,
    )
    assert runner.run_once()
    assert writes[0]["first_open_at"] == (
        first if first and datetime(2017, 1, 1, tzinfo=UTC) <= first < now else None
    )


def test_discovery_does_not_run_without_execution_ownership():
    runner = HistoryBoundsRunner(
        stub(),
        stub(execution=lambda: nullcontext(None)),
        lambda: pytest.fail("must not query exchange"),
    )
    assert runner.run_once() is False
