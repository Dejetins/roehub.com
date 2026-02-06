from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from trading.contexts.market_data.application.dto.reference_data import (
    RefMarketRow,
    WhitelistInstrumentRow,
)
from trading.contexts.market_data.application.use_cases.seed_ref_market import SeedRefMarketUseCase
from trading.contexts.market_data.application.use_cases.sync_whitelist_to_ref_instruments import (
    SyncWhitelistToRefInstrumentsUseCase,
)
from trading.shared_kernel.primitives.instrument_id import InstrumentId
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol
from trading.shared_kernel.primitives.utc_timestamp import UtcTimestamp


@dataclass(frozen=True, slots=True)
class _FakeClock:
    now_value: UtcTimestamp

    def now(self) -> UtcTimestamp:
        return self.now_value


class _FakeMarketWriter:
    def __init__(self, existing: set[int]) -> None:
        self._existing = set(existing)
        self.inserted: list[RefMarketRow] = []

    def existing_market_ids(self, ids) -> set[int]:
        return self._existing.intersection({i.value for i in ids})

    def insert(self, rows) -> None:
        self.inserted.extend(list(rows))


class _FakeInstrumentWriter:
    def __init__(self) -> None:
        self.upserts = []

    def upsert(self, rows) -> None:
        self.upserts.extend(list(rows))


def test_seed_ref_market_inserts_only_missing() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))
    writer = _FakeMarketWriter(existing={1, 3})
    uc = SeedRefMarketUseCase(writer=writer, clock=clock)

    rep = uc.run()
    assert rep.inserted == 2
    assert sorted([r.market_id.value for r in writer.inserted]) == [2, 4]


def test_sync_whitelist_maps_enabled_disabled_and_validates_market_ids() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))
    writer = _FakeInstrumentWriter()
    uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1, 2, 3, 4})  # noqa: E501

    rows = [
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("BTCUSDT")), True),
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("ETHUSDT")), False),
    ]
    rep = uc.run(rows)

    assert rep.rows_total == 2
    assert rep.enabled_count == 1
    assert rep.disabled_count == 1
    assert len(writer.upserts) == 2

    up0 = writer.upserts[0]
    assert up0.status == "ENABLED"
    assert up0.is_tradable == 1

    up1 = writer.upserts[1]
    assert up1.status == "DISABLED"
    assert up1.is_tradable == 0

    bad_uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1})
    with pytest.raises(ValueError):
        bad_uc.run([WhitelistInstrumentRow(InstrumentId(MarketId(2), Symbol("BTCUSDT")), True)])
