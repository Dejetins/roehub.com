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
    """
    Fake InstrumentRefWriter:
    - existing_latest returns state from the provided mapping
    - upsert records writes
    """

    def __init__(self, existing: dict[tuple[int, str], tuple[str, int]] | None = None) -> None:
        self._existing = existing or {}
        self.upserts = []

    def existing_latest(self, *, market_id, symbols):
        out: dict[str, tuple[str, int]] = {}
        for s in symbols:
            key = (market_id.value, str(s))
            if key in self._existing:
                out[str(s)] = self._existing[key]
        return out

    def upsert(self, rows) -> None:
        self.upserts.extend(list(rows))


def test_seed_ref_market_inserts_only_missing() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))
    writer = _FakeMarketWriter(existing={1, 3})

    uc = SeedRefMarketUseCase(writer=writer, clock=clock)
    rep = uc.run()

    assert rep.inserted == 2
    assert sorted([r.market_id.value for r in writer.inserted]) == [2, 4]


def test_sync_whitelist_inserts_only_new_rows() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))
    writer = _FakeInstrumentWriter(existing={})
    uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1, 2, 3, 4})

    rows = [
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("BTCUSDT")), True),
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("ETHUSDT")), False),
    ]
    rep = uc.run(rows)

    assert rep.rows_total == 2
    assert rep.rows_upserted == 2
    assert rep.rows_skipped_unchanged == 0
    assert rep.enabled_count == 1
    assert rep.disabled_count == 1
    assert len(writer.upserts) == 2


def test_sync_whitelist_skips_unchanged_rows() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))

    existing = {
        (1, "BTCUSDT"): ("ENABLED", 1),
        (1, "ETHUSDT"): ("DISABLED", 0),
    }
    writer = _FakeInstrumentWriter(existing=existing)

    uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1, 2, 3, 4})

    rows = [
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("BTCUSDT")), True),
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("ETHUSDT")), False),
    ]
    rep = uc.run(rows)

    assert rep.rows_total == 2
    assert rep.rows_upserted == 0
    assert rep.rows_skipped_unchanged == 2
    assert len(writer.upserts) == 0


def test_sync_whitelist_updates_changed_rows() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))

    # currently enabled in DB, but whitelist disables it => should write 1 update
    existing = {
        (1, "BTCUSDT"): ("ENABLED", 1),
    }
    writer = _FakeInstrumentWriter(existing=existing)

    uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1, 2, 3, 4})

    rows = [
        WhitelistInstrumentRow(InstrumentId(MarketId(1), Symbol("BTCUSDT")), False),
    ]
    rep = uc.run(rows)

    assert rep.rows_total == 1
    assert rep.rows_upserted == 1
    assert rep.rows_skipped_unchanged == 0
    assert len(writer.upserts) == 1

    up = writer.upserts[0]
    assert up.status == "DISABLED"
    assert up.is_tradable == 0


def test_sync_whitelist_rejects_unknown_market_id() -> None:
    clock = _FakeClock(UtcTimestamp(datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)))
    writer = _FakeInstrumentWriter(existing={})

    uc = SyncWhitelistToRefInstrumentsUseCase(writer=writer, clock=clock, known_market_ids={1})

    rows = [WhitelistInstrumentRow(InstrumentId(MarketId(2), Symbol("BTCUSDT")), True)]
    with pytest.raises(ValueError):
        uc.run(rows)
