from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.market_data.application.dto import FundingInstrument
from trading.contexts.market_data.application.use_cases.sync_futures_funding_universe import (
    SyncFuturesFundingUniverseUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


class _Clock:
    def now(self):
        return UtcTimestamp(datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc))


class _Source:
    def list_funding_instruments(self, market_id):
        return [
            FundingInstrument(
                instrument_id=InstrumentId(market_id, Symbol("BTCUSDT")),
                instrument_key=f"test:futures:{market_id.value}:BTCUSDT",
                exchange="test",
                market_type="futures",
                status="TRADING",
                is_tradable=1,
                base_asset="BTC",
                quote_asset="USDT",
                funding_interval_minutes=480 if market_id.value == 2 else None,
                funding_interval_source="test" if market_id.value == 2 else None,
                funding_cap=None,
                funding_floor=None,
                updated_at=UtcTimestamp(datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)),
            )
        ]


class _Store:
    def __init__(self):
        self.rows = []

    def upsert_funding_instruments(self, rows):
        self.rows.extend(rows)

    def list_tradable_funding_instruments(self, *, market_ids):
        return []

    def get_funding_instrument(self, instrument_id):
        return None


def test_sync_reports_missing_interval_metadata() -> None:
    store = _Store()
    use_case = SyncFuturesFundingUniverseUseCase(
        source=_Source(),
        store=store,
        clock=_Clock(),
        market_ids=(MarketId(2), MarketId(4)),
    )

    report = use_case.run()

    assert report.instruments_total == 2
    assert report.instruments_with_interval == 1
    assert report.instruments_missing_interval == 1
    assert len(store.rows) == 2
