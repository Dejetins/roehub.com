from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.funding_instrument_universe_source import (  # noqa: E501
    FundingInstrumentUniverseSource,
)
from trading.contexts.market_data.application.ports.stores.funding_instrument_universe_store import (  # noqa: E501
    FundingInstrumentUniverseStore,
)
from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class FundingUniverseMarketReport:
    market_id: MarketId
    instruments_total: int
    instruments_with_interval: int
    instruments_missing_interval: int


@dataclass(frozen=True, slots=True)
class SyncFuturesFundingUniverseReport:
    markets_total: int
    instruments_total: int
    instruments_with_interval: int
    instruments_missing_interval: int
    rows_written: int
    market_reports: tuple[FundingUniverseMarketReport, ...]


class SyncFuturesFundingUniverseUseCase:
    def __init__(
        self,
        *,
        source: FundingInstrumentUniverseSource,
        store: FundingInstrumentUniverseStore,
        clock: Clock,
        market_ids: Sequence[MarketId],
    ) -> None:
        if source is None:  # type: ignore[truthy-bool]
            raise ValueError("SyncFuturesFundingUniverseUseCase requires source")
        if store is None:  # type: ignore[truthy-bool]
            raise ValueError("SyncFuturesFundingUniverseUseCase requires store")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("SyncFuturesFundingUniverseUseCase requires clock")
        if not market_ids:
            raise ValueError("SyncFuturesFundingUniverseUseCase requires market_ids")
        self._source = source
        self._store = store
        self._clock = clock
        self._market_ids = tuple(market_ids)

    def run(self) -> SyncFuturesFundingUniverseReport:
        all_rows = []
        reports: list[FundingUniverseMarketReport] = []
        for market_id in self._market_ids:
            rows = list(self._source.list_funding_instruments(market_id))
            with_interval = sum(1 for row in rows if row.funding_interval_minutes is not None)
            missing_interval = len(rows) - with_interval
            reports.append(
                FundingUniverseMarketReport(
                    market_id=market_id,
                    instruments_total=len(rows),
                    instruments_with_interval=with_interval,
                    instruments_missing_interval=missing_interval,
                )
            )
            all_rows.extend(rows)

        self._store.upsert_funding_instruments(all_rows)
        return SyncFuturesFundingUniverseReport(
            markets_total=len(self._market_ids),
            instruments_total=len(all_rows),
            instruments_with_interval=sum(r.instruments_with_interval for r in reports),
            instruments_missing_interval=sum(r.instruments_missing_interval for r in reports),
            rows_written=len(all_rows),
            market_reports=tuple(reports),
        )
