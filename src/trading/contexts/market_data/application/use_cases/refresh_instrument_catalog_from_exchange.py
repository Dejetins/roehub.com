from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from trading.contexts.market_data.application.dto import InstrumentRefEnrichmentUpsert
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.instrument_metadata_source import (
    InstrumentMetadataSource,
)
from trading.contexts.market_data.application.ports.stores.catalog_snapshot_writer import (
    CatalogSnapshotWriter,
)
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)
from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class CatalogRefreshReport:
    markets_total: int
    instruments_total: int
    rows_upserted: int


@dataclass(frozen=True, slots=True)
class RefreshInstrumentCatalogFromExchangeUseCase:
    """Refresh bounded exchange metadata without creating user selections or backfills."""

    market_ids: Sequence[MarketId]
    metadata_source: InstrumentMetadataSource
    writer: InstrumentRefWriter
    clock: Clock
    snapshot_writer: CatalogSnapshotWriter | None = None
    before_write: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.market_ids:
            raise ValueError("RefreshInstrumentCatalogFromExchangeUseCase requires market ids")
        if self.metadata_source is None or self.writer is None or self.clock is None:  # type: ignore[truthy-bool]
            raise ValueError("catalog refresh dependencies are required")

    def run(self) -> CatalogRefreshReport:
        now = self.clock.now()
        payload: list[InstrumentRefEnrichmentUpsert] = []
        for market_id in self.market_ids:
            for row in self.metadata_source.list_for_market(market_id):
                payload.append(
                    InstrumentRefEnrichmentUpsert(
                        market_id=row.instrument_id.market_id,
                        symbol=row.instrument_id.symbol,
                        status=row.status,
                        is_tradable=row.is_tradable,
                        base_asset=row.base_asset,
                        quote_asset=row.quote_asset,
                        price_step=row.price_step,
                        qty_step=row.qty_step,
                        min_notional=row.min_notional,
                        updated_at=now,
                    )
                )
        if self.before_write is not None:
            self.before_write()
        self.writer.upsert_enrichment(payload)
        if self.snapshot_writer is not None:
            if self.before_write is not None:
                self.before_write()
            self.snapshot_writer.publish(
                market_ids=self.market_ids, rows=payload, refreshed_at=now.value
            )
        return CatalogRefreshReport(
            markets_total=len(self.market_ids),
            instruments_total=len(payload),
            rows_upserted=len(payload),
        )
