from datetime import datetime
from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import InstrumentRefEnrichmentUpsert
from trading.shared_kernel.primitives import MarketId


class CatalogSnapshotWriter(Protocol):
    """Publish immutable public metadata only after reference writes succeed."""

    def publish(
        self,
        *,
        market_ids: Sequence[MarketId],
        rows: Sequence[InstrumentRefEnrichmentUpsert],
        refreshed_at: datetime,
    ) -> None: ...
