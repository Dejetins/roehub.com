from datetime import datetime
from typing import Mapping, Protocol, Sequence


class CatalogCoverageReader(Protocol):
    """Bounded, aggregate-only reads for the exchange catalog workspace."""

    def counts(
        self, *, market_ids: Sequence[int], start_at: datetime, end_at: datetime
    ) -> Mapping[tuple[int, str], int]: ...

    def latest(self, *, market_id: int, symbol: str, before: datetime) -> datetime | None: ...
