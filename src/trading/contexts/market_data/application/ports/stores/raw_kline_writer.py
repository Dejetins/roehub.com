from __future__ import annotations

from typing import Iterable, Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta


class RawKlineWriter(Protocol):
    """Persist one caller-bounded batch of normalized closed candles.

    Successful return acknowledges durable storage; failures propagate and callers
    retain recovery work. Adapters route raw observations and own canonical/index
    maintenance. PostgreSQL commits those effects atomically; the legacy adapter
    remains for consumers awaiting the coordinated storage switch. Raw replay may
    retain observations, but must not inflate canonical counts or content revisions.
    """

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        ...
