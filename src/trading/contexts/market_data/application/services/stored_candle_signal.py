"""Optional post-write hint. Canonical visibility/reconciliation owns durable readiness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.shared_kernel.primitives import InstrumentId

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredCandleSignalWriter:
    """Wrap a trusted committed writer at composition for REST, WS and historical writes.

    Hints carry coordinates only, never compute/download. The bounded batch already owned
    by ingestion is materialized once. A failed hint must not roll back/retry committed
    candles; the artifact reconciler enumerates actual canonical stored coordinates.
    """

    delegate: RawKlineWriter
    notify: Callable[[InstrumentId], None]

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        batch = tuple(rows)
        self.delegate.write_1m(batch)
        notify_stored_candles(batch, self.notify)


def notify_stored_candles(rows: Iterable[CandleWithMeta], notify: Callable[[InstrumentId], None]):
    for instrument in dict.fromkeys(row.candle.instrument_id for row in rows):
        try:
            notify(instrument)
        except Exception:
            log.warning("stored candle hint unavailable; bounded reconciliation required")
