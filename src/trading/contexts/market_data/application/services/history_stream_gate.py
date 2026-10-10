"""Admit a new stream only after its full closed history has been verified."""

import logging
from datetime import UTC, datetime
from typing import Callable, Sequence

from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.services.history_readiness import HistoryReadiness
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp

log = logging.getLogger(__name__)


class HistoryReadyInstrumentReader:
    def __init__(
        self,
        *,
        source: EnabledInstrumentReader,
        readiness: HistoryReadiness,
        on_ready: Callable[[InstrumentId, datetime], None] = lambda *_: None,
    ) -> None:
        self.source, self.readiness = source, readiness
        self.on_ready = on_ready
        self._admitted: set[InstrumentId] = set()
        self.repairs: list[RestFillTask] = []
        self._states: dict[InstrumentId, str] = {}

    def list_enabled_tradable(self) -> Sequence[InstrumentId]:
        desired = tuple(self.source.list_enabled_tradable())
        self.repairs = []
        for removed in self._states.keys() - set(desired):
            self._event(removed, "disabled")
            del self._states[removed]
        self._admitted.intersection_update(desired)
        pending = [i for i in desired if i not in self._admitted]
        if pending:
            starts = self.readiness.origins.ready_starts(
                market_ids=list({i.market_id.value for i in pending})
            )
            starts = {
                (i.market_id.value, str(i.symbol)): starts[(i.market_id.value, str(i.symbol))]
                for i in pending
                if (i.market_id.value, str(i.symbol)) in starts
            }
            end = datetime.now(UTC).replace(second=0, microsecond=0)
            counts = self.readiness.coverage.counts(starts=starts, end_at=end)
            absent = self.readiness.coverage.source_absent_counts(starts=starts, end_at=end)
            for i in pending:
                key = (i.market_id.value, str(i.symbol))
                if key in starts and 0 < (end - starts[key]).total_seconds() / 60 == counts.get(
                    key, 0
                ) + absent.get(key, 0):
                    self.on_ready(i, end)
                    self._admitted.add(i)
                    self._event(i, "history_ready")
                elif key not in starts:
                    self._event(i, "finding_history_start")
                    self.readiness.origins.read_or_request(
                        market_id=key[0], symbol=key[1], now=datetime.now(UTC)
                    )
                else:
                    self._event(i, "preparing_history")
                    gap = self.readiness.coverage.first_gap(
                        instrument_id=i,
                        time_range=TimeRange(UtcTimestamp(starts[key]), UtcTimestamp(end)),
                    )
                    if gap:
                        self.repairs.append(
                            RestFillTask(
                                i,
                                gap,
                                "stream_history_repair",
                            )
                        )
        # Once admitted, normal WS reconnect/tail repair owns subsequent live gaps.
        return tuple(i for i in desired if i in self._admitted)

    def _event(self, instrument: InstrumentId, state: str) -> None:
        if self._states.get(instrument) == state:
            return
        self._states[instrument] = state
        log.info("stream_transition market_id=%s symbol=%s state=%s",
                 instrument.market_id.value, str(instrument.symbol), state)
