"""Resolve requested instrument metadata without ingesting candle history."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, ContextManager, Mapping, Protocol

from trading.contexts.market_data.application.ports.sources.instrument_history_start_source import (
    InstrumentHistoryStartSource,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


class HistoryBoundsStore(Protocol):
    def claim(self, *, now: datetime) -> Mapping[str, Any] | None: ...
    def finish(
        self, *, job: Mapping[str, Any], first_open_at: datetime | None, now: datetime
    ) -> None: ...


class DiscoveryCoordinator(Protocol):
    def execution(self) -> ContextManager[Callable[[], None] | None]: ...


@dataclass
class HistoryBoundsRunner:
    store: HistoryBoundsStore
    coordinator: DiscoveryCoordinator
    source_factory: Callable[[], InstrumentHistoryStartSource]
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def run_once(self) -> bool:
        with self.coordinator.execution() as check:
            if check is None:
                return False
            check()
            job = self.store.claim(now=self.now())
            if job is None:
                return False
            first = None
            try:
                result = self.source_factory().get_history_start(
                    InstrumentId(MarketId(job["market_id"]), Symbol(job["symbol"]))
                )
                if result is not None:
                    first = result.value.replace(second=0, microsecond=0)
                    if not datetime(2017, 1, 1, tzinfo=UTC) <= first < self.now():
                        first = None
            except Exception:
                # Provider details do not enter browser responses or durable metadata.
                first = None
            check()
            self.store.finish(job=job, first_open_at=first, now=self.now())
            return True
