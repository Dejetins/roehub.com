from __future__ import annotations

from typing import Protocol

from trading.contexts.strategy.domain.entities import StrategySignal


class StrategyExecutionProducer(Protocol):
    def record_signal(self, *, signal: StrategySignal) -> None: ...


class NoOpStrategyExecutionProducer:
    def record_signal(self, *, signal: StrategySignal) -> None:
        _ = signal
