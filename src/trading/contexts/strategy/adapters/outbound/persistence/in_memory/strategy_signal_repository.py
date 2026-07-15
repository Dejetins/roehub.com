from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import StrategySignalRepository
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import OrganizationId, UserId


class InMemoryStrategySignalRepository(StrategySignalRepository):
    def __init__(self) -> None:
        self._signals_by_id: dict[UUID, StrategySignal] = {}

    def record(self, *, signal: StrategySignal) -> StrategySignal:
        self._signals_by_id.setdefault(signal.signal_id, signal)
        return self._signals_by_id[signal.signal_id]

    def list_latest_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        limit: int,
    ) -> tuple[StrategySignal, ...]:
        bounded_limit = max(0, min(int(limit), 100))
        rows = [
            signal
            for signal in self._signals_by_id.values()
            if signal.organization_id == organization_id
            and signal.owner_user_id == owner_user_id
            and signal.strategy_id == strategy_id
        ]
        ordered = sorted(
            rows,
            key=lambda signal: (signal.created_at or signal.bar_ts_close, str(signal.signal_id)),
            reverse=True,
        )
        return tuple(ordered[:bounded_limit])
