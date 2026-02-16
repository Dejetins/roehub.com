from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import StrategyEventRepository
from trading.contexts.strategy.domain.entities import StrategyEvent
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class InMemoryStrategyEventRepository(StrategyEventRepository):
    """
    InMemoryStrategyEventRepository — deterministic append-only in-memory event stream adapter.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_event_repository.py
      - apps/api/wiring/modules/strategy.py
      - tests/unit/contexts/strategy/application
    """

    def __init__(self) -> None:
        """
        Initialize empty append-only in-memory event storage.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Adapter lifetime is process-local and non-persistent.
        Raises:
            None.
        Side Effects:
            Creates mutable in-memory dictionary state.
        """
        self._events_by_id: dict[UUID, StrategyEvent] = {}

    def append(self, *, event: StrategyEvent) -> StrategyEvent:
        """
        Append one immutable event snapshot.

        Args:
            event: Event snapshot to append.
        Returns:
            StrategyEvent: Persisted event snapshot.
        Assumptions:
            Event ids are unique per adapter instance.
        Raises:
            StrategyStorageError: If duplicate event id is appended.
        Side Effects:
            Writes event snapshot to in-memory dictionary.
        """
        if event.event_id in self._events_by_id:
            raise StrategyStorageError("InMemoryStrategyEventRepository duplicate event_id")
        self._events_by_id[event.event_id] = event
        return event

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        List strategy-level events in deterministic ordering by timestamp and event_id.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered event snapshots.
        Assumptions:
            Ordering matches SQL adapter deterministic contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        events = [
            event
            for event in self._events_by_id.values()
            if event.user_id == user_id and event.strategy_id == strategy_id
        ]
        return tuple(sorted(events, key=lambda item: (item.ts, str(item.event_id))))

    def list_for_run(self, *, user_id: UserId, run_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        List run-level events in deterministic ordering by timestamp and event_id.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered run event snapshots.
        Assumptions:
            `run_id`-scoped stream excludes strategy-level events with null run_id.
        Raises:
            None.
        Side Effects:
            None.
        """
        events = [
            event
            for event in self._events_by_id.values()
            if event.user_id == user_id and event.run_id == run_id
        ]
        return tuple(sorted(events, key=lambda item: (item.ts, str(item.event_id))))
