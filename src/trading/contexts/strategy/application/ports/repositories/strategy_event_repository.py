from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import StrategyEvent
from trading.shared_kernel.primitives import UserId


class StrategyEventRepository(Protocol):
    """
    StrategyEventRepository — append-only storage port for Strategy v1 events.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_event.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_event_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def append(self, *, event: StrategyEvent) -> StrategyEvent:
        """
        Persist one append-only strategy event.

        Args:
            event: Event snapshot to append.
        Returns:
            StrategyEvent: Persisted event snapshot.
        Assumptions:
            Repository forbids update/delete operations for event stream.
        Raises:
            ValueError: If repository cannot append event.
        Side Effects:
            Writes one event row.
        """
        ...

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        Read strategy-level event stream in deterministic order.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered event snapshots.
        Assumptions:
            Ordering is deterministic by timestamp and event id.
        Raises:
            ValueError: If repository cannot map one of rows.
        Side Effects:
            Reads zero or more event rows.
        """
        ...

    def list_for_run(self, *, user_id: UserId, run_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        Read run-level event stream in deterministic order.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered event snapshots for run.
        Assumptions:
            `run_id` can be null in table but this query scopes only run-specific events.
        Raises:
            ValueError: If repository cannot map one of rows.
        Side Effects:
            Reads zero or more event rows.
        """
        ...
