from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import Strategy
from trading.shared_kernel.primitives import UserId


class StrategyRepository(Protocol):
    """
    StrategyRepository — storage port for immutable Strategy v1 aggregates.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def create(self, *, strategy: Strategy) -> Strategy:
        """
        Persist new immutable strategy snapshot.

        Args:
            strategy: Strategy aggregate to store.
        Returns:
            Strategy: Persisted strategy snapshot.
        Assumptions:
            `strategy.spec_json` is immutable and cannot be updated after insert.
        Raises:
            ValueError: If repository implementation cannot persist strategy.
        Side Effects:
            Writes one row into strategy storage.
        """
        ...

    def find_by_strategy_id(self, *, user_id: UserId, strategy_id: UUID) -> Strategy | None:
        """
        Load strategy by owner id and strategy id.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            Strategy | None: Persisted strategy or `None`.
        Assumptions:
            Lookup scope is owner-local and deterministic.
        Raises:
            ValueError: If repository implementation cannot map row.
        Side Effects:
            Reads one row from storage.
        """
        ...

    def find_any_by_strategy_id(self, *, strategy_id: UUID) -> Strategy | None:
        """
        Load strategy by identifier without owner filtering for explicit use-case ownership checks.

        Args:
            strategy_id: Strategy identifier.
        Returns:
            Strategy | None: Persisted strategy snapshot or `None`.
        Assumptions:
            Ownership/visibility checks are performed explicitly in application use-cases.
        Raises:
            ValueError: If repository implementation cannot map row.
        Side Effects:
            Reads one row from storage.
        """
        ...

    def list_for_user(
        self,
        *,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        """
        List owner strategies in deterministic order.

        Args:
            user_id: Strategy owner identifier.
            include_deleted: Include soft-deleted rows when `True`.
        Returns:
            tuple[Strategy, ...]: Ordered strategy snapshots.
        Assumptions:
            Result ordering is deterministic and explicitly controlled by storage query.
        Raises:
            ValueError: If repository implementation cannot map one of rows.
        Side Effects:
            Reads zero or more rows from storage.
        """
        ...

    def soft_delete(self, *, user_id: UserId, strategy_id: UUID) -> bool:
        """
        Mark strategy as soft-deleted (`is_deleted=True`) as the only mutable operation.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            bool: `True` when row changed, otherwise `False`.
        Assumptions:
            Strategy spec remains immutable and untouched.
        Raises:
            ValueError: If repository implementation fails to execute update.
        Side Effects:
            Executes one storage update for `is_deleted` field only.
        """
        ...
