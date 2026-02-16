from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import StrategyRepository
from trading.contexts.strategy.domain.entities import Strategy
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class InMemoryStrategyRepository(StrategyRepository):
    """
    InMemoryStrategyRepository — deterministic in-memory StrategyRepository adapter for dev/tests.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - apps/api/wiring/modules/strategy.py
      - tests/unit/contexts/strategy/application
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory storage for immutable strategies.

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
        self._strategies_by_id: dict[UUID, Strategy] = {}

    def create(self, *, strategy: Strategy) -> Strategy:
        """
        Persist new immutable strategy snapshot in memory.

        Args:
            strategy: Strategy aggregate to store.
        Returns:
            Strategy: Persisted strategy snapshot.
        Assumptions:
            Strategy identifier is unique per adapter instance.
        Raises:
            StrategyStorageError: If duplicate strategy id already exists.
        Side Effects:
            Writes strategy snapshot to in-memory dictionary.
        """
        if strategy.strategy_id in self._strategies_by_id:
            raise StrategyStorageError("InMemoryStrategyRepository duplicate strategy_id")
        self._strategies_by_id[strategy.strategy_id] = strategy
        return strategy

    def find_by_strategy_id(self, *, user_id: UserId, strategy_id: UUID) -> Strategy | None:
        """
        Load strategy snapshot by owner and identifier.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            Strategy | None: Matching owned strategy or `None`.
        Assumptions:
            Owner checks are exact `UserId` value-object equality.
        Raises:
            None.
        Side Effects:
            None.
        """
        strategy = self._strategies_by_id.get(strategy_id)
        if strategy is None:
            return None
        if strategy.user_id != user_id:
            return None
        return strategy

    def find_any_by_strategy_id(self, *, strategy_id: UUID) -> Strategy | None:
        """
        Load strategy snapshot by identifier without owner filtering.

        Args:
            strategy_id: Strategy identifier.
        Returns:
            Strategy | None: Matching strategy or `None`.
        Assumptions:
            Use-case layer performs explicit ownership checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._strategies_by_id.get(strategy_id)

    def list_for_user(
        self,
        *,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        """
        List owner strategies in deterministic ordering by created_at and strategy_id.

        Args:
            user_id: Strategy owner identifier.
            include_deleted: Include soft-deleted strategies when `True`.
        Returns:
            tuple[Strategy, ...]: Deterministically ordered strategy snapshots.
        Assumptions:
            Ordering matches SQL adapter deterministic contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        filtered = [
            strategy
            for strategy in self._strategies_by_id.values()
            if strategy.user_id == user_id and (include_deleted or not strategy.is_deleted)
        ]
        return tuple(
            sorted(
                filtered,
                key=lambda item: (item.created_at, str(item.strategy_id)),
            )
        )

    def soft_delete(self, *, user_id: UserId, strategy_id: UUID) -> bool:
        """
        Mark strategy snapshot as deleted (`is_deleted=True`) when owner matches.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            bool: `True` when strategy transitioned to deleted state, else `False`.
        Assumptions:
            Immutable strategy snapshot replacement preserves deterministic fields.
        Raises:
            StrategyStorageError: If strategy replacement violates domain invariants.
        Side Effects:
            Updates strategy snapshot entry in in-memory dictionary.
        """
        strategy = self._strategies_by_id.get(strategy_id)
        if strategy is None:
            return False
        if strategy.user_id != user_id:
            return False
        if strategy.is_deleted:
            return False

        self._strategies_by_id[strategy_id] = strategy.soft_deleted()
        return True
