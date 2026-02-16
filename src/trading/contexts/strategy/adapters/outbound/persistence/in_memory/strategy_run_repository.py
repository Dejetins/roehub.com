from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import StrategyRunRepository
from trading.contexts.strategy.domain.entities import StrategyRun
from trading.contexts.strategy.domain.errors import (
    StrategyActiveRunConflictError,
    StrategyStorageError,
)
from trading.shared_kernel.primitives import UserId


class InMemoryStrategyRunRepository(StrategyRunRepository):
    """
    InMemoryStrategyRunRepository — deterministic in-memory StrategyRunRepository adapter.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_run_repository.py
      - apps/api/wiring/modules/strategy.py
      - tests/unit/contexts/strategy/application
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory storage for strategy run snapshots.

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
        self._runs_by_id: dict[UUID, StrategyRun] = {}

    def create(self, *, run: StrategyRun) -> StrategyRun:
        """
        Persist new run snapshot while enforcing one-active-run invariant.

        Args:
            run: Run snapshot to store.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Active-state uniqueness is enforced in this adapter and use-case layer.
        Raises:
            StrategyActiveRunConflictError: If another active run exists for strategy.
            StrategyStorageError: If duplicate run id is inserted.
        Side Effects:
            Writes run snapshot to in-memory dictionary.
        """
        if run.run_id in self._runs_by_id:
            raise StrategyStorageError("InMemoryStrategyRunRepository duplicate run_id")

        existing_active = self.find_active_for_strategy(
            user_id=run.user_id,
            strategy_id=run.strategy_id,
        )
        if run.is_active() and existing_active is not None and existing_active.run_id != run.run_id:
            raise StrategyActiveRunConflictError(
                "Strategy v1 allows exactly one active run per strategy"
            )

        self._runs_by_id[run.run_id] = run
        return run

    def update(self, *, run: StrategyRun) -> StrategyRun:
        """
        Persist run transition snapshot by replacing existing in-memory entry.

        Args:
            run: Updated immutable run snapshot.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Update target run must already exist.
        Raises:
            StrategyStorageError: If run id does not exist.
        Side Effects:
            Replaces run snapshot in in-memory dictionary.
        """
        if run.run_id not in self._runs_by_id:
            raise StrategyStorageError("InMemoryStrategyRunRepository missing run_id")
        self._runs_by_id[run.run_id] = run
        return run

    def find_by_run_id(self, *, user_id: UserId, run_id: UUID) -> StrategyRun | None:
        """
        Load run snapshot by owner and run identifier.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            StrategyRun | None: Matching owned run or `None`.
        Assumptions:
            Owner checks are exact `UserId` value-object equality.
        Raises:
            None.
        Side Effects:
            None.
        """
        run = self._runs_by_id.get(run_id)
        if run is None:
            return None
        if run.user_id != user_id:
            return None
        return run

    def find_active_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> StrategyRun | None:
        """
        Find active run for strategy in deterministic ordering.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            StrategyRun | None: Active run snapshot or `None`.
        Assumptions:
            Deterministic ordering matches SQL adapter (`started_at DESC, run_id DESC`).
        Raises:
            None.
        Side Effects:
            None.
        """
        active_runs = [
            run
            for run in self._runs_by_id.values()
            if run.user_id == user_id and run.strategy_id == strategy_id and run.is_active()
        ]
        if not active_runs:
            return None
        sorted_runs = sorted(
            active_runs,
            key=lambda item: (item.started_at, str(item.run_id)),
            reverse=True,
        )
        return sorted_runs[0]

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyRun, ...]:
        """
        List strategy runs in deterministic ordering by started_at and run_id.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyRun, ...]: Ordered run snapshots.
        Assumptions:
            Ordering matches SQL adapter deterministic contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        filtered = [
            run
            for run in self._runs_by_id.values()
            if run.user_id == user_id and run.strategy_id == strategy_id
        ]
        return tuple(
            sorted(
                filtered,
                key=lambda item: (item.started_at, str(item.run_id)),
            )
        )
