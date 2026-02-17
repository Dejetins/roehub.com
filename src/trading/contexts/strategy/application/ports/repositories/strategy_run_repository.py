from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import StrategyRun
from trading.shared_kernel.primitives import UserId


class StrategyRunRepository(Protocol):
    """
    StrategyRunRepository — storage port for Strategy v1 run lifecycle snapshots.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def create(self, *, run: StrategyRun) -> StrategyRun:
        """
        Persist new run snapshot.

        Args:
            run: Run snapshot in initial or predefined state.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Storage enforces single active run invariant per strategy.
        Raises:
            ValueError: If repository fails to persist run.
        Side Effects:
            Writes one run row.
        """
        ...

    def update(self, *, run: StrategyRun) -> StrategyRun:
        """
        Persist run state transition snapshot.

        Args:
            run: Updated immutable run snapshot.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Transition validity is pre-validated by domain.
        Raises:
            ValueError: If row does not exist or update fails.
        Side Effects:
            Updates one run row.
        """
        ...

    def find_by_run_id(self, *, user_id: UserId, run_id: UUID) -> StrategyRun | None:
        """
        Load run by owner and run identifier.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            StrategyRun | None: Stored run snapshot or `None`.
        Assumptions:
            Lookup is deterministic and owner-scoped.
        Raises:
            ValueError: If repository cannot map row.
        Side Effects:
            Reads one run row.
        """
        ...

    def find_active_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> StrategyRun | None:
        """
        Load active run for strategy, if any.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            StrategyRun | None: Active run snapshot or `None`.
        Assumptions:
            At most one active run exists by v1 invariant.
        Raises:
            ValueError: If repository cannot map row.
        Side Effects:
            Reads one run row.
        """
        ...

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyRun, ...]:
        """
        List all runs for strategy in deterministic order.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyRun, ...]: Ordered run snapshots.
        Assumptions:
            Ordering is deterministic and controlled by explicit SQL `ORDER BY`.
        Raises:
            ValueError: If repository cannot map one of rows.
        Side Effects:
            Reads zero or more run rows.
        """
        ...

    def list_active_runs(self) -> tuple[StrategyRun, ...]:
        """
        List all active runs across users/strategies in deterministic order.

        Args:
            None.
        Returns:
            tuple[StrategyRun, ...]: Active run snapshots ordered by `(started_at, run_id)`.
        Assumptions:
            Active states are fixed to `starting|warming_up|running|stopping` in Strategy v1.
        Raises:
            ValueError: If repository cannot map one of rows.
        Side Effects:
            Reads zero or more run rows.
        """
        ...
