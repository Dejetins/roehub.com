from __future__ import annotations

from collections.abc import Iterable

from trading.contexts.strategy.domain.entities.strategy_run import StrategyRun
from trading.contexts.strategy.domain.errors import StrategyActiveRunConflictError


def ensure_single_active_run(
    *,
    existing_runs: Iterable[StrategyRun],
    candidate_run: StrategyRun,
) -> None:
    """
    Enforce v1 invariant: one active run per strategy at any given time.

    Args:
        existing_runs: Existing run snapshots for one strategy.
        candidate_run: Candidate run snapshot to validate.
    Returns:
        None.
    Assumptions:
        Caller provides runs for the same `strategy_id` scope.
    Raises:
        StrategyActiveRunConflictError: If candidate introduces second active run.
    Side Effects:
        None.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_run_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """
    if not candidate_run.is_active():
        return

    for existing_run in existing_runs:
        if existing_run.run_id == candidate_run.run_id:
            continue
        if existing_run.is_active():
            raise StrategyActiveRunConflictError(
                "Strategy v1 allows exactly one active run per strategy"
            )
