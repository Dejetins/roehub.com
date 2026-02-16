from __future__ import annotations


class StrategyDomainError(ValueError):
    """
    Base deterministic domain error for Strategy v1 bounded context.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """


class StrategySpecValidationError(StrategyDomainError):
    """
    Raised when StrategySpecV1 payload violates immutable schema-versioned contract.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - src/trading/contexts/strategy/domain/services/strategy_name.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """


class StrategyRunTransitionError(StrategyDomainError):
    """
    Raised when StrategyRun state transition is invalid for v1 state machine.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_run_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """


class StrategyActiveRunConflictError(StrategyDomainError):
    """
    Raised when repository/domain detects a second active run for one strategy.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/services/run_invariants.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """


class StrategyStorageError(StrategyDomainError):
    """
    Raised when storage adapter cannot map deterministic strategy/runs/events row payloads.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_event_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """
