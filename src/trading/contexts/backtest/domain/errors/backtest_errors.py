from __future__ import annotations

from typing import Any, Mapping, Sequence
from uuid import UUID


class BacktestDomainError(ValueError):
    """
    Base deterministic domain error for Backtest v1 bounded context.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/errors.py
      - src/trading/platform/errors/roehub_error.py
      - apps/api/common/errors.py
    """


class BacktestValidationError(BacktestDomainError):
    """
    Raised when backtest request/domain invariants violate v1 contract.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/errors.py
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    """

    def __init__(
        self,
        message: str,
        *,
        errors: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        """
        Build validation error with optional deterministic item payload.

        Args:
            message: Human-readable validation failure description.
            errors: Optional detailed validation items (`path`, `code`, `message`).
        Returns:
            None.
        Assumptions:
            Missing fields are normalized to deterministic fallback values.
        Raises:
            None.
        Side Effects:
            Stores normalized immutable validation items for API mapping layer.
        """
        super().__init__(message)
        normalized_errors: list[dict[str, str]] = []
        if errors is not None:
            for item in errors:
                normalized_errors.append(
                    {
                        "path": str(item.get("path", "unknown")),
                        "code": str(item.get("code", "validation_error")),
                        "message": str(item.get("message", "Validation error")),
                    }
                )
        self._errors = tuple(normalized_errors)

    @property
    def errors(self) -> tuple[Mapping[str, str], ...]:
        """
        Return immutable normalized validation details.

        Args:
            None.
        Returns:
            tuple[Mapping[str, str], ...]: Stable normalized validation details.
        Assumptions:
            Items were normalized during initialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._errors


class BacktestNotFoundError(BacktestDomainError):
    """
    Raised when requested saved strategy snapshot does not exist or is deleted.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/backtest/application/use_cases/errors.py
    """

    def __init__(self, *, strategy_id: UUID) -> None:
        """
        Build deterministic not-found domain error with strategy identity.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            None.
        Assumptions:
            Missing and deleted saved strategies are intentionally not distinguished.
        Raises:
            None.
        Side Effects:
            Stores strategy id for deterministic error-to-API mapping.
        """
        super().__init__("Backtest strategy was not found")
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> UUID:
        """
        Return requested strategy id associated with not-found failure.

        Args:
            None.
        Returns:
            UUID: Strategy identifier.
        Assumptions:
            Value was supplied at construction time.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._strategy_id


class BacktestForbiddenError(BacktestDomainError):
    """
    Raised when saved strategy ownership does not match authenticated user.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/current_user.py
      - src/trading/contexts/backtest/application/use_cases/errors.py
    """

    def __init__(self, *, strategy_id: UUID) -> None:
        """
        Build deterministic forbidden domain error with strategy identity.

        Args:
            strategy_id: Strategy identifier requested by non-owner user.
        Returns:
            None.
        Assumptions:
            Ownership checks are explicit in use-case layer and independent from SQL filters.
        Raises:
            None.
        Side Effects:
            Stores strategy id for deterministic error-to-API mapping.
        """
        super().__init__("Backtest strategy does not belong to current user")
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> UUID:
        """
        Return forbidden strategy id.

        Args:
            None.
        Returns:
            UUID: Forbidden strategy identifier.
        Assumptions:
            Value was supplied at construction time.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._strategy_id


class BacktestConflictError(BacktestDomainError):
    """
    Raised when use-case receives conflicting but syntactically valid payload/state.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/errors.py
      - src/trading/platform/errors/roehub_error.py
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        """
        Build conflict error with deterministic normalized details mapping.

        Args:
            message: Human-readable conflict description.
            details: Optional structured conflict details.
        Returns:
            None.
        Assumptions:
            Details payload is JSON-compatible or can be stringified by upper layer.
        Raises:
            None.
        Side Effects:
            Stores deterministic shallow copy of details for API mapping.
        """
        super().__init__(message)
        self._details = dict(details) if details is not None else {}

    @property
    def details(self) -> Mapping[str, Any]:
        """
        Return deterministic conflict details mapping.

        Args:
            None.
        Returns:
            Mapping[str, Any]: Stored details payload.
        Assumptions:
            Mapping may be empty when no details were provided.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._details


class BacktestStorageError(BacktestDomainError):
    """
    Raised when outbound adapter cannot load deterministic saved-strategy snapshot.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/backtest/application/use_cases/errors.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
    """


class BacktestJobTransitionError(BacktestDomainError):
    """
    Raised when Backtest job lifecycle transition violates state-machine contract.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    """


class BacktestJobLeaseError(BacktestDomainError):
    """
    Raised when Backtest job lease-owner invariants are violated.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_lease_repository.py
    """
