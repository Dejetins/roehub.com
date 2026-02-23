from __future__ import annotations

from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.backtest.domain.errors import (
    BacktestConflictError,
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestStorageError,
    BacktestValidationError,
)
from trading.contexts.indicators.domain.errors import (
    ComputeBudgetExceeded,
    GridValidationError,
    MissingInputSeriesError,
    MissingRequiredSeries,
    UnknownIndicatorError,
)
from trading.platform.errors import RoehubError


def validation_error(
    *,
    message: str,
    errors: Sequence[Mapping[str, str]] | None = None,
) -> RoehubError:
    """
    Build canonical `validation_error` RoehubError with deterministic item ordering.

    Docs:
      - docs/architecture/api/api-errors-and-422-payload-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/domain/errors/backtest_errors.py
      - apps/api/common/errors.py
      - src/trading/platform/errors/roehub_error.py

    Args:
        message: Human-readable validation failure message.
        errors: Optional validation items list.
    Returns:
        RoehubError: Canonical deterministic validation error.
    Assumptions:
        Validation item entries contain `path`, `code`, and `message`.
    Raises:
        None.
    Side Effects:
        None.
    """
    details: dict[str, Any] = {}
    if errors is not None:
        details["errors"] = _sorted_validation_items(items=errors)
    return RoehubError(
        code="validation_error",
        message=message,
        details=details,
    )


def backtest_not_found(*, strategy_id: UUID) -> RoehubError:
    """
    Build deterministic not-found API error for saved strategy lookup failures.

    Args:
        strategy_id: Requested saved strategy identifier.
    Returns:
        RoehubError: Canonical `not_found` payload.
    Assumptions:
        Missing and soft-deleted strategies are intentionally not distinguished.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="not_found",
        message="Backtest strategy was not found",
        details={
            "strategy_id": str(strategy_id),
        },
    )


def backtest_forbidden(*, strategy_id: UUID) -> RoehubError:
    """
    Build deterministic forbidden API error for non-owner saved strategy access.

    Args:
        strategy_id: Saved strategy identifier requested by non-owner.
    Returns:
        RoehubError: Canonical `forbidden` payload.
    Assumptions:
        Ownership checks happen in backtest use-case layer.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="forbidden",
        message="Backtest strategy does not belong to current user",
        details={
            "strategy_id": str(strategy_id),
        },
    )


def backtest_job_not_found(*, job_id: UUID) -> RoehubError:
    """
    Build deterministic not-found API error for Backtest job id lookup failures.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/common/errors.py

    Args:
        job_id: Requested Backtest job identifier.
    Returns:
        RoehubError: Canonical `not_found` payload.
    Assumptions:
        Missing job id always maps to `404 not_found`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="not_found",
        message="Backtest job was not found",
        details={"job_id": str(job_id)},
    )


def backtest_job_forbidden(*, job_id: UUID) -> RoehubError:
    """
    Build deterministic forbidden API error for foreign Backtest job access attempts.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/common/errors.py

    Args:
        job_id: Existing Backtest job identifier requested by non-owner.
    Returns:
        RoehubError: Canonical `forbidden` payload.
    Assumptions:
        Existing foreign jobs must return `403` (not `404`) by EPIC-11 contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="forbidden",
        message="Backtest job does not belong to current user",
        details={"job_id": str(job_id)},
    )


def backtest_conflict(*, message: str, details: Mapping[str, Any]) -> RoehubError:
    """
    Build deterministic conflict API error for request/use-case state conflicts.

    Args:
        message: Human-readable conflict description.
        details: Structured conflict details mapping.
    Returns:
        RoehubError: Canonical `conflict` payload.
    Assumptions:
        Conflict details are JSON-compatible or safely stringified by RoehubError.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="conflict",
        message=message,
        details=dict(details),
    )


def map_backtest_exception(*, error: Exception) -> RoehubError:
    """
    Map known backtest/indicator exceptions to canonical RoehubError contract.

    Docs:
      - docs/architecture/api/api-errors-and-422-payload-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/errors/backtest_errors.py
      - src/trading/contexts/indicators/domain/errors/__init__.py

    Args:
        error: Caught exception.
    Returns:
        RoehubError: Canonical mapped error.
    Assumptions:
        Unknown exceptions are mapped to generic `unexpected_error`.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(error, RoehubError):
        return error

    if isinstance(error, BacktestValidationError):
        normalized_errors = error.errors if len(error.errors) > 0 else None
        return validation_error(message=str(error), errors=normalized_errors)

    if isinstance(error, (GridValidationError, ComputeBudgetExceeded)):
        return validation_error(message=str(error))

    if isinstance(error, (MissingInputSeriesError, MissingRequiredSeries, UnknownIndicatorError)):
        return validation_error(message=str(error))

    if isinstance(error, BacktestNotFoundError):
        return backtest_not_found(strategy_id=error.strategy_id)

    if isinstance(error, BacktestForbiddenError):
        return backtest_forbidden(strategy_id=error.strategy_id)

    if isinstance(error, BacktestConflictError):
        return backtest_conflict(message=str(error), details=error.details)

    if isinstance(error, BacktestStorageError):
        return RoehubError(
            code="unexpected_error",
            message="Backtest storage operation failed",
            details={"reason": str(error)},
        )

    if isinstance(error, ValueError):
        return validation_error(message=str(error))

    return RoehubError(
        code="unexpected_error",
        message="Unexpected backtest operation error",
        details={"reason": str(error)},
    )


def _sorted_validation_items(*, items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """
    Normalize and deterministically sort validation item list by path/code/message.

    Args:
        items: Validation item sequence.
    Returns:
        list[dict[str, str]]: Deterministically sorted normalized list.
    Assumptions:
        Missing fields are replaced with deterministic fallback literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    normalized_items: list[dict[str, str]] = []
    for item in items:
        normalized_items.append(
            {
                "path": str(item.get("path", "unknown")),
                "code": str(item.get("code", "validation_error")),
                "message": str(item.get("message", "Validation error")),
            }
        )

    return sorted(
        normalized_items,
        key=lambda row: (row["path"], row["code"], row["message"]),
    )
