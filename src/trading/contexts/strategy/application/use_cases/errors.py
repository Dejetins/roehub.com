from __future__ import annotations

from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.strategy.domain.errors import (
    StrategyActiveRunConflictError,
    StrategyRunTransitionError,
    StrategySpecValidationError,
    StrategyStorageError,
)
from trading.platform.errors import RoehubError


def validation_error(
    *,
    message: str,
    errors: Sequence[Mapping[str, str]] | None = None,
) -> RoehubError:
    """
    Build canonical `validation_error` RoehubError with deterministic sorted errors list.

    Args:
        message: Human-readable validation failure description.
        errors: Optional validation item list (`path`, `code`, `message`).
    Returns:
        RoehubError: Canonical validation error object.
    Assumptions:
        Each validation item is JSON-serializable and contains string fields.
    Raises:
        None.
    Side Effects:
        None.
    """
    details: dict[str, Any] = {}
    if errors is not None:
        normalized_errors = _sorted_validation_items(items=errors)
        details["errors"] = normalized_errors
    return RoehubError(
        code="validation_error",
        message=message,
        details=details,
    )



def strategy_not_found(*, strategy_id: UUID) -> RoehubError:
    """
    Build deterministic not-found error for missing or soft-deleted strategy.

    Args:
        strategy_id: Requested strategy identifier.
    Returns:
        RoehubError: Not-found error contract.
    Assumptions:
        Missing and deleted strategies are intentionally not distinguished in HTTP status.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="not_found",
        message="Strategy was not found",
        details={
            "strategy_id": str(strategy_id),
        },
    )



def strategy_forbidden(*, strategy_id: UUID) -> RoehubError:
    """
    Build deterministic forbidden error for non-owner strategy access attempts.

    Args:
        strategy_id: Strategy identifier attempted by non-owner.
    Returns:
        RoehubError: Forbidden error contract.
    Assumptions:
        Owner checks are explicit in use-cases and independent from SQL filtering shortcuts.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="forbidden",
        message="Strategy does not belong to current user",
        details={
            "strategy_id": str(strategy_id),
        },
    )



def strategy_conflict(*, message: str, details: Mapping[str, Any]) -> RoehubError:
    """
    Build deterministic conflict error for run-state or concurrent-operation violations.

    Args:
        message: Human-readable conflict description.
        details: Structured conflict details mapping.
    Returns:
        RoehubError: Conflict error contract.
    Assumptions:
        Conflict payload is used for deterministic client behavior and retries.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(code="conflict", message=message, details=dict(details))



def map_strategy_exception(*, error: Exception) -> RoehubError:
    """
    Map known Strategy domain/storage exceptions to canonical RoehubError variants.

    Args:
        error: Caught strategy exception.
    Returns:
        RoehubError: Canonical mapped error object.
    Assumptions:
        Unknown exceptions are mapped to generic `unexpected_error` response contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(error, StrategySpecValidationError):
        return validation_error(message=str(error))
    if isinstance(error, StrategyRunTransitionError):
        return strategy_conflict(
            message="Strategy run transition is invalid",
            details={"reason": str(error)},
        )
    if isinstance(error, StrategyActiveRunConflictError):
        return strategy_conflict(
            message="Strategy already has active run",
            details={"reason": str(error)},
        )
    if isinstance(error, StrategyStorageError):
        return RoehubError(
            code="unexpected_error",
            message="Strategy storage operation failed",
            details={"reason": str(error)},
        )

    return RoehubError(
        code="unexpected_error",
        message="Unexpected strategy operation error",
        details={"reason": str(error)},
    )



def _sorted_validation_items(*, items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """
    Normalize and deterministically sort validation item list.

    Args:
        items: Input validation item sequence.
    Returns:
        list[dict[str, str]]: Sorted normalized list.
    Assumptions:
        Missing fields are replaced with deterministic fallbacks.
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
