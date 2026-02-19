from __future__ import annotations

from uuid import UUID

from trading.contexts.backtest.application.use_cases import map_backtest_exception
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestValidationError,
)


def test_map_backtest_exception_sorts_validation_items_deterministically() -> None:
    """
    Verify validation error mapping emits deterministic sorted `details.errors` payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Sorting key order is `(path, code, message)` as in global API error contract.
    Raises:
        AssertionError: If mapped payload order is non-deterministic.
    Side Effects:
        None.
    """
    error = BacktestValidationError(
        "Invalid backtest request",
        errors=(
            {
                "path": "body.template.preselect",
                "code": "out_of_range",
                "message": "Must be <= 20000",
            },
            {
                "path": "body.template.direction_mode",
                "code": "unsupported",
                "message": "Unknown direction mode",
            },
        ),
    )

    mapped = map_backtest_exception(error=error)

    assert mapped.code == "validation_error"
    assert mapped.details == {
        "errors": [
            {
                "path": "body.template.direction_mode",
                "code": "unsupported",
                "message": "Unknown direction mode",
            },
            {
                "path": "body.template.preselect",
                "code": "out_of_range",
                "message": "Must be <= 20000",
            },
        ]
    }


def test_map_backtest_exception_maps_forbidden_to_canonical_payload() -> None:
    """
    Verify forbidden domain error maps to canonical `forbidden` Roehub payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved strategy ownership checks are performed in use-case layer.
    Raises:
        AssertionError: If mapped code/details differ from contract.
    Side Effects:
        None.
    """
    strategy_id = UUID("00000000-0000-0000-0000-000000000303")
    mapped = map_backtest_exception(
        error=BacktestForbiddenError(strategy_id=strategy_id),
    )

    assert mapped.code == "forbidden"
    assert mapped.message == "Backtest strategy does not belong to current user"
    assert mapped.details == {"strategy_id": str(strategy_id)}

