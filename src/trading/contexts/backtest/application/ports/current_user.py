from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """
    Current authenticated user context for ownership-aware backtest saved mode.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/shared_kernel/primitives/user_id.py
    """

    user_id: UserId

    def __post_init__(self) -> None:
        """
        Validate required ownership principal payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Backtest saved-mode checks require non-null `UserId`.
        Raises:
            ValueError: If `user_id` is missing.
        Side Effects:
            None.
        """
        if self.user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("CurrentUser.user_id is required")

