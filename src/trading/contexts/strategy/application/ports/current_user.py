from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """
    CurrentUser — authenticated owner context consumed by Strategy use-cases.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/use_cases
      - apps/api/routes/strategies.py
      - apps/api/wiring/modules/strategy.py
    """

    user_id: UserId

    def __post_init__(self) -> None:
        """
        Validate current-user context invariants for Strategy ownership checks.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Strategy owner checks require non-null `UserId` value object.
        Raises:
            ValueError: If `user_id` is missing.
        Side Effects:
            None.
        """
        if self.user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("CurrentUser.user_id is required")


class CurrentUserProvider(Protocol):
    """
    CurrentUserProvider — Strategy port that resolves authenticated CurrentUser context.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - apps/api/wiring/modules/strategy.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    """

    def require_current_user(self) -> CurrentUser:
        """
        Resolve current authenticated user for Strategy ownership-aware operations.

        Args:
            None.
        Returns:
            CurrentUser: Authenticated strategy owner context.
        Assumptions:
            API adapter handles authentication and guarantees deterministic principal payload.
        Raises:
            ValueError: If provider implementation cannot resolve authenticated user.
        Side Effects:
            None.
        """
        ...
