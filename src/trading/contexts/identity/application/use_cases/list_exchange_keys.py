from __future__ import annotations

from trading.contexts.identity.application.ports import ExchangeKeysRepository
from trading.contexts.identity.application.use_cases.exchange_keys_models import (
    ExchangeKeyView,
    to_exchange_key_view,
)
from trading.shared_kernel.primitives import UserId


class ListExchangeKeysUseCase:
    """
    ListExchangeKeysUseCase — return active exchange keys for authenticated user.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - src/trading/contexts/identity/application/use_cases/exchange_keys_models.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    def __init__(self, *, repository: ExchangeKeysRepository) -> None:
        """
        Initialize use-case with exchange keys repository dependency.

        Args:
            repository: Exchange keys storage port.
        Returns:
            None.
        Assumptions:
            Repository returns active rows for given user.
        Raises:
            ValueError: If repository dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ListExchangeKeysUseCase requires repository")
        self._repository = repository

    def list_for_user(self, *, user_id: UserId) -> tuple[ExchangeKeyView, ...]:
        """
        Fetch active keys and return deterministic API-safe projection list.

        Args:
            user_id: Owner identity user id.
        Returns:
            tuple[ExchangeKeyView, ...]: Deterministically sorted non-secret key projections.
        Assumptions:
            Soft-deleted rows are excluded by repository implementation.
        Raises:
            ValueError: If repository returns invalid domain rows.
        Side Effects:
            Reads storage records.
        """
        rows = self._repository.list_active_for_user(user_id=user_id)
        sorted_rows = tuple(
            sorted(
                rows,
                key=lambda item: (item.created_at, str(item.key_id)),
            )
        )
        return tuple(to_exchange_key_view(entity=item) for item in sorted_rows)
