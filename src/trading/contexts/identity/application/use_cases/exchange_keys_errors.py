from __future__ import annotations


class ExchangeKeysOperationError(ValueError):
    """
    ExchangeKeysOperationError — deterministic application error for exchange keys flows.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/application/use_cases/delete_exchange_key.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        """
        Initialize deterministic operation error fields for HTTP mapping.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable deterministic message.
            status_code: HTTP status expected by inbound adapters.
        Returns:
            None.
        Assumptions:
            Error mapping is one-to-one with API responses.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def payload(self) -> dict[str, str]:
        """
        Build deterministic payload with stable key order.

        Args:
            None.
        Returns:
            dict[str, str]: `{"error": "...", "message": "..."}` payload.
        Assumptions:
            Payload is used as HTTPException detail.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "error": self.code,
            "message": self.message,
        }


class ExchangeKeyValidationError(ExchangeKeysOperationError):
    """
    ExchangeKeyValidationError — deterministic 422 for invalid exchange key payload.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
    """

    def __init__(self, *, message: str = "Exchange key payload is invalid.") -> None:
        """
        Initialize deterministic 422 validation error.

        Args:
            message: Stable non-secret validation message.
        Returns:
            None.
        Assumptions:
            Message never contains secret values.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="invalid_exchange_key_payload",
            message=message,
            status_code=422,
        )


class ExchangeKeyAlreadyExistsError(ExchangeKeysOperationError):
    """
    ExchangeKeyAlreadyExistsError — deterministic 409 for active duplicate key.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
    """

    def __init__(self) -> None:
        """
        Initialize deterministic duplicate-key conflict error.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Active key uniqueness is enforced by repository invariants.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="exchange_key_already_exists",
            message="Exchange API key already exists.",
            status_code=409,
        )


class ExchangeKeyNotFoundError(ExchangeKeysOperationError):
    """
    ExchangeKeyNotFoundError — deterministic 404 for missing/deleted/not-owned key.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/delete_exchange_key.py
      - src/trading/contexts/identity/application/ports/exchange_keys_repository.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic not-found error for delete operation.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Missing, non-owned, and already-deleted states are intentionally collapsed.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="exchange_key_not_found",
            message="Exchange API key was not found.",
            status_code=404,
        )
