from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountSettingsRepository,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases.create_exchange_key import (
    CreateExchangeKeyUseCase,
)
from trading.contexts.identity.application.use_cases.delete_exchange_key import (
    DeleteExchangeKeyUseCase,
)
from trading.contexts.identity.application.use_cases.exchange_keys_errors import (
    ExchangeKeysOperationError,
)
from trading.contexts.identity.application.use_cases.exchange_keys_models import ExchangeKeyView
from trading.contexts.identity.application.use_cases.list_exchange_keys import (
    ListExchangeKeysUseCase,
)

_RECENT_AUTH_WINDOW = timedelta(minutes=10)


class CreateExchangeKeyRequest(BaseModel):
    """
    CreateExchangeKeyRequest — API payload for `POST /exchange-keys`.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - apps/api/routes/identity.py
    """

    exchange_name: Literal["binance", "bybit"]
    market_type: Literal["spot", "futures"]
    label: str | None = None
    permissions: Literal["read", "trade"]
    api_key: str
    api_secret: str
    passphrase: str | None = None


class ExchangeKeyResponse(BaseModel):
    """
    ExchangeKeyResponse — API-safe exchange key response without secret fields.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/exchange_keys_models.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      - apps/api/routes/identity.py
    """

    key_id: str
    exchange_name: Literal["binance", "bybit"]
    market_type: Literal["spot", "futures"]
    label: str | None
    permissions: Literal["read", "trade"]
    api_key: str
    created_at: datetime
    updated_at: datetime


def build_exchange_keys_router(
    *,
    create_use_case: CreateExchangeKeyUseCase,
    list_use_case: ListExchangeKeysUseCase,
    delete_use_case: DeleteExchangeKeyUseCase,
    current_user_dependency: RequireCurrentUserDependency,
    audit_events_repository: AccountSettingsRepository,
    clock: IdentityClock,
) -> APIRouter:
    """
    Build router exposing exchange keys create/list/delete endpoints for authenticated users.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - apps/api/routes/identity.py

    Args:
        create_use_case: Exchange key create use-case.
        list_use_case: Exchange key list use-case.
        delete_use_case: Exchange key delete use-case.
        current_user_dependency: Current-user auth dependency.
    Returns:
        APIRouter: Configured exchange keys router.
    Assumptions:
        Auth dependency resolves authenticated principal or raises deterministic 401 payload.
    Raises:
        ValueError: If required dependencies are missing.
    Side Effects:
        None.
    """
    if create_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires create_use_case")
    if list_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires list_use_case")
    if delete_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires delete_use_case")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires current_user_dependency")
    if audit_events_repository is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires audit_events_repository")
    if clock is None:  # type: ignore[truthy-bool]
        raise ValueError("build_exchange_keys_router requires clock")

    router = APIRouter(tags=["identity"])

    @router.post("/exchange-keys", response_model=ExchangeKeyResponse, status_code=201)
    def post_exchange_key(
        http_request: Request,
        request: CreateExchangeKeyRequest,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ExchangeKeyResponse:
        """
        Create one exchange key for current user with encrypted secret storage.

        Args:
            request: Exchange key create payload.
            principal: Authenticated principal.
        Returns:
            ExchangeKeyResponse: API-safe created key projection.
        Assumptions:
            Secret fields are encrypted before persistence and never returned in response.
        Raises:
            HTTPException: Deterministic 4xx payload for validation/conflict conditions.
        Side Effects:
            Writes one exchange key row in storage.
        """
        _enforce_exchange_mutation_csrf(request=http_request)
        now = clock.now()
        _enforce_recent_auth(principal=principal, now=now)
        try:
            result = create_use_case.create(
                user_id=principal.user_id,
                exchange_name=request.exchange_name,
                market_type=request.market_type,
                label=request.label,
                permissions=request.permissions,
                api_key=request.api_key,
                api_secret=request.api_secret,
                passphrase=request.passphrase,
            )
        except ExchangeKeysOperationError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.payload(),
            ) from error
        audit_events_repository.append_audit_event(
            owner_user_id=principal.user_id,
            event_type="exchange_key_created",
            summary="Exchange API key created",
            metadata={
                "surface": "api",
                "key_id": str(result.key_id),
                "exchange_name": result.exchange_name,
                "market_type": result.market_type,
                "permissions": result.permissions,
            },
            created_at=now,
        )
        return _to_exchange_key_response(view=result)

    @router.get("/exchange-keys", response_model=list[ExchangeKeyResponse])
    def get_exchange_keys(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> list[ExchangeKeyResponse]:
        """
        List active exchange keys for current user in deterministic order.

        Args:
            principal: Authenticated principal.
        Returns:
            list[ExchangeKeyResponse]: API-safe key projections without secret fields.
        Assumptions:
            Result ordering is deterministic (`created_at ASC, key_id ASC`).
        Raises:
            HTTPException: Deterministic 4xx payload for operation errors.
        Side Effects:
            Reads exchange key rows from storage.
        """
        try:
            result = list_use_case.list_for_user(user_id=principal.user_id)
        except ExchangeKeysOperationError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.payload(),
            ) from error
        return [_to_exchange_key_response(view=item) for item in result]

    @router.delete("/exchange-keys/{key_id}", status_code=204, response_model=None)
    def delete_exchange_key(
        request: Request,
        key_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> Response:
        """
        Soft-delete one active exchange key owned by current user.

        Args:
            key_id: Exchange key identifier path parameter.
            principal: Authenticated principal.
        Returns:
            Response: Empty 204 response when key is deleted.
        Assumptions:
            Delete operation performs soft-delete (`is_deleted=true`, `deleted_at=now`).
        Raises:
            HTTPException: Deterministic 4xx payload for missing key and other errors.
        Side Effects:
            Updates one exchange key row in storage.
        """
        _enforce_exchange_mutation_csrf(request=request)
        now = clock.now()
        _enforce_recent_auth(principal=principal, now=now)
        try:
            delete_use_case.delete(user_id=principal.user_id, key_id=key_id)
        except ExchangeKeysOperationError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.payload(),
            ) from error
        audit_events_repository.append_audit_event(
            owner_user_id=principal.user_id,
            event_type="exchange_key_deleted",
            summary="Exchange API key deleted",
            metadata={
                "surface": "api",
                "key_id": str(key_id),
            },
            created_at=now,
        )
        return Response(status_code=204)

    return router


def _enforce_exchange_mutation_csrf(*, request: Request) -> None:
    rejection_reason = same_origin_rejection_reason(
        request=request,
        fail_closed_without_origin=True,
    )
    if rejection_reason is None:
        return
    raise HTTPException(
        status_code=403,
        detail={
            "error": "csrf_required",
            "message": "CSRF protection is required for exchange credential mutations.",
            "reason": rejection_reason,
        },
    )


def _enforce_recent_auth(*, principal: CurrentUserPrincipal, now: datetime) -> None:
    if principal.session_created_at is None:
        raise _recent_auth_required()
    if principal.session_created_at + _RECENT_AUTH_WINDOW < now:
        raise _recent_auth_required()


def _recent_auth_required() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "error": "recent_auth_required",
            "message": "Recent Keycloak authentication is required.",
        },
    )



def _to_exchange_key_response(*, view: ExchangeKeyView) -> ExchangeKeyResponse:
    """
    Convert application-layer projection into strict API response DTO.

    Args:
        view: Application non-secret exchange key projection.
    Returns:
        ExchangeKeyResponse: Strict response model value.
    Assumptions:
        `view.api_key` is already masked and safe for response.
    Raises:
        ValueError: If enum literals are outside API contract.
    Side Effects:
        None.
    """
    return ExchangeKeyResponse(
        key_id=str(view.key_id),
        exchange_name=_to_exchange_name_literal(value=view.exchange_name),
        market_type=_to_market_type_literal(value=view.market_type),
        label=view.label,
        permissions=_to_permissions_literal(value=view.permissions),
        api_key=view.api_key,
        created_at=view.created_at,
        updated_at=view.updated_at,
    )



def _to_exchange_name_literal(*, value: str) -> Literal["binance", "bybit"]:
    """
    Convert runtime exchange value into strict API literal.

    Args:
        value: Runtime exchange string.
    Returns:
        Literal["binance", "bybit"]: Strict exchange literal.
    Assumptions:
        Runtime values already normalized by application layer.
    Raises:
        ValueError: If value is outside API contract.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if normalized not in {"binance", "bybit"}:
        raise ValueError(f"Unsupported exchange_name value: {value!r}")
    return normalized  # type: ignore[return-value]



def _to_market_type_literal(*, value: str) -> Literal["spot", "futures"]:
    """
    Convert runtime market type into strict API literal.

    Args:
        value: Runtime market type string.
    Returns:
        Literal["spot", "futures"]: Strict market type literal.
    Assumptions:
        Runtime values already normalized by application layer.
    Raises:
        ValueError: If value is outside API contract.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if normalized not in {"spot", "futures"}:
        raise ValueError(f"Unsupported market_type value: {value!r}")
    return normalized  # type: ignore[return-value]



def _to_permissions_literal(*, value: str) -> Literal["read", "trade"]:
    """
    Convert runtime permissions into strict API literal.

    Args:
        value: Runtime permissions string.
    Returns:
        Literal["read", "trade"]: Strict permissions literal.
    Assumptions:
        Runtime values already normalized by application layer.
    Raises:
        ValueError: If value is outside API contract.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if normalized not in {"read", "trade"}:
        raise ValueError(f"Unsupported permissions value: {value!r}")
    return normalized  # type: ignore[return-value]
