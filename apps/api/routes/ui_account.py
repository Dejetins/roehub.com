from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_account import (
    AccountAuditEventResponse,
    AccountAuditEventsResponse,
    AccountIntegrationResponse,
    AccountIntegrationsResponse,
    AccountLimitsResponse,
    AccountNotificationsResponse,
    AccountPreferencesResponse,
    AccountProfileResponse,
    AccountSessionResponse,
    AccountSessionsResponse,
    UpdateAccountIntegrationsRequest,
    UpdateAccountNotificationsRequest,
    UpdateAccountPreferencesRequest,
    UpdateAccountProfileRequest,
)
from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountAuditEvent,
    AccountIntegration,
    AccountPreferences,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import (
    AccountAuditEventsPage,
    AccountLimitsView,
    AccountProfileView,
    AccountSessionsPage,
    AccountSessionView,
    AccountSettingsOperationError,
    AccountSettingsUseCase,
    ExchangeKeysOperationError,
    ListExchangeKeysUseCase,
)
from trading.platform.errors import RoehubError

_MUTATION_METHODS = {"DELETE", "PATCH", "POST", "PUT"}


def build_ui_account_router(
    *,
    account_settings_use_case: AccountSettingsUseCase,
    list_exchange_keys_use_case: ListExchangeKeysUseCase,
    current_user_dependency: RequireCurrentUserDependency,
    allowed_ui_origins: tuple[str, ...],
) -> APIRouter:
    if account_settings_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires account_settings_use_case")
    if list_exchange_keys_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires list_exchange_keys_use_case")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires current_user_dependency")

    mutation_guard = _build_mutation_guard(allowed_ui_origins=allowed_ui_origins)
    router = APIRouter(prefix="/ui/account", tags=["ui-account"])

    def require_account_user(request: Request) -> CurrentUserPrincipal:
        try:
            return current_user_dependency(request)
        except HTTPException as error:
            if error.status_code == 401:
                raise RoehubError(
                    code="auth.required",
                    message="Authentication is required",
                    details={},
                ) from error
            raise

    @router.get("/profile", response_model=AccountProfileResponse)
    def get_profile(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountProfileResponse:
        view = account_settings_use_case.get_profile(
            owner_user_id=principal.user_id,
            paid_level=principal.paid_level,
        )
        return _to_profile_response(view=view)

    @router.put(
        "/profile",
        response_model=AccountProfileResponse,
        dependencies=[Depends(mutation_guard)],
    )
    def put_profile(
        request: UpdateAccountProfileRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountProfileResponse:
        try:
            view = account_settings_use_case.update_profile(
                owner_user_id=principal.user_id,
                paid_level=principal.paid_level,
                display_name=request.display_name,
                timezone_name=request.timezone,
            )
        except AccountSettingsOperationError as error:
            raise _to_roehub_error(error=error) from error
        return _to_profile_response(view=view)

    @router.get("/limits", response_model=AccountLimitsResponse)
    def get_limits(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountLimitsResponse:
        try:
            exchange_keys_used = len(
                list_exchange_keys_use_case.list_for_user(user_id=principal.user_id)
            )
        except ExchangeKeysOperationError as error:
            raise RoehubError(
                code="unexpected_error",
                message=error.message,
                details={"source": "exchange_keys"},
            ) from error
        view = account_settings_use_case.get_limits(
            paid_level=principal.paid_level,
            exchange_keys_used=exchange_keys_used,
        )
        return _to_limits_response(view=view)

    @router.get("/integrations", response_model=AccountIntegrationsResponse)
    def get_integrations(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountIntegrationsResponse:
        items = account_settings_use_case.list_integrations(
            owner_user_id=principal.user_id,
        )
        return _to_integrations_response(items=items)

    @router.put(
        "/integrations",
        response_model=AccountIntegrationsResponse,
        dependencies=[Depends(mutation_guard)],
    )
    def put_integrations(
        request: UpdateAccountIntegrationsRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountIntegrationsResponse:
        try:
            items = account_settings_use_case.update_integrations(
                owner_user_id=principal.user_id,
                integrations=tuple(
                    (item.provider, item.enabled) for item in request.integrations
                ),
            )
        except AccountSettingsOperationError as error:
            raise _to_roehub_error(error=error) from error
        return _to_integrations_response(items=items)

    @router.get("/notifications", response_model=AccountNotificationsResponse)
    def get_notifications(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountNotificationsResponse:
        preferences = account_settings_use_case.get_preferences(
            owner_user_id=principal.user_id,
        )
        return _to_notifications_response(preferences=preferences)

    @router.put(
        "/notifications",
        response_model=AccountNotificationsResponse,
        dependencies=[Depends(mutation_guard)],
    )
    def put_notifications(
        request: UpdateAccountNotificationsRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountNotificationsResponse:
        preferences = account_settings_use_case.update_notifications(
            owner_user_id=principal.user_id,
            email_notifications_enabled=request.email_notifications_enabled,
            trade_alerts_enabled=request.trade_alerts_enabled,
            product_updates_enabled=request.product_updates_enabled,
        )
        return _to_notifications_response(preferences=preferences)

    @router.get("/preferences", response_model=AccountPreferencesResponse)
    def get_preferences(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountPreferencesResponse:
        preferences = account_settings_use_case.get_preferences(
            owner_user_id=principal.user_id,
        )
        return _to_preferences_response(preferences=preferences)

    @router.put(
        "/preferences",
        response_model=AccountPreferencesResponse,
        dependencies=[Depends(mutation_guard)],
    )
    def put_preferences(
        request: UpdateAccountPreferencesRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountPreferencesResponse:
        updates = {
            field_name: field_value
            for field_name, field_value in request.model_dump(exclude_unset=True).items()
            if field_value is not None
        }
        try:
            preferences = account_settings_use_case.update_preferences(
                owner_user_id=principal.user_id,
                updates=updates,
            )
        except AccountSettingsOperationError as error:
            raise _to_roehub_error(error=error) from error
        return _to_preferences_response(preferences=preferences)

    @router.get("/sessions", response_model=AccountSessionsResponse)
    def get_sessions(
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountSessionsResponse:
        try:
            page = account_settings_use_case.list_sessions(
                owner_user_id=principal.user_id,
                cursor=cursor,
                limit=limit,
            )
        except AccountSettingsOperationError as error:
            raise _to_roehub_error(error=error) from error
        return _to_sessions_response(page=page)

    @router.get("/audit-events", response_model=AccountAuditEventsResponse)
    def get_audit_events(
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountAuditEventsResponse:
        try:
            page = account_settings_use_case.list_audit_events(
                owner_user_id=principal.user_id,
                cursor=cursor,
                limit=limit,
            )
        except AccountSettingsOperationError as error:
            raise _to_roehub_error(error=error) from error
        return _to_audit_response(page=page)

    return router


def _build_mutation_guard(
    *,
    allowed_ui_origins: tuple[str, ...],
) -> Callable[[Request], None]:
    allowed = {
        _normalize_origin(origin)
        for origin in allowed_ui_origins
        if _normalize_origin(origin) is not None
    }

    def require_same_origin_mutation(request: Request) -> None:
        if request.method.upper() not in _MUTATION_METHODS:
            return
        origin = _normalize_origin(request.headers.get("origin"))
        if origin is not None and origin in allowed:
            return
        referer = _normalize_origin(request.headers.get("referer"))
        if referer is not None and referer in allowed:
            return
        raise RoehubError(
            code="forbidden",
            message="Same-origin mutation guard failed",
            details={"reason": "csrf_origin"},
        )

    return require_same_origin_mutation


def _normalize_origin(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    parsed = urlparse(raw_value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _to_roehub_error(*, error: AccountSettingsOperationError) -> RoehubError:
    return RoehubError(
        code=error.code,
        message=error.message,
        details={"field_errors": error.field_errors} if error.field_errors else {},
    )


def _to_profile_response(*, view: AccountProfileView) -> AccountProfileResponse:
    return AccountProfileResponse(
        user_id=str(view.owner_user_id),
        paid_level=str(view.paid_level),
        display_name=view.display_name,
        timezone=view.timezone,
        updated_at=view.updated_at,
    )


def _to_limits_response(*, view: AccountLimitsView) -> AccountLimitsResponse:
    return AccountLimitsResponse(
        paid_level=str(view.paid_level),
        exchange_keys_used=view.exchange_keys_used,
        exchange_keys_limit=view.exchange_keys_limit,
        active_strategies_used=view.active_strategies_used,
        active_strategies_limit=view.active_strategies_limit,
        webhook_events_used=view.webhook_events_used,
        webhook_events_limit=view.webhook_events_limit,
    )


def _to_integrations_response(
    *,
    items: tuple[AccountIntegration, ...],
) -> AccountIntegrationsResponse:
    return AccountIntegrationsResponse(
        integrations=[
            AccountIntegrationResponse(
                provider=item.provider,  # type: ignore[arg-type]
                enabled=item.enabled,
                updated_at=item.updated_at,
            )
            for item in items
        ]
    )


def _to_notifications_response(
    *,
    preferences: AccountPreferences,
) -> AccountNotificationsResponse:
    return AccountNotificationsResponse(
        email_notifications_enabled=preferences.email_notifications_enabled,
        trade_alerts_enabled=preferences.trade_alerts_enabled,
        product_updates_enabled=preferences.product_updates_enabled,
        updated_at=preferences.updated_at,
    )


def _to_preferences_response(
    *,
    preferences: AccountPreferences,
) -> AccountPreferencesResponse:
    return AccountPreferencesResponse(
        theme=preferences.theme,  # type: ignore[arg-type]
        locale=preferences.locale,  # type: ignore[arg-type]
        density=preferences.density,  # type: ignore[arg-type]
        updated_at=preferences.updated_at,
    )


def _to_sessions_response(*, page: AccountSessionsPage) -> AccountSessionsResponse:
    return AccountSessionsResponse(
        items=[_to_session_response(item=item) for item in page.items],
        next_cursor=page.next_cursor,
    )


def _to_session_response(*, item: AccountSessionView) -> AccountSessionResponse:
    return AccountSessionResponse(
        session_id=str(item.session_id),
        created_at=item.created_at,
        last_seen_at=item.last_seen_at,
        idle_expires_at=item.idle_expires_at,
        absolute_expires_at=item.absolute_expires_at,
        revoked_at=item.revoked_at,
        status=item.status,  # type: ignore[arg-type]
    )


def _to_audit_response(*, page: AccountAuditEventsPage) -> AccountAuditEventsResponse:
    return AccountAuditEventsResponse(
        items=[_to_audit_event_response(item=item) for item in page.items],
        next_cursor=page.next_cursor,
    )


def _to_audit_event_response(*, item: AccountAuditEvent) -> AccountAuditEventResponse:
    return AccountAuditEventResponse(
        event_id=str(item.event_id),
        event_type=item.event_type,
        metadata=dict(item.metadata),
        created_at=item.created_at,
    )
