from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_account import (
    AccountAuditEventResponse,
    AccountAuditEventsResponse,
    AccountAutorefreshPreferenceResponse,
    AccountIntegrationResponse,
    AccountIntegrationsResponse,
    AccountLimitsResponse,
    AccountNotificationResponse,
    AccountNotificationsResponse,
    AccountPreferencesResponse,
    AccountProfileResponse,
    AccountSessionResponse,
    AccountSessionsResponse,
    UpdateAccountIntegrationRequest,
    UpdateAccountNotificationRequest,
    UpdateAccountPreferencesRequest,
    UpdateAccountProfileRequest,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountAuditEvent,
    AccountIntegrationSettings,
    AccountNotificationSettings,
    AccountPreferences,
    AccountProfileSettings,
    AccountSessionView,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
    AccountSettingsValidationError,
)
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_ui_account_router(
    *,
    account_settings: AccountSettingsUseCase,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    if account_settings is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires account_settings")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires current_user_dependency")

    router = APIRouter(tags=["ui-account"])

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

    @router.get("/ui/account/profile", response_model=AccountProfileResponse)
    def get_profile(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountProfileResponse:
        profile = account_settings.get_profile(owner_user_id=principal.user_id)
        preferences = account_settings.get_preferences(owner_user_id=principal.user_id)
        return _profile_response(
            profile=profile,
            preferences=preferences,
            subscription_status=str(principal.paid_level),
        )

    @router.put("/ui/account/profile", response_model=AccountProfileResponse)
    def put_profile(
        request: Request,
        payload: UpdateAccountProfileRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountProfileResponse:
        _enforce_same_origin_mutation(request=request)
        try:
            profile = account_settings.update_profile(
                owner_user_id=principal.user_id,
                username=payload.username,
                email=payload.email,
                timezone=payload.timezone,
                telegram_discord=payload.telegram_discord,
            )
        except AccountSettingsValidationError as error:
            raise _validation_error(error=error) from error
        preferences = account_settings.get_preferences(owner_user_id=principal.user_id)
        return _profile_response(
            profile=profile,
            preferences=preferences,
            subscription_status=str(principal.paid_level),
        )

    @router.get("/ui/account/limits", response_model=AccountLimitsResponse)
    def get_limits(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountLimitsResponse:
        _ = principal
        return AccountLimitsResponse(
            plan="pro",
            exchange_connections_used=0,
            exchange_connections_limit=10,
            api_keys_used=0,
            api_keys_limit=10,
            active_strategies_used=0,
            active_strategies_limit=50,
            webhook_events_used=88,
            webhook_events_limit=100,
        )

    @router.get("/ui/account/integrations", response_model=AccountIntegrationsResponse)
    def get_integrations(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountIntegrationsResponse:
        return AccountIntegrationsResponse(
            items=[
                _integration_response(integration=integration)
                for integration in account_settings.list_integrations(
                    owner_user_id=principal.user_id
                )
            ]
        )

    @router.put("/ui/account/integrations", response_model=AccountIntegrationResponse)
    def put_integration(
        request: Request,
        payload: UpdateAccountIntegrationRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountIntegrationResponse:
        _enforce_same_origin_mutation(request=request)
        try:
            integration = account_settings.update_integration(
                owner_user_id=principal.user_id,
                integration_key=payload.integration_key,
                mode=payload.mode,
                webhook_url=payload.webhook_url,
            )
        except AccountSettingsValidationError as error:
            raise _validation_error(error=error) from error
        return _integration_response(integration=integration)

    @router.get("/ui/account/notifications", response_model=AccountNotificationsResponse)
    def get_notifications(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountNotificationsResponse:
        return AccountNotificationsResponse(
            items=[
                _notification_response(notification=notification)
                for notification in account_settings.list_notifications(
                    owner_user_id=principal.user_id
                )
            ]
        )

    @router.put("/ui/account/notifications", response_model=AccountNotificationResponse)
    def put_notification(
        request: Request,
        payload: UpdateAccountNotificationRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountNotificationResponse:
        _enforce_same_origin_mutation(request=request)
        try:
            notification = account_settings.update_notification(
                owner_user_id=principal.user_id,
                channel_key=payload.channel_key,
                mode=payload.mode,
            )
        except AccountSettingsValidationError as error:
            raise _validation_error(error=error) from error
        return _notification_response(notification=notification)

    @router.get("/ui/account/preferences", response_model=AccountPreferencesResponse)
    def get_preferences(
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountPreferencesResponse:
        return _preferences_response(
            preferences=account_settings.get_preferences(owner_user_id=principal.user_id)
        )

    @router.put("/ui/account/preferences", response_model=AccountPreferencesResponse)
    def put_preferences(
        request: Request,
        payload: UpdateAccountPreferencesRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountPreferencesResponse:
        _enforce_same_origin_mutation(request=request)
        try:
            preferences = account_settings.update_preferences(
                owner_user_id=principal.user_id,
                theme=payload.theme,
                locale=payload.locale,
                density=payload.density,
                autorefresh_preset=payload.autorefresh_preset,
                refresh_interval_seconds=payload.refresh_interval_seconds,
            )
        except AccountSettingsValidationError as error:
            raise _validation_error(error=error) from error
        return _preferences_response(preferences=preferences)

    @router.get("/ui/account/sessions", response_model=AccountSessionsResponse)
    def get_sessions(
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountSessionsResponse:
        page = account_settings.list_sessions(
            owner_user_id=principal.user_id,
            cursor=cursor,
            limit=limit,
        )
        return AccountSessionsResponse(
            items=[_session_response(session=session) for session in page.items],
            next_cursor=page.next_cursor,
        )

    @router.get("/ui/account/audit-events", response_model=AccountAuditEventsResponse)
    def get_audit_events(
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> AccountAuditEventsResponse:
        page = account_settings.list_audit_events(
            owner_user_id=principal.user_id,
            cursor=cursor,
            limit=limit,
        )
        return AccountAuditEventsResponse(
            items=[_audit_event_response(event=event) for event in page.items],
            next_cursor=page.next_cursor,
        )

    return router


def _profile_response(
    *,
    profile: AccountProfileSettings,
    preferences: AccountPreferences,
    subscription_status: str,
) -> AccountProfileResponse:
    return AccountProfileResponse(
        user_id=str(profile.owner_user_id),
        username=profile.username,
        email=profile.email,
        timezone=profile.timezone,
        locale=preferences.locale,
        telegram_discord=profile.telegram_discord,
        subscription_status=subscription_status,
        updated_at=profile.updated_at,
    )


def _preferences_response(*, preferences: AccountPreferences) -> AccountPreferencesResponse:
    return AccountPreferencesResponse(
        theme=preferences.theme,
        locale=preferences.locale,
        density=preferences.density,
        autorefresh=AccountAutorefreshPreferenceResponse(
            preset_key=preferences.autorefresh_preset,
            refresh_interval_seconds=preferences.refresh_interval_seconds,
            allowed_presets=["off", "10s", "15s", "30s", "1m", "5m"],
            min_custom_interval_seconds=10,
            max_custom_interval_seconds=1800,
        ),
        updated_at=preferences.updated_at,
    )


def _integration_response(
    *,
    integration: AccountIntegrationSettings,
) -> AccountIntegrationResponse:
    label_by_key = {
        "telegram": "Telegram Bot",
        "discord": "Discord",
        "slack": "Slack",
    }
    return AccountIntegrationResponse(
        integration_key=integration.integration_key,
        label=label_by_key[integration.integration_key],
        status="disconnected" if integration.mode == "off" else "connected",
        mode=integration.mode,
        webhook_url_masked=integration.webhook_url_masked,
        updated_at=integration.updated_at,
    )


def _notification_response(
    *,
    notification: AccountNotificationSettings,
) -> AccountNotificationResponse:
    label_by_key = {
        "telegram": "Telegram notifications",
        "email": "Email notifications",
        "push": "Push notifications",
        "trade_fills": "Trade fill events",
        "risk_alerts": "Risk alerts",
        "daily_report": "Daily report",
        "system": "System notifications",
    }
    return AccountNotificationResponse(
        channel_key=notification.channel_key,
        label=label_by_key[notification.channel_key],
        mode=notification.mode,
        updated_at=notification.updated_at,
    )


def _session_response(*, session: AccountSessionView) -> AccountSessionResponse:
    return AccountSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        last_seen_at=session.last_seen_at,
        idle_expires_at=session.idle_expires_at,
        absolute_expires_at=session.absolute_expires_at,
        revoked_at=session.revoked_at,
        device=session.device,
        ip_address=session.ip_address,
        location=session.location,
        is_current=session.is_current,
    )


def _audit_event_response(*, event: AccountAuditEvent) -> AccountAuditEventResponse:
    return AccountAuditEventResponse(
        event_id=str(event.event_id),
        created_at=event.created_at,
        event_type=event.event_type,
        summary=event.summary,
        metadata=event.metadata,
    )


def _validation_error(*, error: AccountSettingsValidationError) -> RoehubError:
    return RoehubError(
        code="validation_error",
        message=error.message,
        details={
            "errors": [
                {
                    "path": error.field,
                    "code": error.code,
                    "message": error.message,
                }
            ]
        },
    )


def _enforce_same_origin_mutation(*, request: Request) -> None:
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    if origin is None and referer is None:
        return
    expected_host = request.headers.get("host", "")
    for candidate in (origin, referer):
        if candidate is None:
            continue
        parsed = urlparse(candidate)
        if parsed.netloc and parsed.netloc != expected_host:
            raise RoehubError(
                code="forbidden",
                message="Mutation origin is not allowed",
                details={"reason": "csrf_origin_mismatch"},
            )
