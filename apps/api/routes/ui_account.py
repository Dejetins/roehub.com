from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Literal
from uuid import UUID

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
    CreateExchangeConnectionRequest,
    ExchangeConnectionResponse,
    ExchangeConnectionsResponse,
    RotateExchangeConnectionRequest,
    UpdateAccountIntegrationRequest,
    UpdateAccountNotificationRequest,
    UpdateAccountPreferencesRequest,
    UpdateAccountProfileRequest,
)
from apps.api.exchange_control_client import (
    ExchangeConnectionCommandResult,
    ExchangeControlClient,
    ExchangeControlClientError,
)
from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountAuditEvent,
    AccountIntegrationSettings,
    AccountNotificationSettings,
    AccountPreferences,
    AccountProfileSettings,
    AccountSessionView,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
    AccountSettingsValidationError,
)
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]
_RECENT_AUTH_WINDOW = timedelta(minutes=10)
ExchangeConnectionStatusFilter = Literal["active", "disabled", "archived", "all"]


def build_ui_account_router(
    *,
    account_settings: AccountSettingsUseCase,
    current_user_dependency: CurrentUserDependency,
    clock: IdentityClock,
    exchange_control_client: ExchangeControlClient | None = None,
) -> APIRouter:
    if account_settings is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires account_settings")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires current_user_dependency")
    if clock is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires clock")

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
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            connections = client.list_connections(
                owner_user_id=str(principal.user_id),
                request_id="apps-api-account-limits-exchange-connections",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        active_connections = tuple(row for row in connections if row.status == "active")
        base_limits = account_settings.get_limits(
            owner_user_id=principal.user_id,
            plan=str(principal.paid_level),
            exchange_connections_used=len(active_connections),
            api_keys_used=len(active_connections),
        )
        return AccountLimitsResponse(
            plan=str(base_limits["plan"]),
            exchange_connections_used=int(base_limits["exchange_connections_used"]),
            exchange_connections_limit=int(base_limits["exchange_connections_limit"]),
            api_keys_used=int(base_limits["api_keys_used"]),
            api_keys_limit=int(base_limits["api_keys_limit"]),
            active_strategies_used=int(base_limits["active_strategies_used"]),
            active_strategies_limit=int(base_limits["active_strategies_limit"]),
            webhook_events_used=int(base_limits["webhook_events_used"]),
            webhook_events_limit=int(base_limits["webhook_events_limit"]),
        )

    @router.get(
        "/ui/account/exchange-connections",
        response_model=ExchangeConnectionsResponse,
    )
    def get_exchange_connections(
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=50),
        status: ExchangeConnectionStatusFilter = Query(default="active"),
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionsResponse:
        _ = cursor, limit
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            rows = client.list_connections(
                owner_user_id=str(principal.user_id),
                request_id="apps-api-list-exchange-connections",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        filtered_rows = tuple(
            row
            for row in rows
            if _matches_connection_status_filter(row=row, status=status)
        )
        return ExchangeConnectionsResponse(
            items=[_exchange_connection_response(row=row) for row in filtered_rows],
            next_cursor=None,
        )

    @router.post(
        "/ui/account/exchange-connections",
        response_model=ExchangeConnectionResponse,
        status_code=201,
    )
    def post_exchange_connection(
        request: Request,
        payload: CreateExchangeConnectionRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionResponse:
        _enforce_same_origin_mutation(request=request)
        _enforce_recent_auth(principal=principal, now=clock.now())
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            row = client.create_connection(
                owner_user_id=str(principal.user_id),
                exchange_name=payload.exchange_name,
                market_type=payload.market_type,
                environment=payload.environment,
                label=payload.label,
                permissions=payload.permissions,
                api_key=payload.api_key,
                api_secret=payload.api_secret,
                request_id="apps-api-create-exchange-connection",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        return _exchange_connection_response(row=row)

    @router.post(
        "/ui/account/exchange-connections/{connection_id}/rotate",
        response_model=ExchangeConnectionResponse,
    )
    def post_exchange_connection_rotation(
        connection_id: UUID,
        request: Request,
        payload: RotateExchangeConnectionRequest,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionResponse:
        _enforce_same_origin_mutation(request=request)
        _enforce_recent_auth(principal=principal, now=clock.now())
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            row = client.rotate_connection(
                owner_user_id=str(principal.user_id),
                connection_id=str(connection_id),
                api_key=payload.api_key,
                api_secret=payload.api_secret,
                request_id="apps-api-rotate-exchange-connection",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        return _exchange_connection_response(row=row)

    @router.post(
        "/ui/account/exchange-connections/{connection_id}/disable",
        response_model=ExchangeConnectionResponse,
    )
    def post_exchange_connection_disable(
        connection_id: UUID,
        request: Request,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionResponse:
        _enforce_same_origin_mutation(request=request)
        _enforce_recent_auth(principal=principal, now=clock.now())
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            row = client.disable_connection(
                owner_user_id=str(principal.user_id),
                connection_id=str(connection_id),
                request_id="apps-api-disable-exchange-connection",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        return _exchange_connection_response(row=row)

    @router.post(
        "/ui/account/exchange-connections/{connection_id}/archive",
        response_model=ExchangeConnectionResponse,
    )
    def post_exchange_connection_archive(
        connection_id: UUID,
        request: Request,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionResponse:
        _enforce_same_origin_mutation(request=request, fail_closed_without_origin=True)
        _enforce_recent_auth(principal=principal, now=clock.now())
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            before_rows = client.list_connections(
                owner_user_id=str(principal.user_id),
                request_id="apps-api-archive-exchange-connection-read",
            )
            row = client.archive_connection(
                owner_user_id=str(principal.user_id),
                connection_id=str(connection_id),
                request_id="apps-api-archive-exchange-connection",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        previous_row = next(
            (item for item in before_rows if item.connection_id == str(connection_id)),
            None,
        )
        if previous_row is None or previous_row.status != "archived":
            account_settings.record_exchange_connection_archive(
                owner_user_id=principal.user_id,
                connection_id=row.connection_id,
                exchange_name=row.exchange_name,
                market_type=row.market_type,
                environment=row.environment,
                previous_status=previous_row.status if previous_row else "disabled",
                new_status=row.status,
                reason=row.status_reason or "user_archived",
            )
        return _exchange_connection_response(row=row)

    @router.post(
        "/ui/account/exchange-connections/{connection_id}/validate",
        response_model=ExchangeConnectionResponse,
    )
    def post_exchange_connection_validate(
        connection_id: UUID,
        request: Request,
        principal: CurrentUserPrincipal = Depends(require_account_user),
    ) -> ExchangeConnectionResponse:
        _enforce_same_origin_mutation(request=request)
        client = _require_exchange_control_client(client=exchange_control_client)
        try:
            row = client.validate_connection(
                owner_user_id=str(principal.user_id),
                connection_id=str(connection_id),
                request_id="apps-api-validate-exchange-connection",
            )
        except ExchangeControlClientError as error:
            raise _exchange_control_unavailable(error=error) from error
        account_settings.record_exchange_connection_validation(
            owner_user_id=principal.user_id,
            exchange_name=row.exchange_name,
            validation_status=row.validation_status,
        )
        return _exchange_connection_response(row=row)

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


def _require_exchange_control_client(
    *,
    client: ExchangeControlClient | None,
) -> ExchangeControlClient:
    if client is None:
        raise RoehubError(
            code="exchange_control_unavailable",
            message="Exchange-control internal client is not configured",
            details={},
        )
    return client


def _exchange_control_unavailable(*, error: ExchangeControlClientError) -> RoehubError:
    message = str(error)
    if "exchange_connection_not_found" in message:
        return RoehubError(
            code="exchange_connection_not_found",
            message="Exchange connection was not found.",
            details={},
        )
    if "exchange_connection_not_owned" in message:
        return RoehubError(
            code="exchange_connection_not_owned",
            message="Exchange connection is not owned by current user.",
            details={},
        )
    if "exchange_connection_already_exists" in message:
        return RoehubError(
            code="exchange_connection_already_exists",
            message="Exchange connection already exists.",
            details={},
        )
    if "exchange_connection_not_disabled" in message:
        return RoehubError(
            code="exchange_connection_not_disabled",
            message="Exchange connection must be disabled before archive.",
            details={},
        )
    if "exchange_connection_invalid" in message:
        return RoehubError(
            code="validation_error",
            message="Exchange connection request is invalid",
            details={
                "errors": [
                    {
                        "path": "exchange_connection",
                        "code": "exchange_connection_invalid",
                        "message": "Exchange connection request is invalid.",
                    }
                ]
            },
        )
    return RoehubError(
        code="exchange_control_unavailable",
        message="Exchange-control internal request failed",
        details={},
    )


def _exchange_connection_response(
    *,
    row: ExchangeConnectionCommandResult,
) -> ExchangeConnectionResponse:
    return ExchangeConnectionResponse(
        connection_id=row.connection_id,
        credential_version_id=row.credential_version_id,
        exchange_name=_exchange_name_literal(value=row.exchange_name),
        market_type=_market_type_literal(value=row.market_type),
        environment=_environment_literal(value=row.environment),
        label=row.label,
        permissions=_permissions_literal(value=row.permissions),
        api_key=row.api_key,
        status=_connection_status_literal(value=row.status),
        status_reason=row.status_reason,
        validation_status=_validation_status_literal(value=row.validation_status),
        validation_reason=row.validation_reason,
        ip_restriction_status=row.ip_restriction_status,
        last_validated_at=row.last_validated_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        disabled_at=row.disabled_at,
        archived_at=row.archived_at,
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


def _enforce_same_origin_mutation(
    *,
    request: Request,
    fail_closed_without_origin: bool = False,
) -> None:
    rejection_reason = same_origin_rejection_reason(
        request=request,
        fail_closed_without_origin=fail_closed_without_origin,
    )
    if rejection_reason is None:
        return
    raise RoehubError(
        code="forbidden",
        message="Mutation origin is not allowed",
        details={"reason": rejection_reason},
    )


def _enforce_recent_auth(*, principal: CurrentUserPrincipal, now: datetime) -> None:
    if principal.session_created_at is None:
        raise RoehubError(
            code="recent_auth_required",
            message="Recent Keycloak authentication is required.",
            details={},
        )
    if principal.session_created_at + _RECENT_AUTH_WINDOW < now:
        raise RoehubError(
            code="recent_auth_required",
            message="Recent Keycloak authentication is required.",
            details={},
        )


def _exchange_name_literal(*, value: str) -> Literal["binance", "bybit"]:
    if value == "binance" or value == "bybit":
        return value
    raise ValueError(f"Unsupported exchange_name value: {value!r}")


def _market_type_literal(*, value: str) -> Literal["spot", "futures"]:
    if value == "spot" or value == "futures":
        return value
    raise ValueError(f"Unsupported market_type value: {value!r}")


def _environment_literal(*, value: str) -> Literal["mainnet", "testnet"]:
    if value == "mainnet" or value == "testnet":
        return value
    raise ValueError(f"Unsupported environment value: {value!r}")


def _permissions_literal(*, value: str) -> Literal["read", "trade"]:
    if value == "read" or value == "trade":
        return value
    raise ValueError(f"Unsupported permissions value: {value!r}")


def _matches_connection_status_filter(
    *,
    row: ExchangeConnectionCommandResult,
    status: ExchangeConnectionStatusFilter,
) -> bool:
    if status == "all":
        return True
    return row.status == status


def _connection_status_literal(*, value: str) -> Literal["active", "disabled", "archived"]:
    if value == "active" or value == "disabled" or value == "archived":
        return value
    raise ValueError(f"Unsupported connection status value: {value!r}")


def _validation_status_literal(
    *,
    value: str,
) -> Literal[
    "valid_readonly",
    "valid_trade_enabled",
    "invalid_credentials",
    "invalid_permissions",
    "invalid_ip_restriction",
    "unsupported_account_mode",
    "skipped_external_validation",
]:
    allowed = {
        "valid_readonly",
        "valid_trade_enabled",
        "invalid_credentials",
        "invalid_permissions",
        "invalid_ip_restriction",
        "unsupported_account_mode",
        "skipped_external_validation",
    }
    if value in allowed:
        return value  # type: ignore[return-value]
    raise ValueError(f"Unsupported validation status value: {value!r}")
