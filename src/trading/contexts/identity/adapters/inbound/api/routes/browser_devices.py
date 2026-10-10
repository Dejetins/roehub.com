"""Account-owned browser sessions; opaque device IDs are not bearer credentials."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.browser_device import register_browser
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.inbound.api.routes.auth_local import (
    _enforce_authenticated_mutation,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal, IdentityClock
from trading.contexts.identity.application.ports.session_repository import SessionRepository


class BrowserDeviceResponse(BaseModel):
    device_id: UUID
    label: str
    last_seen_at: datetime
    expires_at: datetime
    active: bool
    is_current: bool


class BrowserDevicesResponse(BaseModel):
    items: list[BrowserDeviceResponse]
    next_cursor: str | None
    legacy_active_count: int


class DeviceRevokedResponse(BaseModel):
    revoked: bool = True
    current: bool


def build_browser_devices_router(
    *,
    repository: SessionRepository,
    current_user_dependency: RequireCurrentUserDependency,
    clock: IdentityClock,
    cookie_secure: bool,
    cookie_name: str,
    cookie_path: str,
) -> APIRouter:
    router = APIRouter(prefix="/ui/account/devices", tags=["identity"])

    @router.get("", response_model=BrowserDevicesResponse)
    def get_devices(
        request: Request,
        response: Response,
        cursor: UUID | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BrowserDevicesResponse:
        if principal.session_id is None:
            raise HTTPException(status_code=401, detail="missing_session")
        now = clock.now()
        # Adopt only the proven current legacy session. No UA/IP-based historical merging.
        register_browser(
            request=request,
            response=response,
            repository=repository,
            session_id=principal.session_id,
            now=now,
            secure=cookie_secure,
        )
        rows = repository.list_devices(
            user_id=principal.user_id, now=now, before=cursor, limit=limit + 1
        )
        return BrowserDevicesResponse(
            items=[
                BrowserDeviceResponse(
                    device_id=row.device_id,
                    label=row.label,
                    last_seen_at=row.last_seen_at,
                    expires_at=row.expires_at,
                    active=row.active,
                    is_current=row.session_id == principal.session_id,
                )
                for row in rows[:limit]
            ],
            next_cursor=str(rows[limit - 1].device_id) if len(rows) > limit else None,
            legacy_active_count=repository.legacy_session_count(user_id=principal.user_id, now=now),
        )

    @router.post("/legacy/revoke", response_model=DeviceRevokedResponse)
    def revoke_legacy(
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> DeviceRevokedResponse:
        _enforce_authenticated_mutation(request=request)
        if principal.session_id is None:
            raise HTTPException(status_code=401, detail="missing_session")
        now = clock.now()
        register_browser(
            request=request,
            response=response,
            repository=repository,
            session_id=principal.session_id,
            now=now,
            secure=cookie_secure,
        )
        repository.revoke_device(user_id=principal.user_id, device_id=None, now=now)
        return DeviceRevokedResponse(current=False)

    @router.post("/{device_id}/revoke", response_model=DeviceRevokedResponse)
    def revoke_device(
        device_id: UUID,
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> DeviceRevokedResponse:
        _enforce_authenticated_mutation(request=request)
        if principal.session_id is None:
            raise HTTPException(status_code=401, detail="missing_session")
        now = clock.now()
        current_id = register_browser(
            request=request,
            response=response,
            repository=repository,
            session_id=principal.session_id,
            now=now,
            secure=cookie_secure,
        )
        repository.revoke_device(user_id=principal.user_id, device_id=device_id, now=now)
        current = current_id == device_id
        if current:
            response.delete_cookie(
                cookie_name, path=cookie_path, secure=cookie_secure, httponly=True, samesite="lax"
            )
            response.delete_cookie(
                "roehub_csrf", path=cookie_path, secure=cookie_secure, samesite="lax"
            )
        response.headers["Cache-Control"] = "no-store"
        return DeviceRevokedResponse(current=current)

    return router
