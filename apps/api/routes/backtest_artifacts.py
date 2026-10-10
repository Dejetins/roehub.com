"""Authenticated, bounded artifact management HTTP routes."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.routing import APIRoute
from pydantic import ValidationError

from apps.api.dto.backtest_artifacts import (
    Command,
    CommandItem,
    CommandResult,
    Defaults,
    Detail,
    Events,
    InventoryPage,
    RequestPage,
    RequestView,
    SafeStorage,
    SinglePolicy,
    State,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    ManagementError,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_PRICE_TIMEFRAMES_V2,
)
from trading.contexts.identity.adapters.inbound.api.csrf import same_origin_rejection_reason


class BoundedArtifactRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def bounded(request):
            chunks = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 1024 * 1024:
                    raise HTTPException(413, {"code": "request_too_large"})
                chunks.append(chunk)
            request._body = b"".join(chunks)
            return await original(request)

        return bounded


def artifact_error(error):
    reason = str(error) if isinstance(error, ManagementError) else "service_unavailable"
    allowed = {
        "forbidden",
        "installation_forbidden",
        "coordinate_forbidden",
        "not_found",
        "stale_version",
        "idempotency_conflict",
        "immutable_profile_conflict",
        "catalog_changed",
        "management_unavailable",
        "storage_unconfigured",
        "quota_exceeded",
        "invalid_command",
        "invalid_bulk",
        "manual_policy_mismatch",
        "resource_conflict",
    }
    if reason not in allowed:
        reason = "service_unavailable" if isinstance(error, psycopg.Error) else "invalid_request"
    status = (
        403
        if reason in {"forbidden", "installation_forbidden", "coordinate_forbidden"}
        else (
            404
            if reason == "not_found"
            else 409
            if reason
            in {
                "stale_version",
                "idempotency_conflict",
                "immutable_profile_conflict",
                "catalog_changed",
                "manual_policy_mismatch",
            }
            else 503
            if reason in {"management_unavailable", "storage_unconfigured", "service_unavailable"}
            else 429
            if reason in {"quota_exceeded", "resource_conflict"}
            else 422
        )
    )
    return HTTPException(
        status, {"code": reason}, headers={"Retry-After": "30"} if status in (429, 503) else None
    )


def require_mutation(request, key):
    if same_origin_rejection_reason(request=request, fail_closed_without_origin=True):
        raise HTTPException(403, {"code": "csrf_required"})
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", key) is None:
        raise HTTPException(422, {"code": "invalid_idempotency_key"})


def build_artifact_router(*, service, current_user_dependency, scope_resolver):
    router = APIRouter(
        prefix="/api/backtest-artifacts",
        tags=["backtest-artifacts"],
        route_class=BoundedArtifactRoute,
    )

    def scope(principal):
        if service is None or scope_resolver is None:
            raise HTTPException(
                503, {"code": "management_unavailable"}, headers={"Retry-After": "30"}
            )
        try:
            return scope_resolver.resolve(
                user_id=principal.user_id
            ).organization_id.value, principal.user_id.value
        except PermissionError as error:
            raise HTTPException(403, {"code": "forbidden"}) from error

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ManagementError, psycopg.Error, ValidationError) as error:
            raise artifact_error(error) from error

    @router.get("/defaults", response_model=Defaults)
    def defaults(principal=Depends(current_user_dependency)):
        org, actor = scope(principal)
        settings = call(service.settings, org, actor)
        member = service.repo._one(
            "SELECT role FROM identity_memberships WHERE organization_id=%(org)s AND "
            "user_id=%(actor)s",
            org=org,
            actor=actor,
        )
        return Defaults(
            profile=service.repo.catalog.default_profile(),
            schedule=settings["schedule"],
            base_timeframes=ARTIFACT_PRICE_TIMEFRAMES_V2,
            unavailable_indicator_ids=tuple(sorted(service.repo.catalog.unavailable)),
            can_mutate=not settings["paused"] and member["role"] in ("owner", "admin", "operator"),
        )

    @router.get("/storage", response_model=SafeStorage)
    def storage(principal=Depends(current_user_dependency)):
        org, actor = scope(principal)
        settings = call(service.settings, org, actor, active=False)
        state = (
            "ready"
            if settings["capability_enabled"] and settings["legacy_handoff"] == "complete"
            else "unavailable"
        )
        if settings["paused"]:
            state = "paused"
        return SafeStorage(state=state)

    @router.get("/inventory", response_model=InventoryPage)
    def inventory(
        exchange: str = Query(max_length=16),
        market_type: str = Query(max_length=16),
        search: str = Query(default="", max_length=64),
        cursor: str | None = Query(default=None, max_length=256),
        limit: int = Query(default=50, ge=1, le=100),
        principal=Depends(current_user_dependency),
    ):
        org, actor = scope(principal)
        return call(service.inventory, org, actor, exchange, market_type, search, cursor, limit)

    @router.get("/instruments/{exchange}/{market_type}/{symbol}", response_model=Detail)
    def detail(
        exchange: str, market_type: str, symbol: str, principal=Depends(current_user_dependency)
    ):
        org, actor = scope(principal)
        return call(
            lambda: service.detail(
                org, actor, Coordinate(exchange=exchange, market_type=market_type, symbol=symbol)
            )
        )

    @router.put(
        "/instruments/{exchange}/{market_type}/{symbol}/profile",
        response_model=CommandResult,
        status_code=202,
    )
    @router.put(
        "/instruments/{exchange}/{market_type}/{symbol}/schedule",
        response_model=CommandResult,
        status_code=202,
    )
    def single(
        exchange: str,
        market_type: str,
        symbol: str,
        body: SinglePolicy,
        request: Request,
        key: str = Header(alias="Idempotency-Key"),
        principal=Depends(current_user_dependency),
    ):
        require_mutation(request, key)
        org, actor = scope(principal)
        return call(
            lambda: service.submit(
                org,
                actor,
                key,
                Command(
                    items=(
                        CommandItem(
                            coordinate=Coordinate(
                                exchange=exchange, market_type=market_type, symbol=symbol
                            ),
                            expected_version=body.expected_version,
                            patch=body.patch,
                        ),
                    )
                ),
                "policy",
            )
        )

    @router.post("/build-requests", response_model=CommandResult, status_code=202)
    def build(
        body: Command,
        request: Request,
        key: str = Header(alias="Idempotency-Key"),
        principal=Depends(current_user_dependency),
    ):
        require_mutation(request, key)
        org, actor = scope(principal)
        return call(service.submit, org, actor, key, body, "update_now")

    @router.post("/policies:bulk", response_model=CommandResult, status_code=202)
    def bulk(
        body: Command,
        request: Request,
        key: str = Header(alias="Idempotency-Key"),
        principal=Depends(current_user_dependency),
    ):
        require_mutation(request, key)
        org, actor = scope(principal)
        return call(service.submit, org, actor, key, body, "policy")

    @router.get("/requests", response_model=RequestPage)
    def requests(
        cursor: UUID | None = None,
        state: State | None = None,
        kind: Literal["base", "derivatives"] | None = None,
        limit: int = Query(default=30, ge=1, le=50),
        principal=Depends(current_user_dependency),
    ):
        org, actor = scope(principal)
        return call(service.requests, org, actor, cursor, limit, state, kind)

    @router.get("/requests/{request_id}", response_model=RequestView)
    def job(request_id: UUID, principal=Depends(current_user_dependency)):
        org, actor = scope(principal)
        return call(service.request, org, actor, request_id)

    @router.get("/requests/{request_id}/events", response_model=Events)
    def events(
        request_id: UUID,
        limit: int = Query(default=30, ge=1, le=50),
        principal=Depends(current_user_dependency),
    ):
        org, actor = scope(principal)
        return Events(items=call(service.events, org, actor, request_id, limit))

    @router.get("/commands/{key}", response_model=CommandResult)
    def command(key: str, principal=Depends(current_user_dependency)):
        org, actor = scope(principal)
        if len(key) > 128:
            raise HTTPException(422, {"code": "invalid_idempotency_key"})
        result = call(service.repo.command, organization_id=org, actor_id=actor, key=key)
        if result is None:
            raise HTTPException(404, {"code": "not_found"})
        return result["response"]

    return router
