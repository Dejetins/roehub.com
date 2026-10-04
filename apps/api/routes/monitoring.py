"""Read-only operational projection behind the existing operations permission."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.operational_health_client import (
    OperationalHealthClient,
    OperationalHealthClientError,
    OperationalHistoryClient,
)
from apps.api.routes.market_data_reference import _resolve_research_scope
from apps.monitoring.operational_health import OperationalStatus
from trading.contexts.backtest.application.ports import ResearchOrganizationScopeResolver
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases.organizations import (
    OrganizationAccessError,
    OrganizationAccessService,
)
from trading.platform.errors import RoehubError

GROUPS = ("core", "data", "compute", "trading", "security", "extensions")
RUNBOOKS = Path(__file__).resolve().parents[3] / "docs/runbooks/generated/ru"


def _group(item: OperationalStatus) -> str:
    capability = item.capability
    if capability.startswith(("secrets.", "identity.", "auth.")):
        return "security"
    if capability.startswith(("market_data.", "storage.clickhouse")):
        return "data"
    if capability.startswith(("research.", "ml.")):
        return "compute"
    if capability.startswith("trading."):
        return "trading"
    if capability.startswith(("extensions.", "notifications.")):
        return "extensions"
    return "core"


def _project(item: OperationalStatus, now: datetime) -> dict[str, Any]:
    age = max(0, (now - item.observed_at).total_seconds())
    state = (
        "stale"
        if item.detail_code == "probe.snapshot_stale" or age > 30
        else {
            "ready": "healthy",
            "stopped": "unavailable",
            "degraded": "degraded",
            "unknown": "unknown",
        }[item.state]
    )
    runbook = (
        f"/runbooks/{item.runbook_id}"
        if re.fullmatch(r"[a-z][a-z0-9.-]{1,127}", item.runbook_id)
        and (RUNBOOKS / f"{item.runbook_id}.md").is_file()
        else None
    )
    return dict(
        service_id=item.service_id,
        group=_group(item),
        capability=item.capability,
        state=state,
        observed_at=item.observed_at,
        detail_code=item.detail_code,
        signal_source="operational-health",
        runbook_path=runbook,
        required=item.required,
    )


def build_monitoring_router(
    *,
    current_user_dependency: Callable[[Request], CurrentUserPrincipal],
    scope_resolver: ResearchOrganizationScopeResolver | None,
    organization_service: OrganizationAccessService,
    health: OperationalHealthClient | None,
) -> APIRouter:
    router = APIRouter(tags=["monitoring"])

    def read(principal: CurrentUserPrincipal) -> dict[str, Any]:
        org = _resolve_research_scope(resolver=scope_resolver, principal=principal).organization_id
        try:
            organization_service.require_operation_read(principal=principal, organization_id=org)
        except OrganizationAccessError as error:
            raise RoehubError(code=error.code, message=error.message) from error
        if health is None:
            raise HTTPException(503, "Operational observer is not configured")
        try:
            snapshot = health.snapshot()
        except OperationalHealthClientError as error:
            raise HTTPException(503, "Operational observer is unavailable") from error
        now = datetime.now(UTC)
        events = []
        history_state = "unavailable"
        if isinstance(health, OperationalHistoryClient):
            try:
                ids = {s.service_id for s in snapshot.services}
                events = [
                    _project(e, e.observed_at) for e in health.events()[:200] if e.service_id in ids
                ]
                history_state = "available"
            except OperationalHealthClientError:
                pass
        return dict(
            generated_at=snapshot.generated_at,
            groups=GROUPS,
            services=[_project(s, now) for s in snapshot.services],
            events=events,
            history_state=history_state,
            history_scope="observer_lifetime",
            commands=[],
        )

    @router.get("/ui/monitoring")
    def inventory(principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        return read(principal)

    @router.get("/ui/monitoring/{service_id}")
    def detail(service_id: str, principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        snapshot = read(principal)
        row = next((s for s in snapshot["services"] if s["service_id"] == service_id), None)
        if row is None:
            raise HTTPException(404, "Configured service not found")
        return dict(
            service=row,
            generated_at=snapshot["generated_at"],
            events=[e for e in snapshot["events"] if e["service_id"] == service_id][:50],
            history_state=snapshot["history_state"],
            history_scope="observer_lifetime",
            metrics=[],
            dependencies=[],
            commands=[],
        )

    return router
