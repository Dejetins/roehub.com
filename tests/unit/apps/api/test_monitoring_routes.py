from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.operational_health_client import OperationalHealthClientError
from apps.api.routes.monitoring import _project, build_monitoring_router
from apps.monitoring.operational_health import OperationalStatus
from trading.shared_kernel.primitives import OrganizationId


def stub(**values: Any) -> Any:
    """Partial dependencies expose only the capabilities exercised by each route test."""
    return SimpleNamespace(**values)


def status(**values):
    return OperationalStatus(
        service_id="postgresql",
        capability="storage.postgresql",
        state="unknown",
        detail_code="probe.reachable_no_readiness",
        runbook_id="runtime.service-degraded",
        action_ref="diagnostics",
        required=True,
        observed_at=values.get("observed_at", datetime.now(UTC)),
    )


def test_projection_never_turns_reachability_or_old_observation_into_healthy():
    now = datetime.now(UTC)
    assert _project(status(), now)["state"] == "unknown"
    assert _project(status(observed_at=now - timedelta(minutes=2)), now)["state"] == "stale"
    assert "target" not in _project(status(), now)


def test_monitoring_denies_before_observer_and_reports_unavailable_without_fake_success():
    reads = []
    access = []

    def principal(request: Request):
        if not request.headers.get("x-authorized"):
            raise HTTPException(401)
        return stub(user_id=uuid4())

    def require_operation_read(**kwargs):
        access.append(kwargs)
        if len(access) == 1:
            raise HTTPException(403)

    def snapshot():
        reads.append(1)
        raise OperationalHealthClientError("must not expose raw internal failure")

    app = FastAPI()
    app.include_router(
        build_monitoring_router(
            current_user_dependency=principal,
            scope_resolver=stub(resolve=lambda **_: stub(organization_id=OrganizationId(uuid4()))),
            organization_service=stub(require_operation_read=require_operation_read),
            health=stub(snapshot=snapshot),
        )
    )
    client = TestClient(app)
    assert client.get("/ui/monitoring").status_code == 401
    assert client.get("/ui/monitoring", headers={"x-authorized": "1"}).status_code == 403
    assert reads == []
    r = client.get("/ui/monitoring", headers={"x-authorized": "1"})
    assert r.status_code == 503 and "raw internal" not in r.text
