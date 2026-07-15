"""Synthetic, secret-free browser QA boundary for the Stage 19 admin console."""

from __future__ import annotations

from http.cookies import SimpleCookie
from uuid import UUID, uuid4, uuid5

import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app

ORGANIZATION_ID = UUID("10000000-0000-0000-0000-000000000019")
OWNER_ID = UUID("20000000-0000-0000-0000-000000000019")
MEMBER_ID = UUID("30000000-0000-0000-0000-000000000019")
PLUGIN_ID = UUID("40000000-0000-0000-0000-000000000019")
PACKAGE_ID = UUID("50000000-0000-0000-0000-000000000019")
INSTANCE_ID = UUID("60000000-0000-0000-0000-000000000019")

_PERMISSIONS = {
    "owner": [
        "organization.read",
        "organization.update",
        "members.read",
        "members.manage",
        "roles.manage",
        "plugins.read",
        "plugins.manage",
        "operations.execute",
        "trading.execute",
        "mainnet.approve",
        "audit.read",
    ],
    "admin": [
        "organization.read",
        "organization.update",
        "members.read",
        "members.manage",
        "roles.manage",
        "plugins.read",
        "plugins.manage",
        "operations.execute",
        "audit.read",
    ],
    "operator": [
        "organization.read",
        "members.read",
        "plugins.read",
        "operations.execute",
        "audit.read",
    ],
    "trader": [
        "organization.read",
        "members.read",
        "plugins.read",
        "trading.execute",
    ],
    "viewer": ["organization.read", "members.read", "plugins.read"],
}


def _cookies(request: httpx.Request) -> dict[str, str]:
    parsed = SimpleCookie()
    parsed.load(request.headers.get("cookie", ""))
    return {key: value.value for key, value in parsed.items()}


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _api_handler(request: httpx.Request) -> httpx.Response:
    values = _cookies(request)
    role = values.get("qa_role", "owner")
    if role not in _PERMISSIONS:
        role = "viewer"
    recent = values.get("qa_recent", "1") == "1"
    unknown = values.get("qa_unknown", "0") == "1"
    health_state = values.get("qa_health", "ready")
    if health_state not in {"ready", "degraded", "stopped", "unknown"}:
        health_state = "unknown"
    path = request.url.path

    if request.method == "GET" and path in {"/v1/organizations", "/api/v1/organizations"}:
        return _json_response(
            200,
            [
                {
                    "organization": {
                        "organization_id": str(ORGANIZATION_ID),
                        "installation_id": "70000000-0000-0000-0000-000000000019",
                        "slug": "qa-organization",
                        "display_name": "QA Organization",
                        "status": "active",
                        "created_at": "2026-07-13T09:00:00Z",
                    },
                    "role": role,
                    "permissions": _PERMISSIONS[role],
                }
            ],
        )

    if request.method == "GET" and path.endswith("/snapshot"):
        can_operate = "operations.execute" in _PERMISSIONS[role]
        events = (
            [
                {
                    "event_id": "80000000-0000-0000-0000-000000000019",
                    "category": "identity",
                    "action": "membership.role_changed",
                    "target_type": "membership",
                    "target_id": str(MEMBER_ID),
                    "outcome": "succeeded",
                    "created_at": "2026-07-13T10:00:00Z",
                }
            ]
            if "audit.read" in _PERMISSIONS[role]
            else []
        )
        return _json_response(
            200,
            {
                "schema": "io.roehub.admin-snapshot/v1alpha1",
                "organization_id": str(ORGANIZATION_ID),
                "organization_name": "QA Organization",
                "role": role,
                "permissions": _PERMISSIONS[role],
                "recent_auth": recent,
                "installation_owner": role == "owner",
                "members": [
                    {
                        "user_id": str(OWNER_ID),
                        "role": "owner",
                        "status": "active",
                        "created_at": "2026-07-13T09:00:00Z",
                    },
                    {
                        "user_id": str(MEMBER_ID),
                        "role": role if role != "owner" else "admin",
                        "status": "active",
                        "created_at": "2026-07-13T09:05:00Z",
                    },
                ],
                "plugin_installations": [
                    {
                        "plugin_installation_id": str(PLUGIN_ID),
                        "plugin_id": "qa.market-data",
                        "package_id": str(PACKAGE_ID),
                        "rollback_available": True,
                        "granted_permissions": ["data.read"],
                        "status": "enabled",
                        "updated_at": "2026-07-13T09:30:00Z",
                    }
                ],
                "plugin_instances": [
                    {
                        "instance_id": str(INSTANCE_ID),
                        "plugin_installation_id": str(PLUGIN_ID),
                        "name": "QA market feed",
                        "config_revision": 2,
                        "status": "enabled",
                        "updated_at": "2026-07-13T09:30:00Z",
                    }
                ],
                "plugin_operations": [],
                "events": events,
                "capabilities": {
                    "providers": "available",
                    "backups": "deferred",
                    "updates": "ready" if can_operate else "degraded",
                    "services": "ready" if can_operate else "degraded",
                    "observability": (
                        "ready" if health_state == "ready" else "degraded"
                    ),
                },
                "operational_health": {
                    "schema": "io.roehub.admin-operational-health/v1alpha1",
                    "profile": "trading",
                    "overall_state": health_state,
                    "generated_at": "2026-07-14T12:00:00Z",
                    "grafana_path": None,
                    "services": [
                        {
                            "service_id": "api",
                            "capability": "product.web_api",
                            "state": health_state,
                            "detail_code": (
                                "probe.http_ready"
                                if health_state == "ready"
                                else "probe.connection_refused"
                            ),
                            "runbook_id": "web.api-health-degraded",
                            "runbook_path": "/runbooks/web.api-health-degraded",
                            "action_ref": "restart_service",
                            "observed_at": "2026-07-14T12:00:00Z",
                        },
                        {
                            "service_id": "openbao",
                            "capability": "secrets.openbao",
                            "state": (
                                "degraded"
                                if health_state == "stopped"
                                else health_state
                            ),
                            "detail_code": (
                                "probe.openbao_ready"
                                if health_state == "ready"
                                else (
                                    "probe.unknown"
                                    if health_state == "unknown"
                                    else "probe.openbao_not_ready"
                                )
                            ),
                            "runbook_id": "auth.openbao-unavailable",
                            "runbook_path": "/runbooks/auth.openbao-unavailable",
                            "action_ref": "restart_service",
                            "observed_at": "2026-07-14T12:00:00Z",
                        },
                    ],
                },
            },
        )

    if request.method == "PATCH" and "/members/" in path:
        if path.endswith(str(OWNER_ID)):
            return _json_response(
                409,
                {
                    "error": {
                        "code": "last_owner",
                        "message": "The last organization owner cannot be demoted",
                        "details": {},
                    }
                },
            )
        return _json_response(
            200,
            {
                "organization_id": str(ORGANIZATION_ID),
                "user_id": str(MEMBER_ID),
                "role": "operator",
                "status": "active",
                "created_at": "2026-07-13T09:05:00Z",
            },
        )

    if request.method == "POST" and path.endswith("/plugins/bundles:validate"):
        return _json_response(
            200,
            {
                "contract": "ValidatedPluginBundle/v1alpha1",
                "plugin_id": "qa.market-data",
                "version": "1.1.0",
                "package_digest": f"sha256:{'1' * 64}",
                "image_digest": f"sha256:{'2' * 64}",
                "publisher_key_id": "qa-publisher",
                "permissions": ["data.read", "network.egress"],
            },
        )

    if request.method == "POST" and path.endswith("/plugins/installations"):
        return _json_response(
            202,
            {
                "contract": "PluginOperation/v1alpha1",
                "operation_id": str(uuid4()),
                "organization_id": str(ORGANIZATION_ID),
                "kind": "update",
                "target_id": "qa.market-data",
                "status": "pending",
                "result": {"contract": "PluginOperation/v1alpha1"},
                "created_at": "2026-07-13T10:01:00Z",
                "updated_at": "2026-07-13T10:01:00Z",
            },
        )

    if request.method == "POST" and path.endswith("/operations"):
        operation_id = _scoped_operation_id(request)
        if operation_id is None:
            return _json_response(
                422,
                {
                    "error": {
                        "code": "admin.idempotency_key_invalid",
                        "message": "Idempotency-Key is invalid",
                        "details": {},
                    }
                },
            )
        return _json_response(
            202,
            {
                "schema": "io.roehub.control-operation-result/v1alpha1",
                "operation_id": str(operation_id),
                "action": "restart",
                "profile": "base",
                "state": "unknown" if unknown else "accepted",
                "detail_code": (
                    "operation.state_unknown" if unknown else "operation.accepted"
                ),
                "active_services": [],
                "journal_sequence": 1,
            },
        )

    if request.method == "POST" and path.endswith(":reconcile"):
        operation_id = _scoped_operation_id(request)
        if operation_id is None or str(operation_id) not in path:
            return _json_response(
                404,
                {
                    "error": {
                        "code": "admin.operation_not_found",
                        "message": "Administrative operation is not found",
                        "details": {},
                    }
                },
            )
        return _json_response(
            200,
            {
                "schema": "io.roehub.control-operation-result/v1alpha1",
                "operation_id": str(operation_id),
                "action": "restart",
                "profile": "base",
                "state": "succeeded",
                "detail_code": "operation.reconciled",
                "active_services": ["api", "web"],
                "journal_sequence": 2,
            },
        )

    if request.method == "GET" and "/operations/" in path:
        operation_id = _scoped_operation_id(request)
        if operation_id is None or str(operation_id) not in path:
            return _json_response(
                404,
                {
                    "error": {
                        "code": "admin.operation_not_found",
                        "message": "Administrative operation is not found",
                        "details": {},
                    }
                },
            )
        return _json_response(
            200,
            {
                "schema": "io.roehub.control-operation-result/v1alpha1",
                "operation_id": str(operation_id),
                "action": "restart",
                "profile": "base",
                "state": "unknown" if unknown else "succeeded",
                "detail_code": (
                    "operation.state_unknown" if unknown else "operation.completed"
                ),
                "active_services": ["api", "web"],
                "journal_sequence": 2,
            },
        )

    return _json_response(
        404,
        {"error": {"code": "not_found", "message": "QA route not found", "details": {}}},
    )


def _scoped_operation_id(request: httpx.Request) -> UUID | None:
    idempotency_key = request.headers.get("Idempotency-Key", "")
    if len(idempotency_key) < 8:
        return None
    return uuid5(ORGANIZATION_ID, f"admin-operation:{idempotency_key}")


app = create_app(
    environ={
        "WEB_API_BASE_URL": "http://127.0.0.1:8765",
        "WEB_API_UPSTREAM_URL": "http://admin-qa.local",
        "ROEHUB_ASSET_VERSION": "stage19-qa",
    }
)
app.state.current_user_api_client = type(
    "SyntheticCurrentUserClient",
    (),
    {
        "fetch_current_user": lambda self, *, cookie_header: CurrentUserApiResult(
            status_code=200,
            user=WebCurrentUser(user_id="qa-browser-user", paid_level="free"),
            error_message=None,
        )
    },
)()
app.state.api_proxy_transport = httpx.MockTransport(_api_handler)


@app.get("/__qa/admin/setup", include_in_schema=False)
def configure_admin_qa(
    request: Request,
    role: str = "owner",
    recent: bool = True,
    unknown: bool = False,
    health: str = "ready",
) -> RedirectResponse:
    _ = request
    effective_role = role if role in _PERMISSIONS else "viewer"
    response = RedirectResponse(url="/admin")
    response.set_cookie("qa_role", effective_role, httponly=True, samesite="strict")
    response.set_cookie("qa_recent", "1" if recent else "0", httponly=True, samesite="strict")
    response.set_cookie("qa_unknown", "1" if unknown else "0", httponly=True, samesite="strict")
    effective_health = (
        health if health in {"ready", "degraded", "stopped", "unknown"} else "unknown"
    )
    response.set_cookie("qa_health", effective_health, httponly=True, samesite="strict")
    return response
