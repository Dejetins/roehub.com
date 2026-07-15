from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app

_INSTANCE_ID = "00000000-0000-0000-0000-000000000713"
_REQUESTS: list[dict[str, Any]] = []


def _frame(*, state: str) -> dict[str, Any]:
    now = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    rows = [
        {
            "timestamp": (now + timedelta(minutes=index * 5)).isoformat(),
            "pnl": round(105.0 + index * 3.4 - (index % 3) * 2.1, 2),
            "drawdown": round(-1.5 - (index % 4) * 0.7, 2),
        }
        for index in range(12)
    ]
    if state == "empty":
        rows = []
    partial = state == "partial"
    degraded = state == "degraded"
    return {
        "contract": "RoehubDataFrame/v1",
        "frame_id": f"qa.portfolio.{state}",
        "title": "Sanitized portfolio fixture",
        "columns": [
            {
                "key": "timestamp",
                "label": "Observed at",
                "data_type": "timestamp",
                "role": "dimension",
                "unit": {"kind": "timestamp", "symbol": "UTC", "scale": 1.0},
                "nullable": False,
            },
            {
                "key": "pnl",
                "label": "Portfolio PnL",
                "data_type": "number",
                "role": "measure",
                "unit": {"kind": "currency", "symbol": "USD", "scale": 1.0},
                "nullable": False,
            },
            {
                "key": "drawdown",
                "label": "Drawdown",
                "data_type": "number",
                "role": "measure",
                "unit": {"kind": "percent", "symbol": "%", "scale": 1.0},
                "nullable": False,
            },
        ],
        "rows": rows,
        "metadata": {
            "source_label": "Controlled external database fixture",
            "query_label": "Portfolio PnL by five-minute interval",
            "generated_at": now.isoformat(),
            "attributes": {"fixture": True, "environment": "isolated"},
        },
        "freshness": {
            "status": "stale" if degraded else "fresh",
            "observed_at": now.isoformat(),
            "age_seconds": 180 if degraded else 2,
            "max_age_seconds": 60,
        },
        "notices": (
            [
                {
                    "level": "warning",
                    "code": "fixture.freshness_degraded",
                    "message": "The sanitized fixture is older than its freshness target.",
                }
            ]
            if degraded
            else []
        ),
        "partial": partial,
        "errors": (
            [
                {
                    "code": "fixture.segment_unavailable",
                    "message": "One sanitized segment is unavailable.",
                    "retryable": True,
                    "field": None,
                }
            ]
            if partial
            else []
        ),
    }


def _api_handler(request: httpx.Request) -> httpx.Response:
    state = request.url.params.get("fixture_state", "success")
    try:
        payload = json.loads(request.content)
    except (UnicodeError, json.JSONDecodeError):
        payload = {}
    _REQUESTS.append(
        {
            "method": request.method,
            "path": request.url.path,
            "state": state,
            "read_only": payload.get("read_only"),
            "has_organization_authority": any(
                key in payload for key in ("organization_id", "organization", "tenant")
            ),
        }
    )
    if state == "error":
        return httpx.Response(
            503,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "code": "data_source.gateway_unavailable",
                    "message": "Controlled fixture is unavailable",
                    "details": {},
                }
            },
        )
    time.sleep(0.35)
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json=_frame(state=state),
    )


app = create_app(
    environ={
        "WEB_API_BASE_URL": "http://127.0.0.1:8765",
        "WEB_API_UPSTREAM_URL": "http://stage13-api.invalid",
        "ROEHUB_PLUGIN_PANEL_LAB": "true",
        "ROEHUB_PLUGIN_PANEL_LAB_INSTANCE_ID": _INSTANCE_ID,
        "WEB_ASSET_VERSION": "stage13-browser-proof",
    }
)
app.state.current_user_api_client = SimpleNamespace(
    fetch_current_user=lambda *, cookie_header: CurrentUserApiResult(
        status_code=200,
        user=WebCurrentUser(user_id="stage13-disposable-user", paid_level="free"),
        error_message=None,
    )
)
app.state.api_proxy_transport = httpx.MockTransport(_api_handler)


@app.get("/__qa/plugin-panels/status", include_in_schema=False)
def get_status() -> dict[str, Any]:
    return {
        "contract": "PluginPanelBrowserProof/v1",
        "requests": list(_REQUESTS),
    }
