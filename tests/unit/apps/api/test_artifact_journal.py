"""Journal query validation and backward-compatible filter forwarding."""

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.backtest_artifacts import build_artifact_router


def test_journal_filters_are_optional_and_closed():
    org, actor = uuid4(), uuid4()
    service = Mock()
    service.requests.return_value = {"items": [], "next_cursor": None}
    resolver = Mock()
    resolver.resolve.return_value = SimpleNamespace(organization_id=SimpleNamespace(value=org))
    app = FastAPI()
    app.include_router(
        build_artifact_router(
            service=service,
            current_user_dependency=lambda: SimpleNamespace(user_id=SimpleNamespace(value=actor)),
            scope_resolver=resolver,
        )
    )
    client = TestClient(app)
    url = "/api/backtest-artifacts/requests"
    assert client.get(url).status_code == 200
    service.requests.assert_called_with(org, actor, None, 30, None, None)
    cursor = uuid4()
    assert (
        client.get(
            url, params={"state": "error", "kind": "base", "cursor": str(cursor)}
        ).status_code
        == 200
    )
    service.requests.assert_called_with(org, actor, cursor, 30, "error", "base")
    service.requests.reset_mock()
    for params in ({"state": "unknown"}, {"kind": "other"}, {"limit": 51}):
        assert client.get(url, params=params).status_code == 422
    service.requests.assert_not_called()
