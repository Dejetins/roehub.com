from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.wiring.modules.identity import build_identity_api_module


def _client():
    app = FastAPI()
    app.include_router(build_identity_api_module(environ={"ROEHUB_ENV": "test"}).router)
    return TestClient(app)


def test_current_user_dependency_rejects_missing_cookie():
    response = _client().get("/auth/current-user")
    assert response.status_code == 401


def test_current_user_dependency_rejects_unknown_session_cookie():
    client = _client()
    client.cookies.set("roehub_session_id", "00000000-0000-0000-0000-000000000001")
    assert client.get("/auth/current-user").status_code == 401


def test_external_auth_is_not_exposed_and_local_auth_remains():
    client = _client()
    for path in ("status", "login", "callback", "link"):
        assert client.get(f"/auth/oidc/{path}").status_code == 404
    assert client.get("/auth/local/status").status_code == 200
