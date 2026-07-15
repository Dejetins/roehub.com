"""Real-network Web fixture for the Stage 20 operational admin browser proof."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import RedirectResponse

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app

app = create_app(
    environ={
        "WEB_API_BASE_URL": "http://127.0.0.1:8766",
        "WEB_API_UPSTREAM_URL": "http://127.0.0.1:8765",
        "ROEHUB_ASSET_VERSION": "stage20-real-chain",
    }
)
app.state.current_user_api_client = type(
    "SyntheticCurrentUserClient",
    (),
    {
        "fetch_current_user": lambda self, *, cookie_header: CurrentUserApiResult(
            status_code=200,
            user=WebCurrentUser(user_id="stage20-browser-user", paid_level="free"),
            error_message=None,
        )
    },
)()


@app.get("/__qa/admin/setup", include_in_schema=False)
def configure_admin_qa(
    request: Request,
    role: str = "owner",
    recent: bool = True,
) -> RedirectResponse:
    _ = request
    allowed_roles = {"owner", "admin", "operator", "trader", "viewer"}
    effective_role = role if role in allowed_roles else "viewer"
    response = RedirectResponse(url="/admin")
    response.set_cookie("roehub_session_id", "stage20-local", httponly=True, samesite="strict")
    response.set_cookie("roehub_locale", "ru", samesite="strict")
    response.set_cookie("qa_role", effective_role, httponly=True, samesite="strict")
    response.set_cookie("qa_recent", "1" if recent else "0", httponly=True, samesite="strict")
    return response
