"""Independent local Navigator client; reference build and services stay unchanged."""
from importlib import import_module
from pathlib import Path

from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates


def create_app():
    web = import_module("apps.web.main.app")
    setattr(
        web, "DEFAULT_PLATFORM_DIST",
        Path(__file__).resolve().parents[2] / "apps/navigator-web/dist",
    )
    app = web.create_app()
    if app.state.platform_assets is None:
        raise RuntimeError("Navigator requires its independent platform client build")
    app.state.client_routes.append("/dashboard")
    app.state.client_routes.extend(["/settings", "/connections", "/data", "/monitoring"])
    templates = Jinja2Templates(directory=str(web._TEMPLATES_PATH))

    @app.middleware("http")
    async def settings_compatibility(request: Request, call_next):
        if request.url.path == "/settings":
            target = {
                "market-data": "/data", "market_data": "/data",
                "api": "/connections", "integrations": "/connections",
                "preferences": "/settings/preferences",
                "notifications": "/settings/notifications", "security": "/settings/security",
            }.get(request.query_params.get("tab", ""), "/settings/profile")
            return RedirectResponse(target, status_code=308)
        return await call_next(request)

    @app.get("/settings/{category}")
    def settings_page(request: Request, category: str):
        if category not in {"profile", "preferences", "notifications", "security"}:
            raise web.HTTPException(status_code=404)
        return web._render_protected_page(
            request=request, templates=templates, page_path=request.url.path,
            active_path="/settings", page_title_key="page.settings.title",
        )

    @app.get("/data")
    @app.get("/data/ingestion")
    def data_page(request: Request):
        return web._render_protected_page(
            request=request, templates=templates, page_path=request.url.path,
            active_path="/data", page_title_key="settings.tabs.market_data",
        )

    @app.get("/monitoring/{service_id}")
    def service_page(request: Request, service_id: str):
        return web._render_protected_page(
            request=request, templates=templates, page_path=request.url.path,
            active_path="/monitoring", page_title_key="page.monitoring.title",
        )
    return app
