"""Explicit local-only portfolio UI preview; keep default dashboard routing intact."""
from apps.web.main.app import create_app as create_web_app


def create_app():
    app = create_web_app()
    if app.state.platform_assets is None:
        raise RuntimeError("Overview preview requires the platform client build")
    app.state.client_routes.append("/dashboard")
    return app
