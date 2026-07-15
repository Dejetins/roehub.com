from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter


def _build_ping_router(*, path: str) -> APIRouter:
    """
    Build minimal ping router used by API app composition tests.

    Args:
        path: Route path to expose.
    Returns:
        APIRouter: Router with one deterministic `GET` endpoint.
    Assumptions:
        Endpoint is used only to inspect router inclusion semantics.
    Raises:
        None.
    Side Effects:
        None.
    """
    router = APIRouter()

    @router.get(path)
    def _ping() -> dict[str, str]:
        """
        Return deterministic ping payload for router-inclusion tests.

        Args:
            None.
        Returns:
            dict[str, str]: Static ok payload.
        Assumptions:
            Handler is not executed in this test module.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"ok": "1"}

    return router


def _load_app_module(*, monkeypatch):
    """
    Import API app module with deterministic low-thread Numba env for CI stability.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        module: Imported `apps.api.main.app` module object.
    Assumptions:
        Importing module executes `app = create_app()` once at module load.
    Raises:
        Exception: Propagates import-time runtime/configuration errors.
    Side Effects:
        Sets process env variables controlling Numba threads for this test scope.
    """
    numba_threads = "1"
    try:
        import numba
        from numba.np.ufunc import parallel

        if parallel._is_initialized:
            numba_threads = str(numba.get_num_threads())
    except Exception:
        numba_threads = "1"
    monkeypatch.setenv("ROEHUB_NUMBA_NUM_THREADS", numba_threads)
    monkeypatch.setenv("NUMBA_NUM_THREADS", numba_threads)
    monkeypatch.setenv("STRATEGY_PG_DSN", "postgresql://user:pass@localhost:5432/roehub")
    monkeypatch.setenv("IDENTITY_FAIL_FAST", "false")
    original_read_text = Path.read_text

    def _safe_read_text(self: Path, *args, **kwargs) -> str:
        if self == Path("/etc/roehub/roehub.env"):
            return ""
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _safe_read_text)
    return importlib.import_module("apps.api.main.app")


def _patch_create_app_dependencies(*, app_module, monkeypatch, strategy_enabled: bool) -> None:
    """
    Patch heavy runtime dependencies in API app factory for isolated router-toggle checks.

    Args:
        strategy_enabled: Desired runtime result for `is_strategy_api_enabled`.
    Returns:
        None.
    Assumptions:
        Patched stubs are sufficient for `create_app` wiring code path.
    Raises:
        None.
    Side Effects:
        Replaces module-level function references in `apps.api.main.app`.
    """
    identity_router = _build_ping_router(path="/identity/ping")
    strategy_router = _build_ping_router(path="/strategies/ping")
    indicators_router = _build_ping_router(path="/indicators/ping")
    market_data_router = _build_ping_router(path="/market-data/markets")
    extensions_router = _build_ping_router(path="/plugins/ping")
    admin_router = _build_ping_router(path="/admin/ping")
    ui_dashboard_router = _build_ping_router(path="/ui/dashboard/summary")

    monkeypatch.setattr(app_module, "build_indicators_registry", lambda *, environ: object())
    monkeypatch.setattr(
        app_module,
        "load_indicators_compute_numba_config",
        lambda *, environ: SimpleNamespace(
            max_variants_per_compute=10,
            max_compute_bytes_total=10,
        ),
    )
    monkeypatch.setattr(
        app_module,
        "build_indicators_compute",
        lambda *, environ, config: object(),
    )
    monkeypatch.setattr(
        app_module,
        "bind_indicators_runtime_dependencies",
        lambda *, app_state, compute, candle_feed: None,
    )
    monkeypatch.setattr(
        app_module,
        "build_indicators_router",
        lambda *,
        registry,
        current_user_dependency,
        organization_scope_resolver,
        compute,
        max_variants_per_compute,
        max_compute_bytes_total: indicators_router,
    )
    monkeypatch.setattr(
        app_module,
        "build_identity_api_module",
        lambda *, environ: SimpleNamespace(
            router=identity_router,
            current_user_dependency=lambda _request: None,
            organization_repository=object(),
            organization_access_service=object(),
            clock=object(),
        ),
    )
    monkeypatch.setattr(
        app_module,
        "build_extensions_api_module",
        lambda *, environ, current_user_dependency, organization_repository: SimpleNamespace(
            router=extensions_router,
            service=object(),
        ),
    )
    monkeypatch.setattr(
        app_module,
        "build_admin_router",
        lambda **_kwargs: admin_router,
    )
    monkeypatch.setattr(
        app_module,
        "is_strategy_api_enabled",
        lambda *, environ: strategy_enabled,
    )
    monkeypatch.setattr(
        app_module,
        "build_strategy_router",
        lambda *, environ, current_user_dependency, live_execution_services=None: strategy_router,
    )
    monkeypatch.setattr(
        app_module,
        "build_market_data_reference_router",
        lambda *, environ, current_user_dependency: market_data_router,
    )
    monkeypatch.setattr(
        app_module,
        "build_ui_dashboard_router",
        lambda *, environ, current_user_dependency: ui_dashboard_router,
    )


def test_create_app_includes_strategy_router_when_enabled(monkeypatch) -> None:
    """
    Verify API app includes Strategy router when strategy runtime toggle is enabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `is_strategy_api_enabled` represents effective runtime toggle state.
    Raises:
        AssertionError: If Strategy routes are missing when toggle is enabled.
    Side Effects:
        None.
    """
    app_module = _load_app_module(monkeypatch=monkeypatch)
    _patch_create_app_dependencies(
        app_module=app_module,
        monkeypatch=monkeypatch,
        strategy_enabled=True,
    )

    app = app_module.create_app(environ={})
    paths = {
        str(getattr(route, "path"))
        for route in app.routes
        if hasattr(route, "path")
    }

    assert "/health" in paths
    assert "/metrics" in paths
    assert "/strategies/ping" in paths
    assert "/market-data/markets" in paths
    assert "/admin/ping" in paths
    assert "/ui/dashboard/summary" in paths


def test_create_app_skips_strategy_router_when_disabled(monkeypatch) -> None:
    """
    Verify API app skips Strategy router when strategy runtime toggle is disabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Disabled Strategy module must not break other API routers.
    Raises:
        AssertionError: If Strategy routes are present while toggle is disabled.
    Side Effects:
        None.
    """
    app_module = _load_app_module(monkeypatch=monkeypatch)
    _patch_create_app_dependencies(
        app_module=app_module,
        monkeypatch=monkeypatch,
        strategy_enabled=False,
    )

    app = app_module.create_app(environ={})
    paths = {
        str(getattr(route, "path"))
        for route in app.routes
        if hasattr(route, "path")
    }

    assert "/health" in paths
    assert "/metrics" in paths
    assert "/strategies/ping" not in paths
    assert "/identity/ping" in paths
    assert "/indicators/ping" in paths
    assert "/market-data/markets" in paths
    assert "/ui/dashboard/summary" in paths
