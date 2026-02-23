from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import APIRouter

from apps.api.wiring.modules import market_data_reference as market_data_reference_module


class _DummySettingsLoader:
    """
    Dummy ClickHouse settings loader returning deterministic database settings.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/apps/api/test_market_data_reference_wiring_module.py
      - apps/api/wiring/modules/market_data_reference.py
      - apps/cli/wiring/db/clickhouse.py
    """

    def __init__(self, environ) -> None:  # noqa: ANN001
        """
        Store environ argument for compatibility with production loader API.

        Parameters:
        - environ: runtime environment mapping.

        Returns:
        - None.

        Assumptions/Invariants:
        - Value is unused in this test double.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        self._environ = environ

    def load(self):  # noqa: ANN001
        """
        Return minimal settings namespace expected by wiring module.

        Parameters:
        - None.

        Returns:
        - Settings object with `database` field.

        Assumptions/Invariants:
        - Wiring uses only `database` in this test.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = self._environ
        return SimpleNamespace(database="market_data")


class _DummyFactory:
    """
    Generic constructor-compatible stub storing kwargs for wiring assertions.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/apps/api/test_market_data_reference_wiring_module.py
      - apps/api/wiring/modules/market_data_reference.py
      - tests/unit/apps/api/test_backtest_wiring_module.py
    """

    def __init__(self, **kwargs) -> None:
        """
        Store constructor kwargs on instance.

        Parameters:
        - **kwargs: arbitrary constructor arguments.

        Returns:
        - None.

        Assumptions/Invariants:
        - Stub behavior is not executed in these wiring tests.

        Errors/Exceptions:
        - None.

        Side effects:
        - Stores kwargs for optional debug inspection.
        """
        self.kwargs = kwargs


def test_build_market_data_reference_router_mounts_expected_paths(monkeypatch) -> None:
    """
    Verify wiring module composes and returns router with reference API endpoints.

    Parameters:
    - monkeypatch: pytest monkeypatch fixture.

    Returns:
    - None.

    Assumptions/Invariants:
    - Route builder dependency can be stubbed with deterministic ping router.

    Errors/Exceptions:
    - None.

    Side effects:
    - Monkeypatches wiring module collaborators.
    """
    monkeypatch.setattr(
        market_data_reference_module,
        "ClickHouseSettingsLoader",
        _DummySettingsLoader,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "_clickhouse_client",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "ThreadLocalClickHouseConnectGateway",
        _DummyFactory,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "ClickHouseEnabledMarketReader",
        _DummyFactory,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "ClickHouseEnabledTradableInstrumentSearchReader",
        _DummyFactory,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "ListEnabledMarketsUseCase",
        _DummyFactory,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "SearchEnabledTradableInstrumentsUseCase",
        _DummyFactory,
    )
    monkeypatch.setattr(
        market_data_reference_module,
        "build_market_data_reference_api_router",
        lambda **_kwargs: _build_ping_router(),
    )

    router = market_data_reference_module.build_market_data_reference_router(
        environ={},
        current_user_dependency=lambda _request: None,  # type: ignore[arg-type]
    )
    paths = _paths_from_router(router=router)

    assert "/market-data/markets" in paths
    assert "/market-data/instruments" in paths


def test_build_market_data_reference_router_requires_auth_dependency() -> None:
    """
    Verify missing auth dependency is rejected at wiring stage.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Wiring must fail-fast on startup when required dependencies are missing.

    Errors/Exceptions:
    - Expects `ValueError`.

    Side effects:
    - None.
    """
    with pytest.raises(
        ValueError,
        match="build_market_data_reference_router requires current_user_dependency",
    ):
        market_data_reference_module.build_market_data_reference_router(
            environ={},
            current_user_dependency=None,  # type: ignore[arg-type]
        )


def _build_ping_router() -> APIRouter:
    """
    Build deterministic router exposing market-data reference endpoint paths.

    Parameters:
    - None.

    Returns:
    - APIRouter with two static ping handlers.

    Assumptions/Invariants:
    - Handlers are used only for route-path assertions.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    router = APIRouter()

    @router.get("/market-data/markets")
    def _markets_ping() -> dict[str, str]:
        """
        Return static payload for markets path assertion.

        Parameters:
        - None.

        Returns:
        - Static dictionary payload.

        Assumptions/Invariants:
        - Handler body is not relevant for this test.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return {"ok": "1"}

    @router.get("/market-data/instruments")
    def _instruments_ping() -> dict[str, str]:
        """
        Return static payload for instruments path assertion.

        Parameters:
        - None.

        Returns:
        - Static dictionary payload.

        Assumptions/Invariants:
        - Handler body is not relevant for this test.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return {"ok": "1"}

    return router


def _paths_from_router(*, router: APIRouter) -> set[str]:
    """
    Extract deterministic non-empty route paths from APIRouter collection.

    Parameters:
    - router: router object under test.

    Returns:
    - Set of non-empty route path strings.

    Assumptions/Invariants:
    - Route objects expose `.path` attribute in runtime implementation.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return {
        path
        for route in router.routes
        for path in (str(getattr(route, "path", "")),)
        if path
    }
