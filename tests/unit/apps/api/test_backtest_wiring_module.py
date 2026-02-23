from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import APIRouter

from apps.api.wiring.modules import backtest as backtest_module
from trading.contexts.backtest.adapters.outbound import (
    BacktestExecutionRuntimeConfig,
    BacktestJobsRuntimeConfig,
    BacktestReportingRuntimeConfig,
    BacktestRuntimeConfig,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.indicators.application.ports.compute import IndicatorCompute


class _DummyFactory:
    """
    Helper object accepting arbitrary constructor kwargs for wiring monkeypatches.
    """

    def __init__(self, **kwargs) -> None:
        """
        Store constructor kwargs for optional assertions in tests.

        Args:
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None.
        Assumptions:
            Tests do not execute behavior on instances.
        Raises:
            None.
        Side Effects:
            Stores kwargs on instance for debugging.
        """
        self.kwargs = kwargs


class _DummyDefaultsProvider:
    """
    Dummy defaults provider class exposing `from_environ` constructor API.
    """

    @classmethod
    def from_environ(cls, *, environ):
        """
        Return opaque defaults provider fixture object.

        Args:
            environ: Environment mapping.
        Returns:
            object: Opaque fixture object.
        Assumptions:
            build_backtest_router only needs a truthy provider object.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = environ
        return object()


class _DummyStrategyReader:
    """
    Dummy strategy reader wrapper used in wiring router-toggle tests.
    """

    def __init__(self, *, repository) -> None:
        """
        Store repository fixture object.

        Args:
            repository: Strategy repository fixture.
        Returns:
            None.
        Assumptions:
            Reader behavior is not executed in router-composition tests.
        Raises:
            None.
        Side Effects:
            Stores repository for debugging.
        """
        self.repository = repository



def _paths_from_router(*, router: APIRouter) -> set[str]:
    """
    Extract deterministic non-empty route paths from APIRouter route collection.

    Args:
        router: Built router instance.
    Returns:
        set[str]: Route path set.
    Assumptions:
        Route objects may not expose `path` attribute in static `BaseRoute` typing.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        path
        for route in router.routes
        for path in (str(getattr(route, "path", "")),)
        if path
    }



def _build_ping_router(*, path: str) -> APIRouter:
    """
    Build minimal ping router used by backtest wiring composition tests.

    Args:
        path: Route path literal.
    Returns:
        APIRouter: Router exposing one deterministic endpoint.
    Assumptions:
        Handlers are not executed during these tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    router = APIRouter()

    @router.get(path)
    def _ping() -> dict[str, str]:
        """
        Return deterministic static payload.

        Args:
            None.
        Returns:
            dict[str, str]: Static OK payload.
        Assumptions:
            Handler body is not relevant for routing assertions.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"ok": "1"}

    return router



def _runtime_config(*, jobs_enabled: bool) -> BacktestRuntimeConfig:
    """
    Build minimal runtime config fixture for backtest wiring router-toggle tests.

    Args:
        jobs_enabled: Jobs toggle value.
    Returns:
        BacktestRuntimeConfig: Valid runtime config fixture.
    Assumptions:
        Scalar defaults match production contracts and pass constructor validation.
    Raises:
        ValueError: If fixture setup violates runtime config invariants.
    Side Effects:
        None.
    """
    return BacktestRuntimeConfig(
        version=1,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        reporting=BacktestReportingRuntimeConfig(top_trades_n_default=3),
        execution=BacktestExecutionRuntimeConfig(
            init_cash_quote_default=10000.0,
            fixed_quote_default=100.0,
            safe_profit_percent_default=30.0,
            slippage_pct_default=0.01,
            fee_pct_default_by_market_id={1: 0.075},
        ),
        jobs=BacktestJobsRuntimeConfig(
            enabled=jobs_enabled,
            top_k_persisted_default=300,
            max_active_jobs_per_user=3,
            claim_poll_seconds=1.0,
            lease_seconds=60,
            heartbeat_seconds=15,
            parallel_workers=1,
            snapshot_seconds=30,
            snapshot_variants_step=1000,
        ),
    )



def _patch_backtest_wiring_dependencies(*, monkeypatch, jobs_enabled: bool) -> None:
    """
    Patch heavy backtest wiring dependencies for isolated router-composition checks.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        jobs_enabled: Jobs toggle fixture value.
    Returns:
        None.
    Assumptions:
        Patched stubs are sufficient for build_backtest_router composition flow.
    Raises:
        None.
    Side Effects:
        Replaces module-level function/class references in wiring module.
    """
    runtime_config = _runtime_config(jobs_enabled=jobs_enabled)

    monkeypatch.setattr(
        backtest_module,
        "resolve_backtest_config_path",
        lambda *, environ: Path("configs/test/backtest.yaml"),
    )
    monkeypatch.setattr(
        backtest_module,
        "load_backtest_runtime_config",
        lambda _path: runtime_config,
    )
    monkeypatch.setattr(
        backtest_module,
        "build_backtest_runtime_config_hash",
        lambda *, config: "f" * 64,
    )
    monkeypatch.setattr(backtest_module, "YamlBacktestGridDefaultsProvider", _DummyDefaultsProvider)
    monkeypatch.setattr(backtest_module, "_build_strategy_repository", lambda *, settings: object())
    monkeypatch.setattr(
        backtest_module,
        "StrategyRepositoryBacktestStrategyReader",
        _DummyStrategyReader,
    )
    monkeypatch.setattr(backtest_module, "_build_backtest_candle_feed", lambda *, environ: object())
    monkeypatch.setattr(backtest_module, "RunBacktestUseCase", _DummyFactory)
    monkeypatch.setattr(
        backtest_module,
        "build_backtests_router",
        lambda **kwargs: _build_ping_router(path="/backtests/ping"),
    )

    monkeypatch.setattr(backtest_module, "_build_jobs_gateway", lambda *, settings: object())
    monkeypatch.setattr(backtest_module, "PostgresBacktestJobRepository", _DummyFactory)
    monkeypatch.setattr(backtest_module, "PostgresBacktestJobResultsRepository", _DummyFactory)
    monkeypatch.setattr(backtest_module, "CreateBacktestJobUseCase", _DummyFactory)
    monkeypatch.setattr(backtest_module, "GetBacktestJobStatusUseCase", _DummyFactory)
    monkeypatch.setattr(backtest_module, "GetBacktestJobTopUseCase", _DummyFactory)
    monkeypatch.setattr(backtest_module, "ListBacktestJobsUseCase", _DummyFactory)
    monkeypatch.setattr(backtest_module, "CancelBacktestJobUseCase", _DummyFactory)
    monkeypatch.setattr(
        backtest_module,
        "build_backtest_jobs_router",
        lambda **kwargs: _build_ping_router(path="/backtests/jobs/ping"),
    )



def test_build_backtest_router_skips_jobs_routes_when_toggle_disabled(monkeypatch) -> None:
    """
    Verify jobs routes are not mounted when `backtest.jobs.enabled=false`.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Sync `/backtests` route remains mounted regardless of jobs toggle state.
    Raises:
        AssertionError: If jobs route appears while toggle is disabled.
    Side Effects:
        None.
    """
    _patch_backtest_wiring_dependencies(monkeypatch=monkeypatch, jobs_enabled=False)

    router = backtest_module.build_backtest_router(
        environ={},
        current_user_dependency=cast(
            RequireCurrentUserDependency,
            lambda _request: None,
        ),
        indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
    )
    paths = _paths_from_router(router=router)

    assert "/backtests/ping" in paths
    assert "/backtests/jobs/ping" not in paths



def test_build_backtest_router_mounts_jobs_routes_when_toggle_enabled(monkeypatch) -> None:
    """
    Verify jobs routes are mounted when `backtest.jobs.enabled=true`.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Wiring composes sync and jobs routers under one backtest module router.
    Raises:
        AssertionError: If jobs routes are missing while toggle is enabled.
    Side Effects:
        None.
    """
    _patch_backtest_wiring_dependencies(monkeypatch=monkeypatch, jobs_enabled=True)

    router = backtest_module.build_backtest_router(
        environ={"STRATEGY_PG_DSN": "postgresql://user:pass@localhost:5432/roehub"},
        current_user_dependency=cast(
            RequireCurrentUserDependency,
            lambda _request: None,
        ),
        indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
    )
    paths = _paths_from_router(router=router)

    assert "/backtests/ping" in paths
    assert "/backtests/jobs/ping" in paths



def test_build_backtest_router_fails_fast_when_jobs_enabled_without_dsn(monkeypatch) -> None:
    """
    Verify wiring fails fast when jobs toggle is enabled and STRATEGY_PG_DSN is missing.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        EPIC-11 jobs repositories require Postgres DSN at startup.
    Raises:
        AssertionError: If wiring does not raise deterministic ValueError.
    Side Effects:
        None.
    """
    _patch_backtest_wiring_dependencies(monkeypatch=monkeypatch, jobs_enabled=True)

    with pytest.raises(ValueError, match="STRATEGY_PG_DSN"):
        backtest_module.build_backtest_router(
            environ={},
            current_user_dependency=cast(
                RequireCurrentUserDependency,
                lambda _request: None,
            ),
            indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
        )
