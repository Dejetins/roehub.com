from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import APIRouter

from apps.api.wiring.modules import backtest as backtest_module
from trading.contexts.backtest.adapters.outbound import (
    BacktestCpuRuntimeConfig,
    BacktestExecutionRuntimeConfig,
    BacktestGuardsRuntimeConfig,
    BacktestJobsRuntimeConfig,
    BacktestReportingRuntimeConfig,
    BacktestRuntimeConfig,
    BacktestSyncRuntimeConfig,
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

    def compute_defaults(self, *, indicator_id: str):
        """
        Return no compute defaults for router-composition wiring tests.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            None: No compute defaults are needed in these tests.
        Assumptions:
            Router-composition tests do not execute staged grid planning.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = indicator_id
        return None

    def signal_param_defaults(self, *, indicator_id: str):
        """
        Return empty signal defaults mapping for router-composition wiring tests.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            dict[str, object]: Empty defaults mapping.
        Assumptions:
            Runtime defaults response does not read signal defaults.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = indicator_id
        return {}

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return deterministic supported indicator catalog for wiring tests.

        Args:
            None.
        Returns:
            tuple[str, ...]: Static supported indicator ids.
        Assumptions:
            Router-composition tests only need deterministic launch catalog values.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("ma.sma",)

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Return deterministic source catalog for one indicator id.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            tuple[str, ...]: Static source values for supported indicator ids.
        Assumptions:
            Unsupported ids can return an empty tuple in these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("close", "hlc3") if indicator_id.strip().lower() == "ma.sma" else ()

    @classmethod
    def from_environ(cls, *, environ):
        """
        Return opaque defaults provider fixture object.

        Args:
            environ: Environment mapping.
        Returns:
            _DummyDefaultsProvider: Fixture provider instance.
        Assumptions:
            build_backtest_router only needs a truthy provider object.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = environ
        return cls()


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


class _DummyArtifactsRuntimeConfig:
    """
    Dummy artifact runtime config exposing only `artifact_root_path()` for wiring tests.
    """

    def __init__(self, *, artifact_root: Path) -> None:
        """
        Store deterministic artifact root used by path-builder assertions.

        Args:
            artifact_root: Filesystem root returned by `artifact_root_path()`.
        Returns:
            None.
        Assumptions:
            Wiring tests only need artifact root and not the full config surface.
        Raises:
            None.
        Side Effects:
            Stores root on instance for later access.
        """
        self._artifact_root = artifact_root

    def artifact_root_path(self) -> Path:
        """
        Return deterministic artifact root path for builder composition tests.

        Args:
            None.
        Returns:
            Path: Configured artifact store root.
        Assumptions:
            Router wiring treats this as the source-of-truth root literal.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._artifact_root


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



def _runtime_config(
    *,
    jobs_enabled: bool,
    max_variants_per_compute: int = 600000,
    max_compute_bytes_total: int = 5 * 1024**3,
    max_numba_threads: int = 4,
    sync_deadline_seconds: float = 55.0,
    eager_top_reports_enabled: bool = False,
) -> BacktestRuntimeConfig:
    """
    Build minimal runtime config fixture for backtest wiring router-toggle tests.

    Args:
        jobs_enabled: Jobs toggle value.
        max_variants_per_compute: Variants guard limit.
        max_compute_bytes_total: Memory guard limit.
        max_numba_threads: Runtime Numba threads cap.
        sync_deadline_seconds: Sync route cooperative hard deadline in seconds.
        eager_top_reports_enabled: Feature flag for eager top reports in sync response.
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
        reporting=BacktestReportingRuntimeConfig(
            top_trades_n_default=3,
            eager_top_reports_enabled=eager_top_reports_enabled,
        ),
        execution=BacktestExecutionRuntimeConfig(
            init_cash_quote_default=10000.0,
            fixed_quote_default=100.0,
            safe_profit_percent_default=30.0,
            slippage_pct_default=0.01,
            fee_pct_default_by_market_id={1: 0.075},
        ),
        guards=BacktestGuardsRuntimeConfig(
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        ),
        cpu=BacktestCpuRuntimeConfig(max_numba_threads=max_numba_threads),
        sync=BacktestSyncRuntimeConfig(sync_deadline_seconds=sync_deadline_seconds),
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
        "_load_backtest_artifacts_runtime_config",
        lambda *, environ: _DummyArtifactsRuntimeConfig(
            artifact_root=Path("artifacts/backtest/v2")
        ),
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
        "CreateAndRunBacktestSyncInlineUseCase",
        _DummyFactory,
    )
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


def test_build_backtest_router_passes_sync_half_guards_to_run_use_case(monkeypatch) -> None:
    """
    Verify wiring passes sync half-guards and sync deadline into composed dependencies.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Jobs mode keeps full guard values; only sync route uses half budgets.
    Raises:
        AssertionError: If run use-case kwargs do not contain expected halved guard values.
    Side Effects:
        None.
    """
    captured_kwargs: dict[str, object] = {}
    captured_sync_inline_kwargs: dict[str, object] = {}
    captured_backtests_router_kwargs: dict[str, object] = {}
    runtime_config = _runtime_config(
        jobs_enabled=False,
        max_variants_per_compute=101,
        max_compute_bytes_total=1001,
        max_numba_threads=7,
        sync_deadline_seconds=42.5,
        eager_top_reports_enabled=False,
    )

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
        "_load_backtest_artifacts_runtime_config",
        lambda *, environ: _DummyArtifactsRuntimeConfig(
            artifact_root=Path("artifacts/backtest/v2")
        ),
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

    class _CaptureRunBacktestUseCase:
        """
        Capture run use-case constructor kwargs for guard/CPU assertions.
        """

        def __init__(self, **kwargs) -> None:
            """
            Store kwargs for deterministic assertions.

            Args:
                **kwargs: Constructor kwargs from wiring module.
            Returns:
                None.
            Assumptions:
                Captured kwargs are not mutated by router builder.
            Raises:
                None.
            Side Effects:
                Stores kwargs in enclosing test scope.
            """
            captured_kwargs.update(kwargs)

    class _CaptureCreateAndRunBacktestSyncInlineUseCase:
        """
        Capture persisted sync-inline orchestrator constructor kwargs for wiring assertions.
        """

        def __init__(self, **kwargs) -> None:
            """
            Store kwargs for deterministic assertions.

            Args:
                **kwargs: Constructor kwargs from wiring module.
            Returns:
                None.
            Assumptions:
                Captured kwargs are not mutated by router builder.
            Raises:
                None.
            Side Effects:
                Stores kwargs in enclosing test scope.
            """
            captured_sync_inline_kwargs.update(kwargs)

    monkeypatch.setattr(backtest_module, "RunBacktestUseCase", _CaptureRunBacktestUseCase)
    monkeypatch.setattr(
        backtest_module,
        "CreateAndRunBacktestSyncInlineUseCase",
        _CaptureCreateAndRunBacktestSyncInlineUseCase,
    )
    monkeypatch.setattr(
        backtest_module,
        "build_backtests_router",
        lambda **kwargs: captured_backtests_router_kwargs.update(kwargs)
        or _build_ping_router(path="/backtests/ping"),
    )

    router = backtest_module.build_backtest_router(
        environ={"STRATEGY_PG_DSN": "postgresql://user:pass@localhost:5432/roehub"},
        current_user_dependency=cast(
            RequireCurrentUserDependency,
            lambda _request: None,
        ),
        indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
    )
    assert "/backtests/ping" in _paths_from_router(router=router)
    assert captured_kwargs["max_variants_per_compute"] == 50
    assert captured_kwargs["max_compute_bytes_total"] == 500
    assert captured_kwargs["max_numba_threads"] == 7
    assert captured_kwargs["eager_top_reports_enabled"] is False
    assert captured_sync_inline_kwargs["backtest_runtime_config_hash"] == "f" * 64
    assert captured_sync_inline_kwargs["engine_version"] == "signal_tf + 1m_risk"
    assert captured_backtests_router_kwargs["sync_deadline_seconds"] == 42.5
    assert captured_backtests_router_kwargs["eager_top_reports_enabled"] is False



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
        environ={"STRATEGY_PG_DSN": "postgresql://user:pass@localhost:5432/roehub"},
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


def test_build_backtest_router_uses_artifact_root_from_artifact_config(monkeypatch) -> None:
    """
    Verify jobs wiring builds artifact path resolver from strict artifact config root.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Backtest jobs mode composes artifact loader only after artifact config is loaded.
    Raises:
        AssertionError: If configured artifact root is not passed into the path builder.
    Side Effects:
        None.
    """
    captured_builder_root: Path | None = None
    runtime_config = _runtime_config(jobs_enabled=True)

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
        "_load_backtest_artifacts_runtime_config",
        lambda *, environ: _DummyArtifactsRuntimeConfig(
            artifact_root=Path("custom/artifacts/backtest/v2")
        ),
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
        "CreateAndRunBacktestSyncInlineUseCase",
        _DummyFactory,
    )
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

    class _CapturePathBuilder:
        """
        Capture path-builder root used during artifact loader composition.
        """

        def __init__(self, *, root: Path) -> None:
            """
            Store path-builder root for deterministic assertions.

            Args:
                root: Artifact root path injected by wiring module.
            Returns:
                None.
            Assumptions:
                Wiring passes `root` as a keyword argument when composing jobs loader.
            Raises:
                None.
            Side Effects:
                Stores captured root in enclosing test scope.
            """
            nonlocal captured_builder_root
            captured_builder_root = root

    monkeypatch.setattr(backtest_module, "BacktestArtifactPathBuilderV2", _CapturePathBuilder)
    monkeypatch.setattr(
        backtest_module,
        "YamlBacktestArtifactLoaderV2",
        lambda *, path_resolver: _DummyFactory(path_resolver=path_resolver),
    )

    router = backtest_module.build_backtest_router(
        environ={"STRATEGY_PG_DSN": "postgresql://user:pass@localhost:5432/roehub"},
        current_user_dependency=cast(
            RequireCurrentUserDependency,
            lambda _request: None,
        ),
        indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
    )

    assert "/backtests/jobs/ping" in _paths_from_router(router=router)
    assert captured_builder_root == Path("custom/artifacts/backtest/v2")



def test_build_backtest_router_fails_fast_when_persisted_sync_storage_dsn_is_missing(
    monkeypatch,
) -> None:
    """
    Verify wiring fails fast when persisted sync storage DSN is missing at startup.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        R7-02 sync-inline persistence reuses the jobs storage family and requires Postgres DSN.
    Raises:
        AssertionError: If wiring does not raise deterministic ValueError.
    Side Effects:
        None.
    """
    _patch_backtest_wiring_dependencies(monkeypatch=monkeypatch, jobs_enabled=False)

    with pytest.raises(ValueError, match="STRATEGY_PG_DSN"):
        backtest_module.build_backtest_router(
            environ={},
            current_user_dependency=cast(
                RequireCurrentUserDependency,
                lambda _request: None,
            ),
            indicator_compute=cast(IndicatorCompute, SimpleNamespace()),
        )
