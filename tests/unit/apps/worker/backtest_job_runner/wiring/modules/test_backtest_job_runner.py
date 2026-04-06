from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from apps.worker.backtest_job_runner.main import main as main_module
from apps.worker.backtest_job_runner.wiring.modules import (
    backtest_job_runner as worker_module,
)
from apps.worker.backtest_job_runner.wiring.modules import build_backtest_job_runner_app


def test_build_backtest_job_runner_app_requires_strategy_pg_dsn() -> None:
    """
    Verify worker wiring fails fast when `STRATEGY_PG_DSN` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime config file exists and loads before DSN validation.
    Raises:
        AssertionError: If missing DSN does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="STRATEGY_PG_DSN"):
        build_backtest_job_runner_app(
            config_path="configs/dev/backtest.yaml",
            environ={},
            instance_index=0,
            metrics_port=9204,
        )


def test_build_backtest_job_runner_app_skips_clickhouse_wiring_for_artifact_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify worker app wiring no longer constructs ClickHouse timeline dependencies in R8-01.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Claimed worker execution is artifact-only and therefore startup should not require CH.
    Raises:
        AssertionError: If wiring still forwards legacy live-timeline dependencies.
    Side Effects:
        Monkeypatches runtime/wiring constructors for isolated startup regression coverage.
    """
    captured_runner_kwargs: dict[str, object] = {}

    runtime_config = SimpleNamespace(
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20_000,
        reporting=SimpleNamespace(top_trades_n_default=3),
        ranking=SimpleNamespace(
            primary_metric_default="total_return_pct",
            secondary_metric_default=None,
        ),
        jobs=SimpleNamespace(
            top_k_persisted_default=300,
            lease_seconds=60,
            heartbeat_seconds=15,
            worker_processes=3,
            snapshot_seconds=None,
            snapshot_variants_step=None,
            claim_poll_seconds=5.0,
        ),
        execution=SimpleNamespace(
            init_cash_quote_default=10_000.0,
            fixed_quote_default=100.0,
            safe_profit_percent_default=30.0,
            slippage_pct_default=0.01,
            fee_pct_default_by_market_id={1: 0.075},
        ),
        guards=SimpleNamespace(
            max_variants_per_compute=600_000,
            max_compute_bytes_total=1024,
        ),
        cpu=SimpleNamespace(max_numba_threads=1),
        contracts=SimpleNamespace(
            allowed_request_timeframes=("15m", "30m", "1h"),
            forbidden_request_timeframes=("1m", "5m"),
        ),
        execution_profiles=SimpleNamespace(),
        adaptive_selector_policy=SimpleNamespace(mode="shadow"),
    )
    artifact_runtime_config = SimpleNamespace(
        artifact_root_path=lambda: Path("/tmp/backtest-artifacts-test")
    )

    monkeypatch.setattr(worker_module, "load_backtest_runtime_config", lambda _path: runtime_config)
    monkeypatch.setattr(
        worker_module,
        "load_backtest_artifacts_runtime_config",
        lambda _path: artifact_runtime_config,
    )
    monkeypatch.setattr(
        worker_module,
        "resolve_backtest_artifacts_config_path",
        lambda *, environ: Path("/tmp/backtest-artifacts.yaml"),
    )
    monkeypatch.setattr(
        worker_module,
        "PsycopgBacktestPostgresGateway",
        lambda *, dsn: SimpleNamespace(dsn=dsn),
    )
    monkeypatch.setattr(
        worker_module,
        "PostgresBacktestJobRepository",
        lambda *, gateway: SimpleNamespace(gateway=gateway),
    )
    monkeypatch.setattr(
        worker_module,
        "PostgresBacktestJobLeaseRepository",
        lambda *, gateway: SimpleNamespace(gateway=gateway),
    )
    monkeypatch.setattr(
        worker_module,
        "PostgresBacktestJobResultsRepository",
        lambda *, gateway: SimpleNamespace(gateway=gateway),
    )
    monkeypatch.setattr(
        worker_module,
        "build_indicators_compute",
        lambda *, environ: SimpleNamespace(environ=environ),
    )
    monkeypatch.setattr(
        worker_module.YamlBacktestGridDefaultsProvider,
        "from_environ",
        staticmethod(lambda *, environ: SimpleNamespace(environ=environ)),
    )
    monkeypatch.setattr(
        worker_module,
        "BacktestArtifactPathBuilderV2",
        lambda *, root: SimpleNamespace(root=root),
    )
    monkeypatch.setattr(
        worker_module,
        "YamlBacktestArtifactLoaderV2",
        lambda *, path_resolver: SimpleNamespace(path_resolver=path_resolver),
    )
    monkeypatch.setattr(
        worker_module,
        "ArtifactSlotResolverV2",
        lambda *, artifact_loader: SimpleNamespace(artifact_loader=artifact_loader),
    )
    monkeypatch.setattr(
        worker_module,
        "BacktestArtifactRuntimePlannerV2",
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )

    def _fake_runner_use_case(**kwargs: object) -> object:
        captured_runner_kwargs.update(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(worker_module, "RunBacktestJobRunnerV1", _fake_runner_use_case)

    app = build_backtest_job_runner_app(
        config_path="configs/dev/backtest.yaml",
        environ={"STRATEGY_PG_DSN": "postgresql://local/test"},
        instance_index=2,
        metrics_port=9206,
    )

    artifact_slot_resolver = cast(Any, captured_runner_kwargs["artifact_slot_resolver"])
    assert app.instance_index == 2
    assert app.metrics_port == 9206
    assert "hostname=" in app.locked_by
    assert ";pid=" in app.locked_by
    assert app.locked_by.endswith("instance_index=2")
    assert "candle_timeline_builder" not in captured_runner_kwargs
    assert artifact_slot_resolver.artifact_loader.path_resolver.root == Path(
        "/tmp/backtest-artifacts-test"
    )


def test_build_backtest_job_runner_app_rejects_instance_index_outside_worker_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify worker startup fails fast when instance_index exceeds configured worker cardinality.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime config loader already resolved jobs.worker_processes before heavier wiring begins.
    Raises:
        AssertionError: If invalid instance index does not raise ValueError.
    Side Effects:
        Monkeypatches runtime config loader for isolated startup invariant coverage.
    """
    runtime_config = SimpleNamespace(
        jobs=SimpleNamespace(worker_processes=2),
    )

    monkeypatch.setattr(worker_module, "load_backtest_runtime_config", lambda _path: runtime_config)

    with pytest.raises(ValueError, match="instance_index must be < worker_processes"):
        build_backtest_job_runner_app(
            config_path="configs/dev/backtest.yaml",
            environ={"STRATEGY_PG_DSN": "postgresql://local/test"},
            instance_index=2,
            metrics_port=9204,
        )


def test_build_locked_by_includes_hostname_pid_and_instance_index() -> None:
    """
    Verify locked_by identity is deterministic and readable for fleet operations.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Hostname, pid, and instance index together uniquely identify one worker process.
    Raises:
        AssertionError: If locked_by omits one of the identity components.
    Side Effects:
        None.
    """
    locked_by = worker_module._build_locked_by(
        instance_index=4,
        hostname="worker-host",
        pid=321,
    )

    assert locked_by == "hostname=worker-host;pid=321;instance_index=4"


def test_run_async_passes_instance_index_to_worker_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify worker entrypoint forwards explicit instance_index and per-instance metrics_port.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Entry point resolves a deterministic per-instance metrics_port before starting the app.
    Raises:
        AssertionError: If instance-aware identity or metrics binding is not forwarded correctly.
    Side Effects:
        Monkeypatches startup dependencies and runs the async entrypoint once.
    """
    captured_build_kwargs: dict[str, object] = {}

    class _FakeApp:
        """
        Minimal worker app stub that exits immediately after startup wiring.
        """

        locked_by = "hostname=test-host;pid=123;instance_index=3"

        async def run(self, stop_event: object) -> None:
            """
            Record one startup invocation without entering a long-lived claim loop.

            Args:
                stop_event: Cooperative shutdown signal passed by the entrypoint.
            Returns:
                None.
            Assumptions:
                Startup test only needs to observe that the app would have been run.
            Raises:
                None.
            Side Effects:
                Stores the received stop event for later assertions.
            """
            captured_build_kwargs["stop_event"] = stop_event

    monkeypatch.setattr(
        main_module,
        "_resolve_config_path",
        lambda *, config_path, environ: Path(config_path or "configs/dev/backtest.yaml"),
    )
    monkeypatch.setattr(
        main_module,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(jobs=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(main_module, "_install_signal_handlers", lambda _stop_event: None)
    monkeypatch.setattr(
        main_module,
        "build_backtest_job_runner_app",
        lambda **kwargs: captured_build_kwargs.update(kwargs) or _FakeApp(),
    )

    exit_code = main_module.asyncio.run(
        main_module._run_async(
            config_path="configs/dev/backtest.yaml",
            metrics_port=9304,
            instance_index=3,
        )
    )

    assert exit_code == 0
    assert captured_build_kwargs["config_path"] == "configs/dev/backtest.yaml"
    assert captured_build_kwargs["instance_index"] == 3
    assert captured_build_kwargs["metrics_port"] == 9307


def test_run_async_uses_default_metrics_port_base_plus_instance_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify worker entrypoint derives default per-instance metrics_port from base port and index.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        When CLI does not override the base port, the worker uses the default base plus
        `instance_index`.
    Raises:
        AssertionError: If default metrics_port does not remain deterministic per instance.
    Side Effects:
        Monkeypatches startup dependencies and runs the async entrypoint once.
    """
    captured_build_kwargs: dict[str, object] = {}

    class _FakeApp:
        """
        Minimal worker app stub that exits immediately after startup wiring.
        """

        locked_by = "hostname=test-host;pid=123;instance_index=2"

        async def run(self, stop_event: object) -> None:
            """
            Record one startup invocation without entering a long-lived claim loop.

            Args:
                stop_event: Cooperative shutdown signal passed by the entrypoint.
            Returns:
                None.
            Assumptions:
                Startup test only needs to observe that the app would have been run.
            Raises:
                None.
            Side Effects:
                Stores the received stop event for later assertions.
            """
            captured_build_kwargs["stop_event"] = stop_event

    monkeypatch.setattr(
        main_module,
        "_resolve_config_path",
        lambda *, config_path, environ: Path(config_path or "configs/dev/backtest.yaml"),
    )
    monkeypatch.setattr(
        main_module,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(jobs=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(main_module, "_install_signal_handlers", lambda _stop_event: None)
    monkeypatch.setattr(
        main_module,
        "build_backtest_job_runner_app",
        lambda **kwargs: captured_build_kwargs.update(kwargs) or _FakeApp(),
    )

    exit_code = main_module.asyncio.run(
        main_module._run_async(
            config_path="configs/dev/backtest.yaml",
            metrics_port=None,
            instance_index=2,
        )
    )

    assert exit_code == 0
    assert captured_build_kwargs["instance_index"] == 2
    assert captured_build_kwargs["metrics_port"] == 9206
