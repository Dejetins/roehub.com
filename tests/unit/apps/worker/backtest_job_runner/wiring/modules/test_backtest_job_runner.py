from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

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
    runtime_planner = cast(Any, captured_runner_kwargs["runtime_planner"])
    assert app.instance_index == 2
    assert app.metrics_port == 9206
    assert "hostname=" in app.locked_by
    assert ";pid=" in app.locked_by
    assert app.locked_by.endswith("instance_index=2")
    assert "candle_timeline_builder" not in captured_runner_kwargs
    assert runtime_planner.kwargs == {
        "execution_profiles": runtime_config.execution_profiles,
        "adaptive_selector_policy": runtime_config.adaptive_selector_policy,
    }
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


def test_backtest_job_runner_app_keeps_single_claim_loop_and_one_claimed_job_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify one worker process alternates `claim_next(...)` and claimed-job execution sequentially.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Queue concurrency must remain at the fleet level, so one process must not claim a second
        job before finishing the first claimed attempt.
    Raises:
        AssertionError: If the app reorders claim/process events or leaves the active gauge set.
    Side Effects:
        Monkeypatches metrics HTTP startup for isolated claim-loop coverage.
    """
    events: list[tuple[str, object]] = []
    claimed_jobs = [
        SimpleNamespace(job_id=UUID("00000000-0000-0000-0000-000000000101"), attempt=1),
        SimpleNamespace(job_id=UUID("00000000-0000-0000-0000-000000000102"), attempt=2),
    ]
    stop_event = asyncio.Event()

    class _FakeLeaseRepository:
        """
        Lease repository fake returning two claimed jobs in deterministic order.
        """

        def claim_next(
            self,
            *,
            now: object,
            locked_by: str,
            lease_seconds: int,
        ) -> object | None:
            """
            Return the next claimed job and record claim-loop ordering metadata.

            Args:
                now: Claim timestamp payload.
                locked_by: Active lease owner literal.
                lease_seconds: Lease TTL.
            Returns:
                object | None: Next claimed job fixture or `None` when exhausted.
            Assumptions:
                Test drives stop_event before the app would need to poll after the second job.
            Raises:
                AssertionError: If the claim loop mutates the worker identity unexpectedly.
            Side Effects:
                Appends the observed claim event to the in-memory event log.
            """
            _ = now, lease_seconds
            assert locked_by == "worker-test-1"
            if not claimed_jobs:
                events.append(("claim_next", None))
                return None
            claimed_job = claimed_jobs.pop(0)
            events.append(("claim_next", claimed_job.job_id))
            return claimed_job

    class _FakeRunnerUseCase:
        """
        Runner fake recording single-job processing order for one worker process.
        """

        def process_claimed_job(
            self,
            *,
            job: object,
            locked_by: str,
        ) -> worker_module.BacktestJobRunReportV1:
            """
            Record one claimed-job execution and stop after the second attempt.

            Args:
                job: Already-claimed job fixture.
                locked_by: Active lease owner literal.
            Returns:
                BacktestJobRunReportV1: Succeeded report for the provided claimed job.
            Assumptions:
                The app should not invoke this method concurrently for more than one claimed job.
            Raises:
                AssertionError: If the worker identity changes across attempts.
            Side Effects:
                Appends the observed process event and sets stop_event after the second job.
            """
            assert locked_by == "worker-test-1"
            claimed_job = cast(Any, job)
            events.append(("process_claimed_job", claimed_job.job_id))
            if claimed_job.job_id == UUID("00000000-0000-0000-0000-000000000102"):
                stop_event.set()
            return worker_module.BacktestJobRunReportV1(
                job_id=claimed_job.job_id,
                attempt=claimed_job.attempt,
                status="succeeded",
            )

    monkeypatch.setattr(worker_module, "start_http_server", lambda *args, **kwargs: None)
    app = worker_module.BacktestJobRunnerApp(
        claim_poll_seconds=5.0,
        lease_seconds=60,
        instance_index=0,
        locked_by="worker-test-1",
        lease_repository=cast(Any, _FakeLeaseRepository()),
        runner_use_case=cast(Any, _FakeRunnerUseCase()),
        metrics=worker_module.BacktestJobRunnerMetrics(
            registry=worker_module.CollectorRegistry()
        ),
        metrics_port=9204,
    )

    asyncio.run(asyncio.wait_for(app.run(stop_event), timeout=1.0))

    assert events == [
        ("claim_next", UUID("00000000-0000-0000-0000-000000000101")),
        ("process_claimed_job", UUID("00000000-0000-0000-0000-000000000101")),
        ("claim_next", UUID("00000000-0000-0000-0000-000000000102")),
        ("process_claimed_job", UUID("00000000-0000-0000-0000-000000000102")),
    ]
    active_claimed_jobs_metric = next(
        metric
        for metric in app.metrics.registry.collect()
        if metric.name == "backtest_job_runner_active_claimed_jobs"
    )
    assert active_claimed_jobs_metric.samples[0].value == 0.0


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
