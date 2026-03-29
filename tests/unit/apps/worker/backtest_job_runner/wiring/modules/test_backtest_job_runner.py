from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

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

    def _fake_runner_use_case(**kwargs: object) -> object:
        captured_runner_kwargs.update(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(worker_module, "RunBacktestJobRunnerV1", _fake_runner_use_case)

    app = build_backtest_job_runner_app(
        config_path="configs/dev/backtest.yaml",
        environ={"STRATEGY_PG_DSN": "postgresql://local/test"},
        metrics_port=9204,
    )

    artifact_slot_resolver = cast(Any, captured_runner_kwargs["artifact_slot_resolver"])
    assert app.metrics_port == 9204
    assert "candle_timeline_builder" not in captured_runner_kwargs
    assert artifact_slot_resolver.artifact_loader.path_resolver.root == Path(
        "/tmp/backtest-artifacts-test"
    )
