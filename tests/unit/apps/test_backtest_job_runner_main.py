from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.worker.backtest_job_runner.main import main as backtest_job_runner_main


class _NoOpApp:
    """
    No-op app stub used to isolate backtest job-runner entrypoint tests.
    """

    async def run(self, _stop_event: asyncio.Event) -> None:
        """
        Complete immediately without side effects.

        Args:
            _stop_event: Cooperative stop event (unused in this stub).
        Returns:
            None.
        Assumptions:
            Entrypoint tests validate argument passing, not worker loop behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


def test_run_async_exits_zero_when_jobs_worker_disabled(monkeypatch) -> None:
    """
    Verify worker entrypoint exits with code 0 when `backtest.jobs.enabled=false`.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Disabled mode must skip app wiring and runtime loop startup.
    Raises:
        AssertionError: If app build is called or exit code differs from zero.
    Side Effects:
        None.
    """
    calls = {"build": 0}

    monkeypatch.setattr(
        backtest_job_runner_main,
        "_resolve_config_path",
        lambda *, config_path, environ: Path("configs/dev/backtest.yaml"),
    )
    monkeypatch.setattr(
        backtest_job_runner_main,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=False),
        ),
    )

    def _build_app(**_kwargs) -> _NoOpApp:
        calls["build"] += 1
        return _NoOpApp()

    monkeypatch.setattr(backtest_job_runner_main, "build_backtest_job_runner_app", _build_app)
    monkeypatch.setattr(
        backtest_job_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    exit_code = asyncio.run(
        backtest_job_runner_main._run_async(
            config_path=None,
            metrics_port=None,
        )
    )

    assert exit_code == 0
    assert calls["build"] == 0


def test_run_async_metrics_port_cli_override_has_priority(monkeypatch) -> None:
    """
    Verify CLI `--metrics-port` override wins over backtest worker default metrics port.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime config already passed validation before app wiring.
    Raises:
        AssertionError: If build function receives wrong metrics ports.
    Side Effects:
        None.
    """
    received_metrics_ports: list[int] = []

    monkeypatch.setattr(
        backtest_job_runner_main,
        "_resolve_config_path",
        lambda *, config_path, environ: Path("configs/dev/backtest.yaml"),
    )
    monkeypatch.setattr(
        backtest_job_runner_main,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=True),
        ),
    )

    def _build_app(*, config_path: str, environ, metrics_port: int) -> _NoOpApp:
        _ = config_path, environ
        received_metrics_ports.append(metrics_port)
        return _NoOpApp()

    monkeypatch.setattr(backtest_job_runner_main, "build_backtest_job_runner_app", _build_app)
    monkeypatch.setattr(
        backtest_job_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    cli_override_exit_code = asyncio.run(
        backtest_job_runner_main._run_async(
            config_path=None,
            metrics_port=9400,
        )
    )
    default_exit_code = asyncio.run(
        backtest_job_runner_main._run_async(
            config_path=None,
            metrics_port=None,
        )
    )

    assert cli_override_exit_code == 0
    assert default_exit_code == 0
    assert received_metrics_ports == [9400, backtest_job_runner_main._DEFAULT_METRICS_PORT]


def test_run_async_rejects_non_positive_metrics_port(monkeypatch) -> None:
    """
    Verify worker entrypoint rejects non-positive CLI metrics port override.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime config enables worker path and port validation happens before app build.
    Raises:
        AssertionError: If invalid metrics port does not raise ValueError.
    Side Effects:
        None.
    """
    monkeypatch.setattr(
        backtest_job_runner_main,
        "_resolve_config_path",
        lambda *, config_path, environ: Path("configs/dev/backtest.yaml"),
    )
    monkeypatch.setattr(
        backtest_job_runner_main,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=True),
        ),
    )

    with pytest.raises(ValueError, match="--metrics-port"):
        asyncio.run(
            backtest_job_runner_main._run_async(
                config_path=None,
                metrics_port=0,
            )
        )
