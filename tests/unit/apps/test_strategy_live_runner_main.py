from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from apps.worker.strategy_live_runner.main import main as strategy_live_runner_main


class _NoOpApp:
    """
    No-op app stub used to isolate strategy worker entrypoint tests.
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


def test_run_async_exits_zero_when_live_worker_disabled(monkeypatch) -> None:
    """
    Verify worker entrypoint exits with code 0 when `strategy.live_worker.enabled=false`.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Disabled mode must skip app wiring and runtime loop startup.
    Raises:
        AssertionError: If build function is called or exit code differs from 0.
    Side Effects:
        None.
    """
    calls = {"build": 0}

    monkeypatch.setattr(
        strategy_live_runner_main,
        "resolve_strategy_config_path",
        lambda *, environ, cli_config_path: Path("configs/dev/strategy.yaml"),
    )
    monkeypatch.setattr(
        strategy_live_runner_main,
        "load_strategy_live_runner_runtime_config",
        lambda path, *, environ: SimpleNamespace(
            live_worker_enabled=False,
            metrics_port=9203,
        ),
    )

    def _build_app(**_kwargs) -> _NoOpApp:
        calls["build"] += 1
        return _NoOpApp()

    monkeypatch.setattr(strategy_live_runner_main, "build_strategy_live_runner_app", _build_app)
    monkeypatch.setattr(
        strategy_live_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    exit_code = asyncio.run(
        strategy_live_runner_main._run_async(
            config_path=None,
            metrics_port=None,
        )
    )

    assert exit_code == 0
    assert calls["build"] == 0


def test_run_async_metrics_port_cli_override_has_priority(monkeypatch) -> None:
    """
    Verify CLI `--metrics-port` override wins over runtime config metrics port.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime config already passed validation before app wiring.
    Raises:
        AssertionError: If build function receives wrong metrics port.
    Side Effects:
        None.
    """
    received_metrics_ports: list[int] = []

    monkeypatch.setattr(
        strategy_live_runner_main,
        "resolve_strategy_config_path",
        lambda *, environ, cli_config_path: Path("configs/dev/strategy.yaml"),
    )
    monkeypatch.setattr(
        strategy_live_runner_main,
        "load_strategy_live_runner_runtime_config",
        lambda path, *, environ: SimpleNamespace(
            live_worker_enabled=True,
            metrics_port=9300,
        ),
    )

    def _build_app(*, config_path: str, environ, metrics_port: int) -> _NoOpApp:
        received_metrics_ports.append(metrics_port)
        return _NoOpApp()

    monkeypatch.setattr(strategy_live_runner_main, "build_strategy_live_runner_app", _build_app)
    monkeypatch.setattr(
        strategy_live_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    cli_override_exit_code = asyncio.run(
        strategy_live_runner_main._run_async(
            config_path=None,
            metrics_port=9400,
        )
    )
    yaml_default_exit_code = asyncio.run(
        strategy_live_runner_main._run_async(
            config_path=None,
            metrics_port=None,
        )
    )

    assert cli_override_exit_code == 0
    assert yaml_default_exit_code == 0
    assert received_metrics_ports == [9400, 9300]
