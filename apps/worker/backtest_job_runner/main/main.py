from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Mapping

from apps.worker.backtest_job_runner.wiring.modules import build_backtest_job_runner_app
from trading.contexts.backtest.adapters.outbound import (
    load_backtest_runtime_config,
    resolve_backtest_config_path,
)

_DEFAULT_METRICS_PORT = 9204


def _configure_logging() -> None:
    """
    Configure process-wide logging defaults for Backtest job-runner worker.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logging is configured once at process startup.
    Raises:
        None.
    Side Effects:
        Sets root logging handlers and format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for Backtest job-runner worker process.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    Assumptions:
        Config path defaults to environment-based runtime resolver.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(prog="backtest-job-runner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to backtest runtime config (backtest.yaml).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Prometheus metrics HTTP port (CLI override has highest priority).",
    )
    return parser


def _resolve_config_path(*, config_path: str | None, environ: Mapping[str, str]) -> Path:
    """
    Resolve runtime config path from CLI override or environment contract.

    Args:
        config_path: Optional CLI config path.
        environ: Runtime environment mapping.
    Returns:
        Path: Resolved `backtest.yaml` path.
    Assumptions:
        CLI override has highest precedence over env-based resolution.
    Raises:
        ValueError: If environment resolver receives unsupported env values.
    Side Effects:
        None.
    """
    if config_path is not None and config_path.strip():
        return Path(config_path)
    return resolve_backtest_config_path(environ=environ)


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """
    Install SIGTERM/SIGINT handlers that trigger cooperative shutdown.

    Args:
        stop_event: Shared shutdown event.
    Returns:
        None.
    Assumptions:
        Function runs inside active asyncio event loop.
    Raises:
        None.
    Side Effects:
        Registers process signal handlers.
    """
    loop = asyncio.get_running_loop()

    def _mark_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _mark_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: _mark_stop())


async def _run_async(config_path: str | None, metrics_port: int | None) -> int:
    """
    Build and run Backtest job-runner worker until stop signal.

    Args:
        config_path: Optional CLI runtime config path override.
        metrics_port: Optional CLI Prometheus endpoint port override.
    Returns:
        int: Process exit code.
    Assumptions:
        Required storage/compute credentials are provided in environment variables.
    Raises:
        Exception: Propagates wiring/runtime errors to caller.
    Side Effects:
        Loads runtime config and starts worker runtime loop.
    """
    resolved_config_path = _resolve_config_path(config_path=config_path, environ=os.environ)
    runtime_config = load_backtest_runtime_config(resolved_config_path)
    if not runtime_config.jobs.enabled:
        logging.getLogger(__name__).info(
            "component=backtest-job-runner status=disabled config_path=%s",
            resolved_config_path,
        )
        return 0

    if metrics_port is not None and metrics_port <= 0:
        raise ValueError("--metrics-port must be > 0 when provided")
    effective_metrics_port = (
        metrics_port if metrics_port is not None else _DEFAULT_METRICS_PORT
    )

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_backtest_job_runner_app(
        config_path=str(resolved_config_path),
        environ=os.environ,
        metrics_port=effective_metrics_port,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Entrypoint for Backtest job-runner worker process.

    Args:
        argv: Optional command-line arguments without program name.
    Returns:
        int: Process exit code.
    Assumptions:
        Function is executed in standalone process context.
    Raises:
        None.
    Side Effects:
        Initializes logging and runs asyncio loop.
    """
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(
            _run_async(
                config_path=args.config,
                metrics_port=args.metrics_port,
            )
        )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("backtest-job-runner failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
