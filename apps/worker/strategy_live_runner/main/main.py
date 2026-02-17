from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.worker.strategy_live_runner.wiring.modules import build_strategy_live_runner_app


def _configure_logging() -> None:
    """
    Configure process-wide logging defaults for strategy live-runner worker.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logging is configured once at process start.
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
    Build CLI parser for strategy live-runner worker process.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    Assumptions:
        Defaults point to local dev runtime config.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(prog="strategy-live-runner")
    parser.add_argument(
        "--config",
        default="configs/dev/strategy_live_runner.yaml",
        help="Path to strategy_live_runner.yaml",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9203,
        help="Prometheus metrics HTTP port",
    )
    return parser


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


async def _run_async(config_path: str, metrics_port: int) -> int:
    """
    Build and run strategy live-runner worker until stop signal.

    Args:
        config_path: Runtime config path.
        metrics_port: Prometheus endpoint port.
    Returns:
        int: Process exit code.
    Assumptions:
        Required storage/redis credentials are provided in environment variables.
    Raises:
        Exception: Propagates wiring/runtime errors to caller.
    Side Effects:
        Starts worker runtime loop and metrics endpoint.
    """
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_strategy_live_runner_app(
        config_path=config_path,
        environ=os.environ,
        metrics_port=metrics_port,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Entrypoint for strategy live-runner worker process.

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
        return asyncio.run(_run_async(config_path=args.config, metrics_port=args.metrics_port))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("strategy-live-runner failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
