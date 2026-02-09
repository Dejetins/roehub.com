from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.worker.market_data_ws.wiring.modules import build_market_data_ws_app


def _configure_logging() -> None:
    """
    Configure process-wide structured logging defaults.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Logging is configured once at process start.

    Errors/Exceptions:
    - None.

    Side effects:
    - Sets root logging handlers/format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for market-data websocket worker process.

    Parameters:
    - None.

    Returns:
    - Configured argument parser.

    Assumptions/Invariants:
    - Defaults target local development setup.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    parser = argparse.ArgumentParser(prog="market-data-ws-worker")
    parser.add_argument(
        "--config",
        default="configs/dev/market_data.yaml",
        help="Path to market_data.yaml",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9201,
        help="Prometheus metrics HTTP port",
    )
    return parser


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """
    Install SIGTERM/SIGINT handlers that trigger cooperative shutdown.

    Parameters:
    - stop_event: shutdown event shared with runtime tasks.

    Returns:
    - None.

    Assumptions/Invariants:
    - Called from main asyncio event loop.

    Errors/Exceptions:
    - None.

    Side effects:
    - Registers process signal handlers.
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
    Build and run websocket worker until termination signal.

    Parameters:
    - config_path: runtime config path.
    - metrics_port: metrics endpoint port.

    Returns:
    - Process exit code.

    Assumptions/Invariants:
    - ClickHouse settings are supplied through environment variables.

    Errors/Exceptions:
    - Propagates wiring/runtime exceptions to caller.

    Side effects:
    - Starts websocket worker runtime and metrics endpoint.
    """
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    app = build_market_data_ws_app(
        config_path=config_path,
        environ=os.environ,
        metrics_port=metrics_port,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Entrypoint for market-data websocket worker process.

    Parameters:
    - argv: optional command-line arguments excluding program name.

    Returns:
    - Process exit code.

    Assumptions/Invariants:
    - This function is executed in standalone process context.

    Errors/Exceptions:
    - Returns non-zero on runtime failures.

    Side effects:
    - Initializes logging and runs asyncio loop.
    """
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_async(config_path=args.config, metrics_port=args.metrics_port))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("market-data-ws-worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

