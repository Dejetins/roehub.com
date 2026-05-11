from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.worker.backtest_job_runner.wiring.modules import (
    build_backtest_job_runner_app,
    load_backtest_job_runner_runtime_config,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest-job-runner")
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Prometheus metrics HTTP port (CLI override has highest priority)",
    )
    return parser


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _mark_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _mark_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: _mark_stop())


async def _run_async(metrics_port: int | None) -> int:
    runtime_config = load_backtest_job_runner_runtime_config(environ=os.environ)
    if not runtime_config.enabled:
        logging.getLogger(__name__).info("backtest-job-runner disabled by config")
        return 0
    if metrics_port is not None and metrics_port <= 0:
        raise ValueError("--metrics-port must be > 0 when provided")

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_backtest_job_runner_app(
        environ=os.environ,
        runtime_config=runtime_config,
        metrics_port=metrics_port or runtime_config.metrics_port,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_async(metrics_port=args.metrics_port))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("backtest-job-runner failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
