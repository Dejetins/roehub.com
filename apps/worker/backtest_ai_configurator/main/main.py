from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.worker.backtest_ai_configurator.wiring.modules import (
    build_backtest_ai_configurator_worker_app,
    load_backtest_ai_configurator_worker_runtime_config,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest-ai-configurator-worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one claimed job, then exit",
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


async def _run_async(*, once: bool) -> int:
    runtime_config = load_backtest_ai_configurator_worker_runtime_config(
        environ=os.environ
    )
    if not runtime_config.enabled:
        logging.getLogger(__name__).info(
            "backtest-ai-configurator-worker disabled by config"
        )
        return 0
    if once:
        runtime_config = type(runtime_config)(
            enabled=runtime_config.enabled,
            poll_interval_seconds=runtime_config.poll_interval_seconds,
            empty_backoff_seconds=runtime_config.empty_backoff_seconds,
            heartbeat_interval_seconds=runtime_config.heartbeat_interval_seconds,
            max_jobs_per_process=1,
        )
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_backtest_ai_configurator_worker_app(
        environ=os.environ,
        runtime_config=runtime_config,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_async(once=args.once))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("backtest-ai-configurator-worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
