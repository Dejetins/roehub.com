from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from apps.worker.notification_dispatcher.wiring.modules.notification_dispatcher import (
    build_notification_dispatcher_app,
    load_notification_dispatcher_runtime_config,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notification-dispatcher")
    parser.add_argument(
        "--config",
        default="configs/dev/notifications.yaml",
        help="Path to notifications.yaml",
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


async def _run_async(config_path: Path) -> int:
    runtime_config = load_notification_dispatcher_runtime_config(config_path=config_path)
    if not runtime_config.enabled:
        logging.getLogger(__name__).info("notification-dispatcher disabled by config")
        return 0

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_notification_dispatcher_app(config_path=config_path, environ=os.environ)
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_async(config_path=Path(args.config)))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("notification-dispatcher failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
