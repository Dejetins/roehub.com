from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from apps.common.runtime_health import RuntimeHealthServer, RuntimeHealthState
from apps.worker.telegram_bot_worker.wiring.modules.telegram_bot_worker import (
    load_telegram_bot_worker_runtime_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telegram-bot-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9212)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_telegram_bot_worker_runtime_config(config_path=args.config)
    if config.enabled:
        raise RuntimeError("Telegram worker activation requires owner-provided OpenBao inputs")
    stop = threading.Event()
    for current in (signal.SIGINT, signal.SIGTERM):
        signal.signal(current, lambda *_args: stop.set())
    server = RuntimeHealthServer(
        host="0.0.0.0",
        port=args.port,
        state=RuntimeHealthState(
            service="telegram-bot-worker",
            ready=True,
            mode="disabled",
            reason="disabled_by_safe_default",
        ),
    )
    server.start()
    try:
        stop.wait()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
