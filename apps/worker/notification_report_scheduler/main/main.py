from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path
from typing import Mapping

import yaml

from apps.common.runtime_health import RuntimeHealthServer, RuntimeHealthState
from apps.worker.notification_report_scheduler.wiring import (
    build_notification_report_scheduler,
)


def _config(path: Path) -> tuple[bool, int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("notification report scheduler config must be a mapping")
    notifications = payload.get("notifications") or {}
    if not isinstance(notifications, Mapping):
        raise ValueError("notifications config must be a mapping")
    scheduler = notifications.get("report_scheduler") or {}
    if not isinstance(scheduler, Mapping):
        raise ValueError("notifications.report_scheduler must be a mapping")
    enabled = scheduler.get("enabled", False)
    poll_interval = scheduler.get("poll_interval_seconds", 60)
    if not isinstance(enabled, bool):
        raise ValueError("report scheduler enabled must be bool")
    if not isinstance(poll_interval, int) or isinstance(poll_interval, bool) or poll_interval <= 0:
        raise ValueError("report scheduler poll interval must be positive")
    return enabled, poll_interval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notification-report-scheduler")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9211)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    enabled, _poll_interval = _config(args.config)
    if enabled:
        raise RuntimeError(
            "notification report scheduler requires an installed stats source adapter"
        )
    if not callable(build_notification_report_scheduler):
        raise RuntimeError("notification report scheduler wiring is unavailable")
    stop = threading.Event()
    for current in (signal.SIGINT, signal.SIGTERM):
        signal.signal(current, lambda *_args: stop.set())
    server = RuntimeHealthServer(
        host="0.0.0.0",
        port=args.port,
        state=RuntimeHealthState(
            service="notification-report-scheduler",
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
