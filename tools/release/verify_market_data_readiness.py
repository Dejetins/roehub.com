#!/usr/bin/env python3
"""Prove live market-data ingestion at a Docker Compose boundary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


class MarketDataReadinessError(RuntimeError):
    """Raised when live public-market ingestion has not reached readiness."""


def _metrics_by_name(payload: str) -> dict[str, float]:
    """Aggregate Prometheus samples by metric name without retaining labels."""
    metrics: dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name")
        metrics[name] = metrics.get(name, 0.0) + float(match.group("value"))
    return metrics


def _readiness_snapshot(
    *,
    ws_metrics: dict[str, float],
    scheduler_metrics: dict[str, float],
    newest_candle_timestamp: float | None,
    now_timestamp: float,
    max_candle_age_seconds: float,
    min_ws_messages: int,
) -> dict[str, Any]:
    """Return a serialisable verdict for the required live-ingestion signals."""
    candle_age_seconds = (
        None
        if newest_candle_timestamp is None
        else max(0.0, now_timestamp - newest_candle_timestamp)
    )
    checks = {
        "ws_connection": ws_metrics.get("ws_connected", 0.0) >= 1.0,
        "ws_messages": ws_metrics.get("ws_messages_total", 0.0) >= min_ws_messages,
        "raw_insert": ws_metrics.get("insert_rows_total", 0.0) >= 1.0,
        "scheduler_sync_and_enrichment": scheduler_metrics.get("scheduler_job_runs_total", 0.0)
        >= 2.0,
        "ws_errors": ws_metrics.get("ws_errors_total", 0.0) == 0.0,
        "insert_errors": ws_metrics.get("insert_errors_total", 0.0) == 0.0,
        "rest_fill_errors": ws_metrics.get("rest_fill_errors_total", 0.0) == 0.0,
        "scheduler_errors": scheduler_metrics.get("scheduler_job_errors_total", 0.0) == 0.0,
        "fresh_candle": candle_age_seconds is not None
        and candle_age_seconds <= max_candle_age_seconds,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "ws": {
            name: ws_metrics.get(name, 0.0)
            for name in (
                "ws_connected",
                "ws_messages_total",
                "insert_rows_total",
                "ws_errors_total",
                "insert_errors_total",
                "rest_fill_errors_total",
            )
        },
        "scheduler": {
            name: scheduler_metrics.get(name, 0.0)
            for name in (
                "scheduler_job_runs_total",
                "scheduler_job_errors_total",
            )
        },
        "newest_candle_timestamp": newest_candle_timestamp,
        "candle_age_seconds": candle_age_seconds,
    }


def _run(command: list[str], *, timeout: float = 30.0) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        raise MarketDataReadinessError(
            f"command failed ({error.returncode}): {' '.join(command)}; "
            f"stderr={(error.stderr or '').strip()!r}"
        ) from error
    return completed.stdout


def _compose(project: str, files: Sequence[Path]) -> list[str]:
    command = ["docker", "compose", "-p", project]
    for compose_file in files:
        command.extend(["-f", str(compose_file)])
    return command


def _service_metrics(compose: list[str], service: str, port: int) -> dict[str, float]:
    payload = _run(
        [
            *compose,
            "exec",
            "-T",
            service,
            "python",
            "-c",
            (
                "from urllib.request import urlopen; "
                f"print(urlopen('http://127.0.0.1:{port}/metrics', timeout=3).read().decode())"
            ),
        ]
    )
    return _metrics_by_name(payload)


def _newest_candle_timestamp(compose: list[str]) -> float | None:
    output = _run(
        [
            *compose,
            "exec",
            "-T",
            "clickhouse",
            "sh",
            "-ec",
            (
                "clickhouse-client --user default "
                "--password \"$(cat /run/roehub-secrets/clickhouse-password)\" "
                "--query \"SELECT toUnixTimestamp(max(ts_close)) "
                "FROM roehub.canonical_candles_1m\""
            ),
        ]
    ).strip()
    if output in {"", "0"}:
        return None
    return float(output)


def _collect(compose: list[str], args: argparse.Namespace) -> dict[str, Any]:
    now = time.time()
    return _readiness_snapshot(
        ws_metrics=_service_metrics(compose, "market-data-ws", 9201),
        scheduler_metrics=_service_metrics(compose, "market-data-scheduler", 9202),
        newest_candle_timestamp=_newest_candle_timestamp(compose),
        now_timestamp=now,
        max_candle_age_seconds=args.max_candle_age_seconds,
        min_ws_messages=args.min_ws_messages,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--compose",
        type=Path,
        action="append",
        required=True,
        help="Compose file; pass in the exact precedence order used for the runtime.",
    )
    parser.add_argument("--max-candle-age-seconds", type=float, default=180.0)
    parser.add_argument("--min-ws-messages", type=int, default=1)
    parser.add_argument("--wait-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    if args.max_candle_age_seconds <= 0 or args.min_ws_messages < 1:
        parser.error("candle age must be positive and minimum messages must be at least one")
    if args.wait_seconds < 0 or args.poll_seconds <= 0:
        parser.error("wait seconds must be non-negative and poll seconds must be positive")
    return args


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    compose = _compose(args.project, args.compose)
    deadline = time.monotonic() + args.wait_seconds
    last_snapshot: dict[str, Any] | None = None
    while True:
        try:
            snapshot = _collect(compose, args)
        except MarketDataReadinessError as error:
            snapshot = {"ready": False, "error": str(error)}
        snapshot["checked_at"] = datetime.now(UTC).isoformat()
        last_snapshot = snapshot
        if snapshot["ready"] or time.monotonic() >= deadline:
            break
        time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))

    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(last_snapshot, indent=2, sort_keys=True) + "\n")
    print(json.dumps(last_snapshot, indent=2, sort_keys=True))
    if not last_snapshot["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
