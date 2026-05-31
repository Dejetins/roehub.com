from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from apps.exchange_execution.main.app import (
    EXCHANGE_EXECUTION_DEFAULT_HOST,
    EXCHANGE_EXECUTION_METRICS_PORT,
    resolve_runtime_settings,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exchange-execution")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="HTTP metrics/readiness port")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        settings = resolve_runtime_settings(environ=os.environ)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("exchange-execution config validation failed")
        return 1
    uvicorn.run(
        "apps.exchange_execution.main.app:app",
        host=args.host or settings.bind_host or EXCHANGE_EXECUTION_DEFAULT_HOST,
        port=args.port or settings.metrics_port or EXCHANGE_EXECUTION_METRICS_PORT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
