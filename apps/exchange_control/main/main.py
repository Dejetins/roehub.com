from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from trading.contexts.exchange_control.adapters.inbound.http.app import (
    EXCHANGE_CONTROL_DEFAULT_HOST,
    EXCHANGE_CONTROL_METRICS_PORT,
    ExchangeControlRuntimeConfig,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exchange-control")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="HTTP metrics/readiness port")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        config = ExchangeControlRuntimeConfig.from_environ(
            environ=os.environ,
            bind_host=args.host or EXCHANGE_CONTROL_DEFAULT_HOST,
            metrics_port=args.port or EXCHANGE_CONTROL_METRICS_PORT,
        )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("exchange-control config validation failed")
        return 1

    uvicorn.run(
        "apps.exchange_control.main.app:app",
        host=config.bind_host,
        port=config.metrics_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
