from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestAiConfigRepository,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiTrainingExportUseCase,
)

_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest-ai-configurator-training-export")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of labeled rows to export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write scrubbed JSONL to this path instead of stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dsn = os.environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    if not dsn:
        raise ValueError(f"{_STRATEGY_PG_DSN_KEY} is required for training export")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")
    repository = PostgresBacktestAiConfigRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=dsn)
    )
    payload = BacktestAiTrainingExportUseCase(source=repository).export_jsonl(
        limit=args.limit
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

