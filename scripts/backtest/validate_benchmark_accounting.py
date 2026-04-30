from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.application.services.v2.benchmark_accounting import (  # noqa: E402
    validate_canonical_benchmark_json,
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate backtest benchmark runner accounting stage rules."
    )
    parser.add_argument(
        "--canonical-json",
        type=Path,
        default=DEFAULT_CANONICAL_JSON,
        help="Canonical notebook benchmark_results.json path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for the validation evidence JSON.",
    )
    args = parser.parse_args(argv)

    summary = validate_canonical_benchmark_json(args.canonical_json)
    rendered = _render_json(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _render_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
