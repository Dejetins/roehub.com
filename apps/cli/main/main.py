from __future__ import annotations

import logging
import sys

from apps.cli.commands.backfill_1m import Backfill1mCli


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = argv if argv is not None else sys.argv[1:]
    return Backfill1mCli().run(args)


if __name__ == "__main__":
    raise SystemExit(main())
