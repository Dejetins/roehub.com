from __future__ import annotations

import logging
import sys

from apps.cli.commands.backfill_1m import Backfill1mCli
from apps.cli.commands.rest_catchup_1m import RestCatchUp1mCli
from apps.cli.commands.sync_instruments import SyncInstrumentsCli


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = argv if argv is not None else sys.argv[1:]

    # Backward compatibility:
    # если команда не указана — считаем что это backfill-1m (старое поведение).
    if not args:
        print(
            "Usage:\n"
            "  backfill-1m [args...]\n"
            "  sync-instruments [args...]\n"
            "  rest-catchup [args...]\n"
            "\n"
            "Back-compat: if no command is provided, arguments are passed to backfill-1m."
        )
        return 2

    cmd = args[0]
    rest = args[1:]

    if cmd == "backfill-1m":
        return Backfill1mCli().run(rest)
    if cmd == "sync-instruments":
        return SyncInstrumentsCli().run(rest)
    if cmd == "rest-catchup":
        return RestCatchUp1mCli().run(rest)

    # back-compat
    return Backfill1mCli().run(args)


if __name__ == "__main__":
    raise SystemExit(main())
