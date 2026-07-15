from __future__ import annotations

import logging
import sys

from apps.cli.commands.artifacts import ArtifactsCli
from apps.cli.commands.backfill_1m import Backfill1mCli
from apps.cli.commands.backtest_artifact_publish import BacktestArtifactPublishCli
from apps.cli.commands.funding_rate_catchup import FundingRateCatchupCli
from apps.cli.commands.local_auth_bootstrap import LocalAuthBootstrapCli
from apps.cli.commands.plugins import PluginsCli
from apps.cli.commands.providers import ProvidersCli
from apps.cli.commands.rest_catchup_1m import RestCatchUp1mCli
from apps.cli.commands.sync_instruments import SyncInstrumentsCli
from apps.cli.commands.telegram import TelegramCli


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
            "  artifacts install <bundle> [args...]\n"
            "  backtest-artifact-publish [args...]\n"
            "  sync-instruments [args...]\n"
            "  rest-catchup [args...]\n"
            "  funding-rate-catchup [args...]\n"
            "  local-auth-bootstrap --output-file PATH\n"
            "  providers install|add [args...]\n"
            "  plugins init|validate|install|update|rollback|doctor [args...]\n"
            "  telegram connect [args...]\n"
            "\n"
            "Back-compat: if no command is provided, arguments are passed to backfill-1m."
        )
        return 2

    cmd = args[0]
    rest = args[1:]

    if cmd == "artifacts":
        return ArtifactsCli().run(rest)
    if cmd == "backfill-1m":
        return Backfill1mCli().run(rest)
    if cmd == "backtest-artifact-publish":
        return BacktestArtifactPublishCli().run(rest)
    if cmd == "sync-instruments":
        return SyncInstrumentsCli().run(rest)
    if cmd == "rest-catchup":
        return RestCatchUp1mCli().run(rest)
    if cmd == "funding-rate-catchup":
        return FundingRateCatchupCli().run(rest)
    if cmd == "local-auth-bootstrap":
        return LocalAuthBootstrapCli().run(rest)
    if cmd == "providers":
        return ProvidersCli().run(rest)
    if cmd == "plugins":
        return PluginsCli().run(rest)
    if cmd == "telegram":
        return TelegramCli().run(rest)

    # back-compat
    return Backfill1mCli().run(args)


if __name__ == "__main__":
    raise SystemExit(main())
