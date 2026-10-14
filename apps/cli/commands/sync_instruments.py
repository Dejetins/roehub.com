from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from apps.cli.wiring.db.postgres import market_gateway
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.config.whitelist import (
    load_whitelist_rows_from_csv,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import (
    PostgresInstrumentRefWriter,
    PostgresMarketRefWriter,
)
from trading.contexts.market_data.application.dto.reference_data import WhitelistInstrumentRow
from trading.contexts.market_data.application.use_cases.seed_ref_market import SeedRefMarketUseCase
from trading.contexts.market_data.application.use_cases.sync_whitelist_to_ref_instruments import (
    SyncWhitelistToRefInstrumentsUseCase,
)
from trading.platform.time.system_clock import SystemClock

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyncInstrumentsReport:
    ref_market_inserted: int
    ref_instruments_rows_total: int
    ref_instruments_upserted: int
    ref_instruments_skipped_unchanged: int
    enabled_count: int
    disabled_count: int


class SyncInstrumentsCli:
    def run(self, argv: Sequence[str]) -> int:
        parser = _build_parser()
        ns = parser.parse_args(list(argv))

        config_path = Path(ns.config)
        whitelist_path = Path(ns.whitelist)

        cfg = load_market_data_runtime_config(config_path)
        known_market_ids = set(cfg.market_ids())

        clock = SystemClock()

        # PostgreSQL (reuse existing CLI wiring env loader)
        gateway = market_gateway(os.environ, workload="reference")

        market_writer = PostgresMarketRefWriter(gateway=gateway)
        instr_writer = PostgresInstrumentRefWriter(gateway=gateway)

        # 1) seed ref_market
        seed_uc = SeedRefMarketUseCase(writer=market_writer, clock=clock)
        seed_report = seed_uc.run()

        # 2) sync whitelist -> ref_instruments (including disabled)
        wl_rows = load_whitelist_rows_from_csv(whitelist_path)
        rows = [
            WhitelistInstrumentRow(instrument_id=r.instrument_id, is_enabled=r.is_enabled)
            for r in wl_rows
        ]

        sync_uc = SyncWhitelistToRefInstrumentsUseCase(
            writer=instr_writer,
            clock=clock,
            known_market_ids=known_market_ids,
        )
        sync_report = sync_uc.run(rows)

        report = SyncInstrumentsReport(
            ref_market_inserted=seed_report.inserted,
            ref_instruments_rows_total=sync_report.rows_total,
            ref_instruments_upserted=sync_report.rows_upserted,
            ref_instruments_skipped_unchanged=sync_report.rows_skipped_unchanged,
            enabled_count=sync_report.enabled_count,
            disabled_count=sync_report.disabled_count,
        )

        if ns.report_format == "json":
            print(json.dumps(asdict(report), ensure_ascii=False))
        else:
            print(
                "sync-instruments report:\n"
                f"- ref_market inserted: {report.ref_market_inserted}\n"
                f"- ref_instruments rows_total: {report.ref_instruments_rows_total}\n"
                f"- ref_instruments upserted: {report.ref_instruments_upserted}\n"
                f"- ref_instruments skipped_unchanged: {report.ref_instruments_skipped_unchanged}\n"
                f"- enabled: {report.enabled_count}\n"
                f"- disabled: {report.disabled_count}\n"
            )

        return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sync-instruments")
    p.add_argument(
        "--config",
        default="configs/dev/market_data.yaml",
        help="Path to market_data.yaml (default: configs/dev/market_data.yaml)",
    )
    p.add_argument(
        "--whitelist",
        default="configs/dev/whitelist.csv",
        help="Path to whitelist.csv (default: configs/dev/whitelist.csv)",
    )
    p.add_argument(
        "--report-format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    return p
