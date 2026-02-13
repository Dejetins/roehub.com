from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_index_reader import (  # noqa: E501
    ClickHouseCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.application.use_cases.rest_catchup_1m import (
    RestCatchUp1mReport,
    RestCatchUp1mUseCase,
)
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RestCatchUpCliReport:
    instruments_total: int
    instruments_ok: int
    instruments_failed: int


class RestCatchUp1mCli:
    def run(self, argv: Sequence[str]) -> int:
        """
        Execute rest-catchup command for one instrument or whole enabled universe.

        Parameters:
        - argv: command-line args without program name.

        Returns:
        - Process exit code (`0` on success, `2` when at least one instrument failed).

        Assumptions/Invariants:
        - Runtime config contains valid ClickHouse and ingestion settings.
        - Inter-instrument delay is read from `ingestion.rest_inter_instrument_delay_s`.

        Errors/Exceptions:
        - Propagates argument/config/database initialization errors.

        Side effects:
        - Executes REST catchup use-case and writes reports to stdout/logs.
        - Sleeps between instruments when configured and running in all-instruments mode.
        """
        p = _build_parser()
        ns = p.parse_args(list(argv))

        cfg = load_market_data_runtime_config(Path(ns.config))
        clock = SystemClock()
        http = RequestsHttpClient()

        # ClickHouse
        settings = ClickHouseSettingsLoader(os.environ).load()
        client = _clickhouse_client(settings)
        gw = ClickHouseConnectGateway(client)

        index = ClickHouseCanonicalCandleIndexReader(gateway=gw, database=settings.database)
        writer = ClickHouseRawKlineWriter(gateway=gw, database=settings.database)

        ingest_id = uuid4()

        source = RestCandleIngestSource(cfg=cfg, clock=clock, http=http, ingest_id=ingest_id)

        uc = RestCatchUp1mUseCase(
            index=index,
            source=source,
            writer=writer,
            clock=clock,
            max_days_per_insert=cfg.backfill.max_days_per_insert,
            batch_size=int(ns.batch_size),
            ingest_id=ingest_id,
        )

        if ns.all_from_ref_instruments:
            instruments = _load_enabled_instruments(gw, settings.database)
            ok = 0
            fail = 0
            for idx, inst in enumerate(instruments):
                try:
                    rep = uc.run(inst)
                    ok += 1
                    _print_report(rep, fmt=ns.report_format, instrument_id=inst)
                except Exception:  # noqa: BLE001
                    log.exception("rest-catchup failed for %s", inst)
                    fail += 1
                _sleep_between_instruments(
                    delay_s=cfg.ingestion.rest_inter_instrument_delay_s,
                    has_next=idx < len(instruments) - 1,
                )

            summary = RestCatchUpCliReport(
                instruments_total=len(instruments),
                instruments_ok=ok,
                instruments_failed=fail,
            )
            if ns.report_format == "json":
                print(json.dumps(asdict(summary), ensure_ascii=False))
            else:
                print(
                    "rest-catchup summary:\n"
                    f"- instruments_total: {summary.instruments_total}\n"
                    f"- ok: {summary.instruments_ok}\n"
                    f"- failed: {summary.instruments_failed}\n"
                )
            return 0 if fail == 0 else 2

        if ns.market_id is None or ns.symbol is None:
            raise SystemExit("Either --all-from-ref-instruments or (--market-id and --symbol) must be provided")  # noqa: E501

        inst = InstrumentId(MarketId(int(ns.market_id)), Symbol(str(ns.symbol)))
        rep = uc.run(inst)
        _print_report(rep, fmt=ns.report_format, instrument_id=inst)
        return 0


def _sleep_between_instruments(*, delay_s: float, has_next: bool) -> None:
    """
    Sleep between instrument runs in sequential CLI mode.

    Parameters:
    - delay_s: configured pause duration in seconds.
    - has_next: whether loop has another instrument after current one.

    Returns:
    - None.

    Assumptions/Invariants:
    - Delay is non-negative and validated by runtime config parser.

    Errors/Exceptions:
    - Propagates `ValueError` from `time.sleep` if called with invalid delay.

    Side effects:
    - Blocks current thread for `delay_s` seconds when conditions are met.
    """
    if delay_s <= 0 or not has_next:
        return
    time.sleep(delay_s)


def _load_enabled_instruments(gw, database: str) -> list[InstrumentId]:
    q = f"""
    SELECT market_id, symbol
    FROM {database}.ref_instruments
    WHERE is_tradable = 1
    """
    rows = gw.select(q, {})
    out: list[InstrumentId] = []
    for r in rows:
        out.append(InstrumentId(MarketId(int(r["market_id"])), Symbol(str(r["symbol"]))))
    return out


def _print_report(rep: RestCatchUp1mReport, *, fmt: str, instrument_id: InstrumentId) -> None:
    """
    Print one rest-catchup report in text or JSON format.

    Parameters:
    - rep: rest catch-up report DTO returned by use-case.
    - fmt: output format, either `text` or `json`.
    - instrument_id: instrument for which the report was produced.

    Returns:
    - None.

    Assumptions/Invariants:
    - report structure matches `RestCatchUp1mReport`.
    - `to_dict()` contains only JSON primitives.

    Errors/Exceptions:
    - Propagates `TypeError` if a non-serializable value appears unexpectedly.

    Side effects:
    - Writes report to stdout.
    """
    payload = rep.to_dict()
    payload["instrument_id"] = str(instrument_id)

    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False))
        return

    print(
        "rest-catchup report:\n"
        f"- instrument_id: {payload['instrument_id']}\n"
        f"- tail_start: {payload['tail_start']}\n"
        f"- tail_end: {payload['tail_end']}\n"
        f"- tail_rows_read: {payload['tail_rows_read']}\n"
        f"- tail_rows_written: {payload['tail_rows_written']}\n"
        f"- tail_batches: {payload['tail_batches']}\n"
        f"- gap_scan_start: {payload['gap_scan_start']}\n"
        f"- gap_scan_end: {payload['gap_scan_end']}\n"
        f"- gap_days_scanned: {payload['gap_days_scanned']}\n"
        f"- gap_days_with_gaps: {payload['gap_days_with_gaps']}\n"
        f"- gap_ranges_filled: {payload['gap_ranges_filled']}\n"
        f"- gap_rows_read: {payload['gap_rows_read']}\n"
        f"- gap_rows_written: {payload['gap_rows_written']}\n"
        f"- gap_rows_skipped_existing: {payload['gap_rows_skipped_existing']}\n"
        f"- gap_batches: {payload['gap_batches']}\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rest-catchup")
    p.add_argument(
        "--config",
        default="configs/dev/market_data.yaml",
        help="Path to market_data.yaml (default: configs/dev/market_data.yaml)",
    )
    p.add_argument("--market-id", type=int, default=None, help="MarketId (when running single instrument)")  # noqa: E501
    p.add_argument("--symbol", type=str, default=None, help="Symbol (when running single instrument)")  # noqa: E501
    p.add_argument(
        "--all-from-ref-instruments",
        action="store_true",
        help="Run rest-catchup for all enabled instruments in ClickHouse ref_instruments",
    )
    p.add_argument("--batch-size", type=int, default=10000, help="Raw insert batch size (default: 10000)")  # noqa: E501
    p.add_argument("--report-format", choices=("text", "json"), default="text", help="Output format")  # noqa: E501
    return p
