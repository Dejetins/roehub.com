from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.funding_rate_history_source import (
    RestFundingRateHistorySource,
)
from trading.contexts.market_data.adapters.outbound.config.instrument_key import (
    build_instrument_key,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.funding_rate_store import (  # noqa: E501
    ClickHouseFundingRateStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.application.dto import FundingInstrument
from trading.contexts.market_data.application.use_cases import (
    BackfillFundingRatesUseCase,
    SyncFuturesFundingUniverseUseCase,
)
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class FundingRateCatchupCli:
    def run(self, argv: Sequence[str]) -> int:
        parser = _build_parser()
        ns = parser.parse_args(list(argv))

        cfg = load_market_data_runtime_config(Path(ns.config))
        funding_cfg = cfg.scheduler.jobs.funding_rate_catchup
        clock = SystemClock()
        http = RequestsHttpClient()

        settings = ClickHouseSettingsLoader(os.environ).load()
        gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
        store = ClickHouseFundingRateStore(gateway=gateway, database=settings.database)
        source = RestFundingRateHistorySource(
            cfg=cfg,
            http=http,
            clock=clock,
            ingest_id=uuid4(),
            binance_standard_interval_hours=funding_cfg.binance_standard_interval_hours,
            allow_binance_funding_info_failure_fallback=(
                funding_cfg.allow_binance_funding_info_failure_fallback
            ),
        )

        sync_uc = SyncFuturesFundingUniverseUseCase(
            source=source,
            store=store,
            clock=clock,
            market_ids=(MarketId(2), MarketId(4)),
        )
        if ns.sync_universe:
            sync_report = sync_uc.run()
            if ns.report_format == "text":
                print(
                    "funding-universe sync:\n"
                    f"- instruments_total: {sync_report.instruments_total}\n"
                    f"- with_interval: {sync_report.instruments_with_interval}\n"
                    f"- missing_interval: {sync_report.instruments_missing_interval}\n"
                )

        catchup_uc = BackfillFundingRatesUseCase(
            source=source,
            writer=store,
            universe_store=store,
            clock=clock,
            tail_lookback_intervals=funding_cfg.tail_lookback_intervals,
            settlement_lag_minutes=funding_cfg.settlement_lag_minutes,
        )

        if ns.all_due:
            report = catchup_uc.run_due_universe(
                market_ids=(MarketId(2), MarketId(4)),
                dry_run=ns.dry_run,
            )
            _print_report(report.to_dict(), fmt=ns.report_format)
            return 0 if report.instruments_failed == 0 else 2

        if ns.market_id is None or ns.symbol is None:
            raise SystemExit("Either --all-due or (--market-id and --symbol) must be provided")

        instrument_id = InstrumentId(MarketId(int(ns.market_id)), Symbol(str(ns.symbol)))
        instrument = _load_or_build_instrument(
            cfg=cfg,
            store=store,
            clock=clock,
            instrument_id=instrument_id,
            funding_interval_minutes=ns.funding_interval_minutes,
            funding_interval_source=ns.funding_interval_source,
        )
        time_range = _parse_optional_time_range(start=ns.start, end=ns.end)
        report = catchup_uc.run_single(
            instrument=instrument,
            time_range=time_range,
            dry_run=ns.dry_run,
        )
        _print_report(report.to_dict(), fmt=ns.report_format)
        return 0 if report.instruments_failed == 0 else 2


def _load_or_build_instrument(
    *,
    cfg,
    store: ClickHouseFundingRateStore,
    clock: SystemClock,
    instrument_id: InstrumentId,
    funding_interval_minutes: int | None,
    funding_interval_source: str | None,
) -> FundingInstrument:
    stored = store.get_funding_instrument(instrument_id)
    if stored is not None:
        return stored
    if funding_interval_minutes is None:
        raise SystemExit(
            "Funding interval metadata is mandatory. Run --sync-universe first or pass "
            "--funding-interval-minutes for a bounded manual dry/run."
        )
    market = cfg.market_by_id(instrument_id.market_id)
    return FundingInstrument(
        instrument_id=instrument_id,
        instrument_key=build_instrument_key(cfg=cfg, instrument_id=instrument_id),
        exchange=market.exchange,
        market_type=market.market_type,
        status="MANUAL",
        is_tradable=1,
        base_asset=None,
        quote_asset=None,
        funding_interval_minutes=int(funding_interval_minutes),
        funding_interval_source=funding_interval_source or "manual_cli",
        funding_cap=None,
        funding_floor=None,
        updated_at=clock.now(),
    )


def _parse_optional_time_range(*, start: str | None, end: str | None) -> TimeRange | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise SystemExit("--start and --end must be provided together")
    return TimeRange(start=UtcTimestamp(_parse_dt(start)), end=UtcTimestamp(_parse_dt(end)))


def _parse_dt(raw: str) -> datetime:
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit(f"timestamp must include timezone: {raw}")
    return parsed.astimezone(timezone.utc)


def _print_report(payload: dict[str, object], *, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(
        "funding-rate-catchup report:\n"
        f"- instruments_total: {payload['instruments_total']}\n"
        f"- due: {payload['instruments_due']}\n"
        f"- ok: {payload['instruments_ok']}\n"
        f"- skipped: {payload['instruments_skipped']}\n"
        f"- failed: {payload['instruments_failed']}\n"
        f"- rows_read: {payload['rows_read']}\n"
        f"- rows_written: {payload['rows_written']}\n"
        f"- dry_run: {payload['dry_run']}\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="funding-rate-catchup")
    parser.add_argument(
        "--config",
        default="configs/dev/market_data.yaml",
        help="Path to market_data.yaml",
    )
    parser.add_argument("--sync-universe", action="store_true")
    parser.add_argument("--all-due", action="store_true")
    parser.add_argument("--market-id", type=int, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="UTC ISO start, inclusive")
    parser.add_argument("--end", type=str, default=None, help="UTC ISO end, exclusive")
    parser.add_argument("--funding-interval-minutes", type=int, default=None)
    parser.add_argument("--funding-interval-source", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-format", choices=("text", "json"), default="text")
    return parser
