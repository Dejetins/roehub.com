from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from prometheus_client import CollectorRegistry

from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
    MarketDataSchedulerApp,
    MarketDataSchedulerMetrics,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.application.use_cases.backfill_funding_rates import (
    BackfillFundingRatesReport,
    FundingCatchupInstrumentReport,
)
from trading.contexts.market_data.application.use_cases.sync_futures_funding_universe import (
    FundingUniverseMarketReport,
    SyncFuturesFundingUniverseReport,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


def test_funding_metrics_do_not_use_symbol_label() -> None:
    metrics = MarketDataSchedulerMetrics(registry=CollectorRegistry())
    funding_metrics = [
        metrics.scheduler_funding_catchup_instruments_total,
        metrics.scheduler_funding_catchup_rows_written_total,
        metrics.scheduler_funding_catchup_lag_seconds,
        metrics.scheduler_funding_catchup_last_success_timestamp_seconds,
        metrics.scheduler_funding_catchup_universe_instruments,
    ]

    for metric in funding_metrics:
        assert "symbol" not in metric._labelnames


def test_funding_startup_bootstrap_runs_before_heavy_startup_scan(tmp_path: Path) -> None:
    app = MarketDataSchedulerApp(
        config=_config(tmp_path),
        whitelist_path=str(tmp_path / "whitelist.csv"),
        seed_use_case=cast(Any, object()),
        sync_use_case=cast(Any, object()),
        enrich_use_case=cast(Any, object()),
        instrument_reader=cast(Any, object()),
        index_reader=cast(Any, object()),
        rest_fill_queue=cast(Any, object()),
        backfill_planner=cast(Any, object()),
        rest_catchup_use_case=cast(Any, object()),
        metrics=MarketDataSchedulerMetrics(registry=CollectorRegistry()),
        metrics_port=9202,
        funding_sync_use_case=cast(Any, object()),
        funding_catchup_use_case=cast(Any, object()),
    )

    assert [job.name for job in app._startup_jobs()] == [
        "sync_whitelist",
        "enrich",
        "funding_rate_catchup",
        "startup_scan",
    ]


def test_funding_job_does_not_refresh_full_universe_every_wake(tmp_path: Path) -> None:
    class _Clock:
        def now(self):
            return UtcTimestamp(datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc))

    class _FundingSync:
        def __init__(self):
            self.calls = 0

        def run(self):
            self.calls += 1
            return SyncFuturesFundingUniverseReport(
                markets_total=2,
                instruments_total=2,
                instruments_with_interval=2,
                instruments_missing_interval=0,
                rows_written=2,
                market_reports=(
                    FundingUniverseMarketReport(MarketId(2), 1, 1, 0),
                    FundingUniverseMarketReport(MarketId(4), 1, 1, 0),
                ),
            )

    class _FundingCatchup:
        def __init__(self):
            self.calls = 0

        def run_due_universe(self, *, market_ids, dry_run):
            self.calls += 1
            assert tuple(int(m.value) for m in market_ids) == (2, 4)
            assert dry_run is False
            return BackfillFundingRatesReport(
                instruments_total=1,
                instruments_due=0,
                instruments_ok=0,
                instruments_skipped=1,
                instruments_failed=0,
                rows_read=0,
                rows_written=0,
                dry_run=False,
                instrument_reports=(
                    FundingCatchupInstrumentReport(
                        instrument_id=InstrumentId(MarketId(2), Symbol("BTCUSDT")),
                        exchange="binance",
                        market_type="futures",
                        status="not_due",
                        start=None,
                        end=None,
                        rows_read=0,
                        rows_written=0,
                        lag_seconds=None,
                        reason="not due",
                    ),
                ),
            )

    async def _scenario() -> None:
        sync = _FundingSync()
        catchup = _FundingCatchup()
        app = MarketDataSchedulerApp(
            config=_config(tmp_path),
            whitelist_path=str(tmp_path / "whitelist.csv"),
            seed_use_case=cast(Any, object()),
            sync_use_case=cast(Any, object()),
            enrich_use_case=cast(Any, object()),
            instrument_reader=cast(Any, object()),
            index_reader=cast(Any, object()),
            rest_fill_queue=cast(Any, object()),
            backfill_planner=cast(Any, object()),
            rest_catchup_use_case=cast(Any, object()),
            metrics=MarketDataSchedulerMetrics(registry=CollectorRegistry()),
            metrics_port=9202,
            funding_sync_use_case=cast(Any, sync),
            funding_catchup_use_case=cast(Any, catchup),
        )
        app._clock = _Clock()

        await app._run_funding_rate_catchup_job()
        await app._run_funding_rate_catchup_job()

        assert sync.calls == 1
        assert catchup.calls == 2

    asyncio.run(_scenario())


def _config(tmp_path: Path):
    path = tmp_path / "market_data.yaml"
    path.write_text(
        """
version: 1
market_data:
  markets:
    - market_id: 2
      exchange: binance
      market_type: futures
      market_code: binance:futures
      rest:
        base_url: "https://fapi.binance.com"
        earliest_available_ts_utc: "2019-09-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.01, max_s: 0.01, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: "wss://x"
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
    - market_id: 4
      exchange: bybit
      market_type: futures
      market_code: bybit:futures
      rest:
        base_url: "https://api.bybit.com"
        earliest_available_ts_utc: "2018-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.01, max_s: 0.01, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: "wss://x"
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 200
  ingestion: { flush_interval_ms: 250, max_buffer_rows: 1000 }
  scheduler:
    jobs:
      funding_rate_catchup:
        universe_refresh_interval_seconds: 21600
  backfill: { max_days_per_insert: 7, chunk_align: utc_day }
""".strip(),
        encoding="utf-8",
    )
    return load_market_data_runtime_config(path)
