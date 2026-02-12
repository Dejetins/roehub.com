from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from prometheus_client import REGISTRY, CollectorRegistry

from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
    MarketDataSchedulerApp,
    MarketDataSchedulerMetrics,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    DailyTsOpenCount,
)
from trading.contexts.market_data.application.services import (
    AsyncRestFillQueue,
    SchedulerBackfillPlanner,
)
from trading.contexts.market_data.application.use_cases import (
    EnrichRefInstrumentsFromExchangeUseCase,
    RestCatchUp1mReport,
    RestCatchUp1mUseCase,
    SeedRefMarketUseCase,
    SyncWhitelistToRefInstrumentsUseCase,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


@dataclass(frozen=True, slots=True)
class _FixedClock:
    now_value: UtcTimestamp

    def now(self) -> UtcTimestamp:
        """Return deterministic current timestamp for scheduler tests."""
        return self.now_value


class _SeedUseCase:
    def __init__(self) -> None:
        """Initialize seed invocation counter."""
        self.calls = 0

    def run(self):  # noqa: ANN001
        """Increment invocation counter for each scheduler seed run."""
        self.calls += 1


class _SyncUseCase:
    def __init__(self) -> None:
        """Initialize sync invocation counter."""
        self.calls = 0

    def run(self, rows):  # noqa: ANN001
        """Increment invocation counter and ignore payload details."""
        _ = rows
        self.calls += 1


class _EnrichUseCase:
    def __init__(self) -> None:
        """Initialize enrich invocation counter."""
        self.calls = 0

    def run(self):  # noqa: ANN001
        """Increment invocation counter and return minimal report-like object."""
        self.calls += 1
        return type(
            "_Report",
            (),
            {"instruments_total": 1, "rows_upserted": 1, "symbols_missing_metadata": 0},
        )()


class _InstrumentReader:
    def list_enabled_tradable(self):  # noqa: ANN001
        """Return one enabled instrument for startup scan planning."""
        return [InstrumentId(MarketId(1), Symbol("BTCUSDT"))]


class _MultiInstrumentReader:
    def list_enabled_tradable(self):  # noqa: ANN001
        """Return multiple enabled instruments for periodic insurance tests."""
        return [
            InstrumentId(MarketId(1), Symbol("BTCUSDT")),
            InstrumentId(MarketId(1), Symbol("ETHUSDT")),
        ]


class _BaseIndexReader:
    """Base canonical-index fake with protocol-compatible helper methods."""

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Return full canonical bounds when known.

        Parameters:
        - instrument_id: requested instrument id.

        Returns:
        - `None` because startup-scan tests only require `bounds_1m`.
        """
        _ = instrument_id
        return None

    def max_ts_open_lt(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> UtcTimestamp | None:
        """
        Return latest canonical minute before upper bound.

        Parameters:
        - instrument_id: requested instrument id.
        - before: exclusive upper bound.

        Returns:
        - `None` because startup-scan tests do not use this method.
        """
        _ = instrument_id
        _ = before
        return None

    def daily_counts(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> tuple[DailyTsOpenCount, ...]:
        """
        Return day-count rows for gap scanning.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested time range.

        Returns:
        - Empty tuple for startup-scan focused tests.
        """
        _ = instrument_id
        _ = time_range
        return ()

    def distinct_ts_opens(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> tuple[UtcTimestamp, ...]:
        """
        Return distinct canonical minute opens.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested time range.

        Returns:
        - Empty tuple for startup-scan focused tests.
        """
        _ = instrument_id
        _ = time_range
        return ()


class _IndexReader(_BaseIndexReader):
    def bounds_1m(self, *, instrument_id, before):  # noqa: ANN001
        """Return canonical bounds causing both historical and tail tasks."""
        _ = instrument_id
        _ = before
        return (
            UtcTimestamp(datetime(2026, 2, 9, 13, 55, tzinfo=timezone.utc)),
            UtcTimestamp(datetime(2026, 2, 9, 13, 58, tzinfo=timezone.utc)),
        )


class _TailOnlyIndexReader(_BaseIndexReader):
    def bounds_1m(self, *, instrument_id, before):  # noqa: ANN001
        """Return canonical tail-only bounds to reproduce production bug scenario."""
        _ = instrument_id
        _ = before
        return (
            UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc)),
            UtcTimestamp(datetime(2026, 2, 9, 13, 59, tzinfo=timezone.utc)),
        )


class _NoSeedIndexReader(_BaseIndexReader):
    def bounds_1m(self, *, instrument_id, before):  # noqa: ANN001
        """Return empty canonical bounds to force bootstrap planning path."""
        _ = instrument_id
        _ = before
        return (None, None)


class _RestQueue:
    def __init__(self) -> None:
        """Initialize queue lifecycle and enqueue tracking containers."""
        self.started = 0
        self.closed = 0
        self.enqueued = []

    async def start(self) -> None:
        """Record queue start call."""
        self.started += 1

    async def close(self) -> None:
        """Record queue close call."""
        self.closed += 1

    async def enqueue(self, task) -> bool:  # noqa: ANN001
        """Capture enqueued task and report acceptance."""
        self.enqueued.append(task)
        return True


class _RestCatchUpUseCase:
    def __init__(self) -> None:
        """Initialize deterministic rest-catchup fake state."""
        self.calls: list[InstrumentId] = []

    def run(self, instrument_id: InstrumentId) -> RestCatchUp1mReport:
        """Record invocation and return fixed report payload."""
        self.calls.append(instrument_id)
        return RestCatchUp1mReport(
            tail_start=UtcTimestamp(datetime(2026, 2, 9, 13, 55, tzinfo=timezone.utc)),
            tail_end=UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)),
            tail_rows_read=5,
            tail_rows_written=5,
            tail_batches=1,
            gap_scan_start=UtcTimestamp(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)),
            gap_scan_end=UtcTimestamp(datetime(2026, 2, 9, 13, 55, tzinfo=timezone.utc)),
            gap_days_scanned=8,
            gap_days_with_gaps=2,
            gap_ranges_filled=3,
            gap_rows_read=10,
            gap_rows_written=7,
            gap_rows_skipped_existing=3,
            gap_batches=1,
        )


class _NoSeedRestCatchUpUseCase:
    def __init__(self) -> None:
        """Initialize invocation tracking for missing-seed scenario."""
        self.calls: list[InstrumentId] = []

    def run(self, instrument_id: InstrumentId) -> RestCatchUp1mReport:
        """Always emulate missing canonical seed error."""
        self.calls.append(instrument_id)
        raise ValueError("Run initial backfill first")


def _config(tmp_path: Path):
    """
    Build runtime config fixture for scheduler startup scan test.

    Parameters:
    - tmp_path: pytest temporary directory fixture.

    Returns:
    - Parsed market-data runtime config.
    """
    yaml_text = """
version: 1
market_data:
  markets:
    - market_id: 1
      exchange: binance
      market_type: spot
      market_code: binance:spot
      rest:
        base_url: https://api.binance.com
        earliest_available_ts_utc: "2017-01-01T00:00:00Z"
        timeout_s: 10.0
        retries: 0
        backoff: { base_s: 0.1, max_s: 0.1, jitter_s: 0.0 }
        limiter: { mode: autodetect, safety_factor: 0.8, max_concurrency: 1 }
      ws:
        url: wss://stream.binance.com:9443/stream
        ping_interval_s: 20.0
        pong_timeout_s: 10.0
        reconnect: { min_delay_s: 0.5, max_delay_s: 30.0, factor: 1.7, jitter_s: 0.2 }
        max_symbols_per_connection: 50
  ingestion:
    flush_interval_ms: 250
    max_buffer_rows: 1000
    rest_concurrency_instruments: 2
    tail_lookback_minutes: 180
  scheduler:
    jobs:
      sync_whitelist: { interval_seconds: 3600 }
      enrich: { interval_seconds: 3600 }
      rest_insurance_catchup: { interval_seconds: 3600 }
  backfill:
    max_days_per_insert: 7
    chunk_align: utc_day
"""
    cfg_path = tmp_path / "market_data.yaml"
    cfg_path.write_text(yaml_text.strip(), encoding="utf-8")
    return load_market_data_runtime_config(cfg_path)


def test_scheduler_runs_startup_scan_once_and_enqueues_tasks(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """
    Ensure startup scan runs once per process start and enqueues planned tasks.

    Parameters:
    - tmp_path: pytest temporary path fixture.
    - monkeypatch: pytest monkeypatch fixture.

    Returns:
    - None.
    """

    async def _scenario() -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(
            "apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler.start_http_server",
            lambda _port: None,
        )

        seed = _SeedUseCase()
        sync = _SyncUseCase()
        enrich = _EnrichUseCase()
        queue = _RestQueue()
        registry = CollectorRegistry()
        app = MarketDataSchedulerApp(
            config=_config(tmp_path),
            whitelist_path=str(tmp_path / "missing.csv"),
            seed_use_case=cast(SeedRefMarketUseCase, seed),
            sync_use_case=cast(SyncWhitelistToRefInstrumentsUseCase, sync),
            enrich_use_case=cast(EnrichRefInstrumentsFromExchangeUseCase, enrich),
            instrument_reader=_InstrumentReader(),
            index_reader=_IndexReader(),
            rest_fill_queue=cast(AsyncRestFillQueue, queue),
            backfill_planner=SchedulerBackfillPlanner(tail_lookback_minutes=180),
            rest_catchup_use_case=cast(RestCatchUp1mUseCase, _RestCatchUpUseCase()),
            metrics=MarketDataSchedulerMetrics(registry=registry),
            metrics_port=9202,
        )
        app._clock = _FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)))

        whitelist = tmp_path / "missing.csv"
        whitelist.write_text("market_id,symbol,is_enabled\n1,BTCUSDT,1\n", encoding="utf-8")

        stop_event = asyncio.Event()
        stop_event.set()

        await app.run(stop_event)

        assert seed.calls == 1
        assert sync.calls == 1
        assert enrich.calls == 1
        assert queue.started == 1
        assert queue.closed == 1
        assert len(queue.enqueued) == 2
        reasons = {task.reason for task in queue.enqueued}
        assert reasons == {"historical_backfill", "scheduler_tail"}
        assert app._metrics.scheduler_startup_scan_instruments_total._value.get() == 1
        assert app._metrics.scheduler_tasks_planned_total.labels(
            reason="historical_backfill"
        )._value.get() == 1
        assert app._metrics.scheduler_tasks_enqueued_total.labels(
            reason="historical_backfill"
        )._value.get() == 1
        assert app._metrics.scheduler_tasks_planned_total.labels(reason="scheduler_tail")._value.get() == 1  # noqa: E501
        assert app._metrics.scheduler_tasks_enqueued_total.labels(reason="scheduler_tail")._value.get() == 1  # noqa: E501
        assert "startup_scan: instruments_scanned=1" in caplog.text
        assert "startup_scan planned_task[1]" in caplog.text

    asyncio.run(_scenario())


def test_scheduler_startup_scan_tail_only_canonical_still_enqueues_historical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    Ensure startup scan enqueues historical task in tail-only canonical state.

    Parameters:
    - tmp_path: pytest temporary path fixture.
    - monkeypatch: pytest monkeypatch fixture.

    Returns:
    - None.
    """

    async def _scenario() -> None:
        monkeypatch.setattr(
            "apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler.start_http_server",
            lambda _port: None,
        )

        queue = _RestQueue()
        app = MarketDataSchedulerApp(
            config=_config(tmp_path),
            whitelist_path=str(tmp_path / "missing.csv"),
            seed_use_case=cast(SeedRefMarketUseCase, _SeedUseCase()),
            sync_use_case=cast(SyncWhitelistToRefInstrumentsUseCase, _SyncUseCase()),
            enrich_use_case=cast(EnrichRefInstrumentsFromExchangeUseCase, _EnrichUseCase()),
            instrument_reader=_InstrumentReader(),
            index_reader=_TailOnlyIndexReader(),
            rest_fill_queue=cast(AsyncRestFillQueue, queue),
            backfill_planner=SchedulerBackfillPlanner(tail_lookback_minutes=180),
            rest_catchup_use_case=cast(RestCatchUp1mUseCase, _RestCatchUpUseCase()),
            metrics=MarketDataSchedulerMetrics(registry=CollectorRegistry()),
            metrics_port=9202,
        )
        app._clock = _FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)))

        whitelist = tmp_path / "missing.csv"
        whitelist.write_text("market_id,symbol,is_enabled\n1,BTCUSDT,1\n", encoding="utf-8")

        stop_event = asyncio.Event()
        stop_event.set()

        await app.run(stop_event)

        reasons = {task.reason for task in queue.enqueued}
        assert "historical_backfill" in reasons
        historical = next(task for task in queue.enqueued if task.reason == "historical_backfill")
        assert str(historical.time_range.start) == "2017-01-01T00:00:00.000Z"
        assert str(historical.time_range.end) == "2026-02-09T13:50:00.000Z"

    asyncio.run(_scenario())


def test_rest_insurance_catchup_runs_for_all_enabled_instruments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    Ensure periodic insurance catchup executes rest-catchup for every enabled instrument.

    Parameters:
    - tmp_path: pytest temporary path fixture.
    - monkeypatch: pytest monkeypatch fixture.

    Returns:
    - None.
    """

    async def _scenario() -> None:
        monkeypatch.setattr(
            "apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler.start_http_server",
            lambda _port: None,
        )

        queue = _RestQueue()
        rest_catchup = _RestCatchUpUseCase()
        app = MarketDataSchedulerApp(
            config=_config(tmp_path),
            whitelist_path=str(tmp_path / "missing.csv"),
            seed_use_case=cast(SeedRefMarketUseCase, _SeedUseCase()),
            sync_use_case=cast(SyncWhitelistToRefInstrumentsUseCase, _SyncUseCase()),
            enrich_use_case=cast(EnrichRefInstrumentsFromExchangeUseCase, _EnrichUseCase()),
            instrument_reader=_MultiInstrumentReader(),
            index_reader=_IndexReader(),
            rest_fill_queue=cast(AsyncRestFillQueue, queue),
            backfill_planner=SchedulerBackfillPlanner(tail_lookback_minutes=180),
            rest_catchup_use_case=cast(RestCatchUp1mUseCase, rest_catchup),
            metrics=MarketDataSchedulerMetrics(registry=CollectorRegistry()),
            metrics_port=9202,
        )
        app._clock = _FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)))

        await app._run_rest_insurance_job()

        assert len(rest_catchup.calls) == 2
        assert {str(inst.symbol) for inst in rest_catchup.calls} == {"BTCUSDT", "ETHUSDT"}
        assert not any(task.reason == "scheduler_tail" for task in queue.enqueued)
        assert app._metrics.scheduler_rest_catchup_instruments_total.labels(status="ok")._value.get() == 2  # noqa: E501
        assert app._metrics.scheduler_rest_catchup_tail_rows_written_total._value.get() == 10
        assert app._metrics.scheduler_rest_catchup_gap_days_scanned_total._value.get() == 16

    asyncio.run(_scenario())


def test_rest_insurance_catchup_tracks_skipped_no_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    Ensure periodic insurance marks no-seed instruments as skipped without failing the job.

    Parameters:
    - tmp_path: pytest temporary path fixture.
    - monkeypatch: pytest monkeypatch fixture.

    Returns:
    - None.
    """

    async def _scenario() -> None:
        monkeypatch.setattr(
            "apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler.start_http_server",
            lambda _port: None,
        )

        queue = _RestQueue()
        rest_catchup = _NoSeedRestCatchUpUseCase()
        app = MarketDataSchedulerApp(
            config=_config(tmp_path),
            whitelist_path=str(tmp_path / "missing.csv"),
            seed_use_case=cast(SeedRefMarketUseCase, _SeedUseCase()),
            sync_use_case=cast(SyncWhitelistToRefInstrumentsUseCase, _SyncUseCase()),
            enrich_use_case=cast(EnrichRefInstrumentsFromExchangeUseCase, _EnrichUseCase()),
            instrument_reader=_InstrumentReader(),
            index_reader=_NoSeedIndexReader(),
            rest_fill_queue=cast(AsyncRestFillQueue, queue),
            backfill_planner=SchedulerBackfillPlanner(tail_lookback_minutes=180),
            rest_catchup_use_case=cast(RestCatchUp1mUseCase, rest_catchup),
            metrics=MarketDataSchedulerMetrics(registry=CollectorRegistry()),
            metrics_port=9202,
        )
        app._clock = _FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)))

        await app._run_rest_insurance_job()

        assert len(rest_catchup.calls) == 1
        assert app._metrics.scheduler_rest_catchup_instruments_total.labels(
            status="skipped_no_seed"
        )._value.get() == 1
        assert any(task.reason == "scheduler_bootstrap" for task in queue.enqueued)

    asyncio.run(_scenario())


def test_scheduler_metrics_register_in_default_registry_when_not_provided() -> None:
    """
    Ensure scheduler metrics are exposed on global Prometheus registry by default.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Default scheduler runtime uses global Prometheus registry via `start_http_server`.

    Errors/Exceptions:
    - None.

    Side effects:
    - Registers and then unregisters scheduler collectors in global registry.
    """
    metrics = MarketDataSchedulerMetrics()
    collectors = [
        metrics.scheduler_job_runs_total,
        metrics.scheduler_job_errors_total,
        metrics.scheduler_job_duration_seconds,
        metrics.scheduler_tasks_planned_total,
        metrics.scheduler_tasks_enqueued_total,
        metrics.scheduler_startup_scan_instruments_total,
        metrics.scheduler_rest_catchup_instruments_total,
        metrics.scheduler_rest_catchup_tail_minutes_total,
        metrics.scheduler_rest_catchup_tail_rows_written_total,
        metrics.scheduler_rest_catchup_gap_days_scanned_total,
        metrics.scheduler_rest_catchup_gap_days_with_gaps_total,
        metrics.scheduler_rest_catchup_gap_ranges_filled_total,
        metrics.scheduler_rest_catchup_gap_rows_written_total,
    ]
    try:
        metrics.scheduler_job_runs_total.labels(job="startup_scan").inc()
        assert (
            REGISTRY.get_sample_value(
                "scheduler_job_runs_total",
                labels={"job": "startup_scan"},
            )
            == 1.0
        )
    finally:
        for collector in collectors:
            REGISTRY.unregister(collector)
