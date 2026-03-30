from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import cast

from prometheus_client import CollectorRegistry

from apps.scheduler.backtest_artifact_publisher.wiring.modules import (
    BacktestArtifactPublisherApp,
    BacktestArtifactPublisherMetrics,
    BacktestArtifactPublisherSchedule,
)
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactStageRebuildStatsCollectionV2,
    ArtifactStageRebuildStatsV2,
    ArtifactTailRebuildBarsV2,
)
from trading.contexts.backtest.application.use_cases import (
    PublishBacktestArtifactsV2Request,
    PublishBacktestArtifactsV2Result,
    PublishBacktestArtifactsV2UseCase,
    PublishBacktestArtifactsV2ValidationSummary,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


@dataclass(slots=True)
class _FakeInstrumentReader:
    """
    Deterministic enabled-instrument reader used by artifact publisher scheduler tests.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py
    """

    instruments: tuple[InstrumentId, ...]

    def list_enabled_tradable(self) -> tuple[InstrumentId, ...]:
        """
        Return the configured enabled+tradable snapshot without extra transformation.

        Args:
            None.
        Returns:
            tuple[InstrumentId, ...]: Deterministic instrument tuple for the test scenario.
        Assumptions:
            Scheduler tests control ordering explicitly through this fake reader.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        return self.instruments


@dataclass(slots=True)
class _FakeHostLock:
    """
    In-memory host lock stub with configurable acquisition outcome.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py
    """

    acquired: bool = True
    try_calls: int = 0
    release_calls: int = 0

    def try_acquire(self) -> bool:
        """
        Return the configured acquisition outcome while tracking call count.

        Args:
            None.
        Returns:
            bool: Configured lock acquisition result.
        Assumptions:
            Tests use the same fake for both successful and blocked scheduler scenarios.
        Raises:
            None.
        Side Effects:
            Increments the in-memory acquisition counter.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        self.try_calls += 1
        return self.acquired

    def release(self) -> None:
        """
        Record one release call without touching any external resource.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Tests only need to verify that release happens after successful acquisition.
        Raises:
            None.
        Side Effects:
            Increments the in-memory release counter.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        self.release_calls += 1


@dataclass(slots=True)
class _FakePublishUseCase:
    """
    Recording publish use-case double returning pre-seeded results.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py
    """

    results: list[PublishBacktestArtifactsV2Result]
    requests: list[PublishBacktestArtifactsV2Request] = field(default_factory=list)

    def run(self, request: PublishBacktestArtifactsV2Request) -> PublishBacktestArtifactsV2Result:
        """
        Record the request and return the next configured publish result.

        Args:
            request: Shared publish request forwarded by the scheduler.
        Returns:
            PublishBacktestArtifactsV2Result: Next deterministic result from the configured queue.
        Assumptions:
            Ordering assertions inspect the recorded requests after the scheduler cycle completes.
        Raises:
            AssertionError: If the test forgot to seed a result for the next request.
        Side Effects:
            Appends one request to the in-memory call log.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        self.requests.append(request)
        if not self.results:
            raise AssertionError("missing fake publish result")
        return self.results.pop(0)


def _result(*, exchange: str, market_type: str, symbol: str) -> PublishBacktestArtifactsV2Result:
    """
    Build one compact successful publish result for scheduler unit tests.

    Args:
        exchange: Exchange literal for the artifact coordinates.
        market_type: Market type literal for the artifact coordinates.
        symbol: Symbol literal for the artifact coordinates.
    Returns:
        PublishBacktestArtifactsV2Result: Deterministic successful publish result.
    Assumptions:
        App-layer tests only need a stable success DTO and do not exercise full filesystem details.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    coordinates = ArtifactCoordinatesV2(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
    )
    return PublishBacktestArtifactsV2Result(
        status="succeeded",
        publish_mode="incremental",
        coordinates=coordinates,
        previous_active_slot="slot_a",
        previous_slot_generation=1,
        previous_manifest_sha256="1" * 64,
        published_active_slot="slot_b",
        published_slot_generation=2,
        published_manifest_sha256="2" * 64,
        asof_date="2026-03-30",
        published_at_utc="2026-03-30T00:05:10Z",
        requested_start_utc="2026-03-29T00:00:00Z",
        requested_end_utc="2026-03-30T00:00:00Z",
        source_start_utc="2026-03-29T00:00:00Z",
        source_end_utc="2026-03-30T00:00:00Z",
        source_candle_count=1440,
        reused_prefix_bars=1438,
        rewritten_tail_bars=2,
        blocking_active_run_count=0,
        validation=PublishBacktestArtifactsV2ValidationSummary(
            slot_manifest_path=None,
            manifest_sha256="2" * 64,
            price_timeframes=("1m", "15m"),
            mapping_timeframes=("15m",),
            signal_artifacts=(("15m", "ma.ema"),),
            signal_manifest_count=1,
            hit_times_manifest_present=True,
            diagnostics_count=0,
        ),
        stage_rebuild_stats=ArtifactStageRebuildStatsCollectionV2(
            prices=ArtifactStageRebuildStatsV2(reused_prefix_bars=1438, rewritten_tail_bars=2),
            mappings=ArtifactStageRebuildStatsV2(reused_prefix_bars=95, rewritten_tail_bars=3),
            signals=ArtifactStageRebuildStatsV2(reused_prefix_bars=94, rewritten_tail_bars=4),
            hit_times=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=1435,
                rewritten_tail_bars=5,
            ),
        ),
        tail_rebuild_bars=ArtifactTailRebuildBarsV2(
            prices=2,
            mappings=3,
            signals=4,
            hit_times=5,
        ),
    )


def test_schedule_next_run_after_handles_moscow_boundaries() -> None:
    """
    Ensure the daily schedule stays anchored to `03:05 Europe/Moscow` around day boundaries.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Moscow remains a fixed explicit timezone in the scheduler contract.
    Raises:
        AssertionError: If next-run computation drifts from the agreed cadence.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    schedule = BacktestArtifactPublisherSchedule()

    before_run = schedule.next_run_after(
        now_utc=datetime(2026, 3, 29, 23, 59, tzinfo=timezone.utc),
        last_run_local_date=None,
    )
    exact_run = schedule.next_run_after(
        now_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
        last_run_local_date=None,
    )
    after_run = schedule.next_run_after(
        now_utc=datetime(2026, 3, 30, 0, 6, tzinfo=timezone.utc),
        last_run_local_date=None,
    )

    assert before_run == datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc)
    assert exact_run == datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc)
    assert after_run == datetime(2026, 3, 31, 0, 5, tzinfo=timezone.utc)


def test_schedule_does_not_trigger_twice_after_restart_within_same_minute() -> None:
    """
    Ensure a restart after `03:05` within the same minute moves the next run to the next day.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The scheduler intentionally does not perform a catch-up fire after the minute has passed.
    Raises:
        AssertionError: If the next-run calculation stays on the current Moscow date.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    schedule = BacktestArtifactPublisherSchedule()

    restarted = schedule.next_run_after(
        now_utc=datetime(2026, 3, 30, 0, 5, 10, tzinfo=timezone.utc),
        last_run_local_date=None,
    )
    same_day_already_processed = schedule.next_run_after(
        now_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
        last_run_local_date=date(2026, 3, 30),
    )

    assert restarted == datetime(2026, 3, 31, 0, 5, tzinfo=timezone.utc)
    assert same_day_already_processed == datetime(2026, 3, 31, 0, 5, tzinfo=timezone.utc)


def test_run_cycle_uses_sorted_enabled_tradable_universe() -> None:
    """
    Ensure one scheduler cycle sorts the enabled+tradable universe deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Deterministic ordering uses `(market_id, symbol)` before bridging into artifact coordinates.
    Raises:
        AssertionError: If requests are forwarded to the shared publish use-case out of order.
    Side Effects:
        Executes one in-memory scheduler cycle.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    fake_use_case = _FakePublishUseCase(
        results=[
            _result(exchange="binance", market_type="spot", symbol="ADAUSDT"),
            _result(exchange="binance", market_type="spot", symbol="BTCUSDT"),
            _result(exchange="binance", market_type="futures", symbol="ETHUSDT"),
        ]
    )
    registry = CollectorRegistry()
    app = BacktestArtifactPublisherApp(
        publish_use_case=cast(PublishBacktestArtifactsV2UseCase, fake_use_case),
        instrument_reader=_FakeInstrumentReader(
            instruments=(
                InstrumentId(market_id=MarketId(2), symbol=Symbol("ETHUSDT")),
                InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
                InstrumentId(market_id=MarketId(1), symbol=Symbol("ADAUSDT")),
            )
        ),
        metrics=BacktestArtifactPublisherMetrics(registry=registry),
        host_lock=_FakeHostLock(acquired=True),
        metrics_port=9203,
        now_provider=lambda: datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    )

    app._run_publish_cycle(
        stop_requested=lambda: False,
        scheduled_for_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    )

    assert [request.coordinates for request in fake_use_case.requests] == [
        ArtifactCoordinatesV2(exchange="binance", market_type="spot", symbol="ADAUSDT"),
        ArtifactCoordinatesV2(exchange="binance", market_type="spot", symbol="BTCUSDT"),
        ArtifactCoordinatesV2(exchange="binance", market_type="futures", symbol="ETHUSDT"),
    ]
    assert (
        app.metrics.backtest_artifact_publish_symbols_total.labels(status="planned")._value.get()
        == 3
    )


def test_app_starts_metrics_and_exits_cleanly_when_stop_already_set(
    monkeypatch,
) -> None:
    """
    Ensure the long-running app starts its metrics endpoint and exits cleanly on immediate stop.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Cooperative shutdown before the loop starts must not trigger any publish attempt.
    Raises:
        AssertionError: If the app runs a publish cycle despite the stop event already being set.
    Side Effects:
        Patches the in-process Prometheus HTTP server bootstrap.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    metrics_ports: list[int] = []
    monkeypatch.setattr(
        "apps.scheduler.backtest_artifact_publisher.wiring.modules.backtest_artifact_publisher.start_http_server",
        lambda port, registry=None: metrics_ports.append(port),
    )
    fake_use_case = _FakePublishUseCase(results=[])
    app = BacktestArtifactPublisherApp(
        publish_use_case=cast(PublishBacktestArtifactsV2UseCase, fake_use_case),
        instrument_reader=_FakeInstrumentReader(instruments=()),
        metrics=BacktestArtifactPublisherMetrics(registry=CollectorRegistry()),
        host_lock=_FakeHostLock(acquired=True),
        metrics_port=9203,
        now_provider=lambda: datetime(2026, 3, 30, 0, 0, tzinfo=timezone.utc),
    )

    async def _scenario() -> None:
        """
        Run the scheduler with an already-set stop event.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The app loop should skip scheduling work when the stop event is already set.
        Raises:
            None.
        Side Effects:
            Executes one short-lived scheduler runtime.
        Docs:
          - docs/runbooks/mac-studio-native-backend-operations.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        stop_event = asyncio.Event()
        stop_event.set()
        await app.run(stop_event)

    asyncio.run(_scenario())

    assert metrics_ports == [9203]
    assert fake_use_case.requests == []
