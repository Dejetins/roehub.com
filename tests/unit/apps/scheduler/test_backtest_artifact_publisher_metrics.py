from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast

from prometheus_client import CollectorRegistry

from apps.scheduler.backtest_artifact_publisher.wiring.modules import (
    BacktestArtifactPublisherApp,
    BacktestArtifactPublisherMetrics,
)
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactSlotPublishErrorV2,
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
    Fixed enabled-instrument reader used in scheduler metric tests.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
    """

    instruments: tuple[InstrumentId, ...]

    def list_enabled_tradable(self) -> tuple[InstrumentId, ...]:
        """
        Return the configured enabled+tradable snapshot for the current test.

        Args:
            None.
        Returns:
            tuple[InstrumentId, ...]: Deterministic instrument tuple for the scenario.
        Assumptions:
            Metric tests do not need ClickHouse and therefore use a fixed in-memory snapshot.
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
    In-memory host lock double for scheduler metric scenarios.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
    """

    acquired: bool = True
    release_calls: int = 0

    def try_acquire(self) -> bool:
        """
        Return the configured lock acquisition outcome.

        Args:
            None.
        Returns:
            bool: Configured acquisition result.
        Assumptions:
            Metric tests only need success or immediate contention outcomes.
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
        return self.acquired

    def release(self) -> None:
        """
        Record one release call after a successful acquisition.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Only successful lock acquisitions should reach the release path.
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
class _SequencedPublishUseCase:
    """
    Recording publish use-case double consuming a queue of results or exceptions.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
    """

    outcomes: list[object]
    requests: list[PublishBacktestArtifactsV2Request] = field(default_factory=list)

    def run(self, request: PublishBacktestArtifactsV2Request) -> PublishBacktestArtifactsV2Result:
        """
        Return the next configured outcome after recording the forwarded request.

        Args:
            request: Shared publish request from the scheduler.
        Returns:
            PublishBacktestArtifactsV2Result: Next configured successful result.
        Assumptions:
            Tests seed enough outcomes for every publish call in the scheduler cycle.
        Raises:
            Exception: Re-raises a queued exception outcome for failure scenarios.
            AssertionError: If the test forgot to seed the next outcome.
        Side Effects:
            Appends one request to the in-memory call log.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("missing queued outcome for fake publish use-case")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return cast(PublishBacktestArtifactsV2Result, outcome)


def _successful_result(
    *,
    symbol: str,
    tail_rebuild_bars: ArtifactTailRebuildBarsV2,
) -> PublishBacktestArtifactsV2Result:
    """
    Build one deterministic successful publish result for scheduler metric assertions.

    Args:
        symbol: Symbol literal for the returned artifact coordinates.
        tail_rebuild_bars: Stage-level bounded tail counters to expose through metrics.
    Returns:
        PublishBacktestArtifactsV2Result: Stable successful publish result.
    Assumptions:
        Metric tests focus on counters and gauges rather than full filesystem details.
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
    return PublishBacktestArtifactsV2Result(
        status="succeeded",
        publish_mode="incremental",
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol=symbol,
        ),
        previous_active_slot="slot_a",
        previous_slot_generation=1,
        previous_manifest_sha256="1" * 64,
        published_active_slot="slot_b",
        published_slot_generation=2,
        published_manifest_sha256="2" * 64,
        asof_date="2026-03-30",
        published_at_utc="2026-03-30T00:10:00Z",
        requested_start_utc="2026-03-29T00:00:00Z",
        requested_end_utc="2026-03-30T00:00:00Z",
        source_start_utc="2026-03-29T00:00:00Z",
        source_end_utc="2026-03-30T00:00:00Z",
        source_candle_count=1440,
        reused_prefix_bars=1438,
        rewritten_tail_bars=tail_rebuild_bars.prices,
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
            prices=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=1438,
                rewritten_tail_bars=tail_rebuild_bars.prices,
            ),
            mappings=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=95,
                rewritten_tail_bars=tail_rebuild_bars.mappings,
            ),
            signals=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=94,
                rewritten_tail_bars=tail_rebuild_bars.signals,
            ),
            hit_times=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=1435,
                rewritten_tail_bars=tail_rebuild_bars.hit_times,
            ),
        ),
        tail_rebuild_bars=tail_rebuild_bars,
    )


def test_run_cycle_records_lock_held_metrics() -> None:
    """
    Ensure a host-lock contention updates the blocked and run counters without any publish attempt.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `lock_held` is a run-level failure before the universe scan starts.
    Raises:
        AssertionError: If the scheduler records incorrect metrics for lock contention.
    Side Effects:
        Executes one in-memory blocked scheduler cycle.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    fake_use_case = _SequencedPublishUseCase(outcomes=[])
    metrics = BacktestArtifactPublisherMetrics(registry=CollectorRegistry())
    app = BacktestArtifactPublisherApp(
        publish_use_case=cast(PublishBacktestArtifactsV2UseCase, fake_use_case),
        instrument_reader=_FakeInstrumentReader(instruments=()),
        metrics=metrics,
        host_lock=_FakeHostLock(acquired=False),
        metrics_port=9203,
        now_provider=lambda: datetime(2026, 3, 30, 0, 10, tzinfo=timezone.utc),
    )

    app._run_publish_cycle(
        stop_requested=lambda: False,
        scheduled_for_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    )

    assert metrics.backtest_artifact_publish_runs_total.labels(status="lock_held")._value.get() == 1
    assert (
        metrics.backtest_artifact_publish_blocked_total.labels(reason="lock_held")._value.get()
        == 1
    )
    assert fake_use_case.requests == []
    assert metrics.backtest_artifact_publish_last_success_unixtime._value.get() == 0


def test_run_cycle_records_validation_failed_counts_and_last_success() -> None:
    """
    Ensure mixed success plus validation failure updates counters, blocked reasons, and freshness.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Last-success freshness may advance when the cycle produced at least one successful publish.
    Raises:
        AssertionError: If counters or gauges drift from the agreed metric semantics.
    Side Effects:
        Executes one in-memory scheduler cycle with mixed symbol outcomes.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    now_utc = datetime(2026, 3, 30, 0, 10, tzinfo=timezone.utc)
    metrics = BacktestArtifactPublisherMetrics(registry=CollectorRegistry())
    fake_lock = _FakeHostLock(acquired=True)
    fake_use_case = _SequencedPublishUseCase(
        outcomes=[
            _successful_result(
                symbol="BTCUSDT",
                tail_rebuild_bars=ArtifactTailRebuildBarsV2(
                    prices=2,
                    mappings=3,
                    signals=4,
                    hit_times=5,
                ),
            ),
            ArtifactSlotPublishErrorV2(
                code="slot_validation_failed",
                message="slot validation failed",
            ),
        ]
    )
    app = BacktestArtifactPublisherApp(
        publish_use_case=cast(PublishBacktestArtifactsV2UseCase, fake_use_case),
        instrument_reader=_FakeInstrumentReader(
            instruments=(
                InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
                InstrumentId(market_id=MarketId(1), symbol=Symbol("ETHUSDT")),
            )
        ),
        metrics=metrics,
        host_lock=fake_lock,
        metrics_port=9203,
        now_provider=lambda: now_utc,
    )

    app._run_publish_cycle(
        stop_requested=lambda: False,
        scheduled_for_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    )

    assert metrics.backtest_artifact_publish_runs_total.labels(status="validation_failed")._value.get() == 1  # noqa: E501
    assert metrics.backtest_artifact_publish_symbols_total.labels(status="planned")._value.get() == 2  # noqa: E501
    assert metrics.backtest_artifact_publish_symbols_total.labels(status="succeeded")._value.get() == 1  # noqa: E501
    assert metrics.backtest_artifact_publish_symbols_total.labels(
        status="validation_failed"
    )._value.get() == 1
    assert metrics.backtest_artifact_publish_blocked_total.labels(
        reason="validation_failed"
    )._value.get() == 1
    assert metrics.backtest_artifact_tail_rebuild_bars_total.labels(stage="prices")._value.get() == 2  # noqa: E501
    assert metrics.backtest_artifact_tail_rebuild_bars_total.labels(stage="mappings")._value.get() == 3  # noqa: E501
    assert metrics.backtest_artifact_tail_rebuild_bars_total.labels(stage="signals")._value.get() == 4  # noqa: E501
    assert metrics.backtest_artifact_tail_rebuild_bars_total.labels(stage="hit_times")._value.get() == 5  # noqa: E501
    assert (
        metrics.backtest_artifact_publish_last_success_unixtime._value.get()
        == now_utc.timestamp()
    )
    assert fake_lock.release_calls == 1


def test_run_cycle_records_unexpected_error_status() -> None:
    """
    Ensure unexpected publish exceptions increment the dedicated `unexpected_error` labels.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Unexpected exceptions are not counted as blocked reasons.
    Raises:
        AssertionError: If unexpected exceptions drift into the wrong metric labels.
    Side Effects:
        Executes one in-memory failing scheduler cycle.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    metrics = BacktestArtifactPublisherMetrics(registry=CollectorRegistry())
    app = BacktestArtifactPublisherApp(
        publish_use_case=cast(
            PublishBacktestArtifactsV2UseCase,
            _SequencedPublishUseCase(outcomes=[RuntimeError("boom")]),
        ),
        instrument_reader=_FakeInstrumentReader(
            instruments=(InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),)
        ),
        metrics=metrics,
        host_lock=_FakeHostLock(acquired=True),
        metrics_port=9203,
        now_provider=lambda: datetime(2026, 3, 30, 0, 10, tzinfo=timezone.utc),
    )

    app._run_publish_cycle(
        stop_requested=lambda: False,
        scheduled_for_utc=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    )

    assert metrics.backtest_artifact_publish_runs_total.labels(status="unexpected_error")._value.get() == 1  # noqa: E501
    assert metrics.backtest_artifact_publish_symbols_total.labels(
        status="unexpected_error"
    )._value.get() == 1
    assert metrics.backtest_artifact_publish_last_success_unixtime._value.get() == 0
