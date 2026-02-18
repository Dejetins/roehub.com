from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Sequence
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
)
from trading.contexts.strategy.application import (
    StrategyLiveCandleMessage,
    StrategyLiveRunner,
    StrategyRealtimeEventV1,
    StrategyRealtimeMetricV1,
    StrategyRealtimeOutputRecordV1,
    StrategyTelegramNotificationV1,
    TelegramNotificationPolicy,
)
from trading.contexts.strategy.application.services import (
    TimeframeRollupPolicy,
    TimeframeRollupProgress,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun, StrategySpecV1
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)


class _FixedClock:
    """
    Deterministic UTC clock stub returning the same timestamp for every call.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize fixed clock value.

        Args:
            now_value: UTC datetime returned by `now()`.
        Returns:
            None.
        Assumptions:
            Value is timezone-aware UTC datetime.
        Raises:
            None.
        Side Effects:
            Stores immutable timestamp value.
        """
        self._now_value = now_value

    def now(self) -> datetime:
        """
        Return fixed UTC datetime value.

        Args:
            None.
        Returns:
            datetime: Fixed UTC datetime.
        Assumptions:
            Returned value satisfies strategy clock contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


class _SleeperProbe:
    """
    Deterministic sleeper probe recording requested backoff durations.
    """

    def __init__(self) -> None:
        """
        Initialize empty backoff duration log.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Test code inspects `calls` list for retry behavior assertions.
        Raises:
            None.
        Side Effects:
            Creates mutable in-memory call log.
        """
        self.calls: list[float] = []

    def sleep(self, *, seconds: float) -> None:
        """
        Record one requested sleep duration.

        Args:
            seconds: Requested backoff duration.
        Returns:
            None.
        Assumptions:
            Backoff policy passes non-negative values.
        Raises:
            None.
        Side Effects:
            Appends value to internal call log.
        """
        self.calls.append(seconds)


class _StreamStub:
    """
    Deterministic live candle stream stub with one-shot read buffers per instrument.
    """

    def __init__(
        self,
        *,
        messages_by_instrument: Mapping[str, Sequence[StrategyLiveCandleMessage]],
    ) -> None:
        """
        Initialize stream buffers from deterministic message mapping.

        Args:
            messages_by_instrument: Per-instrument message sequences.
        Returns:
            None.
        Assumptions:
            Messages are already deterministic test fixtures.
        Raises:
            None.
        Side Effects:
            Stores mutable one-shot message buffers.
        """
        self._buffers = {key: list(value) for key, value in messages_by_instrument.items()}
        self.acks: list[tuple[str, str]] = []

    def read_closed_1m(self, *, instrument_key: str) -> tuple[StrategyLiveCandleMessage, ...]:
        """
        Return buffered messages for one instrument and clear buffer.

        Args:
            instrument_key: Canonical instrument key.
        Returns:
            tuple[StrategyLiveCandleMessage, ...]: Buffered messages.
        Assumptions:
            Each buffer is consumed exactly once per test iteration.
        Raises:
            None.
        Side Effects:
            Empties internal buffer for requested instrument key.
        """
        buffered = self._buffers.get(instrument_key, [])
        self._buffers[instrument_key] = []
        return tuple(buffered)

    def ack(self, *, instrument_key: str, message_id: str) -> None:
        """
        Record ack call for assertion.

        Args:
            instrument_key: Instrument key.
            message_id: Message id acknowledged by runner.
        Returns:
            None.
        Assumptions:
            Runner acks each processed message exactly once.
        Raises:
            None.
        Side Effects:
            Appends tuple into ack call log.
        """
        self.acks.append((instrument_key, message_id))


class _CanonicalReaderStub:
    """
    Deterministic canonical reader stub using queued response batches per call.
    """

    def __init__(self, *, responses: Sequence[Sequence[CandleWithMeta]]) -> None:
        """
        Initialize queued canonical responses.

        Args:
            responses: Ordered sequence of result batches returned per read call.
        Returns:
            None.
        Assumptions:
            Tests configure enough response batches for expected calls.
        Raises:
            None.
        Side Effects:
            Stores mutable queue and call log.
        """
        self._responses = list(responses)
        self.calls: list[TimeRange] = []

    def read_1m(self, instrument_id: InstrumentId, time_range: TimeRange):
        """
        Return next queued response batch and record requested range.

        Args:
            instrument_id: Instrument identifier.
            time_range: Requested canonical time range.
        Returns:
            Iterator[CandleWithMeta]: Response iterator.
        Assumptions:
            Instrument id is ignored in stub and checked by test fixture setup.
        Raises:
            None.
        Side Effects:
            Pops one response batch from queue and records requested time range.
        """
        _ = instrument_id
        self.calls.append(time_range)
        if not self._responses:
            return iter(())
        return iter(self._responses.pop(0))


class _TrackingRunRepository(InMemoryStrategyRunRepository):
    """
    In-memory run repository with update-order tracking for deterministic ordering assertions.
    """

    def __init__(self, *, reverse_active_runs: bool = False) -> None:
        """
        Initialize tracking repository.

        Args:
            reverse_active_runs: Return active runs in reverse order to test runner sorting.
        Returns:
            None.
        Assumptions:
            Base repository semantics are preserved.
        Raises:
            None.
        Side Effects:
            Enables mutable update-order log.
        """
        super().__init__()
        self._reverse_active_runs = reverse_active_runs
        self.update_order: list[str] = []

    def update(self, *, run: StrategyRun) -> StrategyRun:
        """
        Track run update order and delegate persistence to base repository.

        Args:
            run: Updated run snapshot.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Update order should reflect deterministic runner processing order.
        Raises:
            Exception: Propagates base repository errors.
        Side Effects:
            Appends run id to update-order log.
        """
        self.update_order.append(str(run.run_id))
        return super().update(run=run)

    def list_active_runs(self) -> tuple[StrategyRun, ...]:
        """
        List active runs, optionally reversing base deterministic ordering.

        Args:
            None.
        Returns:
            tuple[StrategyRun, ...]: Active run snapshots.
        Assumptions:
            Reverse mode is used only to verify runner-side ordering safeguards.
        Raises:
            None.
        Side Effects:
            None.
        """
        rows = super().list_active_runs()
        if self._reverse_active_runs:
            return tuple(reversed(rows))
        return rows


class _RealtimeOutputProbe:
    """
    In-memory realtime output publisher probe recording publish batches.
    """

    def __init__(self) -> None:
        """
        Initialize empty publish batches collection.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Probe is used to verify live-runner integration points.
        Raises:
            None.
        Side Effects:
            Creates mutable in-memory storage for publish calls.
        """
        self.batches: list[tuple[StrategyRealtimeOutputRecordV1, ...]] = []

    def publish_records_v1(
        self, *, records: Sequence[StrategyRealtimeOutputRecordV1]
    ) -> None:
        """
        Record one realtime publish batch.

        Args:
            records: Published realtime output records.
        Returns:
            None.
        Assumptions:
            Records are already validated by application service before publishing.
        Raises:
            None.
        Side Effects:
            Appends immutable batch snapshot to probe storage.
        """
        self.batches.append(tuple(records))


class _TelegramNotifierProbe:
    """
    In-memory Telegram notifier probe recording notifications and optional injected failures.
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        """
        Initialize Telegram notifier probe.

        Args:
            error: Optional error raised by notify call.
        Returns:
            None.
        Assumptions:
            Probe is used for deterministic best-effort behavior assertions.
        Raises:
            None.
        Side Effects:
            Creates mutable notification call log.
        """
        self._error = error
        self.notifications: list[StrategyTelegramNotificationV1] = []

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        """
        Record notification payload and optionally raise injected error.

        Args:
            notification: Telegram notification payload.
        Returns:
            None.
        Assumptions:
            Notifier is called only after policy filtering.
        Raises:
            Exception: Injected error configured by test.
        Side Effects:
            Appends notification to in-memory call log.
        """
        self.notifications.append(notification)
        if self._error is not None:
            raise self._error


class _FailingRollupPolicy(TimeframeRollupPolicy):
    """
    Timeframe rollup policy stub that always fails for deterministic failure-path tests.
    """

    def advance(
        self,
        *,
        timeframe: Timeframe,
        progress: TimeframeRollupProgress,
        candle_ts_open: datetime,
    ):
        """
        Raise deterministic failure from rollup policy advance.

        Args:
            timeframe: Strategy timeframe value object.
            progress: Current rollup progress.
            candle_ts_open: Candle timestamp.
        Returns:
            TimeframeRollupProgress: This method never returns.
        Assumptions:
            Live-runner catches processing-stage exceptions and marks run failed.
        Raises:
            RuntimeError: Always raised for failure-path assertions.
        Side Effects:
            None.
        """
        _ = (timeframe, progress, candle_ts_open)
        raise RuntimeError("rollup policy failed")


def test_live_runner_checkpoint_monotonicity_ignores_duplicates_and_out_of_order() -> None:
    """
    Ensure runner ignores candles with `ts_open <= checkpoint`
    and advances only for contiguous minute.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000901")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    run = _create_running_run(
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        checkpoint_ts_open=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
    )
    run_repository.create(run=run)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-1", _candle_at(datetime(2026, 2, 17, 9, 59, tzinfo=timezone.utc))),
                _message("m-2", _candle_at(datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc))),
                _message("m-3", _candle_at(datetime(2026, 2, 17, 10, 1, tzinfo=timezone.utc))),
                _message("m-4", _candle_at(datetime(2026, 2, 17, 10, 1, tzinfo=timezone.utc))),
                _message("m-5", _candle_at(datetime(2026, 2, 17, 10, 2, tzinfo=timezone.utc))),
            ),
        }
    )
    canonical_reader = _CanonicalReaderStub(responses=())
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=canonical_reader,
        retry_attempts=0,
    )

    report = runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=run.run_id)

    assert persisted is not None
    assert persisted.checkpoint_ts_open == datetime(2026, 2, 17, 10, 2, tzinfo=timezone.utc)
    assert canonical_reader.calls == []
    assert report.read_messages == 5
    assert report.acked_messages == 5
    assert len(stream.acks) == 5


def test_live_runner_gap_repair_retries_and_advances_only_after_full_continuity() -> None:
    """
    Ensure gap repair retries canonical reads and advances checkpoint only after full continuity.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000902")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    run = _create_running_run(
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        checkpoint_ts_open=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
    )
    run_repository.create(run=run)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-gap", _candle_at(datetime(2026, 2, 17, 10, 3, tzinfo=timezone.utc))),
            ),
        }
    )
    canonical_reader = _CanonicalReaderStub(
        responses=(
            (_candle_at(datetime(2026, 2, 17, 10, 1, tzinfo=timezone.utc)),),
            (
                _candle_at(datetime(2026, 2, 17, 10, 1, tzinfo=timezone.utc)),
                _candle_at(datetime(2026, 2, 17, 10, 2, tzinfo=timezone.utc)),
            ),
        )
    )
    sleeper = _SleeperProbe()
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=canonical_reader,
        retry_attempts=1,
        sleeper=sleeper,
    )

    runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=run.run_id)

    assert persisted is not None
    assert persisted.checkpoint_ts_open == datetime(2026, 2, 17, 10, 3, tzinfo=timezone.utc)
    assert len(canonical_reader.calls) == 2
    assert sleeper.calls == [1.0]


def test_live_runner_gap_repair_keeps_checkpoint_when_continuity_not_restored() -> None:
    """
    Ensure runner does not advance checkpoint if canonical repair cannot restore full missing range.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000903")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    run = _create_running_run(
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        checkpoint_ts_open=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
    )
    run_repository.create(run=run)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-gap", _candle_at(datetime(2026, 2, 17, 10, 3, tzinfo=timezone.utc))),
            ),
        }
    )
    canonical_reader = _CanonicalReaderStub(
        responses=(
            (_candle_at(datetime(2026, 2, 17, 10, 1, tzinfo=timezone.utc)),),
        )
    )
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=canonical_reader,
        retry_attempts=0,
    )

    runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=run.run_id)

    assert persisted is not None
    assert persisted.checkpoint_ts_open == datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)


def test_live_runner_computes_warmup_and_transitions_to_running() -> None:
    """
    Ensure warmup metadata uses `numeric_max_param_v1` and transitions run to running after warmup.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000904")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    started_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000E904"),
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc),
        metadata_json={},
    )
    run_repository.create(run=started_run)

    messages = tuple(
        _message(
            f"m-{index}",
            _candle_at(
                datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)
                + timedelta(minutes=index)
            ),
        )
        for index in range(50)
    )
    stream = _StreamStub(messages_by_instrument={strategy.spec.instrument_key: messages})
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=_CanonicalReaderStub(responses=()),
        retry_attempts=0,
    )

    runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=started_run.run_id)

    assert persisted is not None
    assert persisted.state == "running"
    assert persisted.checkpoint_ts_open == datetime(2026, 2, 17, 12, 49, tzinfo=timezone.utc)
    assert persisted.metadata_json["warmup"] == {
        "algorithm": "numeric_max_param_v1",
        "bars": 50,
        "processed_bars": 50,
        "satisfied": True,
    }


def test_timeframe_rollup_policy_closes_bucket_only_when_all_1m_present() -> None:
    """
    Ensure rollup closes derived timeframe bucket only after all 1m candles inside bucket exist.
    """
    policy = TimeframeRollupPolicy()
    timeframe = Timeframe("5m")

    progress = TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0)
    closures: list[bool] = []
    for offset in range(5):
        step = policy.advance(
            timeframe=timeframe,
            progress=progress,
            candle_ts_open=(
                datetime(2026, 2, 17, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=offset)
            ),
        )
        closures.append(step.bucket_closed)
        progress = step.progress

    assert closures == [False, False, False, False, True]

    progress = TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0)
    partial_closures: list[bool] = []
    for offset in range(3):
        step = policy.advance(
            timeframe=timeframe,
            progress=progress,
            candle_ts_open=(
                datetime(2026, 2, 17, 14, 2, tzinfo=timezone.utc) + timedelta(minutes=offset)
            ),
        )
        partial_closures.append(step.bucket_closed)
        progress = step.progress

    assert partial_closures == [False, False, False]


def test_live_runner_enforces_deterministic_run_processing_order_per_instrument() -> None:
    """
    Ensure runner processes runs per instrument in deterministic `(started_at, run_id)` order.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000905")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    second_strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository(reverse_active_runs=True)
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)
    strategy_repository.create(strategy=second_strategy)

    run_first = _create_running_run(
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        run_id=UUID("00000000-0000-0000-0000-00000000A905"),
        started_at=datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc),
        checkpoint_ts_open=datetime(2026, 2, 17, 14, 59, tzinfo=timezone.utc),
    )
    run_second = _create_running_run(
        user_id=user_id,
        strategy_id=second_strategy.strategy_id,
        run_id=UUID("00000000-0000-0000-0000-00000000B905"),
        started_at=datetime(2026, 2, 17, 15, 1, tzinfo=timezone.utc),
        checkpoint_ts_open=datetime(2026, 2, 17, 14, 59, tzinfo=timezone.utc),
    )
    run_repository.create(run=run_first)
    run_repository.create(run=run_second)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-order", _candle_at(datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc))),
            ),
        }
    )
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=_CanonicalReaderStub(responses=()),
        retry_attempts=0,
    )

    runner.run_once()
    first_occurrence = {
        run_id: run_repository.update_order.index(run_id)
        for run_id in {str(run_first.run_id), str(run_second.run_id)}
    }
    assert first_occurrence[str(run_first.run_id)] < first_occurrence[str(run_second.run_id)]


def test_live_runner_publishes_realtime_output_records_after_persistence() -> None:
    """
    Ensure live-runner publishes realtime metric/event records via injected publisher probe.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000906")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    started_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000E906"),
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 2, 17, 16, 0, tzinfo=timezone.utc),
        metadata_json={},
    )
    run_repository.create(run=started_run)
    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-rt", _candle_at(datetime(2026, 2, 17, 16, 0, tzinfo=timezone.utc))),
            ),
        }
    )
    realtime_output_probe = _RealtimeOutputProbe()
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=_CanonicalReaderStub(responses=()),
        retry_attempts=0,
        realtime_output_probe=realtime_output_probe,
    )

    runner.run_once()

    emitted = [record for batch in realtime_output_probe.batches for record in batch]
    metric_types = [
        record.metric_type
        for record in emitted
        if isinstance(record, StrategyRealtimeMetricV1)
    ]
    event_types = [
        record.event_type
        for record in emitted
        if isinstance(record, StrategyRealtimeEventV1)
    ]

    assert "checkpoint_ts_open" in metric_types
    assert "candles_processed_total" in metric_types
    assert "rollup_bucket_closed" in metric_types
    assert "run_state_changed" in event_types
    assert all(
        event_type in {"run_state_changed", "run_stopped", "run_failed"}
        for event_type in event_types
    )


def test_live_runner_emits_failed_telegram_notification_on_failed_run() -> None:
    """
    Ensure live-runner emits Telegram `failed` notification payload after run failure transition.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000907")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    started_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000E907"),
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 2, 17, 17, 0, tzinfo=timezone.utc),
        metadata_json={},
    )
    run_repository.create(run=started_run)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-fail", _candle_at(datetime(2026, 2, 17, 17, 0, tzinfo=timezone.utc))),
            ),
        }
    )
    notifier_probe = _TelegramNotifierProbe()
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=_CanonicalReaderStub(responses=()),
        retry_attempts=0,
        telegram_notifier_probe=notifier_probe,
        rollup_policy=_FailingRollupPolicy(),
    )

    runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=started_run.run_id)

    assert persisted is not None
    assert persisted.state == "failed"
    assert len(notifier_probe.notifications) == 1
    notification = notifier_probe.notifications[0]
    assert notification.event_type == "failed"
    assert notification.message_text == (
        f"FAILED | strategy_id={strategy.strategy_id} "
        f"| run_id={started_run.run_id} "
        "| error=rollup policy failed"
    )


def test_live_runner_keeps_best_effort_when_telegram_notifier_raises() -> None:
    """
    Ensure Telegram notifier exceptions never break live-runner iteration processing.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000908")
    strategy = _create_strategy(user_id=user_id, timeframe_code="1m")
    run_repository = _TrackingRunRepository()
    strategy_repository = InMemoryStrategyRepository()
    strategy_repository.create(strategy=strategy)

    started_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000E908"),
        user_id=user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 2, 17, 18, 0, tzinfo=timezone.utc),
        metadata_json={},
    )
    run_repository.create(run=started_run)

    stream = _StreamStub(
        messages_by_instrument={
            strategy.spec.instrument_key: (
                _message("m-fail", _candle_at(datetime(2026, 2, 17, 18, 0, tzinfo=timezone.utc))),
            ),
        }
    )
    runner = _build_runner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        stream=stream,
        canonical_reader=_CanonicalReaderStub(responses=()),
        retry_attempts=0,
        telegram_notifier_probe=_TelegramNotifierProbe(error=RuntimeError("telegram down")),
        rollup_policy=_FailingRollupPolicy(),
    )

    report = runner.run_once()
    persisted = run_repository.find_by_run_id(user_id=user_id, run_id=started_run.run_id)

    assert persisted is not None
    assert persisted.state == "failed"
    assert report.failed_runs == 1
    assert report.acked_messages == 1


def _build_runner(
    *,
    strategy_repository: InMemoryStrategyRepository,
    run_repository: _TrackingRunRepository,
    stream: _StreamStub,
    canonical_reader: _CanonicalReaderStub,
    retry_attempts: int,
    sleeper: _SleeperProbe | None = None,
    realtime_output_probe: _RealtimeOutputProbe | None = None,
    telegram_notifier_probe: _TelegramNotifierProbe | None = None,
    telegram_policy: TelegramNotificationPolicy | None = None,
    warmup_estimator: Callable[..., int] | None = None,
    rollup_policy: TimeframeRollupPolicy | None = None,
) -> StrategyLiveRunner:
    """
    Build StrategyLiveRunner with deterministic test doubles.

    Args:
        strategy_repository: In-memory strategy storage adapter.
        run_repository: Tracking run repository adapter.
        stream: Stream consumer stub.
        canonical_reader: Canonical reader stub.
        retry_attempts: Gap repair retries.
        sleeper: Optional sleeper probe.
        realtime_output_probe: Optional realtime output publisher probe.
        telegram_notifier_probe: Optional Telegram notifier probe.
        telegram_policy: Optional Telegram notification policy override.
        warmup_estimator: Optional warmup estimator override.
        rollup_policy: Optional timeframe rollup policy override.
    Returns:
        StrategyLiveRunner: Configured runner instance.
    Assumptions:
        Clock returns constant UTC timestamp for deterministic `updated_at`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return StrategyLiveRunner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        live_candle_stream=stream,
        canonical_candle_reader=canonical_reader,
        clock=_FixedClock(now_value=datetime(2026, 2, 17, 23, 0, tzinfo=timezone.utc)),
        sleeper=sleeper or _SleeperProbe(),
        repair_retry_attempts=retry_attempts,
        repair_backoff_seconds=1.0,
        realtime_output_publisher=realtime_output_probe,
        telegram_notifier=telegram_notifier_probe,
        telegram_notification_policy=telegram_policy,
        warmup_estimator=warmup_estimator,
        rollup_policy=rollup_policy,
    )


def _create_strategy(*, user_id: UserId, timeframe_code: str) -> Strategy:
    """
    Create deterministic immutable strategy fixture.

    Args:
        user_id: Strategy owner id.
        timeframe_code: Strategy timeframe code.
    Returns:
        Strategy: Persistable immutable strategy aggregate.
    Assumptions:
        Fixture indicator params produce warmup bars = 50.
    Raises:
        ValueError: If fixture values violate domain invariants.
    Side Effects:
        None.
    """
    spec = StrategySpecV1(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        timeframe=Timeframe(timeframe_code),
        indicators=(
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            },
        ),
        signal_template="MA(20,50)",
    )
    return Strategy.create(
        user_id=user_id,
        spec=spec,
        created_at=datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc),
    )


def _create_running_run(
    *,
    user_id: UserId,
    strategy_id: UUID,
    checkpoint_ts_open: datetime,
    run_id: UUID | None = None,
    started_at: datetime | None = None,
) -> StrategyRun:
    """
    Create deterministic run fixture in `running` state with normalized metadata.

    Args:
        user_id: Strategy owner id.
        strategy_id: Strategy id.
        checkpoint_ts_open: Last processed candle timestamp.
        run_id: Optional explicit run id.
        started_at: Optional run start timestamp.
    Returns:
        StrategyRun: Running run snapshot.
    Assumptions:
        Metadata mirrors normalized warmup/rollup payload shape.
    Raises:
        ValueError: If state transitions violate run invariants.
    Side Effects:
        None.
    """
    effective_run_id = run_id or UUID("00000000-0000-0000-0000-00000000C905")
    effective_started_at = started_at or datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
    starting = StrategyRun.start(
        run_id=effective_run_id,
        user_id=user_id,
        strategy_id=strategy_id,
        started_at=effective_started_at,
        metadata_json={
            "warmup": {
                "algorithm": "numeric_max_param_v1",
                "bars": 50,
                "processed_bars": 50,
                "satisfied": True,
            },
            "rollup": {
                "timeframe": "1m",
                "bucket_open_ts": None,
                "bucket_count_1m": 0,
            },
        },
    )
    warming = starting.transition_to(
        next_state="warming_up",
        changed_at=effective_started_at + timedelta(minutes=1),
        checkpoint_ts_open=None,
        last_error=None,
    )
    return warming.transition_to(
        next_state="running",
        changed_at=effective_started_at + timedelta(minutes=2),
        checkpoint_ts_open=checkpoint_ts_open,
        last_error=None,
    )


def _message(message_id: str, candle: CandleWithMeta) -> StrategyLiveCandleMessage:
    """
    Build deterministic stream message fixture.

    Args:
        message_id: Redis stream message id.
        candle: Parsed candle payload.
    Returns:
        StrategyLiveCandleMessage: Message fixture.
    Assumptions:
        Message id uniqueness is controlled by test fixture.
    Raises:
        None.
    Side Effects:
        None.
    """
    return StrategyLiveCandleMessage(message_id=message_id, candle=candle)


def _candle_at(ts_open: datetime) -> CandleWithMeta:
    """
    Build deterministic 1m candle fixture at requested open timestamp.

    Args:
        ts_open: Candle open timestamp in UTC.
    Returns:
        CandleWithMeta: Deterministic candle payload.
    Assumptions:
        Candle close is exactly one minute after open.
    Raises:
        ValueError: If candle invariants are violated.
    Side Effects:
        None.
    """
    instrument = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
    candle = Candle(
        instrument_id=instrument,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume_base=10.0,
        volume_quote=1005.0,
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=UtcTimestamp(ts_open + timedelta(minutes=1)),
        ingest_id=UUID("00000000-0000-0000-0000-00000000D905"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )
    return CandleWithMeta(candle=candle, meta=meta)
