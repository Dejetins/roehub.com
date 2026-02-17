from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.contexts.strategy.application.ports import (
    EventTypeV1,
    MetricTypeV1,
    NoOpStrategyRealtimeOutputPublisher,
    StrategyClock,
    StrategyLiveCandleStream,
    StrategyRealtimeEventV1,
    StrategyRealtimeMetricV1,
    StrategyRealtimeOutputPublisher,
    StrategyRepository,
    StrategyRunnerSleeper,
    StrategyRunRepository,
    serialize_realtime_event_payload_json,
)
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyRun,
    StrategyRunState,
    StrategySpecV1,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

from .timeframe_rollup import TimeframeRollupPolicy, TimeframeRollupProgress
from .warmup_estimator import estimate_strategy_warmup_bars

log = logging.getLogger(__name__)

_ONE_MINUTE = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerIterationReport:
    """
    StrategyLiveRunnerIterationReport — deterministic counters for one live-runner poll iteration.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - tests/unit/contexts/strategy/application/test_strategy_live_runner.py
    """

    polled_runs: int
    active_instruments: int
    read_messages: int
    acked_messages: int
    failed_runs: int


@dataclass(slots=True)
class _RunContext:
    """
    _RunContext — mutable processing context for one active run and immutable strategy snapshot.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/domain/entities/strategy.py
    """

    run: StrategyRun
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class _WarmupState:
    """
    _WarmupState — normalized warmup metadata snapshot kept in `strategy_runs.metadata_json`.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/application/services/warmup_estimator.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
    """

    bars: int
    processed_bars: int
    satisfied: bool


class StrategyLiveRunner:
    """
    StrategyLiveRunner — single-instance Strategy v1 live execution orchestrator.

    Core responsibilities:
    - poll active runs from Postgres-backed repository;
    - consume Redis Streams `md.candles.1m.<instrument_key>`;
    - enforce strict checkpoint monotonicity on `strategy_runs.checkpoint_ts_open`;
    - perform gap repair by read-only ClickHouse canonical reader;
    - compute/persist warmup metadata (`numeric_max_param_v1`);
    - apply run state transitions (`starting -> warming_up -> running`, stopping/failure).

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/application/ports/repositories/strategy_run_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """

    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository,
        run_repository: StrategyRunRepository,
        live_candle_stream: StrategyLiveCandleStream,
        canonical_candle_reader: CanonicalCandleReader,
        clock: StrategyClock,
        sleeper: StrategyRunnerSleeper,
        repair_retry_attempts: int,
        repair_backoff_seconds: float,
        realtime_output_publisher: StrategyRealtimeOutputPublisher | None = None,
        warmup_estimator: Callable[..., int] | None = None,
        rollup_policy: TimeframeRollupPolicy | None = None,
    ) -> None:
        """
        Initialize live-runner dependencies and deterministic retry policy.

        Args:
            strategy_repository: Strategy aggregate repository port.
            run_repository: Strategy run repository port.
            live_candle_stream: Live Redis stream consumer port.
            canonical_candle_reader: Canonical ClickHouse candle reader port.
            clock: UTC clock port.
            sleeper: Sleep abstraction for retry backoff.
            repair_retry_attempts: Maximum canonical repair retries per gap.
            repair_backoff_seconds: Base retry backoff in seconds.
            realtime_output_publisher: Optional realtime output publisher port.
            warmup_estimator: Optional warmup estimator override.
            rollup_policy: Optional timeframe rollup closure policy override.
        Returns:
            None.
        Assumptions:
            Worker process is single-instance in v1 deployment model.
        Raises:
            ValueError: If required dependencies or retry settings are invalid.
        Side Effects:
            None.
        """
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires strategy_repository")
        if run_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires run_repository")
        if live_candle_stream is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires live_candle_stream")
        if canonical_candle_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires canonical_candle_reader")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires clock")
        if sleeper is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyLiveRunner requires sleeper")
        if repair_retry_attempts < 0:
            raise ValueError("StrategyLiveRunner.repair_retry_attempts must be >= 0")
        if repair_backoff_seconds < 0:
            raise ValueError("StrategyLiveRunner.repair_backoff_seconds must be >= 0")

        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._live_candle_stream = live_candle_stream
        self._canonical_candle_reader = canonical_candle_reader
        self._clock = clock
        self._sleeper = sleeper
        self._repair_retry_attempts = repair_retry_attempts
        self._repair_backoff_seconds = repair_backoff_seconds
        self._realtime_output_publisher = (
            realtime_output_publisher
            if realtime_output_publisher is not None
            else NoOpStrategyRealtimeOutputPublisher()
        )
        self._warmup_estimator = warmup_estimator or estimate_strategy_warmup_bars
        self._rollup_policy = rollup_policy or TimeframeRollupPolicy()
        self._candles_processed_total_by_run: dict[str, int] = {}
        self._dropped_non_contiguous_total_by_run: dict[str, int] = {}

    def run_once(self) -> StrategyLiveRunnerIterationReport:
        """
        Execute one full poll/process iteration across all currently active strategy runs.

        Args:
            None.
        Returns:
            StrategyLiveRunnerIterationReport: Deterministic iteration counters.
        Assumptions:
            Active runs are processed in deterministic order by `(started_at, run_id)`.
        Raises:
            Exception: Unexpected storage/stream errors that should fail current iteration.
        Side Effects:
            Reads active runs and stream messages.
            Persists run checkpoint/state/metadata updates.
            Acknowledges processed Redis stream messages.
        """
        active_runs = tuple(
            sorted(
                self._run_repository.list_active_runs(),
                key=lambda run: (run.started_at, str(run.run_id)),
            )
        )
        contexts_by_instrument: dict[str, list[_RunContext]] = {}
        failed_runs = 0

        for run in active_runs:
            strategy = self._strategy_repository.find_by_strategy_id(
                user_id=run.user_id,
                strategy_id=run.strategy_id,
            )
            if strategy is None or strategy.is_deleted:
                run = self._mark_failed(
                    run=run,
                    error=RuntimeError("strategy not found or deleted"),
                    spec=None,
                )
                _ = run
                failed_runs += 1
                continue

            prepared_run = self._prepare_run(run=run, spec=strategy.spec)
            if prepared_run.is_active():
                contexts_by_instrument.setdefault(strategy.spec.instrument_key, []).append(
                    _RunContext(run=prepared_run, strategy=strategy)
                )

        read_messages = 0
        acked_messages = 0
        for instrument_key in sorted(contexts_by_instrument):
            contexts = sorted(
                contexts_by_instrument[instrument_key],
                key=lambda row: (row.run.started_at, str(row.run.run_id)),
            )
            messages = tuple(
                sorted(
                    self._live_candle_stream.read_closed_1m(instrument_key=instrument_key),
                    key=lambda message: (
                        message.candle.candle.ts_open.value,
                        message.message_id,
                    ),
                )
            )
            read_messages += len(messages)

            for message in messages:
                for context in contexts:
                    if context.run.state == "stopping":
                        continue
                    if context.run.state not in {"warming_up", "running", "starting"}:
                        continue
                    try:
                        context.run = self._process_candle(
                            run=context.run,
                            spec=context.strategy.spec,
                            candle=message.candle,
                        )
                    except Exception as error:  # noqa: BLE001
                        context.run = self._mark_failed(
                            run=context.run,
                            error=error,
                            spec=context.strategy.spec,
                        )
                        failed_runs += 1

                self._live_candle_stream.ack(
                    instrument_key=instrument_key,
                    message_id=message.message_id,
                )
                acked_messages += 1

            for context in contexts:
                if context.run.state == "stopping":
                    context.run = self._transition_state(
                        run=context.run,
                        spec=context.strategy.spec,
                        next_state="stopped",
                        checkpoint_ts_open=context.run.checkpoint_ts_open,
                        last_error=None,
                        metadata_json=context.run.metadata_json,
                    )

        return StrategyLiveRunnerIterationReport(
            polled_runs=len(active_runs),
            active_instruments=len(contexts_by_instrument),
            read_messages=read_messages,
            acked_messages=acked_messages,
            failed_runs=failed_runs,
        )

    def _prepare_run(self, *, run: StrategyRun, spec: StrategySpecV1) -> StrategyRun:
        """
        Normalize warmup metadata and apply pre-processing state transitions for one run.

        Args:
            run: Active run snapshot from storage.
            spec: Immutable strategy specification.
        Returns:
            StrategyRun: Persisted run snapshot ready for candle processing.
        Assumptions:
            Runner is responsible for transitioning `starting -> warming_up`.
        Raises:
            Exception: Storage/domain errors when persisting updated snapshot.
        Side Effects:
            May update run metadata.
            May transition run to `warming_up` or `running`.
        """
        metadata_json, warmup_state = self._normalized_metadata(
            metadata_json=run.metadata_json,
            spec=spec,
        )
        if metadata_json != run.metadata_json:
            run = self._persist_same_state(
                run=run,
                checkpoint_ts_open=run.checkpoint_ts_open,
                last_error=run.last_error,
                metadata_json=metadata_json,
            )
            self._publish_snapshot_metrics(run=run, spec=spec, ts=run.updated_at)

        if run.state == "starting":
            run = self._transition_state(
                run=run,
                spec=spec,
                next_state="warming_up",
                checkpoint_ts_open=run.checkpoint_ts_open,
                last_error=None,
                metadata_json=run.metadata_json,
            )

        if run.state == "warming_up" and warmup_state.satisfied:
            run = self._transition_state(
                run=run,
                spec=spec,
                next_state="running",
                checkpoint_ts_open=run.checkpoint_ts_open,
                last_error=None,
                metadata_json=run.metadata_json,
            )
        return run

    def _process_candle(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        candle: CandleWithMeta,
    ) -> StrategyRun:
        """
        Process one stream candle against strict checkpoint rules and repair policy.

        Args:
            run: Current active run snapshot.
            spec: Immutable strategy specification.
            candle: Parsed live candle payload.
        Returns:
            StrategyRun: Persisted run snapshot after processing.
        Assumptions:
            Input candle belongs to run instrument stream.
        Raises:
            Exception: Storage/domain/canonical-reader errors.
        Side Effects:
            May read canonical candles for gap repair.
            May update checkpoint/metadata/state in run repository.
        """
        if run.state == "starting":
            run = self._transition_state(
                run=run,
                spec=spec,
                next_state="warming_up",
                checkpoint_ts_open=run.checkpoint_ts_open,
                last_error=None,
                metadata_json=run.metadata_json,
            )
        if run.state not in {"warming_up", "running"}:
            return run

        ts_open = candle.candle.ts_open.value
        checkpoint = run.checkpoint_ts_open
        if checkpoint is not None and ts_open <= checkpoint:
            self._increment_dropped_non_contiguous_total(run_id=run.run_id)
            return run

        if checkpoint is not None and ts_open > checkpoint + _ONE_MINUTE:
            run, is_continuous = self._repair_gap(
                run=run,
                spec=spec,
                target_ts_open=ts_open,
            )
            if not is_continuous:
                return run

        return self._accept_contiguous_candle(run=run, spec=spec, candle=candle)

    def _accept_contiguous_candle(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        candle: CandleWithMeta,
    ) -> StrategyRun:
        """
        Persist one contiguous candle and update warmup/rollup progress deterministically.

        Args:
            run: Current active run snapshot.
            spec: Immutable strategy specification.
            candle: Candle to apply.
        Returns:
            StrategyRun: Persisted run snapshot with advanced checkpoint.
        Assumptions:
            Candle is contiguous to current checkpoint or starts checkpoint sequence.
        Raises:
            Exception: Storage/domain errors while persisting updated run.
        Side Effects:
            Updates run checkpoint and metadata in Postgres storage.
        """
        ts_open = candle.candle.ts_open.value
        checkpoint = run.checkpoint_ts_open
        if checkpoint is not None:
            expected = checkpoint + _ONE_MINUTE
            if ts_open != expected:
                self._increment_dropped_non_contiguous_total(run_id=run.run_id)
                return run

        metadata_json, warmup_state = self._normalized_metadata(
            metadata_json=run.metadata_json,
            spec=spec,
        )
        rollup_progress = _read_rollup_progress(
            metadata_json=metadata_json,
            timeframe_code=spec.timeframe.code,
        )
        rollup_step = self._rollup_policy.advance(
            timeframe=spec.timeframe,
            progress=rollup_progress,
            candle_ts_open=ts_open,
        )
        processed_bars = warmup_state.processed_bars + (1 if rollup_step.bucket_closed else 0)
        normalized_warmup = _WarmupState(
            bars=warmup_state.bars,
            processed_bars=processed_bars,
            satisfied=processed_bars >= warmup_state.bars,
        )
        metadata_json["warmup"] = _serialize_warmup(warmup_state=normalized_warmup)
        metadata_json["rollup"] = _serialize_rollup_progress(
            progress=rollup_step.progress,
            timeframe_code=spec.timeframe.code,
        )

        run = self._persist_same_state(
            run=run,
            checkpoint_ts_open=ts_open,
            last_error=run.last_error,
            metadata_json=metadata_json,
        )
        candles_processed_total = self._increment_candles_processed_total(run_id=run.run_id)
        self._publish_snapshot_metrics(
            run=run,
            spec=spec,
            ts=run.updated_at,
            extra_metrics=(
                ("candles_processed_total", str(candles_processed_total)),
                ("rollup_bucket_closed", _bool_as_flag(rollup_step.bucket_closed)),
                ("gap_detected", "0"),
                ("repair_missing_bars", "0"),
                ("repair_attempt", "0"),
                ("repair_continuous", "1"),
            ),
        )
        if run.state == "warming_up" and normalized_warmup.satisfied:
            run = self._transition_state(
                run=run,
                spec=spec,
                next_state="running",
                checkpoint_ts_open=run.checkpoint_ts_open,
                last_error=None,
                metadata_json=run.metadata_json,
            )
        return run

    def _repair_gap(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        target_ts_open: datetime,
    ) -> tuple[StrategyRun, bool]:
        """
        Attempt to repair missing contiguous candles from canonical ClickHouse storage.

        Args:
            run: Current active run snapshot.
            spec: Immutable strategy specification.
            target_ts_open: Incoming stream candle `ts_open` that exposed the gap.
        Returns:
            tuple[StrategyRun, bool]: Updated run and continuity flag.
        Assumptions:
            Repair is read-only and does not trigger ingestion writes.
        Raises:
            Exception: Canonical reader/storage errors.
        Side Effects:
            Reads canonical candles and may advance run checkpoint via repository updates.
            Sleeps between attempts using configured backoff policy.
        """
        if run.checkpoint_ts_open is None:
            return run, False

        first_expected_start = run.checkpoint_ts_open + _ONE_MINUTE
        missing_bars = _missing_bars_until_target(
            expected_start=first_expected_start,
            target_ts_open=target_ts_open,
        )

        for attempt in range(self._repair_retry_attempts + 1):
            expected_start = run.checkpoint_ts_open + _ONE_MINUTE
            if expected_start >= target_ts_open:
                return run, True

            repaired_rows = tuple(
                sorted(
                    self._canonical_candle_reader.read_1m(
                        spec.instrument_id,
                        TimeRange(
                            start=UtcTimestamp(expected_start),
                            end=UtcTimestamp(target_ts_open),
                        ),
                    ),
                    key=lambda row: row.candle.ts_open.value,
                )
            )

            contiguous_rows: list[CandleWithMeta] = []
            expected_cursor = expected_start
            for repaired in repaired_rows:
                row_ts_open = repaired.candle.ts_open.value
                if row_ts_open < expected_cursor:
                    continue
                if row_ts_open != expected_cursor:
                    break
                contiguous_rows.append(repaired)
                expected_cursor = expected_cursor + _ONE_MINUTE

            if expected_cursor >= target_ts_open:
                for repaired in contiguous_rows:
                    run = self._accept_contiguous_candle(run=run, spec=spec, candle=repaired)
                self._publish_snapshot_metrics(
                    run=run,
                    spec=spec,
                    ts=run.updated_at,
                    extra_metrics=(
                        ("gap_detected", "1"),
                        ("repair_missing_bars", str(missing_bars)),
                        ("repair_attempt", str(attempt + 1)),
                        ("repair_continuous", "1"),
                    ),
                )
                return run, True

            if attempt == self._repair_retry_attempts:
                break
            backoff_seconds = float(attempt + 1) * self._repair_backoff_seconds
            if backoff_seconds > 0:
                self._sleeper.sleep(seconds=backoff_seconds)

        return run, False

    def _persist_same_state(
        self,
        *,
        run: StrategyRun,
        checkpoint_ts_open: datetime | None,
        last_error: str | None,
        metadata_json: Mapping[str, Any],
    ) -> StrategyRun:
        """
        Persist run snapshot update without changing run state literal.

        Args:
            run: Current run snapshot.
            checkpoint_ts_open: Updated checkpoint timestamp.
            last_error: Updated error text.
            metadata_json: Updated metadata mapping.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Run remains in the same state and `updated_at` is monotonic.
        Raises:
            Exception: Domain/storage validation errors.
        Side Effects:
            Updates one run row in repository.
        """
        snapshot = StrategyRun(
            run_id=run.run_id,
            user_id=run.user_id,
            strategy_id=run.strategy_id,
            state=run.state,
            started_at=run.started_at,
            stopped_at=run.stopped_at,
            checkpoint_ts_open=checkpoint_ts_open,
            last_error=last_error,
            updated_at=_monotonic_changed_at(now=self._clock.now(), previous=run.updated_at),
            metadata_json=metadata_json,
        )
        return self._run_repository.update(run=snapshot)

    def _transition_state(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1 | None,
        next_state: StrategyRunState,
        checkpoint_ts_open: datetime | None,
        last_error: str | None,
        metadata_json: Mapping[str, Any],
    ) -> StrategyRun:
        """
        Transition run state and persist metadata/checkpoint snapshot atomically.

        Args:
            run: Current run snapshot.
            spec: Strategy specification snapshot for realtime output payload routing.
            next_state: Target state literal.
            checkpoint_ts_open: Target checkpoint value.
            last_error: Target error value.
            metadata_json: Target metadata payload.
        Returns:
            StrategyRun: Persisted transitioned snapshot.
        Assumptions:
            Transition graph is validated by `StrategyRun.transition_to`.
        Raises:
            Exception: Domain/storage validation errors.
        Side Effects:
            Updates one run row in repository.
        """
        transitioned = run.transition_to(
            next_state=next_state,
            changed_at=_monotonic_changed_at(now=self._clock.now(), previous=run.updated_at),
            checkpoint_ts_open=checkpoint_ts_open,
            last_error=last_error,
        )
        if transitioned.metadata_json != metadata_json:
            transitioned = StrategyRun(
                run_id=transitioned.run_id,
                user_id=transitioned.user_id,
                strategy_id=transitioned.strategy_id,
                state=transitioned.state,
                started_at=transitioned.started_at,
                stopped_at=transitioned.stopped_at,
                checkpoint_ts_open=transitioned.checkpoint_ts_open,
                last_error=transitioned.last_error,
                updated_at=transitioned.updated_at,
                metadata_json=metadata_json,
            )
        persisted = self._run_repository.update(run=transitioned)
        if spec is not None:
            self._publish_snapshot_metrics(run=persisted, spec=spec, ts=persisted.updated_at)
            self._publish_transition_events(previous=run, current=persisted, spec=spec)
        return persisted

    def _mark_failed(
        self,
        *,
        run: StrategyRun,
        error: Exception,
        spec: StrategySpecV1 | None,
    ) -> StrategyRun:
        """
        Transition run into `failed` state and store deterministic last_error text.

        Args:
            run: Current run snapshot.
            error: Caught processing error.
            spec: Optional strategy specification snapshot for realtime output routing.
        Returns:
            StrategyRun: Persisted failed run snapshot.
        Assumptions:
            Active-run states allow transition to `failed` in Strategy v1.
        Raises:
            Exception: Domain/storage errors while persisting failed state.
        Side Effects:
            Updates run row to terminal failed state.
        """
        if run.state == "failed":
            return run
        message = str(error).strip() or error.__class__.__name__
        log.exception("strategy live-runner failed run_id=%s reason=%s", run.run_id, message)
        return self._transition_state(
            run=run,
            spec=spec,
            next_state="failed",
            checkpoint_ts_open=run.checkpoint_ts_open,
            last_error=message,
            metadata_json=run.metadata_json,
        )

    def _publish_snapshot_metrics(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        ts: datetime,
        extra_metrics: tuple[tuple[MetricTypeV1, str], ...] = (),
    ) -> None:
        """
        Publish deterministic realtime metric snapshot after successful run persistence.

        Args:
            run: Persisted run snapshot.
            spec: Strategy specification used for routing fields.
            ts: Publish timestamp tied to persisted run update.
            extra_metrics: Additional metric values for current processing step.
        Returns:
            None.
        Assumptions:
            Method is called only after successful repository update.
        Raises:
            None.
        Side Effects:
            Emits best-effort metric records via realtime output publisher port.
        """
        metric_values = self._snapshot_metric_values(run=run, spec=spec)
        for metric_type, value in extra_metrics:
            metric_values[metric_type] = value

        records = tuple(
            self._metric_record(
                run=run,
                spec=spec,
                ts=ts,
                metric_type=metric_type,
                value=value,
            )
            for metric_type, value in sorted(metric_values.items(), key=lambda row: row[0])
        )
        self._realtime_output_publisher.publish_records_v1(records=records)

    def _snapshot_metric_values(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
    ) -> dict[MetricTypeV1, str]:
        """
        Build baseline metric value set for one persisted run snapshot.

        Args:
            run: Persisted run snapshot.
            spec: Strategy specification for warmup/rollup normalization.
        Returns:
            dict[MetricTypeV1, str]: Baseline metric values keyed by fixed v1 metric types.
        Assumptions:
            Metadata payload follows Strategy live-runner normalization contract.
        Raises:
            ValueError: If timestamp normalization fails.
        Side Effects:
            None.
        """
        estimated_bars = max(int(self._warmup_estimator(spec=spec)), 1)
        warmup_state = _read_warmup_state(
            metadata_json=run.metadata_json,
            estimated_bars=estimated_bars,
        )
        rollup_progress = _read_rollup_progress(
            metadata_json=run.metadata_json,
            timeframe_code=spec.timeframe.code,
        )
        return {
            "warmup_processed_bars": str(warmup_state.processed_bars),
            "checkpoint_ts_open": (
                _isoformat_utc(run.checkpoint_ts_open)
                if run.checkpoint_ts_open is not None
                else ""
            ),
            "lag_seconds": str(
                _lag_seconds(
                    now=self._clock.now(),
                    checkpoint_ts_open=run.checkpoint_ts_open,
                )
            ),
            "candles_processed_total": str(self._candles_processed_total(run_id=run.run_id)),
            "warmup_required_bars": str(warmup_state.bars),
            "warmup_satisfied": _bool_as_flag(warmup_state.satisfied),
            "run_state": run.state,
            "rollup_bucket_count_1m": str(rollup_progress.bucket_count_1m),
            "rollup_bucket_open_ts": (
                _isoformat_utc(rollup_progress.bucket_open_ts)
                if rollup_progress.bucket_open_ts is not None
                else ""
            ),
            "dropped_non_contiguous_total": str(
                self._dropped_non_contiguous_total(run_id=run.run_id)
            ),
        }

    def _metric_record(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        ts: datetime,
        metric_type: MetricTypeV1,
        value: str,
    ) -> StrategyRealtimeMetricV1:
        """
        Build one realtime metric record with fixed routing fields for run and strategy.

        Args:
            run: Persisted run snapshot.
            spec: Strategy specification snapshot.
            ts: Publish timestamp.
            metric_type: Fixed metric type literal.
            value: Metric value encoded as string.
        Returns:
            StrategyRealtimeMetricV1: Validated metric record.
        Assumptions:
            `value` is already serialized to wire-format string.
        Raises:
            ValueError: If record construction violates realtime metric contract.
        Side Effects:
            None.
        """
        return StrategyRealtimeMetricV1(
            user_id=run.user_id,
            ts=ts,
            strategy_id=run.strategy_id,
            run_id=run.run_id,
            metric_type=metric_type,
            value=value,
            instrument_key=spec.instrument_key,
            timeframe=spec.timeframe.code,
        )

    def _publish_transition_events(
        self,
        *,
        previous: StrategyRun,
        current: StrategyRun,
        spec: StrategySpecV1,
    ) -> None:
        """
        Publish user-facing realtime events for run state transitions.

        Args:
            previous: Run snapshot before transition persistence.
            current: Persisted run snapshot after transition.
            spec: Strategy specification used for routing fields.
        Returns:
            None.
        Assumptions:
            Only fixed v1 event types are emitted.
        Raises:
            None.
        Side Effects:
            Emits best-effort event records via realtime output publisher port.
        """
        if previous.state == current.state:
            return

        transition_payload = serialize_realtime_event_payload_json(
            payload={
                "from": previous.state,
                "to": current.state,
            }
        )
        events: list[StrategyRealtimeEventV1] = [
            self._event_record(
                run=current,
                spec=spec,
                ts=current.updated_at,
                event_type="run_state_changed",
                payload_json=transition_payload,
            )
        ]
        if current.state == "stopped":
            events.append(
                self._event_record(
                    run=current,
                    spec=spec,
                    ts=current.updated_at,
                    event_type="run_stopped",
                    payload_json=serialize_realtime_event_payload_json(payload={}),
                )
            )
        if current.state == "failed":
            events.append(
                self._event_record(
                    run=current,
                    spec=spec,
                    ts=current.updated_at,
                    event_type="run_failed",
                    payload_json=serialize_realtime_event_payload_json(
                        payload={"error": current.last_error or ""}
                    ),
                )
            )
        self._realtime_output_publisher.publish_records_v1(records=tuple(events))

    def _event_record(
        self,
        *,
        run: StrategyRun,
        spec: StrategySpecV1,
        ts: datetime,
        event_type: EventTypeV1,
        payload_json: str,
    ) -> StrategyRealtimeEventV1:
        """
        Build one realtime event record with fixed routing fields for run and strategy.

        Args:
            run: Persisted run snapshot.
            spec: Strategy specification snapshot.
            ts: Publish timestamp.
            event_type: Fixed event type literal.
            payload_json: Canonical JSON object string payload.
        Returns:
            StrategyRealtimeEventV1: Validated event record.
        Assumptions:
            `payload_json` is serialized with deterministic sort order and ASCII output.
        Raises:
            ValueError: If record construction violates realtime event contract.
        Side Effects:
            None.
        """
        return StrategyRealtimeEventV1(
            user_id=run.user_id,
            ts=ts,
            strategy_id=run.strategy_id,
            run_id=run.run_id,
            event_type=event_type,
            payload_json=payload_json,
            instrument_key=spec.instrument_key,
            timeframe=spec.timeframe.code,
        )

    def _increment_candles_processed_total(self, *, run_id: UUID) -> int:
        """
        Increase in-memory contiguous candle counter for one run.

        Args:
            run_id: Run identifier.
        Returns:
            int: Updated counter value.
        Assumptions:
            Counter scope is process-local for current runner instance.
        Raises:
            None.
        Side Effects:
            Mutates internal counter dictionary.
        """
        key = str(run_id)
        value = self._candles_processed_total_by_run.get(key, 0) + 1
        self._candles_processed_total_by_run[key] = value
        return value

    def _candles_processed_total(self, *, run_id: UUID) -> int:
        """
        Read process-local contiguous candle counter value for one run.

        Args:
            run_id: Run identifier.
        Returns:
            int: Current counter value.
        Assumptions:
            Missing run id means counter value `0`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._candles_processed_total_by_run.get(str(run_id), 0)

    def _increment_dropped_non_contiguous_total(self, *, run_id: UUID) -> int:
        """
        Increase in-memory dropped non-contiguous candle counter for one run.

        Args:
            run_id: Run identifier.
        Returns:
            int: Updated counter value.
        Assumptions:
            Counter scope is process-local for current runner instance.
        Raises:
            None.
        Side Effects:
            Mutates internal counter dictionary.
        """
        key = str(run_id)
        value = self._dropped_non_contiguous_total_by_run.get(key, 0) + 1
        self._dropped_non_contiguous_total_by_run[key] = value
        return value

    def _dropped_non_contiguous_total(self, *, run_id: UUID) -> int:
        """
        Read process-local dropped non-contiguous candle counter value for one run.

        Args:
            run_id: Run identifier.
        Returns:
            int: Current counter value.
        Assumptions:
            Missing run id means counter value `0`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._dropped_non_contiguous_total_by_run.get(str(run_id), 0)

    def _normalized_metadata(
        self,
        *,
        metadata_json: Mapping[str, Any],
        spec: StrategySpecV1,
    ) -> tuple[dict[str, Any], _WarmupState]:
        """
        Normalize runtime metadata payload for warmup and rollup deterministic fields.

        Args:
            metadata_json: Existing persisted metadata mapping.
            spec: Immutable strategy specification.
        Returns:
            tuple[dict[str, Any], _WarmupState]:
                Normalized metadata mapping and parsed warmup state.
        Assumptions:
            Warmup bars are computed by `numeric_max_param_v1` estimator from spec indicators.
        Raises:
            ValueError: If estimator returns invalid warmup bars value.
        Side Effects:
            None.
        """
        normalized = dict(metadata_json)
        estimated_bars = max(int(self._warmup_estimator(spec=spec)), 1)
        warmup_state = _read_warmup_state(
            metadata_json=normalized,
            estimated_bars=estimated_bars,
        )
        normalized["warmup"] = _serialize_warmup(warmup_state=warmup_state)
        normalized["rollup"] = _serialize_rollup_progress(
            progress=_read_rollup_progress(
                metadata_json=normalized,
                timeframe_code=spec.timeframe.code,
            ),
            timeframe_code=spec.timeframe.code,
        )
        return normalized, warmup_state


def _missing_bars_until_target(*, expected_start: datetime, target_ts_open: datetime) -> int:
    """
    Calculate missing 1m bars count between expected checkpoint cursor and target timestamp.

    Args:
        expected_start: Expected next contiguous 1m timestamp.
        target_ts_open: Incoming candle timestamp that exposed potential gap.
    Returns:
        int: Non-negative count of missing 1m bars before target timestamp.
    Assumptions:
        Timestamps are UTC and normalized to minute granularity.
    Raises:
        None.
    Side Effects:
        None.
    """
    if target_ts_open <= expected_start:
        return 0
    missing_seconds = (target_ts_open - expected_start).total_seconds()
    return max(int(missing_seconds // 60), 0)


def _lag_seconds(*, now: datetime, checkpoint_ts_open: datetime | None) -> int:
    """
    Compute non-negative lag in seconds between current time and run checkpoint timestamp.

    Args:
        now: Current clock timestamp.
        checkpoint_ts_open: Run checkpoint timestamp.
    Returns:
        int: Lag in whole seconds, or `0` when checkpoint is absent.
    Assumptions:
        Inputs are timezone-aware UTC datetimes.
    Raises:
        ValueError: If datetime normalization fails.
    Side Effects:
        None.
    """
    if checkpoint_ts_open is None:
        return 0
    now_value = UtcTimestamp(now).value
    checkpoint_value = UtcTimestamp(checkpoint_ts_open).value
    lag = int((now_value - checkpoint_value).total_seconds())
    return max(lag, 0)


def _bool_as_flag(value: bool) -> str:
    """
    Convert boolean value into deterministic wire-format `\"0\"`/`\"1\"` string.

    Args:
        value: Boolean source value.
    Returns:
        str: `\"1\"` when value is true, otherwise `\"0\"`.
    Assumptions:
        Realtime metric schema stores boolean-like values as strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    return "1" if value else "0"


def _read_warmup_state(*, metadata_json: Mapping[str, Any], estimated_bars: int) -> _WarmupState:
    """
    Read and normalize warmup metadata payload with deterministic defaults.

    Args:
        metadata_json: Existing run metadata payload.
        estimated_bars: Estimator-derived warmup bars count.
    Returns:
        _WarmupState: Normalized warmup metadata state.
    Assumptions:
        Missing or malformed warmup payload falls back to deterministic defaults.
    Raises:
        ValueError: If estimated bars is invalid.
    Side Effects:
        None.
    """
    if estimated_bars <= 0:
        raise ValueError("warmup estimated bars must be > 0")

    raw_warmup = metadata_json.get("warmup")
    processed_bars = 0
    if isinstance(raw_warmup, Mapping):
        raw_processed = raw_warmup.get("processed_bars", 0)
        processed_bars = _safe_non_negative_int(raw_processed)

    satisfied = processed_bars >= estimated_bars
    return _WarmupState(
        bars=estimated_bars,
        processed_bars=processed_bars,
        satisfied=satisfied,
    )


def _serialize_warmup(*, warmup_state: _WarmupState) -> dict[str, Any]:
    """
    Serialize warmup state into deterministic metadata payload shape.

    Args:
        warmup_state: Normalized warmup state.
    Returns:
        dict[str, Any]: Warmup metadata mapping.
    Assumptions:
        Algorithm literal for v1 is fixed to `numeric_max_param_v1`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "algorithm": "numeric_max_param_v1",
        "bars": warmup_state.bars,
        "processed_bars": warmup_state.processed_bars,
        "satisfied": warmup_state.satisfied,
    }


def _read_rollup_progress(
    *,
    metadata_json: Mapping[str, Any],
    timeframe_code: str,
) -> TimeframeRollupProgress:
    """
    Read and normalize rollup progress state from metadata payload.

    Args:
        metadata_json: Existing run metadata payload.
        timeframe_code: Strategy timeframe code.
    Returns:
        TimeframeRollupProgress: Normalized rollup state.
    Assumptions:
        Rollup state resets when timeframe code changes or payload is malformed.
    Raises:
        ValueError: If serialized timestamp cannot be parsed.
    Side Effects:
        None.
    """
    raw_rollup = metadata_json.get("rollup")
    if not isinstance(raw_rollup, Mapping):
        return TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0)

    raw_timeframe = raw_rollup.get("timeframe")
    if str(raw_timeframe) != timeframe_code:
        return TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0)

    bucket_count = _safe_non_negative_int(raw_rollup.get("bucket_count_1m", 0))
    raw_bucket_open = raw_rollup.get("bucket_open_ts")
    if raw_bucket_open in (None, ""):
        return TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0)

    bucket_open_text = str(raw_bucket_open).strip()
    bucket_open_ts = _parse_iso_utc(text=bucket_open_text)
    return TimeframeRollupProgress(
        bucket_open_ts=bucket_open_ts,
        bucket_count_1m=bucket_count,
    )


def _serialize_rollup_progress(
    *,
    progress: TimeframeRollupProgress,
    timeframe_code: str,
) -> dict[str, Any]:
    """
    Serialize rollup progress into deterministic metadata payload shape.

    Args:
        progress: Normalized rollup progress.
        timeframe_code: Strategy timeframe code.
    Returns:
        dict[str, Any]: Rollup metadata mapping.
    Assumptions:
        `bucket_open_ts` is serialized with UTC `Z` suffix when present.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "timeframe": timeframe_code,
        "bucket_open_ts": (
            _isoformat_utc(progress.bucket_open_ts) if progress.bucket_open_ts is not None else None
        ),
        "bucket_count_1m": progress.bucket_count_1m,
    }


def _safe_non_negative_int(value: Any) -> int:
    """
    Convert arbitrary value into non-negative integer with deterministic fallback to zero.

    Args:
        value: Raw value from metadata payload.
    Returns:
        int: Parsed non-negative integer.
    Assumptions:
        Malformed values are treated as zero instead of raising hard failure.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    if parsed < 0:
        return 0
    return parsed


def _monotonic_changed_at(*, now: datetime, previous: datetime) -> datetime:
    """
    Enforce monotonic non-decreasing timestamp for run snapshot persistence.

    Args:
        now: Current clock value.
        previous: Previous persisted `updated_at` value.
    Returns:
        datetime: Monotonic timestamp used for new snapshot.
    Assumptions:
        Both inputs are timezone-aware UTC datetimes.
    Raises:
        ValueError: If `now` or `previous` cannot be normalized into UtcTimestamp.
    Side Effects:
        None.
    """
    normalized_now = UtcTimestamp(now).value
    normalized_previous = UtcTimestamp(previous).value
    if normalized_now < normalized_previous:
        return normalized_previous
    return normalized_now


def _parse_iso_utc(*, text: str) -> datetime:
    """
    Parse ISO UTC datetime string with `Z` suffix support.

    Args:
        text: ISO datetime string.
    Returns:
        datetime: Parsed timezone-aware datetime value.
    Assumptions:
        UtcTimestamp provides final UTC normalization.
    Raises:
        ValueError: If text is not valid ISO datetime.
    Side Effects:
        None.
    """
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    return UtcTimestamp(datetime.fromisoformat(normalized)).value


def _isoformat_utc(value: datetime) -> str:
    """
    Serialize UTC datetime into millisecond ISO format with `Z` suffix.

    Args:
        value: Timezone-aware UTC datetime.
    Returns:
        str: ISO string like `2026-02-10T12:34:00.000Z`.
    Assumptions:
        UtcTimestamp enforces UTC and millisecond precision.
    Raises:
        ValueError: If input datetime is invalid for UtcTimestamp.
    Side Effects:
        None.
    """
    return str(UtcTimestamp(value))
