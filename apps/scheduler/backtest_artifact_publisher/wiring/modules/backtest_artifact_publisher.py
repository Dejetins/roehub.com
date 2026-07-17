from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import IO, Callable, Literal, Mapping, Protocol
from zoneinfo import ZoneInfo

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from apps.api.wiring.modules.indicators import (
    build_artifact_precompute_indicators_compute,
    build_indicators_registry,
)
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.backtest_artifacts.adapters.outbound import (
    AtomicArtifactCurrentPointerWriterV2,
    BacktestArtifactPathBuilderV2,
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.application.services import (
    ArtifactCoordinatesV2,
    ArtifactSlotPublishErrorV2,
    ArtifactTailRebuildBarsV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
    BacktestSignalRulesEngineV2,
    YamlBacktestArtifactLoaderV2,
    artifact_coordinates_from_market_id_v2,
)
from trading.contexts.backtest_artifacts.application.use_cases import (
    PublishBacktestArtifactsV2Request,
    PublishBacktestArtifactsV2UseCase,
)
from trading.contexts.indicators.application.services import GridBuilder
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.application.ports.stores import EnabledInstrumentReader
from trading.platform.time.system_clock import SystemClock  # noqa: F401

log = logging.getLogger(__name__)

type PublisherRunStatus = Literal[
    "succeeded",
    "inactive_slot_pinned",
    "validation_failed",
    "lock_held",
    "unexpected_error",
]
NowProvider = Callable[[], datetime]
StopRequested = Callable[[], bool]


def _default_now_provider() -> datetime:
    """
    Return the default timezone-aware UTC wall clock for the scheduler runtime.

    Args:
        None.
    Returns:
        datetime: Timezone-aware UTC datetime.
    Assumptions:
        Scheduler cadence, logs, and metrics use UTC internally and convert to Moscow explicitly.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    return datetime.now(timezone.utc)


def _validated_utc_datetime(value: datetime) -> datetime:
    """
    Require a timezone-aware UTC datetime from scheduler clock dependencies.

    Args:
        value: Datetime candidate returned by a scheduler clock dependency.
    Returns:
        datetime: The same datetime normalized to UTC.
    Assumptions:
        Scheduler cadence must never depend on ambiguous host-local timezone math.
    Raises:
        ValueError: If the datetime is naive or not UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    utc_offset = value.utcoffset()
    if value.tzinfo is None or utc_offset is None:
        raise ValueError(
            "backtest artifact publisher scheduler now_provider must return "
            "timezone-aware UTC datetime"
        )
    if utc_offset.total_seconds() != 0:
        raise ValueError(
            "backtest artifact publisher scheduler now_provider must return UTC datetime"
        )
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class BacktestArtifactPublisherSchedule:
    """
    Deterministic daily schedule anchored to `03:05 Europe/Moscow`.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - infra/macos/launchd/com.roehub.backtest-artifact-publisher.plist
    """

    timezone_name: str = "Europe/Moscow"
    hour: int = 3
    minute: int = 5
    enabled: bool = True

    def __post_init__(self) -> None:
        """
        Validate the fixed Moscow daily scheduler contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            EPIC R11-02 fixes the runtime cadence at `03:05 Europe/Moscow`.
        Raises:
            ValueError: If timezone name or clock fields are invalid.
        Side Effects:
            Validates that the configured timezone exists.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        ZoneInfo(self.timezone_name)
        if not isinstance(self.enabled, bool):
            raise ValueError("BacktestArtifactPublisherSchedule.enabled must be a bool")
        if not 0 <= self.hour <= 23:
            raise ValueError("BacktestArtifactPublisherSchedule.hour must be in [0, 23]")
        if not 0 <= self.minute <= 59:
            raise ValueError("BacktestArtifactPublisherSchedule.minute must be in [0, 59]")

    def next_run_after(
        self,
        *,
        now_utc: datetime,
        last_run_local_date: date | None,
    ) -> datetime:
        """
        Compute the next scheduler fire time in UTC using explicit Moscow wall clock rules.

        Args:
            now_utc: Current timezone-aware UTC datetime.
            last_run_local_date: Moscow calendar date already processed by this process.
        Returns:
            datetime: Next run timestamp in UTC.
        Assumptions:
            The scheduler fires at most once per Moscow calendar date inside one process.
        Raises:
            ValueError: If `now_utc` is naive or not UTC.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py
        """
        current_utc = _validated_utc_datetime(now_utc)
        timezone_info = ZoneInfo(self.timezone_name)
        local_now = current_utc.astimezone(timezone_info)
        candidate_local = datetime.combine(
            local_now.date(),
            time(hour=self.hour, minute=self.minute),
            tzinfo=timezone_info,
        )
        if local_now > candidate_local or last_run_local_date == candidate_local.date():
            candidate_local += timedelta(days=1)
        return candidate_local.astimezone(timezone.utc)

    def local_run_date(self, *, run_at_utc: datetime) -> date:
        """
        Resolve the Moscow calendar date corresponding to one scheduled UTC fire time.

        Args:
            run_at_utc: Scheduled run timestamp in UTC.
        Returns:
            date: Moscow calendar date of the scheduled run.
        Assumptions:
            The schedule date is the deduplication key for one process lifetime.
        Raises:
            ValueError: If `run_at_utc` is naive or not UTC.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py
        """
        return _validated_utc_datetime(run_at_utc).astimezone(ZoneInfo(self.timezone_name)).date()


class BacktestArtifactPublisherHostLock(Protocol):
    """
    Host-level lock contract guarding scheduler runs across concurrent processes.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """

    def try_acquire(self) -> bool:
        """Attempt a fail-fast non-blocking lock acquisition."""
        ...

    def release(self) -> None:
        """Release a previously acquired host-level lock."""
        ...


@dataclass(slots=True)
class FileBacktestArtifactPublisherHostLock:
    """
    Non-blocking advisory file lock used by the Mac Studio scheduler service.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/runbooks/mac-studio-native-backend-operations.md
    Related:
      - infra/macos/launchd/com.roehub.backtest-artifact-publisher.plist
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """

    path: Path
    _handle: IO[str] | None = field(default=None, init=False, repr=False)

    def try_acquire(self) -> bool:
        """
        Acquire the host-level advisory lock without waiting.

        Args:
            None.
        Returns:
            bool: `True` when the current process acquired the lock, otherwise `False`.
        Assumptions:
            Production target is macOS/Linux where `fcntl.flock` releases on process exit.
        Raises:
            RuntimeError: If advisory file locks are unavailable on the current platform.
            OSError: If the lock file cannot be created or written.
        Side Effects:
            Creates the lock file parent directory and writes the current process id on success.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        if self._handle is not None:
            return True
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - unsupported on current CI targets
            raise RuntimeError(
                "backtest artifact publisher host lock requires fcntl advisory locks"
            ) from error

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        """
        Release the advisory lock when it is currently held by this process.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Double-release is treated as a no-op to simplify `finally` blocks.
        Raises:
            RuntimeError: If advisory file locks are unavailable on the current platform.
            OSError: If unlocking or closing the lock file fails.
        Side Effects:
            Unlocks and closes the underlying file descriptor.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        handle = self._handle
        if handle is None:
            return
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - unsupported on current CI targets
            raise RuntimeError(
                "backtest artifact publisher host lock requires fcntl advisory locks"
            ) from error
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


class BacktestArtifactPublisherMetrics:
    """
    Prometheus metric bundle for the dedicated artifact publisher scheduler service.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/runbooks/prod-dashboard-metrics-reference-ru.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml
    """

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        """
        Register scheduler metrics in the provided or default Prometheus registry.

        Args:
            registry: Optional registry override used by tests.
        Returns:
            None.
        Assumptions:
            Metric names are fixed by EPIC R11-02 and must remain stable.
        Raises:
            ValueError: Propagated by Prometheus client on duplicate metric registration.
        Side Effects:
            Registers metric collectors in the selected registry.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
        """
        self.registry = registry or REGISTRY
        self.backtest_artifact_publish_runs_total = Counter(
            "backtest_artifact_publish_runs_total",
            "Backtest artifact publisher scheduler runs grouped by final status",
            labelnames=("status",),
            registry=self.registry,
        )
        self.backtest_artifact_publish_duration_seconds = Histogram(
            "backtest_artifact_publish_duration_seconds",
            "Backtest artifact publisher scheduler run duration in seconds",
            buckets=(0.1, 1.0, 5.0, 15.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0),
            registry=self.registry,
        )
        self.backtest_artifact_publish_symbols_total = Counter(
            "backtest_artifact_publish_symbols_total",
            "Backtest artifact publisher symbols processed grouped by status",
            labelnames=("status",),
            registry=self.registry,
        )
        self.backtest_artifact_publish_blocked_total = Counter(
            "backtest_artifact_publish_blocked_total",
            "Backtest artifact publisher blocked runs grouped by reason",
            labelnames=("reason",),
            registry=self.registry,
        )
        self.backtest_artifact_publish_last_success_unixtime = Gauge(
            "backtest_artifact_publish_last_success_unixtime",
            "Unix timestamp of the last scheduler cycle that produced at least "
            "one successful publish",
            registry=self.registry,
        )
        self.backtest_artifact_tail_rebuild_bars_total = Counter(
            "backtest_artifact_tail_rebuild_bars_total",
            "Bounded tail rebuild bars grouped by artifact stage",
            labelnames=("stage",),
            registry=self.registry,
        )

    def observe_tail_rebuild_bars(self, *, counts: ArtifactTailRebuildBarsV2) -> None:
        """
        Record stage-level bounded tail rewrite counters from one successful publish result.

        Args:
            counts: Shared publish tail counters grouped by artifact stage.
        Returns:
            None.
        Assumptions:
            Stage labels are finite and stable: `prices`, `mappings`, `signals`, `hit_times`.
        Raises:
            None.
        Side Effects:
            Increments Prometheus counters for non-zero stage values.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        for stage, value in counts.as_dict().items():
            if value <= 0:
                continue
            self.backtest_artifact_tail_rebuild_bars_total.labels(stage=stage).inc(value)


@dataclass(frozen=True, slots=True)
class BacktestArtifactPublisherApp:
    """
    Long-running scheduler runtime for daily backtest artifact publish on Mac Studio.

    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/main/main.py
      - infra/macos/launchd/com.roehub.backtest-artifact-publisher.plist
    """

    publish_use_case: PublishBacktestArtifactsV2UseCase
    instrument_reader: EnabledInstrumentReader
    metrics: BacktestArtifactPublisherMetrics
    host_lock: BacktestArtifactPublisherHostLock
    metrics_port: int
    schedule: BacktestArtifactPublisherSchedule = BacktestArtifactPublisherSchedule()
    now_provider: NowProvider = _default_now_provider

    def __post_init__(self) -> None:
        """
        Validate runtime dependencies and scalar scheduler settings.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Wiring must fail fast on startup when one dependency or scalar is invalid.
        Raises:
            ValueError: If one dependency or scalar setting is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/mac-studio-native-backend-operations.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        if self.publish_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPublisherApp.publish_use_case is required")
        if self.instrument_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPublisherApp.instrument_reader is required")
        if self.metrics is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPublisherApp.metrics is required")
        if self.host_lock is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactPublisherApp.host_lock is required")
        if self.metrics_port <= 0:
            raise ValueError("BacktestArtifactPublisherApp.metrics_port must be > 0")

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Run the long-lived scheduler loop until cooperative shutdown is requested.

        Args:
            stop_event: Cooperative shutdown signal shared with the process entrypoint.
        Returns:
            None.
        Assumptions:
            Publish work is synchronous and therefore runs in a worker thread
            between schedule waits.
        Raises:
            None. Unexpected cycle errors are logged and the scheduler loop continues.
        Side Effects:
            Starts a Prometheus `/metrics` endpoint and performs scheduled publish work.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/runbooks/mac-studio-native-backend-operations.md
        Related:
          - apps/scheduler/backtest_artifact_publisher/main/main.py
        """
        start_http_server(self.metrics_port, registry=self.metrics.registry)
        log.info(
            "event=metrics_started component=backtest-artifact-publisher "
            "metrics_port=%s timezone=%s schedule=03:05",
            self.metrics_port,
            self.schedule.timezone_name,
        )
        if not self.schedule.enabled:
            log.info(
                "event=schedule_disabled component=backtest-artifact-publisher "
                "reason=capacity_gate"
            )
            await stop_event.wait()
            return
        last_run_local_date: date | None = None

        while not stop_event.is_set():
            now_utc = _validated_utc_datetime(self.now_provider())
            next_run_utc = self.schedule.next_run_after(
                now_utc=now_utc,
                last_run_local_date=last_run_local_date,
            )
            wait_seconds = max((next_run_utc - now_utc).total_seconds(), 0.0)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
                break
            except TimeoutError:
                pass

            if stop_event.is_set():
                break

            try:
                await asyncio.to_thread(
                    self._run_publish_cycle,
                    stop_requested=stop_event.is_set,
                    scheduled_for_utc=next_run_utc,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "event=run_cycle_crashed component=backtest-artifact-publisher "
                    "scheduled_for_utc=%s",
                    next_run_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            finally:
                last_run_local_date = self.schedule.local_run_date(run_at_utc=next_run_utc)

    def _run_publish_cycle(
        self,
        *,
        stop_requested: StopRequested,
        scheduled_for_utc: datetime,
    ) -> None:
        """
        Execute one due daily publish cycle under host-level locking and metrics accounting.

        Args:
            stop_requested: Callback returning whether cooperative shutdown was requested.
            scheduled_for_utc: UTC timestamp of the due scheduler fire time.
        Returns:
            None.
        Assumptions:
            One cycle processes the full enabled+tradable universe in deterministic order.
        Raises:
            None. All symbol-level and run-level failures are converted into logs and metrics.
        Side Effects:
            Acquires the host lock, reads `market_data.ref_instruments`, and publishes artifacts.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        started = perf_counter()
        scheduled_utc = _validated_utc_datetime(scheduled_for_utc)
        if not self.host_lock.try_acquire():
            self.metrics.backtest_artifact_publish_blocked_total.labels(reason="lock_held").inc()
            self.metrics.backtest_artifact_publish_runs_total.labels(status="lock_held").inc()
            self.metrics.backtest_artifact_publish_duration_seconds.observe(
                max(perf_counter() - started, 0.0)
            )
            log.warning(
                "event=run_blocked component=backtest-artifact-publisher "
                "reason=lock_held scheduled_for_utc=%s",
                scheduled_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return

        run_status: PublisherRunStatus = "succeeded"
        successful_symbols = 0
        try:
            coordinates = self._load_publish_universe()
            self.metrics.backtest_artifact_publish_symbols_total.labels(status="planned").inc(
                len(coordinates)
            )
            log.info(
                "event=run_started component=backtest-artifact-publisher "
                "scheduled_for_utc=%s timezone=%s universe_total=%s",
                scheduled_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                self.schedule.timezone_name,
                len(coordinates),
            )

            for coordinate in coordinates:
                if stop_requested():
                    log.info(
                        "event=run_stop_requested component=backtest-artifact-publisher "
                        "scheduled_for_utc=%s last_symbol=%s/%s/%s",
                        scheduled_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        coordinate.exchange,
                        coordinate.market_type,
                        coordinate.symbol,
                    )
                    break
                symbol_status = self._publish_one_coordinate(coordinate=coordinate)
                run_status = _promote_run_status(current=run_status, candidate=symbol_status)
                if symbol_status == "succeeded":
                    successful_symbols += 1

            if successful_symbols > 0 or len(coordinates) == 0:
                self.metrics.backtest_artifact_publish_last_success_unixtime.set(
                    _validated_utc_datetime(self.now_provider()).timestamp()
                )
            self.metrics.backtest_artifact_publish_runs_total.labels(status=run_status).inc()
            log.info(
                "event=run_finished component=backtest-artifact-publisher "
                "scheduled_for_utc=%s status=%s successful_symbols=%s",
                scheduled_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                run_status,
                successful_symbols,
            )
        finally:
            self.host_lock.release()
            self.metrics.backtest_artifact_publish_duration_seconds.observe(
                max(perf_counter() - started, 0.0)
            )

    def _load_publish_universe(self) -> tuple[ArtifactCoordinatesV2, ...]:
        """
        Load and deterministically order the effective collector publish universe.

        Args:
            None.
        Returns:
            tuple[ArtifactCoordinatesV2, ...]: Ordered artifact coordinates for the daily run.
        Assumptions:
            Universe source-of-truth is the global effective collector set: explicit
            organization choices plus active strategy pins. It must never be inferred
            from every legacy reference row.
        Raises:
            ValueError: If one market id cannot be bridged into artifact coordinates.
            Exception: Propagates storage reader errors from the enabled instrument reader.
        Side Effects:
            Reads the global effective collector set from PostgreSQL.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/market_data/adapters/outbound/persistence/postgres/
            instrument_selection_repository.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        instruments = tuple(self.instrument_reader.list_enabled_tradable())
        ordered_instruments = tuple(
            sorted(
                instruments,
                key=lambda instrument: (instrument.market_id.value, str(instrument.symbol)),
            )
        )
        return tuple(
            artifact_coordinates_from_market_id_v2(
                market_id=instrument.market_id.value,
                symbol=str(instrument.symbol),
            )
            for instrument in ordered_instruments
        )

    def _publish_one_coordinate(
        self,
        *,
        coordinate: ArtifactCoordinatesV2,
    ) -> PublisherRunStatus:
        """
        Execute one shared publish use-case call and convert its outcome into stable metrics labels.

        Args:
            coordinate: Explicit artifact coordinate from the deterministic enabled universe.
        Returns:
            PublisherRunStatus: Stable final status label for the symbol publish attempt.
        Assumptions:
            Scheduler must reuse the shared R11-01 publish use-case instead of
            reimplementing publish logic.
        Raises:
            None. Publish failures are converted into status labels, logs, and metrics.
        Side Effects:
            Updates per-symbol counters/tail metrics and emits one final stage-level rebuild log.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        try:
            result = self.publish_use_case.run(
                PublishBacktestArtifactsV2Request(coordinates=coordinate)
            )
        except ArtifactSlotPublishErrorV2 as error:
            status = _publish_error_status(error.code)
            self.metrics.backtest_artifact_publish_symbols_total.labels(status=status).inc()
            if status in {"inactive_slot_pinned", "validation_failed"}:
                self.metrics.backtest_artifact_publish_blocked_total.labels(reason=status).inc()
            log.warning(
                "event=symbol_publish_blocked component=backtest-artifact-publisher "
                "exchange=%s market_type=%s symbol=%s reason=%s",
                coordinate.exchange,
                coordinate.market_type,
                coordinate.symbol,
                status,
            )
            return status
        except Exception:  # noqa: BLE001
            self.metrics.backtest_artifact_publish_symbols_total.labels(
                status="unexpected_error"
            ).inc()
            log.exception(
                "event=symbol_publish_failed component=backtest-artifact-publisher "
                "exchange=%s market_type=%s symbol=%s reason=unexpected_error",
                coordinate.exchange,
                coordinate.market_type,
                coordinate.symbol,
            )
            return "unexpected_error"

        self.metrics.backtest_artifact_publish_symbols_total.labels(status="succeeded").inc()
        self.metrics.observe_tail_rebuild_bars(counts=result.tail_rebuild_bars)
        log.info(
            "event=symbol_publish_succeeded component=backtest-artifact-publisher "
            "exchange=%s market_type=%s symbol=%s publish_mode=%s "
            "active_slot=%s tail_rebuild_bars=%s stage_rebuild_stats=%s",
            coordinate.exchange,
            coordinate.market_type,
            coordinate.symbol,
            result.publish_mode,
            result.published_active_slot,
            result.tail_rebuild_bars.as_dict(),
            result.stage_rebuild_stats.as_dict(),
        )
        return "succeeded"


def _publish_error_status(code: str) -> PublisherRunStatus:
    """
    Collapse shared publish error codes into the finite scheduler status label set.

    Args:
        code: Shared publish error code from `ArtifactSlotPublishErrorV2`.
    Returns:
        PublisherRunStatus: Stable scheduler status label.
    Assumptions:
        Scheduler metrics must stay finite even if shared publish code emits new messages.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if code == "inactive_slot_pinned":
        return "inactive_slot_pinned"
    if code in {"slot_validation_failed", "artifact_slot_validation_failed"}:
        return "validation_failed"
    return "unexpected_error"


def _promote_run_status(
    *,
    current: PublisherRunStatus,
    candidate: PublisherRunStatus,
) -> PublisherRunStatus:
    """
    Promote one scheduler run status to the most severe symbol outcome seen so far.

    Args:
        current: Current aggregate run status.
        candidate: Candidate status from one symbol publish attempt.
    Returns:
        PublisherRunStatus: Higher-severity status according to the fixed scheduler ordering.
    Assumptions:
        `unexpected_error` dominates `validation_failed`, which dominates `inactive_slot_pinned`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    severity = {
        "succeeded": 0,
        "inactive_slot_pinned": 1,
        "validation_failed": 2,
        "unexpected_error": 3,
        "lock_held": 4,
    }
    return current if severity[current] >= severity[candidate] else candidate


def build_backtest_artifact_publisher_app(
    *,
    config_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
    lock_path: str | None = None,
) -> BacktestArtifactPublisherApp:
    """
    Build the fully wired backtest artifact publisher scheduler application.

    Args:
        config_path: Path to `backtest_artifacts.yaml`.
        environ: Runtime environment mapping with ClickHouse and Postgres credentials.
        metrics_port: Prometheus metrics HTTP port.
        lock_path: Optional override for the host-level lock file path.
    Returns:
        BacktestArtifactPublisherApp: Ready-to-run scheduler app instance.
    Assumptions:
        Startup must fail fast when config, ClickHouse, or Postgres dependencies are invalid.
    Raises:
        ValueError: If config/env/DSN contracts are invalid.
        FileNotFoundError: If the config path does not exist.
        RuntimeError: If ClickHouse client dependency is not installed.
    Side Effects:
        Reads config files and initializes storage/compute adapters used by the scheduler.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/runbooks/mac-studio-native-backend-operations.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
      - apps/scheduler/backtest_artifact_publisher/main/main.py
    """
    artifact_runtime_config = load_backtest_artifacts_runtime_config(Path(config_path))
    path_builder = BacktestArtifactPathBuilderV2(root=artifact_runtime_config.artifact_root_path())
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=path_builder)
    pointer_writer = AtomicArtifactCurrentPointerWriterV2(path_resolver=path_builder)
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )
    strategy_postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not strategy_postgres_dsn:
        raise ValueError("STRATEGY_PG_DSN is required for backtest-artifact-publisher scheduler")

    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=environ,
        artifact_config_path=config_path,
    )
    indicator_registry = build_indicators_registry(
        environ=environ,
        artifact_config_path=config_path,
    )
    indicator_compute = build_artifact_precompute_indicators_compute(
        environ=environ,
        artifact_config_path=config_path,
    )
    artifact_config_hash = build_backtest_artifacts_runtime_config_hash(
        config=artifact_runtime_config
    )
    precompute_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=artifact_runtime_config.to_precompute_runtime_settings(
            config_sha256=artifact_config_hash
        ),
        artifact_loader=artifact_loader,
        canonical_candle_reader=ClickHouseCanonicalCandleReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
        indicator_compute=indicator_compute,
        indicator_grid_builder=GridBuilder(registry=indicator_registry),
    )
    slot_publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=artifact_loader,
        current_pointer_writer=pointer_writer,
        job_repository=PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=strategy_postgres_dsn)
        ),
    )
    publish_use_case = PublishBacktestArtifactsV2UseCase(
        canonical_candle_index_reader=ClickHouseCanonicalCandleIndexReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        ),
        precompute_runner=precompute_runner,
        slot_publisher=slot_publisher,
        validation_spec=artifact_runtime_config.to_validation_spec(),
    )
    resolved_lock_path = (
        Path(lock_path)
        if lock_path is not None
        else artifact_runtime_config.artifact_root_path() / ".backtest_artifact_publisher.lock"
    )
    return BacktestArtifactPublisherApp(
        publish_use_case=publish_use_case,
        instrument_reader=PostgresInstrumentSelectionRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=strategy_postgres_dsn),
        ),
        metrics=BacktestArtifactPublisherMetrics(),
        host_lock=FileBacktestArtifactPublisherHostLock(path=resolved_lock_path),
        metrics_port=metrics_port,
        schedule=BacktestArtifactPublisherSchedule(
            enabled=artifact_runtime_config.publish_schedule.enabled,
        ),
    )


__all__ = [
    "BacktestArtifactPublisherApp",
    "BacktestArtifactPublisherMetrics",
    "BacktestArtifactPublisherSchedule",
    "build_backtest_artifact_publisher_app",
]
