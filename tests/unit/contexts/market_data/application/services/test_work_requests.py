from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from trading.contexts.market_data.application.services.work_requests import (
    MarketDataWorkRequestRunner,
    WorkRequest,
)
from trading.contexts.market_data.application.use_cases.rest_fill_range_1m import (
    RestFillRange1mUseCase,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def test_work_request_bounds_reject_unsupported_or_unclosed_ranges():
    valid = WorkRequest("candle_ingestion", 2, ("BTCUSDT",), NOW - timedelta(hours=1), NOW)
    assert valid.validate(now=NOW) == 60
    for command in (
        WorkRequest("candle_ingestion", 2, ("BTCUSDT",), datetime(2016, 1, 1, tzinfo=UTC), NOW),
        WorkRequest("candle_ingestion", 2, ("BTCUSDT",), NOW, NOW + timedelta(minutes=1)),
        WorkRequest("candle_ingestion", 2, ("BTCUSDT", "BTCUSDT"), NOW - timedelta(hours=1), NOW),
        WorkRequest("candle_ingestion", 2, ("BTCUSDT",), NOW - timedelta(hours=1), NOW, "5m"),
        WorkRequest("catalog_refresh", 2, ("BTCUSDT",)),
    ):
        with pytest.raises(ValueError):
            command.validate(now=NOW)


@dataclass
class Fill:
    run: object
    writer: object = None


class Store:
    def __init__(self):
        self.job: dict[str, Any] = dict(
            job_id=uuid4(),
            worker_token=uuid4(),
            kind="candle_ingestion",
            market_id=2,
            symbols=["BTCUSDT"],
            start_at=NOW - timedelta(hours=3),
            end_at=NOW,
        )
        self.states = []
        self.outcome: dict[str, Any] | None = None
        self.cancel = False
        self.pause = False

    def execution(self):
        return nullcontext(lambda: None)

    def recover(self, **kwargs):
        pass

    def claim(self, **kwargs):
        return self.job

    def checkpoint(self, **kwargs):
        self.states.append(kwargs)
        return "cancel_requested" if self.cancel else "pause_requested" if self.pause else "running"

    def release(self, **kwargs):
        self.released = True
        if self.cancel:
            self.outcome = {"state": "cancelled"}
        last = self.states[-1]
        self.job.update(
            completed_units=last["units"],
            rows_read=last["rows_read"],
            rows_written=last["rows_written"],
        )

    def defer(self, **kwargs):
        self.outcome = {**kwargs, "state": "retry_wait"}

    def finish(self, **kwargs):
        self.outcome = kwargs


def test_runner_cancellation_acknowledges_persisted_window_not_future_work():
    store = Store()
    windows = []

    def fill(task):
        windows.append(task.time_range)
        store.cancel = True
        return SimpleNamespace(rows_read=60, rows_written=55)

    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None, now=lambda: NOW
    )
    assert runner.run_once()
    assert len(windows) == 1
    assert store.outcome is not None
    assert store.outcome["state"] == "cancelled"
    assert store.states[-1]["units"] == 60
    assert store.states[-1]["rows_written"] == 55


def test_runner_progress_counts_work_but_does_not_invent_rows_or_retry_failure():
    store = Store()
    calls = []

    def fill(task):
        calls.append(task)
        if len(calls) == 2:
            raise RuntimeError("private-provider-payload-must-not-be-saved")
        return SimpleNamespace(rows_read=0, rows_written=0)

    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None, now=lambda: NOW
    )
    runner.run_once()
    assert len(calls) == 2
    assert store.states[-1]["units"] == 60
    assert store.states[-1]["rows_read"] == 0
    assert store.outcome is not None
    assert store.outcome["state"] == "failed"
    assert store.outcome["error_code"] == "source_or_storage_unavailable"
    assert "private-provider" not in str(store.outcome)


def test_full_history_is_accepted_and_bounded_batches_resume_across_symbols():
    start = datetime(2017, 1, 1, tzinfo=UTC)
    request = WorkRequest("candle_ingestion", 2, ("BTCUSDT", "ETHUSDT"), start, NOW)
    assert request.validate(now=NOW) == int((NOW - start).total_seconds() // 60) * 2
    store = Store()
    store.job.update(symbols=["BTCUSDT", "ETHUSDT"], start_at=NOW - timedelta(hours=3))
    windows = []

    def fill(task):
        windows.append((str(task.instrument_id.symbol), task.time_range))
        return SimpleNamespace(rows_read=60, rows_written=60)

    runner = MarketDataWorkRequestRunner(
        store,
        cast(RestFillRange1mUseCase, Fill(fill)),
        lambda *_: None,
        now=lambda: NOW,
        max_windows=2,
    )
    runner.run_once()
    assert store.released and store.outcome is None
    assert store.job["completed_units"] == 120
    runner.run_once()
    assert store.job["completed_units"] == 240
    runner.run_once()
    assert store.outcome is not None
    assert store.outcome["state"] == "succeeded"
    assert [name for name, _ in windows] == ["BTCUSDT"] * 3 + ["ETHUSDT"] * 3
    assert len({(name, span.start.value) for name, span in windows}) == 6
    assert store.states[-1]["units"] == 360
    assert store.states[-1]["rows_written"] == 360


def test_pause_checkpoints_current_window_and_resume_uses_same_cursor():
    store = Store()
    windows = []

    def fill(task):
        windows.append(task.time_range.start.value)
        store.pause = True
        return SimpleNamespace(rows_read=60, rows_written=55)

    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None, now=lambda: NOW
    )
    runner.run_once()
    assert store.outcome is None and store.job["completed_units"] == 60
    store.pause = False
    runner.run_once()
    assert windows == [NOW - timedelta(hours=3), NOW - timedelta(hours=2)]
    assert store.job["rows_written"] == 110


def test_temporary_error_keeps_checkpoint_and_provider_cooldown():
    from trading.contexts.market_data.application.services.source_retry import TemporarySourceError

    store = Store()
    calls = []

    def fill(task):
        calls.append(task)
        if len(calls) == 2:
            raise TemporarySourceError(rate_limited=True, retry_after_s=90)
        return SimpleNamespace(rows_read=60, rows_written=57)

    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None, now=lambda: NOW
    )
    runner.run_once()
    assert store.states[-1]["units"] == 60
    assert store.outcome is not None
    assert store.outcome["state"] == "retry_wait"
    assert store.outcome["error_code"] == "source_rate_limited"
    assert store.outcome["retry_after_s"] == 90


def test_transient_checkpoint_failure_retries_without_turning_into_terminal_failure(caplog):
    store = Store()
    original = store.checkpoint
    def checkpoint(**kwargs):
        if kwargs["units"] == 60:
            raise ConnectionError("private-connection-details")
        return original(**kwargs)
    store.checkpoint = checkpoint
    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(lambda _: SimpleNamespace(
            rows_read=60, rows_written=60))), lambda *_: None, now=lambda: NOW,
        retryable_storage_error=lambda e: isinstance(e, ConnectionError),
    )
    runner.run_once()
    assert store.outcome is not None
    assert store.outcome["state"] == "retry_wait"
    assert store.outcome["error_code"] == "storage_unavailable"
    assert "phase=checkpoint error_type=ConnectionError" in caplog.text
    assert "private-connection-details" not in caplog.text


def test_runtime_storage_classifier_excludes_permanent_errors():
    import psycopg
    from clickhouse_connect.driver.exceptions import OperationalError

    from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
        _retryable_work_storage_error,
    )
    for error in [psycopg.OperationalError(), OperationalError(), TimeoutError()]:
        assert _retryable_work_storage_error(error)
    for error in [psycopg.errors.InvalidPassword(), psycopg.errors.SyntaxError(), ValueError()]:
        assert not _retryable_work_storage_error(error)


def test_clickhouse_memory_error_recovers_but_schema_and_auth_errors_do_not():
    from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

    from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
        _retryable_work_storage_error,
    )
    for error in [DatabaseError('Code: 241. memory limit'),
                  DatabaseError('HTTPDriver for private-url received ClickHouse error code 241')]:
        store = Store()
        store.job.update(completed_units=60, rows_read=60, rows_written=60)
        seen = []

        def fill(task):
            seen.append(task.time_range.start.value)
            raise error

        runner = MarketDataWorkRequestRunner(
            store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None,
            now=lambda: NOW, retryable_storage_error=_retryable_work_storage_error,
        )
        runner.run_once()
        assert seen == [store.job['start_at'] + timedelta(minutes=60)]
        assert store.outcome is not None
        assert store.outcome['error_code'] == 'storage_memory_pressure'
        assert store.outcome['job']['error_phase'] == 'fill_window'
        assert store.states[-1]['units'] == 60
    for code in [60, 62, 81, 516, 497]:
        for cls in [DatabaseError, OperationalError]:
            assert not _retryable_work_storage_error(cls(f'Code: {code}. permanent'))
    assert not _retryable_work_storage_error(DatabaseError('unknown failure'))


def test_bounded_larger_window_preserves_cursor_and_yields_after_confirmed_insert():
    store = Store()
    store.job.update(start_at=NOW-timedelta(days=3), completed_units=75,
                     rows_read=75, rows_written=75)
    windows = []

    def fill(task):
        windows.append(task.time_range)
        return SimpleNamespace(rows_read=1440, rows_written=1440)

    runner = MarketDataWorkRequestRunner(
        store, cast(RestFillRange1mUseCase, Fill(fill)), lambda *_: None, now=lambda: NOW,
        window_minutes=1440, max_windows=1,
    )
    runner.run_once()
    assert len(windows) == 1
    assert windows[0].start.value == NOW-timedelta(days=3)+timedelta(minutes=75)
    assert store.job['completed_units'] == 1515
    assert store.job['rows_written'] == 1515
