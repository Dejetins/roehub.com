from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.dto import encode_backtest_runs_cursor
from apps.api.routes import build_backtest_runs_router
from trading.contexts.backtest.application.dto import BacktestMetricRowV1, BacktestReportV1
from trading.contexts.backtest.application.ports import BacktestJobListPage
from trading.contexts.backtest.application.services import (
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestRunProgressSnapshotBuilder,
    BacktestRunTopReadResult,
    backtest_run_forbidden,
    backtest_run_not_found,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobExecutionMode,
    BacktestJobTopVariant,
    TradeV1,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.shared_kernel.primitives import PaidLevel, UserId

_BENCHMARK_CORPUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "perf_smoke"
    / "contexts"
    / "backtest"
    / "fixtures"
    / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)


class _HeaderCurrentUserDependency:
    """
    Request dependency resolving authenticated principal from `X-User-Id` header.
    """

    def __call__(self, request: Request):
        """
        Resolve principal or raise deterministic HTTP 401 payload.

        Args:
            request: HTTP request object.
        Returns:
            object: CurrentUserPrincipal-compatible object.
        Assumptions:
            Header contains UUID string when provided.
        Raises:
            HTTPException: If authentication header is missing.
        Side Effects:
            None.
        """
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )

        from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


def _benchmark_corpus():
    """
    Load the committed benchmark corpus used by public runs route ETA fallback tests.

    Args:
        None.
    Returns:
        object: Typed committed benchmark corpus fixture.
    Assumptions:
        Route tests may read the committed corpus fixture directly because the production rule is
        only that the request path itself must avoid file IO.
    Raises:
        OSError: If the committed benchmark fixture is missing.
        ValueError: If the fixture payload violates the typed corpus contract.
    Side Effects:
        Reads one repository fixture file.
    """
    return load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_PATH
    )


@dataclass
class _StatusUseCaseFake:
    """
    Deterministic status use-case fake returning preconfigured run or error.
    """

    run: BacktestJob | None = None
    error: Exception | None = None

    def execute(self, *, run_id: UUID, current_user):
        """
        Return configured run snapshot or raise configured exception.

        Args:
            run_id: Requested run identifier.
            current_user: Authenticated user payload.
        Returns:
            BacktestJob: Configured run snapshot.
        Assumptions:
            `run_id/current_user` are irrelevant for static fake responses.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = run_id, current_user
        if self.error is not None:
            raise self.error
        if self.run is None:  # pragma: no cover - guarded by fixtures
            raise ValueError("status fake requires run")
        return self.run


@dataclass
class _TopUseCaseFake:
    """
    Deterministic top use-case fake returning preconfigured result or error.
    """

    result: BacktestRunTopReadResult | None = None
    error: Exception | None = None

    def execute(self, *, run_id: UUID, current_user, limit: int | None):
        """
        Return configured top result payload or raise configured exception.

        Args:
            run_id: Requested run identifier.
            current_user: Authenticated user payload.
            limit: Optional top rows limit.
        Returns:
            BacktestRunTopReadResult: Configured top payload.
        Assumptions:
            Fake does not enforce limit semantics.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = run_id, current_user, limit
        if self.error is not None:
            raise self.error
        if self.result is None:  # pragma: no cover - guarded by fixtures
            raise ValueError("top fake requires result")
        return self.result


@dataclass
class _ListUseCaseFake:
    """
    Deterministic list use-case fake returning preconfigured page payload or error.
    """

    page: BacktestJobListPage
    error: Exception | None = None
    last_cursor: BacktestJobListCursor | None = None
    last_state: str | None = None

    def execute(self, *, current_user, state: str | None, limit: int, cursor):
        """
        Return configured list page and record decoded cursor/state arguments.

        Args:
            current_user: Authenticated user payload.
            state: Optional state filter.
            limit: Page size.
            cursor: Optional decoded keyset cursor.
        Returns:
            BacktestJobListPage: Configured page fixture.
        Assumptions:
            Fake does not validate state/limit values.
        Raises:
            Exception: Configured exception.
        Side Effects:
            Stores last cursor and state payloads for assertions.
        """
        _ = current_user, limit
        self.last_cursor = cursor
        self.last_state = state
        if self.error is not None:
            raise self.error
        return self.page


@dataclass
class _CancelUseCaseFake:
    """
    Deterministic cancel use-case fake returning preconfigured run or error.
    """

    run: BacktestJob | None = None
    error: Exception | None = None

    def execute(self, *, run_id: UUID, current_user):
        """
        Return configured cancel status payload or raise configured exception.

        Args:
            run_id: Requested run identifier.
            current_user: Authenticated user payload.
        Returns:
            BacktestJob: Configured updated run snapshot.
        Assumptions:
            Fake ignores input ids and always returns fixture run.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = run_id, current_user
        if self.error is not None:
            raise self.error
        if self.run is None:  # pragma: no cover - guarded by fixtures
            raise ValueError("cancel fake requires run")
        return self.run


@dataclass
class _VariantReportUseCaseFake:
    """
    Deterministic run-scoped variant-report fake returning preconfigured report or error.
    """

    report: BacktestReportV1 | None = None
    error: Exception | None = None
    last_run_id: UUID | None = None
    last_include_trades: bool | None = None

    def execute(
        self,
        *,
        run_id: UUID,
        current_user,
        variant_payload,
        include_trades: bool,
        run_control=None,
    ) -> BacktestReportV1:
        """
        Return configured report payload or raise configured exception.

        Args:
            run_id: Requested persisted run identifier.
            current_user: Authenticated user payload.
            variant_payload: Explicit selected variant payload.
            include_trades: Include-trades flag.
            run_control: Optional cooperative cancellation handle.
        Returns:
            BacktestReportV1: Configured report payload.
        Assumptions:
            Route tests verify wiring and error mapping, not report compute logic.
        Raises:
            Exception: Configured fake exception.
        Side Effects:
            Stores last `run_id` and `include_trades` for assertions.
        """
        _ = current_user, variant_payload, run_control
        self.last_run_id = run_id
        self.last_include_trades = include_trades
        if self.error is not None:
            raise self.error
        if self.report is None:  # pragma: no cover - guarded by fixtures
            raise ValueError("variant_report fake requires report")
        return self.report


def _build_client(
    *,
    status_use_case: _StatusUseCaseFake | None = None,
    top_use_case: _TopUseCaseFake | None = None,
    list_use_case: _ListUseCaseFake | None = None,
    cancel_use_case: _CancelUseCaseFake | None = None,
    variant_report_use_case: _VariantReportUseCaseFake | None = None,
    run_progress_builder: BacktestRunProgressSnapshotBuilder | None = None,
) -> tuple[TestClient, _ListUseCaseFake, _VariantReportUseCaseFake]:
    """
    Build minimal FastAPI TestClient with public runs router and shared error handlers.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/apps/api/test_backtest_runs_routes.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    Args:
        status_use_case: Optional status use-case fake.
        top_use_case: Optional top use-case fake.
        list_use_case: Optional list use-case fake.
        cancel_use_case: Optional cancel use-case fake.
        run_progress_builder: Optional deterministic progress/ETA builder for payload tests.
    Returns:
        tuple[TestClient, _ListUseCaseFake, _VariantReportUseCaseFake]:
            Configured client and resolved fake dependencies.
    Assumptions:
        Shared API error handlers provide deterministic Roehub/422 payloads.
    Raises:
        ValueError: If router dependencies are invalid.
    Side Effects:
        None.
    """
    base_run = _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000930"))
    resolved_list_use_case = list_use_case or _ListUseCaseFake(
        page=BacktestJobListPage(items=(base_run,), next_cursor=None)
    )
    resolved_variant_report_use_case = variant_report_use_case or _VariantReportUseCaseFake(
        report=_variant_report()
    )
    resolved_progress_builder = run_progress_builder or BacktestRunProgressSnapshotBuilder(
        now_provider=lambda: datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc)
    )

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtest_runs_router(
            get_status_use_case=status_use_case or _StatusUseCaseFake(run=base_run),  # type: ignore[arg-type]
            get_top_use_case=top_use_case
            or _TopUseCaseFake(result=_top_result(run=base_run)),  # type: ignore[arg-type]
            list_use_case=resolved_list_use_case,  # type: ignore[arg-type]
            cancel_use_case=cancel_use_case or _CancelUseCaseFake(run=base_run),  # type: ignore[arg-type]
            variant_report_use_case=resolved_variant_report_use_case,  # type: ignore[arg-type]
            current_user_dependency=_HeaderCurrentUserDependency(),
            sync_deadline_seconds=55.0,
            run_progress_builder=resolved_progress_builder,
        )
    )
    return TestClient(app), resolved_list_use_case, resolved_variant_report_use_case


def test_get_backtest_runs_returns_history_page_with_public_run_metadata() -> None:
    """
    Verify `GET /backtests/runs` returns deterministic history payload with `run_id` vocabulary.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public history list hides internal hashes and exposes persisted metadata fields.
    Raises:
        AssertionError: If status code, cursor decoding, or payload shape drifts.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 3, 29, 11, 30, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000931"),
    )
    list_fake = _ListUseCaseFake(
        page=BacktestJobListPage(
            items=(
                _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000932")),
            ),
            next_cursor=cursor,
        )
    )
    client, resolved_list_fake, _ = _build_client(list_use_case=list_fake)

    response = client.get(
        "/backtests/runs",
        params={
            "state": "RUNNING",
            "cursor": encode_backtest_runs_cursor(cursor=cursor),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["run_id"] == "00000000-0000-0000-0000-000000000932"
    assert body["items"][0]["execution_mode"] == "sync_inline"
    assert body["items"][0]["execution_profile_mode"] == "exact_small"
    assert body["items"][0]["market_id"] == 1
    assert body["items"][0]["symbol"] == "BTCUSDT"
    assert body["items"][0]["requested_top_n"] == 25
    assert body["items"][0]["progress_percent"] == 0
    assert body["items"][0]["eta_seconds"] is None
    assert "ranking_secondary_metric" not in body["items"][0]
    assert body["next_cursor"] == encode_backtest_runs_cursor(cursor=cursor)
    assert resolved_list_fake.last_state == "running"
    assert resolved_list_fake.last_cursor == cursor


def test_get_backtest_runs_prefers_additive_execution_profile_metadata() -> None:
    """
    Verify history route projects `execution_profile_mode` from additive run metadata first.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        New persisted rows no longer need `request_json.execution_profile_mode` for list/history
        rendering once additive metadata fields are available.
    Raises:
        AssertionError: If the route falls back to catalog defaults despite additive metadata.
    Side Effects:
        None.
    """
    list_fake = _ListUseCaseFake(
        page=BacktestJobListPage(
            items=(
                replace(
                    _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000934")),
                    execution_profile_mode_hint="exact_parallel",
                    effective_execution_profile_mode="exact_parallel",
                ),
            ),
            next_cursor=None,
        )
    )
    client, _, _ = _build_client(list_use_case=list_fake)

    response = client.get(
        "/backtests/runs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["run_id"] == "00000000-0000-0000-0000-000000000934"
    assert body["items"][0]["execution_profile_mode"] == "exact_parallel"
    assert body["items"][0]["progress_percent"] == 0
    assert body["items"][0]["eta_seconds"] is None


def test_get_backtest_run_status_returns_failed_payload_with_public_fields() -> None:
    """
    Verify `GET /backtests/runs/{run_id}` returns failure payload and persisted metadata.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs payload exposes owner-visible state metadata but not internal hashes.
    Raises:
        AssertionError: If payload shape drifts from the R7-03 contract.
    Side Effects:
        None.
    """
    failed_run = _failed_run(run_id=UUID("00000000-0000-0000-0000-000000000933"))
    client, _, _ = _build_client(status_use_case=_StatusUseCaseFake(run=failed_run))

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000933",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "00000000-0000-0000-0000-000000000933"
    assert body["state"] == "failed"
    assert body["execution_mode"] == "sync_inline"
    assert body["execution_profile_mode"] == "exact_small"
    assert body["progress_percent"] == 0
    assert body["eta_seconds"] is None
    assert body["artifact_slot"] == "slot_b"
    assert body["artifact_slot_generation"] == 11
    assert body["artifact_asof_date"] == "2026-03-29"
    assert body["last_error_json"] == {
        "code": "unexpected_error",
        "message": "Execution failed",
        "details": {"stage": "stage_b"},
    }
    assert "request_hash" not in body
    assert "ranking_secondary_metric" not in body


def test_get_backtest_run_top_returns_summary_only_rows_with_persisted_metrics() -> None:
    """
    Verify `GET /backtests/runs/{run_id}/top` exposes deterministic summary-only row fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public `/top` rows expose `summary_metrics_json/best_tp_pct/best_sl_pct` and no details.
    Raises:
        AssertionError: If payload shape drifts from the persisted summary contract.
    Side Effects:
        None.
    """
    run = _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000934"))
    client, _, _ = _build_client(top_use_case=_TopUseCaseFake(result=_top_result(run=run)))

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000934/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "00000000-0000-0000-0000-000000000934"
    assert body["execution_mode"] == "sync_inline"
    assert body["items"] == [
        {
            "rank": 1,
            "variant_key": "a" * 64,
            "indicator_variant_key": "b" * 64,
            "variant_index": 0,
            "total_return_pct": 10.0,
            "payload": {"schema_version": 1},
            "summary_metrics_json": {
                "total_return_pct": 10.0,
                "profit_factor": 1.5,
            },
            "best_tp_pct": 4.0,
            "best_sl_pct": 2.0,
        }
    ]


def test_get_backtest_run_status_returns_additive_progress_eta_and_profile_fields() -> None:
    """
    Verify running status payload exposes additive weighted progress, ETA, and profile fields.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/apps/api/test_backtest_runs_routes.py
      - apps/api/routes/backtest_runs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs status remains backward-compatible while adding UI-facing progress metadata.
    Raises:
        AssertionError: If additive fields or their deterministic values drift.
    Side Effects:
        None.
    """
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000938"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 30, tzinfo=timezone.utc),
        request_json={
            "time_range": {
                "start": "2026-03-28T00:00:00+00:00",
                "end": "2026-03-28T01:00:00+00:00",
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1h",
            },
            "top_k": 25,
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-29",
        ),
        execution_mode="sync_inline",
        execution_profile_mode_hint="exact_parallel",
        effective_execution_profile_mode="exact_parallel",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    running = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 5, tzinfo=timezone.utc),
    ).update_progress(
        changed_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=20,
    )
    client, _, _ = _build_client(status_use_case=_StatusUseCaseFake(run=running))

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000938",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "running"
    assert body["stage"] == "stage_b"
    assert body["processed_units"] == 10
    assert body["total_units"] == 20
    assert body["execution_profile_mode"] == "exact_parallel"
    assert body["progress_percent"] == 70
    assert body["eta_seconds"] == 26
    assert "ranking_secondary_metric" not in body


def test_get_backtest_run_status_uses_benchmark_eta_fallback_when_timeline_signal_is_too_early(
) -> None:
    """
    Verify status route returns benchmark-backed ETA when current throughput is not defensible.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The run has started and exposed Stage B counters, but elapsed time is still too small for
        the timeline-only ETA path to publish a throughput-based estimate.
    Raises:
        AssertionError: If route wiring drops the benchmark-backed ETA fallback.
    Side Effects:
        None.
    """
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000939"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 59, 55, tzinfo=timezone.utc),
        request_json={
            "top_k": 25,
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        execution_mode="sync_inline",
        execution_profile_mode_hint="exact_parallel",
        effective_execution_profile_mode="exact_parallel",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    running = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 5, tzinfo=timezone.utc),
    ).update_progress(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=0,
        total_units=48,
    )
    client, _, _ = _build_client(
        status_use_case=_StatusUseCaseFake(run=running),
        run_progress_builder=BacktestRunProgressSnapshotBuilder(
            benchmark_corpus=_benchmark_corpus(),
            now_provider=lambda: datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        ),
    )

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000939",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["execution_profile_mode"] == "exact_parallel"
    assert "ranking_secondary_metric" not in body
    assert body["progress_percent"] == 45
    assert body["eta_seconds"] == 34


def test_get_backtest_run_top_returns_persisted_rows_in_deterministic_order() -> None:
    """
    Verify `GET /backtests/runs/{run_id}/top` returns persisted rows in canonical order.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/api/routes/backtest_runs.py
      - apps/api/dto/backtest_runs.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case result is already backed by persisted storage rows ordered as
        `rank ASC, variant_key ASC`.
    Raises:
        AssertionError: If route wiring changes payload order or status code for owner runs.
    Side Effects:
        None.
    """
    run = _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000936"))
    result = BacktestRunTopReadResult(
        job=run,
        rows=(
            BacktestJobTopVariant(
                job_id=run.job_id,
                rank=1,
                variant_key="a" * 64,
                indicator_variant_key="b" * 64,
                variant_index=0,
                total_return_pct=10.0,
                payload_json={"schema_version": 1, "label": "first"},
                summary_metrics_json={"total_return_pct": 10.0, "profit_factor": 1.5},
                best_tp_pct=4.0,
                best_sl_pct=2.0,
                report_table_md=None,
                trades_json=None,
                updated_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
            ),
            BacktestJobTopVariant(
                job_id=run.job_id,
                rank=2,
                variant_key="c" * 64,
                indicator_variant_key="d" * 64,
                variant_index=1,
                total_return_pct=8.0,
                payload_json={"schema_version": 1, "label": "second"},
                summary_metrics_json={"total_return_pct": 8.0, "profit_factor": 1.2},
                best_tp_pct=3.0,
                best_sl_pct=1.5,
                report_table_md=None,
                trades_json=None,
                updated_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
            ),
        ),
    )
    client, _, _ = _build_client(top_use_case=_TopUseCaseFake(result=result))

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000936/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["rank"] for item in body["items"]] == [1, 2]
    assert [item["variant_key"] for item in body["items"]] == ["a" * 64, "c" * 64]
    assert body["items"][0]["payload"] == {"schema_version": 1, "label": "first"}
    assert body["items"][1]["payload"] == {"schema_version": 1, "label": "second"}


def test_post_backtest_run_cancel_returns_updated_status_snapshot() -> None:
    """
    Verify `POST /backtests/runs/{run_id}/cancel` returns updated public status payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cancel use-case returns updated owner snapshot.
    Raises:
        AssertionError: If endpoint status or payload drifts from the R7-03 contract.
    Side Effects:
        None.
    """
    cancelled_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000935")
    ).request_cancel(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
    )
    client, _, _ = _build_client(cancel_use_case=_CancelUseCaseFake(run=cancelled_run))

    response = client.post(
        "/backtests/runs/00000000-0000-0000-0000-000000000935/cancel",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    assert response.json()["run_id"] == "00000000-0000-0000-0000-000000000935"


def test_get_backtest_runs_returns_background_modes_and_cancel_marker() -> None:
    """
    Verify public history list preserves both background execution modes and running cancel marker.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R8-03 public history keeps `queued/running` visibility for both background launch modes.
    Raises:
        AssertionError: If execution mode or `cancel_requested_at` fields are dropped.
    Side Effects:
        None.
    """
    running = _running_run(
        run_id=UUID("00000000-0000-0000-0000-000000000936"),
        execution_mode="background_manual_legacy",
    ).request_cancel(
        changed_at=datetime(2026, 3, 29, 12, 0, 30, tzinfo=timezone.utc)
    )
    queued = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000937"),
        execution_mode="background_auto",
    )
    list_fake = _ListUseCaseFake(
        page=BacktestJobListPage(items=(running, queued), next_cursor=None)
    )
    client, _, _ = _build_client(list_use_case=list_fake)

    response = client.get(
        "/backtests/runs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["execution_mode"] == "background_manual_legacy"
    assert body["items"][0]["execution_profile_mode"] == "exact_small"
    assert body["items"][0]["state"] == "running"
    assert body["items"][0]["cancel_requested_at"] == "2026-03-29T12:00:30Z"
    assert body["items"][1]["execution_mode"] == "background_auto"
    assert body["items"][1]["execution_profile_mode"] == "exact_small"
    assert body["items"][1]["state"] == "queued"


def test_get_backtest_run_status_maps_foreign_and_missing_errors() -> None:
    """
    Verify public status route preserves explicit `403` for foreign and `404` for missing runs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case already produced canonical public runs errors.
    Raises:
        AssertionError: If route/error-handler mapping changes status codes or payloads.
    Side Effects:
        None.
    """
    forbidden_client, _, _ = _build_client(
        status_use_case=_StatusUseCaseFake(
            error=backtest_run_forbidden(
                run_id=UUID("00000000-0000-0000-0000-000000000936")
            )
        )
    )
    not_found_client, _, _ = _build_client(
        status_use_case=_StatusUseCaseFake(
            error=backtest_run_not_found(
                run_id=UUID("00000000-0000-0000-0000-000000000937")
            )
        )
    )

    forbidden = forbidden_client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000936",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )
    not_found = not_found_client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000937",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["details"]["run_id"] == (
        "00000000-0000-0000-0000-000000000936"
    )
    assert not_found.status_code == 404
    assert not_found.json()["error"]["details"]["run_id"] == (
        "00000000-0000-0000-0000-000000000937"
    )


def test_get_backtest_runs_rejects_invalid_state_filter() -> None:
    """
    Verify `GET /backtests/runs` rejects unknown state values with deterministic 422 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs reuse the legacy state decoder and unified deterministic 422 mapping.
    Raises:
        AssertionError: If invalid state is accepted or payload changes.
    Side Effects:
        None.
    """
    client, _, _ = _build_client()

    response = client.get(
        "/backtests/runs",
        params={"state": "done"},
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Invalid runs state filter",
            "details": {
                "errors": [
                    {
                        "path": "query.state",
                        "code": "invalid_value",
                        "message": (
                            "state must be one of: queued, running, succeeded, failed, cancelled"
                        ),
                    }
                ]
            },
        }
    }


def test_get_backtest_run_top_maps_validation_error_for_invalid_limit() -> None:
    """
    Verify `GET /backtests/runs/{run_id}/top` maps use-case validation errors to 422.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Top-limit validation stays in application layer and flows through shared handlers.
    Raises:
        AssertionError: If validation error stops mapping to canonical 422 payload.
    Side Effects:
        None.
    """
    client, _, _ = _build_client(
        top_use_case=_TopUseCaseFake(
            error=BacktestValidationError(
                "Top rows limit must be > 0",
                errors=(
                    {
                        "path": "query.limit",
                        "code": "greater_than",
                        "message": "limit must be > 0",
                    },
                ),
            )
        )
    )

    response = client.get(
        "/backtests/runs/00000000-0000-0000-0000-000000000938/top?limit=0",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Top rows limit must be > 0",
            "details": {
                "errors": [
                    {
                        "path": "query.limit",
                        "code": "greater_than",
                        "message": "limit must be > 0",
                    }
                ]
            },
        }
    }


def test_post_backtest_run_variant_report_returns_report_for_one_selected_variant() -> None:
    """
    Verify run-scoped variant-report endpoint returns deterministic detail payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Client sends only `run_id` path param plus explicit selected variant payload.
    Raises:
        AssertionError: If status code, response shape, or forwarded arguments drift.
    Side Effects:
        None.
    """
    variant_report_fake = _VariantReportUseCaseFake(report=_variant_report())
    client, _, resolved_variant_report_fake = _build_client(
        variant_report_use_case=variant_report_fake
    )

    response = client.post(
        "/backtests/runs/00000000-0000-0000-0000-000000000939/variant-report",
        json=_variant_report_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "rows": [{"metric": "Total Return [%]", "value": "12.00"}],
        "table_md": "|Metric|Value|\n|---|---|\n|Total Return [%]|12.00|",
        "trades": [
            {
                "trade_id": 1,
                "direction": "long",
                "entry_bar_index": 0,
                "exit_bar_index": 1,
                "entry_fill_price": 100.0,
                "exit_fill_price": 101.0,
                "qty_base": 1.0,
                "entry_quote_amount": 100.0,
                "exit_quote_amount": 101.0,
                "entry_fee_quote": 0.0,
                "exit_fee_quote": 0.0,
                "gross_pnl_quote": 1.0,
                "net_pnl_quote": 1.0,
                "locked_profit_quote": 0.0,
                "exit_reason": "signal_exit",
            }
        ],
    }
    assert resolved_variant_report_fake.last_run_id == UUID(
        "00000000-0000-0000-0000-000000000939"
    )
    assert resolved_variant_report_fake.last_include_trades is True


def test_post_backtest_run_variant_report_maps_owner_and_missing_errors() -> None:
    """
    Verify run-scoped variant-report endpoint preserves explicit `403` and `404` semantics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case remains source of truth for owner-backed persisted run access policy.
    Raises:
        AssertionError: If route/error mapping drifts from public runs contract.
    Side Effects:
        None.
    """
    forbidden_client, _, _ = _build_client(
        variant_report_use_case=_VariantReportUseCaseFake(
            error=backtest_run_forbidden(
                run_id=UUID("00000000-0000-0000-0000-000000000940")
            )
        )
    )
    not_found_client, _, _ = _build_client(
        variant_report_use_case=_VariantReportUseCaseFake(
            error=backtest_run_not_found(
                run_id=UUID("00000000-0000-0000-0000-000000000941")
            )
        )
    )

    forbidden = forbidden_client.post(
        "/backtests/runs/00000000-0000-0000-0000-000000000940/variant-report",
        json=_variant_report_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )
    not_found = not_found_client.post(
        "/backtests/runs/00000000-0000-0000-0000-000000000941/variant-report",
        json=_variant_report_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["details"]["run_id"] == (
        "00000000-0000-0000-0000-000000000940"
    )
    assert not_found.status_code == 404
    assert not_found.json()["error"]["details"]["run_id"] == (
        "00000000-0000-0000-0000-000000000941"
    )


def test_post_backtest_run_variant_report_rejects_invalid_payload_with_422() -> None:
    """
    Verify run-scoped variant-report endpoint rejects invalid body shape deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        New route requires only `variant` and optional `include_trades` in request body.
    Raises:
        AssertionError: If invalid payload no longer maps to canonical validation error.
    Side Effects:
        None.
    """
    client, _, _ = _build_client()

    response = client.post(
        "/backtests/runs/00000000-0000-0000-0000-000000000942/variant-report",
        json={
            "include_trades": True,
            "template": {"unexpected": "full-run-envelope-is-forbidden-here"},
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Validation failed",
            "details": {
                "errors": [
                    {
                        "path": "body.template",
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    },
                    {
                        "path": "body.variant",
                        "code": "required",
                        "message": "Field required",
                    },
                ]
            },
        }
    }


def _top_result(*, run: BacktestJob) -> BacktestRunTopReadResult:
    """
    Build deterministic top-use-case result fixture for public runs route tests.

    Args:
        run: Persisted run fixture used for response state and metadata.
    Returns:
        BacktestRunTopReadResult: Deterministic summary-only top result fixture.
    Assumptions:
        Persisted rows remain summary-only under the R7-03 contract.
    Raises:
        ValueError: If top-row fixture violates entity invariants.
    Side Effects:
        None.
    """
    row = BacktestJobTopVariant(
        job_id=run.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        summary_metrics_json={"total_return_pct": 10.0, "profit_factor": 1.5},
        best_tp_pct=4.0,
        best_sl_pct=2.0,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )
    return BacktestRunTopReadResult(job=run, rows=(row,))


def _queued_run(
    *,
    run_id: UUID,
    execution_mode: BacktestJobExecutionMode = "sync_inline",
) -> BacktestJob:
    """
    Build deterministic queued persisted run fixture for public route tests.

    Args:
        run_id: Deterministic persisted run identifier.
        execution_mode: Persisted execution-mode literal exposed by public routes.
    Returns:
        BacktestJob: Queued persisted run fixture.
    Assumptions:
        Run belongs to the request principal used in route tests.
    Raises:
        ValueError: If fixture violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=run_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 30, tzinfo=timezone.utc),
        request_json={
            "time_range": {
                "start": "2026-03-28T00:00:00+00:00",
                "end": "2026-03-28T01:00:00+00:00",
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1h",
            },
            "top_k": 25,
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-29",
        ),
        execution_mode=execution_mode,
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )


def _running_run(
    *,
    run_id: UUID,
    execution_mode: BacktestJobExecutionMode = "sync_inline",
) -> BacktestJob:
    """
    Build deterministic running persisted run fixture for lifecycle route tests.

    Args:
        run_id: Deterministic persisted run identifier.
        execution_mode: Persisted execution-mode literal exposed by public routes.
    Returns:
        BacktestJob: Running persisted run fixture.
    Assumptions:
        Lease fields remain valid and non-expired for running state.
    Raises:
        ValueError: If fixture violates lifecycle invariants.
    Side Effects:
        None.
    """
    queued = _queued_run(run_id=run_id, execution_mode=execution_mode)
    return queued.claim(
        changed_at=datetime(2026, 3, 29, 11, 35, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_expires_at=datetime(2026, 3, 29, 11, 36, tzinfo=timezone.utc),
    )


def _failed_run(*, run_id: UUID) -> BacktestJob:
    """
    Build deterministic failed persisted run fixture with Roehub-like error payload.

    Args:
        run_id: Deterministic persisted run identifier.
    Returns:
        BacktestJob: Failed persisted run fixture.
    Assumptions:
        Failed state includes both short and structured error payload fields.
    Raises:
        ValueError: If fixture violates lifecycle invariants.
    Side Effects:
        None.
    """
    running = _running_run(run_id=run_id)
    return running.finish(
        next_state="failed",
        changed_at=datetime(2026, 3, 29, 11, 37, tzinfo=timezone.utc),
        last_error="Execution failed",
        last_error_json=BacktestJobErrorPayload(
            code="unexpected_error",
            message="Execution failed",
            details={"stage": "stage_b"},
        ),
    )


def _variant_report_payload() -> dict[str, object]:
    """
    Build deterministic request body for run-scoped variant-report route tests.

    Args:
        None.
    Returns:
        dict[str, object]: Minimal strict request payload with explicit selected variant.
    Assumptions:
        `run_id` is passed via route path and must not be duplicated in body.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "include_trades": True,
        "variant": {
            "indicator_selections": [
                {
                    "indicator_id": "ma.sma",
                    "inputs": {"source": "close"},
                    "params": {"window": 20},
                }
            ],
            "signal_params": {"ma.sma": {"cross_up": 0.5}},
            "risk_params": {
                "sl_enabled": True,
                "sl_pct": 2.0,
                "tp_enabled": True,
                "tp_pct": 4.0,
            },
            "execution_params": {
                "init_cash_quote": 10000.0,
                "fee_pct": 0.075,
                "slippage_pct": 0.01,
                "fixed_quote": 100.0,
                "safe_profit_percent": 30.0,
            },
            "direction_mode": "long-short",
            "sizing_mode": "all_in",
        },
    }


def _variant_report() -> BacktestReportV1:
    """
    Build deterministic detail-report fixture for run-scoped route tests.

    Args:
        None.
    Returns:
        BacktestReportV1: Report fixture with rows, markdown table, and one trade.
    Assumptions:
        One trade item is enough to validate strict response serialization.
    Raises:
        ValueError: If fixture violates report entity invariants.
    Side Effects:
        None.
    """
    return BacktestReportV1(
        rows=(BacktestMetricRowV1(metric="Total Return [%]", value="12.00"),),
        table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|12.00|",
        trades=(
            TradeV1(
                trade_id=1,
                direction="long",
                entry_bar_index=0,
                exit_bar_index=1,
                entry_fill_price=100.0,
                exit_fill_price=101.0,
                qty_base=1.0,
                entry_quote_amount=100.0,
                exit_quote_amount=101.0,
                entry_fee_quote=0.0,
                exit_fee_quote=0.0,
                gross_pnl_quote=1.0,
                net_pnl_quote=1.0,
                locked_profit_quote=0.0,
                exit_reason="signal_exit",
            ),
        ),
    )
