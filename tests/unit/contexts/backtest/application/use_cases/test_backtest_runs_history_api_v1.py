from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.dto import (
    BacktestMetricRowV1,
    BacktestReportV1,
    BacktestVariantPayloadV1,
    RunBacktestRequest,
    RunBacktestSavedOverrides,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    CurrentUser,
)
from trading.contexts.backtest.application.services import (
    ArtifactPinnedIdentityV2,
    ExecutionProfilesCatalogV2,
    default_execution_profiles_catalog_v2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestRunProgressSnapshotBuilder,
    BuildBacktestRunVariantReportUseCase,
    CancelBacktestRunUseCase,
    GetBacktestRunStatusUseCase,
    GetBacktestRunTopUseCase,
    ListBacktestRunsUseCase,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobExecutionMode,
    BacktestJobStageWeights,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)

_BENCHMARK_CORPUS_PATH = (
    Path(__file__).resolve().parents[6]
    / "tests"
    / "perf_smoke"
    / "contexts"
    / "backtest"
    / "fixtures"
    / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)


class _FakeJobRepository:
    """
    Deterministic in-memory fake for persisted-run repository use-case tests.
    """

    def __init__(
        self,
        *,
        jobs_by_id: Mapping[UUID, BacktestJob] | None = None,
        list_page: BacktestJobListPage | None = None,
    ) -> None:
        """
        Initialize fake repository with deterministic state and optional fixtures.

        Args:
            jobs_by_id: Optional seeded runs mapping.
            list_page: Optional deterministic list page fixture.
        Returns:
            None.
        Assumptions:
            Tests mutate fake state directly through repository methods.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory maps and query logs for assertions.
        """
        self.jobs_by_id = dict(jobs_by_id or {})
        self.list_page = list_page or BacktestJobListPage(items=tuple(), next_cursor=None)
        self.last_cancel_call: tuple[UUID, UserId] | None = None
        self.last_list_query: BacktestJobListQuery | None = None

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Reject unexpected plain create calls in public runs history unit tests.

        Args:
            job: Queued job snapshot.
        Returns:
            BacktestJob: Never returns because this path is out of scope.
        Assumptions:
            Public runs history use-cases do not create new persisted rows.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = job
        raise AssertionError("create is not expected in these tests")

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
    ) -> BacktestJob:
        """
        Reject unexpected sync-inline persistence calls in public runs history unit tests.

        Args:
            job: Terminal job snapshot.
            top_variants: Summary-only top rows.
        Returns:
            BacktestJob: Never returns because this path is out of scope.
        Assumptions:
            Public runs history use-cases operate on already-persisted rows.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = job, top_variants
        raise AssertionError("create_with_top_variants is not expected in these tests")

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Read one run from in-memory store with optional owner filter.

        Args:
            job_id: Requested run identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Matching snapshot or `None`.
        Assumptions:
            Owner filter semantics match repository contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        run = self.jobs_by_id.get(job_id)
        if run is None:
            return None
        if user_id is not None and run.user_id != user_id:
            return None
        return run

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        """
        Return preconfigured list page and record query for assertions.

        Args:
            query: List query payload.
        Returns:
            BacktestJobListPage: Preconfigured deterministic page fixture.
        Assumptions:
            Tests validate query fields separately.
        Raises:
            None.
        Side Effects:
            Records last list query payload.
        """
        self.last_list_query = query
        return self.list_page

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        """
        Simulate deterministic cancel semantics for in-memory owner runs.

        Args:
            job_id: Requested run identifier.
            user_id: Owner identifier.
            cancel_requested_at: Cancel timestamp.
        Returns:
            BacktestJob | None: Updated snapshot or `None`.
        Assumptions:
            Fake uses domain helper `request_cancel` for lifecycle behavior.
        Raises:
            None.
        Side Effects:
            Mutates in-memory run state and records cancel call args.
        """
        self.last_cancel_call = (job_id, user_id)
        run = self.jobs_by_id.get(job_id)
        if run is None or run.user_id != user_id:
            return None
        updated = run.request_cancel(changed_at=cancel_requested_at)
        self.jobs_by_id[job_id] = updated
        return updated

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Return zero active runs because quota semantics are out of scope here.

        Args:
            user_id: Owner identifier.
        Returns:
            int: Always `0`.
        Assumptions:
            Public runs history use-cases do not exercise create-time quotas.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = user_id
        return 0

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return zero because publish-guard counting is out of scope for these tests.

        Args:
            market_id: Canonical market id.
            symbol: Canonical symbol.
            artifact_slot: Candidate slot literal.
            artifact_manifest_hash: Candidate manifest hash.
        Returns:
            int: Always `0`.
        Assumptions:
            Public runs history use-cases do not exercise inactive-slot publish guards.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


class _FakeResultsRepository:
    """
    Deterministic in-memory fake for persisted summary rows repository tests.
    """

    def __init__(self, *, rows: tuple[BacktestJobTopVariant, ...]) -> None:
        """
        Initialize fake results repository with fixed top rows tuple.

        Args:
            rows: Deterministic top rows fixture.
        Returns:
            None.
        Assumptions:
            Rows are already sorted by repository ordering contract.
        Raises:
            None.
        Side Effects:
            Stores last requested limit for assertions.
        """
        self.rows = rows
        self.last_limit: int | None = None

    def list_top_variants(self, *, job_id: UUID, limit: int) -> tuple[BacktestJobTopVariant, ...]:
        """
        Return deterministic slice of preconfigured top rows fixture.

        Args:
            job_id: Requested run identifier.
            limit: Top limit value.
        Returns:
            tuple[BacktestJobTopVariant, ...]: Deterministic rows subset.
        Assumptions:
            Fake ignores job_id because tests control fixture scope.
        Raises:
            None.
        Side Effects:
            Records requested limit value.
        """
        _ = job_id
        self.last_limit = limit
        return self.rows[:limit]

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[BacktestJobTopVariant, ...],
    ) -> bool:
        """
        Satisfy repository protocol for worker-only snapshot writes in history tests.

        Args:
            job_id: Run identifier.
            now: Snapshot timestamp.
            locked_by: Expected worker owner identity.
            rows: Summary rows payload.
        Returns:
            bool: Always `True`.
        Assumptions:
            Public runs history use-cases never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = job_id, now, locked_by, rows
        return True

    def save_stage_a_shortlist(self, *, job_id: UUID, now: datetime, locked_by: str, shortlist):
        """
        Satisfy repository protocol for worker-only shortlist writes in history tests.

        Args:
            job_id: Run identifier.
            now: Snapshot timestamp.
            locked_by: Expected worker owner identity.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: Always `True`.
        Assumptions:
            Public runs history use-cases never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = job_id, now, locked_by, shortlist
        return True

    def get_stage_a_shortlist(self, *, job_id: UUID):
        """
        Satisfy repository protocol for worker-only shortlist reads in history tests.

        Args:
            job_id: Run identifier.
        Returns:
            None: Always `None`.
        Assumptions:
            Public runs history use-cases never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = job_id
        return None


class _FakeRequestDecoder:
    """
    Deterministic persisted-request decoder fake for run-scoped detail use-case tests.
    """

    def __init__(self, *, request: RunBacktestRequest) -> None:
        """
        Store deterministic decoded request fixture.

        Args:
            request: Decoded request DTO returned for every payload.
        Returns:
            None.
        Assumptions:
            Tests assert only that application use-case consumes canonical decoded request.
        Raises:
            None.
        Side Effects:
            Stores last payload for assertions.
        """
        self.request = request
        self.last_payload: Mapping[str, object] | None = None

    def decode(self, *, payload: Mapping[str, object]) -> RunBacktestRequest:
        """
        Return configured request DTO and record input payload.

        Args:
            payload: Persisted `request_json` mapping payload.
        Returns:
            RunBacktestRequest: Configured decoded request fixture.
        Assumptions:
            Payload validation itself is outside the scope of this fake.
        Raises:
            None.
        Side Effects:
            Stores last payload for assertions.
        """
        self.last_payload = payload
        return self.request


class _FakeRunUseCase:
    """
    Deterministic report-builder fake capturing reconstructed template request context.
    """

    def __init__(self, *, report: BacktestReportV1) -> None:
        """
        Store deterministic report fixture.

        Args:
            report: Report payload returned for every invocation.
        Returns:
            None.
        Assumptions:
            Tests verify upstream reconstruction and pinning, not real report computation.
        Raises:
            None.
        Side Effects:
            Stores last invocation kwargs for assertions.
        """
        self.report = report
        self.last_requested_time_range: TimeRange | None = None
        self.last_template: RunBacktestTemplate | None = None
        self.last_warmup_bars: int | None = None
        self.last_variant_payload: BacktestVariantPayloadV1 | None = None
        self.last_include_trades: bool | None = None
        self.last_artifact_context: object | None = None

    def build_variant_report_for_template(
        self,
        *,
        requested_time_range: TimeRange,
        template: RunBacktestTemplate,
        warmup_bars: int | None,
        variant_payload: BacktestVariantPayloadV1,
        include_trades: bool = False,
        run_control=None,
        artifact_context=None,
        template_root_path: str = "body.template",
        template_already_validated: bool = False,
    ) -> BacktestReportV1:
        """
        Capture reconstructed request context and return deterministic report fixture.

        Args:
            requested_time_range: Original request time range.
            template: Reconstructed effective template.
            warmup_bars: Effective warmup override.
            variant_payload: Explicit selected variant payload.
            include_trades: Include-trades flag.
            run_control: Optional cooperative cancellation handle.
            artifact_context: Pinned artifact context object.
            template_root_path: Validation root path literal.
            template_already_validated: Validation short-circuit flag.
        Returns:
            BacktestReportV1: Configured report fixture.
        Assumptions:
            Fake does not execute runtime validation or compute.
        Raises:
            AssertionError: If route/use-case unexpectedly mutates required arguments.
        Side Effects:
            Stores last invocation arguments for assertions.
        """
        _ = run_control, template_root_path, template_already_validated
        self.last_requested_time_range = requested_time_range
        self.last_template = template
        self.last_warmup_bars = warmup_bars
        self.last_variant_payload = variant_payload
        self.last_include_trades = include_trades
        self.last_artifact_context = artifact_context
        return self.report


class _FakeArtifactSlotResolver:
    """
    Deterministic slot resolver fake returning one pinned artifact context marker.
    """

    def __init__(self, *, artifact_context: object) -> None:
        """
        Store deterministic pinned artifact context fixture.

        Args:
            artifact_context: Object returned for every resolve call.
        Returns:
            None.
        Assumptions:
            Tests assert resolver inputs separately from downstream runtime behavior.
        Raises:
            None.
        Side Effects:
            Stores last resolve arguments for assertions.
        """
        self.artifact_context = artifact_context
        self.last_coordinates = None
        self.last_pinned_identity: ArtifactPinnedIdentityV2 | None = None

    def resolve_pinned_context(self, coordinates, pinned_identity: ArtifactPinnedIdentityV2):
        """
        Return configured artifact context and record pin metadata.

        Args:
            coordinates: Artifact coordinates derived from reconstructed template.
            pinned_identity: Persisted immutable artifact identity.
        Returns:
            object: Configured artifact context marker.
        Assumptions:
            Fake resolver does not validate filesystem state.
        Raises:
            None.
        Side Effects:
            Stores last resolve arguments for assertions.
        """
        self.last_coordinates = coordinates
        self.last_pinned_identity = pinned_identity
        return self.artifact_context


def test_get_status_use_case_returns_403_for_foreign_and_404_for_missing_run() -> None:
    """
    Verify owner policy returns `403` for foreign existing run and `404` for missing run.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs use-case reads storage without owner SQL filter first.
    Raises:
        AssertionError: If error mapping violates the R7-03 owner contract.
    Side Effects:
        None.
    """
    owner_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000950"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000999"),
    )
    repository = _FakeJobRepository(jobs_by_id={owner_run.job_id: owner_run})
    use_case = GetBacktestRunStatusUseCase(job_repository=repository)

    with pytest.raises(RoehubError) as forbidden_error:
        use_case.execute(
            run_id=owner_run.job_id,
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )
    assert forbidden_error.value.code == "forbidden"
    assert forbidden_error.value.details is not None
    assert forbidden_error.value.details["run_id"] == str(owner_run.job_id)

    with pytest.raises(RoehubError) as not_found_error:
        use_case.execute(
            run_id=UUID("00000000-0000-0000-0000-000000000951"),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )
    assert not_found_error.value.code == "not_found"
    assert not_found_error.value.details is not None
    assert not_found_error.value.details["run_id"] == "00000000-0000-0000-0000-000000000951"


def test_get_top_use_case_validates_limit_and_reads_rows_for_run() -> None:
    """
    Verify top use-case validates limit and returns deterministic persisted summary rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository returns rows ordered by `rank ASC, variant_key ASC`.
    Raises:
        AssertionError: If limit validation or rows retrieval contract breaks.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000952"),
        user_id=owner_user_id,
    )
    row = BacktestJobTopVariant(
        job_id=owner_run.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        summary_metrics_json={"total_return_pct": 10.0, "profit_factor": 1.2},
        best_tp_pct=4.0,
        best_sl_pct=2.0,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )

    repository = _FakeJobRepository(jobs_by_id={owner_run.job_id: owner_run})
    results_repository = _FakeResultsRepository(rows=(row,))
    use_case = GetBacktestRunTopUseCase(
        job_repository=repository,
        results_repository=results_repository,
        top_k_persisted_default=5,
    )

    result = use_case.execute(
        run_id=owner_run.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
        limit=1,
    )
    assert result.rows == (row,)
    assert results_repository.last_limit == 1

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            run_id=owner_run.job_id,
            current_user=CurrentUser(user_id=owner_user_id),
            limit=6,
        )
    assert error_info.value.code == "validation_error"


def test_get_top_use_case_returns_persisted_rows_in_repository_order() -> None:
    """
    Verify persisted `/top` rows are returned unchanged in `rank ASC, variant_key ASC` order.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - apps/api/routes/backtest_runs.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository already applies canonical persisted ordering for
        `GET /backtests/runs/{run_id}/top`.
    Raises:
        AssertionError: If use-case changes ordering or default persisted limit semantics.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000962"),
        user_id=owner_user_id,
    )
    first_row = BacktestJobTopVariant(
        job_id=owner_run.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        summary_metrics_json={"total_return_pct": 10.0, "profit_factor": 1.2},
        best_tp_pct=4.0,
        best_sl_pct=2.0,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )
    second_row = BacktestJobTopVariant(
        job_id=owner_run.job_id,
        rank=2,
        variant_key="c" * 64,
        indicator_variant_key="d" * 64,
        variant_index=1,
        total_return_pct=8.5,
        payload_json={"schema_version": 1, "variant": "second"},
        summary_metrics_json={"total_return_pct": 8.5, "profit_factor": 1.1},
        best_tp_pct=3.5,
        best_sl_pct=1.5,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    )

    repository = _FakeJobRepository(jobs_by_id={owner_run.job_id: owner_run})
    results_repository = _FakeResultsRepository(rows=(first_row, second_row))
    use_case = GetBacktestRunTopUseCase(
        job_repository=repository,
        results_repository=results_repository,
        top_k_persisted_default=2,
    )

    result = use_case.execute(
        run_id=owner_run.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
        limit=None,
    )

    assert result.job == owner_run
    assert result.rows == (first_row, second_row)
    assert results_repository.last_limit == 2


def test_run_progress_snapshot_builder_uses_request_profile_override_for_weights() -> None:
    """
    Verify additive progress builder honors persisted `execution_profile_mode` override.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted runs may later carry explicit profile literals in `request_json` without
        changing the public route shape or storage schema.
    Raises:
        AssertionError: If profile resolution, weighted progress, or ETA projection drifts.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000963"),
        user_id=owner_user_id,
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={
            "mode": "template",
            "top_k": 25,
            "execution_profile_mode": "exact_parallel",
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    running = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-test-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    ).update_progress(
        changed_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=20,
    )

    progress = BacktestRunProgressSnapshotBuilder(
        benchmark_corpus=_benchmark_corpus(),
        now_provider=lambda: datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc)
    ).build(run=running)

    assert progress.execution_profile_mode == "exact_parallel"
    assert progress.progress_percent == 65
    assert progress.eta_seconds == 33


def test_run_progress_snapshot_builder_falls_back_to_catalog_default_mode() -> None:
    """
    Verify additive progress builder falls back to the configured default profile when missing.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        A2 does not require persisted profile selection on every historical row.
    Raises:
        AssertionError: If default-profile fallback drifts from the execution-profile catalog.
    Side Effects:
        None.
    """
    owner_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000964"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
    )

    progress = BacktestRunProgressSnapshotBuilder().build(run=owner_run)

    assert progress.execution_profile_mode == "exact_small"
    assert progress.progress_percent == 0
    assert progress.eta_seconds is None


def test_run_progress_snapshot_builder_uses_benchmark_fallback_before_throughput_is_defensible(
) -> None:
    """
    Verify persisted run ETA falls back to the committed benchmark corpus before throughput exists.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The current run has started and exposes stage counters, but elapsed time is still too
        small for the timeline-only ETA path to defend a throughput-based estimate.
    Raises:
        AssertionError: If benchmark fallback does not produce the expected deterministic ETA.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000966"),
        user_id=owner_user_id,
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 59, 55, tzinfo=timezone.utc),
        request_json={
            "mode": "template",
            "top_k": 25,
            "execution_profile_mode": "exact_parallel",
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    running = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-test-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    ).update_progress(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=0,
        total_units=48,
    )

    progress = BacktestRunProgressSnapshotBuilder(
        benchmark_corpus=_benchmark_corpus(),
        now_provider=lambda: datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    ).build(run=running)

    assert progress.execution_profile_mode == "exact_parallel"
    assert progress.progress_percent == 35
    assert progress.eta_seconds == 34


def test_run_progress_snapshot_builder_reads_weights_from_execution_profile_catalog() -> None:
    """
    Verify progress/ETA semantics use profile-contract weights instead of a hardcoded side table.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        B3 moves deterministic stage weights into the execution-profile catalog consumed by both
        launch and persisted-run read paths.
    Raises:
        AssertionError: If read path still ignores custom catalog weights.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000965"),
        user_id=owner_user_id,
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={
            "mode": "template",
            "top_k": 25,
            "execution_profile_mode": "exact_parallel",
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    running = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-test-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    ).update_progress(
        changed_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=20,
    )

    default_catalog = default_execution_profiles_catalog_v2()
    exact_parallel = replace(
        default_catalog.profile_for_mode(mode="exact_parallel"),
        progress_weights=BacktestJobStageWeights(stage_a=10, stage_b=85, finalizing=5),
    )
    custom_catalog = ExecutionProfilesCatalogV2(
        default_mode=default_catalog.default_mode,
        available_profiles=tuple(
            exact_parallel if profile.mode == "exact_parallel" else profile
            for profile in default_catalog.available_profiles
        ),
    )

    progress = BacktestRunProgressSnapshotBuilder(
        execution_profiles=custom_catalog,
        now_provider=lambda: datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    ).build(run=running)

    assert progress.execution_profile_mode == "exact_parallel"
    assert progress.progress_percent == 52
    assert progress.eta_seconds == 56


def test_cancel_use_case_returns_updated_owner_run_snapshot() -> None:
    """
    Verify cancel use-case returns idempotent status payload after owner validation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fake repository uses domain `request_cancel` lifecycle method.
    Raises:
        AssertionError: If cancel operation does not update state snapshot.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_run = _queued_run(
        run_id=UUID("00000000-0000-0000-0000-000000000953"),
        user_id=owner_user_id,
    )
    repository = _FakeJobRepository(jobs_by_id={owner_run.job_id: owner_run})
    use_case = CancelBacktestRunUseCase(
        job_repository=repository,
        now_provider=lambda: datetime(2026, 3, 29, 12, 5, tzinfo=timezone.utc),
    )

    cancelled = use_case.execute(
        run_id=owner_run.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
    )

    assert cancelled.state == "cancelled"
    assert repository.last_cancel_call == (owner_run.job_id, owner_user_id)


def test_cancel_use_case_keeps_running_background_auto_visible() -> None:
    """
    Verify running `background_auto` cancel stays visible as `running + cancel_requested_at`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R8-03 keeps running cancel best-effort while public runs API must expose the marker until
        worker finalization.
    Raises:
        AssertionError: If cancel use-case hides the running state or background execution mode.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_run = _running_run(
        run_id=UUID("00000000-0000-0000-0000-000000000954"),
        user_id=owner_user_id,
        execution_mode="background_auto",
    )
    repository = _FakeJobRepository(jobs_by_id={owner_run.job_id: owner_run})
    cancel_requested_at = datetime(2026, 3, 29, 12, 5, tzinfo=timezone.utc)
    use_case = CancelBacktestRunUseCase(
        job_repository=repository,
        now_provider=lambda: cancel_requested_at,
    )

    updated = use_case.execute(
        run_id=owner_run.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
    )

    assert updated.state == "running"
    assert updated.execution_mode == "background_auto"
    assert updated.cancel_requested_at == cancel_requested_at
    assert updated.finished_at is None
    assert repository.last_cancel_call == (owner_run.job_id, owner_user_id)


def test_build_variant_report_use_case_reconstructs_saved_run_from_persisted_snapshot() -> None:
    """
    Verify run-scoped detail use-case rebuilds saved-mode template from persisted run storage.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved-mode request_json may omit template and must be reconstructed from spec snapshot.
    Raises:
        AssertionError: If request reconstruction, artifact pinning, or forwarded args drift.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    saved_run = _saved_run(
        run_id=UUID("00000000-0000-0000-0000-000000000955"),
        user_id=owner_user_id,
    )
    repository = _FakeJobRepository(jobs_by_id={saved_run.job_id: saved_run})
    decoder = _FakeRequestDecoder(request=_saved_request())
    report = BacktestReportV1(rows=(BacktestMetricRowV1(metric="Total Return [%]", value="12.00"),))
    run_use_case = _FakeRunUseCase(report=report)
    artifact_context = object()
    resolver = _FakeArtifactSlotResolver(artifact_context=artifact_context)
    use_case = BuildBacktestRunVariantReportUseCase(
        job_repository=repository,
        request_decoder=decoder,  # type: ignore[arg-type]
        run_use_case=run_use_case,  # type: ignore[arg-type]
        artifact_slot_resolver=resolver,  # type: ignore[arg-type]
    )

    result = use_case.execute(
        run_id=saved_run.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
        variant_payload=_variant_payload(),
        include_trades=True,
    )

    assert result == report
    assert decoder.last_payload == saved_run.request_json
    assert run_use_case.last_requested_time_range == _saved_request().time_range
    assert run_use_case.last_warmup_bars == 144
    assert run_use_case.last_include_trades is True
    assert run_use_case.last_artifact_context is artifact_context
    assert run_use_case.last_template is not None
    assert run_use_case.last_template.instrument_id == InstrumentId(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
    )
    assert run_use_case.last_template.timeframe == Timeframe("1h")
    assert run_use_case.last_template.direction_mode == "short-only"
    assert run_use_case.last_template.execution_params == {
        "fee_pct": 0.1,
        "fixed_quote": 100.0,
        "init_cash_quote": 10000.0,
    }
    assert resolver.last_pinned_identity is not None
    assert resolver.last_pinned_identity.artifact_slot == "slot_b"
    assert resolver.last_pinned_identity.slot_generation == 11
    assert resolver.last_pinned_identity.artifact_asof_date == "2026-03-29"
    assert resolver.last_pinned_identity.artifact_manifest_hash == "d" * 64


def test_build_variant_report_use_case_keeps_403_and_404_owner_policy() -> None:
    """
    Verify run-scoped detail use-case preserves explicit foreign/missing owner semantics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case reads storage without owner filter first, matching other runs endpoints.
    Raises:
        AssertionError: If `403` or `404` contract drifts.
    Side Effects:
        None.
    """
    foreign_run = _saved_run(
        run_id=UUID("00000000-0000-0000-0000-000000000956"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000999"),
    )
    repository = _FakeJobRepository(jobs_by_id={foreign_run.job_id: foreign_run})
    use_case = BuildBacktestRunVariantReportUseCase(
        job_repository=repository,
        request_decoder=_FakeRequestDecoder(request=_saved_request()),  # type: ignore[arg-type]
        run_use_case=_FakeRunUseCase(
            report=BacktestReportV1(
                rows=(BacktestMetricRowV1(metric="Total Return [%]", value="12.00"),)
            )
        ),  # type: ignore[arg-type]
        artifact_slot_resolver=_FakeArtifactSlotResolver(artifact_context=object()),  # type: ignore[arg-type]
    )

    with pytest.raises(RoehubError) as forbidden_error:
        use_case.execute(
            run_id=foreign_run.job_id,
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
            variant_payload=_variant_payload(),
        )
    assert forbidden_error.value.code == "forbidden"
    assert forbidden_error.value.details is not None
    assert forbidden_error.value.details["run_id"] == str(foreign_run.job_id)

    with pytest.raises(RoehubError) as not_found_error:
        use_case.execute(
            run_id=UUID("00000000-0000-0000-0000-000000000957"),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
            variant_payload=_variant_payload(),
        )
    assert not_found_error.value.code == "not_found"
    assert not_found_error.value.details is not None
    assert not_found_error.value.details["run_id"] == (
        "00000000-0000-0000-0000-000000000957"
    )


def test_build_variant_report_use_case_rejects_missing_artifact_pin() -> None:
    """
    Verify run-scoped detail use-case fails deterministically when pin metadata is absent.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Lazy detail must be pinned to persisted artifact identity of original run.
    Raises:
        AssertionError: If missing artifact pin is silently ignored.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    broken_run = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000958"),
        user_id=owner_user_id,
        mode="saved",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={"strategy_id": "00000000-0000-0000-0000-000000000321"},
        request_hash="a" * 64,
        spec_hash="b" * 64,
        spec_payload_json=_saved_spec_payload(),
        engine_params_hash="c" * 64,
        backtest_runtime_config_hash="e" * 64,
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
    repository = _FakeJobRepository(jobs_by_id={broken_run.job_id: broken_run})
    use_case = BuildBacktestRunVariantReportUseCase(
        job_repository=repository,
        request_decoder=_FakeRequestDecoder(request=_saved_request()),  # type: ignore[arg-type]
        run_use_case=_FakeRunUseCase(
            report=BacktestReportV1(
                rows=(BacktestMetricRowV1(metric="Total Return [%]", value="12.00"),)
            )
        ),  # type: ignore[arg-type]
        artifact_slot_resolver=_FakeArtifactSlotResolver(artifact_context=object()),  # type: ignore[arg-type]
    )

    with pytest.raises(BacktestValidationError, match="slot-pinned artifact metadata"):
        use_case.execute(
            run_id=broken_run.job_id,
            current_user=CurrentUser(user_id=owner_user_id),
            variant_payload=_variant_payload(),
        )


def test_list_use_case_passes_keyset_query_to_repository_for_runs() -> None:
    """
    Verify list use-case forwards state/limit/cursor into repository keyset query object.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs reuse the same repository query object as legacy jobs history.
    Raises:
        AssertionError: If query forwarding contract is broken.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000954"),
    )
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    page = BacktestJobListPage(
        items=(_queued_run(run_id=cursor.job_id, user_id=owner_user_id),),
        next_cursor=None,
    )
    repository = _FakeJobRepository(list_page=page)
    use_case = ListBacktestRunsUseCase(job_repository=repository)

    result = use_case.execute(
        current_user=CurrentUser(user_id=owner_user_id),
        state="queued",
        limit=25,
        cursor=cursor,
    )

    assert result == page
    assert repository.last_list_query is not None
    assert repository.last_list_query.state == "queued"
    assert repository.last_list_query.limit == 25
    assert repository.last_list_query.cursor == cursor


def _benchmark_corpus():
    """
    Load the committed benchmark corpus used by persisted runs ETA fallback unit tests.

    Args:
        None.
    Returns:
        object: Typed committed benchmark corpus fixture.
    Assumptions:
        Unit tests may read the committed corpus fixture directly because request-path file IO is
        the production constraint, not the test harness.
    Raises:
        OSError: If the committed benchmark fixture is missing.
        ValueError: If the fixture payload violates the typed corpus contract.
    Side Effects:
        Reads one repository fixture file.
    """
    return load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_PATH
    )


def _queued_run(
    *,
    run_id: UUID,
    user_id: UserId,
    execution_mode: BacktestJobExecutionMode = "sync_inline",
) -> BacktestJob:
    """
    Build deterministic queued persisted run fixture for R7-03 use-case unit tests.

    Args:
        run_id: Deterministic persisted run identifier.
        user_id: Run owner identifier.
        execution_mode: Persisted execution-mode literal exposed by public runs surfaces.
    Returns:
        BacktestJob: Queued persisted run snapshot fixture.
    Assumptions:
        Hash literals are valid lowercase SHA-256 placeholders.
    Raises:
        ValueError: If one fixture field violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=run_id,
        user_id=user_id,
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={"mode": "template", "top_k": 25},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
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
    user_id: UserId,
    execution_mode: BacktestJobExecutionMode = "background_manual_legacy",
) -> BacktestJob:
    """
    Build deterministic running persisted run fixture for public lifecycle tests.

    Args:
        run_id: Deterministic persisted run identifier.
        user_id: Run owner identifier.
        execution_mode: Persisted background execution mode literal.
    Returns:
        BacktestJob: Running persisted run fixture with active lease metadata.
    Assumptions:
        Lease fields stay valid for public status/cancel tests.
    Raises:
        ValueError: If fixture violates domain lifecycle invariants.
    Side Effects:
        None.
    """
    queued = _queued_run(run_id=run_id, user_id=user_id, execution_mode=execution_mode)
    return queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-test-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    )


def _saved_run(*, run_id: UUID, user_id: UserId) -> BacktestJob:
    """
    Build deterministic saved-mode persisted run fixture for lazy detail use-case tests.

    Args:
        run_id: Deterministic persisted run identifier.
        user_id: Run owner identifier.
    Returns:
        BacktestJob: Saved-mode persisted run snapshot fixture.
    Assumptions:
        `request_json` omits template and relies on `spec_payload_json + overrides`.
    Raises:
        ValueError: If fixture violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=run_id,
        user_id=user_id,
        mode="saved",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={
            "time_range": {
                "start": "2026-03-28T00:00:00+00:00",
                "end": "2026-03-28T01:00:00+00:00",
            },
            "strategy_id": "00000000-0000-0000-0000-000000000321",
            "overrides": {
                "direction_mode": "short-only",
                "execution": {"fee_pct": 0.1},
            },
            "warmup_bars": 144,
            "top_k": 25,
            "top_trades_n": 3,
        },
        request_hash="a" * 64,
        spec_hash="b" * 64,
        spec_payload_json=_saved_spec_payload(),
        engine_params_hash="c" * 64,
        backtest_runtime_config_hash="e" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-29",
        ),
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )


def _saved_request() -> RunBacktestRequest:
    """
    Build deterministic decoded saved-mode request fixture for lazy detail tests.

    Args:
        None.
    Returns:
        RunBacktestRequest: Saved-mode request without template payload.
    Assumptions:
        Decoder output mirrors canonical persisted `request_json` shape.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 3, 28, 1, 0, tzinfo=timezone.utc)),
        ),
        strategy_id=UUID("00000000-0000-0000-0000-000000000321"),
        overrides=RunBacktestSavedOverrides(
            direction_mode="short-only",
            execution_params={"fee_pct": 0.1},
        ),
        warmup_bars=144,
        top_k=25,
        top_trades_n=3,
    )


def _saved_spec_payload() -> dict[str, object]:
    """
    Build deterministic saved strategy snapshot payload used for template reconstruction tests.

    Args:
        None.
    Returns:
        dict[str, object]: Minimal Strategy-spec payload compatible with saved snapshot helper.
    Assumptions:
        One indicator definition is sufficient for template reconstruction coverage.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "schema_version": 1,
        "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
        "timeframe": "1h",
        "indicators": [
            {
                "indicator_id": "ma.sma",
                "inputs": {"source": "close"},
                "params": {"window": 20},
            }
        ],
        "signal_grids": {"ma.sma": {"cross_up": {"mode": "explicit", "values": [0.5]}}},
        "risk": {
            "sl_enabled": True,
            "sl_pct": 2.0,
            "tp_enabled": True,
            "tp_pct": 4.0,
        },
        "execution": {
            "fee_pct": 0.075,
            "fixed_quote": 100.0,
            "init_cash_quote": 10000.0,
        },
        "direction_mode": "long-short",
        "sizing_mode": "all_in",
    }


def _variant_payload() -> BacktestVariantPayloadV1:
    """
    Build deterministic explicit selected variant payload for lazy detail tests.

    Args:
        None.
    Returns:
        BacktestVariantPayloadV1: One selected variant payload fixture.
    Assumptions:
        Variant identity keeps v1 scalar semantics unchanged.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return BacktestVariantPayloadV1(
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        signal_params={"ma.sma": {"cross_up": 0.5}},
        risk_params={"sl_enabled": True, "sl_pct": 2.0, "tp_enabled": True, "tp_pct": 4.0},
        execution_params={
            "fee_pct": 0.1,
            "fixed_quote": 100.0,
            "init_cash_quote": 10000.0,
        },
        direction_mode="short-only",
        sizing_mode="all_in",
    )
