from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    CurrentUser,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestRunProgressSnapshotBuilder,
    CancelBacktestRunUseCase,
    GetBacktestRunStatusUseCase,
    GetBacktestRunTopUseCase,
    ListBacktestRunsUseCase,
)
from trading.contexts.backtest.application.use_cases import (
    backtest_runs_history_api_v1 as backtest_runs_history_module,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobExecutionMode,
    BacktestJobStageAShortlist,
    BacktestJobStageWeights,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.backtest_artifacts.application.services.v2.benchmark_corpus_v2 import (
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.execution_profile_v2 import (
    ExecutionProfilesCatalogV2,
    default_execution_profiles_catalog_v2,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId

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
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        """
        Reject unexpected sync-inline persistence calls in public runs history unit tests.

        Args:
            job: Terminal job snapshot.
            top_variants: Summary-only top rows.
            stage_a_shortlist: Optional Stage A shortlist snapshot from sync persistence.
        Returns:
            BacktestJob: Never returns because this path is out of scope.
        Assumptions:
            Public runs history use-cases operate on already-persisted rows.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = job, top_variants, stage_a_shortlist
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def test_run_progress_snapshot_builder_uses_additive_effective_profile_for_weights() -> None:
    """
    Verify additive progress builder honors persisted additive effective profile metadata.

    Docs:
      - docs/architecture/backtest/README.md
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
        New rows expose one stable public `execution_profile_mode` while storing its backing
        source in additive execution-profile metadata outside `request_json`.
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
    assert progress.progress_percent == 70
    assert progress.eta_seconds == 26


def test_run_progress_snapshot_builder_keeps_legacy_request_json_profile_fallback() -> None:
    """
    Verify additive progress builder keeps explicit legacy `request_json` profile fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Historical rows may still predate additive profile columns and must remain readable until
        backfill or retention cleanup completes.
    Raises:
        AssertionError: If the compatibility branch for legacy rows disappears.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000968"),
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
        now_provider=lambda: datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    ).build(run=running)

    assert progress.execution_profile_mode == "exact_parallel"
    assert progress.progress_percent == 70
    assert progress.eta_seconds == 26


def test_run_progress_snapshot_builder_falls_back_to_catalog_default_mode() -> None:
    """
    Verify additive progress builder falls back to the configured default profile when missing.

    Docs:
      - docs/architecture/backtest/README.md
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
    assert progress.progress_percent == 45
    assert progress.eta_seconds == 34


def test_run_progress_snapshot_builder_keeps_progress_bounded_and_monotonic_across_public_stages(
) -> None:
    """
    Verify retained-frontier-aware default weights stay bounded and monotonic across public stages.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Internal row prefilter, combo prefilter, and retained-candidate exact work may reshape
        the Stage A share, but persisted history must still expose monotonic `stage_a -> stage_b`
        progress on the stable public stage vocabulary.
    Raises:
        AssertionError: If progress leaves `[0, 100]` or regresses across the public stages.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000967"),
        user_id=owner_user_id,
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
        request_json={
            "mode": "template",
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
    claimed = queued.claim(
        changed_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        locked_by="worker-test-1",
        lease_expires_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
    )
    running_stage_a = claimed.update_progress(
        changed_at=datetime(2026, 3, 29, 12, 0, 20, tzinfo=timezone.utc),
        stage="stage_a",
        processed_units=8,
        total_units=20,
    )
    running_stage_b_start = running_stage_a.update_progress(
        changed_at=datetime(2026, 3, 29, 12, 0, 40, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=0,
        total_units=48,
    )
    running_stage_b_mid = running_stage_b_start.update_progress(
        changed_at=datetime(2026, 3, 29, 12, 1, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=12,
        total_units=24,
    )

    builder = BacktestRunProgressSnapshotBuilder()
    stage_a_progress = builder.build(run=running_stage_a).progress_percent
    stage_b_start_progress = builder.build(run=running_stage_b_start).progress_percent
    stage_b_mid_progress = builder.build(run=running_stage_b_mid).progress_percent

    assert 0 <= stage_a_progress < stage_b_start_progress < stage_b_mid_progress <= 100


def test_later_backtest_job_stages_keeps_public_stage_vocabulary_stable() -> None:
    """
    Verify later-stage projection keeps public `stage_a/stage_b/finalizing` vocabulary stable.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Internal runtime sub-stage details may evolve, but public history ETA/progress projection
        must remain fixed to the stable stage literals.
    Raises:
        AssertionError: If later-stage projection leaks non-public stage literals.
    Side Effects:
        None.
    """
    assert backtest_runs_history_module._later_backtest_job_stages(stage="stage_a") == (
        "stage_b",
        "finalizing",
    )
    assert backtest_runs_history_module._later_backtest_job_stages(stage="stage_b") == (
        "finalizing",
    )
    assert (
        backtest_runs_history_module._later_backtest_job_stages(stage="finalizing") == tuple()
    )
    assert (
        backtest_runs_history_module._later_backtest_job_stages(stage="stage_a_prefilter")
        == tuple()
    )


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
    execution_mode: BacktestJobExecutionMode = "background_auto",
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
