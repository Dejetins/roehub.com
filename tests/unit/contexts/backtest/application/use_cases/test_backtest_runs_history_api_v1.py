from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    CurrentUser,
)
from trading.contexts.backtest.application.use_cases import (
    CancelBacktestRunUseCase,
    GetBacktestRunStatusUseCase,
    GetBacktestRunTopUseCase,
    ListBacktestRunsUseCase,
)
from trading.contexts.backtest.domain.entities import BacktestJob, BacktestJobTopVariant
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


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


def _queued_run(*, run_id: UUID, user_id: UserId) -> BacktestJob:
    """
    Build deterministic queued persisted run fixture for R7-03 use-case unit tests.

    Args:
        run_id: Deterministic persisted run identifier.
        user_id: Run owner identifier.
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
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )
