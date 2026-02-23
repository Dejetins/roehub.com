from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestStrategySnapshot,
    CurrentUser,
)
from trading.contexts.backtest.application.use_cases import (
    CancelBacktestJobUseCase,
    CreateBacktestJobCommand,
    CreateBacktestJobUseCase,
    GetBacktestJobStatusUseCase,
    GetBacktestJobTopUseCase,
    ListBacktestJobsUseCase,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobStageAShortlist,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
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


class _FakeJobRepository:
    """
    Deterministic in-memory fake for Backtest job repository use-case tests.
    """

    def __init__(
        self,
        *,
        active_total: int = 0,
        jobs_by_id: Mapping[UUID, BacktestJob] | None = None,
        list_page: BacktestJobListPage | None = None,
    ) -> None:
        """
        Initialize fake repository with deterministic state and optional fixtures.

        Args:
            active_total: Initial active jobs count returned by quota read.
            jobs_by_id: Optional seeded jobs mapping.
            list_page: Optional deterministic list page fixture.
        Returns:
            None.
        Assumptions:
            Tests mutate fake state directly through repository methods.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory maps/counters for test assertions.
        """
        self.active_total = active_total
        self.jobs_by_id = dict(jobs_by_id or {})
        self.list_page = list_page or BacktestJobListPage(items=tuple(), next_cursor=None)
        self.last_create_job: BacktestJob | None = None
        self.last_cancel_call: tuple[UUID, UserId] | None = None
        self.last_list_query: BacktestJobListQuery | None = None

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Persist job snapshot into in-memory store.

        Args:
            job: Job snapshot to persist.
        Returns:
            BacktestJob: Persisted job snapshot.
        Assumptions:
            Job ids are unique in test setup.
        Raises:
            None.
        Side Effects:
            Mutates in-memory jobs map and records last create call.
        """
        self.last_create_job = job
        self.jobs_by_id[job.job_id] = job
        return job

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Read one job from in-memory store with optional owner filter.

        Args:
            job_id: Requested job identifier.
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
        job = self.jobs_by_id.get(job_id)
        if job is None:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

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
        Simulate deterministic cancel semantics for in-memory owner jobs.

        Args:
            job_id: Requested job identifier.
            user_id: Owner identifier.
            cancel_requested_at: Cancel timestamp.
        Returns:
            BacktestJob | None: Updated snapshot or `None`.
        Assumptions:
            Fake uses domain helper `request_cancel` for lifecycle behavior.
        Raises:
            None.
        Side Effects:
            Mutates in-memory job state and records cancel call args.
        """
        _ = cancel_requested_at
        self.last_cancel_call = (job_id, user_id)
        job = self.jobs_by_id.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        updated = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs_by_id[job_id] = updated
        return updated

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Return deterministic active jobs counter fixture.

        Args:
            user_id: Owner identifier.
        Returns:
            int: Active jobs count fixture.
        Assumptions:
            Counter fixture is configured by test setup.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = user_id
        return self.active_total


class _FakeResultsRepository:
    """
    Deterministic in-memory fake for Backtest job results repository tests.
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
            job_id: Requested job identifier.
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
        Satisfy repository protocol for worker-only snapshot writes in use-case tests.

        Args:
            job_id: Job identifier.
            now: Snapshot timestamp.
            locked_by: Lease owner marker.
            rows: Replacement top rows.
        Returns:
            bool: Always `True` for this in-memory fake.
        Assumptions:
            EPIC-11 API use-case tests never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (job_id, now, locked_by, rows)
        return True

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: BacktestJobStageAShortlist,
    ) -> bool:
        """
        Satisfy repository protocol for worker shortlist writes in use-case tests.

        Args:
            job_id: Job identifier.
            now: Upsert timestamp.
            locked_by: Lease owner marker.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: Always `True` for this in-memory fake.
        Assumptions:
            EPIC-11 API use-case tests never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (job_id, now, locked_by, shortlist)
        return True

    def get_stage_a_shortlist(self, *, job_id: UUID) -> BacktestJobStageAShortlist | None:
        """
        Satisfy repository protocol for worker shortlist reads in use-case tests.

        Args:
            job_id: Job identifier.
        Returns:
            BacktestJobStageAShortlist | None: Always `None` in this fake.
        Assumptions:
            EPIC-11 API use-case tests do not depend on shortlist payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = job_id
        return None


@dataclass(frozen=True, slots=True)
class _FakeStrategyReader:
    """
    Deterministic strategy reader fake returning one optional snapshot fixture.
    """

    snapshot: BacktestStrategySnapshot | None

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Return preconfigured snapshot independent from requested strategy id.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Configured snapshot fixture.
        Assumptions:
            Use-case tests focus on create-flow policy and hashes.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return self.snapshot



def test_create_backtest_job_use_case_persists_effective_snapshot_and_hashes() -> None:
    """
    Verify template-mode create flow stores effective scalar defaults/execution and hashes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request payload follows strict API envelope shape expected by worker decoder.
    Raises:
        AssertionError: If create snapshot payload or hashes are inconsistent.
    Side Effects:
        None.
    """
    repository = _FakeJobRepository(active_total=0)
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="c" * 64,
        now_provider=lambda: datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000901"),
    )

    run_request = RunBacktestRequest(
        time_range=_time_range(),
        template=_template(),
        top_k=5,
        top_trades_n=2,
    )
    command = CreateBacktestJobCommand(
        run_request=run_request,
        request_payload=_template_request_payload(),
    )

    created = use_case.execute(
        command=command,
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.job_id == UUID("00000000-0000-0000-0000-000000000901")
    assert created.mode == "template"
    assert created.state == "queued"
    assert created.request_json["warmup_bars"] == 200
    assert created.request_json["top_k"] == 5
    assert created.request_json["preselect"] == 20000
    assert created.request_json["top_trades_n"] == 2
    assert created.request_json["template"]["execution"] == {
        "fee_pct": 0.075,
        "fixed_quote": 100.0,
        "init_cash_quote": 10000.0,
        "safe_profit_percent": 30.0,
        "slippage_pct": 0.01,
    }
    assert created.spec_hash is None
    assert len(created.request_hash) == 64
    assert len(created.engine_params_hash) == 64
    assert created.backtest_runtime_config_hash == "c" * 64



def test_create_backtest_job_use_case_saved_mode_persists_spec_hash_and_snapshot() -> None:
    """
    Verify saved-mode create flow persists `spec_hash/spec_payload_json` and effective overrides.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved strategy snapshot fixture belongs to current user.
    Raises:
        AssertionError: If saved-mode reproducibility fields are missing.
    Side Effects:
        None.
    """
    strategy_snapshot = _strategy_snapshot(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
    )
    repository = _FakeJobRepository(active_total=0)
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=strategy_snapshot),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="d" * 64,
        now_provider=lambda: datetime(2026, 2, 23, 12, 1, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000902"),
    )

    strategy_id = UUID("00000000-0000-0000-0000-000000000501")
    run_request = RunBacktestRequest(
        time_range=_time_range(),
        strategy_id=strategy_id,
    )
    command = CreateBacktestJobCommand(
        run_request=run_request,
        request_payload={
            "time_range": {
                "start": "2026-02-21T00:00:00+00:00",
                "end": "2026-02-21T01:00:00+00:00",
            },
            "strategy_id": str(strategy_id),
        },
    )

    created = use_case.execute(
        command=command,
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.mode == "saved"
    assert created.spec_hash is not None
    assert created.spec_payload_json == strategy_snapshot.spec_payload
    assert created.request_json["strategy_id"] == str(strategy_id)
    assert created.request_json["overrides"]["execution"]["fee_pct"] == 0.075



def test_create_backtest_job_use_case_rejects_top_k_above_persisted_cap() -> None:
    """
    Verify create flow returns deterministic `validation_error` for `top_k` over persisted cap.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        EPIC-11 `top_k <= top_k_persisted_default` invariant is mandatory.
    Raises:
        AssertionError: If error code/details are not deterministic.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=0),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="e" * 64,
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(
                    time_range=_time_range(),
                    template=_template(),
                    top_k=11,
                ),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body.top_k",
                "code": "max_value",
                "message": "top_k must be <= 10",
            }
        ]
    }



def test_create_backtest_job_use_case_rejects_active_quota_exceeded() -> None:
    """
    Verify create flow returns deterministic `validation_error` when active quota is exceeded.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Active jobs are counted as `queued + running`.
    Raises:
        AssertionError: If quota violation does not map to deterministic error.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=3),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="f" * 64,
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(time_range=_time_range(), template=_template()),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body",
                "code": "quota_exceeded",
                "message": "active jobs limit reached (3/3)",
            }
        ]
    }



def test_get_status_use_case_returns_403_for_foreign_and_404_for_missing() -> None:
    """
    Verify owner policy returns `403` for foreign existing job and `404` for missing job.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case reads job without owner SQL filter first.
    Raises:
        AssertionError: If error mapping violates EPIC-11 owner contract.
    Side Effects:
        None.
    """
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000999"),
    )
    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    use_case = GetBacktestJobStatusUseCase(job_repository=repository)

    with pytest.raises(RoehubError) as forbidden_error:
        use_case.execute(
            job_id=owner_job.job_id,
            current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
        )
    assert forbidden_error.value.code == "forbidden"

    with pytest.raises(RoehubError) as not_found_error:
        use_case.execute(
            job_id=UUID("00000000-0000-0000-0000-000000000811"),
            current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
        )
    assert not_found_error.value.code == "not_found"



def test_get_top_use_case_validates_limit_and_reads_rows() -> None:
    """
    Verify top use-case validates limit against persisted cap and returns deterministic rows.

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
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000820"),
        user_id=owner_user_id,
    )
    row = BacktestJobTopVariant(
        job_id=owner_job.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
    )

    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    results_repository = _FakeResultsRepository(rows=(row,))
    use_case = GetBacktestJobTopUseCase(
        job_repository=repository,
        results_repository=results_repository,
        top_k_persisted_default=5,
    )

    result = use_case.execute(
        job_id=owner_job.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
        limit=1,
    )
    assert result.rows == (row,)
    assert results_repository.last_limit == 1

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            job_id=owner_job.job_id,
            current_user=CurrentUser(user_id=owner_user_id),
            limit=6,
        )
    assert error_info.value.code == "validation_error"



def test_cancel_use_case_returns_updated_owner_snapshot() -> None:
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
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000830"),
        user_id=owner_user_id,
    )
    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    use_case = CancelBacktestJobUseCase(
        job_repository=repository,
        now_provider=lambda: datetime(2026, 2, 23, 12, 5, tzinfo=timezone.utc),
    )

    cancelled = use_case.execute(
        job_id=owner_job.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
    )

    assert cancelled.state == "cancelled"
    assert repository.last_cancel_call == (owner_job.job_id, owner_user_id)



def test_list_use_case_passes_keyset_query_to_repository() -> None:
    """
    Verify list use-case forwards state/limit/cursor into repository keyset query object.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository query object validates deterministic limit bounds.
    Raises:
        AssertionError: If query forwarding contract is broken.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000840"),
    )
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    page = BacktestJobListPage(
        items=(_queued_job(job_id=cursor.job_id, user_id=owner_user_id),),
        next_cursor=None,
    )
    repository = _FakeJobRepository(list_page=page)
    use_case = ListBacktestJobsUseCase(job_repository=repository)

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



def _template_request_payload() -> Mapping[str, Any]:
    """
    Build minimal valid API template payload used by create use-case command fixture.

    Args:
        None.
    Returns:
        Mapping[str, Any]: API transport payload fixture.
    Assumptions:
        Shape is compatible with `BacktestsPostRequest` strict DTO.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-21T00:00:00+00:00",
            "end": "2026-02-21T01:00:00+00:00",
        },
        "template": {
            "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
            "timeframe": "1m",
            "indicator_grids": [
                {
                    "indicator_id": "ma.sma",
                    "params": {"window": {"mode": "explicit", "values": [20]}},
                }
            ],
        },
    }



def _queued_job(*, job_id: UUID, user_id: UserId) -> BacktestJob:
    """
    Build deterministic queued job fixture for EPIC-11 use-case unit tests.

    Args:
        job_id: Deterministic job identifier.
        user_id: Job owner identifier.
    Returns:
        BacktestJob: Queued domain job snapshot fixture.
    Assumptions:
        Hash literals are valid lowercase SHA-256 placeholders.
    Raises:
        ValueError: If one fixture field violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=job_id,
        user_id=user_id,
        mode="template",
        created_at=datetime(2026, 2, 23, 11, 55, tzinfo=timezone.utc),
        request_json={"mode": "template", "top_k": 5},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
    )



def _template() -> RunBacktestTemplate:
    """
    Build deterministic ad-hoc template fixture for create-use-case tests.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Minimal valid template fixture.
    Assumptions:
        One indicator axis is sufficient for create-flow tests.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
    )



def _strategy_snapshot(*, user_id: UserId) -> BacktestStrategySnapshot:
    """
    Build deterministic saved strategy snapshot fixture for create-use-case tests.

    Args:
        user_id: Snapshot owner identifier.
    Returns:
        BacktestStrategySnapshot: Valid saved snapshot fixture.
    Assumptions:
        Snapshot spec payload contains non-empty JSON object.
    Raises:
        ValueError: If fixture violates snapshot invariants.
    Side Effects:
        None.
    """
    return BacktestStrategySnapshot(
        strategy_id=UUID("00000000-0000-0000-0000-000000000501"),
        user_id=user_id,
        is_deleted=False,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        spec_payload={
            "schema_version": 1,
            "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
            "timeframe": "1m",
            "indicators": [
                {
                    "id": "ma.sma",
                    "inputs": {"source": "close"},
                    "params": {"window": 20},
                }
            ],
        },
    )



def _time_range() -> TimeRange:
    """
    Build deterministic UTC half-open time range fixture for request DTO tests.

    Args:
        None.
    Returns:
        TimeRange: Shared time range fixture.
    Assumptions:
        Range start is strictly before range end.
    Raises:
        ValueError: If fixture violates `TimeRange` invariants.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 21, 0, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 21, 1, 0, tzinfo=timezone.utc)),
    )
