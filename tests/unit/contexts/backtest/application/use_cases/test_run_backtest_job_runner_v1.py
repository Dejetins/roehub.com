from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.backtest.application.services import (
    BacktestRiskVariantV1,
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.application.use_cases import RunBacktestJobRunnerV1
from trading.contexts.backtest.domain.entities import BacktestJob, TradeV1
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)


class _FakeRequestDecoder:
    """
    Deterministic request decoder stub returning predefined request payload.
    """

    def __init__(self, *, request: RunBacktestRequest) -> None:
        """
        Initialize decoder stub with fixed request payload.

        Args:
            request: Prebuilt backtest request fixture.
        Returns:
            None.
        Assumptions:
            Worker tests control the request fixture shape.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._request = request

    def decode(self, *, payload: Mapping[str, Any]) -> RunBacktestRequest:
        """
        Return predefined request payload regardless of persisted JSON content.

        Args:
            payload: Persisted request payload mapping.
        Returns:
            RunBacktestRequest: Prebuilt request fixture.
        Assumptions:
            Decoder behavior is isolated from DTO validation in these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = payload
        return self._request


class _FakeTimelineBuilder:
    """
    Timeline builder stub returning minimal deterministic timeline payload.
    """

    def build(
        self,
        *,
        market_id: MarketId,
        symbol: Symbol,
        timeframe: Timeframe,
        requested_time_range: TimeRange,
        warmup_bars: int,
    ) -> Any:
        """
        Build minimal timeline payload required by job-runner use-case.

        Args:
            market_id: Requested market identifier.
            symbol: Requested symbol.
            timeframe: Requested timeframe.
            requested_time_range: Requested time range.
            warmup_bars: Warmup bars count.
        Returns:
            Any: Timeline-like payload with `candles` and `target_slice`.
        Assumptions:
            Staged scorer fake does not inspect candle arrays.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = market_id, symbol, timeframe, requested_time_range, warmup_bars
        return SimpleNamespace(candles=object(), target_slice=slice(0, 1))


class _NoOpIndicatorCompute:
    """
    Indicator compute placeholder used to satisfy constructor dependency.
    """

    def estimate(self, grid: Any, *, max_variants_guard: int) -> Any:
        """
        Return no-op estimate payload.

        Args:
            grid: Grid payload.
            max_variants_guard: Variants guard.
        Returns:
            Any: Placeholder payload.
        Assumptions:
            Test grid-builder fake bypasses indicator estimate calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = grid, max_variants_guard
        return None

    def compute(self, req: Any) -> Any:
        """
        Return no-op compute payload.

        Args:
            req: Compute request payload.
        Returns:
            Any: Placeholder payload.
        Assumptions:
            Test grid-builder fake bypasses indicator compute calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = req
        return None

    def warmup(self) -> None:
        """
        Execute no-op warmup.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is not relevant for these unit tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _FakeGridContext:
    """
    Minimal staged grid context fixture for Stage-A/Stage-B loops.
    """

    def __init__(
        self,
        *,
        base_variants: tuple[BacktestStageABaseVariant, ...],
        risk_variants: tuple[BacktestRiskVariantV1, ...],
    ) -> None:
        """
        Initialize deterministic staged grid context payload.

        Args:
            base_variants: Stage-A base variants.
            risk_variants: Stage-B risk variants.
        Returns:
            None.
        Assumptions:
            Stage-B total is `len(base_variants) * len(risk_variants)`.
        Raises:
            ValueError: If one fixture array is empty.
        Side Effects:
            None.
        """
        if len(base_variants) == 0:
            raise ValueError("_FakeGridContext requires at least one base variant")
        if len(risk_variants) == 0:
            raise ValueError("_FakeGridContext requires at least one risk variant")
        self._base_variants = base_variants
        self.risk_variants = risk_variants
        self.stage_a_variants_total = len(base_variants)
        self.stage_b_variants_total = len(base_variants) * len(risk_variants)

    def iter_stage_a_variants(self) -> tuple[BacktestStageABaseVariant, ...]:
        """
        Return deterministic Stage-A base variants sequence.

        Args:
            None.
        Returns:
            tuple[BacktestStageABaseVariant, ...]: Base variants fixture.
        Assumptions:
            Fixture order is deterministic and controlled by test data.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._base_variants


class _FakeGridBuilder:
    """
    Grid builder stub returning predefined staged grid context.
    """

    def __init__(self, *, context: _FakeGridContext) -> None:
        """
        Initialize grid builder stub with fixed context.

        Args:
            context: Prebuilt staged grid context.
        Returns:
            None.
        Assumptions:
            Context values match use-case request preselect settings.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._context = context

    def build(
        self,
        *,
        template: RunBacktestTemplate,
        candles: Any,
        indicator_compute: Any,
        preselect: int,
        defaults_provider: Any,
        max_variants_per_compute: int,
        max_compute_bytes_total: int,
    ) -> _FakeGridContext:
        """
        Return predefined staged grid context.

        Args:
            template: Run template payload.
            candles: Candle arrays payload.
            indicator_compute: Indicator compute dependency.
            preselect: Stage-A preselect value.
            defaults_provider: Optional defaults provider.
            max_variants_per_compute: Variants guard.
            max_compute_bytes_total: Memory guard.
        Returns:
            _FakeGridContext: Prebuilt staged grid context.
        Assumptions:
            Guard checks are out of scope for this fake.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (
            template,
            candles,
            indicator_compute,
            preselect,
            defaults_provider,
            max_variants_per_compute,
            max_compute_bytes_total,
        )
        return self._context


class _DeterministicScorerWithDetails:
    """
    Deterministic scorer fake for Stage-A/Stage-B and finalizing details calls.
    """

    def __init__(self, *, stage_a_scores: Mapping[str, float]) -> None:
        """
        Initialize scorer with explicit Stage-A score mapping.

        Args:
            stage_a_scores: Mapping `base_variant_key -> total_return_pct`.
        Returns:
            None.
        Assumptions:
            Stage-B scores use one fixed value to keep ranking tie-break deterministic.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._stage_a_scores = dict(stage_a_scores)
        self._stage_b_score = 7.0

    def score_variant(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic ranking metric payload for Stage-A and Stage-B.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Metric payload with `Total Return [%]`.
        Assumptions:
            Stage-B deterministic tie-break is handled by variant-key sorting.
        Raises:
            ValueError: If Stage-A score mapping is missing requested variant key.
        Side Effects:
            None.
        """
        _ = candles, indicator_selections, signal_params, risk_params, indicator_variant_key
        if stage == "stage_a":
            return {"Total Return [%]": float(self._stage_a_scores[variant_key])}
        return {"Total Return [%]": self._stage_b_score}

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Any:
        """
        Return minimal details payload used by finalizing step.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Any: Details-like object with deterministic metrics payload.
        Assumptions:
            Reporting service fake ignores execution/risk payload structure.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (
            stage,
            candles,
            indicator_selections,
            signal_params,
            risk_params,
            indicator_variant_key,
            variant_key,
        )
        return SimpleNamespace(
            metrics={"Total Return [%]": self._stage_b_score},
            target_slice=slice(0, 1),
            execution_params={},
            risk_params={},
            execution_outcome={},
        )


class _FakeReportingService:
    """
    Reporting service fake recording include-trades policy decisions.
    """

    def __init__(self) -> None:
        """
        Initialize reporting service fake with empty calls log.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `build_report` is called once per finalized persisted variant.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.calls: list[dict[str, Any]] = []

    def build_report(
        self,
        *,
        requested_time_range: TimeRange,
        candles: Any,
        target_slice: slice,
        execution_params: Any,
        execution_outcome: Any,
        include_table_md: bool,
        include_trades: bool,
    ) -> Any:
        """
        Return deterministic report payload with optional trades fixture.

        Args:
            requested_time_range: Requested time range.
            candles: Candle arrays payload.
            target_slice: Reporting target slice.
            execution_params: Execution params payload.
            execution_outcome: Execution outcome payload.
            include_table_md: Include markdown table flag.
            include_trades: Include trades payload flag.
        Returns:
            Any: Report-like object with `table_md` and `trades` fields.
        Assumptions:
            Finalizing requires non-empty markdown table for persisted variants.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory calls log.
        """
        _ = requested_time_range, candles, target_slice, execution_params, execution_outcome
        self.calls.append(
            {
                "include_table_md": include_table_md,
                "include_trades": include_trades,
            }
        )
        return SimpleNamespace(
            table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|7.00|",
            trades=(_sample_trade(),) if include_trades else None,
        )


class _FakeJobRepository:
    """
    Job repository fake for deterministic cancel polling behavior.
    """

    def __init__(
        self,
        *,
        default_job: BacktestJob,
        scripted_get_results: tuple[BacktestJob | None, ...] = (),
    ) -> None:
        """
        Initialize fake repository with scripted `get` responses.

        Args:
            default_job: Fallback job payload for unscripted reads.
            scripted_get_results: Optional queued `get` responses.
        Returns:
            None.
        Assumptions:
            Worker use-case reads only `get(job_id=...)` for cancel checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._default_job = default_job
        self._scripted_get_results = list(scripted_get_results)

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Return scripted or default job snapshot for cancel checks.

        Args:
            job_id: Job identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Job snapshot payload.
        Assumptions:
            `job_id` always matches configured test fixture id.
        Raises:
            None.
        Side Effects:
            Pops one scripted response when queue is non-empty.
        """
        _ = job_id, user_id
        if self._scripted_get_results:
            return self._scripted_get_results.pop(0)
        return self._default_job


class _FakeLeaseRepository:
    """
    Lease repository fake recording progress/heartbeat/finish calls.
    """

    def __init__(self) -> None:
        """
        Initialize fake lease repository with empty call logs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Call order is deterministic for one claimed attempt.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.update_progress_calls: list[dict[str, Any]] = []
        self.heartbeat_calls: list[dict[str, Any]] = []
        self.finish_calls: list[dict[str, Any]] = []

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Return no claimed jobs (unused in use-case unit tests).

        Args:
            now: Claim timestamp.
            locked_by: Worker owner id.
            lease_seconds: Lease TTL.
        Returns:
            BacktestJob | None: Always `None`.
        Assumptions:
            Claim loop is out of scope for these use-case tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = now, locked_by, lease_seconds
        return None

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> Any:
        """
        Record heartbeat call and return non-null payload.

        Args:
            job_id: Job id.
            now: Heartbeat timestamp.
            locked_by: Worker owner id.
            lease_seconds: Lease TTL.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful heartbeat is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.heartbeat_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "lease_seconds": lease_seconds,
            }
        )
        return object()

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: str,
        processed_units: int,
        total_units: int,
    ) -> Any:
        """
        Record progress call and return non-null payload.

        Args:
            job_id: Job id.
            now: Progress timestamp.
            locked_by: Worker owner id.
            stage: Stage literal.
            processed_units: Processed units.
            total_units: Total units.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful progress write is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.update_progress_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "stage": stage,
                "processed_units": processed_units,
                "total_units": total_units,
            }
        )
        return object()

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: str,
        last_error: str | None = None,
        last_error_json: Any = None,
    ) -> Any:
        """
        Record finish call and return non-null payload.

        Args:
            job_id: Job id.
            now: Finish timestamp.
            locked_by: Worker owner id.
            next_state: Target terminal state.
            last_error: Optional short error text.
            last_error_json: Optional structured error payload.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful finish write is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.finish_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "next_state": next_state,
                "last_error": last_error,
                "last_error_json": last_error_json,
            }
        )
        return object()


class _FakeResultsRepository:
    """
    Results repository fake with optional lease-loss simulation on snapshot writes.
    """

    def __init__(self, *, fail_replace_call_numbers: tuple[int, ...] = ()) -> None:
        """
        Initialize results repository fake.

        Args:
            fail_replace_call_numbers: 1-based replace call numbers returning lease lost.
        Returns:
            None.
        Assumptions:
            Lease-lost is represented by `False` from replace method.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._fail_replace_call_numbers = set(fail_replace_call_numbers)
        self.replace_calls: list[dict[str, Any]] = []
        self.shortlist_calls: list[dict[str, Any]] = []

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[Any, ...],
    ) -> bool:
        """
        Record snapshot replace call and optionally simulate lease loss.

        Args:
            job_id: Job id.
            now: Snapshot timestamp.
            locked_by: Worker owner id.
            rows: Snapshot rows payload.
        Returns:
            bool: `False` when configured call number simulates lease loss.
        Assumptions:
            One call can represent running snapshot or finalizing snapshot write.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.replace_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "rows": rows,
            }
        )
        call_number = len(self.replace_calls)
        return call_number not in self._fail_replace_call_numbers

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: Any,
    ) -> bool:
        """
        Record Stage-A shortlist save call and report success.

        Args:
            job_id: Job id.
            now: Save timestamp.
            locked_by: Worker owner id.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: Always `True`.
        Assumptions:
            Lease-loss simulation for shortlist save is not required in these tests.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.shortlist_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "shortlist": shortlist,
            }
        )
        return True


@dataclass
class _NowProvider:
    """
    Monotonic UTC now-provider fixture for deterministic use-case tests.
    """

    current: datetime
    step_seconds: int = 1

    def __call__(self) -> datetime:
        """
        Return current timestamp and advance internal cursor by fixed step.

        Args:
            None.
        Returns:
            datetime: Current UTC timestamp.
        Assumptions:
            Fixed step is positive and small enough for tests.
        Raises:
            None.
        Side Effects:
            Mutates internal timestamp cursor.
        """
        now = self.current
        self.current = now + timedelta(seconds=self.step_seconds)
        return now


def test_process_claimed_job_persists_stage_progress_and_finalizing_policy() -> None:
    """
    Verify succeeded flow writes stage progress semantics and finalizing trades/report policy.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted cap is `min(top_k, top_k_persisted_default)` and finalizing uses top-trades cap.
    Raises:
        AssertionError: If stage progress or finalizing snapshot policy is violated.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 1.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    reporting_service = _FakeReportingService()
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=reporting_service,
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 0, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert lease_repository.finish_calls[-1]["next_state"] == "succeeded"
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_a",
        processed_units=0,
        total_units=2,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_b",
        processed_units=0,
        total_units=4,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_b",
        processed_units=4,
        total_units=4,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="finalizing",
        processed_units=0,
        total_units=1,
    )

    running_rows = results_repository.replace_calls[0]["rows"]
    assert all(row.report_table_md is None for row in running_rows)
    assert all(row.trades_json is None for row in running_rows)

    final_rows = results_repository.replace_calls[-1]["rows"]
    assert len(final_rows) == 2
    assert all(row.report_table_md is not None for row in final_rows)
    assert final_rows[0].trades_json is not None
    assert final_rows[1].trades_json is None


def test_process_claimed_job_cancels_on_batch_boundary() -> None:
    """
    Verify cancel detection on batch boundary transitions job to `cancelled` and stops writes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cancel is checked before next batch and before finalizing.
    Raises:
        AssertionError: If cancel flow writes extra snapshots or misses terminal transition.
    Side Effects:
        None.
    """
    job = _build_running_job()
    cancelled_job = job.request_cancel(changed_at=job.updated_at + timedelta(seconds=1))
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 2.0,
            base_variants[1].base_variant_key: 1.0,
        }
    )
    job_repository = _FakeJobRepository(
        default_job=cancelled_job,
        scripted_get_results=(job, cancelled_job),
    )
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 10, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "cancelled"
    assert len(lease_repository.finish_calls) == 1
    assert lease_repository.finish_calls[0]["next_state"] == "cancelled"
    assert results_repository.shortlist_calls == []
    assert results_repository.replace_calls == []


def test_process_claimed_job_stops_when_lease_lost_during_snapshot_write() -> None:
    """
    Verify lease-lost snapshot write aborts processing immediately without terminal finish write.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Lease-lost is surfaced as `False` from results repository replace call.
    Raises:
        AssertionError: If worker continues writing after lease-lost condition.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 2.0,
            base_variants[1].base_variant_key: 1.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository(fail_replace_call_numbers=(1,))
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 20, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "lease_lost"
    assert len(results_repository.replace_calls) == 1
    assert lease_repository.finish_calls == []


def test_process_claimed_job_persists_running_snapshots_by_time_trigger() -> None:
    """
    Verify running snapshots are persisted when `snapshot_seconds` threshold is reached.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variants-step trigger is effectively disabled by large `snapshot_variants_step`.
    Raises:
        AssertionError: If time trigger does not persist running snapshots.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 3.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=1,
        snapshot_variants_step=10_000,
        stage_batch_size=1,
        now_provider=_NowProvider(
            current=_utc(2026, 2, 23, 10, 30, 0),
            step_seconds=2,
        ),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    running_snapshots = [
        call
        for call in results_repository.replace_calls
        if all(row.report_table_md is None for row in call["rows"])
    ]
    assert report.status == "succeeded"
    assert len(running_snapshots) >= 2


def test_process_claimed_job_persists_running_snapshots_by_variants_step() -> None:
    """
    Verify running snapshots are persisted when `snapshot_variants_step` threshold is reached.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Time trigger is effectively disabled by large `snapshot_seconds` value.
    Raises:
        AssertionError: If running snapshots are not persisted incrementally.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 3.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 30, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    running_snapshots = [
        call
        for call in results_repository.replace_calls
        if all(row.report_table_md is None for row in call["rows"])
    ]
    assert report.status == "succeeded"
    assert len(running_snapshots) >= 2


def _build_use_case(
    *,
    request: RunBacktestRequest,
    job_repository: _FakeJobRepository,
    lease_repository: _FakeLeaseRepository,
    results_repository: _FakeResultsRepository,
    grid_context: _FakeGridContext,
    scorer: _DeterministicScorerWithDetails,
    reporting_service: _FakeReportingService,
    top_k_persisted_default: int,
    snapshot_seconds: int | None,
    snapshot_variants_step: int | None,
    stage_batch_size: int,
    now_provider: _NowProvider,
) -> RunBacktestJobRunnerV1:
    """
    Build job-runner use-case with deterministic fakes for unit tests.

    Args:
        request: Decoded request fixture.
        job_repository: Fake job repository.
        lease_repository: Fake lease repository.
        results_repository: Fake results repository.
        grid_context: Fake staged grid context.
        scorer: Fake scorer with details.
        reporting_service: Fake reporting service.
        top_k_persisted_default: Persisted cap for top rows.
        snapshot_seconds: Optional time trigger threshold.
        snapshot_variants_step: Optional variants-step trigger threshold.
        stage_batch_size: Batch boundary size.
        now_provider: Monotonic now-provider fixture.
    Returns:
        RunBacktestJobRunnerV1: Prepared use-case instance.
    Assumptions:
        Request decoder fake returns the provided request for any payload.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RunBacktestJobRunnerV1(
        job_repository=cast(Any, job_repository),
        lease_repository=cast(Any, lease_repository),
        results_repository=cast(Any, results_repository),
        request_decoder=cast(Any, _FakeRequestDecoder(request=request)),
        candle_timeline_builder=cast(Any, _FakeTimelineBuilder()),
        indicator_compute=cast(Any, _NoOpIndicatorCompute()),
        grid_builder=cast(Any, _FakeGridBuilder(context=grid_context)),
        reporting_service=cast(Any, reporting_service),
        staged_scorer=cast(Any, scorer),
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20_000,
        top_trades_n_default=3,
        top_k_persisted_default=top_k_persisted_default,
        heartbeat_seconds=1_000,
        lease_seconds=60,
        snapshot_seconds=snapshot_seconds,
        snapshot_variants_step=snapshot_variants_step,
        stage_batch_size=stage_batch_size,
        now_provider=now_provider,
    )


def _build_request(*, top_k: int, preselect: int, top_trades_n: int) -> RunBacktestRequest:
    """
    Build deterministic template-mode request fixture for worker use-case tests.

    Args:
        top_k: Requested top-k value.
        preselect: Requested Stage-A preselect value.
        top_trades_n: Requested trades payload cap.
    Returns:
        RunBacktestRequest: Template-mode request fixture.
    Assumptions:
        Indicator template uses one explicit grid and one explicit selection.
    Raises:
        None.
    Side Effects:
        None.
    """
    template = RunBacktestTemplate(
        instrument_id=InstrumentId(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
        ),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ema"),
                params={"length": ExplicitValuesSpec(name="length", values=(10,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ema",
                inputs={"source": "close"},
                params={"length": 10},
            ),
        ),
        signal_grids={},
        execution_params={"slippage_pct": 0.01},
    )
    return RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(_utc(2026, 2, 1, 0, 0, 0)),
            end=UtcTimestamp(_utc(2026, 2, 2, 0, 0, 0)),
        ),
        template=template,
        top_k=top_k,
        preselect=preselect,
        top_trades_n=top_trades_n,
    )


def _build_running_job() -> BacktestJob:
    """
    Build deterministic running Backtest job fixture.

    Args:
        None.
    Returns:
        BacktestJob: Running claimed job fixture.
    Assumptions:
        Hash literals use valid sha256-like 64-char lowercase strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    created_at = _utc(2026, 2, 23, 9, 0, 0)
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000910"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=created_at,
        request_json={"mode": "template"},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
    )
    return queued.claim(
        changed_at=created_at + timedelta(seconds=5),
        locked_by="worker-test-1",
        lease_expires_at=created_at + timedelta(seconds=65),
    )


def _build_stage_a_variants() -> tuple[BacktestStageABaseVariant, ...]:
    """
    Build deterministic Stage-A variants fixture.

    Args:
        None.
    Returns:
        tuple[BacktestStageABaseVariant, ...]: Stage-A variants fixture.
    Assumptions:
        Base variant keys are unique canonical 64-char literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    selection_a = IndicatorVariantSelection(
        indicator_id="ema",
        inputs={"source": "close"},
        params={"length": 10},
    )
    selection_b = IndicatorVariantSelection(
        indicator_id="ema",
        inputs={"source": "close"},
        params={"length": 20},
    )
    return (
        BacktestStageABaseVariant(
            stage_a_index=0,
            indicator_selections=(selection_a,),
            signal_params={"ema": {"threshold": 1}},
            indicator_variant_key="1" * 64,
            base_variant_key="a" * 64,
        ),
        BacktestStageABaseVariant(
            stage_a_index=1,
            indicator_selections=(selection_b,),
            signal_params={"ema": {"threshold": 2}},
            indicator_variant_key="2" * 64,
            base_variant_key="b" * 64,
        ),
    )


def _build_risk_variants() -> tuple[BacktestRiskVariantV1, ...]:
    """
    Build deterministic Stage-B risk variants fixture.

    Args:
        None.
    Returns:
        tuple[BacktestRiskVariantV1, ...]: Risk variants fixture.
    Assumptions:
        Risk payload shape follows v1 keys `sl_enabled/sl_pct/tp_enabled/tp_pct`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        BacktestRiskVariantV1(
            risk_index=0,
            risk_params={
                "sl_enabled": False,
                "sl_pct": None,
                "tp_enabled": False,
                "tp_pct": None,
            },
        ),
        BacktestRiskVariantV1(
            risk_index=1,
            risk_params={
                "sl_enabled": True,
                "sl_pct": 1.0,
                "tp_enabled": True,
                "tp_pct": 2.0,
            },
        ),
    )


def _sample_trade() -> TradeV1:
    """
    Build deterministic trade fixture for finalizing trades payload tests.

    Args:
        None.
    Returns:
        TradeV1: Deterministic closed-trade snapshot.
    Assumptions:
        Numeric values satisfy `TradeV1` domain invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return TradeV1(
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
        exit_reason="signal",
    )


def _has_progress_call(
    *,
    calls: list[dict[str, Any]],
    stage: str,
    processed_units: int,
    total_units: int,
) -> bool:
    """
    Check whether progress calls list contains expected stage counters.

    Args:
        calls: Recorded progress calls.
        stage: Expected stage.
        processed_units: Expected processed units.
        total_units: Expected total units.
    Returns:
        bool: `True` when matching call exists.
    Assumptions:
        Calls list order is deterministic but exact index is not asserted.
    Raises:
        None.
    Side Effects:
        None.
    """
    for call in calls:
        if (
            call["stage"] == stage
            and call["processed_units"] == processed_units
            and call["total_units"] == total_units
        ):
            return True
    return False


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    """
    Build timezone-aware UTC datetime helper for fixtures.

    Args:
        year: Year component.
        month: Month component.
        day: Day component.
        hour: Hour component.
        minute: Minute component.
        second: Second component.
    Returns:
        datetime: UTC-aware datetime value.
    Assumptions:
        Input components form a valid calendar datetime.
    Raises:
        ValueError: If datetime components are invalid.
    Side Effects:
        None.
    """
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
