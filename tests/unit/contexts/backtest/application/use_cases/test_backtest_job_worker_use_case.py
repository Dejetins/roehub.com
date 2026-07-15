from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestCostEstimate,
    BacktestNoRiskTopResult,
    BacktestPreflightResult,
    BacktestRuntimeGuardrails,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services.v2 import (
    BacktestJobExecutionResult,
    BacktestPreflightService,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobHeavyPromotion,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobCancellationRequested,
    BacktestJobWorkerUseCase,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStage,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-0000-0000-000000000001")


def test_worker_claims_updates_progress_executes_and_finishes_job() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    lease_repository = _LeaseRepository(repository=repository)
    executor = _Executor()
    use_case = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=executor,
        lease_seconds=60,
        locked_by="test-worker",
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.job is not None
    assert result.job.state == "succeeded"
    assert repository.job.state == "succeeded"
    assert len(repository.top_rows) == 1
    assert lease_repository.progress_updates == (("stage_a", 0, 1),)
    assert executor.calls == (job.job_id,)


def test_worker_replays_admitted_job_with_configured_validation_guardrails() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    preflight_service = _GuardrailCapturingPreflightService()
    use_case = BacktestJobWorkerUseCase(
        lease_repository=_LeaseRepository(repository=repository),
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, preflight_service),
        executor=_Executor(),
        lease_seconds=60,
        locked_by="test-worker",
        validation_guardrails=BacktestRuntimeGuardrails(max_top_n=100),
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.job is not None
    assert result.job.state == "succeeded"
    assert preflight_service.validation_guardrails is not None
    assert preflight_service.validation_guardrails.max_top_n == 100


def test_worker_persists_failed_state_when_executor_raises() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    use_case = BacktestJobWorkerUseCase(
        lease_repository=_LeaseRepository(repository=repository),
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=_FailingExecutor(),
        lease_seconds=60,
        locked_by="test-worker",
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.job is not None
    assert result.job.state == "failed"
    assert result.job.last_error == "boom"
    assert result.job.last_error_json is not None
    assert result.job.last_error_json.code == "unexpected_error"


def test_worker_heartbeats_active_lease_while_executor_runs() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    lease_repository = _LeaseRepository(repository=repository)
    heartbeat_seen = threading.Event()
    executor = _BlockingExecutor(heartbeat_seen=heartbeat_seen)
    use_case = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=executor,
        lease_seconds=60,
        heartbeat_interval_seconds=0.001,
        locked_by="test-worker",
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.lease_lost is False
    assert result.job is not None
    assert result.job.state == "succeeded"
    assert lease_repository.heartbeat_calls >= 1


def test_worker_cancels_running_child_when_cancel_requested_during_heartbeat() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    heartbeat_seen = threading.Event()
    lease_repository = _LeaseRepository(
        repository=repository,
        heartbeat_event=heartbeat_seen,
        request_cancel_on_heartbeat=True,
    )
    executor = _CancellableExecutor()
    use_case = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=executor,
        lease_seconds=60,
        heartbeat_interval_seconds=0.001,
        locked_by="test-worker",
    )

    result = use_case.run_next()

    assert heartbeat_seen.is_set()
    assert executor.cancel_seen is True
    assert result.claimed is True
    assert result.status == "cancelled"
    assert result.job is not None
    assert result.job.state == "cancelled"
    assert repository.job.state == "cancelled"
    assert repository.top_rows == ()


def test_worker_requeues_light_candidate_as_heavy_without_terminal_commit() -> None:
    job = _queued_job()
    repository = _Repository(job=job)
    lease_repository = _LeaseRepository(repository=repository)
    use_case = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=_PromotingExecutor(),
        lease_seconds=60,
        locked_by="test-worker",
        scheduling_classes=("light_candidate",),
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.status == "requeued_heavy"
    assert result.job is not None
    assert result.job.state == "queued"
    assert result.job.request_json["scheduling"]["scheduling_class"] == "heavy"
    assert repository.top_rows == ()
    assert repository.terminal_commits == 0
    assert lease_repository.promotions == 1


def test_worker_marks_lease_lost_when_terminal_commit_guard_fails() -> None:
    job = _queued_job()
    repository = _Repository(job=job, finish_returns_none=True)
    use_case = BacktestJobWorkerUseCase(
        lease_repository=_LeaseRepository(repository=repository),
        job_repository=cast(BacktestJobRepository, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        executor=_Executor(),
        lease_seconds=60,
        locked_by="test-worker",
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.lease_lost is True
    assert result.job is None
    assert repository.top_rows == ()
    assert repository.terminal_commits == 0


@dataclass
class _LeaseRepository:
    repository: "_Repository"
    progress_updates: tuple[tuple[BacktestJobStage, int, int], ...] = ()
    heartbeat_calls: int = 0
    heartbeat_event: threading.Event | None = None
    promotions: int = 0
    request_cancel_on_heartbeat: bool = False

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
        scheduling_classes: tuple[str, ...] | None = None,
    ) -> BacktestJob | None:
        _ = scheduling_classes
        if self.repository.job.state != "queued":
            return None
        claimed = self.repository.job.claim(
            changed_at=now,
            locked_by=locked_by,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self.repository.job = claimed
        return claimed

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        _ = job_id, now, locked_by, lease_seconds
        self.heartbeat_calls += 1
        if self.request_cancel_on_heartbeat and self.repository.job.cancel_requested_at is None:
            self.repository.job = self.repository.job.request_cancel(changed_at=now)
        if self.heartbeat_event is not None:
            self.heartbeat_event.set()
        return self.repository.job

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: BacktestJobStage,
        processed_units: int,
        total_units: int,
    ) -> BacktestJob | None:
        _ = locked_by
        if self.repository.job.job_id != job_id or self.repository.job.state != "running":
            return None
        updated = self.repository.job.update_progress(
            changed_at=now,
            stage=stage,
            processed_units=processed_units,
            total_units=total_units,
        )
        self.repository.job = updated
        self.progress_updates = (
            *self.progress_updates,
            (stage, processed_units, total_units),
        )
        return updated

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        _ = locked_by
        if self.repository.job.job_id != job_id:
            return None
        finished = self.repository.job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.repository.job = finished
        return finished

    def promote_to_heavy_and_requeue(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        estimated_combinations_upper_bound: int,
        actual_combinations: int,
        reason: str,
    ) -> BacktestJob | None:
        _ = locked_by, estimated_combinations_upper_bound, actual_combinations, reason
        if self.repository.job.job_id != job_id or self.repository.job.state != "running":
            return None
        request = dict(self.repository.job.request_json)
        request["scheduling"] = {
            "scheduling_class": "heavy",
            "estimated_combinations_upper_bound": estimated_combinations_upper_bound,
            "actual_combinations": actual_combinations,
            "promotion_reason": reason,
        }
        requeued = BacktestJob.create_queued(
            job_id=self.repository.job.job_id,
            organization_id=self.repository.job.organization_id,
            user_id=self.repository.job.user_id,
            mode=self.repository.job.mode,
            created_at=self.repository.job.created_at,
            request_json=request,
            request_hash=self.repository.job.request_hash,
            spec_hash=self.repository.job.spec_hash,
            spec_payload_json=self.repository.job.spec_payload_json,
            engine_params_hash=self.repository.job.engine_params_hash,
            backtest_runtime_config_hash=self.repository.job.backtest_runtime_config_hash,
            artifact_pin=self.repository.job.artifact_pin,
            execution_mode=self.repository.job.execution_mode,
            market_id=self.repository.job.market_id,
            symbol=self.repository.job.symbol,
            timeframe=self.repository.job.timeframe,
            requested_top_n=self.repository.job.requested_top_n,
            ranking_primary_metric=self.repository.job.ranking_primary_metric,
            ranking_secondary_metric=self.repository.job.ranking_secondary_metric,
        )
        object.__setattr__(requeued, "attempt", self.repository.job.attempt)
        object.__setattr__(requeued, "updated_at", now)
        self.repository.job = requeued
        self.promotions += 1
        return requeued


@dataclass
class _Repository:
    job: BacktestJob
    top_rows: tuple[BacktestJobTopVariant, ...] = ()
    finish_returns_none: bool = False
    terminal_commits: int = 0

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        _ = locked_by
        if (
            self.job.job_id != job_id
            or self.job.organization_id != organization_id
            or self.job.user_id != user_id
        ):
            return None
        if self.finish_returns_none:
            return None
        self.job = self.job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.top_rows = top_variants
        self.terminal_commits += 1
        return self.job

    def create(self, *, job: BacktestJob) -> BacktestJob:
        self.job = job
        return job

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = stage_a_shortlist
        self.job = job
        self.top_rows = top_variants
        return job


@dataclass
class _Executor:
    calls: tuple[UUID, ...] = ()

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
    ) -> BacktestJobExecutionResult:
        _ = cancel_event
        self.calls = (*self.calls, job_id)
        top_result = BacktestNoRiskTopResult(
            rank=1,
            score=12.5,
            indicator_rows={"ma.dema": 7},
            metrics={"total_return_pct": 12.5},
            metadata={"ma.dema.source": "close", "ma.dema.window": 5},
        )
        assembly = BacktestTopResultAssemblyService().assemble(
            job_id=job_id,
            normalized_request=preflight.normalized_request,
            top_results=(top_result,),
            updated_at=updated_at,
        )
        return BacktestJobExecutionResult(
            top_variants=assembly.top_variants,
            stage_timings=assembly.stage_timings,
            summary_hash=assembly.summary_hash,
            cleanup_evidence={"result_contains_heavy_references": False},
        )


class _FailingExecutor:
    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
    ) -> BacktestJobExecutionResult:
        _ = job_id, preflight, updated_at, cancel_event
        raise RuntimeError("boom")


class _PromotingExecutor:
    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
    ) -> BacktestJobHeavyPromotion:
        _ = job_id, preflight, updated_at, cancel_event
        return BacktestJobHeavyPromotion(
            estimated_combinations_upper_bound=10,
            actual_combinations=100000,
        )


@dataclass
class _BlockingExecutor:
    heartbeat_seen: threading.Event

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
    ) -> BacktestJobExecutionResult:
        _ = cancel_event
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self.heartbeat_seen.wait(timeout=0.001):
                break
        top_result = BacktestNoRiskTopResult(
            rank=1,
            score=12.5,
            indicator_rows={"ma.dema": 7},
            metrics={"total_return_pct": 12.5},
            metadata={"ma.dema.source": "close", "ma.dema.window": 5},
        )
        assembly = BacktestTopResultAssemblyService().assemble(
            job_id=job_id,
            normalized_request=preflight.normalized_request,
            top_results=(top_result,),
            updated_at=updated_at,
        )
        return BacktestJobExecutionResult(
            top_variants=assembly.top_variants,
            stage_timings=assembly.stage_timings,
            summary_hash=assembly.summary_hash,
            cleanup_evidence={"result_contains_heavy_references": False},
        )


@dataclass
class _CancellableExecutor:
    cancel_seen: bool = False

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
    ) -> BacktestJobExecutionResult:
        _ = job_id, preflight, updated_at
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.wait(timeout=0.001):
                self.cancel_seen = True
                raise BacktestJobCancellationRequested("cancel requested")
        raise RuntimeError("cancel event was not observed")


class _PreflightService:
    def execute(self, payload: Any) -> BacktestPreflightResult:
        return BacktestPreflightResult(
            normalized_request=dict(payload),
            request_hash="d" * 64,
            result_config_hash="e" * 64,
            artifact_metadata=_artifact_metadata(),
            cost_estimate=BacktestCostEstimate(
                indicator_rows=1,
                candidate_combinations=1,
                tp_sl_cells=0,
                cost_class="small",
            ),
        )


class _GuardrailCapturingPreflightService:
    validation_guardrails: BacktestRuntimeGuardrails | None = None

    def execute(
        self,
        payload: Any,
        *,
        validation_guardrails: BacktestRuntimeGuardrails | None = None,
    ) -> BacktestPreflightResult:
        self.validation_guardrails = validation_guardrails
        return _PreflightService().execute(payload)


def _queued_job() -> BacktestJob:
    metadata = _artifact_metadata()
    request = _request()
    request["artifact_metadata"] = metadata.as_mapping()
    return BacktestJob.create_queued(
        job_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000401"),
        mode="template",
        created_at=datetime.now(UTC) - timedelta(seconds=1),
        request_json=request,
        request_hash="d" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="e" * 64,
        backtest_runtime_config_hash="e" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_a",
            artifact_slot_generation=4,
            artifact_manifest_hash=metadata.artifact_manifest_hash,
            artifact_asof_date=metadata.artifact_asof_date,
        ),
        execution_mode="background_auto",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="15m",
        requested_top_n=100,
        ranking_primary_metric="total_return_pct",
    )


def _artifact_metadata() -> BacktestArtifactMetadata:
    return BacktestArtifactMetadata(
        artifact_slot="slot_a",
        artifact_slot_generation=4,
        artifact_manifest_hash="a" * 64,
        artifact_asof_date="2026-03-25",
        hit_times_manifest_hash="b" * 64,
        published_at_utc="2026-03-25T02:00:00Z",
    )


def _request() -> dict[str, Any]:
    return {
        "coordinates": BacktestCoordinates("binance", "spot", "BTCUSDT").as_mapping(),
        "timeframe": "15m",
        "time_range": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"},
        "indicators": [{"indicator_id": "ma.dema", "sources": ["close"]}],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": 100,
    }
