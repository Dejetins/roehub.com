from __future__ import annotations

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
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services.v2 import (
    BacktestJobExecutionResult,
    BacktestPreflightService,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobWorkerUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStage,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.shared_kernel.primitives import UserId


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


@dataclass
class _LeaseRepository:
    repository: "_Repository"
    progress_updates: tuple[tuple[BacktestJobStage, int, int], ...] = ()

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
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


@dataclass
class _Repository:
    job: BacktestJob
    top_rows: tuple[BacktestJobTopVariant, ...] = ()

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        _ = locked_by
        if self.job.job_id != job_id or self.job.user_id != user_id:
            return None
        self.job = self.job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.top_rows = top_variants
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
    ) -> BacktestJobExecutionResult:
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
    ) -> BacktestJobExecutionResult:
        _ = job_id, preflight, updated_at
        raise RuntimeError("boom")


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


def _queued_job() -> BacktestJob:
    metadata = _artifact_metadata()
    request = _request()
    request["artifact_metadata"] = metadata.as_mapping()
    return BacktestJob.create_queued(
        job_id=uuid4(),
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
