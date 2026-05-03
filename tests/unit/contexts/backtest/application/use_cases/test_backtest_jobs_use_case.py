from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import pytest

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestCostEstimate,
    BacktestLazyTradesDetailReadModel,
    BacktestNoRiskTopResult,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestJobExecutionResult,
    BacktestRuntimeConfig,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobsUseCase,
    BacktestJobWorkerUseCase,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


def test_trades_resolves_public_variant_key_only() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000301")
    job, row = _job_and_row(user_id=user_id)
    repository = _Repository(job=job, top_rows=(row,))
    lazy_service = _LazyService()
    use_case = _use_case(repository=repository, lazy_service=lazy_service)

    result = use_case.trades(
        user_id=user_id,
        job_id=job.job_id,
        variant_key=str(row.payload_json["public_variant_key"]),
    )

    assert result.variant_key == row.payload_json["public_variant_key"]
    assert result.variant_hash == row.variant_key
    assert lazy_service.requests == ((row.payload_json["public_variant_key"], row.variant_key),)


def test_trades_does_not_resolve_raw_storage_sha_as_public_key() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000302")
    job, row = _job_and_row(user_id=user_id)
    use_case = _use_case(repository=_Repository(job=job, top_rows=(row,)))

    with pytest.raises(RoehubError) as exc_info:
        use_case.trades(user_id=user_id, job_id=job.job_id, variant_key=row.variant_key)

    assert exc_info.value.code == "backtest.not_found"


def test_trades_enforces_ownership_before_variant_lookup() -> None:
    owner_id = UserId.from_string("00000000-0000-0000-0000-000000000303")
    foreign_id = UserId.from_string("00000000-0000-0000-0000-000000000304")
    job, row = _job_and_row(user_id=owner_id)
    repository = _Repository(job=job, top_rows=(row,))
    use_case = _use_case(repository=repository)

    with pytest.raises(RoehubError) as exc_info:
        use_case.trades(
            user_id=foreign_id,
            job_id=job.job_id,
            variant_key=str(row.payload_json["public_variant_key"]),
        )

    assert exc_info.value.code == "backtest.forbidden"
    assert repository.public_variant_lookups == ()


def test_create_persists_queued_background_job_without_inline_execution() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000305")
    seed_job, _row = _job_and_row(user_id=user_id)
    repository = _Repository(job=seed_job, top_rows=())
    trigger = _Trigger()
    use_case = _use_case(repository=repository, execution_trigger=trigger)

    result = use_case.create(
        user_id=user_id,
        payload=_request(),
        idempotency_key="stable-key",
    )

    assert result.job.state == "queued"
    assert result.job.progress.pipeline_stage == "queued"
    assert repository.job.state == "queued"
    assert repository.job.execution_mode == "background_auto"
    assert repository.top_rows == ()
    assert trigger.job_ids == (repository.job.job_id,)


def test_create_idempotency_replay_does_not_enqueue_duplicate_work() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000306")
    seed_job, _row = _job_and_row(user_id=user_id)
    repository = _Repository(job=seed_job, top_rows=())
    trigger = _Trigger()
    use_case = _use_case(repository=repository, execution_trigger=trigger)

    first = use_case.create(user_id=user_id, payload=_request(), idempotency_key="stable-key")
    replay = use_case.create(user_id=user_id, payload=_request(), idempotency_key="stable-key")

    assert first.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert replay.job.job_id == first.job.job_id
    assert trigger.job_ids == (repository.job.job_id,)


def test_cancel_is_deterministic_for_queued_running_and_terminal_jobs() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000307")
    seed_job, _row = _job_and_row(user_id=user_id)
    repository = _Repository(job=seed_job, top_rows=())
    use_case = _use_case(repository=repository)
    created = use_case.create(user_id=user_id, payload=_request(), idempotency_key=None)

    queued_cancel = use_case.cancel(user_id=user_id, job_id=UUID(created.job.job_id))
    terminal_cancel = use_case.cancel(user_id=user_id, job_id=UUID(created.job.job_id))

    assert queued_cancel.state == "cancelled"
    assert terminal_cancel.state == "cancelled"

    running_job, _row = _job_and_row(user_id=user_id)
    repository.job = running_job.claim(
        changed_at=running_job.created_at + timedelta(seconds=1),
        locked_by="test-worker",
        lease_expires_at=running_job.created_at + timedelta(seconds=901),
    )

    running_cancel = use_case.cancel(user_id=user_id, job_id=repository.job.job_id)

    assert running_cancel.state == "running"
    assert repository.job.cancel_requested_at is not None


def test_worker_claims_updates_and_finishes_queued_job() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000308")
    seed_job, _row = _job_and_row(user_id=user_id)
    repository = _Repository(job=seed_job, top_rows=())
    use_case = _use_case(repository=repository)
    created = use_case.create(user_id=user_id, payload=_request(), idempotency_key=None)
    worker = BacktestJobWorkerUseCase(
        job_repository=repository,
        lease_repository=repository,
        executor=_WorkerExecutor(),
        lease_seconds=900,
    )

    result = worker.run_next(locked_by="test-worker")

    assert result.claimed_job_id == UUID(created.job.job_id)
    assert result.state == "succeeded"
    assert result.executed is True
    assert result.lease_lost is False
    assert repository.job.state == "succeeded"
    assert repository.progress_updates == (("stage_a", 0, 1),)
    assert len(repository.top_rows) == 1


def _use_case(
    *,
    repository: "_Repository",
    execution_trigger: Any | None = None,
    lazy_service: Any | None = None,
) -> BacktestJobsUseCase:
    runtime_config = _runtime_config()
    return BacktestJobsUseCase(
        job_repository=repository,
        preflight_service=cast(Any, _PreflightService()),
        runtime_config=runtime_config,
        execution_trigger=execution_trigger,
        lazy_trades_service=cast(Any, lazy_service or _LazyService()),
    )


def _runtime_config() -> BacktestRuntimeConfig:
    return BacktestRuntimeConfig(
        hit_times_tp_levels_pct=(2.0,),
        hit_times_sl_levels_pct=(1.0,),
        artifact_config_hash="e" * 64,
    )


@dataclass
class _PreflightService:
    def execute(self, payload: dict[str, Any]) -> BacktestPreflightResult:
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


@dataclass
class _Trigger:
    job_ids: tuple[UUID, ...] = ()

    def enqueue(self, *, job: BacktestJob) -> None:
        self.job_ids = (*self.job_ids, job.job_id)


@dataclass
class _WorkerExecutor:
    def execute(
        self,
        *,
        job_id: UUID,
        preflight: Any,
        updated_at: datetime,
    ) -> BacktestJobExecutionResult:
        top_result = BacktestNoRiskTopResult(
            rank=1,
            score=12.5,
            indicator_rows={"ma.dema": 7},
            metrics={"total_return_pct": 12.5, "trade_count": 2.0},
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
class _LazyService:
    requests: tuple[tuple[str, str], ...] = ()

    def execute(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> BacktestLazyTradesDetailReadModel:
        self.requests = (*self.requests, (public_variant_key, row.variant_key))
        return BacktestLazyTradesDetailReadModel(
            job_id=str(job.job_id),
            variant_key=public_variant_key,
            variant_hash=row.variant_key,
            request_hash=job.request_hash,
            engine_params_hash=job.engine_params_hash,
            artifact_manifest_hash=str(job.request_json["artifact_metadata"]["artifact_manifest_hash"]),
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=dict(row.payload_json["canonical_variant_params"]),
            readable_params=dict(row.payload_json["readable_params"]),
            trades=(),
            chart_overlay={"schema": "backtest_chart_overlay_v1", "markers": [], "segments": []},
            cache={"status": "miss"},
            timing={"lazy_trades_compute": 0.0},
        )


@dataclass
class _Repository:
    job: BacktestJob
    top_rows: tuple[BacktestJobTopVariant, ...]
    public_variant_lookups: tuple[str, ...] = ()
    progress_updates: tuple[tuple[str, int, int], ...] = ()

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        if self.job.job_id != job_id:
            return None
        if user_id is not None and self.job.user_id != user_id:
            return None
        return self.job

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        self.public_variant_lookups = (*self.public_variant_lookups, public_variant_key)
        if self.job.job_id != job_id:
            return None
        for row in self.top_rows:
            if row.payload_json.get("public_variant_key") == public_variant_key:
                return row
        return None

    def list_top_variants(self, *, job_id: UUID) -> tuple[BacktestJobTopVariant, ...]:
        return self.top_rows if self.job.job_id == job_id else ()

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        return BacktestJobListPage(items=(self.job,), next_cursor=None)

    def create(self, *, job: BacktestJob) -> BacktestJob:
        self.job = job
        self.top_rows = ()
        return job

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        idempotency = dict(self.job.request_json).get("idempotency")
        if (
            self.job.user_id == user_id
            and self.job.created_at >= created_after
            and isinstance(idempotency, Mapping)
            and idempotency.get("key_hash") == idempotency_key_hash
        ):
            return self.job
        return None

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        if self.job.state != "queued":
            return None
        claimed = self.job.claim(
            changed_at=now,
            locked_by=locked_by,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self.job = claimed
        return claimed

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: str,
        processed_units: int,
        total_units: int,
    ) -> BacktestJob | None:
        if self.job.job_id != job_id or self.job.locked_by != locked_by:
            return None
        progressed = self.job.update_progress(
            changed_at=now,
            stage=cast(Any, stage),
            processed_units=processed_units,
            total_units=total_units,
        )
        self.job = progressed
        self.progress_updates = (
            *self.progress_updates,
            (stage, processed_units, total_units),
        )
        return progressed

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        if self.job.job_id != job_id or self.job.locked_by != locked_by:
            return None
        renewed = self.job.renew_lease(
            changed_at=now,
            locked_by=locked_by,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self.job = renewed
        return renewed

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
        if self.job.job_id != job_id or self.job.locked_by != locked_by:
            return None
        finished = self.job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.job = finished
        return finished

    def claim_for_inline_execution(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob | None:
        _ = job_id, user_id, now, locked_by, lease_expires_at
        return None

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
        if self.job.job_id != job_id or self.job.user_id != user_id:
            return None
        if self.job.state != "running" or self.job.locked_by != locked_by:
            return None
        finished = self.job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.job = finished
        self.top_rows = top_variants if next_state == "succeeded" else ()
        return finished

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = top_variants, stage_a_shortlist
        return job

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        if self.job.job_id != job_id or self.job.user_id != user_id:
            return None
        self.job = self.job.request_cancel(changed_at=cancel_requested_at)
        return self.job

    def count_active_for_user(self, *, user_id: UserId) -> int:
        _ = user_id
        return 0

    def count_active_global(self) -> int:
        return 0

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


def _job_and_row(*, user_id: UserId) -> tuple[BacktestJob, BacktestJobTopVariant]:
    job_id = uuid4()
    created_at = datetime.now(UTC) - timedelta(seconds=1)
    request = _request()
    metadata = _artifact_metadata()
    request["artifact_metadata"] = metadata.as_mapping()
    job = BacktestJob.create_queued(
        job_id=job_id,
        user_id=user_id,
        mode="template",
        created_at=created_at,
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
    top_result = BacktestNoRiskTopResult(
        rank=1,
        score=12.5,
        indicator_rows={"ma.dema": 7},
        metrics={"total_return_pct": 12.5, "trade_count": 2.0},
        metadata={"ma.dema.source": "close", "ma.dema.window": 5},
    )
    row = BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=request,
        top_results=(top_result,),
        updated_at=created_at,
    ).top_variants[0]
    return job, row


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
