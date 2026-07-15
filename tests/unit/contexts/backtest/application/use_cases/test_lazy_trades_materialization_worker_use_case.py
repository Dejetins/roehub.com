from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import BacktestLazyTradesDetailReadModel
from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestLazyTradesMaterializationExecutionResult,
    BacktestLazyTradesMaterializationWorkerUseCase,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


def test_lazy_trades_worker_claims_executes_and_finishes_task() -> None:
    task = _task()
    repository = _MaterializationRepository(task=task)
    jobs = _JobRepository(
        organization_id=task.organization_id,
        owner_user_id=task.owner_user_id,
        job_id=task.job_id,
    )
    executor = _LazyTradesChildExecutor()
    use_case = BacktestLazyTradesMaterializationWorkerUseCase(
        materialization_repository=cast(Any, repository),
        job_repository=cast(Any, jobs),
        lazy_trades_service=cast(Any, _LazyTradesService()),
        lease_seconds=60,
        locked_by="test-runner",
        executor=executor,
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.lease_lost is False
    assert result.task is not None
    assert result.task.status == "completed"
    assert result.cache_status == "miss"
    assert repository.finished_completed == (task.task_id,)
    assert executor.calls == (task.task_id,)
    assert jobs.owner_scoped_gets == ()


def test_lazy_trades_worker_marks_task_failed_when_variant_missing() -> None:
    task = _task()
    repository = _MaterializationRepository(task=task)
    jobs = _JobRepository(
        organization_id=task.organization_id,
        owner_user_id=task.owner_user_id,
        job_id=task.job_id,
        row=None,
    )
    use_case = BacktestLazyTradesMaterializationWorkerUseCase(
        materialization_repository=cast(Any, repository),
        job_repository=cast(Any, jobs),
        lazy_trades_service=cast(Any, _LazyTradesService()),
        lease_seconds=60,
        locked_by="test-runner",
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.task is not None
    assert result.task.status == "failed"
    assert repository.last_error_json is not None
    assert repository.last_error_json["code"] == "backtest.variant_not_found"
    assert repository.last_error_json["details"]["retryable"] is False


def test_lazy_trades_worker_maps_child_failure_to_failed_task() -> None:
    task = _task()
    repository = _MaterializationRepository(task=task)
    executor = _LazyTradesChildExecutor(raise_error=RuntimeError("child failed"))
    use_case = BacktestLazyTradesMaterializationWorkerUseCase(
        materialization_repository=cast(Any, repository),
        job_repository=cast(
            Any,
            _JobRepository(
                organization_id=task.organization_id,
                owner_user_id=task.owner_user_id,
                job_id=task.job_id,
            ),
        ),
        lazy_trades_service=cast(Any, _LazyTradesService()),
        lease_seconds=60,
        locked_by="test-runner",
        executor=executor,
    )

    result = use_case.run_next()

    assert result.claimed is True
    assert result.task is not None
    assert result.task.status == "failed"
    assert repository.last_error_json is not None
    assert repository.last_error_json["code"] == "unexpected_error"
    assert repository.last_error_json["details"]["retryable"] is True


@dataclass
class _MaterializationRepository:
    task: BacktestLazyTradesMaterializationTask
    finished_completed: tuple[UUID, ...] = ()
    last_error_json: dict[str, Any] | None = None

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestLazyTradesMaterializationTask | None:
        _ = lease_seconds
        if self.task.status != "queued":
            return None
        self.task = _replace_task(
            self.task,
            status="running",
            started_at=now,
            locked_by=locked_by,
            locked_at=now,
            lease_expires_at=now + timedelta(seconds=60),
            heartbeat_at=now,
            attempt=self.task.attempt + 1,
        )
        return self.task

    def heartbeat(
        self,
        *,
        task_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestLazyTradesMaterializationTask | None:
        _ = task_id, now, locked_by, lease_seconds
        return self.task

    def finish_completed(
        self,
        *,
        task_id: UUID,
        owner_user_id: UserId,
        now: datetime,
        locked_by: str,
        cache_status: str,
        cache_path: str | None,
    ) -> BacktestLazyTradesMaterializationTask | None:
        _ = owner_user_id, locked_by, cache_path
        self.finished_completed = (*self.finished_completed, task_id)
        self.task = _replace_task(
            self.task,
            status="completed",
            updated_at=now,
            finished_at=now,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            cache_status=cache_status,
        )
        return self.task

    def finish_failed(
        self,
        *,
        task_id: UUID,
        owner_user_id: UserId,
        now: datetime,
        locked_by: str,
        last_error: str,
        last_error_json: Mapping[str, Any],
    ) -> BacktestLazyTradesMaterializationTask | None:
        _ = task_id, owner_user_id, locked_by
        self.last_error_json = dict(last_error_json)
        self.task = _replace_task(
            self.task,
            status="failed",
            updated_at=now,
            finished_at=now,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        return self.task


@dataclass
class _JobRepository:
    organization_id: OrganizationId
    owner_user_id: UserId
    job_id: UUID
    row: Any = field(default_factory=lambda: SimpleNamespace(variant_key="b" * 64))
    owner_scoped_gets: tuple[tuple[UUID, OrganizationId, UserId | None], ...] = ()

    def get(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId | None = None,
    ) -> Any:
        self.owner_scoped_gets = (
            *self.owner_scoped_gets,
            (job_id, organization_id, user_id),
        )
        if (
            job_id != self.job_id
            or organization_id != self.organization_id
            or user_id != self.owner_user_id
        ):
            return None
        return SimpleNamespace(
            job_id=job_id,
            organization_id=organization_id,
            user_id=user_id,
        )

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        public_variant_key: str,
    ) -> Any:
        _ = public_variant_key
        if job_id != self.job_id or organization_id != self.organization_id:
            return None
        return self.row


@dataclass
class _LazyTradesService:
    calls: tuple[tuple[UUID, str], ...] = ()

    def execute(self, *, job: Any, row: Any, public_variant_key: str) -> Any:
        _ = row
        self.calls = (*self.calls, (job.job_id, public_variant_key))
        return BacktestLazyTradesDetailReadModel(
            job_id=str(job.job_id),
            variant_key=public_variant_key,
            variant_hash="b" * 64,
            request_hash="c" * 64,
            engine_params_hash="d" * 64,
            artifact_manifest_hash="e" * 64,
            summary_metrics={},
            canonical_variant_params={},
            readable_params={},
            trades=(),
            chart_overlay={},
            cache={"status": "miss"},
            timing={"lazy_trades_compute": 0.001},
        )


@dataclass
class _LazyTradesChildExecutor:
    calls: tuple[UUID, ...] = ()
    raise_error: Exception | None = None

    def execute(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
    ) -> BacktestLazyTradesMaterializationExecutionResult:
        self.calls = (*self.calls, task.task_id)
        if self.raise_error is not None:
            raise self.raise_error
        return BacktestLazyTradesMaterializationExecutionResult(
            cache_status="miss",
            cache_path="/tmp/trades-cache",
        )


def _task() -> BacktestLazyTradesMaterializationTask:
    now = datetime.now(UTC)
    return BacktestLazyTradesMaterializationTask(
        task_id=uuid4(),
        organization_id=OrganizationId.from_string(
            "00000000-0000-0000-0000-000000000001"
        ),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000401"),
        job_id=uuid4(),
        public_variant_key="ma.dema:source=close,window=5",
        variant_hash="b" * 64,
        request_hash="c" * 64,
        engine_params_hash="d" * 64,
        artifact_manifest_hash="e" * 64,
        cache_key="f" * 64,
        status="queued",
        priority_class="interactive",
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=5),
        started_at=None,
        finished_at=None,
        locked_by=None,
        locked_at=None,
        lease_expires_at=None,
        heartbeat_at=None,
        attempt=0,
        last_error=None,
        last_error_json=None,
        cache_status="miss",
        cache_path=None,
        ttl_seconds=1209600,
    )


def _replace_task(
    task: BacktestLazyTradesMaterializationTask,
    **overrides: Any,
) -> BacktestLazyTradesMaterializationTask:
    payload = {
        "task_id": task.task_id,
        "organization_id": task.organization_id,
        "owner_user_id": task.owner_user_id,
        "job_id": task.job_id,
        "public_variant_key": task.public_variant_key,
        "variant_hash": task.variant_hash,
        "request_hash": task.request_hash,
        "engine_params_hash": task.engine_params_hash,
        "artifact_manifest_hash": task.artifact_manifest_hash,
        "cache_key": task.cache_key,
        "status": task.status,
        "priority_class": task.priority_class,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "locked_by": task.locked_by,
        "locked_at": task.locked_at,
        "lease_expires_at": task.lease_expires_at,
        "heartbeat_at": task.heartbeat_at,
        "attempt": task.attempt,
        "last_error": task.last_error,
        "last_error_json": task.last_error_json,
        "cache_status": task.cache_status,
        "cache_path": task.cache_path,
        "ttl_seconds": task.ttl_seconds,
    }
    payload.update(overrides)
    return BacktestLazyTradesMaterializationTask(**payload)
