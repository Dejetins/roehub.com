from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestCostEstimate,
    BacktestLazyTradesDetailReadModel,
    BacktestLazyTradesMaterializationReadModel,
    BacktestNoRiskTopResult,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestLazyTradesMaterializationRequest,
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestAdmissionService,
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import PaidLevel, UserId


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


def test_trades_cache_miss_enqueues_materialization_without_lazy_compute() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000308")
    job, row = _job_and_row(user_id=user_id)
    repository = _Repository(job=job, top_rows=(row,))
    materializations = _MaterializationRepository()
    lazy_service = _LazyService(cache_hit=False)
    use_case = _use_case(
        repository=repository,
        lazy_service=lazy_service,
        materialization_repository=materializations,
    )

    result = use_case.trades(
        user_id=user_id,
        job_id=job.job_id,
        variant_key=str(row.payload_json["public_variant_key"]),
    )

    assert isinstance(result, BacktestLazyTradesMaterializationReadModel)
    assert result.status == "queued"
    assert result.variant_key == row.payload_json["public_variant_key"]
    assert result.variant_hash == row.variant_key
    assert result.materialization["retry_after_seconds"] == 2
    assert result.materialization["correlation_id"] == result.materialization["task_id"]
    assert result.cache["status"] == "miss"
    assert result.pagination == {"mode": "none"}
    assert lazy_service.execute_calls == 0
    assert len(materializations.tasks) == 1


def test_trades_cache_miss_materialization_is_idempotent_by_request_identity() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000309")
    job, row = _job_and_row(user_id=user_id)
    materializations = _MaterializationRepository()
    use_case = _use_case(
        repository=_Repository(job=job, top_rows=(row,)),
        lazy_service=_LazyService(cache_hit=False),
        materialization_repository=materializations,
    )
    public_key = str(row.payload_json["public_variant_key"])

    first = use_case.trades(user_id=user_id, job_id=job.job_id, variant_key=public_key)
    replay = use_case.trades(user_id=user_id, job_id=job.job_id, variant_key=public_key)

    assert isinstance(first, BacktestLazyTradesMaterializationReadModel)
    assert isinstance(replay, BacktestLazyTradesMaterializationReadModel)
    assert replay.materialization["task_id"] == first.materialization["task_id"]
    assert len(materializations.tasks) == 1


def test_trades_does_not_resolve_raw_storage_sha_as_public_key() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000302")
    job, row = _job_and_row(user_id=user_id)
    use_case = _use_case(repository=_Repository(job=job, top_rows=(row,)))

    with pytest.raises(RoehubError) as exc_info:
        use_case.trades(user_id=user_id, job_id=job.job_id, variant_key=row.variant_key)

    assert exc_info.value.code == "backtest.not_found"


def test_trades_rejects_oversized_public_variant_key_before_lookup() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000310")
    job, row = _job_and_row(user_id=user_id)
    repository = _Repository(job=job, top_rows=(row,))
    use_case = _use_case(repository=repository)

    with pytest.raises(RoehubError) as exc_info:
        use_case.trades(user_id=user_id, job_id=job.job_id, variant_key="x" * 257)

    assert exc_info.value.code == "backtest.invalid_request"
    assert repository.public_variant_lookups == ()


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


def test_create_persists_queued_background_job_and_triggers_execution() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000305")
    repository = _CreateRepository()
    trigger = _Trigger()
    use_case = _create_use_case(repository=repository, trigger=trigger)

    result = use_case.create(user_id=user_id, payload=_request(), idempotency_key="stable-key")

    assert result.job.state == "queued"
    assert result.job.progress.pipeline_stage == "queued"
    assert result.job.refresh_status == "poll"
    assert result.job.retry_after_seconds == 2
    assert result.idempotent_replay is False
    assert len(repository.jobs) == 1
    stored = next(iter(repository.jobs.values()))
    assert stored.execution_mode == "background_auto"
    assert stored.request_hash == "d" * 64
    assert trigger.calls == ((stored.job_id, stored.request_hash),)


def test_create_idempotency_replay_does_not_enqueue_duplicate_work() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000306")
    repository = _CreateRepository()
    trigger = _Trigger()
    use_case = _create_use_case(repository=repository, trigger=trigger)

    first = use_case.create(user_id=user_id, payload=_request(), idempotency_key="stable-key")
    replay = use_case.create(user_id=user_id, payload=_request(), idempotency_key="stable-key")
    changed = _request()
    changed["top_n"] = 50

    with pytest.raises(RoehubError) as exc_info:
        use_case.create(user_id=user_id, payload=changed, idempotency_key="stable-key")

    assert replay.idempotent_replay is True
    assert replay.job.job_id == first.job.job_id
    assert len(repository.jobs) == 1
    assert trigger.calls == ((next(iter(repository.jobs.values())).job_id, "d" * 64),)
    assert exc_info.value.code == "backtest.idempotency_key_conflict"


def test_create_idempotency_replay_does_not_consume_quota_slot() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000311")
    repository = _CreateRepository()
    trigger = _Trigger()
    use_case = _create_use_case(
        repository=repository,
        trigger=trigger,
        admission_service=BacktestAdmissionService(),
    )
    request = _request()
    request["top_n"] = 20

    first = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        payload=request,
        idempotency_key="stable-key",
    )
    replay = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        payload=request,
        idempotency_key="stable-key",
    )

    assert replay.idempotent_replay is True
    assert replay.job.job_id == first.job.job_id
    assert len(repository.jobs) == 1
    assert trigger.calls == ((next(iter(repository.jobs.values())).job_id, "d" * 64),)


def test_create_rejects_free_request_above_tier_top_n() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000312")
    use_case = _create_use_case(
        repository=_CreateRepository(),
        trigger=_Trigger(),
        admission_service=BacktestAdmissionService(),
    )

    with pytest.raises(RoehubError) as exc_info:
        use_case.create(
            user_id=user_id,
            paid_level=PaidLevel.free(),
            payload=_request(),
            idempotency_key=None,
        )

    assert exc_info.value.code == "backtest.request_too_expensive"
    details = exc_info.value.details
    assert isinstance(details, dict)
    assert details["paid_level"] == "free"
    assert details["limit_scope"] == "full_jobs.top_n"


def test_create_rejects_free_active_quota_with_retry_details() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000313")
    repository = _CreateRepository()
    trigger = _Trigger()
    use_case = _create_use_case(
        repository=repository,
        trigger=trigger,
        admission_service=BacktestAdmissionService(),
    )
    request = _request()
    request["top_n"] = 20
    for index in range(2):
        use_case.create(
            user_id=user_id,
            paid_level=PaidLevel.free(),
            payload=request,
            idempotency_key=f"key-{index}",
        )

    with pytest.raises(RoehubError) as exc_info:
        use_case.create(
            user_id=user_id,
            paid_level=PaidLevel.free(),
            payload=request,
            idempotency_key="key-over-limit",
        )

    assert exc_info.value.code == "backtest.rate_limited"
    details = exc_info.value.details
    assert isinstance(details, dict)
    assert details["paid_level"] == "free"
    assert details["limit_scope"] == "full_jobs.active"
    assert details["retry_after_seconds"] == 60


def test_lazy_detail_quota_replay_does_not_consume_new_slot() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000314")
    job, row = _job_and_row(user_id=user_id)
    materializations = _MaterializationRepository()
    use_case = _use_case(
        repository=_Repository(job=job, top_rows=(row,)),
        lazy_service=_LazyService(cache_hit=False),
        materialization_repository=materializations,
        admission_service=BacktestAdmissionService(),
    )
    public_key = str(row.payload_json["public_variant_key"])

    first = use_case.trades(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        job_id=job.job_id,
        variant_key=public_key,
    )
    replay = use_case.trades(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        job_id=job.job_id,
        variant_key=public_key,
    )

    assert isinstance(first, BacktestLazyTradesMaterializationReadModel)
    assert isinstance(replay, BacktestLazyTradesMaterializationReadModel)
    assert replay.materialization["task_id"] == first.materialization["task_id"]
    assert len(materializations.tasks) == 1


def test_cancel_is_deterministic_for_queued_running_and_terminal_jobs() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000307")
    queued, _ = _job_and_row(user_id=user_id)
    repository = _Repository(job=queued, top_rows=())
    use_case = _use_case(repository=repository)

    cancelled = use_case.cancel(user_id=user_id, job_id=queued.job_id)

    assert cancelled.state == "cancelled"
    assert repository.job.state == "cancelled"
    terminal = use_case.cancel(user_id=user_id, job_id=queued.job_id)
    assert terminal.state == "cancelled"

    running = queued.claim(
        changed_at=datetime.now(UTC),
        locked_by="test-worker",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    running_repository = _Repository(job=running, top_rows=())
    running_use_case = _use_case(repository=running_repository)

    still_running = running_use_case.cancel(user_id=user_id, job_id=running.job_id)

    assert still_running.state == "running"
    assert running_repository.job.cancel_requested_at is not None


def _use_case(
    *,
    repository: "_Repository",
    lazy_service: Any | None = None,
    materialization_repository: Any | None = None,
    admission_service: BacktestAdmissionService | None = None,
) -> BacktestJobsUseCase:
    runtime_config = _runtime_config()
    return BacktestJobsUseCase(
        job_repository=repository,
        preflight_service=BacktestPreflightService(
            defaults_provider=None,  # type: ignore[arg-type]
            artifact_context_resolver=None,  # type: ignore[arg-type]
            runtime_config=runtime_config,
        ),
        runtime_config=runtime_config,
        lazy_trades_service=cast(Any, lazy_service or _LazyService()),
        lazy_trades_materialization_repository=materialization_repository,
        admission_service=admission_service,
    )


def _create_use_case(
    *,
    repository: "_CreateRepository",
    trigger: "_Trigger",
    admission_service: BacktestAdmissionService | None = None,
) -> BacktestJobsUseCase:
    runtime_config = _runtime_config()
    return BacktestJobsUseCase(
        job_repository=cast(Any, repository),
        preflight_service=cast(BacktestPreflightService, _PreflightService()),
        runtime_config=runtime_config,
        execution_trigger=trigger,
        admission_service=admission_service,
    )


def _runtime_config() -> BacktestRuntimeConfig:
    return BacktestRuntimeConfig(
        hit_times_tp_levels_pct=(2.0,),
        hit_times_sl_levels_pct=(1.0,),
        artifact_config_hash="e" * 64,
    )


@dataclass
class _LazyService:
    cache_hit: bool = True
    requests: tuple[tuple[str, str], ...] = ()
    execute_calls: int = 0

    def read_cached(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> "_Probe":
        self.requests = (*self.requests, (public_variant_key, row.variant_key))
        detail = BacktestLazyTradesDetailReadModel(
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
            cache={"status": "hit" if self.cache_hit else "miss"},
            timing={"lazy_trades_cache_hit": 0.0} if self.cache_hit else {},
        )
        return _Probe(
            detail=detail if self.cache_hit else None,
            cache_status="hit" if self.cache_hit else "miss",
        )

    def execute(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> BacktestLazyTradesDetailReadModel:
        self.execute_calls += 1
        return self.read_cached(
            job=job,
            row=row,
            public_variant_key=public_variant_key,
        ).detail  # type: ignore[return-value]


@dataclass(frozen=True)
class _CacheKey:
    engine_params_hash: str = "e" * 64
    artifact_manifest_hash: str = "a" * 64
    digest: str = "f" * 64


@dataclass(frozen=True)
class _Probe:
    detail: BacktestLazyTradesDetailReadModel | None
    cache_status: str
    cache_key: _CacheKey = _CacheKey()
    cache_warning: str | None = None
    ttl_seconds: int = 172_800
    cache_lookup_s: float = 0.0


@dataclass
class _MaterializationRepository:
    tasks: dict[tuple[str, UUID, str, str], BacktestLazyTradesMaterializationTask] = field(
        default_factory=dict
    )

    def request_materialization(
        self,
        *,
        request: BacktestLazyTradesMaterializationRequest,
    ) -> BacktestLazyTradesMaterializationTask:
        key = (
            str(request.owner_user_id),
            request.job_id,
            request.public_variant_key,
            request.cache_key,
        )
        existing = self.tasks.get(key)
        if existing is not None:
            return existing
        task = BacktestLazyTradesMaterializationTask(
            task_id=uuid4(),
            owner_user_id=request.owner_user_id,
            job_id=request.job_id,
            public_variant_key=request.public_variant_key,
            variant_hash=request.variant_hash,
            request_hash=request.request_hash,
            engine_params_hash=request.engine_params_hash,
            artifact_manifest_hash=request.artifact_manifest_hash,
            cache_key=request.cache_key,
            status="queued",
            priority_class=request.priority_class,
            created_at=request.requested_at,
            updated_at=request.requested_at,
            started_at=None,
            finished_at=None,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            attempt=0,
            last_error=None,
            last_error_json=None,
            cache_status=request.cache_status,
            cache_path=None,
            ttl_seconds=request.ttl_seconds,
        )
        self.tasks[key] = task
        return task

    def find_by_identity(
        self,
        *,
        owner_user_id: UserId,
        job_id: UUID,
        public_variant_key: str,
        cache_key: str,
    ) -> BacktestLazyTradesMaterializationTask | None:
        return self.tasks.get((str(owner_user_id), job_id, public_variant_key, cache_key))

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        return sum(
            1
            for task in self.tasks.values()
            if task.owner_user_id == owner_user_id and task.status in {"queued", "running"}
        )

    def count_created_for_user_since(
        self,
        *,
        owner_user_id: UserId,
        created_after: datetime,
    ) -> int:
        return sum(
            1
            for task in self.tasks.values()
            if task.owner_user_id == owner_user_id and task.created_at >= created_after
        )

    def count_active_global(self) -> int:
        return sum(1 for task in self.tasks.values() if task.status in {"queued", "running"})


@dataclass
class _Repository:
    job: BacktestJob
    top_rows: tuple[BacktestJobTopVariant, ...]
    public_variant_lookups: tuple[str, ...] = ()

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

    def list_top_variants(
        self,
        *,
        job_id: UUID,
        limit: int | None = None,
    ) -> tuple[BacktestJobTopVariant, ...]:
        rows = self.top_rows if self.job.job_id == job_id else ()
        return rows if limit is None else rows[:limit]

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        return BacktestJobListPage(items=(self.job,), next_cursor=None)

    def create(self, *, job: BacktestJob) -> BacktestJob:
        return job

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        _ = user_id, idempotency_key_hash, created_after
        return None

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
        _ = (
            job_id,
            user_id,
            now,
            locked_by,
            next_state,
            top_variants,
            last_error,
            last_error_json,
        )
        return None

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

    def delete_terminal(self, *, job_id: UUID, user_id: UserId) -> bool:
        return (
            self.job.job_id == job_id
            and self.job.user_id == user_id
            and self.job.state in {"succeeded", "failed", "cancelled"}
        )

    def count_active_for_user(self, *, user_id: UserId) -> int:
        _ = user_id
        return 0

    def count_created_for_user_since(
        self,
        *,
        user_id: UserId,
        created_after: datetime,
    ) -> int:
        _ = user_id, created_after
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


@dataclass
class _CreateRepository:
    jobs: dict[UUID, BacktestJob] = field(default_factory=dict)
    top_rows: dict[UUID, tuple[BacktestJobTopVariant, ...]] = field(default_factory=dict)

    def create(self, *, job: BacktestJob) -> BacktestJob:
        self.jobs[job.job_id] = job
        return job

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        for job in self.jobs.values():
            idempotency = dict(job.request_json).get("idempotency")
            if (
                job.user_id == user_id
                and job.created_at >= created_after
                and isinstance(idempotency, dict)
                and idempotency.get("key_hash") == idempotency_key_hash
            ):
                return job
        return None

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_top_variants(
        self,
        *,
        job_id: UUID,
        limit: int | None = None,
    ) -> tuple[BacktestJobTopVariant, ...]:
        rows = self.top_rows.get(job_id, ())
        return rows if limit is None else rows[:limit]

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        items = tuple(job for job in self.jobs.values() if job.user_id == query.user_id)
        return BacktestJobListPage(items=items, next_cursor=None)

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        _ = job_id, public_variant_key
        return None

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        cancelled = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs[job_id] = cancelled
        return cancelled

    def delete_terminal(self, *, job_id: UUID, user_id: UserId) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id or job.state not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return False
        del self.jobs[job_id]
        self.top_rows.pop(job_id, None)
        return True

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = top_variants, stage_a_shortlist
        return self.create(job=job)

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
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        finished = job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.jobs[job_id] = finished
        self.top_rows[job_id] = top_variants
        return finished

    def count_active_for_user(self, *, user_id: UserId) -> int:
        return sum(1 for job in self.jobs.values() if job.user_id == user_id and job.is_active())

    def count_created_for_user_since(
        self,
        *,
        user_id: UserId,
        created_after: datetime,
    ) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.user_id == user_id and job.created_at >= created_after
        )

    def count_active_global(self) -> int:
        return sum(1 for job in self.jobs.values() if job.is_active())

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


@dataclass
class _Trigger:
    calls: tuple[tuple[UUID, str], ...] = ()

    def enqueue(self, *, job_id: UUID, user_id: UserId, request_hash: str) -> None:
        _ = user_id
        self.calls = (*self.calls, (job_id, request_hash))


class _PreflightService:
    def execute(self, payload: Any) -> BacktestPreflightResult:
        request = dict(payload)
        request.setdefault("top_n", 100)
        request_hash = "c" * 64 if request.get("top_n") == 50 else "d" * 64
        return BacktestPreflightResult(
            normalized_request=request,
            request_hash=request_hash,
            result_config_hash="e" * 64,
            artifact_metadata=_artifact_metadata(),
            cost_estimate=BacktestCostEstimate(
                indicator_rows=1,
                candidate_combinations=1,
                tp_sl_cells=0,
                cost_class="small",
            ),
        )


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
        execution_mode="sync_inline",
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
