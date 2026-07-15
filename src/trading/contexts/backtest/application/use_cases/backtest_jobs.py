from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Mapping, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BacktestJobCreateResult,
    BacktestJobListResult,
    BacktestJobReadModel,
    BacktestJobTopResult,
    BacktestJobTopVariantReadModel,
    BacktestLazyTradesMaterializationReadModel,
    BacktestLazyTradesResultReadModel,
    BacktestPreflightResult,
    build_backtest_job_read_model,
    build_top_variant_read_model,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobExecutionTrigger,
    BacktestJobListQuery,
    BacktestJobRepository,
    BacktestLazyTradesMaterializationRepository,
    BacktestLazyTradesMaterializationRequest,
    BacktestLazyTradesMaterializationTask,
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.contexts.backtest.application.services.research_identity import (
    build_research_idempotency_key_hash,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestAdmissionService,
    BacktestFullJobQuotaSnapshot,
    BacktestLazyDetailQuotaSnapshot,
    BacktestLazyTradesDetailService,
    BacktestPaginatedTradesReadModel,
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestResultSeriesKind,
    BacktestResultSeriesReadModel,
    BacktestResultStatsReadModel,
    BacktestResultSummaryReadModel,
    BacktestRuntimeConfig,
    BacktestTradesCsvReadModel,
    build_result_summary_read_model,
    normalize_csv_max_rows,
    symbol_from_job_request,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
    SCHEDULING_METADATA_KEY,
    scheduling_metadata_from_preflight,
)
from trading.contexts.backtest.domain.entities import (
    BacktestArtifactSlotLiteral,
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

BACKTEST_ERROR_AUTH_REQUIRED = "auth.required"
BACKTEST_ERROR_FORBIDDEN = "backtest.forbidden"
BACKTEST_ERROR_NOT_FOUND = "backtest.not_found"
BACKTEST_ERROR_IDEMPOTENCY_CONFLICT = "backtest.idempotency_key_conflict"
BACKTEST_ERROR_INVALID_REQUEST = "backtest.invalid_request"
BACKTEST_ERROR_JOB_NOT_CANCELLABLE = "backtest.job_not_cancellable"
BACKTEST_ERROR_JOB_NOT_DELETABLE = "backtest.job_not_deletable"
BACKTEST_ERROR_RATE_LIMITED = "backtest.rate_limited"
BACKTEST_ERROR_QUEUE_SATURATED = "backtest.queue_saturated"


@dataclass(frozen=True, slots=True)
class BacktestJobsUseCase:
    job_repository: BacktestJobRepository
    preflight_service: BacktestPreflightService
    runtime_config: BacktestRuntimeConfig
    organization_scope_resolver: ResearchOrganizationScopeResolver
    execution_trigger: BacktestJobExecutionTrigger | None = None
    lazy_trades_service: BacktestLazyTradesDetailService | None = None
    lazy_trades_materialization_repository: BacktestLazyTradesMaterializationRepository | None = (
        None
    )
    admission_service: BacktestAdmissionService | None = None
    idempotency_ttl_seconds: int = 86_400
    light_max_estimated_combinations: int = DEFAULT_LIGHT_ESTIMATED_COMBINATIONS

    def preflight(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel,
        payload: Mapping[str, Any],
    ) -> BacktestPreflightResult:
        scope = self._scope(user_id=user_id)
        preflight = self._preflight(payload=payload, paid_level=paid_level)
        self._enforce_full_job_admission(
            organization_id=scope.organization_id,
            user_id=user_id,
            paid_level=paid_level,
            preflight=preflight,
            now=datetime.now(UTC),
        )
        return preflight

    def create(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        payload: Mapping[str, Any],
        idempotency_key: str | None,
    ) -> BacktestJobCreateResult:
        scope = self._scope(user_id=user_id)
        preflight = self._preflight(payload=payload, paid_level=paid_level)
        key_hash = _idempotency_key_hash(
            organization_id=scope.organization_id,
            idempotency_key=idempotency_key,
        )
        now = datetime.now(UTC)
        if key_hash is not None:
            existing = self.job_repository.find_by_idempotency_key(
                organization_id=scope.organization_id,
                user_id=user_id,
                idempotency_key_hash=key_hash,
                created_after=now - timedelta(seconds=self.idempotency_ttl_seconds),
            )
            if existing is not None:
                if existing.request_hash != preflight.request_hash:
                    raise _error(
                        code=BACKTEST_ERROR_IDEMPOTENCY_CONFLICT,
                        message="Idempotency-Key was already used with a different request",
                        details={"idempotency_key_hash": key_hash},
                    )
                return BacktestJobCreateResult(
                    job=self._job_read_model(job=existing),
                    idempotent_replay=True,
                )

        self._enforce_full_job_admission(
            organization_id=scope.organization_id,
            user_id=user_id,
            paid_level=paid_level,
            preflight=preflight,
            now=now,
        )
        job_id = uuid4()
        queued_job = BacktestJob.create_queued(
            job_id=job_id,
            organization_id=scope.organization_id,
            user_id=user_id,
            mode="template",
            created_at=now,
            request_json=_job_request_json(
                preflight=preflight,
                key_hash=key_hash,
                light_max_estimated_combinations=self.light_max_estimated_combinations,
                strategy_name=_strategy_name_from_payload(payload),
            ),
            request_hash=preflight.request_hash,
            spec_hash=None,
            spec_payload_json=None,
            engine_params_hash=preflight.result_config_hash,
            backtest_runtime_config_hash=preflight.result_config_hash,
            artifact_pin=_artifact_pin(preflight=preflight),
            execution_mode="background_auto",
            market_id=_market_id(preflight=preflight),
            symbol=_symbol(preflight=preflight),
            timeframe=str(preflight.normalized_request["timeframe"]),
            requested_top_n=int(preflight.normalized_request["top_n"]),
            ranking_primary_metric=_ranking_primary_metric(preflight=preflight),
            ranking_secondary_metric=None,
        )
        stored_job = self.job_repository.create(job=queued_job)
        if self.execution_trigger is not None:
            self.execution_trigger.enqueue(
                job_id=stored_job.job_id,
                organization_id=stored_job.organization_id,
                user_id=user_id,
                request_hash=stored_job.request_hash,
            )
        return BacktestJobCreateResult(
            job=build_backtest_job_read_model(job=stored_job),
            idempotent_replay=False,
        )

    def get(self, *, user_id: UserId, job_id: UUID) -> BacktestJobReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        top_count = len(
            self.job_repository.list_top_variants(
                job_id=job_id,
                organization_id=job.organization_id,
            )
        )
        return build_backtest_job_read_model(job=job, top_variants_count=top_count)

    def list(
        self,
        *,
        user_id: UserId,
        state: str | None,
        risk_mode: str | None,
        limit: int,
        cursor: str | None,
    ) -> BacktestJobListResult:
        scope = self._scope(user_id=user_id)
        parsed_state = _parse_state(state=state)
        parsed_cursor = _decode_cursor(cursor=cursor)
        page = self.job_repository.list_for_user(
            query=BacktestJobListQuery(
                organization_id=scope.organization_id,
                user_id=user_id,
                limit=limit,
                state=parsed_state,
                cursor=parsed_cursor,
            )
        )
        items = tuple(
            build_backtest_job_read_model(job=job)
            for job in page.items
            if _risk_mode_matches(job=job, risk_mode=risk_mode)
        )
        next_cursor = _encode_cursor(cursor=page.next_cursor)
        return BacktestJobListResult(items=items, next_cursor=next_cursor)

    def top(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        limit: int | None = None,
    ) -> BacktestJobTopResult:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        if limit is not None and limit <= 0:
            raise _error(
                code=BACKTEST_ERROR_INVALID_REQUEST,
                message="Backtest top variants limit must be positive",
                details={"limit": limit},
            )
        rows = self.job_repository.list_top_variants(
            job_id=job_id,
            organization_id=job.organization_id,
            limit=limit,
        )
        return BacktestJobTopResult(
            items=tuple(
                build_top_variant_read_model(job_id=str(job_id), row=row)
                for row in sorted(rows, key=lambda item: item.rank)
            )
        )

    def variant(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestJobTopVariantReadModel:
        variant_key = _validate_public_variant_key(variant_key=variant_key)
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        row = self.job_repository.get_top_variant_by_public_key(
            job_id=job_id,
            organization_id=job.organization_id,
            public_variant_key=variant_key,
        )
        if row is None:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest variant was not found",
                details={"job_id": str(job_id), "variant_key": variant_key},
            )
        return build_top_variant_read_model(job_id=str(job_id), row=row)

    def summary(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        top_limit: int | None = None,
    ) -> BacktestResultSummaryReadModel:
        job = self.get(user_id=user_id, job_id=job_id)
        top = self.top(user_id=user_id, job_id=job_id, limit=top_limit)
        return build_result_summary_read_model(job=job, top_variants=top)

    def trades(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestLazyTradesResultReadModel:
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        return context.probe.detail  # type: ignore[return-value]

    def variant_series(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
        kind: BacktestResultSeriesKind,
        points: int,
    ) -> BacktestResultSeriesReadModel | BacktestLazyTradesMaterializationReadModel:
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        cache_read = context.lazy_trades_service.cache.read_series(
            cache_key=context.probe.cache_key,
            now=datetime.now(UTC),
            ttl_seconds=context.probe.ttl_seconds,
            kind=kind,
            points=points,
        )
        if not cache_read.is_hit or cache_read.payload is None:
            return self._request_lazy_materialization(context=context)
        return _series_read_model_from_cache(payload=cache_read.payload, kind=kind)

    def monthly_stats(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestResultStatsReadModel | BacktestLazyTradesMaterializationReadModel:
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        cache_read = context.lazy_trades_service.cache.read_monthly_stats(
            cache_key=context.probe.cache_key,
            now=datetime.now(UTC),
            ttl_seconds=context.probe.ttl_seconds,
        )
        if not cache_read.is_hit or cache_read.payload is None:
            return self._request_lazy_materialization(context=context)
        return _stats_read_model_from_cache(payload=cache_read.payload, kind="monthly")

    def symbol_stats(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestResultStatsReadModel | BacktestLazyTradesMaterializationReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        cache_read = context.lazy_trades_service.cache.read_symbol_stats(
            cache_key=context.probe.cache_key,
            now=datetime.now(UTC),
            ttl_seconds=context.probe.ttl_seconds,
            symbol=symbol_from_job_request(job.request_json),
        )
        if not cache_read.is_hit or cache_read.payload is None:
            return self._request_lazy_materialization(context=context)
        return _stats_read_model_from_cache(payload=cache_read.payload, kind="symbol")

    def paginated_trades(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
        page: int,
        page_size: int,
    ) -> BacktestPaginatedTradesReadModel | BacktestLazyTradesMaterializationReadModel:
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        cache_read = context.lazy_trades_service.cache.read_page(
            cache_key=context.probe.cache_key,
            now=datetime.now(UTC),
            ttl_seconds=context.probe.ttl_seconds,
            page=page,
            page_size=page_size,
        )
        if not cache_read.is_hit or cache_read.payload is None:
            return self._request_lazy_materialization(context=context)
        return _paginated_trades_read_model_from_cache(payload=cache_read.payload)

    def trades_csv(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None = None,
        job_id: UUID,
        variant_key: str,
        max_rows: int | None = None,
    ) -> BacktestTradesCsvReadModel | BacktestLazyTradesMaterializationReadModel:
        context = self._lazy_trades_cache_context_or_materialization(
            user_id=user_id,
            paid_level=paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        if isinstance(context, BacktestLazyTradesMaterializationReadModel):
            return context
        effective_max_rows = normalize_csv_max_rows(max_rows)
        cache_read = context.lazy_trades_service.cache.read_csv(
            cache_key=context.probe.cache_key,
            now=datetime.now(UTC),
            ttl_seconds=context.probe.ttl_seconds,
            max_rows=effective_max_rows,
        )
        if not cache_read.is_hit or cache_read.payload is None:
            return self._request_lazy_materialization(context=context)
        return _trades_csv_read_model_from_cache(payload=cache_read.payload)

    def cancel(self, *, user_id: UserId, job_id: UUID) -> BacktestJobReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        if job.state in {"succeeded", "failed", "cancelled"}:
            return self.get(user_id=user_id, job_id=job_id)
        cancelled = self.job_repository.cancel(
            job_id=job_id,
            organization_id=job.organization_id,
            user_id=user_id,
            cancel_requested_at=datetime.now(UTC),
        )
        if cancelled is None:
            raise _error(
                code=BACKTEST_ERROR_JOB_NOT_CANCELLABLE,
                message="Backtest job is not cancellable",
                details={"job_id": str(job_id), "state": job.state},
            )
        return build_backtest_job_read_model(job=cancelled)

    def delete(self, *, user_id: UserId, job_id: UUID) -> None:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        if job.state not in {"succeeded", "failed", "cancelled"}:
            raise _error(
                code=BACKTEST_ERROR_JOB_NOT_DELETABLE,
                message="Backtest job must be terminal before it can be deleted",
                details={"job_id": str(job_id), "state": job.state},
            )
        deleted = self.job_repository.delete_terminal(
            job_id=job_id,
            organization_id=job.organization_id,
            user_id=user_id,
        )
        if not deleted:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest job was not found",
                details={"job_id": str(job_id)},
            )

    def _lazy_trades_cache_context_or_materialization(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel | None,
        job_id: UUID,
        variant_key: str,
    ) -> "_LazyTradesCacheContext | BacktestLazyTradesMaterializationReadModel":
        variant_key = _validate_public_variant_key(variant_key=variant_key)
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        row = self.job_repository.get_top_variant_by_public_key(
            job_id=job_id,
            organization_id=job.organization_id,
            public_variant_key=variant_key,
        )
        if row is None:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest variant was not found",
                details={"job_id": str(job_id), "variant_key": variant_key},
            )
        if self.lazy_trades_service is None:
            raise _error(
                code=BACKTEST_ERROR_QUEUE_SATURATED,
                message="Backtest lazy trades service is not configured",
                details={"reason": "lazy_trades_service_unavailable"},
            )
        probe = self.lazy_trades_service.read_cached(
            job=job,
            row=row,
            public_variant_key=variant_key,
        )
        context = _LazyTradesCacheContext(
            user_id=user_id,
            paid_level=paid_level,
            job=job,
            row=row,
            public_variant_key=variant_key,
            lazy_trades_service=self.lazy_trades_service,
            probe=probe,
        )
        if probe.detail is not None:
            return context
        return self._request_lazy_materialization(context=context)

    def _request_lazy_materialization(
        self,
        *,
        context: "_LazyTradesCacheContext",
    ) -> BacktestLazyTradesMaterializationReadModel:
        if self.lazy_trades_materialization_repository is None:
            raise _error(
                code=BACKTEST_ERROR_QUEUE_SATURATED,
                message="Backtest lazy trades materialization queue is not configured",
                details={"reason": "lazy_trades_materialization_repository_unavailable"},
            )
        probe = context.probe
        existing_task = self.lazy_trades_materialization_repository.find_by_identity(
            organization_id=context.job.organization_id,
            owner_user_id=context.user_id,
            job_id=context.job.job_id,
            public_variant_key=context.public_variant_key,
            cache_key=probe.cache_key.digest,
        )
        if existing_task is not None and not _materialization_should_requeue(task=existing_task):
            return _materialization_read_model(
                task=existing_task,
                cache_warning=probe.cache_warning,
                cache_lookup_s=probe.cache_lookup_s,
            )
        requested_at = datetime.now(UTC)
        request = BacktestLazyTradesMaterializationRequest(
            organization_id=context.job.organization_id,
            owner_user_id=context.user_id,
            job_id=context.job.job_id,
            public_variant_key=context.public_variant_key,
            variant_hash=context.row.variant_key,
            request_hash=context.job.request_hash,
            engine_params_hash=probe.cache_key.engine_params_hash,
            artifact_manifest_hash=probe.cache_key.artifact_manifest_hash,
            cache_key=probe.cache_key.digest,
            cache_status=probe.cache_status,
            ttl_seconds=probe.ttl_seconds,
            requested_at=requested_at,
        )
        if (
            existing_task is None
            and self.admission_service is not None
            and context.paid_level is not None
        ):
            now = datetime.now(UTC)
            self.admission_service.ensure_lazy_detail_quota_allowed(
                paid_level=context.paid_level,
                snapshot=BacktestLazyDetailQuotaSnapshot(
                    active_lazy_detail_tasks_for_user=(
                        self.lazy_trades_materialization_repository.count_active_for_user(
                            organization_id=context.job.organization_id,
                            owner_user_id=context.user_id,
                        )
                    ),
                    lazy_detail_creates_in_window=(
                        self.lazy_trades_materialization_repository.count_created_for_user_since(
                            organization_id=context.job.organization_id,
                            owner_user_id=context.user_id,
                            created_after=now - timedelta(hours=1),
                        )
                    ),
                    active_lazy_detail_tasks_global=(
                        self.lazy_trades_materialization_repository.count_active_global()
                    ),
                ),
            )
        task = self.lazy_trades_materialization_repository.request_materialization(
            request=request,
        )
        return _materialization_read_model(
            task=task,
            cache_warning=probe.cache_warning,
            cache_lookup_s=probe.cache_lookup_s,
        )

    def _preflight(
        self,
        *,
        payload: Mapping[str, Any],
        paid_level: PaidLevel | None = None,
    ) -> BacktestPreflightResult:
        try:
            if (
                self.admission_service is not None
                and paid_level is not None
                and isinstance(self.preflight_service, BacktestPreflightService)
            ):
                return self.preflight_service.execute(
                    payload,
                    validation_guardrails=(
                        self.admission_service.preflight_validation_guardrails(
                            base_guardrails=self.runtime_config.guardrails,
                        )
                    ),
                )
            return self.preflight_service.execute(payload)
        except BacktestPreflightRejected as error:
            raise _error(
                code=error.error_code,
                message=error.message,
                details=error.details(),
            ) from error

    def _enforce_full_job_admission(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        paid_level: PaidLevel | None,
        preflight: BacktestPreflightResult,
        now: datetime,
    ) -> None:
        if self.admission_service is not None:
            if paid_level is None:
                return
            self.admission_service.ensure_full_job_request_allowed(
                paid_level=paid_level,
                preflight=preflight,
            )
            self.admission_service.ensure_full_job_quota_allowed(
                paid_level=paid_level,
                snapshot=BacktestFullJobQuotaSnapshot(
                    active_full_jobs_for_user=(
                        self.job_repository.count_active_for_user(
                            organization_id=organization_id,
                            user_id=user_id,
                        )
                    ),
                    full_job_creates_in_window=(
                        self.job_repository.count_created_for_user_since(
                            organization_id=organization_id,
                            user_id=user_id,
                            created_after=now - timedelta(hours=1),
                        )
                    ),
                    active_full_jobs_global=self.job_repository.count_active_global(),
                ),
            )
            return

        guardrails = self.runtime_config.guardrails
        active_for_user = self.job_repository.count_active_for_user(
            organization_id=organization_id,
            user_id=user_id,
        )
        user_active_limit = (
            guardrails.max_active_jobs_per_user + guardrails.max_queued_jobs_per_user
        )
        if active_for_user >= user_active_limit:
            raise _error(
                code=BACKTEST_ERROR_RATE_LIMITED,
                message="Backtest active job limit was reached for current user",
                details={"active_jobs": active_for_user},
            )
        active_global = self.job_repository.count_active_global()
        if active_global >= guardrails.max_active_jobs_global:
            raise _error(
                code=BACKTEST_ERROR_QUEUE_SATURATED,
                message="Backtest queue is saturated",
                details={"active_jobs": active_global},
            )

    def _job_read_model(self, *, job: BacktestJob) -> BacktestJobReadModel:
        top_count = None
        if job.state in {"succeeded", "failed", "cancelled"}:
            top_count = len(
                self.job_repository.list_top_variants(
                    job_id=job.job_id,
                    organization_id=job.organization_id,
                )
            )
        return build_backtest_job_read_model(job=job, top_variants_count=top_count)

    def _scope(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return self.organization_scope_resolver.resolve(user_id=user_id)

    def _require_visible_job(self, *, user_id: UserId, job_id: UUID) -> BacktestJob:
        scope = self._scope(user_id=user_id)
        job = self.job_repository.get(
            job_id=job_id,
            organization_id=scope.organization_id,
        )
        if job is None:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest job was not found",
                details={"job_id": str(job_id)},
            )
        if job.user_id != user_id:
            raise _error(
                code=BACKTEST_ERROR_FORBIDDEN,
                message="Backtest job does not belong to current user",
                details={"job_id": str(job_id)},
            )
        return job


@dataclass(frozen=True, slots=True)
class _LazyTradesCacheContext:
    user_id: UserId
    paid_level: PaidLevel | None
    job: BacktestJob
    row: BacktestJobTopVariant
    public_variant_key: str
    lazy_trades_service: BacktestLazyTradesDetailService
    probe: Any


def _materialization_read_model(
    *,
    task: BacktestLazyTradesMaterializationTask,
    cache_warning: str | None,
    cache_lookup_s: float,
) -> BacktestLazyTradesMaterializationReadModel:
    retry_after_seconds = _materialization_retry_after_seconds(status=task.status)
    cache: dict[str, Any] = {
        "status": task.cache_status,
        "cache_key": task.cache_key,
        "cache_path": task.cache_path,
        "ttl_seconds": task.ttl_seconds,
    }
    if cache_warning is not None:
        cache["warning"] = cache_warning
    materialization = {
        "task_id": str(task.task_id),
        "correlation_id": str(task.task_id),
        "status": task.status,
        "retryable": _materialization_retryable(task=task),
        "retry_after_seconds": retry_after_seconds,
        "priority_class": task.priority_class,
        "created_at": _format_datetime(task.created_at),
        "updated_at": _format_datetime(task.updated_at),
        "started_at": _format_datetime(task.started_at),
        "finished_at": _format_datetime(task.finished_at),
        "attempt": task.attempt,
        "last_error": task.last_error,
        "last_error_json": dict(task.last_error_json) if task.last_error_json else None,
        "request_identity": {
            "request_hash": task.request_hash,
            "cache_key": task.cache_key,
        },
    }
    return BacktestLazyTradesMaterializationReadModel(
        job_id=str(task.job_id),
        variant_key=task.public_variant_key,
        variant_hash=task.variant_hash,
        request_hash=task.request_hash,
        status=task.status,
        materialization=materialization,
        cache=cache,
        timing={"cache_lookup_s": cache_lookup_s},
        pagination={"mode": "none"},
    )


def _series_read_model_from_cache(
    *,
    payload: Mapping[str, Any],
    kind: BacktestResultSeriesKind,
) -> BacktestResultSeriesReadModel:
    return BacktestResultSeriesReadModel(
        job_id=str(payload["job_id"]),
        variant_key=str(payload["variant_key"]),
        variant_hash=str(payload["variant_hash"]),
        kind=kind,
        points=tuple(dict(item) for item in payload.get("points", ()) if isinstance(item, Mapping)),
        requested_points=int(payload["requested_points"]),
        max_points=int(payload["max_points"]),
        returned_points=int(payload["returned_points"]),
        source_points=int(payload["source_points"]),
        downsampled=bool(payload["downsampled"]),
        cache=_mapping(payload.get("cache")),
        timing=_mapping(payload.get("timing")),
    )


def _stats_read_model_from_cache(
    *,
    payload: Mapping[str, Any],
    kind: Literal["monthly", "symbol"],
) -> BacktestResultStatsReadModel:
    return BacktestResultStatsReadModel(
        job_id=str(payload["job_id"]),
        variant_key=str(payload["variant_key"]),
        variant_hash=str(payload["variant_hash"]),
        kind=kind,
        items=tuple(dict(item) for item in payload.get("items", ()) if isinstance(item, Mapping)),
        bounds=_mapping(payload.get("bounds")),
        cache=_mapping(payload.get("cache")),
        timing=_mapping(payload.get("timing")),
    )


def _paginated_trades_read_model_from_cache(
    *,
    payload: Mapping[str, Any],
) -> BacktestPaginatedTradesReadModel:
    return BacktestPaginatedTradesReadModel(
        job_id=str(payload["job_id"]),
        variant_key=str(payload["variant_key"]),
        variant_hash=str(payload["variant_hash"]),
        items=tuple(dict(item) for item in payload.get("items", ()) if isinstance(item, Mapping)),
        pagination=_mapping(payload.get("pagination")),
        summary_metrics=_mapping(payload.get("summary_metrics")),
        cache=_mapping(payload.get("cache")),
        timing=_mapping(payload.get("timing")),
    )


def _trades_csv_read_model_from_cache(
    *,
    payload: Mapping[str, Any],
) -> BacktestTradesCsvReadModel:
    return BacktestTradesCsvReadModel(
        content=str(payload.get("content", "")),
        row_count=int(payload.get("row_count") or 0),
        total_rows=int(payload.get("total_rows") or 0),
        max_rows=int(payload.get("max_rows") or 0),
        truncated=bool(payload.get("truncated")),
        sort=str(payload.get("sort", "trade_index_asc")),
        cache=_mapping(payload.get("cache")),
        timing=_mapping(payload.get("timing")),
    )


def _materialization_retry_after_seconds(*, status: str) -> int:
    if status == "queued":
        return 2
    if status == "running":
        return 5
    return 30


def _materialization_retryable(*, task: BacktestLazyTradesMaterializationTask) -> bool:
    if task.status in {"queued", "running"}:
        return True
    if task.status != "failed" or task.last_error_json is None:
        return False
    details = task.last_error_json.get("details")
    return isinstance(details, Mapping) and details.get("retryable") is True


def _materialization_should_requeue(*, task: BacktestLazyTradesMaterializationTask) -> bool:
    if task.status == "completed":
        return True
    if task.status == "cancelled":
        return True
    return task.status == "failed" and _materialization_retryable(task=task)


def _validate_public_variant_key(*, variant_key: str) -> str:
    normalized = variant_key.strip()
    if not normalized:
        raise _error(
            code=BACKTEST_ERROR_INVALID_REQUEST,
            message="Backtest variant_key must be non-empty",
            details={"variant_key": variant_key},
        )
    if len(normalized) > 256:
        raise _error(
            code=BACKTEST_ERROR_INVALID_REQUEST,
            message="Backtest variant_key is too long",
            details={"max_length": 256},
        )
    return normalized


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _job_request_json(
    *,
    preflight: BacktestPreflightResult,
    key_hash: str | None,
    light_max_estimated_combinations: int,
    strategy_name: str | None,
) -> dict[str, Any]:
    payload = dict(preflight.normalized_request)
    if strategy_name:
        payload["ui_metadata"] = {"strategy_name": strategy_name}
    payload["artifact_metadata"] = preflight.artifact_metadata.as_mapping()
    payload[SCHEDULING_METADATA_KEY] = scheduling_metadata_from_preflight(
        preflight=preflight,
        light_max_estimated_combinations=light_max_estimated_combinations,
    )
    if key_hash is not None:
        payload["idempotency"] = {"key_hash": key_hash}
    return payload


def _strategy_name_from_payload(payload: Mapping[str, Any]) -> str | None:
    raw = payload.get("strategy_name")
    if not isinstance(raw, str):
        return None
    normalized = " ".join(raw.strip().split())
    if not normalized:
        return None
    return normalized[:96]


def _artifact_pin(*, preflight: BacktestPreflightResult) -> BacktestJobArtifactPin:
    metadata = preflight.artifact_metadata
    return BacktestJobArtifactPin(
        artifact_slot=cast(BacktestArtifactSlotLiteral, metadata.artifact_slot),
        artifact_slot_generation=metadata.artifact_slot_generation,
        artifact_manifest_hash=metadata.artifact_manifest_hash,
        artifact_asof_date=metadata.artifact_asof_date,
    )


def _symbol(*, preflight: BacktestPreflightResult) -> str:
    coordinates = _coordinates(preflight=preflight)
    return str(coordinates["symbol"])


def _ranking_primary_metric(*, preflight: BacktestPreflightResult) -> str:
    ranking = preflight.normalized_request["ranking"]
    if not isinstance(ranking, Mapping):
        raise _error(
            code="backtest.invalid_request",
            message="Normalized request ranking must be an object",
            details={"path": "ranking"},
        )
    return str(ranking["primary_metric"])


def _market_id(*, preflight: BacktestPreflightResult) -> int:
    coordinates = _coordinates(preflight=preflight)
    return artifact_market_id_from_coordinates_v2(
        ArtifactCoordinatesV2(
            exchange=str(coordinates["exchange"]),
            market_type=str(coordinates["market_type"]),
            symbol=str(coordinates["symbol"]),
        )
    )


def _coordinates(*, preflight: BacktestPreflightResult) -> Mapping[str, Any]:
    coordinates = preflight.normalized_request["coordinates"]
    if not isinstance(coordinates, Mapping):
        raise _error(
            code="backtest.invalid_request",
            message="Normalized request coordinates must be an object",
            details={"path": "coordinates"},
        )
    return coordinates


def _idempotency_key_hash(
    *,
    organization_id: OrganizationId,
    idempotency_key: str | None,
) -> str | None:
    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    return build_research_idempotency_key_hash(
        organization_id=organization_id,
        idempotency_key=normalized,
    )


def _parse_state(*, state: str | None) -> BacktestJobState | None:
    if state is None or not state.strip():
        return None
    normalized = state.strip().lower()
    if normalized not in {"queued", "running", "succeeded", "failed", "cancelled"}:
        raise _error(
            code="backtest.invalid_request",
            message="Unsupported backtest job state filter",
            details={"state": state},
        )
    return cast(BacktestJobState, normalized)


def _risk_mode_matches(*, job: BacktestJob, risk_mode: str | None) -> bool:
    if risk_mode is None or not risk_mode.strip():
        return True
    risk = dict(job.request_json).get("risk")
    return isinstance(risk, Mapping) and risk.get("mode") == risk_mode.strip()


def _encode_cursor(*, cursor: BacktestJobListCursor | None) -> str | None:
    if cursor is None:
        return None
    rendered = json.dumps(dict(cursor.to_payload()), sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(rendered.encode("utf-8")).decode("ascii")


def _decode_cursor(*, cursor: str | None) -> BacktestJobListCursor | None:
    if cursor is None or not cursor.strip():
        return None
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as error:  # noqa: BLE001
        raise _error(
            code="backtest.invalid_request",
            message="Invalid backtest jobs cursor",
            details={"cursor": cursor},
        ) from error
    if not isinstance(payload, Mapping):
        raise _error(
            code="backtest.invalid_request",
            message="Invalid backtest jobs cursor",
            details={"cursor": cursor},
        )
    return BacktestJobListCursor.from_payload(payload=payload)


def _error(*, code: str, message: str, details: Mapping[str, Any]) -> RoehubError:
    return RoehubError(code=code, message=message, details=dict(details))


__all__ = [
    "BACKTEST_ERROR_AUTH_REQUIRED",
    "BACKTEST_ERROR_FORBIDDEN",
    "BACKTEST_ERROR_IDEMPOTENCY_CONFLICT",
    "BACKTEST_ERROR_INVALID_REQUEST",
    "BACKTEST_ERROR_JOB_NOT_CANCELLABLE",
    "BACKTEST_ERROR_JOB_NOT_DELETABLE",
    "BACKTEST_ERROR_NOT_FOUND",
    "BACKTEST_ERROR_QUEUE_SATURATED",
    "BACKTEST_ERROR_RATE_LIMITED",
    "BacktestJobsUseCase",
]
