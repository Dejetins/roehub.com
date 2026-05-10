from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BacktestJobCreateResult,
    BacktestJobListResult,
    BacktestJobReadModel,
    BacktestJobTopResult,
    BacktestJobTopVariantReadModel,
    BacktestLazyTradesDetailReadModel,
    BacktestPreflightResult,
    build_backtest_job_read_model,
    build_top_variant_read_model,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobExecutionTrigger,
    BacktestJobListQuery,
    BacktestJobRepository,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestLazyTradesDetailService,
    BacktestPaginatedTradesReadModel,
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestResultSeriesKind,
    BacktestResultSeriesReadModel,
    BacktestResultStatsReadModel,
    BacktestResultSummaryReadModel,
    BacktestRuntimeConfig,
    build_monthly_stats_read_model,
    build_paginated_trades_read_model,
    build_result_series_read_model,
    build_result_summary_read_model,
    build_symbol_stats_read_model,
    build_trades_csv,
    symbol_from_job_request,
)
from trading.contexts.backtest.domain.entities import (
    BacktestArtifactSlotLiteral,
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobState,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId

BACKTEST_ERROR_AUTH_REQUIRED = "auth.required"
BACKTEST_ERROR_FORBIDDEN = "backtest.forbidden"
BACKTEST_ERROR_NOT_FOUND = "backtest.not_found"
BACKTEST_ERROR_IDEMPOTENCY_CONFLICT = "backtest.idempotency_key_conflict"
BACKTEST_ERROR_JOB_NOT_CANCELLABLE = "backtest.job_not_cancellable"
BACKTEST_ERROR_JOB_NOT_DELETABLE = "backtest.job_not_deletable"
BACKTEST_ERROR_RATE_LIMITED = "backtest.rate_limited"
BACKTEST_ERROR_QUEUE_SATURATED = "backtest.queue_saturated"


@dataclass(frozen=True, slots=True)
class BacktestJobsUseCase:
    job_repository: BacktestJobRepository
    preflight_service: BacktestPreflightService
    runtime_config: BacktestRuntimeConfig
    execution_trigger: BacktestJobExecutionTrigger | None = None
    lazy_trades_service: BacktestLazyTradesDetailService | None = None
    idempotency_ttl_seconds: int = 86_400

    def create(
        self,
        *,
        user_id: UserId,
        payload: Mapping[str, Any],
        idempotency_key: str | None,
    ) -> BacktestJobCreateResult:
        preflight = self._preflight(payload=payload)
        key_hash = _idempotency_key_hash(idempotency_key=idempotency_key)
        now = datetime.now(UTC)
        if key_hash is not None:
            existing = self.job_repository.find_by_idempotency_key(
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

        self._enforce_guardrails(user_id=user_id)
        job_id = uuid4()
        queued_job = BacktestJob.create_queued(
            job_id=job_id,
            user_id=user_id,
            mode="template",
            created_at=now,
            request_json=_job_request_json(preflight=preflight, key_hash=key_hash),
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
                user_id=user_id,
                request_hash=stored_job.request_hash,
            )
        return BacktestJobCreateResult(
            job=build_backtest_job_read_model(job=stored_job),
            idempotent_replay=False,
        )

    def get(self, *, user_id: UserId, job_id: UUID) -> BacktestJobReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        top_count = len(self.job_repository.list_top_variants(job_id=job_id))
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
        parsed_state = _parse_state(state=state)
        parsed_cursor = _decode_cursor(cursor=cursor)
        page = self.job_repository.list_for_user(
            query=BacktestJobListQuery(
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

    def top(self, *, user_id: UserId, job_id: UUID) -> BacktestJobTopResult:
        self._require_visible_job(user_id=user_id, job_id=job_id)
        rows = self.job_repository.list_top_variants(job_id=job_id)
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
        self._require_visible_job(user_id=user_id, job_id=job_id)
        row = self.job_repository.get_top_variant_by_public_key(
            job_id=job_id,
            public_variant_key=variant_key,
        )
        if row is None:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest variant was not found",
                details={"job_id": str(job_id), "variant_key": variant_key},
        )
        return build_top_variant_read_model(job_id=str(job_id), row=row)

    def summary(self, *, user_id: UserId, job_id: UUID) -> BacktestResultSummaryReadModel:
        job = self.get(user_id=user_id, job_id=job_id)
        top = self.top(user_id=user_id, job_id=job_id)
        return build_result_summary_read_model(job=job, top_variants=top)

    def trades(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestLazyTradesDetailReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        row = self.job_repository.get_top_variant_by_public_key(
            job_id=job_id,
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
        return self.lazy_trades_service.execute(
            job=job,
            row=row,
            public_variant_key=variant_key,
        )

    def variant_series(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
        kind: BacktestResultSeriesKind,
        points: int,
    ) -> BacktestResultSeriesReadModel:
        detail = self.trades(user_id=user_id, job_id=job_id, variant_key=variant_key)
        return build_result_series_read_model(
            detail=detail,
            kind=kind,
            requested_points=points,
        )

    def monthly_stats(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestResultStatsReadModel:
        detail = self.trades(user_id=user_id, job_id=job_id, variant_key=variant_key)
        return build_monthly_stats_read_model(detail=detail)

    def symbol_stats(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestResultStatsReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        detail = self.trades(user_id=user_id, job_id=job_id, variant_key=variant_key)
        return build_symbol_stats_read_model(
            detail=detail,
            symbol=symbol_from_job_request(job.request_json),
        )

    def paginated_trades(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
        page: int,
        page_size: int,
    ) -> BacktestPaginatedTradesReadModel:
        detail = self.trades(user_id=user_id, job_id=job_id, variant_key=variant_key)
        return build_paginated_trades_read_model(
            detail=detail,
            page=page,
            page_size=page_size,
        )

    def trades_csv(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> str:
        detail = self.trades(user_id=user_id, job_id=job_id, variant_key=variant_key)
        return build_trades_csv(detail=detail)

    def cancel(self, *, user_id: UserId, job_id: UUID) -> BacktestJobReadModel:
        job = self._require_visible_job(user_id=user_id, job_id=job_id)
        if job.state in {"succeeded", "failed", "cancelled"}:
            return self.get(user_id=user_id, job_id=job_id)
        cancelled = self.job_repository.cancel(
            job_id=job_id,
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
        deleted = self.job_repository.delete_terminal(job_id=job_id, user_id=user_id)
        if not deleted:
            raise _error(
                code=BACKTEST_ERROR_NOT_FOUND,
                message="Backtest job was not found",
                details={"job_id": str(job_id)},
            )

    def _preflight(self, *, payload: Mapping[str, Any]) -> BacktestPreflightResult:
        try:
            return self.preflight_service.execute(payload)
        except BacktestPreflightRejected as error:
            raise _error(
                code=error.error_code,
                message=error.message,
                details=error.details(),
            ) from error

    def _enforce_guardrails(self, *, user_id: UserId) -> None:
        guardrails = self.runtime_config.guardrails
        active_for_user = self.job_repository.count_active_for_user(user_id=user_id)
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
            top_count = len(self.job_repository.list_top_variants(job_id=job.job_id))
        return build_backtest_job_read_model(job=job, top_variants_count=top_count)

    def _require_visible_job(self, *, user_id: UserId, job_id: UUID) -> BacktestJob:
        job = self.job_repository.get(job_id=job_id)
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


def _job_request_json(
    *,
    preflight: BacktestPreflightResult,
    key_hash: str | None,
) -> dict[str, Any]:
    payload = dict(preflight.normalized_request)
    payload["artifact_metadata"] = preflight.artifact_metadata.as_mapping()
    if key_hash is not None:
        payload["idempotency"] = {"key_hash": key_hash}
    return payload


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


def _idempotency_key_hash(*, idempotency_key: str | None) -> str | None:
    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    "BACKTEST_ERROR_JOB_NOT_CANCELLABLE",
    "BACKTEST_ERROR_JOB_NOT_DELETABLE",
    "BACKTEST_ERROR_NOT_FOUND",
    "BACKTEST_ERROR_QUEUE_SATURATED",
    "BACKTEST_ERROR_RATE_LIMITED",
    "BacktestJobsUseCase",
]
