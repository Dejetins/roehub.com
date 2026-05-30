from __future__ import annotations

from typing import Any, Callable, Mapping
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from apps.api.dto import (
    BacktestJobResponse,
    BacktestJobsListResponse,
    BacktestLazyTradesMaterializationResponse,
    BacktestLazyTradesResponse,
    BacktestPaginatedTradesResponse,
    BacktestPreflightResponse,
    BacktestResultSeriesResponse,
    BacktestResultStatsResponse,
    BacktestResultSummaryResponse,
    BacktestRuntimeDefaultsResponse,
    BacktestTopVariantResponse,
    BacktestTopVariantsResponse,
    build_backtest_job_response,
    build_backtest_jobs_list_response,
    build_backtest_lazy_trades_materialization_response,
    build_backtest_lazy_trades_response,
    build_backtest_paginated_trades_response,
    build_backtest_preflight_response,
    build_backtest_result_series_response,
    build_backtest_result_stats_response,
    build_backtest_result_summary_response,
    build_backtest_runtime_defaults_response,
    build_backtest_top_variant_response,
    build_backtest_top_variants_response,
)
from apps.api.monitoring import record_strategy_variant_launch
from trading.contexts.backtest.application.dto import (
    BacktestLazyTradesMaterializationReadModel,
)
from trading.contexts.backtest.application.services.v2 import (
    DEFAULT_BACKTEST_RESULT_POINTS,
    DEFAULT_BACKTEST_TRADES_CSV_MAX_ROWS,
    MAX_BACKTEST_RESULT_POINTS,
    MAX_BACKTEST_TRADES_CSV_MAX_ROWS,
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.application import (
    CreateStrategyFromBacktestVariantResult,
    CreateStrategyFromBacktestVariantUseCase,
    CurrentUser,
)
from trading.contexts.strategy.domain.entities import Strategy
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]

_STRATEGY_VARIANT_LAUNCH_REJECTED_REASONS = frozenset(
    {
        "forbidden",
        "idempotency_key_conflict",
        "idempotency_key_required",
        "not_found",
        "not_launchable",
        "strategy_variant_launch_unavailable",
        "unavailable",
        "unexpected_error",
    }
)
_STRATEGY_VARIANT_LAUNCH_DUPLICATE_REASONS = frozenset(
    {
        "duplicate",
        "idempotent_replay",
        "source_variant_exists",
    }
)


class BacktestVariantStrategyProvenanceResponse(BaseModel):
    source_job_id: UUID
    source_variant_key: str
    source_variant_hash: str
    source_indicator_variant_hash: str | None
    strategy_spec_hash: str
    launch_request_hash: str


class BacktestVariantStrategyResponse(BaseModel):
    status: str
    duplicate: bool
    duplicate_reason: str | None
    strategy: dict[str, Any]
    provenance: BacktestVariantStrategyProvenanceResponse


def build_backtests_router(
    *,
    runtime_defaults_service: BacktestRuntimeDefaultsService,
    preflight_service: BacktestPreflightService,
    current_user_dependency: CurrentUserDependency,
    jobs_use_case: BacktestJobsUseCase | None = None,
    create_strategy_from_variant_use_case: CreateStrategyFromBacktestVariantUseCase | None = None,
) -> APIRouter:
    """
    Build Iteration 1 public backtests API shell.
    """
    if runtime_defaults_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires runtime_defaults_service")
    if preflight_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires preflight_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["backtests"])

    def require_backtest_user(request: Request) -> CurrentUserPrincipal:
        try:
            return current_user_dependency(request)
        except HTTPException as error:
            if error.status_code == 401:
                raise RoehubError(
                    code="auth.required",
                    message="Authentication is required",
                    details={},
                ) from error
            raise

    def require_jobs_use_case() -> BacktestJobsUseCase:
        if jobs_use_case is None:
            raise RoehubError(
                code="backtest.queue_saturated",
                message="Backtest jobs service is not configured",
                details={"reason": "job_repository_unavailable"},
            )
        return jobs_use_case

    def require_create_strategy_from_variant_use_case() -> CreateStrategyFromBacktestVariantUseCase:
        if create_strategy_from_variant_use_case is None:
            raise RoehubError(
                code="strategy_variant_launch.unavailable",
                message="Create strategy from variant service is not configured",
                details={"reason": "strategy_variant_launch_unavailable"},
            )
        return create_strategy_from_variant_use_case

    @router.get("/backtests/runtime-defaults", response_model=BacktestRuntimeDefaultsResponse)
    def get_backtest_runtime_defaults(
        _principal: CurrentUserPrincipal = Depends(require_backtest_user),
    ) -> BacktestRuntimeDefaultsResponse:
        defaults = runtime_defaults_service.execute()
        return build_backtest_runtime_defaults_response(defaults=defaults)

    @router.post("/backtests/preflight", response_model=BacktestPreflightResponse)
    def post_backtest_preflight(
        payload: Any = Body(...),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
    ) -> BacktestPreflightResponse:
        if jobs_use_case is not None:
            if not isinstance(payload, Mapping):
                raise RoehubError(
                    code="backtest.invalid_request",
                    message="Backtest preflight request must be a JSON object",
                    details={
                        "errors": [
                            {
                                "path": "body",
                                "code": "invalid_type",
                                "message": "Request body must be a JSON object",
                            }
                        ]
                    },
                )
            result = jobs_use_case.preflight(
                user_id=principal.user_id,
                paid_level=principal.paid_level,
                payload=payload,
            )
            return build_backtest_preflight_response(result=result)
        try:
            result = preflight_service.execute(payload)
        except BacktestPreflightRejected as error:
            raise RoehubError(
                code=error.error_code,
                message=error.message,
                details=error.details(),
            ) from error
        return build_backtest_preflight_response(result=result)

    @router.post(
        "/backtests/jobs",
        response_model=BacktestJobResponse,
        status_code=201,
    )
    def post_backtest_job(
        response: Response,
        payload: Any = Body(...),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestJobResponse:
        if not isinstance(payload, Mapping):
            raise RoehubError(
                code="backtest.invalid_request",
                message="Backtest job request must be a JSON object",
                details={
                    "errors": [
                        {
                            "path": "body",
                            "code": "invalid_type",
                            "message": "Request body must be a JSON object",
                        }
                    ]
                },
            )
        result = use_case.create(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        if result.idempotent_replay:
            response.status_code = 200
        return build_backtest_job_response(result=result)

    @router.get("/backtests/jobs", response_model=BacktestJobsListResponse)
    def get_backtest_jobs(
        state: str | None = Query(default=None),
        risk_mode: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=250),
        cursor: str | None = Query(default=None),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestJobsListResponse:
        result = use_case.list(
            user_id=principal.user_id,
            state=state,
            risk_mode=risk_mode,
            limit=limit,
            cursor=cursor,
        )
        return build_backtest_jobs_list_response(result=result)

    @router.get("/backtests/jobs/{job_id}", response_model=BacktestJobResponse)
    def get_backtest_job(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestJobResponse:
        result = use_case.get(user_id=principal.user_id, job_id=job_id)
        return build_backtest_job_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/summary",
        response_model=BacktestResultSummaryResponse,
    )
    def get_backtest_job_summary(
        job_id: UUID,
        top_limit: int | None = Query(default=None, ge=1, le=20),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestResultSummaryResponse:
        result = use_case.summary(user_id=principal.user_id, job_id=job_id, top_limit=top_limit)
        return build_backtest_result_summary_response(result=result)

    @router.get("/backtests/jobs/{job_id}/top", response_model=BacktestTopVariantsResponse)
    def get_backtest_job_top(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestTopVariantsResponse:
        result = use_case.top(user_id=principal.user_id, job_id=job_id)
        return build_backtest_top_variants_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}",
        response_model=BacktestTopVariantResponse,
    )
    def get_backtest_job_variant(
        job_id: UUID,
        variant_key: str,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestTopVariantResponse:
        result = use_case.variant(
            user_id=principal.user_id,
            job_id=job_id,
            variant_key=variant_key,
        )
        return build_backtest_top_variant_response(result=result)

    @router.post(
        "/backtests/jobs/{job_id}/variants/{variant_key}/strategies",
        response_model=BacktestVariantStrategyResponse,
        status_code=201,
    )
    def post_backtest_job_variant_strategy(
        response: Response,
        job_id: UUID,
        variant_key: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: CreateStrategyFromBacktestVariantUseCase = Depends(
            require_create_strategy_from_variant_use_case
        ),
    ) -> BacktestVariantStrategyResponse:
        try:
            result = use_case.execute(
                current_user=CurrentUser(user_id=principal.user_id),
                job_id=job_id,
                variant_key=variant_key,
                idempotency_key=idempotency_key,
            )
        except RoehubError as error:
            record_strategy_variant_launch(
                result="rejected",
                reason=_strategy_variant_launch_rejected_metric_reason(error=error),
            )
            raise
        if result.duplicate:
            response.status_code = 200
            record_strategy_variant_launch(
                result="duplicate",
                reason=_strategy_variant_launch_duplicate_metric_reason(
                    reason=result.duplicate_reason
                ),
            )
        else:
            record_strategy_variant_launch(result="created")
        return _to_variant_strategy_response(result=result)

    @router.post(
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades",
        response_model=BacktestLazyTradesResponse,
    )
    def post_backtest_job_variant_trades(
        response: Response,
        job_id: UUID,
        variant_key: str,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestLazyTradesResponse:
        result = use_case.trades(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_lazy_trades_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}/equity",
        response_model=BacktestResultSeriesResponse | BacktestLazyTradesMaterializationResponse,
    )
    def get_backtest_job_variant_equity(
        response: Response,
        job_id: UUID,
        variant_key: str,
        points: int | None = Query(default=None, ge=10, le=MAX_BACKTEST_RESULT_POINTS),
        max_points: int | None = Query(default=None, ge=10, le=MAX_BACKTEST_RESULT_POINTS),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestResultSeriesResponse | BacktestLazyTradesMaterializationResponse:
        resolved_points = _resolve_result_points(points=points, max_points=max_points)
        result = use_case.variant_series(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
            kind="equity",
            points=resolved_points,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_result_series_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
        response_model=BacktestResultSeriesResponse | BacktestLazyTradesMaterializationResponse,
    )
    def get_backtest_job_variant_drawdown(
        response: Response,
        job_id: UUID,
        variant_key: str,
        points: int | None = Query(default=None, ge=10, le=MAX_BACKTEST_RESULT_POINTS),
        max_points: int | None = Query(default=None, ge=10, le=MAX_BACKTEST_RESULT_POINTS),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestResultSeriesResponse | BacktestLazyTradesMaterializationResponse:
        resolved_points = _resolve_result_points(points=points, max_points=max_points)
        result = use_case.variant_series(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
            kind="drawdown",
            points=resolved_points,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_result_series_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
        response_model=BacktestResultStatsResponse | BacktestLazyTradesMaterializationResponse,
    )
    def get_backtest_job_variant_monthly_stats(
        response: Response,
        job_id: UUID,
        variant_key: str,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestResultStatsResponse | BacktestLazyTradesMaterializationResponse:
        result = use_case.monthly_stats(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_result_stats_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
        response_model=BacktestResultStatsResponse | BacktestLazyTradesMaterializationResponse,
    )
    def get_backtest_job_variant_symbol_stats(
        response: Response,
        job_id: UUID,
        variant_key: str,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestResultStatsResponse | BacktestLazyTradesMaterializationResponse:
        result = use_case.symbol_stats(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_result_stats_response(result=result)

    @router.get(
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades",
        response_model=BacktestPaginatedTradesResponse | BacktestLazyTradesMaterializationResponse,
    )
    def get_backtest_job_variant_trades(
        response: Response,
        job_id: UUID,
        variant_key: str,
        page: int = Query(default=1, ge=1, le=10_000),
        page_size: int = Query(default=50, ge=1, le=100),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestPaginatedTradesResponse | BacktestLazyTradesMaterializationResponse:
        result = use_case.paginated_trades(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
            page=page,
            page_size=page_size,
        )
        _apply_materialization_status_code(response=response, result=result)
        return build_backtest_paginated_trades_response(result=result)

    @router.get("/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv")
    def get_backtest_job_variant_trades_csv(
        job_id: UUID,
        variant_key: str,
        max_rows: int = Query(
            default=DEFAULT_BACKTEST_TRADES_CSV_MAX_ROWS,
            ge=1,
            le=MAX_BACKTEST_TRADES_CSV_MAX_ROWS,
        ),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> Response:
        content = use_case.trades_csv(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            job_id=job_id,
            variant_key=variant_key,
            max_rows=max_rows,
        )
        if isinstance(content, BacktestLazyTradesMaterializationReadModel):
            return JSONResponse(
                status_code=202,
                content=build_backtest_lazy_trades_materialization_response(
                    result=content
                ).model_dump(mode="json"),
            )
        return Response(
            content=content.content,
            media_type="text/csv; charset=utf-8",
            headers={
                "content-disposition": (
                    f'attachment; filename="backtest-{job_id}-{variant_key}-trades.csv"'
                ),
                "x-roehub-trades-row-count": str(content.row_count),
                "x-roehub-trades-total-rows": str(content.total_rows),
                "x-roehub-trades-max-rows": str(content.max_rows),
                "x-roehub-trades-truncated": str(content.truncated).lower(),
                "x-roehub-trades-sort": content.sort,
                "x-roehub-cache-status": str(content.cache.get("status", "unknown")),
            },
        )

    @router.post("/backtests/jobs/{job_id}/cancel", response_model=BacktestJobResponse)
    def post_backtest_job_cancel(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestJobResponse:
        result = use_case.cancel(user_id=principal.user_id, job_id=job_id)
        return build_backtest_job_response(result=result)

    @router.delete("/backtests/jobs/{job_id}", status_code=204, response_model=None)
    def delete_backtest_job(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> Response:
        use_case.delete(user_id=principal.user_id, job_id=job_id)
        return Response(status_code=204)

    return router


def _apply_materialization_status_code(*, response: Response, result: Any) -> None:
    if isinstance(result, BacktestLazyTradesMaterializationReadModel):
        response.status_code = 202


def _to_variant_strategy_response(
    *,
    result: CreateStrategyFromBacktestVariantResult,
) -> BacktestVariantStrategyResponse:
    provenance = result.provenance
    return BacktestVariantStrategyResponse(
        status="duplicate" if result.duplicate else "created",
        duplicate=result.duplicate,
        duplicate_reason=result.duplicate_reason,
        strategy=_strategy_response_mapping(strategy=result.strategy),
        provenance=BacktestVariantStrategyProvenanceResponse(
            source_job_id=provenance.source_job_id,
            source_variant_key=provenance.source_variant_key,
            source_variant_hash=provenance.source_variant_hash,
            source_indicator_variant_hash=provenance.source_indicator_variant_hash,
            strategy_spec_hash=provenance.strategy_spec_hash,
            launch_request_hash=provenance.launch_request_hash,
        ),
    )


def _strategy_response_mapping(*, strategy: Strategy) -> dict[str, Any]:
    return {
        "strategy_id": str(strategy.strategy_id),
        "user_id": str(strategy.user_id),
        "name": strategy.name,
        "created_at": strategy.created_at.isoformat(),
        "is_deleted": strategy.is_deleted,
        "spec": strategy.spec.to_json(),
    }


def _resolve_result_points(*, points: int | None, max_points: int | None) -> int:
    if points is not None and max_points is not None and points != max_points:
        raise RoehubError(
            code="backtest.invalid_request",
            message="Use either points or max_points for result series, not conflicting values",
            details={"points": points, "max_points": max_points},
        )
    return max_points if max_points is not None else points or DEFAULT_BACKTEST_RESULT_POINTS


def _strategy_variant_launch_rejected_metric_reason(*, error: RoehubError) -> str:
    details = error.details or {}
    candidate = str(details.get("reason", ""))
    if candidate in _STRATEGY_VARIANT_LAUNCH_REJECTED_REASONS:
        return candidate
    if error.code.startswith("strategy_variant_launch."):
        suffix = error.code.rsplit(".", 1)[-1]
        if suffix in _STRATEGY_VARIANT_LAUNCH_REJECTED_REASONS:
            return suffix
    if error.code == "unexpected_error":
        return "unexpected_error"
    return "unexpected_error"


def _strategy_variant_launch_duplicate_metric_reason(*, reason: str | None) -> str:
    if reason in _STRATEGY_VARIANT_LAUNCH_DUPLICATE_REASONS:
        return reason
    return "duplicate"


__all__ = ["build_backtests_router"]
