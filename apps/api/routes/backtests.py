from __future__ import annotations

from typing import Any, Callable, Mapping
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request, Response

from apps.api.dto import (
    BacktestJobResponse,
    BacktestJobsListResponse,
    BacktestPreflightResponse,
    BacktestRuntimeDefaultsResponse,
    BacktestTopVariantResponse,
    BacktestTopVariantsResponse,
    build_backtest_job_response,
    build_backtest_jobs_list_response,
    build_backtest_preflight_response,
    build_backtest_runtime_defaults_response,
    build_backtest_top_variant_response,
    build_backtest_top_variants_response,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtests_router(
    *,
    runtime_defaults_service: BacktestRuntimeDefaultsService,
    preflight_service: BacktestPreflightService,
    current_user_dependency: CurrentUserDependency,
    jobs_use_case: BacktestJobsUseCase | None = None,
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

    @router.get("/backtests/runtime-defaults", response_model=BacktestRuntimeDefaultsResponse)
    def get_backtest_runtime_defaults(
        _principal: CurrentUserPrincipal = Depends(require_backtest_user),
    ) -> BacktestRuntimeDefaultsResponse:
        defaults = runtime_defaults_service.execute()
        return build_backtest_runtime_defaults_response(defaults=defaults)

    @router.post("/backtests/preflight", response_model=BacktestPreflightResponse)
    def post_backtest_preflight(
        payload: Any = Body(...),
        _principal: CurrentUserPrincipal = Depends(require_backtest_user),
    ) -> BacktestPreflightResponse:
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

    @router.post("/backtests/jobs/{job_id}/cancel", response_model=BacktestJobResponse)
    def post_backtest_job_cancel(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestJobResponse:
        result = use_case.cancel(user_id=principal.user_id, job_id=job_id)
        return build_backtest_job_response(result=result)

    return router


__all__ = ["build_backtests_router"]
