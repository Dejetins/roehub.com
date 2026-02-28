"""
Backtest Jobs API routes for EPIC-11 async create/status/top/list/cancel workflow.

Docs:
  - docs/architecture/backtest/backtest-jobs-api-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from typing import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from apps.api.dto import (
    BacktestJobsListResponse,
    BacktestJobStatusResponse,
    BacktestJobTopResponse,
    BacktestsPostRequest,
    build_backtest_job_status_response,
    build_backtest_job_top_response,
    build_backtest_jobs_list_response,
    build_backtest_run_request,
    decode_backtest_jobs_cursor,
    decode_backtest_jobs_state,
)
from trading.contexts.backtest.application.ports import CurrentUser
from trading.contexts.backtest.application.use_cases import (
    CancelBacktestJobUseCase,
    CreateBacktestJobCommand,
    CreateBacktestJobUseCase,
    GetBacktestJobStatusUseCase,
    GetBacktestJobTopUseCase,
    ListBacktestJobsUseCase,
    map_backtest_exception,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]



def build_backtest_jobs_router(
    *,
    create_use_case: CreateBacktestJobUseCase,
    get_status_use_case: GetBacktestJobStatusUseCase,
    get_top_use_case: GetBacktestJobTopUseCase,
    list_use_case: ListBacktestJobsUseCase,
    cancel_use_case: CancelBacktestJobUseCase,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    """
    Build EPIC-11 jobs router exposing create/status/top/list/cancel endpoints.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtest_jobs.py
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

    Args:
        create_use_case: Jobs create use-case implementation.
        get_status_use_case: Jobs status use-case implementation.
        get_top_use_case: Jobs top use-case implementation.
        list_use_case: Jobs list use-case implementation.
        cancel_use_case: Jobs cancel use-case implementation.
        current_user_dependency: Identity dependency resolving authenticated principal.
    Returns:
        APIRouter: Configured jobs router.
    Assumptions:
        Owner-only policy and deterministic errors are enforced in use-case layer.
    Raises:
        ValueError: If one required dependency is missing.
    Side Effects:
        None.
    """
    if create_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires create_use_case")
    if get_status_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires get_status_use_case")
    if get_top_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires get_top_use_case")
    if list_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires list_use_case")
    if cancel_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires cancel_use_case")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_jobs_router requires current_user_dependency")

    router = APIRouter(tags=["backtest"])

    @router.post("/backtests/jobs", response_model=BacktestJobStatusResponse, status_code=201)
    def post_backtest_job(
        request: BacktestsPostRequest,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestJobStatusResponse:
        """
        Create queued Backtest job from sync envelope with EPIC-11 validations/quota checks.

        Docs:
          - docs/architecture/backtest/backtest-jobs-api-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - apps/api/dto/backtests.py
          - apps/api/dto/backtest_jobs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

        Args:
            request: Strict parsed API request envelope (`strategy_id xor template`).
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestJobStatusResponse: Created queued job status snapshot.
        Assumptions:
            Endpoint preserves existing sync request envelope semantics.
        Raises:
            RoehubError: Canonical mapped validation/forbidden/not_found/conflict errors.
        Side Effects:
            Persists one queued job row in jobs storage.
        """
        try:
            command = CreateBacktestJobCommand(
                run_request=build_backtest_run_request(request=request),
                request_payload=request.model_dump(mode="json", exclude_none=True),
            )
            created = create_use_case.execute(
                command=command,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtest_job_status_response(job=created)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.get("/backtests/jobs/{job_id}", response_model=BacktestJobStatusResponse)
    def get_backtest_job_status(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestJobStatusResponse:
        """
        Read owner job status/progress snapshot with explicit `403` vs `404` semantics.

        Docs:
          - docs/architecture/backtest/backtest-jobs-api-v1.md
          - docs/architecture/roadmap/milestone-5-epics-v1.md
        Related:
          - apps/api/dto/backtest_jobs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/common/errors.py

        Args:
            job_id: Requested Backtest job identifier.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestJobStatusResponse: Owner job status payload.
        Assumptions:
            Existing foreign job must map to `403 forbidden`.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads one jobs row from storage.
        """
        try:
            job = get_status_use_case.execute(
                job_id=job_id,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtest_job_status_response(job=job)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.get("/backtests/jobs/{job_id}/top", response_model=BacktestJobTopResponse)
    def get_backtest_job_top(
        job_id: UUID,
        limit: int | None = None,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestJobTopResponse:
        """
        Read owner persisted ranking summary top rows with lazy-details policy.

        Docs:
          - docs/architecture/backtest/backtest-jobs-api-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
        Related:
          - apps/api/dto/backtest_jobs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/common/errors.py

        Args:
            job_id: Requested Backtest job identifier.
            limit: Optional top rows limit (defaults to persisted cap).
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestJobTopResponse: Owner top rows payload.
        Assumptions:
            Rows are always sorted by `rank ASC, variant_key ASC` and exclude report/trades details.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads jobs row and top-variants rows from storage.
        """
        try:
            result = get_top_use_case.execute(
                job_id=job_id,
                current_user=CurrentUser(user_id=principal.user_id),
                limit=limit,
            )
            return build_backtest_job_top_response(result=result)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.get("/backtests/jobs", response_model=BacktestJobsListResponse)
    def list_backtest_jobs(
        state: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestJobsListResponse:
        """
        List owner jobs using deterministic keyset pagination with opaque cursor transport.

        Docs:
          - docs/architecture/backtest/backtest-jobs-api-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - apps/api/dto/backtest_jobs.py
          - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

        Args:
            state: Optional state filter query value.
            limit: Page size (validated in use-case/query object).
            cursor: Opaque `base64url(json)` keyset cursor.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestJobsListResponse: Deterministic page payload.
        Assumptions:
            Ordering is fixed to `created_at DESC, job_id DESC`.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads jobs page from storage.
        """
        try:
            state_value = decode_backtest_jobs_state(state=state)
            cursor_value = decode_backtest_jobs_cursor(cursor=cursor)
            page = list_use_case.execute(
                current_user=CurrentUser(user_id=principal.user_id),
                state=state_value,
                limit=limit,
                cursor=cursor_value,
            )
            return build_backtest_jobs_list_response(
                items=page.items,
                next_cursor=page.next_cursor,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.post(
        "/backtests/jobs/{job_id}/cancel",
        response_model=BacktestJobStatusResponse,
    )
    def cancel_backtest_job(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestJobStatusResponse:
        """
        Request owner job cancel and return idempotent updated status payload.

        Docs:
          - docs/architecture/backtest/backtest-jobs-api-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - apps/api/dto/backtest_jobs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/common/errors.py

        Args:
            job_id: Requested Backtest job identifier.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestJobStatusResponse: Updated owner job status snapshot.
        Assumptions:
            Endpoint returns status payload instead of `204` by EPIC-11 contract.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Writes cancel marker/state transition for owner active jobs.
        """
        try:
            cancelled = cancel_use_case.execute(
                job_id=job_id,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtest_job_status_response(job=cancelled)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    return router


__all__ = ["build_backtest_jobs_router"]
