"""
Public Backtest runs history/status/top/cancel API routes over persisted run storage.

Docs:
  - docs/architecture/backtest/README.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from typing import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from apps.api.dto import (
    BacktestRunsListResponse,
    BacktestRunStatusResponse,
    BacktestRunTopResponse,
    build_backtest_run_status_response,
    build_backtest_run_top_response,
    build_backtest_runs_list_response,
    decode_backtest_runs_cursor,
    decode_backtest_runs_state,
)
from trading.contexts.backtest.application.ports import CurrentUser
from trading.contexts.backtest.application.use_cases import (
    BacktestRunProgressSnapshotBuilder,
    CancelBacktestRunUseCase,
    GetBacktestRunStatusUseCase,
    GetBacktestRunTopUseCase,
    ListBacktestRunsUseCase,
    map_backtest_exception,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtest_runs_router(
    *,
    get_status_use_case: GetBacktestRunStatusUseCase,
    get_top_use_case: GetBacktestRunTopUseCase,
    list_use_case: ListBacktestRunsUseCase,
    cancel_use_case: CancelBacktestRunUseCase,
    current_user_dependency: CurrentUserDependency,
    sync_deadline_seconds: float,
    run_progress_builder: BacktestRunProgressSnapshotBuilder | None = None,
) -> APIRouter:
    """
    Build public runs router exposing history/status/top/cancel endpoints.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtest_runs.py
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py

    Args:
        get_status_use_case: Runs status use-case implementation.
        get_top_use_case: Runs top-summary use-case implementation.
        list_use_case: Runs history list use-case implementation.
        cancel_use_case: Runs cancel use-case implementation.
        current_user_dependency: Identity dependency resolving authenticated principal.
        sync_deadline_seconds: Reserved router parameter retained for signature compatibility.
        run_progress_builder: Optional additive progress/ETA projector for public runs payloads.
    Returns:
        APIRouter: Configured public runs router.
    Assumptions:
        Owner-only policy and deterministic errors are enforced in use-case layer.
    Raises:
        ValueError: If one required dependency is missing.
    Side Effects:
        None.
    """
    if get_status_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_runs_router requires get_status_use_case")
    if get_top_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_runs_router requires get_top_use_case")
    if list_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_runs_router requires list_use_case")
    if cancel_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_runs_router requires cancel_use_case")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_runs_router requires current_user_dependency")
    if sync_deadline_seconds <= 0.0:
        raise ValueError("build_backtest_runs_router requires sync_deadline_seconds > 0")

    router = APIRouter(tags=["backtest"])
    resolved_run_progress_builder = (
        run_progress_builder or BacktestRunProgressSnapshotBuilder()
    )

    @router.get("/backtests/runs/{run_id}", response_model=BacktestRunStatusResponse)
    def get_backtest_run_status(
        run_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRunStatusResponse:
        """
        Read owner persisted run status/metadata snapshot with explicit `403` vs `404`.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - apps/api/dto/backtest_runs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/common/errors.py

        Args:
            run_id: Requested persisted run identifier.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestRunStatusResponse: Owner run status payload.
        Assumptions:
            Existing foreign run must map to `403 forbidden`.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads one persisted run row from storage.
        """
        try:
            run = get_status_use_case.execute(
                run_id=run_id,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtest_run_status_response(
                run=run,
                progress=resolved_run_progress_builder.build(run=run),
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.get("/backtests/runs/{run_id}/top", response_model=BacktestRunTopResponse)
    def get_backtest_run_top(
        run_id: UUID,
        limit: int | None = None,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRunTopResponse:
        """
        Read owner summary-only persisted top rows with deterministic ordering.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - apps/api/dto/backtest_runs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/common/errors.py

        Args:
            run_id: Requested persisted run identifier.
            limit: Optional top rows limit (defaults to persisted cap).
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestRunTopResponse: Owner summary-only top rows payload.
        Assumptions:
            Rows are always sorted by `rank ASC, variant_key ASC` and exclude report/trades bodies.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads one run row and zero or more summary rows from storage.
        """
        try:
            result = get_top_use_case.execute(
                run_id=run_id,
                current_user=CurrentUser(user_id=principal.user_id),
                limit=limit,
            )
            return build_backtest_run_top_response(result=result)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.get("/backtests/runs", response_model=BacktestRunsListResponse)
    def list_backtest_runs(
        state: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRunsListResponse:
        """
        List owner persisted runs using deterministic keyset pagination.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - apps/api/dto/backtest_runs.py
          - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py

        Args:
            state: Optional state filter query value.
            limit: Page size (validated in use-case/query object).
            cursor: Opaque `base64url(json)` keyset cursor.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestRunsListResponse: Deterministic page payload.
        Assumptions:
            Ordering is fixed to `created_at DESC, job_id DESC`.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Reads one persisted runs page from storage.
        """
        try:
            state_value = decode_backtest_runs_state(state=state)
            cursor_value = decode_backtest_runs_cursor(cursor=cursor)
            page = list_use_case.execute(
                current_user=CurrentUser(user_id=principal.user_id),
                state=state_value,
                limit=limit,
                cursor=cursor_value,
            )
            progress_by_run_id = {
                item.job_id: resolved_run_progress_builder.build(run=item)
                for item in page.items
            }
            return build_backtest_runs_list_response(
                items=page.items,
                next_cursor=page.next_cursor,
                progress_by_run_id=progress_by_run_id,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.post("/backtests/runs/{run_id}/cancel", response_model=BacktestRunStatusResponse)
    def cancel_backtest_run(
        run_id: UUID,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRunStatusResponse:
        """
        Request cancel for one owner persisted run and return the latest status snapshot.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - apps/api/dto/backtest_runs.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/common/errors.py

        Args:
            run_id: Requested persisted run identifier.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestRunStatusResponse: Updated or already-terminal owner run snapshot.
        Assumptions:
            Cancel is idempotent for terminal persisted runs.
        Raises:
            RoehubError: Canonical mapped errors.
        Side Effects:
            Writes cancel marker/state transition for the owner run.
        """
        try:
            run = cancel_use_case.execute(
                run_id=run_id,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtest_run_status_response(
                run=run,
                progress=resolved_run_progress_builder.build(run=run),
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    return router


__all__ = ["build_backtest_runs_router"]
