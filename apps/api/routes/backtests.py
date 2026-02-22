"""
Backtests API route for synchronous `POST /backtests` runs.

Docs:
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Request

from apps.api.dto import (
    BacktestsPostRequest,
    BacktestsPostResponse,
    build_backtest_run_request,
    build_backtests_post_response,
)
from trading.contexts.backtest.application.ports import BacktestStrategyReader, CurrentUser
from trading.contexts.backtest.application.use_cases import (
    RunBacktestUseCase,
    map_backtest_exception,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtests_router(
    *,
    run_use_case: RunBacktestUseCase,
    strategy_reader: BacktestStrategyReader,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    """
    Build `POST /backtests` router for saved/ad-hoc synchronous backtest runs.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        run_use_case: Backtest application use-case implementation.
        strategy_reader: Saved-strategy reader ACL port for reproducibility `spec_hash`.
        current_user_dependency: Identity dependency resolving authenticated principal.
    Returns:
        APIRouter: Configured backtests router.
    Assumptions:
        Ownership checks for saved mode are enforced in use-case, not in route SQL layer.
    Raises:
        ValueError: If required dependencies are missing.
    Side Effects:
        None.
    """
    if run_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires run_use_case")
    if strategy_reader is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires strategy_reader")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["backtest"])

    @router.post("/backtests", response_model=BacktestsPostResponse)
    def post_backtests(
        request: BacktestsPostRequest,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestsPostResponse:
        """
        Execute deterministic sync backtest flow for saved/ad-hoc request envelope.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - apps/api/dto/backtests.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/common/errors.py

        Args:
            request: Parsed strict API request payload.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestsPostResponse: Deterministic top-K response with reproducibility hashes.
        Assumptions:
            `POST /backtests` mode selection is `strategy_id xor template`.
        Raises:
            RoehubError: Deterministic mapped validation/not_found/forbidden/conflict errors.
        Side Effects:
            Invokes backtest use-case and reads saved strategy snapshot for `spec_hash`.
        """
        try:
            strategy_snapshot = None
            if request.strategy_id is not None:
                strategy_snapshot = strategy_reader.load_any(strategy_id=request.strategy_id)

            use_case_request = build_backtest_run_request(request=request)
            use_case_response = run_use_case.execute(
                request=use_case_request,
                current_user=CurrentUser(user_id=principal.user_id),
            )
            return build_backtests_post_response(
                request=request,
                response=use_case_response,
                strategy_snapshot=strategy_snapshot,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    return router


__all__ = ["build_backtests_router"]
