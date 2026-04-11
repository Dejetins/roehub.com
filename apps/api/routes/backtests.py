"""
Backtests API routes for summary-only launch, runtime-defaults, and compatibility detail wiring.

Docs:
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, Depends, Request, Response

from apps.api.dto import (
    BacktestReportResponse,
    BacktestRuntimeDefaultsResponse,
    BacktestsPostRequest,
    BacktestsPostResponse,
    BacktestsVariantReportPostRequest,
    build_backtest_run_request,
    build_backtest_variant_report_payload,
    build_backtest_variant_report_response,
    build_backtest_variant_report_run_request,
    build_backtests_post_response,
)
from trading.contexts.backtest.application.ports import BacktestStrategyReader, CurrentUser
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.application.use_cases import (
    BacktestRunsApiUseCase,
    map_backtest_exception,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]
_SYNC_DISCONNECT_POLL_SECONDS = 0.2


def build_backtests_router(
    *,
    run_use_case: BacktestRunsApiUseCase,
    strategy_reader: BacktestStrategyReader,
    runtime_defaults_response: BacktestRuntimeDefaultsResponse,
    current_user_dependency: CurrentUserDependency,
    sync_deadline_seconds: float,
    eager_top_reports_enabled: bool,
) -> APIRouter:
    """
    Build backtests router for sync runs and runtime-defaults browser prefill endpoint.

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
        runtime_defaults_response: Prebuilt deterministic runtime defaults response payload.
        current_user_dependency: Identity dependency resolving authenticated principal.
        sync_deadline_seconds: Hard wall-time deadline for cooperative sync cancellation.
        eager_top_reports_enabled:
            Retained legacy flag ignored after summary-only launch cutover.
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
    if runtime_defaults_response is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires runtime_defaults_response")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires current_user_dependency")
    if sync_deadline_seconds <= 0.0:
        raise ValueError("build_backtests_router requires sync_deadline_seconds > 0")
    if not isinstance(eager_top_reports_enabled, bool):
        raise ValueError("build_backtests_router requires eager_top_reports_enabled bool")

    router = APIRouter(tags=["backtest"])

    @router.get(
        "/backtests/runtime-defaults",
        response_model=BacktestRuntimeDefaultsResponse,
    )
    def get_backtests_runtime_defaults(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRuntimeDefaultsResponse:
        """
        Return deterministic non-secret Backtest runtime defaults used by `/backtests` UI.

        Docs:
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - apps/api/dto/backtest_runtime_defaults.py
          - apps/api/wiring/modules/backtest.py
          - apps/web/dist/backtest_ui.js

        Args:
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestRuntimeDefaultsResponse:
                Startup-prebuilt deterministic defaults payload, including additive
                frozen R0 `contracts.*` fields.
        Assumptions:
            Runtime defaults payload is derived from validated startup config.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = principal
        return runtime_defaults_response

    @router.post("/backtests", response_model=BacktestsPostResponse)
    async def post_backtests(
        request: BacktestsPostRequest,
        http_request: Request,
        http_response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestsPostResponse:
        """
        Launch deterministic `/backtests` flow for saved/ad-hoc request envelope.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - apps/api/dto/backtests.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/common/errors.py

        Args:
            request: Parsed strict API request payload.
            http_response: Mutable HTTP response used for `200` vs `202 Accepted`.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestsPostResponse:
                Deterministic sync-inline result or explicit queued `background_auto` launch
                response with reproducibility hashes.
        Assumptions:
            `POST /backtests` mode selection is `strategy_id xor template`.
            Internal sync execution-profile selection remains server-owned and stays hidden from
            the public transport contract after the redesigned sync-engine cutover.
        Raises:
            RoehubError: Deterministic mapped validation/not_found/forbidden/conflict errors.
        Side Effects:
            Invokes backtest use-case and reads saved strategy snapshot for `spec_hash`.
        """
        try:
            use_case_request = build_backtest_run_request(request=request)
            request_payload = request.model_dump(mode="json", exclude_none=True)
            run_control = BacktestRunControlV1(deadline_seconds=sync_deadline_seconds)
            run_task = asyncio.create_task(
                asyncio.to_thread(
                    run_use_case.execute,
                    request=use_case_request,
                    current_user=CurrentUser(user_id=principal.user_id),
                    request_payload=request_payload,
                    run_control=run_control,
                )
            )
            while not run_task.done():
                if await http_request.is_disconnected():
                    run_control.cancel(reason="client_disconnected")
                await asyncio.sleep(_SYNC_DISCONNECT_POLL_SECONDS)
            use_case_response = await run_task
            strategy_snapshot = None
            if request.strategy_id is not None and use_case_response.spec_hash is None:
                strategy_snapshot = strategy_reader.load_any(strategy_id=request.strategy_id)
            if use_case_response.execution_mode == "background_auto":
                http_response.status_code = 202
            return build_backtests_post_response(
                request=request,
                response=use_case_response,
                strategy_snapshot=strategy_snapshot,
                include_reports=False,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    @router.post(
        "/backtests/variant-report",
        response_model=BacktestReportResponse,
    )
    async def post_backtests_variant_report(
        request: BacktestsVariantReportPostRequest,
        http_request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestReportResponse:
        """
        Build one compatibility report via legacy `POST /api/backtests/variant-report`.

        Docs:
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - apps/api/dto/backtests.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/common/errors.py

        Args:
            request: Parsed strict API payload with run-context and explicit variant block.
            principal: Authenticated user principal resolved by identity dependency.
        Returns:
            BacktestReportResponse:
                Deterministic report payload (`rows/table_md/trades`) for one explicit variant.
        Assumptions:
            Active full-detail UX should prefer the run-scoped
            `POST /api/backtests/runs/{run_id}/variant-report` flow; this endpoint remains
            behavior-compatible for older clients that still send the full run envelope.
        Raises:
            RoehubError: Deterministic mapped validation/not_found/forbidden/conflict errors.
        Side Effects:
            Executes one variant scoring pass and report build in application layer.
        """
        try:
            use_case_request = build_backtest_variant_report_run_request(request=request)
            variant_payload = build_backtest_variant_report_payload(request=request.variant)
            run_control = BacktestRunControlV1(deadline_seconds=sync_deadline_seconds)
            run_task = asyncio.create_task(
                asyncio.to_thread(
                    run_use_case.build_variant_report,
                    request=use_case_request,
                    current_user=CurrentUser(user_id=principal.user_id),
                    variant_payload=variant_payload,
                    include_trades=request.include_trades,
                    run_control=run_control,
                )
            )
            while not run_task.done():
                if await http_request.is_disconnected():
                    run_control.cancel(reason="client_disconnected")
                await asyncio.sleep(_SYNC_DISCONNECT_POLL_SECONDS)
            use_case_response = await run_task
            return build_backtest_variant_report_response(report=use_case_response)
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    return router


__all__ = ["build_backtests_router"]
