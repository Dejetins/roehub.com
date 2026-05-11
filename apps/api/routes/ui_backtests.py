from __future__ import annotations

from typing import Callable, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_backtests import BacktestWorkstationResponse
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class BacktestWorkstationService(Protocol):
    def get_workstation(
        self,
        *,
        principal: CurrentUserPrincipal,
        cursor: str | None,
        state: str | None,
        query: str,
        exchange: str | None,
        market_type: str | None,
        instrument_exchange: str | None,
        instrument_market_type: str | None,
        symbol: str | None,
        launched_from: str | None,
        launched_to: str | None,
        refresh: Literal["initial", "auto", "manual"],
    ) -> BacktestWorkstationResponse:
        ...


def build_ui_backtests_router(
    *,
    workstation_service: BacktestWorkstationService,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    if workstation_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires workstation_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["ui-backtests"])

    def require_backtest_workstation_user(request: Request) -> CurrentUserPrincipal:
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

    @router.get("/ui/backtests/workstation", response_model=BacktestWorkstationResponse)
    def get_backtest_workstation(
        cursor: str | None = Query(default=None),
        state: str | None = Query(default=None),
        query: str = Query(default=""),
        exchange: str | None = Query(default=None),
        market_type: str | None = Query(default=None),
        instrument_exchange: str | None = Query(default=None),
        instrument_market_type: str | None = Query(default=None),
        symbol: str | None = Query(default=None),
        launched_from: str | None = Query(default=None),
        launched_to: str | None = Query(default=None),
        refresh: Literal["initial", "auto", "manual"] = Query(default="initial"),
        principal: CurrentUserPrincipal = Depends(require_backtest_workstation_user),
    ) -> BacktestWorkstationResponse:
        normalized_cursor = cursor.strip() if cursor else None
        normalized_state = state.strip() if state else None
        normalized_query = query.strip()
        normalized_exchange = exchange.strip().casefold() if exchange else None
        normalized_market_type = market_type.strip().casefold() if market_type else None
        normalized_instrument_exchange = (
            instrument_exchange.strip().casefold() if instrument_exchange else None
        )
        normalized_instrument_market_type = (
            instrument_market_type.strip().casefold() if instrument_market_type else None
        )
        normalized_symbol = symbol.strip().upper() if symbol else None
        normalized_launched_from = launched_from.strip() if launched_from else None
        normalized_launched_to = launched_to.strip() if launched_to else None
        return workstation_service.get_workstation(
            principal=principal,
            cursor=normalized_cursor or None,
            state=normalized_state or None,
            query=normalized_query,
            exchange=normalized_exchange or None,
            market_type=normalized_market_type or None,
            instrument_exchange=normalized_instrument_exchange or None,
            instrument_market_type=normalized_instrument_market_type or None,
            symbol=normalized_symbol or None,
            launched_from=normalized_launched_from or None,
            launched_to=normalized_launched_to or None,
            refresh=refresh,
        )

    return router


__all__ = ["build_ui_backtests_router"]
