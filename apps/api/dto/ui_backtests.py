from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from trading.contexts.backtest.application.dto import BacktestJobCountersResult


class UiBacktestCountersResponse(BaseModel):
    """
    Lightweight browser UI counters for the backtests toolbar.
    """

    active_jobs: int
    max_active_jobs: int
    max_active_jobs_global: int
    can_create: bool
    links: dict[str, Any]


def build_ui_backtest_counters_response(
    *,
    result: BacktestJobCountersResult,
) -> UiBacktestCountersResponse:
    payload = result.as_mapping()
    payload["links"] = {
        "history": "/backtests/jobs",
        "create": "/backtests/jobs",
    }
    return UiBacktestCountersResponse.model_validate(payload)


__all__ = [
    "UiBacktestCountersResponse",
    "build_ui_backtest_counters_response",
]
