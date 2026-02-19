from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)


def _build_time_range() -> TimeRange:
    """
    Build deterministic one-hour UTC time range fixture for request tests.

    Args:
        None.
    Returns:
        TimeRange: Half-open UTC range `[start, end)`.
    Assumptions:
        Timestamps are timezone-aware UTC values.
    Raises:
        ValueError: If underlying primitives reject provided timestamp values.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 16, 1, 0, tzinfo=timezone.utc)),
    )


def _build_template() -> RunBacktestTemplate:
    """
    Build minimal valid ad-hoc template fixture for Backtest DTO invariant tests.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Valid template-mode payload.
    Assumptions:
        One explicit indicator grid/selection is enough for BKT-EPIC-01 skeleton.
    Raises:
        ValueError: If any underlying primitive/grid contract is invalid.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
        ),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(20,)),
                },
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
    )


def test_run_backtest_request_accepts_saved_mode() -> None:
    """
    Verify request accepts saved mode with `strategy_id` and without template payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved and template modes are mutually exclusive by DTO contract.
    Raises:
        AssertionError: If request mode or values mismatch expected contract.
    Side Effects:
        None.
    """
    request = RunBacktestRequest(
        time_range=_build_time_range(),
        strategy_id=UUID("00000000-0000-0000-0000-000000000101"),
    )

    assert request.mode == "saved"
    assert request.template is None


def test_run_backtest_request_accepts_template_mode() -> None:
    """
    Verify request accepts ad-hoc mode with template payload and without strategy id.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Template mode carries instrument/timeframe/indicator-grid information.
    Raises:
        AssertionError: If mode or payload shape does not match expected contract.
    Side Effects:
        None.
    """
    request = RunBacktestRequest(
        time_range=_build_time_range(),
        template=_build_template(),
    )

    assert request.mode == "template"
    assert request.strategy_id is None


def test_run_backtest_request_rejects_missing_both_modes() -> None:
    """
    Verify request rejects payload when neither saved nor template mode is provided.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case requires exactly one mode to avoid ambiguous behavior.
    Raises:
        AssertionError: If invalid payload does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="exactly one mode"):
        RunBacktestRequest(time_range=_build_time_range())


def test_run_backtest_request_rejects_both_modes_simultaneously() -> None:
    """
    Verify request rejects payload when saved and template modes are both set.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mode ambiguity must fail fast before use-case orchestration.
    Raises:
        AssertionError: If invalid mixed-mode payload does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="exactly one mode"):
        RunBacktestRequest(
            time_range=_build_time_range(),
            strategy_id=UUID("00000000-0000-0000-0000-000000000202"),
            template=_build_template(),
        )

