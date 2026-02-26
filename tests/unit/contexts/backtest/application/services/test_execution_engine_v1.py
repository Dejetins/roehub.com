from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    SIGNAL_CODE_LONG_V1,
    SIGNAL_CODE_NEUTRAL_V1,
    SIGNAL_CODE_SHORT_V1,
    BacktestExecutionEngineV1,
    encode_signal_array_v1,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1, RiskParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


@pytest.fixture(name="engine")
def _engine() -> BacktestExecutionEngineV1:
    """
    Build deterministic execution engine fixture for close-fill behavior tests.

    Args:
        None.
    Returns:
        BacktestExecutionEngineV1: Execution engine instance.
    Assumptions:
        Engine has no mutable global state between runs.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestExecutionEngineV1()


def test_execution_engine_v1_edge_gating_prevents_reentry_after_sl(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify persisted signal does not reopen position after SL close without new edge.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        First target bar assumes previous signal is NEUTRAL.
    Raises:
        AssertionError: If engine re-enters on persisted LONG after SL close.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 100.0, 98.0, 110.0, 115.0))
    signals = _signal_array(("LONG", "LONG", "LONG", "LONG", "LONG"))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 5),
        final_signal=signals,
        execution_params=_execution_params(direction_mode="long-short", sizing_mode="all_in"),
        risk_params=RiskParamsV1(sl_enabled=True, sl_pct=1.0, tp_enabled=False, tp_pct=None),
    )

    assert len(outcome.trades) == 1
    assert outcome.trades[0].exit_reason == "sl"
    assert outcome.total_return_pct == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("direction_mode", "signals"),
    (
        ("long-only", ("LONG", "SHORT", "SHORT")),
        ("short-only", ("SHORT", "LONG", "LONG")),
    ),
)
def test_execution_engine_v1_direction_modes_apply_exit_only_semantics(
    engine: BacktestExecutionEngineV1,
    direction_mode: str,
    signals: tuple[str, str, str],
) -> None:
    """
    Verify forbidden opposite signal closes current position but cannot open opposite side.

    Args:
        engine: Execution engine fixture.
        direction_mode: Direction mode under test.
        signals: Target signal path for mode-specific scenario.
    Returns:
        None.
    Assumptions:
        Opposite signal is treated as exit-only in one-side modes.
    Raises:
        AssertionError: If forbidden side position is opened.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 105.0, 90.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 3),
        final_signal=_signal_array(signals),
        execution_params=_execution_params(direction_mode=direction_mode, sizing_mode="all_in"),
        risk_params=RiskParamsV1(sl_enabled=False, sl_pct=None, tp_enabled=False, tp_pct=None),
    )

    assert len(outcome.trades) == 1
    assert outcome.trades[0].exit_reason == "signal_exit"


def test_execution_engine_v1_fee_and_slippage_apply_on_entry_and_exit(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify buy/sell slippage and fee are applied on both entry and exit operations.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Buy fill uses `+slippage`, sell fill uses `-slippage`.
    Raises:
        AssertionError: If final equity does not match deterministic fee/slippage math.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 100.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 2),
        final_signal=_signal_array(("LONG", "NEUTRAL")),
        execution_params=_execution_params(
            direction_mode="long-short",
            sizing_mode="all_in",
            init_cash_quote=1000.0,
            fee_pct=1.0,
            slippage_pct=1.0,
        ),
        risk_params=RiskParamsV1(sl_enabled=False, sl_pct=None, tp_enabled=False, tp_pct=None),
    )

    assert outcome.equity_end_quote == pytest.approx(960.3960396039604)
    assert outcome.total_return_pct == pytest.approx(-3.9603960396039604)
    assert len(outcome.trades) == 1


def test_execution_engine_v1_sl_tp_tie_prioritizes_sl(engine: BacktestExecutionEngineV1) -> None:
    """
    Verify SL priority is applied when SL and TP triggers are both true on one bar.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        `sl_pct=0` and `tp_pct=0` produce deterministic SL/TP tie on unchanged close.
    Raises:
        AssertionError: If close reason is not `sl`.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 100.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 2),
        final_signal=_signal_array(("LONG", "LONG")),
        execution_params=_execution_params(direction_mode="long-short", sizing_mode="all_in"),
        risk_params=RiskParamsV1(sl_enabled=True, sl_pct=0.0, tp_enabled=True, tp_pct=0.0),
    )

    assert len(outcome.trades) == 1
    assert outcome.trades[0].exit_reason == "sl"


def test_execution_engine_v1_forced_close_on_last_target_bar(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify still-open position is force-closed on last bar of target slice.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Forced close runs after signal exit/entry logic on last target bar.
    Raises:
        AssertionError: If trade is not closed with `forced_close` reason.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 105.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 2),
        final_signal=_signal_array(("LONG", "LONG")),
        execution_params=_execution_params(direction_mode="long-short", sizing_mode="all_in"),
        risk_params=RiskParamsV1(sl_enabled=False, sl_pct=None, tp_enabled=False, tp_pct=None),
    )

    assert len(outcome.trades) == 1
    assert outcome.trades[0].exit_reason == "forced_close"
    assert outcome.trades[0].exit_bar_index == 1


def test_execution_engine_v1_profit_lock_reduces_next_entry_budget(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify profit-lock mode freezes part of positive net PnL before next same-bar entry.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Lock amount equals `net_pnl_quote * safe_profit_percent / 100`.
    Raises:
        AssertionError: If second entry budget or safe balance is incorrect.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 110.0, 100.0, 110.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 4),
        final_signal=_signal_array(("LONG", "NEUTRAL", "LONG", "NEUTRAL")),
        execution_params=_execution_params(
            direction_mode="long-short",
            sizing_mode="strategy_compound_profit_lock",
            safe_profit_percent=50.0,
        ),
        risk_params=RiskParamsV1(sl_enabled=False, sl_pct=None, tp_enabled=False, tp_pct=None),
    )

    assert len(outcome.trades) == 2
    assert outcome.trades[1].entry_quote_amount == pytest.approx(1050.0)
    assert outcome.safe_quote == pytest.approx(102.5)
    assert outcome.equity_end_quote == pytest.approx(1205.0)


def test_execution_engine_v1_short_accounting_is_symmetric(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify synthetic short accounting produces symmetric deterministic PnL behavior.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Short mode uses margin-style reserve/release without cash-sale explosion.
    Raises:
        AssertionError: If short trade PnL and final equity are incorrect.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 90.0))
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 2),
        final_signal=_signal_array(("SHORT", "NEUTRAL")),
        execution_params=_execution_params(direction_mode="short-only", sizing_mode="all_in"),
        risk_params=RiskParamsV1(sl_enabled=False, sl_pct=None, tp_enabled=False, tp_pct=None),
    )

    assert len(outcome.trades) == 1
    assert outcome.trades[0].gross_pnl_quote == pytest.approx(100.0)
    assert outcome.equity_end_quote == pytest.approx(1100.0)


def test_execution_engine_v1_is_deterministic_across_repeated_runs(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify same input payload yields identical trades and final equity on repeated runs.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Engine loop ordering and trade-id generation are deterministic.
    Raises:
        AssertionError: If repeated outcomes differ.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 103.0, 99.0, 104.0, 102.0))
    signals = _signal_array(("LONG", "LONG", "SHORT", "SHORT", "NEUTRAL"))
    execution_params = _execution_params(
        direction_mode="long-short",
        sizing_mode="all_in",
        fee_pct=0.05,
        slippage_pct=0.01,
    )
    risk_params = RiskParamsV1(sl_enabled=True, sl_pct=2.0, tp_enabled=True, tp_pct=6.0)

    first = engine.run(
        candles=candles,
        target_slice=slice(0, 5),
        final_signal=signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )
    second = engine.run(
        candles=candles,
        target_slice=slice(0, 5),
        final_signal=signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )

    assert first.trades == second.trades
    assert first.equity_end_quote == second.equity_end_quote
    assert first.total_return_pct == second.total_return_pct


def test_execution_engine_v1_compact_and_legacy_signals_have_equivalent_outcome(
    engine: BacktestExecutionEngineV1,
) -> None:
    """
    Verify compact int8 and legacy string signals yield identical execution metrics.

    Args:
        engine: Execution engine fixture.
    Returns:
        None.
    Assumptions:
        Execution loop normalizes legacy inputs once and then uses compact codes internally.
    Raises:
        AssertionError: If compact and legacy paths diverge on deterministic payloads.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 102.0, 98.0, 104.0, 101.0))
    legacy_signals = _signal_array(("LONG", "LONG", "SHORT", "SHORT", "NEUTRAL"))
    compact_signals = _signal_code_array((1, 1, -1, -1, 0))
    execution_params = _execution_params(
        direction_mode="long-short",
        sizing_mode="all_in",
        fee_pct=0.05,
        slippage_pct=0.01,
    )
    risk_params = RiskParamsV1(sl_enabled=True, sl_pct=2.0, tp_enabled=True, tp_pct=5.0)

    legacy_outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 5),
        final_signal=legacy_signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )
    compact_outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 5),
        final_signal=compact_signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )

    assert compact_signals.dtype == np.int8
    assert legacy_outcome.trades == compact_outcome.trades
    assert legacy_outcome.equity_end_quote == compact_outcome.equity_end_quote
    assert legacy_outcome.total_return_pct == compact_outcome.total_return_pct


def test_encode_signal_array_v1_returns_same_buffer_for_canonical_int8_codes() -> None:
    """
    Verify canonical compact signal vectors use zero-copy fast path in signal encoding.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical vector is one-dimensional, `np.int8`, C-contiguous, and code-valid.
    Raises:
        AssertionError: If fast path allocates a new buffer.
    Side Effects:
        None.
    """
    signal_codes = _signal_code_array((1, 0, -1, 1, 0))

    encoded = encode_signal_array_v1(signals=signal_codes)

    assert encoded is signal_codes


def test_encode_signal_array_v1_legacy_labels_map_to_canonical_codes() -> None:
    """
    Verify legacy `LONG|SHORT|NEUTRAL` vectors normalize to canonical compact int8 codes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy labels are case-sensitive canonical literals in this fixture.
    Raises:
        AssertionError: If label-to-code mapping diverges from v1 contract.
    Side Effects:
        None.
    """
    encoded = encode_signal_array_v1(
        signals=np.asarray(("LONG", "NEUTRAL", "SHORT", "LONG"), dtype="U7")
    )

    assert encoded.dtype == np.int8
    assert encoded.tolist() == [
        int(SIGNAL_CODE_LONG_V1),
        int(SIGNAL_CODE_NEUTRAL_V1),
        int(SIGNAL_CODE_SHORT_V1),
        int(SIGNAL_CODE_LONG_V1),
    ]


def test_execution_engine_v1_skips_risk_evaluation_when_risk_is_disabled(
    engine: BacktestExecutionEngineV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify engine loop does not call risk-exit resolver when both SL and TP are disabled.

    Args:
        engine: Execution engine fixture.
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Stage-A risk policy disables both SL and TP for every variant.
    Raises:
        AssertionError: If risk resolver is called or deterministic outcome changes.
    Side Effects:
        Monkeypatches module-level `_risk_exit_reason` during the test.
    """
    candles = _candles_from_closes((100.0, 102.0, 101.0, 103.0))
    signals = _signal_array(("LONG", "LONG", "LONG", "NEUTRAL"))
    execution_params = _execution_params(direction_mode="long-short", sizing_mode="all_in")
    risk_params = RiskParamsV1(
        sl_enabled=False,
        sl_pct=None,
        tp_enabled=False,
        tp_pct=None,
    )
    reference = engine.run(
        candles=candles,
        target_slice=slice(0, 4),
        final_signal=signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )

    calls = 0

    def _counting_risk_exit_reason(
        *,
        position: object,
        close_price: float,
        risk_params: RiskParamsV1,
    ) -> str | None:
        """
        Count risk resolver calls while preserving deterministic resolver semantics.

        Args:
            position: Open-position payload (ignored by this test helper).
            close_price: Current close price (ignored by this test helper).
            risk_params: Risk settings payload.
        Returns:
            str | None: Always `None` in this instrumentation helper.
        Assumptions:
            Disabled risk policy should prevent this helper from being called.
        Raises:
            None.
        Side Effects:
            Increments local call counter.
        """
        nonlocal calls
        _ = position, close_price, risk_params
        calls += 1
        return None

    monkeypatch.setattr(
        "trading.contexts.backtest.application.services.execution_engine_v1._risk_exit_reason",
        _counting_risk_exit_reason,
    )
    outcome = engine.run(
        candles=candles,
        target_slice=slice(0, 4),
        final_signal=signals,
        execution_params=execution_params,
        risk_params=risk_params,
    )

    assert calls == 0
    assert outcome.trades == reference.trades
    assert outcome.equity_end_quote == reference.equity_end_quote
    assert outcome.total_return_pct == reference.total_return_pct


def _execution_params(
    *,
    direction_mode: str,
    sizing_mode: str,
    init_cash_quote: float = 1000.0,
    fixed_quote: float = 100.0,
    safe_profit_percent: float = 30.0,
    fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> ExecutionParamsV1:
    """
    Build execution-params fixture object for deterministic engine behavior tests.

    Args:
        direction_mode: Direction mode literal.
        sizing_mode: Sizing mode literal.
        init_cash_quote: Initial quote balance.
        fixed_quote: Fixed quote notional.
        safe_profit_percent: Profit-lock percent for lock mode.
        fee_pct: Fee percent.
        slippage_pct: Slippage percent.
    Returns:
        ExecutionParamsV1: Immutable execution params fixture.
    Assumptions:
        Percent arguments use human percent units.
    Raises:
        ValueError: If one input violates execution-parameter invariants.
    Side Effects:
        None.
    """
    return ExecutionParamsV1(
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        init_cash_quote=init_cash_quote,
        fixed_quote=fixed_quote,
        safe_profit_percent=safe_profit_percent,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )


def _signal_array(values: tuple[str, ...]) -> np.ndarray:
    """
    Convert tuple of signal literals into deterministic unicode numpy vector.

    Args:
        values: Signal literals tuple.
    Returns:
        np.ndarray: Unicode signal array.
    Assumptions:
        Caller passes canonical `LONG|SHORT|NEUTRAL` literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    return np.asarray(values, dtype="U7")


def _signal_code_array(values: tuple[int, ...]) -> np.ndarray:
    """
    Convert tuple of compact signal codes into deterministic int8 numpy vector.

    Args:
        values: Compact signal codes (`-1`, `0`, `1`).
    Returns:
        np.ndarray: Compact int8 signal array.
    Assumptions:
        Codes are canonical and follow backtest compact-signal contract.
    Raises:
        ValueError: If one value is outside canonical compact code set.
    Side Effects:
        None.
    """
    signal_codes = np.asarray(values, dtype=np.int8)
    if not np.isin(
        signal_codes,
        (SIGNAL_CODE_SHORT_V1, SIGNAL_CODE_NEUTRAL_V1, SIGNAL_CODE_LONG_V1),
    ).all():
        raise ValueError("values must contain only compact signal codes -1, 0, 1")
    return signal_codes


def _candles_from_closes(closes: tuple[float, ...]) -> CandleArrays:
    """
    Build dense candle arrays fixture from close-price tuple.

    Args:
        closes: Close-price tuple.
    Returns:
        CandleArrays: Deterministic dense 1m candle arrays.
    Assumptions:
        Open/high/low are set equal to close for scenario simplicity.
    Raises:
        ValueError: If primitive constructors reject generated timestamps.
    Side Effects:
        None.
    """
    bars = len(closes)
    ts_open = np.arange(bars, dtype=np.int64) * int(_ONE_MINUTE / timedelta(milliseconds=1))
    close_array = np.asarray(closes, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(_EPOCH_UTC),
            end=UtcTimestamp(_EPOCH_UTC + (_ONE_MINUTE * bars)),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=close_array.copy(),
        high=close_array.copy(),
        low=close_array.copy(),
        close=close_array,
        volume=np.ones(bars, dtype=np.float32),
    )
