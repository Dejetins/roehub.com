from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    BacktestExecutionEngineV1,
    CloseFillBacktestStagedScorerV1,
    IndicatorSignalEvaluationInputV1,
    evaluate_and_aggregate_signals_v1,
)
from trading.contexts.backtest.application.services.close_fill_scorer_v1 import (
    _signal_cache_key,
    _SignalCacheValue,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1, RiskParamsV1
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    IndicatorVariantSelection,
    TensorMeta,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


class _FixedSeriesIndicatorCompute:
    """
    IndicatorCompute fake returning deterministic fixed series by indicator id.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py
    """

    def __init__(self, *, outputs: dict[str, np.ndarray]) -> None:
        """
        Initialize deterministic output map and compute call counter.

        Args:
            outputs: Indicator id to output series mapping.
        Returns:
            None.
        Assumptions:
            Output arrays are one-dimensional and timeline-aligned.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._outputs = outputs
        self.compute_calls = 0
        self.requested_layout_preferences: list[Layout | None] = []

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Return placeholder estimate payload for protocol compatibility.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Deterministic single-variant estimate payload.
        Assumptions:
            Scorer tests call `compute` directly and do not rely on estimate semantics.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = max_variants_guard
        return EstimateResult(
            indicator_id=grid.indicator_id,
            axes=(AxisDef(name="window", values_int=(1,)),),
            variants=1,
            max_variants_guard=1,
        )

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Return deterministic tensor with one variant honoring requested layout.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Deterministic tensor with one variant.
        Assumptions:
            Requested indicator id exists in outputs mapping.
        Raises:
            KeyError: If requested indicator id is not configured in fake outputs.
        Side Effects:
            Increments in-memory compute call counter.
        """
        self.compute_calls += 1
        self.requested_layout_preferences.append(req.grid.layout_preference)
        indicator_id = req.grid.indicator_id.value
        output = self._outputs[indicator_id]
        requested_layout = req.grid.layout_preference or Layout.TIME_MAJOR
        if requested_layout is Layout.VARIANT_MAJOR:
            values = np.ascontiguousarray(output.reshape(1, output.shape[0]), dtype=np.float32)
        else:
            values = np.ascontiguousarray(output.reshape(output.shape[0], 1), dtype=np.float32)
        return IndicatorTensor(
            indicator_id=IndicatorId(indicator_id),
            layout=requested_layout,
            axes=(AxisDef(name="variant", values_int=(0,)),),
            values=values,
            meta=TensorMeta(t=output.shape[0], variants=1, nan_policy="propagate", compute_ms=0),
        )

    def warmup(self) -> None:
        """
        Provide no-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is outside scorer unit-test scope.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


def test_close_fill_scorer_v1_applies_stage_risk_policy() -> None:
    """
    Verify Stage A disables SL/TP while Stage B applies close-based risk settings.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persistent LONG signal after SL exit must not reopen without new edge.
    Raises:
        AssertionError: If Stage A/B metrics do not reflect documented risk semantics.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 90.0, 110.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.roc": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 3),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
    )
    selection = _selection(indicator_id="momentum.roc")

    stage_a = scorer.score_variant_metric(
        stage="stage_a",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": True, "sl_pct": 1.0, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )
    stage_b = scorer.score_variant_metric(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": True, "sl_pct": 1.0, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )

    assert stage_a["Total Return [%]"] == pytest.approx(10.0)
    assert stage_b["Total Return [%]"] == pytest.approx(-10.0)


def test_close_fill_scorer_v1_metric_api_and_legacy_api_do_not_call_details_path() -> None:
    """
    Verify metric-only and legacy scoring APIs avoid implicit details-path execution.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `score_variant` is backward-compatible shim over `score_variant_metric`.
    Raises:
        AssertionError: If metric/legacy API attempts to call details scorer path.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 102.0, 101.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.roc": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 3),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
    )
    selection = _selection(indicator_id="momentum.roc")

    def _raise_on_details_call(**_: object) -> object:
        raise AssertionError("score_variant_with_details must not be called")

    setattr(scorer, "score_variant_with_details", _raise_on_details_call)
    metric_payload = scorer.score_variant_metric(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )
    legacy_payload = scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )

    assert metric_payload == legacy_payload


def test_close_fill_scorer_v1_reuses_signal_cache_for_same_base_variant() -> None:
    """
    Verify scorer caches aggregated signal vector for repeated risk-only Stage B variants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cache key is based on indicator variant key and signal parameters only.
    Raises:
        AssertionError: If compute port is called more than once for identical signal payload.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 102.0, 101.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.roc": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 3),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
    )
    selection = _selection(indicator_id="momentum.roc")

    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )
    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": True, "sl_pct": 1.0, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )

    assert compute.compute_calls == 1


def test_close_fill_scorer_v1_cache_is_bounded_and_stores_compact_signals() -> None:
    """
    Verify scorer signal cache is bounded by entry count and stores compact int8 vectors.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Different `signal_params` payloads create different cache keys.
    Raises:
        AssertionError: If cache exceeds configured capacity or stores non-int8 arrays.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 102.0, 101.0, 103.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.roc": np.asarray((1.0, 1.0, 1.0, 1.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 4),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        signals_cache_max_entries=1,
        signals_cache_max_bytes=1024,
    )
    selection = _selection(indicator_id="momentum.roc")

    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )
    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={"momentum.roc": {"debug_toggle": 1}},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )

    assert len(scorer._signals_cache) == 1
    assert all(value.final_signal.dtype == np.int8 for value in scorer._signals_cache.values())


def test_close_fill_scorer_v1_cache_respects_byte_limit() -> None:
    """
    Verify cache byte limit evicts oversized signal vectors and prevents unbounded growth.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        One compact signal vector for 3 bars occupies 3 bytes (`np.int8`).
    Raises:
        AssertionError: If oversized entries are retained despite byte budget.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 101.0, 102.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.roc": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 3),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        signals_cache_max_entries=8,
        signals_cache_max_bytes=2,
    )
    selection = _selection(indicator_id="momentum.roc")

    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )
    scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )

    assert compute.compute_calls == 2
    assert len(scorer._signals_cache) == 0


def test_signal_cache_value_keeps_canonical_int8_signal_without_copy() -> None:
    """
    Verify canonical compact signal vectors are cached without allocating a second buffer.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical final-signal vector is one-dimensional, `np.int8`, C-contiguous.
    Raises:
        AssertionError: If cache value normalization allocates another array.
    Side Effects:
        None.
    """
    final_signal = np.asarray((1, 0, -1, 1), dtype=np.int8)

    cache_value = _SignalCacheValue(final_signal=final_signal)

    assert cache_value.final_signal is final_signal


def test_signal_cache_key_is_hash_stable_and_separates_distinct_signal_params() -> None:
    """
    Verify hashed cache key is deterministic and collision-safe for distinct signal params.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical payload semantics stay identical after switching to SHA-256 key format.
    Raises:
        AssertionError: If deterministic key equivalence or separation is violated.
    Side Effects:
        None.
    """
    key_a = _signal_cache_key(
        indicator_variant_key="A" * 64,
        signal_params={
            "momentum.rsi": {"long_threshold": 30, "short_threshold": 70},
        },
    )
    key_b = _signal_cache_key(
        indicator_variant_key="a" * 64,
        signal_params={
            "momentum.rsi": {"short_threshold": 70, "long_threshold": 30},
        },
    )
    key_c = _signal_cache_key(
        indicator_variant_key="a" * 64,
        signal_params={
            "momentum.rsi": {"long_threshold": 70, "short_threshold": 30},
        },
    )

    assert len(key_a) == 64
    assert key_a == key_b
    assert key_a != key_c


def test_close_fill_scorer_v1_cache_does_not_cross_talk_between_signal_params() -> None:
    """
    Verify cache key isolates variants with same indicator key but different signal params.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `momentum.rsi` threshold params deterministically alter LONG/SHORT decision.
    Raises:
        AssertionError: If cached signals leak between different params payloads.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 105.0, 95.0))
    compute = _FixedSeriesIndicatorCompute(
        outputs={
            "momentum.rsi": np.asarray((20.0, 20.0, 20.0), dtype=np.float32),
        }
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 3),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
    )
    selection = _selection(indicator_id="momentum.rsi")

    long_metric = scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={"momentum.rsi": {"long_threshold": 30, "short_threshold": 70}},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )["Total Return [%]"]
    short_metric = scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={"momentum.rsi": {"long_threshold": 70, "short_threshold": 30}},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="c" * 64,
    )["Total Return [%]"]

    assert compute.compute_calls == 2
    assert long_metric != short_metric


def test_close_fill_scorer_v1_matches_legacy_total_return_pct() -> None:
    """
    Verify scorer compact path keeps `Total Return [%]` equal to legacy signal execution.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy path builds string signals through `evaluate_and_aggregate_signals_v1`.
    Raises:
        AssertionError: If scorer metric diverges from legacy execution outcome.
    Side Effects:
        None.
    """
    candles = _candles_from_closes((100.0, 105.0, 98.0, 106.0))
    indicator_series = np.asarray((1.0, 1.0, -1.0, -1.0), dtype=np.float32)
    compute = _FixedSeriesIndicatorCompute(outputs={"momentum.roc": indicator_series})
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"init_cash_quote": 1000.0, "fee_pct": 0.0, "slippage_pct": 0.0},
        market_id=1,
        target_slice=slice(0, 4),
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
    )
    selection = _selection(indicator_id="momentum.roc")

    scorer_metric = scorer.score_variant(
        stage="stage_b",
        candles=candles,
        indicator_selections=(selection,),
        signal_params={},
        risk_params={"sl_enabled": False, "sl_pct": None, "tp_enabled": False, "tp_pct": None},
        indicator_variant_key="a" * 64,
        variant_key="b" * 64,
    )["Total Return [%]"]

    legacy_input = IndicatorSignalEvaluationInputV1(
        indicator_id="momentum.roc",
        primary_output=indicator_series,
    )
    legacy_signals = evaluate_and_aggregate_signals_v1(
        candles=candles,
        indicator_inputs=(legacy_input,),
    ).final_signal
    legacy_outcome = BacktestExecutionEngineV1().run(
        candles=candles,
        target_slice=slice(0, 4),
        final_signal=legacy_signals,
        execution_params=ExecutionParamsV1(
            direction_mode="long-short",
            sizing_mode="all_in",
            init_cash_quote=1000.0,
            fixed_quote=100.0,
            safe_profit_percent=30.0,
            fee_pct=0.0,
            slippage_pct=0.0,
        ),
        risk_params=RiskParamsV1(
            sl_enabled=False,
            sl_pct=None,
            tp_enabled=False,
            tp_pct=None,
        ),
    )

    assert scorer_metric == pytest.approx(legacy_outcome.total_return_pct)


def _selection(*, indicator_id: str) -> IndicatorVariantSelection:
    """
    Build explicit indicator-variant selection fixture for scorer tests.

    Args:
        indicator_id: Indicator identifier.
    Returns:
        IndicatorVariantSelection: Explicit selection with one window parameter.
    Assumptions:
        Signal-rule evaluation for `momentum.roc` depends only on primary output sign.
    Raises:
        ValueError: If selection invariants are violated.
    Side Effects:
        None.
    """
    return IndicatorVariantSelection(
        indicator_id=indicator_id,
        inputs={},
        params={"window": 5},
    )


def _candles_from_closes(closes: tuple[float, ...]) -> CandleArrays:
    """
    Build deterministic candle arrays fixture from close-price tuple.

    Args:
        closes: Close-price tuple.
    Returns:
        CandleArrays: Dense 1m candle arrays.
    Assumptions:
        Open/high/low values equal close for fixture simplicity.
    Raises:
        ValueError: If primitive constructors reject generated timestamps.
    Side Effects:
        None.
    """
    bars = len(closes)
    timeframe_ms = int(_ONE_MINUTE / timedelta(milliseconds=1))
    ts_open = np.arange(bars, dtype=np.int64) * timeframe_ms
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
