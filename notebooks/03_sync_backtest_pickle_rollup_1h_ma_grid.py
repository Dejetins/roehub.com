"""Notebook mirror: Pickle -> Rollup 1h -> MA Grid backtests.

This file exists so `ruff`/`pyright` can type-check the same logic that is
stored in `notebooks/03_sync_backtest_pickle_rollup_1h_ma_grid.ipynb`.

Docs:
  - notebooks/README.md
Related:
  - notebooks/03_sync_backtest_pickle_rollup_1h_ma_grid.ipynb
"""

from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading.contexts.backtest.adapters.outbound.config.backtest_runtime_config import (
    load_backtest_runtime_config,
)
from trading.contexts.backtest.application.dto import RunBacktestTemplate
from trading.contexts.backtest.application.services.close_fill_scorer_v1 import (
    CloseFillBacktestStagedScorerV1,
)
from trading.contexts.backtest.application.services.staged_runner_v1 import BacktestStagedRunnerV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.platform.config.indicators_compute_numba import IndicatorsComputeNumbaConfig
from trading.shared_kernel.primitives import InstrumentId, Timeframe, TimeRange, UtcTimestamp

PICKLE_PATH = Path("/ABS/PATH/TO/candles.pkl")
ROLLUP_TO = Timeframe("1h")
WINDOWS = tuple(range(5, 201))
INDICATORS = ("ma.sma", "ma.ema")

PRESELECT = 60
TOP_K = 30
TOP_TRADES_N = 3

NUMBA_NUM_THREADS = 1
NUMBA_CACHE_DIR = Path(".cache/numba/notebooks")


def _utc_dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _to_int64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.int64)


def _to_ts_open_ms(values: object) -> np.ndarray:
    """Normalize time_open/ts_open payload into epoch milliseconds int64."""
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.datetime64):
        return np.ascontiguousarray(raw.astype("datetime64[ms]").astype(np.int64))
    if raw.dtype == object:
        out = np.empty(int(raw.shape[0]), dtype=np.int64)
        for index, item in enumerate(raw.tolist()):
            if item is None:
                raise ValueError("time_open contains None")
            value_attr = getattr(item, "value", None)
            if isinstance(value_attr, (int, np.integer)):
                out[index] = int(int(value_attr) // 1_000_000)
                continue
            if isinstance(item, datetime):
                dt = item if item.tzinfo is not None else item.replace(tzinfo=timezone.utc)
                out[index] = int(dt.timestamp() * 1000.0)
                continue
            out[index] = int(item)
        return np.ascontiguousarray(out, dtype=np.int64)
    return np.ascontiguousarray(raw, dtype=np.int64)


def load_candles_from_pickle(path: Path) -> CandleArrays:
    """Load CandleArrays from a pickle file.

    Supports these pickle payloads:
    - CandleArrays
    - dict with keys: ts_open (or time_open), open, high, low, close, volume
      (optional: market_id, symbol, timeframe)
    - pandas.DataFrame with columns: ts_open (or time_open) + open/high/low/close/volume
    """

    try:
        obj = pickle.loads(path.read_bytes())
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "pandas":
            raise RuntimeError(
                "This pickle requires pandas to unpickle. "
                "Either install pandas (e.g. `uv pip install pandas`) "
                "or re-save the pickle as CandleArrays / dict-of-arrays."
            ) from exc
        raise
    if isinstance(obj, CandleArrays):
        return obj

    if isinstance(obj, dict):
        time_key: str | None = None
        if "ts_open" in obj:
            time_key = "ts_open"
        elif "time_open" in obj:
            time_key = "time_open"

        required = {"open", "high", "low", "close", "volume"}
        if time_key is not None and required.issubset(obj.keys()):
            from trading.shared_kernel.primitives import MarketId, Symbol

            ts_open = _to_ts_open_ms(obj[time_key])
            tf = Timeframe(str(obj.get("timeframe", "1m")))
            tf_ms = int(tf.duration().total_seconds() * 1000)
            start = _utc_dt_from_ms(int(ts_open[0]))
            end = _utc_dt_from_ms(int(ts_open[-1]) + tf_ms)
            return CandleArrays(
                market_id=MarketId(int(obj.get("market_id", 1))),
                symbol=Symbol(str(obj.get("symbol", "BTCUSDT"))),
                time_range=TimeRange(UtcTimestamp(start), UtcTimestamp(end)),
                timeframe=tf,
                ts_open=ts_open,
                open=_to_float32(np.asarray(obj["open"])),
                high=_to_float32(np.asarray(obj["high"])),
                low=_to_float32(np.asarray(obj["low"])),
                close=_to_float32(np.asarray(obj["close"])),
                volume=_to_float32(np.asarray(obj["volume"])),
            )

    raise TypeError(f"unsupported pickle payload type: {type(obj)!r}")


def rollup_candles_to_1h(candles: CandleArrays) -> CandleArrays:
    """Roll up candles to 1h using deterministic OHLCV reduceat operations."""
    if candles.ts_open.shape[0] == 0:
        raise ValueError("candles are empty")

    tf_out = ROLLUP_TO
    hour_ms = int(tf_out.duration().total_seconds() * 1000)
    ts = _to_int64(candles.ts_open)
    buckets = (ts // hour_ms).astype(np.int64)
    change = np.nonzero(buckets[1:] != buckets[:-1])[0] + 1
    starts = np.concatenate((np.asarray((0,), dtype=np.int64), change.astype(np.int64)))
    ends = np.concatenate((starts[1:], np.asarray((ts.shape[0],), dtype=np.int64)))

    open_ = _to_float32(candles.open)
    high = _to_float32(candles.high)
    low = _to_float32(candles.low)
    close = _to_float32(candles.close)
    volume = _to_float32(candles.volume)

    ts_open_1h = (buckets[starts] * hour_ms).astype(np.int64)
    open_1h = open_[starts]
    high_1h = np.maximum.reduceat(high, starts)
    low_1h = np.minimum.reduceat(low, starts)
    volume_1h = np.add.reduceat(volume, starts)
    close_1h = close[(ends - 1).astype(np.int64)]

    start = _utc_dt_from_ms(int(ts_open_1h[0]))
    end = _utc_dt_from_ms(int(ts_open_1h[-1]) + hour_ms)
    return CandleArrays(
        market_id=candles.market_id,
        symbol=candles.symbol,
        time_range=TimeRange(UtcTimestamp(start), UtcTimestamp(end)),
        timeframe=tf_out,
        ts_open=_to_int64(ts_open_1h),
        open=_to_float32(open_1h),
        high=_to_float32(high_1h),
        low=_to_float32(low_1h),
        close=_to_float32(close_1h),
        volume=_to_float32(volume_1h),
    )


def run_ma_grid(*, candles: CandleArrays, indicator_id: str) -> dict[str, object]:
    rt = load_backtest_runtime_config("configs/dev/backtest.yaml")

    bars = int(candles.close.shape[0])
    warmup = int(rt.warmup_bars_default)
    if bars <= warmup + 10:
        raise ValueError(f"not enough bars after rollup: bars={bars}, warmup={warmup}")

    try:
        from trading.contexts.indicators.adapters.outbound.compute_numba.engine import (
            NumbaIndicatorCompute,
        )
        from trading.contexts.indicators.domain.definitions import all_defs
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Numba compute engine is not available: {exc!r}") from exc

    numba_cfg = IndicatorsComputeNumbaConfig(
        numba_num_threads=NUMBA_NUM_THREADS,
        numba_cache_dir=NUMBA_CACHE_DIR,
        max_compute_bytes_total=rt.guards.max_compute_bytes_total,
        max_variants_per_compute=rt.guards.max_variants_per_compute,
    )
    indicator_compute = NumbaIndicatorCompute(defs=all_defs(), config=numba_cfg)
    indicator_compute.warmup()

    grid = GridSpec(
        indicator_id=IndicatorId(indicator_id),
        source=ExplicitValuesSpec(name="source", values=("close",)),
        params={
            "window": ExplicitValuesSpec(name="window", values=WINDOWS),
        },
    )
    template = RunBacktestTemplate(
        instrument_id=InstrumentId(candles.market_id, candles.symbol),
        timeframe=candles.timeframe,
        indicator_grids=(grid,),
        indicator_selections=(),
        signal_grids=None,
        risk_grid=None,
        direction_mode="long-short",
        sizing_mode="all_in",
        risk_params=None,
        execution_params=None,
    )
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=indicator_compute,
        direction_mode=template.direction_mode,
        sizing_mode=template.sizing_mode,
        execution_params=template.execution_params or {},
        market_id=candles.market_id.value,
        target_slice=slice(warmup, bars),
        init_cash_quote_default=rt.execution.init_cash_quote_default,
        fixed_quote_default=rt.execution.fixed_quote_default,
        safe_profit_percent_default=rt.execution.safe_profit_percent_default,
        slippage_pct_default=rt.execution.slippage_pct_default,
        fee_pct_default_by_market_id=rt.execution.fee_pct_default_by_market_id,
        max_variants_guard=rt.guards.max_variants_per_compute,
        max_compute_bytes_total=rt.guards.max_compute_bytes_total,
    )
    runner = BacktestStagedRunnerV1(parallel_workers=1)
    t0 = time.perf_counter()
    res = runner.run(
        template=template,
        candles=candles,
        preselect=PRESELECT,
        top_k=TOP_K,
        indicator_compute=indicator_compute,
        scorer=scorer,
        defaults_provider=None,
        max_variants_per_compute=rt.guards.max_variants_per_compute,
        max_compute_bytes_total=rt.guards.max_compute_bytes_total,
        requested_time_range=candles.time_range,
        top_trades_n=TOP_TRADES_N,
    )
    dt = time.perf_counter() - t0
    top = [
        {
            "rank": i,
            "variant_key": v.variant_key,
            "indicator_variant_key": v.indicator_variant_key,
            "total_return_pct": float(v.total_return_pct),
        }
        for i, v in enumerate(res.variants, start=1)
    ]
    return {
        "indicator_id": indicator_id,
        "bars": bars,
        "warmup": warmup,
        "stage_a_variants_total": int(res.stage_a_variants_total),
        "stage_b_variants_total": int(res.stage_b_variants_total),
        "elapsed_s": float(dt),
        "top": top,
    }


def main() -> list[dict[str, object]]:
    candles_raw = load_candles_from_pickle(PICKLE_PATH)
    candles_1h = rollup_candles_to_1h(candles_raw)
    return [
        run_ma_grid(candles=candles_1h, indicator_id=indicator_id) for indicator_id in INDICATORS
    ]


if __name__ == "__main__":  # pragma: no cover
    print(main())
