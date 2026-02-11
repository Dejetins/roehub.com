from __future__ import annotations

from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.definitions.ma import defs as ma_defs
from trading.contexts.indicators.domain.definitions.momentum import defs as momentum_defs
from trading.contexts.indicators.domain.definitions.structure import defs as structure_defs
from trading.contexts.indicators.domain.definitions.trend import defs as trend_defs
from trading.contexts.indicators.domain.definitions.volatility import defs as volatility_defs
from trading.contexts.indicators.domain.definitions.volume import defs as volume_defs


def test_all_defs_keep_group_order_and_unique_ids() -> None:
    """
    Verify global registry keeps fixed group order and unique indicator ids.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `all_defs()` concatenates groups in fixed order:
        ma->trend->volatility->momentum->volume->structure.
    Raises:
        AssertionError: If ordering or uniqueness constraints are violated.
    Side Effects:
        None.
    """
    merged = all_defs()
    merged_ids = [item.indicator_id.value for item in merged]

    expected = [
        *[item.indicator_id.value for item in ma_defs()],
        *[item.indicator_id.value for item in trend_defs()],
        *[item.indicator_id.value for item in volatility_defs()],
        *[item.indicator_id.value for item in momentum_defs()],
        *[item.indicator_id.value for item in volume_defs()],
        *[item.indicator_id.value for item in structure_defs()],
    ]
    assert merged_ids == expected
    assert len(set(merged_ids)) == len(merged_ids)


def test_each_group_defs_are_sorted_by_indicator_id() -> None:
    """
    Verify deterministic alphabetical ordering of ids inside every group.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Group-level defs are expected to be sorted by indicator_id.
    Raises:
        AssertionError: If any group order differs from sorted ids.
    Side Effects:
        None.
    """
    groups = (
        ma_defs(),
        trend_defs(),
        volatility_defs(),
        momentum_defs(),
        volume_defs(),
        structure_defs(),
    )
    for group_defs in groups:
        group_ids = [item.indicator_id.value for item in group_defs]
        assert group_ids == sorted(group_ids)


def test_expanded_baseline_contains_required_indicator_ids() -> None:
    """
    Verify expanded baseline includes required simple and complex indicator ids.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Expanded baseline must keep mandatory ids from follow-up scope.
    Raises:
        AssertionError: If required ids are missing.
    Side Effects:
        None.
    """
    ids = {item.indicator_id.value for item in all_defs()}

    required = {
        "ma.dema",
        "ma.ema",
        "ma.hma",
        "ma.lwma",
        "ma.rma",
        "ma.sma",
        "ma.tema",
        "ma.vwma",
        "ma.wma",
        "ma.zlema",
        "trend.adx",
        "trend.aroon",
        "trend.chandelier_exit",
        "trend.donchian",
        "trend.ichimoku",
        "trend.keltner",
        "trend.linreg_slope",
        "trend.psar",
        "trend.supertrend",
        "trend.vortex",
        "volatility.atr",
        "volatility.bbands",
        "volatility.bbands_bandwidth",
        "volatility.bbands_percent_b",
        "volatility.hv",
        "volatility.stddev",
        "volatility.tr",
        "volatility.variance",
        "momentum.cci",
        "momentum.fisher",
        "momentum.macd",
        "momentum.ppo",
        "momentum.roc",
        "momentum.rsi",
        "momentum.stoch",
        "momentum.stoch_rsi",
        "momentum.trix",
        "momentum.williams_r",
        "volume.ad_line",
        "volume.cmf",
        "volume.mfi",
        "volume.obv",
        "volume.volume_sma",
        "volume.vwap",
        "volume.vwap_deviation",
        "structure.candle_stats",
        "structure.candle_stats_atr_norm",
        "structure.heikin_ashi",
        "structure.percent_rank",
        "structure.pivots",
        "structure.zscore",
    }

    assert required.issubset(ids)
