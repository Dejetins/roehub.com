from .backtest import build_backtest_router
from .identity import build_identity_api_module, build_identity_router
from .indicators import (
    bind_indicators_runtime_dependencies,
    build_indicators_candle_feed,
    build_indicators_compute,
    build_indicators_registry,
)
from .market_data_reference import build_market_data_reference_router
from .strategy import build_strategy_router, is_strategy_api_enabled

__all__ = [
    "build_backtest_router",
    "build_identity_api_module",
    "build_identity_router",
    "bind_indicators_runtime_dependencies",
    "build_indicators_candle_feed",
    "build_indicators_compute",
    "build_indicators_registry",
    "build_market_data_reference_router",
    "build_strategy_router",
    "is_strategy_api_enabled",
]
