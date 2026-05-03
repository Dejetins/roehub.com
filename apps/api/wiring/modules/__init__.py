from .backtest import build_backtests_router
from .identity import build_identity_api_module, build_identity_router
from .indicators import (
    bind_indicators_runtime_dependencies,
    build_indicators_candle_feed,
    build_indicators_compute,
    build_indicators_registry,
)
from .market_data_reference import build_market_data_reference_router
from .strategy import build_strategy_api_module, build_strategy_router, is_strategy_api_enabled
from .ui_account import build_ui_account_api_module
from .ui_backtests import build_ui_backtests_router as build_ui_backtests_api_router
from .ui_dashboard import build_ui_dashboard_module
from .ui_strategies_monitoring import build_ui_strategy_monitoring_api_module

__all__ = [
    "build_backtests_router",
    "build_identity_api_module",
    "build_identity_router",
    "bind_indicators_runtime_dependencies",
    "build_indicators_candle_feed",
    "build_indicators_compute",
    "build_indicators_registry",
    "build_market_data_reference_router",
    "build_strategy_api_module",
    "build_strategy_router",
    "build_ui_account_api_module",
    "build_ui_backtests_api_router",
    "build_ui_dashboard_module",
    "build_ui_strategy_monitoring_api_module",
    "is_strategy_api_enabled",
]
