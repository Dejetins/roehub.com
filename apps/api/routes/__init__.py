from .backtest_ai_config import build_backtest_ai_config_router
from .backtests import build_backtests_router
from .identity import build_identity_router
from .indicators import build_indicators_router
from .market_data_reference import build_market_data_reference_router
from .operations import build_operations_router
from .strategies import build_strategies_router
from .ui_account import build_ui_account_router
from .ui_backtests import build_ui_backtests_router
from .ui_dashboard import build_ui_dashboard_router
from .ui_strategies_dashboard import build_ui_strategies_dashboard_router

__all__ = [
    "build_backtests_router",
    "build_backtest_ai_config_router",
    "build_identity_router",
    "build_indicators_router",
    "build_market_data_reference_router",
    "build_operations_router",
    "build_strategies_router",
    "build_ui_account_router",
    "build_ui_backtests_router",
    "build_ui_dashboard_router",
    "build_ui_strategies_dashboard_router",
]
