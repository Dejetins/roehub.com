from .backtest import build_backtests_router
from .extensions import build_extensions_api_module
from .identity import build_identity_api_module, build_identity_router
from .indicators import (
    bind_indicators_runtime_dependencies,
    build_indicators_candle_feed,
    build_indicators_compute,
    build_indicators_registry,
)
from .live_execution import (
    LiveExecutionServices,
    build_live_execution_services,
    build_ui_execution_router_module,
)
from .market_data_reference import build_market_data_reference_router
from .research_tenancy import build_research_organization_scope_resolver
from .strategy import build_strategy_router, is_strategy_api_enabled
from .ui_account import build_account_settings_use_case, build_ui_account_router
from .ui_backtests import build_ui_backtests_router
from .ui_dashboard import build_dashboard_summary_service, build_ui_dashboard_router
from .ui_strategies_dashboard import (
    build_strategy_dashboard_service,
    build_ui_strategies_dashboard_router,
)

__all__ = [
    "build_backtests_router",
    "build_extensions_api_module",
    "build_account_settings_use_case",
    "build_dashboard_summary_service",
    "build_strategy_dashboard_service",
    "build_identity_api_module",
    "build_identity_router",
    "bind_indicators_runtime_dependencies",
    "build_indicators_candle_feed",
    "build_indicators_compute",
    "build_indicators_registry",
    "build_market_data_reference_router",
    "build_research_organization_scope_resolver",
    "LiveExecutionServices",
    "build_live_execution_services",
    "build_ui_execution_router_module",
    "build_strategy_router",
    "build_ui_account_router",
    "build_ui_backtests_router",
    "build_ui_dashboard_router",
    "build_ui_strategies_dashboard_router",
    "is_strategy_api_enabled",
]
