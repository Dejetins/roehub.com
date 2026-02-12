"""
Outbound adapters for indicators bounded context.
"""

from .compute_numba import (
    ComputeNumbaWarmupRunner,
    NumbaIndicatorCompute,
    apply_numba_runtime_config,
    ensure_numba_cache_dir_writable,
)
from .compute_numpy import (
    compute_ma_grid_f32 as compute_ma_grid_numpy_f32,
)
from .compute_numpy import (
    is_supported_ma_indicator as is_supported_ma_indicator_numpy,
)
from .config import (
    IndicatorDefaultsValidationError,
    load_indicator_defaults_yaml,
    validate_indicator_defaults,
)
from .feeds import MarketDataCandleFeed
from .registry import YamlIndicatorRegistry

__all__ = [
    "ComputeNumbaWarmupRunner",
    "IndicatorDefaultsValidationError",
    "MarketDataCandleFeed",
    "NumbaIndicatorCompute",
    "YamlIndicatorRegistry",
    "apply_numba_runtime_config",
    "ensure_numba_cache_dir_writable",
    "compute_ma_grid_numpy_f32",
    "is_supported_ma_indicator_numpy",
    "load_indicator_defaults_yaml",
    "validate_indicator_defaults",
]
