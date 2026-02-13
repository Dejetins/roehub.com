"""
Numpy oracle adapters for indicators compute validation.

Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels.ma
"""

from .ma import compute_ma_grid_f32, is_supported_ma_indicator
from .momentum import compute_momentum_grid_f32, is_supported_momentum_indicator
from .structure import compute_structure_grid_f32, is_supported_structure_indicator
from .trend import compute_trend_grid_f32, is_supported_trend_indicator
from .volatility import compute_volatility_grid_f32, is_supported_volatility_indicator
from .volume import compute_volume_grid_f32, is_supported_volume_indicator

__all__ = [
    "compute_ma_grid_f32",
    "is_supported_ma_indicator",
    "compute_momentum_grid_f32",
    "is_supported_momentum_indicator",
    "compute_structure_grid_f32",
    "is_supported_structure_indicator",
    "compute_trend_grid_f32",
    "is_supported_trend_indicator",
    "compute_volatility_grid_f32",
    "is_supported_volatility_indicator",
    "compute_volume_grid_f32",
    "is_supported_volume_indicator",
]
